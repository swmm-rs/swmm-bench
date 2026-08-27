from __future__ import annotations

import dataclasses
import hashlib
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swmm_bench.comparator import (
    DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
    compare_all,
    compare_all_outputs,
)
from swmm_bench.models import (
    BenchmarkResult,
    EngineResult,
    InterfaceArtifact,
    InterfaceFamilyResult,
)
from swmm_bench.runner import run_engine
from swmm_bench.suite import REGRESSION_SUITE_NAME, SuiteModel, materialize_models

INTERFACE_SUITE_NAME = "interface-suite"


@dataclass(frozen=True)
class InterfaceCase:
    family: str
    consumer_model: str
    artifact_name: str
    use_kind: str
    save_kind: str


_CASES = (
    InterfaceCase(
        "rainfall",
        "rainfall-use.inp",
        "rainfall-interface.bin",
        "RAINFALL",
        "RAINFALL",
    ),
    InterfaceCase(
        "runoff",
        "runoff-use.inp",
        "runoff-interface.bin",
        "RUNOFF",
        "RUNOFF",
    ),
    InterfaceCase(
        "hotstart",
        "hotstart-use.inp",
        "hotstart-interface.bin",
        "HOTSTART",
        "HOTSTART",
    ),
    InterfaceCase(
        "rdii",
        "rdii-use.inp",
        "rdii-interface.bin",
        "RDII",
        "RDII",
    ),
    InterfaceCase(
        "routing",
        "inflows-use-toy.inp",
        "inflows-interface.dat",
        "INFLOWS",
        "OUTFLOWS",
    ),
)
INTERFACE_FAMILIES = tuple(case.family for case in _CASES)


class InterfaceSelectionError(ValueError):
    pass


def select_interface_cases(
    families: Sequence[str] | None = None,
) -> tuple[InterfaceCase, ...]:
    if not families:
        return _CASES
    requested = set(families)
    unknown = requested - set(INTERFACE_FAMILIES)
    if unknown:
        valid = ", ".join(INTERFACE_FAMILIES)
        raise InterfaceSelectionError(
            f"Unknown interface family {sorted(unknown)[0]!r}. Valid families: {valid}."
        )
    return tuple(case for case in _CASES if case.family in requested)


def _engine_labels(engine_paths: Sequence[str]) -> list[str]:
    paths = [Path(path) for path in engine_paths]
    basename_counts = Counter(path.name for path in paths)
    labels = [
        (
            f"{path.parent.name}/{path.name}"
            if basename_counts[path.name] > 1
            else path.name
        )
        for path in paths
    ]
    label_counts: dict[str, int] = {}
    for index, label in enumerate(labels):
        label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts[label] > 1:
            labels[index] = f"{label}-{label_counts[label]}"
    return labels


def _copy_model_directory(inp_path: Path, destination: Path) -> Path:
    shutil.copytree(inp_path.parent, destination)
    return destination / inp_path.name


def _replace_use_with_save(text: str, case: InterfaceCase) -> str:
    escaped_use_kind = re.escape(case.use_kind)
    pattern = rf'(?im)^[ \t]*USE[ \t]+{escaped_use_kind}[ \t]+"[^"]+"[ \t]*$'
    replacement = f'SAVE {case.save_kind} "{case.artifact_name}"'
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(
            f"Expected one USE {case.use_kind} directive in {case.consumer_model}"
        )
    return updated


def _remove_files_section(text: str) -> str:
    updated, count = re.subn(
        r"(?ims)^\[FILES\][ \t]*\n.*?(?=^\[|\Z)",
        "",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected one [FILES] section")
    return updated


def _set_option(text: str, name: str, value: str) -> str:
    updated, count = re.subn(
        rf"(?im)^([ \t]*{name}[ \t]+)\S+[ \t]*$",
        rf"\g<1>{value}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Expected one {name} option")
    return updated


def _prepare_derived_models(
    case: InterfaceCase,
    consumer_template: Path,
    preparation_root: Path,
) -> tuple[Path, Path, Path]:
    source_inp = _copy_model_directory(
        consumer_template,
        preparation_root / case.family / "source",
    )
    consumer_inp = _copy_model_directory(
        consumer_template,
        preparation_root / case.family / "consumer",
    )
    baseline_inp = _copy_model_directory(
        consumer_template,
        preparation_root / case.family / "baseline",
    )

    source_text = _replace_use_with_save(
        source_inp.read_text(encoding="utf-8"),
        case,
    )
    baseline_text = _remove_files_section(baseline_inp.read_text(encoding="utf-8"))
    if case.family == "hotstart":
        source_text = _set_option(source_text, "START_DATE", "01/07/1975")
        source_text = _set_option(source_text, "REPORT_START_DATE", "01/07/1975")
        source_text = _set_option(source_text, "END_DATE", "01/08/1975")
        baseline_text = _set_option(baseline_text, "START_DATE", "01/07/1975")
        baseline_text = _set_option(
            baseline_text,
            "REPORT_START_DATE",
            "01/08/1975",
        )

    source_inp.write_text(source_text, encoding="utf-8")
    baseline_inp.write_text(baseline_text, encoding="utf-8")
    for root in (source_inp.parent, baseline_inp.parent):
        artifact = root / case.artifact_name
        if artifact.exists():
            artifact.unlink()
    return source_inp, consumer_inp, baseline_inp


def _prepare_routing_models(
    consumer_template: Path,
    generator_template: Path,
    baseline_template: Path,
    preparation_root: Path,
) -> tuple[Path, Path, Path]:
    source_inp = _copy_model_directory(
        generator_template,
        preparation_root / "routing" / "source",
    )
    consumer_inp = _copy_model_directory(
        consumer_template,
        preparation_root / "routing" / "consumer",
    )
    baseline_inp = _copy_model_directory(
        baseline_template,
        preparation_root / "routing" / "baseline",
    )
    artifact = source_inp.parent / "inflows-interface.dat"
    if artifact.exists():
        artifact.unlink()
    return source_inp, consumer_inp, baseline_inp


def _append_error(result: EngineResult, message: str) -> None:
    result.error = f"{result.error}; {message}" if result.error else message


def _artifact_from_generator(
    result: EngineResult,
    artifact_name: str,
) -> InterfaceArtifact | None:
    if result.exit_code != 0:
        _append_error(result, f"Engine exited with code {result.exit_code}")
        return None
    if result.error or not result.rpt_path:
        return None
    artifact_path = Path(result.rpt_path).parent / "model" / artifact_name
    if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
        _append_error(
            result,
            f"Engine did not produce a non-empty interface file: {artifact_name}",
        )
        return None
    with artifact_path.open("rb") as artifact_file:
        digest = hashlib.file_digest(artifact_file, "sha256").hexdigest()
    return InterfaceArtifact(
        path=str(artifact_path.resolve()),
        size_bytes=artifact_path.stat().st_size,
        sha256=digest,
    )


def _successful(result: EngineResult) -> bool:
    return result.exit_code == 0 and result.error is None


def run_interface_suite(
    *,
    source_engine: str,
    target_engines: Sequence[str],
    families: Sequence[str] | None,
    work_dir: Path,
    name: str,
    timeout: float | None,
    option_overrides: Mapping[str, str] | None = None,
    parse_workers: int = 4,
    output_workers: int = 1,
    report_target_mb: int = 100,
    output_typical_weight: float = DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
    platform: dict[str, Any],
    progress_callback: Callable[[EngineResult], None] | None = None,
    run_started_callback: Callable[[str, str], None] | None = None,
) -> BenchmarkResult:
    source_label = Path(source_engine).expanduser().resolve().name
    target_labels = _engine_labels(target_engines)
    cases = select_interface_cases(families)
    work_dir.mkdir(parents=True, exist_ok=True)
    engine_results: list[EngineResult] = []
    family_results: list[InterfaceFamilyResult] = []
    comparison_results: list[EngineResult] = []

    use_models = [
        SuiteModel(
            REGRESSION_SUITE_NAME,
            "use_interfaces",
            f"use_interfaces/{case.consumer_model}",
        )
        for case in cases
    ]
    with (
        ExitStack() as stack,
        tempfile.TemporaryDirectory(
            prefix="swmm-interface-preparation-"
        ) as temporary_directory,
    ):
        materialized_use = stack.enter_context(materialize_models(use_models))
        use_paths = {
            item.model.relative_path: item.inp_path for item in materialized_use
        }
        routing_paths: dict[str, Path] = {}
        if any(case.family == "routing" for case in cases):
            routing_models = (
                SuiteModel(
                    INTERFACE_SUITE_NAME,
                    "routing",
                    "routing/generator.inp",
                ),
                SuiteModel(
                    INTERFACE_SUITE_NAME,
                    "routing",
                    "routing/baseline.inp",
                ),
            )
            materialized_routing = stack.enter_context(
                materialize_models(routing_models)
            )
            routing_paths = {
                item.model.relative_path: item.inp_path for item in materialized_routing
            }

        preparation_root = Path(temporary_directory)
        for case in cases:
            consumer_template = use_paths[f"use_interfaces/{case.consumer_model}"]
            if case.family == "routing":
                source_inp, consumer_inp, baseline_inp = _prepare_routing_models(
                    consumer_template,
                    routing_paths["routing/generator.inp"],
                    routing_paths["routing/baseline.inp"],
                    preparation_root,
                )
            else:
                source_inp, consumer_inp, baseline_inp = _prepare_derived_models(
                    case,
                    consumer_template,
                    preparation_root,
                )

            generator_identity = f"interface://{case.family}/generator"
            consumer_identity = f"interface://{case.family}/consumer"
            baseline_identity = f"interface://{case.family}/baseline"
            if run_started_callback:
                run_started_callback(source_label, f"{case.family}/generator")
            generator_result = run_engine(
                source_engine,
                source_inp,
                work_dir,
                timeout,
                inp_name=f"{case.family}/generator",
                inp_identity=generator_identity,
                option_overrides=option_overrides,
            )
            engine_results.append(generator_result)
            if progress_callback:
                progress_callback(generator_result)

            artifact = _artifact_from_generator(
                generator_result,
                case.artifact_name,
            )
            self_identities = [
                f"interface://{case.family}/self/{index}"
                for index in range(len(target_engines))
            ]
            family_results.append(
                InterfaceFamilyResult(
                    family=case.family,
                    generator_identity=generator_identity,
                    consumer_identity=consumer_identity,
                    baseline_identity=baseline_identity,
                    self_comparison_identities=self_identities,
                    artifact=artifact,
                )
            )
            if artifact is None:
                continue

            shutil.copy2(artifact.path, consumer_inp.parent / case.artifact_name)
            consumers: list[EngineResult] = []
            baselines: list[EngineResult] = []
            for target_engine, target_label in zip(
                target_engines, target_labels, strict=True
            ):
                if run_started_callback:
                    run_started_callback(
                        target_label, f"{case.family}/interface-consumer"
                    )
                consumer = run_engine(
                    target_engine,
                    consumer_inp,
                    work_dir,
                    timeout,
                    inp_name=f"{case.family}/interface-consumer",
                    inp_identity=consumer_identity,
                    option_overrides=option_overrides,
                    engine_name=target_label,
                )
                if progress_callback:
                    progress_callback(consumer)
                if run_started_callback:
                    run_started_callback(
                        target_label, f"{case.family}/direct-baseline"
                    )
                baseline = run_engine(
                    target_engine,
                    baseline_inp,
                    work_dir,
                    timeout,
                    inp_name=f"{case.family}/direct-baseline",
                    inp_identity=baseline_identity,
                    option_overrides=option_overrides,
                    engine_name=target_label,
                )
                if progress_callback:
                    progress_callback(baseline)
                consumers.append(consumer)
                baselines.append(baseline)
                engine_results.extend((consumer, baseline))

            comparison_results.extend(
                result for result in (*consumers, *baselines) if _successful(result)
            )
            for self_identity, consumer, baseline in zip(
                self_identities,
                consumers,
                baselines,
                strict=True,
            ):
                if not _successful(consumer) or not _successful(baseline):
                    continue
                comparison_results.extend(
                    (
                        dataclasses.replace(
                            consumer,
                            engine_name=f"{consumer.engine_name} interface",
                            inp_path=self_identity,
                            inp_name=f"{case.family}: {consumer.engine_name}",
                        ),
                        dataclasses.replace(
                            baseline,
                            engine_name=f"{baseline.engine_name} direct",
                            inp_path=self_identity,
                            inp_name=f"{case.family}: {baseline.engine_name}",
                        ),
                    )
                )

    return BenchmarkResult(
        schema_version="6",
        name=name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        platform=platform,
        engine_results=engine_results,
        comparisons=compare_all(
            comparison_results, parse_workers=parse_workers
        ),
        output_comparisons=compare_all_outputs(
            comparison_results,
            parse_workers=output_workers,
            report_size_mb=report_target_mb,
            typical_weight=output_typical_weight,
        ),
        interface_families=family_results,
    )
