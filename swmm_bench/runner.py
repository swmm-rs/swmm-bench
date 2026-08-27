from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import psutil  # pyright: ignore[reportMissingModuleSource]
from swmm.pandas import Report  # pyright: ignore[reportMissingImports]

from swmm_bench.models import BenchmarkSample, EngineResult


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _display_name(inp_path: Path) -> str:
    try:
        return str(inp_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return inp_path.name


def _analysis_duration_seconds(rpt_path: Path) -> float | None:
    try:
        duration = Report(str(rpt_path)).analysis_duration.total_seconds()
        return duration if math.isfinite(duration) else None
    except Exception:
        # A run can still produce a useful partial report without timing metadata.
        return None


def _copy_model_tree(
    inp_path: Path,
    destination_root: Path,
    excluded_roots: tuple[Path, ...] = (),
) -> Path:
    source_root = inp_path.parent.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    copied_inp = destination_root / inp_path.name
    excluded_paths = {path.expanduser().resolve() for path in excluded_roots}
    excluded_paths.add(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    def is_excluded(path: Path) -> bool:
        absolute_path = path.absolute()
        return any(
            absolute_path == excluded_path or excluded_path in absolute_path.parents
            for excluded_path in excluded_paths
        )

    for directory, directory_names, file_names in os.walk(
        source_root,
        topdown=True,
        followlinks=False,
    ):
        source_directory = Path(directory)
        target_directory = destination_root / source_directory.relative_to(source_root)
        target_directory.mkdir(parents=True, exist_ok=True)

        directory_names[:] = [
            name for name in directory_names if not is_excluded(source_directory / name)
        ]
        for name in directory_names:
            (target_directory / name).mkdir(parents=True, exist_ok=True)

        for name in file_names:
            source = source_directory / name
            if not is_excluded(source):
                shutil.copy2(source, target_directory / name)

    if not copied_inp.exists():
        shutil.copy2(inp_path, copied_inp)

    return copied_inp


def _set_option(inp_path: Path, option_name: str, value: str) -> None:
    if (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", option_name) is None
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            "INP option overrides require a valid name and single-token value"
        )
    with inp_path.open("r", encoding="utf-8", newline="") as inp_file:
        text = inp_file.read()
    options = re.search(
        r"(?ims)^\[OPTIONS\][ \t]*(?:;[^\r\n]*)?(?=\r?$).*?(?=^\[|\Z)",
        text,
    )
    if options is None:
        return
    section, replacements = re.subn(
        rf"(?im)^([ \t]*{re.escape(option_name)}[ \t]+)\S+"
        r"([ \t]*(?:;[^\r\n]*)?)(?=\r?$)",
        lambda match: f"{match.group(1)}{value}{match.group(2)}",
        options.group(),
    )
    if replacements == 0:
        header = re.search(
            r"(?im)^\[OPTIONS\][ \t]*(?:;[^\r\n]*)?(?=\r?$)",
            section,
        )
        if header is None:
            raise ValueError("Matched [OPTIONS] section has no valid section header")
        newline = "\r\n" if "\r\n" in text else "\n"
        section = (
            section[: header.end()]
            + f"{newline}{option_name:<20} {value}"
            + section[header.end() :]
        )
    with inp_path.open("w", encoding="utf-8", newline="") as inp_file:
        inp_file.write(text[: options.start()] + section + text[options.end() :])


def run_engine(
    engine_path: str,
    inp_path: Path,
    work_dir: Path,
    timeout: float | None,
    inp_name: str | None = None,
    inp_identity: str | None = None,
    engine_name: str | None = None,
    option_overrides: Mapping[str, str] | None = None,
    excluded_roots: tuple[Path, ...] = (),
) -> EngineResult:
    engine = Path(engine_path).expanduser().resolve()
    resolved_inp = inp_path.expanduser().resolve()
    display_name = inp_name or _display_name(resolved_inp)
    engine_name = engine_name or engine.name
    case_dir = work_dir / _safe_name(engine_name) / _safe_name(display_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    copied_inp = _copy_model_tree(
        resolved_inp,
        case_dir / "model",
        excluded_roots=excluded_roots,
    )
    for option_name, value in (
        {"THREADS": "1"} if option_overrides is None else option_overrides
    ).items():
        _set_option(copied_inp, option_name, value)
    execution_cwd = case_dir / "model"

    rpt_candidate = case_dir / "raw.rpt"
    out_candidate = case_dir / "raw.out"
    final_rpt = case_dir / "result.rpt"
    final_out = case_dir / "result.out"

    peak_rss_bytes = 0
    process_done = threading.Event()

    def poll_memory(process: psutil.Process) -> None:
        nonlocal peak_rss_bytes
        while not process_done.is_set():
            try:
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    rss += child.memory_info().rss
                peak_rss_bytes = max(peak_rss_bytes, rss)
            except (psutil.Error, ProcessLookupError):
                if process_done.is_set():
                    break
            time.sleep(0.1)

    stdout = ""
    stderr = ""
    exit_code: int | None = None
    error: str | None = None
    command_duration_s: float | None = None
    memory_thread: threading.Thread | None = None

    try:
        for artifact in (rpt_candidate, out_candidate, final_rpt, final_out):
            if artifact.exists() or artifact.is_symlink():
                if artifact.is_symlink() or not artifact.is_file():
                    raise RuntimeError(f"Expected output artifact file at {artifact}")
                artifact.unlink()
        command_started_at = perf_counter()
        process = subprocess.Popen(
            [str(engine), str(copied_inp), str(rpt_candidate), str(out_candidate)],
            cwd=str(execution_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ps_process = psutil.Process(process.pid)
        memory_thread = threading.Thread(
            target=poll_memory, args=(ps_process,), daemon=True
        )
        memory_thread.start()

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        # pi-lens-ignore: ast-grep:no-bare-except
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            exit_code = process.returncode
            error = f"Timed out after {timeout} seconds"
        command_duration_s = perf_counter() - command_started_at
    except OSError as exc:
        error = str(exc)
    finally:
        process_done.set()
        if memory_thread is not None:
            memory_thread.join(timeout=1.0)

    duration_s: float | None = None
    peak_memory_mb = peak_rss_bytes / (1024 * 1024) if peak_rss_bytes else None
    rpt_path: str | None = None
    out_path: str | None = None

    if rpt_candidate.exists() and rpt_candidate.stat().st_size > 0:
        shutil.copy2(rpt_candidate, final_rpt)
        rpt_path = str(final_rpt.resolve())
        duration_s = _analysis_duration_seconds(final_rpt)
        if duration_s is None and exit_code == 0 and error is None:
            duration_s = command_duration_s
    elif error is None:
        error = "Engine did not produce a non-empty report file"

    if out_candidate.exists() and out_candidate.stat().st_size > 0:
        shutil.copy2(out_candidate, final_out)
        out_path = str(final_out.resolve())

    return EngineResult(
        engine_path=str(engine),
        engine_name=engine_name,
        inp_path=inp_identity or str(resolved_inp),
        inp_name=display_name,
        duration_s=duration_s,
        peak_memory_mb=peak_memory_mb,
        exit_code=exit_code,
        rpt_path=rpt_path,
        stdout=stdout,
        stderr=stderr,
        error=error,
        out_path=out_path,
    )


def _aggregate_results(sample_results: list[EngineResult]) -> EngineResult:
    if not sample_results:
        raise ValueError("Cannot aggregate an empty benchmark sample set")

    successful = [
        (index, result)
        for index, result in enumerate(sample_results)
        if result.exit_code == 0 and result.error is None
    ]
    timed = [
        (index, result)
        for index, result in successful
        if result.duration_s is not None
    ]
    duration_s = median(result.duration_s for _, result in timed) if timed else None
    if duration_s is not None:
        representative_index, representative = min(
            timed,
            key=lambda item: (
                abs(item[1].duration_s - duration_s),  # type: ignore[operator]
                item[0],
            ),
        )
    elif successful:
        representative_index, representative = successful[0]
    else:
        representative_index, representative = 0, sample_results[0]

    memory_samples = [
        result.peak_memory_mb
        for _, result in successful
        if result.peak_memory_mb is not None
    ]
    failures = len(sample_results) - len(successful)
    return replace(
        representative,
        duration_s=duration_s,
        peak_memory_mb=median(memory_samples) if memory_samples else None,
        error=(
            f"{failures} of {len(sample_results)} runs failed"
            if failures
            else None
        ),
        samples=[
            BenchmarkSample(
                duration_s=result.duration_s,
                peak_memory_mb=result.peak_memory_mb,
                exit_code=result.exit_code,
                error=result.error,
            )
            for result in sample_results
        ],
        representative_sample=representative_index + 1,
    )


def run_benchmark(
    engines: list[str],
    inp_files: list[Path],
    work_dir: Path,
    benchmark_name: str,
    timeout: float | None,
    option_overrides: Mapping[str, str] | None = None,
    progress_callback: Callable[[EngineResult], None] | None = None,
    inp_names: dict[Path, str] | None = None,
    inp_identities: dict[Path, str] | None = None,
    run_started_callback: Callable[[str, str], None] | None = None,
    runs: int = 1,
) -> list[EngineResult]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    results: list[EngineResult] = []
    resolved_names = {
        path.expanduser().resolve(): name for path, name in (inp_names or {}).items()
    }
    resolved_identities = {
        path.expanduser().resolve(): identity
        for path, identity in (inp_identities or {}).items()
    }
    labelled_inputs = [
        (
            resolved_names.get(path.expanduser().resolve(), _display_name(path)),
            path,
            resolved_identities.get(path.expanduser().resolve()),
        )
        for path in inp_files
    ]

    benchmark_dir = work_dir / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if runs == 1:
        for engine_path in engines:
            engine_name = Path(engine_path).expanduser().resolve().name
            for display_name, inp_path, inp_identity in labelled_inputs:
                if run_started_callback is not None:
                    run_started_callback(engine_name, display_name)
                result = run_engine(
                    engine_path=engine_path,
                    inp_path=inp_path,
                    work_dir=benchmark_dir,
                    timeout=timeout,
                    inp_name=display_name,
                    inp_identity=inp_identity,
                    option_overrides=option_overrides,
                    excluded_roots=(work_dir,),
                )
                results.append(result)
                if progress_callback is not None:
                    progress_callback(result)
        return results

    sample_root = benchmark_dir / ".samples"
    if sample_root.exists():
        shutil.rmtree(sample_root)
    samples_by_case: dict[tuple[int, int], list[EngineResult]] = {
        (engine_index, input_index): []
        for engine_index in range(len(engines))
        for input_index in range(len(labelled_inputs))
    }

    for input_index, (display_name, inp_path, inp_identity) in enumerate(
        labelled_inputs
    ):
        for sample_index in range(runs):
            first_engine = sample_index % len(engines)
            engine_indices = list(range(first_engine, len(engines))) + list(
                range(first_engine)
            )
            for engine_index in engine_indices:
                engine_path = engines[engine_index]
                engine_name = Path(engine_path).expanduser().resolve().name
                if run_started_callback is not None:
                    run_started_callback(engine_name, display_name)
                result = run_engine(
                    engine_path=engine_path,
                    inp_path=inp_path,
                    work_dir=sample_root / str(sample_index + 1),
                    timeout=timeout,
                    inp_name=display_name,
                    inp_identity=inp_identity,
                    option_overrides=option_overrides,
                    excluded_roots=(work_dir,),
                )
                samples_by_case[(engine_index, input_index)].append(result)
                if progress_callback is not None:
                    progress_callback(result)

    for engine_index, engine_path in enumerate(engines):
        engine_name = Path(engine_path).expanduser().resolve().name
        for input_index, (display_name, _inp_path, _inp_identity) in enumerate(
            labelled_inputs
        ):
            sample_results = samples_by_case[(engine_index, input_index)]
            aggregate = _aggregate_results(sample_results)
            representative_index = aggregate.representative_sample
            assert representative_index is not None
            source_case = (
                sample_root
                / str(representative_index)
                / _safe_name(engine_name)
                / _safe_name(display_name)
            )
            canonical_case = (
                benchmark_dir / _safe_name(engine_name) / _safe_name(display_name)
            )
            if canonical_case.exists():
                shutil.rmtree(canonical_case)
            canonical_case.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_case), canonical_case)

            rpt_path = canonical_case / "result.rpt"
            out_path = canonical_case / "result.out"
            aggregate.rpt_path = str(rpt_path.resolve()) if rpt_path.is_file() else None
            aggregate.out_path = str(out_path.resolve()) if out_path.is_file() else None
            (canonical_case / "samples.json").write_text(
                json.dumps(
                    {
                        "representative_sample": representative_index,
                        "samples": [sample.to_dict() for sample in aggregate.samples],
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            results.append(aggregate)

    shutil.rmtree(sample_root)
    return results
