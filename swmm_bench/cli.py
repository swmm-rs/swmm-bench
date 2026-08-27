from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import typer  # pyright: ignore[reportMissingImports]
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from swmm_bench import __version__
from swmm_bench.comparator import (
    DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
    ComparisonProgress,
    compare_all,
    compare_all_outputs,
)
from swmm_bench.discovery import discover_inp_files
from swmm_bench.interfaces import (
    INTERFACE_FAMILIES,
    InterfaceSelectionError,
    run_interface_suite,
    select_interface_cases,
)
from swmm_bench.models import BenchmarkResult, BenchmarkSample, EngineResult
from swmm_bench.runner import (
    _aggregate_results,
    _analysis_duration_seconds,
    run_benchmark,
)
from swmm_bench.reporter import print_summary, render_html, save_json
from swmm_bench.suite import (
    BENCHMARK_SUITE_NAME,
    REGRESSION_SUITE_NAME,
    SuiteSelectionError,
    catalog_models,
    materialize_models,
    select_models,
)

app = typer.Typer(help="Benchmark SWMM executables.")
test_app = typer.Typer(help="Regression-test SWMM executables.")
console = Console()


def _platform_info() -> dict[str, str]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "host": socket.gethostname(),
    }


def _default_benchmark_name(prefix: str = "bench") -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def _validate_engines(engines: list[str]) -> list[str]:
    validated: list[str] = []
    for raw_engine in engines:
        has_path_component = any(
            separator and separator in raw_engine for separator in (os.sep, os.altsep)
        )
        resolved_engine = raw_engine if has_path_component else shutil.which(raw_engine)
        engine_path = Path(resolved_engine or raw_engine).expanduser().resolve()
        if not engine_path.exists():
            if has_path_component:
                raise typer.BadParameter(f"Engine path does not exist: {engine_path}")
            raise typer.BadParameter(
                f"Engine executable not found on PATH: {raw_engine}"
            )
        if not engine_path.is_file():
            raise typer.BadParameter(f"Engine path is not a file: {engine_path}")
        try:
            is_executable = os.access(engine_path, os.X_OK)
        except OSError as exc:
            raise typer.BadParameter(
                f"Could not inspect engine executable: {engine_path}"
            ) from exc
        if not is_executable:
            raise typer.BadParameter(f"Engine path is not executable: {engine_path}")
        validated.append(str(engine_path))
    return validated


def _parse_inp_option_overrides(
    raw_overrides: list[str] | None,
) -> dict[str, str]:
    overrides = {"THREADS": "1"}
    for raw_override in raw_overrides or ():
        name, separator, value = raw_override.partition("=")
        name = name.strip().upper()
        value = value.strip()
        if (
            not separator
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None
            or not value
            or any(character.isspace() for character in value)
        ):
            raise typer.BadParameter(
                "INP option overrides must use NAME=VALUE with a single-token value."
            )
        overrides[name] = value
    return overrides


def _comparison_pair_count(engine_results: list[EngineResult]) -> int:
    counts: dict[str, int] = defaultdict(int)
    for result in engine_results:
        counts[result.inp_path] += 1
    return sum(count * (count - 1) // 2 for count in counts.values())


def _execute_benchmark(
    *,
    engines: list[str],
    inp_files: list[Path],
    name: str | None,
    default_name_prefix: str,
    output_dir: Path,
    timeout: float,
    html: bool,
    option_overrides: dict[str, str],
    parse_workers: int,
    output_parse_workers: int,
    report_size_mb: int,
    output_typical_weight: float,
    json_out: Path | None,
    inp_names: dict[Path, str] | None = None,
    inp_identities: dict[Path, str] | None = None,
    runs: int = 1,
) -> None:
    benchmark_name = name or _default_benchmark_name(default_name_prefix)
    validated_engines = _validate_engines(engines)
    resolved_output_dir = output_dir.expanduser().resolve()
    benchmark_dir = resolved_output_dir / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        json_out.expanduser().resolve() if json_out else benchmark_dir / "results.json"
    )

    total_runs = len(validated_engines) * len(inp_files) * runs
    run_label = "tests" if default_name_prefix == "regression" else "benchmarks"
    html_path: Path | None = None
    interactive = console.is_interactive
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        disable=not interactive,
    ) as progress:
        run_task = progress.add_task(f"Running {run_label}", total=total_runs)

        def run_started(engine_name: str, inp_name: str) -> None:
            description = f"Running {run_label}: {engine_name} · {inp_name}"
            progress.update(run_task, description=description)
            if interactive:
                progress.refresh()
            else:
                console.print(
                    f"[{int(progress.tasks[run_task].completed) + 1}/{total_runs}] {description}"
                )

        engine_results = run_benchmark(
            engines=validated_engines,
            inp_files=inp_files,
            work_dir=resolved_output_dir,
            benchmark_name=benchmark_name,
            timeout=timeout,
            option_overrides=option_overrides,
            progress_callback=lambda _result: progress.advance(run_task),
            run_started_callback=run_started,
            inp_names=inp_names,
            inp_identities=inp_identities,
            runs=runs,
        )

        pair_total = _comparison_pair_count(engine_results)
        report_task = progress.add_task(
            "Comparing reports",
            total=max(pair_total, 1),
        )
        output_load_task = progress.add_task(
            "Loading binary outputs",
            total=1,
            visible=False,
        )
        output_pair_task = progress.add_task(
            "Comparing binary outputs",
            total=max(pair_total, 1),
        )
        output_detail_task = progress.add_task(
            "Comparing output series",
            total=1,
            visible=False,
        )

        def comparison_progress(event: ComparisonProgress) -> None:
            pair_label = ""
            if event.inp_name and event.engine_a and event.engine_b:
                pair_label = (
                    f": {event.inp_name} ({event.engine_a} vs {event.engine_b})"
                )
            if event.phase == "report":
                description = f"Comparing reports{pair_label}"
                progress.update(
                    report_task,
                    description=description,
                    completed=event.completed,
                    total=max(event.total, 1),
                )
            elif event.phase == "output-load":
                item = Path(event.item_name).name if event.item_name else "output"
                description = f"Loading binary output: {item}"
                progress.update(
                    output_load_task,
                    description=description,
                    completed=event.completed,
                    total=max(event.total, 1),
                    visible=True,
                )
            elif event.phase == "output-pair":
                description = f"Comparing binary outputs{pair_label}"
                progress.update(
                    output_pair_task,
                    description=description,
                    completed=event.completed,
                    total=max(event.total, 1),
                )
            elif event.phase == "output-series":
                description = f"Comparing output series{pair_label}"
                progress.update(
                    output_detail_task,
                    description=description,
                    completed=event.completed,
                    total=max(event.total, 1),
                    visible=True,
                )
            else:
                description = f"Preparing chart data{pair_label}"
                progress.update(
                    output_detail_task,
                    description=description,
                    completed=event.completed,
                    total=max(event.total, 1),
                    visible=True,
                )
            if not interactive and event.status == "started":
                console.print(
                    f"[{event.completed + 1}/{max(event.total, 1)}] {description}"
                )

        comparisons = compare_all(
            engine_results,
            progress_callback=comparison_progress,
            parse_workers=parse_workers,
        )
        if pair_total == 0:
            progress.update(
                report_task,
                description="Comparing reports: no engine pairs",
                completed=1,
            )
            if not interactive:
                console.print("Comparing reports: no engine pairs")

        output_comparisons = compare_all_outputs(
            engine_results,
            progress_callback=comparison_progress,
            parse_workers=output_parse_workers,
            report_size_mb=report_size_mb,
            typical_weight=output_typical_weight,
        )
        if pair_total == 0:
            progress.update(
                output_pair_task,
                description="Comparing binary outputs: no engine pairs",
                completed=1,
            )
            if not interactive:
                console.print("Comparing binary outputs: no engine pairs")

        result = BenchmarkResult(
            schema_version="7" if runs > 1 else "5",
            name=benchmark_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            platform=_platform_info(),
            engine_results=engine_results,
            comparisons=comparisons,
            output_comparisons=output_comparisons,
            run_count=runs if runs > 1 else None,
            run_order="interleaved" if runs > 1 else None,
        )

        json_task = progress.add_task("Writing JSON results", total=1)
        if not interactive:
            console.print("Writing JSON results")
        save_json(result, json_path)
        progress.advance(json_task)

        if html:
            html_path = benchmark_dir / "report.html"
            html_task = progress.add_task("Rendering HTML report", total=1)
            if not interactive:
                console.print("Rendering HTML report")
            render_html(result, html_path)
            progress.advance(html_task)

    print_summary(result)
    console.print(f"Saved JSON results to [bold]{json_path}[/bold]")
    if html_path is not None:
        console.print(f"Saved HTML report to [bold]{html_path}[/bold]")


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", help="Show the application version and exit."
    ),
) -> None:
    if version:
        console.print(f"swmm-bench {__version__}")
        raise typer.Exit()


@test_app.callback()
def test_main(
    version: bool = typer.Option(
        False, "--version", help="Show the application version and exit."
    ),
) -> None:
    if version:
        console.print(f"swmm-test {__version__}")
        raise typer.Exit()


@app.command()
def run(
    engines: list[str] = typer.Argument(..., help="One or more SWMM executable paths."),
    inp: list[str] | None = typer.Option(
        None,
        "--inp",
        help="Model file or directory. Uses bundled stress models when omitted.",
        show_default=False,
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Run one exact bundled category-relative .inp model path.",
    ),
    name: str | None = typer.Option(None, "--name", help="Benchmark name."),
    output_dir: Path = typer.Option(
        Path("swmm-bench-results"),
        "--output-dir",
        help="Directory for benchmark outputs.",
    ),
    inp_option: list[str] | None = typer.Option(
        None,
        "--inp-option",
        help="Override an INP [OPTIONS] value as NAME=VALUE; repeat as needed. Defaults to THREADS=1.",
        show_default=False,
    ),
    runs: int = typer.Option(
        1,
        "--runs",
        min=1,
        help="Measured runs per engine and model; repeated runs are interleaved.",
    ),
    parse_workers: int = typer.Option(
        4,
        "--parse-workers",
        min=1,
        help="Processes used to parse report files.",
    ),
    output_parse_workers: int = typer.Option(
        1,
        "--output-parse-workers",
        min=1,
        help="Processes used to parse binary outputs; higher values use more memory.",
    ),
    report_size_mb: int = typer.Option(
        100,
        "--report-size-mb",
        min=1,
        help="Approximate maximum HTML report size in MB.",
    ),
    output_typical_weight: float = typer.Option(
        DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
        "--output-typical-weight",
        min=0.0,
        max=1.0,
        help="Weight for typical-over-time output distance; the remaining weight scores event differences.",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Recurse into directories looking for models."
    ),
    pattern: str = typer.Option(
        "*.inp", "--pattern", help="Model discovery glob pattern."
    ),
    timeout: float = typer.Option(
        300.0, "--timeout", help="Per-run timeout in seconds."
    ),
    html: bool = typer.Option(
        True, "--html/--no-html", help="Generate an HTML report."
    ),
    json_out: Path | None = typer.Option(
        None, "--json-out", help="Path for the JSON results file."
    ),
) -> None:
    option_overrides = _parse_inp_option_overrides(inp_option)

    if inp and model:
        raise typer.BadParameter("Specify either --inp or --model, not both.")

    if inp:
        inp_files = discover_inp_files(inp, recursive=recursive, pattern=pattern)
        if not inp_files:
            raise typer.BadParameter(
                "No input models found for the supplied --inp paths/pattern."
            )

        _execute_benchmark(
            engines=engines,
            inp_files=inp_files,
            name=name,
            default_name_prefix="bench",
            output_dir=output_dir,
            timeout=timeout,
            option_overrides=option_overrides,
            parse_workers=parse_workers,
            output_parse_workers=output_parse_workers,
            report_size_mb=report_size_mb,
            output_typical_weight=output_typical_weight,
            html=html,
            json_out=json_out,
            runs=runs,
        )
        return

    try:
        selected_models = select_models(
            model=model,
            suite_name=BENCHMARK_SUITE_NAME,
        )
    except SuiteSelectionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    with materialize_models(selected_models) as materialized_models:
        inp_files = [item.inp_path for item in materialized_models]
        _execute_benchmark(
            engines=engines,
            inp_files=inp_files,
            name=name,
            default_name_prefix="bench",
            output_dir=output_dir,
            timeout=timeout,
            option_overrides=option_overrides,
            parse_workers=parse_workers,
            output_parse_workers=output_parse_workers,
            report_size_mb=report_size_mb,
            output_typical_weight=output_typical_weight,
            html=html,
            json_out=json_out,
            runs=runs,
            inp_names={
                item.inp_path: item.model.relative_path for item in materialized_models
            },
            inp_identities={
                item.inp_path: item.model.identity for item in materialized_models
            },
        )


def _rebuild_platform_info(run_directory: Path) -> dict[str, str]:
    unavailable = {"host": "Unavailable", "os": "Unavailable", "python": "Unavailable"}
    for filename in ("results.json", "report.json"):
        try:
            data = json.loads((run_directory / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        saved = data.get("platform")
        if isinstance(saved, dict):
            platform_info = unavailable.copy()
            for key in platform_info:
                value = saved.get(key)
                if isinstance(value, str) and value:
                    platform_info[key] = value
            return platform_info
    return unavailable


def _rebuild_engine_results(run_directory: Path) -> list[EngineResult]:
    engine_results = []
    for engine_directory in sorted(
        path for path in run_directory.iterdir() if path.is_dir()
    ):
        for case_directory in sorted(
            path
            for path in engine_directory.iterdir()
            if path.is_dir() and (path / "model").is_dir()
        ):
            rpt = case_directory / "result.rpt"
            out = case_directory / "result.out"
            rpt_path = (
                str(rpt.resolve()) if rpt.is_file() and rpt.stat().st_size else None
            )
            out_path = (
                str(out.resolve()) if out.is_file() and out.stat().st_size else None
            )
            result = EngineResult(
                engine_path=engine_directory.name,
                engine_name=engine_directory.name,
                inp_path=case_directory.name,
                inp_name=case_directory.name,
                duration_s=_analysis_duration_seconds(rpt) if rpt_path else None,
                peak_memory_mb=None,
                exit_code=None,
                rpt_path=rpt_path,
                stdout="",
                stderr="",
                error=None
                if rpt_path
                else "Saved run did not produce a non-empty report file",
                out_path=out_path,
            )
            manifest_path = case_directory / "samples.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    samples = [
                        BenchmarkSample.from_dict(item)
                        for item in manifest["samples"]
                    ]
                    if not samples:
                        raise ValueError("sample list is empty")
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise typer.BadParameter(
                        f"Could not read benchmark samples from {manifest_path}: {exc}"
                    ) from exc
                result = _aggregate_results(
                    [
                        EngineResult(
                            engine_path=result.engine_path,
                            engine_name=result.engine_name,
                            inp_path=result.inp_path,
                            inp_name=result.inp_name,
                            duration_s=sample.duration_s,
                            peak_memory_mb=sample.peak_memory_mb,
                            exit_code=sample.exit_code,
                            rpt_path=rpt_path,
                            stdout="",
                            stderr="",
                            error=sample.error,
                            out_path=out_path,
                        )
                        for sample in samples
                    ]
                )
                result.rpt_path = rpt_path
                result.out_path = out_path
            engine_results.append(result)
    return engine_results


@app.command()
def rebuild(
    run_directory: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=False,
        readable=True,
        help="Prior swmm-bench result directory; defaults to the current directory.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path for rebuilt report JSON."
    ),
    outputs: bool = typer.Option(
        False,
        "--outputs",
        help="Recalculate binary-output comparisons with detailed chart data; this can be slow.",
    ),
    all_comparisons: bool = typer.Option(
        False,
        "--all-comparisons",
        help="Retain output chart data at every distance; requires --outputs.",
    ),
    parse_workers: int = typer.Option(
        4,
        "--parse-workers",
        min=1,
        help="Processes used to parse report files.",
    ),
    output_parse_workers: int = typer.Option(
        1,
        "--output-parse-workers",
        min=1,
        help="Processes used to parse binary outputs; higher values use more memory.",
    ),
    report_size_mb: int = typer.Option(
        100,
        "--report-size-mb",
        min=1,
        help="Approximate eventual HTML report size in MB.",
    ),
    output_typical_weight: float = typer.Option(
        DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
        "--output-typical-weight",
        min=0.0,
        max=1.0,
        help="Weight for typical-over-time output distance; the remaining weight scores event differences.",
    ),
) -> None:
    run_directory = run_directory.expanduser().resolve()
    if all_comparisons and not outputs:
        raise typer.BadParameter("--all-comparisons requires --outputs.")
    engine_results = _rebuild_engine_results(run_directory)
    if not engine_results:
        raise typer.BadParameter(
            "No saved engine case directories containing a model/ directory were found."
        )
    pair_total = _comparison_pair_count(engine_results)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        report_task = progress.add_task("Comparing reports", total=max(pair_total, 1))
        output_load_task = progress.add_task(
            "Loading binary outputs", total=1, visible=False
        )
        output_pair_task = progress.add_task(
            "Comparing binary outputs", total=max(pair_total, 1), visible=outputs
        )
        output_detail_task = progress.add_task(
            "Comparing output series", total=1, visible=False
        )

        def comparison_progress(event: ComparisonProgress) -> None:
            pair_label = ""
            if event.inp_name and event.engine_a and event.engine_b:
                pair_label = (
                    f": {event.inp_name} ({event.engine_a} vs {event.engine_b})"
                )
            if event.phase == "report":
                progress.update(
                    report_task,
                    description=f"Comparing reports{pair_label}",
                    completed=event.completed,
                    total=max(event.total, 1),
                )
            elif outputs and event.phase == "output-load":
                item = Path(event.item_name).name if event.item_name else "output"
                progress.update(
                    output_load_task,
                    description=f"Loading binary output: {item}",
                    completed=event.completed,
                    total=max(event.total, 1),
                    visible=True,
                )
            elif outputs and event.phase == "output-pair":
                progress.update(
                    output_pair_task,
                    description=f"Comparing binary outputs{pair_label}",
                    completed=event.completed,
                    total=max(event.total, 1),
                )
            elif outputs and event.phase == "output-series":
                progress.update(
                    output_detail_task,
                    description=f"Comparing output series{pair_label}",
                    completed=event.completed,
                    total=max(event.total, 1),
                    visible=True,
                )

        comparisons = compare_all(
            engine_results,
            progress_callback=comparison_progress,
            parse_workers=parse_workers,
        )
        if pair_total == 0:
            progress.update(
                report_task,
                description="Comparing reports: no engine pairs",
                completed=1,
            )

        output_comparisons = []
        if outputs:
            output_comparisons = compare_all_outputs(
                engine_results,
                progress_callback=comparison_progress,
                include_all_comparisons=all_comparisons,
                parse_workers=output_parse_workers,
                report_size_mb=report_size_mb,
                typical_weight=output_typical_weight,
            )
            if pair_total == 0:
                progress.update(
                    output_pair_task,
                    description="Comparing binary outputs: no engine pairs",
                    completed=1,
                )

        run_count = max((len(result.samples) for result in engine_results), default=0)
        repeated = run_count > 1
        result = BenchmarkResult(
            schema_version="7" if repeated else "5",
            name=run_directory.name,
            timestamp=datetime.fromtimestamp(
                run_directory.stat().st_mtime, timezone.utc
            ).isoformat(),
            platform=_rebuild_platform_info(run_directory),
            engine_results=engine_results,
            comparisons=comparisons,
            output_comparisons=output_comparisons,
            run_count=run_count if repeated else None,
            run_order="interleaved" if repeated else None,
        )
        json_path = (
            output.expanduser().resolve() if output else run_directory / "report.json"
        )
        json_task = progress.add_task("Writing JSON results", total=1)
        save_json(result, json_path)
        progress.advance(json_task)
    console.print(f"Saved JSON results to [bold]{json_path}[/bold]")


@test_app.command("list")
def list_suite() -> None:
    grouped_models: dict[str, list[str]] = defaultdict(list)
    for model in catalog_models(REGRESSION_SUITE_NAME):
        grouped_models[model.category].append(model.relative_path)

    console.print("[bold]Bundled regression suite[/bold]")
    for category, model_paths in grouped_models.items():
        console.print(f"\n[bold]{category}[/bold] ({len(model_paths)})")
        for model_path in model_paths:
            console.print(f"  {model_path}")

    console.print(
        "\nRun the suite with one or more compatible SWMM executables "
        "(ENGINE input.inp report.rpt output.out):"
    )
    console.print("  swmm-test run /path/to/swmm")
    console.print(
        "  swmm-test run /path/to/swmm-a /path/to/swmm-b --category hydrology"
    )
    console.print(
        "  swmm-test run /path/to/swmm-a /path/to/swmm-b "
        "--model water-quality/waterquality-events_example.inp"
    )


@test_app.command("run")
def run_suite(
    engines: list[str] = typer.Argument(..., help="One or more SWMM executable paths."),
    category: str | None = typer.Option(
        None, "--category", help="Run every model in this suite category."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Run one exact category-relative .inp model path."
    ),
    name: str | None = typer.Option(None, "--name", help="Regression run name."),
    output_dir: Path = typer.Option(
        Path("swmm-test-results"),
        "--output-dir",
        help="Directory for regression outputs.",
    ),
    inp_option: list[str] | None = typer.Option(
        None,
        "--inp-option",
        help="Override an INP [OPTIONS] value as NAME=VALUE; repeat as needed. Defaults to THREADS=1.",
        show_default=False,
    ),
    parse_workers: int = typer.Option(
        4,
        "--parse-workers",
        min=1,
        help="Processes used to parse report files.",
    ),
    output_parse_workers: int = typer.Option(
        1,
        "--output-parse-workers",
        min=1,
        help="Processes used to parse binary outputs; higher values use more memory.",
    ),
    report_size_mb: int = typer.Option(
        100,
        "--report-size-mb",
        min=1,
        help="Approximate maximum HTML report size in MB.",
    ),
    output_typical_weight: float = typer.Option(
        DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
        "--output-typical-weight",
        min=0.0,
        max=1.0,
        help="Weight for typical-over-time output distance; the remaining weight scores event differences.",
    ),
    timeout: float = typer.Option(
        300.0, "--timeout", help="Per-run timeout in seconds."
    ),
    html: bool = typer.Option(
        True, "--html/--no-html", help="Generate an HTML report."
    ),
    json_out: Path | None = typer.Option(
        None, "--json-out", help="Path for the JSON results file."
    ),
) -> None:
    option_overrides = _parse_inp_option_overrides(inp_option)

    try:
        selected_models = select_models(
            category=category,
            model=model,
            suite_name=REGRESSION_SUITE_NAME,
        )
    except SuiteSelectionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    with materialize_models(selected_models) as materialized_models:
        inp_files = [item.inp_path for item in materialized_models]
        inp_names = {
            item.inp_path: item.model.relative_path for item in materialized_models
        }
        inp_identities = {
            item.inp_path: item.model.identity for item in materialized_models
        }
        _execute_benchmark(
            engines=engines,
            inp_files=inp_files,
            name=name,
            default_name_prefix="regression",
            output_dir=output_dir,
            timeout=timeout,
            option_overrides=option_overrides,
            parse_workers=parse_workers,
            output_parse_workers=output_parse_workers,
            report_size_mb=report_size_mb,
            output_typical_weight=output_typical_weight,
            html=html,
            json_out=json_out,
            inp_names=inp_names,
            inp_identities=inp_identities,
        )


@test_app.command("interface")
def run_interfaces(
    source: str = typer.Argument(
        ..., help="SWMM executable that generates interfaces."
    ),
    targets: list[str] = typer.Argument(
        ...,
        help="One or more SWMM executables that consume the generated interfaces.",
    ),
    family: list[str] | None = typer.Option(
        None,
        "--family",
        help=f"Interface family to run; repeat as needed. Valid: {', '.join(INTERFACE_FAMILIES)}.",
    ),
    name: str | None = typer.Option(None, "--name", help="Interface run name."),
    output_dir: Path = typer.Option(
        Path("swmm-test-results"),
        "--output-dir",
        help="Directory for interface test outputs.",
    ),
    inp_option: list[str] | None = typer.Option(
        None,
        "--inp-option",
        help="Override an INP [OPTIONS] value as NAME=VALUE; repeat as needed. Defaults to THREADS=1.",
        show_default=False,
    ),
    parse_workers: int = typer.Option(
        4,
        "--parse-workers",
        min=1,
        help="Processes used to parse report files.",
    ),
    output_parse_workers: int = typer.Option(
        1,
        "--output-parse-workers",
        min=1,
        help="Processes used to parse binary outputs; higher values use more memory.",
    ),
    report_size_mb: int = typer.Option(
        100,
        "--report-size-mb",
        min=1,
        help="Approximate maximum HTML report size in MB.",
    ),
    output_typical_weight: float = typer.Option(
        DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
        "--output-typical-weight",
        min=0.0,
        max=1.0,
        help="Weight for typical-over-time output distance; the remaining weight scores event differences.",
    ),
    timeout: float = typer.Option(
        300.0,
        "--timeout",
        help="Per-run timeout in seconds.",
    ),
    html: bool = typer.Option(
        True,
        "--html/--no-html",
        help="Generate an HTML report.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Path for the JSON results file.",
    ),
) -> None:
    option_overrides = _parse_inp_option_overrides(inp_option)

    try:
        selected_cases = select_interface_cases(family)
    except InterfaceSelectionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    validated_source = _validate_engines([source])[0]
    validated_targets = _validate_engines(targets)
    run_name = name or _default_benchmark_name("interface")
    result_dir = output_dir.expanduser().resolve() / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(selected_cases) * (1 + 2 * len(validated_targets))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running interface pipelines", total=total_runs)

        def run_started(engine_name: str, inp_name: str) -> None:
            progress.update(
                task,
                description=f"Running interfaces: {engine_name} · {inp_name}",
            )
            progress.refresh()

        result = run_interface_suite(
            source_engine=validated_source,
            target_engines=validated_targets,
            families=[case.family for case in selected_cases],
            work_dir=result_dir,
            name=run_name,
            timeout=timeout,
            option_overrides=option_overrides,
            parse_workers=parse_workers,
            output_workers=output_parse_workers,
            report_target_mb=report_size_mb,
            output_typical_weight=output_typical_weight,
            platform=_platform_info(),
            progress_callback=lambda _result: progress.advance(task),
            run_started_callback=run_started,
        )
        actual_runs = len(result.engine_results)
        progress.update(task, completed=actual_runs, total=max(actual_runs, 1))

    json_path = (
        json_out.expanduser().resolve() if json_out else result_dir / "results.json"
    )
    save_json(result, json_path)
    html_path: Path | None = None
    if html:
        html_path = result_dir / "report.html"
        render_html(result, html_path)

    print_summary(result)
    console.print(f"Saved JSON results to [bold]{json_path}[/bold]")
    if html_path is not None:
        console.print(f"Saved HTML report to [bold]{html_path}[/bold]")
    if any(
        row.exit_code != 0 or row.error is not None for row in result.engine_results
    ):
        raise typer.Exit(1)


@app.command()
def report(
    results_json: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a saved results.json file.",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Path for the generated HTML report."
    ),
    engine_order: list[str] | None = typer.Option(
        None,
        "--engine-order",
        help="Place this engine before unlisted engines in report charts, tables, and distance controls. Repeat to specify multiple engines.",
    ),
    open_report: bool = typer.Option(
        False, "--open", help="Open the generated report in the default browser."
    ),
) -> None:
    try:
        data = json.loads(
            results_json.expanduser().resolve().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Could not read results JSON: {exc}") from exc
    result = BenchmarkResult.from_dict(data)
    html_path = (
        output.expanduser().resolve()
        if output
        else results_json.expanduser().resolve().with_name("report.html")
    )
    try:
        render_html(result, html_path, engine_order=engine_order)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--engine-order") from exc
    console.print(f"Saved HTML report to [bold]{html_path}[/bold]")

    if open_report:
        webbrowser.open(html_path.as_uri())
