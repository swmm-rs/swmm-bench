from __future__ import annotations

import json
import itertools
from math import inf
from pathlib import Path
from statistics import mean
from typing import Any

from jinja2 import Environment, PackageLoader  # pyright: ignore[reportMissingImports]
from rich.console import Console
from rich.table import Table

from swmm_bench.models import (
    BenchmarkResult,
    OUTPUT_DISTANCE_METRIC_COMPOSITE,
    EngineResult,
    ModelComparison,
    OUTPUT_DISTANCE_METRIC_LEGACY,
    OUTPUT_DISTANCE_METRIC_NRMSE,
    OUTPUT_DISTANCE_METRIC_NRMSE_V1,
    OutputComparison,
)

console = Console()

_HTML_COMPARISON_DISTANCE_THRESHOLD = 0.01
_HTML_GRAPHICAL_SERIES_LIMIT = 5_000


def _distance_style(distance: float) -> str:
    if distance <= 0.03:
        return "green"
    if distance <= 0.1:
        return "yellow"
    return "red"


def _output_metric_info(metric: str) -> dict[str, str]:
    if metric == OUTPUT_DISTANCE_METRIC_COMPOSITE:
        return {
            "id": metric,
            "kind": "composite",
            "label": "Typical/event composite + missing penalty",
            "short_label": "Composite distance",
        }
    if metric == OUTPUT_DISTANCE_METRIC_NRMSE:
        return {
            "id": metric,
            "kind": "nrmse",
            "label": "Symmetric NRMSE + missing penalty",
            "short_label": "NRMSE + missing",
        }
    if metric == OUTPUT_DISTANCE_METRIC_NRMSE_V1:
        return {
            "id": metric,
            "kind": "nrmse-v1",
            "label": "Symmetric NRMSE + all-timestamp missing penalty",
            "short_label": "NRMSE + all missing",
        }
    if metric == OUTPUT_DISTANCE_METRIC_LEGACY:
        return {
            "id": metric,
            "kind": "legacy",
            "label": "Legacy pointwise relative distance",
            "short_label": "Legacy distance",
        }
    return {
        "id": metric,
        "kind": "unknown",
        "label": f"Unknown output metric ({metric})",
        "short_label": "Unknown metric",
    }


def _output_timeline_summary(comparison: OutputComparison) -> str | None:
    coverage = comparison.timeline_coverage
    shared = coverage.shared_timestamp_count
    if shared is None:
        return None

    trailing = []
    for engine, count in (
        (comparison.engine_a, coverage.trailing_timestamp_count_a),
        (comparison.engine_b, coverage.trailing_timestamp_count_b),
    ):
        if count:
            suffix = "timestamp" if count == 1 else "timestamps"
            trailing.append(f"{engine} has {count} additional trailing {suffix}")
    if not trailing:
        return None

    shared_suffix = "timestamp" if shared == 1 else "timestamps"
    return (
        f"{shared} shared {shared_suffix}; {'; '.join(trailing)}. "
        "Trailing-only timestamps do not affect value distance."
    )


def _uses_report_analysis_duration(result: BenchmarkResult) -> bool:
    try:
        return int(result.schema_version) >= 5
    except ValueError:
        return False


def print_summary(result: BenchmarkResult) -> None:
    uses_analysis_duration = _uses_report_analysis_duration(result)
    duration_name = "Analysis" if uses_analysis_duration else "Runtime"
    if result.run_count and result.run_count > 1:
        duration_name = f"Median {duration_name.lower()}"
    performance_table = Table(title=f"Benchmark: {result.name}")
    performance_table.add_column("Engine")
    performance_table.add_column("Model")
    performance_table.add_column(f"{duration_name} (s)", justify="right")
    performance_table.add_column("Peak RSS (MB)", justify="right")
    performance_table.add_column("Exit")
    performance_table.add_column("Status")

    for engine_result in result.engine_results:
        status = engine_result.error or "ok"
        performance_table.add_row(
            engine_result.engine_name,
            engine_result.inp_name,
            (
                "-"
                if engine_result.duration_s is None
                else f"{engine_result.duration_s:.3f}"
            ),
            (
                "-"
                if engine_result.peak_memory_mb is None
                else f"{engine_result.peak_memory_mb:.2f}"
            ),
            "-" if engine_result.exit_code is None else str(engine_result.exit_code),
            status,
        )

    console.print(performance_table)

    by_model: dict[str, list[Any]] = {}
    for engine_result in result.engine_results:
        by_model.setdefault(engine_result.inp_name, []).append(engine_result)

    timing_kind = "analysis timing" if uses_analysis_duration else "runtime"
    speed_table = Table(title=f"Model {timing_kind} highlights")
    speed_table.add_column("Model")
    speed_table.add_column("Fastest")
    speed_table.add_column("Slowest")

    for model_name, rows in sorted(by_model.items()):
        completed = [
            row
            for row in rows
            if row.exit_code == 0 and row.error is None and row.duration_s is not None
        ]
        if not completed:
            continue
        fastest = min(
            completed, key=lambda row: row.duration_s if row.duration_s is not None else inf
        )
        slowest = max(completed, key=lambda row: row.duration_s or 0.0)
        speed_table.add_row(
            model_name,
            f"{fastest.engine_name} ({fastest.duration_s:.3f}s)",
            f"{slowest.engine_name} ({slowest.duration_s:.3f}s)",
        )

    console.print(speed_table)

    comparison_table = Table(title="Pairwise report distances")
    comparison_table.add_column("Model")
    comparison_table.add_column("Engines")
    comparison_table.add_column("Distance", justify="right")

    if result.comparisons:
        for comparison in result.comparisons:
            comparison_table.add_row(
                comparison.inp_name,
                f"{comparison.engine_a} vs {comparison.engine_b}",
                f"[{_distance_style(comparison.overall_distance)}]{comparison.overall_distance:.4f}[/{_distance_style(comparison.overall_distance)}]",
            )
    else:
        comparison_table.add_row("-", "-", "No comparable reports")

    console.print(comparison_table)

    output_comparison_table = Table(title="Pairwise output distances")
    output_comparison_table.add_column("Model")
    output_comparison_table.add_column("Engines")
    output_comparison_table.add_column("Metric")
    output_comparison_table.add_column("Distance", justify="right")

    if result.output_comparisons:
        for comparison in result.output_comparisons:
            metric_info = _output_metric_info(comparison.metric)
            output_comparison_table.add_row(
                comparison.inp_name,
                f"{comparison.engine_a} vs {comparison.engine_b}",
                metric_info["short_label"],
                f"[{_distance_style(comparison.overall_distance)}]{comparison.overall_distance:.4f}[/{_distance_style(comparison.overall_distance)}]",
            )
    else:
        output_comparison_table.add_row("-", "-", "-", "No comparable outputs")

    console.print(output_comparison_table)


def save_json(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _engine_pair_reason(
    result_a: EngineResult, result_b: EngineResult, artifact_name: str
) -> str | None:
    for result in (result_a, result_b):
        if result.exit_code not in (0, None):
            return f"engine {result.engine_name} did not complete simulation"
        artifact = result.rpt_path if artifact_name == "report" else result.out_path
        if not artifact:
            return f"engine {result.engine_name} did not produce {artifact_name}"
        if result.error:
            return result.error
    return None


def _failed_model_reason(result_a: EngineResult, result_b: EngineResult) -> str | None:
    failed_engines = [
        result.engine_name
        for result in (result_a, result_b)
        if result.exit_code not in (0, None) or result.error
    ]
    if not failed_engines:
        return None
    if len(failed_engines) == 1:
        return f"Comparison cannot be made: {failed_engines[0]} model failed."
    return f"Comparison cannot be made: {' and '.join(failed_engines)} models failed."


def _report_error_reason(comparison: ModelComparison) -> str:
    failed_engines = [
        engine
        for engine, errors in (
            (comparison.engine_a, comparison.report_errors_a),
            (comparison.engine_b, comparison.report_errors_b),
        )
        if errors
    ]
    if len(failed_engines) == 1:
        return f"Comparison cannot be made: {failed_engines[0]} model failed."
    if len(failed_engines) > 1:
        return (
            f"Comparison cannot be made: {' and '.join(failed_engines)} models failed."
        )
    return "Comparison cannot be made: report errors indicate a model failed."


def _comparison_pairs(
    result: BenchmarkResult,
) -> list[tuple[str, EngineResult, EngineResult]]:
    grouped: dict[str, list[EngineResult]] = {}
    for engine_result in result.engine_results:
        grouped.setdefault(engine_result.inp_path, []).append(engine_result)
    return [
        (inp_path, result_a, result_b)
        for inp_path, rows in grouped.items()
        for result_a, result_b in itertools.combinations(rows, 2)
    ]


def _comparison_key(comparison: ModelComparison) -> tuple[str, str, str]:
    return (comparison.inp_path, comparison.engine_a, comparison.engine_b)


def _below_threshold_reason(distance: float) -> str:
    return (
        f"{distance:.6f} hidden as matching "
        f"(≤{_HTML_COMPARISON_DISTANCE_THRESHOLD:.6f})"
    )


def _output_placeholder_reason(comparison: OutputComparison) -> str | None:
    if (
        comparison.overall_distance > _HTML_COMPARISON_DISTANCE_THRESHOLD
        or comparison.graphical_series
    ):
        return None
    unavailable_reason = comparison.graphical_unavailable_reason
    if unavailable_reason and "overall distance did not exceed" not in unavailable_reason:
        return None
    return _below_threshold_reason(comparison.overall_distance)


def _failed_placeholder(reason: str | None) -> bool:
    return bool(reason and "hidden as matching" not in reason)


def _view_severity(comparison: ModelComparison, placeholder_reason: str | None) -> str:
    if comparison.report_errors or _failed_placeholder(placeholder_reason):
        return "failed"
    if comparison.overall_distance <= 0.01:
        return "green"
    if comparison.overall_distance <= 0.1:
        return "yellow"
    return "red"


def _diagnostic_rows(
    comparison: ModelComparison,
    rpt_path_a: str | None = None,
    rpt_path_b: str | None = None,
) -> list[dict[str, str | None]]:
    def compact_legacy(message: str, kind: str) -> str:
        if not message.startswith("Report "):
            return message
        for marker in (f" {kind.lower()} ", f" {kind.lower()}s "):
            if marker in message:
                return f"{kind.upper()} {message.rsplit(marker, 1)[1]}"
        if " table " in message:
            return f"{kind.upper()}: table {message.rsplit(' table ', 1)[1]}"
        return message

    def split_legacy(messages: list[str], kind: str) -> tuple[list[str], list[str]]:
        messages_a, messages_b = [], []
        for message in messages:
            compact_message = compact_legacy(message, kind)
            if rpt_path_a and message.startswith(f"Report {rpt_path_a} "):
                messages_a.append(compact_message)
            elif rpt_path_b and message.startswith(f"Report {rpt_path_b} "):
                messages_b.append(compact_message)
            else:
                messages_a.append(compact_message)
        return messages_a, messages_b

    errors_a, errors_b = (
        (comparison.report_errors_a, comparison.report_errors_b)
        if comparison.report_errors_a or comparison.report_errors_b
        else split_legacy(comparison.report_errors, "Error")
    )
    warnings_a, warnings_b = (
        (comparison.report_warnings_a, comparison.report_warnings_b)
        if comparison.report_warnings_a or comparison.report_warnings_b
        else split_legacy(comparison.report_warnings, "Warning")
    )
    rows = []
    for kind, messages_a, messages_b in (
        ("Error", errors_a, errors_b),
        ("Warning", warnings_a, warnings_b),
    ):
        rows.extend(
            {
                "kind": kind,
                "message_a": message_a,
                "message_b": message_b,
            }
            for message_a, message_b in itertools.zip_longest(messages_a, messages_b)
        )
    return rows


def _prioritize_engines(
    engines: list[str],
    engine_order: list[str] | None,
    *,
    available_engines: list[str] | None = None,
) -> list[str]:
    requested = engine_order or []
    duplicates = [
        engine for index, engine in enumerate(requested) if engine in requested[:index]
    ]
    if duplicates:
        raise ValueError(
            f"Engine order contains duplicate engine: {duplicates[0]!r}."
        )

    known_engines = available_engines if available_engines is not None else engines
    unknown = [engine for engine in requested if engine not in known_engines]
    if unknown:
        available = ", ".join(known_engines) or "none"
        raise ValueError(
            f"Unknown engine in engine order: {unknown[0]!r}. "
            f"Available engines: {available}."
        )

    requested_set = set(requested)
    return [
        *(engine for engine in requested if engine in engines),
        *(engine for engine in engines if engine not in requested_set),
    ]


def _build_template_context(
    result: BenchmarkResult, *, engine_order: list[str] | None = None
) -> dict[str, Any]:
    uses_analysis_duration = _uses_report_analysis_duration(result)
    duration_name = "analysis duration" if uses_analysis_duration else "runtime"
    all_engines = list(
        dict.fromkeys(
            itertools.chain(
                (row.engine_name for row in result.engine_results),
                (
                    engine
                    for comparison in itertools.chain(
                        result.comparisons, result.output_comparisons
                    )
                    for engine in (comparison.engine_a, comparison.engine_b)
                ),
            )
        )
    )
    engines = _prioritize_engines(
        list(dict.fromkeys(row.engine_name for row in result.engine_results)),
        engine_order,
        available_engines=all_engines,
    )
    models = sorted({row.inp_name for row in result.engine_results})
    matrix: dict[str, dict[str, dict[str, Any]]] = {engine: {} for engine in engines}

    for row in result.engine_results:
        successful_samples = [
            sample
            for sample in row.samples
            if sample.exit_code == 0
            and sample.error is None
            and sample.duration_s is not None
        ]
        sample_durations = [
            sample.duration_s for sample in successful_samples if sample.duration_s is not None
        ]
        matrix.setdefault(row.engine_name, {})[row.inp_name] = {
            "duration_s": row.duration_s,
            "peak_memory_mb": row.peak_memory_mb,
            "exit_code": row.exit_code,
            "error": row.error,
            "sample_count": len(row.samples),
            "successful_sample_count": len(successful_samples),
            "minimum_duration_s": min(sample_durations) if sample_durations else None,
            "maximum_duration_s": max(sample_durations) if sample_durations else None,
        }

    average_durations = []
    for engine in engines:
        durations = [
            cell["duration_s"]
            for cell in matrix.get(engine, {}).values()
            if cell.get("duration_s") is not None
        ]
        average_durations.append(round(mean(durations), 6) if durations else 0.0)

    pairs = _comparison_pairs(result)
    report_by_key = {
        _comparison_key(comparison): comparison for comparison in result.comparisons
    }
    output_by_key = {
        _comparison_key(comparison): comparison
        for comparison in result.output_comparisons
    }
    report_paths = {
        (row.inp_path, row.engine_name): row.rpt_path for row in result.engine_results
    }

    comparison_rows: list[dict[str, Any]] = []
    seen_reports: set[tuple[str, str, str]] = set()
    for inp_path, result_a, result_b in pairs:
        key = (inp_path, result_a.engine_name, result_b.engine_name)
        comparison = report_by_key.get(key)
        placeholder_reason = None
        run_failure_reason = _failed_model_reason(result_a, result_b)
        if comparison is None:
            placeholder_reason = run_failure_reason or _engine_pair_reason(
                result_a, result_b, "report"
            )
            if placeholder_reason is None:
                continue
            comparison = ModelComparison(
                inp_path=inp_path,
                inp_name=result_a.inp_name,
                engine_a=result_a.engine_name,
                engine_b=result_b.engine_name,
                overall_distance=0.0,
                section_comparisons=[],
            )
        elif run_failure_reason:
            placeholder_reason = run_failure_reason
        elif comparison.report_errors:
            placeholder_reason = _report_error_reason(comparison)
        elif (
            comparison.overall_distance <= _HTML_COMPARISON_DISTANCE_THRESHOLD
            and not comparison.report_warnings
        ):
            placeholder_reason = _below_threshold_reason(comparison.overall_distance)
        comparison_rows.append(
            {
                "comparison": comparison,
                "placeholder_reason": placeholder_reason,
                "severity": _view_severity(comparison, placeholder_reason),
                "diagnostic_rows": _diagnostic_rows(
                    comparison, result_a.rpt_path, result_b.rpt_path
                ),
            }
        )
        seen_reports.add(key)
    for comparison in result.comparisons:
        key = _comparison_key(comparison)
        if key in seen_reports:
            continue
        if comparison.report_errors:
            placeholder_reason = _report_error_reason(comparison)
        else:
            placeholder_reason = (
                _below_threshold_reason(comparison.overall_distance)
                if comparison.overall_distance <= _HTML_COMPARISON_DISTANCE_THRESHOLD
                and not comparison.report_warnings
                else None
            )
        comparison_rows.append(
            {
                "comparison": comparison,
                "placeholder_reason": placeholder_reason,
                "severity": _view_severity(comparison, placeholder_reason),
                "diagnostic_rows": _diagnostic_rows(
                    comparison,
                    report_paths.get((comparison.inp_path, comparison.engine_a)),
                    report_paths.get((comparison.inp_path, comparison.engine_b)),
                ),
            }
        )
    comparison_rows.sort(
        key=lambda item: (
            item["severity"] == "failed",
            item["placeholder_reason"] is None,
            item["comparison"].overall_distance,
        ),
        reverse=True,
    )
    for index, item in enumerate(comparison_rows):
        item["dom_id"] = f"report-comparison-{index}"

    output_comparison_rows: list[dict[str, Any]] = []
    seen_outputs: set[tuple[str, str, str]] = set()
    for inp_path, result_a, result_b in pairs:
        key = (inp_path, result_a.engine_name, result_b.engine_name)
        comparison = output_by_key.get(key)
        placeholder_reason = None
        if comparison is None:
            placeholder_reason = _engine_pair_reason(result_a, result_b, "output")
            if placeholder_reason is None:
                continue
            comparison = OutputComparison(
                inp_path=inp_path,
                inp_name=result_a.inp_name,
                engine_a=result_a.engine_name,
                engine_b=result_b.engine_name,
                overall_distance=0.0,
                section_comparisons=[],
            )
        else:
            placeholder_reason = _output_placeholder_reason(comparison)
        output_comparison_rows.append(
            {
                "comparison": comparison,
                "placeholder_reason": placeholder_reason,
                "severity": _view_severity(comparison, placeholder_reason),
            }
        )
        seen_outputs.add(key)
    for comparison in result.output_comparisons:
        key = _comparison_key(comparison)
        if key in seen_outputs:
            continue
        placeholder_reason = _output_placeholder_reason(comparison)
        output_comparison_rows.append(
            {
                "comparison": comparison,
                "placeholder_reason": placeholder_reason,
                "severity": _view_severity(comparison, placeholder_reason),
            }
        )
    output_comparison_rows.sort(
        key=lambda item: (
            item["severity"] == "failed",
            item["placeholder_reason"] is None,
            item["comparison"].overall_distance,
        ),
        reverse=True,
    )

    def graph_payload(comparison: Any) -> tuple[list[dict[str, Any]], str | None]:
        saved_series = getattr(comparison, "graphical_series", [])
        series = sorted(
            (
                item
                for item in saved_series
                if not item.sampled
                and len(item.timestamps) == item.source_point_count
            ),
            key=lambda item: item.distance,
            reverse=True,
        )
        note = getattr(comparison, "graphical_unavailable_reason", None)
        omitted_sampled_count = len(saved_series) - len(series)
        if omitted_sampled_count:
            omission_note = (
                "Legacy sampled chart data was omitted. Rebuild output comparisons "
                "to retain complete time series."
            )
            note = f"{note} {omission_note}" if note else omission_note
        if len(series) > _HTML_GRAPHICAL_SERIES_LIMIT:
            truncation_note = (
                f"Showing the {_HTML_GRAPHICAL_SERIES_LIMIT} highest-distance graph series "
                "in HTML. The JSON results retain every series."
            )
            note = f"{note} {truncation_note}" if note else truncation_note
            series = series[:_HTML_GRAPHICAL_SERIES_LIMIT]
        return [item.to_dict() for item in series], note

    output_drilldowns = []
    for index, item in enumerate(output_comparison_rows):
        comparison = item["comparison"]
        if item["placeholder_reason"]:
            graphical_series, graphical_note = [], None
        else:
            graphical_series, graphical_note = graph_payload(comparison)
        output_drilldowns.append(
            {
                "dom_id": f"output-comparison-{index}",
                "comparison": comparison,
                "placeholder_reason": item["placeholder_reason"],
                "severity": item["severity"],
                "metric_info": _output_metric_info(comparison.metric),
                "timeline_summary": _output_timeline_summary(comparison),
                "details_retained": getattr(comparison, "details_retained", True),
                "graphical_series": graphical_series,
                "graphical_note": graphical_note,
                "graphical_unavailable_reason": (
                    getattr(comparison, "graphical_unavailable_reason", None)
                    if graphical_series
                    else graphical_note
                    or getattr(comparison, "graphical_unavailable_reason", None)
                ),
            }
        )
    issue_queue = []
    for item in comparison_rows:
        comparison = item["comparison"]
        if (
            item["severity"] == "green"
            and not comparison.report_errors
            and not comparison.report_warnings
        ):
            continue
        issue_queue.append(
            {
                "kind": "Report",
                "comparison": comparison,
                "severity": item["severity"],
                "diagnostics": len(comparison.report_errors)
                + len(comparison.report_warnings),
                "diagnostic_rows": item["diagnostic_rows"],
                "engine_pair": f"{comparison.engine_a} vs {comparison.engine_b}",
                "explorer_id": "report-comparison-explorer",
                "dom_id": item["dom_id"],
            }
        )
    for item in output_drilldowns:
        if item["severity"] == "green":
            continue
        comparison = item["comparison"]
        issue_queue.append(
            {
                "kind": "Output",
                "comparison": comparison,
                "severity": item["severity"],
                "diagnostics": None,
                "graphical_series": item["graphical_series"],
                "graphical_unavailable_reason": item["graphical_unavailable_reason"],
                "engine_pair": f"{comparison.engine_a} vs {comparison.engine_b}",
                "explorer_id": "output-comparison-explorer",
                "dom_id": item["dom_id"],
            }
        )
    issue_queue.sort(
        key=lambda item: (
            {"failed": 3, "red": 2, "yellow": 1, "green": 0}[item["severity"]],
            item["comparison"].overall_distance,
        ),
        reverse=True,
    )
    issue_groups_by_model: dict[str, list[dict[str, Any]]] = {}
    for item in issue_queue:
        issue_groups_by_model.setdefault(item["comparison"].inp_name, []).append(item)
    issue_groups = [
        {
            "model": model,
            "issues": items,
            "severity": items[0]["severity"],
            "kinds": sorted({item["kind"] for item in items}),
            "engine_pair": items[0]["engine_pair"],
            "distance": max(item["comparison"].overall_distance for item in items),
            "diagnostics": sum(item["diagnostics"] or 0 for item in items),
        }
        for model, items in issue_groups_by_model.items()
    ]
    issue_counts = {
        severity: sum(item["severity"] == severity for item in issue_queue)
        for severity in ("failed", "red", "yellow")
    }
    issue_counts["total"] = len(issue_queue)

    interface_groups = []
    for family in result.interface_families:
        self_identities = set(family.self_comparison_identities)

        def report_rows(*identities: str) -> list[Any]:
            selected = set(identities)
            return [
                item
                for item in comparison_rows
                if item["comparison"].inp_path in selected
            ]

        def output_rows(*identities: str) -> list[dict[str, Any]]:
            selected = set(identities)
            return [
                item
                for item in output_drilldowns
                if item["comparison"].inp_path in selected
            ]

        run_rows = [
            {"role": role, "result": row}
            for identity, role in (
                (family.generator_identity, "Generator"),
                (family.consumer_identity, "Interface consumer"),
                (family.baseline_identity, "Direct baseline"),
            )
            for row in result.engine_results
            if row.inp_path == identity
        ]
        interface_groups.append(
            {
                "family": family.family,
                "artifact": family.artifact,
                "runs": run_rows,
                "comparison_views": (
                    {
                        "title": "Interface consumer",
                        "reports": report_rows(family.consumer_identity),
                        "outputs": output_rows(family.consumer_identity),
                    },
                    {
                        "title": "Direct baseline",
                        "reports": report_rows(family.baseline_identity),
                        "outputs": output_rows(family.baseline_identity),
                    },
                    {
                        "title": "Interface vs direct",
                        "reports": report_rows(*self_identities),
                        "outputs": output_rows(*self_identities),
                    },
                ),
            }
        )

    report_placeholder_count = sum(
        1 for item in comparison_rows if item["placeholder_reason"]
    )
    output_placeholder_count = sum(
        1 for item in output_drilldowns if item["placeholder_reason"]
    )
    report_error_count = sum(
        len(item["comparison"].report_errors) for item in comparison_rows
    )
    report_warning_count = sum(
        len(item["comparison"].report_warnings) for item in comparison_rows
    )

    distance_chart_rows = []
    for kind, rows in (
        ("Report", comparison_rows),
        ("Output", output_drilldowns),
    ):
        for item in rows:
            comparison = item["comparison"]
            distance_chart_rows.append(
                {
                    "kind": kind,
                    "model": comparison.inp_name,
                    "engine_a": comparison.engine_a,
                    "engine_b": comparison.engine_b,
                    "distance": (
                        None
                        if _failed_placeholder(item["placeholder_reason"])
                        else comparison.overall_distance
                    ),
                    "dom_id": item["dom_id"],
                }
            )
    distance_chart_engines = _prioritize_engines(
        list(
            dict.fromkeys(
                itertools.chain(
                    engines,
                    (
                        engine
                        for item in distance_chart_rows
                        for engine in (item["engine_a"], item["engine_b"])
                    ),
                )
            )
        ),
        engine_order,
        available_engines=all_engines,
    )
    distance_chart_models = sorted({item["model"] for item in distance_chart_rows})


    return {
        "result": result,
        "engines": engines,
        "models": models,
        "matrix": matrix,
        "comparison_rows": comparison_rows,
        "output_drilldowns": output_drilldowns,
        "issue_groups": issue_groups,
        "interface_groups": interface_groups,
        "issue_counts": issue_counts,
        "comparison_exclusion_threshold": _HTML_COMPARISON_DISTANCE_THRESHOLD,
        "excluded_report_comparison_count": report_placeholder_count,
        "excluded_output_comparison_count": output_placeholder_count,
        "report_error_count": report_error_count,
        "report_warning_count": report_warning_count,
        "chart_labels": engines,
        "chart_values": average_durations,
        "distance_chart_rows": distance_chart_rows,
        "distance_chart_engines": distance_chart_engines,
        "distance_chart_models": distance_chart_models,
        "duration_heading": (
            f"Mean of per-model median {duration_name}s by engine"
            if result.run_count and result.run_count > 1
            else f"Average {duration_name} by engine"
        ),
        "duration_chart_label": (
            f"Mean model median {duration_name} (s)"
            if result.run_count and result.run_count > 1
            else f"Average {duration_name} (s)"
        ),
    }


def render_html(
    result: BenchmarkResult, path: Path, *, engine_order: list[str] | None = None
) -> None:
    environment = Environment(
        loader=PackageLoader("swmm_bench", "templates"),
        autoescape=True,
    )
    environment.filters["distance_class"] = _distance_style
    template = environment.get_template("report.html.j2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        template.render(
            **_build_template_context(result, engine_order=engine_order)
        ),
        encoding="utf-8",
    )
