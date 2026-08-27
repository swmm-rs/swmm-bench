from __future__ import annotations

import dataclasses
import dis
import inspect
import itertools
import json
import math
import multiprocessing
import numbers
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, get_type_hints

from pandas import DataFrame, Timedelta, isna  # pyright: ignore[reportMissingImports]

from swmm_bench.models import (
    OUTPUT_DISTANCE_METRIC_COMPOSITE,
    EngineResult,
    ModelComparison,
    OutputComparison,
    OutputSectionComparison,
    OutputSeriesComparison,
    OutputTimelineCoverage,
    SectionComparison,
)
from swmm_bench.output import extract_output_frame, output_series_name
from swmm.pandas import Report  # pyright: ignore[reportMissingImports]

_MAX_RETAINED_DIFFERENCES = 100
_MAX_GRAPHICAL_SERIES_PER_COMPARISON = 5_000
_ESTIMATED_GRAPH_POINT_BYTES = 96
_ESTIMATED_GRAPH_SERIES_BYTES = 256
_DEFAULT_GRAPH_PAYLOAD_BUDGET_BYTES = 12_000 * _ESTIMATED_GRAPH_POINT_BYTES
_GRAPHICAL_OUTPUT_DISTANCE_THRESHOLD = 0.01
_BYTES_PER_MEBIBYTE = 1024 * 1024
_MIN_REPORT_RESERVE_BYTES = 5 * _BYTES_PER_MEBIBYTE
_MAX_REPORT_RESERVE_BYTES = 10 * _BYTES_PER_MEBIBYTE
_DEFAULT_REPORT_PARSE_WORKERS = 4
_DEFAULT_OUTPUT_PARSE_WORKERS = 1
DEFAULT_COMPOSITE_TYPICAL_WEIGHT = 0.75


def _parse_executor(max_workers: int, *, use_processes: bool) -> Executor:
    if use_processes:
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return ThreadPoolExecutor(max_workers=max_workers)


def _report_graph_payload_budget(report_size_mb: int) -> int:
    if report_size_mb < 1:
        raise ValueError("report_size_mb must be at least 1")
    target_bytes = report_size_mb * _BYTES_PER_MEBIBYTE
    reserve_bytes = min(
        target_bytes // 2,
        max(
            _MIN_REPORT_RESERVE_BYTES,
            min(target_bytes // 10, _MAX_REPORT_RESERVE_BYTES),
        ),
    )
    return target_bytes - reserve_bytes


def _graphical_series_payload_bytes(
    graphical_series: Sequence[OutputSeriesComparison],
) -> int:
    payload = [series.to_dict() for series in graphical_series]
    return len(
        json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _estimated_graph_payload_demand(frame_a: DataFrame, frame_b: DataFrame) -> int:
    series_count = min(
        len(_output_union_columns(frame_a, frame_b)),
        _MAX_GRAPHICAL_SERIES_PER_COMPARISON,
    )
    if not series_count:
        return 0
    period_count = len(frame_a.index.union(frame_b.index))
    return series_count * (
        _ESTIMATED_GRAPH_SERIES_BYTES
        + period_count * _ESTIMATED_GRAPH_POINT_BYTES
    )


def _allocate_graph_payload_budgets(
    pair_paths: Sequence[tuple[Path | None, Path | None]],
    frames_by_path: Mapping[Path, DataFrame | None],
    total_budget_bytes: int,
) -> list[int]:
    demands = []
    for out_a, out_b in pair_paths:
        frame_a = frames_by_path.get(out_a) if out_a is not None else None
        frame_b = frames_by_path.get(out_b) if out_b is not None else None
        demands.append(
            _estimated_graph_payload_demand(frame_a, frame_b)
            if frame_a is not None and frame_b is not None
            else 0
        )
    total_demand = sum(demands)
    if not total_demand:
        return [0] * len(pair_paths)
    return [
        total_budget_bytes * demand // total_demand
        for demand in demands
    ]


@dataclasses.dataclass(frozen=True)
class ComparisonProgress:
    phase: Literal[
        "report", "output-load", "output-pair", "output-series", "output-graph"
    ]
    completed: int
    total: int
    inp_name: str | None = None
    engine_a: str | None = None
    engine_b: str | None = None
    item_name: str | None = None
    status: Literal["started", "completed", "skipped"] = "completed"


@dataclasses.dataclass
class _PreparedOutputSeries:
    values_a: list[Any]
    values_b: list[Any]
    numeric_a: list[float | None]
    numeric_b: list[float | None]


@dataclasses.dataclass(frozen=True)
class _OutputTimeline:
    comparison_index: Any
    coverage: OutputTimelineCoverage


ProgressCallback = Callable[[ComparisonProgress], None]


def _is_null(value: Any) -> bool:
    try:
        return bool(isna(value))
    except (TypeError, ValueError):
        return False


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, Timedelta):
        return value.total_seconds()
    if isinstance(value, timedelta):
        return value.total_seconds()
    return None


def _cell_distance(value_a: Any, value_b: Any) -> tuple[float, float | None]:
    null_a = _is_null(value_a)
    null_b = _is_null(value_b)
    if null_a or null_b:
        return (0.0, None) if null_a and null_b else (1.0, None)

    try:
        if bool(value_a == value_b):
            return 0.0, None
    except (TypeError, ValueError):
        pass

    duration_a = _duration_seconds(value_a)
    duration_b = _duration_seconds(value_b)
    if duration_a is not None and duration_b is not None:
        absolute = abs(duration_a - duration_b)
        scale = max(abs(duration_a), abs(duration_b))
        return (0.0 if scale == 0.0 else min(absolute / scale, 1.0)), absolute

    if isinstance(value_a, bool) or isinstance(value_b, bool):
        return 1.0, None

    if isinstance(value_a, numbers.Real) and isinstance(value_b, numbers.Real):
        try:
            numeric_a = float(value_a)
            numeric_b = float(value_b)
        except (TypeError, ValueError, OverflowError):
            return 1.0, None
        if not math.isfinite(numeric_a) or not math.isfinite(numeric_b):
            return 1.0, None
        absolute = abs(numeric_a - numeric_b)
        scale = max(abs(numeric_a), abs(numeric_b))
        return (0.0 if scale == 0.0 else min(absolute / scale, 1.0)), absolute

    return 1.0, None


def _json_value(value: Any) -> Any:
    if _is_null(value):
        return None
    if isinstance(value, (Timedelta, timedelta)):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return numeric if math.isfinite(numeric) else None
    return value


def _validate_unique_labels(
    table_name: str,
    frame: DataFrame,
    report_path: str | Path | None = None,
) -> None:
    location = f"Report {report_path} table" if report_path is not None else "Table"
    if not frame.index.is_unique:
        raise ValueError(f"{location} {table_name!r} has duplicate index labels")
    if not frame.columns.is_unique:
        raise ValueError(f"{location} {table_name!r} has duplicate column labels")


def _compare_table(
    table_name: str,
    table_a: DataFrame | None,
    table_b: DataFrame | None,
) -> tuple[SectionComparison, float, int]:
    frame_a = table_a if table_a is not None else DataFrame()
    frame_b = table_b if table_b is not None else DataFrame()
    _validate_unique_labels(table_name, frame_a)
    _validate_unique_labels(table_name, frame_b)

    union_index = frame_a.index.union(frame_b.index, sort=False)
    union_columns = frame_a.columns.union(frame_b.columns, sort=False)
    aligned_a = frame_a.reindex(index=union_index, columns=union_columns)
    aligned_b = frame_b.reindex(index=union_index, columns=union_columns)

    distance_sum = 0.0
    cell_count = len(union_index) * len(union_columns)
    differences: list[dict[str, Any]] = []
    difference_count = 0
    for row in union_index:
        for column in union_columns:
            value_a = aligned_a.at[row, column]
            value_b = aligned_b.at[row, column]
            present_a = row in frame_a.index and column in frame_a.columns
            present_b = row in frame_b.index and column in frame_b.columns
            if present_a != present_b:
                distance, absolute = 1.0, None
            else:
                distance, absolute = _cell_distance(value_a, value_b)
            distance_sum += distance
            if distance > 0.0:
                difference_count += 1
                differences.append(
                    {
                        "row": _json_value(row),
                        "column": _json_value(column),
                        "value_a": _json_value(value_a),
                        "value_b": _json_value(value_b),
                        "abs_diff": _json_value(absolute),
                        "rel_diff": distance,
                    }
                )
                differences.sort(
                    key=lambda item: (
                        -item["rel_diff"],
                        repr(item["row"]),
                        repr(item["column"]),
                    )
                )
                del differences[_MAX_RETAINED_DIFFERENCES:]

    comparison = SectionComparison(
        section_name=table_name,
        distance=distance_sum / cell_count if cell_count else 0.0,
        row_count_a=len(frame_a.index),
        row_count_b=len(frame_b.index),
        differences=differences,
        difference_count=difference_count,
        differences_truncated=difference_count > len(differences),
    )
    return comparison, distance_sum, cell_count


def _missing_table_comparison(
    table_name: str,
    table_a: DataFrame | None,
    table_b: DataFrame | None,
    engine_a: str = "A",
    engine_b: str = "B",
    note: str | None = None,
) -> tuple[SectionComparison, float, int]:
    present = table_a if table_a is not None else table_b
    cell_count = len(present.index) * len(present.columns) if present is not None else 0
    if note is None:
        note = (
            f"Table exists for {engine_a} but not {engine_b}."
            if table_a is not None
            else f"Table exists for {engine_b} but not {engine_a}."
        )
    return (
        SectionComparison(
            section_name=table_name,
            distance=1.0 if cell_count else 0.0,
            row_count_a=len(table_a.index) if table_a is not None else 0,
            row_count_b=len(table_b.index) if table_b is not None else 0,
            differences=[],
            note=note,
        ),
        cell_count * 1.0,
        cell_count,
    )


def _compare_tables(
    tables_a: Mapping[str, DataFrame],
    tables_b: Mapping[str, DataFrame],
    engine_a: str = "A",
    engine_b: str = "B",
) -> tuple[list[SectionComparison], float]:
    comparisons: list[SectionComparison] = []
    total_distance = 0.0
    total_cells = 0
    for table_name in sorted(set(tables_a) | set(tables_b)):
        table_a = tables_a.get(table_name)
        table_b = tables_b.get(table_name)
        if table_a is None or table_b is None:
            comparison, distance_sum, cell_count = _missing_table_comparison(
                table_name, table_a, table_b, engine_a, engine_b
            )
        else:
            comparison, distance_sum, cell_count = _compare_table(
                table_name,
                table_a,
                table_b,
            )
        comparisons.append(comparison)
        total_distance += distance_sum
        total_cells += cell_count
    return comparisons, total_distance / total_cells if total_cells else 0.0


def _dataframe_property_names(report_type: type[Any]) -> list[str]:
    names: list[str] = []
    for name, member in inspect.getmembers(report_type):
        if not isinstance(member, property) or member.fget is None:
            continue
        try:
            return_type = get_type_hints(member.fget).get("return")
        except (NameError, TypeError) as exc:
            raise TypeError(
                f"Cannot resolve return annotation for Report property {name!r}"
            ) from exc
        if return_type is DataFrame:
            names.append(name)
    return names


def _is_missing_section_lookup(error: KeyError, report: Any) -> bool:
    if not error.args or not hasattr(report, "_sections"):
        return False
    traceback = error.__traceback__
    if traceback is None:
        return False
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    instruction = next(
        (
            item
            for item in dis.get_instructions(traceback.tb_frame.f_code)
            if item.offset == traceback.tb_lasti
        ),
        None,
    )
    return bool(
        instruction is not None
        and instruction.opname in {"BINARY_SUBSCR", "BINARY_OP"}
        and traceback.tb_frame.f_locals.get("self") is report
        and error.args[0] not in report._sections
    )


def _report_messages(report: Any, property_name: str, label: str) -> list[str]:
    try:
        messages = getattr(report, property_name)
    except AttributeError:
        return []
    except Exception as exc:
        return [f"{label.upper()}: could not be parsed: {exc}"]
    return [
        f"{label.upper()} {getattr(message, 'code', '-')}: {getattr(message, 'message', message)}"
        for message in messages
    ]


def _extract_report_tables(
    rpt_path: str | Path,
) -> tuple[dict[str, DataFrame], dict[str, str], list[str], list[str]]:
    path = Path(rpt_path)
    report = Report(str(path))
    report_errors = _report_messages(report, "errors", "error")
    report_warnings = _report_messages(report, "warnings", "warning")
    tables: dict[str, DataFrame] = {}
    skipped: dict[str, str] = {}
    for property_name in _dataframe_property_names(type(report)):
        if property_name in (
            "routing_time_step_summary",
            "highest_flow_instability_indexes",
        ):
            continue
        try:
            value = getattr(report, property_name)
            if not isinstance(value, DataFrame):
                raise TypeError(f"returned {type(value).__name__}, expected DataFrame")
            _validate_unique_labels(property_name, value, path)
        except KeyError as exc:
            if _is_missing_section_lookup(exc, report):
                continue
            message = f"WARNING: table {property_name!r} could not be parsed: {exc}"
        except Exception as exc:
            message = f"WARNING: table {property_name!r} could not be parsed: {exc}"
        else:
            tables[property_name] = value
            continue
        warnings.warn(
            f"Report {path} {message.removeprefix('WARNING: ')}", stacklevel=2
        )
        skipped[property_name] = message
    return tables, skipped, report_errors, report_warnings


def _extract_report_tables_for_cache(
    rpt_path: str | Path,
) -> tuple[
    tuple[dict[str, DataFrame], dict[str, str], list[str], list[str]],
    list[str],
]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = _extract_report_tables(rpt_path)
    return report, [str(item.message) for item in caught]


def _compare_report_tables(
    report_a: tuple[dict[str, DataFrame], dict[str, str], list[str], list[str]],
    report_b: tuple[dict[str, DataFrame], dict[str, str], list[str], list[str]],
    engine_a: str,
    engine_b: str,
    inp_path: str,
    inp_name: str,
) -> ModelComparison:
    cached_tables_a, skipped_a, errors_a, warnings_a = report_a
    cached_tables_b, skipped_b, errors_b, warnings_b = report_b
    tables_a = cached_tables_a.copy()
    tables_b = cached_tables_b.copy()
    skipped_comparisons: list[SectionComparison] = []
    for table_name in sorted(set(skipped_a) | set(skipped_b)):
        table_a = tables_a.pop(table_name, None)
        table_b = tables_b.pop(table_name, None)
        if table_name in skipped_a and table_name in skipped_b:
            note = f"Table could not be parsed for {engine_a} or {engine_b}."
        elif table_name in skipped_a:
            note = (
                f"Table could not be parsed for {engine_a}; parsed only for {engine_b}."
            )
        else:
            note = (
                f"Table could not be parsed for {engine_b}; parsed only for {engine_a}."
            )
        comparison, _, _ = _missing_table_comparison(
            table_name, table_a, table_b, engine_a, engine_b, note
        )
        skipped_comparisons.append(comparison)
    table_comparisons, overall_distance = _compare_tables(
        tables_a, tables_b, engine_a, engine_b
    )
    table_comparisons.extend(skipped_comparisons)
    return ModelComparison(
        inp_path=inp_path,
        inp_name=inp_name,
        engine_a=engine_a,
        engine_b=engine_b,
        overall_distance=overall_distance,
        section_comparisons=table_comparisons,
        report_warnings=list(skipped_a.values())
        + list(skipped_b.values())
        + warnings_a
        + warnings_b,
        report_errors=errors_a + errors_b,
        report_warnings_a=list(skipped_a.values()) + warnings_a,
        report_warnings_b=list(skipped_b.values()) + warnings_b,
        report_errors_a=errors_a,
        report_errors_b=errors_b,
    )


def compare_rpts(
    rpt_a: str | Path,
    rpt_b: str | Path,
    engine_a: str,
    engine_b: str,
    inp_path: str,
    inp_name: str,
) -> ModelComparison:
    return _compare_report_tables(
        _extract_report_tables(rpt_a),
        _extract_report_tables(rpt_b),
        engine_a,
        engine_b,
        inp_path,
        inp_name,
    )


def _chart_number(value: Any) -> float | None:
    if (
        _is_null(value)
        or isinstance(value, bool)
        or not isinstance(value, numbers.Real)
    ):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _output_diagnostic_value(value: Any) -> Any:
    if _is_null(value):
        return None
    numeric = _chart_number(value)
    return numeric if numeric is not None else str(value)


def _output_column(raw_column: Any) -> tuple[str, str, str]:
    values = tuple(str(item) for item in raw_column)
    if len(values) != 3:
        raise ValueError(f"Expected three output column levels, received {len(values)}")
    return values[0], values[1], values[2]


def _output_union_index(frame_a: DataFrame, frame_b: DataFrame) -> Any:
    if len(frame_a.index):
        return frame_a.index.union(frame_b.index).sort_values()
    return frame_b.index


def _output_timeline(frame_a: DataFrame, frame_b: DataFrame) -> _OutputTimeline:
    index_a = frame_a.index
    index_b = frame_b.index
    shared_timestamp_count = len(index_a.intersection(index_b))
    comparison_index = _output_union_index(frame_a, frame_b)
    trailing_timestamp_count_a = 0
    trailing_timestamp_count_b = 0

    if len(index_a) and len(index_b):
        common_end = min(max(index_a), max(index_b))
        trailing_timestamp_count_a = sum(
            timestamp > common_end for timestamp in index_a
        )
        trailing_timestamp_count_b = sum(
            timestamp > common_end for timestamp in index_b
        )
        comparison_index = comparison_index[comparison_index <= common_end]

    return _OutputTimeline(
        comparison_index=comparison_index,
        coverage=OutputTimelineCoverage(
            timestamp_count_a=len(index_a),
            timestamp_count_b=len(index_b),
            shared_timestamp_count=shared_timestamp_count,
            trailing_timestamp_count_a=trailing_timestamp_count_a,
            trailing_timestamp_count_b=trailing_timestamp_count_b,
        ),
    )


def _output_union_columns(frame_a: DataFrame, frame_b: DataFrame) -> Any:
    if len(frame_a.columns):
        return frame_a.columns.union(frame_b.columns, sort=False)
    return frame_b.columns


def _missing_output_reason(
    *,
    column_in_a: bool,
    column_in_b: bool,
    timestamp_in_a: bool,
    timestamp_in_b: bool,
    null_a: bool,
    null_b: bool,
    numeric_a: float | None,
    numeric_b: float | None,
) -> str:
    if not column_in_a:
        return "semantic series absent in A"
    if not column_in_b:
        return "semantic series absent in B"
    if not timestamp_in_a:
        return "timestamp missing in A"
    if not timestamp_in_b:
        return "timestamp missing in B"
    if null_a and not null_b:
        return "null value in A"
    if null_b and not null_a:
        return "null value in B"
    if numeric_a is None and numeric_b is None:
        return "invalid values in A and B"
    if numeric_a is None:
        return "invalid value in A"
    return "invalid value in B"


def _retain_output_diagnostic(
    diagnostics: list[dict[str, Any]],
    diagnostic: dict[str, Any],
) -> None:
    diagnostics.append(diagnostic)
    diagnostics.sort(
        key=lambda item: (
            0 if item["issue"] != "numeric difference" else 1,
            -item["rel_diff"],
            repr(item["row"]),
        )
    )
    del diagnostics[_MAX_RETAINED_DIFFERENCES:]


def _compare_output_series(
    raw_column: Any,
    frame_a: DataFrame,
    frame_b: DataFrame,
    union_index: Any,
    timestamp_presence_a: list[bool],
    timestamp_presence_b: list[bool],
    prepared: _PreparedOutputSeries,
    *,
    typical_weight: float,
    include_details: bool = True,
) -> OutputSectionComparison:
    column = _output_column(raw_column)
    series_name = output_series_name(column)
    column_in_a = raw_column in frame_a.columns
    column_in_b = raw_column in frame_b.columns
    finite_values_a: list[float] = []
    finite_values_b: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    numeric_difference_count = 0
    missing_count = 0
    both_null_count = 0

    for position, timestamp in enumerate(union_index):
        timestamp_in_a = timestamp_presence_a[position]
        timestamp_in_b = timestamp_presence_b[position]
        present_a = column_in_a and timestamp_in_a
        present_b = column_in_b and timestamp_in_b
        value_a = prepared.values_a[position] if present_a else None
        value_b = prepared.values_b[position] if present_b else None
        null_a = present_a and _is_null(value_a)
        null_b = present_b and _is_null(value_b)
        numeric_a = prepared.numeric_a[position] if present_a else None
        numeric_b = prepared.numeric_b[position] if present_b else None

        if present_a and present_b and null_a and null_b:
            both_null_count += 1
            continue
        if present_a and present_b and numeric_a is not None and numeric_b is not None:
            finite_values_a.append(numeric_a)
            finite_values_b.append(numeric_b)
            if numeric_a != numeric_b:
                numeric_difference_count += 1
            continue

        missing_count += 1
        if include_details:
            _retain_output_diagnostic(
                diagnostics,
                {
                    "row": _json_value(timestamp),
                    "column": _json_value(raw_column),
                    "value_a": _output_diagnostic_value(value_a),
                    "value_b": _output_diagnostic_value(value_b),
                    "abs_diff": None,
                    "rel_diff": 1.0,
                    "issue": _missing_output_reason(
                        column_in_a=column_in_a,
                        column_in_b=column_in_b,
                        timestamp_in_a=timestamp_in_a,
                        timestamp_in_b=timestamp_in_b,
                        null_a=null_a,
                        null_b=null_b,
                        numeric_a=numeric_a,
                        numeric_b=numeric_b,
                    ),
                },
            )

    finite_pair_count = len(finite_values_a)
    max_magnitude = max(
        (
            max(abs(value_a), abs(value_b))
            for value_a, value_b in zip(finite_values_a, finite_values_b)
        ),
        default=0.0,
    )
    numeric_distance = 0.0
    typical_numeric_distance = 0.0
    normalized_scale = 0.0
    if finite_pair_count and max_magnitude:
        rmse = math.sqrt(
            math.fsum(
                ((value_a / max_magnitude) - (value_b / max_magnitude)) ** 2
                for value_a, value_b in zip(finite_values_a, finite_values_b)
            )
            / finite_pair_count
        )
        rms_a = math.sqrt(
            math.fsum((value / max_magnitude) ** 2 for value in finite_values_a)
            / finite_pair_count
        )
        rms_b = math.sqrt(
            math.fsum((value / max_magnitude) ** 2 for value in finite_values_b)
            / finite_pair_count
        )
        normalized_scale = max(rms_a, rms_b)
        if normalized_scale:
            numeric_distance = min(rmse / normalized_scale, 1.0)
        typical_numeric_distance = min(
            math.fsum(
                abs(value_a - value_b) / max_magnitude
                for value_a, value_b in zip(finite_values_a, finite_values_b)
            )
            / finite_pair_count,
            1.0,
        )

    if (
        include_details
        and numeric_difference_count
        and len(diagnostics) < _MAX_RETAINED_DIFFERENCES
    ):
        for position, timestamp in enumerate(union_index):
            if not (
                column_in_a
                and column_in_b
                and timestamp_presence_a[position]
                and timestamp_presence_b[position]
            ):
                continue
            value_a = prepared.numeric_a[position]
            value_b = prepared.numeric_b[position]
            if value_a is None or value_b is None or value_a == value_b:
                continue
            absolute = abs(value_a - value_b)
            normalized_absolute = 0.0
            if max_magnitude and normalized_scale:
                normalized_absolute = min(
                    abs((value_a / max_magnitude) - (value_b / max_magnitude))
                    / normalized_scale,
                    1.0,
                )
            _retain_output_diagnostic(
                diagnostics,
                {
                    "row": _json_value(timestamp),
                    "column": _json_value(raw_column),
                    "value_a": value_a,
                    "value_b": value_b,
                    "abs_diff": absolute if math.isfinite(absolute) else None,
                    "rel_diff": normalized_absolute,
                    "issue": "numeric difference",
                },
            )

    difference_count = missing_count + numeric_difference_count
    timestamp_count = len(union_index)
    if timestamp_count:
        missing_fraction = missing_count / timestamp_count
    elif column_in_a != column_in_b:
        missing_fraction = 1.0
    else:
        missing_fraction = 0.0
    typical_distance = min(
        max(
            missing_fraction
            + (1.0 - missing_fraction) * typical_numeric_distance,
            0.0,
        ),
        1.0,
    )
    event_distance = min(
        max(
            missing_fraction + (1.0 - missing_fraction) * numeric_distance,
            0.0,
        ),
        1.0,
    )
    distance = (
        typical_weight * typical_distance
        + (1.0 - typical_weight) * event_distance
    )

    return OutputSectionComparison(
        section_name=series_name,
        distance=distance,
        row_count_a=len(frame_a.index) if column_in_a else 0,
        row_count_b=len(frame_b.index) if column_in_b else 0,
        differences=diagnostics,
        difference_count=difference_count,
        differences_truncated=difference_count > len(diagnostics),
        numeric_distance=numeric_distance,
        typical_distance=typical_distance,
        event_distance=event_distance,
        missing_fraction=missing_fraction,
        finite_pair_count=finite_pair_count,
        missing_count=missing_count,
        both_null_count=both_null_count,
        timestamp_count=timestamp_count,
    )


def _build_graphical_series(
    union_index: Any,
    sorted_columns: list[Any],
    frame_a: DataFrame,
    frame_b: DataFrame,
    section_comparisons: Sequence[OutputSectionComparison],
    *,
    inp_name: str,
    engine_a: str,
    engine_b: str,
    payload_budget_bytes: int | None = None,
    progress_callback: ProgressCallback | None = None,
    existing_series: Sequence[OutputSeriesComparison] = (),
) -> tuple[list[OutputSeriesComparison], str | None]:
    series_count = len(sorted_columns)
    if not series_count:
        return [], None

    sections_by_name = {
        section.section_name: section for section in section_comparisons
    }
    effective_payload_budget = (
        payload_budget_bytes
        if payload_budget_bytes is not None
        else _DEFAULT_GRAPH_PAYLOAD_BUDGET_BYTES
    )
    candidate_columns = sorted(
        sorted_columns,
        key=lambda raw_column: (
            -sections_by_name[
                output_series_name(_output_column(raw_column))
            ].distance,
            -sections_by_name[
                output_series_name(_output_column(raw_column))
            ].difference_count,
            output_series_name(_output_column(raw_column)),
        ),
    )[:_MAX_GRAPHICAL_SERIES_PER_COMPARISON]

    graphical_series = list(existing_series)
    used_payload_bytes = (
        _graphical_series_payload_bytes(graphical_series)
        if graphical_series
        else 2  # Opening and closing brackets around the JSON list.
    )
    for position, raw_column in enumerate(
        candidate_columns[len(graphical_series) :],
        start=len(graphical_series) + 1,
    ):
        column = _output_column(raw_column)
        series_name = output_series_name(column)
        section = sections_by_name[series_name]
        values_a = (
            frame_a.loc[:, raw_column].reindex(union_index).tolist()
            if raw_column in frame_a.columns
            else [None] * len(union_index)
        )
        values_b = (
            frame_b.loc[:, raw_column].reindex(union_index).tolist()
            if raw_column in frame_b.columns
            else [None] * len(union_index)
        )
        numeric_a = [_chart_number(value) for value in values_a]
        numeric_b = [_chart_number(value) for value in values_b]
        positions = range(len(union_index))
        candidate = OutputSeriesComparison(
            element_type=column[0],
            element_name=column[1],
            attribute=column[2],
            distance=section.distance,
            row_count_a=section.row_count_a,
            row_count_b=section.row_count_b,
            timestamps=[str(_json_value(union_index[item])) for item in positions],
            values_a=[numeric_a[item] for item in positions],
            values_b=[numeric_b[item] for item in positions],
            source_point_count=len(union_index),
            sampled=False,
            typical_distance=section.typical_distance,
            event_distance=section.event_distance,
        )
        candidate_bytes = _graphical_series_payload_bytes([candidate]) - 2
        separator_bytes = 1 if graphical_series else 0
        if (
            used_payload_bytes + separator_bytes + candidate_bytes
            > effective_payload_budget
        ):
            break
        graphical_series.append(candidate)
        used_payload_bytes += separator_bytes + candidate_bytes
        if progress_callback is not None:
            progress_callback(
                ComparisonProgress(
                    phase="output-graph",
                    completed=position,
                    total=len(candidate_columns),
                    inp_name=inp_name,
                    engine_a=engine_a,
                    engine_b=engine_b,
                    item_name=series_name,
                )
            )

    if not graphical_series:
        return (
            [],
            "Graphical data was not retained because the report-size target "
            "left insufficient chart payload budget.",
        )
    if len(graphical_series) < series_count:
        return (
            graphical_series,
            f"Retained complete chart data for {len(graphical_series):,} of "
            f"{series_count:,} highest-distance output series within the "
            "report-size target.",
        )
    return graphical_series, None


def _validate_typical_weight(typical_weight: float) -> float:
    if not math.isfinite(typical_weight) or not 0.0 <= typical_weight <= 1.0:
        raise ValueError("typical_weight must be a finite number from 0 to 1")
    return typical_weight


def _compare_output_frames(
    frame_a: DataFrame,
    frame_b: DataFrame,
    engine_a: str,
    engine_b: str,
    inp_path: str,
    inp_name: str,
    *,
    progress_callback: ProgressCallback | None = None,
    retain_tabular: bool = True,
    retain_graphical: bool = True,
    include_all_comparisons: bool = False,
    retain_graph_sections: bool = False,
    graph_payload_budget_bytes: int | None = None,
    typical_weight: float = DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
) -> OutputComparison:
    typical_weight = _validate_typical_weight(typical_weight)
    union_columns = _output_union_columns(frame_a, frame_b)
    if not len(union_columns):
        raise ValueError("Neither output file contains reporting periods")
    timeline = _output_timeline(frame_a, frame_b)
    comparison_index: Any = timeline.comparison_index
    graphical_index: Any = _output_union_index(frame_a, frame_b)
    sorted_columns = sorted(union_columns)
    timestamp_presence_a = comparison_index.isin(frame_a.index).tolist()
    timestamp_presence_b = comparison_index.isin(frame_b.index).tolist()
    table_comparisons: list[OutputSectionComparison] = []

    for position, raw_column in enumerate(sorted_columns, start=1):
        values_a = (
            frame_a.loc[:, raw_column].reindex(comparison_index).tolist()
            if raw_column in frame_a.columns
            else [None] * len(comparison_index)
        )
        values_b = (
            frame_b.loc[:, raw_column].reindex(comparison_index).tolist()
            if raw_column in frame_b.columns
            else [None] * len(comparison_index)
        )
        prepared = _PreparedOutputSeries(
            values_a=values_a,
            values_b=values_b,
            numeric_a=[_chart_number(value) for value in values_a],
            numeric_b=[_chart_number(value) for value in values_b],
        )
        table_comparisons.append(
            _compare_output_series(
                raw_column,
                frame_a,
                frame_b,
                comparison_index,
                timestamp_presence_a,
                timestamp_presence_b,
                prepared,
                typical_weight=typical_weight,
                include_details=retain_tabular,
            )
        )
        if progress_callback is not None:
            progress_callback(
                ComparisonProgress(
                    phase="output-series",
                    completed=position,
                    total=len(sorted_columns),
                    inp_name=inp_name,
                    engine_a=engine_a,
                    engine_b=engine_b,
                    item_name=output_series_name(_output_column(raw_column)),
                )
            )

    overall_distance = sum(
        comparison.distance for comparison in table_comparisons
    ) / len(table_comparisons)
    typical_distance = sum(
        comparison.typical_distance or 0.0 for comparison in table_comparisons
    ) / len(table_comparisons)
    event_distance = sum(
        comparison.event_distance or 0.0 for comparison in table_comparisons
    ) / len(table_comparisons)
    graphical_series: list[OutputSeriesComparison] = []
    unavailable_reason: str | None = None
    if retain_graphical and (
        include_all_comparisons
        or overall_distance > _GRAPHICAL_OUTPUT_DISTANCE_THRESHOLD
    ):
        graphical_series, unavailable_reason = _build_graphical_series(
            graphical_index,
            sorted_columns,
            frame_a,
            frame_b,
            table_comparisons,
            inp_name=inp_name,
            engine_a=engine_a,
            engine_b=engine_b,
            payload_budget_bytes=graph_payload_budget_bytes,
            progress_callback=progress_callback,
        )
    elif retain_graphical:
        unavailable_reason = (
            f"Graphical data was not retained because the overall distance "
            f"did not exceed {_GRAPHICAL_OUTPUT_DISTANCE_THRESHOLD:.6f}."
        )
    else:
        unavailable_reason = "Detailed output comparisons were not retained."
    return OutputComparison(
        inp_path=inp_path,
        inp_name=inp_name,
        engine_a=engine_a,
        engine_b=engine_b,
        overall_distance=overall_distance,
        section_comparisons=(
            table_comparisons if retain_tabular or retain_graph_sections else []
        ),
        graphical_series=graphical_series,
        details_retained=bool(graphical_series),
        graphical_unavailable_reason=unavailable_reason,
        metric=OUTPUT_DISTANCE_METRIC_COMPOSITE,
        typical_weight=typical_weight,
        typical_distance=typical_distance,
        event_distance=event_distance,
        timeline_coverage=timeline.coverage,
    )


def compare_outs(
    out_a: str | Path,
    out_b: str | Path,
    engine_a: str,
    engine_b: str,
    inp_path: str,
    inp_name: str,
    *,
    typical_weight: float = DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
) -> OutputComparison:
    return _compare_output_frames(
        extract_output_frame(out_a),
        extract_output_frame(out_b),
        engine_a,
        engine_b,
        inp_path,
        inp_name,
        retain_tabular=False,
        typical_weight=typical_weight,
    )


def compare_all(
    engine_results: list[EngineResult],
    *,
    progress_callback: ProgressCallback | None = None,
    parse_workers: int = _DEFAULT_REPORT_PARSE_WORKERS,
) -> list[ModelComparison]:
    comparisons: list[ModelComparison] = []
    grouped: dict[str, list[EngineResult]] = {}

    for result in engine_results:
        grouped.setdefault(result.inp_path, []).append(result)

    pairs = [
        (inp_path, result_a, result_b)
        for inp_path, results in grouped.items()
        for result_a, result_b in itertools.combinations(results, 2)
    ]

    def report_path(result: EngineResult) -> Path | None:
        if not result.rpt_path:
            return None
        path = Path(result.rpt_path)
        if not path.exists() or path.stat().st_size == 0:
            return None
        return path

    pair_paths = [
        (report_path(result_a), report_path(result_b))
        for _, result_a, result_b in pairs
    ]
    paths_to_load = list(
        dict.fromkeys(
            path
            for paths in pair_paths
            if all(path is not None for path in paths)
            for path in paths
            if path is not None
        )
    )
    if parse_workers < 1:
        raise ValueError("parse_workers must be at least 1")
    reports_by_path = {}
    if paths_to_load:
        with _parse_executor(
            min(parse_workers, len(paths_to_load)),
            use_processes=parse_workers > 1 and len(paths_to_load) > 1,
        ) as executor:
            loaded_reports = executor.map(
                _extract_report_tables_for_cache, paths_to_load
            )
            for path, (report, warning_messages) in zip(
                paths_to_load, loaded_reports
            ):
                reports_by_path[path] = report
                for message in warning_messages:
                    warnings.warn(message, stacklevel=2)

    for position, ((inp_path, result_a, result_b), (rpt_a, rpt_b)) in enumerate(
        zip(pairs, pair_paths), start=1
    ):
        status: Literal["completed", "skipped"] = "skipped"
        if progress_callback is not None:
            progress_callback(
                ComparisonProgress(
                    phase="report",
                    completed=position - 1,
                    total=len(pairs),
                    inp_name=result_a.inp_name,
                    engine_a=result_a.engine_name,
                    engine_b=result_b.engine_name,
                    status="started",
                )
            )
        try:
            if rpt_a is None or rpt_b is None:
                continue
            comparisons.append(
                _compare_report_tables(
                    reports_by_path[rpt_a],
                    reports_by_path[rpt_b],
                    result_a.engine_name,
                    result_b.engine_name,
                    inp_path,
                    result_a.inp_name,
                )
            )
            status = "completed"
        finally:
            if progress_callback is not None:
                progress_callback(
                    ComparisonProgress(
                        phase="report",
                        completed=position,
                        total=len(pairs),
                        inp_name=result_a.inp_name,
                        engine_a=result_a.engine_name,
                        engine_b=result_b.engine_name,
                        status=status,
                    )
                )

    return comparisons


def compare_all_outputs(
    engine_results: list[EngineResult],
    *,
    progress_callback: ProgressCallback | None = None,
    retain_graphical: bool = True,
    include_all_comparisons: bool = False,
    parse_workers: int = _DEFAULT_OUTPUT_PARSE_WORKERS,
    report_size_mb: int = 100,
    typical_weight: float = DEFAULT_COMPOSITE_TYPICAL_WEIGHT,
) -> list[OutputComparison]:
    typical_weight = _validate_typical_weight(typical_weight)
    comparisons: list[OutputComparison] = []
    comparison_frames: list[tuple[DataFrame, DataFrame]] = []
    grouped: dict[str, list[EngineResult]] = {}
    frames_by_path: dict[Path, DataFrame | None] = {}

    for result in engine_results:
        grouped.setdefault(result.inp_path, []).append(result)

    pairs = [
        (inp_path, result_a, result_b)
        for inp_path, results in grouped.items()
        for result_a, result_b in itertools.combinations(results, 2)
    ]
    total_graph_payload_bytes = (
        _report_graph_payload_budget(report_size_mb) if retain_graphical else 0
    )

    def output_path(result: EngineResult) -> Path | None:
        if result.exit_code not in (0, None) or not result.out_path:
            return None
        path = Path(result.out_path)
        if not path.exists() or path.stat().st_size == 0:
            return None
        return path

    pair_paths = [
        (output_path(result_a), output_path(result_b))
        for _, result_a, result_b in pairs
    ]
    paths_to_load = list(
        dict.fromkeys(
            path
            for paths in pair_paths
            if all(path is not None for path in paths)
            for path in paths
            if path is not None
        )
    )
    if parse_workers < 1:
        raise ValueError("parse_workers must be at least 1")
    completed_loads = 0
    with _parse_executor(
        min(parse_workers, len(paths_to_load) or 1),
        use_processes=parse_workers > 1 and len(paths_to_load) > 1,
    ) as executor:
        futures_by_path = {}
        for path in paths_to_load:
            if progress_callback is not None:
                progress_callback(
                    ComparisonProgress(
                        phase="output-load",
                        completed=completed_loads,
                        total=len(paths_to_load),
                        item_name=str(path),
                        status="started",
                    )
                )
            futures_by_path[path] = executor.submit(extract_output_frame, path)

        for path in paths_to_load:
            future = futures_by_path[path]
            status: Literal["completed", "skipped"] = "completed"
            try:
                frames_by_path[path] = future.result()
            except (OSError, TypeError, ValueError) as exc:
                warnings.warn(f"Skipping unreadable output {path}: {exc}", stacklevel=2)
                frames_by_path[path] = None
                status = "skipped"
            completed_loads += 1
            if progress_callback is not None:
                progress_callback(
                    ComparisonProgress(
                        phase="output-load",
                        completed=completed_loads,
                        total=len(paths_to_load),
                        item_name=str(path),
                        status=status,
                    )
                )

    pair_graph_budgets = _allocate_graph_payload_budgets(
        pair_paths,
        frames_by_path,
        total_graph_payload_bytes,
    )
    for position, (
        (inp_path, result_a, result_b),
        (out_a, out_b),
        pair_graph_budget,
    ) in enumerate(
        zip(pairs, pair_paths, pair_graph_budgets),
        start=1,
    ):
        status: Literal["completed", "skipped"] = "skipped"
        if progress_callback is not None:
            progress_callback(
                ComparisonProgress(
                    phase="output-pair",
                    completed=position - 1,
                    total=len(pairs),
                    inp_name=result_a.inp_name,
                    engine_a=result_a.engine_name,
                    engine_b=result_b.engine_name,
                    status="started",
                )
            )
        try:
            if out_a is None or out_b is None:
                continue
            frame_a = frames_by_path.get(out_a)
            frame_b = frames_by_path.get(out_b)
            if frame_a is None or frame_b is None:
                continue
            try:
                comparison = _compare_output_frames(
                    frame_a,
                    frame_b,
                    result_a.engine_name,
                    result_b.engine_name,
                    inp_path,
                    result_a.inp_name,
                    progress_callback=progress_callback,
                    retain_tabular=False,
                    retain_graphical=retain_graphical,
                    include_all_comparisons=include_all_comparisons,
                    retain_graph_sections=True,
                    graph_payload_budget_bytes=pair_graph_budget,
                    typical_weight=typical_weight,
                )
                comparisons.append(comparison)
                comparison_frames.append((frame_a, frame_b))
                status = "completed"
            except ValueError as exc:
                warnings.warn(
                    f"Skipping output comparison for {inp_path}: {exc}",
                    stacklevel=2,
                )
        finally:
            if progress_callback is not None:
                progress_callback(
                    ComparisonProgress(
                        phase="output-pair",
                        completed=position,
                        total=len(pairs),
                        inp_name=result_a.inp_name,
                        engine_a=result_a.engine_name,
                        engine_b=result_b.engine_name,
                        status=status,
                    )
                )

    retained_payload_bytes = sum(
        _graphical_series_payload_bytes(comparison.graphical_series)
        for comparison in comparisons
        if comparison.graphical_series
    )
    remaining_graph_payload_bytes = max(
        0, total_graph_payload_bytes - retained_payload_bytes
    )
    for position, comparison in enumerate(comparisons):
        if not remaining_graph_payload_bytes:
            break
        if not retain_graphical or not (
            include_all_comparisons
            or comparison.overall_distance > _GRAPHICAL_OUTPUT_DISTANCE_THRESHOLD
        ):
            continue
        frame_a, frame_b = comparison_frames[position]
        union_columns = _output_union_columns(frame_a, frame_b)
        series_count = min(
            len(union_columns),
            _MAX_GRAPHICAL_SERIES_PER_COMPARISON,
        )
        if len(comparison.graphical_series) >= series_count:
            continue
        current_payload_bytes = (
            _graphical_series_payload_bytes(comparison.graphical_series)
            if comparison.graphical_series
            else 0
        )
        graphical_series, unavailable_reason = _build_graphical_series(
            _output_union_index(frame_a, frame_b),
            sorted(union_columns),
            frame_a,
            frame_b,
            comparison.section_comparisons,
            inp_name=comparison.inp_name,
            engine_a=comparison.engine_a,
            engine_b=comparison.engine_b,
            payload_budget_bytes=(
                current_payload_bytes + remaining_graph_payload_bytes
            ),
            existing_series=comparison.graphical_series,
        )
        new_payload_bytes = (
            _graphical_series_payload_bytes(graphical_series)
            if graphical_series
            else 0
        )
        if new_payload_bytes <= current_payload_bytes:
            continue
        remaining_graph_payload_bytes -= new_payload_bytes - current_payload_bytes
        comparisons[position] = dataclasses.replace(
            comparison,
            graphical_series=graphical_series,
            details_retained=True,
            graphical_unavailable_reason=unavailable_reason,
        )

    return [
        dataclasses.replace(comparison, section_comparisons=[])
        for comparison in comparisons
    ]
