from __future__ import annotations

import math
import threading
import unittest
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from pandas import DataFrame, Series, Timedelta  # pyright: ignore[reportMissingImports]

from swmm_bench.comparator import (
    _cell_distance,
    _compare_tables,
    _dataframe_property_names,
    _extract_report_tables,
    compare_all,
    compare_rpts,
)
from swmm_bench.models import BenchmarkResult, EngineResult, ModelComparison
from swmm_bench.reporter import save_json


class CellDistanceTests(unittest.TestCase):
    def test_numeric_distance_is_symmetric(self) -> None:
        forward, forward_abs = _cell_distance(2.0, 4.0)
        reverse, reverse_abs = _cell_distance(4.0, 2.0)

        self.assertEqual(forward, 0.5)
        self.assertEqual(reverse, 0.5)
        self.assertEqual(forward_abs, 2.0)
        self.assertEqual(reverse_abs, 2.0)

    def test_null_string_boolean_nonfinite_and_timedelta_values(self) -> None:
        self.assertEqual(_cell_distance(float("nan"), float("nan")), (0.0, None))
        self.assertEqual(_cell_distance(float("nan"), 1.0), (1.0, None))
        self.assertEqual(_cell_distance("OPEN", "OPEN"), (0.0, None))
        self.assertEqual(_cell_distance("OPEN", "CLOSED"), (1.0, None))
        self.assertEqual(_cell_distance(True, False), (1.0, None))
        self.assertEqual(_cell_distance(math.inf, math.inf), (0.0, None))
        self.assertEqual(_cell_distance(math.inf, -math.inf), (1.0, None))
        self.assertEqual(
            _cell_distance(Timedelta(hours=1), timedelta(hours=2)),
            (0.5, 3600.0),
        )

    def test_sign_crossing_distances_are_capped_at_one(self) -> None:
        self.assertEqual(_cell_distance(-1.0, 1.0), (1.0, 2.0))
        self.assertEqual(
            _cell_distance(Timedelta(hours=-1), Timedelta(hours=1)),
            (1.0, 7200.0),
        )


class TableDistanceTests(unittest.TestCase):
    def test_identical_tables_have_zero_distance(self) -> None:
        table = DataFrame({"flow": [1.0, 2.0]}, index=["N1", "N2"])

        comparisons, overall = _compare_tables(
            {"node_flow": table},
            {"node_flow": table.copy()},
        )

        self.assertEqual(overall, 0.0)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].distance, 0.0)
        self.assertEqual(comparisons[0].differences, [])

    def test_overall_distance_weights_cells_not_tables(self) -> None:
        changed_a = DataFrame({"value": [0.0]}, index=["row"])
        changed_b = DataFrame({"value": [1.0]}, index=["row"])
        unchanged = DataFrame(
            {"a": [1.0], "b": [2.0], "c": [3.0]},
            index=["row"],
        )

        comparisons, overall = _compare_tables(
            {"changed": changed_a, "unchanged": unchanged},
            {"changed": changed_b, "unchanged": unchanged.copy()},
        )

        self.assertEqual(overall, 0.25)
        self.assertEqual(
            {item.section_name: item.distance for item in comparisons},
            {"changed": 1.0, "unchanged": 0.0},
        )

    def test_missing_table_row_column_and_value_score_union_cells(self) -> None:
        present = DataFrame({"a": [1.0, 2.0]}, index=["R1", "R2"])
        comparisons, overall = _compare_tables({"only_a": present}, {})
        self.assertEqual(overall, 1.0)
        self.assertEqual(comparisons[0].row_count_a, 2)
        self.assertEqual(comparisons[0].row_count_b, 0)
        self.assertEqual(comparisons[0].differences, [])
        self.assertEqual(comparisons[0].note, "Table exists for A but not B.")

        left = DataFrame({"left": [1.0]}, index=["left-row"])
        right = DataFrame({"right": [1.0]}, index=["right-row"])
        comparisons, overall = _compare_tables({"mixed": left}, {"mixed": right})
        self.assertEqual(overall, 0.5)
        self.assertEqual(comparisons[0].distance, 0.5)

        left_null = DataFrame({"value": [None]}, index=["row"])
        right_value = DataFrame({"value": [3.0]}, index=["row"])
        _, overall = _compare_tables({"null": left_null}, {"null": right_value})
        self.assertEqual(overall, 1.0)

    def test_structurally_missing_null_cell_scores_one(self) -> None:
        present_null = DataFrame({"value": [None]}, index=["row"])

        comparisons, overall = _compare_tables({"only_a": present_null}, {})

        self.assertEqual(overall, 1.0)
        self.assertEqual(comparisons[0].distance, 1.0)

    def test_differences_are_sorted_largest_first(self) -> None:
        left = DataFrame({"value": [9.0, 1.0]}, index=["small", "large"])
        right = DataFrame({"value": [10.0, 10.0]}, index=["small", "large"])

        comparisons, _ = _compare_tables({"table": left}, {"table": right})

        self.assertEqual(
            [item["row"] for item in comparisons[0].differences],
            ["large", "small"],
        )
        self.assertEqual(
            [item["rel_diff"] for item in comparisons[0].differences],
            [0.9, 0.1],
        )

    def test_difference_details_are_capped_without_changing_distance(self) -> None:
        left = DataFrame({"value": [0.0] * 101})
        right = DataFrame({"value": [1.0] * 101})

        comparisons, overall = _compare_tables({"table": left}, {"table": right})

        self.assertEqual(overall, 1.0)
        self.assertEqual(comparisons[0].difference_count, 101)
        self.assertEqual(len(comparisons[0].differences), 100)
        self.assertTrue(comparisons[0].differences_truncated)

    def test_nonfinite_report_diagnostics_remain_json_safe(self) -> None:
        left = DataFrame(
            {"value": [math.inf, 1e308]}, index=["infinite", "overflow"]
        )
        right = DataFrame(
            {"value": [1.0, -1e308]}, index=["infinite", "overflow"]
        )
        sections, overall_distance = _compare_tables({"summary": left}, {"summary": right})
        comparison = ModelComparison(
            inp_path="model.inp",
            inp_name="model.inp",
            engine_a="a",
            engine_b="b",
            overall_distance=overall_distance,
            section_comparisons=sections,
        )
        result = BenchmarkResult(
            schema_version="5",
            name="nonfinite-report-values",
            timestamp="2026-08-18T00:00:00+00:00",
            platform={},
            engine_results=[],
            comparisons=[comparison],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "results.json"
            save_json(result, output)

        self.assertEqual(sections[0].distance, 1.0)
        self.assertIsNone(sections[0].differences[0]["value_a"])
        self.assertIsNone(sections[0].differences[1]["abs_diff"])

    def test_duplicate_labels_are_rejected(self) -> None:
        duplicate_rows = DataFrame(
            {"value": [1.0, 2.0]},
            index=["row", "row"],
        )

        with self.assertRaisesRegex(ValueError, "duplicate index labels"):
            _compare_tables(
                {"table": duplicate_rows},
                {"table": duplicate_rows.copy()},
            )


def _write_report(path: Path, dry_weather_inflow: float) -> None:
    lines = [
        "  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2",
        "  ",
        "  *************",
        "  Element Count",
        "  *************",
        "  Number of nodes ........... 1",
        "  ",
        "  ",
        "  ****************",
        "  Raingage Summary",
        "  ****************",
        "  Name                 Data Source                    Type       Interval",
        "  ------------------------------------------------------------------------",
        "  Rain1                Rain1                          VOLUME       5 min.",
        "  ",
        "  ",
        "  **************************        Volume        Volume",
        "  Flow Routing Continuity        acre-feet      10^6 gal",
        "  **************************     ---------     ---------",
        f"  Dry Weather Inflow .......        {dry_weather_inflow:0.3f}        1.000",
        "  External Outflow .........         2.000         2.000",
        "  Continuity Error (%) .....         0.000",
        "  ",
        "  ",
        "  Analysis begun on:  Sun Jul 19 12:50:39 2026",
        "  Analysis ended on:  Sun Jul 19 12:50:40 2026",
        "  Total elapsed time: 00:00:01",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FakeReport:
    @property
    def table(self) -> DataFrame:
        return DataFrame({"value": [1.0]})

    @property
    def metadata(self) -> Series:
        return Series(["ignored"])


class ReportExtractionTests(unittest.TestCase):
    def test_discovers_only_dataframe_annotated_properties(self) -> None:
        self.assertEqual(_dataframe_property_names(FakeReport), ["table"])

    def test_routing_time_step_summary_is_excluded(self) -> None:
        class RoutingTimeStepReport:
            def __init__(self, path: str):
                self.path = path

            @property
            def routing_time_step_summary(self) -> DataFrame:
                value = "00:00:01.0" if self.path == "a.rpt" else "00:00:01.1"
                return DataFrame({"value": [value]})

        with patch("swmm_bench.comparator.Report", RoutingTimeStepReport):
            comparison = compare_rpts(
                "a.rpt", "b.rpt", "a", "b", "model.inp", "model.inp"
            )

        self.assertEqual(comparison.overall_distance, 0.0)
        self.assertEqual(comparison.section_comparisons, [])

    def test_missing_optional_sections_are_skipped(self) -> None:
        class OptionalReport:
            def __init__(self, _path: str):
                self._sections: dict[str, str] = {}

            @property
            def optional(self) -> DataFrame:
                return self._sections["Optional Table"]

        with patch("swmm_bench.comparator.Report", OptionalReport):
            self.assertEqual(_extract_report_tables("missing.rpt"), ({}, {}, [], []))

    def test_available_table_parse_failures_warn_and_skip_table(self) -> None:
        class BrokenReport:
            def __init__(self, _path: str):
                self._sections = {"Broken Table": "present"}

            @property
            def broken(self) -> DataFrame:
                raise ValueError("bad table")

        with patch("swmm_bench.comparator.Report", BrokenReport):
            with self.assertWarnsRegex(UserWarning, "missing.rpt.*broken.*bad table"):
                tables, skipped, errors, report_warnings = _extract_report_tables(
                    "missing.rpt"
                )

        self.assertEqual(tables, {})
        self.assertIn("broken", skipped)
        self.assertIn("bad table", skipped["broken"])
        self.assertEqual(errors, [])
        self.assertEqual(report_warnings, [])

    def test_duplicate_label_failure_warns_and_skips_table(self) -> None:
        class DuplicateReport:
            def __init__(self, _path: str):
                self._sections: dict[str, str] = {}

            @property
            def duplicate(self) -> DataFrame:
                return DataFrame({"value": [1.0, 2.0]}, index=["row", "row"])

        with patch("swmm_bench.comparator.Report", DuplicateReport):
            with self.assertWarnsRegex(
                UserWarning,
                "duplicate.rpt.*duplicate.*duplicate index labels",
            ):
                tables, skipped, errors, report_warnings = _extract_report_tables(
                    "duplicate.rpt"
                )

        self.assertEqual(tables, {})
        self.assertIn("duplicate", skipped)
        self.assertEqual(errors, [])
        self.assertEqual(report_warnings, [])

    def test_compare_rpts_captures_report_errors_and_warnings(self) -> None:
        class Message:
            def __init__(self, code: str, message: str):
                self.code = code
                self.message = message

        class DiagnosticReport:
            def __init__(self, path: str):
                self.path = path

            @property
            def table(self) -> DataFrame:
                return DataFrame({"value": [1.0]})

            @property
            def errors(self):
                return (
                    [Message("101", "simulation failed")]
                    if self.path.endswith("a.rpt")
                    else []
                )

            @property
            def warnings(self):
                return [Message("09", "model warning")]

        with patch("swmm_bench.comparator.Report", DiagnosticReport):
            comparison = compare_rpts(
                "a.rpt", "b.rpt", "a", "b", "model.inp", "model.inp"
            )

        self.assertIn("ERROR 101: simulation failed", comparison.report_errors)
        self.assertIn("WARNING 09: model warning", comparison.report_warnings)
        self.assertEqual(
            comparison.report_errors_a,
            ["ERROR 101: simulation failed"],
        )
        self.assertEqual(comparison.report_errors_b, [])
        self.assertEqual(
            comparison.report_warnings_a,
            ["WARNING 09: model warning"],
        )
        self.assertEqual(
            comparison.report_warnings_b,
            ["WARNING 09: model warning"],
        )

    def test_compare_rpts_warns_and_excludes_unparsed_tables(self) -> None:
        class PartlyBrokenReport:
            def __init__(self, path: str):
                self.path = path

            @property
            def good(self) -> DataFrame:
                return DataFrame({"value": [1.0]})

            @property
            def broken(self) -> DataFrame:
                if self.path.endswith("a.rpt"):
                    raise ValueError("bad table")
                return DataFrame({"value": [99.0]})

        with patch("swmm_bench.comparator.Report", PartlyBrokenReport):
            with self.assertWarnsRegex(UserWarning, "a.rpt.*broken.*bad table"):
                comparison = compare_rpts(
                    "a.rpt",
                    "b.rpt",
                    "a",
                    "b",
                    "model.inp",
                    "model.inp",
                )

        self.assertEqual(comparison.overall_distance, 0.0)
        self.assertEqual(
            [
                (section.section_name, section.note)
                for section in comparison.section_comparisons
            ],
            [
                ("good", None),
                ("broken", "Table could not be parsed for a; parsed only for b."),
            ],
        )
        self.assertIn("bad table", comparison.report_warnings[0])

    def test_compare_rpts_uses_swmm_pandas_tables_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_a = root / "a.rpt"
            report_b = root / "b.rpt"
            _write_report(report_a, 1.0)
            _write_report(report_b, 2.0)

            forward = compare_rpts(
                report_a,
                report_b,
                "a",
                "b",
                "model.inp",
                "model.inp",
            )
            reverse = compare_rpts(
                report_b,
                report_a,
                "b",
                "a",
                "model.inp",
                "model.inp",
            )

        self.assertGreater(forward.overall_distance, 0.0)
        self.assertEqual(forward.overall_distance, reverse.overall_distance)
        self.assertIn(
            "flow_routing_continuity",
            [item.section_name for item in forward.section_comparisons],
        )

    def test_compare_all_parses_each_unique_report_once_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_paths = [root / f"{name}.rpt" for name in ("a", "b", "c")]
            for report_path in report_paths:
                report_path.write_text("report\n", encoding="utf-8")

            barrier = threading.Barrier(len(report_paths))

            def extract(_path: Path):
                barrier.wait(timeout=2)
                return {}, {}, [], []

            def result(engine_name: str, report_path: Path) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=str(report_path),
                    stdout="",
                    stderr="",
                    error=None,
                )

            with (
                patch(
                    "swmm_bench.comparator._extract_report_tables", side_effect=extract
                ) as extract_report,
                patch(
                    "swmm_bench.comparator._parse_executor",
                    side_effect=lambda max_workers, **_kwargs: ThreadPoolExecutor(
                        max_workers
                    ),
                ),
            ):
                comparisons = compare_all(
                    [
                        result(engine_name, report_path)
                        for engine_name, report_path in zip(
                            ("a", "b", "c"), report_paths
                        )
                    ],
                    parse_workers=3,
                )

        self.assertEqual(extract_report.call_count, 3)
        self.assertEqual(
            [(item.engine_a, item.engine_b) for item in comparisons],
            [("a", "b"), ("a", "c"), ("b", "c")],
        )

    def test_compare_all_reemits_worker_warnings_in_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_paths = [root / f"{name}.rpt" for name in ("a", "b")]
            for report_path in report_paths:
                report_path.write_text("report\n", encoding="utf-8")

            barrier = threading.Barrier(len(report_paths))
            second_finished = threading.Event()

            def load(path: Path):
                barrier.wait(timeout=2)
                if path == report_paths[0]:
                    self.assertTrue(second_finished.wait(timeout=2))
                else:
                    second_finished.set()
                return ({}, {}, [], []), [f"warning-{path.stem}"]

            def result(engine_name: str, report_path: Path) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=str(report_path),
                    stdout="",
                    stderr="",
                    error=None,
                )

            with (
                patch(
                    "swmm_bench.comparator._extract_report_tables_for_cache",
                    side_effect=load,
                ),
                patch(
                    "swmm_bench.comparator._parse_executor",
                    side_effect=lambda max_workers, **_kwargs: ThreadPoolExecutor(
                        max_workers
                    ),
                ),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                comparisons = compare_all(
                    [
                        result(engine_name, report_path)
                        for engine_name, report_path in zip(
                            ("a", "b"), report_paths
                        )
                    ],
                    parse_workers=2,
                )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(
            [str(item.message) for item in caught], ["warning-a", "warning-b"]
        )

    def test_compare_all_process_pool_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_paths = [root / f"{name}.rpt" for name in ("a", "b")]
            for report_path in report_paths:
                _write_report(report_path, 1.0)

            def result(engine_name: str, report_path: Path) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=str(report_path),
                    stdout="",
                    stderr="",
                    error=None,
                )

            comparisons = compare_all(
                [
                    result(engine_name, report_path)
                    for engine_name, report_path in zip(("a", "b"), report_paths)
                ],
                parse_workers=2,
            )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].overall_distance, 0.0)

    def test_compare_all_reports_pair_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "same.rpt"
            _write_report(report, 1.0)

            def result(engine_name: str) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=str(report),
                    stdout="",
                    stderr="",
                    error=None,
                )

            events = []
            comparisons = compare_all(
                [result("a"), result("b")],
                progress_callback=events.append,
            )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual([event.status for event in events], ["started", "completed"])
        self.assertEqual(events[-1].completed, events[-1].total)

    def test_identical_reports_have_zero_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "same.rpt"
            _write_report(report, 1.0)
            comparison = compare_rpts(
                report,
                report,
                "a",
                "b",
                "model.inp",
                "model.inp",
            )

        self.assertEqual(comparison.overall_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
