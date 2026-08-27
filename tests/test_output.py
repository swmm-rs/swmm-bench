from __future__ import annotations

import math
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast
from unittest.mock import patch

from pandas import (  # pyright: ignore[reportMissingImports]
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Timestamp,
    date_range,
)
from pandas.testing import assert_frame_equal  # pyright: ignore[reportMissingImports]
from swmm.pandas import example_out_path  # pyright: ignore[reportMissingImports]

from swmm_bench.comparator import (
    _allocate_graph_payload_budgets,
    _cell_distance,
    _compare_output_frames,
    _graphical_series_payload_bytes,
    _report_graph_payload_budget,
    compare_all_outputs,
    compare_outs,
)
from swmm_bench.models import (
    EngineResult,
    OUTPUT_DISTANCE_METRIC_COMPOSITE,
)
from swmm_bench.output import extract_output_frame, extract_output_series


class OutputExtractionTests(unittest.TestCase):
    def test_report_size_reserves_space_for_non_graph_content(self) -> None:
        self.assertEqual(_report_graph_payload_budget(200), 190 * 1024 * 1024)
        self.assertEqual(_report_graph_payload_budget(100), 90 * 1024 * 1024)
        self.assertEqual(_report_graph_payload_budget(1), 512 * 1024)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            _report_graph_payload_budget(0)

    def test_graph_budget_allocation_is_order_independent_and_skips_invalid_pairs(
        self,
    ) -> None:
        index = date_range("2020-01-01", periods=3, freq="min")
        small_columns = MultiIndex.from_tuples([("node", "N1", "depth")])
        large_columns = MultiIndex.from_tuples(
            [("node", f"N{position}", "depth") for position in range(4)]
        )
        small = DataFrame([[1.0]] * 3, index=index, columns=small_columns)
        large = DataFrame([[1.0] * 4] * 3, index=index, columns=large_columns)
        paths = [Path(f"{name}.out") for name in ("a", "b", "c", "d")]
        frames = {paths[0]: small, paths[1]: small, paths[2]: large, paths[3]: large}
        pair_paths = [(paths[0], paths[1]), (None, paths[3]), (paths[2], paths[3])]

        budgets = _allocate_graph_payload_budgets(pair_paths, frames, 10_000)
        reversed_budgets = _allocate_graph_payload_budgets(
            list(reversed(pair_paths)), frames, 10_000
        )
        larger_budgets = _allocate_graph_payload_budgets(pair_paths, frames, 20_000)

        self.assertEqual(budgets[1], 0)
        self.assertGreater(budgets[2], budgets[0])
        self.assertEqual(budgets, list(reversed(reversed_budgets)))
        self.assertTrue(
            all(larger >= smaller for smaller, larger in zip(budgets, larger_budgets))
        )

    def test_unused_pair_graph_budget_is_redistributed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_paths = [root / f"{name}.out" for name in ("a", "b", "c")]
            for output_path in output_paths:
                output_path.write_bytes(b"output")

            columns = MultiIndex.from_tuples(
                [("node", f"N{position}", "depth") for position in range(12)]
            )
            index = date_range("2020-01-01", periods=50, freq="min")
            matching = DataFrame([[1.0] * 12] * 50, index=index, columns=columns)
            different = DataFrame(
                [[float(position + 2) for position in range(12)]] * 50,
                index=index,
                columns=columns,
            )
            frames = {
                output_paths[0]: matching,
                output_paths[1]: matching.copy(),
                output_paths[2]: different,
            }

            def result(engine_name: str, output_path: Path) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=None,
                    stdout="",
                    stderr="",
                    error=None,
                    out_path=str(output_path),
                )

            graph_budget = 30_000
            with (
                patch(
                    "swmm_bench.comparator.extract_output_frame",
                    side_effect=lambda path: frames[path],
                ),
                patch(
                    "swmm_bench.comparator._report_graph_payload_budget",
                    return_value=graph_budget,
                ),
            ):
                comparisons = compare_all_outputs(
                    [
                        result(engine_name, output_path)
                        for engine_name, output_path in zip(
                            ("a", "b", "c"), output_paths
                        )
                    ],
                    report_size_mb=1,
                )

        retained_payload_bytes = sum(
            _graphical_series_payload_bytes(comparison.graphical_series)
            for comparison in comparisons
            if comparison.graphical_series
        )
        self.assertGreater(retained_payload_bytes, 25_000)
        self.assertLessEqual(retained_payload_bytes, graph_budget)

    def test_package_fixture_extracts_a_semantic_wide_frame(self) -> None:
        preloaded = extract_output_frame(example_out_path)
        streamed = extract_output_frame(example_out_path, preload=False)

        self.assertIsInstance(preloaded.index, DatetimeIndex)
        self.assertEqual(preloaded.index.name, "datetime")
        self.assertIsInstance(preloaded.columns, MultiIndex)
        self.assertEqual(
            preloaded.columns.names,
            ["element_type", "element_name", "attribute"],
        )
        self.assertEqual(
            set(preloaded.columns.get_level_values("element_type")),
            {"subcatchment", "node", "link", "system"},
        )
        self.assertIn(("node", "JUNC1", "hydraulic_head"), preloaded.columns)
        self.assertIn(("system", "system", "outfall_flows"), preloaded.columns)
        assert_frame_equal(preloaded, streamed, check_dtype=False)

    def test_output_series_preserve_each_semantic_column(self) -> None:
        frame = extract_output_frame(example_out_path)
        series = extract_output_series(example_out_path)

        self.assertEqual(len(series), len(frame.columns))
        self.assertTrue(
            all(
                table.columns.equals(frame.loc[:, [column]].columns)
                for column, table in zip(frame.columns, series.values())
            )
        )

    def test_same_output_has_zero_distance(self) -> None:
        comparison = compare_outs(
            example_out_path,
            example_out_path,
            "a",
            "b",
            "model.inp",
            "model.inp",
        )

        self.assertEqual(comparison.metric, OUTPUT_DISTANCE_METRIC_COMPOSITE)
        self.assertEqual(comparison.overall_distance, 0.0)
        self.assertEqual(comparison.section_comparisons, [])
        self.assertEqual(comparison.graphical_series, [])
        self.assertFalse(comparison.details_retained)

    def test_graphical_series_can_be_forced_at_zero_distance(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        frame = DataFrame(
            [[1.0], [2.0]],
            index=date_range("2026-08-03", periods=2, freq="5min"),
            columns=columns,
        )

        comparison = _compare_output_frames(
            frame,
            frame,
            "a",
            "b",
            "model.inp",
            "model.inp",
            retain_tabular=False,
            include_all_comparisons=True,
        )

        self.assertEqual(comparison.overall_distance, 0.0)
        self.assertEqual(len(comparison.graphical_series), 1)
        self.assertTrue(comparison.details_retained)

    def test_graphical_series_aligns_semantic_values_and_missing_timestamps(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "hydraulic_head")],
            names=["element_type", "element_name", "attribute"],
        )
        frame_a = DataFrame(
            [[1.0], [2.0]],
            index=DatetimeIndex(
                [Timestamp("2020-01-01"), Timestamp("2020-01-01 00:05")]
            ),
            columns=columns,
        )
        frame_b = DataFrame(
            [[2.5], [3.0]],
            index=DatetimeIndex(
                [Timestamp("2020-01-01 00:05"), Timestamp("2020-01-01 00:10")]
            ),
            columns=columns,
        )

        comparison = _compare_output_frames(
            frame_a,
            frame_b,
            "a",
            "b",
            "model.inp",
            "model.inp",
        )

        series = comparison.graphical_series[0]
        self.assertEqual(
            (series.element_type, series.element_name, series.attribute),
            ("node", "J1", "hydraulic_head"),
        )
        self.assertEqual(
            series.timestamps,
            [
                "2020-01-01T00:00:00",
                "2020-01-01T00:05:00",
                "2020-01-01T00:10:00",
            ],
        )
        self.assertEqual(series.values_a, [1.0, 2.0, None])
        self.assertEqual(series.values_b, [None, 2.5, 3.0])
        self.assertEqual(series.source_point_count, 3)
        section = comparison.section_comparisons[0]
        self.assertIsNotNone(section.numeric_distance)
        self.assertIsNotNone(section.missing_fraction)
        self.assertAlmostEqual(cast(float, section.numeric_distance), 0.2)
        self.assertAlmostEqual(cast(float, section.missing_fraction), 0.5)
        self.assertAlmostEqual(section.distance, 0.5 + 0.5 * 0.2)
        self.assertEqual(section.finite_pair_count, 1)
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(comparison.timeline_coverage.trailing_timestamp_count_b, 1)
        self.assertEqual(series.distance, section.distance)
        self.assertFalse(series.sampled)

    def test_trailing_periods_are_coverage_not_value_distance(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        shared_index = date_range("2020-01-01", periods=3, freq="5min")
        extended_index = date_range("2020-01-01", periods=6, freq="5min")
        frame_a = DataFrame([1.0, 2.0, 3.0], index=shared_index, columns=columns)
        frame_b = DataFrame(
            [1.0, 2.0, 3.0, 40.0, 50.0, 60.0],
            index=extended_index,
            columns=columns,
        )

        forward = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        reverse = _compare_output_frames(frame_b, frame_a, "b", "a", "m", "m")
        frame_b_with_shared_difference = frame_b.copy()
        frame_b_with_shared_difference.iloc[1, 0] = 20.0
        differing = _compare_output_frames(
            frame_a, frame_b_with_shared_difference, "a", "b", "m", "m"
        )

        self.assertEqual(forward.metric, OUTPUT_DISTANCE_METRIC_COMPOSITE)
        self.assertEqual(forward.overall_distance, 0.0)
        self.assertEqual(reverse.overall_distance, 0.0)
        self.assertEqual(forward.section_comparisons[0].missing_fraction, 0.0)
        self.assertEqual(forward.timeline_coverage.timestamp_count_a, 3)
        self.assertEqual(forward.timeline_coverage.timestamp_count_b, 6)
        self.assertEqual(forward.timeline_coverage.shared_timestamp_count, 3)
        self.assertEqual(forward.timeline_coverage.trailing_timestamp_count_a, 0)
        self.assertEqual(forward.timeline_coverage.trailing_timestamp_count_b, 3)
        self.assertEqual(reverse.timeline_coverage.trailing_timestamp_count_a, 3)
        self.assertEqual(reverse.timeline_coverage.trailing_timestamp_count_b, 0)
        self.assertGreater(differing.overall_distance, 0.0)

    def test_internal_gap_is_penalized_while_trailing_periods_are_coverage(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        frame_a = DataFrame(
            [1.0, 3.0, 4.0],
            index=DatetimeIndex(
                [
                    Timestamp("2020-01-01 00:00"),
                    Timestamp("2020-01-01 00:10"),
                    Timestamp("2020-01-01 00:15"),
                ]
            ),
            columns=columns,
        )
        frame_b = DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=date_range("2020-01-01", periods=5, freq="5min"),
            columns=columns,
        )

        forward = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        reverse = _compare_output_frames(frame_b, frame_a, "b", "a", "m", "m")

        self.assertEqual(forward.overall_distance, 0.25)
        self.assertEqual(reverse.overall_distance, 0.25)
        section = forward.section_comparisons[0]
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(section.timestamp_count, 4)
        self.assertEqual(forward.timeline_coverage.trailing_timestamp_count_b, 1)
        self.assertEqual(reverse.timeline_coverage.trailing_timestamp_count_a, 1)

    def test_output_distance_blends_typical_and_event_error(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame_a = DataFrame([1.0, 2.0, 3.0], index=index, columns=columns)
        frame_b = DataFrame([1.0, 2.0, 4.0], index=index, columns=columns)

        forward = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        reverse = _compare_output_frames(frame_b, frame_a, "b", "a", "m", "m")
        scaled = _compare_output_frames(
            frame_a * 1000.0,
            frame_b * 1000.0,
            "a",
            "b",
            "m",
            "m",
        )

        event_distance = 1.0 / math.sqrt(21.0)
        typical_distance = 1.0 / 12.0
        expected = 0.75 * typical_distance + 0.25 * event_distance
        self.assertAlmostEqual(forward.overall_distance, expected)
        self.assertAlmostEqual(reverse.overall_distance, expected)
        self.assertAlmostEqual(scaled.overall_distance, expected)
        section = forward.section_comparisons[0]
        self.assertAlmostEqual(cast(float, section.typical_distance), typical_distance)
        self.assertAlmostEqual(cast(float, section.event_distance), event_distance)
        self.assertAlmostEqual(cast(float, section.numeric_distance), event_distance)

        typical_only = _compare_output_frames(
            frame_a, frame_b, "a", "b", "m", "m", typical_weight=1.0
        )
        event_only = _compare_output_frames(
            frame_a, frame_b, "a", "b", "m", "m", typical_weight=0.0
        )
        self.assertAlmostEqual(typical_only.overall_distance, typical_distance)
        self.assertAlmostEqual(event_only.overall_distance, event_distance)

    def test_event_scale_prevents_near_zero_noise_from_dominating(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow_rate")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("1977-07-15", periods=8, freq="10min")
        values_a = [
            0.0,
            -237.9263,
            -415.4236,
            27.6930,
            0.03813,
            0.02513,
            -0.03188,
            0.00914,
        ]
        values_b = [
            0.0,
            -236.7247,
            -417.1129,
            27.7395,
            0.00182,
            -0.05181,
            0.00210,
            0.01837,
        ]
        frame_a = DataFrame(values_a, index=index, columns=columns)
        frame_b = DataFrame(values_b, index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        legacy_distance = sum(
            _cell_distance(value_a, value_b)[0]
            for value_a, value_b in zip(values_a, values_b)
        ) / len(values_a)

        self.assertGreater(legacy_distance, 0.4)
        self.assertLess(comparison.overall_distance, 0.01)
        self.assertEqual(comparison.section_comparisons[0].missing_fraction, 0.0)

    def test_paired_nulls_are_neutral_and_one_sided_nulls_are_missing(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame_a = DataFrame([0.0, None, 1.0], index=index, columns=columns)
        frame_b = DataFrame([0.0, None, None], index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        section = comparison.section_comparisons[0]

        self.assertEqual(section.numeric_distance, 0.0)
        self.assertIsNotNone(section.missing_fraction)
        self.assertAlmostEqual(cast(float, section.missing_fraction), 1.0 / 3.0)
        self.assertAlmostEqual(section.distance, 1.0 / 3.0)
        self.assertEqual(section.finite_pair_count, 1)
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(section.both_null_count, 1)

    def test_absent_series_has_full_penalty_and_series_are_equally_weighted(
        self,
    ) -> None:
        columns_a = MultiIndex.from_tuples(
            [("node", "J1", "depth"), ("node", "J1", "volume")],
            names=["element_type", "element_name", "attribute"],
        )
        columns_b = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=4, freq="min")
        frame_a = DataFrame(
            [[1.0, 10.0], [1.0, 11.0], [1.0, 12.0], [1.0, 13.0]],
            index=index,
            columns=columns_a,
        )
        frame_b = DataFrame([1.0, 1.0, 1.0, 1.0], index=index, columns=columns_b)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        distances = {
            section.section_name: section.distance
            for section in comparison.section_comparisons
        }

        self.assertEqual(distances['["node","J1","depth"]'], 0.0)
        self.assertEqual(distances['["node","J1","volume"]'], 1.0)
        self.assertEqual(comparison.overall_distance, 0.5)

    def test_nonfinite_output_value_is_missing_and_scores_finitely(self) -> None:
        columns = MultiIndex.from_tuples(
            [("system", "system", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=1, freq="min")
        frame_a = DataFrame([math.inf], index=index, columns=columns)
        frame_b = DataFrame([1.0], index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        section = comparison.section_comparisons[0]

        self.assertEqual(section.distance, 1.0)
        self.assertEqual(section.missing_fraction, 1.0)
        self.assertTrue(math.isfinite(comparison.overall_distance))
        self.assertEqual(section.differences[0]["value_a"], "inf")

    def test_output_diagnostics_are_bounded_and_keep_missing_values_first(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=150, freq="min")
        frame_a = DataFrame([None] * 10 + [0.0] * 140, index=index, columns=columns)
        frame_b = DataFrame([1.0] * 150, index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        section = comparison.section_comparisons[0]

        self.assertEqual(section.difference_count, 150)
        self.assertEqual(len(section.differences), 100)
        self.assertTrue(section.differences_truncated)
        self.assertTrue(
            all(row["issue"] == "null value in A" for row in section.differences[:10])
        )
        self.assertTrue(
            all(
                row["issue"] == "numeric difference" for row in section.differences[10:]
            )
        )

    def test_output_comparison_does_not_use_report_table_metric(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=2, freq="min")
        frame = DataFrame([1.0, 2.0], index=index, columns=columns)

        with patch("swmm_bench.comparator._compare_tables", side_effect=AssertionError):
            comparison = _compare_output_frames(frame, frame.copy(), "a", "b", "m", "m")

        self.assertEqual(comparison.overall_distance, 0.0)

    def test_graphical_series_retains_every_source_timestep(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=600, freq="min")
        values_a = [0.0] * 600
        values_b = [0.0] * 600
        values_b[300] = 1.0
        frame_a = DataFrame(values_a, index=index, columns=columns)
        frame_b = DataFrame(values_b, index=index, columns=columns)

        comparison = _compare_output_frames(
            frame_a,
            frame_b,
            "a",
            "b",
            "model.inp",
            "model.inp",
        )

        self.assertEqual(comparison.overall_distance, 0.25125)
        self.assertEqual(comparison.graphical_series[0].distance, 0.25125)
        self.assertEqual(comparison.graphical_series[0].typical_distance, 1.0 / 600.0)
        self.assertEqual(comparison.graphical_series[0].event_distance, 1.0)
        self.assertEqual(len(comparison.graphical_series[0].timestamps), 600)
        self.assertFalse(comparison.graphical_series[0].sampled)

    def test_report_size_budget_retains_highest_distance_series(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", f"N{index}", "depth") for index in range(5)],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=100, freq="min")
        frame = DataFrame([[1.0] * 5] * 100, index=index, columns=columns)
        compared = DataFrame(
            [[1.0, 1.1, 2.0, 10.0, 100.0]] * 100,
            index=index,
            columns=columns,
        )
        payload_budget = 8_000

        comparison = _compare_output_frames(
            frame,
            compared,
            "a",
            "b",
            "model.inp",
            "model.inp",
            graph_payload_budget_bytes=payload_budget,
        )

        self.assertEqual(
            [series.element_name for series in comparison.graphical_series],
            ["N4", "N3"],
        )
        self.assertTrue(
            all(len(series.timestamps) == 100 for series in comparison.graphical_series)
        )
        self.assertTrue(
            all(not series.sampled for series in comparison.graphical_series)
        )
        self.assertLessEqual(
            _graphical_series_payload_bytes(comparison.graphical_series),
            payload_budget,
        )
        self.assertIn(
            "2 of 5 highest-distance output series",
            comparison.graphical_unavailable_reason or "",
        )

    def test_graphical_payload_is_omitted_when_complete_series_exceed_budget(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [
                ("node", "J1", "depth"),
                ("node", "J1", "flow"),
                ("node", "J1", "volume"),
            ],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame = DataFrame(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]],
            index=index,
            columns=columns,
        )
        compared = frame.copy()
        compared.iloc[0, 0] = 10.0
        with patch("swmm_bench.comparator._DEFAULT_GRAPH_PAYLOAD_BUDGET_BYTES", 4):
            comparison = _compare_output_frames(
                frame,
                compared,
                "a",
                "b",
                "model.inp",
                "model.inp",
            )

        self.assertEqual(comparison.graphical_series, [])
        self.assertIn(
            "insufficient chart payload budget",
            comparison.graphical_unavailable_reason or "",
        )
        self.assertEqual(len(comparison.section_comparisons), 3)
        self.assertGreater(comparison.overall_distance, 0.01)

    def test_graph_payload_is_omitted_instead_of_hiding_interior_divergence(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth"), ("node", "J2", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame_a = DataFrame(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            index=index,
            columns=columns,
        )
        frame_b = DataFrame(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            index=index,
            columns=columns,
        )

        with patch("swmm_bench.comparator._DEFAULT_GRAPH_PAYLOAD_BUDGET_BYTES", 4):
            comparison = _compare_output_frames(
                frame_a,
                frame_b,
                "a",
                "b",
                "model.inp",
                "model.inp",
            )

        self.assertEqual(comparison.overall_distance, 0.5)
        self.assertEqual(comparison.graphical_series, [])
        self.assertIn(
            "chart payload budget", comparison.graphical_unavailable_reason or ""
        )

    def test_summary_omits_series_details(self) -> None:
        columns = MultiIndex.from_tuples([("node", "J1", "depth")])
        frame_a = DataFrame([[1.0], [2.0]], columns=columns)
        frame_b = DataFrame([[1.0], [3.0]], columns=columns)

        comparison = _compare_output_frames(
            frame_a,
            frame_b,
            "a",
            "b",
            "model.inp",
            "model.inp",
            retain_tabular=False,
            retain_graphical=False,
        )

        self.assertGreater(comparison.overall_distance, 0.0)
        self.assertEqual(comparison.section_comparisons, [])
        self.assertEqual(comparison.graphical_series, [])
        self.assertFalse(comparison.details_retained)

    def test_output_pairing_uses_available_output_artifacts(self) -> None:
        def result(engine_name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=None,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
                out_path=str(example_out_path),
            )

        comparisons = compare_all_outputs([result("a"), result("b")])

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].overall_distance, 0.0)

    def test_output_files_are_loaded_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_paths = [root / f"{name}.out" for name in ("a", "b")]
            for output_path in output_paths:
                output_path.write_bytes(b"output")

            frame = DataFrame(
                [[1.0]],
                index=DatetimeIndex([Timestamp("2020-01-01")], name="datetime"),
                columns=MultiIndex.from_tuples(
                    [("node", "N1", "depth")],
                    names=["element_type", "element_name", "attribute"],
                ),
            )
            barrier = threading.Barrier(len(output_paths))
            second_finished = threading.Event()
            events = []

            def extract(path: Path) -> DataFrame:
                barrier.wait(timeout=2)
                if path == output_paths[0]:
                    self.assertTrue(second_finished.wait(timeout=2))
                else:
                    second_finished.set()
                return frame.copy()

            def result(engine_name: str, output_path: Path) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=None,
                    stdout="",
                    stderr="",
                    error=None,
                    out_path=str(output_path),
                )

            with (
                patch(
                    "swmm_bench.comparator.extract_output_frame", side_effect=extract
                ) as extract_output,
                patch(
                    "swmm_bench.comparator._parse_executor",
                    side_effect=lambda max_workers, **_kwargs: ThreadPoolExecutor(
                        max_workers
                    ),
                ),
            ):
                comparisons = compare_all_outputs(
                    [
                        result(engine_name, output_path)
                        for engine_name, output_path in zip(
                            ("a", "b"), output_paths
                        )
                    ],
                    progress_callback=events.append,
                    retain_graphical=False,
                    parse_workers=2,
                )

        self.assertEqual(extract_output.call_count, 2)
        load_events = [event for event in events if event.phase == "output-load"]
        self.assertEqual(
            [event.item_name for event in load_events],
            [str(output_paths[0]), str(output_paths[1])] * 2,
        )
        self.assertEqual(
            [event.status for event in load_events],
            ["started", "started", "completed", "completed"],
        )
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].overall_distance, 0.0)

    def test_output_process_pool_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_paths = [root / f"{name}.out" for name in ("a", "b")]
            for output_path in output_paths:
                shutil.copy2(example_out_path, output_path)

            def result(engine_name: str, output_path: Path) -> EngineResult:
                return EngineResult(
                    engine_path=f"/{engine_name}",
                    engine_name=engine_name,
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path=None,
                    stdout="",
                    stderr="",
                    error=None,
                    out_path=str(output_path),
                )

            comparisons = compare_all_outputs(
                [
                    result(engine_name, output_path)
                    for engine_name, output_path in zip(("a", "b"), output_paths)
                ],
                retain_graphical=False,
                parse_workers=2,
            )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].overall_distance, 0.0)

    def test_output_comparison_reports_loading_series_and_graph_progress(self) -> None:
        def result(engine_name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
                out_path=str(example_out_path),
            )

        events = []
        comparisons = compare_all_outputs(
            [result("a"), result("b")],
            progress_callback=events.append,
        )

        self.assertEqual(len(comparisons), 1)
        load_events = [event for event in events if event.phase == "output-load"]
        self.assertEqual(
            [(event.completed, event.total) for event in load_events],
            [(0, 1), (1, 1)],
        )
        series_events = [event for event in events if event.phase == "output-series"]
        graph_events = [event for event in events if event.phase == "output-graph"]
        self.assertTrue(series_events)
        self.assertEqual(series_events[-1].completed, series_events[-1].total)
        self.assertEqual(graph_events, [])
        pair_events = [event for event in events if event.phase == "output-pair"]
        self.assertEqual(
            [event.status for event in pair_events], ["started", "completed"]
        )

    def test_failed_engine_output_is_not_compared(self) -> None:
        failed = EngineResult(
            engine_path="/failed",
            engine_name="failed",
            inp_path="model.inp",
            inp_name="model.inp",
            duration_s=1.0,
            peak_memory_mb=1.0,
            exit_code=1,
            rpt_path=None,
            stdout="",
            stderr="",
            error=None,
            out_path=str(example_out_path),
        )
        succeeded = EngineResult(
            engine_path="/succeeded",
            engine_name="succeeded",
            inp_path="model.inp",
            inp_name="model.inp",
            duration_s=1.0,
            peak_memory_mb=1.0,
            exit_code=0,
            rpt_path=None,
            stdout="",
            stderr="",
            error=None,
            out_path=str(example_out_path),
        )

        self.assertEqual(compare_all_outputs([failed, succeeded]), [])

    def test_unreadable_output_is_skipped_without_aborting_other_results(self) -> None:
        def result(engine_name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
                out_path=str(example_out_path),
            )

        with patch(
            "swmm_bench.comparator.extract_output_frame",
            side_effect=ValueError("truncated"),
        ):
            with self.assertWarnsRegex(UserWarning, "Skipping unreadable output"):
                comparisons = compare_all_outputs([result("a"), result("b")])

        self.assertEqual(comparisons, [])

    def test_two_zero_period_outputs_are_not_reported_as_identical(self) -> None:
        with patch(
            "swmm_bench.comparator.extract_output_frame", return_value=DataFrame()
        ):
            with self.assertRaisesRegex(ValueError, "Neither output file"):
                compare_outs("a.out", "b.out", "a", "b", "model.inp", "model.inp")

    def test_zero_period_output_never_queries_series(self) -> None:
        class ZeroPeriodOutput:
            period = 0

            def __init__(self, _path: str, *, preload: bool) -> None:
                self.preload = preload

            def __enter__(self) -> "ZeroPeriodOutput":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        with patch("swmm_bench.output.Output", ZeroPeriodOutput):
            frame = extract_output_frame("empty.out")

        self.assertTrue(frame.empty)

    def test_empty_categories_are_skipped_and_singletons_are_semantic(self) -> None:
        class SingletonOutput:
            period = 2
            subcatchments: tuple[str, ...] = ()
            nodes = ("N1",)
            links: tuple[str, ...] = ()
            project_size = (0, 1, 0, 0, 0)

            def __init__(self, _path: str, *, preload: bool) -> None:
                self.preload = preload

            def __enter__(self) -> "SingletonOutput":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def node_series(
                self,
                node: str,
                *,
                attribute: None,
                columns: str,
            ) -> DataFrame:
                if node != "N1" or attribute is not None or columns != "attr":
                    raise AssertionError("unexpected node-series query")
                return DataFrame(
                    {"hydraulic_head": [1.0, 2.0]},
                    index=[Timestamp("2020-01-01"), Timestamp("2020-01-01 00:05")],
                )

        with patch("swmm_bench.output.Output", SingletonOutput):
            frame = extract_output_frame(Path("singleton.out"))

        self.assertEqual(
            list(frame.columns),
            [("node", "N1", "hydraulic_head")],
        )
        self.assertEqual(frame.index.name, "datetime")


if __name__ == "__main__":
    unittest.main()
