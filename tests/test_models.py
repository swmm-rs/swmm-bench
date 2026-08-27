from __future__ import annotations

import unittest

from swmm_bench.models import (
    BenchmarkResult,
    BenchmarkSample,
    EngineResult,
    OUTPUT_DISTANCE_METRIC_COMPOSITE,
    OUTPUT_DISTANCE_METRIC_LEGACY,
    OUTPUT_DISTANCE_METRIC_NRMSE,
    OutputComparison,
    OutputSectionComparison,
    OutputSeriesComparison,
    OutputTimelineCoverage,
)


class ModelCompatibilityTests(unittest.TestCase):
    def test_schema_one_payload_loads_without_output_fields(self) -> None:
        data = {
            "schema_version": "1",
            "name": "legacy",
            "timestamp": "2026-07-19T00:00:00+00:00",
            "platform": {"host": "test"},
            "engine_results": [
                {
                    "engine_path": "/tmp/swmm",
                    "engine_name": "swmm",
                    "inp_path": "model.inp",
                    "inp_name": "model.inp",
                    "duration_s": 1.0,
                    "peak_memory_mb": 2.0,
                    "exit_code": 0,
                    "rpt_path": "/tmp/result.rpt",
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                }
            ],
            "comparisons": [
                {
                    "inp_path": "model.inp",
                    "inp_name": "model.inp",
                    "engine_a": "a",
                    "engine_b": "b",
                    "overall_distance": 0.5,
                    "section_comparisons": [
                        {
                            "section_name": "table",
                            "distance": 0.5,
                            "row_count_a": 1,
                            "row_count_b": 1,
                            "differences": [{"row": "row"}],
                        }
                    ],
                }
            ],
        }

        result = BenchmarkResult.from_dict(data)

        self.assertIsNone(result.engine_results[0].out_path)
        self.assertEqual(result.output_comparisons, [])
        comparison = result.comparisons[0].section_comparisons[0]
        self.assertEqual(comparison.difference_count, 1)
        self.assertFalse(comparison.differences_truncated)


    def test_repeated_benchmark_samples_round_trip(self) -> None:
        result = BenchmarkResult(
            schema_version="7",
            name="repeated",
            timestamp="2026-08-22T00:00:00+00:00",
            platform={},
            engine_results=[
                EngineResult(
                    engine_path="/engine",
                    engine_name="engine",
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=2.0,
                    peak_memory_mb=10.0,
                    exit_code=0,
                    rpt_path="/result.rpt",
                    stdout="",
                    stderr="",
                    error=None,
                    samples=[BenchmarkSample(2.0, 10.0, 0, None)],
                    representative_sample=1,
                )
            ],
            comparisons=[],
            run_count=1,
            run_order="interleaved",
        )

        loaded = BenchmarkResult.from_dict(result.to_dict())

        self.assertEqual(loaded.run_count, 1)
        self.assertEqual(loaded.run_order, "interleaved")
        self.assertEqual(loaded.engine_results[0].samples[0].duration_s, 2.0)
        self.assertEqual(loaded.engine_results[0].representative_sample, 1)


    def test_schema_two_output_comparison_loads_without_graphical_series(self) -> None:
        result = BenchmarkResult.from_dict(
            {
                "schema_version": "2",
                "name": "legacy-output",
                "timestamp": "2026-07-19T00:00:00+00:00",
                "platform": {},
                "engine_results": [],
                "comparisons": [],
                "output_comparisons": [
                    {
                        "inp_path": "model.inp",
                        "inp_name": "model.inp",
                        "engine_a": "a",
                        "engine_b": "b",
                        "overall_distance": 0.25,
                        "section_comparisons": [],
                    }
                ],
            }
        )

        comparison = result.output_comparisons[0]
        self.assertIsInstance(comparison, OutputComparison)
        self.assertEqual(comparison.graphical_series, [])
        self.assertEqual(comparison.metric, OUTPUT_DISTANCE_METRIC_LEGACY)

    def test_invalid_graphical_payload_is_omitted(self) -> None:
        result = BenchmarkResult.from_dict(
            {
                "schema_version": "3",
                "name": "invalid-graph",
                "timestamp": "2026-07-20T00:00:00+00:00",
                "platform": {},
                "engine_results": [],
                "comparisons": [],
                "output_comparisons": [
                    {
                        "inp_path": "model.inp",
                        "inp_name": "model.inp",
                        "engine_a": "a",
                        "engine_b": "b",
                        "overall_distance": 0.5,
                        "section_comparisons": [],
                        "graphical_series": [
                            {
                                "element_type": 1,
                                "element_name": "J1",
                                "attribute": "depth",
                                "distance": 0.5,
                                "row_count_a": 1,
                                "row_count_b": 1,
                                "timestamps": None,
                                "values_a": [float("nan")],
                                "values_b": [1.0],
                                "source_point_count": 1,
                            }
                        ],
                    }
                ],
            }
        )

        comparison = result.output_comparisons[0]
        self.assertEqual(comparison.graphical_series, [])
        self.assertIn(
            "persisted payload is invalid",
            comparison.graphical_unavailable_reason or "",
        )
        self.assertEqual(comparison.metric, OUTPUT_DISTANCE_METRIC_LEGACY)

    def test_empty_graphical_series_is_omitted(self) -> None:
        comparison = OutputComparison.from_dict(
            {
                "inp_path": "model.inp",
                "inp_name": "model.inp",
                "engine_a": "a",
                "engine_b": "b",
                "overall_distance": 0.0,
                "section_comparisons": [],
                "graphical_series": [
                    {
                        "element_type": "node",
                        "element_name": "J1",
                        "attribute": "depth",
                        "distance": 0.0,
                        "row_count_a": 0,
                        "row_count_b": 0,
                        "timestamps": [],
                        "values_a": [],
                        "values_b": [],
                        "source_point_count": 0,
                    }
                ],
            }
        )

        self.assertEqual(comparison.graphical_series, [])
        self.assertIn(
            "persisted payload is invalid",
            comparison.graphical_unavailable_reason or "",
        )

    def test_schema_four_metric_and_graphical_series_round_trip(self) -> None:
        result = BenchmarkResult(
            schema_version="4",
            name="graphical-output",
            timestamp="2026-07-20T00:00:00+00:00",
            platform={},
            engine_results=[],
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.5,
                    section_comparisons=[
                        OutputSectionComparison(
                            section_name='["node","J1","depth"]',
                            distance=0.5,
                            row_count_a=3,
                            row_count_b=3,
                            differences=[],
                            numeric_distance=0.4,
                            missing_fraction=1.0 / 6.0,
                            finite_pair_count=2,
                            missing_count=1,
                            both_null_count=0,
                            timestamp_count=3,
                        )
                    ],
                    timeline_coverage=OutputTimelineCoverage(
                        timestamp_count_a=3,
                        timestamp_count_b=5,
                        shared_timestamp_count=3,
                        trailing_timestamp_count_b=2,
                    ),
                    graphical_series=[
                        OutputSeriesComparison(
                            element_type="node",
                            element_name="J1",
                            attribute="depth",
                            distance=0.5,
                            row_count_a=3,
                            row_count_b=3,
                            timestamps=["2026-01-01T00:00:00", "2026-01-01T00:05:00"],
                            values_a=[1.0, None],
                            values_b=[1.5, 2.0],
                            source_point_count=3,
                            sampled=True,
                        )
                    ],
                )
            ],
        )

        loaded = BenchmarkResult.from_dict(result.to_dict())

        comparison = loaded.output_comparisons[0]
        self.assertEqual(comparison.metric, OUTPUT_DISTANCE_METRIC_NRMSE)
        self.assertEqual(comparison.timeline_coverage.timestamp_count_a, 3)
        self.assertEqual(comparison.timeline_coverage.timestamp_count_b, 5)
        self.assertEqual(comparison.timeline_coverage.shared_timestamp_count, 3)
        self.assertEqual(
            comparison.timeline_coverage.trailing_timestamp_count_b, 2
        )
        section = comparison.section_comparisons[0]
        self.assertEqual(section.numeric_distance, 0.4)
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(section.timestamp_count, 3)
        series = comparison.graphical_series[0]
        self.assertEqual(series.element_type, "node")
        self.assertEqual(series.values_a, [1.0, None])
        self.assertEqual(series.source_point_count, 3)
        self.assertTrue(series.sampled)

    def test_composite_output_metric_round_trips_components_and_weight(self) -> None:
        result = BenchmarkResult(
            schema_version="5",
            name="composite-output",
            timestamp="2026-08-20T00:00:00+00:00",
            platform={},
            engine_results=[],
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.25,
                    section_comparisons=[],
                    metric=OUTPUT_DISTANCE_METRIC_COMPOSITE,
                    typical_weight=0.0,
                    typical_distance=0.1,
                    event_distance=0.25,
                    graphical_series=[
                        OutputSeriesComparison(
                            element_type="node",
                            element_name="J1",
                            attribute="depth",
                            distance=0.25,
                            row_count_a=1,
                            row_count_b=1,
                            timestamps=["2026-01-01T00:00:00"],
                            values_a=[1.0],
                            values_b=[0.0],
                            source_point_count=1,
                            typical_distance=0.1,
                            event_distance=0.25,
                        )
                    ],
                )
            ],
        )

        comparison = BenchmarkResult.from_dict(result.to_dict()).output_comparisons[0]

        self.assertEqual(comparison.metric, OUTPUT_DISTANCE_METRIC_COMPOSITE)
        self.assertEqual(comparison.typical_weight, 0.0)
        self.assertEqual(comparison.typical_distance, 0.1)
        self.assertEqual(comparison.event_distance, 0.25)
        self.assertEqual(comparison.graphical_series[0].typical_distance, 0.1)
        self.assertEqual(comparison.graphical_series[0].event_distance, 0.25)

    def test_timeline_coverage_rejects_invalid_counts(self) -> None:
        for field, value in (
            ("timestamp_count_a", -1),
            ("shared_timestamp_count", True),
            ("trailing_timestamp_count_b", None),
            ("trailing_timestamp_count_a", "3"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    OutputTimelineCoverage.from_dict({field: value})

    def test_unknown_output_metric_identifier_is_preserved(self) -> None:
        comparison = OutputComparison.from_dict(
            {
                "inp_path": "model.inp",
                "inp_name": "model.inp",
                "engine_a": "a",
                "engine_b": "b",
                "overall_distance": 0.2,
                "section_comparisons": [],
                "graphical_series": [],
                "metric": "future-metric-v9",
            }
        )

        self.assertEqual(comparison.metric, "future-metric-v9")


if __name__ == "__main__":
    unittest.main()
