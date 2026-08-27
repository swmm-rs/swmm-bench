from __future__ import annotations

import math
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console

from swmm_bench.models import (
    BenchmarkResult,
    BenchmarkSample,
    EngineResult,
    OUTPUT_DISTANCE_METRIC_COMPOSITE,
    OUTPUT_DISTANCE_METRIC_LEGACY,
    ModelComparison,
    OutputComparison,
    OutputSectionComparison,
    OutputSeriesComparison,
    OutputTimelineCoverage,
    SectionComparison,
)
from swmm_bench.reporter import (
    _build_template_context,
    print_summary,
    render_html,
    save_json,
)


class ReporterTests(unittest.TestCase):
    def test_timing_highlights_exclude_failed_analyses(self) -> None:
        def engine_result(
            engine_name: str,
            model_name: str,
            duration_s: float,
            exit_code: int,
            error: str | None = None,
        ) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path=model_name,
                inp_name=model_name,
                duration_s=duration_s,
                peak_memory_mb=1.0,
                exit_code=exit_code,
                rpt_path=None,
                stdout="",
                stderr="failed" if exit_code or error else "",
                error=error,
            )

        result = BenchmarkResult(
            schema_version="5",
            name="failed-timing",
            timestamp="2026-07-20T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[
                engine_result("failed-fast", "mixed.inp", 0.25, 1),
                engine_result("completed", "mixed.inp", 1.0, 0),
                engine_result(
                    "failed-only", "failed-only.inp", 0.5, 0, "analysis failed"
                ),
            ],
            comparisons=[],
        )
        mocked_console = Mock()
        with patch("swmm_bench.reporter.console", mocked_console):
            print_summary(result)

        speed_table = mocked_console.print.call_args_list[1].args[0]
        output = StringIO()
        Console(file=output, width=120).print(speed_table)
        highlights = output.getvalue()

        self.assertIn("completed (1.000s)", highlights)
        self.assertNotIn("failed-fast", highlights)
        self.assertNotIn("failed-only.inp", highlights)


    def test_repeated_benchmark_renders_median_sample_statistics(self) -> None:
        result = BenchmarkResult(
            schema_version="7",
            name="repeated",
            timestamp="2026-08-22T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[
                EngineResult(
                    engine_path="/engine",
                    engine_name="engine",
                    inp_path="model.inp",
                    inp_name="model.inp",
                    duration_s=2.0,
                    peak_memory_mb=11.0,
                    exit_code=0,
                    rpt_path=None,
                    stdout="",
                    stderr="",
                    error=None,
                    samples=[
                        BenchmarkSample(1.0, 10.0, 0, None),
                        BenchmarkSample(3.0, 12.0, 0, None),
                        BenchmarkSample(2.0, 11.0, 0, None),
                    ],
                    representative_sample=3,
                )
            ],
            comparisons=[],
            run_count=3,
            run_order="interleaved",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.html"
            render_html(result, report_path)
            html = report_path.read_text(encoding="utf-8")

        self.assertIn("3 per engine/model", html)
        self.assertIn("interleaved", html)
        self.assertIn("2.000 s median", html)
        self.assertIn("1.000–3.000 s · 3/3 successful", html)
        self.assertIn("Mean of per-model median analysis durations", html)


    def test_summary_only_output_explains_chart_retention_reason(self) -> None:
        result = BenchmarkResult(
            schema_version="5",
            name="chart-budget",
            timestamp="2026-08-18T00:00:00+00:00",
            platform={},
            engine_results=[],
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.0,
                    section_comparisons=[],
                    details_retained=False,
                    graphical_unavailable_reason=(
                        "Graphical data was not retained because the report-size "
                        "target left insufficient chart payload budget."
                    ),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("report-size target left insufficient", html)
        self.assertNotIn("0.000000 hidden as matching", html)
        self.assertNotIn("Detailed output comparison data was not saved.", html)

    def test_legacy_sampled_series_are_omitted_from_output_explorer(self) -> None:
        result = BenchmarkResult(
            schema_version="5",
            name="legacy-sampled-chart",
            timestamp="2026-08-18T00:00:00+00:00",
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
                    section_comparisons=[],
                    graphical_series=[
                        OutputSeriesComparison(
                            element_type="node",
                            element_name="legacy-sampled-node",
                            attribute="depth",
                            distance=0.5,
                            row_count_a=3,
                            row_count_b=3,
                            timestamps=["2020-01-01T00:00:00", "2020-01-01T00:10:00"],
                            values_a=[1.0, 2.0],
                            values_b=[1.1, 2.1],
                            source_point_count=3,
                            sampled=True,
                        )
                    ],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("Legacy sampled chart data was omitted", html)
        self.assertNotIn("legacy-sampled-node", html)

    def test_duration_label_tracks_schema_semantics(self) -> None:
        for schema_version, expected in (
            ("4", "Average runtime by engine"),
            ("5", "Average analysis duration by engine"),
        ):
            with self.subTest(schema_version=schema_version):
                result = BenchmarkResult(
                    schema_version=schema_version,
                    name="duration-label",
                    timestamp="2026-07-20T00:00:00+00:00",
                    platform={"host": "test", "os": "test", "python": "test"},
                    engine_results=[],
                    comparisons=[],
                )
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output = Path(temporary_directory) / "report.html"
                    render_html(result, output)
                    html = output.read_text(encoding="utf-8")

                self.assertIn(expected, html)

    def test_output_comparison_reports_trailing_timeline_coverage(self) -> None:
        result = BenchmarkResult(
            schema_version="5",
            name="timeline-coverage",
            timestamp="2026-08-18T00:00:00+00:00",
            platform={},
            engine_results=[],
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="epaswmm",
                    engine_b="runswmmrs",
                    overall_distance=0.0,
                    section_comparisons=[],
                    timeline_coverage=OutputTimelineCoverage(
                        timestamp_count_a=3,
                        timestamp_count_b=6,
                        shared_timestamp_count=3,
                        trailing_timestamp_count_b=3,
                    ),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn('data-severity="green"', html)
        self.assertIn("timeline coverage differs · matching", html)
        self.assertNotIn("timeline coverage differs · not comparable", html)
        self.assertIn("3 shared timestamps", html)
        self.assertIn("runswmmrs has 3 additional trailing timestamps", html)
        self.assertIn("do not affect value distance", html)

    def test_comparison_drilldown_labels_report_tables(self) -> None:
        result = BenchmarkResult(
            schema_version="4",
            name="table-diff",
            timestamp="2026-07-19T00:00:00+00:00",
            platform={
                "host": "test",
                "system": "test",
                "release": "test",
                "machine": "test",
                "python": "test",
            },
            engine_results=[],
            comparisons=[
                ModelComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.5,
                    section_comparisons=[
                        SectionComparison(
                            section_name="node_depth_summary",
                            distance=0.0,
                            row_count_a=1,
                            row_count_b=1,
                            differences=[
                                {
                                    "row": "J1",
                                    "column": "depth",
                                    "value_a": 1.0,
                                    "value_b": 2.0,
                                    "abs_diff": 1.0,
                                    "rel_diff": 1.0,
                                }
                            ],
                        )
                    ],
                ),
                ModelComparison(
                    inp_path="matching-report.inp",
                    inp_name="matching-report.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.01,
                    section_comparisons=[],
                ),
                ModelComparison(
                    inp_path="warning-report.inp",
                    inp_name="warning-report.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.0,
                    section_comparisons=[
                        SectionComparison(
                            section_name="bad_table",
                            distance=1.0,
                            row_count_a=0,
                            row_count_b=1,
                            differences=[],
                            note="Table could not be parsed for a; parsed only for b.",
                        )
                    ],
                    report_warnings=[
                        "WARNING: table 'bad_table' could not be parsed: bad table"
                    ],
                ),
                ModelComparison(
                    inp_path="error-report.inp",
                    inp_name="error-report.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.0,
                    section_comparisons=[],
                    report_errors=[
                        "ERROR 101: simulation failed",
                        "ERROR 102: output failed",
                    ],
                    report_warnings=[
                        "WARNING 09: timestep reduced",
                        "WARNING 10: continuity warning",
                    ],
                    report_errors_a=["ERROR 101: simulation failed"],
                    report_errors_b=["ERROR 102: output failed"],
                    report_warnings_a=["WARNING 09: timestep reduced"],
                    report_warnings_b=["WARNING 10: continuity warning"],
                ),
            ],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.5,
                    section_comparisons=[
                        OutputSectionComparison(
                            section_name='["node","J1","hydraulic_head"]',
                            distance=0.5,
                            row_count_a=2,
                            row_count_b=2,
                            differences=[
                                {
                                    "row": "2020-01-01T00:05:00",
                                    "value_a": None,
                                    "value_b": 2.0,
                                    "issue": "null value in A",
                                    "abs_diff": None,
                                    "rel_diff": 1.0,
                                }
                            ],
                            difference_count=1,
                            numeric_distance=0.4,
                            missing_fraction=1.0 / 6.0,
                            finite_pair_count=1,
                            missing_count=1,
                            both_null_count=0,
                            timestamp_count=2,
                        )
                    ],
                    graphical_series=[
                        OutputSeriesComparison(
                            element_type="node",
                            element_name='J1 & "north"',
                            attribute="hydraulic_head",
                            distance=0.5,
                            row_count_a=2,
                            row_count_b=2,
                            timestamps=["2020-01-01T00:00:00", "2020-01-01T00:05:00"],
                            values_a=[1.0, None],
                            values_b=[1.5, 2.0],
                            source_point_count=2,
                        )
                    ],
                ),
                OutputComparison(
                    inp_path="matching-output.inp",
                    inp_name="matching-output.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.0,
                    section_comparisons=[],
                ),
                OutputComparison(
                    inp_path="summary-only-output.inp",
                    inp_name="summary-only-output.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.001,
                    section_comparisons=[
                        OutputSectionComparison(
                            section_name="summary-only-series",
                            distance=0.001,
                            row_count_a=1,
                            row_count_b=1,
                            differences=[],
                        )
                    ],
                    details_retained=False,
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("<th>Table</th>", html)
        self.assertNotIn("<th>Series</th>", html)
        self.assertNotIn(">Tabular</button>", html)
        self.assertIn("1 plotted series", html)
        self.assertIn("Element type", html)
        self.assertIn("Search element or attribute", html)
        self.assertIn("Overall output distance", html)
        self.assertIn("Selected series distance", html)
        self.assertIn("Symmetric NRMSE + missing penalty", html)
        self.assertIn("About output distance", html)
        self.assertIn('aria-hidden="true">ⓘ</span>', html)
        self.assertIn("numeric distance = min(RMSE(A − B)", html)
        self.assertIn("series distance = missing fraction", html)
        self.assertIn("equal-weight arithmetic mean", html)
        self.assertIn("Report-table comparisons use a separate", html)
        self.assertIn('type="application/json"', html)
        self.assertIn('J1 \\u0026 \\"north\\"', html)
        self.assertNotIn('J1 & "north"', html)
        self.assertIn("pointRadius: selected.timestamps.length < 2 ? 4 : 0", html)
        self.assertIn("View plotted timesteps as a table", html)
        self.assertIn("data-sample-table-body", html)
        self.assertIn("outline: 3px solid var(--accent)", html)
        self.assertIn("data-graph-error", html)
        self.assertIn("data-chart-frame", html)
        self.assertIn("No matching output series", html)
        self.assertIn("chart.destroy();", html)
        self.assertIn("chartjs-plugin-zoom@2.2.0", html)
        self.assertIn("hammerjs@2.0.8", html)
        self.assertIn("data-reset-zoom", html)
        self.assertIn("Reset zoom", html)
        self.assertIn("Ctrl + wheel or pinch to zoom time", html)
        self.assertIn("modifierKey: 'ctrl'", html)
        self.assertIn("modifierKey: 'shift'", html)
        self.assertIn("modifierKey: 'alt'", html)
        self.assertIn("chart.resetZoom();", html)
        self.assertIn("What this report includes", html)
        self.assertIn("distance is above 1%", html)
        self.assertIn("Comparison overview", html)
        self.assertIn('data-theme="solarized-light"', html)
        self.assertIn(':root[data-theme="solarized-dark"]', html)
        self.assertIn("data-theme-toggle", html)
        self.assertIn("swmm-bench.report.theme.v1", html)
        self.assertIn("swmm-bench:themechange", html)
        self.assertIn("applyChartTheme", html)
        self.assertIn("Issues", html)
        self.assertNotIn('class="storm-trace"', html)
        self.assertIn("data-issue-queue", html)
        self.assertIn('data-label="Model"', html)
        self.assertIn("data-issue-kind", html)
        self.assertIn("data-issue-chip", html)
        self.assertIn("data-issue-sort", html)
        self.assertIn("data-explorer-show-matches", html)
        self.assertIn("data-section-only-differences", html)
        self.assertIn("data-section-sort-distance", html)
        self.assertIn("data-copy-table-name", html)
        self.assertIn("data-issue-toggle", html)
        self.assertNotIn('data-label="Evidence"', html)
        self.assertNotIn("issue-summary-filter", html)
        self.assertIn('class="severity-key"', html)
        self.assertIn("issue-toolbar", html)
        self.assertIn("More filters", html)
        self.assertIn("Performance table (0 engines × 0 models)", html)
        self.assertIn("data-performance-table", html)
        self.assertIn("data-sort-model", html)
        self.assertIn("data-sort-index", html)
        self.assertIn("data-performance-index", html)
        self.assertIn("Graph JSON is parsed only after opening", html)
        self.assertIn("Report comparison explorer", html)
        self.assertIn("Search model or engine", html)
        self.assertIn("data-comparison-explorer", html)
        self.assertIn("data-explorer-target", html)
        self.assertIn('class="section-nav"', html)
        self.assertIn('href="#overview"', html)
        self.assertIn('href="#performance"', html)
        self.assertIn("data-explorer-engine-pair", html)
        self.assertIn("data-explorer-min-distance", html)
        self.assertIn("data-explorer-max-distance", html)
        self.assertIn("data-engine-pair=", html)
        self.assertIn("data-distance=", html)
        self.assertIn("Not comparable or matching comparisons: 2 report-table; 2 output", html)
        self.assertIn("Distance by selected baseline", html)
        self.assertIn("data-distance-baseline", html)
        self.assertIn("data-distance-kind", html)
        self.assertIn("data-distance-engines", html)
        self.assertIn("data-distance-scale", html)
        self.assertIn('<option value="logarithmic">Logarithmic</option>', html)
        self.assertIn("data-distance-chart", html)
        self.assertIn("distance-overview-data", html)
        self.assertIn("data-distance-reset", html)
        self.assertIn("data-distance-zoom-mode", html)
        self.assertIn("openExplorerItem(record.dom_id)", html)
        self.assertIn("row.model === model", html)
        self.assertIn("Clustered bar chart of report or output distances", html)
        self.assertIn("text: 'Model Run'", html)
        self.assertNotIn("ticks: { display: false }", html)
        self.assertIn("mode: axisMode", html)
        self.assertIn(
            "limits: { y: { min: scaleType === 'linear' ? 0 : 'original' } }",
            html,
        )
        self.assertIn("type: scaleType", html)
        self.assertIn("beginAtZero: scaleType === 'linear'", html)
        self.assertIn("min: scaleType === 'linear' ? 0 : undefined", html)
        self.assertIn("warning-report.inp", html)
        self.assertIn(
            "WARNING: table &#39;bad_table&#39; could not be parsed: bad table",
            html,
        )
        self.assertIn('data-has-warnings="true"', html)
        self.assertIn("warning-badge", html)
        self.assertIn("error-report.inp", html)
        self.assertIn('data-severity="failed"', html)
        self.assertIn(">ERROR 101: simulation failed</td>", html)
        self.assertIn(">ERROR 102: output failed</td>", html)
        self.assertIn(">WARNING 09: timestep reduced</td>", html)
        self.assertIn(">WARNING 10: continuity warning</td>", html)
        self.assertIn("Comparison cannot be made: a and b models failed.", html)
        self.assertIn("Diagnostics", html)
        self.assertIn("2 errors", html)
        self.assertIn("3 warnings", html)
        self.assertIn("missing table", html)
        self.assertIn("Table could not be parsed for a; parsed only for b.", html)
        self.assertIn("<th>Rows a</th>", html)
        self.assertIn("<th>Rows b</th>", html)
        self.assertIn("<th>a</th>", html)
        self.assertIn("<th>b</th>", html)
        self.assertIn("matching-report.inp", html)
        self.assertIn("matching-output.inp", html)
        self.assertIn("0.010000 hidden as matching (≤0.010000)", html)
        self.assertIn("0.000000 hidden as matching (≤0.010000)", html)
        self.assertIn("summary-only-output.inp", html)
        self.assertIn("0.001000 hidden as matching (≤0.010000)", html)
        self.assertNotIn("summary-only-series", html)

    def test_composite_output_metric_explains_and_displays_components(self) -> None:
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
                    overall_distance=0.2575,
                    section_comparisons=[],
                    metric=OUTPUT_DISTANCE_METRIC_COMPOSITE,
                    typical_weight=0.0,
                    typical_distance=0.01,
                    event_distance=1.0,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("Typical/event composite + missing penalty", html)
        self.assertIn("The typical weight is 0%", html)
        self.assertIn("the event weight is 100%", html)
        self.assertIn("Typical-over-time <strong>0.010000</strong>", html)
        self.assertIn("event <strong>1.000000</strong>", html)

    def test_report_engine_order_prioritizes_requested_engines(self) -> None:
        def engine(name: str, duration_s: float) -> EngineResult:
            return EngineResult(
                engine_path=f"/{name}",
                engine_name=name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=duration_s,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
            )

        result = BenchmarkResult(
            schema_version="5",
            name="ordered-report",
            timestamp="2026-08-03T00:00:00+00:00",
            platform={},
            engine_results=[
                engine("openswmm", 1.0),
                engine("swmmrs", 2.0),
                engine("runswmm", 3.0),
            ],
            comparisons=[],
        )

        context = _build_template_context(result, engine_order=["runswmm"])

        self.assertEqual(
            context["engines"], ["runswmm", "openswmm", "swmmrs"]
        )
        self.assertEqual(context["chart_labels"], context["engines"])
        self.assertEqual(context["chart_values"], [3.0, 1.0, 2.0])
        self.assertEqual(
            context["distance_chart_engines"], context["engines"]
        )
        with self.assertRaisesRegex(ValueError, "Unknown engine"):
            _build_template_context(result, engine_order=["missing"])
        with self.assertRaisesRegex(ValueError, "duplicate engine"):
            _build_template_context(result, engine_order=["runswmm", "runswmm"])

    def test_report_controls_follow_engine_order_and_use_tabs(self) -> None:
        def engine(name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{name}",
                engine_name=name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=f"/tmp/{name}.rpt",
                stdout="",
                stderr="",
                error=None,
                out_path=None,
            )

        result = BenchmarkResult(
            schema_version="5",
            name="ordered-controls",
            timestamp="2026-08-03T00:00:00+00:00",
            platform={},
            engine_results=[engine("runswmm"), engine("openswmm")],
            comparisons=[
                ModelComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="runswmm",
                    engine_b="openswmm",
                    overall_distance=0.2,
                    section_comparisons=[],
                    report_warnings=["WARNING 09: timestep reduced"],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        overview_data = html.split('id="distance-overview-data">', 1)[1].split(
            "</script>", 1
        )[0]
        distance_kind = html.split("<select data-distance-kind>", 1)[1].split(
            "</select>", 1
        )[0]
        self.assertIn('"engines": ["runswmm", "openswmm"]', overview_data)
        self.assertLess(
            distance_kind.index('value="Output"'),
            distance_kind.index('value="Report"'),
        )
        self.assertIn('role="tablist"', html)
        self.assertIn('data-comparison-tab="comparison"', html)
        self.assertIn('data-comparison-tab-panel="diagnostics"', html)

    def test_html_retains_all_above_threshold_output_graphs(self) -> None:
        comparisons = [
            OutputComparison(
                inp_path=f"model-{index}.inp",
                inp_name=f"model-{index}.inp",
                engine_a="a",
                engine_b="b",
                section_comparisons=[],
                overall_distance=0.02,
                graphical_series=[
                    OutputSeriesComparison(
                        element_type="node",
                        element_name="J1",
                        attribute="depth",
                        distance=0.02,
                        row_count_a=2,
                        row_count_b=2,
                        timestamps=["2020-01-01T00:00:00", "2020-01-01T00:05:00"],
                        values_a=[1.0, 2.0],
                        values_b=[1.1, 2.1],
                        source_point_count=2,
                    )
                ],
            )
            for index in range(13)
        ]
        result = BenchmarkResult(
            schema_version="5",
            name="all-output-graphs",
            timestamp="2026-07-27T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[],
            comparisons=[],
            output_comparisons=comparisons,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertEqual(html.count(" data-output-series-data>"), 13)
        self.assertNotIn("Graphical data was omitted from this HTML report", html)

    def test_issue_queue_includes_failed_engine_pairs(self) -> None:
        engine_results = [
            EngineResult(
                engine_path=f"/{name}",
                engine_name=name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0 if exit_code == 0 else None,
                peak_memory_mb=1.0 if exit_code == 0 else None,
                exit_code=exit_code,
                rpt_path=f"/tmp/{name}.rpt" if exit_code == 0 else None,
                stdout="",
                stderr="failed" if exit_code else "",
                error="failed" if exit_code else None,
                out_path=f"/tmp/{name}.out" if exit_code == 0 else None,
            )
            for name, exit_code in (("a", 0), ("b", 0), ("failed", 1))
        ]
        result = BenchmarkResult(
            schema_version="5",
            name="priority-output",
            timestamp="2026-07-27T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=engine_results,
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.5,
                    section_comparisons=[],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        issue_queue = html.split("data-issue-queue", maxsplit=1)[1].split(
            "</table>", maxsplit=1
        )[0]
        self.assertIn("a vs b", issue_queue)
        self.assertIn("a vs failed", issue_queue)
        self.assertEqual(
            issue_queue.count('data-label="Model">model.inp</td>'),
            issue_queue.count("data-issue-row"),
        )
        self.assertIn("b vs failed", issue_queue)

    def test_report_explorer_includes_failed_engine_placeholders(self) -> None:
        result = BenchmarkResult(
            schema_version="5",
            name="failed-engine",
            timestamp="2026-07-27T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[
                EngineResult(
                    engine_path="/good",
                    engine_name="openswmm",
                    inp_path="complex/rtk.inp",
                    inp_name="complex/rtk.inp",
                    duration_s=1.0,
                    peak_memory_mb=1.0,
                    exit_code=0,
                    rpt_path="/tmp/good.rpt",
                    stdout="",
                    stderr="",
                    error=None,
                    out_path="/tmp/good.out",
                ),
                EngineResult(
                    engine_path="/bad",
                    engine_name="runswmmrs",
                    inp_path="complex/rtk.inp",
                    inp_name="complex/rtk.inp",
                    duration_s=None,
                    peak_memory_mb=None,
                    exit_code=1,
                    rpt_path=None,
                    stdout="",
                    stderr="failed",
                    error="failed",
                    out_path=None,
                ),
            ],
            output_comparisons=[],
            comparisons=[
                ModelComparison(
                    inp_path="complex/rtk.inp",
                    inp_name="complex/rtk.inp",
                    engine_a="openswmm",
                    engine_b="runswmmrs",
                    overall_distance=1.0,
                    section_comparisons=[
                        SectionComparison(
                            section_name="failed_table",
                            distance=1.0,
                            row_count_a=1,
                            row_count_b=1,
                            differences=[],
                        )
                    ],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("complex/rtk.inp", html)
        self.assertIn("openswmm vs runswmmrs", html)
        self.assertIn("Comparison cannot be made: runswmmrs model failed.", html)
        self.assertIn("Not comparable or matching comparisons: 1 report-table; 1 output", html)
        self.assertIn('data-severity="failed"', html)
        self.assertIn('class="graph-empty distance-failed"', html)
        self.assertIn("Check the engine exit code.", html)
        self.assertNotIn("failed_table", html)

    def test_included_comparison_does_not_inherit_prior_placeholder(self) -> None:
        def engine(name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{name}",
                engine_name=name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=f"/tmp/{name}.rpt",
                stdout="",
                stderr="",
                error=None,
                out_path=None,
            )

        result = BenchmarkResult(
            schema_version="5",
            name="placeholder-leak",
            timestamp="2026-07-27T00:00:00+00:00",
            platform={},
            engine_results=[engine("a"), engine("b"), engine("c")],
            comparisons=[
                ModelComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.001,
                    section_comparisons=[],
                ),
                ModelComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="c",
                    overall_distance=0.07,
                    section_comparisons=[
                        SectionComparison(
                            section_name="included_table",
                            distance=0.07,
                            row_count_a=1,
                            row_count_b=1,
                            differences=[],
                        )
                    ],
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("included_table", html)

    def test_report_autoescapes_persisted_text_and_attributes(self) -> None:
        result = BenchmarkResult(
            schema_version="3",
            name="<script>alert('name')</script>",
            timestamp="2026-07-20T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[],
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="<img src=x onerror=alert('model')>",
                    engine_a='a" onmouseover="alert(1)',
                    engine_b="b",
                    overall_distance=0.5,
                    section_comparisons=[],
                    metric=OUTPUT_DISTANCE_METRIC_LEGACY,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertNotIn("<script>alert('name')</script>", html)
        self.assertNotIn("<img src=x onerror=alert('model')>", html)
        self.assertIn("&lt;script&gt;alert(&#39;name&#39;)&lt;/script&gt;", html)
        self.assertIn("&lt;img src=x onerror=alert(&#39;model&#39;)&gt;", html)
        self.assertIn(
            'data-engine-a="a&#34; onmouseover=&#34;alert(1)"',
            html,
        )

    def test_json_writer_rejects_nonfinite_values(self) -> None:
        result = BenchmarkResult(
            schema_version="3",
            name="nonfinite",
            timestamp="2026-07-20T00:00:00+00:00",
            platform={"invalid": math.nan},
            engine_results=[],
            comparisons=[],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "results.json"
            with self.assertRaisesRegex(ValueError, "Out of range float values"):
                save_json(result, output)

    def test_legacy_output_comparison_renders_graphical_fallback(self) -> None:
        result = BenchmarkResult(
            schema_version="2",
            name="legacy-output",
            timestamp="2026-07-19T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[],
            comparisons=[],
            output_comparisons=[
                OutputComparison(
                    inp_path="model.inp",
                    inp_name="model.inp",
                    engine_a="a",
                    engine_b="b",
                    overall_distance=0.5,
                    section_comparisons=[],
                    metric=OUTPUT_DISTANCE_METRIC_LEGACY,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.html"
            render_html(result, output)
            html = output.read_text(encoding="utf-8")

        self.assertNotIn(">Tabular</button>", html)
        self.assertNotIn(">Graphical</button>", html)
        self.assertIn("No complete time-series data was saved for this comparison", html)
        self.assertIn("Legacy pointwise relative distance", html)
        self.assertIn(
            "Near-zero values can therefore produce large relative scores", html
        )
        self.assertNotIn("Symmetric NRMSE + missing penalty", html)
        self.assertNotIn(" data-output-series-data>", html)


if __name__ == "__main__":
    unittest.main()
