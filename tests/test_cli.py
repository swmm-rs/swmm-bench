from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.text import Text
from typer.testing import CliRunner  # pyright: ignore[reportMissingImports]
from swmm.pandas import example_out_path  # pyright: ignore[reportMissingImports]

from swmm_bench.cli import app, test_app as regression_app
from swmm_bench.models import (
    BenchmarkResult,
    EngineResult,
    OutputComparison,
    OutputSeriesComparison,
    OUTPUT_DISTANCE_METRIC_COMPOSITE,
)


class RegressionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_suite_list_groups_models_and_shows_usage(self) -> None:
        result = self.runner.invoke(regression_app, ["list"])

        self.assertEqual(result.exit_code, 0, result.output)
        plain_output = Text.from_ansi(result.output).plain
        self.assertIn("hydrology (5)", plain_output)
        self.assertIn("water-quality/waterquality-events_example.inp", plain_output)
        self.assertIn("swmm-test run /path/to/swmm", plain_output)
        self.assertIn("--category hydrology", plain_output)
        self.assertIn("--model", plain_output)
        self.assertIn("water-quality/waterquality-events_example.inp", plain_output)

    def test_suite_run_forwards_inp_option_overrides(self) -> None:
        with patch("swmm_bench.cli._execute_benchmark") as execute_benchmark:
            result = self.runner.invoke(
                regression_app,
                [
                    "run",
                    "fake-swmm",
                    "--model",
                    "complex/rtk.inp",
                    "--inp-option",
                    "VARIABLE_STEP=0",
                    "--inp-option",
                    "IGNORE_QUALITY=YES",
                    "--parse-workers",
                    "2",
                    "--output-parse-workers",
                    "3",
                    "--report-size-mb",
                    "150",
                    "--output-typical-weight",
                    "0.9",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)

        self.assertEqual(
            execute_benchmark.call_args.kwargs["option_overrides"],
            {
                "THREADS": "1",
                "VARIABLE_STEP": "0",
                "IGNORE_QUALITY": "YES",
            },
        )
        self.assertEqual(execute_benchmark.call_args.kwargs["parse_workers"], 2)
        self.assertEqual(
            execute_benchmark.call_args.kwargs["output_parse_workers"], 3
        )
        self.assertEqual(execute_benchmark.call_args.kwargs["report_size_mb"], 150)
        self.assertEqual(
            execute_benchmark.call_args.kwargs["output_typical_weight"], 0.9
        )


    def test_suite_run_does_not_accept_benchmark_repetitions(self) -> None:
        result = self.runner.invoke(
            regression_app, ["run", "fake-swmm", "--runs", "2"]
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No such option: --runs", result.output)


    def test_suite_run_records_stable_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "fake-swmm"
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            output_dir = root / "results"

            result = self.runner.invoke(
                regression_app,
                [
                    "run",
                    str(engine),
                    "--model",
                    "complex/rtk.inp",
                    "--name",
                    "suite-test",
                    "--output-dir",
                    str(output_dir),
                    "--no-html",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            plain_output = " ".join(Text.from_ansi(result.output).plain.split())
            self.assertIn(
                "Running tests: fake-swmm · complex/rtk.inp", plain_output
            )
            results_json = output_dir / "suite-test" / "results.json"
            data = json.loads(results_json.read_text(encoding="utf-8"))
            engine_result = data["engine_results"][0]
            self.assertEqual(data["schema_version"], "5")
            self.assertIsNone(engine_result["out_path"])
            self.assertEqual(data["output_comparisons"], [])
            self.assertEqual(
                engine_result["inp_name"],
                "complex/rtk.inp",
            )
            self.assertEqual(
                engine_result["inp_path"],
                "bundled://regression-suite/complex/rtk.inp",
            )


class BenchmarkCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_run_forwards_inp_option_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inp_path = Path(temporary_directory) / "model.inp"
            inp_path.write_text("[OPTIONS]\nVARIABLE_STEP 0.75\n", encoding="utf-8")

            with patch("swmm_bench.cli._execute_benchmark") as execute_benchmark:
                result = self.runner.invoke(
                    app,
                    [
                        "run",
                        "fake-swmm",
                        "--inp",
                        str(inp_path),
                        "--inp-option",
                        "VARIABLE_STEP=0.25",
                        "--inp-option",
                        "THREADS=4",
                        "--parse-workers",
                        "3",
                        "--output-parse-workers",
                        "2",
                        "--report-size-mb",
                        "125",
                        "--runs",
                        "3",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(
                execute_benchmark.call_args.kwargs["option_overrides"],
                {"THREADS": "4", "VARIABLE_STEP": "0.25"},
            )
            self.assertEqual(execute_benchmark.call_args.kwargs["parse_workers"], 3)
            self.assertEqual(
                execute_benchmark.call_args.kwargs["output_parse_workers"], 2
            )
            self.assertEqual(
                execute_benchmark.call_args.kwargs["report_size_mb"], 125
            )
            self.assertEqual(execute_benchmark.call_args.kwargs["runs"], 3)

    def test_run_rejects_malformed_inp_option_override(self) -> None:
        result = self.runner.invoke(
            app,
            ["run", "fake-swmm", "--inp-option", "VARIABLE_STEP"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("must use NAME=VALUE", result.output)


    def test_run_selects_one_bundled_benchmark_model(self) -> None:
        with patch("swmm_bench.cli._execute_benchmark") as execute_benchmark:
            result = self.runner.invoke(
                app,
                ["run", "fake-swmm", "--model", "stress/fredericksburg.inp"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            list(execute_benchmark.call_args.kwargs["inp_names"].values()),
            ["stress/fredericksburg.inp"],
        )
        self.assertEqual(
            list(execute_benchmark.call_args.kwargs["inp_identities"].values()),
            ["bundled://benchmarks/stress/fredericksburg.inp"],
        )

    def test_run_rejects_model_with_inp(self) -> None:
        result = self.runner.invoke(
            app,
            [
                "run",
                "fake-swmm",
                "--inp",
                "model.inp",
                "--model",
                "stress/fredericksburg.inp",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Specify either --inp or --model", result.output)


    def test_run_uses_bundled_benchmarks_when_inp_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "fake-swmm"
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            output_dir = root / "results"

            result = self.runner.invoke(
                app,
                [
                    "run",
                    str(engine),
                    "--name",
                    "bundled-benchmark",
                    "--output-dir",
                    str(output_dir),
                    "--no-html",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            data = json.loads(
                (output_dir / "bundled-benchmark" / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                {item["inp_path"] for item in data["engine_results"]},
                {
                    "bundled://benchmarks/stress/10033-hydraulic.inp",
                    "bundled://benchmarks/stress/10860-nodes.inp",
                    "bundled://benchmarks/stress/126000-groundwater-lid.inp",
                    "bundled://benchmarks/stress/17100-dummy-links.inp",
                    "bundled://benchmarks/stress/4569-nodes.inp",
                    "bundled://benchmarks/stress/ddc-24hr-100yr.inp",
                    "bundled://benchmarks/stress/fredericksburg.inp",
                    "bundled://benchmarks/stress/terreno.inp",
                },
            )

    def test_rebuild_restores_report_json_and_html_from_saved_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "previous-run"
            case_directory = run_directory / "fake-swmm" / "model-a"
            (case_directory / "model").mkdir(parents=True)
            (case_directory / "model" / "Model.inp").write_text(
                "[TITLE]\n", encoding="utf-8"
            )
            (case_directory / "result.rpt").write_text("report\n", encoding="utf-8")

            def compare_outputs(
                *_args: object,
                retain_graphical: bool = True,
                include_all_comparisons: bool = False,
                **_kwargs: object,
            ) -> list[OutputComparison]:
                include_details = retain_graphical and include_all_comparisons
                return [
                    OutputComparison(
                        inp_path="model-a",
                        inp_name="model-a",
                        engine_a="fake-swmm-a",
                        engine_b="fake-swmm-b",
                        overall_distance=0.0,
                        section_comparisons=[],
                        graphical_series=[
                            OutputSeriesComparison(
                                element_type="node",
                                element_name="J1",
                                attribute="depth",
                                distance=0.0,
                                row_count_a=1,
                                row_count_b=1,
                                timestamps=["2026-08-03T00:00:00"],
                                values_a=[1.0],
                                values_b=[1.0],
                                source_point_count=1,
                            )
                        ]
                        if include_details
                        else [],
                        details_retained=include_details,
                    )
                ]

            with patch(
                "swmm_bench.cli.compare_all_outputs", side_effect=compare_outputs
            ) as compare_outputs_mock:
                rebuild_result = self.runner.invoke(
                    app,
                    [
                        "rebuild",
                        str(run_directory),
                        "--outputs",
                        "--all-comparisons",
                        "--report-size-mb",
                        "175",
                        "--output-typical-weight",
                        "0.9",
                    ],
                )

            self.assertEqual(rebuild_result.exit_code, 0, rebuild_result.output)
            plain_output = Text.from_ansi(rebuild_result.output).plain
            self.assertIn("Comparing reports: no engine pairs", plain_output)
            self.assertIn("Comparing binary outputs: no engine pairs", plain_output)
            self.assertIn("Writing JSON results", plain_output)
            report_json = run_directory / "report.json"
            data = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(data["name"], "previous-run")
            self.assertEqual(
                data["platform"],
                {"host": "Unavailable", "os": "Unavailable", "python": "Unavailable"},
            )
            self.assertEqual(data["engine_results"][0]["engine_name"], "fake-swmm")
            self.assertEqual(data["engine_results"][0]["inp_name"], "model-a")
            self.assertEqual(len(data["output_comparisons"][0]["graphical_series"]), 1)
            self.assertTrue(data["output_comparisons"][0]["details_retained"])
            self.assertNotIn(
                "retain_graphical", compare_outputs_mock.call_args.kwargs
            )
            self.assertTrue(
                compare_outputs_mock.call_args.kwargs["include_all_comparisons"]
            )
            self.assertEqual(
                compare_outputs_mock.call_args.kwargs["report_size_mb"], 175
            )
            self.assertEqual(
                compare_outputs_mock.call_args.kwargs["typical_weight"], 0.9
            )

            report_html = root / "report.html"
            report_result = self.runner.invoke(
                app, ["report", str(report_json), "--output", str(report_html)]
            )

            self.assertEqual(report_result.exit_code, 0, report_result.output)
            self.assertTrue(report_html.exists())
            report_text = report_html.read_text(encoding="utf-8")
            self.assertNotIn(
                "Detailed output comparison data was not saved.", report_text
            )
            self.assertNotIn(
                "0.000000 hidden as matching (≤0.010000)", report_text
            )
            self.assertIn("data-output-series-data", report_text)


    def test_rebuild_restores_repeated_benchmark_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "repeated-run"
            case_directory = run_directory / "fake-swmm" / "model-a"
            (case_directory / "model").mkdir(parents=True)
            (case_directory / "result.rpt").write_text("report\n", encoding="utf-8")
            (run_directory / "results.json").write_text(
                json.dumps(
                    {
                        "platform": {
                            "host": "benchmark-host",
                            "os": "benchmark-os",
                            "python": "3.14",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (case_directory / "samples.json").write_text(
                json.dumps(
                    {
                        "representative_sample": 3,
                        "samples": [
                            {
                                "duration_s": 3.0,
                                "peak_memory_mb": 12.0,
                                "exit_code": 0,
                                "error": None,
                            },
                            {
                                "duration_s": 1.0,
                                "peak_memory_mb": 10.0,
                                "exit_code": 0,
                                "error": None,
                            },
                            {
                                "duration_s": 2.0,
                                "peak_memory_mb": 11.0,
                                "exit_code": 0,
                                "error": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rebuild_result = self.runner.invoke(
                app, ["rebuild", str(run_directory)]
            )

            self.assertEqual(rebuild_result.exit_code, 0, rebuild_result.output)
            data = json.loads(
                (run_directory / "report.json").read_text(encoding="utf-8")
            )
            result = data["engine_results"][0]
            self.assertEqual(data["schema_version"], "7")
            self.assertEqual(data["run_count"], 3)
            self.assertEqual(data["run_order"], "interleaved")
            self.assertEqual(
                data["platform"],
                {
                    "host": "benchmark-host",
                    "os": "benchmark-os",
                    "python": "3.14",
                },
            )
            self.assertEqual(result["duration_s"], 2.0)
            self.assertEqual(result["peak_memory_mb"], 11.0)
            self.assertEqual(len(result["samples"]), 3)
            report_path = run_directory / "report.html"
            report_result = self.runner.invoke(
                app,
                [
                    "report",
                    str(run_directory / "report.json"),
                    "--output",
                    str(report_path),
                ],
            )
            self.assertEqual(report_result.exit_code, 0, report_result.output)
            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("2.000 s median", report_html)
            self.assertIn("1.000–3.000 s · 3/3 successful", report_html)
            self.assertIn("benchmark-host", report_html)
            self.assertIn("benchmark-os", report_html)


    def test_run_records_output_comparisons_without_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engines = []
            for name in ("fake-swmm-a", "fake-swmm-b"):
                engine = root / name
                engine.write_text(
                    "#!/usr/bin/env python3\n"
                    "import pathlib\n"
                    "import shutil\n"
                    "import sys\n"
                    f"shutil.copyfile({str(example_out_path)!r}, sys.argv[3])\n",
                    encoding="utf-8",
                )
                engine.chmod(0o755)
                engines.append(engine)

            output_dir = root / "results"
            result = self.runner.invoke(
                app,
                [
                    "run",
                    *(str(engine) for engine in engines),
                    "--inp",
                    str(inp_path),
                    "--name",
                    "output-test",
                    "--output-dir",
                    str(output_dir),
                    "--no-html",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            plain_output = Text.from_ansi(result.output).plain
            normalized_output = " ".join(plain_output.split())
            self.assertIn(
                "[1/2] Running benchmarks: fake-swmm-a · Model.inp",
                normalized_output,
            )
            self.assertIn(
                "Running benchmarks: fake-swmm-b · Model.inp", normalized_output
            )
            self.assertIn("Loading binary output", plain_output)
            self.assertNotIn("Preparing chart data", plain_output)
            self.assertIn("Writing JSON results", plain_output)
            results_json = output_dir / "output-test" / "results.json"
            data = json.loads(results_json.read_text())
            self.assertEqual(data["schema_version"], "5")
            self.assertEqual(data["comparisons"], [])
            self.assertEqual(len(data["output_comparisons"]), 1)
            output_comparison = data["output_comparisons"][0]
            self.assertEqual(output_comparison["overall_distance"], 0.0)
            self.assertEqual(
                output_comparison["metric"], OUTPUT_DISTANCE_METRIC_COMPOSITE
            )
            self.assertEqual(output_comparison["typical_weight"], 0.75)
            self.assertEqual(output_comparison["section_comparisons"], [])
            self.assertEqual(output_comparison["graphical_series"], [])
            self.assertFalse(output_comparison["details_retained"])
            self.assertTrue(all(row["out_path"] for row in data["engine_results"]))

            for row in data["engine_results"]:
                Path(row["out_path"]).unlink()
            report_path = root / "regenerated-report.html"
            report_result = self.runner.invoke(
                app,
                ["report", str(results_json), "--output", str(report_path)],
            )

            self.assertEqual(report_result.exit_code, 0, report_result.output)
            report_html = report_path.read_text(encoding="utf-8")
            self.assertNotIn('<script type="application/json"', report_html)
            self.assertIn("What this report includes", report_html)
            self.assertIn(
                "Not comparable or matching comparisons: 1 report-table; 1 output",
                report_html,
            )
            self.assertIn(
                "0.000000 hidden as matching (≤0.010000)", report_html
            )

    def test_report_command_prioritizes_requested_engines(self) -> None:
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

        benchmark = BenchmarkResult(
            schema_version="5",
            name="ordered-report",
            timestamp="2026-08-03T00:00:00+00:00",
            platform={"host": "test", "os": "test", "python": "test"},
            engine_results=[
                engine("openswmm", 1.0),
                engine("runswmm", 2.0),
            ],
            comparisons=[],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results_json = root / "results.json"
            results_json.write_text(json.dumps(benchmark.to_dict()), encoding="utf-8")
            report_path = root / "report.html"

            result = self.runner.invoke(
                app,
                [
                    "report",
                    str(results_json),
                    "--engine-order",
                    "runswmm",
                    "--output",
                    str(report_path),
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            html = report_path.read_text(encoding="utf-8")

        self.assertIn('labels: ["runswmm", "openswmm"]', html)
        table_head = html.split('data-performance-table>', 1)[1].split(
            "</thead>", 1
        )[0]
        self.assertLess(table_head.index("runswmm"), table_head.index("openswmm"))

    def test_report_command_autoescapes_saved_json_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results_json = root / "results.json"
            results_json.write_text(
                json.dumps(
                    {
                        "schema_version": "3",
                        "name": "<script>alert('name')</script>",
                        "timestamp": "2026-07-20T00:00:00+00:00",
                        "platform": {"host": "test", "os": "test", "python": "test"},
                        "engine_results": [],
                        "comparisons": [],
                        "output_comparisons": [
                            {
                                "inp_path": "model.inp",
                                "inp_name": "<img src=x onerror=alert('model')>",
                                "engine_a": 'a" onmouseover="alert(1)',
                                "engine_b": "b",
                                "overall_distance": 0.5,
                                "section_comparisons": [],
                                "graphical_series": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report_path = root / "report.html"

            result = self.runner.invoke(
                app,
                ["report", str(results_json), "--output", str(report_path)],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            html = report_path.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert('name')</script>", html)
            self.assertNotIn("<img src=x onerror=alert('model')>", html)
            self.assertIn("&lt;script&gt;alert(&#39;name&#39;)&lt;/script&gt;", html)
            self.assertIn(
                'data-engine-a="a&#34; onmouseover=&#34;alert(1)"',
                html,
            )


class RegressionEngineCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_suite_run_resolves_engine_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            engine = bin_dir / "fake-swmm"
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            output_dir = root / "results"
            env = {
                **os.environ,
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            }

            result = self.runner.invoke(
                regression_app,
                [
                    "run",
                    "fake-swmm",
                    "--model",
                    "complex/rtk.inp",
                    "--name",
                    "path-test",
                    "--output-dir",
                    str(output_dir),
                    "--no-html",
                ],
                env=env,
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((output_dir / "path-test" / "results.json").exists())

    def test_suite_run_rejects_conflicting_selectors(self) -> None:
        result = self.runner.invoke(
            regression_app,
            [
                "run",
                "/not/an/executable",
                "--category",
                "hydrology",
                "--model",
                "hydrology/lid-example_lid_rb.inp",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Specify either a category or a model", result.output)


if __name__ == "__main__":
    unittest.main()
