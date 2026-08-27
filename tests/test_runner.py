from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from swmm_bench.models import BenchmarkResult, EngineResult
from swmm_bench.reporter import save_json
from swmm_bench.runner import _copy_model_tree, _set_option, run_benchmark


class RunBenchmarkTests(unittest.TestCase):
    def test_nested_output_directory_is_not_copied_into_model_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_directory = root / "model"
            model_directory.mkdir()
            inp_path = model_directory / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            auxiliary_path = model_directory / "rainfall.dat"
            auxiliary_path.write_text("rainfall\n", encoding="utf-8")

            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n"
                "pathlib.Path(sys.argv[3]).write_bytes(b'output')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            output_root = model_directory / "benches"
            results = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=output_root,
                benchmark_name="bench",
                timeout=5.0,
            )

            staged_model = (
                output_root / "bench" / engine_path.name / inp_path.name / "model"
            )
            self.assertIsNone(results[0].error)
            out_path = results[0].out_path
            self.assertIsNotNone(out_path)
            assert out_path is not None
            self.assertEqual(Path(out_path).read_bytes(), b"output")
            self.assertEqual(
                Path(out_path).name,
                "result.out",
            )
            self.assertEqual(
                (staged_model / inp_path.name).read_text(encoding="utf-8"), "[TITLE]\n"
            )
            self.assertEqual(
                (staged_model / auxiliary_path.name).read_text(encoding="utf-8"),
                "rainfall\n",
            )
            self.assertFalse((staged_model / output_root.name).exists())

    def test_duration_comes_from_report_analysis_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text(\n"
                "    '  Total elapsed time: 00:00:02\\n', encoding='utf-8'\n"
                ")\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            result = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=root / "results",
                benchmark_name="bench",
                timeout=5.0,
            )[0]

            self.assertEqual(result.duration_s, 2.0)

    def test_duration_falls_back_to_command_duration_for_unparseable_reports(
        self,
    ) -> None:
        report_bodies = (
            "  Analysis complete.\n",
            "  Total elapsed time: unavailable\n",
        )
        for index, report_body in enumerate(report_bodies):
            with self.subTest(report_body=report_body):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    inp_path = root / "Model.inp"
                    inp_path.write_text("[TITLE]\n", encoding="utf-8")
                    engine_path = root / "fake-swmm"
                    engine_path.write_text(
                        "#!/usr/bin/env python3\n"
                        "import pathlib\n"
                        "import sys\n"
                        f"pathlib.Path(sys.argv[2]).write_text({report_body!r}, encoding='utf-8')\n",
                        encoding="utf-8",
                    )
                    engine_path.chmod(0o755)

                    with patch(
                        "swmm_bench.runner.perf_counter",
                        side_effect=(10.0 + index, 10.25 + index),
                    ):
                        result = run_benchmark(
                            engines=[str(engine_path)],
                            inp_files=[inp_path],
                            work_dir=root / "results",
                            benchmark_name="bench",
                            timeout=5.0,
                        )[0]

                    self.assertEqual(result.duration_s, 0.25)

    def test_nonfinite_report_duration_falls_back_and_remains_json_safe(self) -> None:
        class NonFiniteDuration:
            def total_seconds(self) -> float:
                return float("nan")

        class NonFiniteReport:
            analysis_duration = NonFiniteDuration()

            def __init__(self, _path: str) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            with (
                patch("swmm_bench.runner.Report", NonFiniteReport),
                patch("swmm_bench.runner.perf_counter", side_effect=(10.0, 10.25)),
            ):
                result = run_benchmark(
                    engines=[str(engine_path)],
                    inp_files=[inp_path],
                    work_dir=root / "results",
                    benchmark_name="bench",
                    timeout=5.0,
                )[0]

            benchmark = BenchmarkResult(
                schema_version="5",
                name="nonfinite-duration",
                timestamp="2026-08-18T00:00:00+00:00",
                platform={},
                engine_results=[result],
                comparisons=[],
            )
            save_json(benchmark, root / "results.json")

            self.assertEqual(result.duration_s, 0.25)

    def test_failed_run_does_not_fall_back_to_command_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text(\n"
                "    'ERROR 101: simulation failed\\n', encoding='utf-8'\n"
                ")\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            with patch(
                "swmm_bench.runner.perf_counter", side_effect=(10.0, 10.01)
            ):
                result = run_benchmark(
                    engines=[str(engine_path)],
                    inp_files=[inp_path],
                    work_dir=root / "results",
                    benchmark_name="bench",
                    timeout=5.0,
                )[0]

            self.assertEqual(result.exit_code, 1)
            self.assertIsNotNone(result.rpt_path)
            self.assertIsNone(result.duration_s)

    def test_repeat_run_does_not_reuse_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n"
                "pathlib.Path(sys.argv[3]).write_bytes(b'output')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)
            output_root = root / "results"

            first = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=output_root,
                benchmark_name="bench",
                timeout=5.0,
            )[0]
            self.assertIsNotNone(first.rpt_path)
            self.assertIsNotNone(first.out_path)

            engine_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            engine_path.chmod(0o755)
            second = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=output_root,
                benchmark_name="bench",
                timeout=5.0,
            )[0]

            self.assertIsNone(second.rpt_path)
            self.assertIsNone(second.out_path)
            self.assertEqual(
                second.error, "Engine did not produce a non-empty report file"
            )

    def test_benchmark_preserves_supplied_input_name_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            started_runs = []
            results = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=root / "results",
                benchmark_name="bench",
                timeout=5.0,
                inp_names={inp_path: "suite/Model.inp"},
                inp_identities={inp_path: "bundled://regression-suite/suite/Model.inp"},
                run_started_callback=lambda engine_name, inp_name: started_runs.append(
                    (engine_name, inp_name)
                ),
            )

            self.assertEqual(started_runs, [("fake-swmm", "suite/Model.inp")])
            self.assertEqual(results[0].inp_name, "suite/Model.inp")
            self.assertEqual(
                results[0].inp_path, "bundled://regression-suite/suite/Model.inp"
            )
            self.assertIsNone(results[0].out_path)


    def test_repeated_runs_interleave_and_keep_median_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engines = [str(root / "engine-a"), str(root / "engine-b")]
            durations = {
                "engine-a": [3.0, 1.0, 2.0],
                "engine-b": [10.0, 30.0, 20.0],
            }
            started_runs = []

            def fake_run_engine(**kwargs: object) -> EngineResult:
                engine_name = Path(str(kwargs["engine_path"])).name
                display_name = str(kwargs["inp_name"])
                work_dir = Path(str(kwargs["work_dir"]))
                sample_index = int(work_dir.name)
                case_dir = work_dir / engine_name / display_name
                (case_dir / "model").mkdir(parents=True)
                (case_dir / "result.rpt").write_text(
                    f"{engine_name}-{sample_index}", encoding="utf-8"
                )
                (case_dir / "result.out").write_bytes(b"output")
                return EngineResult(
                    engine_path=str(kwargs["engine_path"]),
                    engine_name=engine_name,
                    inp_path=str(inp_path),
                    inp_name=display_name,
                    duration_s=durations[engine_name][sample_index - 1],
                    peak_memory_mb=float(sample_index),
                    exit_code=0,
                    rpt_path=str((case_dir / "result.rpt").resolve()),
                    stdout="",
                    stderr="",
                    error=None,
                    out_path=str((case_dir / "result.out").resolve()),
                )

            with patch("swmm_bench.runner.run_engine", side_effect=fake_run_engine):
                results = run_benchmark(
                    engines=engines,
                    inp_files=[inp_path],
                    work_dir=root / "results",
                    benchmark_name="bench",
                    timeout=5.0,
                    runs=3,
                    run_started_callback=lambda engine_name, inp_name: started_runs.append(
                        (engine_name, inp_name)
                    ),
                )

            self.assertEqual(
                [engine for engine, _model in started_runs],
                ["engine-a", "engine-b", "engine-b", "engine-a", "engine-a", "engine-b"],
            )
            self.assertEqual([result.duration_s for result in results], [2.0, 20.0])
            self.assertEqual([result.peak_memory_mb for result in results], [2.0, 2.0])
            self.assertEqual(
                [result.representative_sample for result in results], [3, 3]
            )
            self.assertEqual([len(result.samples) for result in results], [3, 3])
            self.assertEqual(
                Path(results[0].rpt_path or "").read_text(encoding="utf-8"),
                "engine-a-3",
            )
            self.assertTrue(
                (
                    root
                    / "results"
                    / "bench"
                    / "engine-a"
                    / "Model.inp"
                    / "samples.json"
                ).is_file()
            )
            self.assertFalse((root / "results" / "bench" / ".samples").exists())

    def test_model_copy_does_not_follow_directory_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_directory = root / "model"
            model_directory.mkdir()
            inp_path = model_directory / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            loop_path = model_directory / "loop"
            try:
                loop_path.symlink_to(model_directory, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlinks are unavailable: {exc}")

            staged_model = root / "staged"
            _copy_model_tree(inp_path, staged_model)

            self.assertEqual(
                (staged_model / inp_path.name).read_text(encoding="utf-8"), "[TITLE]\n"
            )
            self.assertTrue((staged_model / loop_path.name).is_dir())
            self.assertEqual(list((staged_model / loop_path.name).iterdir()), [])

    def test_option_override_does_not_parse_or_rewrite_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inp_path = Path(temporary_directory) / "Model.inp"
            inp_path.write_text(
                "[OPTIONS]\nTHREADS              4\n\n[EVENT]\n01/01/2020 1\n",
                encoding="utf-8",
            )

            _set_option(inp_path, "THREADS", "2")

            self.assertEqual(
                inp_path.read_text(encoding="utf-8"),
                "[OPTIONS]\nTHREADS              2\n\n[EVENT]\n01/01/2020 1\n",
            )

    def test_option_override_handles_comments_duplicates_and_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inp_path = Path(temporary_directory) / "Model.inp"
            inp_path.write_bytes(
                b"[OPTIONS] ; settings\r\n"
                b"VARIABLE_STEP 0.75 ; first\r\n"
                b"VARIABLE_STEP 0.25\r\n"
                b"\r\n[EVENT]\r\n01/01/2020 1\r\n"
            )

            _set_option(inp_path, "VARIABLE_STEP", "0.5")

            self.assertEqual(
                inp_path.read_bytes(),
                b"[OPTIONS] ; settings\r\n"
                b"VARIABLE_STEP 0.5 ; first\r\n"
                b"VARIABLE_STEP 0.5\r\n"
                b"\r\n[EVENT]\r\n01/01/2020 1\r\n",
            )

    def test_option_override_does_not_rewrite_the_rest_of_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inp_path = Path(temporary_directory) / "Model.inp"
            inp_path.write_text(
                "[OPTIONS]\nVARIABLE_STEP        0.75\n\n[EVENT]\n01/01/2020 1\n",
                encoding="utf-8",
            )

            _set_option(inp_path, "VARIABLE_STEP", "0.0")

            self.assertEqual(
                inp_path.read_text(encoding="utf-8"),
                "[OPTIONS]\nVARIABLE_STEP        0.0\n\n[EVENT]\n01/01/2020 1\n",
            )

    def test_option_override_adds_missing_option(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inp_path = Path(temporary_directory) / "Model.inp"
            inp_path.write_text(
                "[OPTIONS]\nFLOW_UNITS CFS\n\n[EVENT]\n01/01/2020 1\n",
                encoding="utf-8",
            )

            _set_option(inp_path, "MINIMUM_STEP", "0.25")

            self.assertEqual(
                inp_path.read_text(encoding="utf-8"),
                "[OPTIONS]\nMINIMUM_STEP         0.25\nFLOW_UNITS CFS\n\n"
                "[EVENT]\n01/01/2020 1\n",
            )

    def test_option_override_rejects_invalid_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid name"):
            _set_option(Path("unused"), "BAD NAME", "1")


if __name__ == "__main__":
    unittest.main()
