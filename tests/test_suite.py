from __future__ import annotations

import unittest

from swmm_bench.suite import (
    BENCHMARK_SUITE_NAME,
    SuiteSelectionError,
    catalog_models,
    categories,
    materialize_models,
    select_models,
)


class SuiteTests(unittest.TestCase):
    def test_catalog_has_expected_categories_and_models(self) -> None:
        models = catalog_models()

        self.assertEqual(len(models), 21)
        self.assertEqual(
            categories(),
            (
                "complex",
                "controls",
                "hydraulics",
                "hydrology",
                "routing",
                "use_interfaces",
                "water-quality",
            ),
        )
        self.assertEqual(
            [model.relative_path for model in models],
            sorted(model.relative_path for model in models),
        )

    def test_selects_all_category_or_exact_model(self) -> None:
        self.assertEqual(len(select_models()), 21)
        self.assertEqual(len(select_models(category="hydrology")), 5)

        selected = select_models(model="routing/kinwave-routing_kinwave.inp")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].category, "routing")
        self.assertEqual(
            selected[0].identity,
            "bundled://regression-suite/routing/kinwave-routing_kinwave.inp",
        )

    def test_rejects_invalid_or_conflicting_selection(self) -> None:
        with self.assertRaisesRegex(
            SuiteSelectionError, "either a category or a model"
        ):
            select_models(
                category="hydrology", model="hydrology/lid-example_lid_rb.inp"
            )
        with self.assertRaisesRegex(SuiteSelectionError, "Valid categories"):
            select_models(category="not-a-category")
        with self.assertRaisesRegex(SuiteSelectionError, "Unknown suite model"):
            select_models(model="../hydrology/lid-example_lid_rb.inp")

    def test_materialization_copies_water_quality_sidecar(self) -> None:
        selected = select_models(category="water-quality")

        with materialize_models(selected) as materialized:
            self.assertEqual(len(materialized), 2)
            self.assertTrue(
                (materialized[0].inp_path.parent / "events_example.dat").is_file()
            )
            for item in materialized:
                self.assertTrue(item.inp_path.is_file())
                model_text = item.inp_path.read_text(encoding="utf-8")
                self.assertIn('"events_example.dat"', model_text)
                self.assertNotIn("D:\\SWMMandSoftware", model_text)

    def test_catalogs_stress_benchmarks_and_materializes_fredericksburg(
        self,
    ) -> None:
        benchmark_models = catalog_models(BENCHMARK_SUITE_NAME)
        self.assertEqual(
            [model.relative_path for model in benchmark_models],
            [
                "stress/10033-hydraulic.inp",
                "stress/10860-nodes.inp",
                "stress/126000-groundwater-lid.inp",
                "stress/17100-dummy-links.inp",
                "stress/4569-nodes.inp",
                "stress/ddc-24hr-100yr.inp",
                "stress/fredericksburg.inp",
                "stress/terreno.inp",
            ],
        )

        selected = select_models(
            model="stress/fredericksburg.inp",
            suite_name=BENCHMARK_SUITE_NAME,
        )
        with materialize_models(selected) as materialized:
            model_text = materialized[0].inp_path.read_text(encoding="utf-8")

        self.assertIn("START_DATE           04/01/2021", model_text)
        self.assertIn("END_DATE             09/30/2021", model_text)
        self.assertIn("REPORT_STEP          01:00:00", model_text)
        self.assertIn("SCS_24h_Type_II_1in", model_text)


if __name__ == "__main__":
    unittest.main()
