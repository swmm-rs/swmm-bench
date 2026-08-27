from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from importlib.resources.abc import Traversable


REGRESSION_SUITE_NAME = "regression-suite"
BENCHMARK_SUITE_NAME = "benchmarks"


@dataclass(frozen=True)
class SuiteModel:
    suite_name: str
    category: str
    relative_path: str

    @property
    def identity(self) -> str:
        return f"bundled://{self.suite_name}/{self.relative_path}"


@dataclass(frozen=True)
class MaterializedSuiteModel:
    model: SuiteModel
    inp_path: Path


class SuiteSelectionError(ValueError):
    pass


def _suite_root(suite_name: str) -> Traversable:
    return resources.files("swmm_bench").joinpath("resources", suite_name)


def catalog_models(
    suite_name: str = REGRESSION_SUITE_NAME,
) -> tuple[SuiteModel, ...]:
    models: list[SuiteModel] = []
    for category_resource in sorted(
        _suite_root(suite_name).iterdir(), key=lambda item: item.name
    ):
        if not category_resource.is_dir():
            continue
        category = category_resource.name
        for model_resource in sorted(
            category_resource.iterdir(), key=lambda item: item.name
        ):
            if model_resource.is_file() and model_resource.name.endswith(".inp"):
                models.append(
                    SuiteModel(
                        suite_name,
                        category,
                        f"{category}/{model_resource.name}",
                    )
                )
    return tuple(models)


def categories(suite_name: str = REGRESSION_SUITE_NAME) -> tuple[str, ...]:
    return tuple(sorted({model.category for model in catalog_models(suite_name)}))


def select_models(
    *,
    category: str | None = None,
    model: str | None = None,
    suite_name: str = REGRESSION_SUITE_NAME,
) -> tuple[SuiteModel, ...]:
    if category is not None and model is not None:
        raise SuiteSelectionError("Specify either a category or a model, not both.")

    models = catalog_models(suite_name)
    if category is not None:
        selected = tuple(item for item in models if item.category == category)
        if selected:
            return selected
        valid_categories = ", ".join(categories(suite_name))
        raise SuiteSelectionError(
            f"Unknown category {category!r}. Valid categories: {valid_categories}."
        )

    if model is not None:
        selected = tuple(item for item in models if item.relative_path == model)
        if selected:
            return selected
        raise SuiteSelectionError(
            f"Unknown suite model {model!r}. Use a category-relative .inp path from the list command."
        )

    return models


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_resource_tree(child, destination / child.name)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file)


@contextmanager
def materialize_models(
    models: Sequence[SuiteModel],
) -> Iterator[list[MaterializedSuiteModel]]:
    suite_names = {model.suite_name for model in models}
    if len(suite_names) != 1:
        raise ValueError("Models must belong to one bundled suite.")
    suite_name = suite_names.pop()
    selected_categories = {model.category for model in models}

    with tempfile.TemporaryDirectory(
        prefix=f"swmm-{suite_name}-"
    ) as temporary_directory:
        root = Path(temporary_directory)
        suite_root = _suite_root(suite_name)
        for category in sorted(selected_categories):
            _copy_resource_tree(suite_root.joinpath(category), root / category)

        yield [
            MaterializedSuiteModel(model=item, inp_path=root / item.relative_path)
            for item in models
        ]
