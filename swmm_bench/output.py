from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pandas import DataFrame, DatetimeIndex, MultiIndex, concat
from swmm.pandas import Output

_OUTPUT_COLUMN_NAMES = ("element_type", "element_name", "attribute")


def _validate_frame(path: Path, frame: DataFrame) -> None:
    if not isinstance(frame.index, DatetimeIndex):
        raise TypeError(f"Output {path} did not produce a DatetimeIndex")
    if not frame.index.is_unique:
        raise ValueError(f"Output {path} has duplicate reporting timestamps")
    if not isinstance(frame.columns, MultiIndex):
        raise TypeError(f"Output {path} did not produce semantic MultiIndex columns")
    if frame.columns.names != list(_OUTPUT_COLUMN_NAMES):
        raise ValueError(f"Output {path} has unexpected semantic column levels")
    if not frame.columns.is_unique:
        raise ValueError(f"Output {path} has duplicate semantic output columns")


def _element_frame(
    path: Path,
    element_type: str,
    elements: Sequence[str],
    series_getter: Callable[..., DataFrame],
) -> list[DataFrame]:
    frames: list[DataFrame] = []
    for element_name in elements:
        try:
            frame = series_getter(element_name, attribute=None, columns="attr")
        except Exception as exc:
            raise ValueError(
                f"Failed to extract {element_type} {element_name!r} from output {path}"
            ) from exc
        if not isinstance(frame, DataFrame):
            raise TypeError(
                f"Output {path} {element_type} {element_name!r} returned "
                f"{type(frame).__name__}, expected DataFrame"
            )
        frame = frame.copy()
        frame.index = frame.index.rename("datetime")
        frame.columns = MultiIndex.from_tuples(
            [
                (element_type, str(element_name), str(attribute))
                for attribute in frame.columns
            ],
            names=_OUTPUT_COLUMN_NAMES,
        )
        frames.append(frame)
    return frames


def extract_output_frame(out_path: str | Path, *, preload: bool = True) -> DataFrame:
    """Read an output file into one semantic, wide result frame.

    Columns identify values by element type, public element name, and public
    attribute name. The reader's private positional output layout is never used.
    """

    path = Path(out_path)
    try:
        with Output(str(path), preload=preload) as output:
            if output.period == 0:
                return DataFrame()

            frames: list[DataFrame] = []
            if output.subcatchments:
                frames.extend(
                    _element_frame(
                        path,
                        "subcatchment",
                        output.subcatchments,
                        output.subcatch_series,
                    )
                )
            if output.nodes:
                frames.extend(
                    _element_frame(path, "node", output.nodes, output.node_series)
                )
            if output.links:
                frames.extend(
                    _element_frame(path, "link", output.links, output.link_series)
                )

            if output.project_size[3]:
                try:
                    system = output.system_series(attribute=None)
                except Exception as exc:
                    raise ValueError(f"Failed to extract system results from output {path}") from exc
                if not isinstance(system, DataFrame):
                    raise TypeError(
                        f"Output {path} system results returned "
                        f"{type(system).__name__}, expected DataFrame"
                    )
                system = system.copy()
                system.index = system.index.rename("datetime")
                system.columns = MultiIndex.from_tuples(
                    [("system", "system", str(attribute)) for attribute in system.columns],
                    names=_OUTPUT_COLUMN_NAMES,
                )
                frames.append(system)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to open output {path}") from exc

    if not frames:
        return DataFrame()

    frame = concat(frames, axis=1)
    _validate_frame(path, frame)
    return frame


def output_series_name(column: tuple[str, str, str]) -> str:
    return json.dumps(column, separators=(",", ":"))


def split_output_frame(frame: DataFrame, source: str | Path = "output frame") -> dict[str, DataFrame]:
    """Split a semantic output frame into one-column comparison tables."""

    if frame.empty and not len(frame.columns):
        return {}

    tables: dict[str, DataFrame] = {}
    for column in frame.columns:
        name = output_series_name(column)
        if name in tables:
            raise ValueError(f"Output {source} has duplicate semantic series {name}")
        tables[name] = frame.loc[:, [column]]
    return tables


def extract_output_series(out_path: str | Path) -> dict[str, DataFrame]:
    """Return each semantic output column as a one-column comparison table."""

    return split_output_frame(extract_output_frame(out_path), out_path)
