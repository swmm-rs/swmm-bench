from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Any, Generic, TypeVar


OUTPUT_DISTANCE_METRIC_COMPOSITE = "typical-event-composite-v1"
OUTPUT_DISTANCE_METRIC_NRMSE = "normalized-rmse-shared-timeline-v2"
OUTPUT_DISTANCE_METRIC_NRMSE_V1 = "normalized-rmse-missing-v1"
OUTPUT_DISTANCE_METRIC_LEGACY = "legacy-cell-distance-v1"


def _graph_string(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Graphical comparison field {name!r} must be a string")
    return value


def _graph_count(data: dict[str, Any], name: str, minimum: int = 0) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Graphical comparison field {name!r} must be an integer >= {minimum}"
        )
    return value


def _timeline_count(
    data: dict[str, Any], name: str, default: int | None
) -> int | None:
    value = data.get(name, default)
    if value is None and default is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Timeline coverage field {name!r} must be a nonnegative integer")
    return value


def _graph_number(data: dict[str, Any], name: str) -> float:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Graphical comparison field {name!r} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Graphical comparison field {name!r} must be numeric"
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Graphical comparison field {name!r} must be finite")
    return numeric


def _graph_values(data: dict[str, Any], name: str) -> list[float | None]:
    values = data.get(name)
    if not isinstance(values, list):
        raise ValueError(f"Graphical comparison field {name!r} must be a list")
    normalized: list[float | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
        elif isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"Graphical comparison field {name!r} must contain finite numbers or null"
            )
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Graphical comparison field {name!r} must contain finite numbers or null"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(
                    f"Graphical comparison field {name!r} must contain finite numbers or null"
                )
            normalized.append(numeric)
    return normalized


@dataclasses.dataclass
class BenchmarkSample:
    duration_s: float | None
    peak_memory_mb: float | None
    exit_code: int | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSample":
        return cls(**data)


@dataclasses.dataclass
class EngineResult:
    engine_path: str
    engine_name: str
    inp_path: str
    inp_name: str
    duration_s: float | None
    peak_memory_mb: float | None
    exit_code: int | None
    rpt_path: str | None
    stdout: str
    stderr: str
    error: str | None
    out_path: str | None = None
    samples: list[BenchmarkSample] = dataclasses.field(default_factory=list)
    representative_sample: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineResult":
        values = dict(data)
        values["samples"] = [
            BenchmarkSample.from_dict(item) for item in data.get("samples", [])
        ]
        return cls(**values)


@dataclasses.dataclass
class SectionComparison:
    section_name: str
    distance: float
    row_count_a: int
    row_count_b: int
    differences: list[dict[str, Any]]
    difference_count: int = 0
    differences_truncated: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionComparison":
        return cls(
            section_name=data["section_name"],
            distance=data["distance"],
            row_count_a=data["row_count_a"],
            row_count_b=data["row_count_b"],
            differences=data.get("differences", []),
            difference_count=data.get(
                "difference_count", len(data.get("differences", []))
            ),
            differences_truncated=data.get("differences_truncated", False),
            note=data.get("note"),
        )


@dataclasses.dataclass
class OutputSectionComparison(SectionComparison):
    numeric_distance: float | None = None
    typical_distance: float | None = None
    event_distance: float | None = None
    missing_fraction: float | None = None
    finite_pair_count: int | None = None
    missing_count: int | None = None
    both_null_count: int | None = None
    timestamp_count: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputSectionComparison":
        return cls(
            section_name=data["section_name"],
            distance=data["distance"],
            row_count_a=data["row_count_a"],
            row_count_b=data["row_count_b"],
            differences=data.get("differences", []),
            difference_count=data.get(
                "difference_count", len(data.get("differences", []))
            ),
            differences_truncated=data.get("differences_truncated", False),
            note=data.get("note"),
            numeric_distance=data.get("numeric_distance"),
            typical_distance=data.get("typical_distance"),
            event_distance=data.get("event_distance"),
            missing_fraction=data.get("missing_fraction"),
            finite_pair_count=data.get("finite_pair_count"),
            missing_count=data.get("missing_count"),
            both_null_count=data.get("both_null_count"),
            timestamp_count=data.get("timestamp_count"),
        )


ComparisonSection = TypeVar("ComparisonSection", bound=SectionComparison)


@dataclasses.dataclass
class ModelComparison(Generic[ComparisonSection]):
    inp_path: str
    inp_name: str
    engine_a: str
    engine_b: str
    overall_distance: float
    section_comparisons: list[ComparisonSection]
    report_warnings: list[str] = dataclasses.field(default_factory=list)
    report_errors: list[str] = dataclasses.field(default_factory=list)
    report_warnings_a: list[str] = dataclasses.field(default_factory=list)
    report_warnings_b: list[str] = dataclasses.field(default_factory=list)
    report_errors_a: list[str] = dataclasses.field(default_factory=list)
    report_errors_b: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "ModelComparison[Any]":
        return ModelComparison[SectionComparison](
            inp_path=data["inp_path"],
            inp_name=data["inp_name"],
            engine_a=data["engine_a"],
            engine_b=data["engine_b"],
            overall_distance=data["overall_distance"],
            section_comparisons=[
                SectionComparison.from_dict(item)
                for item in data.get("section_comparisons", [])
            ],
            report_warnings=list(data.get("report_warnings", [])),
            report_errors=list(data.get("report_errors", [])),
            report_warnings_a=list(data.get("report_warnings_a", [])),
            report_warnings_b=list(data.get("report_warnings_b", [])),
            report_errors_a=list(data.get("report_errors_a", [])),
            report_errors_b=list(data.get("report_errors_b", [])),
        )


@dataclasses.dataclass
class OutputSeriesComparison:
    element_type: str
    element_name: str
    attribute: str
    distance: float
    row_count_a: int
    row_count_b: int
    timestamps: list[str]
    values_a: list[float | None]
    values_b: list[float | None]
    source_point_count: int
    sampled: bool = False
    typical_distance: float | None = None
    event_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputSeriesComparison":
        timestamps = data.get("timestamps")
        if not isinstance(timestamps, list) or not all(
            isinstance(timestamp, str) for timestamp in timestamps
        ):
            raise ValueError(
                "Graphical comparison field 'timestamps' must be a list of strings"
            )
        if not timestamps:
            raise ValueError(
                "Graphical comparison field 'timestamps' must retain at least one sample"
            )
        values_a = _graph_values(data, "values_a")
        values_b = _graph_values(data, "values_b")
        if len(timestamps) != len(values_a) or len(timestamps) != len(values_b):
            raise ValueError(
                "Graphical comparison timestamps and value arrays must have equal lengths"
            )
        source_point_count = _graph_count(
            data,
            "source_point_count",
            minimum=len(timestamps),
        )
        sampled = data.get("sampled", False)
        if not isinstance(sampled, bool):
            raise ValueError("Graphical comparison field 'sampled' must be boolean")
        return cls(
            element_type=_graph_string(data, "element_type"),
            element_name=_graph_string(data, "element_name"),
            attribute=_graph_string(data, "attribute"),
            distance=_graph_number(data, "distance"),
            row_count_a=_graph_count(data, "row_count_a"),
            row_count_b=_graph_count(data, "row_count_b"),
            timestamps=timestamps,
            values_a=values_a,
            values_b=values_b,
            source_point_count=source_point_count,
            sampled=sampled,
            typical_distance=data.get("typical_distance"),
            event_distance=data.get("event_distance"),
        )


@dataclasses.dataclass
class OutputTimelineCoverage:
    timestamp_count_a: int | None = None
    timestamp_count_b: int | None = None
    shared_timestamp_count: int | None = None
    trailing_timestamp_count_a: int = 0
    trailing_timestamp_count_b: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputTimelineCoverage":
        return cls(
            timestamp_count_a=_timeline_count(data, "timestamp_count_a", None),
            timestamp_count_b=_timeline_count(data, "timestamp_count_b", None),
            shared_timestamp_count=_timeline_count(
                data, "shared_timestamp_count", None
            ),
            trailing_timestamp_count_a=_timeline_count(
                data, "trailing_timestamp_count_a", 0
            )
            or 0,
            trailing_timestamp_count_b=_timeline_count(
                data, "trailing_timestamp_count_b", 0
            )
            or 0,
        )


@dataclasses.dataclass
class OutputComparison(ModelComparison[OutputSectionComparison]):
    graphical_series: list[OutputSeriesComparison] = dataclasses.field(
        default_factory=list
    )
    graphical_unavailable_reason: str | None = None
    details_retained: bool = True
    metric: str = OUTPUT_DISTANCE_METRIC_NRMSE
    typical_weight: float | None = None
    typical_distance: float | None = None
    event_distance: float | None = None
    timeline_coverage: OutputTimelineCoverage = dataclasses.field(
        default_factory=OutputTimelineCoverage
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputComparison":
        unavailable_reason = data.get("graphical_unavailable_reason")
        if not isinstance(unavailable_reason, str):
            unavailable_reason = None
        raw_graphical_series = data.get("graphical_series", [])
        try:
            if not isinstance(raw_graphical_series, list):
                raise ValueError("Graphical comparison series must be a list")
            graphical_series = [
                OutputSeriesComparison.from_dict(item) for item in raw_graphical_series
            ]
        except (AttributeError, KeyError, TypeError, ValueError):
            graphical_series = []
            unavailable_reason = (
                "Graphical data could not be loaded because its persisted payload "
                "is invalid."
            )
        return cls(
            inp_path=data["inp_path"],
            inp_name=data["inp_name"],
            engine_a=data["engine_a"],
            engine_b=data["engine_b"],
            overall_distance=data["overall_distance"],
            section_comparisons=[
                OutputSectionComparison.from_dict(item)
                for item in data.get("section_comparisons", [])
            ],
            report_warnings=list(data.get("report_warnings", [])),
            report_errors=list(data.get("report_errors", [])),
            graphical_series=graphical_series,
            graphical_unavailable_reason=unavailable_reason,
            details_retained=(
                data["details_retained"]
                if isinstance(data.get("details_retained"), bool)
                else True
            ),
            metric=(
                data["metric"]
                if isinstance(data.get("metric"), str)
                else OUTPUT_DISTANCE_METRIC_LEGACY
            ),
            typical_weight=data.get("typical_weight"),
            typical_distance=data.get("typical_distance"),
            event_distance=data.get("event_distance"),
            timeline_coverage=(
                OutputTimelineCoverage.from_dict(data["timeline_coverage"])
                if isinstance(data.get("timeline_coverage"), dict)
                else OutputTimelineCoverage()
            ),
        )


@dataclasses.dataclass
class InterfaceArtifact:
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterfaceArtifact":
        return cls(
            path=data["path"],
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
        )


@dataclasses.dataclass
class InterfaceFamilyResult:
    family: str
    generator_identity: str
    consumer_identity: str
    baseline_identity: str
    self_comparison_identities: list[str]
    artifact: InterfaceArtifact | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterfaceFamilyResult":
        artifact = data.get("artifact")
        return cls(
            family=data["family"],
            generator_identity=data["generator_identity"],
            consumer_identity=data["consumer_identity"],
            baseline_identity=data["baseline_identity"],
            self_comparison_identities=data.get("self_comparison_identities", []),
            artifact=InterfaceArtifact.from_dict(artifact) if artifact else None,
        )


@dataclasses.dataclass
class BenchmarkResult:
    schema_version: str
    name: str
    timestamp: str
    platform: dict[str, Any]
    engine_results: list[EngineResult]
    comparisons: list[ModelComparison]
    output_comparisons: list[OutputComparison] = dataclasses.field(default_factory=list)
    interface_families: list[InterfaceFamilyResult] = dataclasses.field(
        default_factory=list
    )
    run_count: int | None = None
    run_order: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if not self.interface_families:
            data.pop("interface_families")
        if self.run_count is None:
            data.pop("run_count")
        if self.run_order is None:
            data.pop("run_order")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            timestamp=data["timestamp"],
            platform=data["platform"],
            engine_results=[
                EngineResult.from_dict(item) for item in data.get("engine_results", [])
            ],
            comparisons=[
                ModelComparison.from_dict(item) for item in data.get("comparisons", [])
            ],
            output_comparisons=[
                OutputComparison.from_dict(item)
                for item in data.get("output_comparisons", [])
            ],
            interface_families=[
                InterfaceFamilyResult.from_dict(item)
                for item in data.get("interface_families", [])
            ],
            run_count=data.get("run_count"),
            run_order=data.get("run_order"),
        )
