"""Target-neutral graph layout planning and exact structural accounting."""
from __future__ import annotations

from merlin.perf.physical_layout import (
    CapacityDemand,
    CapacityLimit,
    ConversionCapability,
    LayoutGraph,
    LayoutOp,
    LayoutRefusal,
    LayoutValue,
    OperatorEncoding,
    PhysicalEncoding,
    plan_physical_layout,
)


def _encoding(identifier: str, storage_bytes: int, *, space: str = "external",
              capacity: bool = False) -> PhysicalEncoding:
    return PhysicalEncoding(identifier, identifier, space, storage_bytes, capacity,
                            provenance=("fixture exact static address function",))


def _value(name: str, *, padded_bytes: int = 64, capacity: bool = False) -> LayoutValue:
    return LayoutValue(name, (1, 2, 2, 4), "i8", (
        _encoding("channels_first", 16, capacity=capacity),
        _encoding("channels_last", padded_bytes, capacity=capacity),
    ))


def _option(identifier: str, *, preferred: int = 0, demand: int = 0) -> OperatorEncoding:
    demands = ((CapacityDemand("local", demand, "fixture complete live working set"),)
               if demand else ())
    return OperatorEncoding(f"impl_{identifier}", identifier, preferred,
                            capacity_demands=demands,
                            provenance=("fixture exact operator layout",))


def _coupled(name: str, inputs: tuple[str, ...], outputs: tuple[str, ...],
             *, preferred_last: bool = False) -> LayoutOp:
    return LayoutOp(name, "layout_polymorphic", inputs, outputs, True, (
        _option("channels_first"),
        _option("channels_last", preferred=1 if preferred_last else 0),
    ))


def _boundary(name: str, inputs: tuple[str, ...] = (),
              outputs: tuple[str, ...] = ()) -> LayoutOp:
    return LayoutOp(name, "fixed_boundary", inputs, outputs, False,
                    (_option("channels_first"),))


def _copies(*, padded_full_write: bool = False) -> tuple[ConversionCapability, ...]:
    return (
        ConversionCapability(
            "first_to_last", "channels_first", "channels_last",
            "full_physical", "full_physical" if padded_full_write else "logical_payload",
            required_capabilities=("exact_layout_copy",),
            provenance=("fixture bit-preserving index map",),
        ),
        ConversionCapability(
            "last_to_first", "channels_last", "channels_first",
            "full_physical", "logical_payload",
            required_capabilities=("exact_layout_copy",),
            provenance=("fixture inverse bit-preserving index map",),
        ),
    )


def test_conv_elementwise_residual_graph_propagates_one_preferred_layout() -> None:
    names = ("x", "stem", "left", "right", "sum", "y")
    graph = LayoutGraph(
        tuple(_value(name) for name in names),
        (
            _boundary("input", outputs=("x",)),
            _coupled("stem_conv", ("x",), ("stem",), preferred_last=True),
            _coupled("left_conv", ("stem",), ("left",), preferred_last=True),
            _coupled("right_conv", ("stem",), ("right",), preferred_last=True),
            _coupled("residual_add", ("left", "right"), ("sum",)),
            _coupled("activation", ("sum",), ("y",)),
            _boundary("output", inputs=("y",)),
        ),
        ("exact_layout_copy",),
        _copies(),
    )
    result = plan_physical_layout(graph)
    assert result.plan is not None
    assert set(dict(result.plan.assignments).values()) == {"channels_last"}
    assert [(item.op, item.port) for item in result.plan.conversions] == [
        ("input", "output"), ("output", "input")]
    assert result.plan.cost.preferred_weight == 3
    assert result.exhausted
    assert result.to_dict()["performance_claim"] == "UNPROVEN"


def test_conversion_reports_logical_physical_movement_and_materialization() -> None:
    graph = LayoutGraph(
        (_value("x", padded_bytes=64), _value("y", padded_bytes=64)),
        (
            _boundary("input", outputs=("x",)),
            _coupled("accelerated", ("x",), ("y",), preferred_last=True),
        ),
        ("exact_layout_copy",),
        _copies(padded_full_write=True),
    )
    result = plan_physical_layout(graph)
    assert result.plan is not None
    conversion = result.plan.conversions[0]
    assert conversion.logical_payload_bytes == 16
    assert conversion.source_physical_bytes == 16
    assert conversion.destination_physical_bytes == 64
    assert conversion.physical_read_bytes == 16
    assert conversion.physical_write_bytes == 64
    assert conversion.movement_bytes == 80
    assert conversion.materialized_bytes == 64
    assert conversion.temporary_bytes == 16


def test_capacity_uses_lifetimes_plus_complete_operator_working_set() -> None:
    values = (
        LayoutValue("x", (16,), "i8", (_encoding(
            "resident", 16, space="local", capacity=True),)),
        LayoutValue("y", (16,), "i8", (_encoding(
            "resident", 16, space="local", capacity=True),)),
    )
    op = LayoutOp("compute", "accelerated", ("x",), ("y",), True,
                  (_option("resident", preferred=1, demand=8),))
    fit = plan_physical_layout(LayoutGraph(
        values, (op,), (), (), (CapacityLimit("local", 40, "fixture local capacity"),)))
    assert fit.plan is not None
    assert dict(fit.plan.peak_capacity_bytes) == {"local": 40}
    assert [(row.value, row.first_op, row.last_op) for row in fit.plan.lifetimes] == [
        ("x", 0, 0), ("y", 0, 0)]

    overflow = plan_physical_layout(LayoutGraph(
        values, (op,), (), (), (CapacityLimit("local", 39, "fixture local capacity"),)))
    assert overflow.plan is None
    assert "capacity_exceeded" in overflow.to_dict()["refusal_histogram"]

    unproved = plan_physical_layout(LayoutGraph(values, (op,), (), ()))
    assert unproved.plan is None
    assert "capacity_unproven" in unproved.to_dict()["refusal_histogram"]


def test_missing_conversion_and_capability_refuse_with_structured_reasons() -> None:
    values = (_value("x"), _value("y"))
    ops = (
        _boundary("input", outputs=("x",)),
        _coupled("accelerated", ("x",), ("y",), preferred_last=True),
    )
    missing_conversion = plan_physical_layout(LayoutGraph(values, ops, (), ()))
    assert missing_conversion.plan is not None
    assert dict(missing_conversion.plan.assignments)["x"] == "channels_first"
    assert "missing_exact_conversion" in missing_conversion.to_dict()["refusal_histogram"]

    missing_capability = plan_physical_layout(LayoutGraph(values, ops, (), _copies()))
    assert missing_capability.plan is not None
    assert dict(missing_capability.plan.assignments)["x"] == "channels_first"
    assert "conversion_capability_unavailable" in missing_capability.to_dict()[
        "refusal_histogram"]


def test_assignment_limit_refuses_instead_of_claiming_partial_optimality() -> None:
    graph = LayoutGraph(
        (_value("x"), _value("y")),
        (
            LayoutOp("use_x", "polymorphic", ("x",), (), True,
                     (_option("channels_first"), _option("channels_last"))),
            LayoutOp("use_y", "polymorphic", ("y",), (), True,
                     (_option("channels_first"), _option("channels_last"))),
        ),
        (), (), max_assignments=3,
    )
    result = plan_physical_layout(graph)
    assert result.plan is None
    assert result.candidate_assignments == 4
    assert result.searched_assignments == 0
    assert not result.exhausted
    assert result.to_dict()["optimal_for_supplied_exact_alternatives"] is False
    assert "assignment_limit_exceeded" in result.to_dict()["refusal_histogram"]


def test_capture_blocker_prevents_a_partial_graph_optimality_claim() -> None:
    value = LayoutValue("x", (16,), "i8", (_encoding("channels_first", 16),))
    op = LayoutOp("use", "fixed", ("x",), (), False, (_option("channels_first"),))
    result = plan_physical_layout(LayoutGraph(
        (value,), (op,), (), (),
        target_refusals=(LayoutRefusal(
            "source_op", "layout_contract_missing", "source op was not represented", (), True),),
    ))
    assert result.plan is not None
    assert result.to_dict()["optimal_for_supplied_exact_alternatives"] is False
