"""Explicit layout transitions use source/type proofs, not guessed cycle costs."""
from dataclasses import replace
from itertools import product
import pytest

from merlin.perf.global_planner import optimize_program
from merlin.perf.structural_planner import StructuralAxis, StructuralEvaluation, optimize_structure
from merlin.perf.structural_transitions import (
    StaticStridedLayout, StructuralQuantity, strided_copy_transition,
)
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation, CycleInterval, RegionAlternative, TransitionAlternative,
)
from merlin.xdsl_dialects.lowering.global_plan_emission import (
    BoundaryMapping, EmittedComponent, GlobalPlanEmission, dispatch_digest,
    verify_global_plan_emission,
)

AXES = (StructuralAxis("copy_payload", "bytes"), StructuralAxis("storage", "bytes"),
        StructuralAxis("scalar_sites", "sites"))


class LayoutAdapter:
    def __init__(self, shape, dtype, source, destination, placement):
        self.shape, self.dtype = shape, dtype
        self.source, self.destination, self.placement = source, destination, placement
        self.native = source.representation(dtype=dtype, placement=placement)
        self.packed = destination.representation(dtype=dtype, placement=placement)

    def program(self):
        return DispatchProgram("forward", [0], {
            name: Buffer(name, list(self.shape), self.dtype, "arg" if name == "x" else "intermediate",
                         0 if name == "x" else None) for name in ("x", "a", "y")},
            [Node("dispatch", "producer", ["x"], ["a"], captures=[]),
             Node("dispatch", "consumer", ["a"], ["y"], captures=[])], ["y"])

    def region_alternatives(self, program):
        def region(name, index, inputs, outputs):
            return RegionAlternative(name, (index,), name, self.placement,
                CycleInterval.unknown("no timing calibration"), inputs=inputs, outputs=outputs)
        return (region("producer", 0, (BufferRepresentation("x", self.native),),
                       (BufferRepresentation("a", self.native),)),
                region("consumer_native", 1, (BufferRepresentation("a", self.native),),
                       (BufferRepresentation("y", self.native),)),
                region("consumer_encoded", 1, (BufferRepresentation("a", self.packed),),
                       (BufferRepresentation("y", self.native),)))

    def boundary_representation(self, *args):
        return self.native

    def transition(self, *args, **kwargs):
        # The legacy cycle path cannot acquire an uncalibrated structural transition.
        return None

    def structural_transition(self, program, *, buffer, producer, consumer, source, destination):
        return strided_copy_transition(id="pack", buffer=buffer,
            producer=producer.id, consumer=consumer.id, source_layout=self.source,
            destination_layout=self.destination, dtype=self.dtype, placement=self.placement,
            provenance=("static address functions supplied by this capability profile",))

    def evaluate_structure(self, program, selected, transitions, axes):
        # Explicit same-unit composition belongs to the adapter; independent axes
        # are not added together or converted to cycles by the core.
        quantities = {item.name: item.amount for transition in transitions for item in transition.quantities}
        return StructuralEvaluation((quantities.get("scalar_load_payload", 0)+quantities.get("scalar_store_payload", 0),
            quantities.get("destination_storage", 0), 2 if transitions else 5),
            ("fixture static scalar implementation counts; not target timing",))


@pytest.mark.parametrize("source,destination,dtype,placement,payload,storage", [
    (StaticStridedLayout((2, 3), (3, 1), 12), StaticStridedLayout((2, 3), (4, 1), 16),
     "i16", "shared_sram", 24, 16),
    (StaticStridedLayout((3, 5), (5, 1), 60), StaticStridedLayout((3, 5), (1, 3), 60),
     "f32", "host_heap", 120, 60),
])
def test_distinct_capability_profiles_preserve_bits_and_count_real_address_work(
        source, destination, dtype, placement, payload, storage):
    adapter = LayoutAdapter(source.shape, dtype, source, destination, placement)
    result = optimize_structure(adapter.program(), adapter, axes=AXES)
    assert result.complete_plans == 2 and len(result.frontier) == 2
    encoded = next(item for item in result.frontier if item.plan.transitions)
    assert encoded.evaluation.values == (payload, storage, 2)
    transition = encoded.plan.transitions[0]
    assert not transition.cycles.resolved and not encoded.plan.cycles.resolved
    assert transition.to_dict()["dram_bytes"] is None
    assert not optimize_program(adapter.program(), adapter).resolved
    before = {}
    after = {}
    for ordinal, coordinate in enumerate(product(*(range(dim) for dim in source.shape))):
        address = source.offset_elements+sum(i*s for i, s in zip(coordinate, source.strides_elements))
        before[address] = ordinal-7  # include signed values; copies preserve raw scalar payload
    for coordinate in product(*(range(dim) for dim in source.shape)):
        src = source.offset_elements+sum(i*s for i, s in zip(coordinate, source.strides_elements))
        dst = destination.offset_elements+sum(i*s for i, s in zip(coordinate, destination.strides_elements))
        assert dst not in after
        after[dst] = before[src]
    assert sorted(after.values()) == sorted(before.values())


@pytest.mark.parametrize("layout", [StaticStridedLayout((2, 3), (2, 1), 12),
    StaticStridedLayout((2, 3), (4, 1), 12), StaticStridedLayout((2, -1), (3, 1), 12)])
def test_alias_out_of_bounds_and_dynamic_shapes_refuse(layout):
    with pytest.raises(ValueError):
        layout.validate("i16")


def test_exact_dependency_and_logical_type_mismatch_refuse():
    source, destination = StaticStridedLayout((2, 3), (3, 1), 12), StaticStridedLayout((2, 3), (4, 1), 16)
    class WrongEdge(LayoutAdapter):
        def structural_transition(self, *args, **kwargs):
            return replace(super().structural_transition(*args, **kwargs), consumer="wrong")
    adapter = WrongEdge(source.shape, "i16", source, destination, "memory")
    result = optimize_structure(adapter.program(), adapter, axes=AXES)
    assert len(result.frontier) == 1
    assert any("requested dependency" in refusal for refusal in result.refusals)
    program = adapter.program()
    program.buffers["a"].dtype = "i32"
    result = optimize_structure(program, LayoutAdapter(source.shape, "i16", source, destination, "memory"), axes=AXES)
    assert any("shape/dtype" in refusal for refusal in result.refusals)


def test_legacy_transition_unknown_cost_still_refuses():
    layout = StaticStridedLayout((2, 3), (3, 1), 12)
    rep = layout.representation(dtype="i16", placement="memory")
    with pytest.raises(ValueError, match="unresolved"):
        TransitionAlternative("copy", "copy", "x", None, "consumer", rep, rep,
                              CycleInterval.unknown("uncalibrated"))


def test_no_implicit_dtype_quantization_or_transport_conversion():
    a = StaticStridedLayout((2, 3), (3, 1), 12)
    b = StaticStridedLayout((2, 3), (4, 1), 16)
    transition = strided_copy_transition(id="t", buffer="a", producer="p", consumer="c",
        source_layout=a, destination_layout=b, dtype="i16", placement="memory", provenance=("static contract",))
    for representation in (replace(transition.destination, dtype="i8"),
                           replace(transition.destination, quantization="scale:2"),
                           replace(transition.destination, placement="other_memory")):
        with pytest.raises(ValueError):
            replace(transition, destination=representation)
    with pytest.raises(ValueError, match="exact derived"):
        replace(transition, quantities=())


def test_measured_axis_needs_receipt_and_cannot_smuggle_timing():
    with pytest.raises(ValueError, match="artifact hash"):
        StructuralQuantity("reads", 4, "bytes", "measured", "profile without receipt")
    with pytest.raises(ValueError, match="timing"):
        StructuralQuantity("latency", 4, "ms", "derived", "not a byte quantity")


def test_unknown_cycle_transition_uses_existing_exact_emission_accounting():
    adapter = LayoutAdapter((2, 3), "i16", StaticStridedLayout((2, 3), (3, 1), 12),
                            StaticStridedLayout((2, 3), (4, 1), 16), "memory")
    logical = adapter.program()
    result = optimize_structure(logical, adapter, axes=AXES)
    plan = next(item.plan for item in result.frontier if item.plan.transitions)
    buffers = {name: replace(buffer) for name, buffer in logical.buffers.items()}
    buffers["packed"] = Buffer("packed", [2, 3], "i16", "intermediate")
    dispatch = DispatchProgram("forward", [0], buffers,
        [Node("dispatch", "producer", ["x"], ["a"], captures=[]),
         Node("dispatch", "strided_copy", ["a"], ["packed"], captures=[]),
         Node("dispatch", "consumer_encoded", ["packed"], ["y"], captures=[])], ["y"])
    receipt = GlobalPlanEmission(dispatch, plan.digest, dispatch_digest(logical),
        (EmittedComponent("producer", (0,), (("x", "x"),), (("a", "a"),)),
         EmittedComponent("consumer_encoded", (2,), (("a", "packed"),), (("y", "y"),))),
        (EmittedComponent("pack", (1,), (("a", "a"),), (("a", "packed"),)),),
        (BoundaryMapping("input", "x", "x"), BoundaryMapping("output", "y", "y")),
        ("protocol fixture only; no claim of physical compiler emission",))
    assert verify_global_plan_emission(logical, plan, receipt) == []
    missing = replace(receipt, transitions=(replace(receipt.transitions[0], node_indices=()),))
    assert any("emitted no node" in item for item in verify_global_plan_emission(logical, plan, missing))
    receipt.dispatch.nodes[2].inputs = ["a"]
    bypassed = replace(receipt, regions=(receipt.regions[0], replace(
        receipt.regions[1], inputs=(("a", "a"),))))
    assert any("bypasses" in item for item in verify_global_plan_emission(logical, plan, bypassed))
