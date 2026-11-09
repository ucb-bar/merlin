"""Public mechanism checks for the target-independent native selector."""

from __future__ import annotations

import builtins
import itertools
import json
import os
import random
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.semantic_compiler import extract as native_extract
from merlin.semantic_compiler import search as native_search
from merlin.semantic_compiler.allocate import (
    AllocationResult,
    CandidateGraph,
    Reservation,
    StorageBank,
    Value,
    allocate,
    check_assignment,
    instruction_schedule,
    live_ranges,
    lower_candidate,
    may_prune_interference,
    topological_orders,
)
from merlin.semantic_compiler.egg_bridge import EGraphTimeout, explore
from merlin.semantic_compiler.extract import ExtractionTimeout, enumerate_candidates
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.rules import (
    AddressConstraint,
    AxisBound,
    AxisEquality,
    ConcatViewDescriptor,
    InstructionDescriptor,
    IntegerAttributeBound,
    ViewDescriptor,
    generate_rules,
)
from merlin.semantic_compiler.search import SearchAblations, SearchLimits, select_and_allocate
from merlin.semantic_compiler.snapshot import NativeTargetProfile, build_native_snapshot, open_native_snapshot
from merlin.semantic_compiler.verify import check_selection, check_timing


@pytest.fixture(scope="session")
def bridge(tmp_path_factory: pytest.TempPathFactory) -> Path:
    manifest = repo_root() / "src/merlin/semantic_compiler/egg_bridge/Cargo.toml"
    target = tmp_path_factory.mktemp("merlin-egg-build")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest)], check=True, env=env)
    binary = target / "release/merlin-egg-bridge"
    assert binary.is_file()
    return binary


def _type(policy: str = "exact") -> TensorType:
    return TensorType((2, 2), "i8", policy)


def _descriptor(
    name: str,
    computation: str,
    input_storages: tuple[str, ...],
    output_storage: str,
    output_dtype: str,
    numerical_policy: str,
    ranks: tuple[int, ...],
    required_attrs: tuple[tuple[str, str | int | float | bool], ...] = (),
    **kwargs: object,
) -> InstructionDescriptor:
    kwargs.setdefault("input_dtypes", (output_dtype,) * len(input_storages))
    kwargs.setdefault("input_numerical_policies", (numerical_policy,) * len(input_storages))
    return InstructionDescriptor(
        name,
        computation,
        input_storages,
        output_storage,
        output_dtype,
        numerical_policy,
        ranks,
        required_attrs,
        **kwargs,
    )


def _request() -> KernelRequest:
    tensor = _type()
    return KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("v", "identity", ("x",), tensor),
            SemanticNode("a", "use_a", ("v",), tensor),
            SemanticNode("b", "use_b", ("v",), tensor),
        ),
        outputs=("a", "b"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-revision-1",
    )


def _descriptors() -> tuple[InstructionDescriptor, ...]:
    return (
        _descriptor("load_a", "identity", ("external",), "a", "i8", "exact", (2,)),
        _descriptor("load_b", "identity", ("external",), "b", "i8", "exact", (2,)),
        _descriptor("finish_a", "use_a", ("a",), "external", "i8", "exact", (2,)),
        _descriptor("finish_b", "use_b", ("b",), "external", "i8", "exact", (2,)),
    )


def _banks() -> tuple[StorageBank, ...]:
    return (
        StorageBank("external", "dram", 4, "word"),
        StorageBank("a", "reg_a", 2, "word"),
        StorageBank("b", "reg_b", 2, "word"),
    )


def test_multi_result_instruction_shares_one_issue_and_two_physical_outputs(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("left", "pair_left", ("x",), tensor),
            SemanticNode("right", "pair_right", ("x",), tensor),
        ),
        outputs=("right", "left"),
        output_storages=("right_bank", "left_bank"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-pair-v1",
    )
    descriptors = (
        _descriptor(
            "pair_left_result",
            "pair_left",
            ("external",),
            "left_bank",
            "i8",
            "exact",
            (2,),
            input_read_offsets=(0,),
            completion_offset=1,
            bundle_name="pair",
            bundle_result_index=0,
            bundle_result_count=2,
        ),
        _descriptor(
            "pair_right_result",
            "pair_right",
            ("external",),
            "right_bank",
            "i8",
            "exact",
            (2,),
            input_read_offsets=(0,),
            completion_offset=1,
            bundle_name="pair",
            bundle_result_index=1,
            bundle_result_count=2,
        ),
    )
    banks = (
        StorageBank("external", "dram", 1, "word"),
        StorageBank("left_bank", "tensor_rf", 2, "slot"),
        StorageBank("right_bank", "tensor_rf", 2, "slot"),
    )
    result = select_and_allocate(
        request,
        descriptors,
        banks,
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(1, 0),
        limits=SearchLimits(candidate_nodes=1, candidates=8),
    )
    assert result.status == "selected", result.reason
    assert result.candidate is not None and result.graph is not None and result.allocation is not None
    assert result.candidate.bundles_complete()
    assert result.rules is not None and result.exploration is not None
    assert InstructionDescriptor.from_record(descriptors[0].record()) == descriptors[0]
    assert result.candidate.instruction_count() == 1
    assert [value.kind for value in result.graph.values] == ["input", "instruction", "projection"]
    assert result.graph.outputs == (2, 1)
    assert result.graph.value(2).projection_of == 1
    assert result.allocation.order == (1,)
    assert result.allocation.addresses[2] == 1 and result.allocation.addresses[1] == 0
    assert check_timing(result.graph, result.allocation.order, result.allocation.issue_times)[0]
    invalid = {**result.allocation.addresses, 2: 0}
    assert not check_assignment(
        result.graph,
        result.allocation.order,
        invalid,
        banks,
        fixed_inputs={"x": 0},
    )[0]
    for capacity in (1, 2, 3):
        physical = (
            banks[0],
            replace(banks[1], capacity=capacity),
            replace(banks[2], capacity=capacity),
        )
        expected = any(left != right for left, right in itertools.product(range(capacity), repeat=2))
        placement = allocate(result.graph, result.allocation.order, physical, fixed_inputs={"x": 0})
        assert (placement.status == "feasible") == expected
    aligned = replace(descriptors[1], validity=(AddressConstraint("aligned", "out", value=2),))
    aligned_result = select_and_allocate(
        request,
        (descriptors[0], aligned),
        (banks[0], replace(banks[1], capacity=3), replace(banks[2], capacity=3)),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(2, 0),
        limits=SearchLimits(candidate_nodes=1, candidates=8),
    )
    assert aligned_result.status == "selected", aligned_result.reason
    assert aligned_result.graph is not None and aligned_result.allocation is not None
    assert not check_assignment(
        aligned_result.graph,
        aligned_result.allocation.order,
        {**aligned_result.allocation.addresses, aligned_result.graph.outputs[0]: 1},
        (banks[0], replace(banks[1], capacity=3), replace(banks[2], capacity=3)),
        fixed_inputs={"x": 0},
        fixed_outputs=(1, 0),
    )[0]
    changed = CandidateGraph(
        (*result.graph.values[:2], replace(result.graph.value(2), projection_of=0)),
        result.graph.outputs,
    )
    assert not check_selection(
        request,
        descriptors,
        result.rules,
        result.exploration,
        result.candidate,
        changed,
        result.allocation,
        banks,
        fixed_inputs={"x": 0},
        fixed_outputs=(1, 0),
    ).valid

    absent = replace(request, nodes=request.nodes[:2], outputs=("left",), output_storages=("left_bank",))
    assert not any(item["kind"] == "instruction" for item in generate_rules(absent, descriptors).symbols.values())
    with pytest.raises(ValueError, match="incomplete"):
        generate_rules(request, descriptors[:1])
    with pytest.raises(ValueError, match="shared operands or timing"):
        generate_rules(request, (descriptors[0], replace(descriptors[1], input_read_offsets=(1,))))

    # A bundle projection cannot be silently paired with an unrelated
    # standalone instruction for the other source result.
    mixed_request = replace(request, output_storages=("alternate", "left_bank"))
    standalone = _descriptor(
        "standalone_right",
        "pair_right",
        ("external",),
        "alternate",
        "i8",
        "exact",
        (2,),
    )
    mixed_banks = (*banks, StorageBank("alternate", "other", 1, "slot"))
    mixed = select_and_allocate(
        mixed_request,
        (*descriptors, standalone),
        mixed_banks,
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert mixed.status != "selected"

    used_request = replace(
        request,
        nodes=(*request.nodes, SemanticNode("used", "consume_right", ("right",), tensor)),
        outputs=("used", "left"),
        output_storages=("external", "left_bank"),
    )
    consumer = _descriptor(
        "consume_right",
        "consume_right",
        ("right_bank",),
        "external",
        "i8",
        "exact",
        (2,),
    )
    used = select_and_allocate(
        used_request,
        (*descriptors, consumer),
        (replace(banks[0], capacity=2), *banks[1:]),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(1, 0),
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert used.status == "selected", used.reason
    assert used.graph is not None and used.allocation is not None
    owners = [value.id for value in used.graph.values if value.kind == "instruction"]
    assert len(owners) == 2 and used.allocation.order == tuple(owners)
    assert any(value.kind == "projection" and value.projection_of == owners[0] for value in used.graph.values)


def test_multi_result_projection_reserves_storage_from_shared_issue(bridge: Path) -> None:
    # A bundle may write either result anywhere in [issue, completion], so a
    # later result must not reuse a produced operand read before completion.
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("p", "prep", ("x",), tensor),
            SemanticNode("left", "pair_left", ("p",), tensor),
            SemanticNode("right", "pair_right", ("p",), tensor),
        ),
        outputs=("left", "right"),
        output_storages=("rf", "rf"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-pair-alias",
    )
    timing = {"input_read_offsets": (0,), "completion_offset": 3}
    descriptors = (
        _descriptor("prep", "prep", ("external",), "rf", "i8", "exact", (2,)),
        _descriptor(
            "pair_left_result",
            "pair_left",
            ("rf",),
            "rf",
            "i8",
            "exact",
            (2,),
            **timing,
            bundle_name="pair",
            bundle_result_index=0,
            bundle_result_count=2,
        ),
        _descriptor(
            "pair_right_result",
            "pair_right",
            ("rf",),
            "rf",
            "i8",
            "exact",
            (2,),
            **timing,
            bundle_name="pair",
            bundle_result_index=1,
            bundle_result_count=2,
        ),
    )

    def banks(capacity: int) -> tuple[StorageBank, ...]:
        return (StorageBank("external", "dram", 4, "word"), StorageBank("rf", "rf", capacity, "slot"))

    def select(capacity: int) -> native_search.SearchResult:
        return select_and_allocate(
            request,
            descriptors,
            banks(capacity),
            bridge=bridge,
            fixed_inputs={"x": 0},
            limits=SearchLimits(candidate_nodes=3, candidates=16),
        )

    assert select(2).status != "selected"
    result = select(3)
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    graph, allocation = result.graph, result.allocation
    kinds = {value.source_node: value for value in graph.values}
    prep, owner, projection = kinds["p"], kinds["left"], kinds["right"]
    assert owner.kind == "instruction" and projection.kind == "projection"
    assert projection.projection_of == owner.id
    ranges = live_ranges(graph, allocation.order)
    schedule = instruction_schedule(graph, allocation.order)
    assert ranges[projection.id][0] == schedule[owner.id][0]
    addresses = allocation.addresses
    assert len({addresses[prep.id], addresses[owner.id], addresses[projection.id]}) == 3
    aliased = {**addresses, projection.id: addresses[prep.id]}
    assert not check_assignment(graph, allocation.order, aliased, banks(3), fixed_inputs={"x": 0})[0]


def test_allocator_chooses_canonical_addresses_for_replay() -> None:
    graph = CandidateGraph(
        values=(
            Value(0, "first", "register", 1, (), None, "instruction"),
            Value(1, "second", "register", 1, (), None, "instruction"),
        ),
        outputs=(0, 1),
    )
    bank = (StorageBank("register", "physical_registers", 2, "register"),)
    results = [allocate(graph, (0, 1), bank) for _ in range(12)]
    assert all(result.status == "feasible" for result in results)
    assert all(result.addresses == {0: 0, 1: 1} for result in results)


def test_typed_graph_rejects_unknown_initialization_and_policy_merges() -> None:
    with pytest.raises(ValueError, match="positive static"):
        TensorType((0, 2), "i8", "exact")
    with pytest.raises(ValueError, match="stateful computation"):
        SemanticNode("bad", "dma_launch", (), _type(), effect="state")
    assert (
        SemanticNode("v", "sum", (), _type("ordered"), attrs=(("axis", 1),)).semantic_key()
        != SemanticNode("other", "sum", (), _type("reassociated"), attrs=(("axis", 1),)).semantic_key()
    )


def test_immutable_kernel_identity_is_reused_without_changing_serialization(monkeypatch) -> None:
    from merlin.semantic_compiler import model

    source = _request()
    record = source.record()
    original = model._json_bytes
    calls = []

    def serialize(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(model, "_json_bytes", serialize)
    first = source.digest()
    assert source.digest() == first
    assert len(calls) == 1
    assert source.record() == record
    assert KernelRequest.from_record(record).digest() == first
    assert replace(source, target_identity="another-profile").digest() != first


@pytest.mark.parametrize("container", ["outputs", "attrs", "attribute_pair", "map_rows"])
def test_mutable_source_containers_never_reuse_a_stale_digest(container) -> None:
    source = _request()
    if container == "outputs":
        values = list(source.outputs)
        source = replace(source, outputs=values)
        mutate = values.reverse
    elif container in {"attrs", "attribute_pair"}:
        pair = ["note", "before"] if container == "attribute_pair" else ("note", "before")
        values = (pair,) if container == "attribute_pair" else [pair]
        node = replace(source.nodes[1], attrs=values)
        source = replace(source, nodes=(source.nodes[0], node, *source.nodes[2:]))
        mutate = (lambda: pair.__setitem__(1, "after")) if container == "attribute_pair" else (
            lambda: values.__setitem__(0, ("note", "after"))
        )
    else:
        rows = [[1, 0], [0, 1]]
        mapping = IndexMap(2, rows, (0, 0))
        node = replace(source.nodes[1], index_maps=(mapping, mapping))
        source = replace(source, nodes=(source.nodes[0], node, *source.nodes[2:]))
        def mutate():
            rows[0][1] = 1
    first = source.digest()
    mutate()
    assert source.digest() != first


def test_semantic_kernel_round_trip_preserves_outputs_policies_and_rejects_unknown_fields() -> None:
    request = _request()
    encoded = request.record()
    assert KernelRequest.from_record(encoded).record() == encoded
    assert KernelRequest.from_record(encoded).digest() == request.digest()
    altered = dict(encoded)
    altered["outputs"] = list(reversed(encoded["outputs"]))
    assert KernelRequest.from_record(altered).digest() != request.digest()
    altered = dict(encoded)
    altered["unexpected"] = "executable escape hatch"
    with pytest.raises(ValueError, match="schema or fields"):
        KernelRequest.from_record(altered)


@pytest.mark.parametrize("physical", ("logical", "extents", "layouts"))
def test_input_boundary_order_survives_sorted_json_round_trip(physical: str) -> None:
    names = ("arg6", "arg7", "arg8", "arg9", "arg10")
    nodes = tuple(
        SemanticNode(name, "input", (), TensorType((i + 1,), "i64", "exact"), effect="input")
        for i, name in enumerate(names)
    )
    request = KernelRequest(
        nodes,
        tuple(reversed(names)),
        ("external",) * len(names),
        tuple((name, "external") for name in names),
        "synthetic-input-order",
    )
    if physical != "logical":
        request = replace(request, physical_extents=tuple((name, i + 1) for i, name in enumerate(names)))
    if physical == "layouts":
        request = replace(
            request,
            boundary_layouts=tuple((name, "contiguous") for name in names),
            output_layouts=("contiguous",) * len(names),
        )
    restored = KernelRequest.from_record(json.loads(json.dumps(request.record(), sort_keys=True)))
    assert restored.input_storages == request.input_storages
    assert restored == request
    assert restored.digest() == request.digest()
    reordered = replace(request, input_storages=tuple(reversed(request.input_storages)))
    assert reordered.digest() != request.digest()
    assert KernelRequest.from_record(reordered.record()) == reordered
    for invalid in (["arg6"] * len(names), list(names[:-1]), list(names[:-1]) + ["absent"]):
        encoded = request.record()
        encoded["input_order"] = invalid
        with pytest.raises(ValueError, match="input order"):
            KernelRequest.from_record(encoded)


def test_layout_annotated_copy_keeps_legacy_requests_usable(bridge: Path) -> None:
    tensor = _type()
    legacy = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "identity", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("register",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-layout-copy",
    )
    descriptor = _descriptor(
        "copy",
        "identity",
        ("external",),
        "register",
        "i8",
        "exact",
        (2,),
        value_preserving_copy=True,
        input_layouts=("external-row-major",),
        output_layout="register-row-major",
    )
    banks = (StorageBank("external", "dram", 1, "slot"), StorageBank("register", "regs", 1, "slot"))
    for request in (
        legacy,
        replace(legacy, physical_extents=(("x", 1),)),
        replace(
            legacy,
            physical_extents=(("x", 1),),
            boundary_layouts=(("x", "external-row-major"),),
            output_layouts=("register-row-major",),
        ),
    ):
        selected = select_and_allocate(
            request,
            (descriptor,),
            banks,
            bridge=bridge,
            fixed_inputs={"x": 0},
            fixed_outputs=(0,),
        )
        assert selected.status == "selected", (request.record()["schema"], selected.reason)

    wrong = replace(
        legacy,
        physical_extents=(("x", 1),),
        boundary_layouts=(("x", "wrong-layout"),),
        output_layouts=("register-row-major",),
    )
    rejected = select_and_allocate(
        wrong,
        (descriptor,),
        banks,
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(0,),
    )
    assert rejected.status != "selected"


def test_explicit_boundary_extents_survive_selection_and_constrain_allocation(bridge: Path) -> None:
    tensor = _type()
    legacy = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("c", "constant", (), tensor, effect="constant"),
            SemanticNode("y", "add", ("x", "c"), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-boundary-extent-v1",
        constants=(ConstantBinding("c", "i8", "01020304"),),
    )
    request = replace(legacy, physical_extents=(("x", 2), ("c", 2)))
    assert legacy.record()["schema"] == "merlin.semantic_kernel.v2"
    assert request.record()["schema"] == "merlin.semantic_kernel.v3"
    assert KernelRequest.from_record(request.record()) == request
    assert request.digest() != legacy.digest()
    with pytest.raises(ValueError, match="cover every input and constant"):
        replace(request, physical_extents=(("x", 2),))
    with pytest.raises(ValueError, match="cover every input and constant"):
        replace(request, physical_extents=(("x", True), ("c", 2)))

    descriptor = _descriptor("add", "add", ("external", "external"), "external", "i8", "exact", (2,))
    small_bank = (StorageBank("external", "memory", 4, "word"),)
    legacy_result = select_and_allocate(legacy, (descriptor,), small_bank, bridge=bridge, fixed_inputs={"x": 0})
    assert legacy_result.status == "selected", legacy_result.reason
    cramped = select_and_allocate(request, (descriptor,), small_bank, bridge=bridge, fixed_inputs={"x": 0})
    assert cramped.status == "compile_error" and cramped.rejected_allocation > 0, cramped.reason
    roomy = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "memory", 5, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0, "c": 2},
    )
    assert roomy.status == "selected", roomy.reason
    assert roomy.graph is not None and roomy.rules is not None and roomy.exploration is not None
    assert roomy.candidate is not None and roomy.allocation is not None
    assert {value.source_node: value.extent for value in roomy.graph.values if value.kind in {"input", "constant"}} == {
        "x": 2,
        "c": 2,
    }
    constant = next(value for value in roomy.graph.values if value.kind == "constant")
    assert roomy.allocation.addresses[constant.id] == 2
    moved_constant = dict(roomy.allocation.addresses)
    moved_constant[constant.id] = 3
    valid, reason = check_assignment(
        roomy.graph,
        roomy.allocation.order,
        moved_constant,
        (StorageBank("external", "memory", 5, "word"),),
        fixed_inputs={"x": 0, "c": 2},
    )
    assert not valid and "fixed external address" in reason
    invalid = allocate(
        roomy.graph,
        roomy.allocation.order,
        (StorageBank("external", "memory", 5, "word"),),
        fixed_inputs={"x": 0, "c": 4},
    )
    assert invalid.status == "modeling_failure" and "physical storage" in invalid.reason
    tampered = replace(request, physical_extents=(("x", 1), ("c", 2)))
    checked = check_selection(
        tampered,
        (descriptor,),
        roomy.rules,
        roomy.exploration,
        roomy.candidate,
        roomy.graph,
        roomy.allocation,
        (StorageBank("external", "memory", 5, "word"),),
        fixed_inputs={"x": 0, "c": 2},
    )
    assert not checked.valid and "physical extent differs" in checked.reason


def test_independent_exact_i32_reference_keeps_outputs_and_declared_constants_separate() -> None:
    tensor = TensorType((2, 2), "i32", "exact-i32")
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("c", "constant", (), tensor, effect="constant"),
            SemanticNode("sum", "add", ("x", "c"), tensor),
            SemanticNode("product", "multiply", ("sum", "x"), tensor),
        ),
        outputs=("product", "sum"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-exact-i32-reference-1",
        constants=(
            ConstantBinding(
                "c",
                "i32-le",
                b"".join(element.to_bytes(4, "little", signed=True) for element in (2, -3, 4, 0)).hex(),
            ),
        ),
    )
    constant = TensorValue(tensor, (2, -3, 4, 0))
    first = evaluate_graph(request, {"x": TensorValue(tensor, (1, 2, 3, 4))}, constants={"c": constant})
    assert [value.elements for value in first] == [(3, -2, 21, 16), (3, -1, 7, 4)]
    second = evaluate_graph(request, {"x": TensorValue(tensor, (5, 6, 7, 8))}, constants={"c": constant})
    assert [value.elements for value in second] == [(35, 18, 77, 64), (7, 3, 11, 8)]
    with pytest.raises(ValueError, match="declared constants"):
        evaluate_graph(request, {"x": TensorValue(tensor, (1, 2, 3, 4))})
    with pytest.raises(ValueError, match="compiler bytes"):
        evaluate_graph(
            request, {"x": TensorValue(tensor, (1, 2, 3, 4))}, constants={"c": TensorValue(tensor, (2, -3, 4, 1))}
        )
    with pytest.raises(ValueError, match="outside its admitted domain"):
        TensorValue(tensor, (1 << 31, 0, 0, 0))


def test_raw_byte_movement_reference_preserves_all_outputs_and_rejects_arithmetic() -> None:
    raw = TensorType((4,), "i8", "raw-byte-copy")
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), raw, effect="input"),
            SemanticNode("a", "identity", ("x",), raw),
            SemanticNode("b", "identity", ("a",), raw),
        ),
        outputs=("a", "b"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="movement-reference",
    )
    values = (0, 127, 128, 255)
    evaluated = evaluate_graph(request, {"x": TensorValue(raw, values)})
    assert [item.elements for item in evaluated] == [values, values]
    with pytest.raises(ValueError, match="outside its admitted domain"):
        TensorValue(raw, (0, 1, 2, 256))
    arithmetic = KernelRequest(
        nodes=(request.nodes[0], SemanticNode("a", "add", ("x", "x"), raw)),
        outputs=("a",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="movement-reference",
    )
    with pytest.raises(ValueError, match="raw-byte reference admits only"):
        evaluate_graph(arithmetic, {"x": TensorValue(raw, values)})


def test_bf16_sign_clamp_relu_keeps_positive_bits_and_rejects_special_inputs() -> None:
    from merlin.semantic_compiler.model import pointwise_identity_maps

    tensor = TensorType((2, 2), "bf16", "raw-bit-pattern")
    attrs = (("input_domain", "finite-normal-or-zero-bf16"), ("numeric_semantics", "bf16-sign-clamp-v1"))
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "relu", ("x",), tensor, attrs=attrs, index_maps=pointwise_identity_maps(2, 1)),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="bf16-sign-clamp-reference",
    )
    values = (0x8000, 0x0000, 0xBF80, 0x3E80)
    assert evaluate_graph(request, {"x": TensorValue(tensor, values)})[0].elements == (
        0x0000,
        0x0000,
        0x0000,
        0x3E80,
    )
    for special in (0x0001, 0x8001, 0x7F80, 0x7FC1, 0xFF80):
        with pytest.raises(ValueError, match="outside finite-normal-or-zero"):
            evaluate_graph(request, {"x": TensorValue(tensor, (special, *values[1:]))})
    changed = replace(request, nodes=(request.nodes[0], replace(request.nodes[1], attrs=())))
    with pytest.raises(ValueError, match="semantic contract"):
        evaluate_graph(changed, {"x": TensorValue(tensor, values)})


def test_verifier_owned_numeric_primitive_can_feed_raw_bf16_relu() -> None:
    from merlin.semantic_compiler.model import pointwise_identity_maps

    fp8 = TensorType((1, 1), "fp8_e4m3", "raw-bit-pattern")
    tagged = TensorType((1, 1), "bf16", "synthetic-ordered-v1")
    raw = TensorType((1, 1), "bf16", "raw-bit-pattern")
    request = KernelRequest(
        nodes=(
            SemanticNode("a", "input", (), fp8, effect="input"),
            SemanticNode("w", "input", (), fp8, effect="input"),
            SemanticNode("dot", "matmul", ("a", "w"), tagged),
            SemanticNode(
                "clamped",
                "relu",
                ("dot",),
                raw,
                attrs=(("input_domain", "finite-normal-or-zero-bf16"), ("numeric_semantics", "bf16-sign-clamp-v1")),
                index_maps=pointwise_identity_maps(2, 1),
            ),
        ),
        outputs=("dot", "clamped"),
        output_storages=("external", "external"),
        input_storages=(("a", "external"), ("w", "external")),
        target_identity="reference-primitive-test",
    )
    inputs = {"a": TensorValue(fp8, (0x38,)), "w": TensorValue(fp8, (0xB8,))}
    with pytest.raises(ValueError, match="no numerical policy"):
        evaluate_graph(request, inputs)

    def verify_dot(node: SemanticNode, operands: tuple[TensorValue, ...]) -> TensorValue:
        assert node.op == "matmul" and tuple(item.elements for item in operands) == ((0x38,), (0xB8,))
        return TensorValue(tagged, (0xBF80,))

    outputs = evaluate_graph(request, inputs, primitives={("matmul", tagged.numerical_policy): verify_dot})
    assert [result.elements for result in outputs] == [(0xBF80,), (0x0000,)]
    with pytest.raises(ValueError, match="wrong type"):
        evaluate_graph(
            request,
            inputs,
            primitives={("matmul", tagged.numerical_policy): lambda _node, _operands: TensorValue(raw, (0xBF80,))},
        )


def test_unqualified_source_reference_needs_an_explicit_diagnostic_callback() -> None:
    raw = TensorType((1, 1), "f32", "raw-bit-pattern")
    source = SemanticNode(
        "sum",
        "source_f32_reduce_add_unqualified",
        ("x", "init"),
        raw,
        attrs=(("numeric_semantics", "source-f32-add-unqualified-v1"), ("reduction_order", "unresolved")),
    )
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), raw, effect="input"),
            SemanticNode("init", "input", (), raw, effect="input"),
            source,
        ),
        outputs=("sum",),
        output_storages=("external",),
        input_storages=(("x", "external"), ("init", "external")),
        target_identity="source-reference-boundary",
    )
    inputs = {"x": TensorValue(raw, (0x3F800000,)), "init": TensorValue(raw, (0,))}
    key = source.semantic_key()
    observed: list[tuple[int, int]] = []

    def diagnostic(node: SemanticNode, operands: tuple[TensorValue, ...]) -> TensorValue:
        assert node is source
        observed.append(tuple(operand.elements[0] for operand in operands))
        return TensorValue(raw, (0x3F800000,))

    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(request, inputs, primitives={(source.op, raw.numerical_policy): diagnostic})
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(request, inputs, source_primitives={"different-signature": diagnostic})
    assert evaluate_graph(request, inputs, source_primitives={key: diagnostic})[0].elements == (0x3F800000,)
    assert observed == [(0x3F800000, 0)]
    changed = replace(
        request,
        nodes=(
            *request.nodes[:-1],
            replace(
                source,
                attrs=(
                    ("numeric_semantics", "source-f32-add-unqualified-v1"),
                    ("reduction_order", "axis-ascending"),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(changed, inputs, source_primitives={key: diagnostic})
    with pytest.raises(ValueError, match="wrong type"):
        evaluate_graph(request, inputs, source_primitives={key: lambda _node, _operands: None})


@pytest.mark.parametrize(
    ("dtype", "encoding", "values", "width"),
    (
        ("i8", "i8", (0, 127, 128, 255), 1),
        ("fp8_e4m3", "fp8_e4m3-raw", (0x00, 0x80, 0xFF, 0x7F), 1),
        ("bf16", "bf16-le-raw", (0x0000, 0x8000, 0x7F80, 0xFFFF), 2),
        ("f32", "f32-le-raw", (0x00000000, 0x80000000, 0x7F800000, 0x7FC00001), 4),
    ),
)
def test_opaque_constant_bytes_preserve_exact_bit_patterns(
    dtype: str, encoding: str, values: tuple[int, ...], width: int
) -> None:
    policy = "raw-byte-copy" if dtype == "i8" else "raw-bit-pattern"
    tensor = TensorType((2, 2), dtype, policy)
    data = b"".join(value.to_bytes(width, "little") for value in values).hex()
    transpose_maps = (
        IndexMap(2, ((0, 1), (1, 0)), (0, 0)),
        IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
    )
    request = KernelRequest(
        nodes=(
            SemanticNode("c", "constant", (), tensor, effect="constant"),
            SemanticNode("moved", "transpose", ("c",), tensor, index_maps=transpose_maps),
        ),
        outputs=("c", "moved"),
        output_storages=("external", "external"),
        input_storages=(),
        target_identity="opaque-bit-pattern-reference",
        constants=(ConstantBinding("c", encoding, data),),
    )
    outputs = evaluate_graph(request, {}, constants={"c": TensorValue(tensor, values)})
    assert tuple(output.elements for output in outputs) == (values, (values[0], values[2], values[1], values[3]))
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    with pytest.raises(ValueError, match="compiler bytes"):
        evaluate_graph(request, {}, constants={"c": TensorValue(tensor, (1, *values[1:]))})
    with pytest.raises(ValueError, match="tensor type"):
        replace(request, constants=(ConstantBinding("c", encoding, data[:-2]),))
    if dtype != "i8":
        arithmetic = replace(
            request,
            nodes=(request.nodes[0], SemanticNode("moved", "add", ("c", "c"), tensor)),
            outputs=("moved",),
            output_storages=("external",),
        )
        with pytest.raises(ValueError, match="raw-bit reference admits only"):
            evaluate_graph(arithmetic, {}, constants={"c": TensorValue(tensor, values)})


@pytest.mark.parametrize(
    ("dtype", "encoding", "data"),
    (("fp8_e4m3", "fp8_e4m3-raw", "0080ff7f"), ("bf16", "bf16-le-raw", "00000080807fffff")),
)
def test_native_selection_places_opaque_atlas_constant_through_generated_copy(
    bridge: Path, dtype: str, encoding: str, data: str
) -> None:
    tensor = TensorType((2, 2), dtype, "raw-bit-pattern")
    request = KernelRequest(
        nodes=(SemanticNode("c", "constant", (), tensor, effect="constant"),),
        outputs=("c",),
        output_storages=("register",),
        input_storages=(),
        target_identity="synthetic-opaque-copy",
        constants=(ConstantBinding("c", encoding, data),),
    )
    descriptor = _descriptor(
        "copy_to_register",
        "identity",
        ("external",),
        "register",
        dtype,
        "raw-bit-pattern",
        (2,),
        extent=1,
        value_preserving_copy=True,
    )
    result = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "dram", 2, "word"), StorageBank("register", "tensor_rf", 1, "word")),
        bridge=bridge,
        fixed_outputs=(0,),
    )
    assert result.status == "selected", result.reason
    assert result.candidate is not None and result.candidate.instruction_count() == 1
    assert result.graph is not None
    assert {value.kind for value in result.graph.values} == {"constant", "instruction"}


def test_constant_bytes_are_exact_compiler_inputs_and_change_request_identity() -> None:
    tensor = TensorType((2,), "i32", "exact-i32")
    nodes = (SemanticNode("c", "constant", (), tensor, effect="constant"),)
    with pytest.raises(ValueError, match="exactly one declared byte binding"):
        KernelRequest(nodes, ("c",), ("external",), (), "synthetic")
    first = KernelRequest(
        nodes,
        ("c",),
        ("external",),
        (),
        "synthetic",
        constants=(ConstantBinding("c", "i32-le", "01000000feffffff"),),
    )
    second = replace(first, constants=(ConstantBinding("c", "i32-le", "02000000feffffff"),))
    assert first.digest() != second.digest()
    assert KernelRequest.from_record(first.record()).record() == first.record()
    with pytest.raises(ValueError, match="tensor type"):
        replace(first, constants=(ConstantBinding("c", "i32-le", "01000000"),))
    with pytest.raises(ValueError, match="canonical lowercase hex"):
        ConstantBinding("c", "i32-le", "AB000000")


def test_native_selection_retains_declared_constant_identity_and_placement(bridge: Path) -> None:
    tensor = TensorType((1,), "i32", "exact-i32")
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("c", "constant", (), tensor, effect="constant"),
            SemanticNode("y", "add", ("x", "c"), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-constant-1",
        constants=(ConstantBinding("c", "i32-le", "feffffff"),),
    )
    descriptor = _descriptor("add", "add", ("external", "external"), "external", "i32", "exact-i32", (1,))
    bank = (StorageBank("external", "dram", 3, "word"),)
    result = select_and_allocate(
        request,
        (descriptor,),
        bank,
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(2,),
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    placed = [value for value in result.graph.values if value.kind == "constant"]
    assert len(placed) == 1 and placed[0].source_node == "c"
    assert result.allocation.addresses[placed[0].id] == 1
    altered = replace(request, constants=(ConstantBinding("c", "i32-le", "fdffffff"),))
    again = select_and_allocate(
        altered,
        (descriptor,),
        bank,
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(2,),
    )
    assert again.status == "selected" and again.request_digest != result.request_digest
    assert again.check_fingerprint != result.check_fingerprint


def test_exact_i32_reference_rejects_unknown_arithmetic_and_shape_errors() -> None:
    left_type = TensorType((1, 2), "i32", "exact-i32")
    right_type = TensorType((2, 1), "i32", "exact-i32")
    output_type = TensorType((1, 1), "i32", "exact-i32")
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), left_type, effect="input"),
            SemanticNode("y", "input", (), right_type, effect="input"),
            SemanticNode("z", "matmul", ("x", "y"), output_type),
        ),
        outputs=("z",),
        output_storages=("external",),
        input_storages=(("x", "external"), ("y", "external")),
        target_identity="synthetic-matmul-reference-1",
    )
    values = {"x": TensorValue(left_type, (2, -3)), "y": TensorValue(right_type, (4, 5))}
    assert evaluate_graph(request, values)[0].elements == (-7,)
    unknown = replace(
        request,
        nodes=(
            *request.nodes[:-1],
            replace(request.nodes[-1], op="unknown"),
        ),
    )
    with pytest.raises(ValueError, match="no semantics"):
        evaluate_graph(unknown, values)
    bad_shape = replace(
        request,
        nodes=(
            *request.nodes[:-1],
            replace(request.nodes[-1], type=left_type),
        ),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_graph(bad_shape, values)


def test_logical_index_maps_are_typed_and_part_of_semantic_identity() -> None:
    tensor = _type()
    identity = IndexMap(2, ((1, 0), (0, 1)), (0, 0))
    transposed = IndexMap(2, ((0, 1), (1, 0)), (0, 0))
    assert identity.apply((1, 0)) == (1, 0)
    assert transposed.apply((1, 0)) == (0, 1)
    a = SemanticNode("a", "copy", ("x",), tensor, index_maps=(identity, identity))
    b = SemanticNode("b", "copy", ("x",), tensor, index_maps=(transposed, identity))
    assert a.semantic_key() != b.semantic_key()
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"), a),
        outputs=("a",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-index-map-1",
    )
    assert KernelRequest.from_record(request.record()).record() == request.record()
    absent_map = _descriptor("copy", "copy", ("external",), "external", "i8", "exact", (2,))
    assert not absent_map.accepts(a, (request.node("x"),))
    bound_map = _descriptor(
        "copy",
        "copy",
        ("external",),
        "external",
        "i8",
        "exact",
        (2,),
        index_maps=(identity, identity),
    )
    assert bound_map.accepts(a, (request.node("x"),))
    assert not bound_map.accepts(b, (request.node("x"),))
    bad_rank = IndexMap(2, ((1, 0),), (0,))
    with pytest.raises(ValueError, match="result rank"):
        KernelRequest(
            nodes=(
                SemanticNode("x", "input", (), tensor, effect="input"),
                SemanticNode("a", "copy", ("x",), tensor, index_maps=(bad_rank, identity)),
            ),
            outputs=("a",),
            output_storages=("external",),
            input_storages=(("x", "external"),),
            target_identity="synthetic-index-map-1",
        )


def test_fixed_cross_shape_instruction_matches_only_declared_port_shapes(bridge: Path) -> None:
    source_type = TensorType((32, 32), "i8", "exact")
    result_type = TensorType((64, 16), "i8", "exact")
    source = SemanticNode("x", "input", (), source_type, effect="input")
    reduction = SemanticNode("y", "reduce_column", ("x",), result_type)
    descriptor = _descriptor(
        "fixed_reduction",
        "reduce_column",
        ("register",),
        "register",
        "i8",
        "exact",
        (2,),
        input_ranks=(2,),
        shape_contract="bounded",
        output_axis_bounds=(AxisBound(0, 64, 64), AxisBound(1, 16, 16)),
        input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
    )
    assert InstructionDescriptor.from_record(descriptor.record()) == descriptor
    assert descriptor.accepts(reduction, (source,))
    request = KernelRequest(
        nodes=(source, reduction),
        outputs=("y",),
        output_storages=("register",),
        input_storages=(("x", "register"),),
        target_identity="synthetic-cross-shape-1",
    )
    assert any(rule.descriptor_name == "fixed_reduction" for rule in generate_rules(request, (descriptor,)).rewrites)
    selected = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("register", "regs", 2, "tile"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(1,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.graph is not None and selected.allocation is not None
    assert selected.allocation.addresses[selected.graph.outputs[0]] == 1
    assert not descriptor.accepts(replace(reduction, type=TensorType((32, 32), "i8", "exact")), (source,))
    assert not descriptor.accepts(reduction, (replace(source, type=TensorType((32, 33), "i8", "exact")),))
    with pytest.raises(ValueError, match="exact bounds on every port axis"):
        replace(descriptor, output_axis_bounds=(AxisBound(0, 64, 65), AxisBound(1, 16, 16)))
    with pytest.raises(ValueError, match="exact bounds on every port axis"):
        replace(descriptor, input_axis_bounds=((AxisBound(0, 32, 32),),))


def test_generated_rules_depend_on_descriptor_and_numerical_policy() -> None:
    request = _request()
    with pytest.raises(ValueError, match="explicit dtype and numerical policy"):
        InstructionDescriptor("missing_input_contract", "identity", ("external",), "a", "i8", "exact", (2,))
    rules = generate_rules(request, _descriptors())
    assert len(rules.rewrites) == 5
    wrong_shape = replace(request.node("v"), type=TensorType((2, 3), "i8", "exact"))
    assert not _descriptors()[0].accepts(wrong_shape, (request.node("x"),))
    with pytest.raises(ValueError, match="axis exceeds"):
        _descriptor(
            "invalid_shape",
            "identity",
            ("external",),
            "a",
            "i8",
            "exact",
            (2,),
            input_ranks=(2,),
            shape_contract="relations",
            shape_equalities=(AxisEquality("in0", 2, "out", 0),),
        )
    changed = _descriptor("load_a", "identity", ("external",), "a", "i8", "exact", (2,), (("missing", 1),))
    assert len(generate_rules(request, (changed,)).rewrites) == 1
    altered_source = replace(
        request,
        nodes=(*request.nodes[:2], replace(request.nodes[2], attrs=(("rounding", "different"),)), request.nodes[3]),
    )
    assert len(generate_rules(altered_source, _descriptors()).rewrites) == 4
    wrong_policy = _descriptor("load_a", "identity", ("external",), "a", "i8", "rounded", (2,))
    assert len(generate_rules(request, (wrong_policy,)).rewrites) == 1
    wrong_input = _descriptor("load_a", "identity", ("external",), "a", "i8", "exact", (2,), input_dtypes=("bf16",))
    assert len(generate_rules(request, (wrong_input,)).rewrites) == 1
    bounded = _descriptor(
        "load_a",
        "identity",
        ("external",),
        "a",
        "i8",
        "exact",
        (2,),
        output_axis_bounds=(AxisBound(0, minimum=2, maximum=16, multiple=2),),
    )
    assert len(generate_rules(request, (bounded,)).rewrites) == 2
    changed_geometry = _descriptor(
        "load_a",
        "identity",
        ("external",),
        "a",
        "i8",
        "exact",
        (2,),
        output_axis_bounds=(AxisBound(0, minimum=4, maximum=16, multiple=2),),
    )
    assert len(generate_rules(request, (changed_geometry,)).rewrites) == 1
    delayed = replace(_descriptors()[0], input_read_offsets=(3,), completion_offset=4)
    assert InstructionDescriptor.from_record(delayed.record()) == delayed
    ordinary_rule = next(rule for rule in rules.rewrites if rule.descriptor_name == "load_a")
    delayed_rule = next(
        rule for rule in generate_rules(request, (delayed,)).rewrites if rule.descriptor_name == "load_a"
    )
    assert delayed_rule.descriptor_digest != ordinary_rule.descriptor_digest
    with pytest.raises(ValueError, match="duplicate required attribute"):
        _descriptor(
            "invalid",
            "identity",
            ("external",),
            "a",
            "i8",
            "exact",
            (2,),
            required_attrs=(("axis", 0), ("axis", 1)),
        )


def test_parameterized_integer_attribute_is_bound_and_replayed(bridge: Path) -> None:
    tensor = _type()
    descriptor = _descriptor(
        "scaled_pack",
        "pack",
        ("external",),
        "external",
        "i8",
        "exact",
        (2,),
        required_attrs=(("rounding", "nearest"),),
        parameter_attrs=(IntegerAttributeBound("scale_code", 122, 128),),
    )
    assert InstructionDescriptor.from_record(descriptor.record()) == descriptor
    assert "parameter_attrs" not in _descriptors()[0].record()

    def make_request(code: int | bool) -> KernelRequest:
        return KernelRequest(
            nodes=(
                SemanticNode("x", "input", (), tensor, effect="input"),
                SemanticNode("y", "pack", ("x",), tensor, attrs=(("rounding", "nearest"), ("scale_code", code))),
            ),
            outputs=("y",),
            output_storages=("external",),
            input_storages=(("x", "external"),),
            target_identity="synthetic-parameter-1",
        )

    for code in (122, 128):
        request = make_request(code)
        program = generate_rules(request, (descriptor,))
        selected_symbols = [symbol for symbol, metadata in program.symbols.items() if metadata["kind"] == "instruction"]
        assert len(selected_symbols) == 1
        assert program.symbols[selected_symbols[0]]["instantiated_attrs"]["scale_code"] == code
        selected = select_and_allocate(
            request,
            (descriptor,),
            (StorageBank("external", "memory", 2, "tile"),),
            bridge=bridge,
            fixed_inputs={"x": 0},
            fixed_outputs=(1,),
        )
        assert selected.status == "selected", selected.reason
        assert selected.rules is not None and selected.exploration is not None
        assert selected.candidate is not None and selected.graph is not None and selected.allocation is not None
        altered_symbols = {key: dict(value) for key, value in selected.rules.symbols.items()}
        altered_symbols[selected_symbols[0]]["instantiated_attrs"] = {
            "rounding": "nearest",
            "scale_code": 127 if code == 122 else 122,
        }
        tampered = check_selection(
            request,
            (descriptor,),
            replace(selected.rules, symbols=altered_symbols),
            selected.exploration,
            selected.candidate,
            selected.graph,
            selected.allocation,
            (StorageBank("external", "memory", 2, "tile"),),
            fixed_inputs={"x": 0},
            fixed_outputs=(1,),
        )
        assert not tampered.valid and "changed attribute substitution" in tampered.reason

    for invalid in (121, 129, True):
        assert not any(
            rule.descriptor_name == "scaled_pack"
            for rule in generate_rules(make_request(invalid), (descriptor,)).rewrites
        )
    with pytest.raises(ValueError, match="conflicting parameter attribute"):
        replace(descriptor, parameter_attrs=(IntegerAttributeBound("rounding", 0, 1),))
    with pytest.raises(ValueError, match="ordered finite interval"):
        IntegerAttributeBound("scale_code", 129, 122)


def test_input_axis_precondition_rejects_a_different_reduction_length() -> None:
    source_type = TensorType((2, 3), "i8", "exact")
    output_type = TensorType((2, 2), "i8", "exact")
    nodes = (
        SemanticNode("x", "input", (), source_type, effect="input"),
        SemanticNode("y", "reduce", ("x",), output_type),
    )
    request = KernelRequest(nodes, ("y",), ("external",), (("x", "external"),), "synthetic-input-bound-1")
    descriptor = _descriptor(
        "reduce_k3",
        "reduce",
        ("external",),
        "external",
        "i8",
        "exact",
        (2,),
        input_ranks=(2,),
        input_axis_bounds=((AxisBound(1, 3, 3),),),
        shape_contract="relations",
        shape_equalities=(AxisEquality("in0", 0, "out", 0),),
    )
    assert InstructionDescriptor.from_record(descriptor.record()) == descriptor
    assert descriptor.accepts(nodes[1], (nodes[0],))
    changed = replace(request, nodes=(replace(nodes[0], type=TensorType((2, 4), "i8", "exact")), nodes[1]))
    assert not any(rule.descriptor_name == "reduce_k3" for rule in generate_rules(changed, (descriptor,)).rewrites)
    k4 = replace(descriptor, input_axis_bounds=((AxisBound(1, 4, 4),),))
    assert any(rule.descriptor_name == "reduce_k3" for rule in generate_rules(changed, (k4,)).rewrites)
    with pytest.raises(ValueError, match="explicit rank"):
        replace(descriptor, input_ranks=())
    with pytest.raises(ValueError, match="duplicated"):
        replace(descriptor, input_axis_bounds=((AxisBound(1, 3, 3), AxisBound(1, 3, 3)),))


def test_two_consumers_extract_same_value_into_different_banks(bridge: Path) -> None:
    request = _request()
    rules = generate_rules(request, _descriptors())
    graph = explore(rules, bridge=bridge)
    candidates = list(enumerate_candidates(graph, request, rules, node_budget=4, max_candidates=10))
    assert candidates
    result = select_and_allocate(request, _descriptors(), _banks(), bridge=bridge, fixed_inputs={"x": 0})
    assert result.status == "selected", result.reason
    assert result.engine == "merlin_native"
    assert result.graph is not None and result.allocation is not None
    assert {value.storage for value in result.graph.values if value.kind == "instruction"} == {"a", "b", "external"}
    assert result.allocation.addresses
    assert result.check_fingerprint
    assert result.candidate is not None and result.rules is not None and result.exploration is not None
    checked = check_selection(
        request,
        _descriptors(),
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        result.allocation,
        _banks(),
        fixed_inputs={"x": 0},
    )
    assert checked.valid and checked.fingerprint == result.check_fingerprint
    changed = (replace(_descriptors()[0], extent=2), *_descriptors()[1:])
    tampered = check_selection(
        request,
        changed,
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        result.allocation,
        _banks(),
        fixed_inputs={"x": 0},
    )
    assert not tampered.valid and "differs from the target descriptor" in tampered.reason
    moved_issue = replace(result.allocation, issue_times=((result.allocation.order[0], 99),))
    invalid_schedule = check_selection(
        request,
        _descriptors(),
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        moved_issue,
        _banks(),
        fixed_inputs={"x": 0},
    )
    assert not invalid_schedule.valid and "schedule differs" in invalid_schedule.reason


def test_bit_preserving_rewrite_exposes_instruction_without_copy(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("view", "identity", ("x",), tensor),
            SemanticNode("y", "consume", ("view",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-rewrite-1",
    )
    consume = _descriptor("consume", "consume", ("external",), "external", "i8", "exact", (2,))
    result = select_and_allocate(
        request,
        (consume,),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None
    assert [value.kind for value in result.graph.values] == ["input", "instruction"]
    assert result.rules is not None
    assert any(rule.name.startswith("structural_identity_v1") for rule in result.rules.rewrites)
    ablated = select_and_allocate(
        request,
        (consume,),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
        ablations=SearchAblations(disable_structural_rewrites=True),
    )
    assert ablated.status != "selected"
    assert ablated.diagnostic_ablations == ("disable_structural_rewrites",)
    assert ablated.diagnostic_only
    assert ablated.rules is not None
    assert not any(rule.descriptor_name == "<structural>" for rule in ablated.rules.rewrites)


def test_value_copy_source_precondition_changes_generated_rule() -> None:
    tensor = _type()
    source = SemanticNode("x", "input", (), tensor, attrs=(("input_domain", "seven"),), effect="input")
    result = SemanticNode("y", "identity", ("x",), tensor)
    request = KernelRequest(
        (source, result),
        ("y",),
        ("external",),
        (("x", "external"),),
        "synthetic-copy-precondition-v1",
    )
    descriptor = _descriptor(
        "qualified_load",
        "identity",
        ("external",),
        "register",
        "i8",
        "exact",
        (2,),
        value_preserving_copy=True,
        source_required_attrs=(("input_domain", "seven"),),
    )
    assert InstructionDescriptor.from_record(descriptor.record()) == descriptor
    selected = generate_rules(request, (descriptor,))
    assert [rule.source_node for rule in selected.rewrites if rule.descriptor_name == "qualified_load"] == ["x"]
    changed = replace(request, nodes=(replace(source, attrs=(("input_domain", "other"),)), result))
    assert not any(rule.descriptor_name == "qualified_load" for rule in generate_rules(changed, (descriptor,)).rewrites)
    with pytest.raises(ValueError, match="source attributes require"):
        replace(descriptor, value_preserving_copy=False)
    with pytest.raises(ValueError, match="duplicate source"):
        replace(descriptor, source_required_attrs=(("input_domain", "seven"), ("input_domain", "other")))


def test_generated_operand_policy_alternative_keeps_source_identity(bridge: Path) -> None:
    produced = TensorType((2, 2), "i8", "rounded-by-producer")
    consumed = TensorType((2, 2), "i8", "raw-bit-pattern")
    request = KernelRequest(
        (
            SemanticNode("x", "input", (), produced, effect="input"),
            SemanticNode("y", "consume", ("x",), consumed),
        ),
        ("y",),
        ("external",),
        (("x", "external"),),
        "synthetic-operand-policy-v1",
    )
    load = _descriptor(
        "load",
        "identity",
        ("external",),
        "register",
        "i8",
        "rounded-by-producer",
        (2,),
        value_preserving_copy=True,
    )
    consume = _descriptor(
        "consume",
        "consume",
        ("register",),
        "external",
        "i8",
        "raw-bit-pattern",
        (2,),
        input_policy_alternatives=(("rounded-by-producer",),),
    )
    assert InstructionDescriptor.from_record(consume.record()) == consume
    assert consume.accepts_input_policy(0, "raw-bit-pattern")
    assert consume.accepts_input_policy(0, "rounded-by-producer")
    assert not consume.accepts_input_policy(0, "unrelated")
    assert not consume.accepts_input_policy(1, "rounded-by-producer")
    with pytest.raises(ValueError, match="alternatives must match"):
        replace(consume, input_policy_alternatives=((), ()))
    with pytest.raises(ValueError, match="distinct, nonempty"):
        replace(consume, input_policy_alternatives=(("raw-bit-pattern",),))
    with pytest.raises(ValueError, match="lists of policy names"):
        InstructionDescriptor.from_record({**consume.record(), "input_policy_alternatives": ["rounded-by-producer"]})
    with pytest.raises(ValueError, match="value-preserving copy"):
        replace(load, input_policy_alternatives=(("raw-bit-pattern",),))

    generated = generate_rules(request, (load, consume))
    assert any(rule.descriptor_name == "consume" for rule in generated.rewrites)
    without_alternative = replace(consume, input_policy_alternatives=())
    assert not any(
        rule.descriptor_name == "consume" for rule in generate_rules(request, (load, without_alternative)).rewrites
    )
    changed_input = replace(request, nodes=(replace(request.nodes[0], type=_type("unrelated")), request.nodes[1]))
    assert not any(
        rule.descriptor_name == "consume" for rule in generate_rules(changed_input, (load, consume)).rewrites
    )
    selected = select_and_allocate(
        request,
        (load, consume),
        (StorageBank("external", "dram", 4, "tile"), StorageBank("register", "registers", 1, "tile")),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(3,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.graph is not None and selected.rules is not None
    assert selected.candidate is not None and selected.allocation is not None
    assert selected.exploration is not None
    assert request.node("x").type.numerical_policy == "rounded-by-producer"
    checked = check_selection(
        request,
        (load, consume),
        selected.rules,
        selected.exploration,
        selected.candidate,
        selected.graph,
        selected.allocation,
        (StorageBank("external", "dram", 4, "tile"), StorageBank("register", "registers", 1, "tile")),
        fixed_inputs={"x": 0},
        fixed_outputs=(3,),
    )
    assert checked.valid
    changed = check_selection(
        request,
        (load, without_alternative),
        selected.rules,
        selected.exploration,
        selected.candidate,
        selected.graph,
        selected.allocation,
        (StorageBank("external", "dram", 4, "tile"), StorageBank("register", "registers", 1, "tile")),
        fixed_inputs={"x": 0},
        fixed_outputs=(3,),
    )
    assert not changed.valid


def test_generated_value_copy_materializes_direct_operand_without_source_copy(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "consume", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-value-copy-1",
    )
    copy = _descriptor("load", "identity", ("external",), "register", "i8", "exact", (2,), value_preserving_copy=True)
    consume = _descriptor("consume", "consume", ("register",), "external", "i8", "exact", (2,))
    with pytest.raises(ValueError, match="value-preserving copy"):
        replace(consume, value_preserving_copy=True)
    no_copy = generate_rules(request, (replace(copy, value_preserving_copy=False), consume))
    assert not any(rule.name.startswith("materialize_") for rule in no_copy.rewrites)
    wrong_policy = replace(copy, numerical_policy="other", input_numerical_policies=("other",))
    assert not any(rule.name.startswith("materialize_") for rule in generate_rules(request, (wrong_policy,)).rewrites)

    program = generate_rules(request, (copy, consume))
    assert len([rule for rule in program.rewrites if rule.name.startswith("materialize_")]) == 2
    result = select_and_allocate(
        request,
        (copy, consume),
        (StorageBank("external", "dram", 4, "tile"), StorageBank("register", "registers", 1, "tile")),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(3,),
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.rules is not None and result.allocation is not None
    selected = [
        result.rules.symbols[value.symbol]["descriptor"]["name"]
        for value in result.graph.values
        if value.kind == "instruction"
    ]
    assert selected == ["load", "consume"]
    assert result.allocation.addresses[result.graph.outputs[0]] == 3
    assert result.check_fingerprint
    assert result.candidate is not None and result.exploration is not None
    changed_symbols = dict(result.rules.symbols)
    copy_symbol = next(
        symbol
        for symbol, metadata in changed_symbols.items()
        if metadata.get("realization") == "value_preserving_copy_v1"
    )
    changed_symbols[copy_symbol] = dict(changed_symbols[copy_symbol], realization="unknown_copy")
    replay = check_selection(
        request,
        (copy, consume),
        replace(result.rules, symbols=changed_symbols),
        result.exploration,
        result.candidate,
        result.graph,
        result.allocation,
        (StorageBank("external", "dram", 4, "tile"), StorageBank("register", "registers", 1, "tile")),
        fixed_inputs={"x": 0},
        fixed_outputs=(3,),
    )
    assert not replay.valid and "unknown realization" in replay.reason


def test_one_semantic_value_gets_two_declared_storage_realizations(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "consume_a", ("x",), tensor),
            SemanticNode("b", "consume_b", ("x",), tensor),
        ),
        outputs=("a", "b"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-dual-storage-copy-1",
    )
    descriptors = (
        _descriptor("load_a", "identity", ("external",), "a", "i8", "exact", (2,), value_preserving_copy=True),
        _descriptor("load_b", "identity", ("external",), "b", "i8", "exact", (2,), value_preserving_copy=True),
        _descriptor("consume_a", "consume_a", ("a",), "external", "i8", "exact", (2,)),
        _descriptor("consume_b", "consume_b", ("b",), "external", "i8", "exact", (2,)),
    )
    result = select_and_allocate(
        request,
        descriptors,
        (
            StorageBank("external", "dram", 4, "tile"),
            StorageBank("a", "reg_a", 1, "tile"),
            StorageBank("b", "reg_b", 1, "tile"),
        ),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(2, 3),
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.rules is not None and result.allocation is not None
    names = [
        result.rules.symbols[value.symbol]["descriptor"]["name"]
        for value in result.graph.values
        if value.kind == "instruction"
    ]
    assert set(names) == {"load_a", "load_b", "consume_a", "consume_b"}
    assert len(result.graph.values) == 5
    assert tuple(result.allocation.addresses[value] for value in result.graph.outputs) == (2, 3)
    assert result.check_fingerprint


def test_two_outputs_share_one_instruction_under_one_node_budget(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"), SemanticNode("y", "copy", ("x",), tensor)),
        outputs=("y", "y"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-shared-output-1",
    )
    descriptor = _descriptor("copy", "copy", ("external",), "external", "i8", "exact", (2,))
    program = generate_rules(request, (descriptor,))
    exploration = explore(program, bridge=bridge)
    candidates = list(enumerate_candidates(exploration, request, program, node_budget=1, max_candidates=2))
    assert len(candidates) == 1 and candidates[0].instruction_count() == 1
    selected = lower_candidate(candidates[0], program)
    assert selected.outputs == (1, 1) and len(selected.values) == 2


def test_diamond_charges_shared_producer_once_within_one_root(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("prepared", "prepare", ("x",), tensor),
            SemanticNode("y", "combine", ("prepared", "prepared"), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-shared-diamond-1",
    )
    descriptors = (
        _descriptor("prepare", "prepare", ("external",), "register", "i8", "exact", (2,)),
        _descriptor("combine", "combine", ("register", "register"), "external", "i8", "exact", (2,)),
    )
    program = generate_rules(request, descriptors)
    exploration = explore(program, bridge=bridge)
    assert not list(enumerate_candidates(exploration, request, program, node_budget=1, max_candidates=2))
    candidates = list(enumerate_candidates(exploration, request, program, node_budget=2, max_candidates=2))
    assert len(candidates) == 1 and candidates[0].instruction_count() == 2
    graph = lower_candidate(candidates[0], program)
    assert len(graph.values) == 3
    assert graph.values[-1].children == (1, 1)


def test_repeated_instruction_signature_keeps_each_source_correspondence(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "input", (), tensor, effect="input"),
            SemanticNode("a", "copy", ("x",), tensor),
            SemanticNode("b", "copy", ("y",), tensor),
        ),
        outputs=("a", "b"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"), ("y", "external")),
        target_identity="synthetic-repeated-signature-1",
    )
    descriptor = _descriptor("copy", "copy", ("external",), "external", "i8", "exact", (2,))
    program = generate_rules(request, (descriptor,))
    selected_symbols = {
        symbol: metadata["source_node"]
        for symbol, metadata in program.symbols.items()
        if metadata["kind"] == "instruction"
    }
    assert len(selected_symbols) == 2 and set(selected_symbols.values()) == {"a", "b"}
    result = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "dram", 4, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0, "y": 1},
    )
    assert result.status == "selected", result.reason
    assert result.check_fingerprint


def test_identical_pure_expressions_share_an_instruction(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "copy", ("x",), tensor),
            SemanticNode("b", "copy", ("x",), tensor),
        ),
        outputs=("a", "b"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-identical-sources-1",
    )
    descriptor = _descriptor("copy", "copy", ("external",), "external", "i8", "exact", (2,))
    program = generate_rules(request, (descriptor,))
    exploration = explore(program, bridge=bridge)
    assert exploration.roots[0] == exploration.roots[1]
    selected = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
    )
    assert selected.status == "selected", selected.reason
    assert selected.graph is not None and selected.graph.outputs == (1, 1)


def test_ordered_output_abi_is_solved_and_checked_independently(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"), SemanticNode("y", "copy", ("x",), tensor)),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-fixed-output-1",
    )
    descriptor = _descriptor("copy", "copy", ("external",), "external", "i8", "exact", (2,))
    banks = (StorageBank("external", "dram", 3, "word"),)
    result = select_and_allocate(
        request, (descriptor,), banks, bridge=bridge, fixed_inputs={"x": 0}, fixed_outputs=(2,)
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    output_id = result.graph.outputs[0]
    assert result.allocation.addresses[output_id] == 2
    assert result.rules is not None and result.exploration is not None and result.candidate is not None
    changed_abi = check_selection(
        request,
        (descriptor,),
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        result.allocation,
        banks,
        fixed_inputs={"x": 0},
    )
    assert changed_abi.valid and changed_abi.fingerprint != result.check_fingerprint
    moved = dict(result.allocation.addresses)
    moved[output_id] = 1
    checked, reason = check_assignment(
        result.graph, result.allocation.order, moved, banks, fixed_inputs={"x": 0}, fixed_outputs=(2,)
    )
    assert not checked and "fixed external address" in reason
    assert (
        select_and_allocate(
            request, (descriptor,), banks, bridge=bridge, fixed_inputs={"x": 0}, fixed_outputs=(0,)
        ).status
        == "compile_error"
    )
    assert (
        select_and_allocate(
            request, (descriptor,), banks, bridge=bridge, fixed_inputs={"x": 0}, fixed_outputs=(0, 1)
        ).status
        == "modeling_failure"
    )
    for inputs, outputs in (
        ({"ghost": 0}, (2,)),
        ({"x": True}, (2,)),
        ({"x": 3}, (2,)),
        ({"x": 0}, (-1,)),
        ({"x": 0}, (True,)),
        ({"x": 0}, (3,)),
    ):
        malformed = select_and_allocate(
            request, (descriptor,), banks, bridge=bridge, fixed_inputs=inputs, fixed_outputs=outputs
        )
        assert malformed.status == "modeling_failure", (inputs, outputs, malformed)
        valid, reason = check_assignment(
            result.graph,
            result.allocation.order,
            result.allocation.addresses,
            banks,
            fixed_inputs=inputs,
            fixed_outputs=outputs,
        )
        assert not valid and "ABI" in reason


def test_in_place_reuse_requires_declared_last_use_and_later_write(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "load", ("x",), tensor),
            SemanticNode("b", "transform", ("a",), tensor),
            SemanticNode("y", "store", ("b",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-in-place-1",
    )
    load = _descriptor("load", "load", ("external",), "register", "i8", "exact", (2,))
    transform = _descriptor(
        "transform",
        "transform",
        ("register",),
        "register",
        "i8",
        "exact",
        (2,),
        input_read_offsets=(0,),
        completion_offset=1,
        in_place_inputs=(0,),
    )
    store = _descriptor("store", "store", ("register",), "external", "i8", "exact", (2,))
    assert InstructionDescriptor.from_record(transform.record()) == transform
    banks = (StorageBank("external", "dram", 2, "word"), StorageBank("register", "mrf", 1, "word"))
    result = select_and_allocate(
        request, (load, transform, store), banks, bridge=bridge, fixed_inputs={"x": 0}, fixed_outputs=(1,)
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    produced = {value.source_node: value.id for value in result.graph.values if value.kind == "instruction"}
    assert result.allocation.addresses[produced["a"]] == result.allocation.addresses[produced["b"]] == 0
    retained_source = CandidateGraph(result.graph.values, (*result.graph.outputs, produced["a"]))
    retained = allocate(retained_source, result.allocation.order, banks, fixed_inputs={"x": 0}, fixed_outputs=(1, 0))
    assert retained.status == "infeasible_candidate"
    wide_values = tuple(
        replace(value, extent=2) if value.id in (produced["a"], produced["b"]) else value
        for value in result.graph.values
    )
    wide_graph = CandidateGraph(wide_values, result.graph.outputs)
    partial = dict(result.allocation.addresses)
    partial[produced["b"]] = 1
    wide_banks = (banks[0], StorageBank("register", "mrf", 3, "word"))
    checked, reason = check_assignment(
        wide_graph, result.allocation.order, partial, wide_banks, fixed_inputs={"x": 0}, fixed_outputs=(1,)
    )
    assert not checked and "overlap" in reason
    refused = select_and_allocate(
        request,
        (load, replace(transform, in_place_inputs=()), store),
        banks,
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(1,),
    )
    assert refused.status == "compile_error"
    with pytest.raises(ValueError, match="later completion"):
        replace(transform, completion_offset=0)
    with pytest.raises(ValueError, match="read before"):
        replace(transform, input_read_offsets=(1,))


def test_missing_rule_and_exploration_limit_have_distinct_statuses(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"), SemanticNode("y", "opaque", ("x",), tensor)),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-status-1",
    )
    bank = (StorageBank("external", "dram", 2, "word"),)
    missing = select_and_allocate(request, (), bank, bridge=bridge)
    assert missing.status == "unsupported_semantics"
    bounded = select_and_allocate(request, (), bank, bridge=bridge, limits=SearchLimits(egraph_nodes=1))
    assert bounded.status == "resource_limit"


def test_search_limits_round_trip_and_exhausted_candidate_budget(bridge: Path) -> None:
    limits = SearchLimits(candidate_nodes=1)
    assert SearchLimits.from_record(limits.record()) == limits
    for invalid in (
        {**limits.record(), "candidate_nodes": True},
        {**limits.record(), "candidate_nodes": 0},
        {**limits.record(), "schema": "merlin.native_search_limits.v0"},
        {key: value for key, value in limits.record().items() if key != "candidates"},
    ):
        with pytest.raises(ValueError, match="search limits"):
            SearchLimits.from_record(invalid)

    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "add", ("x", "x"), tensor),
            SemanticNode("y", "add", ("a", "x"), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-budget-1",
    )
    add = _descriptor("add", "add", ("external", "external"), "external", "i8", "exact", (2,))
    banks = (StorageBank("external", "dram", 3, "word"),)
    bounded = select_and_allocate(
        request, (add,), banks, bridge=bridge, fixed_inputs={"x": 0}, fixed_outputs=(2,), limits=limits
    )
    assert bounded.status == "resource_limit" and bounded.candidate_attempts == 0
    selected = select_and_allocate(
        request,
        (add,),
        banks,
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(2,),
        limits=SearchLimits(candidate_nodes=2),
    )
    assert selected.status == "selected" and selected.candidate is not None
    assert selected.candidate.instruction_count() == 2


def test_native_search_alternates_bounded_exploration_and_extraction(
    bridge: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "add", ("x", "x"), tensor),
            SemanticNode("y", "add", ("a", "x"), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-staged-search-1",
    )
    add = _descriptor("add", "add", ("external", "external"), "external", "i8", "exact", (2,))
    calls: list[tuple[int, int]] = []

    def recorded_explore(program, **kwargs):
        calls.append((kwargs["iterations"], kwargs["node_limit"]))
        result = explore(program, **kwargs)
        # Keep the first graph valid but mark it unfinished, so the controller
        # must retry exploration after the one-instruction extraction budget.
        return replace(result, stop_reason="iteration_limit") if len(calls) == 1 else result

    monkeypatch.setattr(native_search, "explore", recorded_explore)
    selected = select_and_allocate(
        request,
        (add,),
        (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
        fixed_outputs=(2,),
        limits=SearchLimits(iterations=2, egraph_nodes=128, candidate_nodes=2),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 2
    assert calls == [(1, 64), (2, 128)]
    assert selected.exploration_attempts == len(calls)


def test_native_selection_runs_when_act_imports_and_executables_are_blocked(
    bridge: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    original_run = subprocess.run

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"act", "act_backend", "taidl", "taidl_to"}:
            raise AssertionError("native selection attempted an ACT import")
        return original_import(name, *args, **kwargs)

    def guarded_run(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if not isinstance(command, (list, tuple)) or Path(str(command[0])).name != "merlin-egg-bridge":
            raise AssertionError("native selection attempted an undeclared subprocess")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", guarded_run)
    result = select_and_allocate(
        _request(),
        _descriptors(),
        (StorageBank("external", "dram", 7, "word"), *_banks()[1:]),
        bridge=bridge,
        fixed_inputs={"x": 5},
    )
    assert result.status == "selected", result.reason


def test_native_target_snapshots_rebuild_offline_and_bind_target_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    original_run = subprocess.run
    cargo_calls = 0

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"act", "act_backend", "taidl", "taidl_to"}:
            raise AssertionError("native snapshot generation attempted an ACT import")
        return original_import(name, *args, **kwargs)

    def guarded_run(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal cargo_calls
        if not isinstance(command, (list, tuple)) or Path(str(command[0])).name not in {
            "cargo",
            "merlin-egg-bridge",
        }:
            raise AssertionError("native snapshot generation attempted an undeclared subprocess")
        if isinstance(command, (list, tuple)) and command[:2] == ["cargo", "build"]:
            assert "--offline" in command and "--locked" in command
            cargo_calls += 1
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", guarded_run)
    crate = repo_root() / "src/merlin/semantic_compiler/egg_bridge"
    profile = NativeTargetProfile(_request().target_identity, _descriptors(), _banks())
    assert profile.record()["schema"] == "merlin.native_target_profile.v6"
    old_profile = dict(profile.record(), schema="merlin.native_target_profile.v5")
    with pytest.raises(ValueError, match="schema"):
        NativeTargetProfile.from_record(old_profile)
    original = build_native_snapshot(
        profile,
        destination=tmp_path / "target-a",
        crate=crate,
        cargo_target_dir=tmp_path / "cargo",
        source_revision="public-test-revision",
    )
    assert original.select(_request(), fixed_inputs={"x": 0}).status == "selected"
    assert original.manifest["schema"] == "merlin.native_target_snapshot.v6"
    expected_sources = {
        "search.py",
        "rules.py",
        "allocate.py",
        "ordered_mn_tiling.py",
        "ordered_mnk_tiling.py",
        "precision_transform.py",
        "precision_constants.py",
        "precision_values.py",
        "frontends/compile_inputs.py",
        "runtime/fp8_formats.py",
    }
    assert expected_sources <= set(original.manifest["compiler_sources"])
    assert original.manifest["runtime_dependencies"]["z3-library"]
    manifest_path = original.root / "manifest.json"
    saved_manifest = manifest_path.read_bytes()
    altered_manifest = dict(original.manifest)
    altered_manifest["compiler_sources"] = {**altered_manifest["compiler_sources"], "search.py": "0" * 64}
    altered_sources = altered_manifest["compiler_sources"]
    manifest_path.write_text(json.dumps(altered_manifest))
    with pytest.raises(ValueError, match="Python sources differ"):
        open_native_snapshot(original.root)
    manifest_path.write_bytes(saved_manifest)
    altered_manifest = dict(original.manifest)
    altered_manifest["compiler_sources"] = {
        **altered_manifest["compiler_sources"],
        "ordered_mnk_tiling.py": "0" * 64,
    }
    manifest_path.write_text(json.dumps(altered_manifest))
    with pytest.raises(ValueError, match="Python sources differ"):
        open_native_snapshot(original.root)
    manifest_path.write_bytes(saved_manifest)
    altered_manifest = dict(original.manifest)
    altered_manifest["runtime_dependencies"] = {**altered_manifest["runtime_dependencies"], "z3-library": "0.0.0"}
    manifest_path.write_text(json.dumps(altered_manifest))
    with pytest.raises(ValueError, match="runtime dependencies differ"):
        open_native_snapshot(original.root)
    manifest_path.write_bytes(saved_manifest)
    manifest_path.write_text(json.dumps(dict(original.manifest, schema="merlin.native_target_snapshot.v5")))
    with pytest.raises(ValueError, match="invalid manifest"):
        open_native_snapshot(original.root)
    manifest_path.write_bytes(saved_manifest)
    saved_sources = original.manifest["compiler_sources"]
    original.manifest["compiler_sources"] = altered_sources
    with pytest.raises(ValueError, match="Python sources differ"):
        original.select(_request(), fixed_inputs={"x": 0})
    original.manifest["compiler_sources"] = saved_sources
    saved_dependencies = original.manifest["runtime_dependencies"]
    original.manifest["runtime_dependencies"] = altered_manifest["runtime_dependencies"]
    with pytest.raises(ValueError, match="runtime dependencies differ"):
        original.select(_request(), fixed_inputs={"x": 0})
    original.manifest["runtime_dependencies"] = saved_dependencies
    reserved_result = original.select(_request(), fixed_inputs={"x": 0}, reservations=(Reservation("a", 0, 1),))
    assert reserved_result.status == "selected"
    assert reserved_result.graph is not None and reserved_result.allocation is not None
    assert all(
        reserved_result.allocation.addresses[value.id] != 0
        for value in reserved_result.graph.values
        if value.storage == "a"
    )
    smaller = NativeTargetProfile(
        "synthetic-revision-2",
        _descriptors(),
        (StorageBank("external", "dram", 1, "word"), *_banks()[1:]),
    )
    second = build_native_snapshot(
        smaller,
        destination=tmp_path / "target-b",
        crate=crate,
        cargo_target_dir=tmp_path / "cargo",
        source_revision="public-test-revision",
    )
    assert second.profile.digest() != original.profile.digest()
    assert cargo_calls == 2
    with pytest.raises(ValueError, match="target identity"):
        second.select(_request())
    revised_request = replace(_request(), target_identity="synthetic-revision-2")
    assert second.select(revised_request, fixed_inputs={"x": 0}).status == "compile_error"
    profile_path = second.root / "profile.json"
    profile_path.write_text(profile_path.read_text().replace("synthetic-revision-2", "tampered"))
    with pytest.raises(ValueError, match="differs from manifest"):
        open_native_snapshot(second.root)
    with pytest.raises(ValueError, match="changed after"):
        second.select(revised_request)


def test_native_snapshot_reports_missing_solver_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from merlin.semantic_compiler import snapshot as snapshot_module

    metadata = snapshot_module.importlib.metadata
    installed_version = metadata.version

    def missing_solver(name: str) -> str:
        if name == "z3-solver":
            raise metadata.PackageNotFoundError(name)
        return installed_version(name)

    monkeypatch.setattr(metadata, "version", missing_solver)
    with pytest.raises(ValueError, match="runtime dependency is unavailable"):
        snapshot_module._runtime_dependencies()


def test_native_target_snapshots_retarget_storage_geometry(tmp_path: Path) -> None:
    descriptors = tuple(
        _descriptor(f"emit_{name}", f"use_{name}", ("external",), "external", "i8", "exact", (2,))
        for name in ("a", "b")
    )
    tensor = _type()
    snapshots = {}
    requests = {}
    for capacity in (2, 3):
        identity = f"synthetic-capacity-{capacity}"
        profile = NativeTargetProfile(
            identity,
            descriptors,
            (StorageBank("external", "dram", capacity, "tile"),),
        )
        snapshots[capacity] = build_native_snapshot(
            profile,
            destination=tmp_path / f"target-{capacity}",
            crate=repo_root() / "src/merlin/semantic_compiler/egg_bridge",
            cargo_target_dir=tmp_path / "cargo",
            source_revision="public-retarget-test",
        )
        requests[capacity] = KernelRequest(
            (
                SemanticNode("x", "input", (), tensor, effect="input"),
                SemanticNode("a", "use_a", ("x",), tensor),
                SemanticNode("b", "use_b", ("x",), tensor),
            ),
            ("a", "b"),
            ("external", "external"),
            (("x", "external"),),
            identity,
        )

    assert snapshots[2].profile.digest() != snapshots[3].profile.digest()
    assert snapshots[2].manifest["bridge_sha256"] == snapshots[3].manifest["bridge_sha256"]
    assert snapshots[2].manifest["compiler_sources"] == snapshots[3].manifest["compiler_sources"]
    limits = SearchLimits(candidate_nodes=3, candidates=12, orders_per_candidate=6)
    small = snapshots[2].select(requests[2], fixed_inputs={"x": 0}, limits=limits)
    assert small.status == "compile_error" and small.rejected_allocation > 0
    large = snapshots[3].select(requests[3], fixed_inputs={"x": 0}, limits=limits)
    assert large.status == "selected", large.reason
    assert large.graph is not None and large.allocation is not None
    assert tuple(large.allocation.addresses[value] for value in large.graph.outputs) == (1, 2)


def test_alias_overlap_is_rejected_independently(bridge: Path) -> None:
    request = _request()
    rules = generate_rules(request, _descriptors())
    graph = explore(rules, bridge=bridge)
    candidate = next(enumerate_candidates(graph, request, rules, node_budget=4, max_candidates=1))
    lowered = lower_candidate(candidate, rules)
    from merlin.semantic_compiler.allocate import topological_orders

    order = next(topological_orders(lowered, limit=1))
    # Both typed register views alias the same physical slot.
    aliased = (
        StorageBank("external", "dram", 4, "word"),
        StorageBank("a", "registers", 2, "word"),
        StorageBank("b", "registers", 2, "word"),
    )
    addresses = {value.id: 0 for value in lowered.values}
    valid, reason = check_assignment(lowered, order, addresses, aliased, fixed_inputs={"x": 0})
    assert not valid and "overlap" in reason


def test_pair_registers_and_typed_aliases_share_backing_slots() -> None:
    graph = CandidateGraph(
        (
            Value(0, "fp8_input", "fp8_view", 1, (), "x", "input"),
            Value(
                1,
                "bf16_pair",
                "bf16_view",
                2,
                (0,),
                None,
                "instruction",
                (AddressConstraint("aligned", "out", value=2),),
            ),
        ),
        (1,),
    )
    banks = (
        StorageBank("fp8_view", "tensor_registers", 4, "slot"),
        StorageBank("bf16_view", "tensor_registers", 4, "slot"),
    )
    solved = allocate(graph, (1,), banks, fixed_inputs={"x": 3})
    assert solved.status == "feasible" and solved.addresses[1] == 0
    valid, reason = check_assignment(graph, (1,), {0: 3, 1: 3}, banks, fixed_inputs={"x": 3})
    assert not valid and "range" in reason
    valid, reason = check_assignment(graph, (1,), {0: 3, 1: 2}, banks, fixed_inputs={"x": 3})
    assert not valid and "overlap" in reason


def test_unsat_pruning_direction_and_base_identity() -> None:
    small = frozenset({(1, 2)})
    large = frozenset({(1, 2), (2, 3)})
    assert may_prune_interference("same", "same", small, large, failed_status="infeasible_candidate")
    assert not may_prune_interference("same", "same", large, small, failed_status="infeasible_candidate")
    assert not may_prune_interference("old", "new", small, large, failed_status="infeasible_candidate")
    assert not may_prune_interference("same", "same", small, large, failed_status="search_timeout")


def test_native_controller_prunes_only_later_same_base_interference_supersets(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "branch_a", ("x",), tensor),
            SemanticNode("b", "branch_b", ("x",), tensor),
        ),
        outputs=("a", "b"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"),),
        target_identity="synthetic-prune-1",
    )
    descriptors = (
        _descriptor("a", "branch_a", ("external",), "external", "i8", "exact", (2,)),
        _descriptor("b", "branch_b", ("external",), "external", "i8", "exact", (2,)),
    )
    result = select_and_allocate(
        request,
        descriptors,
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
    )
    assert result.status == "compile_error"
    assert result.rejected_allocation == 1 and result.pruned_orders == 1


def test_order_fallback_recovers_a_feasible_schedule() -> None:
    """First ready-order keeps two register values live; second serializes them."""
    graph = CandidateGraph(
        (
            Value(0, "x", "external", 1, (), "x", "input"),
            Value(1, "y", "external", 1, (), "y", "input"),
            Value(2, "produce_x", "register", 1, (0,), None, "instruction"),
            Value(3, "produce_y", "register", 1, (1,), None, "instruction"),
            Value(4, "consume_x", "external", 1, (2,), None, "instruction"),
            Value(5, "consume_y", "external", 1, (3,), None, "instruction"),
        ),
        (4, 5),
    )
    banks = (StorageBank("external", "dram", 4, "word"), StorageBank("register", "regs", 1, "word"))
    orders = list(topological_orders(graph, limit=8))
    results = [allocate(graph, order, banks, fixed_inputs={"x": 0, "y": 1}) for order in orders]
    assert results[0].status == "infeasible_candidate"
    assert any(result.status == "feasible" for result in results[1:])


def test_assignment_rejects_duplicate_and_reversed_instruction_orders() -> None:
    graph = CandidateGraph(
        (
            Value(0, "input", "external", 1, (), "x", "input"),
            Value(1, "first", "register", 1, (0,), None, "instruction"),
            Value(2, "second", "external", 1, (1,), None, "instruction"),
        ),
        (2,),
    )
    banks = (StorageBank("external", "dram", 2, "word"), StorageBank("register", "regs", 1, "word"))
    addresses = {0: 0, 1: 0, 2: 1}
    valid, reason = check_assignment(graph, (1, 1, 2), addresses, banks)
    assert not valid and "duplicates" in reason
    valid, reason = check_assignment(graph, (2, 1), addresses, banks)
    assert not valid and "before its instruction producer" in reason


def test_non_emitting_dependency_preserves_order_and_producer_lifetime() -> None:
    graph = CandidateGraph(
        (
            Value(0, "input", "external", 1, (), "x", "input"),
            Value(1, "producer", "register", 1, (0,), None, "instruction"),
            Value(
                2,
                "view",
                "register",
                1,
                (1,),
                None,
                "view",
                validity=(AddressConstraint("eq_offset", "out", "in0", 0),),
            ),
            Value(3, "consumer", "external", 1, (2,), None, "instruction"),
        ),
        (3,),
    )
    assert list(topological_orders(graph, limit=4)) == [(1, 3)]
    with pytest.raises(ValueError, match="before its instruction producer"):
        instruction_schedule(graph, (3, 1))
    timed, reason = check_timing(graph, (3, 1), ((3, 1), (1, 2)))
    assert not timed and reason == "input read precedes producer completion"
    assert check_timing(graph, (1, 3), ((1, 1), (3, 2))) == (True, "")
    assert live_ranges(graph, (1, 3))[1] == (1, 2)
    assert live_ranges(graph, (1, 3))[2] == (1, 2)
    banks = (StorageBank("external", "dram", 2, "word"), StorageBank("register", "regs", 2, "word"))
    placed = allocate(graph, (1, 3), banks, fixed_inputs={"x": 0}, fixed_outputs=(1,))
    assert placed.status == "feasible", placed.reason
    assert placed.addresses[1] == placed.addresses[2]
    moved = dict(placed.addresses)
    moved[2] = 1 - moved[1]
    assert check_assignment(graph, (1, 3), moved, banks)[1] == "identity view moved from its source address"
    invalid_view = replace(graph.values[2], validity=())
    invalid_graph = replace(graph, values=(*graph.values[:2], invalid_view, graph.values[3]))
    assert allocate(invalid_graph, (1, 3), banks).status == "modeling_failure"
    chained = CandidateGraph(
        (
            *graph.values[:3],
            replace(graph.values[2], id=3, children=(2,)),
            replace(graph.values[3], id=4, children=(3,)),
        ),
        (4,),
    )
    chained_placement = allocate(
        chained,
        (1, 4),
        (banks[0], StorageBank("register", "regs", 1, "word")),
        fixed_inputs={"x": 0},
        fixed_outputs=(1,),
    )
    assert chained_placement.status == "feasible", chained_placement.reason
    assert {chained_placement.addresses[index] for index in (1, 2, 3)} == {0}
    assert check_timing(chained, (4, 1), ((4, 1), (1, 2)))[1] == ("input read precedes producer completion")
    assert check_timing(chained, (1, 4), ((1, 1), (4, 2))) == (True, "")
    assert check_timing(chained, (1,), ((1, 1),))[1] == ("physical schedule omits or duplicates an instruction")
    invalid_dependency = replace(chained.values[3], children=(4,))
    invalid_graph = replace(chained, values=(*chained.values[:3], invalid_dependency, chained.values[4]))
    assert check_timing(invalid_graph, (1, 4), ((1, 1), (4, 2)))[1] == (
        "physical schedule has invalid value dependencies"
    )
    output_view = replace(graph, outputs=(2,))
    assert live_ranges(output_view, (1, 3))[1] == (1, 3)
    with pytest.raises(ValueError, match="topological value order"):
        list(
            topological_orders(
                replace(graph, values=(*graph.values[:2], replace(graph.values[2], children=(3,)), graph.values[3])),
                limit=4,
            )
        )

    clobber_graph = CandidateGraph(
        (
            *graph.values[:3],
            Value(3, "clobber", "register", 1, (), None, "instruction"),
            replace(graph.values[3], id=4),
        ),
        (4,),
    )
    one_register = (banks[0], StorageBank("register", "regs", 1, "word"))
    assert (
        allocate(clobber_graph, (1, 3, 4), one_register, fixed_inputs={"x": 0}, fixed_outputs=(1,)).status
        == "infeasible_candidate"
    )
    assert (
        allocate(clobber_graph, (1, 4, 3), one_register, fixed_inputs={"x": 0}, fixed_outputs=(1,)).status == "feasible"
    )


def test_concat_view_keeps_both_producers_live_and_adjacent() -> None:
    graph = CandidateGraph(
        (
            Value(0, "a", "external", 1, (), "a", "input"),
            Value(1, "b", "external", 1, (), "b", "input"),
            Value(2, "produce_a", "register", 1, (0,), None, "instruction", completion_offset=2),
            Value(3, "produce_b", "register", 1, (1,), None, "instruction", completion_offset=1),
            Value(
                4,
                "concat",
                "register",
                2,
                (2, 3),
                None,
                "view",
                validity=(
                    AddressConstraint("eq_offset", "out", "in0", 0),
                    AddressConstraint("eq_offset", "in1", "out", 1),
                ),
            ),
            Value(5, "consume", "external", 1, (4,), None, "instruction"),
        ),
        (5,),
    )
    banks = (StorageBank("external", "dram", 3, "word"), StorageBank("register", "regs", 2, "word"))
    order = (2, 3, 5)
    allocation = allocate(graph, order, banks, fixed_inputs={"a": 0, "b": 1}, fixed_outputs=(2,))
    assert allocation.status == "feasible", allocation.reason
    assert allocation.addresses[2] == 0
    assert allocation.addresses[3] == 1
    assert allocation.addresses[4] == 0
    assert live_ranges(graph, order)[2][1] >= allocation.issue_times[-1][1]
    assert live_ranges(graph, order)[3][1] >= allocation.issue_times[-1][1]
    assert check_timing(graph, order, allocation.issue_times) == (True, "")
    assert check_timing(graph, (2, 5, 3), ((2, 1), (5, 4), (3, 5)))[1] == ("input read precedes producer completion")
    bad = dict(allocation.addresses)
    bad[3] = bad[2]
    assert check_assignment(graph, order, bad, banks)[1] == "instruction address map or validity failed"


def test_generated_producers_need_layouts_for_a_no_copy_concat(bridge: Path, tmp_path: Path) -> None:
    part = TensorType((2,), "i8", "exact")
    joined = TensorType((4,), "i8", "exact")
    request = KernelRequest(
        nodes=(
            SemanticNode("a", "input", (), part, effect="input"),
            SemanticNode("b", "input", (), part, effect="input"),
            SemanticNode("a_copy", "identity", ("a",), part),
            SemanticNode("b_copy", "identity", ("b",), part),
            SemanticNode("joined", "concat", ("a_copy", "b_copy"), joined, attrs=(("axis", 0),)),
        ),
        outputs=("joined",),
        output_storages=("register",),
        input_storages=(("a", "external"), ("b", "external")),
        target_identity="synthetic-produced-concat-layout",
        physical_extents=(("a", 2), ("b", 2)),
        boundary_layouts=(("a", "contiguous_element_order"), ("b", "contiguous_element_order")),
        output_layouts=("contiguous_element_order",),
    )
    producer = _descriptor(
        "copy",
        "identity",
        ("external",),
        "register",
        "i8",
        "exact",
        (1,),
        extent=2,
        input_layouts=("contiguous_element_order",),
        output_layout="contiguous_element_order",
    )
    assert InstructionDescriptor.from_record(producer.record()) == producer
    with pytest.raises(ValueError, match="typed fields"):
        InstructionDescriptor.from_record({**producer.record(), "input_layouts": "contiguous_element_order"})
    view = ConcatViewDescriptor("join", "register", (part, part), joined, (2, 2), 4)
    banks = (StorageBank("external", "dram", 4, "element"), StorageBank("register", "regs", 4, "element"))
    profile = NativeTargetProfile(request.target_identity, (producer,), banks, (view,))
    snapshot = build_native_snapshot(
        profile,
        destination=tmp_path / "snapshot",
        crate=repo_root() / "src/merlin/semantic_compiler/egg_bridge",
        cargo_target_dir=tmp_path / "cargo",
        source_revision="public-layout-qualified-view-test",
    )
    assert (
        open_native_snapshot(snapshot.root)
        .select(
            request,
            fixed_inputs={"a": 0, "b": 2},
            fixed_outputs=(0,),
        )
        .status
        == "selected"
    )
    selected = select_and_allocate(
        request,
        (producer,),
        banks,
        bridge=bridge,
        views=(view,),
        fixed_inputs={"a": 0, "b": 2},
        fixed_outputs=(0,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 2
    assert selected.allocation is not None and selected.graph is not None
    assert sorted(value.extent for value in selected.graph.values if value.kind == "instruction") == [2, 2]
    assert selected.allocation.addresses[selected.graph.outputs[0]] == 0
    changed = replace(producer, output_layout="interleaved")
    assert (
        select_and_allocate(
            request,
            (changed,),
            banks,
            bridge=bridge,
            views=(view,),
            fixed_inputs={"a": 0, "b": 2},
            fixed_outputs=(0,),
        ).status
        != "selected"
    )


def _row_slice_request(outputs: tuple[str, ...], *, declared: bool) -> KernelRequest:
    source, rows = TensorType((4, 2), "i8", "exact"), TensorType((2, 2), "i8", "exact")
    identity = ((1, 0), (0, 1))
    layouts = (
        {
            "boundary_layouts": (("x", "contiguous_element_order"),),
            "output_layouts": ("contiguous_element_order",) * len(outputs),
        }
        if declared
        else {}
    )
    return KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), source, effect="input"),
            SemanticNode("p", "prep", ("x",), source),
            SemanticNode(
                "s",
                "gather",
                ("p",),
                rows,
                index_maps=(IndexMap(2, identity, (2, 0)), IndexMap(2, identity, (0, 0))),
            ),
        ),
        outputs=outputs,
        output_storages=("rf",) * len(outputs),
        input_storages=(("x", "external"),),
        target_identity="synthetic-row-slice",
        physical_extents=(("x", 8),),
        **layouts,
    )


def _row_slice_view() -> ViewDescriptor:
    return ViewDescriptor(
        "rows",
        "rf",
        TensorType((4, 2), "i8", "exact"),
        TensorType((2, 2), "i8", "exact"),
        4,
        kind="slice",
        source_extent=8,
        physical_offset=4,
    )


def test_view_verification_rechecks_realized_source_extent(bridge: Path) -> None:
    # The slice offset holds only for the 8-unit source the target qualified;
    # a padded 16-unit realisation places row 2 elsewhere.
    request = _row_slice_request(("s",), declared=True)
    layout = "contiguous_element_order"
    descriptors = tuple(
        _descriptor(
            name,
            "prep",
            ("external",),
            "rf",
            "i8",
            "exact",
            (2,),
            extent=extent,
            input_layouts=(layout,),
            output_layout=layout,
        )
        for name, extent in (("prep_dense", 8), ("prep_padded", 16))
    )
    views = (_row_slice_view(),)
    banks = (StorageBank("external", "dram", 64, "word"), StorageBank("rf", "rf", 64, "slot"))
    result = select_and_allocate(
        request,
        descriptors,
        banks,
        bridge=bridge,
        views=views,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert result.status == "selected", result.reason
    assert result.candidate is not None and result.rules is not None and result.exploration is not None
    assert result.allocation is not None
    padded = next(
        symbol
        for symbol, metadata in result.rules.symbols.items()
        if metadata.get("descriptor", {}).get("name") == "prep_padded"
    )
    root = result.candidate.outputs[0]
    forged = native_extract.Candidate((replace(root, children=(replace(root.children[0], symbol=padded),)),))
    graph = lower_candidate(forged, result.rules)
    allocation = allocate(graph, result.allocation.order, banks, fixed_inputs={"x": 0})
    assert allocation.status == "feasible"
    check = check_selection(
        request,
        descriptors,
        result.rules,
        result.exploration,
        forged,
        graph,
        allocation,
        banks,
        views=views,
        fixed_inputs={"x": 0},
    )
    assert not check.valid
    assert check.reason == "selected view source extent differs from the target descriptor"


@pytest.mark.parametrize("outputs", [("p", "s"), ("s", "p")])
def test_view_layout_obligation_is_checked_in_either_output_order(
    bridge: Path,
    outputs: tuple[str, str],
) -> None:
    # The slice requires its source layout, which the producer in turn requires
    # of the legacy input that declares none. Reaching the producer first as a
    # plain output must not skip that obligation.
    request = _row_slice_request(outputs, declared=False)
    layout = "contiguous_element_order"
    descriptors = (
        _descriptor(
            "prep",
            "prep",
            ("external",),
            "rf",
            "i8",
            "exact",
            (2,),
            extent=8,
            input_layouts=(layout,),
            output_layout=layout,
        ),
    )
    views = (_row_slice_view(),)
    banks = (StorageBank("external", "dram", 64, "word"), StorageBank("rf", "rf", 64, "slot"))
    result = select_and_allocate(
        request,
        descriptors,
        banks,
        bridge=bridge,
        views=views,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert result.status != "selected"
    assert result.rules is not None and result.exploration is not None
    classes = {name: result.exploration.class_by_node[index] for index, name in enumerate(("x", "p", "s"))}
    symbols = {
        metadata.get("source_node"): symbol
        for symbol, metadata in result.rules.symbols.items()
        if metadata["kind"] != "semantic"
    }
    x = native_extract.Choice(classes["x"], symbols["x"], "external")
    p = native_extract.Choice(classes["p"], symbols["p"], "rf", (x,))
    s = native_extract.Choice(classes["s"], symbols["s"], "rf", (p,))
    candidate = native_extract.Candidate(tuple({"p": p, "s": s}[name] for name in outputs))
    graph = lower_candidate(candidate, result.rules)
    allocation = allocate(graph, next(topological_orders(graph, limit=1)), banks, fixed_inputs={"x": 0})
    assert allocation.status == "feasible"
    check = check_selection(
        request,
        descriptors,
        result.rules,
        result.exploration,
        candidate,
        graph,
        allocation,
        banks,
        views=views,
        fixed_inputs={"x": 0},
    )
    assert not check.valid
    assert check.reason == "selected physical layout differs from its required representation"


def test_64_tiny_identity_view_placements_match_independent_enumeration() -> None:
    """Enumerate at most 4^4 placements per case without the native checker."""
    graph = CandidateGraph(
        (
            Value(0, "input", "external", 1, (), "x", "input"),
            Value(1, "producer", "register", 1, (0,), None, "instruction"),
            Value(
                2,
                "view",
                "register",
                1,
                (1,),
                None,
                "view",
                validity=(AddressConstraint("eq_offset", "out", "in0", 0),),
            ),
            Value(3, "clobber", "register", 1, (), None, "instruction"),
            Value(4, "consumer", "external", 1, (2,), None, "instruction"),
        ),
        (4,),
    )
    feasible = infeasible = 0
    for external_capacity, register_capacity, clobber_first, reserve_zero in itertools.product(
        range(1, 5), range(1, 5), (False, True), (False, True)
    ):
        banks = (
            StorageBank("external", "dram", external_capacity, "word"),
            StorageBank("register", "regs", register_capacity, "word"),
        )
        order = (1, 3, 4) if clobber_first else (1, 4, 3)
        reservation = (Reservation("register", 0, 1),) if reserve_zero else ()
        pinned_output = 1 if external_capacity > 1 else 0
        # This fixture has serial, zero-latency instructions. The producer
        # remains live through the consumer read, including its identity view.
        producer_birth = order.index(1) + 1
        consumer_read = order.index(4) + 1
        clobber_time = order.index(3) + 1
        expected = any(
            output_address == pinned_output
            and output_address != 0  # Retained external input is fixed at zero.
            and view_address == producer_address
            and (not reserve_zero or producer_address != 0)
            and (not reserve_zero or clobber_address != 0)
            and not (producer_birth <= clobber_time <= consumer_read and producer_address == clobber_address)
            for output_address, producer_address, view_address, clobber_address in itertools.product(
                range(external_capacity), range(register_capacity), range(register_capacity), range(register_capacity)
            )
        )
        result = allocate(
            graph, order, banks, fixed_inputs={"x": 0}, fixed_outputs=(pinned_output,), reservations=reservation
        )
        assert (result.status == "feasible") == expected, (
            external_capacity,
            register_capacity,
            clobber_first,
            reserve_zero,
            result.status,
        )
        feasible += expected
        infeasible += not expected
    assert feasible > 0 and infeasible > 0 and feasible + infeasible == 64


def test_162_tiny_boundary_extent_placements_match_independent_enumeration() -> None:
    """Check 162 finite instances with at most six constant addresses each."""
    checked = 0
    feasible = infeasible = 0
    for input_extent, constant_extent in itertools.product(range(1, 4), repeat=2):
        graph = CandidateGraph(
            (
                Value(0, "x", "external", input_extent, (), "x", "input"),
                Value(1, "c", "external", constant_extent, (), "c", "constant"),
                Value(2, "add", "external", 1, (0, 1), "y", "instruction"),
            ),
            (2,),
        )
        for capacity in range(3, 7):
            banks = (StorageBank("external", "memory", capacity, "word"),)
            for output_address in range(capacity):
                # All three values are live at the sole instruction's issue.
                # Enumerate intervals directly, without the native formula or checker.
                expected = any(
                    all(
                        left + left_size <= right or right + right_size <= left
                        for (left, left_size), (right, right_size) in itertools.combinations(
                            ((0, input_extent), (constant_address, constant_extent), (output_address, 1)), 2
                        )
                    )
                    for constant_address in range(capacity - constant_extent + 1)
                )
                result = allocate(graph, (2,), banks, fixed_inputs={"x": 0}, fixed_outputs=(output_address,))
                assert result.status in {"feasible", "infeasible_candidate"}, result.reason
                assert (result.status == "feasible") == expected
                checked += 1
                feasible += expected
                infeasible += not expected
    assert checked == 162 and feasible and infeasible


def test_delayed_operand_read_extends_lifetime_and_forces_serial_issue() -> None:
    graph = CandidateGraph(
        (
            Value(0, "pointer", "scalar", 1, (), "pointer", "input", preserve_input=False),
            Value(
                1, "delayed_read", "command", 1, (0,), None, "instruction", input_read_offsets=(3,), completion_offset=4
            ),
            Value(2, "reuse_scalar", "scalar", 1, (), None, "instruction"),
        ),
        (2,),
    )
    banks = (StorageBank("scalar", "registers", 1, "word"), StorageBank("command", "cmd", 1, "word"))
    order = (1, 2)
    assert instruction_schedule(graph, order) == {1: (1, 5), 2: (6, 6)}
    assert live_ranges(graph, order)[0] == (0, 4)
    result = allocate(graph, order, banks, fixed_inputs={"pointer": 0})
    assert result.status == "feasible" and result.issue_times == ((1, 1), (2, 6))
    assert check_timing(graph, order, result.issue_times) == (True, "")
    timed, reason = check_timing(graph, order, ((1, 1), (2, 2)))
    assert not timed and "prior completion" in reason
    with pytest.raises(ValueError, match="precede instruction completion"):
        _descriptor(
            "bad_timing",
            "move",
            ("scalar",),
            "command",
            "i8",
            "exact",
            (2,),
            input_read_offsets=(5,),
            completion_offset=4,
        )


def test_generated_timing_reaches_native_selection_and_checked_allocation(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "move", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-delayed-read-1",
    )
    descriptor = _descriptor(
        "delayed_move",
        "move",
        ("external",),
        "external",
        "i8",
        "exact",
        (2,),
        input_read_offsets=(3,),
        completion_offset=4,
    )
    result = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
    )
    assert result.status == "selected" and result.graph is not None and result.allocation is not None
    assert result.allocation.issue_times == ((result.graph.outputs[0], 1),)
    assert live_ranges(result.graph, result.allocation.order)[0] == (0, 4)
    assert result.allocation.addresses[result.graph.outputs[0]] == 1


def test_alternate_instruction_candidate_after_allocation_failure(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "mix", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-variant-1",
    )
    good = _descriptor("good", "mix", ("external",), "external", "i8", "exact", (2,))
    good_program = generate_rules(request, (good,))
    good_symbol = next(symbol for symbol, info in good_program.symbols.items() if info["kind"] == "instruction")
    bad = None
    for number in range(100):
        trial = _descriptor(f"oversized_{number}", "mix", ("external",), "external", "i8", "exact", (2,), extent=3)
        trial_program = generate_rules(request, (trial,))
        trial_symbol = next(symbol for symbol, info in trial_program.symbols.items() if info["kind"] == "instruction")
        if trial_symbol < good_symbol:
            bad = trial
            break
    assert bad is not None
    result = select_and_allocate(
        request,
        (bad, good),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=1, candidates=4),
    )
    assert result.status == "selected", result.reason
    assert result.candidate_attempts == 2
    assert result.rejected_allocation == 1
    assert not result.diagnostic_only
    for policy in (
        SearchAblations(one_shot_extraction=True),
        SearchAblations(disable_candidate_fallback=True),
    ):
        ablated = select_and_allocate(
            request,
            (bad, good),
            (StorageBank("external", "dram", 2, "word"),),
            bridge=bridge,
            fixed_inputs={"x": 0},
            limits=SearchLimits(candidate_nodes=1, candidates=4),
            ablations=policy,
        )
        assert ablated.status == "resource_limit"
        assert ablated.candidate_attempts == 1 and ablated.rejected_allocation == 1
        assert ablated.diagnostic_ablations == policy.active() and ablated.diagnostic_only


def test_target_materialization_rejection_resumes_checked_search(bridge: Path) -> None:
    seen: list[tuple[str, tuple[int, ...]]] = []

    def materialize(selected: native_search.SearchResult) -> tuple[bool, str]:
        assert selected.status == "selected" and selected.check_fingerprint
        assert selected.candidate is not None and selected.allocation is not None
        seen.append((selected.candidate.digest(), selected.allocation.order))
        return (len(seen) > 1, "first physical lowering cannot encode this order")

    result = select_and_allocate(
        _request(),
        _descriptors(),
        _banks(),
        bridge=bridge,
        fixed_inputs={"x": 0},
        accept_materialization=materialize,
    )
    assert result.status == "selected", result.reason
    assert len(seen) == 2 and seen[0] != seen[1]
    assert result.rejected_materialization == 1
    assert result.rejected_allocation == 0
    assert result.ordering_attempts >= 2

    bounded = select_and_allocate(
        _request(),
        _descriptors(),
        _banks(),
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(orders_per_candidate=1),
        accept_materialization=lambda selected: (False, "first order rejected"),
    )
    assert bounded.status == "resource_limit"
    assert bounded.rejected_materialization > 0


def test_target_materialization_can_report_a_checked_physical_reallocation(bridge: Path) -> None:
    observed: list[tuple[dict[int, int], str]] = []

    def materialize(selected: native_search.SearchResult) -> native_search.MaterializationDecision:
        assert selected.graph is not None and selected.allocation is not None
        realized = dict(selected.allocation.addresses)
        moved = [value.id for value in selected.graph.values if value.storage == "a"]
        assert moved
        for value_id in moved:
            realized[value_id] = 1 - realized[value_id]
        valid, reason = check_assignment(
            selected.graph, selected.allocation.order, realized, _banks(), fixed_inputs={"x": 0}
        )
        assert valid, reason
        observed.append((realized, selected.check_fingerprint))
        return native_search.MaterializationDecision(
            True, realized_allocation=replace(selected.allocation, addresses=realized)
        )

    result = select_and_allocate(
        _request(),
        _descriptors(),
        _banks(),
        bridge=bridge,
        fixed_inputs={"x": 0},
        accept_materialization=materialize,
    )
    assert result.status == "selected" and result.allocation is not None
    assert result.allocation.addresses == observed[0][0]
    assert result.check_fingerprint != observed[0][1]

    def invalid_materialization(selected: native_search.SearchResult) -> native_search.MaterializationDecision:
        assert selected.graph is not None and selected.allocation is not None
        invalid = dict(selected.allocation.addresses)
        internal = next(value.id for value in selected.graph.values if value.storage == "a")
        invalid[internal] = 2
        return native_search.MaterializationDecision(
            True, realized_allocation=replace(selected.allocation, addresses=invalid)
        )

    invalid = select_and_allocate(
        _request(),
        _descriptors(),
        _banks(),
        bridge=bridge,
        fixed_inputs={"x": 0},
        accept_materialization=invalid_materialization,
    )
    assert invalid.status == "modeling_failure"
    assert "invalid physical assignment" in invalid.reason


def test_target_materialization_rejection_is_not_unsat_or_success(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "mix", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-materialization-1",
    )
    descriptor = _descriptor("mix", "mix", ("external",), "external", "i8", "exact", (2,))
    result = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
        accept_materialization=lambda selected: (False, "physical encoder lacks this mode"),
    )
    assert result.status == "unqualified_target"
    assert result.rejected_materialization == 1 and result.rejected_allocation == 0
    assert "physical encoder lacks this mode" in result.reason

    with pytest.raises(RuntimeError, match="materializer unavailable"):
        select_and_allocate(
            request,
            (descriptor,),
            (StorageBank("external", "dram", 2, "word"),),
            bridge=bridge,
            fixed_inputs={"x": 0},
            accept_materialization=lambda selected: (_ for _ in ()).throw(RuntimeError("materializer unavailable")),
        )

    def corrupt_assignment(selected: native_search.SearchResult) -> tuple[bool, str]:
        assert selected.allocation is not None
        selected.allocation.addresses.clear()
        return True, ""

    changed = select_and_allocate(
        request,
        (descriptor,),
        (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge,
        fixed_inputs={"x": 0},
        accept_materialization=corrupt_assignment,
    )
    assert changed.status == "modeling_failure"
    assert "changed during target materialization" in changed.reason


def test_search_order_ablation_exposes_required_fallback(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "input", (), tensor, effect="input"),
            SemanticNode("px", "produce_x", ("x",), tensor),
            SemanticNode("py", "produce_y", ("y", "x"), tensor),
            SemanticNode("cx", "consume_x", ("px",), tensor),
            SemanticNode("cy", "consume_y", ("py",), tensor),
        ),
        outputs=("cx", "cy"),
        output_storages=("external", "external"),
        input_storages=(("x", "external"), ("y", "external")),
        target_identity="synthetic-order-ablation-1",
    )
    descriptors = (
        _descriptor("px", "produce_x", ("external",), "register", "i8", "exact", (2,)),
        _descriptor("py", "produce_y", ("external", "external"), "register", "i8", "exact", (2,)),
        _descriptor("cx", "consume_x", ("register",), "external", "i8", "exact", (2,)),
        _descriptor("cy", "consume_y", ("register",), "external", "i8", "exact", (2,)),
    )
    banks = (StorageBank("external", "dram", 4, "word"), StorageBank("register", "regs", 1, "word"))
    full = select_and_allocate(
        request,
        descriptors,
        banks,
        bridge=bridge,
        fixed_inputs={"x": 0, "y": 1},
        limits=SearchLimits(candidate_nodes=4, candidates=8),
    )
    ablated = select_and_allocate(
        request,
        descriptors,
        banks,
        bridge=bridge,
        fixed_inputs={"x": 0, "y": 1},
        limits=SearchLimits(candidate_nodes=4, candidates=8),
        ablations=SearchAblations(disable_alternative_orders=True),
    )
    assert full.status == "selected", full.reason
    assert full.ordering_attempts > 1 and full.rejected_allocation > 0
    assert ablated.status == "resource_limit" and ablated.ordering_attempts == 1
    assert ablated.diagnostic_ablations == ("disable_alternative_orders",)
    assert ablated.diagnostic_only


def test_search_deadlines_preserve_timeout_status(bridge: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    rules = generate_rules(request, _descriptors())
    exploration = explore(rules, bridge=bridge)
    with pytest.raises(ExtractionTimeout):
        list(
            enumerate_candidates(
                exploration,
                request,
                rules,
                node_budget=4,
                max_candidates=4,
                deadline=0,
            )
        )

    clock = [0.0]
    monkeypatch.setattr(native_search, "monotonic", lambda: clock[0])
    monkeypatch.setattr(native_extract, "monotonic", lambda: clock[0])
    monkeypatch.setattr(native_search, "explore", lambda *args, **kwargs: exploration)

    def timed_allocation(graph, order, banks, **kwargs):
        clock[0] = 2.0
        return AllocationResult("infeasible_candidate", order, {})

    monkeypatch.setattr(native_search, "allocate", timed_allocation)
    result = select_and_allocate(
        request,
        _descriptors(),
        _banks(),
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(wall_timeout_s=1),
    )
    assert result.status == "search_timeout"
    assert result.ordering_attempts == 1 and result.candidate is None

    def unavailable(*args, **kwargs):
        raise EGraphTimeout("watchdog elapsed")

    clock[0] = 0.0
    monkeypatch.setattr(native_search, "explore", unavailable)
    expired = select_and_allocate(request, _descriptors(), _banks(), bridge=bridge)
    assert expired.status == "search_timeout" and "watchdog" in expired.reason


def test_egraph_bridge_watchdog_has_a_distinct_timeout_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / "bridge"
    bridge.write_text("placeholder")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="merlin-egg-bridge", timeout=0.01)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(EGraphTimeout, match="timed out"):
        explore(generate_rules(_request(), _descriptors()), bridge=bridge, wall_timeout_s=0.01)


def test_search_keeps_unqualified_target_distinct_from_solver_timeout(
    bridge: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unqualified(graph, order, banks, **kwargs):
        return AllocationResult("unqualified_target", order, {}, "address units disagree")

    monkeypatch.setattr(native_search, "allocate", unqualified)
    result = select_and_allocate(_request(), _descriptors(), _banks(), bridge=bridge)
    assert result.status == "unqualified_target"
    assert "address units disagree" in result.reason
    assert result.rejected_allocation > 0


def test_generated_address_validity_and_independent_checker(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "move", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-address-map-1",
    )
    descriptor = _descriptor(
        "offset_move",
        "move",
        ("external",),
        "external",
        "i8",
        "exact",
        (2,),
        validity=(AddressConstraint("eq_offset", "out", "in0", 1), AddressConstraint("aligned", "out", value=2)),
    )
    bank = (StorageBank("external", "dram", 3, "word"),)
    result = select_and_allocate(request, (descriptor,), bank, bridge=bridge, fixed_inputs={"x": 1})
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    assert 2 in result.allocation.addresses.values()
    tampered = dict(result.allocation.addresses)
    output_id = result.graph.outputs[0]
    tampered[output_id] = 0
    checked, reason = check_assignment(result.graph, result.allocation.order, tampered, bank, fixed_inputs={"x": 1})
    assert not checked and "address map" in reason
    impossible = _descriptor(
        "offset_move",
        "move",
        ("external",),
        "external",
        "i8",
        "exact",
        (2,),
        validity=(AddressConstraint("eq_offset", "out", "in0", 2),),
    )
    failed = select_and_allocate(request, (impossible,), bank, bridge=bridge, fixed_inputs={"x": 1})
    assert failed.status == "compile_error" and failed.rejected_allocation == 1


def test_address_relations_require_explicit_common_units() -> None:
    graph = CandidateGraph(
        (
            Value(0, "input", "bytes", 1, (), "x", "input"),
            Value(
                1,
                "instruction",
                "rows",
                1,
                (0,),
                None,
                "instruction",
                (AddressConstraint("eq_offset", "out", "in0", 1),),
            ),
        ),
        (1,),
    )
    banks = (StorageBank("bytes", "external", 4, "byte"), StorageBank("rows", "scratch", 4, "row"))
    result = allocate(graph, (1,), banks)
    assert result.status == "unqualified_target" and "units" in result.reason
    valid, reason = check_assignment(graph, (1,), {0: 0, 1: 1}, banks)
    assert not valid and "units" in reason


def test_descriptor_address_bounds_are_solved_and_checked(bridge: Path) -> None:
    tensor = _type()
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "move", ("x",), tensor),
        ),
        outputs=("y",),
        output_storages=("register",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-address-bounds-1",
    )
    descriptor = _descriptor(
        "bounded_move",
        "move",
        ("external",),
        "register",
        "i8",
        "exact",
        (2,),
        validity=(AddressConstraint("at_least", "out", value=2), AddressConstraint("at_most", "out", value=2)),
    )
    banks = (StorageBank("external", "dram", 4, "tile"), StorageBank("register", "regfile", 4, "register"))
    result = select_and_allocate(request, (descriptor,), banks, bridge=bridge)
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    output_id = result.graph.outputs[0]
    assert result.allocation.addresses[output_id] == 2
    for address, reason_text in ((1, "lower"), (3, "upper")):
        changed = dict(result.allocation.addresses)
        changed[output_id] = address
        checked, reason = check_assignment(result.graph, result.allocation.order, changed, banks)
        assert not checked and reason_text in reason
    impossible = _descriptor(
        "bounded_move",
        "move",
        ("external",),
        "register",
        "i8",
        "exact",
        (2,),
        validity=(AddressConstraint("at_least", "out", value=4),),
    )
    rejected = select_and_allocate(request, (impossible,), banks, bridge=bridge)
    assert rejected.status == "compile_error" and rejected.rejected_allocation == 1
    with pytest.raises(ValueError, match="nonnegative"):
        AddressConstraint("at_least", "out", value=-1)
    with pytest.raises(ValueError, match="no right-hand"):
        AddressConstraint("at_most", "out", "in0", 2)


def test_boundary_input_survives_later_instruction_writes() -> None:
    graph = CandidateGraph(
        (
            Value(0, "input", "external", 1, (), "x", "input"),
            Value(1, "first", "external", 1, (0,), "a", "instruction"),
            Value(2, "second", "external", 1, (1,), "y", "instruction"),
        ),
        (2,),
    )
    order = (1, 2)
    cramped = (StorageBank("external", "dram", 2, "word"),)
    # The first instruction consumes x, but the later output must not overwrite
    # x merely because that read has completed.
    valid, reason = check_assignment(graph, order, {0: 0, 1: 1, 2: 0}, cramped, fixed_inputs={"x": 0})
    assert not valid and "input" in reason
    assert allocate(graph, order, cramped, fixed_inputs={"x": 0}).status == "infeasible_candidate"

    roomy = (StorageBank("external", "dram", 3, "word"),)
    result = allocate(graph, order, roomy, fixed_inputs={"x": 0})
    assert result.status == "feasible", result.reason
    assert result.addresses[0] == 0
    assert result.addresses[1] != 0 and result.addresses[2] != 0


def test_input_retention_survives_rule_generation_and_controls_allocation(bridge: Path) -> None:
    tensor = TensorType((2, 2), "i8", "exact")

    def request(retention: str) -> KernelRequest:
        return KernelRequest(
            nodes=(
                SemanticNode("x", "input", (), tensor, attrs=(("input_retention", retention),), effect="input"),
                SemanticNode("a", "stage_a", ("x",), tensor),
                SemanticNode("y", "stage_b", ("a",), tensor),
            ),
            outputs=("y",),
            output_storages=("external",),
            input_storages=(("x", "external"),),
            target_identity="synthetic-input-retention",
        )

    descriptors = (
        _descriptor("first", "stage_a", ("external",), "external", "i8", "exact", (2,)),
        _descriptor("second", "stage_b", ("external",), "external", "i8", "exact", (2,)),
    )
    bank = (StorageBank("external", "dram", 2, "word"),)
    retained = request("preserve")
    reusable = request("reusable")
    assert KernelRequest.from_record(reusable.record()) == reusable
    assert retained.digest() != reusable.digest()
    with pytest.raises(ValueError, match="input retention"):
        request("unknown")
    with pytest.raises(ValueError, match="input retention applies only"):
        SemanticNode("a", "stage_a", ("x",), tensor, attrs=(("input_retention", "reusable"),))

    retained_result = select_and_allocate(
        retained,
        descriptors,
        bank,
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert retained_result.status != "selected"
    reusable_result = select_and_allocate(
        reusable,
        descriptors,
        bank,
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert reusable_result.status == "selected", reusable_result.reason
    assert reusable_result.graph is not None and reusable_result.allocation is not None
    source = next(value for value in reusable_result.graph.values if value.kind == "input")
    assert not source.preserve_input
    assert (
        reusable_result.allocation.addresses[source.id]
        == reusable_result.allocation.addresses[reusable_result.graph.outputs[0]]
    )

    roomy = (StorageBank("external", "dram", 3, "word"),)
    stable = select_and_allocate(
        retained,
        descriptors,
        roomy,
        bridge=bridge,
        fixed_inputs={"x": 0},
        limits=SearchLimits(candidate_nodes=2, candidates=8),
    )
    assert stable.status == "selected"
    assert stable.rules is not None and stable.candidate is not None
    assert stable.exploration is not None and stable.allocation is not None
    source_symbol = next(symbol for symbol, row in stable.rules.symbols.items() if row["kind"] == "input")
    changed_symbols = {symbol: row.copy() for symbol, row in stable.rules.symbols.items()}
    changed_symbols[source_symbol]["preserve_input"] = False
    changed_rules = replace(stable.rules, symbols=changed_symbols)
    changed_graph = lower_candidate(stable.candidate, changed_rules)
    checked = check_selection(
        retained,
        descriptors,
        changed_rules,
        stable.exploration,
        stable.candidate,
        changed_graph,
        stable.allocation,
        roomy,
        fixed_inputs={"x": 0},
    )
    assert not checked.valid and "input retention" in checked.reason


def test_reserved_scratch_and_register_aliases_are_checked() -> None:
    banks = (
        StorageBank("fp8_view", "tensor_registers", 4, "register"),
        StorageBank("bf16_view", "tensor_registers", 4, "register", alignment=2),
    )
    reserved = (Reservation("fp8_view", 1, 2),)
    single = CandidateGraph((Value(0, "copy", "fp8_view", 1, (), None, "instruction"),), (0,))
    result = allocate(single, (0,), banks, reservations=reserved)
    assert result.status == "feasible"
    assert result.addresses[0] in {0, 3}
    checked, reason = check_assignment(single, (0,), {0: 1}, banks, reservations=reserved)
    assert not checked and "reserved" in reason

    pair = CandidateGraph((Value(0, "wide", "bf16_view", 2, (), None, "instruction"),), (0,))
    assert allocate(pair, (0,), banks, reservations=reserved).status == "infeasible_candidate"
    checked, reason = check_assignment(pair, (0,), {0: 0}, banks, reservations=reserved)
    assert not checked and "reserved" in reason
    bad = (Reservation("fp8_view", 2, 2), Reservation("bf16_view", 2, 1))
    assert allocate(single, (0,), banks, reservations=bad).status == "unqualified_target"
    assert allocate(single, (0,), banks, reservations=(Reservation("fp8_view", 4, 1),)).status == "unqualified_target"
    with pytest.raises(ValueError, match="reservation"):
        Reservation("fp8_view", True, 1)


def test_final_replay_checks_reservations_and_binds_them_to_fingerprint(bridge: Path) -> None:
    request = _request()
    reserved = (Reservation("a", 0, 1),)
    result = select_and_allocate(
        request,
        _descriptors(),
        _banks(),
        bridge=bridge,
        fixed_inputs={"x": 0},
        reservations=reserved,
    )
    assert result.status == "selected", result.reason
    assert result.graph is not None and result.allocation is not None
    assert result.rules is not None and result.exploration is not None and result.candidate is not None
    value_id = next(value.id for value in result.graph.values if value.storage == "a")
    assert result.allocation.addresses[value_id] == 1
    checked = check_selection(
        request,
        _descriptors(),
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        result.allocation,
        _banks(),
        fixed_inputs={"x": 0},
        reservations=reserved,
    )
    assert checked.valid and checked.fingerprint == result.check_fingerprint
    without_reservation = check_selection(
        request,
        _descriptors(),
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        result.allocation,
        _banks(),
        fixed_inputs={"x": 0},
    )
    assert without_reservation.valid and without_reservation.fingerprint != checked.fingerprint
    tampered_addresses = dict(result.allocation.addresses)
    tampered_addresses[value_id] = 0
    tampered = check_selection(
        request,
        _descriptors(),
        result.rules,
        result.exploration,
        result.candidate,
        result.graph,
        replace(result.allocation, addresses=tampered_addresses),
        _banks(),
        fixed_inputs={"x": 0},
        reservations=reserved,
    )
    assert not tampered.valid and "reserved" in tampered.reason


def test_search_preserves_invalid_resource_contract_status(bridge: Path) -> None:
    result = select_and_allocate(
        _request(),
        _descriptors(),
        _banks(),
        bridge=bridge,
        reservations=(Reservation("missing_store", 0, 1),),
    )
    assert result.status == "unqualified_target"
    assert "unknown storage bank" in result.reason


def _reference_feasible(
    graph: CandidateGraph,
    order: tuple[int, ...],
    banks: tuple[StorageBank, ...],
    fixed: dict[str, int],
    reservations: tuple[Reservation, ...] = (),
    fixed_outputs: tuple[int | None, ...] | None = None,
    trials: list[int] | None = None,
) -> bool:
    """Tiny independent enumerator: at most three values and 4^3 assignments."""
    bank_by_name = {bank.name: bank for bank in banks}
    issue: dict[int, int] = {}
    completed: dict[int, int] = {}
    clock = 0
    for value_id in order:
        instruction = graph.values[value_id]
        clock += 1
        issue[value_id] = clock
        clock += instruction.completion_offset
        completed[value_id] = clock
    lifetimes = []
    domains = []
    for value in graph.values:
        birth = issue.get(value.id, 0)
        death = completed.get(value.id, birth)
        for parent in graph.values:
            for port, child in enumerate(parent.children):
                if child == value.id:
                    offset = parent.input_read_offsets[port] if parent.input_read_offsets else 0
                    death = max(death, issue[parent.id] + offset)
        if value.id in graph.outputs:
            death = clock + 1
        lifetimes.append((birth, death))
        bank = bank_by_name[value.storage]
        possible = range(0, bank.capacity - value.extent + 1)
        domains.append([addr for addr in possible if addr % bank.alignment == 0])
    for assignment in itertools.product(*domains):
        if trials is not None:
            trials[0] += 1
        if any(
            bank_by_name[value.storage].backing == bank_by_name[reserved.storage].backing
            and assignment[value.id] < reserved.start + reserved.extent
            and reserved.start < assignment[value.id] + value.extent
            for value in graph.values
            for reserved in reservations
        ):
            continue
        if any(
            value.kind == "input" and value.source_node in fixed and assignment[value.id] != fixed[value.source_node]
            for value in graph.values
        ):
            continue
        if fixed_outputs is not None and any(
            address is not None and assignment[value_id] != address
            for value_id, address in zip(graph.outputs, fixed_outputs)
        ):
            continue
        legal = True
        for value in graph.values:
            ports = {"out": assignment[value.id]}
            ports.update((f"in{port}", assignment[child]) for port, child in enumerate(value.children))
            for condition in value.validity:
                if condition.kind == "eq_offset" and (ports[condition.lhs] - ports[condition.rhs] != condition.value):
                    legal = False
                elif condition.kind == "aligned" and ports[condition.lhs] % condition.value:
                    legal = False
            if not legal:
                break
        if not legal:
            continue
        for left in graph.values:
            for right in graph.values[left.id + 1 :]:
                if bank_by_name[left.storage].backing != bank_by_name[right.storage].backing:
                    continue
                left_birth, left_death = lifetimes[left.id]
                right_birth, right_death = lifetimes[right.id]
                both_live = left_birth <= right_death and right_birth <= left_death
                left_addr, right_addr = assignment[left.id], assignment[right.id]
                overlap = left_addr < right_addr + right.extent and right_addr < left_addr + left.extent
                retained_input = (left.kind == "input" and left.preserve_input and right.kind == "instruction") or (
                    right.kind == "input" and right.preserve_input and left.kind == "instruction"
                )
                if retained_input and overlap:
                    legal = False
                    break
                if both_live and overlap:
                    exact_reuse = False
                    for child, parent in ((left, right), (right, left)):
                        if child.kind != "instruction" or parent.kind != "instruction":
                            continue
                        if child.id in graph.outputs or child.extent != parent.extent:
                            continue
                        if parent.children.count(child.id) != 1:
                            continue
                        port = parent.children.index(child.id)
                        if port not in parent.in_place_inputs or parent.read_offset(port) >= parent.completion_offset:
                            continue
                        if sum(value.children.count(child.id) for value in graph.values) != 1:
                            continue
                        exact_reuse = left_addr == right_addr
                    if not exact_reuse:
                        legal = False
                        break
            if not legal:
                break
        if legal:
            return True
    return False


def test_200_bounded_allocations_agree_with_independent_enumerator() -> None:
    """200 seeded graphs, 220 order instances, at most 4^3 assignments each.

    Vary physical aliases, extent, alignment, address maps, fixed I/O,
    in-place legality and delayed input reads. The oracle uses only Python
    finite enumeration, never the native constraint builder or checker.
    """
    rng = random.Random(1907)
    outcomes = {"feasible": 0, "infeasible_candidate": 0}
    in_place_only = 0
    offset_values: set[int] = set()
    constrained_outcomes: set[bool] = set()
    delayed_cases = 0
    checked_orders = 0
    two_order_cases = 0
    enumerated_assignments = 0
    max_assignments_in_one_order = 0
    for index in range(200):
        branch = index % 10 == 1
        two_stores = index % 2 == 0
        aliases = two_stores and index % 3 == 0
        # Pin one known in-place-only witness inside the seeded population.
        capacity_a = 2 if index == 7 else rng.randint(1, 4)
        capacity_b = rng.randint(1, 4)
        banks = (StorageBank("a", "shared", capacity_a, "slot"),)
        if two_stores:
            banks += (StorageBank("b", "shared" if aliases else "other", capacity_b, "slot"),)
        storage = "b" if two_stores and index % 4 == 0 else "a"
        values = [Value(0, "source", "a", 1 if index == 7 else rng.randint(1, 2), (), "input", "input")]
        first_conditions = (
            (AddressConstraint("eq_offset", "out", "in0", (index // 6) % 3 - 1),)
            if index % 6 == 0
            else (AddressConstraint("aligned", "out", value=2),)
            if index % 8 == 0
            else ()
        )
        values.append(
            Value(
                1,
                "compute",
                storage,
                1 if index == 7 else rng.randint(1, 2),
                (0,),
                None,
                "instruction",
                validity=first_conditions,
                input_read_offsets=(1,) if index % 13 == 0 else (),
                completion_offset=1 if index % 13 == 0 else 0,
            )
        )
        if index % 5:
            later_completion = 2 if index % 13 == 0 else 1 if index % 7 == 0 else 0
            later_conditions = (
                (AddressConstraint("eq_offset", "out", "in0", (index // 9) % 3 - 1),)
                if index % 9 == 0
                else (AddressConstraint("aligned", "out", value=2),)
                if index % 11 == 0
                else ()
            )
            values.append(
                Value(
                    2,
                    "consume",
                    "a",
                    1 if index == 7 else rng.randint(1, 2),
                    (0,) if branch else (1,),
                    None,
                    "instruction",
                    validity=later_conditions,
                    input_read_offsets=(1,) if index % 13 == 0 else (),
                    completion_offset=later_completion,
                    in_place_inputs=(0,) if index % 7 == 0 else (),
                )
            )
        graph = CandidateGraph(tuple(values), (1, 2) if branch else (values[-1].id,))
        orders = ((1, 2), (2, 1)) if branch else (tuple(value.id for value in values if value.kind == "instruction"),)
        two_order_cases += branch
        fixed = (
            {"input": 0}
            if index == 7
            else (
                {"input": rng.randint(0, min(1, capacity_a - values[0].extent))}
                if index % 3 and capacity_a >= values[0].extent
                else {}
            )
        )
        pinned: list[int | None] = []
        for position, value_id in enumerate(graph.outputs):
            output = values[value_id]
            output_capacity = next(bank.capacity for bank in banks if bank.name == output.storage)
            select = (position == 0 and index % 4 in (0, 1)) or (position == 1 and index % 8 == 1)
            pinned.append(
                rng.randint(0, min(2, output_capacity - output.extent))
                if select and output_capacity >= output.extent
                else None
            )
        fixed_outputs = (
            (1,) if index == 7 else (tuple(pinned) if any(address is not None for address in pinned) else None)
        )
        reservations = (Reservation("a", capacity_a - 1, 1),) if index % 4 == 0 else ()
        conditions = tuple(condition for value in values for condition in value.validity)
        offset_values.update(condition.value for condition in conditions if condition.kind == "eq_offset")
        delayed_cases += any(value.input_read_offsets for value in values)
        for order in orders:
            checked_orders += 1
            trials = [0]
            expected = _reference_feasible(graph, order, banks, fixed, reservations, fixed_outputs, trials)
            enumerated_assignments += trials[0]
            max_assignments_in_one_order = max(max_assignments_in_one_order, trials[0])
            if conditions:
                constrained_outcomes.add(expected)
            if len(values) == 3 and values[-1].in_place_inputs and expected:
                ordinary = CandidateGraph((*values[:2], replace(values[-1], in_place_inputs=())), graph.outputs)
                if not _reference_feasible(ordinary, order, banks, fixed, reservations, fixed_outputs):
                    in_place_only += 1
            result = allocate(
                graph,
                order,
                banks,
                fixed_inputs=fixed,
                reservations=reservations,
                fixed_outputs=fixed_outputs,
                timeout_ms=5000,
            )
            assert result.status in outcomes, (index, order, result)
            outcomes[result.status] += 1
            assert (result.status == "feasible") == expected, (index, order, graph, banks, fixed, fixed_outputs, result)
    assert all(outcomes.values()), outcomes
    assert in_place_only > 0
    assert offset_values == {-1, 0, 1}
    assert constrained_outcomes == {False, True}
    assert delayed_cases > 0
    assert two_order_cases == 20 and checked_orders == 220
    assert 0 < max_assignments_in_one_order <= 64
    print(
        json.dumps(
            {
                "schema": "merlin.native_tiny_allocation_reference.v1",
                "graphs": 200,
                "order_instances": checked_orders,
                "two_order_graphs": two_order_cases,
                "enumerated_assignments": enumerated_assignments,
                "max_assignments_in_one_order": max_assignments_in_one_order,
                "outcomes": outcomes,
                "in_place_only_witnesses": in_place_only,
                "delayed_read_graphs": delayed_cases,
                "address_offsets": sorted(offset_values),
            },
            sort_keys=True,
        )
    )


def test_30_two_step_instruction_choices_agree_with_independent_path_enumeration(bridge: Path) -> None:
    """30 graphs; two unary steps, up to two instruction choices per step, two storage classes."""
    selected_cases = 0
    no_legal_path_cases = 0
    max_enumerated_paths = 0
    for case in range(30):
        dimension = 2 + case % 3
        tensor = TensorType((dimension, 2), "i8", "exact")
        request = KernelRequest(
            nodes=(
                SemanticNode("x", "input", (), tensor, effect="input"),
                SemanticNode("a", "stage_a", ("x",), tensor),
                SemanticNode("y", "stage_b", ("a",), tensor),
            ),
            outputs=("y",),
            output_storages=("external",),
            input_storages=(("x", "external"),),
            target_identity=f"synthetic-choice-{case}",
        )
        first: list[tuple[str, str]] = []
        second: list[tuple[str, str]] = []
        descriptors: list[InstructionDescriptor] = []
        if case % 2 == 0:
            first.append(("a_register", "register"))
            descriptors.append(
                _descriptor(
                    "a_register",
                    "stage_a",
                    ("external",),
                    "register",
                    "i8",
                    "exact",
                    (2,),
                    output_axis_bounds=(AxisBound(0, maximum=3),),
                )
            )
        if case % 3 != 0:
            first.append(("a_external", "external"))
            descriptors.append(
                _descriptor(
                    "a_external",
                    "stage_a",
                    ("external",),
                    "external",
                    "i8",
                    "exact",
                    (2,),
                )
            )
        if case % 5 != 0:
            second.append(("b_register", "register"))
            descriptors.append(
                _descriptor(
                    "b_register",
                    "stage_b",
                    ("register",),
                    "external",
                    "i8",
                    "exact",
                    (2,),
                )
            )
        if case % 7 != 0:
            second.append(("b_external", "external"))
            descriptors.append(
                _descriptor(
                    "b_external",
                    "stage_b",
                    ("external",),
                    "external",
                    "i8",
                    "exact",
                    (2,),
                )
            )
        # Independent finite path oracle. No native matcher, extractor or solver is used.
        legal_paths = {
            (first_name, second_name)
            for first_name, output_storage in first
            for second_name, required_storage in second
            if output_storage == required_storage and (first_name != "a_register" or dimension <= 3)
        }
        max_enumerated_paths = max(max_enumerated_paths, len(first) * len(second))
        banks = (StorageBank("external", "dram", 5, "word"), StorageBank("register", "regs", 2, "word"))
        result = select_and_allocate(
            request,
            tuple(descriptors),
            banks,
            bridge=bridge,
            fixed_inputs={"x": 0},
            limits=SearchLimits(candidate_nodes=2, candidates=8),
        )
        assert (result.status == "selected") == bool(legal_paths), (case, result.status, result.reason)
        if not legal_paths:
            no_legal_path_cases += 1
            continue
        selected_cases += 1
        assert result.graph is not None and result.allocation is not None and result.rules is not None
        instructions = {value.source_node: value for value in result.graph.values if value.kind == "instruction"}
        assert set(instructions) == {"a", "y"}, (case, result.graph)
        source = next(value for value in result.graph.values if value.kind == "input")
        first_value, second_value = instructions["a"], instructions["y"]
        assert first_value.children == (source.id,)
        assert second_value.children == (first_value.id,)
        assert result.graph.outputs == (second_value.id,)
        path = (
            result.rules.symbols[first_value.symbol]["descriptor"]["name"],
            result.rules.symbols[second_value.symbol]["descriptor"]["name"],
        )
        assert path in legal_paths, (case, path, legal_paths)
        assert first_value.storage == dict(first)[path[0]]
        assert second_value.storage == "external"
        # Check the chosen physical witness against this finite problem directly.
        addresses = result.allocation.addresses
        assert addresses[source.id] == 0
        assert 1 <= addresses[second_value.id] < 5
        if first_value.storage == "external":
            assert 1 <= addresses[first_value.id] < 5
            assert addresses[first_value.id] != addresses[second_value.id]
        else:
            assert 0 <= addresses[first_value.id] < 2
    print(
        json.dumps(
            {
                "schema": "merlin.native_tiny_selection_reference.v1",
                "cases": 30,
                "selected_witnesses_checked": selected_cases,
                "no_legal_path_cases": no_legal_path_cases,
                "max_paths_enumerated_per_case": max_enumerated_paths,
                "instruction_steps": 2,
                "storage_classes": 2,
                "shape_first_axis": [2, 3, 4],
            },
            sort_keys=True,
        )
    )
