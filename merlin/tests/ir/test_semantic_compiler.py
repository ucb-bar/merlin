"""Public mechanism checks for the target-independent native selector."""

from __future__ import annotations

import builtins
import itertools
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
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.rules import (
    AddressConstraint,
    AxisBound,
    AxisEquality,
    InstructionDescriptor,
    generate_rules,
)
from merlin.semantic_compiler.search import SearchLimits, select_and_allocate
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


def test_typed_graph_rejects_unknown_initialization_and_policy_merges() -> None:
    with pytest.raises(ValueError, match="positive static"):
        TensorType((0, 2), "i8", "exact")
    with pytest.raises(ValueError, match="stateful computation"):
        SemanticNode("bad", "dma_launch", (), _type(), effect="state")
    assert (
        SemanticNode("v", "sum", (), _type("ordered"), attrs=(("axis", 1),)).semantic_key()
        != SemanticNode("other", "sum", (), _type("reassociated"), attrs=(("axis", 1),)).semantic_key()
    )


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
    )
    constant = TensorValue(tensor, (2, -3, 4, 0))
    first = evaluate_graph(request, {"x": TensorValue(tensor, (1, 2, 3, 4))}, constants={"c": constant})
    assert [value.elements for value in first] == [(3, -2, 21, 16), (3, -1, 7, 4)]
    second = evaluate_graph(request, {"x": TensorValue(tensor, (5, 6, 7, 8))}, constants={"c": constant})
    assert [value.elements for value in second] == [(35, 18, 77, 64), (7, 3, 11, 8)]
    with pytest.raises(ValueError, match="declared constants"):
        evaluate_graph(request, {"x": TensorValue(tensor, (1, 2, 3, 4))})
    with pytest.raises(ValueError, match="outside its admitted domain"):
        TensorValue(tensor, (1 << 31, 0, 0, 0))


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
        if isinstance(command, (list, tuple)) and any("act" in str(part).lower() for part in command):
            raise AssertionError("native selection attempted an ACT subprocess")
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

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"act", "act_backend", "taidl", "taidl_to"}:
            raise AssertionError("native snapshot generation attempted an ACT import")
        return original_import(name, *args, **kwargs)

    def guarded_run(command: object, *args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if isinstance(command, (list, tuple)) and any("act" in str(part).lower() for part in command):
            raise AssertionError("native snapshot generation attempted an ACT subprocess")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", guarded_run)
    crate = repo_root() / "src/merlin/semantic_compiler/egg_bridge"
    profile = NativeTargetProfile(_request().target_identity, _descriptors(), _banks())
    assert profile.record()["schema"] == "merlin.native_target_profile.v3"
    old_profile = dict(profile.record(), schema="merlin.native_target_profile.v2")
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


def test_delayed_operand_read_extends_lifetime_and_forces_serial_issue() -> None:
    graph = CandidateGraph(
        (
            Value(0, "pointer", "scalar", 1, (), "pointer", "input"),
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


def _reference_feasible(
    graph: CandidateGraph,
    order: tuple[int, ...],
    banks: tuple[StorageBank, ...],
    fixed: dict[str, int],
) -> bool:
    """Tiny independent enumerator: at most three values and 4^3 assignments."""
    bank_by_name = {bank.name: bank for bank in banks}
    issue = {value_id: index + 1 for index, value_id in enumerate(order)}
    lifetimes = []
    domains = []
    for value in graph.values:
        birth = issue.get(value.id, 0)
        death = birth
        for parent in graph.values:
            if value.id in parent.children:
                death = max(death, issue[parent.id])
        if value.id in graph.outputs:
            death = len(order) + 1
        lifetimes.append((birth, death))
        bank = bank_by_name[value.storage]
        possible = range(0, bank.capacity - value.extent + 1)
        domains.append([addr for addr in possible if addr % bank.alignment == 0])
    for assignment in itertools.product(*domains):
        if any(
            value.kind == "input" and value.source_node in fixed and assignment[value.id] != fixed[value.source_node]
            for value in graph.values
        ):
            continue
        legal = True
        for left in graph.values:
            for right in graph.values[left.id + 1 :]:
                if bank_by_name[left.storage].backing != bank_by_name[right.storage].backing:
                    continue
                left_birth, left_death = lifetimes[left.id]
                right_birth, right_death = lifetimes[right.id]
                both_live = left_birth <= right_death and right_birth <= left_death
                left_addr, right_addr = assignment[left.id], assignment[right.id]
                overlap = left_addr < right_addr + right.extent and right_addr < left_addr + left.extent
                if both_live and overlap:
                    legal = False
                    break
            if not legal:
                break
        if legal:
            return True
    return False


def test_200_bounded_allocations_agree_with_independent_enumerator() -> None:
    """200 seeded instances; 2-3 values, 1-2 stores, capacity 1-4 slots, extent 1-2."""
    rng = random.Random(1907)
    outcomes = {"feasible": 0, "infeasible_candidate": 0}
    for index in range(200):
        two_stores = index % 2 == 0
        aliases = two_stores and index % 3 == 0
        capacity_a = rng.randint(1, 4)
        capacity_b = rng.randint(1, 4)
        banks = (StorageBank("a", "shared", capacity_a, "slot"),)
        if two_stores:
            banks += (StorageBank("b", "shared" if aliases else "other", capacity_b, "slot"),)
        storage = "b" if two_stores and index % 4 == 0 else "a"
        values = [Value(0, "source", "a", rng.randint(1, 2), (), "input", "input")]
        values.append(Value(1, "compute", storage, rng.randint(1, 2), (0,), None, "instruction"))
        if index % 5:
            values.append(Value(2, "consume", "a", rng.randint(1, 2), (1,), None, "instruction"))
        graph = CandidateGraph(tuple(values), (values[-1].id,))
        order = tuple(value.id for value in values if value.kind == "instruction")
        fixed = {"input": rng.randint(0, 1)} if index % 3 else {}
        expected = _reference_feasible(graph, order, banks, fixed)
        result = allocate(graph, order, banks, fixed_inputs=fixed, timeout_ms=5000)
        assert result.status in outcomes, (index, result)
        outcomes[result.status] += 1
        assert (result.status == "feasible") == expected, (index, graph, banks, fixed, result)
    assert all(outcomes.values()), outcomes


def test_30_two_step_instruction_choices_agree_with_independent_path_enumeration(bridge: Path) -> None:
    """30 graphs; two unary steps, up to two instruction choices per step, two storage classes."""
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
        first: list[str] = []
        second: list[str] = []
        descriptors: list[InstructionDescriptor] = []
        if case % 2 == 0:
            first.append("register")
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
            first.append("external")
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
            second.append("register")
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
            second.append("external")
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
        legal_first = {storage for storage in first if storage != "register" or dimension <= 3}
        expected = any(storage in second for storage in legal_first)
        result = select_and_allocate(
            request,
            tuple(descriptors),
            (StorageBank("external", "dram", 5, "word"), StorageBank("register", "regs", 2, "word")),
            bridge=bridge,
            fixed_inputs={"x": 0},
            limits=SearchLimits(candidate_nodes=2, candidates=8),
        )
        assert (result.status == "selected") == expected, (case, result.status, result.reason)
