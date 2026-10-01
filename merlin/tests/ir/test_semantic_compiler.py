"""Public mechanism checks for the target-independent native selector."""

from __future__ import annotations

import itertools
import os
import random
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.semantic_compiler.allocate import (
    CandidateGraph,
    Reservation,
    StorageBank,
    Value,
    allocate,
    check_assignment,
    lower_candidate,
    may_prune_interference,
    topological_orders,
)
from merlin.semantic_compiler.egg_bridge import explore
from merlin.semantic_compiler.extract import enumerate_candidates
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import AddressConstraint, InstructionDescriptor, generate_rules
from merlin.semantic_compiler.search import SearchLimits, select_and_allocate


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
        InstructionDescriptor("load_a", "identity", ("external",), "a", "i8", "exact", (2,)),
        InstructionDescriptor("load_b", "identity", ("external",), "b", "i8", "exact", (2,)),
        InstructionDescriptor("finish_a", "use_a", ("a",), "external", "i8", "exact", (2,)),
        InstructionDescriptor("finish_b", "use_b", ("b",), "external", "i8", "exact", (2,)),
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
    assert SemanticNode("v", "sum", (), _type("ordered"), attrs=(("axis", 1),)).semantic_key() != SemanticNode(
        "other", "sum", (), _type("reassociated"), attrs=(("axis", 1),)
    ).semantic_key()


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


def test_generated_rules_depend_on_descriptor_and_numerical_policy() -> None:
    request = _request()
    rules = generate_rules(request, _descriptors())
    assert len(rules.rewrites) == 4
    changed = InstructionDescriptor("load_a", "identity", ("external",), "a", "i8", "exact", (2,), (("missing", 1),))
    assert len(generate_rules(request, (changed,)).rewrites) == 0
    wrong_policy = InstructionDescriptor("load_a", "identity", ("external",), "a", "i8", "rounded", (2,))
    assert len(generate_rules(request, (wrong_policy,)).rewrites) == 0
    wrong_input = InstructionDescriptor(
        "load_a", "identity", ("external",), "a", "i8", "exact", (2,), input_dtypes=("bf16",)
    )
    assert len(generate_rules(request, (wrong_input,)).rewrites) == 0
    with pytest.raises(ValueError, match="duplicate required attribute"):
        InstructionDescriptor(
            "invalid", "identity", ("external",), "a", "i8", "exact", (2,),
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
    assert result.graph is not None and result.allocation is not None
    assert {value.storage for value in result.graph.values if value.kind == "instruction"} == {"a", "b", "external"}
    assert result.allocation.addresses


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


def test_unsat_pruning_direction_and_base_identity() -> None:
    small = frozenset({(1, 2)})
    large = frozenset({(1, 2), (2, 3)})
    assert may_prune_interference("same", "same", small, large, failed_status="infeasible_candidate")
    assert not may_prune_interference("same", "same", large, small, failed_status="infeasible_candidate")
    assert not may_prune_interference("old", "new", small, large, failed_status="infeasible_candidate")
    assert not may_prune_interference("same", "same", small, large, failed_status="search_timeout")


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
    good = InstructionDescriptor("good", "mix", ("external",), "external", "i8", "exact", (2,))
    good_program = generate_rules(request, (good,))
    good_symbol = next(symbol for symbol, info in good_program.symbols.items() if info["kind"] == "instruction")
    bad = None
    for number in range(100):
        trial = InstructionDescriptor(
            f"oversized_{number}", "mix", ("external",), "external", "i8", "exact", (2,), extent=3
        )
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
    descriptor = InstructionDescriptor(
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
    impossible = InstructionDescriptor(
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
            Value(1, "instruction", "rows", 1, (0,), None, "instruction", (
                AddressConstraint("eq_offset", "out", "in0", 1),
            )),
        ),
        (1,),
    )
    banks = (StorageBank("bytes", "external", 4, "byte"), StorageBank("rows", "scratch", 4, "row"))
    result = allocate(graph, (1,), banks)
    assert result.status == "unqualified_target" and "units" in result.reason
    valid, reason = check_assignment(graph, (1,), {0: 0, 1: 1}, banks)
    assert not valid and "units" in reason


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


def test_search_preserves_invalid_resource_contract_status(bridge: Path) -> None:
    result = select_and_allocate(
        _request(), _descriptors(), _banks(), bridge=bridge,
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
            bank_by_name[value.storage].backing == bank_by_name[reserved.storage].backing
            and assignment[value.id] < reserved.start + reserved.extent
            and reserved.start < assignment[value.id] + value.extent
            for value in graph.values for reserved in reservations
        ):
            continue
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
        reservations = (Reservation("a", capacity_a - 1, 1),) if index % 4 == 0 else ()
        expected = _reference_feasible(graph, order, banks, fixed, reservations)
        result = allocate(graph, order, banks, fixed_inputs=fixed, reservations=reservations, timeout_ms=5000)
        assert result.status in outcomes, (index, result)
        outcomes[result.status] += 1
        assert (result.status == "feasible") == expected, (index, graph, banks, fixed, result)
    assert all(outcomes.values()), outcomes
