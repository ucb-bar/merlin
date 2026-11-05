"""A finite spill/store/reload path can revisit a logical value's e-class."""

from pathlib import Path

import pytest

from merlin.semantic_compiler import search as native_search
from merlin.semantic_compiler.allocate import StorageBank, allocate, lower_candidate, topological_orders
from merlin.semantic_compiler.egg_bridge import explore
from merlin.semantic_compiler.extract import Candidate, Choice, enumerate_candidates
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import InstructionDescriptor, generate_rules
from merlin.semantic_compiler.verify import check_selection
from merlin.tests.ir.test_semantic_compiler import bridge  # noqa: F401 -- shared session fixture


def test_checked_finite_spill_roundtrip_remains_an_extraction_alternative(bridge: Path) -> None:  # noqa: F811
    tensor = TensorType((1,), "i8", "exact")
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"),
               SemanticNode("p", "add", ("x", "x"), tensor)),
        outputs=("p",), output_storages=("register",), input_storages=(("x", "external"),),
        target_identity="synthetic-spill-roundtrip",
    )
    descriptors = tuple(
        InstructionDescriptor(
            name, op, sources, output, "i8", "exact", (1,),
            input_dtypes=("i8",) * len(sources), input_numerical_policies=("exact",) * len(sources),
            value_preserving_copy=copy,
        )
        for name, op, sources, output, copy in (
            ("add", "add", ("external", "external"), "register", False),
            ("store", "identity", ("register",), "external", True),
            ("load", "identity", ("external",), "register", True),
        )
    )
    program = generate_rules(request, descriptors)
    exploration = explore(program, bridge=bridge)
    source_class, result_class = exploration.class_by_node
    symbols = {
        v["descriptor"]["name"]: k for k, v in program.symbols.items()
        if v["kind"] == "instruction" and v["source_node"] == "p"
    }
    source = Choice(source_class, next(k for k, v in program.symbols.items() if v["kind"] == "input"), "external")
    computed = Choice(result_class, symbols["add"], "register", (source, source))
    stored = Choice(result_class, symbols["store"], "external", (computed,))
    loaded = Choice(result_class, symbols["load"], "register", (stored,))
    witness = Candidate((loaded,))
    graph = lower_candidate(witness, program)
    banks = (StorageBank("external", "dram", 2, "word"), StorageBank("register", "rf", 1, "word"))
    allocation = allocate(graph, next(topological_orders(graph, limit=1)), banks,
                          fixed_inputs={"x": 0}, fixed_outputs=(0,))
    assert allocation.status == "feasible"
    assert check_selection(request, descriptors, program, exploration, witness, graph, allocation, banks,
                           fixed_inputs={"x": 0}, fixed_outputs=(0,)).valid
    candidates = list(enumerate_candidates(exploration, request, program, node_budget=3, max_candidates=64))
    assert candidates[0].instruction_count() == 1
    assert witness.digest() in {candidate.digest() for candidate in candidates}


def test_search_uses_geometric_extraction_budgets(bridge: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    tensor = TensorType((1,), "i8", "exact")
    request = KernelRequest(
        nodes=(SemanticNode("x0", "input", (), tensor, effect="input"),) + tuple(
            SemanticNode(f"x{i}", "add", (f"x{i-1}", "x0"), tensor) for i in range(1, 8)
        ),
        outputs=("x7",), output_storages=("external",), input_storages=(("x0", "external"),),
        target_identity="synthetic-budget-growth",
    )
    descriptor = InstructionDescriptor(
        "add", "add", ("external", "external"), "external", "i8", "exact", (1,),
        input_dtypes=("i8", "i8"), input_numerical_policies=("exact", "exact"),
    )
    budgets = []

    def record(*args, **kwargs):
        budgets.append(kwargs["node_budget"])
        return enumerate_candidates(*args, **kwargs)

    monkeypatch.setattr(native_search, "enumerate_candidates", record)
    result = native_search.select_and_allocate(
        request, (descriptor,), (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge, fixed_inputs={"x0": 0}, fixed_outputs=(1,),
        limits=native_search.SearchLimits(candidate_nodes=7),
    )
    assert result.status == "selected", result.reason
    assert result.candidate.instruction_count() == 7
    assert budgets == [1, 2, 4, 7]
