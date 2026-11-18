"""Inclusive extraction budgets avoid expensive undersized large-DAG stages."""

from pathlib import Path

import pytest

from merlin.semantic_compiler import search as native_search
from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.egg_bridge import ENode, Exploration
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import InstructionDescriptor


def test_large_dag_starts_above_the_64_issue_cliff() -> None:
    limits = native_search.SearchLimits(candidate_nodes=256)
    stages = list(native_search._work_stages(67, limits))
    assert stages[0] == (1, 268, 128)
    assert 64 not in [stage[2] for stage in stages]
    assert stages[-1] == (8, 5000, 256)


def test_small_dags_retain_the_original_geometric_progression() -> None:
    limits = native_search.SearchLimits(candidate_nodes=7)
    assert list(native_search._work_stages(8, limits)) == [
        (1, 64, 1), (2, 128, 2), (4, 256, 4), (8, 512, 7),
        (8, 1024, 7), (8, 2048, 7), (8, 4096, 7), (8, 5000, 7),
    ]
    assert next(native_search._work_stages(63, limits))[2] == 1
    assert next(native_search._work_stages(64, native_search.SearchLimits(candidate_nodes=256)))[2] == 128


@pytest.mark.parametrize("cap", (1, 24, 64, 68, 100, 127))
def test_large_dag_initial_budget_respects_a_lower_declared_cap(cap: int) -> None:
    stages = list(native_search._work_stages(67, native_search.SearchLimits(candidate_nodes=cap)))
    assert stages[0][2] == cap
    assert all(stage[2] == cap for stage in stages)


def test_output_store_estimate_changes_start_without_claiming_a_lower_bound() -> None:
    limits = native_search.SearchLimits(candidate_nodes=300)
    assert next(native_search._work_stages(64, limits, output_count=65))[2] == 256


@pytest.mark.parametrize("source_nodes", (3, 63, 64, 67, 129, 500))
@pytest.mark.parametrize("limits", (
    native_search.SearchLimits(iterations=3, egraph_nodes=73, candidate_nodes=150),
    native_search.SearchLimits(iterations=1, egraph_nodes=1, candidate_nodes=1),
    native_search.SearchLimits(iterations=8, egraph_nodes=5000, candidate_nodes=256),
))
def test_stages_are_monotone_bounded_and_reach_every_declared_cap(source_nodes, limits) -> None:
    stages = list(native_search._work_stages(source_nodes, limits))
    caps = (limits.iterations, limits.egraph_nodes, limits.candidate_nodes)
    assert stages[-1] == caps
    for stage in stages:
        assert all(0 < value <= cap for value, cap in zip(stage, caps))
    assert all(all(a <= b for a, b in zip(before, after)) for before, after in zip(stages, stages[1:]))


def test_large_budget_still_selects_a_smaller_first_valid_candidate(monkeypatch) -> None:
    tensor = TensorType((1,), "i8", "exact")
    request = KernelRequest(
        nodes=(SemanticNode("x0", "input", (), tensor, effect="input"),) + tuple(
            SemanticNode(f"x{i}", "step", (f"x{i-1}",), tensor) for i in range(1, 67)
        ), outputs=("x66",), output_storages=("external",), input_storages=(("x0", "external"),),
        target_identity="synthetic-large-inclusive-budget",
    )
    descriptors = tuple(InstructionDescriptor(
        name, "step", ("external",), "external", "i8", "exact", (1,),
        input_dtypes=("i8",), input_numerical_policies=("exact",),
    ) for name in ("first", "fallback"))

    def explored(program, **kwargs):
        indices = {node.id: index for index, node in enumerate(request.nodes)}
        classes = {index: [ENode(row["symbol"], tuple(row["children"]))]
                   for index, row in enumerate(program.nodes)}
        for symbol, metadata in program.symbols.items():
            if metadata["kind"] == "instruction":
                node = request.node(metadata["source_node"])
                classes[indices[node.id]].append(ENode(symbol, tuple(indices[key] for key in node.inputs)))
        return Exploration(program.roots, tuple(range(len(program.nodes))),
                           {index: tuple(nodes) for index, nodes in classes.items()}, "saturated", 1,
                           sum(map(len, classes.values())))

    monkeypatch.setattr(native_search, "explore", explored)
    budgets = []
    enumerate_candidates = native_search.enumerate_candidates

    def enumerate_and_record(*args, **kwargs):
        budgets.append(kwargs["node_budget"])
        return enumerate_candidates(*args, **kwargs)

    monkeypatch.setattr(native_search, "enumerate_candidates", enumerate_and_record)
    materializations = []

    def materialize(selected):
        assert selected.candidate.instruction_count() == 66
        assert selected.check_fingerprint
        assert not selected.diagnostic_only
        materializations.append(selected.candidate.digest())
        return (len(materializations) > 1, "reject first checked implementation")

    result = native_search.select_and_allocate(
        request, descriptors, (StorageBank("external", "dram", 3, "word"),),
        bridge=Path("unused-synthetic-bridge"), fixed_inputs={"x0": 0}, fixed_outputs=(1,),
        limits=native_search.SearchLimits(candidate_nodes=256, candidates=4, orders_per_candidate=1),
        accept_materialization=materialize,
    )
    assert result.status == "selected", result.reason
    assert budgets == [128]
    assert result.candidate.instruction_count() == 66 < budgets[0]
    assert result.candidate_attempts == 2 and result.rejected_materialization == 1
    assert len(set(materializations)) == 2
    assert not result.diagnostic_ablations
