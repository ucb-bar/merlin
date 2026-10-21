"""Selection identity and bookkeeping for shared instruction DAGs."""

import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from merlin.semantic_compiler.egg_bridge import ENode, Exploration
from merlin.semantic_compiler.extract import (
    Candidate,
    Choice,
    ExtractionSession,
    _cycle_components,
    enumerate_candidates,
)
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import InstructionDescriptor, generate_rules


def legacy_key(choice: Choice) -> tuple:
    base = (
        choice.eclass, choice.symbol, choice.storage,
        tuple(legacy_key(child) for child in choice.children),
    )
    return base if not choice.bundle_instance else (
        *base, choice.bundle_instance, choice.bundle_result_index, choice.bundle_result_count,
    )


def test_shared_dag_keys_do_not_repeatedly_expand_producers() -> None:
    calls = 0

    class CountedChoice(Choice):
        def key(self) -> tuple:
            nonlocal calls
            calls += 1
            return super().key()

    depth = 12
    root = CountedChoice(0, "input", "external")
    for index in range(1, depth + 1):
        root = CountedChoice(index, f"i_{index}", "register", (root, root))
    assert root.key() == legacy_key(root)
    assert Candidate((root, root)).instruction_count() == depth
    for _ in range(5):
        root.key()
    # Shared producer identity should require work proportional to the DAG,
    # rather than the exponentially larger unfolded expression tree.
    assert calls <= 20 * (depth + 1)


def test_keys_preserve_ordered_outputs_sharing_and_bundle_identity() -> None:
    source = Choice(0, "input", "external")
    projections = tuple(
        Choice(index + 1, f"i_result_{index}", "register", (source,), "issue_0", index, 2)
        for index in range(2)
    )
    sink = Choice(3, "i_store", "external", projections)
    candidate = Candidate((sink, projections[0]))
    expected = json.dumps(
        [legacy_key(choice) for choice in candidate.outputs],
        separators=(",", ":"), sort_keys=True,
    )
    assert candidate.digest() == sha256(expected.encode()).hexdigest()
    assert candidate.instruction_count() == 2
    assert candidate.bundles_complete()
    assert not Candidate((projections[0],)).bundles_complete()
    assert candidate.digest() != Candidate(tuple(reversed(candidate.outputs))).digest()
    assert replace(projections[0], storage="external").key() != projections[0].key()
    assert replace(projections[0], bundle_instance="issue_1").physical_instance() != projections[0].physical_instance()
    assert replace(sink, children=tuple(reversed(projections))).key() != sink.key()


def test_choice_children_are_immutable() -> None:
    leaf = Choice(0, "input", "external")
    with pytest.raises(ValueError, match="immutable tuple"):
        Choice(1, "i_use", "register", [leaf])
    with pytest.raises(ValueError, match="selected choices"):
        Choice(1, "i_use", "register", (0,))
    with pytest.raises(FrozenInstanceError):
        leaf.children = (leaf,)


def test_extraction_reuses_repeated_subproblems() -> None:
    visits = 0

    class CountedClasses(dict):
        def get(self, key, default=None):
            nonlocal visits
            visits += 1
            return super().get(key, default)

    depth = 10
    tensor = TensorType((1,), "i8", "exact")
    nodes = (SemanticNode("x0", "input", (), tensor, effect="input"),) + tuple(
        SemanticNode(f"x{i}", "combine", (f"x{i-1}", f"x{i-1}"), tensor)
        for i in range(1, depth + 1)
    )
    request = KernelRequest(
        nodes=nodes, outputs=(f"x{depth}", f"x{depth}"),
        input_storages=(("x0", "external"),), output_storages=("external", "external"),
        target_identity="synthetic-shared-dag",
    )
    descriptor = InstructionDescriptor(
        "combine", "combine", ("external", "external"), "external", "i8", "exact", (1,),
        input_dtypes=("i8", "i8"), input_numerical_policies=("exact", "exact"),
    )
    program = generate_rules(request, (descriptor,))
    symbols = {
        metadata["source_node"]: name for name, metadata in program.symbols.items()
        if metadata["kind"] in {"input", "instruction"}
    }
    classes = CountedClasses({
        i: (ENode(symbols[node.id], () if i == 0 else (i-1, i-1)),)
        for i, node in enumerate(nodes)
    })
    exploration = Exploration((depth, depth), tuple(classes), classes, "saturated", 1, len(classes))
    session = ExtractionSession(exploration, request, program)
    candidates = list(enumerate_candidates(
        exploration, request, program, node_budget=depth, max_candidates=2, session=session,
    ))
    assert len(candidates) == 1
    assert candidates[0].instruction_count() == depth
    assert visits <= 4 * len(classes)
    before = visits
    replay = list(enumerate_candidates(
        exploration, request, program, node_budget=depth, max_candidates=2, session=session,
    ))
    assert [c.digest() for c in replay] == [c.digest() for c in candidates]
    assert visits == before
    for changed in (
        {"exploration": replace(exploration)}, {"request": replace(request)},
        {"program": replace(program)}, {"deadline": 1.0},
    ):
        args = {"exploration": exploration, "request": request, "program": program} | changed
        with pytest.raises(ValueError, match="different graph, request, rules or deadline"):
            list(enumerate_candidates(**args, node_budget=depth, max_candidates=2, session=session))


def test_cyclic_copy_alternatives_keep_their_ancestor_context() -> None:
    tensor = TensorType((1,), "i8", "exact")
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("a", "identity", ("x",), tensor),
            SemanticNode("y", "identity", ("a",), tensor),
        ),
        outputs=("a", "y"), input_storages=(("x", "external"),),
        output_storages=("register", "external"), target_identity="synthetic-copy-cycle",
    )
    descriptors = tuple(
        InstructionDescriptor(
            name, "identity", (source,), output, "i8", "exact", (1,),
            input_dtypes=("i8",), input_numerical_policies=("exact",),
        )
        for name, source, output in (
            ("load", "external", "register"), ("store", "register", "external"),
        )
    )
    program = generate_rules(request, descriptors)
    leaf_symbol = next(k for k, v in program.symbols.items() if v["kind"] == "input")
    symbols = {
        (v["source_node"], v["descriptor"]["name"]): k
        for k, v in program.symbols.items() if v["kind"] == "instruction"
    }
    load, store = symbols["a", "load"], symbols["y", "store"]
    exploration = Exploration(
        (1, 0), (0, 1, 0),
        {0: (ENode(leaf_symbol, ()), ENode(store, (1,))), 1: (ENode(load, (0,)),)},
        "saturated", 1, 3,
    )
    leaf = Choice(0, leaf_symbol, "external")
    loaded = Choice(1, load, "register", (leaf,))
    stored = Choice(0, store, "external", (loaded,))
    expected = [Candidate((loaded, leaf)), Candidate((loaded, stored))]
    session = ExtractionSession(exploration, request, program)
    narrow = list(enumerate_candidates(
        exploration, request, program, node_budget=1, max_candidates=4, session=session,
    ))
    assert [c.digest() for c in narrow] == [expected[0].digest()]
    actual = list(enumerate_candidates(
        exploration, request, program, node_budget=3, max_candidates=4, session=session,
    ))
    assert [c.digest() for c in actual] == [c.digest() for c in expected]


def test_cycle_components_match_all_three_class_graphs() -> None:
    # Independently close all 512 directed graphs on three vertices, including
    # self edges. A wrong component could incorrectly share cycle exclusions.
    for bits in range(1 << 9):
        reachable = [[i == j or bool(bits & (1 << (3 * i + j))) for j in range(3)] for i in range(3)]
        classes = {
            i: (ENode("unused", tuple(j for j in range(3) if bits & (1 << (3 * i + j)))),)
            for i in range(3)
        }
        for middle in range(3):
            for start in range(3):
                for end in range(3):
                    reachable[start][end] |= reachable[start][middle] and reachable[middle][end]
        graph = Exploration((), (), classes, "saturated", 1, 3)
        components = _cycle_components(graph, None)
        assert all(
            (j in components[i]) == (reachable[i][j] and reachable[j][i])
            for i in range(3) for j in range(3)
        ), bits
