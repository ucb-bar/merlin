"""Joint-output lower bounds must not discard a bounded legal instruction DAG."""

import itertools
import random
from functools import cache

import pytest

from merlin.semantic_compiler import extract
from merlin.semantic_compiler.egg_bridge import ENode, Exploration
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import InstructionDescriptor, RuleProgram


def test_insufficient_joint_budget_does_not_expand_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = TensorType((1,), "i8", "exact")
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"),
               SemanticNode("a", "step", ("x",), tensor), SemanticNode("b", "step", ("a",), tensor)),
        outputs=("b", "a"), input_storages=(("x", "external"),),
        output_storages=("external", "external"), target_identity="synthetic-joint-bound",
    )
    descriptor = InstructionDescriptor(
        "step", "step", ("external",), "external", "i8", "exact", (1,),
        input_dtypes=("i8",), input_numerical_policies=("exact",),
    )
    program = RuleProgram((), (), (), {
        "b_x": {"kind": "input", "source_node": "x"},
        "i_a": {"kind": "instruction", "descriptor": descriptor.record()},
        "i_b": {"kind": "instruction", "descriptor": descriptor.record()},
    })
    graph = Exploration((2, 1), (0, 1, 2), {
        0: (ENode("b_x", ()),), 1: (ENode("i_a", (0,)),), 2: (ENode("i_b", (1,)),),
    }, "saturated", 1, 3)
    original = extract._choices

    def forbidden(*args, **kwargs):
        raise AssertionError("jointly insufficient budget expanded an instruction choice")

    monkeypatch.setattr(extract, "_choices", forbidden)
    assert not list(extract.enumerate_candidates(graph, request, program, node_budget=1, max_candidates=2))
    monkeypatch.setattr(extract, "_choices", original)
    candidates = list(extract.enumerate_candidates(graph, request, program, node_budget=2, max_candidates=2))
    assert len(candidates) == 1 and candidates[0].instruction_count() == 2


def abstract_problem(specification, roots):
    classes, symbols = {}, {}
    for index, alternatives in enumerate(specification):
        nodes = []
        for variant, (kind, children) in enumerate(alternatives):
            symbol = ("i_" if kind == "instruction" else "b_") + f"{index}_{variant}"
            symbols[symbol] = {"kind": kind}
            if kind == "instruction":
                symbols[symbol]["descriptor"] = {"output_storage": "external"}
            nodes.append(ENode(symbol, children))
        classes[index] = tuple(nodes)
    graph = Exploration(roots, tuple(classes), classes, "saturated", 1, sum(map(len, classes.values())))
    return graph, RuleProgram((), (), (), symbols)


def lower_bound(graph, program, storages=None):
    return extract._mandatory_instruction_bound(
        graph, program, extract._cycle_components(graph, None), None,
        ("external",) * len(graph.roots) if storages is None else storages,
    )


def test_bound_intersects_alternative_dependencies_and_unions_shared_roots() -> None:
    # Class 2 has a fused one-instruction implementation or a two-step path.
    graph, program = abstract_problem([
        [("input", ())], [("instruction", (0,))],
        [("instruction", (0,)), ("instruction", (1,))],
    ], (2,))
    assert lower_bound(graph, program) == 1
    graph = Exploration((1, 2, 2), graph.class_by_node, graph.classes, "saturated", 1, graph.node_count)
    assert lower_bound(graph, program) == 2


def test_views_and_copy_cycles_do_not_add_mandatory_instruction_issues() -> None:
    graph, program = abstract_problem([
        [("input", ()), ("instruction", (0,))],
        [("instruction", (0,)), ("instruction", (1,))],
        [("view", (1,)), ("instruction", (2,))],
    ], (1, 2))
    assert lower_bound(graph, program) == 1
    # A bundle can realize separate components in one physical issue. The
    # single-result lower bound deliberately defers these cases.
    program.symbols["i_1_0"]["bundle_instance"] = "pair"
    assert lower_bound(graph, program) == 0


def test_boundary_store_adds_an_issue_without_counting_a_shared_output_twice() -> None:
    graph, program = abstract_problem([
        [("input", ())], [("instruction", (0,)), ("instruction", (1,))],
    ], (1, 1))
    program.symbols["i_1_0"]["descriptor"]["output_storage"] = "register"
    assert lower_bound(graph, program) == 2
    assert lower_bound(graph, program, ("register", "register")) == 1


def bounded_witness_counts():
    """200 graphs, three classes, unary choices, expression depth at most four.

    Enumerate every finite expression and ordered root tuple in that domain.
    Neither the native extractor nor native instruction counting decides the
    expected work; identity is the complete abstract expression tuple.
    """
    rng = random.Random(71)
    feasible = checked = 0
    for case in range(200):
        specification = []
        for _ in range(3):
            alternatives = []
            for _ in range(rng.randint(1, 2)):
                kind = rng.choice(("input", "instruction", "instruction", "view"))
                children = () if kind == "input" or rng.randrange(4) == 0 else (rng.randrange(3),)
                alternatives.append((kind, children))
            specification.append(alternatives)
        roots = (case % 3,) if case % 2 else (case % 3, (case + 1) % 3)
        graph, program = abstract_problem(specification, roots)
        bound = lower_bound(graph, program)

        @cache
        def trees(node, depth):
            result = []
            for alternative in graph.classes[node]:
                if not alternative.children:
                    result.append((alternative.symbol, ()))
                elif depth:
                    for child in trees(alternative.children[0], depth - 1):
                        result.append((alternative.symbol, (child,)))
            return tuple(result)

        def instructions(tree):
            symbol, children = tree
            result = {tree} if symbol.startswith("i_") else set()
            for child in children:
                result.update(instructions(child))
            return result

        witnesses = 0
        for outputs in itertools.product(*(trees(root, 4) for root in roots)):
            actual = set().union(*(instructions(output) for output in outputs))
            assert bound <= len(actual), (case, bound, outputs)
            witnesses += 1
        feasible += witnesses > 0
        checked += witnesses
    return feasible, checked


def test_bound_preserves_all_bounded_witnesses_in_200_tiny_choice_graphs() -> None:
    feasible, checked = bounded_witness_counts()
    assert (feasible, checked) == (145, 760)
