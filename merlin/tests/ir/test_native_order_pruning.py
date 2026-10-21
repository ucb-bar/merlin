"""Small independent checks for ordering and interference pruning.

The main semantic-compiler suite already enumerates 200 placement problems and
30 instruction-choice problems. These cases cover the remaining order and
pruning implications without repeating that allocation corpus.
"""

from itertools import combinations, permutations, product

from merlin.semantic_compiler.allocate import CandidateGraph, Value, may_prune_interference, topological_orders


def test_readonly_boundary_copies_move_to_first_use_without_losing_orders():
    graph = CandidateGraph(
        (
            Value(0, "a", "external", 1, (), "a", "input"),
            Value(1, "load_a", "register", 2, (0,), None, "instruction", value_preserving_copy=True),
            Value(2, "b", "external", 1, (), "b", "input"),
            Value(3, "load_b", "register", 2, (2,), None, "instruction", value_preserving_copy=True),
            Value(4, "unary_b", "register", 2, (3,), None, "instruction"),
            Value(5, "combine", "register", 2, (1, 4), None, "instruction"),
        ),
        (5,),
    )
    assert list(topological_orders(graph, limit=2)) == [(1, 3, 4, 5), (3, 4, 1, 5)]
    expected = {
        order
        for order in permutations((1, 3, 4, 5))
        if order.index(3) < order.index(4) < order.index(5) and order.index(1) < order.index(5)
    }
    assert set(topological_orders(graph, limit=24)) == expected
    assert (1, 3, 4, 5) in expected
    assert len(list(topological_orders(graph, limit=2))) == 2


def test_ordinary_boundary_computations_are_not_treated_as_loads():
    graph = CandidateGraph(
        (
            Value(0, "source", "external", 1, (), "x", "input"),
            Value(1, "first", "register", 2, (0,), None, "instruction"),
            Value(2, "second", "register", 2, (0,), None, "instruction"),
            Value(3, "third", "register", 2, (2,), None, "instruction"),
        ),
        (1, 3),
    )
    assert next(topological_orders(graph, limit=1)) == (1, 2, 3)


def test_64_instruction_dags_enumerate_every_legal_order() -> None:
    """Three instructions have at most six orders; no native order helper builds the oracle."""
    checked = multiple_orders = 0
    for dependency_bits, extent_bits in product(range(8), repeat=2):
        parents = (
            (0,),
            (1,) if dependency_bits & 1 else (0,),
            tuple(parent for bit, parent in ((2, 1), (4, 2)) if dependency_bits & bit) or (0,),
        )
        graph = CandidateGraph(
            (
                Value(0, "source", "external", 1, (), "x", "input"),
                *(
                    Value(
                        number,
                        f"instruction_{number}",
                        "register",
                        1 + bool(extent_bits & (1 << (number - 1))),
                        parents[number - 1],
                        None,
                        "instruction",
                    )
                    for number in (1, 2, 3)
                ),
            ),
            (3,),
        )
        expected = {
            order
            for order in permutations((1, 2, 3))
            if all(
                order.index(parent) < order.index(child)
                for child in (1, 2, 3)
                for parent in parents[child - 1]
                if parent != 0
            )
        }
        assert set(topological_orders(graph, limit=6)) == expected
        checked += 1
        multiple_orders += len(expected) > 1
    assert checked == 64 and multiple_orders > 0


def test_unsat_interference_prunes_only_same_base_supersets() -> None:
    """Fixed addresses 0 and 1 leave no third slot under both z interferences."""
    universe = {(0, 1), (0, 2), (1, 2)}
    failed = frozenset({(0, 2), (1, 2)})

    def satisfiable(edges: frozenset[tuple[int, int]]) -> bool:
        return any(
            all(addresses[left] != addresses[right] for left, right in edges)
            for z in (0, 1)
            for addresses in ((0, 1, z),)
        )

    assert not satisfiable(failed)
    for size in range(4):
        for edges in combinations(sorted(universe), size):
            current = frozenset(edges)
            implied_unsat = failed <= current
            assert satisfiable(current) != implied_unsat
            assert (
                may_prune_interference(
                    "fixed-x0-y1-domain-2",
                    "fixed-x0-y1-domain-2",
                    failed,
                    current,
                    failed_status="infeasible_candidate",
                )
                == implied_unsat
            )
            assert not may_prune_interference(
                "fixed-x0-y1-domain-2",
                "changed-address-domain",
                failed,
                current,
                failed_status="infeasible_candidate",
            )
            assert not may_prune_interference(
                "fixed-x0-y1-domain-2",
                "fixed-x0-y1-domain-2",
                failed,
                current,
                failed_status="search_timeout",
            )
