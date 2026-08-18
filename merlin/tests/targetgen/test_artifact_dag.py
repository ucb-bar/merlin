"""The graph exists to make a reuse claim falsifiable, so the tests are mostly attempts to falsify it.

The design doc claims a datapath addition invalidates the capability contract, routing, the unit's
lowering/codegen/CCA/tests, and *nothing* in the parent's schedule, the generic lowering, board support or
the elementwise path. In this model that is not a promise about intent — it is the claim that no path
exists between those nodes. So it is checked as a path query, and if someone later adds an edge that
creates one, this file fails rather than the paragraph quietly becoming untrue.

The rest guard the arithmetic against the ways a reuse number gets flattered: a denominator nobody
declared, a node whose sources could not be read counted as reuse, and a cycle turning the closure into
everything.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import artifact_dag as AD
from merlin.targetgen.artifact_dag import (UNKNOWN_HASH, ArtifactGraph, Node, TargetDelta,
                                           changed_from_hashes, content_hashes, pipeline_graph)

#: The nodes a hardware change must never reach. Each is either reused literally (guarded in-tree by a
#: byte-identity invariant) or target-independent.
_PARENT_NODES = ("parent_schedule", "generic_lowering", "runtime_board_support", "elementwise_path")

#: What a datapath addition touches: the hardware facts and the contract derived from them.
_HARDWARE_DELTA = frozenset({"hardware_evidence", "capability_contract"})


@pytest.fixture(scope="module")
def graph() -> ArtifactGraph:
    return pipeline_graph()


class TestTheGraphIsWellFormed:
    def test_it_validates_on_construction(self, graph):
        assert graph.problems() == ()

    def test_a_dangling_input_is_refused(self):
        # An edge to a node that does not exist propagates no invalidation, so it is worse than absent.
        with pytest.raises(ValueError, match="unknown input"):
            ArtifactGraph.of([Node("a", "k", ("ghost",))])

    def test_a_cycle_is_refused(self):
        # In a cycle the closure grows to everything reachable and the reuse ratio collapses, which reads
        # as a conservative answer instead of a broken graph.
        with pytest.raises(ValueError, match="cycle"):
            ArtifactGraph.of([Node("a", "k", ("b",)), Node("b", "k", ("a",))])

    def test_a_duplicate_node_is_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            ArtifactGraph.of([Node("a", "k"), Node("a", "k")])

    def test_every_node_is_reachable_in_a_topological_order(self, graph):
        order = graph.topo_order()
        assert len(order) == len(graph.nodes)
        seen: set[str] = set()
        for name in order:
            assert all(dep in seen for dep in graph.nodes[name].inputs), name
            seen.add(name)


class TestTheReuseClaimIsFalsifiable:
    """These are the tests that would break if the delta stopped being additive."""

    @pytest.mark.parametrize("parent", _PARENT_NODES)
    def test_a_hardware_change_does_not_reach_the_parent(self, graph, parent):
        for changed in sorted(_HARDWARE_DELTA):
            assert not graph.reaches(changed, parent), (
                f"{changed} now reaches {parent}: the delta is no longer additive, and the design's "
                "reuse claim is what has become false")

    @pytest.mark.parametrize("expected", ["capability_routing", "unit_lowering", "unit_codegen",
                                          "unit_cca", "unit_certification"])
    def test_a_hardware_change_does_reach_what_it_should(self, graph, expected):
        # The mirror of the test above. A graph where the delta reached nothing would satisfy every
        # "must not reach" test perfectly, and mean nothing at all.
        assert expected in TargetDelta(_HARDWARE_DELTA).invalidated(graph)

    def test_the_parent_nodes_are_roots(self, graph):
        # They are reused literally, so nothing in the generation pipeline may feed them.
        for name in _PARENT_NODES:
            assert graph.nodes[name].inputs == (), f"{name} acquired an input"

    def test_the_certification_surface_is_downstream_of_the_codegen_it_certifies(self, graph):
        # If it were not, a codegen change could leave a stale pass on the record.
        assert graph.reaches("unit_codegen", "unit_certification")


class TestTheGraphTracksThePipeline:
    def test_every_emit_layer_the_pipeline_offers_has_a_node(self, graph):
        # Otherwise a layer added upstream is silently outside the invalidation model, and a stale
        # artifact would be reported as reuse.
        from merlin.targetgen.pipeline import EMIT_LAYERS
        have = set(graph.of_kind("emit"))
        want = {f"emit_{layer.replace('-', '_')}" for layer in EMIT_LAYERS}
        assert want == have, f"missing {want - have}, unexpected {have - want}"

    def test_every_emit_node_is_downstream_of_a_plan(self, graph):
        plans = set(graph.of_kind("plan")) | {"capability_contract"}
        for name in graph.of_kind("emit"):
            assert set(graph.nodes[name].inputs) & plans, name

    def test_every_node_names_at_least_one_real_source(self, graph):
        # A node with no readable source hashes to UNKNOWN forever, so its reuse can never be measured.
        got = content_hashes(graph, repo_root())
        unknown = sorted(n for n, v in got.items() if v == UNKNOWN_HASH)
        assert not unknown, f"these nodes have no readable sources: {unknown}"


class TestChangeIsMeasuredNotDeclared:
    def test_a_differing_hash_is_a_change(self):
        assert changed_from_hashes({"a": "1"}, {"a": "2"}) == {"a"}

    def test_an_identical_hash_is_not(self):
        assert changed_from_hashes({"a": "1"}, {"a": "1"}) == frozenset()

    def test_a_new_node_counts_as_changed(self):
        assert changed_from_hashes({}, {"a": "1"}) == {"a"}

    def test_a_disappeared_node_counts_as_changed(self):
        assert changed_from_hashes({"a": "1"}, {}) == {"a"}

    @pytest.mark.parametrize("recorded,current", [
        ({"a": UNKNOWN_HASH}, {"a": "1"}),
        ({"a": "1"}, {"a": UNKNOWN_HASH}),
        ({"a": UNKNOWN_HASH}, {"a": UNKNOWN_HASH}),
    ])
    def test_an_unreadable_source_never_counts_as_reuse(self, recorded, current):
        # Two UNKNOWNs are equal as strings. Treating that as "unchanged" would hand back free reuse for
        # precisely the nodes nothing is known about.
        assert changed_from_hashes(recorded, current) == {"a"}

    def test_a_content_change_moves_the_hash(self, tmp_path):
        (tmp_path / "src").mkdir()
        f = tmp_path / "src" / "x.py"
        f.write_text("a = 1", encoding="utf-8")
        g = ArtifactGraph.of([Node("n", "k", (), ("src/x.py",))])
        before = content_hashes(g, tmp_path)
        f.write_text("a = 2", encoding="utf-8")
        after = content_hashes(g, tmp_path)
        assert before["n"] != after["n"]
        assert changed_from_hashes(before, after) == {"n"}

    def test_a_directory_source_notices_a_new_file(self, tmp_path):
        d = tmp_path / "pkg"
        d.mkdir()
        (d / "a.py").write_text("a", encoding="utf-8")
        g = ArtifactGraph.of([Node("n", "k", (), ("pkg",))])
        before = content_hashes(g, tmp_path)
        (d / "b.py").write_text("b", encoding="utf-8")
        assert content_hashes(g, tmp_path)["n"] != before["n"]

    def test_a_missing_source_is_unknown_not_empty(self, tmp_path):
        g = ArtifactGraph.of([Node("n", "k", (), ("absent.py",))])
        assert content_hashes(g, tmp_path)["n"] == UNKNOWN_HASH

    def test_a_node_with_no_sources_is_unknown(self, tmp_path):
        g = ArtifactGraph.of([Node("n", "k", (), ())])
        assert content_hashes(g, tmp_path)["n"] == UNKNOWN_HASH


class TestTheDeltaArithmetic:
    @pytest.fixture
    def chain(self):
        # a -> b -> c, with an unrelated root d.
        return ArtifactGraph.of([Node("a", "k"), Node("b", "k", ("a",)),
                                 Node("c", "k", ("b",)), Node("d", "k")])

    def test_invalidation_is_transitive_and_includes_the_changed_node(self, chain):
        assert TargetDelta({"a"}).invalidated(chain) == {"a", "b", "c"}

    def test_an_unrelated_root_is_not_invalidated(self, chain):
        assert "d" not in TargetDelta({"a"}).invalidated(chain)

    def test_a_leaf_change_invalidates_only_itself(self, chain):
        assert TargetDelta({"c"}).invalidated(chain) == {"c"}

    def test_a_delta_naming_an_unknown_node_raises(self, chain):
        # Silently ignoring it would understate the invalidation set, i.e. overstate reuse.
        with pytest.raises(ValueError, match="absent from the graph"):
            TargetDelta({"ghost"}).invalidated(chain)

    def test_the_denominator_must_be_declared(self, chain):
        with pytest.raises(ValueError, match="denominator"):
            TargetDelta({"a"}).measure(chain, [])

    def test_the_relevant_set_must_be_in_the_graph(self, chain):
        with pytest.raises(ValueError, match="absent from the graph"):
            TargetDelta({"a"}).measure(chain, ["a", "ghost"])

    def test_the_ratio_is_reused_over_relevant(self, chain):
        m = TargetDelta({"a"}, label="L").measure(chain, ["a", "b", "c", "d"])
        assert m.reused == ("d",) and m.reuse_ratio == 0.25
        assert m.label == "L"

    def test_the_measurement_carries_the_sets_that_produced_it(self, chain):
        d = TargetDelta({"a"}, label="L").measure(chain, chain.nodes).to_dict()
        assert d["n_reused"] + d["n_invalidated"] == d["n_relevant"]
        assert set(d["reused"]).isdisjoint(d["invalidated"])
        assert d["changed"] == ["a"]

    def test_a_narrower_relevant_set_gives_a_different_ratio(self, chain):
        # The whole reason the denominator is an argument: it changes the answer, so it has to be stated.
        wide = TargetDelta({"a"}).measure(chain, ["a", "b", "c", "d"]).reuse_ratio
        narrow = TargetDelta({"a"}).measure(chain, ["a", "b"]).reuse_ratio
        assert wide != narrow


class TestTheRealDeltaMeasurement:
    def test_the_hardware_delta_reuses_exactly_the_parent_nodes_and_the_manifest(self, graph):
        m = TargetDelta(_HARDWARE_DELTA, label="add a matrix unit").measure(graph, graph.nodes)
        assert set(m.reused) == set(_PARENT_NODES) | {"source_manifest"}

    def test_the_ratio_is_reported_rather_than_tuned(self, graph):
        # This asserts a RANGE, not a value: the point is that the number is whatever the graph says.
        # It is low because every plan reads all of evidence -- a fact about the pipeline's granularity,
        # not about how invasive the delta is. Tuning the graph to raise it would defeat the purpose.
        m = TargetDelta(_HARDWARE_DELTA).measure(graph, graph.nodes)
        assert 0.0 < m.reuse_ratio < 1.0
        assert len(m.invalidated) > len(m.reused), (
            "if this flips, check whether the graph got finer-grained or merely more flattering")
