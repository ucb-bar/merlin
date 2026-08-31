"""The composition axis: how a capsule assembles work across the host/accelerator seam.

Reported separately from the ``(family, dtype, alignment)`` cells because it answers a different
question, and derived from each capsule's own interface rather than declared on it.
"""
from __future__ import annotations

import textwrap

import pytest

from merlin.targetgen import boundary as B


class TestSequenceClassification:
    @pytest.mark.parametrize("seq,kind", [
        ("A", B.A),
        ("AA", B.A_A),
        ("AAA", B.A_A),
        ("HAH", B.H_A_H),
        ("HHAHH", B.H_A_H),
        ("AHA", B.A_H_A),
        ("AHHA", B.A_H_A),
        ("HAHAH", B.ROUTING),
        ("HHH", B.HOST_ONLY),
    ])
    def test_each_shape_is_named(self, seq, kind):
        assert B.classify_sequence(list(seq)) == kind

    def test_an_empty_sequence_is_unknown_not_host_only(self):
        # "no accelerator work" and "we could not tell" are different facts. Collapsing them reports an
        # unread capsule as a proven host-only one.
        assert B.classify_sequence([]) == B.UNKNOWN
        assert B.classify_sequence(["?", "?"]) == B.UNKNOWN

    def test_an_unresolved_region_does_not_manufacture_a_seam(self):
        # Treating an unclassifiable region as host work would turn a plain pair of accelerator regions
        # into an A->H->A island -- unearned coverage of the most expensive shape there is.
        assert B.classify_sequence(["A", "?", "A"]) == B.A_A

    def test_one_host_op_between_accelerator_runs_is_an_island_not_a_mixed_graph(self):
        assert B.classify_sequence(list("AHA")) == B.A_H_A
        assert B.classify_sequence(list("AHAHA")) == B.ROUTING


class TestPatternsPresent:
    def test_a_whole_model_contains_more_shapes_than_its_own_label(self):
        # The requirement side needs every shape a capture CONTAINS. A model labelled `routing` end to
        # end still contains isolated dispatches and host islands, and each is a composition the corpus
        # must exercise somewhere; taking only the whole-model label demands `routing` and nothing else.
        found = B.patterns_in_sequence(list("HAHHAAHHAH"))
        assert B.ROUTING in found
        assert B.A in found and B.A_A in found and B.H_A_H in found and B.A_H_A in found

    def test_an_all_host_sequence_presents_only_the_host_shape(self):
        assert B.patterns_in_sequence(list("HHH")) == {B.HOST_ONLY}

    def test_an_unreadable_sequence_presents_nothing(self):
        assert B.patterns_in_sequence([]) == set()


_IFACE = textwrap.dedent('''\
    module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
      %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
      %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>
      %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
      %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
      %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
      merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
    }
    ''')


class TestInterfaceGrammar:
    def test_a_single_dispatch_is_A(self):
        p = B.profile_iface_text(_IFACE)
        assert p.kind == B.A and p.grammar == "merlin_iface" and p.n_accel_regions == 1

    def test_two_commits_under_one_resident_weight_is_A_to_A(self):
        two = _IFACE.replace(
            '  merlin_iface.evict',
            '  %acc1 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>\n'
            '  %Y1 = merlin_iface.commit %acc1 {name = "Y1", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>\n'
            '  merlin_iface.evict')
        assert B.profile_iface_text(two).kind == B.A_A

    def test_k_accumulation_is_one_dispatch_not_two(self):
        # Several matmuls feeding ONE commit is K-accumulation: the intermediate never becomes a value the
        # host can see, so counting matmuls would report an accumulation capsule as proving composition.
        acc = _IFACE.replace(
            '  %Y0 = merlin_iface.commit',
            '  %acc1 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>\n'
            '  %Y0 = merlin_iface.commit')
        assert B.profile_iface_text(acc).kind == B.A

    def test_the_grammar_cannot_express_a_host_seam(self):
        # Tensors enter and leave through host MEMORY, but host memory is not host COMPUTATION. Counting
        # it as one would label every capsule H->A->H and make the axis say nothing at all.
        assert B.profile_iface_text(_IFACE).kind != B.H_A_H


class TestDroppedOpsAreNotAWrongAnswer:
    def test_an_op_outside_the_frozen_grammar_yields_unknown_with_its_name(self):
        # MEASURED, and the reason this check exists: the canonical parser used to return only the
        # commands it recognised and report no error for the rest. 15 of 160 shipped interface capsules
        # contained such an op -- every `movement` capsule parsed to ZERO commands, and every
        # flash-attention capsule silently lost its second matmul. A shape derived from a partial program
        # is a wrong answer, so an op the grammar does not define must reach UNKNOWN by name.
        #
        # The probe mnemonic is deliberately one nobody will ever implement. Naming a real gap here would
        # make this falsifier expire the moment that gap is closed, which is exactly what happened when
        # it was first written against `movement`: the check went green by losing its subject.
        text = textwrap.dedent('''\
            module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
              %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi8>
              %Y0 = merlin_iface.not_an_op_in_any_grammar %X {name = "Y0"} : (tensor<16x16xi8>) -> tensor<16x16xi8>
            }
            ''')
        p = B.profile_iface_text(text)
        assert p.kind == B.UNKNOWN
        assert "not_an_op_in_any_grammar" in p.detail

    def test_the_probe_mnemonic_really_is_undefined(self):
        # Guards the test above: if the probe ever became a real op, that test would pass for the wrong
        # reason and stop testing anything at all.
        assert "not_an_op_in_any_grammar" not in B.grammar_mnemonics()

    def test_the_known_mnemonics_come_from_the_parser_not_from_a_second_list(self):
        # One authority. If a mnemonic is added to the grammar, this follows for free instead of drifting.
        known = B.grammar_mnemonics()
        assert {"matmul", "commit", "resident_pack", "evict"} <= known
        assert B.undefined_mnemonics(_IFACE) == []

    def test_a_type_is_not_an_op(self):
        # `!merlin_iface.resident` appears on nearly every line of a real capsule; reading it as an op
        # would report the whole shipped corpus as using an undefined mnemonic.
        assert "resident" not in B.iface_mnemonics(_IFACE)
        assert "acc" not in B.iface_mnemonics(_IFACE)

    def test_a_module_attribute_is_not_an_op(self):
        assert "version" not in B.iface_mnemonics(_IFACE)
        assert "target" not in B.iface_mnemonics(_IFACE)


class TestGapReport:
    def test_a_required_shape_with_no_capsule_is_reported(self):
        req = {"by_kind": {B.A: ["m"], B.A_H_A: ["m"], B.ROUTING: ["m"]}}
        corpus = {"by_kind": {B.A: ["c0"], B.ROUTING: ["c1"]}, "unreadable": {}}
        gap = B.uncovered_boundaries(req, corpus)
        assert gap["uncovered"] == [B.A_H_A]
        assert gap["n_covered"] == 2 and gap["n_required"] == 3

    def test_a_shape_the_corpus_has_and_no_model_needs_is_extra_not_missing(self):
        req = {"by_kind": {B.A: ["m"]}}
        corpus = {"by_kind": {B.A: ["c0"], B.HOST_ONLY: ["c1"]}, "unreadable": {}}
        assert B.uncovered_boundaries(req, corpus)["extra_kinds"] == [B.HOST_ONLY]

    def test_unreadable_capsules_travel_with_the_gap(self):
        gap = B.uncovered_boundaries({"by_kind": {}}, {"by_kind": {}, "unreadable": {"c": "why"}})
        assert gap["unreadable_capsules"] == {"c": "why"}
