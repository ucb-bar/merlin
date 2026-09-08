"""A declared epilogue stage the readout does not apply is a correctness defect, not a nuance.

Measured on hardware: a capsule declaring ``COMMIT output_dtype='i32' epilogue=['relu']`` returned
126 of 256 outputs negative with ``min = -85`` -- exactly the raw accumulator -- while its
command-buffer numeric floor (L0) and trace (L1) both passed. The compiler's intent was right; the
device discarded the activation. Its sibling with the identical declaration PASSES its grade,
because the default stimulus is non-negative, so ``max(0, x)`` is the identity on every value that
program can produce and the discard is unobservable.

Nothing in the module under test names a target, a dtype width, or an opcode: the rule is general
(a declared stage must be applied by the readout the program selected) and the capability is the
caller's declaration. These tests use a synthetic two-readout target for the rule, and the real
backend's declaration only where the point is that it fires on real data.
"""
from __future__ import annotations

import pytest

from merlin.verify import epilogue_applicability as EA
from merlin.verify.epilogue_applicability import ReadoutCapability

#: A synthetic target with two readouts: one that requantizes and one that dumps the accumulator.
NARROW = ReadoutCapability("narrow", frozenset({"scale", "activation"}), "applies scale + activation")
WIDE = ReadoutCapability("wide", frozenset(), "writes the raw accumulator")
CAPS = (NARROW, WIDE)


def _cb(*commands):
    return {"commands": list(commands)}


def _commit(readout, epilogue, opcode="COMMIT"):
    attrs = {"epilogue": list(epilogue)}
    if readout is not None:
        attrs["output_dtype"] = readout
    return {"opcode": opcode, "attributes": attrs}


class TestTheRuleIsGeneral:
    def test_a_readout_that_applies_the_stage_passes(self):
        got = EA.assess(_cb(_commit("narrow", ["activation"])), CAPS)
        assert got.status == "applied" and not got.refusing
        assert got.discarded == ()

    def test_a_readout_that_does_not_apply_it_is_DISCARDED(self):
        got = EA.assess(_cb(_commit("wide", ["activation"])), CAPS)
        assert got.status == "discarded" and got.refusing
        assert len(got.discarded) == 1
        assert "computes something other than what it declares" in got.discarded[0].why

    def test_no_epilogue_is_not_applicable_rather_than_a_pass(self):
        """"nothing to apply" and "everything applied" are different facts."""
        got = EA.assess(_cb(_commit("wide", [])), CAPS)
        assert got.status == "not_applicable" and not got.refusing

    def test_an_undescribed_readout_is_UNKNOWN_and_refuses(self):
        """Assuming a readout applies whatever is asked is how the original defect stayed invisible."""
        got = EA.assess(_cb(_commit("mystery", ["activation"])), CAPS)
        assert got.status == "unknown" and got.refusing
        assert "refused rather than assumed" in got.stages[0].why

    def test_a_command_with_an_epilogue_but_no_readout_is_UNKNOWN(self):
        got = EA.assess(_cb(_commit(None, ["activation"])), CAPS)
        assert got.status == "unknown" and got.refusing

    def test_a_partially_applied_epilogue_is_discarded_and_names_the_stage(self):
        got = EA.assess(_cb(_commit("narrow", ["activation", "pool"])), CAPS)
        assert got.status == "discarded"
        assert [v.stage for v in got.discarded] == ["pool"]
        assert [v.stage for v in got.stages if v.applied] == ["activation"]

    def test_every_stage_of_every_command_is_reported_not_just_the_first(self):
        got = EA.assess(_cb(_commit("wide", ["a", "b"]), _commit("narrow", ["activation"])), CAPS)
        assert len(got.stages) == 3 and len(got.discarded) == 2
        assert {v.command_index for v in got.discarded} == {0}

    def test_the_declared_readouts_travel_with_the_verdict(self):
        got = EA.assess(_cb(_commit("wide", ["activation"])), CAPS)
        assert got.readouts_declared == ("narrow", "wide")

    def test_a_target_declaring_no_readouts_refuses_everything_with_an_epilogue(self):
        got = EA.assess(_cb(_commit("narrow", ["activation"])), ())
        assert got.status == "unknown" and got.refusing

    def test_a_buffer_with_no_command_sequence_is_unknown(self):
        assert EA.assess({}, CAPS).status == "unknown"

    def test_a_non_mapping_command_is_skipped_without_crashing(self):
        got = EA.assess({"commands": ["nonsense", _commit("wide", ["activation"])]}, CAPS)
        assert got.status == "discarded"

    def test_every_status_is_in_the_declared_vocabulary(self):
        cases = (_cb(_commit("narrow", ["activation"])), _cb(_commit("wide", ["activation"])),
                 _cb(_commit("wide", [])), _cb(_commit("mystery", ["activation"])), {})
        for buf in cases:
            assert EA.assess(buf, CAPS).status in EA.STATUSES

    def test_the_refusing_set_includes_unknown(self):
        """A readout nobody described is the state the defect hid in."""
        assert "unknown" in EA.REFUSING_STATUSES and "discarded" in EA.REFUSING_STATUSES
        assert "applied" not in EA.REFUSING_STATUSES

    def test_it_serialises_with_its_counts(self):
        d = EA.assess(_cb(_commit("wide", ["a", "b"])), CAPS).to_dict()
        assert d["schema"] == "merlin_epilogue_applicability_v1"
        assert d["status"] == "discarded" and d["refusing"] is True and d["n_discarded"] == 2


class TestItFiresOnTheRealTargetDeclaration:
    """The point of the exercise: it catches the live defect a passing capsule hides."""

    def _caps(self):
        from merlin.runtime.backends import base as B
        backend = B.get_backend("gemmini")
        declared = backend.readout_epilogue_capability()
        return tuple(ReadoutCapability(r["selector"], frozenset(r["applies"]),
                                       r.get("evidence", "")) for r in declared)

    def test_the_target_declares_more_than_one_readout(self):
        caps = self._caps()
        assert len(caps) >= 2, "a single-readout target cannot exhibit this defect"

    def test_exactly_one_declared_readout_applies_nothing(self):
        """The full-width path, which is what makes a declared activation vanish."""
        empty = [c for c in self._caps() if not c.applies]
        assert len(empty) == 1
        assert "raw accumulator" in empty[0].evidence

    def test_the_shipped_relu_declaration_is_diagnosed_as_discarded(self):
        """`output_dtype='i32' epilogue=['relu']` -- what both relu capsules emit today."""
        wide = next(c for c in self._caps() if not c.applies)
        got = EA.assess(_cb(_commit(wide.selector, ["relu"])), self._caps())
        assert got.status == "discarded" and got.refusing

    def test_the_narrowing_declaration_is_diagnosed_as_applied(self):
        """Which is the fix: narrow the readout and the same epilogue becomes real."""
        narrow = next(c for c in self._caps() if c.applies)
        got = EA.assess(_cb(_commit(narrow.selector, ["relu"])), self._caps())
        assert got.status == "applied" and not got.refusing

    def test_requant_is_declared_by_NO_readout(self):
        """merlin's integer round-half-up shift is not what this float scale computes, so it is
        not declared as something any readout applies -- it stays a host-side op."""
        assert all("requant" not in c.applies for c in self._caps())

    def test_the_stage_names_are_the_command_buffer_abi_vocabulary(self):
        from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET
        for cap in self._caps():
            unknown = cap.applies - EPILOGUE_STAGE_SET
            assert not unknown, f"{cap.selector} declares non-ABI stage(s) {sorted(unknown)}"


class TestTheGateIsScopedByWhatATargetDECLARES:
    """A target that has not described its readouts must not be refused for it.

    The rule is general, but its input is a per-target declaration. Refusing every capsule on a
    target that simply has not enumerated its readouts would be overfitting by breakage -- an
    undeclared target is not a broken one. So the grade records ``unavailable`` with the reason, and
    "not checked" can never read as "checked and fine".
    """

    def _backend(self, name):
        from merlin.runtime.backends import base as B
        return B.get_backend(name)

    def test_a_target_that_declares_readouts_activates_the_gate(self):
        assert callable(getattr(self._backend("gemmini"), "readout_epilogue_capability", None))

    def test_a_target_that_declares_none_leaves_the_gate_unavailable(self):
        """Exercised by a real second backend, not a mock, so the branch cannot rot."""
        other = self._backend("muon")
        assert not callable(getattr(other, "readout_epilogue_capability", None))

    def test_the_recorded_block_is_self_describing_even_when_nothing_applies(self):
        """An absent key reads as "this axis does not apply" -- the same shape as the defect.

        So the block a grade records always carries its schema, status and refusing flag, including
        for a program with no epilogue at all.
        """
        block = EA.assess(_cb(_commit("wide", [])), CAPS).to_dict()
        assert block["schema"] == "merlin_epilogue_applicability_v1"
        assert block["status"] == "not_applicable" and block["refusing"] is False
        assert block["n_discarded"] == 0 and "readouts_declared" in block

    def test_a_declared_stage_outside_the_abi_vocabulary_is_caught_by_the_backend_test(self):
        """Guards the declaration itself: a typo'd stage name would silently never be applied."""
        from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET
        for row in self._backend("gemmini").readout_epilogue_capability():
            for stage in row.get("applies") or ():
                assert stage in EPILOGUE_STAGE_SET, stage
