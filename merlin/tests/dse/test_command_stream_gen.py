"""The reorder emitter: an A/B pair that is the same work in a different order.

On a hardware-interlocked target every reordering is correct, so a capsule gated on bit-exactness
passes every candidate. The pair's value is entirely in being provably identical work with a different
schedule, and these tests exist to hold that property -- plus the negative control, without which a
measured rise cannot be attributed to hoisting at all.
"""
from __future__ import annotations

from merlin.perf import command_stream_gen as G


def _c(opcode, **operands):
    return {"opcode": opcode, "operands": operands}


# The shape a real capsule parses to: a resident weight, two computes, two commits, an eviction.
_STREAM = [
    _c("RES_PACK", src="W", dst="W_res"),
    _c("MATMUL_RESIDENT", lhs="A0", rhs="W_res", dst="acc0"),
    _c("COMMIT", src="acc0", dst="Y0"),
    _c("MATMUL_RESIDENT", lhs="A1", rhs="W_res", dst="acc1"),
    _c("COMMIT", src="acc1", dst="Y1"),
    _c("EVICT", handle="W_res"),
]


class TestDependenceFromOperandKeys:
    def test_the_written_operand_is_the_one_named_dst(self):
        assert G.writes(_c("COMMIT", src="acc0", dst="Y0")) == {"Y0"}
        assert G.reads(_c("COMMIT", src="acc0", dst="Y0")) == {"acc0"}

    def test_read_after_write_is_refused(self):
        assert G.depends(_STREAM[1], _STREAM[2]) == "read-after-write"

    def test_write_after_read_is_refused(self):
        # Permuting across a WAR changes which value a third command later observes, even though the
        # hardware would execute either order happily.
        assert G.depends(_c("X", src="t", dst="u"), _c("Y", src="z", dst="t")) == "write-after-read"

    def test_write_after_write_is_refused(self):
        assert G.depends(_c("X", dst="t"), _c("Y", dst="t")) == "write-after-write"

    def test_two_reads_of_the_same_operand_do_not_conflict(self):
        assert G.depends(_STREAM[1], _STREAM[3]) is None


class TestLifetimeCommands:
    def test_a_command_with_no_dst_kills_the_operands_it_names(self):
        # THE BUG THIS EXISTS FOR. An eviction names a handle and produces nothing; treating that
        # handle as an ordinary read let the hoist search move the eviction above every compute using
        # the resident (two reads do not conflict), and the candidate evicted the weight before the
        # matmuls that read it -- reported as identical work.
        evict = _c("EVICT", handle="W_res")
        assert G.writes(evict) == {"W_res"}
        assert G.reads(evict) == frozenset()
        assert G.depends(_STREAM[1], evict) == "write-after-read"

    def test_an_eviction_cannot_be_hoisted_above_a_use_of_its_handle(self):
        out, _crossed, why = G.hoist(_STREAM, 5, 1)
        assert out is None and why == "write-after-read"

    def test_the_rule_is_the_absence_of_a_write_key_not_an_opcode_name(self):
        # A target whose grammar gains another lifetime command is covered with no edit here.
        assert G._is_lifetime_op(_c("SOME_OTHER_RELEASE", thing="t"))
        assert not G._is_lifetime_op(_c("SOME_OTHER_RELEASE", thing="t", dst="u"))


class TestThePair:
    def test_the_pair_is_the_lever(self):
        # Both computes issued before either commit: issue-before-wait, which is this archetype's only
        # lever. The commits are the waits -- the point a value becomes host-visible.
        pair = G.reorder_pair(_STREAM)
        assert [c["opcode"] for c in pair.candidate] == [
            "RES_PACK", "MATMUL_RESIDENT", "MATMUL_RESIDENT", "COMMIT", "COMMIT", "EVICT"]
        assert pair.moved_opcode == "MATMUL_RESIDENT" and pair.moved_from == 3 and pair.moved_to == 1

    def test_the_two_members_are_identical_work(self):
        pair = G.reorder_pair(_STREAM)
        assert pair.identical_work
        assert G.work_fingerprint(pair.baseline) == G.work_fingerprint(pair.candidate)

    def test_a_fingerprint_is_order_insensitive_and_catches_a_dropped_command(self):
        # eta is a ratio, so a candidate that does LESS work raises it without scheduling anything
        # better. The fingerprint is what lets the falsifier refuse that comparison.
        fp = G.work_fingerprint(_STREAM)
        assert G.work_fingerprint(list(reversed(_STREAM))) == fp
        assert G.work_fingerprint(_STREAM[:-1]) != fp

    def test_a_stream_with_nothing_legal_to_move_refuses_rather_than_inventing_a_pair(self):
        single = [_c("RES_PACK", src="W", dst="W_res"),
                  _c("CONV2D", ifm="I", weight="W_res", dst="Y0"),
                  _c("EVICT", handle="W_res")]
        pair = G.reorder_pair(single)
        assert pair.candidate == [] and pair.refusal == G.REFUSED_NO_CANDIDATE
        assert not pair.identical_work

    def test_an_empty_stream_refuses(self):
        assert G.reorder_pair([]).refusal == G.REFUSED_NO_CANDIDATE


class TestNegativeControl:
    def test_a_control_exists_and_names_the_dependence_that_blocks_it(self):
        # Without a pair whose hoist is impossible, a rise cannot be attributed to hoisting rather
        # than to noise or to some unrelated difference between two runs.
        ctl = G.negative_control(_STREAM)
        assert ctl.refusal == "read-after-write"
        assert ctl.candidate == []

    def test_the_control_is_not_the_pair(self):
        pair, ctl = G.reorder_pair(_STREAM), G.negative_control(_STREAM)
        assert pair.candidate and not ctl.candidate


class TestFromInterface:
    def test_a_real_capsule_yields_a_pair_and_a_control(self):
        from merlin.common.paths import repo_root
        p = (repo_root() / "merlin" / "contract" / "capsules" / "isa" / "A6_resident_reuse"
             / "capsule.interface.mlir")
        if not p.is_file():
            import pytest
            pytest.skip("the resident-reuse capsule is not present in this checkout")
        got = G.pair_from_interface(p.read_text(encoding="utf-8"))
        assert got["pair"]["identical_work"] is True
        assert got["negative_control"]["refusal"]
        assert "eta must RISE" in got["pass_condition"]
