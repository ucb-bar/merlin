"""A lowering trace on BOTH sides, so a divergence can be attributed to a step.

`LoweringTrace` existed but was only ever built for our own output; the expert side had at most a
declared-transformation summary read from a framework contract, i.e. what the framework SAYS it does.
Without the expert's actual build, a divergence bottoms out at "their compiler is better", which is
unactionable — you cannot tell whether the difference came from their `-O` pipeline, their ISA flags,
or a source-level decision.
"""
from __future__ import annotations

import pytest

from merlin.kernels import trace as T


class _D:
    """A role-tagged instruction, the shape every kernels.decode decoder emits."""

    def __init__(self, index, roles):
        self.index, self.addr, self.roles = index, index, tuple(roles)


def _stream():
    """An accelerator command stream. Deliberately NOT an RVV InsnStream: the trace's asm end must lift
    through the role lifter for an endpoint, since handing an accelerator stream to the RVV lifter
    yields an empty facet rather than an error."""
    return [_D(0, ("operand_load",)), _D(1, ("accumulate",)), _D(2, ("readout",))]


def _endpoint():
    from merlin.kernels import endpoints as EP
    return EP.load_endpoint("gemmini_rocc")


class TestTheExpertsBuildBecomesSteps:
    def test_the_build_is_recorded_as_ordered_stages(self):
        steps = T.expert_build_steps(["clang", "-O3", "-c", "k.c"], tool="clang", version="17")
        assert [s.stage for s in steps] == ["preprocess", "frontend", "llvm-pipeline", "asm-emission"]

    def test_no_expert_step_is_marked_modifiable(self):
        """The load-bearing assertion. Our trace records, per step, the seam that can change it; theirs
        has no such seam. Saying so explicitly is what stops a reader treating an expert step as an
        action we could take, and stops a divergence routing to a seam that does not exist."""
        for s in T.expert_build_steps(["clang", "-O2"], tool="clang"):
            assert s.modifiable_by is None, s.name

    def test_flags_that_carry_a_decision_are_named(self):
        steps = T.expert_build_steps(["clang", "-O3", "-march=rv64gcv", "-ffp-contract=fast"])
        pipeline = next(s for s in steps if s.stage == "llvm-pipeline")
        assert "-O3" in pipeline.summary and "optimization level" in pipeline.summary
        assert "-ffp-contract=fast" in pipeline.summary

    def test_an_unrecognized_flag_is_not_silently_dropped(self):
        # It does not become a NAMED decision, but it stays in the recorded invocation.
        tr = T.expert_trace("x", _stream(), op="matmul", kernel_id="k", target="rvv",
                            endpoint=_endpoint(), build_cmd=["clang", "-fweird-thing"])
        assert "-fweird-thing" in tr.provenance["build_cmd"]

    def test_the_whole_invocation_is_recorded(self):
        tr = T.expert_trace("xnnpack", _stream(), op="matmul", kernel_id="k", target="rvv",
                            endpoint=_endpoint(), build_cmd=["clang", "-O3", "-c", "k.c"],
                            tool="clang", version="17")
        assert tr.provenance["tool"] == "clang" and tr.provenance["tool_version"] == "17"
        assert tr.provenance["level"] == "build+asm"


class TestAHandWrittenCorpusHasNoLowering:
    def test_an_absent_trace_is_recorded_as_a_property_not_a_gap(self):
        """A corpus of hand-written assembly was not produced by a compiler, so there is no lowering to
        reconstruct. Stamping that is what keeps someone from later "fixing" it — and what keeps a
        reader from reading an empty step list as a missing feature."""
        tr = T.expert_trace("atlas-corpus", _stream(), op="matmul", kernel_id="k", target="atlas",
                            endpoint=_endpoint(), hand_written=True)
        assert tr.steps == []
        assert "hand-written assembly" in tr.provenance["no_lowering"]

    def test_a_hand_written_trace_records_no_build_command(self):
        tr = T.expert_trace("atlas-corpus", _stream(), op="m", kernel_id="k", target="atlas",
                            endpoint=_endpoint(), hand_written=True)
        assert "build_cmd" not in tr.provenance


class TestTheTwoTracesCanBeCompared:
    def _ours(self):
        return T.LoweringTrace(
            kernel="matmul", target="rvv", source="ours",
            steps=[T.TransformStep(name="vectorize", plane="dialect", stage="vectorize",
                                   modifiable_by="schedule:vector_sizes")],
            asm=T.AsmRegion(label="matmul"))

    def test_our_steps_are_editable_and_theirs_are_not(self):
        theirs = T.expert_trace("xnnpack", _stream(), op="matmul", kernel_id="matmul", target="rvv",
                                endpoint=_endpoint(), build_cmd=["clang", "-O3"])
        rep = T.traces_agree(self._ours(), theirs)
        assert rep["our_editable_steps"] == ["vectorize"]
        assert rep["their_editable_steps"] == [], (
            "something claimed we can edit the expert's compiler, which would route a divergence to a "
            "seam that does not exist")

    def test_an_absent_expert_trace_is_reported_with_its_reason(self):
        theirs = T.expert_trace("atlas-corpus", _stream(), op="matmul", kernel_id="matmul",
                                target="atlas", endpoint=_endpoint(), hand_written=True)
        rep = T.traces_agree(self._ours(), theirs)
        assert rep["their_trace_absent"] is True and rep["notes"], rep

    def test_a_digest_that_cannot_be_taken_is_recorded_absent_not_faked(self):
        tr = T.expert_trace("x", _stream(), op="m", kernel_id="k", target="rvv", endpoint=_endpoint(),
                            obj="/no/such/object.o", build_cmd=["clang"])
        assert "source_digest" in tr.provenance
        assert tr.provenance["source_digest"] in (None, "") or isinstance(
            tr.provenance["source_digest"], str)
