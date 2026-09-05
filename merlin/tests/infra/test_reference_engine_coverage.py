"""The correctness oracle must model every op the graded corpus actually uses.

Phase 1 buys a functionally certified baseline so phase 2 can iterate on cycles cheaply. That only
holds while the correctness oracle can still execute the corpus. On 2026-09-04 the phase-1 library
in a launch worktree was replaced wholesale with another branch's copy on the assumption that the
other branch was newer and therefore a superset. It was not: the branches had diverged, and the
replacement silently dropped CONV2D, MOVEMENT, ATTENTION_QK and ATTENTION_PV from the reference
engine while adding BIAS_ADD. Nothing failed at import. It surfaced an hour later as a functional
qualification refusing nine capsules with "no definition for", and it had already been misread once
as a genuine capability gap in the target -- a whole capsule family was recorded blocked over it.

A shrinking oracle is invisible until something asks it to execute the op it lost, so it is asserted
here instead of discovered downstream.
"""
from __future__ import annotations

import pytest

from merlin.runtime.reference import MODELED_OPCODES

#: Ops the graded corpus is known to emit. Each entry is here because a capsule needs it, not
#: because the engine happens to implement it -- losing any one makes those capsules ungradeable
#: on this oracle while every other check still passes.
REQUIRED = {
    "MATMUL": "every contraction capsule",
    "MATMUL_RESIDENT": "the resident-reuse and residency-regime families",
    "COMMIT": "every capsule; also the completion point the barrier families count",
    "RES_PACK": "weight-stationary packing",
    "VECTOR_MAP": "elementwise capsules",
    "VREDUCE": "reduction capsules",
    "CONV2D": "the convolution capsules and the window-reuse performance family",
    "MOVEMENT": "the data-movement capsules",
    "ATTENTION_QK": "the attention capsules",
    "ATTENTION_PV": "the attention capsules",
}


@pytest.mark.parametrize("opcode", sorted(REQUIRED))
def test_the_reference_engine_models_every_opcode_the_corpus_needs(opcode):
    assert opcode in MODELED_OPCODES, (
        f"the correctness oracle no longer models {opcode!r}, needed by {REQUIRED[opcode]}. "
        f"A capsule using it cannot be graded, and the failure appears as an opaque refusal far "
        f"from whatever removed it. Currently modelled: {sorted(MODELED_OPCODES)}")


def test_the_modelled_set_only_grows():
    """A regression here is a capability LOSS, which no amount of downstream evidence recovers."""
    missing = sorted(set(REQUIRED) - set(MODELED_OPCODES))
    assert not missing, (
        f"the reference engine lost {missing}. If an op was removed deliberately, remove it from "
        f"REQUIRED in the same change and say why -- do not let the two drift apart silently.")
