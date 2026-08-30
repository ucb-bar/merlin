"""The closed ROLE vocabulary — what an instruction DOES to a compute endpoint.

This is the generalization of ``lift_asm``. That lifter recognizes RVV by counting mnemonic literals,
which works exactly once: on a derived-ISA target there are no stable mnemonics at all. ``llvm-objdump``
prints ``<unknown>`` for an OPU word and for custom-3, an atlas kernel spells the same instruction four
different ways across its own corpus and its own assembler, and a Gemmini kernel looks IDENTICAL in C
whether it lowers to the hardware loop FSM or to a fine-grained preload/compute sequence — the level is
a property of the emitted stream, not of the source text.

What every endpoint DOES have is a derived encoding table and, per entry, a role. So the lifter keys on
roles, and roles are the only vocabulary shared across targets.

**Roles do double duty**, which is why they live here rather than inside one decoder: they feed the CCA
lifter AND they derive the engine set (:mod:`merlin.kernels.engines`). One derivation with two consumers
beats two lists that must be kept in agreement — the failure mode a duplicated role table already caused
once, when a second copy dropped a condition and let a target that merely pushes weights claim it could
multiply.

The set is CLOSED on purpose. An open vocabulary would let each target invent its own spelling for
"drain the accumulator", which is precisely the per-target lifter this exists to avoid.
"""
from __future__ import annotations

from typing import Any

#: role -> what it means, and what it licenses. Order is documentation, not precedence.
ROLES: dict[str, str] = {
    "accumulate": "the multiply-accumulate itself: the instruction that advances a partial sum",
    "operand_load": "move an activation/operand INTO the endpoint's local storage",
    "weight_load": "move a weight into the endpoint (stationary or streamed)",
    "broadcast": "fan one operand across the endpoint's cells/lanes",
    "move": "relocate data WITHIN the endpoint's own registers/tiles, computing nothing",
    "readout": "drain a result out of the accumulator or tile",
    "commit": "make a result architecturally visible (store/retire the readout)",
    "elementwise": "apply a unary/binary op across lanes — a lane engine's compute, not a contraction",
    "config": "set endpoint state that later instructions inherit (vector length, dataflow, dtype)",
    "loop_descriptor": "hand the endpoint a whole loop nest to run itself (a hardware-loop FSM)",
    "control": "branch/jump/link — the loop structure AROUND the work, computing nothing itself",
    "divergence": "change which LANES are live: thread mask, predicate, split/join",
    "warp_control": "partition work across threads of control: spawn/retire warps",
    "sync": "order or wait: fences, barriers, completion polls",
    "dma": "bulk asynchronous movement between memories, not through the compute datapath",
}

#: Roles that are EVIDENCE of a compute engine, and which facet each evidences.
#:
#: Deliberately narrower than ``ROLES``: ``config``/``sync``/``dma``/``commit`` occur on every endpoint
#: ever built, so their presence distinguishes nothing and they evidence no engine. Mirrors — and is
#: checked against — ``targetgen.capability_derive._ROLE_ENGINE``, which reads the same distinction off
#: the ISA census.
#: ``control`` likewise evidences no engine, and it exists to keep the UNROLED bucket meaningful.
#: Without it, branch and jump instructions sit beside a target's genuinely unexplained custom opcodes
#: in the same "claimed but no role" pile — so a report cannot separate "this is not endpoint work" from
#: "we own this gap and can close it", which is the only distinction that makes the number actionable.
#:
#: ``move`` evidences NO engine on purpose: every endpoint with more than one register can shuffle
#: between them, so its presence distinguishes nothing. It earns a role anyway because a stream full of
#: register moves is a real and diagnosable shape -- an operand ladder being rebuilt per step instead of
#: broadcast -- and lumping those instructions into "the tool named it" hides that shape entirely.
ROLE_EVIDENCES_ENGINE: dict[str, str] = {
    "accumulate": "spatial",
    "weight_load": "spatial",
    "broadcast": "spatial",
    "readout": "spatial",
    "loop_descriptor": "spatial",
    "elementwise": "vector",
    # Both evidence a SIMT engine and nothing else: only a machine with many threads of control can
    # have some of them diverge, or spawn more. This is the first rung in the tree that can observe
    # `simt` at all -- every SIMT declaration was UNCHECKED before, because a role census over typed
    # operands cannot see what makes SIMT what it is.
    "divergence": "simt",
    "warp_control": "simt",
}


#: The DERIVED IsaModel's role vocabulary -> this one. Target-agnostic: both sides are our own
#: vocabularies, so this bridge holds for every self-hosted-ISA target and belongs here rather than in a
#: per-endpoint data file.
#:
#: ``scalar`` maps to NOTHING on purpose. A scalar instruction is the code AROUND the loop (the
#: envelope), not an operation on a compute endpoint, and giving it a role would make every target look
#: like it drives an engine it does not have.
FROM_ISA_ROLE: dict[str, str] = {
    "matmul": "accumulate",
    "weight_load": "weight_load",
    # Seeds the accumulator with a starting value — data moving INTO the endpoint, like a preload-with-D.
    "acc_seed": "operand_load",
    "acc_readout": "readout",
    "acc_readout_scaled": "readout",
    "tensor_compute_unary": "elementwise",
    "tensor_compute_binary": "elementwise",
    # A derived-ISA role named "memory" says an instruction touches memory. It does NOT say the
    # movement is BULK or ASYNCHRONOUS, which is what `dma` means here ("bulk asynchronous movement
    # between memories, not through the compute datapath"). Mapping it to `dma` fabricated that
    # claim: on the one target that sources roles this way the ISA's "memory" role names a
    # VMEM->register load, so every such load was counted as bulk async movement while the target's
    # ACTUAL DMA engine instructions -- which carry no ISA role at all -- were counted as nothing.
    # Three facets read the `dma` role (dispatch.dma_overlap, dispatch.double_buffered_banks,
    # memory.dma_pattern), so all three answered confidently about a DMA engine they had never seen.
    # `operand_load` is what the instruction demonstrably is; a target whose ISA model distinguishes
    # asynchronous channel movement should name that role separately rather than have this one widened
    # back, because the widening is unobservable from the name.
    "memory": "operand_load",
}


def from_isa_role(isa_role: str) -> str | None:
    """This vocabulary's name for a derived-IsaModel role, or None when the role drives no endpoint."""
    return FROM_ISA_ROLE.get(str(isa_role))

#: Roles that must appear for a stream to be a COMPLETE contraction on a compute endpoint.
#:
#: The measured reason this exists: a prior audit counted emitted accumulates and reported success for a
#: kernel that never drained its accumulator, because nothing asked whether the result came back out.
#: A readout with no counter is how an accumulate-without-extraction goes unnoticed.
CONTRACTION_ROLES: tuple[str, ...] = ("operand_load", "accumulate", "readout")


def is_role(name: Any) -> bool:
    return isinstance(name, str) and name in ROLES


def check_roles(names) -> None:
    """Raise for any name outside the closed vocabulary, listing what was offered."""
    bad = sorted({str(n) for n in names if not is_role(n)})
    if bad:
        raise KeyError(f"unknown instruction role(s) {bad}; the vocabulary is closed: {sorted(ROLES)}")


def engine_of(role: str) -> str | None:
    """The engine facet ``role`` evidences, or None for a role that evidences none.

    None is a real answer, not a failure: every endpoint configures, syncs and moves data.
    """
    if not is_role(role):
        raise KeyError(f"unknown instruction role {role!r}; known: {sorted(ROLES)}")
    return ROLE_EVIDENCES_ENGINE.get(role)


def missing_contraction_roles(seen) -> tuple[str, ...]:
    """Which roles a complete contraction needs that ``seen`` does not contain."""
    have = {str(s) for s in seen}
    return tuple(r for r in CONTRACTION_ROLES if r not in have)
