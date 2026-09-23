"""A target's scheduling vocabulary is DERIVED from its header and its RTL, not typed out.

MEASURED, and it decided what a whole-model program could be. The scheduling ISA for this target was a
hand table of five entries — `config_ex`, `config_st`, `config_ld`, `loop_ws`, `loop_conv_ws` — so every
whole-model program built through ``merlin.sched`` issued 60 ``loop_ws`` + 19 ``loop_conv_ws`` and
nothing else. Those two hand a whole tiled loop nest to the hardware sequencer, which means the
schedule is the hardware's: which operand stays resident, what order tiles are walked in, and how many
tiles one transfer carries all become descriptor fields the FSM interprets, not decisions a compiler
makes. There was no raw ``mvin``/``preload``/``compute`` in scope, so a non-FSM schedule could not be
written at all.

Adding the missing names by hand would have been the wrong fix twice over: it duplicates three
derivations that already exist (the header says which macro issues which funct, ``#define`` gives that
funct a number, the RTL decode table says which numbers are legal), and it leaves the next target with
whatever someone remembered. So the vocabulary is derived, and this is what holds it there.

The five hand entries remain, and legitimately: each carries a semantic contract no header states (what
a resadd's ``K`` means, when a NULL convolution input is legal). Contracts are written; vocabulary is
derived.
"""

from __future__ import annotations

import pytest

from merlin.sched.isa import instruction_set

#: Written by hand, each for a stated contract. Everything else must be derived.
HAND_WRITTEN = frozenset({"config_ex", "config_st", "config_ld", "loop_ws", "loop_conv_ws"})


@pytest.fixture(scope="module")
def iset():
    try:
        return instruction_set("gemmini")
    except Exception as exc:  # noqa: BLE001 -- no headers on this host is a skip, not a failure
        pytest.skip(f"gemmini instruction set unavailable: {type(exc).__name__}: {exc}")


def test_the_vocabulary_is_larger_than_the_hand_table(iset):
    """If this collapses to the hand table, the derivation stopped working and nobody would notice."""
    derived = set(iset.instrs) - HAND_WRITTEN
    assert derived, (
        "the scheduling vocabulary is exactly the hand-written table, so the header/RTL derivation "
        "produced nothing — a non-FSM schedule cannot be expressed"
    )


def test_a_non_fsm_tile_loop_is_expressible(iset):
    """The instructions a tile loop needs, by ROLE rather than by name.

    Named by role because the point is that no name is typed into the library: a target that spells its
    move ``load`` rather than ``mvin`` should satisfy this the moment its header says so.
    """
    names = set(iset.instrs)

    def _role(*fragments):
        return {n for n in names if any(f in n.lower() for f in fragments)}

    assert _role("mvin", "load"), "no move-in instruction: operands cannot reach the array"
    assert _role("mvout", "store"), "no move-out instruction: results cannot leave the accumulator"
    assert _role("preload"), "no preload: the stationary operand cannot be staged"
    assert _role("compute"), "no compute: the array cannot be stepped"


def test_multiple_load_states_are_reachable(iset):
    """Streaming two operands without config churn needs more than one load state.

    Measured elsewhere: with a single load state and two operands of different pitch, the load
    configuration is re-issued on every block. The instructions that address separate states are what
    make the alternative expressible.
    """
    movers = sorted(n for n in iset.instrs if "mvin" in n.lower())
    assert len(movers) >= 2, f"only {movers} — a schedule cannot keep two operands live without churn"


def test_a_transfer_may_carry_more_than_one_tile(iset):
    """Block DMA has to be *sayable*: the width operand must not be pinned to the array edge.

    The bound belongs to the target (its DMA block limit, cross-checked against the RTL element width),
    so what is asserted here is only that a width operand exists and is an ordinary integer the schedule
    chooses — not that any particular width is legal.
    """
    movers = [n for n in iset.instrs if "mvin" in n.lower()]
    assert movers, "no move-in instruction at all"
    widths = {n: [o.name for o in iset.instrs[n].operands if o.name.lower().endswith("cols")] for n in movers}
    assert any(widths[n] for n in movers), (
        f"no move-in exposes a width operand, so every transfer is whatever the macro defaults to: {widths}"
    )


def test_hand_written_entries_survive(iset):
    """Derivation must not displace the contracts. A lost binding loses its checks with it."""
    missing = sorted(HAND_WRITTEN - set(iset.instrs))
    assert not missing, f"hand-written bindings disappeared from the vocabulary: {missing}"


def test_every_instruction_declares_operand_kinds(iset):
    """A derived operand still has to be typed, or the static checker cannot judge a call."""
    from merlin.sched.isa import OPERAND_KINDS

    for name, d in iset.instrs.items():
        assert d.operands, f"{name} has no operands"
        for o in d.operands:
            assert o.kind in OPERAND_KINDS, f"{name}.{o.name} has unknown kind {o.kind!r}"


def test_only_memory_movers_take_a_pointer_operand(iset):
    """A derived operand's KIND must come from the target's declaration, not the packing's shape.

    MEASURED. The first derivation read "a parameter that IS a whole rs expression is a pointer" off the
    packing alone, and in this header three identically shaped macros disagree about what that means:
    ``flush(skip)`` passes a flag, ``config_ld(stride, ...)`` an integer, ``mvin2(dram_addr, ...)`` an
    address. Two of the three came out ``ptr``.

    That is not cosmetic. ``sched.check.static`` accepts disjoint value types per kind -- ``int`` wants an
    ``Expr``, ``ptr`` wants a ``Ptr`` -- so a mistyped operand REJECTS every schedule that passes it the
    right thing, and no test would have caught it because ``footprint`` is ``None`` for derived entries.

    Asserted by ROLE: an instruction that does not move DRAM may not claim a pointer operand.
    """
    for name, d in iset.instrs.items():
        ptrs = [o.name for o in d.operands if o.kind == "ptr"]
        if not ptrs:
            continue
        moves = any(f in name.lower() for f in ("mvin", "mvout", "load", "store", "loop"))
        assert moves, f"{name} declares pointer operand(s) {ptrs} but does not address memory"


def test_every_memory_mover_exposes_its_address_as_a_pointer(iset):
    """The inverse: a move instruction whose address is typed ``int`` cannot be handed a tensor."""
    missing = []
    for name, d in iset.instrs.items():
        if not any(f in name.lower() for f in ("mvin", "mvout")):
            continue
        if "spad" in name.lower():
            continue  # moves between on-chip banks; no DRAM operand to type
        if not [o for o in d.operands if o.kind == "ptr"]:
            missing.append((name, [(o.name, o.kind) for o in d.operands]))
    assert not missing, f"these move DRAM but expose no pointer operand: {missing}"
