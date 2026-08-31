"""Reorder a command stream into an A/B pair that is the SAME WORK in a different order.

The scheduling family -- issue a transfer before a wait it does not depend on -- is the only
performance lever a hardware-interlocked, command-driven accelerator has. Its trait gate passes once
the target's elaborated FIRRTL evidences a per-engine completion channel, and it was then blocked on an
emitter declared against the wrong archetype: ``perf.workload_gen`` allocates from a scalar register
file, encodes backward branches, and emits assembly, while a command-driven target declares no
self-hosted program at all. Adding a knob there cannot produce a command stream.

This is the sibling that can. It consumes the ordered command list the frozen interface grammar already
carries -- a real capsule parses to ``RES_PACK, MATMUL_RESIDENT, COMMIT, MATMUL_RESIDENT, COMMIT,
EVICT`` -- so the deliverable is a PERMUTATION of an existing list. That is what makes the two members
the same work by construction rather than by assertion, which matters because η is a ratio: a candidate
that quietly does less work raises it without having scheduled anything better.

**Dependence is read off the operand KEY NAMES, not off an opcode table.** The grammar spells the
written operand ``dst`` and every other role by what it reads (``src``, ``lhs``, ``rhs``, ``q``, ``k``,
``p``, ``v``, ``gamma``, ``ifm``, ``weight``, ``a``, ``w``, ``handle``). So "what does this command
read and write" follows from the keys the parser already produced, and a target that gains an op class
is served without an edit here. An opcode table would be a second authority that drifts.

**What this decides is SEMANTIC, not a hazard question.** The hardware resolves hazards in a
reservation station -- that is precisely why every reordering is correct and why a capsule gated on
bit-exactness learns nothing here. The question this module answers is narrower and sharper: would the
two orders compute the same VALUES. A commit that writes a tensor a later matmul reads may not be moved
after it, whatever the hardware would tolerate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: The operand key that names what a command WRITES. Everything else a command names, it reads.
#: Taken from the grammar's own spelling rather than from a per-opcode table.
_WRITE_KEY = "dst"

#: A reason a hoist was refused. Kept as data so a refusal can be reported rather than inferred from
#: an empty result -- an emitter that returns nothing and an emitter that found nothing legal to do
#: are different states, and only one of them is a coverage gap.
REFUSED_NO_CANDIDATE = "no_candidate"
REFUSED_DEPENDENT = "dependent"


def _is_lifetime_op(cmd: dict) -> bool:
    """True for a command that names operands and produces nothing.

    ⚠️ THIS DISTINCTION IS LOAD-BEARING, and omitting it produced a program-destroying permutation the
    first time this module ran. A command with no ``dst`` has no value to contribute; its only
    observable effect is on the STATE of the operands it names. The grammar's eviction command is
    exactly that -- it releases a resident handle -- and treating its handle as an ordinary read let
    the hoist search move it above every compute that used the resident, because two reads do not
    conflict. The resulting candidate evicted the weight before the matmuls that read it and was
    reported as identical work.

    So: no ``dst`` means the named operands are being consumed, and the command may not cross any use
    of them. Derived from the absence of the write key rather than from an opcode name, so a target
    whose grammar gains another lifetime command is covered without an edit here.
    """
    ops = (cmd or {}).get("operands") or {}
    return bool(ops) and not ops.get(_WRITE_KEY)


def reads(cmd: dict) -> frozenset:
    """Operand names this command consumes without ending their lifetime."""
    if _is_lifetime_op(cmd):
        return frozenset()
    ops = (cmd or {}).get("operands") or {}
    return frozenset(str(v) for k, v in ops.items() if k != _WRITE_KEY and v)


def writes(cmd: dict) -> frozenset:
    """Operand names whose STATE this command changes -- what it produces, or what it consumes.

    A lifetime command's operands count here, so any use of those names conflicts with it in both
    directions and the command cannot be permuted across them.
    """
    ops = (cmd or {}).get("operands") or {}
    if _is_lifetime_op(cmd):
        return frozenset(str(v) for v in ops.values() if v)
    v = ops.get(_WRITE_KEY)
    return frozenset({str(v)} if v else ())


def depends(earlier: dict, later: dict) -> str | None:
    """Why ``later`` may not move above ``earlier``, or ``None`` when it may.

    All three classical dependences are refused, and the reason is returned rather than a bare bool:
    read-after-write is the obvious one, but write-after-read and write-after-write change which value
    a third command later observes, so permuting across them changes the program's meaning even on a
    machine that would happily execute it.
    """
    if writes(earlier) & reads(later):
        return "read-after-write"
    if reads(earlier) & writes(later):
        return "write-after-read"
    if writes(earlier) & writes(later):
        return "write-after-write"
    return None


def work_fingerprint(commands) -> dict:
    """What must be IDENTICAL between the two members, in a form the falsifier can compare.

    The opcode multiset and the operand-name multiset, both order-insensitive. If either differs the
    two runs are not the same work and η -- a ratio -- is not comparable between them; the falsifier
    already refuses to judge when a fingerprint differs, and this is what supplies it.
    """
    codes: dict = {}
    names: dict = {}
    for c in commands or ():
        code = str((c or {}).get("opcode") or "")
        codes[code] = codes.get(code, 0) + 1
        for n in sorted(reads(c) | writes(c)):
            names[n] = names.get(n, 0) + 1
    return {"opcodes": dict(sorted(codes.items())), "operands": dict(sorted(names.items())),
            "n_commands": len(list(commands or ()))}


def hoist(commands, index: int, above: int):
    """Move ``commands[index]`` to position ``above``, or refuse with the dependence that blocks it.

    Returns ``(stream, crossed, reason)``. ``stream`` is ``None`` on refusal; ``crossed`` names the
    commands the move would pass, so a report can say what was in the way.
    """
    seq = list(commands or ())
    if not (0 <= above < index < len(seq)):
        return None, (), REFUSED_NO_CANDIDATE
    mover = seq[index]
    crossed = []
    for i in range(above, index):
        why = depends(seq[i], mover)
        if why is not None:
            return None, tuple(crossed), why
        crossed.append(str((seq[i] or {}).get("opcode") or ""))
    out = seq[:above] + [mover] + seq[above:index] + seq[index + 1:]
    return out, tuple(crossed), None


@dataclass
class ReorderPair:
    """An A/B pair on identical work, plus the facts that justify calling it that."""

    baseline: list = field(default_factory=list)
    candidate: list = field(default_factory=list)
    moved_opcode: str = ""
    moved_from: int = -1
    moved_to: int = -1
    crossed: tuple = ()
    fingerprint: dict = field(default_factory=dict)
    identical_work: bool = False
    refusal: str = ""

    def to_dict(self) -> dict:
        return {"moved_opcode": self.moved_opcode, "moved_from": self.moved_from,
                "moved_to": self.moved_to, "crossed": list(self.crossed),
                "identical_work": self.identical_work, "fingerprint": self.fingerprint,
                "refusal": self.refusal,
                "n_baseline": len(self.baseline), "n_candidate": len(self.candidate)}


def reorder_pair(commands, *, movable=None) -> ReorderPair:
    """The earliest legal hoist in ``commands``, as a pair.

    ``movable`` restricts which opcodes may be hoisted; the default is every command that produces a
    resident operand or moves data, because those are the transfers a schedule can overlap with
    compute. Passing it explicitly is how a caller aims the lever at a particular engine.

    Searches for the LONGEST legal move (hoist as far up as the dependences allow), because a move of
    one position may not change occupancy at all, and a family that cannot show a difference proves
    nothing.
    """
    seq = list(commands or ())
    best = ReorderPair(baseline=seq, candidate=[], refusal=REFUSED_NO_CANDIDATE)
    span = -1
    for index in range(len(seq) - 1, 0, -1):
        code = str((seq[index] or {}).get("opcode") or "")
        if movable is not None and code not in movable:
            continue
        # Try the furthest position first; the first that succeeds is the longest legal move for this
        # command, and a longer move for an earlier command still wins on span below.
        for above in range(0, index):
            out, crossed, why = hoist(seq, index, above)
            if out is None:
                continue
            if (index - above) > span:
                span = index - above
                fp_a, fp_b = work_fingerprint(seq), work_fingerprint(out)
                best = ReorderPair(baseline=seq, candidate=out, moved_opcode=code,
                                   moved_from=index, moved_to=above, crossed=crossed,
                                   fingerprint=fp_a, identical_work=(fp_a == fp_b), refusal="")
            break
    return best


def negative_control(commands) -> ReorderPair:
    """A pair whose hoist is IMPOSSIBLE, and which must therefore show no rise.

    Without this the family cannot attribute a rise to hoisting. A move that crosses a genuine
    dependence is refused here rather than emitted, so the control is "the schedule that could not be
    improved" -- and a run in which it nonetheless shows a rise has measured something other than the
    lever.
    """
    seq = list(commands or ())
    for index in range(1, len(seq)):
        for above in range(0, index):
            out, crossed, why = hoist(seq, index, above)
            if out is None and why not in (None, REFUSED_NO_CANDIDATE):
                return ReorderPair(baseline=seq, candidate=[],
                                   moved_opcode=str((seq[index] or {}).get("opcode") or ""),
                                   moved_from=index, moved_to=above, crossed=crossed,
                                   fingerprint=work_fingerprint(seq),
                                   identical_work=False, refusal=why)
    return ReorderPair(baseline=seq, candidate=[], refusal=REFUSED_NO_CANDIDATE)


def pair_from_interface(mlir_text: str, *, movable=None) -> dict:
    """Parse a capsule's interface MLIR and return the pair plus its negative control.

    Routed through the grammar's own canonical parser, which now fails closed on an op it does not
    define -- so a stream this cannot fully read raises there rather than yielding a permutation of a
    partial program.
    """
    from merlin.targetgen.contract import interface_emit as IE

    cb = IE.parse_interface_mlir(mlir_text)
    cmds = list(cb.get("commands") or ())
    pair = reorder_pair(cmds, movable=movable)
    ctl = negative_control(cmds)
    return {"target": cb.get("target", ""), "n_commands": len(cmds),
            "pair": pair.to_dict(), "negative_control": ctl.to_dict(),
            "pass_condition": ("eta must RISE on the candidate; bit-exactness proves nothing here "
                              "because the hardware resolves hazards, so every reordering is correct"),
            "baseline": pair.baseline, "candidate": pair.candidate}
