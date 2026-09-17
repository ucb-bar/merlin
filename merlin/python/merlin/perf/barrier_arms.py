"""Two arms of one kernel that differ ONLY in how many completion barriers the compiler inserted.

The sibling scheduling family reorders a command stream (:mod:`merlin.perf.command_stream_gen`) on the
theory that issue order is the lever a hardware-interlocked, command-driven accelerator gives a
compiler.  MEASURED on this repo's command-driven target, it is not: a legal permutation of the same
multiset cost **exactly zero** cycles at two depths, with the reorder verified to reach the ELF and the
counter verified to move with work.  Both results have the same cause -- the dispatch queue tracks the
dependence itself -- and read together they say where the lever actually is: not in what order the
commands issue, but in **how many barriers the compiler inserted that the hardware did not need**.

There are two ways to obtain the pair, and this module carries both because they answer different
questions and neither subsumes the other.

BUILD THE PAIR (:func:`pair_from_emitter`).  Ask the TARGET'S OWN emitter for the same kernel twice, at
two settings of its retire knob, and then PROVE the two sources are the same program plus barriers
rather than asserting it:

* the low-barrier arm's line sequence must be a SUBSEQUENCE of the high-barrier arm's -- so every
  instruction of one appears, in order, in the other, and nothing was added, dropped or reworded;
* every line the high-barrier arm adds must be the SAME line -- one repeated statement, which is what
  makes "the lever is barrier count" true by construction rather than by reading.

This is the arm-construction path, and it is what the synchronization family's declared regime asks
for: one shape per pair, and the pair is two emissions of that ONE capsule.  ⚠️ NOTHING HERE KNOWS HOW
A BARRIER IS SPELLED.  The barrier statement is *discovered* as the line the two arms differ by, so a
target whose emitter spells its retire differently is served without an edit, and a target whose two
settings differ by something OTHER than a repeated inserted line is REFUSED with its reason rather
than measured as if the difference were a barrier.  Refusing is the point: a pair that differs by more
than the lever prices the difference and calls it the lever, which is how a neighbouring family came
to compare an ~82-cycle lever against an ~280-cycle uncancelled term.

READ A PAIR THAT ALREADY EXISTS (:func:`count_barriers`, :func:`paired_removal`).  A tuning campaign
does not emit both arms itself: the expensive arm is what the baseline compiler already produced and
the cheap arm is what the candidate produced instead.  Those two command buffers are already in hand,
so the job is to count what each one synchronizes.  Counting is structural.  A completion point is an
ABI opcode in the emitted command buffer, taken from the buffer's own declared vocabulary -- not a
regex over a listing, and not a target ISA constant: the same abstract vocabulary appears for any
backend that speaks this ABI.  A buffer that declares no completion opcode yields UNKNOWN with a
reason rather than a count of zero, because "no barriers found" and "cannot see barriers" must never
read the same.

⚠️ A ZERO-BARRIER PAIR IS A RESULT, NOT A FAILURE.  A kernel with one job has no redundant barrier to
remove, so both settings emit the identical program and ``removed`` is 0.  That member is this family's
NEGATIVE CONTROL: its measured differential must be exactly zero, and it is the only member for which
that is true by construction.

Either way the verdict is the family's own falsifier (:func:`analyze_barrier_claim`): the saving must
GROW with the number of barriers removed.  A single paired point cannot show that, so a cohort of one
is REFUSED rather than scored -- one subtraction is an anecdote, and the pathology this family exists
for is precisely that the cost is per barrier.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BarrierPair", "COMPLETION_OPCODES", "ESTABLISHED", "PER_UNIT_GROWTH_OBSERVATION",
    "REFUSED", "REFUSED_HETEROGENEOUS_INSERTION", "REFUSED_KNOB_UNSUPPORTED",
    "REFUSED_NOT_A_PURE_INSERTION", "REFUTED", "RetireArmsError", "UNKNOWN",
    "analyze_barrier_claim", "count_barriers", "paired_removal", "pair_from_emitter",
]


# --- Building the pair: two emissions of one capsule through the target's own emitter ---

#: The emitter did not accept the retire knob at all, so this target cannot express the lever.
REFUSED_KNOB_UNSUPPORTED = "knob_unsupported"
#: The high-barrier arm is not the low-barrier arm plus inserted lines: something was reworded.
REFUSED_NOT_A_PURE_INSERTION = "not_a_pure_insertion"
#: The inserted lines are not all the same statement, so "the difference" is not one repeated barrier.
REFUSED_HETEROGENEOUS_INSERTION = "heterogeneous_insertion"


class RetireArmsError(RuntimeError):
    """A pair that would not be the same work with a different barrier count. Carries its ``reason``."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class BarrierPair:
    """One kernel emitted at two retire settings, with the difference established rather than assumed."""

    #: Source at the setting that emits the fewest barriers.
    minimal: str
    #: Source at the setting that emits one barrier per unit of work.
    maximal: str
    #: The two settings, in the order (minimal, maximal) they were requested.
    settings: tuple[str, str]
    #: The line the two arms differ by, taken from the diff rather than from any target vocabulary.
    #: ``None`` when the two arms are the identical program (the negative-control member).
    barrier_statement: str | None
    #: How many barriers the lever removes. Zero is a legitimate result, not an error.
    removed: int

    def to_dict(self) -> dict:
        return {"settings": list(self.settings), "barrier_statement": self.barrier_statement,
                "removed": self.removed,
                "identical_programs": self.minimal == self.maximal}


def _inserted_lines(low: Sequence[str], high: Sequence[str]) -> list[str]:
    """The lines ``high`` has and ``low`` does not, or raise when ``low`` is not a subsequence.

    A two-pointer walk, not a diff library and not a pattern: the question is exactly "does every line
    of the short source appear, in order, in the long one", and answering it any other way would admit
    a pair whose shared lines had been reordered.
    """
    extra: list[str] = []
    i = 0
    for line in high:
        if i < len(low) and low[i] == line:
            i += 1
        else:
            extra.append(line)
    if i != len(low):
        raise RetireArmsError(
            REFUSED_NOT_A_PURE_INSERTION,
            f"the low-barrier arm's line {i + 1} ({low[i]!r}) has no match in emission order in the "
            "high-barrier arm; the two settings changed the kernel, not only its barrier count")
    return extra


def pair_from_emitter(emit: Callable[..., str], command_buffer: dict, *,
                      settings: tuple[str, str], knob: str = "retire") -> BarrierPair:
    """Emit ``command_buffer`` twice through ``emit`` and return the established barrier pair.

    ``emit`` is the target's OWN driver emitter, called as ``emit(command_buffer, **{knob: setting})``.
    The knob name is a parameter because it belongs to the emitter's signature, not to this module; a
    target whose emitter does not accept it is refused with :data:`REFUSED_KNOB_UNSUPPORTED` rather than
    silently measured at one setting twice.
    """
    if len(settings) != 2 or settings[0] == settings[1]:
        raise ValueError(f"settings must be two DISTINCT emitter settings, got {settings!r}")
    sources = []
    for setting in settings:
        try:
            sources.append(emit(command_buffer, **{knob: setting}))
        except TypeError as exc:
            raise RetireArmsError(
                REFUSED_KNOB_UNSUPPORTED,
                f"this target's emitter does not accept {knob}={setting!r} ({exc}); it cannot express "
                "the barrier lever, so no pair exists to measure") from exc
    low_src, high_src = sources
    extra = _inserted_lines(low_src.split("\n"), high_src.split("\n"))
    distinct = set(extra)
    if len(distinct) > 1:
        raise RetireArmsError(
            REFUSED_HETEROGENEOUS_INSERTION,
            f"the high-barrier arm adds {len(distinct)} distinct statements {sorted(distinct)!r}; the "
            "difference between the arms is then not one repeated barrier and pricing it as one would "
            "attribute the rest of the difference to the lever")
    return BarrierPair(minimal=low_src, maximal=high_src, settings=(settings[0], settings[1]),
                       barrier_statement=(extra[0] if extra else None), removed=len(extra))


# --- Reading a pair the measurement path already built, and deciding the claim ---

#: Opcodes that force the issuing host to observe that prior work retired.  Named at the ABI level,
#: which is target-independent: a backend that speaks this ABI emits these regardless of its ISA.
COMPLETION_OPCODES = ("COMMIT", "FENCE", "BARRIER")

#: The falsifier ``observation`` a family declares when its claim is PER-UNIT GROWTH rather than a
#: direction.  A direction claim ("this arm is cheaper") and a growth claim ("the saving grows with
#: the count removed") are different assertions and only the second one indicts a PER-UNIT cost, so a
#: family declaring this must have :func:`analyze_barrier_claim` run over its cohort -- deciding it
#: with a direction test alone would report a verdict on a claim nobody evaluated.  Compared as data
#: against the declaration, never matched as a pattern.
PER_UNIT_GROWTH_OBSERVATION = "paired_cycle_saving_by_removed_barrier_count"

ESTABLISHED = "ESTABLISHED"
REFUTED = "REFUTED"
REFUSED = "REFUSED"
UNKNOWN = "UNKNOWN"


def _commands(buffer: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = buffer.get("commands")
    return [r for r in rows if isinstance(r, Mapping)] if isinstance(rows, Sequence) else []


def count_barriers(buffer: object) -> dict[str, Any]:
    """Count completion points in one emitted command buffer, or say why it cannot be counted."""
    if not isinstance(buffer, Mapping):
        return {"status": UNKNOWN, "reason": "the command buffer is not a mapping"}
    rows = _commands(buffer)
    if not rows:
        return {"status": UNKNOWN, "reason": "the command buffer declares no commands"}
    opcodes = [str(r.get("opcode") or "") for r in rows]
    if any(not code for code in opcodes):
        return {"status": UNKNOWN, "reason": "a command declares no opcode, so the stream is unreadable"}
    present = sorted({code for code in opcodes if code in COMPLETION_OPCODES})
    if not present:
        # Distinguishing this from zero is the whole point: a program with no recognised completion
        # opcode has not been shown to synchronize rarely, only to be unreadable by this counter.
        return {"status": UNKNOWN,
                "reason": ("no completion opcode from the ABI vocabulary appears in this program, "
                           "so its synchronization cannot be counted"),
                "vocabulary": list(COMPLETION_OPCODES), "observed_opcodes": sorted(set(opcodes))}
    return {"status": "counted", "barriers": sum(1 for c in opcodes if c in COMPLETION_OPCODES),
            "commands": len(rows), "completion_opcodes": present}


def paired_removal(baseline_buffer: object, candidate_buffer: object) -> dict[str, Any]:
    """How many completion points the candidate removed relative to the baseline."""
    base, cand = count_barriers(baseline_buffer), count_barriers(candidate_buffer)
    if base["status"] != "counted" or cand["status"] != "counted":
        return {"status": UNKNOWN,
                "reason": f"baseline: {base.get('reason', 'ok')}; candidate: {cand.get('reason', 'ok')}"}
    return {"status": "counted", "baseline_barriers": base["barriers"],
            "candidate_barriers": cand["barriers"],
            "removed": base["barriers"] - cand["barriers"]}


def analyze_barrier_claim(points: object) -> dict[str, Any]:
    """Decide whether cycles saved grow with barriers removed.

    ``points`` is a sequence of already-measured pairs, each carrying ``removed`` (from
    :func:`paired_removal`) and ``cycles_saved``.  The claim is directional and per-unit: more
    barriers removed must buy more cycles.  A cohort that never varies the removed count cannot
    show that and is REFUSED, not scored.
    """
    if not isinstance(points, Sequence) or not points:
        return {"verdict": REFUSED, "reason": "no measured pairs were supplied"}
    usable: list[tuple[float, float]] = []
    for row in points:
        if not isinstance(row, Mapping):
            return {"verdict": REFUSED, "reason": "a measured pair is not a mapping"}
        removed, saved = row.get("removed"), row.get("cycles_saved")
        if not isinstance(removed, int) or isinstance(removed, bool):
            return {"verdict": REFUSED, "reason": "a pair does not declare how many barriers it removed"}
        if not isinstance(saved, (int, float)) or isinstance(saved, bool):
            return {"verdict": REFUSED, "reason": "a pair carries no measured cycle saving"}
        usable.append((float(removed), float(saved)))
    if len({r for r, _ in usable}) < 2:
        return {"verdict": REFUSED,
                "reason": ("every pair removed the same number of barriers, so the cohort cannot "
                           "show a saving that GROWS with the removed count -- which is the claim")}
    n = len(usable)
    sx = sum(r for r, _ in usable); sy = sum(s for _, s in usable)
    sxx = sum(r * r for r, _ in usable); sxy = sum(r * s for r, s in usable)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        return {"verdict": REFUSED, "reason": "the removed-barrier count does not vary"}
    per_barrier = (n * sxy - sx * sy) / denominator
    measured = {"cycles_per_removed_barrier": per_barrier, "n_pairs": n,
                "distinct_removed_counts": sorted({r for r, _ in usable})}
    if per_barrier <= 0:
        return {"verdict": REFUTED, "measured": measured,
                "reason": (f"removing barriers bought {per_barrier:.3g} cycles each, so the "
                           f"synchronization this family blames is not costing what it claims")}
    return {"verdict": ESTABLISHED, "measured": measured,
            "reason": f"each removed barrier is worth {per_barrier:.1f} cycles across {n} pairs"}
