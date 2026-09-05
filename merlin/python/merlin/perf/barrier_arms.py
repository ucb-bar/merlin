"""Count synchronization points in an emitted program, and decide a barrier-removal claim.

The synchronization family declares this module as its emitter and it did not exist, so the family's
second arm could never be built and ten capsules measured one arm against nothing.  What follows is
the part that is actually derivable, and it is deliberately narrower than "emit two arms":

A barrier arm is NOT something a corpus file can synthesize.  The expensive arm is what the baseline
compiler already emits -- a completion point after every unit of work -- and the cheap arm is what an
optimizing compiler produces instead.  That is precisely the baseline/candidate pair the measurement
path already builds, so the honest job here is to READ the two emitted programs, count what each one
synchronizes, and decide whether removing those points bought the cycles the claim predicts.

Counting is structural.  A completion point is an ABI opcode in the emitted command buffer, taken
from the buffer's own declared vocabulary -- not a regex over a listing, and not a target ISA
constant: the same abstract vocabulary appears for any backend that speaks this ABI.  A buffer that
declares no completion opcode yields UNKNOWN with a reason rather than a count of zero, because
"no barriers found" and "cannot see barriers" must never read the same.

The claim this decides is the family's own falsifier: the saving must GROW with the number of
barriers removed.  A single paired point cannot show that, so a cohort of one is REFUSED rather than
scored -- one subtraction is an anecdote, and the pathology this family exists for is precisely that
the cost is per barrier.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
