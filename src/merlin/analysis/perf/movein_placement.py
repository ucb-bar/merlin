"""Was the operand movement placed where compute can hide it, or hoisted in front of the nest?

THE LEVER. A tiled contraction can stage its operands two ways. It can HOIST -- issue every transfer
the nest will need as one burst standing in front of the first compute -- or it can place each
transfer just before the position that first reads it, interleaved with the computes. Both emit the
same transfers to the same addresses; only the PLACEMENT differs. Measured on pinned elaborated RTL,
bit-exact against the vendor kernel in the same binary: -22.3% on 64x512x1024 and -12.7% on
64x256x1024 from moving the staging off the front of the nest and into it.

AND IT DOES NOT PAY EVERYWHERE, which is the fact that shapes this family and is the reason it is
NOT a demand made of every member. The same change measured -3.2% and -0.6% on two other shapes. The
two shapes it paid on are LOAD-critical -- the movement side is what the runtime is waiting for --
and the two it did nothing on are MESH-critical, where the array is. The discriminant that separates
those four points is the reload multiplicity (see :func:`merlin.analysis.perf.schedule_proxy._regime`), and it
is not a coincidence that it is also the OTHER lever's lever arm: on a mesh-critical shape the order
that keeps the array's operand resident stages a whole run of moving-operand windows together, so
demanding just-in-time placement there would fight the demand that actually pays. This family
therefore demands the property of load-critical members ONLY, waives it on mesh-critical ones, and
RECORDS the waiver -- see :mod:`merlin.analysis.perf.movein_claim`, which refuses a cohort carrying no waived
member, because a family that demanded this everywhere and a family that conditioned it on the
regime would otherwise be indistinguishable.

THE PROPERTY, and why this narrow form of it is the part that is PROVABLE:

    Movement issued while NO COMPUTE HAS YET ISSUED cannot be overlapped with compute. Not "is
    unlikely to be"; cannot -- there is no compute in flight to hide it behind, whatever the depth of
    the machine's load queue. Every transfer in that prefix beyond the ones the first compute
    position actually reads is work the runtime waits through before any useful cycle happens.

    Hence the demand: **the transfers issued before the first compute must not exceed what a
    declared number of compute positions consume**, that per-position need being read from the
    candidate's own stream (the producers of the operands the first compute and its stager read).

    The admitted depth is a DECLARED number in the family's frozen acceptance, not a constant here,
    and it is not a style preference: at one position there is no prefetch at all, so the contract
    declares the depth at which movement and compute can first be concurrent. A contract that
    declares none gets a REFUSAL, never a default.

WHAT IS DELIBERATELY NOT DEMANDED, and the fact that is missing. The same argument does not extend
to a burst in the MIDDLE of the nest: there, computes are outstanding, and whether the movement
behind them is absorbed depends on how deep the load queue is. That depth is a real device property
-- the target this was measured on declares a load reservation window, and a burst deeper than it
fills the load side -- but it is NOT published in the extracted facts this repo reads. So it is
recorded as UNKNOWN and surfaced (``mid_stream_window`` in every verdict), the peak mid-stream run is
REPORTED beside the verdict as the lever arm, and nothing is failed on it. Guessing the window would
decide a performance verdict on a number nobody measured.

WHAT PASSING DOES NOT PROVE. Passing shows the nest does not stand a burst in front of itself. It is
not a speedup claim, it says nothing about the movement VOLUME (a nest that moves each tile twice
passes as long as it places both well), and -- because the provable part is the prefix -- a nest that
issues one compute and then bursts the rest passes. That last is reported, not hidden: the peak
mid-stream run travels with the verdict so a reader can see it.

FAIL CLOSED EVERYWHERE. The staging, compute and movement vocabularies come from the emitted ABI's
own def-use model, through the same resolution the stationary-residency family uses. A trace with no
compute, a first compute whose operands do not resolve, or a contract declaring no admitted depth
yields ``REFUSED`` with the reason -- never a pass, and never a substituted default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.analysis.perf import stationary_residency as _SR

__all__ = [
    "PASS",
    "FAIL",
    "REFUSED",
    "UNKNOWN_WINDOW",
    "exposure_findings",
    "placement_verdict",
]

PASS = "PASS"
FAIL = "FAIL"
REFUSED = "REFUSED"

#: What the target does not publish: how deep a burst its load path can absorb behind outstanding
#: compute. Carried on every verdict so the missing fact travels with the number instead of living
#: only in this docstring, and so a later extraction that DOES publish it is an obvious upgrade
#: rather than a rediscovery.
UNKNOWN_WINDOW = "UNKNOWN"


def _positive(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def placement_verdict(
    trace: Any,
    *,
    admitted_positions: object,
    vocabulary: _SR.Vocabulary | None = None,
    target: object = None,
) -> dict[str, Any]:
    """Decide the move-in placement property for one emitted trace.

    ``admitted_positions`` is how many compute positions' worth of staging may stand in front of the
    nest, read from the family's frozen acceptance by the caller. ``target`` names whose retain
    sentinel a staging command may carry; without one, a retaining position reads as an operand with
    no producer and the verdict refuses. Returns ``{"verdict", "reason", ...}``; ``FAIL`` reports the
    exposed prefix and the excess, ``REFUSED`` names what could not be derived. A trace with no
    compute at all is REFUSED, not passed.
    """
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    instructions = trace.get("instructions")
    if not isinstance(instructions, Sequence) or isinstance(instructions, str):
        raise TypeError("trace instructions must be a sequence")

    positions = _positive(admitted_positions)
    if positions is None:
        return {
            "verdict": REFUSED,
            "reason": (
                "the family's frozen acceptance declares no admitted staging depth, so how much "
                "movement may stand in front of the nest is UNDECLARED; substituting one would make "
                "this demand a number nobody reviewed"
            ),
        }

    vocab = vocabulary if vocabulary is not None else _SR.stationary_vocabulary(target)
    if vocab is None:
        return {
            "verdict": REFUSED,
            "reason": (
                "the emitted ABI declares no staging relation, so which commands move operands in "
                "and which consume them cannot be told apart and no placement can be read"
            ),
        }
    addressing, why = _SR.resolve_addressing(instructions, vocab)
    if addressing is None:
        return {"verdict": REFUSED, "reason": str(why), "vocabulary": vocab.to_dict()}

    first_compute: int | None = None
    for index, instruction in enumerate(instructions):
        if isinstance(instruction, Mapping) and str(instruction.get("class") or "") in vocab.compute_classes:
            first_compute = index
            break
    if first_compute is None:
        return {
            "verdict": REFUSED,
            "reason": (
                "the trace issues no compute at all, so there is no point at which movement stops "
                "being exposed and nothing this demand could have observed"
            ),
        }

    # slot -> the movement that filled it, so the operands the first compute reads can be traced
    # back to the transfers that are COMPULSORY before it.
    producer_of: dict[int, int] = {}
    prefix: list[int] = []
    runs: list[int] = []
    current_run = 0
    for index, instruction in enumerate(instructions):
        if not isinstance(instruction, Mapping):
            continue
        cls = str(instruction.get("class") or "")
        decoded = instruction.get("decoded")
        decoded = decoded if isinstance(decoded, Mapping) else {}
        if cls in vocab.stage_classes or cls in vocab.compute_classes:
            if cls in vocab.compute_classes and current_run:
                runs.append(current_run)
                current_run = 0
            continue
        base = decoded.get(addressing.define_field)
        if not isinstance(base, int) or isinstance(base, bool):
            continue
        width = decoded.get(vocab.width_field)
        span = width if isinstance(width, int) and not isinstance(width, bool) and width > 0 else 1
        for slot in range(base, base + span):
            producer_of[slot] = index
        current_run += 1
        if index < first_compute:
            prefix.append(index)
    if current_run:
        runs.append(current_run)

    # What the first position genuinely needs: the transfers that produced the operands the first
    # compute reads, and those its stager presents. Read from the stream, never from the shape.
    needed: set[int] = set()
    unresolved: list[str] = []
    # The first compute and the staging command that presents ITS operand. Found by scanning back for
    # the nearest stager rather than assuming the instruction immediately before it: a schedule may
    # place a transfer between the two, and reading that transfer as "no stager" would drop the
    # stationary operand from the compulsory set and tighten the bound onto a schedule that did
    # nothing wrong.
    window = [first_compute]
    for index in range(first_compute - 1, -1, -1):
        instruction = instructions[index]
        if not isinstance(instruction, Mapping):
            continue
        if str(instruction.get("class") or "") in vocab.stage_classes:
            window.append(index)
            break
    for index in window:
        instruction = instructions[index]
        if not isinstance(instruction, Mapping):
            continue
        decoded = instruction.get("decoded")
        decoded = decoded if isinstance(decoded, Mapping) else {}
        for field in addressing.consumed_fields:
            address = decoded.get(field)
            if not isinstance(address, int) or isinstance(address, bool):
                continue
            if vocab.sentinel is not None and address == vocab.sentinel:
                continue
            producer = producer_of.get(address)
            if producer is None:
                unresolved.append(f"#{index}.{field}={address}")
            else:
                needed.add(producer)
    if unresolved:
        return {
            "verdict": REFUSED,
            "reason": (
                "the first compute position reads operands with no observed producer in this trace "
                f"({', '.join(unresolved[:4])}), so what it COMPULSORILY needs staged before it is "
                "UNKNOWN and no transfer ahead of it has been shown to be exposed"
            ),
            "exposed_prefix": len(prefix),
        }
    if not needed:
        return {
            "verdict": REFUSED,
            "reason": (
                "the first compute position reads nothing this ABI resolves to a staged slot, so "
                "the compulsory prefix is UNKNOWN and the exposed one cannot be separated from it"
            ),
            "exposed_prefix": len(prefix),
        }

    per_position = len(needed)
    admitted = positions * per_position
    row: dict[str, Any] = {
        "exposed_prefix": len(prefix),
        "per_position_transfers": per_position,
        "admitted_positions": positions,
        "admitted_prefix": admitted,
        "first_compute_index": first_compute,
        "peak_mid_stream_run": max(runs) if runs else 0,
        "mid_stream_window": UNKNOWN_WINDOW,
        "staged_field": addressing.staged_field,
    }
    if len(prefix) > admitted:
        return {
            **row,
            "verdict": FAIL,
            "reason": (
                f"{len(prefix)} transfers are issued before the first compute, against {admitted} "
                f"admitted ({positions} position(s) x {per_position} transfer(s) the first position "
                f"reads): {len(prefix) - admitted} of them are staged while no compute has issued at "
                "all and so cannot be overlapped with any. This is the signature of the nest's whole "
                "operand slab being hoisted in front of it rather than placed beside the positions "
                "that read it. The peak mid-stream run is "
                f"{row['peak_mid_stream_run']}, reported and NOT demanded: whether a burst behind "
                "outstanding compute is absorbed depends on the load window, which this target does "
                "not publish"
            ),
            "excess": len(prefix) - admitted,
        }
    return {
        **row,
        "verdict": PASS,
        "excess": 0,
        "reason": (
            f"{len(prefix)} transfer(s) stand before the first compute, within the {admitted} "
            f"admitted: the nest does not hoist its operand slab in front of itself. This says the "
            "exposed staging is absent; it is not a claim about the movement volume, nor about the "
            f"mid-stream placement (peak run {row['peak_mid_stream_run']}, not demanded because the "
            "load window is UNKNOWN)"
        ),
    }


def exposure_findings(
    trace: Any, *, admitted_positions: object, vocabulary: _SR.Vocabulary | None = None, target: object = None
) -> list[str]:
    """The verdict as advisory diagnostic lines, for callers that collect findings rather than verdicts.

    A ``PASS`` yields no line. Both ``FAIL`` and ``REFUSED`` yield one, because a refusal that
    produced silence would read exactly like a well-placed nest.
    """
    verdict = placement_verdict(trace, admitted_positions=admitted_positions, vocabulary=vocabulary, target=target)
    if verdict["verdict"] == PASS:
        return []
    return [f"move-in placement ({verdict['verdict'].lower()}): {verdict['reason']}"]
