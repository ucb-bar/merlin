"""Inefficiencies visible in an emitted command buffer, tagged by the optimisation level they live at.

WHY THIS EXISTS. A corpus can only measure the levels it has capsules for. On this target that is the
tile and intra-layer rungs, with two inter-layer members and nothing at the boundary, fusion or
global rungs -- so an agent optimising against measured cycles alone is steered, silently, to ignore
whole classes of inefficiency. It is not asked about them and it cannot be scored on them, so it does
not look.

The command buffer already carries enough structure to SEE several of those inefficiencies without
running anything: it names its tensors, says which op produced each value and which consumed it,
records where a value was drained to memory, and declares what each op fused into its epilogue. That
is enough to point at a round trip through memory, a weight staged twice, or an epilogue left
unfused -- for free, before any oracle time is spent.

WHAT A FINDING IS, AND WHAT IT IS EMPHATICALLY NOT. Every finding here is a STRUCTURAL OBSERVATION:
this program does a thing that costs something. None of them is a cycle count, an estimate, or a
prediction, and nothing here may be read as one. A finding says an inefficiency is present; only a
measurement says removing it was worth anything. That asymmetry is the same one the rest of this
package runs on -- a screen may eliminate, it may never certify -- and it is why these are reported
as findings with a level and a reason rather than as a score to optimise.

A buffer this cannot read yields UNKNOWN and says why. An empty finding list means "none of the
patterns below is present", never "this program is efficient".
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["LEVELS", "COMMIT_OPCODES", "STAGE_OPCODES", "RELEASE_OPCODES", "EPILOGUE_ATTRIBUTE",
           "findings", "level_summary"]

#: The optimisation ladder, coarsest last. These are the level names the capsule corpus itself
#: declares, so a finding can be joined to the family that would measure it -- where one exists.
#:
#: This is a MIRROR of the corpus generator's own level vocabulary, and mirrors drift: it shipped
#: without ``L1_separation_floor``, so a finding could not be tagged at that rung and
#: :func:`level_summary` reported a ladder with a hole in it -- silently, because a rung that is never
#: counted is indistinguishable from a rung with nothing on it, which is exactly the failure mode this
#: module exists to argue against. The vocabulary is NOT imported from the generator that owns it
#: (``merlin/contract/capsules/generate_corpus.py``): that is a contract script and this is library
#: code, and a library reaching into a script to learn its own constants inverts the layering. It is
#: pinned instead by a test that FAILS when the two lists differ
#: (``merlin/tests/dse/test_structural_levels.py``) -- the same discipline as
#: :func:`merlin.perf.attribution.buckets_match_reference`, where a mirror is kept honest by a check
#: that can fail rather than by a comment asking the next editor to remember.
LEVELS = ("L1_tile", "L1_separation_floor", "L2_intra_layer", "L3_inter_layer", "L4_boundary",
          "L5_fusion", "L6_global")

#: ABI vocabulary. These are the emitted program's own opcode names, carried by the buffer rather
#: than assumed about any device: a backend speaking this ABI emits them whatever its ISA.
#: An opcode outside this vocabulary is not an error -- it simply matches no pattern below.
COMMIT_OPCODES = ("COMMIT",)
#: Ops that place a value into on-chip residency, and ops that give that residency back.
STAGE_OPCODES = ("RES_PACK",)
RELEASE_OPCODES = ("EVICT",)
#: The attribute an op carries listing the work it folded into its own output stage.
EPILOGUE_ATTRIBUTE = "epilogue"

UNKNOWN = "UNKNOWN"


def _commands(buffer: object) -> "list[Mapping[str, Any]] | None":
    if not isinstance(buffer, Mapping):
        return None
    rows = buffer.get("commands")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    return [r for r in rows if isinstance(r, Mapping)]


def _operands(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("operands")
    return value if isinstance(value, Mapping) else {}


def _attributes(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("attributes")
    return value if isinstance(value, Mapping) else {}


def _produced(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Value names this command writes. The ABI names a destination ``dst``."""
    dst = _operands(row).get("dst")
    return (str(dst),) if isinstance(dst, str) and dst else ()


def _consumed(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Value names this command reads: every named operand that is not the destination."""
    return tuple(str(v) for k, v in sorted(_operands(row).items())
                 if k != "dst" and isinstance(v, str) and v)


def findings(buffer: object) -> dict[str, Any]:
    """Structural inefficiencies in one emitted program, each tagged with its optimisation level."""
    rows = _commands(buffer)
    if rows is None:
        return {"status": UNKNOWN, "reason": "the command buffer declares no command list"}
    if not rows:
        return {"status": UNKNOWN, "reason": "the command buffer declares no commands"}

    found: list[dict[str, Any]] = []

    # ---- L2: a value given back to on-chip residency and then staged again -------------------
    # Staging a weight, releasing it, and staging the same weight again means the program paid the
    # residency cost twice for one value. Whether that was necessary depends on what else needed the
    # space, which this cannot see -- so it is reported, not condemned.
    staged: dict[str, int] = {}
    released: set[str] = set()
    for index, row in enumerate(rows):
        opcode = str(row.get("opcode") or "")
        if opcode in STAGE_OPCODES:
            for name in _consumed(row):
                if name in released and name in staged:
                    found.append({
                        "level": "L2_intra_layer", "kind": "residency_restaged", "value": name,
                        "at_command": index, "first_staged_at": staged[name],
                        "detail": (f"{name!r} was staged for on-chip residency, released, and "
                                   f"staged again; the program paid to place the same value twice"),
                    })
                staged.setdefault(name, index)
        if opcode in RELEASE_OPCODES:
            released.update(_consumed(row))

    # ---- L3: a value drained to memory and then read back ------------------------------------
    # A value that is committed and later consumed made a round trip the program could have avoided
    # by keeping it. This is the inter-layer inefficiency the corpus has only two members for.
    committed: dict[str, int] = {}
    for index, row in enumerate(rows):
        opcode = str(row.get("opcode") or "")
        if opcode in COMMIT_OPCODES:
            for name in _consumed(row):
                committed.setdefault(name, index)
            continue
        for name in _consumed(row):
            if name in committed:
                found.append({
                    "level": "L3_inter_layer", "kind": "memory_round_trip", "value": name,
                    "at_command": index, "committed_at": committed[name],
                    "detail": (f"{name!r} was drained to memory and then read back by a later "
                               f"command; the value made a round trip instead of staying resident"),
                })

    # ---- L5: a producer with an empty epilogue whose only consumer is the next command --------
    # An op that declares no epilogue and whose result is consumed immediately is the shape a fused
    # epilogue would collapse. Whether this backend CAN fuse that pair is not decidable here, which
    # is exactly why this is a finding and not an instruction.
    consumers: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        for name in _consumed(row):
            consumers.setdefault(name, []).append(index)
    for index, row in enumerate(rows):
        epilogue = _attributes(row).get(EPILOGUE_ATTRIBUTE)
        if not isinstance(epilogue, Sequence) or isinstance(epilogue, (str, bytes)) or epilogue:
            continue
        for name in _produced(row):
            uses = consumers.get(name) or []
            if len(uses) == 1 and uses[0] == index + 1:
                found.append({
                    "level": "L5_fusion", "kind": "unfused_single_consumer", "value": name,
                    "at_command": index, "consumed_at": uses[0],
                    "detail": (f"command {index} declares an empty {EPILOGUE_ATTRIBUTE} and its "
                               f"result {name!r} is read by exactly one command, the next one; "
                               f"that is the shape a fused epilogue would collapse"),
                })

    return {
        "status": "read", "commands": len(rows), "findings": found,
        "by_level": level_summary(found),
        "basis": ("structural patterns in the emitted command buffer; each names an inefficiency "
                  "that is PRESENT, and none of them is a cycle count, an estimate, or a claim "
                  "that removing it would be worth anything"),
        "not_covered": ("nothing here inspects the host boundary or the choice of operand encoding, "
                        "so a finding list without those levels is silence about them, not a pass"),
    }


def level_summary(found: "Sequence[Mapping[str, Any]]") -> dict[str, int]:
    """How many findings sit at each level, including the levels with none."""
    counts = {level: 0 for level in LEVELS}
    for row in found:
        level = str(row.get("level") or "")
        if level in counts:
            counts[level] += 1
    return counts
