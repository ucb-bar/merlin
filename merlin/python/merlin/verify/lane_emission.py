"""Did the emitted program actually CONTAIN the work its lane plan placed?

THE DEFECT, MEASURED. SmolVLA's flow_denoise lane plan places 32 ``batch_matmul`` contractions on
the scalar lane -- 24 f32 and 8 bf16 -- each with a recorded lane and a recorded reason ("the mesh
contracts i8 operands; this region is bf16"). The interface MLIR mentions ``bf16`` **21,971 times**.
The lowered MLIR mentions it **zero** times, and the emitted LLVM has no ``bfloat`` at all, while f32
arithmetic survives in force (2,250 ``fmul float``). The bf16 host-lane arithmetic was not lowered,
and nothing refused.

WHAT THAT COST. The dropped contractions were the cross-attention consuming the prefix KV-cache, so
that input became dead and was eliminated: the export reads it in 33 ``tensor.extract_slice`` ops and
the emitted kernel references it **zero** times. The device then produced a denoise update depending
on nothing that changes -- an exactly constant per-step delta of 0.092030, a trajectory advancing 25x
too little, and a step-0 error 21.5x the capture's own quantization spread. Diagnosing it took a
32-minute whole-model simulation and a five-step forensic chain. **The plan's own bookkeeping said
the work was placed**, so every report downstream described a program that was never emitted.

THE FAILURE IS THE SILENCE, NOT THE GAP. A capability the compiler lacks is an ordinary state; this
repo's rule is to record it and fail closed. A performance campaign over a program that silently
omits work will happily measure it as faster, because it is -- it does less. That is the same shape
the tree already tracks many times over: a step that could not fail reporting success.

WHY DTYPE PRESENCE IS THE CHECK. A region's identity is not recoverable from lowered IR -- names are
gone by then. Its DTYPE is not: a placed region declares one, and if the lowered IR contains no
occurrence of that dtype at all then every region declaring it was dropped. That is sound in the
direction that matters (a present dtype never raises) and it catches wholesale elimination, which is
how a missing capability actually presents. It does not catch one region of a surviving dtype going
missing, and :func:`assess` says so rather than implying completeness.

NOTHING HERE NAMES A TARGET. The dtype vocabulary is the command buffer's own and the spelling tables
below are properties of MLIR and LLVM -- languages, not targets. A caller passes the text it emitted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["LaneEmissionError", "PlacedDtype", "Assessment", "assess", "placed_dtypes",
           "MLIR_DTYPE_TOKENS", "LLVM_DTYPE_TOKENS", "STATUSES", "REFUSING_STATUSES"]


class LaneEmissionError(ValueError):
    """The lane plan cannot be read well enough to check it, so nothing is asserted."""


#: How each dtype a lane plan may declare is spelled in MLIR. A property of the LANGUAGE, not of any
#: target: MLIR prints ``bf16`` and LLVM prints ``bfloat`` for the same type.
#:
#: THE KEYS ARE THE VOCABULARY THE PLANS ACTUALLY USE, censused rather than assumed. Across the four
#: emissions in this tree the plans declare ``f32`` (7,221), ``bf16`` (2,874), ``i8`` (968), ``i64``
#: (74), ``bool`` (72) and ``float64`` (18) -- so the placement records mix ABI spellings with torch
#: ones, and a table carrying only the ABI half reports every ``bool`` region as unplaceable.
MLIR_DTYPE_TOKENS: Mapping[str, tuple[str, ...]] = {
    "bf16": ("bf16",), "f16": ("f16",), "f32": ("f32",), "f64": ("f64",),
    "i1": ("i1",), "i8": ("i8",), "i16": ("i16",), "i32": ("i32",), "i64": ("i64",),
    # torch spellings the lane plans also emit
    "bool": ("i1",), "float64": ("f64",), "float32": ("f32",), "float16": ("f16",),
    "bfloat16": ("bf16",), "int8": ("i8",), "int32": ("i32",), "int64": ("i64",),
}

#: How each dtype a lane plan may declare is spelled in LLVM IR.
LLVM_DTYPE_TOKENS: Mapping[str, tuple[str, ...]] = {
    "bf16": ("bfloat",), "f16": ("half",), "f32": ("float",), "f64": ("double",),
    "i1": ("i1",), "i8": ("i8",), "i16": ("i16",), "i32": ("i32",), "i64": ("i64",),
    "bool": ("i1",), "float64": ("double",), "float32": ("float",), "float16": ("half",),
    "bfloat16": ("bfloat",), "int8": ("i8",), "int32": ("i32",), "int64": ("i64",),
}

STATUSES: tuple[str, ...] = (
    "emitted",       # every dtype the plan placed appears in the emitted text
    "dropped",       # a placed dtype appears NOWHERE: every region declaring it was omitted
    "unknown",       # the plan or the spelling table cannot place a declared dtype; refuse
    "not_applicable",  # the plan places no region, so there is nothing to check
)

#: Statuses a caller must treat as fatal. ``unknown`` is here deliberately: a dtype nobody can spell
#: is the state in which a dropped region is invisible.
REFUSING_STATUSES: frozenset[str] = frozenset({"dropped", "unknown"})


@dataclass(frozen=True)
class PlacedDtype:
    """One dtype the lane plan placed, and how many regions declared it."""

    dtype: str
    regions: int
    lanes: tuple[str, ...]
    #: A few region names, so a refusal can name what went missing rather than only a count.
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "regions": self.regions, "lanes": list(self.lanes),
                "examples": list(self.examples)}


@dataclass
class Assessment:
    """The verdict, the dtypes it checked, and what it is NOT able to conclude."""

    status: str
    detail: str = ""
    placed: list[PlacedDtype] = field(default_factory=list)
    dropped: list[PlacedDtype] = field(default_factory=list)
    unplaceable: list[str] = field(default_factory=list)
    #: Placements carrying no dtype. Counted, because they are outside what this check can decide
    #: and a silent exclusion would let a dtype vanish behind a blank field.
    undeclared_placements: int = 0

    @property
    def refusing(self) -> bool:
        return self.status in REFUSING_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_lane_emission_v1", "status": self.status, "detail": self.detail,
                "refusing": self.refusing,
                "placed": [p.to_dict() for p in self.placed],
                "dropped": [p.to_dict() for p in self.dropped],
                "unplaceable_dtypes": list(self.unplaceable),
                "undeclared_placements": self.undeclared_placements,
                "checks": "whether each PLACED dtype appears in the emitted text at all",
                "does_not_check": ("whether an individual region of a SURVIVING dtype was dropped; "
                                   "region identity is not recoverable from lowered IR")}


def placed_dtypes(command_buffer: Mapping[str, Any]) -> tuple[PlacedDtype, ...]:
    """The dtype census of the lane plan, read from ``params.lane_placement``.

    Each entry declares a region, a lane and a dtype. A plan with no placement yields ``()`` and the
    caller gets ``not_applicable`` -- distinct from a plan that places nothing.
    """
    params = command_buffer.get("params")
    if not isinstance(params, Mapping):
        return ()
    rows = params.get("lane_placement")
    if rows is None:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise LaneEmissionError("params.lane_placement is not a list of placement records")
    counts: dict[str, int] = {}
    lanes: dict[str, set[str]] = {}
    examples: dict[str, list[str]] = {}
    undeclared = 0
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise LaneEmissionError(f"lane_placement[{position}] is not a mapping")
        dtype = row.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            # A placement with no dtype cannot be checked. It is COUNTED and reported rather than
            # either refused (which would make the check unusable -- lstmnetvit has 44 of them) or
            # silently skipped (which would let a whole dtype go missing behind a blank field).
            undeclared += 1
            continue
        counts[dtype] = counts.get(dtype, 0) + 1
        lanes.setdefault(dtype, set()).add(str(row.get("lane") or ""))
        bucket = examples.setdefault(dtype, [])
        if len(bucket) < 4 and isinstance(row.get("region"), str):
            bucket.append(row["region"])
    placed = tuple(PlacedDtype(dtype=d, regions=counts[d], lanes=tuple(sorted(lanes[d])),
                               examples=tuple(examples.get(d, ())))
                   for d in sorted(counts))
    if undeclared and not placed:
        raise LaneEmissionError(
            f"all {undeclared} placement record(s) declare no dtype, so nothing about the emitted "
            f"program can be checked; a pass here would be vacuous")
    return placed


def undeclared_placement_count(command_buffer: Mapping[str, Any]) -> int:
    """Placement records with no dtype: outside this check, and counted so they are visible."""
    params = command_buffer.get("params")
    rows = (params or {}).get("lane_placement") if isinstance(params, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return 0
    return sum(1 for row in rows if isinstance(row, Mapping)
               and not (isinstance(row.get("dtype"), str) and row.get("dtype")))


def _token_present(text: str, token: str) -> bool:
    """Whether ``token`` occurs as a type word, not as part of a longer identifier.

    Scanned structurally: ``i1`` must not match inside ``i16``, and ``float`` must not match inside
    ``floating``. A substring test here would report a dropped dtype as present, which is a false
    pass in the one direction that matters.
    """
    index = 0
    length = len(text)
    span = len(token)
    while True:
        index = text.find(token, index)
        if index < 0:
            return False
        before = text[index - 1] if index else " "
        after = text[index + span] if index + span < length else " "
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        index += 1


def assess(command_buffer: Mapping[str, Any], emitted_text: str, *,
           dtype_tokens: Mapping[str, tuple[str, ...]]) -> Assessment:
    """Whether every dtype the lane plan placed appears in ``emitted_text``.

    ``dtype_tokens`` is the spelling table for the language ``emitted_text`` is written in
    (:data:`MLIR_DTYPE_TOKENS` or :data:`LLVM_DTYPE_TOKENS`). Required and never defaulted: guessing
    the language would silently check for a token the text could not contain, which reports every
    dtype as dropped -- indistinguishable from the check being broken.
    """
    placed = placed_dtypes(command_buffer)
    undeclared = undeclared_placement_count(command_buffer)
    if not placed:
        return Assessment(status="not_applicable", undeclared_placements=undeclared,
                          detail="the command buffer's params place no region on any lane, so "
                                 "there is no placement to hold the emitted program against")
    unplaceable = [p.dtype for p in placed if p.dtype not in dtype_tokens]
    if unplaceable:
        return Assessment(
            status="unknown", placed=list(placed), unplaceable=sorted(set(unplaceable)),
            undeclared_placements=undeclared,
            detail=(f"the plan places region(s) of dtype {sorted(set(unplaceable))}, which this "
                    f"spelling table cannot express (it knows {sorted(dtype_tokens)}); whether "
                    f"their work reached the emitted program is UNKNOWN and is refused rather than "
                    f"assumed"))
    dropped = [p for p in placed
               if not any(_token_present(emitted_text, token) for token in dtype_tokens[p.dtype])]
    if dropped:
        worst = max(dropped, key=lambda p: p.regions)
        return Assessment(
            status="dropped", placed=list(placed), dropped=list(dropped),
            undeclared_placements=undeclared,
            detail=(f"{sum(p.regions for p in dropped)} placed region(s) declare dtype(s) "
                    f"{[p.dtype for p in dropped]} that appear NOWHERE in the emitted program. The "
                    f"largest group is {worst.regions} region(s) of {worst.dtype!r} on lane(s) "
                    f"{list(worst.lanes)} (e.g. {list(worst.examples)}). The plan records them as "
                    f"placed, so every report downstream would describe work the program does not "
                    f"contain -- and a performance measurement would score it as faster, because "
                    f"it is: it does less"))
    return Assessment(status="emitted", placed=list(placed), undeclared_placements=undeclared,
                      detail=(f"every one of the {len(placed)} placed dtype(s) appears in the "
                              f"emitted program"))
