"""A vector engine's compute term, compiled from its own sequencer rather than measured.

Why this exists as a seam rather than a model. The structural envelope needs a rate for every
resource, and a lane-pipeline vector unit does not have one in the usual sense: its cost is a
per-instruction schedule (a fill plus whichever of the read and write streams retires last), not a
throughput that a demand can be divided by. Priced as a rate it is refused, and refusing it is what
left most workloads with no end-to-end bound at all -- the unit is idle in the bound and busy in the
machine.

The machine model itself lives in mlc, compiled from the sequencer's own counters and register
depths. This module is the adapter: it maps a target's instruction stream onto that model's op
classes and sums the per-instruction cost, so the envelope gets a resolved term instead of a refusal.

WHAT IS DERIVED AND WHAT IS NOT. Every constant in the underlying model is a structural row limit or
a pipeline-register depth read out of the RTL -- none fitted. The mapping here is structural too: an
instruction's op class comes from splitting its mnemonic, never from a table of spellings. What this
module CANNOT supply is the element count when a stream lists one instruction where the machine
issues several; that needs the operand's shape, and where the shape is absent the answer is UNKNOWN
rather than a per-listed-op guess that silently under-counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: Mnemonic stems that name the same operation as the model's op class under a different spelling.
#: Split structurally from the mnemonic; a stem absent here and absent from the model is UNKNOWN.
_STEM_ALIASES = {"redsum": "rsum", "redmax": "rmax", "redmin": "rmin", "recip": "rcp"}
#: Immediate-load variants, distinguished by their qualifier rather than by a distinct stem.
_IMMEDIATE_VARIANTS = {"all": "vliAll", "row": "vliRow", "col": "vliCol", "one": "vliOne"}
_PACK = {"pack": "fp8pack", "unpack": "fp8unpack"}


class VectorModelUnavailable(RuntimeError):
    """The vector schedule model could not be reached for this target."""


@dataclass(frozen=True)
class VectorTerm:
    """A vector engine's compiled cycle term for one workload, and what it could not account for."""

    cycles: int
    instructions: int
    #: Mnemonics whose op class the model does not define. Their cost is NOT in :attr:`cycles`.
    unmapped: tuple[str, ...]
    #: True when every instruction mapped, so the term is complete rather than a floor.
    complete: bool
    provenance: str

    def claim(self) -> str:
        if self.complete:
            return f"{self.cycles} vector cycles over {self.instructions} instructions"
        return (f"AT LEAST {self.cycles} vector cycles; {len(self.unmapped)} instruction(s) have no "
                f"op class in the compiled model and are unpriced")


def op_class_for(mnemonic: str, known: "Iterable[str]") -> str | None:
    """The model's op class for one instruction, by splitting the mnemonic -- never by a spelling table.

    A mnemonic is ``<stem>[.<qualifier>...]`` with an optional leading vector marker; the qualifiers
    are datatype and shape decorations except where they select a variant that costs differently
    (an immediate load writing one register versus a pair). Returns None when the model defines no
    class, which is a real answer: an unknown instruction must not be priced as a known one."""
    parts = str(mnemonic).split(".")
    stem = parts[0]
    if stem[:1] == "v" and stem[1:] in _IMMEDIATE_VARIANTS.values():
        return stem
    if stem[:1] == "v":
        stem = stem[1:]
    if stem in _PACK:
        cls = _PACK[stem]
        return cls if cls in known else None
    if stem == "li":
        qualifier = parts[1] if len(parts) > 1 else ""
        cls = _IMMEDIATE_VARIANTS.get(qualifier)
        return cls if cls and cls in known else None
    stem = _STEM_ALIASES.get(stem, stem)
    return stem if stem in known else None


def _model(target: str, *, base: Any = None):
    from merlin.targetgen.rtl import mlc_bridge
    root = base if base is not None else mlc_bridge.mlc_dir()
    if root is None:
        raise VectorModelUnavailable(
            "the machine-model checkout is not resolvable, so the vector schedule cannot be compiled; "
            "this is UNAVAILABLE, not a vector engine that costs nothing")
    with mlc_bridge._mlc_cwd():
        from mlc.passes.compile_vpu_cycles import discover_vpu_facts, predict_op_cycles
        return discover_vpu_facts(target, base=root), predict_op_cycles


def vector_term(target: str, instructions: "Iterable[Mapping[str, Any] | tuple]", *,
                unit: str = "Vector", base: Any = None) -> VectorTerm:
    """Compile the vector cycle term for a workload's instruction stream.

    ``instructions`` are either mappings carrying a unit and a mnemonic, or ``(unit, mnemonic, ...)``
    tuples. Only those on ``unit`` are priced; the rest belong to other engines and are another term's
    business."""
    facts, predict = _model(target, base=base)
    classes = facts.op_classes()
    total = 0
    counted = 0
    unmapped: list[str] = []
    for item in instructions:
        if isinstance(item, Mapping):
            where, mnemonic = item.get("unit"), item.get("mnemonic")
        else:
            seq = tuple(item)
            where, mnemonic = (seq[0], seq[1]) if len(seq) >= 2 else (None, None)
        if where != unit or mnemonic is None:
            continue
        cls = op_class_for(mnemonic, classes)
        if cls is None:
            unmapped.append(str(mnemonic))
            continue
        total += int(predict(cls, facts))
        counted += 1
    return VectorTerm(cycles=total, instructions=counted, unmapped=tuple(unmapped),
                      complete=not unmapped,
                      provenance=(f"mlc.passes.compile_vpu_cycles over {counted} instruction(s); "
                                  f"lanes={facts.lanes}, reduce_stages={facts.reduce_stages} "
                                  "(structural, none fitted)"))
