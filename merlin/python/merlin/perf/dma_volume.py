"""Predict a program's bulk-movement byte volume from the program itself.

The term this replaces is a workload INPUT. A cost model that is handed the byte volume answers
"how long does this take GIVEN the traffic", which is a weaker claim than it looks: on a
movement-bound target the handed-in number carries most of the answer. Deriving it from the
program's own descriptors is what makes the prediction a prediction.

WHY THIS DOES NOT USE THE ROLE TABLE. A movement instruction is identified by its ENCODING -- the
opcode/function fields the target's own ISA declares -- not by a semantic role attached downstream.
Roles are a separate, later mapping and were missing for this family entirely; the encoding was
always there. Identification by encoding is also what keeps this target-agnostic: nothing here names
a target, a mnemonic spelling, or a channel count.

TWO RULES THAT MAKE THE ANSWER HONEST, both enforced below and pinned by tests:

* The size operand is read from the ISA model's own field layout, never from a position. Taking
  "operand 2" because it was operand 2 in an example fails silently on the first form whose layout
  differs -- it returns a number, and the number is wrong.
* A descriptor whose size cannot be resolved makes the WHOLE KERNEL report a lower bound, not merely
  that descriptor report UNKNOWN. Summing only the descriptors that happened to resolve understates
  the footprint, and it understates it in the flattering direction: a smaller predicted footprint
  makes the compiler look better and the model look more accurate at the same time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Field names a size may travel in, strongest first. Consulted against the ISA model's OWN layout;
#: a name absent from the layout is never assumed to exist.
_SIZE_FIELD_ORDER = ("len", "bytes", "size", "nbytes", "rs2")


class DmaVolumeError(RuntimeError):
    """A movement volume was asked for where the evidence cannot support one."""


@dataclass(frozen=True)
class Descriptor:
    """One bulk-movement command, as recovered from the program text."""

    index: int
    form: str                      # the movement form, from the encoding (e.g. a load/store/wait family)
    channel: int | None
    direction: str                 # "read" | "write" | "sync"
    size_bytes: int | None         # None = unresolved; NEVER 0 as a stand-in
    size_field: str | None         # which declared field the size was read from
    unresolved_reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.size_bytes is not None


@dataclass(frozen=True)
class KernelVolume:
    """A kernel's predicted movement volume, and how much of it is actually evidenced."""

    kernel: str
    descriptors: tuple[Descriptor, ...]
    read_bytes: int
    write_bytes: int
    #: True when ANY descriptor is unresolved -- the total is then a floor, not a prediction.
    is_lower_bound: bool
    unresolved: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    def claim(self) -> str:
        """The strongest sentence this evidence supports. Never a bare number."""
        if self.is_lower_bound:
            return (f"AT LEAST {self.total_bytes} bytes moved; {len(self.unresolved)} of "
                    f"{len(self.descriptors)} descriptors did not resolve, so the true volume is "
                    f"higher by an unmeasured amount")
        return f"{self.total_bytes} bytes moved across {len(self.descriptors)} descriptors"


def size_field_for(isa: Any, mnemonic: str) -> str | None:
    """Which DECLARED field of ``mnemonic`` carries a transfer size, or None if none does.

    Read from the ISA model's own layout. Returning None is a real answer -- it means this form does
    not name a size operand, and a caller must record UNKNOWN rather than pick a position."""
    try:
        fields = isa.fields_of(mnemonic)
    except Exception:  # noqa: BLE001 - a form the model cannot lay out tells us nothing
        return None
    if not fields:
        return None
    for name in _SIZE_FIELD_ORDER:
        if name in fields:
            return name
    return None


def propagate_constants(instructions: Sequence[Mapping[str, Any]], *,
                        immediate_forms: Mapping[str, str]) -> list[dict[int, int | None]]:
    """Per-instruction snapshots of which scalar registers hold a known constant.

    Forward propagation with KILL semantics: a register written by anything other than a declared
    immediate form becomes UNKNOWN rather than keeping a stale value. That distinction is the whole
    point -- a program that loads a length once and rewrites the register later must not have the old
    length attributed to the later transfer.

    ``immediate_forms`` maps a form name to the field its constant travels in, so no spelling is
    assumed. A backward branch invalidates every register: a value that differs per iteration is not
    a constant, and treating one as constant is how a loop-carried size becomes a confident lie."""
    state: dict[int, int | None] = {}
    out: list[dict[int, int | None]] = []
    seen_backward = False
    for pos, inst in enumerate(instructions):
        if inst.get("branches_backward"):
            seen_backward = True
        form = str(inst.get("form") or "")
        ops = inst.get("operands") or {}
        dest = ops.get("rd")
        if seen_backward:
            state = {}
        elif form in immediate_forms and isinstance(dest, int):
            imm = ops.get(immediate_forms[form])
            state[dest] = int(imm) if isinstance(imm, int) else None
        elif isinstance(dest, int) and dest != 0:
            state[dest] = None          # written by something we cannot evaluate -> UNKNOWN, not stale
        out.append(dict(state))
    return out


def kernel_volume(kernel: str, descriptors: Sequence[Descriptor]) -> KernelVolume:
    """Fold descriptors into a kernel volume, degrading to a LOWER BOUND on any unresolved one."""
    read = sum(d.size_bytes or 0 for d in descriptors if d.direction == "read" and d.resolved)
    write = sum(d.size_bytes or 0 for d in descriptors if d.direction == "write" and d.resolved)
    unresolved = tuple(f"[{d.index}] {d.form}: {d.unresolved_reason or 'size unresolved'}"
                       for d in descriptors if d.direction != "sync" and not d.resolved)
    return KernelVolume(kernel=kernel, descriptors=tuple(descriptors), read_bytes=read,
                        write_bytes=write, is_lower_bound=bool(unresolved), unresolved=unresolved)


def compare_to_measured(volume: KernelVolume, measured_bytes: int, *, tolerance: float = 0.05
                        ) -> dict[str, Any]:
    """Predicted against measured, refusing a verdict a lower bound cannot support.

    A floor below the measurement is CONSISTENT, never a match: every unresolved descriptor could
    account for the gap. Only a fully resolved prediction can agree or disagree."""
    predicted = volume.total_bytes
    if volume.is_lower_bound:
        verdict = "consistent_lower_bound" if predicted <= measured_bytes else "bound_violated"
        return {"kernel": volume.kernel, "predicted": predicted, "measured": measured_bytes,
                "verdict": verdict, "is_lower_bound": True, "unresolved": volume.unresolved,
                "note": ("a floor cannot match; it can only fail to be exceeded"
                         if verdict == "consistent_lower_bound"
                         else "a LOWER bound above its measurement falsifies an input")}
    err = abs(predicted - measured_bytes) / measured_bytes if measured_bytes else None
    return {"kernel": volume.kernel, "predicted": predicted, "measured": measured_bytes,
            "verdict": "match" if (err is not None and err <= tolerance) else "mismatch",
            "relative_error": err, "is_lower_bound": False, "unresolved": ()}
