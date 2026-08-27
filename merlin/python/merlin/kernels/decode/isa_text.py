"""Role-tag a SELF-HOSTED-ISA instruction stream written as text.

The other decoders read binary words from a disassembler. A self-hosted-ISA target does not go through
one: the expert corpus is hand-written assembly, and the compiler emits its own encoded instructions,
so the stream that matters is text. Same output shape as the binary decoders, so
:func:`kernels.cca.lift_asm_roles` consumes either without knowing which it got.

Spelling is the whole difficulty. Measured on a real corpus: 103 distinct mnemonics across 137 files,
of which the derived model covers 77 — and every one of the 137 contains at least one it does not.
That reads like a coverage gap and is a SPELLING gap: the corpus writes ``VMATPUSH.W.MXU0`` where the
model has ``vmatpush.weight.mxu0``, and ``DMA.CONFIG x5, 0`` where the model puts the channel in the
mnemonic. Resolution therefore goes through the model's own resolver, which already normalizes case and
separators, and anything it cannot place is REPORTED by name rather than dropped — mining the subset
that happened to parse is the recorded silent-drop failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from merlin.kernels import roles as _roles

__all__ = ["TextInsn", "decode_text", "role_histogram", "unresolved_mnemonics"]


@dataclass(frozen=True)
class TextInsn:
    """One instruction, shaped like the binary decoders' output so one lifter reads both."""

    index: int
    addr: int
    identity: str                        # the model's instruction CLASS, or the raw text
    roles: tuple[str, ...] = ()
    from_endpoint: bool = False
    mnemonic: str = ""
    operands: tuple[str, ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)


def _mnemonic_and_operands(line: str) -> tuple[str, tuple[str, ...]]:
    """Split one assembly line structurally. No regex: strip comments, then split on whitespace/commas."""
    for marker in ("//", "#", ";"):
        cut = line.find(marker)
        if cut >= 0:
            line = line[:cut]
    line = line.strip()
    if not line or line.endswith(":"):
        return "", ()                    # blank, or a label
    if line.startswith("."):
        return "", ()                    # an assembler directive, not an instruction
    head, _, rest = line.partition(" ")
    ops = tuple(t for t in (o.strip() for o in rest.replace("\t", " ").split(",")) if t)
    return head.strip(), ops


def decode_text(lines, target: str, endpoint=None) -> list[TextInsn]:
    """Role-tag assembly text against ``target``'s DERIVED IsaModel.

    ``endpoint`` scopes the tagging to one datapath: an array and a lane engine can share a self-hosted
    ISA, and a stream tagged with every role would attribute the lane engine's work to the array. The
    endpoint's role table already carries that split.
    """
    from merlin.targetgen import isa_model as _IM
    try:
        model = _IM.isa_model_for_target(target)
    except Exception:  # noqa: BLE001 — no derived model: nothing can be tagged, and say so
        model = None

    claims = set()
    if endpoint is not None:
        claims = {n for names in endpoint.roles.values() for n in names}

    out: list[TextInsn] = []
    for i, raw in enumerate(lines):
        mnemonic, ops = _mnemonic_and_operands(str(raw))
        if not mnemonic:
            continue
        entry = model.resolve(mnemonic) if model is not None else None
        cls = str((entry or {}).get("class") or "")
        isa_role = str((entry or {}).get("role") or "")
        role = _roles.from_isa_role(isa_role) if isa_role else None
        # An instruction the model places but this endpoint does not claim belongs to the OTHER engine
        # on the same ISA. Recorded, not tagged.
        mine = bool(cls) and (not claims or cls in claims)
        out.append(TextInsn(
            index=i, addr=i, identity=cls or mnemonic,
            roles=((role,) if (role and mine) else ()),
            from_endpoint=mine,
            mnemonic=mnemonic, operands=ops,
            fields={"isa_role": isa_role} if isa_role else {}))
    return out


def unresolved_mnemonics(decoded) -> tuple[str, ...]:
    """Mnemonics the derived model could not place, by name and deduplicated.

    Named rather than counted. "12 unresolved" invites mining the rest; a list of names is a work item,
    and it is what distinguishes a spelling gap from a coverage gap.
    """
    return tuple(sorted({d.mnemonic for d in decoded if not d.identity or d.identity == d.mnemonic}))


def role_histogram(decoded) -> dict[str, int]:
    hist: dict[str, int] = {}
    for d in decoded:
        for r in d.roles:
            hist[r] = hist.get(r, 0) + 1
    return hist
