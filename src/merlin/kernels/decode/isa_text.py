"""Role-tag a SELF-HOSTED-ISA instruction stream written as text.

The other decoders read binary words from a disassembler. A self-hosted-ISA target does not go through
one: the expert corpus is hand-written assembly, and the compiler emits its own encoded instructions,
so the stream that matters is text. Same output shape as the binary decoders, so
:func:`kernels.cca.lift_asm_roles` consumes either without knowing which it got.

Assembly mnemonics resolve through the selected model's own metadata. Undeclared spellings and
unexpanded pseudo-instructions remain explicit, per-occurrence gaps in the decoded stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from merlin.kernels import roles as _roles

__all__ = ["TextInsn", "decode_text", "role_histogram", "unresolved_instructions", "unresolved_mnemonics"]


@dataclass(frozen=True)
class TextInsn:
    """One instruction, shaped like the binary decoders' output so one lifter reads both."""

    index: int
    addr: int
    identity: str  # the model's instruction CLASS, or the raw text
    roles: tuple[str, ...] = ()
    from_endpoint: bool = False
    mnemonic: str = ""
    operands: tuple[str, ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)
    source_line: str = ""
    unresolved_reason: str = ""


def _mnemonic_and_operands(line: str) -> tuple[str, tuple[str, ...]]:
    """Split one assembly line structurally. No regex: strip comments, then split on whitespace/commas."""
    for marker in ("//", "#", ";"):
        cut = line.find(marker)
        if cut >= 0:
            line = line[:cut]
    line = line.strip()
    if not line or line.endswith(":"):
        return "", ()  # blank, or a label
    if line.startswith("."):
        return "", ()  # an assembler directive, not an instruction
    parts = line.split(None, 1)
    head, rest = parts[0], parts[1] if len(parts) > 1 else ""
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
    except Exception as exc:
        raise RuntimeError(f"{target}: selected ISA metadata unavailable: {exc}") from exc
    if model is None or model.is_empty():
        raise RuntimeError(f"{target}: selected ISA metadata unavailable or empty")
    claims = set()
    if endpoint is not None:
        claims = {n for names in endpoint.roles.values() for n in names}

    out: list[TextInsn] = []
    for i, raw in enumerate(lines):
        mnemonic, ops = _mnemonic_and_operands(str(raw))
        if not mnemonic:
            continue
        entry = model.resolve(mnemonic)
        cls = str((entry or {}).get("class") or "")
        isa_role = str((entry or {}).get("role") or "")
        role = _roles.from_isa_role(isa_role) if isa_role else None
        reason = ""
        if entry is None:
            reason = "not declared by selected ISA metadata; no pseudo expansion was performed"
        elif not cls:
            reason = "selected ISA metadata entry has no operation class"
        # An instruction the model places but this endpoint does not claim belongs to the OTHER engine
        # on the same ISA. Recorded, not tagged.
        mine = bool(cls) and (not claims or cls in claims)
        out.append(
            TextInsn(
                index=i,
                addr=i,
                identity=cls or mnemonic,
                roles=((role,) if (role and mine) else ()),
                from_endpoint=mine,
                mnemonic=mnemonic,
                operands=ops,
                fields={"isa_role": isa_role, "operation": entry.get("mnemonic", mnemonic)} if entry else {},
                source_line=str(raw),
                unresolved_reason=reason,
            )
        )
    return out


def unresolved_mnemonics(decoded) -> tuple[str, ...]:
    """Mnemonics the derived model could not place, by name and deduplicated.

    Named rather than counted. "12 unresolved" invites mining the rest; a list of names is a work item,
    and it is what distinguishes a spelling gap from a coverage gap.
    """
    return tuple(sorted({d.mnemonic for d in decoded if d.unresolved_reason}))


def unresolved_instructions(decoded) -> tuple[dict[str, Any], ...]:
    """Every unresolved occurrence, retaining its source line, operands, and reason."""
    return tuple(
        {
            "index": d.index,
            "line": d.index + 1,
            "source_line": d.source_line,
            "mnemonic": d.mnemonic,
            "operands": list(d.operands),
            "reason": d.unresolved_reason,
        }
        for d in decoded
        if d.unresolved_reason
    )


def role_histogram(decoded) -> dict[str, int]:
    hist: dict[str, int] = {}
    for d in decoded:
        for r in d.roles:
            hist[r] = hist.get(r, 0) + 1
    return hist
