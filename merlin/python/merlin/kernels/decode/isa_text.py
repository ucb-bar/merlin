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
from pathlib import Path
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


#: RISC-V field positions used to IDENTIFY an instruction: opcode[6:0] and funct7[31:25]. The
#: instruction FORMAT, not any target's values — the values are read from both sides and compared.
_OPCODE_MASK, _FUNCT7_SHIFT, _FUNCT7_MASK = 0x7F, 25, 0x7F

_BRIDGE_CACHE: dict = {}


def _assembler_bridge(target: str, endpoint) -> dict:
    """corpus mnemonic -> the model entry it encodes to, joined by ENCODING rather than by name.

    A target whose expert corpus is hand-written assembly spells instructions the way its own
    assembler does, and the derived model spells them the way its ISA definition does. Measured, those
    differ by abbreviation (``VMATPUSH.W.MXU0`` vs ``vmatpush.weight.mxu0``) and by which field is in
    the mnemonic, so string matching cannot join them and every core MXU instruction in a 137-kernel
    corpus came back unresolvable.

    The target's assembler is a pure ENCODER: each mnemonic is a function returning the instruction
    word. Calling it with zeroed operands yields a word whose opcode and funct7 identify the
    instruction regardless of the operands, and the model's table carries the same two fields. So the
    correspondence is derived from the target's own sources, and a rename on either side simply drops
    out of the join instead of silently mis-mapping.

    Returns ``{}`` when the assembler is not declared or not readable — the honest degradation, with
    the unresolved mnemonics still reported by name.
    """
    key = (target, getattr(endpoint, "name", ""))
    if key in _BRIDGE_CACHE:
        return _BRIDGE_CACHE[key]
    _BRIDGE_CACHE[key] = out = {}
    try:
        import importlib.util
        import inspect

        from merlin.common import provenance as _prov
        from merlin.kernels import endpoints as _ep
        from merlin.targetgen import isa_model as _IM

        block = ((_ep._spec().get("endpoints") or {}).get(getattr(endpoint, "name", "")) or {})
        decl = ((block.get("encoding") or {}).get("assembler") or {})
        if not decl.get("pin") or not decl.get("path"):
            return out
        root = Path(_prov.verify(str(decl["pin"])).observed.path)
        path = root / str(decl["path"])
        if not path.is_file():
            return out

        model = _IM.isa_model_for_target(target)
        by_ident: dict = {}
        for name in (getattr(model, "by_mnemonic", None) or {}):
            entry = model.resolve(name)
            if entry and entry.get("opcode") is not None:
                by_ident.setdefault((int(entry["opcode"]), int(entry.get("funct7") or 0)), entry)

        spec = importlib.util.spec_from_file_location(f"_asm_{target}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, fn in vars(mod).items():
            if not (name.isupper() and callable(fn)):
                continue
            try:
                n = len(inspect.signature(fn).parameters)
                word = int(fn(*([0] * n)))
            except Exception:  # noqa: BLE001 — not a pure encoder: skip it, never guess its encoding
                continue
            ident = (word & _OPCODE_MASK, (word >> _FUNCT7_SHIFT) & _FUNCT7_MASK)
            entry = by_ident.get(ident)
            if entry is not None:
                out[name.replace("_", ".").upper()] = entry
    except Exception:  # noqa: BLE001 — no assembler / unreadable pin: degrade, and report by name
        return out
    return out


def _prefix_resolve(model, mnemonic: str):
    """Resolve a corpus mnemonic the model spells with an extra, OPERAND-ENCODED suffix.

    Measured shape: the corpus writes ``DMA.CONFIG x5, 0`` — channel as an operand — while the derived
    model carries ``dma.config.ch0`` ... ``dma.config.ch5``, channel in the mnemonic. Neither spelling
    is wrong; they put the same field in different places, and a decoder that only tries exact matches
    reports the whole family as unresolvable.

    Structural, not a hand-written alias table: normalize separators, then look for model mnemonics
    that EXTEND the corpus one. FAILS CLOSED when the candidates disagree about the role — if two
    instructions share a prefix and do different things, the prefix does not identify the operation,
    and guessing would attach a confident wrong meaning to every use of it.
    """
    want = str(mnemonic).replace(".", "_").replace("-", "_").upper()
    if not want:
        return None, ""
    hits = []
    for name in (getattr(model, "by_mnemonic", None) or {}):
        norm = str(name).replace(".", "_").replace("-", "_").upper()
        # A genuine extension, not a coincidental shared start: the next character must be a separator.
        if norm.startswith(want) and (len(norm) == len(want) or norm[len(want)] == "_"):
            hits.append(name)
    if not hits:
        return None, ""
    entries = [model.resolve(h) for h in hits]
    entries = [e for e in entries if e]
    roles = {str((e or {}).get("role") or "") for e in entries}
    if len(roles) != 1:
        return None, (f"{mnemonic!r} prefix-matches {len(hits)} model mnemonic(s) with disagreeing "
                      f"roles {sorted(roles)}; refusing to guess")
    return entries[0], (f"{mnemonic!r} resolved by prefix to {len(hits)} operand-encoded variant(s) "
                        f"(e.g. {hits[0]!r})" if len(hits) > 1 else "")


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

    bridge = _assembler_bridge(target, endpoint) if endpoint is not None else {}
    claims = set()
    if endpoint is not None:
        claims = {n for names in endpoint.roles.values() for n in names}

    out: list[TextInsn] = []
    notes: list[str] = []
    for i, raw in enumerate(lines):
        mnemonic, ops = _mnemonic_and_operands(str(raw))
        if not mnemonic:
            continue
        entry = model.resolve(mnemonic) if model is not None else None
        if entry is None and bridge:
            entry = bridge.get(str(mnemonic).replace("_", ".").upper())
            if entry is not None and "assembler bridge" not in "".join(notes):
                notes.append(f"resolved {len(bridge)} corpus spelling(s) through the target's own "
                             f"assembler bridge (joined by encoding, not by name)")
        if entry is None and model is not None:
            entry, why = _prefix_resolve(model, mnemonic)
            if why:
                notes.append(why)
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
    decode_text.last_notes = tuple(dict.fromkeys(notes))     # surfaced by the audit, never dropped
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
