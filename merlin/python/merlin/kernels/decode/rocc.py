"""Decode a RoCC command stream against the target's OWN derived funct table.

The RoCC accelerator interface reuses one of RISC-V's custom opcodes, so a disassembler that does not
know the accelerator prints ``<unknown>`` or, worse, a plausible-looking base-ISA mnemonic. Everything
here therefore decodes the raw 32-bit word and compares INTEGERS against a table derived from the
target's RTL (``funct_decode_table``: ``custom_opcode`` + ``legal_funct`` + ``names``). No opcode, funct
value or field value is written down in this file.

Two traps this exists to avoid, both previously paid for:

* **``func3`` is not an identity constraint.** In RoCC the three bits at [14:12] are ``xd``/``xs1``/``xs2``
  — whether this particular instruction writes rd and reads rs1/rs2 — so they VARY per instruction of
  the same command. A decoder that pinned them dropped every conformant instruction that happened to
  use a different operand shape. They are decoded and reported, never matched on.
* **A word the table does not claim is reported, never dropped.** A silently discarded instruction is
  how a mis-encoded kernel audits clean. Unclaimed custom-opcode words land in ``unaccounted``.

The instruction LEVEL — a hardware-loop FSM descriptor versus a fine-grained preload/compute sequence —
is read off the decoded role stream, because it is a property of the emitted stream and cannot be seen
in the source: the same C call lowers to either, with the library doing the expansion.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.kernels import roles as _roles
from merlin.kernels.decode.opu import UNKNOWN_MNEMONIC, _word_of

__all__ = ["RoccAudit", "RoccInsn", "audit", "decode_stream", "digest", "fields_of", "funct_table_for"]

#: RISC-V R-type field positions — the instruction FORMAT, not a property of any target. RoCC lays a
#: command over it as funct7[31:25] rs2[24:20] rs1[19:15] xd[14] xs1[13] xs2[12] rd[11:7] opcode[6:0].
_OPCODE_MASK = 0x7F   # derived-ok: bits[6:0] of every 32-bit RISC-V word is a field WIDTH, not a value
_FUNCT7_SHIFT, _FUNCT7_MASK = 25, 0x7F
_XD_SHIFT, _XS1_SHIFT, _XS2_SHIFT = 14, 13, 12
_RD_SHIFT, _RS1_SHIFT, _RS2_SHIFT, _REG_MASK = 7, 15, 20, 0x1F


def fields_of(word: int) -> dict[str, int]:
    """Decode the RoCC fields of one 32-bit word. ``xd``/``xs1``/``xs2`` are reported, never matched."""
    return {
        "opcode": word & _OPCODE_MASK,
        "funct": (word >> _FUNCT7_SHIFT) & _FUNCT7_MASK,
        "xd": (word >> _XD_SHIFT) & 1,
        "xs1": (word >> _XS1_SHIFT) & 1,
        "xs2": (word >> _XS2_SHIFT) & 1,
        "rd": (word >> _RD_SHIFT) & _REG_MASK,
        "rs1": (word >> _RS1_SHIFT) & _REG_MASK,
        "rs2": (word >> _RS2_SHIFT) & _REG_MASK,
    }


def funct_table_for(target: str) -> dict[str, Any]:
    """The target's RTL-derived ``funct_decode_table``, or ``{}`` when facts are unavailable.

    Fail-closed by returning empty rather than a default table: a decoder with no derived table must
    report that it decoded nothing, not decode against a guess.
    """
    from merlin.targetgen.rtl import facts as _F
    body = (_F.load_facts(target) or {}).get("facts") or {}
    return next((i for i in body.get("interfaces", ()) if i.get("name") == "funct_decode_table"), {})


@dataclass(frozen=True)
class RoccInsn:
    """One decoded instruction, named and role-tagged by the target's own derivation."""

    index: int
    addr: int
    identity: str                  # the RTL-derived command name, or the disassembler's text
    from_endpoint: bool
    roles: tuple[str, ...] = ()
    mnemonic: str = ""
    operands: tuple[str, ...] = ()
    fields: dict[str, int] = field(default_factory=dict)


def decode_stream(insns: Sequence[Any], table: Mapping[str, Any],
                  roles_of=None) -> list[RoccInsn]:
    """Name and role-tag every instruction, using the DERIVED ``table`` for the endpoint's own words.

    ``roles_of`` maps a derived command name to EVERY role it carries (normally
    :meth:`merlin.kernels.endpoints.Endpoint.roles_of`). Absent, instructions are named but not tagged —
    an honest degradation, since a role table is a separate derivation.
    """
    opcode = table.get("custom_opcode")
    names = {int(k): str(v) for k, v in (table.get("names") or {}).items()}
    legal = {int(f) for f in (table.get("legal_funct") or ())}
    out: list[RoccInsn] = []
    for i, insn in enumerate(insns):
        word = _word_of(getattr(insn, "hexcode", ""))
        f = fields_of(word) if word is not None else {}
        mine = bool(f) and opcode is not None and f["opcode"] == int(opcode) and f["funct"] in legal
        name = names.get(f["funct"], f"funct_{f['funct']}") if mine else ""
        out.append(RoccInsn(
            index=i, addr=int(getattr(insn, "addr", 0)),
            identity=name or str(getattr(insn, "mnemonic", "")),
            from_endpoint=mine,
            roles=(tuple(roles_of(name)) if (mine and roles_of) else ()),
            mnemonic=str(getattr(insn, "mnemonic", "")),
            operands=tuple(getattr(insn, "operands", ()) or ()),
            fields=f))
    return out


@dataclass(frozen=True)
class RoccAudit:
    counts: dict[str, int] = field(default_factory=dict)
    role_counts: dict[str, int] = field(default_factory=dict)
    total_insns: int = 0
    endpoint_insns: int = 0
    unaccounted: tuple[dict[str, Any], ...] = ()
    level: str = "none"
    missing_roles: tuple[str, ...] = ()
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "role_counts": dict(self.role_counts),
                "total_insns": self.total_insns, "endpoint_insns": self.endpoint_insns,
                "unaccounted": [dict(u) for u in self.unaccounted], "level": self.level,
                "missing_roles": list(self.missing_roles), "digest": self.digest}


def digest(decoded: Sequence[RoccInsn]) -> str:
    """Hash of the DECODED identity stream, so the inert-lever guard can see this endpoint.

    A mnemonic-stream hash is blind here: every command of this endpoint disassembles to the same
    unknown text, so swapping an accumulate for a readout would hash identically and be marked inert.
    Register fields are included for endpoint instructions because for an unnameable word they are the
    only thing distinguishing two uses of one command; addresses are excluded so relocation noise does
    not masquerade as a change.
    """
    lines = []
    for d in decoded:
        if d.from_endpoint:
            f = d.fields
            lines.append(f"{d.identity} rd={f.get('rd')} rs1={f.get('rs1')} rs2={f.get('rs2')}")
        else:
            lines.append(f"{d.mnemonic} {','.join(d.operands)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def level_of(role_counts: Mapping[str, int], levels: Mapping[str, Sequence[str]]) -> str:
    """Which instruction LEVEL this stream used, from the roles it actually emitted.

    Returns ``"both"`` when a stream mixes them — a real and interesting state (a kernel that offloads
    its inner loop but still hand-drives an epilogue), not an error to be resolved into one answer.
    """
    hit = [name for name, need in levels.items() if any(role_counts.get(r) for r in need)]
    return hit[0] if len(hit) == 1 else ("both" if len(hit) > 1 else "none")


def audit(insns: Sequence[Any], target: str, endpoint=None) -> RoccAudit:
    """Audit one disassembly stream against ``target``'s derived RoCC table."""
    table = funct_table_for(target)
    roles_of = endpoint.roles_of if endpoint is not None else None
    decoded = decode_stream(insns, table, roles_of)
    counts: dict[str, int] = {n: 0 for n in (table.get("names") or {}).values()}
    role_counts: dict[str, int] = {}
    unaccounted: list[dict[str, Any]] = []
    opcode = table.get("custom_opcode")

    for d in decoded:
        if d.from_endpoint:
            counts[d.identity] = counts.get(d.identity, 0) + 1
            for r in d.roles:
                role_counts[r] = role_counts.get(r, 0) + 1
            continue
        # A word on the endpoint's own opcode that the derived table does not claim, or one the
        # disassembler declined to name: neither ours nor fine. Recorded so a mis-encoded instruction
        # surfaces instead of being counted as absent.
        if (opcode is not None and d.fields.get("opcode") == int(opcode)) \
                or d.mnemonic == UNKNOWN_MNEMONIC:
            unaccounted.append({"index": d.index, "addr": d.addr, "fields": dict(d.fields)})

    levels = endpoint.levels if endpoint is not None else {}
    return RoccAudit(
        counts=counts, role_counts=role_counts, total_insns=len(decoded),
        endpoint_insns=sum(1 for d in decoded if d.from_endpoint),
        unaccounted=tuple(unaccounted),
        level=level_of(role_counts, levels),
        missing_roles=_roles.missing_contraction_roles(role_counts) if role_counts else (),
        digest=digest(decoded))
