"""Role-tag a STANDARD-ISA stream by the instruction grammar its endpoint declares.

The other decoders read a table derived from one target's RTL. A standard ISA has no such table — it
has a specification and a systematic mnemonic grammar — so the vocabulary lives in the endpoint
declaration and this module only matches against it. No instruction name appears here.

Why this exists at all: without it, the one target the mining loop actually works on is the one whose
assembly carries no declared MEANING. Its lifter counted mnemonic literals inline, so a vector kernel
and an accelerator kernel were measured by different instruments and "expert vs ours" across targets
was not the same comparison twice. Reducing both to the same role histogram is what makes the
comparison language- and target-independent.

Matching is on the STEM — the mnemonic up to its first ``.`` — because a vector ISA encodes the operand
FORM in the suffix (``.vv`` / ``.vf`` / ``.vx`` / ``.vs``) while the operation is the stem. The form is
read separately and reported, since "the scalar operand is broadcast rather than rebuilt per step" is a
real decision and it lives entirely in that suffix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["GrammarInsn", "decode_stream", "role_histogram", "unroled_mnemonics"]

#: Operand-form suffixes that mean "one operand is a SCALAR broadcast across the lanes" rather than a
#: second vector. A property of the ISA's naming grammar, not of any target.
_SCALAR_FORMS = frozenset({"vf", "vx", "vs", "vi", "wf", "wx"})


@dataclass(frozen=True)
class GrammarInsn:
    """One instruction, shaped like every other decoder's output so one lifter reads them all."""

    index: int
    addr: int
    identity: str
    roles: tuple[str, ...] = ()
    from_endpoint: bool = False
    mnemonic: str = ""
    operands: tuple[str, ...] = ()
    form: str = ""                    # the operand-form suffix, when the mnemonic carries one
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def scalar_operand(self) -> bool:
        """Does this instruction take a scalar operand broadcast across the lanes?"""
        return self.form in _SCALAR_FORMS


def _stem_and_form(mnemonic: str) -> tuple[str, str]:
    stem, _, rest = str(mnemonic).partition(".")
    return stem, rest.split(".")[0] if rest else ""


def decode_stream(raws, endpoint) -> list[GrammarInsn]:
    """Role-tag a decoded base-ISA stream against ``endpoint``'s declared grammar."""
    by_stem: dict[str, list[str]] = {}
    for role, names in (getattr(endpoint, "roles", {}) or {}).items():
        for name in names:
            by_stem.setdefault(str(name), []).append(role)

    out: list[GrammarInsn] = []
    for i, raw in enumerate(raws):
        mnemonic = str(getattr(raw, "mnemonic", "") or "")
        stem, form = _stem_and_form(mnemonic)
        roles = tuple(by_stem.get(stem, ()))
        out.append(GrammarInsn(
            index=i, addr=int(getattr(raw, "addr", 0) or 0),
            identity=stem or mnemonic, roles=roles, from_endpoint=bool(roles),
            mnemonic=mnemonic, operands=tuple(getattr(raw, "operands", ()) or ()),
            form=form, fields={"stem": stem, "form": form}))
    return out


def role_histogram(decoded) -> dict[str, int]:
    hist: dict[str, int] = {}
    for d in decoded:
        for r in d.roles:
            hist[r] = hist.get(r, 0) + 1
    return hist


def unroled_mnemonics(decoded) -> tuple[str, ...]:
    """Mnemonics observed in the stream that no declared role covers, deduplicated and NAMED.

    Named rather than counted, because the fix is a line in the endpoint declaration and a count does
    not say which line. This is also the honest counterweight to a grammar that cannot be verified
    against a decode table: the declaration is unverifiable, so what it MISSES is reported instead.
    """
    return tuple(sorted({d.mnemonic for d in decoded if not d.roles and d.mnemonic}))
