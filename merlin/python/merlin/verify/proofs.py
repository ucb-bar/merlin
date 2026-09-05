"""Auditing ``contract.prove``: which proof tokens are backed by evidence, and which are assertions.

The `contract` dialect reads like a proof system — `contract.assume`, `contract.require`,
`contract.prove` producing a `!contract.proof<"requirement">`. Its verifier, however, checks only
that the token's requirement *string* equals the op's requirement *string*
(`xdsl_dialects/contract.py`). So a "discharged obligation" in the IR is one string matching another;
nothing establishes that the property holds.

That is not a bug to fix by tightening the verifier — a verifier cannot prove a semantic property, and
making it reject un-evidenced proofs would break every module that legitimately carries one. The fix
is to make the distinction VISIBLE and MEASURED: this module classifies each proof token by whether a
verification layer actually discharged its requirement, and reports the rest as ``asserted`` rather
than letting them read as proved.

Three states, and ``asserted`` is deliberately not a failure — it is the honest name for where most
of the IR is today, and it is what makes progress countable:

``verified``  a verification layer discharged this requirement for the producing pass
``asserted``  the token exists and names a producer, but no layer has discharged it
``unattributed``  the token names no producing pass, so nothing could discharge it even in principle
"""
from __future__ import annotations

from dataclasses import dataclass

VERIFIED = "verified"
ASSERTED = "asserted"
UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class Proof:
    """One ``contract.prove`` token and its evidential standing."""
    requirement: str
    producer_pass: str | None
    status: str

    @property
    def is_evidence(self) -> bool:
        return self.status == VERIFIED


def _prop(op, name: str) -> str | None:
    attr = op.properties.get(name)
    if attr is None:
        return None
    data = getattr(attr, "data", None)
    return data if isinstance(data, str) else str(attr)


def audit_proofs(module, discharged: dict[str, set[str]] | None = None) -> list[Proof]:
    """Classify every ``contract.prove`` in ``module``.

    ``discharged`` maps a pass name to the set of requirements a verification layer has discharged
    for it — e.g. from the verify log. Absent, everything with a producer is ``asserted``, which is
    the correct answer when no verification has run: unknown is never silently upgraded.
    """
    discharged = discharged or {}
    out: list[Proof] = []
    for op in module.walk():
        if op.name != "contract.prove":
            continue
        requirement = _prop(op, "requirement") or ""
        producer = _prop(op, "producer_pass")
        if not producer:
            status = UNATTRIBUTED
        elif requirement in discharged.get(producer, set()):
            status = VERIFIED
        else:
            status = ASSERTED
        out.append(Proof(requirement=requirement, producer_pass=producer, status=status))
    return out


def summarize(proofs: list[Proof]) -> dict[str, int]:
    """Counts per status — the number that should move as verification coverage grows."""
    counts = {VERIFIED: 0, ASSERTED: 0, UNATTRIBUTED: 0}
    for p in proofs:
        counts[p.status] = counts.get(p.status, 0) + 1
    return counts
