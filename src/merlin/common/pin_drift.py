"""Whether a checkout's drift from its pin changes what is built from it, or only how it is cited.

:func:`merlin.common.provenance.verify` reports every disagreement with a pin. Not all of them make a
result wrong: a checkout on another commit whose every READ byte is the pinned commit's builds the same
thing, and only its citation differs. :func:`materiality` sorts the two, by content, so a flow can refuse
the first kind and record the second. A user who knowingly builds from a materially-off checkout says so
in ``MERLIN_ALLOW_OFF_PIN`` (:func:`allowed_off_pin`); the drift is still recorded. Nothing here moves a
checkout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import provenance as P

#: ``pinned``: nothing material differs. ``unrecorded``: another commit or branch, but every byte the work
#: reads is the pinned commit's (or a local edit the pin declares), so a result is right and only its
#: citation differs. ``material``: a byte the work reads is not the pin's or cannot be told, or a required
#: path is missing: a result built from it is about another revision.
DRIFT_NONE, DRIFT_UNRECORDED, DRIFT_MATERIAL = "pinned", "unrecorded", "material"
#: Pins a user has knowingly accepted off their revision: ``MERLIN_ALLOW_OFF_PIN=<pin>[,<pin>...]``.
ALLOW_OFF_PIN_ENV = "MERLIN_ALLOW_OFF_PIN"


@dataclass(frozen=True)
class Materiality:
    """:func:`materiality`'s answer: the verdict, why, and the verification it came from."""

    pin: str
    verdict: str
    reasons: tuple[str, ...]
    verification: P.Verification

    @property
    def material(self) -> bool:
        return self.verdict == DRIFT_MATERIAL


def materiality(
    name: str,
    *,
    checkout: str | Path | None = None,
    path: str | Path | None = None,
    reads: Sequence[str] | None = None,
) -> Materiality:
    """Is ``name``'s checkout off its pin in a way that changes what is built from it?

    Decided by CONTENT, over ``reads`` (the paths the work reads; the pin's ``requires_paths`` are always
    added): a commit or branch other than the pin's is only ``unrecorded`` while every one of those files
    is byte-identical to the pinned commit's, or to a local edit the pin declares. A file that differs, or
    whose pinned bytes cannot be read, a missing required path, a forbidden one present, a declared build
    product that changed, or a covered pin that is itself material, is ``material``.
    """
    p = P.pin(name, path)
    got = P.verify(name, checkout=checkout, path=path, reads=reads)
    if got.ok:
        return Materiality(name, DRIFT_NONE, (), got)
    reasons: list[str] = []
    if not got.observed.present or got.observed.commit == P.UNKNOWN:
        reasons += list(got.drift) or [f"no readable checkout at {got.observed.path}"]
        return Materiality(name, DRIFT_MATERIAL, tuple(reasons), got)
    reasons += [f"missing required path {rel}" for rel in got.missing_paths]
    reasons += [f"{rel} is present but the pin declares it absent" for rel in got.forbidden_present]
    read_set = list(dict.fromkeys([*(reads or ()), *p.requires_paths]))
    if not read_set:
        reasons.append("the pin declares no read set, so whether the bytes read are the pin's is UNKNOWN")
    for rel in read_set:
        s = P.source_status(name, rel, checkout=got.observed.path, path=path)
        reviewed = s.status == P.OFF_PIN and bool(s.declared_digest) and s.digest == s.declared_digest
        if s.status != P.PINNED and not reviewed:
            reasons.append(f"{s.rel}: {s.status}: {s.reason}")
    reasons += P._build_product_findings(p, Path(got.observed.path))[0]
    for child in got.covered:
        sub = materiality(child.pin, path=path)
        reasons += [f"[{child.pin}] {r}" for r in sub.reasons if sub.material]
    return Materiality(name, DRIFT_MATERIAL if reasons else DRIFT_UNRECORDED, tuple(reasons), got)


def allowed_off_pin(name: str) -> bool:
    """True when the user named ``name`` in ``MERLIN_ALLOW_OFF_PIN`` (process environment, then .env): they
    accept building from a checkout off that pin. The drift is still recorded by whatever is built."""
    from .paths import env as _env

    return name in {part.strip() for part in (_env(ALLOW_OFF_PIN_ENV) or "").split(",")}
