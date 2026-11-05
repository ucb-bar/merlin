"""Which pinned checkouts a matrix unit's facts are derived from, and whether a build may use them.

A support package names, per unit in its matrix contract, the pin its sources are read at (``pin``), the
files the derivation reads (``sources``, ``configs.config_scala``, ``configs.mixin_scala``, relative to the
pin's checkout) and any further pinned block it reads (a mapping inside the unit carrying its own ``pin``
and ``source``/``sources``). This module turns that into ``(pin, reads)`` pairs and decides, by content
(:func:`merlin.common.provenance.materiality`), whether a checkout off its pin changes what a build would
emit:

* ``unrecorded`` (another commit or branch, every byte read is the pin's): the build proceeds and records
  the revision it read;
* ``material`` (a byte read differs, a required file is missing): :func:`require` refuses, with the
  commands that would put the checkout on its pin, unless the user named the pin in
  ``MERLIN_ALLOW_OFF_PIN``.

Nothing here names a unit or a target; both are parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class UnitPinError(RuntimeError):
    """A matrix unit's sources are off their pin in a way that changes what would be built."""


@dataclass(frozen=True)
class UnitPin:
    pin: str
    reads: tuple[str, ...]


def _reads(block: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    sources = block.get("sources")
    if isinstance(sources, Mapping):
        out += [str(v) for v in sources.values()]
    if isinstance(block.get("source"), str):
        out.append(block["source"])
    configs = block.get("configs")
    if isinstance(configs, Mapping):
        for key in ("config_scala", "mixin_scala"):  # relative to the pin's checkout (host_config_scala is not)
            out += [str(p) for p in configs.get(key) or ()]
    return out


def unit_pins_from_contract(contract: Mapping[str, Any], unit: str) -> list[UnitPin]:
    """``(pin, reads)`` for ``unit`` and every pinned block nested in it, from a matrix contract mapping."""
    spec = (contract.get("units") or {}).get(unit)
    if not isinstance(spec, Mapping):
        return []
    out: list[UnitPin] = []
    for block in [spec, *(v for v in spec.values() if isinstance(v, Mapping))]:
        if isinstance(block.get("pin"), str) and block["pin"]:
            out.append(UnitPin(block["pin"], tuple(dict.fromkeys(_reads(block)))))
    return out


def unit_pins(support_target: str, unit: str | None = None) -> list[UnitPin]:
    """The pins ``support_target``'s matrix unit(s) read, from the selected support package. Every unit
    when ``unit`` is None; empty when the package declares no matrix contract."""
    from .providers import ProviderError
    from .target_registry import load_matrix_contract

    try:
        contract = load_matrix_contract(support_target)
    except ProviderError:  # the package declares no matrix contract: nothing is derived from a pin
        return []
    units = [unit] if unit else sorted(contract.get("units") or {})
    out: list[UnitPin] = []
    for u in units:
        for p in unit_pins_from_contract(contract, u):
            if p.pin not in {q.pin for q in out}:
                out.append(p)
    return out


def fix_commands(pin_name: str, path=None) -> list[str]:
    """What would put ``pin_name``'s checkout on its pin, derived from the registry entry (printed, never run)."""
    from merlin.common import provenance as PROV

    p = PROV.pin(pin_name, path)
    where = p.checkout()
    target = str(where) if where is not None else f"${p.root_env}/{p.path}".rstrip("/.")
    checkout = f"checkout -B {p.branch} {p.commit}" if p.branch else f"checkout --detach {p.commit}"
    note = p.repo_observed_note and not p.repo_observed
    lines = [f"# {' '.join(p.repo_observed_note.split())}"] if note else []
    lines += [f"git -C {target} fetch {p.repo_observed or p.repo_canonical} {p.commit}", f"git -C {target} {checkout}"]
    if p.local_edits:
        edits = ", ".join(e for e, _ in p.local_edits)
        lines.append(f"# then apply the reviewed edit the pin declares for {edits} (see the chip example's README)")
    return lines


def check(pins: list[UnitPin]) -> list[Any]:
    """:func:`merlin.common.provenance.materiality` for each pin (the pin's checkout from its root variable)."""
    from merlin.common import pin_drift as D

    return [D.materiality(p.pin, reads=p.reads) for p in pins]


def require(support_target: str, unit: str) -> list[Any]:
    """Refuse a build of ``unit`` when a pin it reads is materially off (and not allowed off); return the
    verdicts otherwise, so the caller can report an ``unrecorded`` drift."""
    from merlin.common import pin_drift as D
    from merlin.common import provenance as PROV

    verdicts = check(unit_pins(support_target, unit))
    blocking = [m for m in verdicts if m.material and not D.allowed_off_pin(m.pin)]
    if blocking:
        lines = [f"{unit}'s encodings and geometry would be derived from sources that are not its pinned revision:"]
        for m in blocking:
            seen = m.verification.observed
            lines.append(f"  {m.pin}: {seen.path} at {seen.commit[:12]} (pin: {PROV.pin(m.pin).commit[:12]})")
            lines += [f"    {r}" for r in m.reasons]
            lines += ["  to put it on the pin:"]
            lines += [f"    {c}" for c in fix_commands(m.pin)]
        lines += [
            "  or, to build from it anyway (the build records the drift): "
            f"export {D.ALLOW_OFF_PIN_ENV}={','.join(m.pin for m in blocking)}"
        ]
        raise UnitPinError("\n".join(lines))
    return verdicts
