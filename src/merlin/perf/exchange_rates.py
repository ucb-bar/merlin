"""Measured exchange rates, per DEVICE, refused unless they close.

WHY THIS EXISTS. :mod:`merlin.perf.cost_terms` withholds a composite whenever a moved term has no
measured rate, which is correct and is why an optimisation loop here can name a trade but not score
one. The missing half was never a model — it was a measurement nobody took and nobody wrote down.
The project's single measured rate lived in a TEST FILE, where it was a fixture for the comparator
rather than a fact anything could cite, check, or attribute to a machine.

This module reads the ledger at ``merlin/contract/perf_exchange_rates.yaml`` and hands
:class:`~merlin.perf.cost_terms.Rate` objects to a caller that has already resolved WHICH DEVICE its
measurement came from. It adds exactly two refusals to what ``cost_terms`` already enforces, and both
exist because their absence would be silent:

**A rate from another device is not a weaker answer, it is a wrong one.** Rates are keyed by the pin
registry's artifact name, which is unique per device — never by the configuration string, because two
bitstreams in this repo elaborate the same config onto the same platform and are different silicon.
:func:`merlin.perf.design_identity.design_string` is what turns a run's design keys into that name.

**A rate that does not close is not a rate; it is a fitted number.** A sweep can always draw a slope
through two points. Only a closure check -- price a program where every term is known and compare the
sum against its measured total -- says the slope means anything. The tolerated residue is DECLARED in
the ledger rather than chosen here, so changing it is a reviewable diff with an argument attached; a
rate whose closure was never measured is unpriced, not assumed to close.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from .cost_terms import MEASURED, UNPRICED, Rate

__all__ = ["ExchangeRatesError", "load_ledger", "policy", "rates_for_design", "unpriced_reasons"]

_LEDGER = ("contract", "perf_exchange_rates.yaml")


class ExchangeRatesError(ValueError):
    """The ledger could not be read, or declared something that is not a rate."""


def _validated(body: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Every path into a ledger goes through here.

    An earlier cut validated only the file, so a caller injecting a ledger -- which every test and
    every future alternate-source caller does -- got a bare ``KeyError`` from deep inside instead of
    a stated reason. That is the same defect this module is written against, one level in: a refusal
    that does not say what is missing.
    """
    if not isinstance(body.get("policy"), Mapping):
        raise ExchangeRatesError(f"{where} declares no policy block; the closure bound is a review decision")
    if "max_closure_residual_fraction" not in body["policy"]:
        raise ExchangeRatesError(
            f"{where}: policy declares no max_closure_residual_fraction. There is no default -- a bound "
            "nobody chose would silently decide which composites are trusted."
        )
    if not isinstance(body.get("rates"), Mapping):
        raise ExchangeRatesError(f"{where} declares no rates mapping")
    return dict(body)


@lru_cache(maxsize=1)
def load_ledger() -> dict[str, Any]:
    import yaml

    from ..common.paths import merlin_dir

    path = merlin_dir().joinpath(*_LEDGER)
    if not path.is_file():
        raise ExchangeRatesError(f"no exchange-rate ledger at {path}; a rate is declared, never assumed")
    return _validated(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, str(path))


def policy() -> dict[str, Any]:
    """The declared closure policy. Read from the ledger; never defaulted."""
    return dict(load_ledger()["policy"])


def _fault(entry: Mapping[str, Any], bound: float, unmeasured_is_unpriced: bool) -> str | None:
    """Why this entry cannot price a composite, or ``None`` when it can."""
    if entry.get("status") != MEASURED:
        return f"status is {entry.get('status')!r}, not {MEASURED!r}"
    if not str(entry.get("provenance") or "").strip():
        # cost_terms refuses this too; refusing here as well means the LEDGER cannot carry a
        # provenance-less rate at all, rather than carrying one that the comparator happens to drop.
        return "no provenance recorded, so the number is a guess wearing a number's clothes"
    value = entry.get("cycles_per_unit")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"cycles_per_unit is {value!r}, which is not a number"
    residual = entry.get("closure_residual_fraction")
    if residual is None:
        return "closure was never measured" if unmeasured_is_unpriced else None
    if not isinstance(residual, (int, float)) or isinstance(residual, bool):
        return f"closure_residual_fraction is {residual!r}, which is not a number"
    if abs(float(residual)) > float(bound):
        return f"closure residual {float(residual):.4f} exceeds the declared bound {float(bound):.4f}"
    return None


def rates_for_design(design: str, *, ledger: Mapping[str, Any] | None = None) -> dict[str, Rate]:
    """Every usable rate for one DEVICE, as ``cost_terms`` Rates.

    ``design`` is the pin registry's artifact name, not a configuration string. A device the ledger
    does not name yields an empty mapping — which makes every moved term unpriced and the composite
    withheld, exactly as if no rate existed, because none does.

    An entry that exists but cannot be used is LEFT OUT rather than returned as ``UNPRICED``: the
    comparator treats a missing rate and an unpriced one identically, and returning it would invite a
    caller to read a number off an object the ledger refused. :func:`unpriced_reasons` says why.
    """
    body = _validated(ledger, "the supplied ledger") if ledger is not None else load_ledger()
    bound = body["policy"]["max_closure_residual_fraction"]
    unmeasured_is_unpriced = bool(body["policy"].get("unmeasured_closure_is_unpriced", True))
    declared = body["rates"].get(design) or {}
    out: dict[str, Rate] = {}
    for term, entry in declared.items():
        if not isinstance(entry, Mapping):
            raise ExchangeRatesError(f"rate {design}/{term} is not a mapping")
        if _fault(entry, bound, unmeasured_is_unpriced) is None:
            out[str(term)] = Rate(float(entry["cycles_per_unit"]), MEASURED, str(entry["provenance"]))
    return out


def unpriced_reasons(design: str, *, ledger: Mapping[str, Any] | None = None) -> dict[str, str]:
    """For every declared-but-unusable rate on this device, the reason it cannot price a composite.

    Reported rather than raised: a ledger may legitimately carry real evidence that is not yet strong
    enough to weigh with, and deleting it would lose the measurement. The caller shows the reason
    beside the withheld composite so a reader can see what is missing instead of what is absent.
    """
    body = _validated(ledger, "the supplied ledger") if ledger is not None else load_ledger()
    bound = body["policy"]["max_closure_residual_fraction"]
    unmeasured_is_unpriced = bool(body["policy"].get("unmeasured_closure_is_unpriced", True))
    declared = body["rates"].get(design) or {}
    reasons: dict[str, str] = {}
    for term, entry in declared.items():
        if not isinstance(entry, Mapping):
            continue
        fault = _fault(entry, bound, unmeasured_is_unpriced)
        if fault is not None:
            reasons[str(term)] = fault
    return reasons


#: Re-exported so a caller can spell the comparator's "no rate" state without a second import.
UNPRICED_STATUS = UNPRICED
