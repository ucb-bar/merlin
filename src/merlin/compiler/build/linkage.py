"""Merge explicit region library requirements without choosing their suppliers."""

from __future__ import annotations

import json

from merlin.development.targetgen.host_linkage_contract import validate_linkage_contract


def merge_linkage_requirements(values):
    """Retain checked archive/runtime identities and reject conflicting symbols.

    These are unresolved linking obligations, not numerical or execution proof.
    Preserve the complete selected-runtime recipe rather than treating every
    accepted contract as an archive contract.
    """
    identities, symbols, suppliers = {}, {}, {}
    for value in values:
        selected = validate_linkage_contract(value)
        identity = {key: item for key, item in selected.items() if key != "symbols"}
        key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        for symbol in selected["symbols"]:
            if symbol in suppliers and suppliers[symbol] != key:
                raise ValueError("compiled regions select different suppliers for the same symbol")
            suppliers[symbol] = key
        identities[key] = identity
        symbols.setdefault(key, set()).update(selected["symbols"])
    return [{**identities[key], "symbols": sorted(symbols[key])} for key in sorted(identities)]
