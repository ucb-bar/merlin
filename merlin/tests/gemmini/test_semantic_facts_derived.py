"""The selected gemmini support provider declares where its behavioural facts are written, and the
readout-width fact is DERIVED from those sources -- no facts document is shipped to go stale."""

from __future__ import annotations

import pytest
import selected_driver

from merlin.targetgen.rtl import semantic_facts as SF

# semantic_probes.yaml is consumed only by merlin.perf.whole_model_gate (full-width readout machines),
# never by Phase 0, corpus generation or readiness; the generic data provider ships no probes, so
# these tests run against an explicitly selected package-owned provider only.
pytestmark = [pytest.mark.target("gemmini"), selected_driver.requires_package_owned_support("gemmini")]
_TARGET = "gemmini"


def test_the_provider_declares_probes_and_ships_no_facts_document():
    support = selected_driver.require_support(_TARGET)
    assert (support / "contracts" / SF.PROBES_FILE).is_file()
    assert not list((support / "contracts").glob("semantic_facts*.json"))
    probes = SF.load_probes(_TARGET)
    assert probes is not None and probes["accumulator_readout_width"]["machines"]


def test_the_readout_width_is_derived_where_its_sources_are_present():
    selected_driver.require_support(_TARGET)
    fact = SF.readout_machines(_TARGET)
    rows = (fact.get("value") or {}).get("machines") or {}
    derived = {m: r for m, r in rows.items() if r.get("status") == SF.DERIVED}
    if not derived:
        pytest.skip(f"no machine's sources are present on this host: {rows}")
    for row in derived.values():
        assert isinstance(row["full_width_readout"], bool) and row["header"]["sha256"]
        assert fact["provenance"]["sources"]
