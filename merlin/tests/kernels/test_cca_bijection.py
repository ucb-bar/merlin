"""WS-C: enforce the CCA ⇄ CompilerAction bijection (the "CCA captures everything, everything maps to a
compiler lever" invariant), as a RATCHET.

The invariant is checked by reflection over the real ``cca.py`` dataclasses and ``action_catalog._ROUTES``
— nothing hardcoded, so schema and router cannot silently drift. Two classes of finding are HARD errors
(never allowlisted): an ``unclassified`` schema field (a CCA field nobody classified) and a malformed
escalation ladder. The two "gap" classes (a LEVER field with no route; a route with no backing field) are
allowlisted in ``cca_contract.KNOWN_OPEN`` while WS-C Phase 2 closes them, so this test is GREEN today and
fails the moment NEW drift appears. The reverse tripwire (``test_known_open_is_not_stale``) fails if an
allowlisted gap is actually already closed — forcing KNOWN_OPEN to shrink to empty as the roadmap lands.
"""
from __future__ import annotations

from merlin.kernels import cca_contract as cc


def test_every_schema_field_is_classified():
    """Capture-completeness: adding a field to cca.py without a FIELD_REGISTRY row is an error."""
    rep = cc.check_bijection("rvv")
    assert rep.unclassified == [], f"unclassified CCA facet fields (add to FIELD_REGISTRY): {rep.unclassified}"


def test_escalation_ladders_are_well_formed():
    """Multi-route axes (e.g. accumulator_resident PASS->CODEGEN) must use distinct action classes."""
    rep = cc.check_bijection("rvv")
    assert rep.ladder_errors == [], rep.ladder_errors


def test_no_unexpected_bijection_drift_rvv():
    """The ratchet: beyond the documented KNOWN_OPEN gaps, there is no orphan field or orphan route."""
    unexpected = cc.check_bijection("rvv").unexpected()
    assert unexpected.clean, (
        f"NEW bijection drift (not in cca_contract.KNOWN_OPEN):\n"
        f"  orphan_fields (LEVER field, no route): {unexpected.orphan_fields}\n"
        f"  orphan_routes (route, no backing field): {unexpected.orphan_routes}\n"
        "Either add the missing route/field, or (if intentionally deferred) document it in KNOWN_OPEN.")


def test_known_open_is_not_stale():
    """Reverse tripwire: every allowlisted gap must STILL be a real gap. When Phase 2 closes one, this
    fails until it is removed from KNOWN_OPEN — so the allowlist can only shrink, never rot."""
    rep = cc.check_bijection("rvv")
    known = cc.KNOWN_OPEN.get("rvv", {})
    stale_fields = sorted(set(known.get("orphan_fields", ())) - set(rep.orphan_fields))
    stale_routes = sorted(set(known.get("orphan_routes", ())) - set(rep.orphan_routes))
    assert not stale_fields, f"KNOWN_OPEN orphan_fields already closed — remove from allowlist: {stale_fields}"
    assert not stale_routes, f"KNOWN_OPEN orphan_routes already closed — remove from allowlist: {stale_routes}"


def test_current_state_matches_documented_gaps():
    """Documents current reality: the live gaps are exactly the KNOWN_OPEN set (no more, no less)."""
    rep = cc.check_bijection("rvv")
    known = cc.KNOWN_OPEN["rvv"]
    assert set(rep.orphan_fields) == set(known["orphan_fields"])
    assert set(rep.orphan_routes) == set(known["orphan_routes"])


def test_contract_records_region_per_lever_axis(tmp_path):
    """C2/C3 tie-in: the bijection contract broadens onto the region taxonomy — every LEVER axis row
    records the compiler region that governs it."""
    import yaml

    p = cc.dump_contract("rvv", tmp_path / "cca_bijection.yaml")
    doc = yaml.safe_load(p.read_text())
    by_axis = {row["axis"]: row for row in doc["axes"]}
    for axis in cc.leverable_axes("rvv"):
        assert by_axis[axis]["region"], f"{axis} has no governing region in the contract"


def test_dump_contract_roundtrips(tmp_path):
    """dump_contract emits a valid YAML contract with one row per lever/routed axis."""
    import yaml

    p = cc.dump_contract("rvv", tmp_path / "cca_bijection.yaml", toolchain_version={"probe": "test"})
    doc = yaml.safe_load(p.read_text())
    assert doc["backend"] == "rvv"
    assert doc["toolchain"] == {"probe": "test"}
    axes = {row["axis"] for row in doc["axes"]}
    # every routed axis and every lever axis appears as a row
    assert cc.routed_axes("rvv") <= axes
    assert cc.leverable_axes("rvv") <= axes
