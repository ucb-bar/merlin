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


def test_no_unexpected_bijection_drift_gemmini():
    """The per-target ratchet: once the generic derivation-driven backend registers the DERIVED routes
    into the agnostic core (for an example target), the spatial lever axes (dataflow, accumulator-
    residency) are backed and there is no orphan. Fixed mesh geometry is IDENTITY, excluded. The routes
    come from the discovered hardware, not per-target Python."""
    from merlin.targetgen import rtl_backend
    rtl_backend.register("gemmini")
    unexpected = cc.check_bijection("gemmini").unexpected()
    assert unexpected.clean, (
        f"NEW bijection drift (not in cca_contract.KNOWN_OPEN['gemmini']):\n"
        f"  orphan_fields (LEVER field, no route): {unexpected.orphan_fields}\n"
        f"  orphan_routes (route, no backing field): {unexpected.orphan_routes}\n"
        "Either the derived routes changed, or document it in KNOWN_OPEN['gemmini'].")


def test_known_open_is_not_stale_gemmini():
    """Reverse tripwire: every allowlisted gap must STILL be a real gap, so as each derived route lands
    the orphan must be removed from KNOWN_OPEN (the checklist only shrinks)."""
    rep = cc.check_bijection("gemmini")
    known = cc.KNOWN_OPEN.get("gemmini", {})
    stale_fields = sorted(set(known.get("orphan_fields", ())) - set(rep.orphan_fields))
    stale_routes = sorted(set(known.get("orphan_routes", ())) - set(rep.orphan_routes))
    assert not stale_fields, f"KNOWN_OPEN['gemmini'] orphan_fields already closed — remove: {stale_fields}"
    assert not stale_routes, f"KNOWN_OPEN['gemmini'] orphan_routes already closed — remove: {stale_routes}"


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


# ---------------------------------------------------------------------------------------------
# Backend-general ratchet
# ---------------------------------------------------------------------------------------------
#
# The two per-backend tests above are hand-written, one function each, and that is exactly how the
# matrix backend's break stayed invisible: `cca_matrix.register_routes` has no caller outside tests, so
# `check_bijection("matrix")` was never evaluated and its two orphan routes were never reported. A
# hand-maintained list of backends to check will always lag the backends that exist. These tests
# DISCOVER them instead, so a backend cannot be added without its bijection being checked.


def _registered_backends() -> list[str]:
    """Every backend that has routes registered, discovered from the router itself.

    `cca_matrix` is registered here because it is the one route provider whose real caller has not
    landed yet (its routes belong to a matrix-unit backend, which arrives with the per-target route
    tables). Registering it in the test keeps its bijection checked in the meantime rather than
    checked never; when the real caller lands, this line becomes redundant, not wrong.
    """
    from merlin.kernels import action_catalog as AC, cca_matrix

    cca_matrix.register_routes()
    return sorted(b for b, routes in AC._ROUTES.items() if routes)


def test_every_registered_backend_is_checked():
    """There is at least one backend beyond rvv, and the discovery actually finds them.

    Without this, a discovery bug that returned `[]` would make every test below vacuously pass — the
    failure mode where a ratchet reports green because it checked nothing.
    """
    backends = _registered_backends()
    assert "rvv" in backends, backends
    assert len(backends) > 1, f"discovery found only {backends} — the sweep below would be vacuous"


def test_no_backend_has_an_unclassified_field_or_malformed_ladder():
    """Hard errors, for every backend: these are never allowlistable."""
    for backend in _registered_backends():
        rep = cc.check_bijection(backend)
        assert rep.unclassified == [], f"{backend}: unclassified CCA fields (add to FIELD_REGISTRY): {rep.unclassified}"
        assert rep.ladder_errors == [], f"{backend}: {rep.ladder_errors}"


def test_no_unexpected_bijection_drift_on_any_backend():
    """The ratchet, generalized: no orphan field or route on ANY backend beyond documented KNOWN_OPEN.

    This is the test that would have caught the matrix break: `compute.accumulator_resident` and
    `compute.contraction_form` were classified for the literal backend list ("rvv", ...), so a
    matrix-unit backend routing them had two routes with no backing field.
    """
    dirty = {}
    for backend in _registered_backends():
        unexpected = cc.check_bijection(backend).unexpected()
        if not unexpected.clean:
            dirty[backend] = (unexpected.orphan_fields, unexpected.orphan_routes)
    assert not dirty, (
        "NEW bijection drift (not in cca_contract.KNOWN_OPEN), per backend "
        "(orphan_fields = LEVER with no route, orphan_routes = route with no backing field):\n"
        + "\n".join(f"  {b}: fields={f} routes={r}" for b, (f, r) in sorted(dirty.items()))
        + "\nEither add the missing route/field, or (if intentionally deferred) document it in KNOWN_OPEN.")


def test_a_family_tagged_axis_is_not_inherited_without_a_route():
    """The safety property that makes family tags usable at all.

    `compute.*` levers are tagged with the FAMILY "compute" so a target-agnostic property does not have
    to enumerate every target that has it. That is only sound because `leverable_axes` gates the
    family-indirect arm on the axis being ROUTED: a backend must never inherit a family axis its
    hardware does not expose, or every backend would owe a route for every compute lever.
    """
    from merlin.kernels import action_catalog as AC

    backend = "family_probe_backend"
    AC._ROUTES.setdefault(backend, []).append(AC._Route(
        axis="compute.contraction_form", when=lambda d: True, action_class="KNOB",
        target_seam="knob:probe", change="probe", forkable_now=False, expected_effect="probe"))
    try:
        leverable = cc.leverable_axes(backend)
        assert "compute.contraction_form" in leverable, "a ROUTED family axis must be leverable"
        # ... but nothing else in the family comes along for the ride.
        assert "compute.epilogue" not in leverable, (
            "an UNROUTED family axis leaked in — family tags would then force every backend to route "
            "every compute lever, which is the opposite of what they are for")
        assert cc.check_bijection(backend).orphan_routes == []
    finally:
        AC._ROUTES.pop(backend, None)


class TestTheContractSeesEveryFacetTheCCAHas:
    """A completeness check whose universe is hand-maintained can only confirm what someone remembered.

    ``FACET_CLASSES`` was a literal dict. Adding ``CommunicationFacet`` to the CCA with seven
    unclassified fields left this whole suite GREEN, because the contract could not see the facet at
    all -- so "every field is classified" was true of a universe that excluded the new fields.
    ``cca_compare._facet_names`` had already learned this and its docstring says so; the same fix
    belongs on both sides of the contract.
    """

    def test_the_contracts_universe_equals_the_ccas_facets(self):
        import dataclasses

        from merlin.kernels import cca as ccamod
        from merlin.kernels.cca_contract import FACET_CLASSES

        facet_types = {n for n, o in vars(ccamod).items()
                       if dataclasses.is_dataclass(o) and n.endswith("Facet")}
        on_cca = {f.name for f in dataclasses.fields(ccamod.CCA)
                  if any(t in str(f.type) for t in facet_types)}
        assert set(FACET_CLASSES) == on_cca, (
            f"the contract sees {sorted(set(FACET_CLASSES))} but the CCA has {sorted(on_cca)}; a "
            "facet the contract cannot see has fields it cannot require to be classified")

    def test_every_field_of_every_facet_has_a_row(self):
        import dataclasses

        from merlin.kernels.cca_contract import FACET_CLASSES, FIELD_REGISTRY

        missing = [f"{fname}.{fld.name}"
                   for fname, cls in FACET_CLASSES.items()
                   for fld in dataclasses.fields(cls)
                   if f"{fname}.{fld.name}" not in FIELD_REGISTRY]
        assert not missing, f"unclassified facet field(s): {missing}"

    def test_the_facet_that_exposed_this_is_visible_without_a_list_entry(self):
        """THE REGRESSION CASE, concretely. ``communication`` was added to the CCA and to no list.

        Under the old hand-written FACET_CLASSES it was invisible and its seven fields went
        unclassified with the suite green. If someone reinstates a literal dict and forgets a facet,
        this fails.
        """
        from merlin.kernels.cca_contract import FACET_CLASSES, FIELD_REGISTRY

        assert "communication" in FACET_CLASSES
        assert any(k.startswith("communication.") for k in FIELD_REGISTRY)
