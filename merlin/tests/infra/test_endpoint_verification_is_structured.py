"""An UNVERIFIED endpoint must be distinguishable from one the target's own table confirmed.

``Endpoint.source`` used to carry the whole distinction as free text ("[unverified: no derived table
available]", "[declared, not verifiable against a decode table]") while the full hand-declared role
binding was kept. A grep over the repo found no consumer that branched on either phrase, so a
declaration and a derivation read identically everywhere they were consumed.
"""

from __future__ import annotations

from merlin.kernels import endpoints as EP


def _ep(verified: bool, source: str = "same source string for both") -> EP.Endpoint:
    return EP.Endpoint(
        name="e",
        target="t",
        engine="vector",
        exposure="x",
        source=source,
        roles={"elementwise": ("VADD",), "operand_load": ("LD",)},
        verified=verified,
        unverified_reason="" if verified else "no derived table available",
    )


def test_an_unverified_endpoint_is_distinguishable_from_a_derived_one():
    """THE MUTATION TEST for fix 3: the distinction survives an IDENTICAL ``source`` string."""
    derived, declared = _ep(True), _ep(False)
    assert derived.source == declared.source  # the old marker carried no information here
    assert derived.verified is not declared.verified  # the new one does
    assert derived.to_dict()["verified"] is True
    assert declared.to_dict()["verified"] is False
    assert declared.to_dict()["unverified_reason"]
    assert derived.to_dict() != declared.to_dict()


def test_a_consumer_refuses_to_treat_a_declaration_as_evidence():
    """``engines_evidenced`` is the consumer inside this module, and it now fails closed."""
    assert _ep(True).engines_evidenced() == frozenset({"vector"})
    assert _ep(False).engines_evidenced() == frozenset()
    assert _ep(True).roles_if_verified() and _ep(False).roles_if_verified() == {}
    # The declaration itself is still reachable — it is just no longer mistaken for evidence.
    assert set(_ep(False).roles) == {"elementwise", "operand_load"}


def test_endpoints_default_to_verified_so_a_derived_binding_is_unchanged():
    ep = EP.Endpoint(name="e", target="t", engine="", exposure="", source="rtl_facts")
    assert ep.verified is True and ep.unverified_reason == ""


def test_the_live_spec_marks_only_the_unverifiable_sources():
    """Every in-tree endpoint that has no decode table to check against is flagged, and only those."""
    seen = {}
    for name in EP.endpoint_names():
        try:
            ep = EP.load_endpoint(name)
        except Exception:  # noqa: BLE001 — an endpoint whose toolchain is absent here is not the subject
            continue
        seen[name] = ep.verified
        if not ep.verified:
            assert ep.unverified_reason, f"{name}: unverified with no recorded reason"
    assert seen, "no endpoint resolved at all"
    # Anything unverified must say so structurally, not only in its source prose.
    for name, ok in seen.items():
        ep = EP.load_endpoint(name)
        if "unverified" in ep.source or "not verifiable" in ep.source:
            assert ok is False, f"{name}: source says unverified but the field says verified"
