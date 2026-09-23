"""Canonical decision owners preserve frozen analyzer identities and refusals."""

import re
import sys
from types import ModuleType

import pytest
from merlin_experiments.phase2.claims import affine, dispatch, paired, pk, pr

from merlin.perf.claim_reach import analyzer_identity


@pytest.mark.parametrize(
    ("declared", "owner"),
    [
        (pk._ACCEPTANCE_BASE["analyzer"], pk),
        (pr.supported_acceptance()["analyzer"], pr),
        (affine.ANALYZER, affine),
        (paired.ANALYZER, paired),
    ],
)
def test_frozen_identity_resolves_canonical_owner_not_legacy_shadow(declared, owner, monkeypatch):
    identity = analyzer_identity({"acceptance": {"analyzer": declared}})
    monkeypatch.setitem(sys.modules, identity.module, ModuleType(identity.module))
    resolved = dispatch.resolve([{"performance": {"acceptance": {"analyzer": declared}}}])
    assert resolved.identity.declared == declared
    assert resolved.identity.module == identity.module
    assert resolved.module is owner
    assert resolved.analyze is getattr(owner, identity.function)
    assert dispatch._registry()[declared] is resolved.analyze


def test_unknown_declared_owner_stays_unavailable():
    with pytest.raises(dispatch.DispatchError, match="unavailable"):
        dispatch.resolve([{"performance": {"acceptance": {"analyzer": "absent_claim_owner.decide/v1"}}}])


def test_explicit_external_module_is_not_rewritten(monkeypatch):
    module = ModuleType("fixture_external_claim")
    module.preflight_external = lambda descriptors: {"ready": True}
    module.decide = lambda descriptors, results: {"verdict": "REFUSED"}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    declared = "fixture_external_claim.decide/v1"
    resolved = dispatch.resolve([{"performance": {"acceptance": {"analyzer": declared}}}])
    assert resolved.module is module
    assert resolved.identity.declared == declared


def test_structural_name_validation_matches_original_ascii_contract():
    samples = [None, 1, "", "valid-name.0_A", "line\n", "é", "中", "a/b"]
    samples.extend("prefix" + chr(codepoint) + "suffix" for codepoint in range(256))
    for name in samples:
        expected = isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9._-]+", name) is not None
        with pytest.raises(pk._Refusal) as raised:
            pk._descriptor_point({"name": name})
        accepted_name = "simple non-empty name" not in str(raised.value)
        assert accepted_name == expected, repr(name)
