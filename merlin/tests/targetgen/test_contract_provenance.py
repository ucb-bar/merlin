"""Two ways a target's DERIVED facts can be absent while everything still looks fine.

1. `hardware_spec.target_contract` was parsed and dropped — no field held it — so a descriptor could
   declare its capability contract and the tooling would read whatever `target_registry.resolve(target)`
   found by name instead. Both failure directions were live in this repo: one target's registry resolved
   NOTHING while its declared contract sat on disk (three of four STARTER_PROMPT.md silently unrendered,
   which then failed the anti-cheat gate on their absence), and another target's two paths resolve to
   genuinely different contracts.
2. `facts.decode_body` accepted an EMPTY facts body as a valid decode body, because the check was only
   "is it a dict". Two targets' cached artifacts are `{"facts": {}, "inputs": {"hw_sha": "missing"}}` —
   nothing was extracted — and that read as a decode body and crashed the generators one layer down with
   `KeyError: 'interfaces'`: a broken-tool symptom for a missing-input cause.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.targetgen.rtl import facts as F
from merlin.targetgen.target_experiment import (declared_vs_resolved_contract,
                                                load_capability_manifest,
                                                load_target_experiment)

MINIMAL_CONTRACT = {
    "capabilities": {"ops": ["matmul"]},
    "compute_units": [{"name": "mesh", "kind": "systolic",
                       "accumulate": [{"in": "i8", "weight": "i8", "acc": "i32"}]}],
}


def _descriptor(tmp_path, *, target, contract_rel=None):
    doc = {"target": target, "capsule_corpus": "merlin/contract/capsules/isa",
           "toolchain": {"sim_via": ""}}
    if contract_rel is not None:
        doc["hardware_spec"] = {"target_contract": contract_rel}
    p = tmp_path / "target_experiment.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_target_experiment(p)


# --- 1. the declared contract ------------------------------------------------------------------------
def test_the_declared_contract_is_no_longer_dropped(tmp_path):
    te = _descriptor(tmp_path, target="nosuch_target_xyz",
                     contract_rel="merlin/contract/schemas/capsule.schema.json")
    assert te.declared_contract == "merlin/contract/schemas/capsule.schema.json"
    assert te.declared_contract_path() is not None


def test_a_declaration_with_nothing_resolving_is_declared_only(tmp_path):
    """radiance's case. The declaration is the only contract there is, so a caller that ignores it
    renders no prompt at all — which is strictly worse than using it."""
    te = _descriptor(tmp_path, target="nosuch_target_xyz",
                     contract_rel="merlin/contract/schemas/capsule.schema.json")
    declared, resolved, verdict = declared_vs_resolved_contract(te)
    assert verdict == "declared_only" and declared is not None and resolved is None


def test_no_contract_anywhere_is_reported_as_none(tmp_path):
    te = _descriptor(tmp_path, target="nosuch_target_xyz")
    assert declared_vs_resolved_contract(te)[2] == "none"


def test_a_declaration_pointing_nowhere_is_not_silently_replaced(tmp_path, monkeypatch):
    """A declared path that does not exist must not read as 'agree' just because something else
    resolves — that is the invisible substitution this whole check exists to surface."""
    real = tmp_path / "resolved.yaml"
    real.write_text(yaml.safe_dump(MINIMAL_CONTRACT), encoding="utf-8")
    te = _descriptor(tmp_path, target="nosuch_target_xyz", contract_rel="does/not/exist.yaml")
    monkeypatch.setattr("merlin.targetgen.target_registry.resolve",
                        lambda name: type("T", (), {"contract_path": real})())
    assert declared_vs_resolved_contract(te)[2] == "stale_declaration"


def test_two_different_contracts_are_a_mismatch_not_a_silent_pick(tmp_path, monkeypatch):
    """The saturn_opu case: the declared and resolved contracts differ in whether the fp8 datapaths are
    NAMED or honestly unnamed. Which is authoritative is the contract owner's call, so this must report
    rather than decide — an agent told the wrong thing about its hardware invalidates the run."""
    declared_f = tmp_path / "declared.yaml"
    resolved_f = tmp_path / "resolved.yaml"
    declared_f.write_text(yaml.safe_dump(MINIMAL_CONTRACT), encoding="utf-8")
    resolved_f.write_text(yaml.safe_dump(MINIMAL_CONTRACT), encoding="utf-8")
    te = _descriptor(tmp_path, target="nosuch_target_xyz",
                     contract_rel=str(declared_f.relative_to(declared_f.anchor)))
    # declared_contract_path resolves against repo_root; point it at the real file directly instead
    monkeypatch.setattr(type(te), "declared_contract_path", lambda self: declared_f)
    monkeypatch.setattr("merlin.targetgen.target_registry.resolve",
                        lambda name: type("T", (), {"contract_path": resolved_f})())
    assert declared_vs_resolved_contract(te)[2] == "mismatch"


def test_an_explicit_contract_path_is_read_instead_of_the_registry(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(MINIMAL_CONTRACT), encoding="utf-8")
    cap = load_capability_manifest("nosuch_target_xyz", contract_path=p)
    assert cap.target == "nosuch_target_xyz"
    assert cap.kind == "systolic"


# --- 2. an empty facts body -------------------------------------------------------------------------
def test_an_empty_facts_body_is_not_a_decode_body():
    """The bug: `{}` is a dict, so it passed. Then `f["interfaces"]` raised KeyError two frames later."""
    artifact = {"schema_version": "2.0", "facts": {},
                "inputs": {"target": "t", "hw_mlir": "t_soc.hw.mlir", "hw_sha": "missing"}}
    with pytest.raises(F.FactsEmpty) as e:
        F.decode_body(artifact, "t", needs="a funct-legality encoder")
    msg = str(e.value)
    assert "MISSING INPUT" in msg and "hw_sha='missing'" in msg


def test_an_empty_body_is_distinct_from_an_isa_less_endpoint():
    """The two must not collapse: one is a blocker (re-run introspection), the other is 'not applicable'
    (a command-buffer tile has no decode table by construction). Reporting the first as the second lets a
    target whose facts were never extracted read as ready for the arms that must be grounded in them."""
    with pytest.raises(NotImplementedError):
        F.decode_body({"schema_version": "2.0"}, "t", needs="x")     # no body at all
    assert not issubclass(F.FactsEmpty, NotImplementedError)


def test_a_real_body_still_passes_through():
    body = {"interfaces": [{"name": "funct_decode_table"}], "target": "t"}
    assert F.decode_body({"facts": body}, "t", needs="x") is body
