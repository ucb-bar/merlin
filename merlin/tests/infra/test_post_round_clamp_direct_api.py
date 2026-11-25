"""Gate ordering over ordinary baremetal seam; no simulated execution claim."""

import json

import pytest
import test_baremetal_saved_model as FX
from test_model_check_full_build import _selected_host_chain, _selected_host_qualification

from merlin.compile import baremetal_model as BM
from merlin.llvmlower import post_round_clamp_stage as S


def test_missing_actual_recipe_refuses_before_target_run(tmp_path, monkeypatch):
    inputs, out = FX._fixture(tmp_path, monkeypatch)
    chain, doc = _selected_host_chain(tmp_path)
    q = _selected_host_qualification(chain, doc, tmp_path, monkeypatch, [])
    monkeypatch.setattr(BM.spike_model, "build", FX._fake_build)
    monkeypatch.setattr(BM.spike_model, "run", lambda *a, **kw: pytest.fail("target ran without final recipe"))
    with pytest.raises(BM.BaremetalModelError, match="actual completed recipe"):
        BM.compile_saved_model(
            **inputs,
            output=out / "missing",
            run="spike",
            reference_file="golden.npy",
            host_llvm_transform_chain=chain,
            host_llvm_link_qualification=q,
        )
    assert json.loads((out / "missing/baremetal_model.json").read_text())["status"] != "verified_complete_output"


def test_actual_supplier_refusal_precedes_target_run_and_success(tmp_path, monkeypatch):
    inputs, out = FX._fixture(tmp_path, monkeypatch)
    chain, doc = _selected_host_chain(tmp_path)
    q = _selected_host_qualification(chain, doc, tmp_path, monkeypatch, [])

    def build(bundle, work, **kw):
        r = FX._fake_build(bundle, work, **kw)
        (work / "compilation_recipe.json").write_text("{}")
        return r

    def refused(self, **kw):
        raise ValueError("changed final external provider")

    monkeypatch.setattr(S.ClosedCalleeQualification, "audit", refused)
    monkeypatch.setattr(BM.spike_model, "build", build)
    monkeypatch.setattr(BM.spike_model, "run", lambda *a, **kw: pytest.fail("target ran after failed provider gate"))
    with pytest.raises(ValueError, match="changed final external provider"):
        BM.compile_saved_model(
            **inputs,
            output=out / "failed",
            run="spike",
            reference_file="golden.npy",
            host_llvm_transform_chain=chain,
            host_llvm_link_qualification=q,
        )
    assert json.loads((out / "failed/baremetal_model.json").read_text())["status"] != "verified_complete_output"


def test_successful_current_audit_occurs_before_target_run(tmp_path, monkeypatch):
    inputs, out = FX._fixture(tmp_path, monkeypatch)
    chain, doc = _selected_host_chain(tmp_path)
    observations = []
    q = _selected_host_qualification(chain, doc, tmp_path, monkeypatch, observations)
    order = []

    def build(bundle, work, **kw):
        r = FX._fake_build(bundle, work, **kw)
        (work / "compilation_recipe.json").write_text("{}")
        return r

    def audit(self, **kw):
        self.record()
        order.append("actual current provider audit")
        return {"unit": "observed audit ordering only"}

    def run(*a, **kw):
        order.append("target run stub")
        return {"console": FX._console()}

    monkeypatch.setattr(S.ClosedCalleeQualification, "audit", audit)
    monkeypatch.setattr(BM.spike_model, "build", build)
    monkeypatch.setattr(BM.spike_model, "run", run)
    monkeypatch.setattr(BM, "_spike_extension_backend", lambda target: None)
    result = BM.compile_saved_model(
        **inputs,
        output=out / "success",
        run="spike",
        reference_file="golden.npy",
        host_llvm_transform_chain=chain,
        host_llvm_link_qualification=q,
    )
    assert result["status"] == "verified_complete_output" and order == [
        "actual current provider audit",
        "target run stub",
    ]
    assert result["output"]["host_provider_qualification"]
