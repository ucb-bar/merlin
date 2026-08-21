"""A whole-model (kind == "model") capsule is graded by running the model ON THE TARGET'S MESH and gating
its output vs the model's golden.

The distinction these tests exist to hold: merlin's own x86 dispatch runtime is a useful DIAGNOSTIC and is
never an oracle for someone else's compiler. A `host` run therefore records what it saw and WITHHOLDS the
verdict (`not_gradeable_no_oracle`) — it can never produce a pass. Only a run whose matmul layers actually
executed on the target mesh can. Before this, a model capsule graded merlin's compiler against the torch
golden and reported `pass` with an empty `tiers` block, which is how a suite could read 40/40 while the
submitted backend never ran at all.
"""
from __future__ import annotations

import merlin.compile_cli as cc
from merlin.targetgen import capsule_runner as R

# Real model capsules declare [L0, L1, L2] (see merlin/contract/capsules/radiance/model/*/capsule.yaml).
# A fixture without them would make every fail-closed gate vacuous and hide exactly what these tests check.
_CAP = {"name": "M0_small_llama", "kind": "model", "label": "public",
        "required_oracle_tiers": ["L0", "L1", "L2"],
        "operation": {"op": "model", "attributes": {"model": "small_llama", "compile_dtype": "fp32"}}}
_CAP_INT8 = {"name": "M0_small_llama_int8", "kind": "model", "label": "public",
             "required_oracle_tiers": ["L0", "L1", "L2"],
             "operation": {"op": "model", "attributes": {"model": "small_llama", "compile_dtype": "int8"}}}


def _mesh(monkeypatch, *, ran, fell, status="verified", gate=True, verify=None, unrouted=0):
    """Patch compile_rvv to look like a mesh run that placed `ran` layers and fell back on `fell`."""
    out = {"status": status, "verify": verify if verify is not None else {"gate_ok": gate},
           "mesh_execution": {"target": "radiance", "matmul_layers_on_mesh": ran,
                              "matmul_layers_host_fallback": fell,
                              "matmul_layers_unrouted": unrouted}}
    monkeypatch.setattr(cc, "compile_rvv", lambda *a, **k: out)


# ---------------------------------------------------------------- host: a withheld verdict, never a pass

def test_a_host_reference_run_never_passes_a_model_capsule(monkeypatch):
    """The host dispatch runtime IS merlin's own compiler. However well it does, it is evidence about us,
    not about the submission — so the numeric verdict is withheld rather than granted."""
    monkeypatch.setattr(cc, "compile_rvv",
                        lambda *a, **k: {"status": "verified", "verify": {"gate_ok": True}})
    r = R._grade_model_capsule(_CAP, timeout=60)          # no target -> host
    assert r["status"] == "not_gradeable_no_oracle"
    assert r["numeric"]["status"] == "skipped"
    assert r["host_reference"]["status"] == "verified", "the diagnostic is recorded, not discarded"


def test_a_host_mismatch_is_also_withheld_not_blamed_on_the_submission(monkeypatch):
    """A host-side mismatch is a defect in OUR reference compiler. Reporting it as the capsule's `fail`
    would attribute it to a backend that never executed."""
    monkeypatch.setattr(cc, "compile_rvv",
                        lambda *a, **k: {"status": "run_mismatch", "verify": {"gate_ok": False}})
    r = R._grade_model_capsule(_CAP, timeout=60)
    assert r["status"] == "not_gradeable_no_oracle"
    assert r["host_reference"]["status"] == "run_mismatch"


# ---------------------------------------------------------------------- mesh: a real, earned verdict

def test_a_model_whose_layers_ran_on_the_mesh_and_matched_passes(monkeypatch):
    _mesh(monkeypatch, ran=15, fell=0)
    r = R._grade_model_capsule(_CAP, target="radiance", timeout=60)
    assert r["status"] == "pass"
    assert r["tiers"]["L2"]["status"] == "pass"
    assert r["tiers"]["L0"]["not_applicable"] and r["tiers"]["L1"]["not_applicable"], \
        "L0/L1 interpret a command buffer; a whole model has none — N/A, not 'did not run'"


def test_no_layer_on_the_mesh_is_not_a_pass(monkeypatch):
    """Every layer fell back to the host: the target never ran this model, so there is nothing to certify."""
    _mesh(monkeypatch, ran=0, fell=0)
    r = R._grade_model_capsule(_CAP, target="radiance", timeout=60)
    assert r["status"] != "pass"
    assert r["tiers"]["L2"]["status"] == "skipped"


def test_a_partial_fallback_is_not_a_pass(monkeypatch):
    """A layer the route CLAIMED and then could not run is a hole in the claim, not a rounding error."""
    _mesh(monkeypatch, ran=12, fell=3)
    r = R._grade_model_capsule(_CAP, target="radiance", timeout=60)
    assert r["status"] != "pass"
    assert r["tiers"]["L2"]["status"] == "fail"


def test_unreachable_mesh_counters_fail_closed(monkeypatch):
    """Counters absent means nobody could tell whether the mesh ran — which must not read as zero."""
    monkeypatch.setattr(cc, "compile_rvv",
                        lambda *a, **k: {"status": "verified", "verify": {"gate_ok": True}})
    r = R._grade_model_capsule(_CAP, target="radiance", timeout=60)
    assert r["status"] != "pass"
    assert r["tiers"]["L2"]["status"] == "unavailable"


def test_a_numeric_mismatch_on_the_mesh_is_a_real_fail(monkeypatch):
    _mesh(monkeypatch, ran=15, fell=0, status="run_mismatch", gate=False)
    r = R._grade_model_capsule(_CAP, target="radiance", timeout=60)
    assert r["status"] == "fail" and r["failure"]["category"] == "FUNCTIONAL_MISMATCH"


def test_toolchain_absent_is_incomplete_never_a_silent_pass(monkeypatch):
    def _boom(*a, **k):
        raise SystemExit("no clang-23 / no m2m venv")
    monkeypatch.setattr(cc, "compile_rvv", _boom)
    r = R._grade_model_capsule(_CAP, target="radiance", timeout=60)
    assert r["status"] == "incomplete" and r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_quantized_drop_within_tolerance_still_passes_when_it_ran_on_the_mesh(monkeypatch):
    """The quant-tolerance allowance survives — but it now also requires the layers to have RUN."""
    _mesh(monkeypatch, ran=15, fell=0, status="run_mismatch",
          verify={"gate_ok": False, "fp32_cos": 0.962})
    r = R._grade_model_capsule(_CAP_INT8, target="radiance", timeout=60)
    assert r["status"] == "pass" and r["numeric"]["quant_tolerance"]["cos"] == 0.962


def test_quantized_gross_defect_still_fails(monkeypatch):
    _mesh(monkeypatch, ran=15, fell=0, status="run_mismatch",
          verify={"gate_ok": False, "fp32_cos": 0.42})
    r = R._grade_model_capsule(_CAP_INT8, target="radiance", timeout=60)
    assert r["status"] == "fail" and r["failure"]["category"] == "FUNCTIONAL_MISMATCH"
