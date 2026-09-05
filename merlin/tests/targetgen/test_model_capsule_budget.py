"""A whole-model capsule must be bounded by a wall-clock BUDGET, not by a per-step timeout.

MEASURED (gemmini arm-4 calibration ``merlincirct_defcal1``, 2026-08-29): the agent round finished
in 40 min; the whole-model capstone was then scheduled (its op-pass gate cleared at 18/22 = 0.82),
ran a cycle-accurate Verilator simulation of the entire model for 5 h 30 m past the round's own 4 h
timeout, wrote not one byte into its run directory, and the round never graded. Its ``--qa-timeout``
was 900 s -- a PER-STEP subprocess cap, which a grade making many such calls can never be bounded by.
"""
from __future__ import annotations

import json
import hashlib
import shutil
import time
from pathlib import Path

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_grade as CGR
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES


_EXACT_ARTIFACTS = ("input_interface", "command_buffer", "lowered_llvm", "kernel_object",
                    "package_kernel_elf", "instruction_trace")


def _copy_model_capsule(tmp_path, name="M3_host_island_seam_gemmini"):
    source = Path(__file__).parents[2] / "contract" / "capsules" / "model" / name
    copied = tmp_path / name
    shutil.copytree(source, copied)
    return CR.load_capsule(copied)


def test_frozen_capsule_runtime_bundle_is_stable_and_resolvable(tmp_path):
    """The public capsule, not a mutable recapture, supplies every whole-model runtime argument."""
    cap = _copy_model_capsule(tmp_path)
    identities = []
    for _ in range(2):
        with CR._model_runtime_bundle(cap, timeout=60) as (bundle, provenance, verify):
            from merlin.runtime.dispatch_runtime import resolve_forward_args

            args = resolve_forward_args(bundle)
            assert [list(a.shape) for a in args] == [[32], [32], [32, 32], [32, 32], [16, 32]]
            assert provenance["construction"] == "frozen_capsule_assets_v1"
            assert provenance["interface_reused_byte_exact"] is True
            assert provenance["live_recapture_used"] is False
            assert provenance["validation"]["golden_validated"] is True
            identities.append(provenance["bundle"]["content_sha256"])
            verify()
    assert identities[0] == identities[1], "npz/container timestamps must not change bundle identity"


def test_frozen_capsule_runtime_bundle_rejects_mutated_weights(tmp_path):
    """Matching shapes/names cannot spoof a model instance: loader and safetensor values must agree."""
    cap = _copy_model_capsule(tmp_path)
    weights = Path(cap["__dir__"]) / "capsule.weights.safetensors"
    mutated = bytearray(weights.read_bytes())
    mutated[-1] ^= 1
    weights.write_bytes(mutated)
    with pytest.raises(ValueError, match="validation failed"):
        with CR._model_runtime_bundle(cap, timeout=60):
            pass


def test_frozen_capsule_runtime_bundle_rejects_asset_spoof(tmp_path):
    """A declared loader may not escape the frozen capsule or arrive through a mutable symlink."""
    cap = _copy_model_capsule(tmp_path)
    cap["pytorch_ref"]["loader"] = "../capsule.pytorch.py"
    with pytest.raises(ValueError, match="must stay inside"):
        with CR._model_runtime_bundle(cap, timeout=60):
            pass


def test_frozen_capsule_runtime_bundle_rejects_missing_golden(tmp_path):
    cap = _copy_model_capsule(tmp_path)
    (Path(cap["__dir__"]) / "golden.yaml").unlink()
    with pytest.raises(ValueError, match="missing or a symlink"):
        with CR._model_runtime_bundle(cap, timeout=60):
            pass


def test_mesh_invocation_identity_binds_real_operands():
    from merlin import compile_cli

    a = compile_cli._mesh_invocation_id("mesh_layer_16x16x16_i8_i32", [[1]], [[2]])
    same = compile_cli._mesh_invocation_id("mesh_layer_16x16x16_i8_i32", [[1]], [[2]])
    different = compile_cli._mesh_invocation_id("mesh_layer_16x16x16_i8_i32", [[3]], [[2]])
    assert a == same
    assert a != different


def test_model_check_rejects_one_run_directory_with_two_contents():
    row = _valid_model_row()
    calls = [x for x in row["mesh_execution"]["dispatch_ledger"] if x["lane"] == "on_mesh"]
    calls[1]["cert_run_id"] = calls[0]["cert_run_id"]
    calls[1]["artifact_identity"]["run_id"] = calls[0]["artifact_identity"]["run_id"]
    check = CGR.model_execution_check(row, {"lanes": {"require": ["on_mesh"]}})
    assert check["status"] == "fail"
    assert "model_call_run_id_content_collision" in check["violations"]


def _identity(tag: str) -> dict:
    artifacts = {name: {"sha256": hashlib.sha256(f"{tag}:{name}".encode()).hexdigest(),
                        "size_bytes": len(tag) + len(name) + 1}
                 for name in _EXACT_ARTIFACTS}
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    return {"version": 1, "run_id": tag,
            "content_sha256": hashlib.sha256(canonical).hexdigest(),
            "artifacts": artifacts, "missing": []}


def _trace(identity: dict) -> dict:
    return {"required": True, "status": "pass", "drives_accelerator": True,
            "n_instructions": 7,
            "artifact_sha256": identity["artifacts"]["instruction_trace"]["sha256"]}


def _mesh_entry(ordinal: int, symbol: str) -> dict:
    identity = _identity(f"call-{ordinal}-{symbol}")
    return {"ordinal": ordinal, "symbol": symbol, "lane": "on_mesh", "status": "pass",
            "cert_run_id": identity["run_id"], "artifact_identity": identity,
            "trace_check": _trace(identity),
            "oracle_evidence": {"result": "pass", "derived_from_rtl": True,
                                "cycle_accurate": True, "kind": "rtl_gsim",
                                "engine": "gsim"}}


def _tile(tag: str) -> dict:
    identity = _identity(tag)
    return {"status": "pass", "oracle_result": "pass", "derived_from_rtl": True,
            "cycle_accurate": True, "oracle_engine": "gsim", "artifact_identity": identity,
            "trace_check": _trace(identity)}


def _valid_model_row() -> dict:
    ledger = [_mesh_entry(0, "mesh0"),
              {"ordinal": 1, "symbol": "host0", "lane": "scalar_rvv_lane", "status": "pass"},
              _mesh_entry(2, "mesh1")]
    return {
        "mesh_execution": {"matmul_layers_routed": 2, "matmul_layers_on_mesh": 2,
                           "matmul_layers_host_fallback": 0,
                           "matmul_layers_oracle_unavailable": 0,
                           "matmul_layers_unrouted": 0,
                           "simulator_requested": "gsim",
                           "mesh_route_symbols": ["mesh0", "mesh1"],
                           "dispatch_ledger": ledger},
        "mesh_tile_verification": {"n_tiles": 2, "n_passed": 2, "n_failed": 0,
                                   "n_unavailable": 0, "n_unsynthesizable": 0,
                                   "per_tile": [_tile("tile0"), _tile("tile1")]},
        "lane_report": {"required": ["on_mesh", "scalar_rvv_lane"],
                        "observed": ["on_mesh", "scalar_rvv_lane"], "unexercised": [],
                        "evidence": "dynamic_dispatch_ledger"},
        "boundary_expectation": {"boundary": "A->H->A", "contains": ["A->H->A"],
                                 "n_unresolved": 0},
        "boundary_execution": {"status": "pass", "boundary": "A->H->A",
                               "contains": ["A", "A->H->A"], "n_unresolved": 0},
    }


def _remove_scalar_lane(row: dict) -> None:
    ledger = row["mesh_execution"]["dispatch_ledger"]
    ledger.pop(1)
    ledger[1]["ordinal"] = 1
    row["lane_report"].update(observed=["on_mesh"], unexercised=["scalar_rvv_lane"])


def test_budget_is_unlimited_by_default(monkeypatch):
    """No ceiling unless one is asked for: an operator certification run legitimately takes hours."""
    monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    assert CR.model_budget_seconds() is None
    for bad in ("", "   ", "not-a-number", "0", "-5"):
        monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", bad)
        assert CR.model_budget_seconds() is None, bad
    monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", "90")
    assert CR.model_budget_seconds() == 90.0


def test_no_budget_runs_inline(monkeypatch):
    """With no budget the grade is the plain in-process call — no subprocess, no behaviour change."""
    seen = {}

    def _inline(capsule, *, target, timeout, package_dir):
        seen.update(capsule=capsule["name"], target=target, timeout=timeout)
        return {"capsule": capsule["name"], "status": "pass"}

    monkeypatch.setattr(CR, "_grade_model_capsule_inline", _inline)
    out = CR._grade_model_capsule({"name": "M0"}, target="t", timeout=7, budget_s=0)
    assert out["status"] == "pass"
    assert seen == {"capsule": "M0", "target": "t", "timeout": 7}


def test_budgeted_grade_pins_capsule_assets_before_starting_child(tmp_path, monkeypatch):
    """Cache GC may remove a materialized capsule while a long model child is still using it.

    The child must receive a private immutable copy, not the cache generation's path.  This is the
    production failure from ``merlincirct_arm4_func_20260901_codex1``: M3 started with a
    ``.gemmini.build.*`` ``__dir__`` and later failed opening ``capsule.yaml`` after cache GC removed
    that generation.
    """
    capsule = _copy_model_capsule(tmp_path / "source")
    original = Path(capsule["__dir__"])
    observed = {}

    class _FinishedChild:
        returncode = 0
        pid = 1

        def __init__(self, cmd, **_kwargs):
            spec_path = Path(cmd[cmd.index("--model-grade") + 1])
            out_path = Path(cmd[cmd.index("--model-grade-out") + 1])
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            pinned = Path(spec["capsule"]["__dir__"])
            observed.update(original=original, pinned=pinned)
            assert pinned != original
            shutil.rmtree(original)
            assert (pinned / "capsule.yaml").is_file()
            out_path.write_text(json.dumps({"capsule": capsule["name"], "status": "pass"}))

        def communicate(self, timeout=None):
            return ("", None)

    import subprocess as sp
    monkeypatch.setattr(sp, "Popen", _FinishedChild)
    out = CR._grade_model_capsule_unlocked(
        capsule, target="gemmini", timeout=10, budget_s=60)

    assert out["status"] == "pass"
    assert observed["pinned"].parent != observed["original"].parent


def test_suite_pins_model_assets_before_build_and_op_grading(tmp_path, monkeypatch):
    """The public-cache generation may be collected while the preceding op phase is still running.

    Pinning only when the model child starts is too late: ``run_suite`` discovers every capsule first,
    then can spend longer than the materializer's GC age building the package and grading ops before it
    reaches the models.
    """
    source = tmp_path / "source" / "M"
    source.mkdir(parents=True)
    (source / "capsule.yaml").write_text("name: M\nkind: model\n", encoding="utf-8")
    original = source.resolve()
    model = {"name": "M", "kind": "model", "__dir__": str(original),
             "gate": {"after_op_pass_fraction": 0.0}}
    op = {"name": "A", "kind": "isa"}
    observed = {}

    monkeypatch.setattr(CR, "load_package", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CR, "build_package", lambda *_args, **_kwargs: shutil.rmtree(original))
    monkeypatch.setattr(CR, "_split_ineligible", lambda caps, _target: (caps, []))
    monkeypatch.setattr(CR._tier_policy, "covering_set", lambda _caps: [])

    def _run(capsule, *_args, **_kwargs):
        if capsule["name"] == "A":
            return {"capsule": "A", "kind": "isa", "status": "pass"}
        pinned = Path(capsule["__dir__"])
        observed["pinned"] = pinned
        assert pinned != original
        assert (pinned / "capsule.yaml").is_file()
        return {"capsule": "M", "kind": "model", "status": "pass"}

    monkeypatch.setattr(CR, "run_capsule", _run)
    out = CR.run_suite([op, model], tmp_path / "package", runs_root=tmp_path / "runs",
                       target="gemmini", max_workers=1)

    assert [row["status"] for row in out] == ["pass", "pass"]
    assert "pinned" in observed
    assert not observed["pinned"].exists(), "suite pin must be cleaned after the model result lands"


@pytest.mark.parametrize("budget", [3.0])
def test_budget_stops_a_grade_that_overruns(tmp_path, monkeypatch, budget):
    """The ceiling is enforced on the CAPSULE. A grade that overruns is stopped and reported."""
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(600)\n")

    import subprocess as sp
    orig = sp.Popen

    def _fake_popen(cmd, **kw):
        # Same contract as the real child (own session, inherited env) but it never finishes.
        return orig(["python3", str(slow)], **kw)

    monkeypatch.setattr(sp, "Popen", _fake_popen)
    t0 = time.monotonic()
    out = CR._grade_model_capsule({"name": "GX0", "label": "public"},
                                  target="gemmini", timeout=900, budget_s=budget)
    elapsed = time.monotonic() - t0
    assert budget <= elapsed < budget + 20, elapsed
    assert out["status"] == "budget_exhausted"
    assert out["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert out["model_budget_s"] == budget
    assert "MERLIN_MODEL_BUDGET_S" in out["failure"]["detail"]


def test_the_result_schema_knows_the_status():
    """A status a runner can WRITE must be one the contract's own schema accepts."""
    from merlin.targetgen.contract import schemas
    schemas.validate({"capsule": "GX0", "status": "budget_exhausted", "contract_version": "0.1",
                      "tiers": {}, "trace_check": {"status": "skipped"},
                      "numeric": {"status": "skipped"},
                      "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                  "detail": "exceeded its budget"}}, "capsule_result")


def test_budget_exhausted_is_neither_numerator_nor_denominator():
    """A capsule stopped by its own clock measured NOTHING about the submission.

    Counting it as a failure is what makes ``all_pass`` unreachable, which disables an agent loop's
    only early exit and turns every run into a fixed-price purchase of its whole round budget.
    """
    assert "budget_exhausted" in NOT_MEASURED_STATUSES
    results = [{"capsule": "A0", "status": "pass"}, {"capsule": "A1", "status": "pass"},
               {"capsule": "GX0", "status": "budget_exhausted", "kind": "model"}]
    graded = [r for r in results if r.get("status") not in NOT_MEASURED_STATUSES]
    assert [r["capsule"] for r in graded] == ["A0", "A1"]
    assert sum(1 for r in graded if r["status"] == "pass") == len(graded)   # all_pass stays reachable


def test_the_stopped_capsule_is_reported_by_name(tmp_path, monkeypatch):
    """Excluded is not hidden: the real scorer names it, counts it, and says why it is not graded."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir(); caps.mkdir()
    results = [{"capsule": "A0", "label": "public", "status": "pass", "tiers": {}},
               {"capsule": "A1", "label": "public", "status": "pass", "tiers": {}},
               {"capsule": "GX0", "label": "public", "kind": "model", "tiers": {},
                "status": "budget_exhausted",
                "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                            "detail": "exceeded its 3600s budget"}}]
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: [])
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: results)

    score = CGR.grade(str(pkg), capsules_root=str(caps), runs_root=str(tmp_path / "runs"),
                      target="gemmini", contract=None, oracle_adapters={})

    assert score["n_budget_exhausted"] == 1
    assert score["budget_exhausted"] == ["GX0"]
    assert "MERLIN_MODEL_BUDGET_S" in score["budget_exhausted_note"]
    # neither numerator nor denominator -- and all_pass therefore still reachable
    assert score["n_capsules"] == 2 and score["n_passed"] == 2
    assert score["functional_pass"] == 1


def test_model_rows_use_execution_evidence_without_fabricating_an_instruction_trace(tmp_path, monkeypatch):
    """A mixed formal cohort has two honest structural obligations, covering one denominator."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir(); caps.mkdir()
    discovered = [
        {"name": "A0", "kind": "op", "label": "public"},
        {"name": "M2", "kind": "model", "label": "public"},
    ]
    results = [
        {"capsule": "A0", "kind": "op", "label": "public", "status": "pass",
         "numeric": {"status": "pass"}, "trace_check": {"status": "pass"},
         "tiers": {"L3": "pass"}},
        {"capsule": "M2", "kind": "model", "label": "public", "status": "pass",
         "numeric": {"status": "pass"}, "tiers": {"L3": "pass"}, **_valid_model_row()},
    ]
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: discovered)
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: results)
    monkeypatch.setattr(CGR.CV, "aggregate", lambda *a, **k: {
        "by_tier_reached": {}, "instruction_class_coverage": {}, "mode_coverage": {},
        "unavailable": {}, "acceleratable_coverage": {},
    })

    score = CGR.grade(str(pkg), capsules_root=str(caps), runs_root=str(tmp_path / "runs"),
                      target="gemmini", contract=None, oracle_adapters={})

    assert score["structural_evidence_scope"] == {
        "n_instruction_trace_capsules": 1, "n_model_execution_capsules": 1}
    assert score["trace_all_pass"] is True
    assert score["model_execution_all_pass"] is True
    assert score["structural_evidence_all_pass"] is True
    model = next(r for r in score["per_capsule"] if r.get("kind") == "model")
    assert model["trace"] is None, "do not fabricate a model-level instruction trace"
    assert model["model_execution_check"]["status"] == "pass"


def test_grade_persists_and_withholds_a_model_pass_from_the_wrong_required_engine(
        tmp_path, monkeypatch):
    """Durable QA must not retain ``status/L3=pass`` when the in-memory proof rejects the engine."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir(); caps.mkdir()
    capsule = {
        "name": "M3", "kind": "model", "label": "public",
        "required_oracle_tiers": ["L3"],
        "lanes": {"require": ["on_mesh", "scalar_rvv_lane"]},
    }
    result = {
        "capsule": "M3", "kind": "model", "label": "public", "status": "pass",
        "numeric": {"status": "pass"},
        "tiers": {"L3": {"status": "pass", "mandatory": True,
                           "derived_from_rtl": True, "cycle_accurate": True}},
        **_valid_model_row(),
    }
    result["mesh_execution"]["simulator_requested"] = "verilator"
    for entry in result["mesh_execution"]["dispatch_ledger"]:
        if entry["lane"] == "on_mesh":
            entry["oracle_evidence"].update(engine="verilator", kind="rtl_verilator")
    for tile in result["mesh_tile_verification"]["per_tile"]:
        tile.update(oracle_engine="verilator")

    runs_root = tmp_path / "grade"
    durable = runs_root / "runs" / "gemmini-capsule-bench" / "M3" / "capsule_result.json"
    durable.parent.mkdir(parents=True)
    durable.write_text(json.dumps(result))
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: [capsule])
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: [result])
    monkeypatch.setattr(CGR.CV, "aggregate", lambda *a, **k: {
        "by_tier_reached": {}, "instruction_class_coverage": {}, "mode_coverage": {},
        "unavailable": {}, "acceleratable_coverage": {},
    })

    score = CGR.grade(str(pkg), capsules_root=str(caps), runs_root=str(runs_root),
                      target="gemmini", contract=None, oracle_adapters={})
    persisted = json.loads(durable.read_text())

    assert persisted["model_execution_check"]["status"] == "fail"
    assert "model_requested_oracle_engine_differs_from_required_engine" in \
        persisted["model_execution_check"]["violations"]
    assert persisted["status"] == "incomplete"
    assert persisted["tiers"]["L3"]["status"] == "unavailable"
    assert score["functional_pass"] == 0
    assert score["gradeable"] is False


def test_model_execution_evidence_fails_closed_on_unmeasured_layer():
    row = _valid_model_row()
    row["mesh_execution"]["matmul_layers_oracle_unavailable"] = 1
    check = CGR.model_execution_check(row)
    assert check["status"] == "fail"
    assert "model_layer_oracle_unavailable" in check["violations"]


@pytest.mark.parametrize("mutation,capsule,reason", [
    (lambda r: r["mesh_execution"].update({"matmul_layers_unrouted": 1}), None,
     "model_contraction_layer_unrouted"),
    (lambda r: r.pop("lane_report", None), {"lanes": {"require": ["on_mesh"]}},
     "required_lane_report_missing"),
    (_remove_scalar_lane,
     {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}},
     "required_model_lane_unexercised"),
    (lambda r: r["mesh_tile_verification"].update({"per_tile": []}), None,
     "model_tile_evidence_missing_or_malformed"),
    (lambda r: r["mesh_tile_verification"]["per_tile"][0].update({"derived_from_rtl": False}),
     None, "model_tile_not_rtl_derived"),
    (lambda r: r["mesh_tile_verification"]["per_tile"][0].update({"cycle_accurate": False}),
     None, "model_tile_not_cycle_accurate"),
    (lambda r: r["mesh_tile_verification"].update({"n_passed": 1}), None,
     "not_all_model_tiles_certified"),
])
def test_model_execution_evidence_rejects_structural_mutations(mutation, capsule, reason):
    row = _valid_model_row()
    mutation(row)
    check = CGR.model_execution_check(row, capsule)
    assert check["status"] == "fail"
    assert reason in check["violations"]


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r["mesh_execution"].pop("dispatch_ledger"),
     "dynamic_dispatch_ledger_missing_or_malformed"),
    (lambda r: r["mesh_execution"]["dispatch_ledger"][0]["trace_check"].update(
        {"drives_accelerator": False}), "model_call_accelerator_trace_missing_or_invalid"),
    (lambda r: r["mesh_execution"]["dispatch_ledger"][0]["artifact_identity"]["artifacts"]
        ["lowered_llvm"].update({"sha256": "0" * 64}),
     "model_call_artifact_identity_digest_mismatch"),
    (lambda r: r["mesh_execution"]["dispatch_ledger"][0]["oracle_evidence"].update(
        {"cycle_accurate": False}), "model_call_oracle_fidelity_missing_or_invalid"),
    (lambda r: r["mesh_execution"]["dispatch_ledger"][0]["oracle_evidence"].update(
        {"engine": "verilator"}), "model_call_oracle_engine_mismatch"),
    (lambda r: r["mesh_tile_verification"]["per_tile"][0].update(
        {"oracle_engine": "verilator"}), "model_tile_oracle_engine_mismatch"),
    (lambda r: r["mesh_execution"].update({"simulator_requested": "verilator"}),
     "model_call_oracle_engine_mismatch"),
    (lambda r: r["mesh_execution"].update({"matmul_layers_on_mesh": 3}),
     "model_mesh_counter_ledger_mismatch"),
    (lambda r: r["mesh_tile_verification"]["per_tile"][0]["trace_check"].update(
        {"drives_accelerator": False}), "model_tile_accelerator_trace_missing_or_invalid"),
    (lambda r: r["boundary_execution"].update(
        {"boundary": "A->A", "contains": ["A->A"]}),
     "model_a_h_a_seam_not_exercised"),
])
def test_model_dynamic_proof_rejects_spoofed_missing_or_mutated_evidence(mutation, reason):
    row = _valid_model_row()
    mutation(row)
    check = CGR.model_execution_check(
        row, {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}})
    assert check["status"] == "fail"
    assert reason in check["violations"]


def test_gsim_is_classified_as_cycle_accurate_rtl_tool():
    from merlin.targetgen import oot_runner

    assert "gsim" in oot_runner._CYCLE_ACCURATE_SIMULATORS


def test_required_gsim_prevents_mixed_engine_model_cycle_rollup(monkeypatch):
    row = _valid_model_row()
    for entry in row["mesh_execution"]["dispatch_ledger"]:
        if entry["lane"] == "on_mesh":
            entry["oracle_evidence"]["cycles"] = 17
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")

    clean = CR._model_tier_evidence(
        row["mesh_execution"], row["mesh_tile_verification"])
    row["mesh_execution"]["dispatch_ledger"][0]["oracle_evidence"]["engine"] = "verilator"
    mixed = CR._model_tier_evidence(
        row["mesh_execution"], row["mesh_tile_verification"])

    assert clean["cycles"] == 34 and clean["cycle_accurate"] is True
    assert clean["measurement_conditions"]["single_engine_cycle_authority"] is True
    assert mixed["cycles"] is None and mixed["cycle_accurate"] is False
    assert mixed["measurement_conditions"]["single_engine_cycle_authority"] is False


def test_exact_cert_identity_changes_when_emitted_artifact_mutates(tmp_path):
    """A reused run pathname is not the identity; changing exact LLVM must change the content ID."""
    from types import SimpleNamespace
    from merlin.targetgen import oot_runner

    generated = tmp_path / "generated"
    generated.mkdir()
    names = {"input.interface.mlir", "command_buffer.json", "lowered.llvm.mlir", "kernel.o",
             "package_kernel.elf", "instruction_trace.json"}
    for name in names:
        (generated / name).write_bytes(("first:" + name).encode())
    paths = SimpleNamespace(generated=generated)
    before = oot_runner._cert_artifact_identity(paths, "same-run")
    (generated / "lowered.llvm.mlir").write_bytes(b"mutated exact llvm")
    after = oot_runner._cert_artifact_identity(paths, "same-run")
    assert before["run_id"] == after["run_id"]
    assert before["content_sha256"] != after["content_sha256"]
    assert (before["artifacts"]["lowered_llvm"]["sha256"]
            != after["artifacts"]["lowered_llvm"]["sha256"])


def test_exact_model_cert_rejects_cpu_only_lowered_llvm(tmp_path, monkeypatch):
    """Correct command-buffer math cannot replace evidence that the emitted artifact drives the mesh."""
    from types import SimpleNamespace
    from merlin.targetgen import oot_runner
    from merlin.targetgen import provenance
    from merlin.runtime import reference, simulator
    from merlin.runtime.backends import base as backends

    package = tmp_path / "pkg"
    package.mkdir()
    tool = package / "tool"
    tool.write_text("stub")
    iface = tmp_path / "model.interface.mlir"
    iface.write_text("module {}")
    pkg = SimpleNamespace(tool=tool, directory=package)

    def _entry(_pkg, name, _input, output_json=None, timeout=0):
        if name == "emit_command_buffer":
            output_json.write_text("{}")
            stdout = ""
        elif name == "lower_target_to_llvm":
            stdout = "module {}"  # deliberately no inline asm / custom opcode
        else:
            stdout = "module {}"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(oot_runner, "load_package", lambda *a, **k: pkg)
    monkeypatch.setattr(oot_runner, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(oot_runner, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(oot_runner, "run_entrypoint", _entry)
    monkeypatch.setattr(oot_runner.schemas, "validate_command_buffer", lambda *a, **k: None)
    monkeypatch.setattr(oot_runner.schemas, "validate", lambda *a, **k: None)
    monkeypatch.setattr(provenance, "toolchain_shas", lambda *a, **k: {})
    monkeypatch.setattr(reference, "reference_outputs", lambda *a, **k: {})
    monkeypatch.setattr(reference, "outputs_match", lambda *a, **k: True)
    monkeypatch.setattr(simulator, "simulate", lambda *a, **k: {"outputs": {}})
    monkeypatch.setattr(backends, "get_backend", lambda *_: SimpleNamespace(available=lambda _s: False))
    monkeypatch.setattr(oot_runner, "_record", lambda *a, **k: None)

    result = oot_runner.certify(package, iface, runs_root=tmp_path / "runs", run_id="cpu-only",
                                simulator="verilator", target="gemmini",
                                require_accelerator_trace=True)
    assert result["status"] == "fail"
    assert result["failure"]["plane"] == "trace_check"
    assert result["trace_check"]["status"] == "fail"
    trace = (tmp_path / "runs" / "runs" / "gemmini-contract" / "cpu-only" / "generated"
             / "instruction_trace.json")
    assert trace.is_file() and json.loads(trace.read_text())["instructions"] == []


class _StubPkg:
    integrity_exempt = False


def test_child_entry_writes_a_result(tmp_path, monkeypatch):
    """``--model-grade`` is the child half: one spec file in, one result file out."""
    spec, out = tmp_path / "spec.json", tmp_path / "res.json"
    spec.write_text(json.dumps({"capsule": {"name": "M0"}, "target": "gemmini",
                                "timeout": 30, "package_dir": None}))
    monkeypatch.setattr(CR, "_grade_model_capsule_inline",
                        lambda c, **kw: {"capsule": c["name"], "status": "pass"})
    assert CR.main(["--model-grade", str(spec), "--model-grade-out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"capsule": "M0", "status": "pass"}


def test_the_kill_reaches_a_grandchild_that_left_the_process_group(tmp_path):
    """A process GROUP is not the tree.

    MEASURED (2026-08-30): the budgeted grade's Verilator child had called ``setsid`` of its own, so it
    sat in neither the grade child's group nor its session. ``killpg`` reaped the python and left the
    simulator running with PPID 1, holding a core on a shared host.
    """
    import subprocess
    import sys as _sys

    # parent -> child that leaves the group and outlives a killpg
    script = tmp_path / "tree.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],\n"
        "                       start_new_session=True)\n"
        "print(kid.pid, flush=True)\n"
        "time.sleep(600)\n")
    proc = subprocess.Popen([_sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    grandchild = int(proc.stdout.readline().strip())
    assert grandchild in CR._descendants(proc.pid)

    t0 = time.monotonic()
    CR._kill_tree(proc.pid, grace=10.0)
    assert time.monotonic() - t0 < 8, "a zombie is not alive; the wait must not sit out its grace"
    proc.wait(timeout=10)
    assert not CR._running(grandchild), "the grandchild outside the group must be dead too"


def test_the_child_dies_when_its_parent_is_killed(tmp_path):
    """The budget protects against an overrun. It does nothing when the PARENT is killed -- the
    parent's cleanup never runs -- so the child asks the kernel to reap it instead.

    MEASURED (2026-08-30): stopping a regrade left the model-grade child and its Verilator alive on a
    shared host, reparented to init, with nothing left that knew to reap them. The guard therefore runs
    BEFORE merlin is imported; a first attempt that set it after the import left a multi-second window
    in which exactly this still happened, and this test caught it.
    """
    import subprocess
    import sys as _sys

    parent_py = tmp_path / "parent.py"
    parent_py.write_text(
        "import subprocess, sys, time\n"
        f"kid = subprocess.Popen([sys.executable, '-c', {CR._CHILD_GUARD + 'print(os.getpid(),flush=True);import time;time.sleep(600)'!r}],\n"
        "    start_new_session=True, stdout=subprocess.PIPE, text=True)\n"
        "print(kid.stdout.readline().strip(), flush=True)\n"   # the child prints once PROTECTED
        "time.sleep(600)\n")

    proc = subprocess.Popen([_sys.executable, str(parent_py)], stdout=subprocess.PIPE, text=True)
    try:
        child = int(proc.stdout.readline().strip())
        assert CR._running(child)
        proc.kill()
        proc.wait(timeout=10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and CR._running(child):
            time.sleep(0.2)
        assert not CR._running(child), (
            "the child outlived its parent — on a shared host that is an hours-long simulator that "
            "nothing is left to reap")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_guard_runs_before_merlin_is_imported():
    """The ordering IS the fix. If the guard ever moves after the import, the window comes back."""
    assert CR._CHILD_PREAMBLE.startswith(CR._CHILD_GUARD)
    assert "prctl" in CR._CHILD_GUARD
    assert CR._CHILD_GUARD.index("prctl") < CR._CHILD_PREAMBLE.index("merlin.targetgen")
