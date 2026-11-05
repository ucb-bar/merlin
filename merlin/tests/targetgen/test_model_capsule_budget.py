"""A whole-model capsule must be bounded by a wall-clock BUDGET, not by a per-step timeout.

MEASURED (gemmini arm-4 calibration ``merlincirct_defcal1``, 2026-08-29): the agent round finished
in 40 min; the whole-model capstone was then scheduled (its op-pass gate cleared at 18/22 = 0.82),
ran a cycle-accurate Verilator simulation of the entire model for 5 h 30 m past the round's own 4 h
timeout, wrote not one byte into its run directory, and the round never graded. Its ``--qa-timeout``
was 900 s -- a PER-STEP subprocess cap, which a grade making many such calls can never be bounded by.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_grade as CGR
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES

_EXACT_ARTIFACTS = (
    "input_interface",
    "command_buffer",
    "lowered_llvm",
    "kernel_object",
    "package_kernel_elf",
    "instruction_trace",
)


def _synthetic_model_capsule(tmp_path, monkeypatch):
    """Author a tiny identity capture with the production capsule writer.

    No mutable corpus answer surfaces or model download are used. The test-only
    validator below checks the frozen bytes and identity golden without claiming
    a real torch.export or accelerator approval.
    """
    import subprocess

    import numpy as np

    from merlin.targetgen import capsule_source as source

    tmp_path.mkdir(parents=True, exist_ok=True)
    loader = tmp_path / "identity_loader.py"
    loader.write_text("# Synthetic identity model fixture; never imported by the test worker.\n")
    weights = tmp_path / "identity.safetensors"
    weights.write_bytes((8).to_bytes(8, "little") + b"{}      ")
    manifest = tmp_path / "identity.safetensors.manifest.json"
    manifest.write_text(json.dumps({"0": {"kind": "input", "name": "I0", "shape": [2, 2]}}))
    linalg = (
        f'builtin.module attributes {{prov.weights_file = "{weights}"}} {{\n'
        "  func.func @forward(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
        "    func.return %arg0 : tensor<2x2xf32>\n  }\n}\n"
    )
    values = [[1.0, 2.0], [3.0, 4.0]]
    artifact = source.CapsuleArtifacts(
        op="model",
        dtype="f32",
        pytorch_src=loader.read_text(),
        linalg_mlir=linalg,
        inputs=[values],
        golden=values,
        weights_path=str(weights),
        meta={
            "weights_manifest": str(manifest),
            "input_abi": [{"shape": [2, 2], "dtype": "f32"}],
            "output_abi": [{"shape": [2, 2], "dtype": "f32"}],
        },
    )
    fake_capture = SimpleNamespace(m2m_dir=tmp_path, capture_loader=lambda *_a, **_k: artifact)
    binding = SimpleNamespace(
        operand_dtype="f32",
        target="synthetic",
        cap_dtype=lambda value: value,
        tiers=["L0", "L1"],
        compare="tolerance_float",
        atol=0.0,
        rtol=0.0,
    )
    monkeypatch.setattr(source, "derived_recipe", lambda *_a: None)
    monkeypatch.setattr(source, "model_accelerator_demand", lambda *_a: (None, []))
    destination = source.write_model_capsule(
        {
            "kind": "model",
            "cat": "model",
            "name": "SY_identity_bundle",
            "model": "synthetic_identity",
            "loader": str(loader),
        },
        binding,
        tmp_path / "capsules",
        source=fake_capture,
    )
    expected_weights = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setattr(source, "_m2m_python", lambda: Path(sys.executable))

    def validate_synthetic_worker(cmd, **_kwargs):
        request = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        frozen = Path(request["weights"])
        inputs = np.load(request["inputs_npz"], allow_pickle=False)
        golden = np.load(request["golden_npy"], allow_pickle=False)
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != expected_weights or not np.array_equal(
            inputs["in0"], golden
        ):
            return subprocess.CompletedProcess(cmd, 1, "", "synthetic fixture validation failed")
        report = {
            "manifest": json.loads(Path(request["captured_manifest"]).read_text()),
            "input_order": {"I0": 0},
            "loader_sha256": hashlib.sha256(Path(request["loader"]).read_bytes()).hexdigest(),
            "weights_sha256": expected_weights,
            "python": str(Path(sys.executable)),
            "python_version": sys.version.split()[0],
            "torch_version": None,
            "torch_export": False,
            "capture_manifest_validated": True,
            "loader_input_count": 1,
            "golden_validated": True,
            "golden_value_replay": True,
            "weights_validated_exact": True,
        }
        return subprocess.CompletedProcess(cmd, 0, "__CAPSULE_BUNDLE_ABI__ " + json.dumps(report) + "\n", "")

    monkeypatch.setattr(subprocess, "run", validate_synthetic_worker)
    return CR.load_capsule(destination)


def test_frozen_capsule_runtime_bundle_is_stable_and_resolvable(tmp_path, monkeypatch):
    """The public capsule, not a mutable recapture, supplies every whole-model runtime argument."""
    cap = _synthetic_model_capsule(tmp_path, monkeypatch)
    identities = []
    for _ in range(2):
        with CR._model_runtime_bundle(cap, timeout=60) as (bundle, provenance, verify):
            from merlin.runtime.dispatch_runtime import resolve_forward_args

            args = resolve_forward_args(bundle)
            assert [list(a.shape) for a in args] == [[2, 2]]
            assert provenance["construction"] == "frozen_capsule_assets_v2"
            assert provenance["interface_reused_byte_exact"] is True
            assert provenance["live_recapture_used"] is False
            assert provenance["validation"]["golden_validated"] is True
            identities.append(provenance["bundle"]["content_sha256"])
            verify()
    assert identities[0] == identities[1], "npz/container timestamps must not change bundle identity"


def test_frozen_capsule_runtime_bundle_rejects_mutated_weights(tmp_path, monkeypatch):
    """Matching shapes/names cannot spoof a model instance: loader and safetensor values must agree."""
    cap = _synthetic_model_capsule(tmp_path, monkeypatch)
    weights = Path(cap["__dir__"]) / "capsule.weights.safetensors"
    mutated = bytearray(weights.read_bytes())
    mutated[-1] = ord("\t")  # valid JSON padding, different frozen bytes
    weights.write_bytes(mutated)
    with pytest.raises(ValueError, match="validation failed"):
        with CR._model_runtime_bundle(cap, timeout=60):
            pass


def test_frozen_capsule_runtime_bundle_rejects_asset_spoof(tmp_path, monkeypatch):
    """A declared loader may not escape the frozen capsule or arrive through a mutable symlink."""
    cap = _synthetic_model_capsule(tmp_path, monkeypatch)
    cap["pytorch_ref"]["loader"] = "../capsule.pytorch.py"
    with pytest.raises(ValueError, match="must stay inside"):
        with CR._model_runtime_bundle(cap, timeout=60):
            pass


def test_frozen_capsule_runtime_bundle_rejects_missing_golden(tmp_path, monkeypatch):
    cap = _synthetic_model_capsule(tmp_path, monkeypatch)
    (Path(cap["__dir__"]) / "golden.yaml").unlink()
    with pytest.raises(ValueError, match="missing or a symlink"):
        with CR._model_runtime_bundle(cap, timeout=60):
            pass


def test_pinned_model_bundle_survives_source_generation_removal(tmp_path, monkeypatch):
    cap = _synthetic_model_capsule(tmp_path / "source", monkeypatch)
    pinned = CR._pin_model_capsule(cap, tmp_path / "private" / "capsule")
    shutil.rmtree(cap["__dir__"])
    with CR._model_runtime_bundle(pinned, timeout=60) as (_, provenance, verify):
        assert provenance["validation"]["golden_validated"] is True
        verify()


@pytest.mark.parametrize("alias", ["source_directory", "asset"])
def test_model_snapshot_does_not_launder_source_aliases(tmp_path, alias):
    source = tmp_path / "source"
    source.mkdir()
    (source / "capsule.yaml").write_text("name: M\nkind: model\n")
    if alias == "source_directory":
        selected = tmp_path / "alias"
        selected.symlink_to(source, target_is_directory=True)
    else:
        selected = source
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"not a declared capsule-local asset")
        (source / "capsule.weights.safetensors").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        CR._pin_model_capsule(
            {"name": "M", "kind": "model", "__dir__": str(selected)},
            tmp_path / "private" / "capsule",
        )


def test_retained_model_snapshot_root_rejects_alias(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "capsule.yaml").write_text("name: M\nkind: model\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(CR, "_run_suite", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="may not traverse a symlink"):
        CR.run_suite(
            [{"name": "M", "kind": "model", "__dir__": str(source)}],
            tmp_path / "package",
            runs_root=tmp_path / "runs",
            model_snapshot_root=alias,
        )
    assert list(outside.iterdir()) == []


def test_model_suite_cleanup_preserves_shared_store_permissions(tmp_path, monkeypatch):
    from merlin.common import content_store

    source = tmp_path / "source"
    source.mkdir()
    declaration = source / "capsule.yaml"
    declaration.write_text("name: M\nkind: model\n")
    declaration.chmod(0o444)
    source.chmod(0o555)
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(content_store, "store_root", lambda namespace=None: store)
    monkeypatch.setattr(CR, "_run_suite", lambda *_args, **_kwargs: [])

    CR.run_suite([{"name": "M", "kind": "model", "__dir__": str(source)}])

    objects = [path for path in store.rglob("*") if path.is_file()]
    assert objects
    assert all(path.stat().st_mode & 0o222 == 0 for path in objects)


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
    artifacts = {
        name: {"sha256": hashlib.sha256(f"{tag}:{name}".encode()).hexdigest(), "size_bytes": len(tag) + len(name) + 1}
        for name in _EXACT_ARTIFACTS
    }
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": 1,
        "run_id": tag,
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifacts": artifacts,
        "missing": [],
    }


def _trace(identity: dict) -> dict:
    return {
        "required": True,
        "status": "pass",
        "drives_accelerator": True,
        "n_instructions": 7,
        "artifact_sha256": identity["artifacts"]["instruction_trace"]["sha256"],
    }


def _mesh_entry(ordinal: int, symbol: str) -> dict:
    identity = _identity(f"call-{ordinal}-{symbol}")
    return {
        "ordinal": ordinal,
        "symbol": symbol,
        "lane": "on_mesh",
        "status": "pass",
        "cert_run_id": identity["run_id"],
        "artifact_identity": identity,
        "trace_check": _trace(identity),
        "oracle_evidence": {
            "result": "pass",
            "derived_from_rtl": True,
            "cycle_accurate": True,
            "kind": "rtl_gsim",
            "engine": "gsim",
        },
    }


def _tile(tag: str) -> dict:
    identity = _identity(tag)
    return {
        "status": "pass",
        "oracle_result": "pass",
        "derived_from_rtl": True,
        "cycle_accurate": True,
        "oracle_engine": "gsim",
        "artifact_identity": identity,
        "trace_check": _trace(identity),
    }


def _valid_model_row() -> dict:
    ledger = [
        _mesh_entry(0, "mesh0"),
        {"ordinal": 1, "symbol": "host0", "lane": "scalar_rvv_lane", "status": "pass"},
        _mesh_entry(2, "mesh1"),
    ]
    return {
        "mesh_execution": {
            "matmul_layers_routed": 2,
            "matmul_layers_on_mesh": 2,
            "matmul_layers_host_fallback": 0,
            "matmul_layers_oracle_unavailable": 0,
            "matmul_layers_unrouted": 0,
            "simulator_requested": "gsim",
            "mesh_route_symbols": ["mesh0", "mesh1"],
            "dispatch_ledger": ledger,
        },
        "mesh_tile_verification": {
            "n_tiles": 2,
            "n_passed": 2,
            "n_failed": 0,
            "n_unavailable": 0,
            "n_unsynthesizable": 0,
            "n_screened": 2,
            "n_screen_passed": 2,
            "n_screen_failed": 0,
            "n_screen_unavailable": 0,
            "per_tile": [_tile("tile0"), _tile("tile1")],
        },
        "lane_report": {
            "required": ["on_mesh", "scalar_rvv_lane"],
            "observed": ["on_mesh", "scalar_rvv_lane"],
            "unexercised": [],
            "scope": ["on_mesh", "scalar_rvv_lane"],
            "evidence": {
                "on_mesh": "dynamic_dispatch_ledger",
                "scalar_rvv_lane": "dynamic_dispatch_ledger",
            },
        },
        "boundary_expectation": {"boundary": "A->H->A", "contains": ["A->H->A"], "n_unresolved": 0},
        "boundary_execution": {
            "status": "pass",
            "boundary": "A->H->A",
            "contains": ["A", "A->H->A"],
            "n_unresolved": 0,
        },
    }


def test_native_host_call_is_accounted_without_claiming_target_host_execution():
    row = _valid_model_row()
    row["mesh_execution"]["dispatch_ledger"][1].update(
        lane="native_cpu", placement="host", executor="native_cpu", target_executed=False
    )
    check = CGR.model_execution_check(row, {"lanes": {"require": ["on_mesh"]}})
    assert "dynamic_dispatch_ledger_entry_missing_or_malformed" not in check["violations"]
    assert "required_model_lane_unexercised" not in check["violations"]


def test_must_accelerate_does_not_grade_the_legacy_outline():
    from merlin.targetgen import capsule_grade as CGR

    capsule = {"semantic": {"must_accelerate": True}}
    row = _valid_model_row()
    missing = CGR.model_execution_check(row, capsule)
    assert missing["kind"] == "candidate_whole_program_execution"
    assert "candidate_source_placement_unverified" in missing["violations"]
    assert "planned_outlined_alignment_unverified" not in missing["violations"]
    assert "dynamic_dispatch_ledger_missing_or_malformed" not in missing["violations"]
    withheld = CGR.enforce_model_execution_check({**row, "status": "pass", "tiers": {}}, capsule, target="gemmini")
    assert withheld["status"] == "incomplete" and withheld["failure"]["plane"] == "candidate_model_execution"

    alignment = {
        "schema": "planned_outlined_alignment_v1",
        "evidence": "runtime_normalized_source_and_outlined_dispatches; static_only",
        "n_planned_accelerator_stages": 1,
        "split_stages": [{"stage": "elementwise"}],
        "unjoined_stages": [],
        "unresolved_groups": [],
        "status": "split",
    }
    row["mesh_execution"]["planned_outlined_alignment"] = alignment
    split = CGR.model_execution_check(row, capsule)
    assert split["violations"] == missing["violations"]

    row["mesh_execution"]["planned_outlined_alignment"] = {**alignment, "status": "matched", "split_stages": []}
    matched = CGR.model_execution_check(row, capsule)
    assert matched["violations"] == missing["violations"]


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

    def _inline(capsule, *, target, timeout, package_dir, numeric_policy, numeric_policy_identity):
        assert numeric_policy is None and numeric_policy_identity is None
        seen.update(capsule=capsule["name"], target=target, timeout=timeout)
        return {"capsule": capsule["name"], "status": "pass"}

    monkeypatch.setattr(CR, "_grade_model_capsule_inline", _inline)
    out = CR._grade_model_capsule({"name": "M0"}, target="t", timeout=7, budget_s=0, numeric_policy=None)
    assert out["status"] == "pass"
    assert seen == {"capsule": "M0", "target": "t", "timeout": 7}


def test_budgeted_grade_pins_capsule_assets_before_starting_child(tmp_path, monkeypatch):
    """Cache GC may remove a materialized capsule while a long model child is still using it.

    The child must receive a private immutable copy, not the cache generation's path.  This is the
    production failure from ``merlincirct_arm4_func_20260901_codex1``: M3 started with a
    ``.gemmini.build.*`` ``__dir__`` and later failed opening ``capsule.yaml`` after cache GC removed
    that generation.
    """
    capsule = _synthetic_model_capsule(tmp_path / "source", monkeypatch)
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
    out = CR._grade_model_capsule_unlocked(capsule, target="gemmini", timeout=10, budget_s=60, numeric_policy=None)

    assert out["status"] == "pass"
    assert observed["pinned"].parent != observed["original"].parent


def test_candidate_grade_uses_the_same_budgeted_child_and_pinned_source(tmp_path, monkeypatch):
    capsule = _synthetic_model_capsule(tmp_path / "source", monkeypatch)
    original = Path(capsule["__dir__"])
    context = {
        "runs_root": str(tmp_path / "runs"),
        "run_id": "one-model",
        "suite": "synthetic",
        "dtype": "f32",
        "contract": None,
        "fourth_output_name": "lowered.llvm.mlir",
    }
    observed = {}

    class _FinishedChild:
        returncode = 0
        pid = 1

        def __init__(self, cmd, **_kwargs):
            spec = json.loads(Path(cmd[cmd.index("--model-grade") + 1]).read_text())
            pinned = Path(spec["capsule"]["__dir__"])
            observed.update(context=spec["candidate_context"], pinned=pinned)
            assert spec["candidate_context"] == context
            assert spec["numeric_policy"] is None
            assert pinned != original and (pinned / "capsule.yaml").is_file()
            Path(cmd[cmd.index("--model-grade-out") + 1]).write_text(
                json.dumps(
                    {"status": "incomplete", "model_execution_check": {"kind": "candidate_whole_program_execution"}}
                )
            )

        def communicate(self, timeout=None):
            assert timeout == 11
            return ("", None)

    import subprocess as sp

    monkeypatch.setattr(sp, "Popen", _FinishedChild)
    result = CR._grade_model_capsule_unlocked(
        capsule, target="synthetic", timeout=7, budget_s=11, candidate_context=context
    )
    assert result["status"] == "incomplete"
    assert result["model_execution_check"]["kind"] == "candidate_whole_program_execution"
    assert observed["pinned"].parent != original.parent


def test_candidate_budget_child_retains_suite_owned_source_pin(tmp_path, monkeypatch):
    capsule = _synthetic_model_capsule(tmp_path / "source", monkeypatch)
    pinned = Path(capsule["__dir__"])
    source_sha = hashlib.sha256((pinned / "capsule.yaml").read_bytes()).hexdigest()
    context = {
        "runs_root": str(tmp_path / "runs"),
        "run_id": "M",
        "suite": "synthetic",
        "dtype": "f32",
        "contract": None,
        "fourth_output_name": "lowered.llvm.mlir",
    }

    class _FinishedChild:
        returncode = 0
        pid = 1

        def __init__(self, cmd, **_kwargs):
            spec = json.loads(Path(cmd[cmd.index("--model-grade") + 1]).read_text())
            assert spec["capsule"]["__dir__"] == str(pinned)
            Path(cmd[cmd.index("--model-grade-out") + 1]).write_text(
                json.dumps({"status": "incomplete", "candidate_source_pin": str(pinned / "capsule.yaml")})
            )

        def communicate(self, timeout=None):
            return ("", None)

    import subprocess as sp

    monkeypatch.setattr(sp, "Popen", _FinishedChild)
    result = CR._grade_model_capsule_unlocked(
        capsule,
        target="synthetic",
        timeout=7,
        budget_s=11,
        candidate_context=context,
        candidate_source_pinned=CR._SUITE_MODEL_PIN,
    )
    source_pin = Path(result["candidate_source_pin"])
    assert source_pin.is_file(), "the candidate source pin must survive child return and postcheck"
    assert hashlib.sha256(source_pin.read_bytes()).hexdigest() == source_sha


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
    model = {
        "name": "M",
        "kind": "model",
        "__dir__": str(original),
        "gate": {"after_op_pass_fraction": 0.0},
        "semantic": {"must_accelerate": True},
    }
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
        assert capsule["__suite_model_pin"] is CR._SUITE_MODEL_PIN
        assert pinned != original
        assert (pinned / "capsule.yaml").is_file()
        return {"capsule": "M", "kind": "model", "status": "pass"}

    monkeypatch.setattr(CR, "run_capsule", _run)
    out = CR.run_suite([op, model], tmp_path / "package", runs_root=tmp_path / "runs", target="gemmini", max_workers=1)

    assert [row["status"] for row in out] == ["pass", "pass"]
    assert "pinned" in observed
    assert not observed["pinned"].exists(), "suite pin must be cleaned after the model result lands"


def test_grade_rechecks_durable_model_source_pin_after_suite_returns(tmp_path, monkeypatch):
    """The score's second native audit must rehash the same source bytes after suite return."""
    from merlin.targetgen import capsule_grade as grade_module
    from merlin.targetgen.native_model_execution import _pinned_native_file

    source = tmp_path / "source" / "M"
    source.mkdir(parents=True)
    (source / "capsule.yaml").write_text("name: M\nkind: model\n", encoding="utf-8")
    (source / "capsule.interface.mlir").write_bytes(b"module { func.func @entry() { return } }\n")
    (source / "golden.yaml").write_text("withheld: [1, 2, 3]\n", encoding="utf-8")
    capsule = {"name": "M", "kind": "model", "label": "public", "__dir__": str(source)}
    package = tmp_path / "package"
    package.mkdir()
    runs_root = tmp_path / "runs"
    observed = {}

    monkeypatch.setattr(grade_module, "load_package", lambda *a, **k: SimpleNamespace(integrity_exempt=False))
    monkeypatch.setattr(grade_module, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(grade_module, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(grade_module.CR, "discover_capsules", lambda *a, **k: [capsule])

    def suite(frozen, *_args, **_kwargs):
        pinned = Path(frozen[0]["__dir__"]) / "capsule.interface.mlir"
        observed["pin"] = pinned
        return [
            {
                "capsule": "M",
                "kind": "model",
                "label": "public",
                "status": "incomplete",
                "tiers": {},
                "candidate_emission": {
                    "source_interface": {
                        "path": str(pinned),
                        "sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
                        "size_bytes": pinned.stat().st_size,
                    }
                },
            }
        ]

    monkeypatch.setattr(grade_module.CR, "_run_suite", suite)
    monkeypatch.setattr(
        grade_module.CV,
        "aggregate",
        lambda *a, **k: {
            "by_tier_reached": {},
            "instruction_class_coverage": {},
            "mode_coverage": {},
            "unavailable": {},
            "acceleratable_coverage": {},
        },
    )

    def second_audit(result, *_args, **_kwargs):
        observed["rehash"] = _pinned_native_file(result["candidate_emission"], "source_interface")
        return result

    monkeypatch.setattr(grade_module, "enforce_model_execution_check", second_audit)
    original_bytes = (source / "capsule.interface.mlir").read_bytes()
    grade_module.grade(
        package,
        capsules_root=source.parent,
        runs_root=runs_root,
        target="synthetic",
        oracle_adapters={},
        model_snapshot_root=runs_root / ".private_model_sources",
    )
    assert observed["rehash"] == observed["pin"]
    assert observed["pin"].is_file(), "native source pin must remain inspectable after grading"
    first_pin = observed["pin"]
    assert first_pin.read_bytes() == original_bytes
    assert first_pin.stat().st_mode & 0o222 == 0
    assert first_pin.parent.stat().st_mode & 0o777 == 0o700
    (source / "capsule.interface.mlir").write_bytes(b"module { func.func @changed() { return } }\n")
    grade_module.grade(
        package,
        capsules_root=source.parent,
        runs_root=runs_root,
        target="synthetic",
        oracle_adapters={},
        model_snapshot_root=runs_root / ".private_model_sources",
    )
    assert observed["pin"] != first_pin
    assert observed["pin"].read_bytes() != original_bytes
    assert first_pin.read_bytes() == original_bytes, "later attempts may not rewrite earlier native source pins"

    # An agent-facing diagnostic gets the exact same second audit while its
    # source closure is temporary; no golden is retained in its output tree.
    diagnostic_root = tmp_path / "selfcheck_out"
    grade_module.grade(
        package,
        capsules_root=source.parent,
        runs_root=diagnostic_root,
        target="synthetic",
        oracle_adapters={},
    )
    assert observed["rehash"] == observed["pin"]
    assert not observed["pin"].exists()
    assert not list(diagnostic_root.rglob("golden.yaml"))


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
    # Exercise the real lock and child supervisor without contending with an
    # operator's live target grade: lock admission precedes the capsule timer.
    target = f"budget-test-{tmp_path.name}"
    out = CR._grade_model_capsule(
        {"name": "GX0", "label": "public"}, target=target, timeout=900, budget_s=budget, numeric_policy=None
    )
    elapsed = time.monotonic() - t0
    assert budget <= elapsed < budget + 20, elapsed
    assert out["status"] == "budget_exhausted"
    assert out["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert out["model_budget_s"] == budget
    assert "MERLIN_MODEL_BUDGET_S" in out["failure"]["detail"]


def test_the_result_schema_knows_the_status():
    """A status a runner can WRITE must be one the contract's own schema accepts."""
    from merlin.targetgen.contract import schemas

    schemas.validate(
        {
            "capsule": "GX0",
            "status": "budget_exhausted",
            "contract_version": "0.1",
            "tiers": {},
            "trace_check": {"status": "skipped"},
            "numeric": {"status": "skipped"},
            "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS", "detail": "exceeded its budget"},
        },
        "capsule_result",
    )


def test_budget_exhausted_is_neither_numerator_nor_denominator():
    """A capsule stopped by its own clock measured NOTHING about the submission.

    Counting it as a failure is what makes ``all_pass`` unreachable, which disables an agent loop's
    only early exit and turns every run into a fixed-price purchase of its whole round budget.
    """
    assert "budget_exhausted" in NOT_MEASURED_STATUSES
    results = [
        {"capsule": "A0", "status": "pass"},
        {"capsule": "A1", "status": "pass"},
        {"capsule": "GX0", "status": "budget_exhausted", "kind": "model"},
    ]
    graded = [r for r in results if r.get("status") not in NOT_MEASURED_STATUSES]
    assert [r["capsule"] for r in graded] == ["A0", "A1"]
    assert sum(1 for r in graded if r["status"] == "pass") == len(graded)  # all_pass stays reachable


def test_the_stopped_capsule_is_reported_by_name(tmp_path, monkeypatch):
    """Excluded is not hidden: the real scorer names it, counts it, and says why it is not graded."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir()
    caps.mkdir()
    results = [
        {"capsule": "A0", "label": "public", "status": "pass", "tiers": {}},
        {"capsule": "A1", "label": "public", "status": "pass", "tiers": {}},
        {
            "capsule": "GX0",
            "label": "public",
            "kind": "model",
            "tiers": {},
            "status": "budget_exhausted",
            "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS", "detail": "exceeded its 3600s budget"},
        },
    ]
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: [])
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: results)

    score = CGR.grade(
        str(pkg),
        capsules_root=str(caps),
        runs_root=str(tmp_path / "runs"),
        target="gemmini",
        contract=None,
        oracle_adapters={},
    )

    assert score["n_budget_exhausted"] == 1
    assert score["budget_exhausted"] == ["GX0"]
    assert "MERLIN_MODEL_BUDGET_S" in score["budget_exhausted_note"]
    # neither numerator nor denominator -- and all_pass therefore still reachable
    assert score["n_capsules"] == 2 and score["n_passed"] == 2
    assert score["functional_pass"] == 1


def test_model_rows_use_execution_evidence_without_fabricating_an_instruction_trace(tmp_path, monkeypatch):
    """A mixed formal cohort has two honest structural obligations, covering one denominator."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir()
    caps.mkdir()
    discovered = [
        {"name": "A0", "kind": "op", "label": "public"},
        {"name": "M2", "kind": "model", "label": "public"},
    ]
    results = [
        {
            "capsule": "A0",
            "kind": "op",
            "label": "public",
            "status": "pass",
            "numeric": {"status": "pass"},
            "trace_check": {"status": "pass"},
            "tiers": {"L3": "pass"},
        },
        {
            "capsule": "M2",
            "kind": "model",
            "label": "public",
            "status": "pass",
            "numeric": {"status": "pass"},
            "tiers": {"L3": "pass"},
            **_valid_model_row(),
        },
    ]
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: discovered)
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: results)
    monkeypatch.setattr(
        CGR.CV,
        "aggregate",
        lambda *a, **k: {
            "by_tier_reached": {},
            "instruction_class_coverage": {},
            "mode_coverage": {},
            "unavailable": {},
            "acceleratable_coverage": {},
        },
    )

    score = CGR.grade(
        str(pkg),
        capsules_root=str(caps),
        runs_root=str(tmp_path / "runs"),
        target="gemmini",
        contract=None,
        oracle_adapters={},
    )

    assert score["structural_evidence_scope"] == {"n_instruction_trace_capsules": 1, "n_model_execution_capsules": 1}
    assert score["trace_all_pass"] is True
    assert score["model_execution_all_pass"] is True
    assert score["structural_evidence_all_pass"] is True
    model = next(r for r in score["per_capsule"] if r.get("kind") == "model")
    assert model["trace"] is None, "do not fabricate a model-level instruction trace"
    assert model["model_execution_check"]["status"] == "pass"


@pytest.mark.parametrize("rtl_tiers", [frozenset({"L3"}), frozenset()])
def test_grade_persists_and_withholds_a_model_pass_from_the_wrong_required_engine(tmp_path, monkeypatch, rtl_tiers):
    """Durable QA must not retain ``status/L3=pass`` when the in-memory proof rejects the engine."""
    pkg, caps = tmp_path / "pkg", tmp_path / "caps"
    pkg.mkdir()
    caps.mkdir()
    capsule = {
        "name": "M3",
        "kind": "model",
        "label": "public",
        "required_oracle_tiers": ["L3"],
        "lanes": {"require": ["on_mesh", "scalar_rvv_lane"]},
    }
    result = {
        "capsule": "M3",
        "kind": "model",
        "label": "public",
        "status": "pass",
        "numeric": {"status": "pass"},
        "tiers": {"L3": {"status": "pass", "mandatory": True, "derived_from_rtl": True, "cycle_accurate": True}},
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
    monkeypatch.setattr(CGR.CR, "_rtl_tiers_of", lambda _target: rtl_tiers)
    monkeypatch.setattr(CGR, "load_package", lambda d, contract=None: _StubPkg())
    monkeypatch.setattr(CGR, "integrity_scan", lambda p: None)
    monkeypatch.setattr(CGR, "build_package", lambda p: None)
    monkeypatch.setattr(CGR.CR, "discover_capsules", lambda *a, **k: [capsule])
    monkeypatch.setattr(CGR.CR, "run_suite", lambda *a, **k: [result])
    monkeypatch.setattr(
        CGR.CV,
        "aggregate",
        lambda *a, **k: {
            "by_tier_reached": {},
            "instruction_class_coverage": {},
            "mode_coverage": {},
            "unavailable": {},
            "acceleratable_coverage": {},
        },
    )

    score = CGR.grade(
        str(pkg), capsules_root=str(caps), runs_root=str(runs_root), target="gemmini", contract=None, oracle_adapters={}
    )
    persisted = json.loads(durable.read_text())

    assert persisted["model_execution_check"]["status"] == "fail"
    assert (
        "model_requested_oracle_engine_differs_from_required_engine" in persisted["model_execution_check"]["violations"]
    )
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


@pytest.mark.parametrize(
    "mutation,capsule,reason",
    [
        (lambda r: r["mesh_execution"].update({"matmul_layers_unrouted": 1}), None, "model_contraction_layer_unrouted"),
        (lambda r: r.pop("lane_report", None), {"lanes": {"require": ["on_mesh"]}}, "required_lane_report_missing"),
        (
            _remove_scalar_lane,
            {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}},
            "required_model_lane_unexercised",
        ),
        (
            lambda r: r["mesh_tile_verification"].update({"per_tile": []}),
            None,
            "model_tile_evidence_missing_or_malformed",
        ),
        (
            lambda r: r["mesh_tile_verification"]["per_tile"][0].update({"derived_from_rtl": False}),
            None,
            "model_tile_not_rtl_derived",
        ),
        (
            lambda r: r["mesh_tile_verification"]["per_tile"][0].update({"cycle_accurate": False}),
            None,
            "model_tile_not_cycle_accurate",
        ),
        (lambda r: r["mesh_tile_verification"].update({"n_passed": 1}), None, "not_all_model_tiles_certified"),
    ],
)
def test_model_execution_evidence_rejects_structural_mutations(mutation, capsule, reason):
    row = _valid_model_row()
    mutation(row)
    check = CGR.model_execution_check(row, capsule)
    assert check["status"] == "fail"
    assert reason in check["violations"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r["mesh_execution"].pop("dispatch_ledger"), "dynamic_dispatch_ledger_missing_or_malformed"),
        (
            lambda r: r["mesh_execution"]["dispatch_ledger"][0]["trace_check"].update({"drives_accelerator": False}),
            "model_call_accelerator_trace_missing_or_invalid",
        ),
        (
            lambda r: r["mesh_execution"]["dispatch_ledger"][0]["artifact_identity"]["artifacts"][
                "lowered_llvm"
            ].update({"sha256": "0" * 64}),
            "model_call_artifact_identity_digest_mismatch",
        ),
        (
            lambda r: r["mesh_execution"]["dispatch_ledger"][0]["oracle_evidence"].update({"cycle_accurate": False}),
            "model_call_oracle_fidelity_missing_or_invalid",
        ),
        (
            lambda r: r["mesh_execution"]["dispatch_ledger"][0]["oracle_evidence"].update({"engine": "verilator"}),
            "model_call_oracle_engine_mismatch",
        ),
        (
            lambda r: r["mesh_tile_verification"]["per_tile"][0].update({"oracle_engine": "verilator"}),
            "model_tile_oracle_engine_mismatch",
        ),
        (
            lambda r: r["mesh_execution"].update({"simulator_requested": "verilator"}),
            "model_call_oracle_engine_mismatch",
        ),
        (lambda r: r["mesh_execution"].update({"matmul_layers_on_mesh": 3}), "model_mesh_counter_ledger_mismatch"),
        (
            lambda r: r["mesh_tile_verification"]["per_tile"][0]["trace_check"].update({"drives_accelerator": False}),
            "model_tile_accelerator_trace_missing_or_invalid",
        ),
        (
            lambda r: r["boundary_execution"].update({"boundary": "A->A", "contains": ["A->A"]}),
            "model_a_h_a_seam_not_exercised",
        ),
    ],
)
def test_model_dynamic_proof_rejects_spoofed_missing_or_mutated_evidence(mutation, reason):
    row = _valid_model_row()
    mutation(row)
    check = CGR.model_execution_check(row, {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}})
    assert check["status"] == "fail"
    assert reason in check["violations"]


def test_gsim_is_classified_as_cycle_accurate_rtl_tool():
    from merlin.targetgen import oot_runner

    assert "gsim" in oot_runner._CYCLE_ACCURATE_SIMULATORS


def test_required_gsim_rejects_mixed_engine_model_evidence(monkeypatch):
    row = _valid_model_row()
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")

    clean = CGR.model_execution_check(row, {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}})
    row["mesh_execution"]["dispatch_ledger"][0]["oracle_evidence"]["engine"] = "verilator"
    mixed = CGR.model_execution_check(row, {"lanes": {"require": ["on_mesh", "scalar_rvv_lane"]}})

    assert clean["status"] == "pass"
    assert mixed["status"] == "fail"
    assert "model_call_oracle_engine_mismatch" in mixed["violations"]


def test_exact_cert_identity_changes_when_emitted_artifact_mutates(tmp_path):
    """A reused run pathname is not the identity; changing exact LLVM must change the content ID."""
    from types import SimpleNamespace

    from merlin.targetgen import oot_runner

    generated = tmp_path / "generated"
    generated.mkdir()
    names = {
        "input.interface.mlir",
        "command_buffer.json",
        "lowered.llvm.mlir",
        "kernel.o",
        "package_kernel.elf",
        "instruction_trace.json",
    }
    for name in names:
        (generated / name).write_bytes(("first:" + name).encode())
    paths = SimpleNamespace(generated=generated)
    before = oot_runner._cert_artifact_identity(paths, "same-run")
    (generated / "lowered.llvm.mlir").write_bytes(b"mutated exact llvm")
    after = oot_runner._cert_artifact_identity(paths, "same-run")
    assert before["run_id"] == after["run_id"]
    assert before["content_sha256"] != after["content_sha256"]
    assert before["artifacts"]["lowered_llvm"]["sha256"] != after["artifacts"]["lowered_llvm"]["sha256"]


def test_exact_model_cert_rejects_cpu_only_lowered_llvm(tmp_path, monkeypatch):
    """Correct command-buffer math cannot replace evidence that the emitted artifact drives the mesh."""
    from types import SimpleNamespace

    from merlin.runtime import reference, simulator
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import oot_runner, provenance
    from merlin.targetgen.rocc import decode as rocc_decode

    package = tmp_path / "pkg"
    package.mkdir()
    tool = package / "tool"
    tool.write_text("stub")
    iface = tmp_path / "model.interface.mlir"
    iface.write_text("module {}")
    pkg = SimpleNamespace(tool=tool, directory=package, manifest={})

    def _entry(_pkg, name, _input, output_json=None, timeout=0, write_bytecode=False):
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
    # This test isolates the certification guard; target ISA extraction has its
    # own tests and is not available from the synthetic package in this fixture.
    monkeypatch.setattr(rocc_decode, "decode_text", lambda *_a, **_k: {"instructions": []})
    monkeypatch.setattr(oot_runner, "_record", lambda *a, **k: None)

    result = oot_runner.certify(
        package,
        iface,
        runs_root=tmp_path / "runs",
        run_id="cpu-only",
        simulator="verilator",
        target="gemmini",
        require_accelerator_trace=True,
    )
    assert result["status"] == "fail"
    assert result["failure"]["plane"] == "trace_check"
    assert result["trace_check"]["status"] == "fail"
    trace = tmp_path / "runs" / "runs" / "gemmini-contract" / "cpu-only" / "generated" / "instruction_trace.json"
    assert trace.is_file() and json.loads(trace.read_text())["instructions"] == []


def test_native_model_console_requires_done_and_every_declared_output():
    """The public backend protocol and independent golden reject partial completion."""
    from merlin.runtime.backends.base import parse_console
    from merlin.targetgen.capsule_golden import compare

    with pytest.raises(RuntimeError, match="did not reach DONE"):
        parse_console("OUT Y0 1 1 7\n")
    with pytest.raises(RuntimeError, match="expected 2 values, got 1"):
        parse_console("OUT Y0 1 2 7\nDONE\n")

    outputs, _ = parse_console("OUT Y0 1 1 7\nDONE\n")
    numeric = compare({"Y0": [[7]], "Y1": [[9]]}, outputs, {"compare": "exact_int"})
    assert numeric["status"] == "fail"
    assert numeric["per_output"]["Y1"]["reason"] == "missing from observed"


class _StubPkg:
    integrity_exempt = False


def test_child_entry_writes_a_result(tmp_path, monkeypatch):
    """``--model-grade`` is the child half: one spec file in, one result file out."""
    spec, out = tmp_path / "spec.json", tmp_path / "res.json"
    spec.write_text(json.dumps({"capsule": {"name": "M0"}, "target": "gemmini", "timeout": 30, "package_dir": None}))
    monkeypatch.setattr(CR, "_grade_model_capsule_inline", lambda c, **kw: {"capsule": c["name"], "status": "pass"})
    assert CR.main(["--model-grade", str(spec), "--model-grade-out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"capsule": "M0", "status": "pass"}


def test_candidate_child_entry_runs_candidate_only(tmp_path, monkeypatch):
    spec, out = tmp_path / "spec.json", tmp_path / "res.json"
    spec.write_text(
        json.dumps(
            {
                "capsule": {"name": "M0"},
                "target": "synthetic",
                "timeout": 30,
                "package_dir": str(tmp_path),
                "candidate_context": {"run_id": "M0"},
            }
        )
    )
    monkeypatch.setattr(CR, "_grade_model_capsule_inline", lambda *a, **kw: pytest.fail("legacy child ran"))
    monkeypatch.setattr(
        CR,
        "_grade_candidate_model_capsule_inline",
        lambda c, **kw: {"capsule": c["name"], "status": "incomplete", "context": kw["context"]},
    )
    assert CR.main(["--model-grade", str(spec), "--model-grade-out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"capsule": "M0", "status": "incomplete", "context": {"run_id": "M0"}}


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
        "time.sleep(600)\n"
    )
    proc = subprocess.Popen([_sys.executable, str(script)], stdout=subprocess.PIPE, text=True, start_new_session=True)
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
        "print(kid.stdout.readline().strip(), flush=True)\n"  # the child prints once PROTECTED
        "time.sleep(600)\n"
    )

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
            "nothing is left to reap"
        )
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_guard_runs_before_merlin_is_imported():
    """The ordering IS the fix. If the guard ever moves after the import, the window comes back."""
    assert CR._CHILD_PREAMBLE.startswith(CR._CHILD_GUARD)
    assert "prctl" in CR._CHILD_GUARD
    assert CR._CHILD_GUARD.index("prctl") < CR._CHILD_PREAMBLE.index("merlin.targetgen")
