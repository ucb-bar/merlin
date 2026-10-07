"""Fail-closed private complete-network gate and host-only input declarations."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_full_models as gate

from merlin.targetgen.sandbox import bwrap, host_surfaces
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_declares_every_complete_program():
    descriptor = Path(__file__).resolve().parents[3] / "examples/gemmini/target/descriptor.yaml"
    assert gate.requirements_for(descriptor) == ("resnet50", "smolvla", "tiny_llama")
    assert gate.program_requirements_for(descriptor) == {
        "resnet50": ("model",),
        "smolvla": ("prefix_encode", "flow_denoise", "action_decode"),
        "tiny_llama": ("prefill", "decode"),
    }
    assert "M2M_LLAMA_LAYERS" in gate.loader_env_requirements_for(descriptor)["tiny_llama"]["forbidden"]
    smol = gate.loader_env_requirements_for(descriptor)["smolvla"]
    assert smol["required"]["M2M_SMOLVLA_PAPER_READY"] == "1"
    assert smol["selected_roles"] == ("checkpoint", "input")


def test_private_capture_records_synthetic_input_without_accuracy_claim(tmp_path):
    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "meta.json").write_text(
        json.dumps(
            {
                "loader_provenance_status": "declared",
                "loader_paper_ready": False,
                "loader_provenance": {
                    "synthetic_inputs": True,
                    "input_source": "synthetic_seed",
                    "input_sha256": "a" * 64,
                    "input_path": "/private/not-for-feedback.npz",
                },
            }
        )
    )
    observed = gate._captured_input_provenance([("model", stage)], {"paper_ready": False, "synthetic_inputs": True})[
        "model"
    ]
    assert observed["paper_ready"] is False and observed["synthetic_inputs"] is True
    assert observed["scope"].endswith("no paper accuracy or full-model numerical result")
    assert "input_path" not in observed
    with pytest.raises(ValueError, match="differs"):
        gate._captured_input_provenance([("model", stage)], {"paper_ready": True})
    (stage / "meta.json").write_text(
        json.dumps(
            {
                "loader_provenance_status": "declared",
                "loader_paper_ready": False,
                "loader_provenance": {
                    "synthetic_tokens": True,
                    "token_source": "synthetic_seed_0",
                    "token_sha256": "b" * 64,
                },
            }
        )
    )
    token_observed = gate._captured_input_provenance(
        [("prefill", stage)], {"paper_ready": False, "synthetic_inputs": True}
    )["prefill"]
    assert token_observed["input_source"] == "synthetic_seed_0"
    assert token_observed["input_sha256"] == "b" * 64
    assert gate._expected_input_provenance({"input_provenance": {"paper_ready": True, "synthetic_inputs": False}}) == {
        "paper_ready": True,
        "synthetic_inputs": False,
    }


def test_complete_requires_current_candidate_all_programs_and_device_work():
    programs = {"a": ("prefix", "decode"), "b": ("model",)}

    def model(name, stages):
        return {
            "model": name,
            "status": "pass",
            "checks": {
                "recipe_derivation": {
                    "evidence_manifest_sha256": "5" * 64,
                    "recipe_sha256": "6" * 64,
                    "recipe_semantic_sha256": "7" * 64,
                    "scope": "current-spec diagnostic recipe derivation only; no model or hardware admission",
                },
                "input_provenance": {
                    stage: {
                        "paper_ready": None,
                        "synthetic_inputs": None,
                        "meta_sha256": "a" * 64,
                        "scope": "input provenance only; no paper accuracy or full-model numerical result",
                    }
                    for stage in stages
                },
                "build": {
                    "status": "capture_lower_codegen_link_verified",
                    "candidate_tree_sha256": "1" * 64,
                    "static_build_board": "build-only-board",
                    "static_build_board_catalog_source": "/operator/board-catalog.yaml",
                    "static_build_board_catalog_sha256": "8" * 64,
                    "static_build_board_dts_source": "/operator/selected.dts",
                    "static_build_board_dts_sha256": "9" * 64,
                    "static_build_board_scope": gate.BUILD_BOARD_SCOPE,
                    "accelerator_rtl_facts_sha256": "a" * 64,
                    "accelerator_rtl_config": "fixture-accelerator-config",
                    "linked_device_groups": len(stages),
                    "programs": [
                        {
                            "program": stage,
                            "status": "capture_lower_codegen_link_verified",
                            "candidate_tree_sha256": "1" * 64,
                            "elf_sha256": "2" * 64,
                            "linked_device_groups": 1,
                            "static_host_compute_audit": [
                                {
                                    "verdict": "clean_static_host_compute_audit",
                                    "artifact_sha256": "3" * 64,
                                    "object_sha256": "4" * 64,
                                    "audit": {
                                        "budget": {"arithmetic_per_element": 1.0},
                                        "groups": [{"verdict": "clean"}],
                                    },
                                }
                            ],
                        }
                        for stage in stages
                    ],
                },
            },
        }

    record = {
        "schema": gate.RESULT_SCHEMA,
        "passed": True,
        "candidate_tree_sha256": "1" * 64,
        "required_models": ["a", "b"],
        "required_programs": {name: list(stages) for name, stages in programs.items()},
        "models": [model(name, stages) for name, stages in programs.items()],
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
    }
    kwargs = {"required_models": ("a", "b"), "required_programs": programs, "candidate_sha256": "1" * 64}
    assert gate.complete(record, **kwargs)
    assert not gate.complete(record, **{**kwargs, "candidate_sha256": "3" * 64})
    record["models"][0]["checks"]["build"]["programs"].pop()
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][1]["checks"]["build"]["linked_device_groups"] = 0
    assert not gate.complete(record, **kwargs)
    record["models"][1] = model("b", programs["b"])
    record["models"][1]["checks"]["build"]["static_build_board_scope"] = "executed"
    assert not gate.complete(record, **kwargs)


def test_official_grade_refuses_manifest_selected_narrow_roster(tmp_path):
    from merlin_experiments.phase1.feedback.certification import _official_grade_result

    (tmp_path / "submission").mkdir()
    (tmp_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "completion": {
                    "formal_grade_complete": True,
                    "required_tier": "L3",
                    "required_full_models": ["a"],
                    "required_full_programs": {"a": ["model"]},
                }
            }
        )
    )
    result = _official_grade_result(
        0,
        tmp_path,
        required_models=("a", "b"),
        required_programs={"a": ("model",), "b": ("prefix", "decode")},
    )
    assert "private_full_model_roster_mismatch" in result["failures"]


def test_built_group_artifact_vetoes_candidate_host_arithmetic(tmp_path, monkeypatch):
    from merlin.llvmlower import device_shim

    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _target: SimpleNamespace(symbol="kernel"))
    device = tmp_path / "device"
    device.mkdir()
    body = "\n".join(
        f'    %v{index} = "llvm.fmul"(%v{index - 1}, %v{index - 1}) : (f32, f32) -> f32' for index in range(1, 65)
    )
    artifact = device / "stem.device.mlir"
    artifact.write_text(
        "\n".join(
            [
                '"builtin.module"() ({',
                '  "llvm.func"() <{sym_name = "kernel", function_type = !llvm.func<void ()>}> ({',
                "  ^entry:",
                '    %v0 = "llvm.mlir.constant"() <{value = 1.000000e+00 : f32}> : () -> f32',
                body,
                '    "llvm.return"() : () -> ()',
                "  }) : () -> ()",
                "}) : () -> ()",
            ]
        )
    )
    (device / "stem.ll").write_text("linked source\n")
    (device / "stem.o").write_bytes(b"linked object")
    with pytest.raises(Exception, match="compute on the host"):
        gate._audit_built_device_host_compute(
            tmp_path,
            [{"symbol": "stem", "group": 1}],
            {"eligible_group_metrics": {1: {"elements": 64, "element_bytes": 4}}},
            "fixture",
        )


def test_multi_program_capture_refuses_slice_and_extra_stage(tmp_path):
    capture = tmp_path / "capture"
    stage_root = capture / "stages"
    names = ("prefix", "decode")
    rows = []
    for name in names:
        stage = stage_root / name
        stage.mkdir(parents=True)
        (stage / "capture_receipt.json").write_text("{}")
        rows.append({"name": name, "ok": True, "opaque": 0, "receipt_sha256": _sha(stage / "capture_receipt.json")})
    contract = {
        "version": 2,
        "stages": list(names),
        "programs": [{"name": name, "bundle": f"stages/{name}"} for name in names],
        "bindings": [{"name": "state"}],
    }
    (capture / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    receipt = {
        "schema": "merlin.model_session_capture.v1",
        "session_contract_sha256": _sha(capture / "session_contract.yaml"),
        "programs": rows,
    }
    (capture / "session-receipt.json").write_text(json.dumps(receipt))
    assert [name for name, _ in gate._captured_programs(capture, names)] == list(names)
    with pytest.raises(ValueError, match="roster"):
        gate._captured_programs(capture, ("decode",))
    (stage_root / "unlisted").mkdir()
    with pytest.raises(ValueError, match="extra"):
        gate._captured_programs(capture, names)


def test_single_complete_capture_may_carry_version_one_execution_contract(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}")
    (capture / "capture_receipt.json").write_text("{}")
    (capture / "session_contract.yaml").write_text("version: 1\nkind: image_stream\nstages: [classify]\n")
    assert gate._captured_programs(capture, ("model",)) == [("model", capture)]
    (capture / "session_contract.yaml").write_text("version: 2\nstages: [classify]\n")
    with pytest.raises(ValueError, match="not version 1"):
        gate._captured_programs(capture, ("model",))
    (capture / "session_contract.yaml").unlink()
    (capture / "stages").mkdir()
    with pytest.raises(ValueError, match="unbound stage"):
        gate._captured_programs(capture, ("model",))


def test_unknown_support_required_operation_is_not_waived():
    assert gate._noncompute_support({"disposition": "support_required", "mlir_operation": "tensor.extract"})
    assert not gate._noncompute_support({"disposition": "support_required", "mlir_operation": "linalg.copy"})
    with pytest.raises(ValueError, match="audited lowering class"):
        gate._noncompute_support({"disposition": "support_required", "mlir_operation": "mystery.perform"})
    with pytest.raises(ValueError, match="audited lowering class"):
        gate._noncompute_support({"disposition": "support_required", "mlir_operation": "tensor.future_unknown"})


def test_selected_input_roster_requires_transitive_dependency_exactly():
    selected = [
        {"role": "checkpoint", "kind": "tree", "guest_member": "hf-hub/main", "tree": {"sha256": "1" * 64}},
        {"role": "input", "kind": "file", "guest_member": "corpus/input.npz", "sha256": "2" * 64},
        {"role": "input", "kind": "tree", "guest_member": "hf-hub/transitive", "tree": {"sha256": "3" * 64}},
    ]
    expected = [
        {"role": "checkpoint", "kind": "tree", "guest_member": "hf-hub/main", "sha256": "1" * 64},
        {"role": "input", "kind": "file", "guest_member": "corpus/input.npz", "sha256": "2" * 64},
        {"role": "input", "kind": "tree", "guest_member": "hf-hub/transitive", "sha256": "3" * 64},
    ]
    gate._require_selected_input_bindings({"selected_input_bindings": expected}, {"selected_inputs": selected})
    with pytest.raises(ValueError, match="omits or changes"):
        gate._require_selected_input_bindings({"selected_input_bindings": expected[:-1]}, {"selected_inputs": selected})
    with pytest.raises(ValueError, match="omits or changes"):
        gate._require_selected_input_bindings(
            {"selected_input_bindings": [*expected[:-1], {**expected[-1], "sha256": "4" * 64}]},
            {"selected_inputs": selected},
        )


def test_static_build_receipt_binds_board_without_execution_claim(tmp_path):
    catalog = tmp_path / "board-catalog.yaml"
    catalog.write_text("selected board\n")
    dts = tmp_path / "selected.dts"
    dts.write_text("actual source DTS\n")
    inputs = {
        "target": "fixture",
        "board": "large-static-board",
        "board_catalog": str(catalog),
        "board_catalog_sha256": _sha(catalog),
        "dts": str(dts),
        "dts_sha256": _sha(dts),
        "run": "none",
    }
    kwargs = {
        "target": "fixture",
        "board": "large-static-board",
        "catalog": catalog,
        "dts": dts,
    }
    gate._require_static_build_inputs({"inputs": inputs}, **kwargs)
    with pytest.raises(ValueError, match="another board"):
        gate._require_static_build_inputs({"inputs": {**inputs, "run": "gsim"}}, **kwargs)
    with pytest.raises(ValueError, match="another board"):
        gate._require_static_build_inputs({"inputs": {**inputs, "dts_sha256": "3" * 64}}, **kwargs)
    with pytest.raises(ValueError, match="another board"):
        gate._require_static_build_inputs({"inputs": {**inputs, "dts": "/unrelated/other.dts"}}, **kwargs)


def test_recipe_derivation_binds_current_sources_and_selected_capture_recipe(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import evidence as evidence_api

    from merlin.targetgen.quant_recipe import digest as recipe_digest

    root = tmp_path / "derived"
    root.mkdir()
    manifest = root / "evidence-manifest.json"
    manifest.write_text("verified by canonical evidence loader")
    body = {"schema": "quant_recipe_v1", "target": "fixture", "format": "int8", "why": {}}
    body["recipe_sha256"] = recipe_digest(body)
    raw = json.dumps(body, sort_keys=True).encode()
    selected = tmp_path / "selected-recipe.json"
    selected.write_bytes(raw)
    member = f"software/quantization-recipes/{_sha(selected)}.json"
    index = {
        "schema": "merlin.phase0.capture_recipes.v1",
        "target": "fixture",
        "software_review": "reviewed",
        "recipes": [{"path": member, "sha256": _sha(selected), "recipe_sha256": body["recipe_sha256"]}],
    }
    sources = {
        "software-spec": "1" * 64,
        "target-contract": "2" * 64,
        "rtl-facts": "3" * 64,
    }
    exported = SimpleNamespace(
        target="fixture",
        source_snapshots=[SimpleNamespace(role=role, sha256=sha) for role, sha in sources.items()],
        archived_artifacts=[
            ("software/quantization-recipes.json", json.dumps(index).encode()),
            (member, raw),
        ],
    )
    monkeypatch.setattr(evidence_api, "load_exported_evidence", lambda _root: exported)
    row = {
        "recipe_derivation_root": str(root),
        "recipe_derivation_manifest_sha256": _sha(manifest),
    }
    plan = {"recipe": {"path": str(selected), "sha256": _sha(selected), "recipe_sha256": body["recipe_sha256"]}}
    kwargs = {
        "target": "fixture",
        "software_sha256": sources["software-spec"],
        "capability_sha256": sources["target-contract"],
        "facts_sha256": sources["rtl-facts"],
    }
    result = gate._recipe_derivation(tmp_path / "operator.yaml", row, plan, **kwargs)
    assert result["recipe_semantic_sha256"] == body["recipe_sha256"]
    assert result["evidence_manifest_sha256"] == _sha(manifest)
    with pytest.raises(ValueError, match="selected software-spec"):
        gate._recipe_derivation(tmp_path / "operator.yaml", row, plan, **{**kwargs, "software_sha256": "4" * 64})
    with pytest.raises(ValueError, match="absent from current-spec derivation"):
        gate._recipe_derivation(
            tmp_path / "operator.yaml",
            row,
            {"recipe": {**plan["recipe"], "recipe_sha256": "5" * 64}},
            **kwargs,
        )
    selected.write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        gate._recipe_derivation(tmp_path / "operator.yaml", row, plan, **kwargs)


def test_private_paths_are_denied_without_copying_capture_into_candidate(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import capture_selection

    selection = tmp_path / "selection" / "capture-selection.json"
    selection.parent.mkdir()
    selection.write_text("{}")
    source = tmp_path / "private-source"
    source.mkdir()
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"private")
    recipe = tmp_path / "selected-recipe.json"
    recipe.write_text("{}")
    evidence = tmp_path / "recipe-derivation"
    evidence.mkdir()
    (evidence / "evidence-manifest.json").write_text("{}")
    run = tmp_path / "sealed-run"
    selected = {
        "schema": capture_selection.SCHEMA_V2,
        "run_dir": str(run),
        "plan": {
            "m2m_root": str(source),
            "workload_root": str(source),
            "selected_inputs": [
                {
                    "source": str(checkpoint),
                    "kind": "file",
                    "role": "checkpoint",
                    "guest_member": "weights.safetensors",
                    "sha256": _sha(checkpoint),
                }
            ],
            "loader_env": {"MODEL_MODE": "complete"},
            "recipe": {"path": str(recipe), "sha256": _sha(recipe), "recipe_sha256": "a" * 64},
        },
    }
    monkeypatch.setattr(capture_selection, "load", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(gate, "_recipe_derivation", lambda *_args, **_kwargs: {})
    for name in ("facts.json", "software.yaml", "capability.yaml"):
        (tmp_path / name).write_text("public target contract\n")
    row = {
        "id": "complete",
        "capture_selection": str(selection),
        "capture_selection_sha256": _sha(selection),
        "capture": str(run / "capture"),
        "capture_execution_attestation": str(run / "attestation.json"),
        "recipe_derivation_root": str(evidence),
        "recipe_derivation_manifest_sha256": _sha(evidence / "evidence-manifest.json"),
        "rtl_facts": str(tmp_path / "facts.json"),
        "rtl_facts_sha256": _sha(tmp_path / "facts.json"),
        "host_package": str(tmp_path / "host"),
        "software_spec": str(tmp_path / "software.yaml"),
        "software_spec_sha256": _sha(tmp_path / "software.yaml"),
        "capability_contract": str(tmp_path / "capability.yaml"),
        "capability_contract_sha256": _sha(tmp_path / "capability.yaml"),
        "host_capabilities": str(tmp_path / "host.yaml"),
        "board_catalog": str(tmp_path / "boards.yaml"),
        "host_dts": str(tmp_path / "host.dts"),
        "deployment_dtype": "int8",
        "input_provenance": {"paper_ready": False, "synthetic_inputs": True},
        "selected_input_bindings": [
            {
                "role": "checkpoint",
                "kind": "file",
                "guest_member": "weights.safetensors",
                "sha256": _sha(checkpoint),
            }
        ],
    }
    spec = tmp_path / "operator.yaml"
    spec.write_text(yaml.safe_dump({"schema": gate.SCHEMA, "target": "fixture", "models": [row]}))
    paths = gate.private_input_paths(spec, target="fixture", required_models=("complete",))
    assert {entry["path"] for entry in paths} >= {
        str(source),
        str(checkpoint),
        str(run),
        str(selection.parent),
        str(evidence),
    }
    assert str(row["software_spec"]) not in {entry["path"] for entry in paths}
    assert str(row["rtl_facts"]) not in {entry["path"] for entry in paths}
    assert str(row["host_package"]) not in {entry["path"] for entry in paths}
    ws = tmp_path / "authoring" / "workspace"
    ws.mkdir(parents=True)
    argv = bwrap.base_argv(
        ws,
        {"allowed": [], "denied": [], "private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    assert str(source) not in argv  # default-denied scratch needs no invalid nested mount
    assert str(run) not in argv  # future output is default-denied, never copied or mounted
    run.mkdir()  # once the parent exists, a broad grant must mask this exact directory
    exposed = bwrap.base_argv(
        ws,
        {"allowed": [{"path": str(tmp_path)}], "denied": [], "private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    assert exposed[exposed.index(str(source)) - 1 : exposed.index(str(source)) + 1] == ["--tmpfs", str(source)]
    assert str(checkpoint) in exposed
    public_spec = Path(row["software_spec"])
    public_spec.write_text("public target contract\n")
    bwrap.base_argv(
        ws,
        {"allowed": [{"path": str(public_spec)}], "denied": [], "private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    alias = tmp_path / "runtime-alias"
    surfaces = host_surfaces.host_input_surfaces(
        ["--ro-bind", str(tmp_path), str(alias)],
        ws,
        {"private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    assert alias / "private-source" in {surface.path for surface in surfaces}
    assert alias / "weights.safetensors" in {surface.path for surface in surfaces}

    from merlin_experiments.phase1 import corpus_inputs as CI

    descriptor = tmp_path / "target.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "phase1_gates": {
                    "private_full_models": {
                        "required": True,
                        "models": ["complete"],
                        "programs": {"complete": ["model"]},
                        "source_workload_dirs": {"complete": source.name},
                        "required_selected_roles": {"complete": ["checkpoint"]},
                        "deployment_dtypes": {"complete": "int8"},
                        "required_loader_env": {"complete": {"MODEL_MODE": "complete"}},
                    }
                },
            }
        )
    )
    authored = tmp_path / "authored.yaml"
    authored_bundle = {"bundle_id": "fixture", "allowed": [], "denied": []}
    authored.write_text(yaml.safe_dump(authored_bundle))
    host_run = tmp_path / "host-run"
    host_run.mkdir()
    monkeypatch.setattr(CI, "stage", lambda *_args, **_kwargs: (dict(authored_bundle), {"fixture": True}))
    prepared = CI.prepare_bundle(
        host_run,
        SimpleNamespace(path=descriptor, target="fixture"),
        authored,
        authored_bundle,
        contract=tmp_path,
        private_full_model_spec=spec,
    )
    assert prepared.private_full_model_record["source"] == str(spec)
    assert prepared.private_full_model_record["path"] in [entry["path"] for entry in prepared.bundle["host_inputs"]]
    assert prepared.bundle["allowed"] == []
    assert {row["path"] for row in prepared.bundle["private_validation_paths"]} >= {
        str(host_run / "grading_private_full_models"),
        str(host_run / "run_manifest.yaml"),
        str(host_run / "environment.yaml"),
    }
    bwrap.materialize_bundle_inputs(ws, prepared.bundle, repo=tmp_path)
    [frozen] = bwrap.snapshot_input_paths(
        ws,
        prepared.bundle,
        [Path(prepared.private_full_model_record["path"])],
        repo=tmp_path,
    )
    assert _sha(frozen) == _sha(spec)


def test_live_bwrap_masks_resumed_private_grade_under_broad_grant(tmp_path):
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    public = tmp_path / "public.txt"
    public.write_text("public\n")
    private = tmp_path / "host-run" / "grading_private_full_models"
    private.mkdir(parents=True)
    (private / "private-result.json").write_text('{"model": "private"}\n')
    manifest = tmp_path / "host-run" / "run_manifest.yaml"
    manifest.write_text("private: true\n")
    bundle = {
        "allowed": [{"path": str(tmp_path)}],
        "denied": [],
        "private_validation_paths": [
            {"path": str(private), "kind": "dir"},
            {"path": str(manifest), "kind": "file"},
        ],
    }
    argv = bwrap.base_argv(
        workspace,
        bundle,
        repo=tmp_path,
        _policy_test_live_inputs=True,
        include_claude_home=False,
        inherit_environment=False,
    )
    alias = Path("/tmp/private-gate-alias")
    argv += ["--dir", str(alias), "--ro-bind", str(tmp_path), str(alias)]
    argv = bwrap.apply_answer_masks(
        argv,
        host_surfaces.host_input_surfaces(argv, workspace, bundle, repo=tmp_path, _policy_test_live_inputs=True),
    )
    probe = subprocess.run(
        [
            *argv,
            "/bin/sh",
            "-c",
            (
                f'test -s "{public}" && test -s "{alias / public.name}" '
                f'&& test ! -s "{private}/private-result.json" '
                f'&& test ! -s "{alias / private.relative_to(tmp_path) / "private-result.json"}" '
                f'&& test ! -s "{manifest}" '
                f'&& test ! -s "{alias / manifest.relative_to(tmp_path)}"'
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "Creating new namespace failed: Operation not permitted" in probe.stderr:
        pytest.skip("bubblewrap user namespace is disabled on this host")
    assert probe.returncode == 0, probe.stderr


def test_late_installed_package_alias_masks_private_grader_and_oracle(tmp_path):
    """A runtime bind must not give installed private imports a second name."""
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    package = tmp_path / "installed" / "merlin_experiments"
    grader = package / "phase1" / "feedback"
    grader.mkdir(parents=True)
    (grader / "private_full_models.py").write_text("private gate\n")
    oracle = package / "phase1" / "private_oracle.py"
    oracle.write_text("private oracle\n")
    public = package / "phase1" / "public_tool.py"
    public.write_text("public tool\n")
    alias = Path("/tmp/installed-private-alias")
    argv = bwrap.base_argv(
        workspace,
        {"allowed": [], "denied": []},
        repo=tmp_path,
        _policy_test_live_inputs=True,
        include_claude_home=False,
        inherit_environment=False,
    )
    argv += ["--dir", str(alias), "--ro-bind", str(package), str(alias)]
    surfaces = [
        AnswerSurface("grader:feedback", grader, "dir", "grader"),
        AnswerSurface("oracle:private_oracle", oracle, "file", "oracle"),
    ]
    masked = bwrap.apply_answer_masks(argv, surfaces)
    assert not bwrap.is_exposed(masked, alias / "phase1/feedback/private_full_models.py")
    assert not bwrap.is_exposed(masked, alias / "phase1/private_oracle.py")
    assert bwrap.is_exposed(masked, alias / "phase1/public_tool.py")
    probe = subprocess.run(
        [
            *masked,
            "/bin/sh",
            "-c",
            (
                f'test -s "{alias / "phase1/public_tool.py"}" '
                f'&& test ! -s "{alias / "phase1/feedback/private_full_models.py"}" '
                f'&& test ! -s "{alias / "phase1/private_oracle.py"}"'
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "Creating new namespace failed: Operation not permitted" in probe.stderr:
        pytest.skip("bubblewrap user namespace is disabled on this host")
    assert probe.returncode == 0, probe.stderr


def test_missing_future_private_output_refuses_live_parent_or_runtime_alias(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = tmp_path / "host-run"
    parent.mkdir()
    future = parent / "grading_private_full_models"
    bundle = {"allowed": [{"path": str(parent)}], "private_validation_paths": [{"path": str(future), "kind": "dir"}]}
    with pytest.raises(RuntimeError, match="future operator-private"):
        bwrap.base_argv(workspace, bundle, repo=tmp_path, _policy_test_live_inputs=True)

    no_public_grant = {"allowed": [], "private_validation_paths": bundle["private_validation_paths"]}
    with pytest.raises(RuntimeError, match="runtime bind could expose a future"):
        host_surfaces.host_input_surfaces(
            ["--ro-bind", str(parent), "/tmp/private-gate-alias"],
            workspace,
            no_public_grant,
            repo=tmp_path,
            _policy_test_live_inputs=True,
        )
