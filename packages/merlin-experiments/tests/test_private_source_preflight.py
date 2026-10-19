"""The private gate's compiler-independent preflight stops before any candidate is involved.

Source admission, capture identity and recipe derivation are substituted over owned harmless
files, as in the linked-ELF selection controls; the preselected-capture path binding and the
stop point are the real gate code. A passing preflight is never a gate result.
"""

from __future__ import annotations

import json
from contextlib import nullcontext

import pytest
from merlin_experiments.phase1.feedback import private_full_models as PFM
from merlin_experiments.phase1.feedback import private_source_preflight as preflight

from merlin.compile import baremetal_model, model_execution_inputs, route_before_build
from merlin.targetgen.contract.build_service import file_digest

TARGET = "owned_fixture"
MODELS = ("first", "second")


def _arguments(tmp_path, monkeypatch, *, misplaced=None, obligations=None):
    from merlin_experiments.phase0 import capture_execution_attestation, capture_selection

    from merlin.targetgen import target_registry

    package = tmp_path / "host-package"
    package.mkdir()
    (package / "manifest.yaml").write_text("{}")
    rows = []
    for name in MODELS:
        run = tmp_path / name / "run"
        stage = run / "capture"
        stage.mkdir(parents=True)
        (stage / "model.mlir").write_text("module {}\n")
        (stage / "capture_receipt.json").write_text("{}")
        selection = tmp_path / name / "selection.json"
        selection.write_text(json.dumps({"schema": capture_selection.SCHEMA_V2, "run_dir": str(run), "plan": {}}))
        declared = stage
        if name == misplaced:
            # A hard-linked relocation of the sealed output: same bytes, not the preselected run.
            declared = tmp_path / name / "relocated" / "capture"
            declared.mkdir(parents=True)
            for member in stage.iterdir():
                (declared / member.name).hardlink_to(member)
        attestation = tmp_path / name / "attestation.json"
        attestation.write_text(
            json.dumps(
                {
                    "issuer": "merlin.sealed_m2m_cpu.v3",
                    "selection": {"path": str(selection), "sha256": file_digest(selection)},
                    "capture": {
                        "kind": "single",
                        "capture_path": str(stage),
                        "model_sha256": file_digest(stage / "model.mlir"),
                        "receipt_sha256": file_digest(stage / "capture_receipt.json"),
                    },
                }
            )
        )
        row = {
            "id": name,
            "source_identity": {"model_id": name, "scope": "complete_network", "role": "private_validation"},
            "capture": str(declared),
            "capture_selection": str(selection),
            "capture_selection_sha256": file_digest(selection),
            "capture_execution_attestation": str(attestation),
            "host_package": str(package),
            "host_package_tree_sha256": model_execution_inputs.strict_tree_sha256(package)["sha256"],
            "deployment_dtype": "i8",
            "board": "owned_board",
            "arena_mb": 1,
            "rtl_config": "owned_config",
            "input_provenance": {"paper_ready": False, "synthetic_inputs": True},
        }
        for field, value in (
            ("software_spec", {"target": TARGET}),
            ("capability_contract", {"name": TARGET}),
            ("host_capabilities", {}),
            ("board_catalog", {}),
            ("host_dts", {}),
            ("rtl_facts", {}),
        ):
            path = tmp_path / (field + ".json")
            path.write_text(json.dumps(value))
            row[field], row[field + "_sha256"] = str(path), file_digest(path)
        rows.append(row)
    spec = tmp_path / "owned-spec.json"
    spec.write_text(json.dumps({"schema": PFM.SCHEMA, "target": TARGET, "models": rows}))
    monkeypatch.setenv("MERLIN_RTL_FACTS", rows[0]["rtl_facts"])
    monkeypatch.setattr(PFM.source_freeze_api, "resolve_optional", lambda *a, **k: {"diagnostic": True})
    monkeypatch.setattr(PFM.source_freeze_api, "claim_binding", lambda *a, **k: {})
    monkeypatch.setattr(PFM, "_require_selected_input_bindings", lambda *a: None)
    monkeypatch.setattr(capture_selection, "load", lambda path, **k: json.loads(path.read_text()))
    monkeypatch.setattr(capture_execution_attestation, "require_verified_execution", lambda *a: None)
    monkeypatch.setattr(PFM, "_capture_tree_bindings", lambda *a: {"diagnostic_source_only": True})
    monkeypatch.setattr(PFM, "_captured_programs", lambda stage, expected: [("model", stage)])
    monkeypatch.setattr(PFM, "_captured_input_provenance", lambda *a: {})
    monkeypatch.setattr(
        PFM, "_authored_file", lambda spec, row, field, frozen: PFM._file(spec, row[field], row[field + "_sha256"])
    )
    monkeypatch.setattr(
        PFM, "_recipe_derivation", lambda *a, **k: ({"diagnostic": True, "recipe_semantic_sha256": "d" * 64}, {}, {})
    )
    monkeypatch.setattr(PFM, "_selected_admission_views", lambda *a, **k: ({"target": TARGET}, {}))
    monkeypatch.setattr(PFM.control_support, "selected_build_observation", lambda *a: {})
    def source_obligations(*args, **kwargs):
        result = (obligations or (lambda *a, **k: {"eligible_groups": [0], "accelerator_source_demand": {}}))(
            *args, **kwargs
        )
        result.setdefault(PFM.linkage_support.FIELD, PFM.linkage_support.begin("d" * 64, "e" * 64, 1))
        return result

    monkeypatch.setattr(PFM, "_source_obligations", source_obligations)
    monkeypatch.setattr(target_registry, "observed_contract", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        model_execution_inputs,
        "selected_firrtl",
        lambda path, **k: {"path": str(path), "sha256": file_digest(path), "target": TARGET, "config": "owned_config"},
    )

    def no_candidate(*args, **kwargs):
        raise AssertionError("a source preflight must not route or build with any compiler")

    monkeypatch.setattr(route_before_build, "plan_before_build", no_candidate)
    monkeypatch.setattr(baremetal_model, "compile_saved_model", no_candidate)
    return {
        "private_spec": spec,
        "target": TARGET,
        "required_models": MODELS,
        "required_programs": {name: ("model",) for name in MODELS},
        "loader_env_requirements": {
            name: {
                "workload_dir": "",
                "selected_roles": (),
                "deployment_dtype": "i8",
                "required": {},
                "forbidden": (),
                "require_integer_contractions": False,
            }
            for name in MODELS
        },
        "out": tmp_path / "preflight",
    }


def test_source_preflight_stops_before_any_candidate_and_is_never_a_gate(tmp_path, monkeypatch):
    result = PFM.run(None, **_arguments(tmp_path, monkeypatch), source_only=True)
    assert [item["status"] for item in result["models"]] == ["source_inputs_verified"] * 2
    assert result["mode"] == "source_preflight" and result["source_preflight_passed"] is True
    assert result["passed"] is False and result["candidate_tree_sha256"] is None
    assert result["scope"] == PFM.SOURCE_PREFLIGHT_SCOPE
    assert all("build" not in item["checks"] for item in result["models"])
    report = tmp_path / "preflight" / preflight.REPORT
    assert json.loads(report.read_text()) == result
    assert not (tmp_path / "preflight" / "private_full_model_gate.json").exists()
    assert not PFM.complete(result, required_models=MODELS, required_programs={}, candidate_sha256="a" * 64)


def test_source_preflight_reports_the_capture_no_compiler_could_pass(tmp_path, monkeypatch):
    result = PFM.run(None, **_arguments(tmp_path, monkeypatch, misplaced="second"), source_only=True)
    first, second = result["models"]
    assert first["status"] == "source_inputs_verified"
    assert second["status"] == "fail"
    assert second["reason"] == "ValueError: full-model capture is not the preselected sealed run output"
    assert result["source_preflight_passed"] is False and result["passed"] is False


def test_unresolved_source_rows_reach_only_the_operator_preflight(tmp_path, monkeypatch):
    rows = [{"op": "owned.dequantize", "classification": "unresolved"}]

    def unresolved(*args, **kwargs):
        raise PFM.SourceAdmissionError(rows)

    arguments = _arguments(tmp_path, monkeypatch, obligations=unresolved)
    result = PFM.run(None, **arguments, source_only=True)
    assert all(
        item["reason"].endswith("1 unaccounted or unjustified operation signature(s)") for item in result["models"]
    )
    assert all(item["operator_unresolved"] == rows for item in result["models"])
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "manifest.yaml").write_text("{}")
    gate = PFM.run(package, **{**arguments, "out": tmp_path / "gate"})
    assert all("operator_unresolved" not in item for item in gate["models"])


def test_source_preflight_takes_no_candidate_or_diagnostic(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no candidate compiler"):
        PFM.run(tmp_path / "host-package", **arguments, source_only=True)


def test_selected_source_partition_keeps_full_roster_and_never_claims_full_success(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, monkeypatch)
    report = PFM.run(None, **arguments, source_only=True, selected_model="first")
    assert report["mode"] == "source_preflight_partition" and report["source_partition_passed"] is True
    assert report["source_preflight_passed"] is False and report["passed"] is False
    assert report["required_models"] == list(MODELS)
    assert report["required_programs"] == {name: ["model"] for name in MODELS}
    assert [row["model"] for row in report["models"]] == ["first"]
    assert report["candidate_tree_sha256"] is None


def test_progress_identifies_program_stage_without_candidate_credit(tmp_path, monkeypatch):
    events = []

    def obligations(*args, progress=None, **kwargs):
        progress({"stage": "normalization"})
        return {"eligible_groups": [0], "accelerator_source_demand": {}}

    arguments = _arguments(tmp_path, monkeypatch, obligations=obligations)
    result = PFM.run(None, **arguments, source_only=True, progress=events.append)
    assert result["source_preflight_passed"] is True and result["passed"] is False
    assert [event for event in events if event["stage"] == "normalization"] == [
        {"stage": "normalization", "model": name, "program": "model"} for name in MODELS
    ]
    assert [event["model"] for event in events if event["stage"] == "model_complete"] == list(MODELS)


def test_progress_failure_cannot_change_source_refusal(tmp_path, monkeypatch):
    def broken(event):
        raise OSError("unavailable observation sink")

    result = PFM.run(None, **_arguments(tmp_path, monkeypatch, misplaced="second"), source_only=True, progress=broken)
    assert result["source_preflight_passed"] is False and result["passed"] is False
    assert [item["status"] for item in result["models"]] == ["source_inputs_verified", "fail"]


def test_native_build_progress_never_claims_receipt_completion_after_compile_failure(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text("{}")
    events = []
    monkeypatch.setattr(route_before_build, "plan_before_build",
                        lambda *a, **k: {"device_routing": object(), "offload": True})
    monkeypatch.setattr(PFM.source_demand, "request", lambda *a: None)
    monkeypatch.setattr(PFM.linked_policy, "require_route", lambda *a: None)

    def refused_compile(**kwargs):
        assert events[-1]["stage"] == "program_build"
        assert events[-1]["program"] == "model"
        raise RuntimeError("intentional compiler refusal")

    monkeypatch.setattr(baremetal_model, "compile_saved_model", refused_compile)
    result = PFM.run(candidate, **arguments, progress=events.append)
    assert result["passed"] is False
    assert [row["status"] for row in result["models"]] == ["fail", "fail"]
    assert [row["model"] for row in events if row["stage"] == "program_build"] == list(MODELS)
    assert not any(row["stage"] == "program_receipt_verified" for row in events)


def test_progress_journal_is_bounded_atomic_and_observational(tmp_path):
    path, seen = tmp_path / preflight.PROGRESS, []
    observe = preflight._observer(seen.append, path)
    observe({"stage": "normalization", "model": "m" * 200, "program": "prefill", "private_error": "secret"})
    first = json.loads(path.read_text())
    assert first["sequence"] == 1 and len(first["model"]) == 128
    assert "secret" not in path.read_text() and path.stat().st_mode & 0o777 == 0o600
    observe({"stage": "host_admission", "current": 128, "total": 300})
    assert json.loads(path.read_text())["current"] == 128
    assert len(seen) == 2 and not path.with_name(path.name + ".pending").exists()
    # A missing journal parent and a broken observer never affect admission.
    preflight._observer(lambda event: (_ for _ in ()).throw(RuntimeError()), tmp_path / "missing" / "record")(
        {"stage": "parse"}
    )


def test_preflight_command_binds_the_launch_inputs(tmp_path, monkeypatch, capsys):
    from merlin_experiments import cli
    from merlin_experiments.phase1.feedback import private_facts, private_full_model_execution, private_source_freeze

    seen = {}
    monkeypatch.setattr(private_source_freeze, "stage", lambda spec, root, *, target: {"staged": str(root)})
    monkeypatch.setattr(private_facts, "selected_input_facts", lambda *a, **k: nullcontext())
    monkeypatch.setattr(PFM, "requirements_for", lambda descriptor: MODELS)
    monkeypatch.setattr(PFM, "program_requirements_for", lambda descriptor: {name: ("model",) for name in MODELS})
    monkeypatch.setattr(PFM, "loader_env_requirements_for", lambda descriptor: {})
    monkeypatch.setattr(private_full_model_execution, "gate_for", lambda descriptor, **k: None)

    def run(submission, spec, **kwargs):
        seen.update(kwargs, submission=submission, spec=spec)
        kwargs["progress"]({"stage": "normalization", "model": "first", "program": "model"})
        return {
            "models": [{"model": "first", "status": "fail", "reason": "ValueError: owned"}],
            "source_preflight_passed": False,
        }

    monkeypatch.setattr(PFM, "run", run)
    out = tmp_path / "out"
    code = cli.main(
        [
            "private-preflight",
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--descriptor",
            str(tmp_path / "d.yaml"),
            "--target",
            TARGET,
            "--out",
            str(out),
        ]
    )
    assert code == 1
    assert seen["submission"] is None and seen["source_only"] is True
    assert seen["source_freeze"] == {"staged": str(out / "sources")} and seen["source_freeze_root"] == out / "sources"
    assert seen["native_result_format_selection"] is None and seen["build_options"] is None
    captured = capsys.readouterr()
    assert "first: fail -- ValueError: owned" in captured.out
    assert '"stage": "normalization"' in captured.err
    journal = json.loads((out / preflight.PROGRESS).read_text())
    assert journal["stage"] == "complete" and journal["status"] == "fail"
    with pytest.raises(ValueError, match="fresh private output"):
        preflight.preflight(tmp_path / "spec.yaml", descriptor=tmp_path / "d.yaml", out=out, target=TARGET)
