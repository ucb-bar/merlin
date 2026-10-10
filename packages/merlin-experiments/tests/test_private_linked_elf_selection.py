"""Actual automatic-caller wiring controls; no model or ISA qualification.

All source/capture admission and expensive build work below are diagnostic
substitutes over owned harmless files. Every attempt remains a failed gate;
these controls prove selection transport/refusal, not instruction legality.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import private_full_models as PFM

from merlin.compile import baremetal_model, model_execution_inputs, route_before_build
from merlin.llvmlower.device_build import DeviceRouting
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def _not_an_isa_evaluator(*, elf, evidence_root):
    raise AssertionError("diagnostic wiring must not issue linked-image admission")


def _other_not_an_isa_evaluator(*, elf, evidence_root):
    raise AssertionError("a substituted callback must not execute")


def _service(target, dependency):
    return LinkedElfAdmissionService(
        target,
        _not_an_isa_evaluator,
        tuple((str(path), file_digest(path)) for path in (Path(__file__).resolve(), dependency)),
    )


def _attempt(tmp_path, monkeypatch, *, fault=None, selected=True, eligible=True):
    """Run the real automatic caller with explicitly unqualified dependencies."""
    from merlin_experiments.phase0 import capture_execution_attestation, capture_selection

    from merlin.targetgen import target_registry

    target = "owned_fixture"
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "manifest.yaml").write_text("{}")
    dependency = tmp_path / "policy-source.txt"
    dependency.write_text("owned selection bytes; no ISA facts\n")
    service = _service(target, dependency) if selected else None
    rows = []
    for name in ("first", "second"):
        stage = tmp_path / name / "capture"
        stage.mkdir(parents=True)
        (stage / "model.mlir").write_text("module {}\n")
        (stage / "capture_receipt.json").write_text("{}")
        selection = stage.parent / "selection.json"
        selection.write_text(
            json.dumps({"schema": capture_selection.SCHEMA_V2, "run_dir": str(stage.parent), "plan": {}})
        )
        attestation = stage.parent / "attestation.json"
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
            "capture": str(stage),
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
            ("software_spec", {"target": target}),
            ("capability_contract", {"name": target}),
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
    spec.write_text(json.dumps({"schema": PFM.SCHEMA, "target": target, "models": rows}))
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
    monkeypatch.setattr(PFM, "_recipe_derivation", lambda *a, **k: ({"diagnostic": True}, {}, {}))
    monkeypatch.setattr(PFM, "_selected_admission_views", lambda *a, **k: ({"target": target}, {}))
    monkeypatch.setattr(PFM.control_support, "selected_build_observation", lambda *a: {})
    monkeypatch.setattr(PFM, "_source_obligations", lambda *a: {"eligible_groups": [0] if eligible else []})
    monkeypatch.setattr(PFM.linkage_support, "symbols", lambda source: [])
    monkeypatch.setattr(target_registry, "observed_contract", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        model_execution_inputs,
        "selected_firrtl",
        lambda path, **k: {"path": str(path), "sha256": file_digest(path), "target": target, "config": "owned_config"},
    )
    seen = {"plans": [], "builds": [], "verification": []}

    def plan(*args, **kwargs):
        seen["plans"].append(kwargs)
        route = DeviceRouting(target, package, "i8", "i32", linked_elf_admission=kwargs.get("linked_elf_admission"))
        if fault == "missing_on_route":
            route = replace(route, linked_elf_admission=None)
        elif fault == "other_service_on_route":
            route = replace(route, linked_elf_admission=_service(target, dependency))
        elif fault == "other_target_on_route":
            route = replace(route, device="other_owned_fixture")
        elif fault == "callback_during_plan":
            object.__setattr__(service, "evaluator", _other_not_an_isa_evaluator)
        return {"device_routing": route, "offload": object()}

    def build(**kwargs):
        seen["builds"].append(kwargs)
        if fault == "route_during_build":
            object.__setattr__(kwargs["device"], "linked_elf_admission", None)
        elif fault == "source_during_build":
            dependency.write_text("changed selected source\n")
        return {"diagnostic_build_only": True}

    def verify(*args, **kwargs):
        seen["verification"].append(kwargs)
        raise ValueError("diagnostic stopped after actual build call; no qualification")

    monkeypatch.setattr(route_before_build, "plan_before_build", plan)
    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    monkeypatch.setattr(PFM, "_verify_compiled_program", verify)
    arguments = {
        "submission": package,
        "private_spec": spec,
        "target": target,
        "required_models": ("first", "second"),
        "required_programs": {name: ("model",) for name in ("first", "second")},
        "loader_env_requirements": {
            name: {
                "workload_dir": "",
                "selected_roles": (),
                "deployment_dtype": "i8",
                "required": {},
                "forbidden": (),
                "require_integer_contractions": False,
            }
            for name in ("first", "second")
        },
        "out": tmp_path / "attempt",
    }
    if selected:
        arguments["linked_elf_admission"] = service
    result = PFM.run(**arguments)
    assert result["passed"] is False and len(result["models"]) == 2
    assert all(row["status"] == "fail" for row in result["models"])
    return result, seen, service


def test_selected_policy_reaches_each_actual_automatic_build_unchanged(tmp_path, monkeypatch):
    result, seen, service = _attempt(tmp_path, monkeypatch)
    assert len(seen["plans"]) == len(seen["builds"]) == len(seen["verification"]) == 2
    assert all(row["linked_elf_admission"] is service for row in seen["plans"])
    assert all(row["device"].linked_elf_admission is service for row in seen["builds"])
    assert all("no qualification" in row["reason"] for row in result["models"])


def test_active_automatic_route_without_policy_refuses_before_build(tmp_path, monkeypatch):
    result, seen, _service_owner = _attempt(tmp_path, monkeypatch, selected=False)
    assert len(seen["plans"]) == 2 and not seen["builds"] and not seen["verification"]
    assert all("requires independent linked ELF admission" in row["reason"] for row in result["models"])


@pytest.mark.parametrize(
    "fault", ["missing_on_route", "other_service_on_route", "other_target_on_route", "callback_during_plan"]
)
def test_planner_cannot_drop_or_replace_the_selected_service(tmp_path, monkeypatch, fault):
    result, seen, _service_owner = _attempt(tmp_path, monkeypatch, fault=fault)
    assert not seen["builds"] and not seen["verification"]
    assert all("linked ELF admission" in row["reason"] for row in result["models"])


@pytest.mark.parametrize("fault", ["route_during_build", "source_during_build"])
def test_build_cannot_change_route_or_selected_policy_bytes(tmp_path, monkeypatch, fault):
    result, seen, _service_owner = _attempt(tmp_path, monkeypatch, fault=fault)
    assert seen["builds"] and not seen["verification"]
    assert all("linked ELF admission" in row["reason"] for row in result["models"])
    assert str(tmp_path) not in " ".join(row["reason"] for row in result["models"])


def test_no_eligible_device_work_preserves_original_source_refusal(tmp_path, monkeypatch):
    result, seen, _service_owner = _attempt(tmp_path, monkeypatch, selected=False, eligible=False)
    assert not seen["plans"] and not seen["builds"]
    assert all("no independently eligible accelerator computation" in row["reason"] for row in result["models"])


@pytest.mark.parametrize("selection", [object(), lambda **kwargs: None])
def test_untyped_policy_cannot_open_original_inputs(tmp_path, selection):
    with pytest.raises(ValueError, match="explicit linked ELF admission service"):
        PFM.run(
            tmp_path / "missing-candidate",
            tmp_path / "missing-spec",
            target="owned_fixture",
            required_models=(),
            required_programs={},
            loader_env_requirements={},
            out=tmp_path / "attempt",
            linked_elf_admission=selection,
        )


def test_wrong_target_policy_cannot_open_original_inputs(tmp_path):
    dependency = tmp_path / "policy-source.txt"
    dependency.write_text("owned source")
    with pytest.raises(ValueError, match="selection is invalid"):
        PFM.run(
            tmp_path / "missing-candidate",
            tmp_path / "missing-spec",
            target="owned_fixture",
            required_models=(),
            required_programs={},
            loader_env_requirements={},
            out=tmp_path / "attempt",
            linked_elf_admission=_service("other_owned_fixture", dependency),
        )
