"""Normal policy transport controls using own synthetic source bytes only.

These test the fixed writer/CLI/freezer dispatch seams. Unrelated paper source
and numerical admission is never qualified by a fixture or a selected pin.
"""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from capture_contraction_inputs import _pin, _selection

from merlin.capture import contraction_formats as F
from merlin.common.artifacts import ProductDir
from merlin.compare import capture_format_policy as P
from merlin.compare import capture_workflow as W
from merlin.compare import cli, freeze
from merlin.compare.paper import PaperStudySpec


def _study(version=3, precision="w8a8"):
    model = {
        "name": "synthetic",
        "capture": "synthetic",
        "checkpoint": "own-source-only",
        "fidelity": "full",
        "precisions": [precision],
        "artifacts": {precision: {"variant": "int8" if precision == "w8a8" else "fp32", "sha256": "unresolved"}},
        "session": {
            "kind": "image_stream",
            "warmups": 0,
            "observations": 2,
            "stages": ["forward"],
            "carried_state": [],
            "measurement_repeats": 3,
            "parameters": {"timed_stages": ["forward"], "paper_primary_scope": "end_to_end"},
        },
        "expected_provenance": {},
        "quality": {"reference": "eager_fp32", "metric": "exact"},
        "memory": {"policies": ["resident"]},
    }
    raw = {
        "version": version,
        "label": "own-source-policy",
        "status": "draft",
        "target": "neutral",
        "primary_precision": "w8a8",
        "control_precision": "fp32",
        "core_counts": [1],
        "development_corpus": {"excluded_models": ["synthetic"], "convergence_sweeps": 2},
        "paper_inputs": {"path": "unresolved", "sha256": "unresolved"},
        "holdout_models": ["synthetic"],
        "freeze": {"forbid_model_name_dispatch": True, "forbid_post_freeze_tuning": True},
        "models": [model],
        "backends": [
            {
                "name": "diagnostic",
                "kind": "frozen_baseline",
                "runtime": "diagnostic",
                "precisions": [precision],
                "quantization": "none",
                "kernel_scope": "whole_model",
                "adapter": "diagnostic",
                "options": {},
            }
        ],
        "reporting": {
            "same_buffer_repeat_is_diagnostic_only": True,
            "execution_order": {
                "policy": "deterministic_block_randomized",
                "block_fields": ["model", "precision", "core_count"],
                "randomized_field": "backend",
                "seed_sha256": "a" * 64,
            },
            "performance_claims": {
                "parity_median_ratio_band": [0.9, 1.1],
                "win_median_ratio_max": 0.9,
                "win_requires_nonoverlapping_observed_ranges": True,
                "win_requires_causal_attribution": True,
                "language": "descriptive_ratio_not_statistical_significance",
            },
            "performance_scope_policy": {
                "primary_table": "end_to_end_continuous_sessions",
                "diagnostic_table": "exact_declared_stage_subsets",
                "e2e_claim_requires_all_stages_timed": True,
                "e2e_win_requires_attribution": True,
            },
        },
    }
    if version == 3:
        raw["contraction_format_requirements"] = {precision: "complete_integer.v1"}
    return raw


def _originals(selected, bundle="."):
    return F.CaptureContractionOriginalSelection(
        (F.OriginalProgramContractionInput("forward", bundle, selected.programs[0].original_graph),)
    )


def _capture(root, form="integer"):
    selected = _selection(root, formats=(form,))
    # No paper/accuracy claim: these independent finite bytes exercise the
    # existing complete source-receipt writer, after the format boundary.
    values = np.asarray([[2.0], [3.0]], dtype=np.float32)
    reference = values * 2
    np.savez(root / "inputs.npz", stream=values)
    np.savez(root / "reference.npz", output=reference)
    np.save(root / "golden.npy", reference)
    contract = {
        "version": 1,
        "streams": [{"key": "stream"}],
        "inputs": "inputs.npz",
        "quality": {
            "scope": "trajectory",
            "reference": "eager_fp32",
            "metric": "exact",
            "golden": "reference.npz",
            "key": "output",
            "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        },
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    return replace(selected, session_contract=_pin(root / "session_contract.yaml"))


def _task(root, original, policy="complete_integer.v1"):
    model = PaperStudySpec.parse(_study()).models[0]
    return W.CaptureTask(
        model,
        "w8a8",
        "int8",
        "int8",
        "synthetic",
        root / "python",
        root,
        ("own-python", "own-writer"),
        {},
        "a" * 64,
        policy,
        original,
    )


def _descriptor(path, original):
    record = F.original_contraction_selection_record(original)
    path.write_text(
        json.dumps(
            {
                "schema": "merlin.capture_contraction_original_selections.v1",
                "cells": [
                    {
                        "model": "synthetic",
                        "precision": "w8a8",
                        "policy": original.policy,
                        "limits": record["limits"],
                        "programs": record["programs"],
                    }
                ],
            }
        )
    )
    return _pin(path)


def test_study_v3_requirement_is_typed_retained_and_changes_identity():
    v2 = PaperStudySpec.parse(_study(2))
    assert "contraction_format_requirements" not in v2.canonical_dict()
    v3 = PaperStudySpec.parse(_study())
    assert v3.canonical_dict()["contraction_format_requirements"] == {"w8a8": "complete_integer.v1"}
    assert v2.sha256() != v3.sha256()


@pytest.mark.parametrize(
    "requirements", [None, {}, [], {"absent": "complete_integer.v1"}, {"w8a8": True}, {"w8a8": "mixed"}]
)
def test_unknown_or_empty_requirement_cannot_declare_eligibility(requirements):
    raw = _study()
    raw["contraction_format_requirements"] = requirements
    with pytest.raises(ValueError, match="membership is incomplete"):
        PaperStudySpec.parse(raw)


@pytest.mark.parametrize("version", [1, 2])
def test_older_cli_spec_cannot_silently_drop_requirement(tmp_path, version):
    raw = _study()
    raw["version"] = version
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="membership is incomplete"):
        cli._load_versioned(path)


def test_version3_missing_requirement_refuses(tmp_path):
    raw = _study()
    del raw["contraction_format_requirements"]
    with pytest.raises(ValueError, match="selection is unavailable"):
        PaperStudySpec.parse(raw)


def test_original_adapter_reopens_original_not_posthoc_trace(tmp_path):
    selected = _capture(tmp_path / "capture")
    original = _originals(selected)
    actual = F.prepare_capture_contraction_selection(tmp_path / "capture", original)
    assert actual == selected
    assert F.require_capture_contraction_formats(tmp_path / "capture", actual)["census"]["eligible"]
    assert "not established" in F.original_contraction_selection_record(original)["scope"]


@pytest.mark.parametrize("defect", ["changed_original", "resigned_trace", "missing_trace", "float"])
def test_changed_unknown_or_floating_emission_cannot_replace_original(tmp_path, defect):
    selected = _capture(tmp_path / "capture", "float" if defect == "float" else "integer")
    original = _originals(selected)
    if defect == "changed_original":
        selected.programs[0].original_graph.path.write_text("{}")
    elif defect == "missing_trace":
        selected.programs[0].frontend_trace.path.unlink()
    elif defect == "resigned_trace":
        path = selected.programs[0].frontend_trace.path
        doc = json.loads(path.read_text())
        doc["graphs"]["original"]["call_count"] = 0
        doc["graphs"]["original"]["sha256"] = "a" * 64
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        F.prepare_capture_contraction_selection(tmp_path / "capture", original)


@pytest.mark.parametrize("bundle", ["../excluded", "/absolute", "alias/../member", "a//b", ["invalid"]])
def test_original_bundle_selection_is_exact_relative_membership(tmp_path, bundle):
    selected = _capture(tmp_path / "capture")
    with pytest.raises(ValueError, match="membership is incomplete"):
        _originals(selected, bundle)


def test_emitted_session_cannot_shrink_preselected_program_roster(tmp_path):
    first = _capture(tmp_path / "capture/first")
    second = _capture(tmp_path / "capture/second", "float")
    original = F.CaptureContractionOriginalSelection(
        (
            F.OriginalProgramContractionInput("first", "first", first.programs[0].original_graph),
            F.OriginalProgramContractionInput("second", "second", second.programs[0].original_graph),
        )
    )
    contract = {"version": 2, "programs": [{"name": "first", "bundle": "first"}]}
    (tmp_path / "capture/session_contract.yaml").write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match="membership is incomplete"):
        F.prepare_capture_contraction_selection(tmp_path / "capture", original)


def test_closed_descriptor_reopens_pins_and_is_not_saved_coverage(tmp_path):
    original = _originals(_capture(tmp_path / "capture"))
    descriptor = _descriptor(tmp_path / "selection.json", original)
    assert P.load_original_format_selections(descriptor) == {("synthetic", "w8a8"): original}
    document = json.loads(descriptor.path.read_text())
    document["eligible"] = True
    descriptor.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="source bytes changed"):
        P.load_original_format_selections(descriptor)
    with pytest.raises(ValueError, match="membership is incomplete"):
        P.load_original_format_selections(_pin(descriptor.path))


@pytest.mark.parametrize("defect", ["stale_graph", "duplicate_cell", "extra_program_field", "bad_budget"])
def test_descriptor_unknown_or_stale_membership_refuses(tmp_path, defect):
    selected = _capture(tmp_path / "capture")
    original = _originals(selected)
    descriptor = _descriptor(tmp_path / "selection.json", original)
    document = json.loads(descriptor.path.read_text())
    if defect == "stale_graph":
        selected.programs[0].original_graph.path.write_bytes(b"{}")
    elif defect == "duplicate_cell":
        document["cells"].append(copy.deepcopy(document["cells"][0]))
    elif defect == "extra_program_field":
        document["cells"][0]["programs"][0]["eligible"] = True
    else:
        document["cells"][0]["limits"]["file_bytes"] = True
    descriptor.path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        P.load_original_format_selections(_pin(descriptor.path))


def test_declared_task_missing_originals_refuses_before_validation(tmp_path):
    task = _task(tmp_path / "unproduced", None)
    with pytest.raises(ValueError, match="selection is unavailable"):
        W._validate_output(task, {}, 0)
    assert not task.output.exists()


def test_task_identity_binds_original_roster_and_rechecks_source(tmp_path):
    selected = _capture(tmp_path / "capture")
    task = _task(tmp_path / "capture", _originals(selected))
    record = task.to_dict()
    assert record["contraction_original_selection_sha256"] == W._json_sha256(record["contraction_original_selection"])
    selected.programs[0].original_graph.path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="source bytes changed"):
        task.to_dict()


@pytest.mark.parametrize("form", ["float", "bf16", "f16"])
def test_normal_output_writer_cannot_register_float_for_declared_integer(tmp_path, form):
    selected = _capture(tmp_path / "capture", form)
    task = _task(tmp_path / "capture", _originals(selected))
    with pytest.raises(ValueError, match="coverage is not established"):
        W._validate_output(task, {}, 0)
    assert not (task.output / "paper_measurement_sources.json").exists()


def test_normal_output_writer_forwards_live_selection_but_not_numerical_authority(tmp_path, monkeypatch):
    selected = _capture(tmp_path / "capture")
    task = _task(tmp_path / "capture", _originals(selected))
    # This unit seam isolates already-existing paper numerical admission. Its
    # fixture cannot certify producer, paper-input or accuracy readiness.
    monkeypatch.setattr(
        W,
        "validate_capture_session",
        lambda *a, **kw: ({"kind": "synthetic", "provenance": {"checkpoint": task.model.checkpoint}}, []),
    )
    result = W._validate_output(task, {}, 0)
    document = json.loads((task.output / "paper_measurement_sources.json").read_text())
    assert document["schema_version"] == 3
    assert document["contraction_formats"]["census"]["macs"]["original"] == 24
    assert "source_producer_authority" in document["contraction_formats"]["census"]["not_established"]
    assert result["measurement_source_receipt"]["sha256"] == _pin(task.output / "paper_measurement_sources.json").sha256


def test_capture_materializer_missing_selection_never_dispatches(tmp_path, monkeypatch):
    raw = _study()
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(W.HostExperimentSpec, "from_yaml", lambda _: SimpleNamespace())
    monkeypatch.setattr(W, "_preflight", lambda *a: ([], {"model2mlir": {}}, {}))
    product_root = tmp_path / "product"
    product_root.mkdir()
    product = ProductDir(
        product_root,
        product_root / "manifest.yaml",
        "synthetic",
        "paper-captures",
        1,
        "abcdef0",
        "20261010T000000Z",
        "neutral",
        [],
        [],
    )
    calls = []
    with pytest.raises(W.CaptureWorkflowNotReady, match="selection is unavailable"):
        W.materialize(source, source, tmp_path, execute=True, product=product, runner=lambda *a: calls.append(a) or 0)
    assert not calls
    assert json.loads((product_root / "capture-plan.json").read_text())["status"] == "blocked"
    assert not (product_root / "staged-study.yaml").exists()


@pytest.mark.parametrize("defect", [None, "changed_original", "float"])
def test_materializer_binds_predispatch_roster_through_normal_writer(tmp_path, monkeypatch, defect):
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(_study()))
    original = _originals(_capture(tmp_path / "original"))
    monkeypatch.setattr(W.HostExperimentSpec, "from_yaml", lambda _: SimpleNamespace(freeze={}))
    monkeypatch.setattr(
        W,
        "_preflight",
        lambda *a: ([], {"model2mlir": {}}, {"models": {"synthetic": {"environment": {"OWN_PAPER_READY": "1"}}}}),
    )
    monkeypatch.setattr(W, "_sanitized_environment", lambda exact: dict(exact))
    monkeypatch.setattr(
        W,
        "validate_capture_session",
        lambda *a, **kw: ({"kind": "synthetic", "provenance": {"checkpoint": "own-source-only"}}, []),
    )
    # No capture/interpreter is launched. The own runner emits finite synthetic
    # bytes; unrelated producer/source/numerical authority remains unqualified.
    interpreter = tmp_path / "workloads/synthetic/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"not executed")
    product_root = tmp_path / "product"
    product_root.mkdir()
    product = ProductDir(
        product_root,
        product_root / "manifest.yaml",
        "synthetic",
        "paper-captures",
        1,
        "abcdef0",
        "20261010T000000Z",
        "neutral",
        [],
        [],
    )

    def own_emitter(task, environment, stdout, stderr):
        assert environment == {"OWN_PAPER_READY": "1"}
        stdout.write_bytes(b"own synthetic writer\n")
        stderr.write_bytes(b"")
        _capture(task.output, "float" if defect == "float" else "integer")
        if defect == "changed_original":
            original.programs[0].original_graph.path.write_bytes(b"{}")
        return 0

    kwargs = dict(
        execute=True,
        product=product,
        runner=own_emitter,
        contraction_originals={("synthetic", "w8a8"): original},
    )
    if defect:
        with pytest.raises(W.CaptureWorkflowNotReady):
            W.materialize(source, source, tmp_path, **kwargs)
        assert not (product_root / "staged-study.yaml").exists()
    else:
        W.materialize(source, source, tmp_path, **kwargs)
        assert yaml.safe_load((product_root / "staged-study.yaml").read_text())["contraction_format_requirements"] == {
            "w8a8": "complete_integer.v1"
        }
    plan = json.loads((product_root / "capture-plan.json").read_text())
    assert (
        plan["tasks"][0]["contraction_original_selection_sha256"]
        == (plan["results"][0]["contraction_original_selection_sha256"])
    )
    assert plan["results"][0]["status"] == ("rejected" if defect else "validated")


def test_normal_cli_freeze_missing_selection_refuses_before_source_work(tmp_path):
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(_study()))
    with pytest.raises(ValueError, match="selection is unavailable"):
        cli.main(
            [
                "--spec",
                str(source),
                "--freeze",
                "--policy",
                str(source),
                "--toolchain-authority",
                str(source),
                "--runtime-path",
                str(source),
                "--frozen-out",
                str(tmp_path / "frozen.yaml"),
            ]
        )
    assert not (tmp_path / "frozen.yaml").exists()


def test_normal_cli_loads_selected_originals_without_saved_status(tmp_path, monkeypatch):
    selected = _capture(tmp_path / "capture", "float")
    original = _originals(selected)
    descriptor = _descriptor(tmp_path / "selection.json", original)
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(_study()))

    # Exercise CLI selection forwarding into the fixed adapter; no upstream
    # package or paper/source authority is mocked into a qualification.
    def consume(spec, **kwargs):
        P.prepare_declared_formats(
            spec.canonical_dict(),
            {("synthetic", "w8a8"): tmp_path / "capture"},
            originals=kwargs["contraction_originals"],
        )
        raise AssertionError("floating source must refuse")

    monkeypatch.setattr(freeze, "freeze_study", consume)
    with pytest.raises(ValueError, match="coverage is not established"):
        cli.main(
            [
                "--spec",
                str(source),
                "--freeze",
                "--policy",
                str(source),
                "--toolchain-authority",
                str(source),
                "--runtime-path",
                str(source),
                "--frozen-out",
                str(tmp_path / "frozen.yaml"),
                "--contraction-originals",
                str(descriptor.path),
                "--contraction-originals-sha256",
                descriptor.sha256,
            ]
        )


def test_fixed_constructor_cannot_omit_declared_requirement(tmp_path):
    from merlin.compare import paper_measurement_freeze as M

    with pytest.raises(ValueError, match="selection is unavailable"):
        M.construct_measurement_evidence(
            _study(),
            capture_roots={},
            output_path=tmp_path / "frozen.yaml",
            toolchain_authority_path=tmp_path / "none",
            toolchain_authority_sha256="a" * 64,
        )
    assert not (tmp_path / ".frozen-measurement-evidence").exists()


@pytest.mark.parametrize("defect", ["missing", "partial", "changed_source", "resigned_status"])
def test_frozen_selection_is_reopened_not_accepted_as_status(tmp_path, defect):
    selected = _capture(tmp_path / "capture")
    raw = _study()
    raw["freeze"]["contraction_format_selections"] = P.selection_records({("synthetic", "w8a8"): selected})
    P.require_frozen_formats(raw, {("synthetic", "w8a8"): tmp_path / "capture"})
    if defect == "missing":
        del raw["freeze"]["contraction_format_selections"]
    elif defect == "partial":
        raw["freeze"]["contraction_format_selections"] = []
    elif defect == "changed_source":
        selected.programs[0].final_mlir.path.write_text("builtin.module {}")
    else:
        raw["freeze"]["contraction_format_selections"][0]["eligible"] = True
    with pytest.raises(ValueError):
        P.require_frozen_formats(raw, {("synthetic", "w8a8"): tmp_path / "capture"})


def test_requirement_is_explicit_not_precision_label_inference(tmp_path):
    raw = _study(precision="fp32")
    selected = _capture(tmp_path / "capture")
    assert P.prepare_declared_formats(
        raw, {("synthetic", "fp32"): tmp_path / "capture"}, originals={("synthetic", "fp32"): _originals(selected)}
    )
    legacy = _study(2)
    assert P.declared_format_cells(legacy) == {}


def test_original_aggregate_budget_precedes_any_graph_parse(tmp_path, monkeypatch):
    selected = _capture(tmp_path / "capture")
    original = replace(_originals(selected), limits=F.ContractionFormatLimits(total_bytes=1))
    monkeypatch.setattr(F, "_original", lambda *a: pytest.fail("budget must refuse before parsing"))
    with pytest.raises(ValueError, match="selected budget"):
        F.verify_original_contraction_selection(original)


@pytest.mark.parametrize("defect", ["program_count", "whole_bytes"])
def test_descriptor_whole_roster_budget_precedes_original_parse(tmp_path, monkeypatch, defect):
    selected = _capture(tmp_path / "capture")
    descriptor = _descriptor(tmp_path / "selection.json", _originals(selected))
    document = json.loads(descriptor.path.read_text())
    cell = document["cells"][0]
    row = cell["programs"][0]
    if defect == "program_count":
        cell["programs"] = [dict(row, program=f"p{index}", bundle=f"p{index}") for index in range(128)]
        document["cells"].append(dict(cell, model="second", programs=[dict(row, program="other", bundle="other")]))
    else:
        path = selected.programs[0].original_graph.path
        raw = path.read_bytes()
        path.write_bytes(b" " * (4 * 1024 * 1024 - len(raw)) + raw)
        row["original_graph"] = {"path": str(path), "sha256": _pin(path).sha256}
        cell["limits"]["total_bytes"] = 32 * 1024 * 1024
        cell["programs"] = [dict(row, program=f"p{index}", bundle=f"p{index}") for index in range(5)]
    descriptor.path.write_text(json.dumps(document))
    monkeypatch.setattr(F, "_original", lambda *a: pytest.fail("whole-roster budget must precede original parsing"))
    with pytest.raises(ValueError, match="selected budget"):
        P.load_original_format_selections(_pin(descriptor.path))
