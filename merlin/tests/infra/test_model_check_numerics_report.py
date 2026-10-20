"""Offline reports reuse pinned sections and score actual shifted targets without changing gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.model_check import numerics_report as N
from merlin.model_check.full_output_audit import SCHEMA as AUDIT_SCHEMA


def test_error_units_zero_energy_and_wide_integer_difference():
    row = N.array_errors(np.array([3.0, 4.0]), np.zeros(2))
    assert row["reference_rms"] == 0 and row["relative_l2"] is None
    assert row["rmse"] == pytest.approx(np.sqrt(12.5))
    assert row["mae"] == 3.5 and row["max_abs_error"] == 4
    row = N.array_errors(np.array([2**62 + 1], np.int64), np.array([2**62], np.int64))
    assert row["rmse"] == 1 and row["mismatched"] == 1
    assert not N.array_errors(np.zeros(1), np.zeros(2))["available"]
    assert not N.array_errors(np.array([np.nan]), np.array([np.nan]))["available"]


def test_ppl_scores_shifted_continuation_excludes_last_and_masks_padding():
    logits = np.array([[[0, 0], [0, np.log(3)], [100, -100]]])
    tokens = np.array([[99, 0, 1, -1]])  # prompt id need not be a target; trailing token is unused
    row = N.perplexity(logits, tokens, 1, 2)
    assert row["scored_tokens"] == 2
    assert row["nll_sum"] == pytest.approx(-np.log(0.5 * 0.75))
    assert row["ppl"] == pytest.approx((0.5 * 0.75) ** -0.5)
    assert row["target_positions"] == [1, 2] and not row["final_decode_logit_scored"]
    masked = N.perplexity(logits, tokens, 1, 2, np.array([[False, True]]))
    assert masked["scored_tokens"] == 1 and masked["ppl"] == pytest.approx(1 / 0.75)
    assert not N.perplexity(logits[:, :2], tokens, 1, 2)["available"]
    assert not N.perplexity(logits, tokens, 1, 2, np.zeros((1, 2), bool))["available"]
    assert not N.perplexity(logits, np.array([[0, 4, 1]]), 1, 2)["available"]


def test_ppl_reduction_is_token_weighted_and_stable():
    logits = np.zeros((2, 3, 2))
    logits[1, 0] = [0, np.log(3)]
    mask = np.array([[True, True], [True, False]])
    row = N.perplexity(logits, np.array([[0, 0, 0], [0, 1, 0]]), 1, 2, mask)
    assert row["scored_tokens"] == 3
    assert row["nll"] == pytest.approx((-2 * np.log(0.5) - np.log(0.75)) / 3)
    extreme = N.perplexity(np.array([[[1000.0, -1000.0], [0, 0]]]), np.array([[0, 1]]), 1, 1)
    assert extreme["available"] and extreme["nll"] == 2000
    assert extreme["ppl"] is None and extreme["ppl_overflow"]


def _fixture(root: Path):
    def write(path, value):
        path.write_text(json.dumps(value))
        return pin(path)

    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    programs = [
        {"name": "prefill", "stage": "prefill", "inputs": [{"from": "tokens", "slice": "prefill"}]},
        {"name": "decode", "stage": "decode", "steps": 2, "inputs": [{"from": "tokens", "slice": "next"}]},
    ]
    judged = {"kind": "next_token_distribution", "outputs": [{"program": p["name"], "output": 0} for p in programs]}
    config = {
        "schema": "merlin.model_check.models.v1",
        "target": "t",
        "models": {
            "m": {
                "workload": "w",
                "chain": {"kind": "causal", "prefill_tokens": 2, "decode_steps": 2},
                "programs": programs,
                "judged": judged,
            }
        },
    }
    original = root / "original.yaml"
    original.write_text(yaml.safe_dump(config))
    config["numerics"] = {"version": N.POLICY, "report_only": True}
    config["models"]["m"]["judged"]["reporting"] = {
        "perplexity": {"exclude_final_decode_logit": True, "padding": "none", "score_tokens": "decode_steps"}
    }
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump(config))
    np.savez(root / "input.npz", tokens=np.array([0, 0, 1, 2]))
    steps = []
    for program, step in [("prefill", 0), ("decode", 0), ("decode", 1)]:
        path = root / f"{program}.{step}.npz"
        np.savez(path, out0=np.zeros((1, 3), np.float32))
        steps.append({"program": program, "step": step, "outputs": 1, "actual": pin(path), "reference": pin(path)})
    bundle = root / "bundle"
    fp = bundle / "m/visible/v00/fp.npz"
    fp.parent.mkdir(parents=True)
    np.savez(fp, logits=np.zeros((3, 3), np.float32))
    manifest = write(bundle / "MANIFEST.json", {})
    sums = bundle / "SHA256SUMS"
    sums.write_text(f"{pin(fp)['sha256']}  m/visible/v00/fp.npz\n")
    compiler = {"model_check_tree_sha256": "c" * 64}
    audit = {
        "schema": AUDIT_SCHEMA,
        "unchanged": True,
        "status": "pass",
        "model": "m",
        "input_id": "visible/v00",
        "submission": compiler,
        "steps": steps,
        "programs": [],
        "float_policy": {"atol": 2e-6, "rtol": 1e-5},
        "selected_identity": {
            "models_config_sha256": pin(original)["sha256"],
            "bundle_manifest_sha256": manifest["sha256"],
            "bundle_sums_sha256": pin(sums)["sha256"],
        },
        "job_pin": write(root / "job.json", {"argv": ["--models", str(original), "--bundle", str(bundle)]}),
        "reference_manifest": write(root / "supplement.json", {"input": pin(root / "input.npz")}),
        "full_report": write(
            root / "full.json",
            {
                "model": "m",
                "submission": {"tree_sha256": "c" * 64},
                "gates": {"past": {"status": "fail"}},
                "programs": [],
            },
        ),
    }
    audit_path = root / "audit.json"
    write(audit_path, audit)
    return audit_path, policy, audit


def test_retained_report_has_section_and_real_ppl_numbers_without_promoting_gates(tmp_path):
    audit, policy, _ = _fixture(tmp_path)
    full_before = (tmp_path / "full.json").read_bytes()
    row = N.report(audit, policy)
    assert row["report_only"] and "gates" not in row
    assert len(row["sections"]) == 3
    assert all(r["compiled_vs_quantized"]["rmse"] == 0 for r in row["sections"])
    assert row["perplexity"]["compiled"]["ppl"] == pytest.approx(3)
    assert (tmp_path / "full.json").read_bytes() == full_before
    json.dumps(row, allow_nan=False)


def test_float_only_stage_can_have_null_quantization_metadata(tmp_path):
    audit_path, policy, audit = _fixture(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"quantization_stats": None}))
    audit["programs"] = [{"program": "prefill", "capture_program": {"path": str(tmp_path / "model.mlir")}}]
    audit_path.write_text(json.dumps(audit))
    assert N.report(audit_path, policy)["capture_attention_sites"] == []


def test_report_rejects_changed_artifact_and_partial_or_rebound_model(tmp_path):
    audit_path, policy, audit = _fixture(tmp_path)
    original_audit = audit_path.read_text()
    audit["steps"] = audit["steps"][:-1]
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="partial"):
        N.report(audit_path, policy)
    audit_path.write_text(original_audit)
    config = yaml.safe_load(policy.read_text())
    config["models"]["m"]["chain"]["prefill_tokens"] = 1
    policy.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="changed evaluated model"):
        N.report(audit_path, policy)


def test_numeric_npz_order_and_byte_pins(tmp_path):
    p = tmp_path / "many.npz"
    np.savez(p, **{f"out{i}": np.array([i]) for i in [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]})
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    reader = N.Reader()
    assert list(reader.arrays({"path": str(p), "sha256": sha})) == [f"out{i}" for i in range(11)]
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest|changed"):
        reader.unchanged()


def test_zero_cosine_is_unavailable_but_absolute_numbers_remain_json_finite():
    row = N._quality("logits", [np.zeros((1, 3), np.float32)], [np.zeros((1, 3), np.float32)], {"topk": [1]})
    assert row["legacy_metrics"]["cosine"] is None
    assert row["errors"]["rmse"] == 0
    json.dumps(row, allow_nan=False)


def test_quality_never_broadcasts_a_single_flow_step_or_accepts_partial_logits():
    actions = np.zeros((1, 50, 6), np.float32)
    one_time = np.zeros((1, 1, 6), np.float32)
    assert not N._quality("per_channel", [actions], [one_time], {"channels": {"first": 0, "last": 5}})["available"]
    logits = np.zeros((1, 8), np.float32)
    assert not N._quality("next_token_distribution", [logits, logits], [logits], {})["available"]


def test_large_finite_diagnostics_never_emit_json_infinity():
    row = N.array_errors(np.full(2, 1e308), np.zeros(2))
    assert row["mae"] == 1e308
    row = N.array_errors(np.array([1e308]), np.array([1e-308]))
    assert row["relative_l2"] is None and row["relative_l2_overflow"]
    json.dumps(row, allow_nan=False)
    row = N.perplexity(np.array([[[1e308, -1e308], [0, 0]]]), np.array([[0, 1]]), 1, 1)
    assert not row["available"]


def test_new_sealed_config_can_already_contain_reporting_and_rejects_changed_semantics(tmp_path):
    audit_path, policy, audit = _fixture(tmp_path)
    original = tmp_path / "original.yaml"
    original.write_bytes(policy.read_bytes())
    audit["selected_identity"]["models_config_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit))
    assert N.report(audit_path, policy)["quality"]["available"]
    config = yaml.safe_load(policy.read_text())
    config["numerics"]["reporting"] = {"reduction_dtype": "float32"}
    policy.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="reporting semantics"):
        N.report(audit_path, policy)


def test_accuracy_label_bytes_are_bound_to_the_input_manifest(tmp_path):
    audit_path, policy, _ = _fixture(tmp_path)
    labels = tmp_path / "visible.json"
    labels.write_text(json.dumps({"v00": {"label": 0}}))
    manifest = {
        "models": {
            "m": {
                "derived": {"visible/v00/input.npz": hashlib.sha256((tmp_path / "input.npz").read_bytes()).hexdigest()},
                "visible": {"visible.json": hashlib.sha256(labels.read_bytes()).hexdigest()},
            }
        }
    }
    (tmp_path / "inputs-manifest.json").write_text(json.dumps(manifest))
    config = yaml.safe_load(policy.read_text())
    config["numerics"]["reporting"] = {"input_manifest": "inputs-manifest.json"}
    config["models"]["m"]["judged"]["reporting"]["labels"] = {
        "path": "visible.json",
        "field": "label",
        "namespace": "toy",
    }
    policy.write_text(yaml.safe_dump(config))
    assert N.report(audit_path, policy)["label_accuracy"]["fp"]["top1"]
    labels.write_text(json.dumps({"v00": {"label": 2}}))
    with pytest.raises(ValueError, match="digest"):
        N.report(audit_path, policy)


def test_missing_fp_still_reports_measurable_compiled_and_quantized_ppl(tmp_path):
    audit_path, policy, audit = _fixture(tmp_path)
    sums = tmp_path / "bundle/SHA256SUMS"
    sums.write_text("")
    audit["selected_identity"]["bundle_sums_sha256"] = hashlib.sha256(sums.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit))
    row = N.report(audit_path, policy)
    assert not row["quality"]["available"] and not row["perplexity"]["fp"]["available"]
    assert row["perplexity"]["compiled"]["ppl"] == pytest.approx(3)
    assert row["perplexity"]["quantized"]["ppl"] == pytest.approx(3)
