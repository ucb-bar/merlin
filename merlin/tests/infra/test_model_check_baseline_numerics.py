"""Small supplied arrays exercise strict offline comparisons, never model execution."""
import hashlib
import json

import numpy as np
import pytest

from merlin.model_check import baseline_numerics as B


def write(path, value):
    path.write_text(B.yaml.safe_dump(value, sort_keys=False))
    return path


def pin(path, root):
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path):
    outputs = [{"dtype": "f32", "shape": [1, 2]}, {"dtype": "i64", "shape": [1]},
               {"dtype": "i1", "shape": [1]}]
    judged = {"kind": "logits", "outputs": [{"program": "model", "output": 0}], "topk": [1],
              "admission": [{"metric": "cosine", "aggregate": "mean", "min": .98}]}
    config = write(tmp_path / "models.json", {"schema": "merlin.model_check.models.v1", "target": "fixture",
        "numerics": {"version": "merlin.model_check.numerics.v1", "report_only": True,
            "functional": {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True,
                           "integer_boolean": "exact", "outputs": "all_declared_every_step"}},
        "models": {"example": {"workload": "example", "chain": {"kind": "single"},
            "programs": [{"name": "model", "inputs": []}], "judged": judged}}})
    protocol = write(tmp_path / "protocol.json", {"schema": "merlin.fixture.experiment_protocol.v1",
        "authority": {"numerical_contract": config.name, "evaluation_input_manifest": "evaluation.json"},
        "profiles": {"quality_v1": {"observations_per_model_per_split": 8}},
        "models": {"example": {"chain": {"kind": "single"}, "selected_chain_steps": 1,
            "declared_arrays_per_chain": 3, "stages": {"model": {"steps": 1, "outputs": outputs}}}}})
    reference = tmp_path / "reference.npz"
    np.savez(reference, out0=np.array([[1., 2.]], np.float32), out1=np.array([2**62], np.int64),
             out2=np.array([True]))
    actual = tmp_path / "actual.npz"
    actual.write_bytes(reference.read_bytes())
    fp = tmp_path / "fp.npz"
    np.savez(fp, out0=np.array([[1., 2.]], np.float32))
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, tokens=np.array([[0, 1, 0, 1]], np.int64))
    definition = B.yaml.safe_load(protocol.read_bytes())
    definition["models"]["example"]["external_evaluation_input"] = {
        "selection": "visible/v00", "sha256": pin(inputs, tmp_path)["sha256"],
        "derived_member": "visible/v00/inputs.npz"}
    write(protocol, definition)
    evaluation = write(tmp_path / "evaluation.json", {"models": {"example": {"derived": {
        f"{split}/{prefix}{i:02}/inputs.npz": pin(inputs, tmp_path)["sha256"]
        for split, prefix in (("visible", "v"), ("sealed", "s")) for i in range(8)}}}})
    ref_manifest = write(tmp_path / "reference-manifest.json", {"identity_only": True})
    row = {"id": "v00", "split": "visible", "draw": 0, "input": pin(inputs, tmp_path),
           "steps": [{"program": "model", "step": 0, "actual": pin(actual, tmp_path),
                      "reference": pin(reference, tmp_path)}], "quality": {"fp32": pin(fp, tmp_path)}}
    manifest = write(tmp_path / "manifest.json", {"schema": B.SCHEMA, "protocol": pin(protocol, tmp_path),
        "numerical_policy": pin(config, tmp_path),
        "evaluation_inputs": pin(evaluation, tmp_path),
        "model": "example", "profile": "functional_v1", "reference_manifest": pin(ref_manifest, tmp_path),
        "cases": [row]})
    return tmp_path, manifest


def change_actual(case, **changes):
    root, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    path = root / spec["cases"][0]["steps"][0]["actual"]["path"]
    with np.load(path, allow_pickle=False) as a:
        arrays = {key: a[key] for key in a.files}
    np.savez(path, **{**arrays, **changes})
    spec["cases"][0]["steps"][0]["actual"] = pin(path, root)
    write(manifest, spec)


def test_complete_functional_arrays_and_explicit_noncertification(case):
    result = B.compare(case[1])
    assert result["numerical_status"] == "pass"
    assert result["functional_passed"] is True
    assert result["quality_admission"]["status"] == "unavailable"
    assert len(result["cases"][0]["comparisons"]) == 3
    assert not any(result[k] for k in ("execution_qualified", "reference_origin_certified", "placement_qualified"))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value,passed", [(1.000005, True), (1.01, False)])
def test_selected_float_policy_is_used(case, value, passed):
    change_actual(case, out0=np.array([[value, 2.]], np.float32))
    assert B.compare(case[1])["functional_passed"] is passed


@pytest.mark.parametrize("change", [
    {"out1": np.array([2**62 + 1], np.int64)}, {"out2": np.array([False])},
    {"out0": np.array([[1., np.nan]], np.float32)}, {"out0": np.array([1., 2.], np.float32)},
    {"out0": np.array([[1., 2.]], np.float64)},
])
def test_integer_bool_nonfinite_shape_and_dtype_are_not_relaxed(case, change):
    change_actual(case, **change)
    result = B.compare(case[1])
    assert result["functional_passed"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changed", ["missing_step", "duplicate_step", "duplicate_case", "bad_pin", "extra_array"])
def test_partial_duplicate_or_changed_inputs_refuse(case, changed):
    root, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    if changed == "missing_step":
        spec["cases"][0]["steps"] = []
    elif changed == "duplicate_step":
        spec["cases"][0]["steps"] *= 2
    elif changed == "duplicate_case":
        spec["cases"] *= 2
    elif changed == "bad_pin":
        spec["reference_manifest"]["sha256"] = "0" * 64
    else:
        path = root / "actual.npz"
        np.savez(path, out0=np.zeros((1, 2), np.float32), out2=np.zeros(1, bool), out9=np.zeros(1))
        spec["cases"][0]["steps"][0]["actual"] = pin(path, root)
    write(manifest, spec)
    with pytest.raises(ValueError):
        B.compare(manifest)


def test_quality_needs_exact_both_split_cohort_and_available_fp(case):
    _, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    spec["profile"] = "quality_v1"
    write(manifest, spec)
    assert B.compare(manifest)["numerical_status"] == "incomplete"
    original = spec["cases"][0]
    spec["cases"] = [{**original, "id": f"{prefix}{i:02}", "split": split}
                     for split, prefix in (("visible", "v"), ("sealed", "s")) for i in range(8)]
    write(manifest, spec)
    assert B.compare(manifest)["quality_admission"]["status"] == "pass"
    spec["cases"][-1] = {**spec["cases"][-1], "quality": {}}
    write(manifest, spec)
    assert B.compare(manifest)["quality_admission"]["status"] == "unavailable"


def test_shape_misaligned_quality_cannot_broadcast(case):
    root, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    fp = root / "wrong-fp.npz"
    np.savez(fp, out0=np.array([1., 2.], np.float32))
    spec["cases"][0]["quality"] = {"fp32": pin(fp, root)}
    write(manifest, spec)
    result = B.compare(manifest)
    assert result["functional_passed"] is True
    assert result["cases"][0]["actual_vs_fp32"]["available"] is False


def test_cli_writes_separate_report_and_never_overwrites(case):
    root, manifest = case
    output = root / "report.json"
    assert B.main(["--manifest", str(manifest), "--out", str(output)]) == 0
    assert B.main(["--manifest", str(manifest), "--out", str(output)]) == 2


def variant(case, kind):
    root, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    protocol_path = root / "protocol.json"
    protocol = B.yaml.safe_load(protocol_path.read_bytes())
    model_path = root / "models.json"
    config = B.yaml.safe_load(model_path.read_bytes())
    model = config["models"]["example"]
    if kind == "tiny":
        chain = {"kind": "causal", "prefill_tokens": 2, "decode_steps": 2}
        stages = {"prefill": {"steps": 1}, "decode": {"steps": 2}}
        shape = [1, 3]
        judged = {"kind": "next_token_distribution", "outputs": [
            {"program": "prefill", "output": 0}, {"program": "decode", "output": 0}],
            "admission": [{"metric": "kl_mean", "aggregate": "mean", "max": .075}],
            "reporting": {"perplexity": {"targets": "teacher_forced_continuation", "score_tokens": "decode_steps",
                "exclude_final_decode_logit": True, "padding": "none", "threshold": "report_only"}}}
        value = np.zeros(shape, np.float32)
    else:
        chain = {"kind": "flow", "steps": 2}
        stages = {"prefix": {"steps": 1}, "flow": {"steps": 2}, "action": {"steps": 1}}
        shape = [1, 2, 3]
        judged = {"kind": "per_channel", "outputs": [{"program": "action", "output": 0}],
            "channels": {"first": 0, "last": 1},
            "reference": {"baseline": "native", "seeds": {"input": "noise", "count": 4}},
            "admission": [{"metric": "rel_rmse_worst", "aggregate": "mean", "max_ratio_to_baseline": 2.},
                          {"metric": "rel_rmse_worst", "aggregate": "p90", "max_ratio_to_baseline": 2.}]}
        value = np.full(shape, 1.1, np.float32)
    model.update(chain=chain, judged=judged, programs=[{"name": p, "steps": s["steps"], "inputs": []}
                                                    for p, s in stages.items()])
    for stage in stages.values():
        stage["outputs"] = [{"dtype": "f32", "shape": shape}]
    count = sum(s["steps"] for s in stages.values())
    protocol["models"]["example"].update(chain=chain, stages=stages, selected_chain_steps=count,
                                          declared_arrays_per_chain=count)
    write(model_path, config)
    write(protocol_path, protocol)
    spec["protocol"] = pin(protocol_path, root)
    spec["numerical_policy"] = pin(model_path, root)
    row = spec["cases"][0]
    row["steps"] = []
    for p, stage in stages.items():
        for step in range(stage["steps"]):
            array = value.copy()
            if kind == "tiny" and p == "decode" and step == 1:
                array[0, 0] = 1000  # excluded last logit would radically alter PPL if scored
            path = root / f"{p}-{step}.npz"
            np.savez(path, out0=array)
            row["steps"].append({"program": p, "step": step, "actual": pin(path, root),
                                  "reference": pin(path, root)})
    fp = root / "fp.npz"
    if kind == "tiny":
        np.savez(fp, out0=np.concatenate([value, value, np.array([[1000., 0, 0]], np.float32)]))
    else:
        np.savez(fp, out0=np.ones(shape, np.float32))
    native = root / "native.npz"
    np.savez(native, out0=value)
    row["quality"] = {"fp32": pin(fp, root), "native_baseline": pin(native, root)}
    write(manifest, spec)
    return spec


def test_true_teacher_forced_ppl_excludes_final_distribution_and_aggregates_tokens(case):
    variant(case, "tiny")
    result = B.compare(case[1])
    score = result["cases"][0]["perplexity"]["actual"]
    assert score["scored_tokens"] == 2
    assert score["ppl"] == pytest.approx(3.)
    assert score["target_positions"] == [2, 3]
    assert score["final_decode_logit_scored"] is False
    assert result["perplexity"]["actual"]["scored_tokens"] == 2
    assert result["perplexity"]["actual"]["report_only"] is True


def test_smol_all_draws_and_native_baseline_are_mandatory_for_quality_admission(case):
    spec = variant(case, "smol")
    original = spec["cases"][0]
    spec["profile"] = "quality_v1"
    spec["cases"] = [{**original, "id": f"{prefix}{i:02}", "split": split, "draw": d}
                     for split, prefix in (("visible", "v"), ("sealed", "s")) for i in range(8) for d in range(4)]
    write(case[1], spec)
    assert B.compare(case[1])["quality_admission"]["status"] == "pass"
    spec["cases"].pop()
    write(case[1], spec)
    assert B.compare(case[1])["quality_admission"]["status"] == "unavailable"
    spec["cases"].append({**original, "id": "s07", "split": "sealed", "draw": 3,
                          "quality": {"fp32": original["quality"]["fp32"]}})
    write(case[1], spec)
    assert B.compare(case[1])["quality_admission"]["status"] == "unavailable"


@pytest.mark.parametrize("changed", ["policy_bytes", "policy_path", "input", "case_id"])
def test_selected_policy_and_functional_input_must_match_protocol(case, changed):
    root, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    if changed == "policy_bytes":
        path = root / "models.json"
        path.write_text(path.read_text() + "\n# changed policy bytes\n")
    elif changed == "policy_path":
        path = root / "other-models.json"
        path.write_bytes((root / "models.json").read_bytes())
        spec["numerical_policy"] = pin(path, root)
    elif changed == "input":
        path = root / "different-input.npz"
        np.savez(path, tokens=np.array([[1, 1, 1, 1]], np.int64))
        spec["cases"][0]["input"] = pin(path, root)
    else:
        spec["cases"][0]["id"] = "v01"
    write(manifest, spec)
    with pytest.raises(ValueError):
        B.compare(manifest)


def test_json_manifest_and_implementation_metadata_are_not_certificates(case):
    _, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    spec["implementation"] = {"name": "example CPU", "revision": "supplied", "numerical_variant": "selected INT8"}
    manifest.write_text(json.dumps(spec))
    result = B.compare(manifest)
    assert result["implementation"] == spec["implementation"]
    assert result["reference_origin_certified"] is result["execution_qualified"] is False


def test_quality_case_relabelling_cannot_replace_selected_observation_bytes(case):
    root, manifest = case
    spec = B.yaml.safe_load(manifest.read_bytes())
    spec["profile"] = "quality_v1"
    path = root / "different-quality-input.npz"
    np.savez(path, tokens=np.array([[9, 8, 7, 6]], np.int64))
    spec["cases"][0]["input"] = pin(path, root)
    write(manifest, spec)
    with pytest.raises(ValueError, match="evaluation manifest"):
        B.compare(manifest)
