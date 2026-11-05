"""Direct full-chain grading selects sealed quantized outputs explicitly and keeps host-twin defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.model_check import bundle as B
from merlin.model_check import cli
from merlin.model_check import full as F
from merlin.model_check import full_build as FB
from merlin.model_check import full_quantized as FQ
from merlin.model_check import report as R
from merlin.model_check.models import Model, ModelsFileError, Program


def _seal(root):
    sums = root / "SHA256SUMS"
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("MANIFEST.json", "SHA256SUMS"))
    sums.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in paths))
    manifest = json.loads((root / "MANIFEST.json").read_text())
    manifest["seal"] = {"sums": "SHA256SUMS", "sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest()}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    (root / "configuration").mkdir(parents=True)
    configuration = root / "configuration/models.yaml"
    configuration.write_text("selected model configuration\n")
    (root / "m/visible/v00").mkdir(parents=True)
    start = np.array([[0, 1, 2]], np.float32)
    frozen = [start + k for k in (1, 2, 3)]
    np.savez(root / "m/visible/v00/int8.npz", **{f"out{i}": v for i, v in enumerate(frozen)})
    programs = (
        Program("prefill", "prefill", ({"from": "input", "key": "x"},)),
        Program("decode", "decode", ({"from": "output", "program": "prefill", "output": 0, "carry": 0},), 2),
    )
    model = Model("m", "w", {"kind": "causal", "prefill_tokens": 1, "decode_steps": 2}, programs,
                  {"kind": "logits", "outputs": [{"program": p.name, "output": 0} for p in programs]},
                  {"full": {"differential": "cached"}})
    models = SimpleNamespace(target="t", path=configuration, model=lambda _: model)
    manifest = {"schema": B.MANIFEST_SCHEMA, "captures": str(tmp_path / "captures"), "models": {"m": {}}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    _seal(root)
    bundle = B.Bundle(root)
    derived = tmp_path / "derived/m/visible/v00"
    derived.mkdir(parents=True)
    np.savez(derived / "inputs.npz", x=start)
    out = tmp_path / "out"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out))
    args = argparse.Namespace(model="m", split="visible", inputs=["visible/v00"], timeout_s=60,
                              decode_steps=None, jobs=1, host_package=tmp_path / "host",
                              submission=tmp_path / "compiler",
                              program=None, work=out / "build/check", full_comparison="quantized-reference",
                              quantized_reference_atol=2e-6, quantized_reference_rtol=1e-5)
    monkeypatch.setattr(FB.BuildSettings, "of", lambda *_: None)
    from merlin.llvmlower import c_runtime

    monkeypatch.setattr(c_runtime, "_out_specs", lambda *_: [([1, 3], "f32")])
    calls = []

    def build(_stage, **kwargs):
        assert kwargs.get("offload", True), "direct mode must never build CPU-only twins"
        calls.append(kwargs["label"])
        return {"status": "linked", "elf_sha256": "image", "elf_bytes": 1, "sections": {}, "build_hash": "build",
                "seconds": 1, "reused": False, "dedup_weights": True, "device_routed": True}

    monkeypatch.setattr(FB, "build", build)
    monkeypatch.setattr(F, "reference_chain", lambda *_a: pytest.fail("direct mode must reuse sealed references"))

    def execute(_target, _record, _stage, inputs, _work, _deadline, **kwargs):
        assert not kwargs.get("twin"), "no host twin may execute under the direct selection"
        return {"values": [np.asarray(inputs["in0"]) + np.float32(1)],
                "row": {"seconds": 0.1, "cycles_functional": 10}}

    monkeypatch.setattr(F, "execute", execute)
    report = R.new_report("check", model="m", split="visible", submission={"tree_sha256": "compiler"})
    report["gates"] = {"build": {"status": "pass", "detail": "kernel build"},
                       "kernel_exact": {"status": "pass", "partial": False, "detail": "exact kernel replay"}}
    report["programs"] = [{"program": p.name, "census": {"denominator": 1, "device": 1},
                            "kernels": [{"symbol": "kernel", "status": "pass"}]} for p in programs]
    return args, report, models, model, bundle, calls


def test_cli_keeps_host_twin_default_and_selects_explicit_float_policy():
    base = ["check", "--submission", "s", "--model", "m", "--split", "visible", "--mode", "full", "--out", "o"]
    args = cli.build_parser().parse_args(base)
    assert args.full_comparison == "host-twin"
    args = cli.build_parser().parse_args(base + ["--full-comparison", "quantized-reference"])
    assert FQ.policy(args) == {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True}
    for invalid in (-1, float("nan"), float("inf")):
        args.quantized_reference_atol = invalid
        with pytest.raises(ModelsFileError, match="finite and nonnegative"):
            F.validate(args)


def test_complete_chain_uses_every_sealed_output_without_twins_or_numpy_recompute(tmp_path, monkeypatch):
    args, report, models, _model, bundle, calls = _setup(tmp_path, monkeypatch)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    R.settle(result)
    assert result["status"] == "pass" and calls == ["m:prefill", "m:decode"]
    assert result["comparison"]["host_twin"] == {"selected": False, "status": "not_run"}
    assert "differential" not in result["gates"]
    gate = result["gates"]["quantized_reference"]
    assert gate["status"] == "pass" and not gate["partial"]
    assert gate["steps_completed"] == gate["steps_expected"] == gate["steps_declared"] == 3
    assert gate["outputs_compared"] == gate["outputs_expected"] == 3
    judged = result["model_metrics"]["per_input"][0]
    assert judged["reference_consistent"] is None
    assert judged["reference_validation"]["unchanged"] and judged["reference_validation"]["strict_arity"]
    assert all(row["elements"] == 3 and row["pass"] for row in judged["quantized_reference"])
    assert judged["fp_quality"] == {"available": False, "detail": judged["fp_quality"]["detail"], "report_only": True}
    assert sum(len(row["runs"]) for row in result["programs"]) == 3
    R.write(result, tmp_path / "full-report.json")


def test_partial_chain_and_failed_kernel_replay_cannot_be_complete(tmp_path, monkeypatch):
    args, report, models, _model, bundle, calls = _setup(tmp_path, monkeypatch)
    args.decode_steps = 1
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert result["gates"]["quantized_reference"]["partial"]
    assert R.settle(result)["status"] == "fail"
    calls.clear()
    report["gates"]["kernel_exact"]["status"] = "fail"
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert not calls and result["gates"]["quantized_reference"]["status"] == "fail"


@pytest.mark.parametrize("change", ["reference", "source_configuration", "seal", "arity", "reference_nonfinite"])
def test_sealed_reference_mismatch_refuses_before_build(tmp_path, monkeypatch, change):
    args, report, models, model, bundle, calls = _setup(tmp_path, monkeypatch)
    path = bundle.root / "m/visible/v00/int8.npz"
    if change == "source_configuration":
        models.path.write_text("different model configuration\n")
    elif change == "seal":
        (bundle.root / "SHA256SUMS").write_text("forged seal\n")
    elif change == "reference":
        np.savez(path, out0=np.ones((1, 3), np.float32))
    else:
        outputs = [np.ones((1, 3), np.float32)] * (2 if change == "arity" else 3)
        if change == "reference_nonfinite":
            outputs[0] = np.full((1, 3), np.nan, np.float32)
        np.savez(path, **{f"out{i}": value for i, value in enumerate(outputs)})
        _seal(bundle.root)
    with pytest.raises(ValueError):
        F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert not calls


def _portable_reference(tmp_path, monkeypatch, *, selection="common"):
    from dataclasses import replace

    from merlin.runtime import linalg_numpy as LN
    from merlin.runtime.host_math import host_math_recipe

    args, report, models, model, bundle, calls = _setup(tmp_path, monkeypatch)
    policy = "portable_trig"
    if selection == "common":
        models.path.write_text(json.dumps({"full": {"host_math_policy": policy}}))
    else:
        model = replace(model, extra={"full": {"host_math_policy": policy}})
        models.model = lambda _: model
    recipe = host_math_recipe(policy)
    path = bundle.root / "m/visible/v00/reference_numeric_contract.json"
    reference = path.with_name("int8.npz")
    contract = {**LN.numeric_contract(), "host_math": {
        "policy": policy, "scalar_source_sha256": hashlib.sha256(recipe.source.encode()).hexdigest(),
        "compile_flags": list(recipe.compile_flags),
    }}
    receipt = {"schema": "merlin.model_check.reference_numeric_contract.v1", "numeric_contract": contract,
               "evaluator_sha256": hashlib.sha256(Path(LN.__file__).read_bytes()).hexdigest(),
               "outputs_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    path.write_text(json.dumps(receipt))
    _seal(bundle.root)
    return args, report, models, model, bundle, calls, path, receipt


@pytest.mark.parametrize("selection", ["common", "model"])
def test_portable_policy_requires_corresponding_sealed_numerical_receipt(tmp_path, monkeypatch, selection):
    args, report, models, _model, bundle, calls, path, receipt = _portable_reference(
        tmp_path, monkeypatch, selection=selection
    )
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "pass" and len(calls) == 2
    proof = result["model_metrics"]["per_input"][0]["reference_validation"]
    assert proof["unchanged"]
    assert proof["numeric_contract"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert proof["numeric_contract"]["numeric_contract"] == receipt["numeric_contract"]


@pytest.mark.parametrize("change", [
    "missing", "unsealed", "changed", "outputs", "version", "policy", "source", "flags",
    "evaluator", "schema", "nonobject", "contract_type",
])
def test_portable_reference_receipt_mismatch_refuses_before_build(tmp_path, monkeypatch, change):
    args, report, models, _model, bundle, calls, path, receipt = _portable_reference(tmp_path, monkeypatch)
    if change == "missing":
        path.unlink()
    elif change == "unsealed":
        payload = path.read_bytes()
        path.unlink()
        _seal(bundle.root)
        path.write_bytes(payload)
    elif change == "changed":
        path.write_text(json.dumps({**receipt, "evaluator_sha256": "a" * 64}))
    else:
        if change in {"outputs", "evaluator"}:
            receipt[f"{change}_sha256"] = "wrong"
        elif change == "version":
            receipt["numeric_contract"]["version"] = "old evaluator policy"
        elif change == "policy":
            receipt["numeric_contract"]["host_math"]["policy"] = "native"
        elif change == "source":
            receipt["numeric_contract"]["host_math"]["scalar_source_sha256"] = "0" * 64
        elif change == "flags":
            receipt["numeric_contract"]["host_math"]["compile_flags"] = ["-fno-builtin"]
        elif change == "schema":
            receipt["schema"] = "unrecognized receipt"
        elif change == "nonobject":
            receipt = []
        elif change == "contract_type":
            receipt["numeric_contract"] = "not an identity"
        path.write_text(json.dumps(receipt))
    if change not in {"unsealed", "changed"}:
        _seal(bundle.root)
    with pytest.raises(ValueError):
        F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert not calls


def test_portable_receipt_mutation_during_execution_is_an_infrastructure_failure(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls, path, _receipt = _portable_reference(tmp_path, monkeypatch)
    original = F.execute

    def execute(*a, **kwargs):
        result = original(*a, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(F, "execute", execute)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "error"
    assert result["failures"][0]["kind"] == "infrastructure"


def test_native_legacy_references_need_no_sidecar_or_unselected_c_source(tmp_path, monkeypatch):
    from merlin.runtime import host_math

    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original = host_math.host_math_recipe

    def recipe(policy="native"):
        assert policy == "native", "native reference validation must not load an unselected C policy"
        return original(policy)

    monkeypatch.setattr(host_math, "host_math_recipe", recipe)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "pass"
    assert "numeric_contract" not in result["model_metrics"]["per_input"][0]["reference_validation"]


def test_changing_and_resealing_config_cannot_relabel_native_reference_as_portable(tmp_path, monkeypatch):
    args, report, models, _model, bundle, calls = _setup(tmp_path, monkeypatch)
    models.path.write_text(json.dumps({"full": {"host_math_policy": "portable_trig"}}))
    _seal(bundle.root)
    with pytest.raises(ValueError, match="bundle seal/configuration binding"):
        F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert not calls


def test_model_native_override_does_not_require_common_portable_receipt(tmp_path, monkeypatch):
    from dataclasses import replace

    args, report, models, model, bundle, _calls = _setup(tmp_path, monkeypatch)
    models.path.write_text(json.dumps({"full": {"host_math_policy": "portable_trig"}}))
    model = replace(model, extra={"full": {"host_math_policy": "native"}})
    models.model = lambda _: model
    _seal(bundle.root)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "pass"


def test_reference_mutation_during_execution_is_not_credited(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original = F.execute

    def execute(*a, **kwargs):
        result = original(*a, **kwargs)
        models.path.write_text("source configuration changed while executing\n")
        return result

    monkeypatch.setattr(F, "execute", execute)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert result["gates"]["quantized_reference"]["status"] == "fail"
    assert R.settle(result)["status"] == "error"


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.bool_])
def test_integer_and_boolean_outputs_are_exact_even_with_large_float_tolerance(dtype):
    a, b = np.array([1, 0], dtype), np.array([1, 1], dtype)
    row = FQ.compare_outputs([a], [b], [{}], {"atol": 1e9, "rtol": 1e9})[0]
    assert not row["pass"] and row["mismatched"] == 1


def test_float_policy_checks_all_elements_and_rejects_nonfinite_shape_dtype_or_arity():
    policy = {"atol": 2e-6, "rtol": 1e-5}
    expected = np.array([1, 0], np.float32)
    accepted = np.array([1.00001, 0.000001], np.float32)
    assert FQ.compare_outputs([accepted], [expected], [{}], policy)[0]["pass"]
    for value in [np.array([1, 0.001], np.float32), np.array([1, np.nan], np.float32),
                  np.array([1, np.inf], np.float32), expected.astype(np.float64), expected.reshape(1, 2)]:
        assert not FQ.compare_outputs([value], [expected], [{}], policy)[0]["pass"]
    assert not FQ.compare_outputs([], [expected], [{}], policy)[0]["pass"]
    assert not FQ.compare_outputs([expected, expected], [expected], [{}], policy)[0]["pass"]
    both_nan = np.array([np.nan], np.float32)
    assert not FQ.compare_outputs([both_nan], [both_nan], [{}], policy)[0]["pass"]


def test_nonfinite_compiled_outputs_fail_selected_gate(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)

    def execute(*_a, **_kwargs):
        return {"values": [np.full((1, 3), np.nan, np.float32)], "row": {"seconds": 0.1, "cycles_functional": 10}}

    monkeypatch.setattr(F, "execute", execute)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "fail"
    assert result["gates"]["quantized_reference"]["status"] == "fail"


def test_lexically_stored_many_reference_arrays_are_selected_in_numeric_order(tmp_path, monkeypatch):
    args, _report, models, model, bundle, _calls = _setup(tmp_path, monkeypatch)
    from dataclasses import replace

    model = replace(model, programs=(model.programs[0], replace(model.programs[1], steps=12)))
    path = bundle.root / "m/visible/v00/int8.npz"
    values = {f"out{i}": np.array([[i]], np.float32) for i in range(13)}
    np.savez(path, **{name: values[name] for name in sorted(values)})
    _seal(bundle.root)
    actual, _proof = FQ.sealed_reference(bundle, model, models.path, "visible/v00")
    assert [value.item() for value in actual] == list(range(13))


def test_sealed_reference_payload_symlinks_are_validated_by_their_bound_bytes(tmp_path, monkeypatch):
    _args, _report, models, model, bundle, _calls = _setup(tmp_path, monkeypatch)
    path = bundle.root / "m/visible/v00/int8.npz"
    payload = tmp_path / "protected-reused-reference.npz"
    path.rename(payload)
    path.symlink_to(payload)
    values, proof = FQ.sealed_reference(bundle, model, models.path, "visible/v00")
    assert len(values) == 3 and proof["strict_arity"]
    np.savez(payload, out0=np.array([99], np.float32))
    with pytest.raises(ValueError, match="reference bytes differ"):
        FQ.sealed_reference(bundle, model, models.path, "visible/v00")


def test_unavailable_fp_quality_is_report_only_for_the_selected_comparison(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    (bundle.root / "m/visible/v00/fp.npz").write_bytes(b"unavailable optional FP data")
    _seal(bundle.root)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "pass"
    fp = result["model_metrics"]["per_input"][0]["fp_quality"]
    assert fp["available"] is False and fp["report_only"] is True


def test_exact_kernel_evidence_cannot_substitute_for_wrong_finite_full_outputs(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original = F.execute

    def execute(*a, **kwargs):
        result = original(*a, **kwargs)
        result["values"][0][0, -1] += np.float32(0.1)
        return result

    monkeypatch.setattr(F, "execute", execute)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert result["gates"]["kernel_exact"]["status"] == "pass"
    assert result["gates"]["quantized_reference"]["status"] == "fail"
    assert R.settle(result)["status"] == "fail"


@pytest.mark.parametrize("failed_step,executions", [(None, 3), (0, 1), (1, 2), (2, 3)])
def test_selected_comparison_stops_after_first_failed_step_and_keeps_evidence(
    tmp_path, monkeypatch, failed_step, executions
):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original = F.execute
    completed = []

    def execute(*a, **kwargs):
        result = original(*a, **kwargs)
        index = len(completed)
        completed.append(index)
        if index == failed_step:
            result["values"][0][0, 0] += np.float32(0.1)
        return result

    monkeypatch.setattr(F, "execute", execute)
    # A stopped prefix must not compute or present FP metrics for a full input.
    if executions < 3:
        monkeypatch.setattr(F, "judged_metrics", lambda *_: pytest.fail("incomplete chain has no FP metrics"))
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert completed == list(range(executions))
    assert R.settle(result)["status"] == ("pass" if failed_step is None else "fail")
    assert all(failure["kind"] == "gate" for failure in result["failures"])
    gate = result["gates"]["quantized_reference"]
    assert gate["steps_completed"] == gate["outputs_compared"] == executions
    assert gate["steps_expected"] == gate["outputs_expected"] == 3
    assert gate["inputs_completed"] == int(executions == 3)
    assert gate["partial"] is (executions < 3)
    judged = result["model_metrics"]["per_input"][0]
    assert judged["reference_validation"]["unchanged"]
    assert judged["complete"] is (executions == 3)
    assert len(judged["quantized_reference"]) == executions
    runs = [run for program in result["programs"] for run in program["runs"]]
    assert len(runs) == executions and all(run["outputs"] for run in runs)
    assert all(run["comparison"]["selected"] == "quantized-reference" for run in runs)
    if failed_step is not None:
        assert not judged["quantized_reference"][-1]["pass"]
        assert result["failures"][0]["step"] == (failed_step - 1 if failed_step else 0)
    R.write(result, tmp_path / "early-stop-report.json")


def test_early_mismatch_still_rechecks_selected_reference_bytes(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original = F.execute
    completed = []

    def execute(*a, **kwargs):
        result = original(*a, **kwargs)
        completed.append(True)
        result["values"][0][0, 0] += np.float32(0.1)
        models.path.write_text("source configuration mutated before early rejection\n")
        return result

    monkeypatch.setattr(F, "execute", execute)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert len(completed) == 1
    assert R.settle(result)["status"] == "error"
    assert result["failures"][0]["kind"] == "infrastructure"
    assert result["gates"]["quantized_reference"]["inputs_completed"] == 0
    first = result["programs"][0]["runs"][0]
    assert first["outputs"][0]["comparison"]["pass"] is False


def test_completed_comparison_survives_later_execution_error(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original = F.execute
    completed = []

    def execute(*a, **kwargs):
        if completed:
            raise RuntimeError("later execution failed")
        completed.append(True)
        return original(*a, **kwargs)

    monkeypatch.setattr(F, "execute", execute)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "error"
    gate = result["gates"]["quantized_reference"]
    assert gate["steps_completed"] == gate["outputs_compared"] == 1
    assert gate["inputs_completed"] == 0 and gate["partial"]
    first = result["programs"][0]["runs"][0]
    assert first["outputs"][0]["comparison"]["pass"] is True


def test_completed_step_callback_cannot_change_default_host_twin_selection(tmp_path, monkeypatch):
    _args, _report, _models, model, _bundle, _calls = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires quantized-reference"):
        F.run_input(model, target="t", builds={}, twins={}, caches={}, stages={}, specs={}, arrays={}, counts={},
                    work=tmp_path, deadline=0, rows=[], twin_pool=None, completed_step=lambda *_: False)


def test_step_comparisons_retain_declared_judged_order_instead_of_execution_order(tmp_path, monkeypatch):
    from dataclasses import replace

    args, report, models, model, bundle, _calls = _setup(tmp_path, monkeypatch)
    model = replace(model, judged={**model.judged, "outputs": list(reversed(model.judged["outputs"]))})
    models.model = lambda _: model
    path = bundle.root / "m/visible/v00/int8.npz"
    with np.load(path) as data:
        expected = [np.asarray(data[f"out{i}"]) for i in (1, 2, 0)]
    np.savez(path, **{f"out{i}": value for i, value in enumerate(expected)})
    _seal(bundle.root)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert R.settle(result)["status"] == "pass"
    comparisons = result["model_metrics"]["per_input"][0]["quantized_reference"]
    assert [(row["program"], row["step"]) for row in comparisons] == [
        ("decode", 0), ("decode", 1), ("prefill", 0)
    ]


def test_direct_comparison_cannot_silently_apply_to_fast_mode(tmp_path):
    output = tmp_path / "usage.json"
    code = cli.main(["check", "--submission", str(tmp_path / "unused-compiler"), "--model", "m",
                     "--split", "visible", "--mode", "fast", "--full-comparison", "quantized-reference",
                     "--out", str(output)])
    assert code == R.EXIT_USAGE
    assert "requires --mode full" in json.loads(output.read_text())["failures"][0]["detail"]


def test_full_check_forwards_loaded_policy_when_cli_models_is_implicit(tmp_path, monkeypatch):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    models.numerics = {"version": "merlin.model_check.numerics.v1", "functional": {
        "atol": .125, "rtol": 0.0, "equal_nan": False, "finite_required": True,
        "integer_boolean": "exact", "outputs": "all_declared_every_step",
    }}
    args.models = None
    args.quantized_reference_atol = args.quantized_reference_rtol = None
    original = F.execute
    executions = 0

    def observed(*a, **kw):
        nonlocal executions
        row = original(*a, **kw)
        if executions == 0:
            row["values"][0] += np.float32(.1)
        executions += 1
        return row

    monkeypatch.setattr(F, "execute", observed)
    result = F.check(args, report, models=models, bundle=bundle, log=lambda *_: None)
    assert result["comparison"]["float_policy"]["atol"] == .125
    assert result["gates"]["quantized_reference"]["status"] == "pass"
    assert executions == 3


@pytest.mark.parametrize("selected", [False, True])
def test_host_fusion_override_reaches_provider_without_changing_reference_configuration(
    tmp_path, monkeypatch, selected
):
    args, report, models, _model, bundle, _calls = _setup(tmp_path, monkeypatch)
    original_configuration = models.path.read_bytes()
    settings = FB.BuildSettings(
        "board", tmp_path / "boards", tmp_path / "host.dts", "scalar", 1, tmp_path / "host",
        host_elementwise_fusion=not selected,
    )
    monkeypatch.setattr(FB.BuildSettings, "of", lambda *_: settings)
    args.host_elementwise_fusion = selected
    observed = []

    def provider(stage, **kwargs):
        observed.append(kwargs["settings"].host_elementwise_fusion)
        return FB.build(stage, **kwargs)

    result = F.check(
        args, report, models=models, bundle=bundle, build_provider=provider, log=lambda *_: None
    )
    assert observed == [selected, selected]
    assert settings.host_elementwise_fusion is not selected
    assert models.path.read_bytes() == original_configuration
    gate = result["gates"]["quantized_reference"]
    assert gate["status"] == "pass" and gate["outputs_compared"] == gate["outputs_expected"] == 3
