"""Explicit full-chain comparison against sealed quantized outputs, without CPU host twins."""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from merlin.model_check import chain as CH
from merlin.model_check import full_reference_cache as RC
from merlin.model_check import metrics as M
from merlin.model_check.models import (
    LEGACY_FLOAT_POLICY,
    ModelsFileError,
    selected_float_policy,
    selected_host_math_policy,
)
from merlin.model_check.models import (
    load as load_models,
)

DEFAULT_ATOL = LEGACY_FLOAT_POLICY["atol"]
DEFAULT_RTOL = LEGACY_FLOAT_POLICY["rtol"]


def policy(args, *, models=None):
    """Use the selected models policy; explicit flags cannot override a declared comparison."""
    if models is None and getattr(args, "models", None) is not None:
        models = load_models(args.models)
    result = selected_float_policy(models)
    for name in ("atol", "rtol"):
        value = getattr(args, "quantized_reference_" + name, None)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ModelsFileError("quantized-reference tolerances must be finite and nonnegative")
        if getattr(models, "numerics", None) is not None and value != result[name]:
            raise ModelsFileError(f"explicit {name} conflicts with selected models numerics.functional.{name}")
        result[name] = value
    return result


def _bindings(model, counts):
    return [
        {"program": ref["program"], "step": step, "output": int(ref["output"])}
        for ref in model.judged["outputs"] for step in range(counts[ref["program"]])
    ]


def _numeric_receipt(path, expected_sha, outputs_sha, selected_policy):
    from merlin.runtime import linalg_numpy as LN
    from merlin.runtime.host_math import host_math_recipe

    payload = path.read_bytes()
    receipt_sha = hashlib.sha256(payload).hexdigest()
    if receipt_sha != expected_sha:
        raise ValueError("sealed reference numerical-contract bytes differ from SHA256SUMS")
    receipt = json.loads(payload)
    if not isinstance(receipt, dict):
        raise ValueError("sealed reference numerical contract must be an object")
    contract = receipt.get("numeric_contract") or {}
    if not isinstance(contract, dict) or not isinstance(contract.get("host_math"), dict):
        raise ValueError("sealed reference numerical contract requires a host math identity")
    host_math = contract["host_math"]
    recipe = host_math_recipe(selected_policy)
    evaluator_sha = receipt.get("evaluator_sha256")
    flags = host_math.get("compile_flags")
    if (
        receipt.get("schema") != "merlin.model_check.reference_numeric_contract.v1"
        or receipt.get("outputs_sha256") != outputs_sha
        or contract.get("version") != LN.NUMERIC_CONTRACT_VERSION
        or host_math.get("policy") != selected_policy
        or recipe.source is None
        or host_math.get("scalar_source_sha256") != hashlib.sha256(recipe.source.encode("utf-8")).hexdigest()
        or not isinstance(flags, list) or any(flag not in flags for flag in recipe.compile_flags)
        or not isinstance(evaluator_sha, str) or len(evaluator_sha) != 64
        or any(char not in "0123456789abcdef" for char in evaluator_sha)
    ):
        raise ValueError("sealed reference numerical contract differs from the selected host math policy")
    return {"path": str(path), "sha256": receipt_sha, "numeric_contract": contract,
            "evaluator_sha256": evaluator_sha}


def sealed_reference(bundle, model, models_path, input_id):
    """Read only sealed, configuration-corresponding reference bytes; retain their identities for recheck."""
    root = bundle.root.resolve()
    manifest_path = root / "MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    seal = manifest.get("seal") or {}
    sums_name = Path(str(seal.get("sums") or "SHA256SUMS"))
    reference_name = Path(model.name) / input_id / "int8.npz"
    if any(name.is_absolute() or ".." in name.parts for name in (sums_name, reference_name)):
        raise ValueError("reference or seal pathname escapes the selected bundle")
    sums = root / sums_name
    path = root / reference_name
    configuration = Path(models_path).read_bytes()
    selected = yaml.safe_load(configuration)
    common = (selected.get("full") or {}) if isinstance(selected, dict) else {}
    if not isinstance(common, dict):
        raise ModelsFileError("models full-mode settings must be a mapping")
    host_math_policy = selected_host_math_policy(model, common)
    receipt_path = path.with_name("reference_numeric_contract.json")
    # Frozen bundles may reuse protected payloads through symlinks. The sealed
    # relative name and bytes bind them; a location-only rejection would refuse valid reuse.
    wanted = {path.relative_to(root).as_posix(), "configuration/models.yaml"}
    if host_math_policy != "native":
        wanted.add(receipt_path.relative_to(root).as_posix())
    expected = {}
    digest = hashlib.sha256()
    with sums.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            value, separator, name = raw.decode().rstrip("\n").partition("  ")
            if separator and name in wanted:
                if name in expected:
                    raise ValueError("duplicate selected reference/configuration entry in bundle seal")
                expected[name] = value
    if digest.hexdigest() != seal.get("sums_sha256") or set(expected) != wanted:
        raise ValueError("selected quantized reference has no valid bundle seal/configuration binding")
    payload = path.read_bytes()
    reference_sha = hashlib.sha256(payload).hexdigest()
    config_sha = hashlib.sha256(configuration).hexdigest()
    if reference_sha != expected[path.relative_to(root).as_posix()]:
        raise ValueError("sealed INT8 reference bytes differ from SHA256SUMS")
    if config_sha != expected["configuration/models.yaml"]:
        raise ValueError("selected models configuration differs from the sealed reference's configuration")
    numeric_receipt = None
    if host_math_policy != "native":
        numeric_receipt = _numeric_receipt(
            receipt_path, expected[receipt_path.relative_to(root).as_posix()], reference_sha, host_math_policy
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        names = [f"out{i}" for i in range(len(data.files))]
        if set(data.files) != set(names):
            raise ValueError("sealed INT8 reference output names/ordering are incomplete")
        values = [np.asarray(data[name]) for name in names]
    counts = {p.name: p.steps for p in model.programs}
    RC.selected_outputs(model, counts, values)  # validates full frozen arity before any prefix selection
    if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in values):
        raise ValueError("sealed INT8 reference must contain supported, entirely finite numeric outputs")
    validation = {"source": "sealed_int8", "path": str(path), "sha256": reference_sha,
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "sums_sha256": digest.hexdigest(), "models_sha256": config_sha,
                  "strict_arity": True, "outputs": [
                      {**binding, "dtype": value.dtype.str, "shape": list(value.shape)}
                      for binding, value in zip(_bindings(model, counts), values, strict=True)
                  ], "unchanged": False}
    if numeric_receipt is not None:
        validation["numeric_contract"] = numeric_receipt
    return values, validation


def recheck_reference(bundle, model, models_path, input_id, before):
    _, after = sealed_reference(bundle, model, models_path, input_id)
    if after != before:
        raise ValueError("sealed reference, model configuration or seal changed during execution")
    return {**before, "unchanged": True}


def compare_outputs(actual, expected, bindings, selected_policy):
    """Compare every element without broadcasting, dtype coercion, nonfinite equality or sampling."""
    if len(actual) != len(expected) or len(bindings) != len(expected):
        return [{"pass": False, "detail": "compiled/reference output arity differs",
                 "actual_outputs": len(actual), "expected_outputs": len(expected)}]
    rows = []
    for value, reference, binding in zip(actual, expected, bindings, strict=True):
        value, reference = np.asarray(value), np.asarray(reference)
        row = {**binding, "dtype": reference.dtype.str, "shape": list(reference.shape),
               "elements": int(reference.size), "pass": False}
        if value.dtype != reference.dtype or value.shape != reference.shape:
            rows.append({**row, "detail": "compiled/reference dtype or shape differs",
                         "actual_dtype": value.dtype.str, "actual_shape": list(value.shape)})
            continue
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all() or not np.isfinite(reference).all():
            rows.append({**row, "detail": "unsupported dtype or nonfinite compiled/reference output"})
            continue
        if value.dtype.kind in "biu":
            matching = value == reference
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                delta = np.abs(value.astype(np.float64) - reference.astype(np.float64))
                bound = selected_policy["atol"] + selected_policy["rtol"] * np.abs(reference.astype(np.float64))
                matching = delta <= bound
            row["max_abs_error"] = (float(delta.max()) if delta.size else 0.0) if np.isfinite(delta).all() else None
        row.update({"pass": bool(np.all(matching)), "mismatched": int(np.count_nonzero(~matching)),
                    "exact": value.tobytes() == reference.tobytes()})
        rows.append(row)
    return rows


def check(args, report, *, models, model, bundle, references, builds, stages, specs, counts,
          inputs, runs, deadline, partial, timing, log=print):
    """Run complete compiled chains and gate only the explicitly selected quantized-reference comparison."""
    from merlin.model_check import full as F

    selected_policy = policy(args, models=models)
    report["comparison"] = {"selected": "quantized-reference", "float_policy": selected_policy,
                            "host_twin": {"selected": False, "status": "not_run"}}
    kernels_built = report["gates"].get("build")
    report["gates"]["build"] = {
        "status": kernels_built["status"] if kernels_built else "pass",
        "detail": f"{len(builds)} program image(s) linked; no CPU host twin selected"
                  + (f"; kernel replay: {kernels_built['detail']}" if kernels_built else ""),
    }

    def one(input_id):
        result = {"input": input_id, "rows": [], "comparisons": [], "complete": False}
        try:
            arrays = F._arrays(F.derived_input(bundle, model.name, input_id))  # noqa: SLF001
            frozen, validation = references[input_id]
            expected = RC.selected_outputs(model, counts, frozen)
            selected = list(zip(_bindings(model, counts), expected, strict=True))
            compared = {}

            def completed_step(program, step, values):
                passed = True
                for index, (binding, reference) in enumerate(selected):
                    if binding["program"] != program or binding["step"] != step:
                        continue
                    output = binding["output"]
                    actual = [values[output]] if output < len(values) else []
                    comparison = compare_outputs(actual, [reference], [binding], selected_policy)[0]
                    compared[index] = {**binding, **comparison}
                    passed = passed and comparison["pass"]
                result["comparisons"] = [compared[index] for index in sorted(compared)]
                result["rows"][-1]["outputs"] = [
                    {"output": i, "dtype": value.dtype.str, "shape": list(value.shape),
                     "comparison": next((row for row in result["comparisons"] if row.get("program") == program
                                         and row.get("step") == step and row.get("output") == i),
                                        {"status": "not_judged"})}
                    for i, value in enumerate(values)
                ]
                return passed

            started = time.monotonic()
            run = F.run_input(
                model, target=models.target, builds=builds, twins={}, caches={}, stages=stages, specs=specs,
                arrays=arrays, counts=counts, work=runs / input_id, deadline=deadline, rows=result["rows"],
                twin_pool=None, comparison="quantized-reference", completed_step=completed_step, log=log,
            )
            result["seconds"] = round(time.monotonic() - started, 1)
            comparisons = result["comparisons"]
            actual_counts = {name: len(values) for name, values in run.steps.items()}
            result["complete"] = actual_counts == counts and len(result["rows"]) == sum(counts.values())
            result["steps_completed"] = actual_counts
            result["pass"] = (result["complete"] and len(comparisons) == len(selected)
                              and bool(comparisons) and all(row["pass"] for row in comparisons))
            result["reference_validation"] = recheck_reference(bundle, model, models.path, input_id, validation)
            # Existing FP data is informational; its absence/failure cannot change the selected gate.
            result["fp_quality"] = {"available": False}
            observed = CH.judged_outputs(model, run)
            if result["complete"] and all(np.isfinite(value).all() for value in observed):
                try:
                    fp = F._ordered(bundle.root / model.name / input_id / "fp.npz")  # noqa: SLF001
                    result["metrics"] = F.judged_metrics(model, run, fp)
                    if model.judged["kind"] == "next_token_distribution":
                        rows = sum(int(np.prod(value.shape[:-1])) for value in expected)
                        fp = [np.asarray(fp[0]).reshape(-1, fp[0].shape[-1])[:rows]]
                    result["reference_metrics"] = M.model_metrics(model.judged["kind"], expected, fp, model.judged)
                    result["fp_quality"] = {"available": True, **F.model_gate(
                        model.judged["kind"], result["metrics"], result["reference_metrics"]
                    )}
                except Exception as error:  # noqa: BLE001 -- this is an explicitly informational result
                    result["fp_quality"] = {"available": False, "detail": f"{type(error).__name__}: {error}"[:500]}
        except Exception as error:  # noqa: BLE001 -- preserve the completed steps and other inputs
            result["error"] = f"{type(error).__name__}: {error}"[:2000]
            log(f"[full] {model.name} {input_id}: {result['error']}")
        return result

    jobs = int(args.jobs) if getattr(args, "jobs", None) else F.default_jobs(len(inputs))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, inputs))
    per_input = []
    for result in results:
        split = result["input"].partition("/")[0]
        if "error" in result:
            report["failures"].append({"kind": "infrastructure", "gate": "quantized_reference",
                                       "input": result["input"], "split": split, "detail": result["error"]})
            continue
        per_input.append({"split": split, "id": result["input"], **result.get("metrics", {}),
                          "complete": result["complete"], "steps_completed": result["steps_completed"],
                          "reference": result.get("reference_metrics"), "reference_consistent": None,
                          "reference_validation": result["reference_validation"], "pass": result["pass"],
                          "quantized_reference": result["comparisons"],
                          "fp_quality": {**result.get("fp_quality", {}), "report_only": True}})
        for failure in (row for row in result["comparisons"] if not row["pass"]):
            report["failures"].append({"kind": "gate", "gate": "quantized_reference",
                                       "input": result["input"], "split": split, **failure,
                                       "detail": failure.get("detail", "output differs from sealed reference")})
        timing["chain_seconds"][result["input"]] = result["seconds"]
    F.record_programs(report, bundle, model, builds, results)
    expected_steps = len(inputs) * sum(counts.values())
    completed_steps = sum(len(result["rows"]) for result in results)
    completed_inputs = sum(result["complete"] and "error" not in result for result in results)
    complete = completed_inputs == len(inputs) and completed_steps == expected_steps
    passed = complete and all(row["pass"] for row in per_input)
    report["gates"]["quantized_reference"] = {
        "status": "pass" if passed else "fail", "partial": partial or not complete,
        "inputs_completed": completed_inputs, "inputs_expected": len(inputs),
        "steps_completed": completed_steps, "steps_expected": expected_steps,
        "steps_declared": len(inputs) * sum(program.steps for program in model.programs),
        "outputs_expected": len(inputs) * len(_bindings(model, counts)), "float_policy": selected_policy,
        "outputs_compared": sum(len(result.get("comparisons", [])) for result in results),
        "detail": f"{completed_inputs}/{len(inputs)} complete compiled input chain(s), "
                  f"{completed_steps}/{expected_steps} steps; completed judged outputs compared immediately "
                  "against the sealed INT8 reference; first mismatch stops that input",
    }
    report["model_metrics"] = {"source": "measured", "inputs": len(per_input), "per_input": per_input,
                               "worst": M.worst_of(per_input), "mean": M.mean_of(per_input),
                               "numerics": {"report_only": True, "partial": partial or not complete,
                                            "detail": "FP-baseline quality is informational"}}
    timing["jobs"] = jobs
    report["timing"].update({key: (round(value, 1) if isinstance(value, float) else value)
                             for key, value in timing.items()})
    return report
