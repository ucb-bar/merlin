"""Offline supplied-output comparison; no execution, placement or reference-origin certificate."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml

from merlin.common.digest import is_sha256
from merlin.model_check import full_quantized as FQ
from merlin.model_check import metrics as M
from merlin.model_check import models as MD
from merlin.model_check import numerics_report as NR

SCHEMA = "merlin.model_check.baseline_outputs.v1"
RESULT_SCHEMA = "merlin.model_check.baseline_comparison.v1"
DTYPES = {"f32": "float32", "f64": "float64", "i1": "bool", "i8": "int8", "i32": "int32", "i64": "int64"}


class Reader:
    def __init__(self, root):
        self.root, self.files = root, {}

    def read(self, record, *, root=None):
        path = Path(record["path"])
        path = ((root or self.root) / path).absolute() if not path.is_absolute() else path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected a regular supplied artifact: {path}")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if not is_sha256(record.get("sha256")) or sha != record["sha256"]:
            raise ValueError(f"supplied artifact SHA256 differs: {path}")
        if str(path) in self.files and self.files[str(path)] != sha:
            raise ValueError("artifact changed between supplied uses")
        self.files[str(path)] = sha
        return path, data

    def arrays(self, record, *, outputs=True):
        _, data = self.read(record)
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("duplicate NPZ array name")
            if outputs and set(archive.files) != {f"out{i}" for i in range(len(archive.files))}:
                raise ValueError("outputs must be exactly out0..outN without gaps")
            names = [f"out{i}" for i in range(len(archive.files))] if outputs else archive.files
            return {k: np.asarray(archive[k]) for k in names}

    def finish(self):
        for path, sha in self.files.items():
            self.read({"path": path, "sha256": sha})
        return [{"path": p, "sha256": sha} for p, sha in sorted(self.files.items())]


def _quality(values, fp, judged):
    result = NR._quality(judged["kind"], values, fp, judged)
    if result["available"]:
        metrics = result["legacy_metrics"]
        if any(not isinstance(metrics.get(r["metric"]), (int, float))
               or not math.isfinite(metrics[r["metric"]]) for r in judged.get("admission", [])):
            return {"available": False, "reason": "required quality metric is missing or nonfinite"}
    return result


def _case(case, reader, model, definition, policy):
    schedule = [(p, s) for p, stage in definition["stages"].items() for s in range(stage["steps"])]
    if [(r["program"], r["step"]) for r in case["steps"]] != schedule:
        raise ValueError("case must contain every selected program/step exactly once in order")
    inputs = reader.arrays(case["input"], outputs=False)
    selected = {(r["program"], r["output"]) for r in model.judged["outputs"]}
    judged_values, rows = {}, []
    for step in case["steps"]:
        program = step["program"]
        outputs = definition["stages"][program]["outputs"]
        actual, reference = reader.arrays(step["actual"]), reader.arrays(step["reference"])
        if len(reference) != len(outputs) or len(actual) != len(outputs):
            raise ValueError("supplied output arity differs from the selected protocol")
        for i, output in enumerate(outputs):
            a, r = actual[f"out{i}"], reference[f"out{i}"]
            if r.dtype != np.dtype(DTYPES[output["dtype"]]) or list(r.shape) != output["shape"]:
                raise ValueError("reference shape/dtype differs from the selected protocol")
            binding = {"program": program, "step": step["step"], "output": i}
            row = FQ.compare_outputs([a], [r], [binding], policy)[0]
            rows.append({**row, "errors": NR.array_errors(a, r)})
            if (program, i) in selected:
                judged_values[(program, step["step"], i)] = (a, r)
    order = [(r["program"], s, r["output"]) for r in model.judged["outputs"]
             for s in range(definition["stages"][r["program"]]["steps"])]
    judged = [judged_values[key][0] for key in order]
    quantized = [judged_values[key][1] for key in order]
    quality = case.get("quality") or {}
    fp = list(reader.arrays(quality["fp32"]).values()) if quality.get("fp32") else []
    result = {"id": case["id"], "split": case["split"], "draw": case["draw"], "comparisons": rows,
              "functional_passed": all(r["pass"] for r in rows),
              "actual_vs_fp32": _quality(judged, fp, model.judged),
              "quantized_reference_vs_fp32": _quality(quantized, fp, model.judged)}
    if quality.get("native_baseline"):
        baseline = list(reader.arrays(quality["native_baseline"]).values())
        result["native_baseline_vs_fp32"] = _quality(baseline, fp, model.judged)
    if model.judged.get("reporting", {}).get("perplexity"):
        selection = model.judged["reporting"]["perplexity"]
        if any(selection.get(k) != v for k, v in {
            "targets": "teacher_forced_continuation", "score_tokens": "decode_steps",
            "exclude_final_decode_logit": True, "padding": "none", "threshold": "report_only",
        }.items()):
            raise ValueError("unsupported continuation PPL reporting policy")
        result["perplexity"] = {}
        for name, arrays in (("actual", judged), ("quantized", quantized), ("fp32", fp)):
            result["perplexity"][name] = (
                NR.perplexity(np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays]),
                              inputs["tokens"], model.chain["prefill_tokens"], model.chain["decode_steps"])
                if arrays and "tokens" in inputs else {"available": False, "reason": "missing logits/tokens"})
    return result


def _admission(rows, model, protocol, field="actual_vs_fp32"):
    cohort = protocol["profiles"]["quality_v1"]
    count = cohort["observations_per_model_per_split"]
    draws = model.judged.get("reference", {}).get("seeds", {}).get("count", 1)
    expected = {(f"{prefix}{i:02}", split, d) for split, prefix in (("visible", "v"), ("sealed", "s"))
                for i in range(count) for d in range(draws)}
    observed = {(r["id"], r["split"], r["draw"]) for r in rows}
    if observed != expected:
        return {"status": "unavailable", "reason": "complete exact visible/sealed id/draw cohort required",
                "expected_chains": len(expected), "observed_chains": len(observed)}
    result = {}
    for split in ("visible", "sealed"):
        group = [r for r in rows if r["split"] == split]
        needs_baseline = any("max_ratio_to_baseline" in rule for rule in model.judged.get("admission", []))
        if any(not r[field]["available"] or (needs_baseline and not
               r.get("native_baseline_vs_fp32", {}).get("available")) for r in group):
            return {"status": "unavailable", "reason": "missing aligned FP/native-baseline quality outputs"}
        result[split] = M.admission(model.judged.get("admission", []),
                                   [r[field]["legacy_metrics"] for r in group],
                                   [r["native_baseline_vs_fp32"]["legacy_metrics"] for r in group]
                                   if needs_baseline else [])
    return {"status": "pass" if all(r["status"] == "pass" for r in result.values()) else "fail",
            "splits": result, "scope": "numerical rules on supplied cohort; origin/execution not certified"}


def compare(manifest_path: Path) -> dict:
    """Compare supplied NPZs using the protocol, shared policy and existing numerical primitives."""
    manifest_path = Path(manifest_path).absolute()
    reader = Reader(manifest_path.parent)
    checker_sources = []
    for path in (Path(__file__), *(Path(module.__file__) for module in (MD, FQ, M, NR))):
        path = path.resolve()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        reader.files[str(path)] = sha
        checker_sources.append({"path": str(path), "sha256": sha})
    _, manifest_bytes = reader.read({"path": str(manifest_path),
                                    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    manifest = yaml.safe_load(manifest_bytes)
    if manifest.get("schema") != SCHEMA or manifest.get("profile") not in {"functional_v1", "quality_v1"}:
        raise ValueError("select a versioned functional_v1 or quality_v1 baseline manifest")
    protocol_path, payload = reader.read(manifest["protocol"])
    protocol = yaml.safe_load(payload)
    schema = protocol.get("schema", "")
    if (not isinstance(schema, str) or not schema.startswith("merlin.")
            or not schema.endswith(".experiment_protocol.v1")):
        raise ValueError("unsupported workload protocol")
    policy_path = (protocol_path.parent / protocol["authority"]["numerical_contract"]).absolute()
    selected_policy, policy_bytes = reader.read(manifest["numerical_policy"])
    if selected_policy.resolve() != policy_path.resolve():
        raise ValueError("numerical policy is not the protocol-linked models file")
    models = MD.load(policy_path, source_bytes=policy_bytes)
    model = models.model(manifest["model"])
    definition = protocol["models"][model.name]
    input_manifest_path = (protocol_path.parent / protocol["authority"]["evaluation_input_manifest"]).absolute()
    selected_inputs, input_manifest_bytes = reader.read(manifest["evaluation_inputs"])
    if selected_inputs.resolve() != input_manifest_path.resolve():
        raise ValueError("evaluation inputs are not the protocol-linked manifest")
    derived = yaml.safe_load(input_manifest_bytes)["models"][model.name]["derived"]
    member = definition["external_evaluation_input"]["derived_member"].split("/", 2)[2]
    if model.chain != definition["chain"] or [(p.name, p.steps) for p in model.programs] != [
            (p, stage["steps"]) for p, stage in definition["stages"].items()]:
        raise ValueError("protocol and numerical model chain/program roster differ")
    if (sum(s["steps"] for s in definition["stages"].values()) != definition["selected_chain_steps"]
            or sum(s["steps"] * len(s["outputs"]) for s in definition["stages"].values())
            != definition["declared_arrays_per_chain"]):
        raise ValueError("protocol total step/output counts disagree")
    policy = MD.selected_float_policy(models)
    reader.read(manifest["reference_manifest"])  # identity only; this is not a trusted-reference certificate
    cases = manifest["cases"]
    keys = [(r["id"], r["split"], r["draw"]) for r in cases]
    if not keys or len(set(keys)) != len(keys) or any(
        s not in {"visible", "sealed"} or type(d) is not int or d < 0 for _, s, d in keys
    ):
        raise ValueError("empty, duplicate or invalid case id/split/draw roster")
    if manifest["profile"] == "functional_v1":
        selection = definition["external_evaluation_input"]
        split, input_id = selection["selection"].split("/")
        if keys != [(input_id, split, 0)] or cases[0]["input"]["sha256"] != selection["sha256"]:
            raise ValueError("functional case/input bytes differ from the protocol selection")
    if any(case["input"]["sha256"] != derived.get(f"{case['split']}/{case['id']}/{member}") for case in cases):
        raise ValueError("case input SHA256 differs from the selected evaluation manifest")
    rows = [_case(case, reader, model, definition, policy) for case in cases]
    functional = all(row["functional_passed"] for row in rows)
    admission = _admission(rows, model, protocol)
    passed = functional and (manifest["profile"] == "functional_v1" or admission["status"] == "pass")
    ppl = {}
    for name in ("actual", "quantized", "fp32"):
        scores = [r.get("perplexity", {}).get(name, {}) for r in rows]
        if scores and all(s.get("available") for s in scores):
            count, total = sum(s["scored_tokens"] for s in scores), sum(s["nll_sum"] for s in scores)
            mean = total / count
            overflow = mean > math.log(np.finfo(np.float64).max)
            ppl[name] = {"available": True, "scored_tokens": count, "nll_sum": total, "nll": mean,
                         "ppl": None if overflow else math.exp(mean), "ppl_overflow": overflow, "report_only": True}
        else:
            ppl[name] = {"available": False, "report_only": True}
    implementation = manifest.get("implementation", {})
    if (not isinstance(implementation, dict) or set(implementation) - {"name", "revision", "numerical_variant"}
            or any(not isinstance(v, str) for v in implementation.values())):
        raise ValueError("unsupported implementation metadata")
    return {"schema": RESULT_SCHEMA, "model": model.name, "profile": manifest["profile"],
            "numerical_status": ("pass" if passed else "fail" if not functional
                                 or admission["status"] == "fail" else "incomplete"),
            "functional_passed": functional, "quality_admission": admission, "cases": rows,
            "quality_admission_selected": manifest["profile"] == "quality_v1", "perplexity": ppl,
            "quantized_reference_quality_admission": _admission(rows, model, protocol, "quantized_reference_vs_fp32"),
            "implementation": implementation, "float_policy": policy, "artifacts": reader.finish(),
            "checker_sources": checker_sources,
            "execution_qualified": False, "reference_origin_certified": False, "placement_qualified": False,
            "scope": "Offline supplied-output checks only; no execution or model qualification."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(args.manifest)
        with args.out.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return 0 if result["numerical_status"] == "pass" else 2
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
