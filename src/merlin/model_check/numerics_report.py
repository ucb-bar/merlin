"""Offline section numerics from retained all-output audits; no model or simulator execution."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from merlin.model_check import metrics as M
from merlin.model_check.full_output_audit import SCHEMA as AUDIT_SCHEMA
from merlin.model_check.full_output_audit import Files, _argument
from merlin.model_check.models import load

SCHEMA = "merlin.model_check.section_numerics.v1"
POLICY = "merlin.model_check.numerics.v1"
REPORTING = {
    "reduction_dtype": "float64",
    "errors": ["reference_rms", "rmse", "mae", "max_abs_error", "relative_l2"],
    "zero_reference_energy": "relative_l2_unavailable_report_absolute_errors",
    "sections": ["program", "step", "output", "channel", "kernel_group", "capture_attention_site"],
    "percentile": "linear",
    "missing_data": "unavailable_never_pass",
    "mixed_units": "no_cross_tensor_error_average",
}


def array_errors(actual, reference) -> dict:
    """Absolute diagnostics, with unavailable relative error for zero reference energy."""
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.dtype != b.dtype or not a.size:
        return {"available": False, "reason": "shape/dtype mismatch or empty array"}
    if a.dtype.kind not in "biuf" or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"available": False, "reason": "unsupported dtype or nonfinite values"}
    # Subtract wide integers exactly before converting the error to float64.
    if a.dtype.kind in "iu" and a.dtype.itemsize == 8:
        error = np.asarray(a.astype(object) - b.astype(object), np.float64)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            error = a.astype(np.float64) - b.astype(np.float64)
    if not np.isfinite(error).all():
        return {"available": False, "reason": "difference exceeds float64 diagnostic range"}

    def rms(values):
        scale = float(np.max(np.abs(values)))
        return scale * float(np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0

    ref_rms, rmse = rms(b.astype(np.float64)), rms(error)
    scale = float(np.max(np.abs(error)))
    relative = rmse / ref_rms if ref_rms else None
    return {
        "available": True,
        "elements": int(a.size),
        "reference_rms": ref_rms,
        "rmse": rmse,
        "mae": scale * float(np.mean(np.abs(error) / scale)) if scale else 0.0,
        "max_abs_error": float(np.max(np.abs(error))),
        "relative_l2": relative if relative is None or math.isfinite(relative) else None,
        "relative_l2_overflow": relative is not None and not math.isfinite(relative),
        "zero_reference_energy": ref_rms == 0.0,
        "mismatched": int(np.count_nonzero(a != b)),
    }


def perplexity(logits, tokens, context: int, decode: int, mask=None) -> dict:
    """Teacher-forced continuation NLL: prefill last + first D-1 decodes, excluding final decode."""
    z, t = np.asarray(logits), np.asarray(tokens)
    if t.ndim == 1:
        t = t[None, :]
    if z.ndim == 2:
        z = z[None, ...]
    if (
        z.ndim != 3
        or t.ndim != 2
        or context < 1
        or decode < 1
        or z.shape[:2] != (t.shape[0], decode + 1)
        or t.shape[1] < context + decode
        or t.dtype.kind not in "iu"
        or not np.isfinite(z).all()
    ):
        return {"available": False, "reason": "incomplete/malformed logits or target token alignment"}
    targets = t[:, context : context + decode]
    valid = np.ones(targets.shape, bool) if mask is None else np.asarray(mask)
    if valid.dtype != np.bool_ or valid.shape != targets.shape or not valid.any():
        return {"available": False, "reason": "invalid or empty score mask"}
    wanted, scored = targets[valid], z[:, :decode, :][valid]
    if np.any(wanted < 0) or np.any(wanted >= z.shape[-1]):
        return {"available": False, "reason": "target token outside vocabulary"}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        losses = -np.take_along_axis(M.log_softmax(scored), wanted[:, None], axis=-1).reshape(-1)
        nll = float(np.sum(losses, dtype=np.float64))
    if not math.isfinite(nll):
        return {"available": False, "reason": "NLL exceeds float64 diagnostic range"}
    mean = nll / losses.size
    return {
        "available": True,
        "scored_tokens": int(losses.size),
        "nll_sum": nll,
        "nll": mean,
        "ppl": math.exp(mean) if mean <= math.log(np.finfo(np.float64).max) else None,
        "ppl_overflow": mean > math.log(np.finfo(np.float64).max),
        "target_positions": [context, context + decode - 1],
        "final_decode_logit_scored": False,
        "token_nll": [float(x) for x in losses],
    }


class Reader(Files):
    """Reopen selected bytes once, retaining hashes; do not rehash checkpoints or whole source trees."""

    def arrays(self, pin):
        path = Path(pin["path"]).resolve()
        self.pin(path, pin["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files
            if len(set(names)) != len(names):
                raise ValueError("duplicate NPZ name")
            if all(n.startswith("out") and n[3:].isdigit() for n in names):
                names = sorted(names, key=lambda n: int(n[3:]))
            return {n: np.asarray(archive[n]) for n in names}

    def pinned_json(self, pin):
        self.selected(pin)
        return self.json(pin["path"])


def _quality(kind, actual, fp, spec):
    if not actual or not fp:
        return {"available": False, "reason": "missing quality outputs"}
    if kind == "next_token_distribution":
        shapes = [v.shape for v in actual]
        aligned = (
            len(fp) == 1
            and fp[0].ndim >= 2
            and all(len(s) >= 2 and s[-1] == fp[0].shape[-1] for s in shapes)
            and sum(int(np.prod(s[:-1])) for s in shapes) == int(np.prod(fp[0].shape[:-1]))
        )
    else:
        aligned = len(actual) == len(fp) and all(a.shape == b.shape for a, b in zip(actual, fp, strict=True))
    if not aligned:
        return {"available": False, "reason": "unaligned quality output shapes/arity; no broadcasting"}
    if any(not np.isfinite(a).all() for a in list(actual) + list(fp)):
        return {"available": False, "reason": "nonfinite quality inputs"}
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            legacy = M.model_metrics(kind, actual, fp, spec)
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}

    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        return None if isinstance(value, float) and not math.isfinite(value) else value

    result = {"available": True, "legacy_metrics": finite(legacy)}
    if kind == "per_channel":
        a, b = actual[-1], fp[0]
        channels = range(int(spec["channels"]["first"]), int(spec["channels"]["last"]) + 1)
        result["channels"] = {
            str(c): array_errors(a[..., c].astype(np.float64), b[..., c].astype(np.float64)) for c in channels
        }
    else:
        a = np.concatenate([v.reshape(-1, v.shape[-1]) for v in actual])
        b = np.concatenate([v.reshape(-1, v.shape[-1]) for v in fp])
        result["errors"] = array_errors(a.astype(np.float64), b.astype(np.float64))
    return result


def report(audit_path: Path, policy_models: Path) -> dict:
    """Add diagnostics to immutable historical evidence, separately pinning the prospective policy."""
    started = time.monotonic()
    files = Reader()
    files.pin(Path(__file__).resolve())
    files.pin(Path(M.__file__).resolve())
    audit = files.json(audit_path.resolve())
    if audit.get("schema") != AUDIT_SCHEMA or not audit.get("unchanged"):
        raise ValueError("expected a retained all-output audit with unchanged artifact identities")
    job = files.pinned_json(audit["job_pin"])
    source = files.pinned_json(audit["full_report"])
    supplement = files.pinned_json(audit["reference_manifest"])
    argv = job["argv"]
    original = Path(_argument(argv, "--models"))
    files.pin(original, audit["selected_identity"]["models_config_sha256"])
    evaluated = load(original).model(audit["model"])
    files.pin(policy_models)
    policy = yaml.safe_load(policy_models.read_text())
    model = load(policy_models).model(audit["model"])
    if (
        model.chain != evaluated.chain
        or model.programs != evaluated.programs
        or {k: v for k, v in model.judged.items() if k != "reporting"}
        != {k: v for k, v in evaluated.judged.items() if k != "reporting"}
    ):
        raise ValueError("reporting policy changed evaluated model bindings, metrics or admission rules")
    if policy.get("numerics", {}).get("version") != POLICY or policy["numerics"].get("report_only") is not True:
        raise ValueError("unknown or non-report-only numerical policy")
    requested = policy["numerics"].get("reporting", {})
    if set(requested) - set(REPORTING) - {"input_manifest"} or any(
        requested.get(k, v) != v for k, v in REPORTING.items()
    ):
        raise ValueError("reporting semantics differ from supported numerical policy v1")
    if (
        source.get("model") != model.name
        or audit["submission"]["model_check_tree_sha256"] != source["submission"]["tree_sha256"]
    ):
        raise ValueError("audit and original report disagree on model or compiler identity")
    expected_steps = [(p.name, s) for p in model.programs for s in range(p.steps)]
    if [(r["program"], r["step"]) for r in audit["steps"]] != expected_steps:
        raise ValueError("section roster is partial, duplicated or reordered")
    bindings = {(r["program"], r["output"]) for r in model.judged["outputs"]}
    observed, section_rows = {}, []
    for row in audit["steps"]:
        actual = files.arrays(row["actual"])
        if row["actual"]["sha256"] == row["reference"]["sha256"]:
            files.selected(row["reference"])
            reference = actual
        else:
            reference = files.arrays(row["reference"])
        names = [f"out{i}" for i in range(row["outputs"])]
        if list(actual) != names or list(reference) != names:
            raise ValueError("section output arity/names differ from original audit")
        for i, name in enumerate(names):
            a, b = actual[name], reference[name]
            section = {
                "program": row["program"],
                "step": row["step"],
                "output": i,
                "dtype": a.dtype.str,
                "shape": list(a.shape),
                "compiled_vs_quantized": array_errors(a, b),
                "fp_quality": {"available": False, "reason": "no aligned FP tensor retained at this boundary"},
            }
            section_rows.append(section)
            if (row["program"], i) in bindings:
                observed[(row["program"], row["step"], i)] = (a, b, section)
    ordered = [
        observed[(r["program"], step, r["output"])]
        for r in model.judged["outputs"]
        for step in range(model.program(r["program"]).steps)
    ]
    judged, quantized = [row[0] for row in ordered], [row[1] for row in ordered]
    bundle = Path(_argument(argv, "--bundle"))
    files.pin(bundle / "MANIFEST.json", audit["selected_identity"]["bundle_manifest_sha256"])
    # Reuse only FP/input bytes already named by the selected seal, without reopening the entire bundle.
    sums = bundle / "SHA256SUMS"
    files.pin(sums, audit["selected_identity"]["bundle_sums_sha256"])
    sealed = {name: sha for sha, name in (line.rstrip("\n").split("  ", 1) for line in sums.read_text().splitlines())}
    relative = f"{model.name}/{audit['input_id']}/fp.npz"
    result = {
        "schema": SCHEMA,
        "report_only": True,
        "model": model.name,
        "input": audit["input_id"],
        "source_audit_status": audit["status"],
        "selected_identity": audit["selected_identity"],
        "submission": audit["submission"],
        "numerical_policy": {"version": POLICY, "report_only": True, "reporting": REPORTING},
        "recorded_float_policy": audit["float_policy"],
        "quality_rules": model.judged.get("admission", []),
        "quality_rule_scope": "per-split cohort; this one-input report is not cohort admission",
        "sections": section_rows,
        "quality": {"available": False, "reason": "no sealed FP outputs"},
        "layerwise_fp": {"available": False, "reason": "no aligned FP intermediate tape retained"},
    }
    input_arrays = files.arrays(supplement["input"])
    input_manifest = None
    if requested.get("input_manifest"):
        input_manifest = files.json((policy_models.parent / requested["input_manifest"]).resolve())
        input_name = f"{audit['input_id']}/{Path(supplement['input']['path']).name}"
        if input_manifest["models"][model.name]["derived"][input_name] != supplement["input"]["sha256"]:
            raise ValueError("input metadata belongs to another evaluated input")
        result["input_selection"] = [
            {
                k: row[k]
                for k in ("id", "split", "article", "member", "episode", "frame", "noise_seed", "label")
                if k in row
            }
            for row in input_manifest["models"][model.name].get("selection", [])
            if row.get("id") == audit["input_id"].split("/")[1]
        ]
    kind, spec = model.judged["kind"], model.judged
    fp = []
    reporting = spec.get("reporting", {})
    if reporting.get("labels"):
        result["label_accuracy"] = {"available": False, "reason": "no finite logits and manifest-bound labels"}
    if relative in sealed:
        fp = list(files.arrays({"path": str(bundle / relative), "sha256": sealed[relative]}).values())
        result["quality"] = {
            "compiled_vs_fp": _quality(kind, judged, fp, spec),
            "quantized_vs_fp": _quality(kind, quantized, fp, spec),
        }
        result["quality"]["available"] = all(row["available"] for row in result["quality"].values())
        if kind == "next_token_distribution" and result["quality"]["available"]:
            fp_rows = fp[0].reshape(-1, fp[0].shape[-1])
            for index, (_, _, section) in enumerate(ordered):
                section["compiled_vs_fp"] = _quality(kind, [judged[index]], [fp_rows[index : index + 1]], spec)
                section["quantized_vs_fp"] = _quality(kind, [quantized[index]], [fp_rows[index : index + 1]], spec)
                section["fp_quality"] = {
                    "available": section["compiled_vs_fp"]["available"] and section["quantized_vs_fp"]["available"]
                }
        elif len(judged) == 1:
            for section in section_rows:
                if (section["program"], section["output"]) in bindings:
                    section["fp_quality"] = {"available": result["quality"]["available"]}
                    section["compiled_vs_fp"] = result["quality"]["compiled_vs_fp"]
                    section["quantized_vs_fp"] = result["quality"]["quantized_vs_fp"]
        if reporting.get("labels") and result["quality"]["available"]:
            label_spec = reporting["labels"]
            split, item = audit["input_id"].split("/")
            labels_path = policy_models.parent / label_spec["path"].format(split=split)
            if labels_path.is_file() and input_manifest is not None:
                files.pin(labels_path.resolve(), input_manifest["models"][model.name][split][labels_path.name])
                labels = files.json(labels_path.resolve())
                label = int(labels[item][label_spec["field"]])
                result["label_accuracy"] = {
                    "available": True,
                    "label_namespace": label_spec["namespace"],
                    "examples": 1,
                }
                for name, values in (("compiled", judged), ("quantized", quantized), ("fp", fp)):
                    logits = np.asarray(values[-1]).reshape(-1)
                    if not 0 <= label < logits.size:
                        raise ValueError("classification label outside output vocabulary")
                    result["label_accuracy"][name] = {
                        f"top{k}": bool(label in np.argsort(-logits)[:k]) for k in spec.get("topk", [1])
                    }
        baseline_name = f"{model.name}/{audit['input_id']}/fp_baseline.npz"
        if baseline_name in sealed:
            baseline = list(
                files.arrays({"path": str(bundle / baseline_name), "sha256": sealed[baseline_name]}).values()
            )
            result["quality"]["baseline_vs_fp"] = _quality(kind, baseline, fp, spec)
    if reporting.get("perplexity"):
        selection = reporting["perplexity"]
        if (
            selection.get("exclude_final_decode_logit") is not True
            or selection.get("padding") != "none"
            or selection.get("score_tokens") != "decode_steps"
        ):
            raise ValueError("unsupported PPL score-mask policy")
        scores = {}
        for name, arrays in (("fp", fp), ("quantized", quantized), ("compiled", judged)):
            if not arrays or "tokens" not in input_arrays:
                scores[name] = {"available": False, "reason": "no retained logits or target tokens"}
            else:
                logits = np.concatenate([v.reshape(-1, v.shape[-1]) for v in arrays])
                scores[name] = perplexity(
                    logits, input_arrays["tokens"], int(model.chain["prefill_tokens"]), int(model.chain["decode_steps"])
                )
        if all(v["available"] for v in scores.values()):
            for name in ("quantized", "compiled"):
                delta = scores[name]["nll"] - scores["fp"]["nll"]
                scores[name]["delta_nll_vs_fp"] = delta
                scores[name]["ppl_ratio_vs_fp"] = math.exp(delta) if delta < 709 else None
        result["perplexity"] = scores
    result["kernel_groups"] = [
        {
            "program": program["program"],
            **{
                k: row[k]
                for k in (
                    "symbol",
                    "status",
                    "cases",
                    "exact",
                    "mismatches",
                    "source_region_id",
                    "source_region_ids",
                    "interface_sha256",
                    "shape_sha256",
                    "form",
                )
                if k in row
            },
        }
        for program in source["programs"]
        for row in program.get("kernels", [])
    ]
    sites = []
    for program in audit["programs"]:
        meta = Path(program["capture_program"]["path"]).with_name("meta.json")
        if meta.is_file():
            stats = (files.json(meta).get("quantization_stats") or {}).get("dynamic_contractions") or {}
            for site in stats.get("sites", []):
                sites.append(
                    {
                        "program": program["program"],
                        "comparison": "isolated_capture_attention_not_propagated_FP_error",
                        **{k: site[k] for k in ("owner", "node", "lineage", "verification") if k in site},
                    }
                )
    result["capture_attention_sites"] = sites
    files.unchanged()
    result["artifacts"] = [{"path": str(p), "sha256": sha} for p, sha in files.pins.items()]
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="existing all-declared-output audit JSON")
    parser.add_argument(
        "--models", type=Path, required=True, help="reporting models config; historical sealed config stays unchanged"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(args.audit, args.models.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['model']}: {len(result['sections'])} sections, {result['elapsed_s']:.3f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
