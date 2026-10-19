"""Read-only consistency check of the public Gemmini experiment definition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen.quant_recipe import digest


def check(spec_path: Path) -> list[str]:
    """Return definition errors; this does not inspect or certify execution."""
    spec = yaml.safe_load(spec_path.read_text())
    root = spec_path.parent
    errors: list[str] = []

    def same(actual: object, expected: object, label: str) -> None:
        if actual != expected:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}")

    same(spec["schema"], "merlin.gemmini.experiment_protocol.v1", "schema")
    models_config = yaml.safe_load((root / spec["authority"]["quality_rules"]).read_text())
    inputs = json.loads((root / spec["authority"]["evaluation_input_manifest"]).read_text())
    launch = yaml.safe_load((root / spec["authority"]["phase1_launch"]).read_text())
    manifest = yaml.safe_load((root / "run-manifest.template.yaml").read_text())
    same(set(spec["models"]), set(models_config["models"]), "model roster")
    total_steps = total_arrays = 0
    for name, model in spec["models"].items():
        source_model = models_config["models"][name]
        same(model["chain"], source_model["chain"], f"{name} chain")
        same(model["quality"]["kind"], source_model["judged"]["kind"], f"{name} metric kind")
        same(model["quality"]["admission"], source_model["judged"]["admission"], f"{name} quality rules")
        same(model["architecture"]["batch"], spec["profiles"]["functional_v1"]["batch"], f"{name} batch")
        evaluation = model["external_evaluation_input"]
        same(
            evaluation["sha256"],
            inputs["models"][name]["derived"][evaluation["derived_member"]],
            f"{name} external input manifest pin",
        )
        stages = model["stages"]
        programs = {p["name"]: p for p in source_model["programs"]}
        same(set(stages), set(programs), f"{name} stages")
        for stage_name, stage in stages.items():
            program = programs[stage_name]
            same(stage["steps"], program.get("steps", 1), f"{name}/{stage_name} invocations")
            same(len(stage["inputs"]), len(program["inputs"]), f"{name}/{stage_name} input arity")
            for tensor in stage["inputs"] + stage["outputs"]:
                if not tensor["shape"] or any(type(n) is not int or n <= 0 for n in tensor["shape"]):
                    errors.append(f"{name}/{stage_name}: invalid tensor shape {tensor}")
                if tensor["dtype"] not in {"f32", "i1", "i64"}:
                    errors.append(f"{name}/{stage_name}: unsupported declared dtype {tensor['dtype']}")
        steps = sum(s["steps"] for s in stages.values())
        arrays = sum(s["steps"] * len(s["outputs"]) for s in stages.values())
        same(steps, model["selected_chain_steps"], f"{name} chain steps")
        same(arrays, model["declared_arrays_per_chain"], f"{name} output count")
        total_steps += steps
        total_arrays += arrays

        quant = model["quantization"]
        recipe_bytes = (root / quant["recipe"]).read_bytes()
        recipe = json.loads(recipe_bytes)
        same(hashlib.sha256(recipe_bytes).hexdigest(), quant["file_sha256"], f"{name} recipe file hash")
        same(digest(recipe), quant["recipe_sha256"], f"{name} recipe content hash")
        same(recipe["recipe_sha256"], quant["recipe_sha256"], f"{name} embedded recipe identity")
        same(recipe["schema"], "quant_recipe_v1", f"{name} recipe schema")
        same(recipe["target"], "gemmini", f"{name} target")
        same(recipe["unit"], "systolic_mesh", f"{name} unit")
        for operand in ("weight", "activation"):
            fields = spec["quantization"][operand]
            actual = recipe[operand]
            for field in ("dtype", "mode", "observer"):
                same(actual[field], fields[field], f"{name}/{operand}/{field}")
            same([actual["quant_min"], actual["quant_max"]], fields["range"], f"{name}/{operand}/range")
            same(actual["symmetric"], True, f"{name}/{operand}/symmetric")
        same(
            recipe["weight"]["channels_per_scale"],
            spec["quantization"]["weight"]["channels_per_scale"],
            f"{name} scale grouping",
        )
        same(recipe["activation"]["granularity"], "tensor", f"{name} activation granularity")
        same(recipe["accumulator_dtype"], spec["quantization"]["accumulator"], f"{name} accumulator")
        same(recipe["epilogue"]["schema"], spec["quantization"]["epilogue"], f"{name} epilogue")
        same(
            recipe["framework_capture_policy"]["observer_epsilon"],
            spec["quantization"]["observer_epsilon"],
            f"{name} observer epsilon",
        )
        template = manifest["models"][name]
        same(template["recipe_file_sha256"], quant["file_sha256"], f"{name} manifest file pin")
        same(template["recipe_sha256"], quant["recipe_sha256"], f"{name} manifest recipe pin")
        same(template["host_math_policy"], model["host_math_policy"], f"{name} manifest math policy")

    same((total_steps, total_arrays), (46, 133), "selected chain totals")
    comparison = spec["functional_comparison"]
    selected = launch["full_model_verification"]
    same(selected["comparison"], "quantized-reference", "selected launch comparison")
    same(selected["output_scope"], "all_declared", "selected launch output scope")
    for field in ("atol", "rtol"):
        same(comparison["floating"][field], selected[field], f"selected launch {field}")
    same(spec["profiles"]["functional_v1"]["fp_quality"], "report_only", "functional FP quality scope")
    same(manifest["protocol"]["id"], spec["id"], "manifest protocol identity")
    ppl = spec["tiny_llama_ppl_v1"]
    same(ppl["implementation"], "missing", "PPL implementation status")
    same(ppl["threshold"], "not_selected", "PPL threshold status")
    same(ppl["phase1_gate"], False, "PPL gate scope")
    tiny_chain = spec["models"]["tiny_llama"]["chain"]
    same(ppl["context_tokens"], tiny_chain["prefill_tokens"], "PPL context matches selected chain")
    same(ppl["continuation_tokens"], tiny_chain["decode_steps"], "PPL continuation matches selected chain")
    same(ppl["teacher_forcing"], True, "PPL teacher forcing")
    same(ppl["reset_KV"], "each_segment", "PPL state reset")
    same(ppl["chunk_tokens"], ppl["context_tokens"] + ppl["continuation_tokens"], "PPL input length")
    same(ppl["scored_target_positions_inclusive"], [32, 63], "PPL target shift")
    same(ppl["scored_logit_positions_inclusive"], [31, 62], "PPL score mask")
    same(ppl["scored_tokens_per_unpadded_segment"], 32, "PPL token denominator")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("experiment.yaml"))
    args = parser.parse_args()
    try:
        errors = check(args.spec.resolve())
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors = [f"invalid definition: {exc}"]
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: definition consistency (3 recipes, 46 invocations, 133 output arrays).")
    print("No capture, numerical evaluation, model execution or certification was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
