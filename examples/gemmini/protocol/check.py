"""Check the shared Gemmini definitions and optionally export a resolved contract."""

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
    calibration = yaml.safe_load((root / spec["authority"]["capsule_calibration_requirements"]).read_bytes())
    same(calibration["schema"], "merlin.gemmini.capsule_calibration_requirements.v1", "capsule calibration schema")
    same(calibration["scope"], "external_public_measurement_requirements", "capsule calibration scope")
    capsule_recipe = yaml.safe_load((root / calibration["source_recipe"]).read_bytes())
    same(
        capsule_recipe.get("column_readout_profile"),
        calibration["coverage"]["activated_column_readout"]["source_profile"],
        "column readout source profile",
    )
    numerical_path = root / spec["authority"]["numerical_contract"]
    numerical_bytes = numerical_path.read_bytes()
    models_config = yaml.safe_load(numerical_bytes)
    from merlin.model_check.models import load, selected_float_policy

    selected_float_policy(load(numerical_path, source_bytes=numerical_bytes))
    inputs = json.loads((root / spec["authority"]["evaluation_input_manifest"]).read_text())
    launch = yaml.safe_load((root / spec["authority"]["phase1_launch"]).read_text())
    manifest = yaml.safe_load((root / "run-manifest.template.yaml").read_text())
    same(set(spec["models"]), set(models_config["models"]), "model roster")
    total_steps = total_arrays = 0
    for name, model in spec["models"].items():
        source_model = models_config["models"][name]
        same(model["chain"], source_model["chain"], f"{name} chain")
        same(model["quality_policy"], f"../numerics/models.yaml#models.{name}.judged", f"{name} quality policy")
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
    comparison = models_config["numerics"]["functional"]
    selected = launch["full_model_verification"]
    same(selected["comparison"], "quantized-reference", "selected launch comparison")
    same(selected["output_scope"], "all_declared", "selected launch output scope")
    for field in ("atol", "rtol"):
        same(comparison[field], selected[field], f"selected launch {field}")
    same(spec["profiles"]["functional_v1"]["fp_quality"], "report_only", "functional FP quality scope")
    same(manifest["protocol"]["id"], spec["id"], "manifest protocol identity")
    same(models_config["numerics"]["report_only"], True, "offline diagnostics scope")
    ppl = models_config["models"]["tiny_llama"]["judged"]["reporting"]["perplexity"]
    same(ppl["exclude_final_decode_logit"], True, "PPL target shift")
    same(ppl["score_tokens"], "decode_steps", "PPL token denominator")
    same(ppl["threshold"], "report_only", "PPL gate scope")
    return errors


def resolved_contract(spec_path: Path) -> dict:
    """Snapshot the checked source definitions; this supplies no execution evidence."""
    errors = check(spec_path)
    if errors:
        raise ValueError("; ".join(errors))
    root = spec_path.parent
    spec = yaml.safe_load(spec_path.read_bytes())
    numerical_path = root / spec["authority"]["numerical_contract"]
    numerical = yaml.safe_load(numerical_path.read_bytes())
    calibration_name = spec["authority"]["capsule_calibration_requirements"]
    calibration = yaml.safe_load((root / calibration_name).read_bytes())
    names = {spec_path.name, "run-manifest.template.yaml", spec["baseline_comparison"]["manifest_template"]}
    names.add(calibration_name)
    names.add(calibration["source_recipe"])
    names.update(spec["authority"][name] for name in (
        "catalog", "phase1_launch", "source_workloads", "numerical_contract",
        "historical_input_sources", "evaluation_input_manifest",
    ))
    names.update(spec["hardware"][name] for name in ("bindings", "board_catalog"))
    recipes = {}
    for name, model in spec["models"].items():
        recipe = model["quantization"]["recipe"]
        names.add(recipe)
        recipes[name] = json.loads((root / recipe).read_bytes())
    return {
        "schema": "merlin.gemmini.resolved_protocol.v1",
        "protocol": spec,
        "numerical_policy": numerical,
        "capsule_calibration_requirements": calibration,
        "recipes": recipes,
        "files": {name: {
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            "bytes": (root / name).stat().st_size,
        } for name in sorted(names)},
        "scope": "Generated snapshot of the shared definitions, not an independently edited policy.",
        "execution_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("experiment.yaml"))
    parser.add_argument("--export", type=Path, help="write resolved JSON outside examples/gemmini")
    args = parser.parse_args()
    try:
        value = resolved_contract(args.spec.resolve()) if args.export else None
        errors = [] if args.export else check(args.spec.resolve())
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors = [f"invalid definition: {exc}"]
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    if args.export:
        output = args.export.resolve()
        if output.is_relative_to(args.spec.resolve().parent.parent):
            print("FAIL: generated snapshots must stay outside the authored Gemmini example")
            return 1
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"FAIL: cannot export definition: {exc}")
            return 1
        print(f"Resolved contract: {output}")
    print("PASS: definition consistency (3 recipes, 46 invocations, 133 output arrays).")
    print("No capture, numerical evaluation, model execution or certification was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
