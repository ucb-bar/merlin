"""Derive additional scalar-weight native recipes from an explicitly selected contract.

This selects the existing per-tensor MinMax weight observer. It neither changes
the per-command physical scale width nor runs calibration or a compiler.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q


def scalar_recipe(source: dict, methodology: dict, output_path: dict, *, model: str) -> dict:
    """Re-derive the same native profile with the existing explicit tensor-scale choice."""
    if source.get("recipe_sha256") != Q.digest(source):
        raise ValueError("source recipe content identity differs")
    original = M.validate(methodology)
    provenance = source.get("methodology") or {}
    if provenance.get("methodology_sha256") != M.digest({k: v for k, v in original.items() if k != "model_variants"}):
        raise ValueError("source recipe methodology differs")
    if provenance.get("output_path_sha256") != M.digest(output_path):
        raise ValueError("selected output path differs from source recipe")
    selected = copy.deepcopy(original)
    selected["scale_granularity"] = "per_tensor"
    # A model variant is more specific than the shared method. Preserve all
    # other selections while applying the same explicit scale choice there.
    for row in selected.get("model_variants", []):
        if row["model"] == model:
            row["changes"].pop("scale_granularity", None)
    result = M.derive(source, selected, output_path, model=model)
    if result.get("status") != "derived":
        raise ValueError(f"scalar-weight recipe is underivable: {result.get('underivable')}")
    output = source["epilogue"]["output_quantization"]
    result = Q.with_native_output_quantization(
        result,
        operators=output["operators"],
        reader_policy=output["reader_policy"],
    )
    result = Q.with_integer_extension(
        result,
        key="projection_input_reuse",
        policy={"schema": "shared_input_projection_v1", "max_nonuniform_readout_blocks": 64},
    )
    if result["weight"]["granularity"] != "tensor" or "channels_per_scale" in result["weight"]:
        raise ValueError("existing derivation did not select one weight scale per tensor")
    # Refuse an unexpected ancillary arithmetic change rather than hide it in
    # a speed variant. Calibration and all independent dynamic policies stay.
    allowed = {"weight", "methodology", "recipe_sha256", "why", "projection_input_reuse"}
    if any(result.get(k) != source.get(k) for k in result.keys() | source.keys() if k not in allowed):
        raise ValueError("scalar-weight derivation changed another arithmetic policy")
    return result


def generate(directory: Path, *, spec: Path, contract: Path) -> dict:
    method = yaml.safe_load(spec.read_bytes())["quantization"]["methodology"]
    path = M.output_path_of(json.loads(contract.read_bytes()))
    if path is None:
        raise ValueError("selected contract has no derived output path")
    records = {}
    for model in ("tiny_llama", "smolvla"):
        source = directory / f"{model}-native-speed.json"
        baseline = source.read_bytes()
        candidate = scalar_recipe(json.loads(baseline), method, path, model=model)
        destination = directory / f"{model}-native-scalar-speed.json"
        raw = (json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        destination.write_bytes(raw)
        records[model] = {
            "baseline": source.name,
            "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
            "candidate": destination.name,
            "candidate_sha256": hashlib.sha256(raw).hexdigest(),
            "recipe_sha256": candidate["recipe_sha256"],
        }
    return records


def main():
    root = repo_root() / "examples/gemmini"
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=root / "target/software-spec.yaml")
    parser.add_argument("--directory", type=Path, default=root / "protocol/recipes")
    args = parser.parse_args()
    print(json.dumps(generate(args.directory, spec=args.spec, contract=args.contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
