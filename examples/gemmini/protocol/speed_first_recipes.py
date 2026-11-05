"""Generate separate speed-first recipes over an explicitly selected contract.

This prepares numerical selections only. Calibration, canonical capture replay,
continuous-chain references and execution remain separate owned operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe_policy as P


def generate(directory: Path, *, spec: Path, contract: Path) -> dict:
    method = yaml.safe_load(spec.read_bytes())["quantization"]["methodology"]
    output_path = M.output_path_of(json.loads(contract.read_bytes()))
    if output_path is None:
        raise ValueError("selected contract has no derived output path")
    records = {}
    for model in ("resnet50", "tiny_llama", "smolvla"):
        path = directory / f"{model}-native-speed.json"
        raw = path.read_bytes()
        source = json.loads(raw)
        policy = {
            "schema": P.SCHEMA,
            "weight_granularity": "tensor",
            "float_quality": "report_only",
            "native_outputs": source["epilogue"]["output_quantization"],
            "dynamic_activation_planes": 1 if "dynamic" in source else None,
            "integer_extensions": {
                key: source[key]
                for key in ("accumulator_residual", "integer_mean", "projection_input_reuse")
                if key in source
            },
        }
        policy["native_outputs"] = {k: v for k, v in policy["native_outputs"].items() if k != "schema"}
        candidate = P.select(source, method, output_path, policy=policy, model=model)
        destination = directory / f"{model}-speed-first.json"
        payload = (json.dumps(candidate, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite a different recipe: {destination.name}")
        destination.write_bytes(payload)
        records[model] = {
            "source": path.name,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate": destination.name,
            "candidate_sha256": hashlib.sha256(payload).hexdigest(),
            "recipe_sha256": candidate["recipe_sha256"],
        }
    return records


def main():
    root = repo_root() / "examples/gemmini"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=root / "target/software-spec.yaml")
    parser.add_argument("--directory", type=Path, default=root / "protocol/recipes")
    args = parser.parse_args()
    print(json.dumps(generate(args.directory, spec=args.spec, contract=args.contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
