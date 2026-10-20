"""Regenerate opt-in numerical recipes; never compiles or runs a workload.

Run from the repository: python examples/gemmini/protocol/native_speed_recipes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen import quant_recipe as Q


def generate(directory: Path) -> dict:
    records = {}
    for model in ("resnet50", "tiny_llama", "smolvla"):
        source = directory / f"{model}.json"
        baseline = json.loads(source.read_bytes())
        candidate = Q.with_native_output_quantization(
            baseline, operators=("linear", "conv2d") if model == "resnet50" else ("linear",),
            reader_policy="compute" if model == "resnet50" else "non_add_compute",
        )
        if model == "resnet50":
            policies = {
                "accumulator_residual": {
                    "schema": "integer_accumulator_residual_v1", "coefficient_rounding": "align_weight_grid",
                    "max_terms": 1, "max_weight_elements": 4194304, "max_mac_ratio": 2.0,
                },
                "integer_mean": {
                    "schema": "integer_mean_contraction_v1", "max_window_elements": 4096,
                    "max_weight_elements": 4096, "max_macs": 1048576,
                },
            }
        else:
            policies = {"projection_input_reuse": {"schema": "shared_input_projection_v1"}}
        for key, policy in policies.items():
            candidate = Q.with_integer_extension(candidate, key=key, policy=policy)
        destination = directory / f"{model}-native-speed.json"
        raw = (json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        destination.write_bytes(raw)
        records[model] = {
            "baseline": source.name, "baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "candidate": destination.name, "candidate_sha256": hashlib.sha256(raw).hexdigest(),
            "recipe_sha256": candidate["recipe_sha256"],
        }
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", type=Path,
                        default=repo_root() / "examples/gemmini/protocol/recipes")
    args = parser.parse_args()
    print(json.dumps(generate(args.directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
