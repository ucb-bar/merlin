"""Eight opt-in floating source cases, using the ordinary capsule builder/writer.

The CLI materializes diagnostic inputs only. It does not admit a Phase 1 corpus,
select hardware facts or launch an authoring agent. Normal campaign generation
can consume the same entries subject to its existing independent admission gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

from merlin.targetgen.component_float import POLICY
from merlin.targetgen.corpus_spec import CorpusBinding, dtype_info

SCHEMA = "merlin.floating_test_suite.v1"
FAMILIES = (
    "ordered_contraction",
    "contraction_relu",
    "residual_fanout",
    "independent_contractions",
    "shared_weight",
    "dependent_contractions",
    "alias_update",
    "rounding_movement",
)


def entries(config):
    """Expand one explicit source policy to eight public cases, never hardware claims."""
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "schema",
            "operand_dtype",
            "accumulator_dtype",
            "shape",
            "pointwise_compute_dtype",
            "pointwise_rounding",
            "reference_budget",
            "required_oracle_tiers",
        }
        or config["schema"] != SCHEMA
    ):
        raise ValueError("floating suite requires its closed v1 configuration")
    shape = config["shape"]
    if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("floating suite shape must be explicit positive M, K, N extents")
    m, k, n = shape
    tiers = config["required_oracle_tiers"]
    if (
        not isinstance(tiers, list)
        or not tiers
        or any(not isinstance(t, str) for t in tiers)
        or len(tiers) != len(set(tiers))
    ):
        raise ValueError("floating suite requires explicit unique oracle tier obligations")
    if k < 4:
        raise ValueError("floating suite needs at least four reduction terms for a nonzero cancellation witness")
    operand, accumulator = (dtype_info(config[key])[1] for key in ("operand_dtype", "accumulator_dtype"))

    def leaf(name, shape, dtype="operand", role="input"):
        return {"name": name, "role": role, "shape": list(shape), "dtype": dtype}

    def node(name, op, *inputs):
        row = {"name": name, "op": op, "inputs": list(inputs)}
        if op == "matmul":
            row.update(compute_dtype=accumulator, rounding="rne")
        elif op in {"add", "update"}:
            row.update(compute_dtype=config["pointwise_compute_dtype"], rounding=config["pointwise_rounding"])
        elif op == "convert":
            row.update(dtype=operand, rounding="rne")
        return row

    def matrix_inputs():
        return [leaf("A", (m, k)), leaf("W", (k, n), role="weight")]

    cases = []
    for family in FAMILIES:
        inputs, nodes, outputs = matrix_inputs(), [node("P", "matmul", "A", "W")], ["P"]
        if family == "contraction_relu":
            nodes += [node("R", "relu", "P")]
            outputs = ["R"]
        elif family == "residual_fanout":
            inputs += [leaf("B", (m, n), "accumulator", "bias"), leaf("C", (m, n), "accumulator")]
            nodes += [node("R", "add", "P", "B"), node("S", "add", "P", "C")]
            outputs = ["P", "R", "S"]
        elif family in {"independent_contractions", "shared_weight"}:
            inputs += [leaf("X", (m, k))]
            weight = "W"
            if family == "independent_contractions":
                inputs += [leaf("V", (k, n), role="weight")]
                weight = "V"
            nodes += [node("Q", "matmul", "X", weight)]
            outputs = ["P", "Q"]
        elif family == "dependent_contractions":
            inputs += [leaf("V", (n, k), role="weight")]
            nodes += [node("Narrow", "convert", "P"), node("Q", "matmul", "Narrow", "V")]
            outputs = ["P", "Narrow", "Q"]
        elif family == "alias_update":
            inputs = [leaf("A", (m, n), "accumulator"), leaf("B", (m, n), "accumulator")]
            nodes = [node("View", "alias", "A"), node("Before", "copy", "View"), node("After", "update", "A", "B")]
            outputs = ["Before", "View", "After"]
        elif family == "rounding_movement":
            inputs = [leaf("A", (m, n), "accumulator"), leaf("B", (m, n), "accumulator")]
            nodes = [
                node("Sum", "add", "A", "B"),
                node("Narrow", "convert", "A"),
                {**node("Wide", "convert", "Narrow"), "dtype": accumulator},
                node("T", "transpose", "Sum"),
                node("Roundtrip", "transpose", "T"),
            ]
            outputs = ["Sum", "Narrow", "Wide", "T", "Roundtrip"]
        palette = []
        for index, row in enumerate(inputs):
            values = [1.0, -0.5, 0.25, -1.0, 0.5, 0.0, -0.25]
            axis = "linear"
            if family == "ordered_contraction":
                from merlin.runtime.fp8_formats import float_format_params, normal_range

                mantissa = float_format_params(accumulator)[1]
                # Keep both factors representable in operand storage. When its
                # exponent range cannot hold the original large factor, scale
                # both factors down equally, preserving the cancellation ratio.
                separation = (mantissa + 2) // 2
                large_exponent = min(separation, math.frexp(normal_range(operand)[1])[1] - 1)
                large = 2.0 ** large_exponent
                small = 2.0 ** (large_exponent - separation)
                values = (
                    [large, small, -large, small, small]
                    if row["name"] == "A"
                    else [large, small, large, small, small]
                )
                axis = 1 if row["name"] == "A" else 0
            if family == "rounding_movement":
                # Derive half-ULP witnesses from the selected formats, not a target name.
                from merlin.runtime.fp8_formats import float_format_params

                operand_mantissa = float_format_params(operand)[1]
                acc_mantissa = float_format_params(accumulator)[1]
                # A midpoint is representable only when the source has more
                # precision. Matching formats exercise exact conversion instead.
                increment = 2.0 ** -min(operand_mantissa + 1, acc_mantissa)
                values = (
                    [1.0, -1.0, 1 + increment, -(1 + increment)]
                    if row["name"] == "A"
                    else [3 * 2.0 ** (-acc_mantissa - 2)]
                )
            # Literal shapes may be smaller than a full stimulus pattern. Only
            # declare values that this input can visit; the four-term ordered
            # prefix still contains cancellation and a nonzero final addend.
            extent = math.prod(row["shape"]) if axis == "linear" else row["shape"][axis]
            values = values[:extent]
            palette.append(
                {
                    "name": row["name"],
                    "axis": axis,
                    "values": values,
                    "offset": 0 if family == "ordered_contraction" else index % len(values),
                }
            )
        cases.append(
            {
                "cat": "floating_components",
                "name": "FP_" + family,
                "kind": "model_slice",
                "op": "component_program",
                "label": "public",
                "source_role": "handauthored_compiler_test",
                "source_reference": family,
                "operand_dtype": config["operand_dtype"],
                "accum_dtype": config["accumulator_dtype"],
                "program": {
                    "floating_point": copy.deepcopy(POLICY),
                    "inputs": inputs,
                    "nodes": nodes,
                    "outputs": [{"name": "out" + str(i), "value": value} for i, value in enumerate(outputs)],
                },
                "input_palette": {"schema": "merlin.component_input_palette.v1", "inputs": palette},
                "reference_budget": copy.deepcopy(config["reference_budget"]),
            }
        )
    return cases


def materialize(config, output):
    """Write diagnostic sources and goldens; never promote them to an admitted corpus."""
    from merlin.targetgen.contract.schemas import validate_capsule

    from .component_float_numerics import preflight
    from .writer import _write_capsule_inner

    output = Path(output)
    if output.exists():
        raise ValueError("floating source diagnostic requires a fresh output directory")
    selected = entries(config)
    binding = CorpusBinding(
        target="source_only",
        tile_dim=None,
        operand_dtype=config["operand_dtype"],
        accum_dtype=config["accumulator_dtype"],
        integer=False,
        tiers=list(config["required_oracle_tiers"]),
        compare="tolerance_float",
        atol=0.0,
        rtol=0.0,
    )
    # Refuse malformed or oversized sources before writing any member.
    for entry in selected:
        preflight(entry, binding=binding)
    output.mkdir(parents=True)
    members = []
    for entry in selected:
        directory = _write_capsule_inner(entry, binding, output)
        validate_capsule(yaml.safe_load((directory / "capsule.yaml").read_bytes()))
        members.append(
            {
                "path": str(directory.relative_to(output)),
                "sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(directory.iterdir())
                    if p.is_file()
                },
            }
        )
    manifest = {
        "schema": "merlin.floating_test_sources.v1",
        "status": "source_only",
        "agent_facing_cases": len(members),
        "hardware_qualified": False,
        "phase1_admitted": False,
        "config": config,
        "members": members,
        "pending": [
            "target_operation_and_numeric_admission",
            "independent_runtime_qualification",
            "physical_scheduling_witnesses",
            "frozen_corpus_release",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "entries.yaml").write_text(yaml.safe_dump({"capsules": selected}, sort_keys=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = materialize(yaml.safe_load(Path(args.config).read_bytes()), args.out)
    print(json.dumps({key: manifest[key] for key in ("status", "agent_facing_cases", "phase1_admitted")}))


if __name__ == "__main__":
    main()
