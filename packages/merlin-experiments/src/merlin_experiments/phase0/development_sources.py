"""Fresh connected public development sources for the ordinary capture route.

These are complete source programs, not model surrogates or admitted accelerator
capsules. Explicit public geometry and arithmetic never come from a validation
capture. The normal capture, exclusion, software screen and grading owners apply.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import prod
from pathlib import Path

from . import workload_roster

SCHEMA = "merlin.phase0.public_development_source.v1"
PATTERN = "grouped_head_integer_mix"
CONTRACT = "signed_i8_i32_round_half_up_saturate_i8"
_AXES = {"steps", "prefix_tokens", "query_tokens", "kv_heads", "query_heads", "head_dim"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def describe(declaration):
    """Derive complete input/result geometry before allocating tensors or files."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "scope", "name", "pattern", "geometry", "arithmetic", "max_logical_elements"}
        or declaration["schema"] != SCHEMA
        or declaration["scope"] != "public_development"
        or declaration["pattern"] != PATTERN
        or not workload_roster._label(declaration["name"])
    ):
        raise ValueError("development source requires an explicit public declaration and supported pattern")
    geometry, arithmetic = declaration["geometry"], declaration["arithmetic"]
    if (
        not isinstance(geometry, dict)
        or set(geometry) != _AXES
        or any(type(value) is not int or not 0 < value <= 65536 for value in geometry.values())
        or geometry["steps"] > 4
        or geometry["query_heads"] > 32
        or geometry["query_heads"] % geometry["kv_heads"]
        or not isinstance(arithmetic, dict)
        or set(arithmetic) != {"contract", "readout_shift"}
        or arithmetic["contract"] != CONTRACT
        or type(arithmetic["readout_shift"]) is not int
        or not 1 <= arithmetic["readout_shift"] <= 30
        or type(declaration["max_logical_elements"]) is not int
        or not 1 <= declaration["max_logical_elements"] <= 2**30
    ):
        raise ValueError("development geometry/arithmetic requires finite explicit supported bounds")
    steps, prefix, tokens, kv, heads, width = (
        geometry[key] for key in ("steps", "prefix_tokens", "query_tokens", "kv_heads", "query_heads", "head_dim")
    )
    # All possible signed-i8 operands, not just the fresh stimulus, must fit i32
    # including the declared readout rounding offset.
    longest = prefix + steps * tokens
    if max(width, longest) * 128**2 + 2 ** (arithmetic["readout_shift"] - 1) > 2**31 - 1:
        raise ValueError("development integer reduction could overflow its exact accumulator contract")
    inputs = [
        {"name": "prefix_k", "shape": [kv, prefix, width], "dtype": "int8"},
        {"name": "prefix_v", "shape": [kv, prefix, width], "dtype": "int8"},
    ]
    outputs = []
    work = 0
    for step in range(steps):
        inputs += [
            {"name": f"q{step}", "shape": [heads, tokens, width], "dtype": "int8"},
            {"name": f"k{step}", "shape": [kv, tokens, width], "dtype": "int8"},
            {"name": f"v{step}", "shape": [kv, tokens, width], "dtype": "int8"},
        ]
        length = prefix + (step + 1) * tokens
        for head in range(heads):
            outputs += [
                {"name": f"scores_{step}_{head}", "shape": [tokens, length], "dtype": "int32"},
                {"name": f"mixed_{step}_{head}", "shape": [tokens, width], "dtype": "int32"},
            ]
        outputs += [
            {"name": f"cache_k{step}", "shape": [kv, length, width], "dtype": "int8"},
            {"name": f"cache_v{step}", "shape": [kv, length, width], "dtype": "int8"},
        ]
        work += 2 * heads * tokens * length * width
    elements = sum(prod(row["shape"]) for row in inputs + outputs)
    if elements > declaration["max_logical_elements"]:
        raise ValueError("complete source input/output roster exceeds its declared logical budget")
    return {
        "declaration": copy.deepcopy(declaration),
        "declaration_sha256": _digest(declaration),
        "inputs": inputs,
        "outputs": outputs,
        "logical_elements": elements,
        "scalar_products": work,
        "contractions": steps * heads * 2,
        "status": "public_source_only_not_model_or_accelerator_qualification",
        "missing": ["softmax", "projection", "normalization", "target_execution", "timing"],
    }


def render(declaration):
    """Unroll grouped contractions, exact readout and carried KV; retain live-outs."""
    description = describe(declaration)
    geometry = declaration["geometry"]
    shift = declaration["arithmetic"]["readout_shift"]
    names = [row["name"] for row in description["inputs"]]
    outputs = [row["name"] for row in description["outputs"]]
    lines = [
        '"""Fresh public integer grouped-head mixing; no softmax or model claim."""',
        "import torch",
        "",
        f"# Public declaration: {description['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def forward(self, " + ", ".join(names) + "):",
        "        cache_k = prefix_k",
        "        cache_v = prefix_v",
    ]
    for step in range(geometry["steps"]):
        lines += [
            f"        cache_k{step} = torch.cat((cache_k, k{step}), dim=1)",
            f"        cache_v{step} = torch.cat((cache_v, v{step}), dim=1)",
        ]
        for head in range(geometry["query_heads"]):
            group = head // (geometry["query_heads"] // geometry["kv_heads"])
            suffix = f"{step}_{head}"
            lines += [
                f"        scores_{suffix} = torch._int_mm(q{step}[{head}].contiguous(), "
                f"cache_k{step}[{group}].transpose(0, 1).contiguous())",
                f"        readout_{suffix} = torch.div(scores_{suffix} + {2 ** (shift - 1)}, "
                f"{2**shift}, rounding_mode='floor').clamp(-128, 127).to(torch.int8)",
                f"        mixed_{suffix} = torch._int_mm(readout_{suffix}, cache_v{step}[{group}].contiguous())",
            ]
        lines += [f"        cache_k = cache_k{step}", f"        cache_v = cache_v{step}"]
    lines += ["        return (" + ", ".join(outputs) + ",)", "", "def get_model_and_inputs():"]
    for ordinal, row in enumerate(description["inputs"]):
        lines.append(
            f"    {row['name']} = (((torch.arange({prod(row['shape'])}, dtype=torch.int64) * "
            f"{1 + (2 * ordinal) % 16} + {ordinal}) % 17) - 8).to(torch.int8).reshape({row['shape']!r})"
        )
    lines += ["    return Program().eval(), (" + ", ".join(names) + ",)"]
    return "\n".join(lines) + "\n"


def capsule_entry(declaration, loader):
    """Declare the exact source arithmetic and every live-out for the normal writer.

    This recipe entry supplies no accelerator demand or performance objective.
    The independent selected software contract must admit the actual captured
    program before it can become a performance member.
    """
    description = describe(declaration)
    return {
        "name": declaration["name"],
        "cat": "model",
        "kind": "model",
        "label": "dev",
        "loader": str(Path(loader).absolute()),
        "capture_quantization": "already_materialized",
        "output_numeric_policy": "captured_per_output_v1",
        "outs": [row["name"] for row in description["outputs"]],
        "source_reference": f"{SCHEMA}:{description['declaration_sha256']}",
    }


def stage(declarations, output_root):
    """Materialize source closures and use the normal Phase 0 roster stage owner.

    Capture selections must still be issued independently. Nothing calls a
    simulator, selects quantization, or publishes an admitted performance score.
    """
    if not isinstance(declarations, list) or not 1 <= len(declarations) <= 32:
        raise ValueError("select one to 32 explicit public development sources")
    descriptions = [describe(declaration) for declaration in declarations]
    if len({row["declaration"]["name"] for row in descriptions}) != len(descriptions):
        raise ValueError("development source names must be unique")
    root = Path(output_root).absolute()
    if root.exists() or not root.parent.is_dir():
        raise ValueError("development staging requires a fresh output with an existing parent")
    root.mkdir(mode=0o700)
    roster = {"schema": workload_roster.SCHEMA, "performance_applications": {}}
    for row in descriptions:
        name = row["declaration"]["name"]
        source = root / "sources" / name
        source.mkdir(parents=True)
        raw = render(row["declaration"]).encode()
        (source / "loader.py").write_bytes(raw)
        row["loader_sha256"] = hashlib.sha256(raw).hexdigest()
        (source / "declaration.json").write_text(json.dumps(row, sort_keys=True, indent=2) + "\n")
        roster["performance_applications"][name] = {"loader": f"sources/{name}/loader.py"}
    roster_path = root / "workload-roster.json"
    roster_path.write_text(json.dumps(roster, sort_keys=True, indent=2) + "\n")
    staged = workload_roster.stage(roster_path, root / "staged")
    result = {"schema": SCHEMA, "members": descriptions, "staging": staged, "phase0_admission": "not_granted"}
    (root / "development-sources.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
