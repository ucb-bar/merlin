"""Stage supplemental public Phase 2 sources through the ordinary workload roster.

No model, compiler, simulator or capture executes during preparation. The selected
readout is source arithmetic, not a claim of equivalence to an external baseline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from math import prod
from pathlib import Path

import yaml

SCHEMA = "merlin.example.public_phase2_benchmarks.v1"
READOUTS = ("native_i8_scale1", "raw_i32")


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def describe(case, readout, *, seed):
    """Complete logical ABI and worst-case accumulator bound, without allocating."""
    if (
        not isinstance(case, dict)
        or set(case) != {"id", "kind", "label", "geometry", "source_case", "bias"}
        or not isinstance(case["id"], str)
        or not case["id"]
        or not case["id"].isascii()
        or not case["id"].replace("_", "").isalnum()
        or not case["id"][0].islower()
        or readout not in READOUTS
        or type(seed) is not int
        or not 0 <= seed < 2**31
    ):
        raise ValueError("benchmark needs explicit identity, geometry, readout and seed")
    g = case["geometry"]
    if not isinstance(g, dict) or type(case["bias"]) is not bool:
        raise ValueError("benchmark geometry/bias is malformed")
    if case["kind"] == "gemm":
        if set(g) != {"M", "N", "K"} or case["bias"]:
            raise ValueError("GEMM is non-transposed MxK by KxN without bias")
        if any(type(n) is not int or n < 1 for n in g.values()):
            raise ValueError("GEMM axes must be positive integers")
        m, n, reduction = (g[key] for key in ("M", "N", "K"))
        inputs = [{"shape": [m, reduction], "dtype": "i8"}, {"shape": [reduction, n], "dtype": "i8"}]
        shape, layout = [m, n], "row_major"
    elif case["kind"] == "conv2d":
        dimensions = {
            "batch",
            "input_height",
            "input_width",
            "kernel_height",
            "kernel_width",
            "input_channels",
            "output_channels",
            "groups",
        }
        if (
            set(g) != dimensions | {"stride", "padding", "dilation"}
            or any(type(g[key]) is not int or g[key] < 1 for key in dimensions)
            or any(
                not isinstance(g[key], list) or any(type(n) is not int for n in g[key])
                for key in ("stride", "padding", "dilation")
            )
            or g["groups"] != 1
            or g["stride"] != [1, 1]
            or g["padding"] != [0, 0]
            or g["dilation"] != [1, 1]
            or not case["bias"]
        ):
            raise ValueError("selected convolution requires explicit valid, unit-stride/dilation geometry and bias")
        b, h, w, r, s, ci, co = (
            g[key]
            for key in (
                "batch",
                "input_height",
                "input_width",
                "kernel_height",
                "kernel_width",
                "input_channels",
                "output_channels",
            )
        )
        if h < r or w < s:
            raise ValueError("valid convolution output is empty")
        inputs = [
            {"shape": [b, h, w, ci], "dtype": "i8"},
            {"shape": [r, s, ci, co], "dtype": "i8"},
            {"shape": [co], "dtype": "i32"},
        ]
        shape, reduction, layout = [b, h - r + 1, w - s + 1, co], r * s * ci, "NHWC_input_HWIO_weight_NHWC_output"
    else:
        raise ValueError("unsupported public benchmark operation")
    if reduction * 128**2 + (128 if case["bias"] else 0) > 2**31 - 1:
        raise ValueError("complete signed-i8 domain could overflow the selected i32 accumulator")
    # Readout variants remain one source group for held-out splitting. No ELF,
    # schedule, recipe or timing identity determines membership.
    return {
        "id": case["id"],
        "readout": readout,
        "inputs": inputs,
        "output": {"shape": shape, "dtype": "i8" if readout == READOUTS[0] else "i32"},
        "scalar_products": prod(shape) * reduction,
        "input_elements": sum(prod(row["shape"]) for row in inputs),
        "output_elements": prod(shape),
        "layout": layout,
        "source_group": digest(
            {
                "operation": case["kind"],
                "geometry": g,
                "bias": case["bias"],
                "input_policy": "seeded_signed_i8_small_v1",
                "seed": seed,
            }
        ),
        "arithmetic": {
            "inputs": "signed_i8",
            "accumulator": "exact_i32_no_overflow",
            "bias": "i32_channel_bias_in_accumulator" if case["bias"] else "none",
            "readout": "min(127,max(-128,acc))" if readout == READOUTS[0] else "acc",
            "scale": 1,
            "activation": "none",
            "compare": "exact_int_all_elements",
        },
    }


def render(case, readout, *, seed):
    description = describe(case, readout, seed=seed)
    g = case["geometry"]
    lines = [
        '"""Independent public operation; no external kernels or schedules."""',
        "import torch",
        "",
        "class Program(torch.nn.Module):",
    ]
    if case["kind"] == "gemm":
        lines += ["    def forward(self, a, b):", "        acc = torch._int_mm(a, b)"]
    else:
        lines += [
            "    def forward(self, x, w, bias):",
            "        acc = torch.nn.functional.conv2d(",
            "            x.permute(0, 3, 1, 2).contiguous().to(torch.int32),",
            "            w.permute(3, 2, 0, 1).contiguous().to(torch.int32), bias,",
            f"            stride={g['stride']!r}, padding={g['padding']!r}, dilation={g['dilation']!r})",
            "        acc = acc.permute(0, 2, 3, 1).contiguous()",
        ]
    if readout == READOUTS[0]:
        lines += [
            "        lower = torch.maximum(acc, torch.full_like(acc, -128))",
            "        return torch.minimum(lower, torch.full_like(lower, 127)).to(torch.int8)",
        ]
    else:
        lines += ["        return acc"]
    lines += ["", "def get_model_and_inputs():"]
    for i, row in enumerate(description["inputs"]):
        if row["dtype"] == "i8":
            value = (
                f"torch.randint(-3, 4, {row['shape']!r}, dtype=torch.int8, "
                f"generator=torch.Generator().manual_seed({seed + 104729 * i}))"
            )
        else:
            value = (
                f"torch.randint(-16, 17, {row['shape']!r}, dtype=torch.int32, "
                f"generator=torch.Generator().manual_seed({seed + 104729 * i}))"
            )
        lines.append(f"    x{i} = {value}")
    lines += ["    return Program().eval(), (" + ", ".join(f"x{i}" for i in range(len(description["inputs"]))) + ",)"]
    return "\n".join(lines) + "\n"


def prepare(profile, *, descriptor, base_roster, software_spec, output, readout=None):
    from merlin_experiments.phase0 import public_programs, workload_roster

    document = yaml.safe_load(Path(profile).read_bytes())
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or document.get("scope") != "supplemental_public_phase2_only"
        or document.get("input_policy") != "seeded_signed_i8_small_v1"
        or document.get("readouts") != list(READOUTS)
    ):
        raise ValueError("explicit supplemental public benchmark profile required")
    readout = document["default_readout"] if readout is None else readout
    rows = [describe(row, readout, seed=document["seed"]) for row in document["benchmarks"]]
    bounds = document["bounds"]
    policy = {
        key: bounds[key] for key in ("max_members", "max_capture_bytes", "max_input_elements", "max_output_elements")
    }
    policy.update(
        schema=public_programs.INTEGER_POLICY_SCHEMA,
        selection="declared_performance_applications",
        source_mode="native_integer_v1",
    )
    public_programs.validate_selection(policy)
    if (
        not rows
        or len(rows) > policy["max_members"]
        or len({r["id"] for r in rows}) != len(rows)
        or type(bounds["max_scalar_products"]) is not int
        or bounds["max_scalar_products"] < 1
        or any(
            r["input_elements"] > policy["max_input_elements"]
            or r["output_elements"] > policy["max_output_elements"]
            or r["scalar_products"] > bounds["max_scalar_products"]
            for r in rows
        )
    ):
        raise ValueError("complete benchmark roster exceeds explicit profile bounds")
    original_roster = workload_roster.load(Path(base_roster))
    selected_descriptor = yaml.safe_load(Path(descriptor).read_bytes())
    roster = {"schema": workload_roster.SCHEMA}
    for section, members in original_roster.items():
        roster[section] = {
            name: {"loader": str(row["loader"]), **({"profile": row["profile"]} if row["profile"] else {})}
            for name, row in members.items()
        }
    performance = roster.setdefault("performance_applications", {})
    if set(performance) & {r["id"] for r in rows}:
        raise ValueError("supplement may not replace an existing public workload")
    if len(performance) + len(rows) > policy["max_members"]:
        raise ValueError("expanded performance roster exceeds its selected member ceiling")
    output = Path(output).absolute()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("preparation requires a fresh output with an existing parent")
    output.mkdir(mode=0o700)
    for case, row in zip(document["benchmarks"], rows, strict=True):
        root = output / "sources" / row["id"]
        root.mkdir(parents=True)
        loader = root / "loader.py"
        loader.write_text(render(case, readout, seed=document["seed"]))
        row["loader"] = pin(loader)
        performance[row["id"]] = {"loader": str(loader)}
    # This is a new descriptor/roster selection; the originals, iteration sources
    # and full-model objective are preserved. It is never a Phase 1 release edit.
    selected_descriptor = copy.deepcopy(selected_descriptor)
    selected_descriptor["workload_spec"]["performance_applications"] = list(performance)
    selected_descriptor["numeric_profile"] = str(output / "recipe.yaml")
    descriptor_copy, roster_copy = output / "descriptor.yaml", output / "workloads.yaml"
    descriptor_copy.write_text(yaml.safe_dump(selected_descriptor, sort_keys=False))
    roster_copy.write_text(yaml.safe_dump(roster, sort_keys=False))
    staged = workload_roster.stage(roster_copy, output / "staged", descriptor=descriptor_copy)
    recipe = {
        "software_spec": str(Path(software_spec).resolve(strict=True)),
        "capsule_policy": "derived_only",
        "capsules": [],
        "performance_source_programs": policy,
        "datapath": {"compare": "exact_int", "required_oracle_tiers": ["L0", "L1", "L2", "L3"]},
    }
    (output / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    report = {
        "schema": "merlin.example.public_phase2_benchmark_preparation.v1",
        "status": "staged_capture_inputs_not_admitted",
        "profile": pin(profile),
        "original_descriptor": pin(descriptor),
        "original_roster": pin(base_roster),
        "software_spec": pin(software_spec),
        "implementation": pin(__file__),
        "upstream_source_pins": pin(Path(profile).with_name("upstream-source-pins.json")),
        "descriptor": pin(descriptor_copy),
        "recipe": pin(output / "recipe.yaml"),
        "staged_roster": pin(output / "staged/staged-workloads.json"),
        "benchmarks": rows,
        "evaluated": False,
        "admitted": False,
        "target_executed": False,
        "baseline_math_equivalence": "pending",
        "full_model_objective_changed": False,
        "comparison_intent": document["comparison"],
        "deferred": document["deferred"],
        "pending": [
            "canonical sealed capture and independent replay for every selected source",
            "ordinary private-exclusion policy choice for public_programs-only intake",
            "ordinary HW/SW derivation, writer and performance generation admission",
            "matched external baseline reference, timing boundary and hardware",
        ],
        "capture_api": "merlin_experiments.phase0.capture_selection.select",
        "capture_kwargs": {"dtype": "int8", "recipe": None, "worker_options": {"source_mode": "native_integer_v1"}},
    }
    for row in rows:
        row["workload_root"] = staged["performance_applications"][row["id"]]["workload_root"]
    (output / "preparation.json").write_bytes(canonical(report))
    return report


def select_capture(
    preparation,
    benchmark,
    *,
    expected_sha256,
    m2m_root,
    worker,
    venv,
    schemas_root,
    run_dir,
    output_dir,
    execution_timeout_seconds=120,
):
    """Opt-in stock preselection only; never issue a model/target execution."""
    from merlin_experiments.phase0.capture_selection import select

    def reopen():
        if pin(preparation)["sha256"] != expected_sha256:
            raise ValueError("selected preparation changed")
        prepared = json.loads(Path(preparation).read_bytes())
        if prepared.get("schema") != "merlin.example.public_phase2_benchmark_preparation.v1":
            raise ValueError("ordinary benchmark preparation required")
        for key in (
            "profile",
            "implementation",
            "upstream_source_pins",
            "descriptor",
            "recipe",
            "staged_roster",
            "original_descriptor",
            "original_roster",
            "software_spec",
        ):
            if pin(prepared[key]["path"]) != prepared[key]:
                raise ValueError("selected preparation input changed: " + key)
        (row,) = [r for r in prepared["benchmarks"] if r["id"] == benchmark]
        profile = yaml.safe_load(Path(prepared["profile"]["path"]).read_bytes())
        (case,) = [r for r in profile["benchmarks"] if r["id"] == benchmark]
        if {k: v for k, v in row.items() if k not in {"loader", "workload_root"}} != describe(
            case, row["readout"], seed=profile["seed"]
        ):
            raise ValueError("selected source description changed")
        staged = json.loads(Path(prepared["staged_roster"]["path"]).read_bytes())
        member = staged["performance_applications"][benchmark]
        root = Path(member["workload_root"])
        if (
            str(root) != row["workload_root"]
            or member["profile_sha256"] is not None
            or member["source_loader"] != row["loader"]["path"]
            or member["loader_sha256"] != row["loader"]["sha256"]
            or pin(root / "loader.py")["sha256"] != row["loader"]["sha256"]
            or pin(row["loader"]["path"]) != row["loader"]
            or (root / "loader.py").read_text() != render(case, row["readout"], seed=profile["seed"])
        ):
            raise ValueError("staged benchmark source changed")
        return root

    root = reopen()
    result = select(
        m2m_root=Path(m2m_root),
        workload_root=root,
        worker=Path(worker),
        venv=Path(venv),
        schemas_root=Path(schemas_root),
        run_dir=Path(run_dir),
        output_dir=Path(output_dir),
        execution_timeout_seconds=execution_timeout_seconds,
        dtype="int8",
        recipe=None,
        worker_options={"source_mode": "native_integer_v1"},
    )
    reopen()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("profile.yaml"))
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--base-roster", type=Path, required=True)
    parser.add_argument("--software-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readout", choices=READOUTS)
    args = parser.parse_args(argv)
    report = prepare(
        args.profile,
        descriptor=args.descriptor,
        base_roster=args.base_roster,
        software_spec=args.software_spec,
        output=args.output,
        readout=args.readout,
    )
    print(json.dumps({"status": report["status"], "benchmarks": len(report["benchmarks"])}))


if __name__ == "__main__":
    main()
