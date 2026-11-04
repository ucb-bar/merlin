#!/usr/bin/env python3
"""Generate selection.yaml from the atlas-kernels manifests and model coverage data.

Every input is read from the pinned commit with ``git show``, so a dirty checkout or a moving branch
cannot change the output:

- ``kernels/*/kernel.yaml`` gives the qualified families, their export paths, and each family's
  ``stage`` (standalone launch or a stage fused into another family's launch), copied as a comment.
- ``coverage/mapping.yaml``, ``coverage/fixture/*.json``, and ``coverage/build_table.py`` give every
  op instance the qualified builds place on a family. The generator places them with the library's own
  ``build_table.place`` and checks them against the rules below.

The manifests' ``select`` shapes are prose, so the semantic rule for each family (``FAMILY_RULES``)
is written here by hand in the field names of examples/radiance/phase1/kernel_library/selection.yaml.
The generator refuses a family without a rule, a rule without a family, a dtype the manifest does not
list, a vector transcendental the family emits without declaring it, and a recorded projection shape
or model projection instance that the rule would refuse.

    python3 examples/atlas/phase1/kernel_library/generate_selection.py --checkout /path/to/atlas-kernels
    python3 examples/atlas/phase1/kernel_library/generate_selection.py --checkout ... --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
SELECTION = HERE.parent / "selection.yaml"

# atlas-kernels feat/kernel-library head. Nicolas updates this pin, and reruns this generator, after
# pushing atlas-kernels to GitHub; source_pr then becomes the pull-request URL.
SOURCE_COMMIT = "6123bef11a3faa265972d72e6165c146a5fdc13e"
SOURCE_PR = "https://github.com/ucb-bar/atlas-kernels"
PACKAGE = "out/artifacts/targets/atlas/kernel_library_v1"
# Library-wide files a family needs to import and build on its own: the make targets, the import hook,
# and lib/ (backend, golden arithmetic, standalone fallbacks). coverage/ is not exported: it holds the
# held-out models' op inventories, which the selection rules do not need.
SHARED_PATHS = ("Makefile", "atlas_kernels_path.py", "lib")

MXU = ("systolic_array", "fp8_mxu", "bf16_accumulate")
MULT32 = {"multiple_of": 32}
PROJECTION_SHAPE = {"M": MULT32, "K": MULT32, "N": MULT32}

# launch: "standalone" families are launches of their own; "fused" families are stages inside another
# family's launch, selected through a compound semantic op (as Radiance's rope_qkv_fused is selected by
# matmul_rope). The schema has no field for this, so it is recorded as a comment on each rule.
FAMILY_RULES = {
    "projection_gemm": {
        "launch": "standalone",
        "semantic_ops": ["matmul"],
        "dtypes": ["fp8_e4m3"],
        "shape": PROJECTION_SHAPE,
        "requires_hardware": list(MXU),
        "priority": 40,
        "note": "the library picks the traffic-reuse recipe for an exact MxKxN (kernel.yaml variants); "
        "any other multiple-of-32 shape uses the default program",
    },
    "epilogue": {
        "launch": "fused",
        "fused_into": "projection_gemm",
        "stage_dtype": "bf16",
        "semantic_ops": [
            "matmul_silu",
            "matmul_mul",
            "matmul_add",
            "matmul_relu",
            "matmul_exp",
            "matmul_reciprocal",
            "matmul_gelu_tanh",
            "matmul_scale",
        ],
        "dtypes": ["fp8_e4m3"],
        "shape": PROJECTION_SHAPE,
        "requires_hardware": [*MXU, "vector_unit", "exp", "reciprocal", "tanh"],
        "priority": 60,
        "note": "its plain add truncates; matmul_add selects rne_add (priority 70) for the exact BF16 sum",
    },
    "rne_add": {
        "launch": "fused",
        "fused_into": "projection_gemm",
        "stage_dtype": "bf16",
        "semantic_ops": ["matmul_add"],
        "dtypes": ["fp8_e4m3"],
        "shape": PROJECTION_SHAPE,
        "requires_hardware": [*MXU, "vector_unit", "tanh"],
        "priority": 70,
        "note": None,
    },
    "rope": {
        "launch": "fused",
        "fused_into": "projection_gemm",
        "stage_dtype": "bf16",
        "semantic_ops": ["matmul_rope"],
        "dtypes": ["fp8_e4m3"],
        "shape": {
            "M": {"min": 32, "max": 128, "multiple_of": 32},
            "K": MULT32,
            "N": {"multiple_of": 64},
            "head_dim": {"eq": 64},
        },
        "requires_hardware": [*MXU, "vector_unit", "tanh"],
        "priority": 100,
        "note": "qualified at M 32 and 128 only; the schema has no set constraint, so M 64 and 96 also match",
    },
    "rmsnorm": {
        "launch": "standalone",
        "semantic_ops": ["rmsnorm"],
        "dtypes": ["bf16"],
        "shape": {"rows": {"min": 32, "max": 128, "multiple_of": 32}, "cols": {"eq": 2048}},
        "requires_hardware": ["vector_unit", "sqrt", "reciprocal"],
        "priority": 80,
        "note": "qualified at rows 32 and 128 only; the schema has no set constraint, so rows 64 and 96 also match",
    },
    "attention": {
        "launch": "standalone",
        "semantic_ops": ["attention_gqa", "attention_gqa_causal", "attention_gqa_masked"],
        "dtypes": ["bf16"],
        "shape": {
            "Sq": {"min": 1, "max": 4096},
            "Sk": {"min": 1, "max": 2048},
            "D": {"eq": 64},
            "QH": {"min": 1},
            "KVH": {"min": 1},
        },
        "requires_hardware": [*MXU, "vector_unit", "exp", "reciprocal"],
        "priority": 100,
        "note": "Sq and Sk are logical (the kernel pads to 32-row tiles); QH must be a multiple of KVH, "
        "which the schema cannot express",
    },
    "smol_fusion": {
        "launch": "standalone",
        "semantic_ops": ["matmul_bias_gelu_tanh"],
        "dtypes": ["fp8_e4m3"],
        "shape": {"M": {"eq": 128}, "K": {"eq": 768}, "N": {"eq": 3072}},
        "requires_hardware": [*MXU, "vector_unit", "tanh"],
        "priority": 90,
        "note": None,
    },
    "resnet_pool": {
        "launch": "standalone",
        "semantic_ops": ["max_pool2d"],
        "dtypes": ["fp8_e4m3"],
        "shape": {
            "kernel": {"eq": 3},
            "stride": {"eq": 2},
            "padding": {"eq": 1},
            "channels": MULT32,
            "height": {"min": 1},
            "width": {"min": 1},
        },
        "requires_hardware": ["vector_unit"],
        "priority": 80,
        "note": "exact only for nonnegative inputs, which the schema cannot express",
    },
    "resnet_patch_gather": {
        "launch": "fused",
        "fused_into": "projection_gemm",
        "stage_dtype": "fp8_e4m3",
        "semantic_ops": ["conv2d_im2col_matmul"],
        "dtypes": ["fp8_e4m3"],
        "shape": {
            "channels": MULT32,
            "kernel": {"min": 1, "max": 3},
            "stride": {"min": 1, "max": 2},
            "out_channels": MULT32,
        },
        "requires_hardware": list(MXU),
        "priority": 80,
        "note": "channels must be a power of two and the kernel odd, which the schema cannot express",
    },
}

# Vector mnemonics whose presence in a family's own generator modules requires a declared capability.
MNEMONIC_FEATURES = {"vexp": "exp", "vrecip": "reciprocal", "vsqrt": "sqrt", "vtanh": "tanh"}

HEADER = """\
# Generated by generate_selection.py from atlas-kernels {short} (kernels/*/kernel.yaml and the
# coverage/ placement of every TinyLlama and SmolVLA op); do not edit by hand. Same schema and field
# names as examples/radiance/phase1/kernel_library/selection.yaml.
#
# The pin is the local feat/kernel-library head. Nicolas updates SOURCE_COMMIT in the generator and
# reruns it after pushing atlas-kernels to GitHub; source_pr then names the pull request. There is no
# atlas_kernels entry in merlin/contract/hardware_pins.yaml yet (see the phase 1 README).
"""

COMPILER_COMMENT = """\
# Compiler-facing registry.  These rows identify reusable schedule strategies from semantic workload
# facts.  A match selects a lowering candidate; it does NOT extend the source library's numeric
# qualification to a new implementation or shape.  Hardware features are supplied by introspection or
# the target contract at the call site; the selector does not manufacture them.
#
# Shapes are the extents a launch receives: callers pad projection, epilogue, RoPE, and norm rows to a
# multiple of 32, and split projection rows into panels; attention takes logical Sq and Sk.  Families
# whose stage is not a launch of their own are selected through compound semantic ops (matmul_<stage>,
# conv2d_im2col_matmul); each rule's comment repeats the manifest's stage.
"""

PROHIBITED_USES = (
    "copying kernel bodies into a submission",
    "linking or calling the reference library from a submission",
    "capsule-name or expected-output dispatch",
)


class SelectionSourceError(ValueError):
    """The pinned atlas-kernels data disagrees with the selection rules."""


def _show(checkout: Path, commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{path}"], check=True, capture_output=True, text=True
    ).stdout


def _tree(checkout: Path, commit: str, path: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(checkout), "ls-tree", "--name-only", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def _load_build_table(source: str):
    with tempfile.TemporaryDirectory(prefix="atlas-build-table.") as td:
        path = Path(td) / "build_table.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("atlas_kernels_build_table", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _refusals(shape: dict, constraints: dict) -> list[str]:
    """Same constraint semantics as the semantic_kernel_family_registry_v1 selector."""
    reasons = []
    for name, constraint in constraints.items():
        if name not in shape:
            continue
        value = shape[name]
        if "eq" in constraint and value != constraint["eq"]:
            reasons.append(f"{name}={value} != {constraint['eq']}")
        if "min" in constraint and value < constraint["min"]:
            reasons.append(f"{name}={value} < {constraint['min']}")
        if "max" in constraint and value > constraint["max"]:
            reasons.append(f"{name}={value} > {constraint['max']}")
        if "multiple_of" in constraint and value % constraint["multiple_of"]:
            reasons.append(f"{name}={value} not a multiple of {constraint['multiple_of']}")
    return reasons


def _pad32(value: int) -> int:
    return (value + 31) // 32 * 32


def collect(checkout: Path, commit: str) -> dict:
    """Read and cross-check the pinned sources; return the data the YAML is rendered from."""
    mapping = yaml.safe_load(_show(checkout, commit, "coverage/mapping.yaml"))
    order = list(mapping["kernels"])
    directories = sorted(_tree(checkout, commit, "kernels"))
    if sorted(order) != directories:
        raise SelectionSourceError(f"coverage kernels {sorted(order)} differ from kernels/ {directories}")
    if sorted(FAMILY_RULES) != directories:
        raise SelectionSourceError(f"FAMILY_RULES {sorted(FAMILY_RULES)} differ from kernels/ {directories}")

    manifests = {}
    for family in order:
        manifest = yaml.safe_load(_show(checkout, commit, f"kernels/{family}/kernel.yaml"))
        if manifest.get("schema") != "atlas.kernel.v1" or manifest.get("family") != family:
            raise SelectionSourceError(f"kernels/{family}/kernel.yaml is not an atlas.kernel.v1 manifest of {family}")
        manifests[family] = manifest
    for family, manifest in manifests.items():
        rule = FAMILY_RULES[family]
        # A fused stage is requested with the dtype of the launch it joins, so its dtypes are checked
        # against that family's manifest and its own manifest must list the dtype the stage consumes.
        owner = manifests[rule.get("fused_into", family)]
        checks = [(dtype, owner) for dtype in rule["dtypes"]]
        if "stage_dtype" in rule:
            checks.append((rule["stage_dtype"], manifest))
        for dtype, source in checks:
            declared = json.dumps([source.get("dtypes"), [row.get("dtype") for row in source.get("select", ())]])
            if dtype not in declared:
                raise SelectionSourceError(f"{family}: dtype {dtype} is not in kernels/{source['family']}/kernel.yaml")
        sources = "".join(
            _show(checkout, commit, f"kernels/{family}/{name}")
            for name in _tree(checkout, commit, f"kernels/{family}")
            if name.endswith(".py") and name not in ("gen.py", "reference.py")
        )
        for mnemonic, feature in MNEMONIC_FEATURES.items():
            if re.search(rf"\b{mnemonic}\b", sources) and feature not in rule["requires_hardware"]:
                raise SelectionSourceError(f"{family}: emits {mnemonic} but does not require {feature}")
        if rule["launch"] == "fused" and (
            rule.get("fused_into") not in manifests or not all("_" in op for op in rule["semantic_ops"])
        ):
            raise SelectionSourceError(f"{family}: a fused stage needs fused_into and compound semantic ops")

    projection = FAMILY_RULES["projection_gemm"]["shape"]
    for variant in manifests["projection_gemm"].get("variants", ()):
        for use in variant.get("uses", ()):
            m, k, n = use["mkn"]
            reasons = _refusals({"M": m, "K": k, "N": n}, projection)
            if reasons:
                raise SelectionSourceError(f"projection_gemm variant {variant['name']} {m}x{k}x{n}: {reasons}")

    table = _load_build_table(_show(checkout, commit, "coverage/build_table.py"))
    instances = {}
    for model in table.MODELS:
        inventory = json.loads(_show(checkout, commit, f"coverage/fixture/{model}.json"))
        placed, unmapped, _ = table.place(model, inventory, mapping["rules"][model], mapping["kernels"])
        if unmapped:
            raise SelectionSourceError(f"{model}: {len(unmapped)} coverage rows are unmapped")
        for row in placed:
            if row["kind"] != "atlas":
                continue
            family = row["where"].split(":", 1)[1]
            instances[family] = instances.get(family, 0) + row["count"]
            if row["op"] == "aten.linear.default":
                k = row["inputs"][0][0][-1]
                n = row["inputs"][1][0][0]
                shape = {"K": _pad32(k), "N": _pad32(n)}
                if family == "smol_fusion":
                    shape = {"K": k, "N": n}
                reasons = _refusals(shape, FAMILY_RULES[family]["shape"])
                if reasons:
                    raise SelectionSourceError(f"{model} {row['module']} on {family}: {reasons}")
    return {"commit": commit, "order": order, "manifests": manifests, "instances": instances}


def _flow(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_flow(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(item) for item in value) + "]"
    return str(value)


def render(data: dict) -> str:
    lines = [HEADER.format(short=data["commit"][:7]).rstrip("\n")]
    lines += [
        "schema: atlas_kernel_library_selection_v1",
        "name: atlas_kernels_model_kernel_library",
        "source_pin: atlas_kernels",
        f"source_commit: {data['commit']}",
        f"source_pr: {SOURCE_PR}",
        f"package: {PACKAGE}",
        "purpose: read-only authoring evidence; submitted packages may not copy, link, or call it",
        "",
        "shared_paths:",
        *(f"  - {path}" for path in SHARED_PATHS),
        "",
        "qualified_families:",
        *(f"  - kernels/{family}" for family in data["order"]),
        "",
        "# Every family runs on npu_model 47b8599 (numerics rtl) in a qualified Atlas build; none is experimental.",
        "experimental_families: []",
        "",
        COMPILER_COMMENT.rstrip("\n"),
        "compiler_selection:",
        "  schema: semantic_kernel_family_registry_v1",
        "  rules:",
    ]
    for family in data["order"]:
        rule = FAMILY_RULES[family]
        stage = " ".join(str(data["manifests"][family]["stage"]).split())
        launch = f"fused into {rule['fused_into']}" if rule["launch"] == "fused" else "standalone launch"
        lines.append(f"    # {launch}; kernel.yaml stage: {stage}")
        if rule["note"]:
            lines.append(f"    # {rule['note']}")
        lines += [
            f"    - family: kernels/{family}",
            f"      semantic_ops: {_flow(rule['semantic_ops'])}",
            f"      dtypes: {_flow(rule['dtypes'])}",
            f"      shape: {_flow(rule['shape'])}",
            f"      requires_hardware: {_flow(rule['requires_hardware'])}",
            f"      priority: {rule['priority']}",
        ]
    lines += ["", "prohibited_uses:", *(f"  - {use}" for use in PROHIBITED_USES)]
    text = "\n".join(lines) + "\n"
    parsed = yaml.safe_load(text)
    for rule in parsed["compiler_selection"]["rules"]:
        expected = FAMILY_RULES[rule["family"].split("/", 1)[1]]
        if any(rule[key] != expected[key] for key in ("semantic_ops", "dtypes", "shape", "requires_hardware")):
            raise AssertionError(f"rendered rule for {rule['family']} does not round-trip")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout",
        default=os.environ.get("MERLIN_ATLAS_KERNELS", ""),
        help="atlas-kernels checkout used as a git object database",
    )
    parser.add_argument("--commit", default=SOURCE_COMMIT)
    parser.add_argument("--output", default=str(SELECTION))
    parser.add_argument("--check", action="store_true", help="fail if --output differs instead of writing it")
    args = parser.parse_args(argv)
    if not args.checkout:
        parser.error("--checkout or MERLIN_ATLAS_KERNELS is required")
    data = collect(Path(args.checkout).resolve(), args.commit)
    text = render(data)
    output = Path(args.output)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != text:
            print(f"{output} differs from the generated selection", file=sys.stderr)
            return 1
    else:
        output.write_text(text, encoding="utf-8")
    print(json.dumps({"families": len(data["order"]), "placed_op_instances": data["instances"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
