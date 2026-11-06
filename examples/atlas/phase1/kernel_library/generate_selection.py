#!/usr/bin/env python3
"""Generate selection.yaml from the atlas-kernels manifests and model coverage data.

Every input is read from the pinned commit with ``git show``, so a dirty checkout or a moving branch
cannot change the output:

- ``kernels/*/kernel.yaml`` gives the qualified families, their export paths, and each family's
  ``stage`` (standalone launch or a stage fused into another family's launch), copied as a comment.
- ``coverage/mapping.yaml``, ``coverage/fixture/*.json``, and ``coverage/build_table.py`` give every
  op instance the qualified builds place on a family. The generator places them with the library's own
  ``build_table.place`` and checks placed Linear K/N against the rules below.

The manifests state shapes in prose. ``manifest_shapes`` interprets the reviewed forms and refuses
unrecognized prose instead of widening the domain. Semantic aliases and hardware requirements remain
reviewed metadata in ``FAMILY_RULES``. This catalog covers producer-fused stages, dimension-64 v2
attention, the documented pool stem, and the manifest RMSNorm width domain. Opt-in standalone stages
and wide/vision-fast attention are outside this catalog. A rule is authoring evidence, not executable
admission: numeric policy, fusion composition, layouts, masks, and arena/resource bounds still need
independent checks. No kernel generator is imported or executed. The library coverage placer only
cross-checks recorded projection variants and placed Linear K/N; it does not prove model coverage.

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
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

from merlin.targetgen.kernel_family_selection import shape_constraint_reasons

HERE = Path(__file__).resolve()
SELECTION = HERE.parent / "selection.yaml"

# Current Qwen full-model bringup pin, reachable from the public origin/feat/kernel-library.
# source-map.json retains the older relocation snapshot, not the current bringup requirement.
SOURCE_COMMIT = "ce465827a977ba8451b585f71910726e7f12b885"
SOURCE_PR = "https://github.com/ucb-bar/atlas-kernels"
PACKAGE = "out/artifacts/targets/atlas/kernel_library_v1"
# Library-wide files a family needs to import and build on its own: the make targets, the import hook,
# and lib/ (backend, golden arithmetic, standalone fallbacks). coverage/ is not exported: it holds the
# held-out models' op inventories, which the selection rules do not need.
SHARED_PATHS = ("Makefile", "atlas_kernels_path.py", "lib", "kernels/common.mk")

MXU = ("systolic_array", "fp8_mxu", "bf16_accumulate")

# launch: "standalone" families are launches of their own; "fused" families are stages inside another
# family's launch, selected through a compound semantic op (as Radiance's rope_qkv_fused is selected by
# matmul_rope). The schema has no field for this, so it is recorded as a comment on each rule.
FAMILY_RULES = {
    "projection_gemm": {
        "launch": "standalone",
        "semantic_ops": ["matmul"],
        "dtypes": ["fp8_e4m3"],
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
        "requires_hardware": [*MXU, "vector_unit", "tanh"],
        "priority": 100,
        "note": "producer post-pass only; standalone BF16 RoPE is outside this catalog",
    },
    "rmsnorm": {
        "launch": "standalone",
        "semantic_ops": ["rmsnorm"],
        "dtypes": ["bf16"],
        "requires_hardware": ["vector_unit", "sqrt", "reciprocal"],
        "priority": 80,
        "note": "physical rows 32 or 128; logical width equals physical width; "
        "epsilon needs separate numeric admission",
    },
    "attention": {
        "launch": "standalone",
        "semantic_ops": ["attention_gqa", "attention_gqa_causal", "attention_gqa_masked"],
        "dtypes": ["bf16"],
        "requires_hardware": [*MXU, "vector_unit", "exp", "reciprocal"],
        "priority": 100,
        "note": "dimension-64 v2, unpacked heads; ceil(Sq/32)*ceil(Sk/32) <= 2048; "
        "masks need separate admission",
    },
    "smol_fusion": {
        "launch": "standalone",
        "semantic_ops": ["matmul_bias_gelu_tanh"],
        "dtypes": ["fp8_e4m3"],
        "requires_hardware": [*MXU, "vector_unit", "tanh"],
        "priority": 90,
        "note": None,
    },
    "resnet_pool": {
        "launch": "standalone",
        "semantic_ops": ["max_pool2d"],
        "dtypes": ["fp8_e4m3"],
        "requires_hardware": [*MXU, "vector_unit"],
        "priority": 80,
        "note": "manifest-qualified stem geometry only; exactness also requires nonnegative normal FP8 inputs",
    },
    "resnet_patch_gather": {
        "launch": "fused",
        "fused_into": "projection_gemm",
        "stage_dtype": "fp8_e4m3",
        "semantic_ops": ["conv2d_im2col_matmul"],
        "dtypes": ["fp8_e4m3"],
        "requires_hardware": list(MXU),
        "priority": 80,
        "note": "1x1/3x3 with same padding; physical M/K, blocked recipe, and arena bounds need separate admission",
    },
}

# Vector mnemonics whose presence in a family's own generator modules requires a declared capability.
MNEMONIC_FEATURES = {"vexp": "exp", "vrecip": "reciprocal", "vsqrt": "sqrt", "vtanh": "tanh"}

# Reviewed semantic aliases for this example, not inferred executable semantics. New aliases need
# an explicit review here as well as an authored FAMILY_RULES entry.
SEMANTIC_OPS = frozenset(
    {
        "matmul",
        "matmul_silu",
        "matmul_mul",
        "matmul_add",
        "matmul_relu",
        "matmul_exp",
        "matmul_reciprocal",
        "matmul_gelu_tanh",
        "matmul_scale",
        "matmul_rope",
        "rmsnorm",
        "attention_gqa",
        "attention_gqa_causal",
        "attention_gqa_masked",
        "matmul_bias_gelu_tanh",
        "max_pool2d",
        "conv2d_im2col_matmul",
    }
)

HEADER = """\
# Generated by generate_selection.py from atlas-kernels {short}; do not edit by hand.
# Shape rules derive from pinned kernels/*/kernel.yaml, with coverage placement cross-checks.
# Radiance v1 field names, plus exact sets, powers of two, divisibility, and tiled-product bounds.
#
# This is the current Qwen full-model bringup pin, reachable from public origin/feat/kernel-library.
# source-map.json records the older relocation snapshot. No information treatment is enabled.
"""

COMPILER_COMMENT = """\
# Currently disabled selection and materialization example with no Atlas compile or runtime consumer.
# These authored rules describe candidate schedule strategies from semantic workload facts. A match
# in the generic matcher does NOT extend the source library's numeric
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


def _tree(checkout: Path, commit: str, path: str, *, directories_only: bool = False) -> list[str]:
    args = ["git", "-C", str(checkout), "ls-tree", "--name-only"]
    if directories_only:
        args.append("-d")
    out = subprocess.run(
        [*args, f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        line for line in out.splitlines()
        if line and not any(word in line.lower() for word in ("hammer", "vlsi"))
    ]


def _load_build_table(source: str):
    with tempfile.TemporaryDirectory(prefix="atlas-build-table.") as td:
        path = Path(td) / "build_table.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("atlas_kernels_build_table", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _refusals(shape: dict, constraints: dict) -> list[str]:
    """Check the dimensions provided by a partial coverage record against the shared matcher."""
    return shape_constraint_reasons(shape, {name: rule for name, rule in constraints.items() if name in shape})


def _pad32(value: int) -> int:
    return (value + 31) // 32 * 32


def check_atlas_mlir_pin(commit: str, checkout: Path | None) -> dict:
    """Compare a selection pin with the provided atlas-mlir HEAD gitlink, without a checkout update."""
    if checkout is None:
        return {"status": "not-checked"}
    probe = subprocess.run(
        ["git", "-C", str(checkout), "ls-tree", "HEAD", "--", "third_party/atlas-kernels"],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = probe.stdout.split()
    if probe.returncode or len(fields) != 4 or fields[:2] != ["160000", "commit"]:
        raise SelectionSourceError("atlas-mlir checkout must have a third_party/atlas-kernels HEAD gitlink")
    gitlink = fields[2]
    if gitlink != commit:
        raise SelectionSourceError(f"atlas-mlir atlas-kernels gitlink {gitlink} differs from selection pin {commit}")
    return {"status": "checked", "source_commit": commit, "gitlink_commit": gitlink}


def _check_rule(family: str, rule: dict) -> None:
    ops = rule["semantic_ops"]
    if not ops or any(op not in SEMANTIC_OPS for op in ops):
        raise SelectionSourceError(f"{family}: semantic_ops must use reviewed aliases")
    if family == "projection_gemm" and not set(MXU).issubset(rule["requires_hardware"]):
        raise SelectionSourceError(f"{family}: requires_hardware must include {list(MXU)}")
    if family == "resnet_pool":
        # Fixed MaxPool geometry and its channel domain (validate requires channels to divide 512).
        # This does not validate the spatial/resource constraints or nonnegative input predicate.
        shape = rule["shape"]
        fixed = {"kernel": 3, "stride": 2, "padding": 1}
        if _refusals(fixed, shape) or not any(
            not _refusals({"channels": channels}, shape) for channels in (32, 64, 128, 256, 512)
        ):
            raise SelectionSourceError(f"{family}: pool geometry cannot match the implemented kernel")


def manifest_shapes(manifests: dict) -> dict:
    """Interpret only reviewed manifest forms; unknown source prose must be reviewed explicitly."""
    def require(family, key, expected):
        value = manifests[family]["constraints"][key]
        if value != expected:
            raise SelectionSourceError(f"{family}: unreviewed manifest constraint {key}: {value!r}")
        return value

    projection = {}
    for source, dimension in (("m", "M"), ("k", "K"), ("n", "N")):
        text = manifests["projection_gemm"]["constraints"][source]
        match = re.fullmatch(r"multiple of ([1-9][0-9]*)", text)
        if not match:
            raise SelectionSourceError(f"projection_gemm: unreviewed {source}: {text!r}")
        projection[dimension] = {"min": int(match[1]), "multiple_of": int(match[1])}

    norm = manifests["rmsnorm"]["constraints"]
    rows = re.fullmatch(r"([1-9][0-9]*) \(prefill\) or ([1-9][0-9]*) \(one decode row padded\)", norm["rows"])
    width = re.fullmatch(
        r"positive multiples of ([1-9][0-9]*) through ([1-9][0-9]*); "
        r"pinned resident path for 960 and 2048; streamed standalone path otherwise",
        norm["width"],
    )
    if not rows or not width:
        raise SelectionSourceError("rmsnorm: unreviewed row/width domain")
    norm_shape = {"rows": {"one_of": sorted(map(int, rows.groups()))},
                  "cols": {"min": int(width[1]), "max": int(width[2]), "multiple_of": int(width[1])}}

    rope = manifests["rope"]
    require("rope", "rows", "producer M 32 or 128; standalone positive physical M multiple of 32")
    producer = [tuple(map(int, shape.split("x"))) for model in rope["models"] for shape in model["shapes"]]
    # The documented producer shapes form a complete Cartesian domain; never widen a correlated set.
    axes = [sorted({shape[index] for shape in producer}) for index in range(3)]
    if set(product(*axes)) != set(producer):
        raise SelectionSourceError("rope: producer shapes need correlated alternatives")
    rope_shape = {dim: {"one_of": values} for dim, values in zip(("M", "K", "N"), axes)}
    rope_shape["head_dim"] = {"eq": rope["constraints"]["head_dimension"]}

    require("attention", "gqa", "query_heads a multiple of kv_heads")
    require(
        "attention", "tiles",
        "dimension 64: at most 64 key tiles and 128 query tiles; wide: key_tiles*head_dim*32 <= 262144 bytes, "
        "at most 128 query tiles, query_tiles*key_tiles <= 2048",
    )
    # stage_v2's 16 KiB descriptor area holds 2048 eight-byte descriptors, including D64.
    attention = {
        "Sq": {"min": 1, "max": 4096, "tile_product_max": {"dimension": "Sk", "tile": 32, "max": 2048}},
        "Sk": {"min": 1, "max": 2048}, "D": {"eq": 64},
        "QH": {"min": 1, "multiple_of_dimension": "KVH"}, "KVH": {"min": 1},
    }

    fusion_shape = manifests["smol_fusion"]["select"][0]["shape"].split()[0]
    fusion = {dim: {"eq": int(value)} for dim, value in zip(("M", "K", "N"), fusion_shape.split("x"), strict=True)}
    # The source validator has correlated register/VMEM geometry. Keep only the qualified stem case.
    pool_shapes = [shape for model in manifests["resnet_pool"]["models"] for shape in model["shapes"]]
    if len(pool_shapes) != 1:
        raise SelectionSourceError("resnet_pool: review new qualified geometries")
    channels, height, width = map(int, pool_shapes[0].split(" -> ")[0].split("x"))
    pool = {"kernel": {"eq": 3}, "stride": {"eq": 2}, "padding": {"eq": 1},
            "channels": {"eq": channels}, "height": {"eq": height}, "width": {"eq": width}}

    require("resnet_patch_gather", "channels", "power of two, multiple of 32")
    require("resnet_patch_gather", "kernel", "odd with same padding")
    require("resnet_patch_gather", "stride", "1 or 2")
    select = manifests["resnet_patch_gather"]["select"][0]["shape"]
    if select != "kernel 1 or 3, stride 1 or 2":
        raise SelectionSourceError("resnet_patch_gather: unreviewed select geometry")
    patch = {"channels": {"min": 32, "multiple_of": 32, "power_of_two": True},
             "kernel": {"one_of": [1, 3]}, "stride": {"one_of": [1, 2]},
             "out_channels": {"min": 32, "multiple_of": 32}}
    return {"projection_gemm": projection, "epilogue": deepcopy(projection), "rne_add": deepcopy(projection),
            "rmsnorm": norm_shape, "rope": rope_shape, "attention": attention,
            "smol_fusion": fusion, "resnet_pool": pool, "resnet_patch_gather": patch}


def collect(checkout: Path, commit: str) -> dict:
    """Read and cross-check the pinned sources; return the data the YAML is rendered from."""
    raw_mlir = os.environ.get("MERLIN_ATLAS_MLIR")
    if raw_mlir is not None and not raw_mlir.strip():
        raise SelectionSourceError("MERLIN_ATLAS_MLIR must name an atlas-mlir checkout")
    pin_check = check_atlas_mlir_pin(commit, Path(raw_mlir) if raw_mlir is not None else None)
    mapping = yaml.safe_load(_show(checkout, commit, "coverage/mapping.yaml"))
    order = list(mapping["kernels"])
    directories = sorted(_tree(checkout, commit, "kernels", directories_only=True))
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
    shapes = manifest_shapes(manifests)
    rules = deepcopy(FAMILY_RULES)
    for family, rule in rules.items():
        rule.setdefault("shape", shapes[family])
    for family, manifest in manifests.items():
        rule = rules[family]
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
        _check_rule(family, rule)

    projection = rules["projection_gemm"]["shape"]
    for variant in manifests["projection_gemm"].get("variants", ()):
        for use in variant.get("uses", ()):
            m, k, n = use["mkn"]
            reasons = _refusals({"M": m, "K": k, "N": n}, projection)
            if reasons:
                raise SelectionSourceError(f"projection_gemm variant {variant['name']} {m}x{k}x{n}: {reasons}")

    for family, rule in rules.items():
        if rule["shape"] != shapes[family]:
            raise SelectionSourceError(f"{family}: shape differs from the pinned manifest domain")

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
                reasons = _refusals(shape, rules[family]["shape"])
                if reasons:
                    raise SelectionSourceError(f"{model} {row['module']} on {family}: {reasons}")
    return {
        "commit": commit,
        "order": order,
        "manifests": manifests,
        "rules": rules,
        "instances": instances,
        "atlas_mlir_pin": pin_check,
    }


def _flow(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_flow(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(item) for item in value) + "]"
    return str(value).lower() if isinstance(value, bool) else str(value)


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
        "# Candidate authoring families, not a claim of numerical qualification at this revision.",
        "experimental_families: []",
        "",
        COMPILER_COMMENT.rstrip("\n"),
        "compiler_selection:",
        "  schema: semantic_kernel_family_registry_v1",
        "  rules:",
    ]
    for family in data["order"]:
        rule = data["rules"][family]
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
        expected = data["rules"][rule["family"].split("/", 1)[1]]
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
    print(
        json.dumps(
            {
                "families": len(data["order"]),
                "placed_op_instances": data["instances"],
                "atlas_mlir_pin": data["atlas_mlir_pin"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
