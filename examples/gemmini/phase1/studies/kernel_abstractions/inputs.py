"""Freeze source workloads for the bounded kernel-abstraction study.

This is an operator preparation recipe, not a compiler policy. Held-out inputs
are independent contracts, never claims about unobserved model executions.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import yaml

from merlin.targetgen.component_program import render
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

GEMMS = [
    "ZA_gemm_512x512x512_native_i8",
    "ZA_gemm_12544x256x64_native_i8",
    "ZA_gemm_12544x64x64_native_i8",
    "ZA_gemm_3136x512x128_native_i8",
    "ZA_gemm_3136x128x512_native_i8",
    "ZA_gemm_784x256x1024_native_i8",
]
CONVS = [f"ZA_conv_3x3x{h}x{h}x{c}x{c}x4_native_i8" for h, c in [(56, 64), (28, 128), (14, 256)]]
ATTENTION = ["X_att_qk_h32_gqa4_m16k64n16", "X_att_pv_h32_gqa4_m16k16n64"]
TINY = [
    "X_mm_down_m16k5632n2048",
    "X_mm_gu_m16k2048n5632x2",
    "X_mm_o_m16k2048n2048",
    "X_mm_qkv_m16k2048n2048_256_256",
    "JT_tiny_decode_m1k2048n32000_raw",
]
SMOL = ["JM_smolvla_flow_denoise_m50k32n720_raw", "CH_kv_heads_smolvla_flow_denoise_q15kv5s50l163d64"]


def pin(path):
    path = Path(path).absolute()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return pin(path)


def expert_targets(study):
    """Project expert values and scope only; never export our scores or schedules."""
    document = json.loads(Path(study).read_text())
    rows = []
    for row in document["rows"]:
        if row["reference"] != "Jack":
            continue
        measured = row.get("evidence") == "retained_FireSim_completed_call" and row.get("outputs_exact") is True
        samples = row.get("ordered_samples", {}).get("reference", []) if measured else []
        rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "shape_mnk": row["shape"],
                "output_contract": row["output"],
                "reference": row["reference"],
                "reference_version": row.get("reference_version", "retained older Jack kernel"),
                "engine": "firesim",
                "status": "measured_matched_outputs" if measured else "unmeasured",
                "ordered_cycles": samples,
                "mean_cycles": sum(samples) / len(samples) if samples else None,
                "source_closure": row.get("source_closure", "unknown"),
                "source_receipt_sha256": [record["sha256"] for record in row.get("receipts", [])[:1]],
                "roi": "expert complete call(s) plus required completion; preparation and comparisons outside",
                "cache": "ordered samples in one image; not independent runs or an all-cold qualification",
                "latest_archive_equivalence": False,
            }
        )
    return {
        "schema": "merlin.expert_performance_targets.v1",
        "expert_only": True,
        "study_sha256": pin(study)["sha256"],
        "shape_convention": "descriptive dense strings M,N,K",
        "hardware": document["hardware"],
        "rows": rows,
        "scope": "trusted reference targets, not numerical/performance qualification of new compiler output",
        "unmatched_historical_estimates_excluded": True,
    }


def prepare_reference_review(reference_manifest, output, *, oracle_study, raw_containers=()):
    """Reuse original reviewed origins and add a sanitized expert-target origin."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    original = Path(reference_manifest)
    document = json.loads(original.read_text())
    for origin in document["origins"]:
        relative = Path(origin["container"])
        source = original.parent / relative
        destination = root / relative
        if origin["kind"] == "user_permitted_archive":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.hardlink_to(source)
        else:
            for item in origin["files"]:
                leaf = destination / item["member"]
                leaf.parent.mkdir(parents=True, exist_ok=True)
                leaf.hardlink_to(source / item["member"])
    targets = expert_targets(oracle_study)
    target_file = root / "expert-targets.json"
    write_json(target_file, targets)
    container = root / "containers/expert-targets.zip"
    with zipfile.ZipFile(container, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(target_file, "expert-targets.json")
    document["review_id"] = "kernel-abstractions-common-raw-and-trusted-targets-20261028"
    document["origins"].append(
        {
            "name": "expert_targets",
            "kind": "user_permitted_archive",
            "container": "containers/expert-targets.zip",
            "identity": pin(container)["sha256"],
            "license": "Operator measurement metadata; no candidate implementation, schedule or score.",
            "allowed_purpose": "reference_reading",
            "content_kind": "source_text",
            "user_authorization": "User requested identical trusted expert measurements in both arms, 2026-10-28.",
            "files": [
                {
                    "member": "expert-targets.json",
                    "destination": "expert-targets.json",
                    "sha256": pin(target_file)["sha256"],
                }
            ],
        }
    )
    for i, original_container in enumerate(raw_containers):
        original_container = Path(original_container).absolute()
        canonical = "_".join(original_container.name.split())
        container = root / "containers" / canonical
        container.hardlink_to(original_container)
        identity = pin(original_container)
        document["origins"].append(
            {
                "name": f"supplied_archive_{i}",
                "kind": "user_permitted_archive",
                "container": "containers/" + canonical,
                "identity": identity["sha256"],
                "license": "User-supplied archive; no separate license supplied.",
                "allowed_purpose": "reference_reading",
                "content_kind": "opaque_container",
                "user_authorization": "User requested the same original archives for both arms, 2026-10-28.",
                "files": [
                    {
                        "member": canonical,
                        "destination": canonical,
                        "sha256": identity["sha256"],
                        "size_bytes": identity["bytes"],
                    }
                ],
            }
        )
    write_json(root / "reference-manifest.json", document)
    return root / "reference-manifest.json"


def capacity_boundary(compiler_root):
    """Execute only the frozen planner's geometry/storage construction prefix.

    The actual selected RHS residency planner decides admission. No inferred
    target capacity or manually chosen boundary N enters the study.
    """
    root = Path(compiler_root).absolute()
    path = root / "dense.py"
    spec = importlib.util.spec_from_file_location("study_frozen_dense", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plan_matmul")
    prefix = []
    for node in function.body:
        prefix.append(copy.deepcopy(node))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "storage" for t in node.targets):
            break
    else:
        raise ValueError("frozen planner storage construction unavailable")
    scope = dict(vars(module))
    # Imports/docstrings and assignments above storage are the original body.
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), str(path), "exec"), scope)
    storage = scope["storage"]
    edge, operand_rows, acc_rows = module._geometry()
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency

    kwargs = dict(
        block=edge,
        operand_rows=operand_rows,
        accumulator_rows=acc_rows,
        rhs_base=storage.rhs.base,
        max_static_rhs_tiles=storage.rhs.capacity // edge,
    )
    admitted = []
    for n in range(edge, storage.rhs.capacity * edge + edge, edge):
        if plan_dense_rhs_residency(784, 256, n, **kwargs) is None:
            if not admitted:
                raise ValueError("frozen planner admits no full RHS boundary")
            return {
                "shape_mk": [784, 256],
                "largest_n": admitted[-1],
                "next_tile_n": n,
                "tile": edge,
                "selected_arguments": kwargs,
                "basis": "frozen dense.plan_matmul storage prefix and existing plan_dense_rhs_residency",
                "pins": [
                    pin(path),
                    pin(root / "resident.py"),
                    pin(root / "rtl_facts/facts.json"),
                    pin(Path(importlib.util.find_spec("merlin.compile.scheduling.dense_rhs_residency").origin)),
                ],
            }
        admitted.append(n)
    raise ValueError("frozen RHS boundary not found within selected storage bound")


def resize_dense(cb, *, m, k=None, ns=None):
    cb = copy.deepcopy(cb)
    tensors = cb["tensors"]
    handles = {c["operands"]["dst"]: c["operands"]["src"] for c in cb["commands"] if c["opcode"] == "RES_PACK"}
    products = {}
    mms = [c for c in cb["commands"] if c["opcode"] == "MATMUL_RESIDENT"]
    for i, command in enumerate(mms):
        operands = command["operands"]
        lhs = operands["lhs"]
        rhs = handles[operands["rhs"]]
        old_k, old_n = tensors[rhs]["shape"]
        depth, width = old_k if k is None else k, old_n if ns is None else ns[i]
        tensors[lhs]["shape"] = [m, depth]
        tensors[rhs]["shape"] = [depth, width]
        products[operands["dst"]] = [m, width]
    for command in cb["commands"]:
        if command["opcode"] == "COMMIT":
            operands = command["operands"]
            if operands["dst"] in tensors:
                tensors[operands["dst"]]["shape"] = products[operands["src"]]
    return cb


def resize_conv(cb, *, batch, output, output_channels=None):
    cb = copy.deepcopy(cb)
    command = next(c for c in cb["commands"] if c["opcode"] == "CONV2D")
    attrs, operands = command["attributes"], command["operands"]
    if attrs["stride"] != [1, 1] or attrs["padding"] != [0, 0, 0, 0] or attrs["kernel"][:2] != [3, 3]:
        raise ValueError("study convolution template must be original valid K3 stride1")
    if output_channels is not None:
        old_channels = attrs["kernel"][3]
        if type(output_channels) is not int or not 0 < output_channels <= old_channels:
            raise ValueError("study convolution channel subset must preserve original readout coefficients")
        handles = {c["operands"]["dst"]: c["operands"]["src"] for c in cb["commands"] if c["opcode"] == "RES_PACK"}
        cb["tensors"][handles[operands["weight"]]]["shape"][-1] = output_channels
        if attrs.get("bias"):
            cb["tensors"][attrs["bias"]]["shape"] = [output_channels]
        for key in ("acc_scale_by_column",):
            if key in attrs:
                attrs[key] = attrs[key][:output_channels]
        attrs["kernel"][3] = output_channels
    cb["tensors"][operands["ifm"]]["shape"][:3] = [batch, output + 2, output + 2]
    cb["tensors"][operands["dst"]]["shape"] = [batch * output * output, attrs["kernel"][3]]
    return cb


def resize_heads(cb, *, query, context, pv):
    cb = copy.deepcopy(cb)
    # Existing original interface records each head and actual shared RHS by SSA.
    for row in cb["tensors"].values():
        if row["role"] == "output":
            row["shape"] = [query, 64 if pv else context]
        elif row["role"] == "weight":
            row["shape"] = [context, 64] if pv else [64, context]
        elif row["role"] == "input":
            row["shape"] = [query, context if pv else 64]
        else:
            raise ValueError("unexpected attention template role")
    return cb


def resize_component(program, *, query, context):
    program = copy.deepcopy(program)
    for row in program["inputs"]:
        name = row["name"]
        row["shape"] = (
            [64, context]
            if name.startswith("KT")
            else [context, 64]
            if name.startswith("V")
            else [query, 64]
            if name.startswith("Q")
            else [query, context]
        )
    return render(program, operand_dtype="i8", accumulator_dtype="i32")


def source_contract(cb):
    from merlin.compile.logical_abi import commit_output_shape

    tensors = copy.deepcopy(cb["tensors"])
    handles = {c["operands"]["dst"]: c["operands"]["src"] for c in cb["commands"] if c["opcode"] == "RES_PACK"}
    products = []
    for c in cb["commands"]:
        if c["opcode"] == "MATMUL_RESIDENT":
            a, b = cb["tensors"][c["operands"]["lhs"]], cb["tensors"][handles[c["operands"]["rhs"]]]
            products.append({"shape_mkn": [a["shape"][0], a["shape"][1], b["shape"][1]], **c["operands"]})
        if c["opcode"] == "COMMIT":
            tensors[c["operands"]["dst"]] = {
                "shape": list(commit_output_shape(cb, c["operands"]["src"], c.get("attributes", {}))),
                "dtype": c.get("attributes", {}).get("output_dtype", "i32"),
                "role": "output",
            }
    return {
        "tensors": tensors,
        "products": products,
        "numerics_and_fusion": [
            {"opcode": c["opcode"], "operands": c["operands"], "attributes": c.get("attributes", {})}
            for c in cb["commands"]
            if c["opcode"] in ("COMMIT", "CONV2D")
        ],
        "scope": "all declared outputs, complete invocation",
        "stimuli": "independent deterministic fixture",
        "captured_model_equivalence": False,
    }


def prepare_inputs(
    output,
    *,
    corpus,
    historical_kernels,
    reference_manifest,
    compiler_root,
    oracle_study,
    base_config=None,
    evidence_root=None,
    raw_containers=(),
    profile="full",
):
    from merlin_experiments.corpus.public_references import stage

    if profile not in ("full", "projection-core", "conv-composition"):
        raise ValueError("unknown study input profile")
    if profile != "full" and raw_containers:
        raise ValueError(f"{profile} reuses the complete pinned reference review")
    output, corpus = Path(output).absolute(), Path(corpus).absolute()
    if output.exists():
        raise ValueError("select a new output directory for immutable study inputs")
    output.mkdir(parents=True)
    oracle_source = pin(oracle_study)
    oracle_copy = output / "oracle-study-frozen.json"
    oracle_copy.write_bytes(Path(oracle_study).read_bytes())
    if pin(oracle_copy)["sha256"] != oracle_source["sha256"]:
        raise ValueError("oracle study changed during input freezing")
    oracle_study = oracle_copy
    templates, capsule_templates = {}, {}
    for p in corpus.rglob("capsule.yaml"):
        if "hidden" in p.relative_to(corpus).parts or "private" in p.relative_to(corpus).parts:
            continue
        doc = yaml.safe_load(p.read_text())
        capsule_templates[doc["name"]] = (p, doc)
    cases = []

    def add(name, family, split, text, template, cap=None, contract=None):
        directory = output / ("training" if split == "train" else "heldout") / name
        directory.mkdir(parents=True)
        path = directory / "capsule.interface.mlir"
        path.write_text(text)
        if contract is None:
            contract = source_contract(parse_interface_mlir(text))
        capsule = (
            copy.deepcopy(cap)
            if cap is not None
            else {
                "kind": "layer",
                "label": "public",
                "source_role": "handauthored_compiler_test",
                "inputs": [dict(v, name=k) for k, v in contract["tensors"].items() if v["role"] != "output"],
                "operation": {"op": "matmul", "attributes": {}},
                "expected": {"instruction_classes": [], "modes": {}},
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
                "required_oracle_tiers": ["L0", "L1", "L2"],
            }
        )
        for key in ("software_screen", "integer_partial_sum_bound", "performance_demand", "chain_demand"):
            capsule.pop(key, None)
        capsule.update(
            name=name,
            interface_mlir=path.name,
            label="public",
            source_reference=(
                "Frozen independent study source; original/generated geometry and numerical "
                "contract pinned in study manifest."
            ),
        )
        if "tensors" in contract:
            capsule["inputs"] = [dict(v, name=k) for k, v in contract["tensors"].items() if v["role"] != "output"]
            if contract["products"]:
                first = contract["products"][0]
                source_cb = parse_interface_mlir(text)
                handles = {
                    c["operands"]["dst"]: c["operands"]["src"]
                    for c in source_cb["commands"]
                    if c["opcode"] == "RES_PACK"
                }
                commits = [c for c in source_cb["commands"] if c["opcode"] == "COMMIT"]
                capsule["operation"] = {
                    "op": "matmul",
                    "attributes": {
                        "lhs": first["lhs"],
                        "weight": handles[first["rhs"]],
                        "out": commits[0]["operands"]["dst"],
                        "outs": [c["operands"]["dst"] for c in commits],
                        **commits[0].get("attributes", {}),
                    },
                }
                capsule["numeric_policy"]["dtype"] = commits[0].get("attributes", {}).get("output_dtype", "i32")
            elif profile == "conv-composition":
                source_cb = parse_interface_mlir(text)
                convs = [c for c in source_cb["commands"] if c["opcode"] == "CONV2D"]
                if len(convs) == 1:
                    conv = convs[0]
                    handles = {
                        c["operands"]["dst"]: c["operands"]["src"]
                        for c in source_cb["commands"] if c["opcode"] == "RES_PACK"
                    }
                    attrs, operands = conv["attributes"], conv["operands"]
                    capsule["operation"] = {"op": "conv2d", "attributes": {
                        **copy.deepcopy(attrs), "ifm": operands["ifm"],
                        "weight": handles[operands["weight"]], "out": operands["dst"],
                        "ci": attrs["kernel"][2], "kh": attrs["kernel"][0], "kw": attrs["kernel"][1],
                    }}
                    capsule["numeric_policy"]["dtype"] = attrs["output_dtype"]
        cap_path = directory / "capsule.yaml"
        cap_path.write_text(yaml.safe_dump(capsule, sort_keys=False))
        cases.append(
            {
                "case": name,
                "family": family,
                "split": split,
                "source": pin(path),
                "capsule": pin(cap_path),
                "template": pin(template),
                "contract": contract,
                "status": "source_ready_admission_pending",
                "original_source_bytes": path.read_bytes() == Path(template).read_bytes(),
                "measurement": "unmeasured; no historical number automatically rebound to generated shapes",
            }
        )

    families = [
        ("dense", GEMMS),
        ("convolution", CONVS),
        ("attention", ATTENTION),
        ("projection", TINY),
        ("smolvla", SMOL),
    ]
    if profile == "projection-core":
        families = [("projection", [TINY[2]])]
    elif profile == "conv-composition":
        families = [("convolution", [CONVS[1]])]
    for family, names in families:
        for name in names:
            cap_path, cap = capsule_templates.get(name, (None, None))
            source = Path(historical_kernels) / name / "input.mlir"
            if not source.is_file() and cap_path is not None:
                source = cap_path.parent / cap["interface_mlir"]
            text = source.read_text()
            if name == SMOL[1]:
                program = cap["operation"]["attributes"]["program"]
                normalized, _ = render(program, operand_dtype="i8", accumulator_dtype="i32")
                contract = {
                    "component_program": normalized,
                    "scope": "30 original rawi32 outputs; independent P, no softmax or full model equivalence",
                }
                templates[name] = (source, program, cap)
                add(name, family, "train", text, source, cap, contract)
            else:
                cb = parse_interface_mlir(text)
                if profile == "projection-core":
                    contract = source_contract(cb)
                    commits = contract["numerics_and_fusion"]
                    if (
                        len(contract["products"]) != 1
                        or contract["products"][0]["shape_mkn"] != [16, 2048, 2048]
                        or len(commits) != 1
                        or commits[0]["opcode"] != "COMMIT"
                        or commits[0]["attributes"] != {"epilogue": [], "output_dtype": "i32"}
                        or any(t["dtype"] != "i8" for t in cb["tensors"].values() if t["role"] != "output")
                    ):
                        raise ValueError("projection-core requires the raw-i32 projection anchor")
                elif profile == "conv-composition":
                    convs = [c for c in cb["commands"] if c["opcode"] == "CONV2D"]
                    if len(convs) != 1:
                        raise ValueError("conv-composition requires one NHWC bias INT8 convolution anchor")
                    conv = convs[0]
                    attrs, operands = conv["attributes"], conv["operands"]
                    if (cb["tensors"][operands["ifm"]]["shape"] != [4, 30, 30, 128]
                        or attrs.get("kernel") != [3, 3, 128, 128]
                        or attrs.get("stride") != [1, 1]
                        or attrs.get("padding") != [0, 0, 0, 0]
                        or attrs.get("dilation") != [1, 1]
                        or attrs.get("layout") != "nhwc"
                        or attrs.get("output_dtype") != "i8"
                        or attrs.get("epilogue") != ["bias_add"]
                        or not attrs.get("bias")
                        or cb["tensors"][attrs["bias"]] != {"shape": [128], "dtype": "i32", "role": "bias"}
                        or cb["tensors"][operands["ifm"]]["dtype"] != "i8"):
                        raise ValueError("conv-composition requires the original valid K3 NHWC bias INT8 anchor")
                templates[name] = (source, cb, cap)
                add(name, family, "train", text, source, cap)
    boundary = None
    counts = {"train": 1, "heldout": 3} if profile != "full" else {"train": 18, "heldout": 30}
    if profile == "projection-core":
        source, cb, cap = templates[TINY[2]]
        for m, k, n in [(32, 2048, 2048), (16, 1024, 2048), (16, 2048, 1024)]:
            add(
                f"holdout_projection_m{m}k{k}n{n}",
                "projection",
                "heldout",
                emit_interface_mlir(resize_dense(cb, m=m, k=k, ns=[n])),
                source,
                cap,
            )
    elif profile == "conv-composition":
        source, cb, cap = templates[CONVS[1]]
        for batch, spatial, channels in [(8, 28, 128), (4, 32, 128), (4, 28, 64)]:
            add(
                f"holdout_conv_b{batch}_out{spatial}_ci128_co{channels}",
                "convolution",
                "heldout",
                emit_interface_mlir(resize_conv(cb, batch=batch, output=spatial, output_channels=channels)),
                source,
                cap,
            )
    else:
        boundary = capacity_boundary(compiler_root)
        dense_shapes = [
            [(511, 512, 512), (513, 512, 512)],
            [(6272, 256, 64), (12544, 257, 64)],
            [(25088, 64, 256), (12544, 64, 255)],
            [(1568, 512, 128), (3136, 512, 129)],
            [(6272, 128, 512), (3136, 127, 512)],
            [(784, 256, boundary["largest_n"]), (784, 256, boundary["next_tile_n"])],
        ]
        for name, shapes in zip(GEMMS, dense_shapes, strict=True):
            source, cb, cap = templates[name]
            for m, k, n in shapes:
                add(
                    f"holdout_dense_m{m}k{k}n{n}",
                    "dense",
                    "heldout",
                    emit_interface_mlir(resize_dense(cb, m=m, k=k, ns=[n])),
                    source,
                    cap,
                )
        for name, h in zip(CONVS, [56, 28, 14], strict=True):
            source, cb, cap = templates[name]
            for b, ho in [(2, h), (4, h + 1)]:
                add(
                    f"holdout_conv_b{b}_out{ho}_c{cb['tensors']['IFM']['shape'][-1]}",
                    "convolution",
                    "heldout",
                    emit_interface_mlir(resize_conv(cb, batch=b, output=ho)),
                    source,
                    cap,
                )
        for name in ATTENTION:
            source, cb, cap = templates[name]
            pv = name == ATTENTION[1]
            for query, context in [(1, 16), (17, 19)]:
                add(
                    f"holdout_attention_{'pv' if pv else 'qk'}_s{query}l{context}",
                    "attention",
                    "heldout",
                    emit_interface_mlir(resize_heads(cb, query=query, context=context, pv=pv)),
                    source,
                    cap,
                )
        for name, m in zip(TINY, [1, 17, 2, 1, 2], strict=True):
            source, cb, cap = templates[name]
            add(
                f"holdout_{name}_m{m}", "projection", "heldout", emit_interface_mlir(resize_dense(cb, m=m)), source, cap
            )
        source, cb, cap = templates[SMOL[0]]
        for m in (49, 51):
            add(
                f"holdout_smol_projection_m{m}k32n720",
                "smolvla",
                "heldout",
                emit_interface_mlir(resize_dense(cb, m=m)),
                source,
                cap,
            )
        source, program, cap = templates[SMOL[1]]
        for query, context in [(50, 165)]:
            normalized, text = resize_component(program, query=query, context=context)
            owncap = copy.deepcopy(cap)
            owncap["operation"]["attributes"]["program"] = {**program, "inputs": normalized["inputs"]}
            owncap["inputs"] = normalized["inputs"]
            owncap["component_program"] = normalized
            add(
                f"holdout_smol_heads_s{query}l{context}",
                "smolvla",
                "heldout",
                text,
                source,
                owncap,
                {
                    "component_program": normalized,
                    "scope": "all30 rawi32 outputs; independent P; no softmax/full model equivalence",
                },
            )
    if (
        len(cases) != sum(counts.values())
        or len({c["case"] for c in cases}) != len(cases)
        or any(sum(c["split"] == split for c in cases) != count for split, count in counts.items())
    ):
        raise ValueError("study sources differ from the selected profile")
    reviewed = (
        Path(reference_manifest)
        if profile != "full"
        else prepare_reference_review(
            reference_manifest, output / "reference-review", oracle_study=oracle_study, raw_containers=raw_containers
        )
    )
    references = stage(reviewed, output / "reference-kernels")
    document = {
        "schema": "private.kernel_abstractions.inputs.v1",
        "cases": cases,
        "counts": counts,
        "profile": profile,
        "training_root": str(output / "training"),
        "heldout_root": str(output / "heldout"),
        "reference_root": str(output / "reference-kernels"),
        "reference_manifest": pin(reviewed),
        "reference_source_manifest": pin(reference_manifest),
        "reference_bank": references,
        "capacity_boundary": boundary,
        "oracle_study": pin(oracle_study),
        "oracle_study_source": oracle_source,
        "early_slot": {
            "case": (CONVS[1] if profile == "conv-composition"
                     else TINY[2] if profile == "projection-core" else TINY[-1]),
            "split": "train",
        },
        "treatment": {
            "raw_and_cca_share_identical_reference_bytes": True,
            "off": "neutral index, no derived CCA map",
            "on": "same raw bank plus existing source/lifting/CCA tooling",
        },
        "admission": "source generation only; executable numerical admission remains required",
    }
    for split, folder in [("train", "training"), ("heldout", "heldout")]:
        write_json(
            output / folder / "selection.json",
            {
                "schema": "private.kernel_abstractions.selection.v1",
                "split": split,
                "cases": [r["case"] for r in cases if r["split"] == split],
            },
        )
    if base_config is not None and evidence_root is not None:
        document["suites"] = prepare_suites(
            document, output=output, base_config=base_config, evidence_root=evidence_root
        )
    write_json(output / "manifest.json", document)
    if profile != "full":
        write_json(
            output / "small-panel.json",
            {
                "schema": "private.kernel_abstractions.panel.v1",
                "inputs_manifest": pin(output / "manifest.json"),
                "training": [c for c in cases if c["split"] == "train"],
                "heldout": [c for c in cases if c["split"] == "heldout"],
                "selection": {
                    key: [c["case"] for c in cases if c["split"] == split]
                    for key, split in [("training_case_ids", "train"), ("heldout_case_ids", "heldout")]
                },
                "suites": document.get("suites", {}),
                "separation": "only train suite is an author input; heldout suite stays operator-only until freeze",
                "numerical_authority": "source admission only; executable numerics and timing unmeasured",
            },
        )
    return document


def prepare_suites(document, *, output, base_config, evidence_root):
    """Use ordinary intake, publication admission, staging and descriptor binding.

    The held-out suite is a separate operator-owned evaluation root. Its public
    contract label is required by the existing suite reader, not an author grant.
    """
    from merlin_experiments.corpus import public_scalability, scalability_intake
    from merlin_experiments.phase0.evidence import load_exported_evidence
    from merlin_experiments.runner import fingerprint

    from merlin.targetgen.target_experiment import load_target_experiment

    config = json.loads(Path(base_config).read_text())
    original_descriptor = Path(config["descriptor"])
    te = load_target_experiment(original_descriptor)
    evidence_root = Path(evidence_root)
    evidence = load_exported_evidence(evidence_root)
    suites = {}
    for split in ("train", "heldout"):
        sources = [Path(row["capsule"]["path"]).parent for row in document["cases"] if row["split"] == split]
        destination = Path(output) / (split + "-intake")
        selection = scalability_intake.prepare(
            sources, destination, target=te.target, evidence=evidence, evidence_root=evidence_root
        )
        staged = public_scalability.stage(selection, Path(output) / (split + "-suite"), te=te)
        experiment = Path(output) / (split + "-experiment")
        shutil.copytree(original_descriptor.parent, experiment)
        descriptor = experiment / original_descriptor.name
        content = yaml.safe_load(descriptor.read_text())
        content["workload_spec"][public_scalability.KEY] = staged
        descriptor.chmod(0o600)
        descriptor.write_text(yaml.safe_dump(content, sort_keys=False))
        selected = load_target_experiment(descriptor)
        public_scalability.selected(selected)
        suites[split] = {
            "root": staged,
            "selection": pin(selection),
            "descriptor": pin(descriptor),
            "operator_only": split == "heldout",
            "fingerprint": fingerprint(Path(staged["path"])),
        }
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "corpus", "historical-kernels", "reference-manifest", "compiler-root", "oracle-study"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--profile", choices=("full", "projection-core", "conv-composition"), default="full")
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--raw-containers", action="append", type=Path, default=[])
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        doc = prepare_inputs(output, **args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)[-1400:]) from None
    print(
        json.dumps(
            {
                "manifest": pin(output / "manifest.json"),
                "counts": doc["counts"],
                "capacity_boundary": doc["capacity_boundary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
