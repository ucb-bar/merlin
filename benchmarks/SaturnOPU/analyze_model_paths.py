#!/usr/bin/env python3
"""Analyze Saturn OPU model compute decomposition from real compile artifacts.

The analyzer intentionally works from the same artifacts used for the paper:

  build/compiled_models/<model>/<target>_<hw>_<basename>/
    sources/*.mlir   - per-dispatch source MLIR
    configs/*.mlir   - configured dispatch MLIR
    files/*.ll, *.s  - linked LLVM IR and final RISC-V assembly
    phases/*.mlir    - whole-module phase dumps, when refreshed

For every registered model we compile through tools/merlin.py with
--dump-artifacts, --dump-phases, and --dump-graph unless complete artifacts
already exist. A missing or failed model is a hard error by default because the
paper metrics should not silently omit a workload.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
MERLIN_ROOT = BENCH_DIR.parent.parent
DEFAULT_ARTIFACT_ROOT = MERLIN_ROOT / "build" / "compiled_models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    mlir_rel: str
    target: str
    hw: str
    output_rel: str | None = None
    reuse_imported_mlir: bool = False


MODELS: list[ModelSpec] = [
    ModelSpec("mlp", "MLP", "models/mlp/mlp.q.int8.mlir", "saturn_opu", "OPU"),
    ModelSpec("mlp_wide", "MLP-Wide", "models/mlp_wide/mlp_wide.q.int8.mlir", "saturn_opu", "OPU"),
    ModelSpec(
        "opu_bench_large_mlp",
        "Large MLP",
        "models/opu_bench_suite/opu_bench_large_mlp.q.int8.mlir",
        "saturn_opu",
        "OPU",
    ),
    ModelSpec(
        "opu_bench_vit_small",
        "ViT-Small",
        "models/opu_bench_suite/opu_bench_vit_small.q.int8.mlir",
        "saturn_opu",
        "OPU",
    ),
    ModelSpec(
        "opu_bench_vit",
        "ViT",
        "models/opu_bench_suite/opu_bench_vit.q.int8.mlir",
        "saturn_opu",
        "OPU_LLM",
    ),
    ModelSpec(
        "opu_bench_hybrid",
        "Hybrid CNN+Trf",
        "models/opu_bench_suite/opu_bench_hybrid.q.int8.mlir",
        "saturn_opu",
        "OPU_LLM",
    ),
    ModelSpec(
        "opu_bench_convnet",
        "ConvNet",
        "models/opu_bench_suite/opu_bench_convnet.q.int8.mlir",
        "saturn_opu",
        "OPU_IM2COL",
    ),
    ModelSpec("dronet", "DroNet", "models/dronet/dronet.q.int8.mlir", "saturn_opu", "OPU_IM2COL"),
    ModelSpec(
        "yolov8_nano",
        "YOLOv8-n",
        "build/compiled_models/yolov8_nano/saturn_opu_OPU_IM2COL_yolov8n.q.int8/yolov8n.q.int8.mlir",
        "saturn_opu",
        "OPU_IM2COL",
        "yolov8_nano/saturn_opu_OPU_IM2COL_yolov8n.q.int8",
        True,
    ),
    ModelSpec(
        "tinyllama",
        "TinyLlama",
        "build/compiled_models/tinyllama/saturn_opu_OPU_LLM_tinyllama.q.int8/tinyllama.q.int8.mlir",
        "saturn_opu",
        "OPU_LLM",
        "tinyllama/saturn_opu_OPU_LLM_tinyllama.q.int8",
        True,
    ),
]


@dataclass
class DispatchRecord:
    model_key: str
    model: str
    idx: int
    symbol: str
    source_file: str
    include_in_model: bool
    op_kind: str
    segment: str
    opu_path: str
    ops: int
    opu_ops: int
    compute_pct: float
    layer_id: str
    shape: str
    m: int | str
    n: int | str
    k: int | str
    b: int | str
    vopacc: int
    opmvinbcast: int
    opu_fetch: int
    rvv_reduction: int
    rvv_sqrt: int
    rvv_gather: int
    tile_m: int | str
    tile_n: int | str
    tile_k: int | str
    tiling_tier: str
    evidence: str


DEFINE_RE = re.compile(r"^define\s+(?:internal\s+)?i32\s+@(?:\"([^\"]+)\"|([^\s(]+))\(", re.MULTILINE)
ASM_TYPE_RE = re.compile(r"^\s*\.type\s+([^,\s]+),@function", re.MULTILINE)
EXPORT_RE = re.compile(r"hal\.executable\.export\s+public\s+@([^\s(]+)")
ASYNC_RE = re.compile(r"\$async_dispatch_(\d+)_(.+)$")
MATMUL_RE = re.compile(r"^matmul(?:_like)?_(\d+)x(\d+)x(\d+)")
BATCH_MATMUL_RE = re.compile(r"^batch_matmul_(\d+)x(\d+)x(\d+)x(\d+)")
MATVEC_RE = re.compile(r"^matvec(?:_like)?_(\d+)x(\d+)")
TENSOR_RE = re.compile(r"tensor<((?:\d+x)+\d+)x[if]\d+")
ITERATION_RE = re.compile(r"iteration_sizes = \[([0-9,\s]+)\]")
DIM_RUN_RE = re.compile(r"(?:^|_)(\d+(?:x\d+)*)(?=(?:x[if]\d+|_[if]\d+|_|$))")
RVV_REDUCTION_RE = re.compile(r"\bv(?:fw)?red[a-z0-9]*\.vs\b|\bvfred[a-z0-9]*\.vs\b")
RVV_GATHER_RE = re.compile(r"\bvrgather(?:ei16)?\.(?:vi|vv)\b")
OPU_VOPACC = ".insn r 87, 2, 81"
OPU_BCAST = ".insn r 87, 6, 89"
OPU_FETCH = ".insn r 87, 6, 93"
UKERNEL_CALL_RE = re.compile(r"call\s+i32\s+@iree_uk_(?:opu_matmul_qdq|opu_matmul|mmt4d)\((.*?)\)", re.DOTALL)
I32_CONST_RE = re.compile(r"\bi32\s+(-?\d+)\b")


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def dims_to_str(dims: list[int]) -> str:
    return "x".join(str(v) for v in dims)


def repo_path(rel: str) -> Path:
    return (MERLIN_ROOT / rel).resolve()


def display_model_keys() -> str:
    return ", ".join(spec.key for spec in MODELS)


def model_output_dir(spec: ModelSpec, artifact_root: Path) -> Path:
    if spec.output_rel:
        output_rel = Path(spec.output_rel)
        if artifact_root == DEFAULT_ARTIFACT_ROOT:
            return artifact_root / output_rel
        return artifact_root / output_rel

    input_path = repo_path(spec.mlir_rel)
    model_name = input_path.parent.name
    basename = input_path.name.removesuffix(".mlir").removesuffix(".onnx")
    hw_suffix = f"_{spec.hw}" if spec.hw else ""
    return artifact_root / model_name / f"{spec.target}{hw_suffix}_{basename}"


def source_path_for(spec: ModelSpec) -> Path:
    path = repo_path(spec.mlir_rel)
    if not path.exists():
        raise FileNotFoundError(f"missing MLIR for {spec.key}: {path}")
    return path


def command_prefix() -> list[str]:
    if os.environ.get("CONDA_DEFAULT_ENV") == "merlin-dev":
        return ["uv", "run"]
    if shutil.which("conda"):
        return ["conda", "run", "-n", "merlin-dev", "uv", "run"]
    return ["uv", "run"]


def artifact_complete(out_dir: Path) -> bool:
    return (
        (out_dir / "sources").is_dir()
        and (out_dir / "files").is_dir()
        and any((out_dir / "files").glob("*.codegen.ll"))
        and any((out_dir / "files").glob("*.s"))
        and (out_dir / "phases").is_dir()
        and any((out_dir / "phases").glob("*.1.input.mlir"))
    )


def compile_model(spec: ModelSpec, artifact_root: Path, refresh: bool) -> Path:
    out_dir = model_output_dir(spec, artifact_root)
    if not refresh and artifact_complete(out_dir):
        return out_dir

    source = source_path_for(spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *command_prefix(),
        "tools/merlin.py",
        "compile",
        str(source),
        "--target",
        spec.target,
        "--hw",
        spec.hw,
        "--dump-artifacts",
        "--dump-phases",
        "--dump-graph",
        "--output-dir",
        str(out_dir),
        "--build-dir",
        "host-merlin-release",
    ]
    if spec.reuse_imported_mlir:
        cmd.append("--reuse-imported-mlir")

    env = os.environ.copy()
    env.setdefault(
        "RISCV_TOOLCHAIN_ROOT",
        str(MERLIN_ROOT / "build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV"),
    )
    env.setdefault("RISCV", env["RISCV_TOOLCHAIN_ROOT"])
    env.setdefault("CHIPYARD_ROOT", "/scratch2/agustin/chipyard")

    print(f"  compiling {spec.key} -> {out_dir}", flush=True)
    res = subprocess.run(
        cmd,
        cwd=MERLIN_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.returncode != 0:
        sys.stdout.write(res.stdout)
        raise RuntimeError(f"compile failed for {spec.key}")
    if not artifact_complete(out_dir):
        raise RuntimeError(f"compile for {spec.key} did not produce complete artifacts under {out_dir}")
    return out_dir


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def first_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return matches[0]


def split_codegen_functions(text: str) -> dict[str, str]:
    matches = list(DEFINE_RE.finditer(text))
    out: dict[str, str] = {}
    for match in matches:
        name = match.group(1) or match.group(2)
        if not name or "$async_dispatch_" not in name:
            continue
        end = text.find("\n}", match.end())
        if end == -1:
            end = len(text)
        out[name] = text[match.end() : end]
    return out


def split_asm_functions(text: str) -> dict[str, str]:
    matches = list(ASM_TYPE_RE.finditer(text))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group(1)
        if "$async_dispatch_" not in name and not name.startswith("_encoding"):
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[name] = text[match.end() : end]
    return out


def export_symbol(source_text: str, path: Path) -> str:
    match = EXPORT_RE.search(source_text)
    if match:
        return match.group(1)
    stem = path.stem
    if stem.startswith("module_"):
        return stem.removeprefix("module_")
    if stem.startswith("configured_module_"):
        return stem.removeprefix("configured_module_")
    return stem


def parse_async(symbol: str) -> tuple[int, str]:
    match = ASYNC_RE.search(symbol)
    if match:
        return int(match.group(1)), match.group(2)
    return -1, symbol


def dims_from_tail(tail: str) -> list[int]:
    for match in DIM_RUN_RE.finditer(tail):
        raw = match.group(1)
        if raw:
            return [int(part) for part in raw.split("x")]
    return []


def tensor_dims(text: str) -> list[list[int]]:
    dims: list[list[int]] = []
    for match in TENSOR_RE.finditer(text):
        shape = [int(part) for part in match.group(1).split("x")]
        dims.append(shape)
    return dims


def largest_tensor_elements(text: str) -> tuple[int, str]:
    shapes = tensor_dims(text)
    if not shapes:
        return 0, ""
    shape = max(shapes, key=product)
    return product(shape), dims_to_str(shape)


def iteration_sizes(text: str) -> tuple[int, int, int] | None:
    match = ITERATION_RE.search(text)
    if not match:
        return None
    values = [int(part.strip()) for part in match.group(1).split(",") if part.strip()]
    if len(values) >= 3:
        return values[0], values[1], values[2]
    return None


def mlir_op_names(source_text: str) -> set[str]:
    """Use MLIR bindings on small source files when available.

    The fallback keeps the analyzer usable if bindings are not installed or a
    dialect parse fails. TinyLlama source files are small enough; huge whole
    module input files are intentionally not parsed here.
    """
    if len(source_text) > 1_000_000:
        return set()
    try:
        import iree.compiler.ir as ir  # type: ignore
    except Exception:
        return set()

    try:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(source_text, context=ctx)
    except Exception:
        return set()

    names: set[str] = set()

    def walk(op) -> None:
        names.add(op.name)
        for region in op.regions:
            for block in region:
                for child in block:
                    walk(child)

    walk(module.operation)
    return names


def classify_generic_body(source_text: str) -> str:
    if "iterator_types" in source_text and "reduction" in source_text:
        if "math.exp" in source_text or "maxnumf" in source_text or "maximumf" in source_text:
            return "softmax_or_reduction"
        if "arith.add" in source_text:
            return "reduction"
    if "tensor.collapse_shape" in source_text and "iree_encoding.set_encoding" in source_text:
        return "im2col_or_pack"
    if "linalg.conv" in source_text:
        return "conv"
    return "elementwise"


def classify_ops(symbol: str, tail: str, source_text: str) -> tuple[str, int, str, dict[str, int | str]]:
    """Return op_kind, analytical ops, shape string, parsed dims."""
    dims: dict[str, int | str] = {"M": "", "N": "", "K": "", "B": ""}
    lowered = tail.lower()

    if match := BATCH_MATMUL_RE.match(lowered):
        b, m, n, k = (int(v) for v in match.groups())
        dims.update({"B": b, "M": m, "N": n, "K": k})
        return "batch_matmul", 2 * b * m * n * k, f"{b}x{m}x{n}x{k}", dims

    if match := MATMUL_RE.match(lowered):
        m, n, k = (int(v) for v in match.groups())
        dims.update({"M": m, "N": n, "K": k})
        return "matmul", 2 * m * n * k, f"{m}x{n}x{k}", dims

    if match := MATVEC_RE.match(lowered):
        n, k = (int(v) for v in match.groups())
        dims.update({"M": 1, "N": n, "K": k})
        return "matvec", 2 * n * k, f"1x{n}x{k}", dims

    if lowered.startswith("softmax"):
        shape = dims_from_tail(lowered)
        if shape:
            reduce_axis = shape[-1]
            outer = product(shape[:-1]) if len(shape) > 1 else 1
            return "softmax", 5 * reduce_axis * outer, dims_to_str(shape), dims

    if lowered.startswith("reduction"):
        shape = dims_from_tail(lowered)
        if shape:
            reduce_axis = shape[-1]
            outer = product(shape[:-1]) if len(shape) > 1 else 1
            return "reduction", max(reduce_axis - 1, 0) * outer, dims_to_str(shape), dims

    if lowered.startswith("elementwise"):
        shape = dims_from_tail(lowered)
        if shape:
            return "elementwise", product(shape), dims_to_str(shape), dims

    if lowered.startswith("slow_memcpy") or "memcpy" in lowered:
        elems, shape = largest_tensor_elements(source_text)
        return "data_movement", elems, shape, dims

    if lowered.startswith("encode") or symbol.startswith("_encoding"):
        elems, shape = largest_tensor_elements(source_text)
        return "data_movement", elems, shape, dims

    if lowered.startswith("conv") or "linalg.conv" in source_text:
        # Prefer explicit conv tensor shapes when present. For conv output O
        # and kernel K, use 2 * O * K per MAC-like contribution.
        shapes = tensor_dims(source_text)
        if len(shapes) >= 2:
            output = max(shapes, key=product)
            kernel = min((s for s in shapes if len(s) >= 4), key=product, default=[])
            kernel_work = product(kernel[-2:]) if kernel else 1
            return "conv", 2 * product(output) * kernel_work, dims_to_str(output), dims
        shape = dims_from_tail(lowered)
        return "conv", 2 * product(shape) if shape else 0, dims_to_str(shape), dims

    if lowered.startswith("generic"):
        generic_class = classify_generic_body(source_text)
        if generic_class == "conv":
            return classify_ops(symbol, "conv", source_text)
        elems, shape = largest_tensor_elements(source_text)
        if generic_class == "reduction" or generic_class == "softmax_or_reduction":
            # Generic reductions usually have one larger input tensor and one
            # smaller output tensor. The delta is the reduction work.
            shapes = sorted(tensor_dims(source_text), key=product, reverse=True)
            if len(shapes) >= 2:
                return "reduction", max(product(shapes[0]) - product(shapes[1]), 0), dims_to_str(shapes[0]), dims
        if generic_class == "im2col_or_pack":
            return "data_movement", elems, shape, dims
        return "elementwise", elems, shape, dims

    if "linalg.matmul" in source_text or "linalg.mmt4d" in source_text:
        sizes = iteration_sizes(source_text)
        if sizes:
            m, n, k = sizes
            dims.update({"M": m, "N": n, "K": k})
            return "matmul", 2 * m * n * k, f"{m}x{n}x{k}", dims

    elems, shape = largest_tensor_elements(source_text)
    return "other", elems, shape, dims


def function_for_idx(functions: dict[str, str], symbol: str, idx: int) -> str:
    if symbol in functions:
        return functions[symbol]
    if idx >= 0:
        needle = f"$async_dispatch_{idx}_"
        for name, body in functions.items():
            if needle in name:
                return body
    return ""


def opcode_counts(asm_body: str) -> Counter[str]:
    return Counter(
        {
            "vopacc": asm_body.count(OPU_VOPACC),
            "opmvinbcast": asm_body.count(OPU_BCAST),
            "opu_fetch": asm_body.count(OPU_FETCH),
            "rvv_reduction": len(RVV_REDUCTION_RE.findall(asm_body)),
            "rvv_sqrt": asm_body.count("vfsqrt.v"),
            "rvv_gather": len(RVV_GATHER_RE.findall(asm_body)),
            "vfslide1down": asm_body.count("vfslide1down.vf"),
        }
    )


def parse_ukernel_tile(codegen_body: str) -> tuple[int, int, int] | None:
    match = UKERNEL_CALL_RE.search(codegen_body)
    if not match:
        return None
    i32_consts = [int(value) for value in I32_CONST_RE.findall(match.group(1))]
    if len(i32_consts) < 4:
        return None
    # Last four i32 arguments are M0, N0, K0, flags.
    return i32_consts[-4], i32_consts[-3], i32_consts[-2]


def int_dim(value: int | str) -> int | None:
    return value if isinstance(value, int) else None


def classify_opu_segment(
    opu_path: str,
    dims: dict[str, int | str],
    tile: tuple[int, int, int] | None,
) -> tuple[str, str]:
    if opu_path not in {"encoding_resolver", "runtime_mmt4d_opu", "fused_qdq", "inline_vopacc"}:
        return opu_path, "none"
    if opu_path == "fused_qdq":
        return "fused_qdq", "fused"
    if opu_path == "inline_vopacc":
        return "inline_vopacc", "inline"

    m = int_dim(dims["M"])
    n = int_dim(dims["N"])
    tile_m, tile_n, tile_k = tile or (0, 0, 0)
    narrow_shape = (
        (m is not None and tile_m > 0 and m < tile_m)
        or (n is not None and tile_n > 0 and n < tile_n)
        or tile_m == 1
        or tile_n == 1
    )

    if opu_path == "encoding_resolver":
        if narrow_shape:
            return "encoding_narrow_tile", f"{tile_m}x{tile_n}x{tile_k}"
        if tile_m >= 32 and tile_n >= 32:
            return "encoding_32x32_tile", f"{tile_m}x{tile_n}x{tile_k}"
        if tile_m >= 16 and tile_n >= 16:
            return "encoding_16x16_tile", f"{tile_m}x{tile_n}x{tile_k}"
        return "encoding_other_tile", f"{tile_m}x{tile_n}x{tile_k}"

    if narrow_shape:
        return "runtime_narrow_tile", f"{tile_m}x{tile_n}x{tile_k}"
    if tile_m >= 32 and tile_n >= 32:
        return "runtime_32x32_tile", f"{tile_m}x{tile_n}x{tile_k}"
    if tile_m >= 16 and tile_n >= 16:
        return "runtime_16x16_tile", f"{tile_m}x{tile_n}x{tile_k}"
    if tile_m == 8 and tile_n == 8:
        return "runtime_8x8_tile", f"{tile_m}x{tile_n}x{tile_k}"
    return "runtime_other_tile", f"{tile_m}x{tile_n}x{tile_k}"


def classify_path(
    op_kind: str,
    codegen_body: str,
    asm_body: str,
) -> tuple[str, str, Counter[str], tuple[int, int, int] | None]:
    counts = opcode_counts(asm_body)
    tile = parse_ukernel_tile(codegen_body)
    has_qdq = "iree_uk_opu_matmul_qdq" in codegen_body
    has_opu_matmul = "iree_uk_opu_matmul" in codegen_body
    has_mmt4d = "iree_uk_mmt4d" in codegen_body
    has_vopacc = counts["vopacc"] > 0

    evidence_parts: list[str] = []
    if has_qdq:
        evidence_parts.append("llvm:iree_uk_opu_matmul_qdq")
    if has_opu_matmul:
        evidence_parts.append("llvm:iree_uk_opu_matmul")
    if has_mmt4d:
        evidence_parts.append("llvm:iree_uk_mmt4d")
    if has_vopacc:
        evidence_parts.append(f"asm:vopacc={counts['vopacc']}")
    if tile:
        evidence_parts.append(f"tile={tile[0]}x{tile[1]}x{tile[2]}")
    if counts["rvv_reduction"]:
        evidence_parts.append(f"asm:rvv_reduction={counts['rvv_reduction']}")
    if counts["rvv_sqrt"]:
        evidence_parts.append(f"asm:vfsqrt={counts['rvv_sqrt']}")
    if counts["rvv_gather"]:
        evidence_parts.append(f"asm:vrgather={counts['rvv_gather']}")

    if has_qdq and has_vopacc:
        return "fused_qdq", ";".join(evidence_parts), counts, tile
    if has_opu_matmul and has_vopacc:
        return "encoding_resolver", ";".join(evidence_parts), counts, tile
    if has_mmt4d and has_vopacc:
        return "runtime_mmt4d_opu", ";".join(evidence_parts), counts, tile
    if has_vopacc:
        return "inline_vopacc", ";".join(evidence_parts), counts, tile
    if op_kind in {"matmul", "batch_matmul", "matvec"}:
        return "rvv_matmul", ";".join(evidence_parts), counts, tile
    if op_kind == "conv":
        return "direct_conv", ";".join(evidence_parts), counts, tile
    if op_kind in {"reduction", "softmax"}:
        return "rvv_reduction_softmax_norm", ";".join(evidence_parts), counts, tile
    if op_kind == "data_movement":
        return "data_movement", ";".join(evidence_parts), counts, tile
    return "elementwise_other", ";".join(evidence_parts), counts, tile


def infer_layer_id(model_key: str, idx: int, op_kind: str, tail: str) -> str:
    if idx < 0:
        return "auxiliary"
    if model_key in {"tinyllama", "opu_bench_vit", "opu_bench_vit_small", "opu_bench_hybrid"}:
        if "softmax" in tail:
            family = "attention_softmax"
        elif op_kind in {"batch_matmul", "matmul", "matvec"}:
            family = "projection_or_ffn"
        elif op_kind in {"reduction", "softmax"}:
            family = "norm_or_softmax"
        else:
            family = op_kind
        return f"{family}:dispatch_{idx}"
    if model_key in {"dronet", "yolov8_nano", "opu_bench_convnet"}:
        if op_kind in {"matmul", "batch_matmul"}:
            return f"conv_im2col_matmul:dispatch_{idx}"
        return f"{op_kind}:dispatch_{idx}"
    return f"{op_kind}:dispatch_{idx}"


def source_files(out_dir: Path) -> list[Path]:
    paths = sorted((out_dir / "sources").glob("*.mlir"))
    if not paths:
        raise FileNotFoundError(f"no source MLIR files under {out_dir / 'sources'}")
    return paths


def analyze_model(spec: ModelSpec, artifact_root: Path, refresh: bool) -> tuple[list[DispatchRecord], Path]:
    out_dir = compile_model(spec, artifact_root, refresh)
    codegen_text = read_text(first_file(out_dir / "files", "*.codegen.ll"))
    asm_text = read_text(first_file(out_dir / "files", "*.s"))
    codegen_functions = split_codegen_functions(codegen_text)
    asm_functions = split_asm_functions(asm_text)

    records: list[DispatchRecord] = []
    for source in source_files(out_dir):
        source_text = read_text(source)
        symbol = export_symbol(source_text, source)
        idx, tail = parse_async(symbol)
        include_in_model = "$async_dispatch_" in symbol and "_initializer_" not in symbol
        op_names = mlir_op_names(source_text)
        if op_names and "linalg.softmax" in op_names and not tail.startswith("softmax"):
            tail = f"softmax_{tail}"

        op_kind, ops, shape, dims = classify_ops(symbol, tail, source_text)
        codegen_body = function_for_idx(codegen_functions, symbol, idx)
        asm_body = function_for_idx(asm_functions, symbol, idx)
        opu_path, evidence, counts, tile = classify_path(op_kind, codegen_body, asm_body)
        segment, tiling_tier = classify_opu_segment(opu_path, dims, tile)
        is_opu = opu_path in {"encoding_resolver", "runtime_mmt4d_opu", "fused_qdq", "inline_vopacc"}
        opu_ops = ops if is_opu else 0
        tile_m, tile_n, tile_k = tile or ("", "", "")
        if not include_in_model:
            ops = 0
            opu_ops = 0
            segment = "data_movement"
            tiling_tier = "none"
            tile_m, tile_n, tile_k = "", "", ""

        records.append(
            DispatchRecord(
                model_key=spec.key,
                model=spec.display_name,
                idx=idx,
                symbol=symbol,
                source_file=str(source.relative_to(MERLIN_ROOT)),
                include_in_model=include_in_model,
                op_kind=op_kind,
                segment=segment,
                opu_path=opu_path,
                ops=ops,
                opu_ops=opu_ops,
                compute_pct=0.0,
                layer_id=infer_layer_id(spec.key, idx, op_kind, tail),
                shape=shape,
                m=dims["M"],
                n=dims["N"],
                k=dims["K"],
                b=dims["B"],
                vopacc=counts["vopacc"],
                opmvinbcast=counts["opmvinbcast"],
                opu_fetch=counts["opu_fetch"],
                rvv_reduction=counts["rvv_reduction"] + counts["vfslide1down"],
                rvv_sqrt=counts["rvv_sqrt"],
                rvv_gather=counts["rvv_gather"],
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                tiling_tier=tiling_tier,
                evidence=evidence,
            )
        )

    total = sum(record.ops for record in records if record.include_in_model)
    if total <= 0:
        raise RuntimeError(f"{spec.key} produced no model compute records")
    for record in records:
        record.compute_pct = 100.0 * record.ops / total if record.include_in_model else 0.0
    return records, out_dir


def write_dispatch_csv(records: list[DispatchRecord], path: Path) -> None:
    fields = [
        "model_key",
        "model",
        "idx",
        "symbol",
        "source_file",
        "include_in_model",
        "layer_id",
        "op_kind",
        "segment",
        "opu_path",
        "ops",
        "opu_ops",
        "compute_pct",
        "shape",
        "M",
        "N",
        "K",
        "B",
        "vopacc",
        "opmvinbcast",
        "opu_fetch",
        "rvv_reduction",
        "rvv_sqrt",
        "rvv_gather",
        "tile_m",
        "tile_n",
        "tile_k",
        "tiling_tier",
        "evidence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model_key": record.model_key,
                    "model": record.model,
                    "idx": record.idx,
                    "symbol": record.symbol,
                    "source_file": record.source_file,
                    "include_in_model": int(record.include_in_model),
                    "layer_id": record.layer_id,
                    "op_kind": record.op_kind,
                    "segment": record.segment,
                    "opu_path": record.opu_path,
                    "ops": record.ops,
                    "opu_ops": record.opu_ops,
                    "compute_pct": f"{record.compute_pct:.8f}",
                    "shape": record.shape,
                    "M": record.m,
                    "N": record.n,
                    "K": record.k,
                    "B": record.b,
                    "vopacc": record.vopacc,
                    "opmvinbcast": record.opmvinbcast,
                    "opu_fetch": record.opu_fetch,
                    "rvv_reduction": record.rvv_reduction,
                    "rvv_sqrt": record.rvv_sqrt,
                    "rvv_gather": record.rvv_gather,
                    "tile_m": record.tile_m,
                    "tile_n": record.tile_n,
                    "tile_k": record.tile_k,
                    "tiling_tier": record.tiling_tier,
                    "evidence": record.evidence,
                }
            )


def write_layer_csv(records: list[DispatchRecord], path: Path) -> None:
    fields = ["model_key", "model", "layer_id", "op_kind", "segment", "dispatches", "ops", "opu_ops", "compute_pct"]
    grouped: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    totals: Counter[str] = Counter()
    for record in records:
        if not record.include_in_model:
            continue
        key = (record.model_key, record.model, record.layer_id, record.op_kind, record.segment)
        grouped.setdefault(key, {"dispatches": 0, "ops": 0, "opu_ops": 0})
        grouped[key]["dispatches"] += 1
        grouped[key]["ops"] += record.ops
        grouped[key]["opu_ops"] += record.opu_ops
        totals[record.model_key] += record.ops

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(grouped.items()):
            model_key, model, layer_id, op_kind, segment = key
            total = totals[model_key] or 1
            writer.writerow(
                {
                    "model_key": model_key,
                    "model": model,
                    "layer_id": layer_id,
                    "op_kind": op_kind,
                    "segment": segment,
                    "dispatches": int(values["dispatches"]),
                    "ops": int(values["ops"]),
                    "opu_ops": int(values["opu_ops"]),
                    "compute_pct": f"{100.0 * values['ops'] / total:.8f}",
                }
            )


def write_summary_csv(records: list[DispatchRecord], artifact_dirs: dict[str, Path], path: Path) -> None:
    sys.path.insert(0, str(BENCH_DIR))
    from palette import OPU_SEGMENTS, SEGMENT_ORDER

    fields = [
        "model_key",
        "model",
        *SEGMENT_ORDER,
        "total_ops",
        "opu_ops",
        "opu_pct",
        "dispatches",
        "opu_dispatches",
        "opu_dispatch_pct",
        "artifact_dir",
    ]
    by_model: dict[str, list[DispatchRecord]] = {}
    for record in records:
        if record.include_in_model:
            by_model.setdefault(record.model_key, []).append(record)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for spec in MODELS:
            model_records = by_model.get(spec.key, [])
            if not model_records:
                continue
            segment_ops = Counter({segment: 0 for segment in SEGMENT_ORDER})
            for record in model_records:
                segment_ops[record.segment] += record.ops
            total = sum(segment_ops.values())
            opu = sum(segment_ops[segment] for segment in OPU_SEGMENTS)
            opu_dispatches = sum(1 for record in model_records if record.opu_ops > 0)
            row = {
                "model_key": spec.key,
                "model": spec.display_name,
                **{segment: int(segment_ops[segment]) for segment in SEGMENT_ORDER},
                "total_ops": int(total),
                "opu_ops": int(opu),
                "opu_pct": f"{100.0 * opu / total if total else 0.0:.6f}",
                "dispatches": len(model_records),
                "opu_dispatches": opu_dispatches,
                "opu_dispatch_pct": f"{100.0 * opu_dispatches / len(model_records) if model_records else 0.0:.6f}",
                "artifact_dir": str(artifact_dirs[spec.key].relative_to(MERLIN_ROOT)),
            }
            writer.writerow(row)


def write_opcode_summary(records: list[DispatchRecord], path: Path) -> None:
    fields = [
        "model_key",
        "model",
        "segment",
        "dispatches",
        "ops",
        "vopacc",
        "opmvinbcast",
        "opu_fetch",
        "rvv_reduction",
        "rvv_sqrt",
        "rvv_gather",
    ]
    grouped: dict[tuple[str, str, str], Counter[str]] = {}
    for record in records:
        if not record.include_in_model:
            continue
        key = (record.model_key, record.model, record.segment)
        grouped.setdefault(key, Counter())
        grouped[key]["dispatches"] += 1
        grouped[key]["ops"] += record.ops
        grouped[key]["vopacc"] += record.vopacc
        grouped[key]["opmvinbcast"] += record.opmvinbcast
        grouped[key]["opu_fetch"] += record.opu_fetch
        grouped[key]["rvv_reduction"] += record.rvv_reduction
        grouped[key]["rvv_sqrt"] += record.rvv_sqrt
        grouped[key]["rvv_gather"] += record.rvv_gather

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (model_key, model, segment), counts in sorted(grouped.items()):
            writer.writerow({"model_key": model_key, "model": model, "segment": segment, **counts})


def print_metrics(records: list[DispatchRecord]) -> None:
    by_model: dict[str, list[DispatchRecord]] = {}
    for record in records:
        if record.include_in_model:
            by_model.setdefault(record.model, []).append(record)

    print()
    print("Saturn OPU model compute metrics")
    print("-" * 112)
    print(
        f"{'Model':<18} {'Dispatches':>10} {'OPU disp':>9} {'Total ops':>16} "
        f"{'OPU ops':>16} {'OPU compute':>11}  Top non-OPU"
    )
    print("-" * 112)
    for model, model_records in by_model.items():
        total = sum(record.ops for record in model_records)
        opu = sum(record.opu_ops for record in model_records)
        opu_dispatches = sum(1 for record in model_records if record.opu_ops > 0)
        non_opu = Counter()
        for record in model_records:
            if record.opu_ops == 0:
                non_opu[record.segment] += record.ops
        top_non = ", ".join(f"{k}:{v / total * 100:.1f}%" for k, v in non_opu.most_common(2)) or "-"
        print(
            f"{model:<18} {len(model_records):>10} {opu_dispatches:>9} "
            f"{total:>16,} {opu:>16,} {format_pct(100.0 * opu / total):>11}  {top_non}"
        )
    print("-" * 112)


def format_pct(value: float) -> str:
    if 99.995 <= value < 100.0:
        return "<100%"
    return f"{value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[spec.key for spec in MODELS], help="Analyze one registered model")
    parser.add_argument("--refresh", action="store_true", help="Force recompilation even if artifacts exist")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--out-dispatch", type=Path, default=BENCH_DIR / "model_dispatch_decomposition.csv")
    parser.add_argument("--out-layer", type=Path, default=BENCH_DIR / "model_layer_decomposition.csv")
    parser.add_argument("--out-summary", type=Path, default=BENCH_DIR / "per_model_summary.csv")
    parser.add_argument("--out-opcodes", type=Path, default=BENCH_DIR / "opu_path_opcode_summary.csv")
    args = parser.parse_args()

    specs = [spec for spec in MODELS if args.model is None or spec.key == args.model]
    if not specs:
        raise SystemExit(f"unknown model; choices: {display_model_keys()}")

    records: list[DispatchRecord] = []
    artifact_dirs: dict[str, Path] = {}
    for spec in specs:
        print(f"analyze {spec.key}")
        model_records, out_dir = analyze_model(spec, args.artifact_root.resolve(), args.refresh)
        records.extend(model_records)
        artifact_dirs[spec.key] = out_dir

    for path in (args.out_dispatch, args.out_layer, args.out_summary, args.out_opcodes):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_dispatch_csv(records, args.out_dispatch)
    write_layer_csv(records, args.out_layer)
    write_summary_csv(records, artifact_dirs, args.out_summary)
    write_opcode_summary(records, args.out_opcodes)
    print_metrics(records)
    print(f"wrote {args.out_dispatch}")
    print(f"wrote {args.out_layer}")
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_opcodes}")


if __name__ == "__main__":
    main()
