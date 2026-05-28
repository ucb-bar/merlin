"""Discover linalg ops in a model and emit a manifest skeleton.

Usage:

    python -m tools.kernels.discover \
        models/dronet/dronet.mlir \
        --target spacemit_x60 --hw RVV \
        --output benchmarks/SaturnOPU/kernels/

The tool runs `iree-compile --compile-to=preprocessing` on the model, walks the
resulting IR, and emits one manifest stub entry per unique linalg op pattern
found. Named ops (`linalg.matmul`, `linalg.conv_2d_nchw_fchw`, etc.) get a
single entry covering all their shape variants. `linalg.generic` ops with
distinct iterator-type signatures get separate entries (since each generic
body needs its own kernel logic).

Output for each discovered op:

  manifest.json                 (or appended to an existing one)
  abi/<op>_workgroup.c          stub with TODO marker
  match/<op>.match.mlir         stub for linalg.generic; named ops use
                                `match.kind: "named_op"` and skip this file

The user fills in the `.c` bodies and (for non-named ops) any DAG body
constraints. Re-running the tool against the same output dir merges new
entries without overwriting existing ones (skips if `name` collides).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import re
import subprocess
import sys

_LOG = logging.getLogger("kernels.discover")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# Linalg named ops we know how to emit a "named_op" manifest entry for.
# The default `outs_from_input` and `op_attrs` (when known) come from
# observed dronet/mobilenet/yolov8 patterns; users can override after the
# stub is generated.
_NAMED_OP_DEFAULTS: dict[str, dict] = {
    "linalg.matmul": {"in_count": 2, "outs_from_input": -1},
    "linalg.matmul_transpose_b": {"in_count": 2, "outs_from_input": -1},
    "linalg.conv_2d_nchw_fchw": {
        "in_count": 2,
        "outs_from_input": 2,  # bias-fused-into-outs convention
        "op_attrs": "{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}",
    },
    "linalg.conv_2d_nhwc_hwcf": {
        "in_count": 2,
        "outs_from_input": 2,
        "op_attrs": "{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}",
    },
    "linalg.depthwise_conv_2d_nchw_chw": {"in_count": 2, "outs_from_input": 2},
    "linalg.pooling_nchw_max": {
        "in_count": 2,
        "outs_from_input": 2,
        "op_attrs": "{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}",
    },
    "linalg.pooling_nchw_sum": {
        "in_count": 2,
        "outs_from_input": 2,
        "op_attrs": "{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}",
    },
    "linalg.broadcast": {"in_count": 1, "outs_from_input": -1},
    "linalg.transpose": {"in_count": 1, "outs_from_input": -1},
}


@dataclasses.dataclass(frozen=True)
class Discovery:
    """One unique op pattern observed in the IR."""

    op_name: str  # e.g. "linalg.conv_2d_nchw_fchw" or "linalg.generic#<body>"
    arg_count: int  # number of tensor inputs (excluding outs)
    arg_types: tuple[str, ...]  # tensor types of the inputs
    out_type: str  # tensor type of the output
    iter_types: tuple[str, ...] = ()  # only set for linalg.generic
    indexing_maps: tuple[str, ...] = ()  # affine_map<...> per operand (in then out)
    body_label: str = ""  # recognized body class ("relu", "addf", etc.) or "" if unknown
    body_text: str = ""  # raw body text for unrecognized generics
    occurrences: int = 1
    elements_per_op: int = 0  # product of output tensor's dims, for sorting


@dataclasses.dataclass(frozen=True)
class FusedDispatch:
    """One fused dispatch observed at the flow phase. Bundles multiple
    body ops into a single executable — replacing it with a custom kernel
    means one read/write per element across the entire chain."""

    dispatch_name: str  # e.g. "elementwise_32x55x55_f32"
    occurrences: int = 1
    in_types: tuple[str, ...] = ()  # iree_tensor_ext.dispatch.tensor types
    out_types: tuple[str, ...] = ()
    # Sequence of body op classes, in order. e.g. ("subf", "mulf", "addf",
    # "cmpf+select"). Used to write the C body and decide whether the chain
    # is auto-templatable.
    body_chain: tuple[str, ...] = ()


def discover_flow(ir_text: str) -> list[FusedDispatch]:
    """Walk a flow.mlir and extract fused dispatches.

    For each `flow.executable`, identifies the fused linalg.generic body's
    op chain (sequence of arith/math ops between `^bb0:` and `linalg.yield`).
    Groups by (dispatch_name, body_chain) so dispatches with identical
    fusion patterns at different shapes share a kernel.
    """
    counts: dict[tuple, FusedDispatch] = {}
    # Split on `flow.executable private` openings; each segment after the
    # first holds one executable's text up to the next executable.
    segments = ir_text.split("flow.executable private ")[1:]
    for seg in segments:
        # Dispatch name lives on the `flow.executable.export public @...`
        # line, which follows the executable opening. Strip the
        # `@main_graph$async_dispatch_<N>_` prefix and the trailing
        # shape+dtype suffix like `_32x55x55_f32` to get the family.
        m = re.search(r"flow\.executable\.export public @main_graph[$]async_dispatch_\d+_" r"([a-zA-Z_0-9]+)\b", seg)
        if not m:
            continue
        full_name = m.group(1)
        family = re.sub(r"_(\d+)(x\d+)*_(f|i)\d+$", "", full_name)
        # Find the func.func signature (between first `func.func @` and `{`).
        sig_m = re.search(r"func\.func @[^(]+\(([^)]*)\)\s*\{", seg)
        sig = sig_m.group(1) if sig_m else ""
        in_types: list[str] = []
        out_types: list[str] = []
        for arg in sig.split(","):
            arg = arg.strip()
            t = re.search(r"tensor<[^>]+>", arg)
            if not t:
                continue
            if "readonly" in arg:
                in_types.append(t.group(0))
            elif "writeonly" in arg or "readwrite" in arg:
                out_types.append(t.group(0))
        # Scan inside any linalg.generic body (between `^bb0(...):` and the
        # subsequent `linalg.yield`) for arith/math ops in chain order.
        chain: list[str] = []
        body_m = re.search(r"linalg\.generic[^{]*\{[^}]*\}[^{]*\{(.*?)\}\s*->\s*tensor", seg, re.DOTALL)
        if body_m:
            body = body_m.group(1)
            for op_m in re.finditer(r"\b(arith|math)\.([a-zA-Z_0-9]+)\b", body):
                op = f"{op_m.group(1)}.{op_m.group(2)}"
                if op == "arith.constant":
                    continue
                chain.append(op)
        key = (family, tuple(chain), tuple(in_types), tuple(out_types))
        if key in counts:
            counts[key] = dataclasses.replace(counts[key], occurrences=counts[key].occurrences + 1)
        else:
            counts[key] = FusedDispatch(
                dispatch_name=family,
                occurrences=1,
                in_types=tuple(in_types),
                out_types=tuple(out_types),
                body_chain=tuple(chain),
            )
    return list(counts.values())


def run_flow(
    input_mlir: pathlib.Path,
    target: str,
    hw: str,
    extra_iree_args: list[str],
) -> str:
    """Compile the model to flow phase (no kernels) and return the IR text."""
    out_dir = REPO_ROOT / "build" / "kernel_discovery_flow_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / "merlin"),
        "compile",
        str(input_mlir),
        "--target",
        target,
        "--hw",
        hw,
        "--no-kernel-embedding",
        "--compile-to",
        "flow",
        "--output-dir",
        str(out_dir),
    ]
    for a in extra_iree_args:
        cmd.append(f"--iree-compile-arg={a}")
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"./merlin compile --compile-to=flow failed:\n"
            f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        )
    candidates = list(out_dir.glob("*flow*.mlir"))
    if not candidates:
        candidates = list(out_dir.glob("*.mlir"))
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0].read_text()


def run_preprocessing(
    input_mlir: pathlib.Path,
    target: str,
    hw: str,
    extra_iree_args: list[str],
) -> str:
    """Run `./merlin compile --compile-to=preprocessing` and return the
    preprocessing-phase MLIR text."""
    out_dir = REPO_ROOT / "build" / "kernel_discovery_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / "merlin"),
        "compile",
        str(input_mlir),
        "--target",
        target,
        "--hw",
        hw,
        "--no-kernel-embedding",
        "--compile-to",
        "preprocessing",
        "--output-dir",
        str(out_dir),
    ]
    for a in extra_iree_args:
        # Use `--flag=value` form so argparse doesn't treat `-`-prefixed
        # values (like "--iree-opt-data-tiling=false") as a new option.
        cmd.append(f"--iree-compile-arg={a}")
    _LOG.info("running: %s", " ".join(cmd))
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"./merlin compile --compile-to=preprocessing failed:\n"
            f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        )
    # The preprocessing output is the model's .mlir under output-dir;
    # `--compile-to` emits it in place of the .vmfb.
    candidates = list(out_dir.glob("*.mlir"))
    candidates = [c for c in candidates if not c.name.startswith("dronet.")]  # original copy
    # Prefer the `.compile_to_<phase>.mlir` style name when present.
    for c in out_dir.glob("*compile_to*.mlir"):
        return c.read_text()
    # Fallback: the largest .mlir in the dir (excluding the input copy).
    if not candidates:
        # Just take whatever's there.
        candidates = list(out_dir.glob("*.mlir"))
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0].read_text()


_NAMED_OP_RE = re.compile(
    r"linalg\.(?P<name>conv_2d_[a-z_]+|pooling_[a-z_]+|depthwise_conv_[a-z_0-9]+|matmul(?:_transpose_[ab])?|broadcast|transpose|fill)"
    r"\b[^\n]*?ins\((?P<ins>[^)]*)\)\s*outs\((?P<outs>[^)]*)\)\s*->\s*(?P<rty>tensor<[^>]+>)"
)
# Capture indexing maps (parenthesized list inside `indexing_maps = [...]`)
# along with iterator types and ins/outs and the body (between `{` and the
# matching `linalg.yield`). Body is non-greedy so we don't run past one
# generic's body into the next.
_GENERIC_RE = re.compile(
    r"linalg\.generic\s*\{[^}]*indexing_maps\s*=\s*\[(?P<maps>[^\]]*)\][^}]*"
    r"iterator_types\s*=\s*\[(?P<iters>[^\]]*)\][^}]*\}\s*"
    r"ins\((?P<ins>[^)]*)\)\s*outs\((?P<outs>[^)]*)\)\s*\{(?P<body>.*?)linalg\.yield",
    re.DOTALL,
)


# --- Body classification --------------------------------------------------
# Each entry: (label, recognizer, in_count_after_match, c_body_template).
# The recognizer is a regex over the linalg.generic body text (between
# `^bb0(...):` and `linalg.yield`). The C body template uses
# `{IN0}`, `{IN1}`, `{IN2}`, `{OUT}` placeholders and assumes the kernel
# operates one element per workgroup with `tid` as the flat output index.

_BODY_RECOGNIZERS: list[tuple[str, str, int, str]] = [
    # NOTE: ReLU's `arith.cmpf ugt + arith.select` body is recognized but
    # the auto-emitted match.mlir doesn't reliably catch the IREE-canonicalized
    # form (the 0.0 constant gets float-hoisted outside the linalg body, and
    # cast_compatible_dag doesn't handle outer-scope SSA captures cleanly).
    # The C body template IS correct; user can still take the auto-emitted
    # `.c` and write the match.mlir by hand. So we LEAVE relu in this list
    # so the C kernel is auto-emitted, but the manifest-side match.mlir will
    # be a stub for now.
    ("relu", r"arith\.cmpf\s+ugt[^\n]*\n[^\n]*arith\.select", 1, "  out_v = (in0_v > 0.0f) ? in0_v : 0.0f;"),
    # math unary ops.
    # Use compiler builtins so the auto-generated kernels work in
    # `-ffreestanding` builds (no libm available).
    ("rsqrt", r"math\.rsqrt\b", 1, "  out_v = 1.0f / __builtin_sqrtf(in0_v);"),
    ("sqrt", r"math\.sqrt\b", 1, "  out_v = __builtin_sqrtf(in0_v);"),
    ("exp", r"math\.exp\b", 1, "  out_v = __builtin_expf(in0_v);"),
    ("log", r"math\.log\b", 1, "  out_v = __builtin_logf(in0_v);"),
    ("absf", r"math\.absf\b", 1, "  out_v = __builtin_fabsf(in0_v);"),
    # arith binary ops (must come AFTER ReLU since ReLU also uses arith).
    ("addf", r"arith\.addf\b", 2, "  out_v = in0_v + in1_v;"),
    ("subf", r"arith\.subf\b", 2, "  out_v = in0_v - in1_v;"),
    ("mulf", r"arith\.mulf\b", 2, "  out_v = in0_v * in1_v;"),
    ("divf", r"arith\.divf\b", 2, "  out_v = in0_v / in1_v;"),
    ("maxf", r"arith\.maximumf\b", 2, "  out_v = (in0_v > in1_v) ? in0_v : in1_v;"),
    ("minf", r"arith\.minimumf\b", 2, "  out_v = (in0_v < in1_v) ? in0_v : in1_v;"),
    # arith unary ops.
    ("negf", r"arith\.negf\b", 1, "  out_v = -in0_v;"),
    # Identity / cast (no arithmetic, just yield the input).
    ("identity", r"linalg\.yield\s+%[a-zA-Z_0-9]+\s*:\s*f32", 1, "  out_v = in0_v;"),
]


def classify_body(body_text: str) -> str | None:
    """Return the label of the first recognized body pattern, or None."""
    for label, pat, _, _ in _BODY_RECOGNIZERS:
        if re.search(pat, body_text):
            return label
    return None


def body_template(label: str) -> tuple[int, str] | None:
    for lab, _, in_count, body in _BODY_RECOGNIZERS:
        if lab == label:
            return (in_count, body)
    return None


def _parse_ins_outs(s: str) -> tuple[list[str], list[str]]:
    # `(%a, %b : tensor<...>, tensor<...>)` -> (ssa_refs, types).
    if ":" not in s:
        return ([], [])
    refs_part, types_part = s.rsplit(":", 1)
    refs = [r.strip() for r in refs_part.split(",") if r.strip()]
    types = [t.strip() for t in types_part.split(",") if t.strip()]
    return (refs, types)


def discover(ir_text: str) -> list[Discovery]:
    counts: dict[tuple, Discovery] = {}

    for m in _NAMED_OP_RE.finditer(ir_text):
        op_name = "linalg." + m.group("name")
        _, in_types = _parse_ins_outs(m.group("ins"))
        _, out_types = _parse_ins_outs(m.group("outs"))
        out_ty = m.group("rty")
        key = (op_name, len(in_types), tuple(in_types), out_ty)
        if key in counts:
            counts[key] = dataclasses.replace(counts[key], occurrences=counts[key].occurrences + 1)
        else:
            counts[key] = Discovery(
                op_name=op_name,
                arg_count=len(in_types),
                arg_types=tuple(in_types),
                out_type=out_ty,
                occurrences=1,
                elements_per_op=_count_elements(out_ty),
            )

    # Generics: distinguish by (iterator-types, body classification).
    for m in _GENERIC_RE.finditer(ir_text):
        iters = tuple(t.strip().strip('"') for t in m.group("iters").split(","))
        _, in_types = _parse_ins_outs(m.group("ins"))
        _, out_types = _parse_ins_outs(m.group("outs"))
        if not out_types:
            continue
        body = m.group("body")
        label = classify_body(body) or "unknown"
        # Indexing maps as a tuple of strings, one per operand.
        maps = tuple(s.strip() for s in m.group("maps").split(",") if s.strip())
        op_label = f"linalg.generic#{label}#{'_'.join(iters)}"
        key = (op_label, len(in_types), tuple(in_types), out_types[0])
        if key in counts:
            counts[key] = dataclasses.replace(counts[key], occurrences=counts[key].occurrences + 1)
        else:
            counts[key] = Discovery(
                op_name=op_label,
                arg_count=len(in_types),
                arg_types=tuple(in_types),
                out_type=out_types[0],
                iter_types=iters,
                indexing_maps=maps,
                body_label=label,
                body_text=body if label == "unknown" else "",
                occurrences=1,
                elements_per_op=_count_elements(out_types[0]),
            )

    return list(counts.values())


def _count_elements(tensor_ty: str) -> int:
    body = tensor_ty.removeprefix("tensor<").rstrip(">")
    if "x" not in body:
        return 1
    parts = body.split("x")
    rank_dims = parts[:-1]
    n = 1
    for d in rank_dims:
        if d == "?":
            return 0
        try:
            n *= int(d)
        except ValueError:
            return 0
    return n


def _shape_to_dynamic(t: str) -> str:
    # tensor<1x32x55x55xf32> -> tensor<?x?x?x?xf32>
    body = t.removeprefix("tensor<").rstrip(">")
    if "x" not in body:
        return t  # rank-0
    parts = body.split("x")
    dtype = parts[-1]
    rank_dims = parts[:-1]
    return f"tensor<{'x'.join(['?'] * len(rank_dims))}x{dtype}>"


def _rank(tensor_ty: str) -> int:
    body = tensor_ty.removeprefix("tensor<").rstrip(">")
    if "x" not in body:
        return 0
    parts = body.split("x")
    return len(parts) - 1  # last part is dtype


def _dim_axis_letters(rank: int) -> list[str]:
    # Stable per-rank dim-name set: D0, D1, D2, ... Easier than guessing N/C/H/W.
    return [f"D{k}" for k in range(rank)]


def _generic_constants_and_aliases(
    arg_types: tuple[str, ...],
    out_type: str,
    indexing_maps: tuple[str, ...],
) -> tuple[list[dict], list[str]]:
    """Emit per-input dim constants for a generic op. Constants are named
    `I<input>_<dim>`. For each output dim, we use input 0's identity-
    corresponding name when available (the canonical elementwise case);
    broadcast inputs (lower rank) just get their own constants and the user
    fills in body-level indexing.
    """
    out_rank = _rank(out_type)
    if out_rank == 0:
        return [], []
    constants: list[dict] = []
    for in_idx, t in enumerate(arg_types):
        r = _rank(t)
        for k in range(r):
            constants.append(
                {
                    "name": f"I{in_idx}_{k}",
                    "type": "i32",
                    "from": {"input": in_idx, "dim": k},
                }
            )
    # output_dims defaults to input 0's first `out_rank` dims when input 0
    # has matching rank (the elementwise identity case). Otherwise leave
    # output_dims empty and let the user supply it.
    if arg_types and _rank(arg_types[0]) == out_rank:
        dim_names = [f"I0_{k}" for k in range(out_rank)]
    else:
        dim_names = []
    return constants, dim_names


def _emit_generic_c_body(
    entry_symbol: str,
    in_count: int,
    out_rank: int,
    dim_names: list[str],
    body_template_str: str,
    op_label: str,
    occurrences: int,
) -> str:
    """Emit a complete scalar C kernel for a recognized linalg.generic body.
    Reads each input as a flat row-major array (first input fully indexed,
    remaining inputs are assumed identity-mapped — broadcast handling is a
    follow-on)."""
    bind_ptrs = []
    for i in range(in_count):
        bind_ptrs.append(f"const float *restrict binding{i}, size_t binding{i}_offset")
    bind_ptrs.append(f"float *restrict binding{in_count}, size_t binding{in_count}_offset")
    dims_args = ", ".join(f"size_t {n}" for n in dim_names)
    sig = ",\n    ".join(bind_ptrs + [dims_args, "size_t tid"])
    total_expr = " * ".join(dim_names) if dim_names else "1"
    in_loads = "\n  ".join(f"float in{i}_v = binding{i}[binding{i}_offset + tid];" for i in range(in_count))
    out_store = f"binding{in_count}[binding{in_count}_offset + tid] = out_v;"
    return (
        f"// AUTO-GENERATED kernel for {op_label} (observed {occurrences}x).\n"
        f"// Body: identity-elementwise scalar — replace with vectorized\n"
        f"// implementation if you need throughput. Uses compiler builtins\n"
        f"// (`__builtin_*`) so the kernel compiles under `-ffreestanding`\n"
        f"// without libm.\n"
        f"#include <stddef.h>\n\n"
        f'__attribute__((visibility("default")))\n'
        f"void {entry_symbol}(\n    {sig}) {{\n"
        f"  if (tid >= {total_expr}) return;\n"
        f"  {in_loads}\n"
        f"  float out_v;\n"
        f"{body_template_str}\n"
        f"  {out_store}\n"
        f"}}\n"
    )


def _emit_generic_match_mlir(
    arg_types: tuple[str, ...],
    out_type: str,
    indexing_maps: tuple[str, ...],
    iter_types: tuple[str, ...],
    body_label: str,
) -> str:
    """Emit a complete linalg_dag match.mlir body for a recognized generic
    body pattern. Uses dynamic shapes so the matcher generalizes over all
    concrete shape variants."""
    out_rank = _rank(out_type)
    dyn_in_types = [_shape_to_dynamic(t) for t in arg_types]
    dyn_out_ty = _shape_to_dynamic(out_type)
    iter_str = ", ".join(f'"{t}"' for t in iter_types)
    map_str = ", ".join(f"affine_map<{m.split('=')[-1].strip()}>" if "=" in m else m for m in indexing_maps)
    bb_args = ", ".join(f"%in{i}: {dyn_in_types[i]}" for i in range(len(arg_types)))
    in_refs = ", ".join(f"%in{i}" for i in range(len(arg_types)))
    in_types_csv = ", ".join(dyn_in_types)
    # Build a fresh empty for outs.
    if out_rank > 0:
        dim_lines = []
        for k in range(out_rank):
            dim_lines.append(f"  %c{k} = arith.constant {k} : index")
            dim_lines.append(f"  %d{k} = tensor.dim %in0, %c{k} : {dyn_in_types[0]}")
        dim_args = ", ".join(f"%d{k}" for k in range(out_rank))
        empty_line = f"  %empty = tensor.empty({dim_args}) " f'{{"match.operation_name_only"}} : {dyn_out_ty}'
        dim_block = "\n".join(dim_lines)
    else:
        empty_line = f"  %empty = tensor.empty() " f'{{"match.operation_name_only"}} : {dyn_out_ty}'
        dim_block = ""

    # Body MLIR — minimal version of each recognized pattern, baked at
    # match-template level so cast_compatible_dag distinguishes them.
    bb_in_args = ", ".join(f"%a{i}: f32" for i in range(len(arg_types)))
    bb_in_args_with_out = bb_in_args + ", %_out: f32"
    body_mlir = {
        "addf": "    %r = arith.addf %a0, %a1 : f32\n    linalg.yield %r : f32",
        "subf": "    %r = arith.subf %a0, %a1 : f32\n    linalg.yield %r : f32",
        "mulf": "    %r = arith.mulf %a0, %a1 : f32\n    linalg.yield %r : f32",
        "divf": "    %r = arith.divf %a0, %a1 : f32\n    linalg.yield %r : f32",
        "maxf": "    %r = arith.maximumf %a0, %a1 : f32\n    linalg.yield %r : f32",
        "minf": "    %r = arith.minimumf %a0, %a1 : f32\n    linalg.yield %r : f32",
        "rsqrt": "    %r = math.rsqrt %a0 : f32\n    linalg.yield %r : f32",
        "sqrt": "    %r = math.sqrt %a0 : f32\n    linalg.yield %r : f32",
        "exp": "    %r = math.exp %a0 : f32\n    linalg.yield %r : f32",
        "log": "    %r = math.log %a0 : f32\n    linalg.yield %r : f32",
        "absf": "    %r = math.absf %a0 : f32\n    linalg.yield %r : f32",
        "negf": "    %r = arith.negf %a0 : f32\n    linalg.yield %r : f32",
        # NOTE: ReLU's `%cst` is hoisted out of the linalg body by canonicalization
        # in real IR. The matcher would need to capture an outer-scope constant —
        # cast_compatible_dag doesn't reliably support that, so we declare ReLU
        # body match as a stub (auto-emitted C is still useful as a starting
        # point). To match real ReLU, write a hand-tuned match with the outer
        # `%cst = arith.constant 0.0` declared in the bb args and referenced
        # inside the linalg body via SSA capture.
        "relu": "    // STUB BODY — see note in discover.py _emit_generic_match_mlir\n"
        "    %c = arith.cmpf ugt, %a0, %a0 : f32\n"
        "    linalg.yield %a0 : f32",
        "identity": "    linalg.yield %a0 : f32",
    }.get(body_label, "    // TODO: body\n    linalg.yield %a0 : f32")

    outer_const = ""
    return (
        f"// AUTO-GENERATED match.mlir for linalg.generic body class '{body_label}'.\n"
        f"// iter_types = [{iter_str}]\n"
        f"// indexing_maps = [{map_str}]\n"
        f"^bb0({bb_args}):\n"
        f"{dim_block}\n"
        f"{outer_const}"
        f"{empty_line}\n"
        f"  %op = linalg.generic\n"
        f"      {{indexing_maps = [{map_str}],\n"
        f"       iterator_types = [{iter_str}]}}\n"
        f"      ins({in_refs} : {in_types_csv})\n"
        f"      outs(%empty : {dyn_out_ty}) {{\n"
        f"    ^bb_inner({bb_in_args_with_out}):\n"
        f"{body_mlir}\n"
        f"  }} -> {dyn_out_ty}\n"
    )


def synthesize_manifest_entry(
    d: Discovery,
    target_key: str,
) -> tuple[dict, str, str | None]:
    """Return (manifest_entry, source_c, match_mlir_or_none).

    For named ops with known defaults: emits a `named_op` manifest entry.
    For linalg.generic with a recognized body pattern: emits a complete
    `linalg_dag` entry + match.mlir + scalar C kernel — runnable as-is.
    For unrecognized bodies: emits a TODO stub the user must finish.
    """
    base_name = d.op_name.replace("linalg.", "").replace(".", "_").replace("#", "_")
    base_name = re.sub(r"[^a-z0-9_]", "_", base_name)
    kernel_name = f"discovered_{base_name}"
    entry_symbol = f"{base_name}_workgroup"

    # Filter out scalar operands (e.g. linalg.fill takes a scalar f32 in,
    # which doesn't go through a tensor binding).
    tensor_arg_types = tuple(t for t in d.arg_types if t.startswith("tensor<"))
    d = dataclasses.replace(d, arg_types=tensor_arg_types, arg_count=len(tensor_arg_types))

    if d.op_name in _NAMED_OP_DEFAULTS:
        spec = _NAMED_OP_DEFAULTS[d.op_name]
        operands = [{"role": "in", "tensor": _shape_to_dynamic(t)} for t in d.arg_types]
        if spec["outs_from_input"] >= 0:
            operands.append({"role": "in", "tensor": _shape_to_dynamic(d.out_type)})
        operands.append({"role": "out", "tensor": _shape_to_dynamic(d.out_type)})
        match_block = {"kind": "named_op", "op_name": d.op_name}
        if spec["outs_from_input"] >= 0:
            match_block["outs_from_input"] = len(d.arg_types)
        if "op_attrs" in spec:
            match_block["op_attrs"] = spec["op_attrs"]
        # Emit per-input dim constants so the stub manifest is at least
        # loadable. Names use I<input>_<dim> to avoid collisions; user can
        # rename + add aliases when filling in the kernel body.
        all_inputs = list(d.arg_types)
        if spec["outs_from_input"] >= 0:
            # The "extra input" we appended for the outs slot also gets dims.
            all_inputs.append(d.out_type)
        sig = {"operands": operands}
        constants: list[dict] = []
        for in_idx, t in enumerate(all_inputs):
            r = _rank(t)
            for k in range(r):
                constants.append(
                    {
                        "name": f"I{in_idx}_{k}",
                        "type": "i32",
                        "from": {"input": in_idx, "dim": k},
                    }
                )
        if constants:
            sig["constants"] = constants
        # Default output_dims to the outs-input's dims when present, else
        # input 0's dims if rank matches.
        out_rank = _rank(d.out_type)
        if out_rank > 0:
            if spec["outs_from_input"] >= 0:
                outs_in_idx = len(d.arg_types)
                sig["output_dims"] = [f"I{outs_in_idx}_{k}" for k in range(out_rank)]
            elif _rank(d.arg_types[0]) == out_rank if d.arg_types else False:
                sig["output_dims"] = [f"I0_{k}" for k in range(out_rank)]
        entry = {
            "name": kernel_name,
            "source": f"abi/{kernel_name}_workgroup.c",
            "source_lang": "c",
            "entry_symbol": entry_symbol,
            "signature": sig,
            "match": match_block,
            "targets": [target_key],
        }
        c_stub = (
            f"// AUTO-GENERATED STUB for {d.op_name} (observed {d.occurrences}x).\n"
            f"// Named-op kernels need a body specific to the op's semantics —\n"
            f"// scalar reference templates are not auto-emitted for these. See\n"
            f"// benchmarks/SaturnOPU/kernels/abi/{base_name}_workgroup.c for an\n"
            f"// example if one exists.\n"
            f"#include <stddef.h>\n\n"
            f'__attribute__((visibility("default")))\n'
            f"void {entry_symbol}(/* TODO: IREE custom-dispatch ABI */) {{\n"
            f"  // TODO\n"
            f"}}\n"
        )
        return entry, c_stub, None

    # linalg.generic path — auto-fill if body is recognized.
    body_template_entry = body_template(d.body_label) if d.body_label else None
    if body_template_entry is None or d.body_label == "unknown":
        # Fallback: stub. Still emit per-input dim constants so the manifest
        # is loadable by spec_gen (which requires constants for any operand
        # with >1 dynamic dim).
        operands = [{"role": "in", "tensor": _shape_to_dynamic(t)} for t in d.arg_types]
        operands.append({"role": "out", "tensor": _shape_to_dynamic(d.out_type)})
        sig = {"operands": operands}
        constants: list[dict] = []
        for in_idx, t in enumerate(d.arg_types):
            for k in range(_rank(t)):
                constants.append(
                    {
                        "name": f"I{in_idx}_{k}",
                        "type": "i32",
                        "from": {"input": in_idx, "dim": k},
                    }
                )
        if constants:
            sig["constants"] = constants
        out_rank_stub = _rank(d.out_type)
        if out_rank_stub > 0 and d.arg_types and _rank(d.arg_types[0]) == out_rank_stub:
            sig["output_dims"] = [f"I0_{k}" for k in range(out_rank_stub)]
        match_block = {
            "kind": "linalg_dag",
            "spec_path": f"match/{kernel_name}.match.mlir",
        }
        entry = {
            "name": kernel_name,
            "source": f"abi/{kernel_name}_workgroup.c",
            "source_lang": "c",
            "entry_symbol": entry_symbol,
            "signature": sig,
            "match": match_block,
            "targets": [target_key],
        }
        c_stub = (
            f"// AUTO-GENERATED STUB for {d.op_name} (observed {d.occurrences}x).\n"
            f"// Body class: {d.body_label or '(unrecognized)'}.\n"
            f"// To auto-generate the body, extend kernels/core/discover.py\n"
            f"// _BODY_RECOGNIZERS with this op chain and re-run.\n"
            f"#include <stddef.h>\n\n"
            f"__attribute__((visibility(\"default\")))\n"
            f"void {entry_symbol}(/* TODO */) {{\n"
            f"  // TODO\n"
            f"}}\n"
        )
        match_text = (
            f"// AUTO-GENERATED STUB for unrecognized body class '{d.body_label}'.\n"
            f"// Body text:\n"
            + "".join(f"//   {ln}\n" for ln in d.body_text.splitlines() if ln.strip())
            + "^bb0():\n  // TODO\n"
        )
        return entry, c_stub, match_text

    # Recognized body — emit complete kernel.
    in_count, body_template_str = body_template_entry
    if in_count != d.arg_count:
        # Body recognized but operand count doesn't match: fall back to stub.
        # (e.g., a 2-input pattern matched to a 1-input occurrence.)
        return synthesize_manifest_entry(dataclasses.replace(d, body_label="unknown"), target_key)
    # If any input has a different rank than the output (broadcast pattern),
    # the auto-emitted C kernel and match.mlir templates can't express it
    # correctly; fall back to a stub the user fills in.
    out_rank_check = _rank(d.out_type)
    if any(_rank(t) != out_rank_check for t in d.arg_types):
        return synthesize_manifest_entry(dataclasses.replace(d, body_label="unknown"), target_key)

    out_rank = _rank(d.out_type)
    constants, dim_names = _generic_constants_and_aliases(d.arg_types, d.out_type, d.indexing_maps)
    operands = [{"role": "in", "tensor": _shape_to_dynamic(t)} for t in d.arg_types]
    operands.append({"role": "out", "tensor": _shape_to_dynamic(d.out_type)})

    sig = {"operands": operands}
    if constants:
        sig["constants"] = constants
        sig["output_dims"] = dim_names

    match_block = {
        "kind": "linalg_dag",
        "spec_path": f"match/{kernel_name}.match.mlir",
    }
    entry = {
        "name": kernel_name,
        "source": f"abi/{kernel_name}_workgroup.c",
        "source_lang": "c",
        "entry_symbol": entry_symbol,
        "signature": sig,
        "match": match_block,
        "targets": [target_key],
    }
    c_text = _emit_generic_c_body(
        entry_symbol=entry_symbol,
        in_count=in_count,
        out_rank=out_rank,
        dim_names=dim_names,
        body_template_str=body_template_str,
        op_label=d.op_name,
        occurrences=d.occurrences,
    )
    match_text = _emit_generic_match_mlir(
        arg_types=d.arg_types,
        out_type=d.out_type,
        indexing_maps=d.indexing_maps,
        iter_types=d.iter_types,
        body_label=d.body_label,
    )
    return entry, c_text, match_text


def write_outputs(
    discoveries: list[Discovery],
    out_dir: pathlib.Path,
    target_key: str,
    *,
    overwrite: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "abi").mkdir(parents=True, exist_ok=True)
    (out_dir / "match").mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        existing_names = {k["name"] for k in manifest.get("kernels", [])}
    else:
        manifest = {"schema_version": 1, "kernels": []}
        existing_names = set()

    # "Biggest coverage with fewest kernels" prioritization: rank by
    # estimated total work covered = occurrences × output element count.
    # We still emit ALL discoveries (so coverage can hit 100%), but the
    # report at the end tells the user which kernels to *invest in
    # optimizing* first.
    def _impact(d: Discovery) -> int:
        return d.occurrences * (d.elements_per_op or 1)

    new_complete = 0
    new_stubs: list[tuple[str, Discovery]] = []
    skipped_no_inputs = 0
    seen_names_in_this_run: set[str] = set()
    stubs_dir = out_dir / "stubs"
    for d in sorted(discoveries, key=lambda d: -_impact(d)):
        # Skip ops with no tensor inputs (e.g. linalg.fill) — they need
        # special spec handling (output-only kernels) the auto-spec doesn't
        # support yet.
        tensor_inputs = sum(1 for t in d.arg_types if t.startswith("tensor<"))
        if tensor_inputs == 0:
            skipped_no_inputs += 1
            continue
        entry, c_text, match_text = synthesize_manifest_entry(d, target_key)
        if entry["name"] in seen_names_in_this_run:
            continue
        seen_names_in_this_run.add(entry["name"])

        is_complete = "// TODO" not in c_text and (match_text is None or "// TODO" not in match_text)

        if is_complete:
            # Only complete entries enter the live manifest (compile.py
            # loads it and any incomplete entry would crash spec_gen or
            # iree-compile).
            if entry["name"] in existing_names and not overwrite:
                continue
            if entry["name"] in existing_names and overwrite:
                manifest["kernels"] = [k for k in manifest["kernels"] if k["name"] != entry["name"]]
            manifest["kernels"].append(entry)
            existing_names.add(entry["name"])
            (out_dir / entry["source"]).write_text(c_text)
            if match_text is not None:
                (out_dir / entry["match"]["spec_path"]).write_text(match_text)
            new_complete += 1
        else:
            # Park stubs in a sibling `stubs/` directory so the user can
            # finish them and move them into the live manifest later.
            stubs_dir.mkdir(parents=True, exist_ok=True)
            (stubs_dir / "abi").mkdir(parents=True, exist_ok=True)
            (stubs_dir / "match").mkdir(parents=True, exist_ok=True)
            entry_copy = dict(entry)
            entry_copy["source"] = f"stubs/{entry['source']}"
            if entry["match"]["kind"] == "linalg_dag" and match_text is not None:
                entry_copy["match"] = dict(entry["match"])
                entry_copy["match"]["spec_path"] = f"stubs/{entry['match']['spec_path']}"
                (stubs_dir / entry["match"]["spec_path"]).write_text(match_text)
            (stubs_dir / entry["source"]).write_text(c_text)
            new_stubs.append((entry["name"], d))

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Emit STUBS.md describing what still needs to be authored to reach
    # 100% coverage. Sorted by impact (biggest coverage win first).
    if new_stubs:
        stubs_md = ["# Kernel stubs to author for full coverage", ""]
        stubs_md.append(
            "Discovery couldn't auto-generate these kernels "
            "(named ops needing custom bodies, broadcast inputs, "
            "or unrecognized linalg.generic body classes)."
        )
        stubs_md.append("")
        stubs_md.append("After writing each kernel:")
        stubs_md.append("1. Move `stubs/abi/<name>.c` → `abi/<name>.c`.")
        stubs_md.append("2. Move `stubs/match/<name>.match.mlir` → `match/<name>.match.mlir` (if present).")
        stubs_md.append("3. Move the entry from `stubs.manifest.json` into `manifest.json`.")
        stubs_md.append("4. Re-run `./merlin compile … --kernels-strict-coverage`.")
        stubs_md.append("")
        stubs_md.append("| Impact | Op | Output type |")
        stubs_md.append("|---:|---|---|")
        for name, d in sorted(new_stubs, key=lambda nd: -_impact(nd[1])):
            stubs_md.append(f"| {d.occurrences * (d.elements_per_op or 1):>12,} | " f"`{d.op_name}` | `{d.out_type}` |")
        (out_dir / "STUBS.md").write_text("\n".join(stubs_md) + "\n")
        # Also write a separate stubs manifest the user can pick from.
        stubs_manifest_path = out_dir / "stubs.manifest.json"
        stubs_manifest = {"schema_version": 1, "kernels": []}
        for name, d in new_stubs:
            entry, _, _ = synthesize_manifest_entry(d, target_key)
            entry["source"] = f"stubs/{entry['source']}"
            if entry["match"]["kind"] == "linalg_dag":
                entry["match"]["spec_path"] = f"stubs/{entry['match']['spec_path']}"
            stubs_manifest["kernels"].append(entry)
        stubs_manifest_path.write_text(json.dumps(stubs_manifest, indent=2) + "\n")

    print(f"Wrote {new_complete} complete kernel entries to {out_dir}/manifest.json")
    if new_stubs:
        print(f"  📝 {len(new_stubs)} stubs need authoring — see {out_dir}/STUBS.md")
    if skipped_no_inputs:
        print(f"  ⏭  {skipped_no_inputs} skipped (no tensor inputs — e.g. linalg.fill)")
    print(f"Total kernels in live manifest: {len(manifest['kernels'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mlir", type=pathlib.Path)
    parser.add_argument("--target", required=True, help="Merlin target name (e.g. spacemit_x60)")
    parser.add_argument("--hw", required=True, help="--hw value (e.g. RVV)")
    parser.add_argument(
        "--output", type=pathlib.Path, required=True, help="Kernel directory to write into (created if missing)"
    )
    parser.add_argument(
        "--target-key",
        default="llvm-cpu-spacemit-x60",
        help="Kernel target key for precompile.py (default: llvm-cpu-spacemit-x60)",
    )
    parser.add_argument(
        "--iree-compile-arg", action="append", default=[], help="Extra flag forwarded to iree-compile (repeatable)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing source/match stubs")
    parser.add_argument(
        "--minimum-cover",
        action="store_true",
        help=(
            "Print the minimum kernel set that, if implemented, would cover "
            "100%% of the model's dispatches. Greedy set-cover over the "
            "discovery output: at each step picks the signature that covers "
            "the most still-uncovered dispatches by total compute. Output is "
            "informational; user picks which to implement."
        ),
    )
    parser.add_argument(
        "--auto-fuse",
        action="store_true",
        help=(
            "Also compile to flow phase and report fused dispatches. Each "
            "fused dispatch is a single executable that bundles multiple "
            "ops; replacing it with one C kernel means one read/write per "
            "element across the whole fused chain (vs N reads/writes when "
            "matching at preprocessing). Emits skeleton fused-kernel stubs "
            "into <output>/fused_stubs/ and a FUSED_STUBS.md report."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    ir_text = run_preprocessing(
        args.input_mlir.resolve(),
        args.target,
        args.hw,
        args.iree_compile_arg,
    )
    discoveries = discover(ir_text)
    if not discoveries:
        print("No linalg ops discovered.", file=sys.stderr)
        return 1
    write_outputs(discoveries, args.output.resolve(), args.target_key, overwrite=args.overwrite)
    print()
    print("Discoveries — ranked by impact (occurrences × output elements):")
    print("  Rule of thumb: cover the biggest amount of the model with the")
    print("  fewest kernels. Top-of-list ops carry the most compute.")
    print()
    total_occ = sum(d.occurrences for d in discoveries)
    cumulative = 0
    for d in sorted(discoveries, key=lambda d: -(d.occurrences * (d.elements_per_op or 1))):
        impact = d.occurrences * (d.elements_per_op or 1)
        cumulative += d.occurrences
        pct = 100.0 * cumulative / total_occ if total_occ else 0.0
        print(f"  {d.occurrences:3}x  impact={impact:>12,}  " f"cum={pct:5.1f}%  {d.op_name}  -> {d.out_type}")

    if args.minimum_cover:
        print()
        print("--minimum-cover: smallest kernel set covering all of the model.")
        print("  Each row is ONE author-unit kernel (dynamic-shape match), which")
        print("  covers every observed shape variant of that (op, body) pair.")
        print("  Greedy: pick highest-impact signature first, accumulate to 100%.")
        print()
        # Collapse Discoveries by (op_name, body_label) — multiple shape
        # variants share a single dynamic-shape kernel.
        from collections import defaultdict as _dd

        by_kernel: dict[tuple[str, str], dict] = _dd(lambda: {"dispatches": 0, "impact": 0, "shapes": set()})
        for d in discoveries:
            key = (d.op_name, d.body_label)
            by_kernel[key]["dispatches"] += d.occurrences
            by_kernel[key]["impact"] += d.occurrences * (d.elements_per_op or 1)
            by_kernel[key]["shapes"].add(d.out_type)
        ranked = sorted(by_kernel.items(), key=lambda kv: -kv[1]["impact"])

        total_dispatches = sum(v["dispatches"] for v in by_kernel.values())
        total_impact = sum(v["impact"] for v in by_kernel.values())
        cum_d = 0
        cum_i = 0
        print(f"  {'#':>2}  {'cov%':>5}  {'cum_disp':>9}  " f"{'shapes':>6}  signature")
        for k, ((op_name, body_label), info) in enumerate(ranked, start=1):
            cum_d += info["dispatches"]
            cum_i += info["impact"]
            cov_pct = 100.0 * cum_i / total_impact if total_impact else 0.0
            print(
                f"  {k:>2}  {cov_pct:>4.1f}%  "
                f"{cum_d:>5}/{total_dispatches:<3}  "
                f"{len(info['shapes']):>6}  {op_name}"
            )
            if cov_pct >= 99.99:
                print()
                print(f"  ──→ {k} kernels = 100% coverage of dronet's compute")
                print(
                    f"       (covers {cum_d} dispatches across "
                    f"{sum(len(v['shapes']) for _, v in ranked[:k])} shape variants)"
                )
                break

    if args.auto_fuse:
        print()
        print("--auto-fuse: also compiling to flow phase to detect IREE-fused dispatches...")
        try:
            flow_ir = run_flow(
                args.input_mlir.resolve(),
                args.target,
                args.hw,
                args.iree_compile_arg,
            )
        except RuntimeError as e:
            print(f"  flow-phase compile failed: {e}", file=sys.stderr)
            return 0
        fused = discover_flow(flow_ir)
        if not fused:
            print("  No fused dispatches detected.")
            return 0
        write_fused_stubs(fused, args.output.resolve(), args.target_key)
        print()
        print(f"Fused dispatches detected at flow phase ({len(fused)} unique signatures):")
        print("  Each fused dispatch is ONE executable — implementing it as a single")
        print("  kernel means one read/write per element across the whole chain.")
        print()
        for f in sorted(fused, key=lambda f: -f.occurrences):
            chain_str = " → ".join(f.body_chain) if f.body_chain else "(no scalar body ops)"
            print(f"  {f.occurrences:3}x  {f.dispatch_name:35s}  body: {chain_str}")
    return 0


def write_fused_stubs(
    fused: list[FusedDispatch],
    out_dir: pathlib.Path,
    target_key: str,
) -> None:
    """Emit one stub per unique fused dispatch signature plus FUSED_STUBS.md."""
    fused_dir = out_dir / "fused_stubs"
    (fused_dir / "abi").mkdir(parents=True, exist_ok=True)
    md = [
        "# Fused-dispatch kernel stubs",
        "",
        "Each entry below is one IREE-fused dispatch detected at the flow",
        "phase. Implementing one C kernel per signature replaces N preprocessing-",
        "level kernel calls with one fused call — fewer dispatches, less memory",
        "bandwidth (one read/write per element across the entire chain).",
        "",
        "| Occurrences | Dispatch family | Body chain | Inputs | Outputs |",
        "|---:|---|---|---|---|",
    ]
    for f in sorted(fused, key=lambda f: -f.occurrences):
        chain_str = " → ".join(f.body_chain) if f.body_chain else "(none)"
        in_str = ", ".join(f"`{t}`" for t in f.in_types)
        out_str = ", ".join(f"`{t}`" for t in f.out_types)
        md.append(f"| {f.occurrences} | `{f.dispatch_name}` | {chain_str} " f"| {in_str} | {out_str} |")
        # Emit a per-signature C stub.
        sym = re.sub(r"[^a-z0-9_]", "_", f"fused_{f.dispatch_name}_{'_'.join(f.body_chain)}")
        stub_path = fused_dir / "abi" / f"{sym}.c"
        if not stub_path.exists():
            stub_path.write_text(_emit_fused_c_stub(f, sym))
    md.append("")
    md.append("To wire one of these into the live manifest:")
    md.append("1. Implement the body in `fused_stubs/abi/<name>.c`.")
    md.append("2. Author a multi-op `match.mlir` capturing the equivalent")
    md.append("   preprocessing-level chain (one `linalg.generic` per body op,")
    md.append("   connected via SSA producer-consumer chain).")
    md.append('3. Add a manifest entry with `match.kind: "linalg_dag"` pointing')
    md.append("   at the new `match.mlir`.")
    md.append("4. Run `./merlin compile … --kernels-dir <dir> --kernels-strict-coverage`")
    md.append("   and confirm the fused kernel fires (look for one `util.call @call_*`")
    md.append("   replacing what was previously N separate calls).")
    (out_dir / "FUSED_STUBS.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {len(fused)} fused-dispatch stubs to {fused_dir}/")
    print(f"  📋 see {out_dir}/FUSED_STUBS.md for the chain-by-chain breakdown")


def _emit_fused_c_stub(f: FusedDispatch, sym: str) -> str:
    in_decl = "\n".join(f"//   binding{i}: {t}  (input)" for i, t in enumerate(f.in_types))
    out_decl = "\n".join(f"//   binding{len(f.in_types) + i}: {t}  (output)" for i, t in enumerate(f.out_types))
    chain = "\n".join(f"//   {op}" for op in f.body_chain)
    return (
        f"// AUTO-GENERATED FUSED stub for IREE-fused dispatch "
        f"`{f.dispatch_name}` (observed {f.occurrences}x).\n"
        f"//\n"
        f"// Bindings:\n{in_decl}\n{out_decl}\n"
        f"//\n"
        f"// Body op chain (executed per output element in IREE's fused form):\n"
        f"{chain}\n"
        f"//\n"
        f"// To implement: one workgroup per output element; each invocation\n"
        f"// reads each input once, runs the chain, writes the output once.\n"
        f"// This replaces N separate preprocessing-level kernel calls (each\n"
        f"// passing the full activation through DRAM) with ONE pass.\n"
        f"\n"
        f"#include <stddef.h>\n\n"
        f'__attribute__((visibility("default")))\n'
        f"void {sym}(/* TODO: per-binding (ptr, offset) + dims + tid */) {{\n"
        f"  // TODO: implement fused chain\n"
        f"}}\n"
    )


if __name__ == "__main__":
    sys.exit(main())
