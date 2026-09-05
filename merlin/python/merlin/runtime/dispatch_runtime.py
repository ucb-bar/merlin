"""Execute a whole model through the Merlin dispatch table (host reference runtime).

This is the capstone of the outliner path: it runs an **entire** captured model by

1. outlining it into per-dispatch kernels (``merlin-outline-dispatches``),
2. compiling each kernel func in isolation to a host ``.so`` (``llvmlower.kernel_backend``,
   deduplicated by kernel body so identical-shape layers compile once), and
3. walking the driver, evaluating the cheap view/glue ops (``expand_shape``,
   ``collapse_shape``, ``extract_slice``, ``concat``, ``splat``, ``constant``) directly in
   numpy and invoking each compiled kernel symbol in order.

The forward arguments (input activations + weights) are bound exactly as the C runtime
binds them — inputs from ``inputs.npz``, weights read from the ``safetensors`` blob via the
manifest ``data_offsets`` (no safetensors library needed). The whole-model output is then
gated against the captured torch golden, proving the outlined dispatch table is faithful
end to end — the same numbers the monolithic compile produces, but through the unified
per-kernel path.

This is the Python reference executor; the deployable C runtime walks the same dispatch
program. Only the view-op semantics needed by the captured models are implemented; an
unrecognized op raises (no silent skips).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_NP = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
       "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8, "i1": np.int8}


class DispatchRuntimeError(RuntimeError):
    pass


# bf16 has no native numpy dtype; we store it as the raw 16-bit pattern (uint16) so the
# compiled kernels (which read/write bf16) see the right bytes, and the numpy view ops
# (reshape/concat/slice — pure data movement) stay byte-correct. Float arithmetic on bf16
# only happens inside compiled kernels, never in the numpy view evaluator.
def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF              # round to nearest even
    return ((u + bias) >> 16).astype(np.uint16)


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f8e4m3fn_to_f32(u8: np.ndarray) -> np.ndarray:
    """Decode OCP float8_e4m3fn (1 sign, 4 exp bias-7, 3 mantissa; no inf, NaN = S.1111.111).

    torchao's ``float8_weight_only_e4m3`` stores weights as ``F8_E4M3`` in the safetensors;
    numpy has no fp8 dtype, so the bytes must be decoded to f32 at load (the MLIR forward arg
    is f32 — the graph multiplies by a separate f32 scale, weight-only style).
    """
    u = np.ascontiguousarray(u8, np.uint8).astype(np.uint32)
    sign = np.where((u >> 7) & 1 == 1, np.float32(-1.0), np.float32(1.0))
    exp = (u >> 3) & 0xF
    man = (u & 0x7).astype(np.float32)
    sub = (man / 8.0) * np.float32(2.0 ** -6)                  # exp == 0: subnormal
    nrm = (1.0 + man / 8.0) * np.exp2((exp.astype(np.float32)) - 7.0)
    val = (sign * np.where(exp == 0, sub, nrm)).astype(np.float32)
    return np.where((exp == 0xF) & (u & 0x7 == 0x7), np.float32(np.nan), val)


def _np_dtype(elem: str):
    """numpy storage dtype for an MLIR element type (bf16 stored as uint16 bit pattern)."""
    if elem == "bf16":
        return np.uint16
    return _NP.get(elem, np.float32)


# --- forward-argument binding (inputs + weights), mirroring llvmlower.c_runtime ----------

def resolve_forward_args(model_dir: str | Path) -> list[np.ndarray]:
    """Numpy array for every ``func @forward`` argument, in signature order."""
    from ..llvmlower.model_runner import parse_forward_signature
    from ..llvmlower.weights_pack import load_safetensors_header

    model_dir = Path(model_dir)
    sig = parse_forward_signature(model_dir / "model.mlir")
    man = json.loads((model_dir / "weights.safetensors.manifest.json").read_text())
    hdr, payload_off = load_safetensors_header(model_dir / "weights.safetensors")
    blob = (model_dir / "weights.safetensors").read_bytes()[payload_off:]
    inputs = np.load(model_dir / "inputs.npz")
    extra_path = model_dir / "extra.npz"
    extra = np.load(extra_path) if extra_path.is_file() else None
    extra_files = list(getattr(extra, "files", []))
    lifted_names = sorted(k for k in extra_files if not k.startswith("buf::"))
    # Per-model input map (manifest arg-name -> inputs.npz index), emitted by the consistent
    # capture; falls back to the legacy fixed map + positional order when absent.
    order_path = model_dir / "input_order.json"
    if order_path.is_file():
        input_order = {k: int(v) for k, v in json.loads(order_path.read_text()).items()}
    else:
        input_order = {"img": 0, "img_mask": 1, "lang_tokens": 2, "lang_masks": 3,
                       "state": 4, "noise": 5, "ids": 0}

    def buffer_array(name: str) -> np.ndarray:
        for k in extra_files:
            if k.startswith("buf::") and "b_" + k[len("buf::"):].replace(".", "_") == name:
                return np.ascontiguousarray(extra[k])
        raise DispatchRuntimeError(f"buffer {name!r} not in {extra_path}")

    args: list[np.ndarray] = []
    n_in = 0
    li = 0
    for i, (shape, dt) in enumerate(sig):
        meta = man[str(i)]
        if meta["kind"] == "param":
            # Quantized-subclass weights are stubbed in the blob (the fused fp32 weight ARG is
            # dead -- the matmul consumes int8 int_data/scale via the qinner channel). Synthesize
            # a zero buffer of the arg's true shape/dtype for the dead descriptor.
            if meta.get("stub"):
                args.append(np.zeros(shape, dtype=_np_dtype(dt)))
                continue
            begin, end = hdr[meta["weight"]]["data_offsets"]
            store_dt = hdr[meta["weight"]].get("dtype")            # safetensors storage dtype
            if store_dt == "F8_E4M3":
                # fp8 weight-only: stored 1-byte e4m3, decoded to the f32 arg the graph expects
                arr = f8e4m3fn_to_f32(np.frombuffer(blob[begin:end], np.uint8)).reshape(shape)
            else:
                arr = np.frombuffer(blob[begin:end], dtype=_np_dtype(dt)).reshape(shape)
            args.append(np.ascontiguousarray(arr))
            continue
        name = meta.get("name", "") or ""
        if meta["kind"] == "buffer":
            # A registered buffer reaches us one of two ways, and only one of them was handled.
            # When the capture EXTERNALIZED it, the manifest names it under "weight" and the
            # bytes are in the safetensors blob like any parameter (BatchNorm running stats, for
            # instance). When it stayed a graph argument, the manifest carries the FX arg "name"
            # and the value is in extra.npz. Prefer the blob when it has the tensor; otherwise
            # fall back to extra.npz -- so neither capture shape is a missing-buffer error.
            wname = meta.get("weight")
            if wname and wname in hdr:
                begin, end = hdr[wname]["data_offsets"]
                arr = np.ascontiguousarray(
                    np.frombuffer(blob[begin:end], dtype=_np_dtype(dt)).reshape(shape))
            else:
                arr = buffer_array(name)
        elif "lifted_tensor" in name:          # lifted get_attr constant (name varies by model)
            arr = np.ascontiguousarray(extra[lifted_names[li]]); li += 1
        elif name in input_order and f"in{input_order[name]}" in inputs.files:
            arr = np.ascontiguousarray(inputs[f"in{input_order[name]}"])
        else:
            arr = np.ascontiguousarray(inputs[f"in{n_in}"]); n_in += 1
        if dt == "bf16":
            args.append(f32_to_bf16(arr).reshape(shape))
        else:
            args.append(arr.astype(_np_dtype(dt)).reshape(shape))
    return args


# --- view-op evaluation ------------------------------------------------------------------

_DYN = -9223372036854775808            # xDSL dynamic-dim sentinel


def _tensor_shape(t) -> tuple[int, ...]:
    from xdsl.dialects.builtin import TensorType

    return tuple(int(d) for d in t.get_shape()) if isinstance(t, TensorType) else ()


def _reshape_shape(rtype) -> tuple[int, ...]:
    """Result shape with dynamic dims as -1 (numpy infers the single unknown)."""
    return tuple(-1 if d <= _DYN // 2 else d for d in _tensor_shape(rtype))


def _elem_str(t) -> str:
    from xdsl.dialects.builtin import TensorType

    et = t.element_type if isinstance(t, TensorType) else t
    return str(et)


def _scalar_dtype(t):
    return _np_dtype(_elem_str(t))


def _fill_value(value, elem: str):
    """Scalar fill value in the storage dtype (bf16 -> its 16-bit pattern as an int)."""
    if elem == "bf16":
        return int(f32_to_bf16(np.array([float(value)], np.float32))[0])
    return value


def _const_value(op):
    val = op.properties["value"]
    data = getattr(val, "value", None)
    if data is not None and hasattr(data, "data"):
        return data.data
    # dense elements constant
    rtype = op.results[0].type
    arr = np.array(list(val.get_values())).reshape(_tensor_shape(rtype))
    if _elem_str(rtype) == "bf16":
        return f32_to_bf16(arr)
    return arr.astype(_scalar_dtype(rtype))


def _pad_value(op, env):
    """The scalar constant a tensor.pad fills with (its region's tensor.yield)."""
    for region in op.regions:
        for block in region.blocks:
            for sub in block.ops:
                if sub.name in ("tensor.yield", "linalg.yield"):
                    v = sub.operands[0]
                    if id(v) in env:
                        return env[id(v)]
                    owner = getattr(v, "owner", None)
                    if owner is not None and owner.name == "arith.constant":
                        return _const_value(owner)
    return 0


def _int_array(attr) -> list[int]:
    return [int(v) for v in attr.get_values()]


def _pad_rank(a: list[int], fill: int, rank: int) -> list[int]:
    """Left-pad a rank-reduced static offset/size/stride array to the source rank."""
    return [fill] * (rank - len(a)) + a if len(a) < rank else a


# Scalar arith evaluated directly (driver-level index/condition glue, not compute kernels).
_CMPI_PRED = {0: lambda a, b: a == b, 1: lambda a, b: a != b,
              2: lambda a, b: a < b, 3: lambda a, b: a <= b,
              4: lambda a, b: a > b, 5: lambda a, b: a >= b,
              6: lambda a, b: a < b, 7: lambda a, b: a <= b,
              8: lambda a, b: a > b, 9: lambda a, b: a >= b}
_SCALAR_ARITH = {
    "arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.remsi",
    "arith.maxsi", "arith.minsi", "arith.andi", "arith.ori", "arith.xori",
    "arith.cmpi", "arith.select", "arith.extsi", "arith.extui", "arith.trunci",
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.sitofp", "arith.fptosi",
}


def _eval_scalar_arith(op, env):
    name = op.name
    a = env[id(op.operands[0])] if op.operands else None
    if name == "arith.select":
        return env[id(op.operands[1])] if a else env[id(op.operands[2])]
    if name in ("arith.extsi", "arith.extui", "arith.trunci", "arith.fptosi"):
        return int(a)
    if name == "arith.sitofp":
        return float(a)
    b = env[id(op.operands[1])]
    if name == "arith.cmpi":
        return bool(_CMPI_PRED[int(op.properties["predicate"].value.data)](a, b))
    return {"arith.addi": lambda: a + b, "arith.subi": lambda: a - b,
            "arith.muli": lambda: a * b, "arith.divsi": lambda: int(a / b),
            "arith.remsi": lambda: int(a - b * int(a / b)),
            "arith.maxsi": lambda: max(a, b), "arith.minsi": lambda: min(a, b),
            "arith.andi": lambda: a & b, "arith.ori": lambda: a | b,
            "arith.xori": lambda: a ^ b,
            "arith.addf": lambda: a + b, "arith.subf": lambda: a - b,
            "arith.mulf": lambda: a * b, "arith.divf": lambda: a / b}[name]()


def _eval_view(op, env: dict[int, Any]):
    name = op.name
    rtype = op.results[0].type
    rshape = _tensor_shape(rtype)
    if name == "arith.constant":
        return _const_value(op)
    if name == "tensor.empty":
        # dynamic dims (tensor<?x...>) take their size from the op's SSA operands, in order
        dyn = list(op.operands)
        j = 0
        shape = []
        for d in _tensor_shape(rtype):
            if d <= _DYN // 2:
                shape.append(int(env[id(dyn[j])])); j += 1
            else:
                shape.append(d)
        return np.zeros(shape, _scalar_dtype(rtype))
    if name in ("tensor.splat", "linalg.fill"):
        elem = _elem_str(rtype)
        return np.full(rshape, _fill_value(env[id(op.operands[0])], elem),
                       dtype=_np_dtype(elem))
    if name in ("tensor.collapse_shape", "tensor.expand_shape"):
        return np.ascontiguousarray(env[id(op.operands[0])]).reshape(_reshape_shape(rtype))
    if name == "tensor.concat":
        dim = int(op.properties["dim"].value.data)
        return np.concatenate([env[id(o)] for o in op.operands], axis=dim)
    if name == "tensor.pad":
        src = env[id(op.operands[0])]
        low = _int_array(op.properties["static_low"])
        high = _int_array(op.properties["static_high"])
        if op.operands[1:]:                       # dynamic pad amounts not supported
            raise DispatchRuntimeError("dynamic tensor.pad amounts not supported")
        return np.pad(src, list(zip(low, high)), mode="constant",
                      constant_values=_pad_value(op, env))
    if name == "tensor.extract_slice":
        seg = _int_array(op.properties["operandSegmentSizes"])
        if seg[1:] != [0, 0, 0]:
            raise DispatchRuntimeError("dynamic extract_slice not supported")
        src = env[id(op.operands[0])]
        rank = src.ndim
        # model2MLIR can emit rank-reduced static arrays (only the kept dims). Left-pad to
        # source rank — offsets 0, sizes 1, strides 1 — matching passes_xdsl._fix_extract_slice.
        offs = _pad_rank(_int_array(op.properties["static_offsets"]), 0, rank)
        sizes = _pad_rank(_int_array(op.properties["static_sizes"]), 1, rank)
        strides = _pad_rank(_int_array(op.properties["static_strides"]), 1, rank)
        idx = tuple(slice(o, o + s * st, st) for o, s, st in zip(offs, sizes, strides))
        return np.ascontiguousarray(src[idx]).reshape(_reshape_shape(rtype))
    if name == "tensor.insert_slice":
        seg = _int_array(op.properties["operandSegmentSizes"])
        if seg[2:] != [0, 0, 0]:
            raise DispatchRuntimeError("dynamic insert_slice not supported")
        src = env[id(op.operands[0])]
        dst = np.array(env[id(op.operands[1])], copy=True)
        rank = dst.ndim
        offs = _pad_rank(_int_array(op.properties["static_offsets"]), 0, rank)
        sizes = _pad_rank(_int_array(op.properties["static_sizes"]), 1, rank)
        strides = _pad_rank(_int_array(op.properties["static_strides"]), 1, rank)
        idx = tuple(slice(o, o + s * st, st) for o, s, st in zip(offs, sizes, strides))
        dst[idx] = np.asarray(src).reshape(dst[idx].shape)
        return dst
    if name in ("arith.index_cast", "arith.index_castui"):
        return int(env[id(op.operands[0])])
    # scalar integer/float arith (loop-index / mask / condition glue in the driver)
    if name in _SCALAR_ARITH:
        return _eval_scalar_arith(op, env)
    if name == "tensor.extract":
        src = env[id(op.operands[0])]
        idx = tuple(int(env[id(o)]) for o in op.operands[1:])
        return src[idx]
    if name == "tensor.insert":
        scalar = env[id(op.operands[0])]
        dst = np.array(env[id(op.operands[1])], copy=True)
        idx = tuple(int(env[id(o)]) for o in op.operands[2:])
        dst[idx] = scalar
        return dst
    if name == "tensor.from_elements":
        elem = _elem_str(rtype)
        vals = [env[id(o)] for o in op.operands]
        if elem == "bf16":
            return f32_to_bf16(np.array(vals, np.float32)).reshape(_reshape_shape(rtype))
        return np.array(vals, dtype=_np_dtype(elem)).reshape(_reshape_shape(rtype))
    raise DispatchRuntimeError(f"no runtime evaluation for view op {name}")


# --- the executor ------------------------------------------------------------------------

def _kernel_io(call):
    """(callee symbol, output (shape,dtype) list) for a func.call."""
    callee = call.callee.string_value() if hasattr(call.callee, "string_value") \
        else call.callee.root_reference.data
    outs = [(_tensor_shape(r.type), _scalar_dtype(r.type)) for r in call.results]
    return callee, outs


def _compile_kernel_so(args: tuple) -> str:
    """Process-pool worker: lower a kernel's MLIR text to a host ``.so`` at ``so_path``.

    Runs in a separate process (xDSL's parser/lowering keeps mutable global state and is NOT
    thread-safe — concurrent parses race and corrupt op-building), so each kernel compiles in
    an isolated interpreter. Takes/returns only strings (picklable)."""
    import shutil
    from pathlib import Path

    from ..llvmlower.lower import lower_model

    text, workdir_str, so_path = args
    wd = Path(workdir_str)
    lower_model(text, wd, targets=("host",))
    shutil.copy2(wd / "model_host.so", so_path)
    return so_path


def mesh_datapath(target: str):
    """``target``'s mesh datapath as its whole ``corpus_spec.CorpusBinding``, DERIVED.

    The mesh boundary below used to be written as ``operand_dtype="int8", accum_dtype="i32"`` with an
    element-type filter of exactly ``("f32", "i8")``. Both are facts about ONE target's datapath, and on a
    float-datapath accelerator they are wrong in the two ways that cancel out into silence: the filter
    never classifies an fp8 layer, so no layer is ever routed, so the hardcoded dtypes are never reached
    and never observed to be wrong. The whole model runs on the host and the run looks clean.

    This RETURNS THE BINDING rather than a hand-picked tuple of four of its fields, and that is the point.
    The projection it used to perform is how ``subnormal_operand_flush`` -- a property already derived from
    the target's RTL, already declared in its profile, already honoured by the capsule golden engine -- was
    dropped on the floor between the deriver and the one consumer that feeds the mesh REAL model operands.
    The fact existed and was correct; nothing carried it the last few lines. Every lossy projection of a
    derived-facts object is that same bug waiting to be written again, so the projection is gone: callers
    take the fields they need off the binding, and a field added later reaches them without a signature
    change. The MLIR spelling the filter wants is ``b.mlir_dtype(b.operand_dtype)``.
    """
    from ..compile_cli import _mesh_tile_binding
    return _mesh_tile_binding(target, None, None)


def boundary_scale(x, fmt: str) -> float:
    """The power-of-two ``s`` placing ``max|x| / s`` just inside ``fmt``'s largest finite magnitude.

    A power of two is the only scale that is FREE here: it shifts the exponent and leaves the mantissa
    untouched, so ``x/s`` rounds into ``fmt`` exactly as ``x`` would have, and multiplying the result back
    afterwards is lossless. Any other factor would inject its own rounding, and we would be grading the
    device against our arithmetic instead of its own.

    Returns 1.0 for an all-zero or non-finite operand -- there is nothing to place.
    """
    from . import fp8_formats as FF
    mx = float(np.abs(x).max()) if x.size else 0.0
    if not (mx > 0.0) or not np.isfinite(mx):
        return 1.0
    _min_normal, max_finite = FF.normal_range(fmt)
    return float(2.0 ** math.ceil(math.log2(mx / max_finite)))


def _representability(x, fmt: str, *, flush: bool) -> dict:
    """How much of ``x`` the operand format can actually carry: values below its smallest normal lose
    mantissa bits (or become zero outright where the datapath flushes them), values above its largest
    finite saturate. Counted and REPORTED rather than assumed away -- an operand the hardware cannot hold
    is a silent wrong answer, and the only reason this went unnoticed for so long is that nobody counted."""
    from . import fp8_formats as FF
    min_normal, max_finite = FF.normal_range(fmt)
    ax = np.abs(np.asarray(x, dtype=np.float64))
    subnormal = int(np.count_nonzero((ax > 0.0) & (ax < min_normal)))
    return {"n_values": int(ax.size),
            "n_subnormal": subnormal,
            "n_flushed_to_zero": subnormal if flush else 0,
            "n_saturating": int(np.count_nonzero(ax > max_finite))}


def float_boundary_operands(a, b, binding) -> tuple[list, list, float, dict]:
    """Operands for a FLOAT mesh datapath, scaled into the format's normal range, plus the scale to undo
    afterwards and a representability record.

    Handing the mesh raw model tensors is what made atlas diverge. A typical layer's weights peak around
    0.088, e4m3's smallest normal is 0.015625, and the MXU flushes a subnormal operand to zero
    (``E4M3Mul.scala``: ``aZero := aExp === 0.U``) -- so most of two layers' weights arrived at the
    multiplier as zeros and the model quietly lost 26% of them. Scaling by a power of two first costs
    nothing and moves the whole tensor into the range the format represents properly.

    Measured over all 15 real atlas layers: mean cos 0.99567 -> 0.99914, worst 0.98765 -> 0.99896, mean
    rel-L2 0.0901 -> 0.0420. That 0.0420 IS the floor -- a control run with subnormal flushing disabled
    entirely scores 0.04220, and per-row/per-column scaling scores the same 0.04195 as per-tensor. What is
    left is e4m3's 3-bit mantissa plus bf16 accumulation, which no boundary can scale away.
    """
    from . import fp8_formats as FF
    fmt = binding.operand_dtype
    af, bf = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    sa, sb = boundary_scale(af, fmt), boundary_scale(bf, fmt)

    # Scaling the operands up scales their products up with them. On a narrow accumulator that trades an
    # operand underflow for an accumulator overflow, which is strictly worse -- fp16 tops out at 65504,
    # and a k-deep reduction of two near-cap fp8 operands passes that in a handful of terms. Back off in
    # whole powers of two (still exact), split between the two operands so neither is pushed back down
    # further than the other, and let the representability record say what the backoff cost.
    ma = float(np.abs(af).max()) if af.size else 0.0
    mb = float(np.abs(bf).max()) if bf.size else 0.0
    try:
        _min_acc, max_acc = FF.normal_range(binding.accum_dtype)
    except KeyError:                                   # integer or unresolvable accumulator: no float cap
        max_acc = None
    if max_acc is not None and ma > 0.0 and mb > 0.0:
        k = max(1, int(af.shape[1])) if af.ndim == 2 else 1
        headroom = max_acc / 2.0                       # half the range, so the sum has somewhere to land
        worst = (ma / sa) * (mb / sb) * k
        if np.isfinite(worst) and worst > headroom:
            back = math.ceil(math.log2(worst / headroom))
            sa *= float(2.0 ** ((back + 1) // 2))
            sb *= float(2.0 ** (back // 2))

    qa, qb = af / sa, bf / sb
    flush = bool(getattr(binding, "subnormal_operand_flush", False))
    rec = {"operand_dtype": fmt, "scale_a": sa, "scale_b": sb,
           "a": _representability(qa, fmt, flush=flush),
           "b": _representability(qb, fmt, flush=flush)}
    return qa.tolist(), qb.tolist(), sa * sb, rec


def _has_contraction(kfn) -> bool:
    """True when a kernel func contains a contraction op at all — used to count the matmuls the mesh
    classifier declined, structurally (no name matching on the symbol)."""
    for op in kfn.walk():
        nm = getattr(op, "name", "")
        if nm in ("linalg.matmul", "linalg.batch_matmul", "linalg.matmul_transpose_b",
                  "linalg.quantized_matmul"):
            return True
    return False


def _classify_mesh_matmul(kfn, accept: tuple[str, ...] = ("f32",)) -> dict | None:
    """Like xnnpack's classify_matmul_kernel but for the MESH route: a plain 2-D ``linalg.matmul`` whose two
    ``ins`` are kernel block args, with an element type this target's mesh can take.

    ``accept`` is derived from the target's own declared operand format (plus ``f32``, the host lane's
    spelling, which the boundary converts). It was the literal pair ``("f32", "i8")``, which silently
    routed nothing at all on any target whose datapath is not int8. Returns ``{"a", "b", "in_dtype"}`` so
    the boundary knows whether the layer arrives already in the mesh's format or needs converting."""
    from xdsl.dialects.builtin import TensorType
    block = kfn.body.blocks[0]
    arg_ids = {id(a): i for i, a in enumerate(block.args)}
    ret = next((o for o in block.ops if o.name == "func.return"), None)
    if ret is None or len(ret.operands) != 1:
        return None
    mm = getattr(ret.operands[0], "owner", None)
    if mm is None or getattr(mm, "name", None) != "linalg.matmul" or len(mm.operands) < 3:
        return None
    a_val, b_val = mm.operands[0], mm.operands[1]

    def _et(v):
        t = v.type
        return str(t.element_type) if isinstance(t, TensorType) and len(t.get_shape()) == 2 else None

    ea, eb = _et(a_val), _et(b_val)
    if ea is None or ea != eb or ea not in accept:
        return None
    if id(a_val) not in arg_ids or id(b_val) not in arg_ids:
        return None                                          # an operand is computed inside the kernel
    return {"a": arg_ids[id(a_val)], "b": arg_ids[id(b_val)], "in_dtype": ea}


#: the canonical 2-D contraction iterator signature: (i, j) parallel over a k reduction.
_MATMUL_ITERS = ("parallel", "parallel", "reduction")


def _is_widening_mac_body(gen) -> bool:
    """True when a ``linalg.generic`` body is a (widening) multiply-accumulate: optional operand
    extensions, then one multiply, one add, and the yield.

    Structural, over the IR's own op vocabulary — not a pattern match on text. Accepts the integer
    (``muli``/``addi``) and float (``mulf``/``addf``) spellings alike, so the same check covers a
    quantized i8xi8->i32 contraction and an f32 one."""
    blocks = getattr(getattr(gen, "body", None), "blocks", None)
    if not blocks:
        return False
    names = [o.name for o in blocks[0].ops]
    if not names or names[-1] != "linalg.yield":
        return False
    core = [n for n in names[:-1] if not n.startswith("arith.ext")]
    return core in (["arith.muli", "arith.addi"], ["arith.mulf", "arith.addf"])


def _classify_mesh_contraction(kfn, accept: tuple[str, ...] = ("f32",)) -> dict | None:
    """The ``linalg.generic`` form of :func:`_classify_mesh_matmul`.

    Quantization rewrites every ``linalg.matmul`` into a generic — measured on an int8 whole model, 0 of
    325 outlined kernels still return a ``linalg.matmul``, and the matmul-only classifier therefore routed
    NOTHING to the mesh: 0 layers on the accelerator, 0 recorded as fallbacks, and a capstone that read as
    a pass. A quantized contraction is exactly what an integer mesh exists for, so it must be recognised.

    Requires the canonical matmul indexing maps, not merely the right iterator shape: a transposed or
    otherwise permuted contraction has the same iterators and computes something else, and routing it to
    the mesh would silently produce the wrong answer. Fails closed when the maps cannot be read."""
    from xdsl.dialects.builtin import TensorType
    block = kfn.body.blocks[0]
    arg_ids = {id(a): i for i, a in enumerate(block.args)}
    ret = next((o for o in block.ops if o.name == "func.return"), None)
    if ret is None or len(ret.operands) != 1:
        return None
    gen = getattr(ret.operands[0], "owner", None)
    if gen is None or getattr(gen, "name", None) != "linalg.generic" or len(gen.inputs) != 2:
        return None
    if tuple(str(getattr(a, "data", a)).rsplit(".", 1)[-1].lower()
             for a in gen.iterator_types) != _MATMUL_ITERS:
        return None
    if not _is_widening_mac_body(gen):
        return None

    a_val, b_val = gen.inputs[0], gen.inputs[1]

    def _et(v):
        t = v.type
        return str(t.element_type) if isinstance(t, TensorType) and len(t.get_shape()) == 2 else None

    ea, eb = _et(a_val), _et(b_val)
    if ea is None or ea != eb or ea not in accept:
        return None
    if isinstance(gen.results[0].type, TensorType) and len(gen.results[0].type.get_shape()) != 2:
        return None
    if id(a_val) not in arg_ids or id(b_val) not in arg_ids:
        return None

    # the maps must be exactly (d0,d2), (d2,d1) -> (d0,d1)
    try:
        maps = [m.data for m in gen.indexing_maps]
        got = tuple(tuple(str(r) for r in mp.results) for mp in maps)
    except Exception:                                        # noqa: BLE001 — unreadable maps: fail closed
        return None
    if got != (("d0", "d2"), ("d2", "d1"), ("d0", "d1")):
        return None
    return {"a": arg_ids[id(a_val)], "b": arg_ids[id(b_val)], "in_dtype": ea,
            "via": "generic"}


def _oracle_unreachable(decline: str) -> bool:
    """Whether a mesh decline describes our MEASUREMENT failing rather than the backend refusing.

    A timed-out simulator, a missing binary, a build that did not produce an oracle: none of these say
    anything about whether the mesh can run the layer. Treating them as refusals turned a whole model
    whose every layer the mesh executes correctly into "15 of 15 fell back to the host", and that number
    then failed a must_accelerate gate -- a compiler blamed for a simulator budget.

    Matched structurally on the oracle's own words (no regex, repo rule); anything unrecognized is
    treated as a genuine refusal, which is the conservative direction: an unknown cause must not be able
    to excuse a real fallback."""
    d = (decline or "").lower()
    if not d:
        return False
    return any(tok in d for tok in ("timed out", "timeout", "invocation failed",
                                    "no such file", "not found", "unreachable",
                                    "build failed", "oracle unavailable"))


def execute(outline_result, arg_arrays: list[np.ndarray], workdir: str | Path,
            entry: str = "forward", cache_dir: str | Path | None = None,
            tap=None, qinner: dict | None = None,
            kernel_backend: str | None = None, mesh_target: str | None = None,
            mesh_package: str | None = None,
            counters: dict | None = None) -> list[np.ndarray]:
    """Run the outlined model on bound arguments; return the driver's result arrays.

    ``cache_dir`` persists compiled kernel ``.so``s keyed by kernel-body hash, so repeated
    runs (debugging, the multi-step action loop) skip recompilation. ``tap(op, value)`` is
    an optional callback invoked after every node with its produced value — the per-op
    bisection hook.
    """
    import hashlib

    from ..llvmlower.abi import HostModel
    from ..llvmlower.kernel_backend import compile_host, extract_kernel
    from ..xdsl_dialects._common import text as to_text

    workdir = Path(workdir)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    module = outline_result.module
    driver = next(op for op in module.walk()
                  if op.name == "func.func" and op.sym_name.data == entry)
    block = driver.body.blocks[0]

    env: dict[int, Any] = {}
    for arg, arr in zip(block.args, arg_arrays):
        env[id(arg)] = arr

    cache: dict[str, Any] = {}      # kernel body text -> HostModel
    compiled: dict[str, Any] = {}   # symbol -> HostModel
    n_unique = 0

    # --- optional XNNPACK kernel-backend routing (default-off) -----------------------------
    # Classify each kernel func once: if it is a plain 2-D f32 linalg.matmul, record the
    # block-arg indices of its A/B operands so run_call can compute it through the XNNPACK
    # host microkernel instead of the Merlin-compiled .so. All other kernels are unaffected.
    xnn_routes: dict[str, dict] = {}
    if kernel_backend == "xnnpack":
        from .backends import xnnpack_host

        if not xnnpack_host.is_available():
            raise DispatchRuntimeError(
                "kernel_backend='xnnpack' requested but the XNNPACK host GEMM lib could not "
                "be built (see MERLIN_XNNPACK_REPO / tmp/kernels/XNNPACK)")
        kfns = {op.sym_name.data: op for op in module.walk()
                if op.name == "func.func" and "$kernel_" in op.sym_name.data}
        for sym, kfn in kfns.items():
            route = xnnpack_host.classify_matmul_kernel(kfn)
            if route is not None:
                xnn_routes[sym] = route
        execute.last_xnn_routed = len(xnn_routes)

    # --- mesh routing: run each matmul kernel on the target accelerator MESH (real activations) ----------
    # Same structural classifier picks the plain matmul kernels; each is executed on the target's mesh
    # oracle via operand injection (run_matmul_on_mesh) with the LIVE activation/weight arrays, and the
    # mesh output threads forward — so the whole model runs with its matmul layers on the real systolic
    # array. A layer that does not already arrive in the mesh's own operand format is converted at the
    # boundary: quantized for an integer datapath (the drop is expected quantization error, covered by the
    # whole-model tolerance), cast for a float one. Which of those applies is DERIVED from the target.
    mesh_routes: dict[str, dict] = {}
    mesh_dp = None                                   # the target's CorpusBinding, whole (see mesh_datapath)
    # Kernel bodies by symbol, for the host-lane contraction count below. Built LAZILY and cached: the
    # two existing `kfns` maps are each bound inside a branch (the xnnpack route, and
    # `kernel_backend == "mesh"`), so neither exists on a plain host run -- reaching for one there is a
    # NameError, not a missing count. One walk, and only if a host kernel actually executes.
    _host_kfn_cache: dict = {}

    def _host_kernel_fn(sym: str):
        if not _host_kfn_cache:
            _host_kfn_cache.update({o.sym_name.data: o for o in module.walk()
                                    if o.name == "func.func" and "$kernel_" in o.sym_name.data})
        return _host_kfn_cache.get(sym)

    mesh_counts: dict = counters if counters is not None else {}   # per-CALL, not
    # a module-global function attribute: run_suite grades capsules on a thread pool.
    if kernel_backend == "mesh":
        if not mesh_target:
            raise DispatchRuntimeError("kernel_backend='mesh' requires mesh_target=<target>")
        mesh_dp = mesh_datapath(mesh_target)
        # The host lane materializes f32; the mesh may also take its own format directly.
        _mesh_et = mesh_dp.mlir_dtype(mesh_dp.operand_dtype)
        _accept = ("f32", _mesh_et) if _mesh_et != "f32" else ("f32",)
        kfns = {op.sym_name.data: op for op in module.walk()
                if op.name == "func.func" and "$kernel_" in op.sym_name.data}
        for sym, kfn in kfns.items():
            # a bare linalg.matmul, or the linalg.generic a quantization rewrite leaves behind
            route = (_classify_mesh_matmul(kfn, _accept)
                     or _classify_mesh_contraction(kfn, _accept))
            if route is not None:
                mesh_routes[sym] = route
        # Matmul-family kernels the classifier REJECTED (bias-fused, transposed, batched-generic,
        # non-f32/i8, operand computed in-kernel). They never reach the mesh branch and so are invisible
        # to `mesh_fell_back`; counting them keeps the coverage claim honest.
        _unrouted = sum(1 for sym, kfn in kfns.items()
                        if sym not in mesh_routes and _has_contraction(kfn))
        # mesh_unavailable seeded to 0 alongside the rest: absent it reads as UNKNOWN downstream,
        # which is the right answer for "nobody could tell" but the wrong one for "none occurred".
        mesh_counts.update(mesh_ran=0, mesh_fell_back=0, mesh_unrouted_matmuls=_unrouted,
                           mesh_routed=len(mesh_routes), mesh_unavailable=0,
                           mesh_route_symbols=sorted(mesh_routes), dispatch_ledger=[])
        execute.last_mesh_routed = len(mesh_routes)
        execute.mesh_ran = 0
        execute.mesh_fell_back = 0
        # A LAYER THE ORACLE COULD NOT MEASURE IS NOT A LAYER THE MESH REFUSED. Kept apart from
        # mesh_fell_back because the two license opposite conclusions: a refusal says something about the
        # backend's shape space, an unavailable oracle says only that our measurement did not finish.
        # Conflating them reported "the model fell back to the host" for a whole model whose every layer
        # the mesh runs correctly -- the simulator had simply timed out.
        execute.mesh_unavailable = 0
        execute.mesh_fallbacks = []                  # per-layer reasons, so a fallback is actionable
        execute.mesh_unavailable_detail = []
        # Layers whose capacity_fit obligation the RUNTIME discharged on the backend's behalf. A
        # whole-model pass that needed this is a statement about runtime+backend, not about the backend.
        execute.mesh_capacity_fit_delegated = []
        # What the mesh's operand format could actually hold, summed over every layer it ran. The whole
        # atlas divergence was invisible because nobody counted this; a run now reports it. The SAME dict
        # rides this call's counters, so a reader takes it off the run's own result rather than off a
        # module-global attribute that a concurrent grade would clobber.
        execute.mesh_operand_repr = {"operand_dtype": mesh_dp.operand_dtype,
                                     "subnormal_operand_flush": bool(mesh_dp.subnormal_operand_flush),
                                     "n_values": 0, "n_subnormal": 0,
                                     "n_flushed_to_zero": 0, "n_saturating": 0}
        if mesh_dp.integer:
            # Only the FLOAT boundary counts representability; an integer datapath saturates rather than
            # underflowing and takes the quantizing branch below. Say that, instead of shipping four
            # zeroes that read as a clean bill of health this check never issued.
            execute.mesh_operand_repr = {
                "operand_dtype": mesh_dp.operand_dtype, "applicable": False,
                "note": "integer datapath: operands are quantized to the mesh's width at the boundary; "
                        "the subnormal/saturation accounting applies to a float datapath only"}
        mesh_counts["mesh_operand_repr"] = execute.mesh_operand_repr
        mesh_counts["mesh_fallbacks"] = execute.mesh_fallbacks
        # Ride this run's own counters, not a module-global attribute a concurrent grade would clobber.
        # Same list object, so the appends below land in both.
        mesh_counts["mesh_capacity_fit_delegated"] = execute.mesh_capacity_fit_delegated

    def _record_dispatch(symbol: str, lane: str, **evidence) -> None:
        """Append one unbounded, ordered entry for one *completed* dynamic kernel call.

        Aggregate counters and a routing plan cannot prove ordering or host/accelerator seams.  This
        ledger is owned by the runtime, populated only after the selected lane completes, and returned
        with this invocation's counters (never a module-global shared across concurrent grades).
        """
        if kernel_backend != "mesh":
            return
        ledger = mesh_counts.setdefault("dispatch_ledger", [])
        ledger.append({"ordinal": len(ledger), "symbol": symbol, "lane": lane,
                       "status": "pass", **evidence})

    def kernel_model(symbol: str):
        if symbol in compiled:
            return compiled[symbol]
        km = extract_kernel(module, symbol)
        key = to_text(km)
        model = cache.get(key)
        nonlocal n_unique
        if model is None:
            if cache_dir is not None:
                import shutil

                digest = hashlib.sha1(key.encode()).hexdigest()[:16]
                so = cache_dir / f"{digest}.so"
                if not so.is_file():
                    compile_host(km, workdir / symbol.replace("$", "_"))
                    shutil.copy2(workdir / symbol.replace("$", "_") / "model_host.so", so)
                model = HostModel.load(str(so))
            else:
                model = compile_host(km, workdir / symbol.replace("$", "_"))
            cache[key] = model
            n_unique += 1
        compiled[symbol] = model
        return model

    def precompile_parallel():
        """Compile all unique kernels concurrently into ``cache_dir`` before execution.

        Lazy per-call compilation is serial — for large models (pi05 ~4k kernels) the
        lowering+clang invocations dominate wall-clock. Each kernel is lowered+compiled in a
        separate PROCESS (xDSL's parser/lowering is not thread-safe — concurrent parses race;
        processes isolate it), then execution only *loads* the cached ``.so``s. The dedup +
        cache check runs in the main process (cheap); only the per-kernel lower+clang fans
        out. No-op without a cache_dir; ``MERLIN_COMPILE_WORKERS=1`` forces serial."""
        if cache_dir is None:
            return
        import concurrent.futures
        import os as _os
        import shutil

        from xdsl.dialects.builtin import ModuleOp, StringAttr

        # Pre-index kernel funcs by symbol in ONE walk -- extract_kernel walks the whole
        # module per call (O(module)), which is O(symbols x module) overall and dominates
        # wall-clock for large graphs (pi05 ~4k symbols x 13k ops). Lookup + clone instead.
        kfuncs = {op.sym_name.data: op for op in module.walk()
                  if op.name == "func.func" and "$kernel_" in op.sym_name.data}

        def _ktext(sym: str) -> str:
            clone = kfuncs[sym].clone()
            clone.properties["sym_name"] = StringAttr(entry)
            clone.properties.pop("sym_visibility", None)
            return to_text(ModuleOp([clone]))

        todo: dict[str, tuple] = {}     # digest -> (kernel_text, symbol); deduped by body
        seen: set[str] = set()
        for op in driver.walk():
            if op.name != "func.call":
                continue
            sym, _ = _kernel_io(op)
            if sym in seen or sym not in kfuncs or sym in xnn_routes:
                continue        # xnn-routed kernels are computed by XNNPACK, never compiled
            seen.add(sym)
            ktext = _ktext(sym)
            digest = hashlib.sha1(ktext.encode()).hexdigest()[:16]
            if digest not in todo and not (cache_dir / f"{digest}.so").is_file():
                todo[digest] = (ktext, sym)
        if not todo:
            return

        jobs = [(ktext, str(workdir / sym.replace("$", "_")), str(cache_dir / f"{digest}.so"))
                for digest, (ktext, sym) in todo.items()]
        workers = int(_os.environ.get("MERLIN_COMPILE_WORKERS", "0")) or max(1, (_os.cpu_count() or 4))
        workers = min(len(jobs), workers)
        if workers <= 1:
            for j in jobs:
                _compile_kernel_so(j)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_compile_kernel_so, jobs))

    precompile_parallel()

    from xdsl.dialects.builtin import TensorType

    from ..llvmlower.abi import ScalarArg

    def run_call(op):
        symbol, outs = _kernel_io(op)
        host_lane = "scalar_rvv_lane"
        host_evidence: dict = {}
        route = xnn_routes.get(symbol)
        if route is not None:
            from .backends import xnnpack_host

            a = np.ascontiguousarray(env[id(op.operands[route["a"]])], np.float32)
            b = np.ascontiguousarray(env[id(op.operands[route["b"]])], np.float32)
            out = xnnpack_host.gemm_f32(a, b).reshape(outs[0][0]).astype(outs[0][1])
            env[id(op.results[0])] = out
            _record_dispatch(symbol, "scalar_rvv_lane", executor="xnnpack_host")
            if tap is not None:
                tap(op, [out])
            return
        mroute = mesh_routes.get(symbol)
        if mroute is not None:                       # run this matmul LAYER on the target mesh (real data)
            from ..compile_cli import run_matmul_on_mesh

            a = np.ascontiguousarray(env[id(op.operands[mroute["a"]])])
            b = np.ascontiguousarray(env[id(op.operands[mroute["b"]])])
            mesh_out = None
            _obs: dict = {}
            binding = mesh_dp or mesh_datapath(mesh_target)
            op_dt, acc_dt, integer = binding.operand_dtype, binding.accum_dtype, bool(binding.integer)
            mesh_et = binding.mlir_dtype(op_dt)
            if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:   # a plain 2-D contraction
                if mroute["in_dtype"] == mesh_et:    # already in the mesh's format — no boundary convert
                    qa, qb, scale = a.tolist(), b.tolist(), 1.0
                elif integer:
                    # f32 -> per-tensor symmetric integer at the mesh boundary. The clip bound comes from
                    # the datapath's own width, not from a literal 127: a narrower or wider integer mesh
                    # would otherwise be fed operands saturated to some other unit's range.
                    from ..targetgen.capsule_dram import dtype_bits as _dbits
                    _bits = _dbits(op_dt)
                    lim = float(2 ** (_bits - 1) - 1)
                    af, bf = a.astype(np.float64), b.astype(np.float64)
                    sa = float(np.abs(af).max()) / lim or 1.0
                    sb = float(np.abs(bf).max()) / lim or 1.0
                    qa = np.clip(np.rint(af / sa), -lim, lim).astype(np.int64).tolist()
                    qb = np.clip(np.rint(bf / sb), -lim, lim).astype(np.int64).tolist()
                    scale = sa * sb
                else:
                    # Float datapath: the mesh takes the values themselves, but only after a power-of-two
                    # shift into the range its operand format can actually hold. Rounding TO the format is
                    # still the ORACLE's job -- doing that here would grade the target against our idea of
                    # its rounding rather than the hardware's -- and a power of two does not round: it
                    # moves the exponent and leaves every mantissa bit where it was.
                    qa, qb, scale, _repr = float_boundary_operands(a, b, binding)
                    _acc = mesh_counts.get("mesh_operand_repr")
                    if _acc is not None:
                        for side in ("a", "b"):
                            for key, val in _repr[side].items():
                                _acc[key] = _acc.get(key, 0) + val
                # `package` is the SUBMISSION under test. Without it the mesh path resolves the target's
                # DEFAULT package and quietly certifies a different compiler than the one being graded.
                # `observed` is this call's OWN record of what the mesh path did -- which tiler chose the
                # extent, why it declined, whether the runtime had to discharge a capacity_fit obligation
                # the backend owes. Read from the call rather than a module global, so a concurrent grade
                # cannot attribute one model's decline to another model's layer.
                # THE BACKEND UNDER TEST MUST REACH THE MESH. Without it this resolved to the
                # target's DEFAULT package -- which for a target that ships none is no oracle at all, so
                # every layer returned None with no verdict and was recorded as a host fallback. That is
                # how a model whose layers the mesh runs correctly reported "15 of 15 fell back": not a
                # refusal, not even a timeout, simply nothing to ask.
                mesh_out = run_matmul_on_mesh(mesh_target, qa, qb,
                                              operand_dtype=op_dt, accum_dtype=acc_dt,
                                              package=mesh_package, observed=_obs)
                _cf = _obs.get("capacity_fit")
                if _cf is not None:
                    _d = getattr(execute, "mesh_capacity_fit_delegated", None)
                    if _d is not None and len(_d) < 64:
                        # WHICH tiler discharged it travels with the record. "the layer did not fit"
                        # and "the backend only built one tile" are different facts about the backend,
                        # and only the second says its shape space is uncovered -- so the score can
                        # report the backend's own coverage apart from runtime+backend.
                        _d.append({"kernel": symbol, "lhs": list(a.shape), "rhs": list(b.shape),
                                   "required_elems": _cf.get("required_elems"),
                                   "capacity_elems": _cf.get("capacity_elems"),
                                   # WHICH tiler chose the extent: a capacity fact read out of the RTL,
                                   # or a probe that halved until the backend stopped refusing. Those are
                                   # different provenance claims and the record used to say neither.
                                   "tile_source": _cf.get("tile_source"),
                                   # ...and WHY it tiled at all (working-set capacity vs a declared
                                   # primitive tile). A separate question from which tiler ran, so both
                                   # ride the record — neither answers the other.
                                   "tiled_by": _cf.get("tiled_by")})
                if mesh_out is not None:
                    om = np.array(mesh_out, np.float64) * scale
                    env[id(op.results[0])] = om.reshape(outs[0][0]).astype(outs[0][1])
                    mesh_counts["mesh_ran"] = mesh_counts.get("mesh_ran", 0) + 1
                    execute.mesh_ran = getattr(execute, "mesh_ran", 0) + 1
                    _record_dispatch(
                        symbol, "on_mesh", lhs_shape=list(a.shape), rhs_shape=list(b.shape),
                        oracle_evidence=_obs.get("oracle_evidence"),
                        trace_check=_obs.get("trace_check"),
                        artifact_identity=_obs.get("artifact_identity"),
                        cert_run_id=_obs.get("cert_run_id"),
                        capacity_fit=_obs.get("capacity_fit"))
                    if tap is not None:
                        tap(op, [env[id(op.results[0])]])
                    return
            # A layer the mesh could not run falls back to the host kernel and is NOT counted as
            # mesh-executed — honest accounting, never a faked mesh result. RECORD WHY. The count alone
            # says a model failed its must_accelerate gate without saying what to fix, which is the same
            # fail-silent shape the rest of this path was built to avoid: a whole-model capsule reported
            # "35 of 37 layers on the mesh, 2 fell back" and nothing anywhere named the two.
            mesh_counts["mesh_fell_back"] = mesh_counts.get("mesh_fell_back", 0) + 1
            if not (a.ndim == 2 and b.ndim == 2):
                _why = f"operand rank {a.ndim}x{b.ndim}: the mesh boundary takes a 2-D contraction"
            elif a.shape[1] != b.shape[0]:
                _why = f"inner dims disagree: {a.shape} @ {b.shape}"
            else:
                # The oracle path records its OWN refusal reason; prefer it, because "unsynthesizable at
                # this shape" and "the oracle was unreachable" are different repairs. THIS call's own
                # `observed` verdict comes first -- it belongs to this layer and nothing else can have
                # overwritten it -- then the module-global last-refusal note, then a generic message that
                # covers both causes and names neither.
                _recorded = (locals().get("_obs") or {}).get("decline")
                if not _recorded:
                    try:
                        from ..compile_cli import _MESH_REFUSAL
                        _recorded = _MESH_REFUSAL.get("reason")
                    except Exception:                              # noqa: BLE001
                        _recorded = None
                _why = (f"mesh oracle declined {a.shape} @ {b.shape} ({op_dt}/{acc_dt}): "
                        f"{str(_recorded)[:300]}" if _recorded else
                        f"mesh oracle returned no result for {a.shape} @ {b.shape} "
                        f"({op_dt}/{acc_dt}) — unsynthesizable at this shape, or the oracle "
                        f"was unreachable")
            # SEPARATE "the oracle could not tell us" FROM "the mesh refused". The host kernel runs
            # either way -- the model must still compute -- but only the second is evidence about the
            # backend. `_oracle_unreachable` keys on the oracle's OWN reported cause, never on a guess.
            # Carried on the counters dict, not the module-global function attributes this check first
            # used: concurrent grades clobber those, and a model verdict now depends on the count.
            _decl = str((locals().get("_obs") or {}).get("decline") or "")
            if _oracle_unreachable(_decl):
                # charged above as a fallback; it is not one -- move it to its own bucket
                mesh_counts["mesh_fell_back"] = max(0, mesh_counts.get("mesh_fell_back", 0) - 1)
                mesh_counts["mesh_unavailable"] = mesh_counts.get("mesh_unavailable", 0) + 1
                _ud = mesh_counts.setdefault("mesh_unavailable_detail", [])
                if len(_ud) < 64:
                    _ud.append({"kernel": symbol, "lhs": list(a.shape), "rhs": list(b.shape),
                                "reason": _why})
                _unavailable = True
                host_lane = "mesh_unavailable"
            else:
                _unavailable = False
                host_lane = "host_fallback"
            host_evidence = {"lhs_shape": list(a.shape), "rhs_shape": list(b.shape),
                             "mesh_decline": _why}
            if not _unavailable:
                # Name the layer, not just the count. "4 layers fell back" gives a reader nothing to act on;
                # the SHAPES say whether the cause is one extent the backend cannot take or four different
                # ones, which is the difference between a tiling gap and a capability gap.
                try:
                    _sh = f"{a.shape[0]}x{a.shape[1]}x{b.shape[1]}" if (a.ndim == 2 and b.ndim == 2) \
                        else f"rank{a.ndim}x{b.ndim}"
                except Exception:                                  # noqa: BLE001
                    _sh = "unknown"
                mesh_counts.setdefault("mesh_fallback_shapes", []).append(f"{_sh}: {_why}")
                # Capture the operands HERE, where they are already numpy arrays, rather than deeper in the
                # oracle path where an earlier attempt to do it failed silently. A layer that fails inside a
                # model and runs standalone differs only in the values it was handed.
                try:
                    import tempfile as _tf
                    _n = len(mesh_counts["mesh_fallback_shapes"])
                    _p = Path(_tf.gettempdir()) / f"fallback_operands_{_n}.npz"
                    np.savez_compressed(_p, A=a, W=b)
                    mesh_counts.setdefault("mesh_fallback_operands", []).append(str(_p))
                except Exception as _oe:                   # noqa: BLE001
                    mesh_counts.setdefault("mesh_fallback_operands", []).append(
                        f"capture failed: {type(_oe).__name__}: {_oe}")
                execute.mesh_fell_back = getattr(execute, "mesh_fell_back", 0) + 1
                _fb = getattr(execute, "mesh_fallbacks", None)
                if _fb is not None and len(_fb) < 64:            # bounded: a diagnostic, not a full trace
                    _fb.append({"kernel": symbol, "lhs": list(a.shape), "rhs": list(b.shape),
                                "reason": _why})
        model = kernel_model(symbol)
        args, keep = [], []
        for o in op.operands:
            if isinstance(o.type, TensorType):
                a = np.ascontiguousarray(env[id(o)])
                keep.append(a)
                args.append((a.ctypes.data, a.shape))
            else:
                args.append(ScalarArg(env[id(o)], str(o.type)))   # by-value scalar arg
        out_arrays = [np.zeros(sh, dt) for sh, dt in outs]
        args += [(o.ctypes.data, o.shape) for o in out_arrays]
        # THE HOST LANE ACTUALLY EXECUTING, which is the one thing nothing counted. `lane_report`
        # corrected `on_mesh` against per-layer mesh accounting but had no equivalent for
        # `scalar_rvv_lane`, so a capsule could satisfy a required host lane on the strength of a
        # ROUTING PLAN that never ran -- the same defect ("a routing plan is not an execution") that
        # was fixed on the mesh side and left open on this one.
        #
        # Contractions are counted SEPARATELY on purpose: the host lane is populated overwhelmingly by
        # norms and activations, so a bare kernel count is satisfied by any model at all and would be
        # as hollow as the plan-only evidence it replaces.
        #
        # Into the per-call `counters` dict, never onto a module-global: run_suite grades on a thread
        # pool and concurrent grades clobber module attributes -- and this count now feeds a verdict.
        mesh_counts["host_kernels_ran"] = mesh_counts.get("host_kernels_ran", 0) + 1
        _hostfn = _host_kernel_fn(symbol)
        if _hostfn is not None and _has_contraction(_hostfn):
            mesh_counts["host_contractions_ran"] = mesh_counts.get("host_contractions_ran", 0) + 1
        model(args)
        for r, o in zip(op.results, out_arrays):
            env[id(r)] = o
        _record_dispatch(symbol, host_lane, **host_evidence)
        if tap is not None:
            tap(op, out_arrays)

    def run_scf_for(op):
        lb, ub, step = (int(env[id(op.operands[i])]) for i in range(3))
        body = op.regions[0].blocks[0]
        ivar, iter_args = body.args[0], list(body.args[1:])
        acc = [env[id(o)] for o in op.operands[3:]]              # iter_args inits
        for i in range(lb, ub, step):
            env[id(ivar)] = i
            for a, v in zip(iter_args, acc):
                env[id(a)] = v
            acc = run_block(body) or acc                        # scf.yield -> new acc
        for r, v in zip(op.results, acc):
            env[id(r)] = v

    def run_scf_if(op):
        cond = env[id(op.operands[0])]
        regions = op.regions
        region = regions[0] if cond else regions[1]
        if region.blocks:
            res = run_block(region.blocks[0])
            for r, v in zip(op.results, res or []):
                env[id(r)] = v

    def _dead_after(blk):
        """Map op-index -> [value-ids] whose LAST use is that op, so the driver can free
        intermediate tensors from ``env`` as soon as they're dead. Without this the numpy
        interpreter keeps every intermediate alive for the whole forward -- fatal for large
        graphs (pi05 ~13k ops). Region-nested uses count against the containing top-level op
        (``op.walk()`` recurses), so a value captured by an scf body stays live until the
        scf op. Disabled when ``tap`` is set (the bisection hook may inspect values later)."""
        last: dict[int, int] = {}
        for i, op in enumerate(blk.ops):
            for sub in op.walk():
                for operand in sub.operands:
                    last[id(operand)] = i
        dead: dict[int, list[int]] = {}
        for vid, i in last.items():
            dead.setdefault(i, []).append(vid)
        return dead

    def run_block(blk, prune=False):
        """Execute a block's ops; returns operand values on a terminator (return/yield)."""
        dead = _dead_after(blk) if prune else {}
        for i, op in enumerate(blk.ops):
            if op.name in ("func.return", "scf.yield"):
                return [env[id(o)] for o in op.operands]
            if op.name == "func.call":
                run_call(op)
            elif op.name == "scf.for":
                run_scf_for(op)
            elif op.name == "scf.if":
                run_scf_if(op)
            else:
                val = _eval_view(op, env)
                # Bind torchao subclass inner tensors (int_data/scale) that m2m elided to an
                # uninitialized empty tagged prov.quant_inner -> the real quantized data.
                if qinner and op.name == "tensor.empty":
                    qi = op.attributes.get("prov.quant_inner")
                    key = getattr(qi, "data", None)
                    if key is not None and key in qinner:
                        val = np.ascontiguousarray(qinner[key])
                env[id(op.results[0])] = val
                if tap is not None:
                    tap(op, val)
            for vid in dead.get(i, ()):                  # free now-dead intermediates
                env.pop(vid, None)
        return None

    results = run_block(block, prune=(tap is None))
    execute.last_unique_kernels = n_unique
    if results is None:
        raise DispatchRuntimeError("driver had no func.return")
    return results


def _propagate_quant_inner(module) -> int:
    """Re-attach quant-inner binding keys onto the (elided, uninitialized) torchao subclass
    inner-tensor ``tensor.empty`` ops, in memory, so ``execute`` can bind the real quantized
    data. xDSL's printer drops attributes on tensor.empty, so m2m parks the keys on the
    CONSUMER ops instead: every op carries ``prov.quant_inner_<i>`` for operand ``i`` whose
    source is an inner-tensor empty. Covers int8 (dequant) and fp8 (mul x scale) patterns.
    Also honors the legacy ``prov.quant_inner_{w,s}`` on the dequant op (operands 0/1)."""
    def _tag_src_empty(val, tag):
        op = getattr(val, "owner", None)
        for _ in range(8):                            # walk through transpose/cast wrappers
            if op is None or not hasattr(op, "name"):
                return
            if op.name == "tensor.empty":
                op.attributes["prov.quant_inner"] = tag
                return
            ins = getattr(op, "operands", ())
            op = getattr(ins[0], "owner", None) if ins else None

    n = 0
    for op in module.walk():
        operands = getattr(op, "operands", ())
        for key, tag in op.attributes.items():
            idx = None
            if key.startswith("prov.quant_inner_"):
                suffix = key[len("prov.quant_inner_"):]
                idx = {"w": 0, "s": 1}.get(suffix)        # legacy dequant aliases
                if idx is None and suffix.isdigit():
                    idx = int(suffix)
            if idx is not None and idx < len(operands):
                _tag_src_empty(operands[idx], tag); n += 1
    return n


def run_model(model_dir: str | Path, workdir: str | Path,
              cache_dir: str | Path | None = None, tap=None,
              int8_compute: bool = False,
              quant_passes: "list[str] | None" = None,
              quant_select=None,
              prequant_gather: bool = False,
              kernel_backend: str | None = None, mesh_target: str | None = None,
              mesh_package: str | None = None) -> dict[str, Any]:
    """Outline + bind + execute a captured model; gate against ``golden.npy``.

    Returns ``{output, golden, cos, rel, ok, n_kernels, n_unique_kernels}``.

    ``int8_compute=True`` runs the integer (W8A8) datapath: each ``dequant(weight)→f32 matmul``
    becomes ``quantize(act)→ i8×i8→i32 matmul → requant`` (real integer contraction on RVV),
    instead of dequantizing the weight to f32 (the default weight-only path).

    ``prequant_gather=True`` additionally moves each gathered activation's quantization to BEFORE
    the gather with a per-tensor scale (the ``quantize_before_gather`` feature). It is meaningful
    only with ``int8_compute=True`` and it is NOT bit-exact against the default per-row scheme.

    ``quant_passes`` / ``quant_select`` narrow that datapath and are meaningful only when
    ``int8_compute=True``. ``quant_passes`` is a subset of ``quant_passes.known()`` (default None =
    all six, byte-identical to the shipped path); ``quant_select`` is an ``(op) -> bool`` predicate
    restricting which ops those passes may rewrite. Together they make the datapath's REACH a
    variable, which is what separates "our arithmetic is wrong" from "we quantize more operations
    than the reference does" when grading against a reference (e.g. torchao, which quantizes
    ``nn.Linear`` only) whose quantization policy is narrower than ours.
    """
    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects.lowering.outline import outline_dispatches

    from ..llvmlower.passes_xdsl import (collapse_overrank_matmul, fix_bool_fptosi,
                                         fix_bool_sitofp, lower_bf16_matmul_f32acc,
                                         lower_quant_ext)
    from ..llvmlower.torchao_affine import lower_torchao_affine_quant

    model_dir = Path(model_dir)
    module = parse_mlir_file(model_dir / "model.mlir")
    # Normalize before outlining so quantized/bf16/over-rank models compute correctly:
    #  - 3-D `aten.linear` matmuls (invalid 2-D-map matmul) -> batched linalg.generic;
    #  - int8 weights: dequantize_per_channel -> linalg.generic (weights stay i8 in memory);
    #  - bf16 matmuls: accumulate in f32 then truncate (matches torch), else ~bf16 round-off;
    #  - bool->float casts: sitofp(i1) (true -> -1.0) -> uitofp (true -> 1.0), matching torch
    #    (fixes the eager-attention causal-mask sign flip; molmoact decoder).
    # An activation-quant capture leaves torchao's choose_qparams/quantize as opaque calls to
    # externs nothing defines; without this the module cannot even be outlined.
    lower_torchao_affine_quant(module)
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)            # dequant prov.quant_inner_{w,s} -> source empties
    if int8_compute:
        # The integer (W8A8) datapath, via the quant-pass registry (the quantization region's
        # edit-point). apply_quant() with the default set runs the six lower_*_int passes in the
        # canonical order — byte-identical to the historical hardcoded sequence, now toggleable.
        from ..llvmlower.quant_passes import apply_quant
        # `prequant_gather` = the `quantize_before_gather` feature. Threaded through the HOST
        # interpreter too, not only the device build: the per-tensor activation scale it introduces is
        # a genuine numeric change, so it has to be gradeable against golden_w8a8.npy here before any
        # board measurement is worth taking.
        # Passed ONLY when asked for: `test_default_reach_passes_no_select_at_all` gates that the
        # default path hands the passes no kwargs at all, so a pass that never learned this flag keeps
        # working and the shipped datapath cannot drift behind a default argument.
        extra = {"prequant_gather": True} if prequant_gather else {}
        apply_quant(module, quant_passes, select=quant_select, **extra)
    lower_quant_ext(module)                   # residual dequants (unconverted) -> f32 fallback
    lower_bf16_matmul_f32acc(module)
    fix_bool_sitofp(module)
    # Keep the interpreter on the SAME bool-cast semantics as the compiled path: `fptosi f32 -> i1`
    # is poison in LLVM but `int(x)` here, so leaving it out is exactly how the two paths diverge.
    fix_bool_fptosi(module)
    outlined = outline_dispatches(module)
    driver = next(op for op in outlined.module.walk()
                  if op.name == "func.func" and "$kernel_" not in op.sym_name.data)
    out_types = list(driver.function_type.outputs.data)
    if len(out_types) != 1:
        raise DispatchRuntimeError(
            f"multi-output models not supported yet ({len(out_types)} results)")
    args = resolve_forward_args(model_dir)
    # Quantized-subclass inner tensors (int_data/scale) the capture extracted under qinner::
    # keys; bind them to the prov.quant_inner-tagged empties m2m left uninitialized.
    qinner = None
    extra_path = model_dir / "extra.npz"
    if extra_path.is_file():
        ex = np.load(extra_path)
        qinner = {k[len("qinner::"):]: ex[k] for k in ex.files if k.startswith("qinner::")}
    import os as _os
    # Per-run mesh counters. A whole-model verdict is decided by these, so they must not
    # live on a module-global function attribute that a concurrent grade can clobber.
    _mesh_counts: dict = {}
    if kernel_backend is None and _os.environ.get("MERLIN_XNNPACK_HOST") == "1":
        kernel_backend = "xnnpack"
    results = execute(outlined, args, Path(workdir), cache_dir=cache_dir, tap=tap,
                      qinner=qinner, kernel_backend=kernel_backend, mesh_target=mesh_target,
                      mesh_package=mesh_package, counters=_mesh_counts)
    # widen bf16 (stored as uint16 bit patterns) to f32 for the golden comparison
    raw = results[0]
    out = (bf16_to_f32(raw) if _elem_str(out_types[0]) == "bf16"
           else np.asarray(raw, dtype=np.float32)).ravel()

    res: dict[str, Any] = {"output": results[0], "n_kernels": outlined.n_kernels,
                           "n_unique_kernels": getattr(execute, "last_unique_kernels", None),
                           "kernel_backend": kernel_backend,
                           "n_xnn_routed": (getattr(execute, "last_xnn_routed", 0)
                                            if kernel_backend == "xnnpack" else 0),
                           **_mesh_counts}
    gpath = model_dir / "golden.npy"
    if gpath.is_file():
        gold = np.load(gpath).astype(np.float32).ravel()
        k = min(len(out), len(gold))
        a, b = out[:k], gold[:k]
        gmax = float(np.abs(b).max())
        abs_err = float(np.abs(a - b).max())
        rel = abs_err / max(1e-9, gmax)
        cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        # Cosine is undefined for a ~zero golden (e.g. a DiT with a zero-init output head):
        # fall back to absolute agreement so a correct zero-output match isn't a false fail.
        ok = rel < 1e-3 if gmax < 1e-6 else (cos > 0.9999 and rel < 1e-3)
        res.update(golden=gold, cos=cos, rel=rel, abs_err=abs_err, gold_absmax=gmax, ok=ok)
    return res
