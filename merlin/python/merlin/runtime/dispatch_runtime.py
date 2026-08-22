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


def _has_contraction(kfn) -> bool:
    """True when a kernel func contains a contraction op at all — used to count the matmuls the mesh
    classifier declined, structurally (no name matching on the symbol)."""
    for op in kfn.walk():
        nm = getattr(op, "name", "")
        if nm in ("linalg.matmul", "linalg.batch_matmul", "linalg.matmul_transpose_b",
                  "linalg.quantized_matmul"):
            return True
    return False


def _classify_mesh_matmul(kfn) -> dict | None:
    """Like xnnpack's classify_matmul_kernel but for the MESH route: a plain 2-D ``linalg.matmul`` whose two
    ``ins`` are kernel block args, element type f32 OR i8 (the int8 systolic datapath). Returns
    ``{"a", "b", "in_dtype"}`` (so the route knows to quantize an f32 layer at the mesh boundary but pass an
    already-int8 layer through untouched), else None."""
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
    if ea is None or ea != eb or ea not in ("f32", "i8"):
        return None
    if id(a_val) not in arg_ids or id(b_val) not in arg_ids:
        return None                                          # an operand is computed inside the kernel
    return {"a": arg_ids[id(a_val)], "b": arg_ids[id(b_val)], "in_dtype": ea}


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
    # mesh output threads forward — so the whole model runs with its matmul layers on the RTL systolic
    # array. f32 operands are quantized to int8 at the boundary (the mesh is an int8 datapath; the drop is
    # expected quantization error, covered by the whole-model quant tolerance).
    mesh_routes: dict[str, dict] = {}
    mesh_counts: dict = counters if counters is not None else {}   # per-CALL, not
    # a module-global function attribute: run_suite grades capsules on a thread pool.
    if kernel_backend == "mesh":
        if not mesh_target:
            raise DispatchRuntimeError("kernel_backend='mesh' requires mesh_target=<target>")
        kfns = {op.sym_name.data: op for op in module.walk()
                if op.name == "func.func" and "$kernel_" in op.sym_name.data}
        for sym, kfn in kfns.items():
            route = _classify_mesh_matmul(kfn)               # f32 OR i8 2-D matmul (mesh route)
            if route is not None:
                mesh_routes[sym] = route
        # Matmul-family kernels the classifier REJECTED (bias-fused, transposed, batched-generic,
        # non-f32/i8, operand computed in-kernel). They never reach the mesh branch and so are invisible
        # to `mesh_fell_back`; counting them keeps the coverage claim honest.
        _unrouted = sum(1 for sym, kfn in kfns.items()
                        if sym not in mesh_routes and _has_contraction(kfn))
        mesh_counts.update(mesh_ran=0, mesh_fell_back=0, mesh_unrouted_matmuls=_unrouted)
        execute.last_mesh_routed = len(mesh_routes)
        execute.mesh_ran = 0
        execute.mesh_fell_back = 0

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
        route = xnn_routes.get(symbol)
        if route is not None:
            from .backends import xnnpack_host

            a = np.ascontiguousarray(env[id(op.operands[route["a"]])], np.float32)
            b = np.ascontiguousarray(env[id(op.operands[route["b"]])], np.float32)
            out = xnnpack_host.gemm_f32(a, b).reshape(outs[0][0]).astype(outs[0][1])
            env[id(op.results[0])] = out
            if tap is not None:
                tap(op, [out])
            return
        mroute = mesh_routes.get(symbol)
        if mroute is not None:                       # run this matmul LAYER on the target mesh (real data)
            from ..compile_cli import run_matmul_on_mesh

            a = np.ascontiguousarray(env[id(op.operands[mroute["a"]])])
            b = np.ascontiguousarray(env[id(op.operands[mroute["b"]])])
            mesh_out = None
            if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:   # a plain 2-D contraction
                from ..compile_cli import mesh_datapath
                op_dt, acc_dt, is_int = mesh_datapath(mesh_target)
                if not is_int:
                    # A float mesh takes the operands as they are. Quantizing here would inject error
                    # into a datapath that declared a float format, and then dequantize it back out.
                    qa, qb, scale = a.astype(np.float64).tolist(), b.astype(np.float64).tolist(), 1.0
                elif mroute["in_dtype"] == "i8":     # already int8 — pass through, NO boundary quant
                    qa, qb, scale = a.astype(np.int64).tolist(), b.astype(np.int64).tolist(), 1.0
                else:                                # f32 -> per-tensor symmetric int8 at the mesh boundary
                    af, bf = a.astype(np.float64), b.astype(np.float64)
                    sa = float(np.abs(af).max()) / 127.0 or 1.0
                    sb = float(np.abs(bf).max()) / 127.0 or 1.0
                    qa = np.clip(np.rint(af / sa), -127, 127).astype(np.int64).tolist()
                    qb = np.clip(np.rint(bf / sb), -127, 127).astype(np.int64).tolist()
                    scale = sa * sb
                mesh_out = run_matmul_on_mesh(mesh_target, qa, qb, operand_dtype=op_dt,
                                              accum_dtype=acc_dt, package=mesh_package)
                if mesh_out is not None:
                    om = np.array(mesh_out, np.float64) * scale
                    env[id(op.results[0])] = om.reshape(outs[0][0]).astype(outs[0][1])
                    mesh_counts["mesh_ran"] = mesh_counts.get("mesh_ran", 0) + 1
                    execute.mesh_ran = getattr(execute, "mesh_ran", 0) + 1
                    if tap is not None:
                        tap(op, [env[id(op.results[0])]])
                    return
            # a layer the mesh could not run at this shape (non-2D or cert-fail) falls back to the host
            # kernel and is NOT counted as mesh-executed — honest accounting, never a faked mesh result.
            mesh_counts["mesh_fell_back"] = mesh_counts.get("mesh_fell_back", 0) + 1
            # Name the layer, not just the count. "4 layers fell back" gives a reader nothing to act on;
            # the SHAPES say whether the cause is one extent the backend cannot take or four different
            # ones, which is the difference between a tiling gap and a capability gap.
            try:
                _sh = f"{a.shape[0]}x{a.shape[1]}x{b.shape[1]}" if (a.ndim == 2 and b.ndim == 2) \
                    else f"rank{a.ndim}x{b.ndim}"
            except Exception:                                  # noqa: BLE001
                _sh = "unknown"
            try:
                from ..compile_cli import _MESH_REFUSAL
                _why = _MESH_REFUSAL.get("reason") or "no reason recorded"
            except Exception:                                  # noqa: BLE001
                _why = "no reason recorded"
            mesh_counts.setdefault("mesh_fallback_shapes", []).append(f"{_sh}: {_why}")
            execute.mesh_fell_back = getattr(execute, "mesh_fell_back", 0) + 1
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
        model(args)
        for r, o in zip(op.results, out_arrays):
            env[id(r)] = o
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
              kernel_backend: str | None = None, mesh_target: str | None = None,
              mesh_package: str | None = None) -> dict[str, Any]:
    """Outline + bind + execute a captured model; gate against ``golden.npy``.

    Returns ``{output, golden, cos, rel, ok, n_kernels, n_unique_kernels}``.

    ``int8_compute=True`` runs the integer (W8A8) datapath: each ``dequant(weight)→f32 matmul``
    becomes ``quantize(act)→ i8×i8→i32 matmul → requant`` (real integer contraction on RVV),
    instead of dequantizing the weight to f32 (the default weight-only path).
    """
    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects.lowering.outline import outline_dispatches

    from ..llvmlower.passes_xdsl import (collapse_overrank_matmul, fix_bool_sitofp,
                                         lower_bf16_matmul_f32acc, lower_quant_ext)

    model_dir = Path(model_dir)
    module = parse_mlir_file(model_dir / "model.mlir")
    # Normalize before outlining so quantized/bf16/over-rank models compute correctly:
    #  - 3-D `aten.linear` matmuls (invalid 2-D-map matmul) -> batched linalg.generic;
    #  - int8 weights: dequantize_per_channel -> linalg.generic (weights stay i8 in memory);
    #  - bf16 matmuls: accumulate in f32 then truncate (matches torch), else ~bf16 round-off;
    #  - bool->float casts: sitofp(i1) (true -> -1.0) -> uitofp (true -> 1.0), matching torch
    #    (fixes the eager-attention causal-mask sign flip; molmoact decoder).
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)            # dequant prov.quant_inner_{w,s} -> source empties
    if int8_compute:
        # The integer (W8A8) datapath, via the quant-pass registry (the quantization region's
        # edit-point). apply_quant() with the default set runs the six lower_*_int passes in the
        # canonical order — byte-identical to the historical hardcoded sequence, now toggleable.
        from ..llvmlower.quant_passes import apply_quant
        apply_quant(module)
    lower_quant_ext(module)                   # residual dequants (unconverted) -> f32 fallback
    lower_bf16_matmul_f32acc(module)
    fix_bool_sitofp(module)
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
