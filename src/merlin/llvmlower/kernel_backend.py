"""Per-kernel backend + checker — compile one outlined dispatch in isolation.

Outlining (``xdsl_dialects.lowering.outline``) keeps the view/glue ops (``extract_slice``,
``expand_shape``, …) in the *driver*; each kernel func is clean linalg + its cloned
accumulator init. That has two payoffs this module realizes:

1. **It routes around the whole-model xDSL printer bug.** The monolithic model cannot be
   re-emitted through xDSL (rank-reducing ``extract_slice`` mis-prints), so the whole-model
   path uses textual preprocessing. A single kernel func contains no ``extract_slice``, so
   it round-trips and compiles through the normal ``lower_model`` path.
2. **Each kernel is independently checkable.** ``check_matmul_kernels`` compiles every
   contraction dispatch and gates it against the analytic numpy reference (``A @ B``) — the
   numerically critical ops, where the whole-model NaN historically lived. First divergence
   localizes a bug to one kernel in seconds instead of bisecting the whole model.

Several kernel libraries coexist in one process safely because ``HostModel.load`` opens
them ``RTLD_LOCAL`` (no trampoline needed at this arity), so their shared
``_mlir_ciface_forward``/``memrefCopy`` symbols don't clash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NP_DTYPE = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.int8,
}


class KernelBackendError(RuntimeError):
    pass


@dataclass
class KernelSignature:
    in_shapes: list[tuple[int, ...]]
    in_dtypes: list[str]
    out_shapes: list[tuple[int, ...]]
    out_dtypes: list[str]


def _shape(t) -> tuple[int, ...]:
    return tuple(int(d) for d in t.get_shape())


def signature_of(func) -> KernelSignature:
    ft = func.function_type
    return KernelSignature(
        in_shapes=[_shape(t) for t in ft.inputs.data],
        in_dtypes=[str(t.element_type) for t in ft.inputs.data],
        out_shapes=[_shape(t) for t in ft.outputs.data],
        out_dtypes=[str(t.element_type) for t in ft.outputs.data],
    )


def standalone_kernel(module, func, entry: str = "forward"):
    """Clone a kernel with the source module's numerical-policy attributes."""
    from xdsl.dialects.builtin import ModuleOp, StringAttr

    clone = func.clone()
    clone.properties["sym_name"] = StringAttr(entry)
    clone.properties.pop("sym_visibility", None)
    return ModuleOp([clone], attributes=dict(module.attributes))


def extract_kernel(module, symbol: str, entry: str = "forward"):
    """A standalone ``builtin.module`` holding just ``symbol``, renamed to ``entry``."""
    func = next((op for op in module.walk() if op.name == "func.func" and op.sym_name.data == symbol), None)
    if func is None:
        raise KernelBackendError(f"kernel @{symbol} not found")
    return standalone_kernel(module, func, entry)


def host_scalar_result_dtype(kernel_module, entry: str = "forward") -> str | None:
    """Derive the by-value result left after tensor results become out-params.

    Multiple scalars need an aggregate C return ABI, which this runner does not
    support. Rank-zero tensors are still tensor output descriptors.
    """
    from xdsl.dialects.builtin import TensorType

    from .abi import scalar_result_ctype

    function = next((op for op in kernel_module.walk() if op.name == "func.func" and op.sym_name.data == entry), None)
    if function is None:
        return None  # Ordinary lowering retains its missing-entry refusal.
    scalars = [t for t in function.function_type.outputs if not isinstance(t, TensorType)]
    if len(scalars) > 1:
        raise KernelBackendError("host kernel aggregate scalar results are unsupported")
    dtype = str(scalars[0]) if scalars else None
    scalar_result_ctype(dtype)
    return dtype


def compile_host(kernel_module, workdir: str | Path):
    """Lower one kernel module to a host ``.so`` and load it (RTLD_LOCAL)."""
    from ..xdsl_dialects._common import text as to_text
    from .abi import HostModel, PrivateHostImagePolicy
    from .lower import lower_model

    # Lowering creates a missing build directory. Resolve its parent aliases
    # first so the selected private sibling retains the compiler's exact origin.
    result_dtype = host_scalar_result_dtype(kernel_module)
    workdir = Path(workdir).resolve()
    res = lower_model(to_text(kernel_module), workdir, targets=("host",))
    return HostModel.load(
        str(res.host_so),
        image_policy=PrivateHostImagePolicy(workdir.resolve(strict=True)),
        scalar_result_dtype=result_dtype,
    )


def compile_deployable_kernel(
    kernel_module,
    workdir: str | Path,
    *,
    entry: str,
    extra_cflags: tuple[str, ...],
    vectorize: bool = False,
    codegen_target: str = "rvv",
) -> dict:
    """Compile one closed tensor kernel for the explicit LP64 compute-worker ABI.

    This initial deployable subset refuses heap/LLVM-stack allocation and OpenMP
    rather than borrow an unbounded allocator or an implicit worker team. The
    selected target flags, compiler layout and exact emitted ABI are recorded.
    It does not build a parent forward, launch hardware or supply runtime libs.
    """
    import hashlib

    from merlin.common.digest import sha256_file

    from ..xdsl_dialects._common import text
    from . import codegen, target_data_layout, toolchain
    from .lower import lower_model

    if (
        not isinstance(entry, str)
        or not entry.isascii()
        or not entry.isidentifier()
        or not isinstance(extra_cflags, tuple)
        or not extra_cflags
        or any(not isinstance(flag, str) or not flag or "\0" in flag for flag in extra_cflags)
    ):
        raise ValueError("explicit safe kernel entry and literal target flag tuple required")
    functions = [op for op in kernel_module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or functions[0].sym_name.data != entry:
        raise ValueError("deployable kernel requires one independently named entry")
    from .deployable_outline import static_tensor_contract

    ranks = [
        len(static_tensor_contract(t)[0])
        for t in (*functions[0].function_type.inputs.data, *functions[0].function_type.outputs.data)
    ]
    flags = (*codegen.RISCV_FLAGS, *extra_cflags)
    observation = target_data_layout.observe_index_width(toolchain.clang(), flags)
    if observation["index_bits"] != 64 or not observation["data_layout"].startswith("e"):
        raise ValueError("deployable descriptors require observed little-endian LP64/index64 lowering")
    work = Path(workdir)
    result = lower_model(
        text(kernel_module),
        work,
        targets=(),
        vectorize=vectorize,
        codegen_target=codegen_target,
        parallel=False,
        features=frozenset(),
        static_arena=False,
        data_layout=observation["data_layout"],
        index_bits=64,
    )
    llvm = result.ll_path.read_text()
    if any(
        token in llvm
        for token in ("@malloc(", "@calloc(", "@realloc(", "@aligned_alloc(", "@free(", "__kmpc_", "alloca ")
    ):
        raise ValueError("deployable kernel needs unsupported internal allocation or OpenMP team")
    raw_prefix = f"define void @{entry}("
    wrapper_prefix = f"define void @_mlir_ciface_{entry}("
    raw = [line for line in llvm.splitlines() if line.startswith(raw_prefix)]
    wrapper = [line for line in llvm.splitlines() if line.startswith(wrapper_prefix)]
    expected = [kind for rank in ranks for kind in ("ptr", "ptr", "i64", *(["i64"] * (2 * rank)))]
    if len(raw) != 1 or len(wrapper) != 1:
        raise ValueError("missing unique void kernel and C-interface entries")
    actual = [item.strip().split()[0] for item in raw[0][len(raw_prefix) :].split(")", 1)[0].split(",")]
    pointer_args = [item.strip().split()[0] for item in wrapper[0][len(wrapper_prefix) :].split(")", 1)[0].split(",")]
    if actual != expected or pointer_args != ["ptr"] * len(ranks):
        raise ValueError("emitted kernel ABI disagrees with rank-exact descriptor contract")
    obj = codegen.compile_ll(result.ll_path, work / "kernel.o", extra_flags=extra_cflags)
    return {
        "entry": entry,
        "ciface_entry": "_mlir_ciface_" + entry,
        "object": str(obj),
        "object_sha256": sha256_file(obj),
        "llvm": str(result.ll_path),
        "llvm_sha256": sha256_file(result.ll_path),
        "source_ir_sha256": hashlib.sha256(text(kernel_module).encode()).hexdigest(),
        "argument_ranks": ranks,
        "index_observation": observation,
        "effective_cflags": list(flags),
        "vectorize": vectorize,
        "codegen_target": codegen_target,
        "heap_allocations": False,
        "LLVM_stack_allocations": False,
        "OpenMP": False,
    }


def run_random(model, sig: KernelSignature, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Invoke the kernel on seeded random inputs; return (inputs, outputs)."""
    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    for shape, dt in zip(sig.in_shapes, sig.in_dtypes):
        npdt = _NP_DTYPE.get(dt)
        if npdt is None:
            raise KernelBackendError(f"unsupported dtype {dt}")
        if np.issubdtype(npdt, np.floating):
            inputs.append(rng.standard_normal(shape).astype(npdt))
        else:
            inputs.append(rng.integers(-4, 5, size=shape).astype(npdt))
    outputs = [np.zeros(shape, _NP_DTYPE[dt]) for shape, dt in zip(sig.out_shapes, sig.out_dtypes)]
    args = [(a.ctypes.data, a.shape) for a in inputs]
    args += [(o.ctypes.data, o.shape) for o in outputs]
    model(args)
    return inputs, outputs


@dataclass
class KernelCheck:
    symbol: str
    ok: bool
    max_abs: float
    shapes: str


def check_matmul_kernels(
    outline_result, workdir: str | Path, seed: int = 0, rtol: float = 1e-4, atol: float = 1e-3
) -> list[KernelCheck]:
    """Compile every plain ``linalg.matmul`` kernel and gate it against ``A @ B``.

    Returns one :class:`KernelCheck` per matmul dispatch. The analytic reference is exact
    for the standard contraction these kernels carry (model2MLIR emits a separate
    ``linalg.transpose`` kernel for ``transposed_b`` layers, so the matmul itself is plain).
    """
    workdir = Path(workdir)
    module = outline_result.module
    results: list[KernelCheck] = []
    for d in outline_result.dispatches:
        if d.root_op != "linalg.matmul":
            continue
        km = extract_kernel(module, d.symbol)
        func = next(op for op in km.walk() if op.name == "func.func")
        sig = signature_of(func)
        if len(sig.in_shapes) != 2 or len(sig.out_shapes) != 1:
            continue  # only plain 2-operand matmul kernels have an analytic reference
        model = compile_host(km, workdir / d.symbol.replace("$", "_"))
        (a, b), (y,) = run_random(model, sig, seed=seed)
        ref = a.astype(np.float32) @ b.astype(np.float32)
        max_abs = float(np.abs(y - ref).max())
        ok = bool(np.allclose(y, ref, rtol=rtol, atol=atol))
        results.append(
            KernelCheck(symbol=d.symbol, ok=ok, max_abs=max_abs, shapes=f"{sig.in_shapes[0]}x{sig.in_shapes[1]}")
        )
    return results
