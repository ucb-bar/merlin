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
from typing import Any

import numpy as np

_NP_DTYPE = {
    "f32": np.float32, "f64": np.float64, "f16": np.float16,
    "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8, "i1": np.int8,
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
        out_dtypes=[str(t.element_type) for t in ft.outputs.data])


def extract_kernel(module, symbol: str, entry: str = "forward"):
    """A standalone ``builtin.module`` holding just ``symbol``, renamed to ``entry``."""
    from xdsl.dialects.builtin import ModuleOp, StringAttr

    func = next((op for op in module.walk()
                 if op.name == "func.func" and op.sym_name.data == symbol), None)
    if func is None:
        raise KernelBackendError(f"kernel @{symbol} not found")
    clone = func.clone()
    clone.properties["sym_name"] = StringAttr(entry)
    clone.properties.pop("sym_visibility", None)
    return ModuleOp([clone])


def compile_host(kernel_module, workdir: str | Path):
    """Lower one kernel module to a host ``.so`` and load it (RTLD_LOCAL)."""
    from .abi import HostModel
    from .lower import lower_model
    from ..xdsl_dialects._common import text as to_text

    workdir = Path(workdir)
    res = lower_model(to_text(kernel_module), workdir, targets=("host",))
    return HostModel.load(str(res.host_so))


def run_random(model, sig: KernelSignature, seed: int = 0
               ) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
    outputs = [np.zeros(shape, _NP_DTYPE[dt])
               for shape, dt in zip(sig.out_shapes, sig.out_dtypes)]
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


def check_matmul_kernels(outline_result, workdir: str | Path, seed: int = 0,
                         rtol: float = 1e-4, atol: float = 1e-3) -> list[KernelCheck]:
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
        results.append(KernelCheck(symbol=d.symbol, ok=ok, max_abs=max_abs,
                                   shapes=f"{sig.in_shapes[0]}x{sig.in_shapes[1]}"))
    return results
