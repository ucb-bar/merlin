"""Explicit host-only PyTorch 2.16 MKL VML sin/cos reference route.

This module never participates in RV64 lowering.  It accepts only the attested
Torch CPU wheel used for the TorchAO capture; the wheel's MKL VML implementation
is the source arithmetic for contiguous f32 ``aten.sin`` and ``aten.cos``.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np

TORCH_VERSION = "2.16.0.dev20261104+cpu"
TORCH_COMMIT = "7f24a19c0967db04f2d5ba44c1e56eca489932ec"
LIBTORCH_CPU_SHA256 = "3e1e4050d94c3ac68b9c8e457e2be860406ee06a573e93d4173cf837e4e94e4a"
QUANTIZATION = "int8_dyn_act_int8_weight"


class TorchVmlUnavailable(RuntimeError):
    """The exact captured host math library is unavailable."""


def load_attested_vml():
    """Return the captured wheel's shared library after checking its source and bytes."""
    try:
        import torch
    except ImportError as error:
        raise TorchVmlUnavailable("the captured PyTorch CPU wheel is required") from error
    if (
        torch.__version__ != TORCH_VERSION
        or torch.version.git_version != TORCH_COMMIT
        or torch.backends.cpu.get_cpu_capability() != "AVX2"
        or not torch.backends.mkl.is_available()
    ):
        raise TorchVmlUnavailable("PyTorch source, AVX2 dispatch, or MKL build differs from the capture")
    library = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
    if not library.is_file():
        raise TorchVmlUnavailable("captured libtorch_cpu.so is absent")
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != LIBTORCH_CPU_SHA256:
        raise TorchVmlUnavailable("libtorch_cpu.so bytes differ from the captured wheel")
    return ctypes.CDLL(str(library)), digest


def classify_kernel(kfn) -> dict[str, object] | None:
    """Recognize only a one-input, identity-layout, provenance-marked f32 sin/cos."""
    from xdsl.dialects.builtin import TensorType

    block = kfn.body.blocks[0]
    arg_ids = {id(arg): i for i, arg in enumerate(block.args)}
    ret = next((op for op in block.ops if op.name == "func.return"), None)
    if ret is None or len(ret.operands) != 1:
        return None
    generic = getattr(ret.operands[0], "owner", None)
    if generic is None or generic.name != "linalg.generic" or len(generic.inputs) != 1:
        return None
    aten = getattr(generic.attributes.get("prov.aten"), "data", None)
    kind = {"aten.sin.default": "sin", "aten.cos.default": "cos"}.get(aten)
    if kind is None or getattr(generic.attributes.get("prov.op"), "data", None) != kind:
        return None
    source, result = generic.inputs[0], generic.results[0]
    if id(source) not in arg_ids:
        return None
    st, rt = source.type, result.type
    if not isinstance(st, TensorType) or not isinstance(rt, TensorType):
        return None
    if st.element_type != rt.element_type or str(st.element_type) != "f32" or st.get_shape() != rt.get_shape():
        return None
    shape = st.get_shape()
    try:
        maps = [m.data for m in generic.indexing_maps]
        identity = tuple(f"d{i}" for i in range(len(shape)))
        if len(maps) != 2 or any(tuple(str(x) for x in m.results) != identity for m in maps):
            return None
        if len(generic.iterator_types) != len(shape) or any(
            "parallel" not in str(x).lower() for x in generic.iterator_types
        ):
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    body = list(generic.body.blocks[0].ops)
    if [op.name for op in body] != [f"math.{kind}", "linalg.yield"]:
        return None
    if body[0].operands[0] is not generic.body.blocks[0].args[0] or body[1].operands[0] is not body[0].results[0]:
        return None
    return {"input": arg_ids[id(source)], "kind": kind}


def evaluate(lib, kind: str, values: np.ndarray) -> np.ndarray:
    """Evaluate contiguous f32 values through the attested MKL high-accuracy VML symbol."""
    if kind not in {"sin", "cos"}:
        raise ValueError("only sin and cos are available")
    if values.dtype != np.float32:
        raise TypeError("VML route requires f32")
    source = np.ascontiguousarray(values)
    result = np.empty_like(source)
    if source.size > 2**31 - 1:
        raise ValueError("VML LP64 count exceeds int32")
    fn = getattr(lib, "vmsSin" if kind == "sin" else "vmsCos")
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    fn.restype = None
    # VML_HA is 1. The other PyTorch VML flags govern denormal control and
    # error reporting; on the attested wheel this mode produces the same values,
    # including subnormal and nonfinite holdouts. The library hash bounds this claim.
    fn(source.size, source.ctypes.data, result.ctypes.data, 1)
    return result
