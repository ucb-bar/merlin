"""Host-side execution of the lowered model via ctypes (the verification oracle).

The lowered module exposes ``_mlir_ciface_forward`` (llvm.emit_c_interface): one
pointer per memref argument, each a rank-N descriptor {alloc, aligned, offset,
sizes[N], strides[N]}, result buffers appended last (buffer-results-to-out-params).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Sequence


_SCALAR_CTYPE = {
    "i64": ctypes.c_int64, "i32": ctypes.c_int32, "i16": ctypes.c_int16,
    "i8": ctypes.c_int8, "i1": ctypes.c_bool,
    "f64": ctypes.c_double, "f32": ctypes.c_float,
}


class ScalarArg:
    """A non-tensor kernel argument passed **by value** through the ciface.

    ``emit_c_interface`` lowers memref args to descriptor pointers but leaves scalar args
    (e.g. a ``cumsum`` accumulator-init ``i64``) passed directly by value — they must not
    be wrapped in a descriptor.
    """

    __slots__ = ("value", "dtype")

    def __init__(self, value, dtype: str):
        self.value, self.dtype = value, dtype

    def to_ctype(self):
        ct = _SCALAR_CTYPE.get(self.dtype)
        if ct is None:
            raise ValueError(f"unsupported scalar arg dtype {self.dtype}")
        return ct(int(self.value) if self.dtype.startswith("i") else float(self.value))


def make_descriptor(rank: int):
    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p),
                    ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64),
                    ("sizes", ctypes.c_int64 * rank),
                    ("strides", ctypes.c_int64 * rank)]
    return MemRefDescriptor


def descriptor(buf_ptr: int, shape: Sequence[int]):
    """Descriptor for a dense row-major buffer at buf_ptr."""
    rank = len(shape)
    desc_t = make_descriptor(rank)
    strides = [1] * rank
    for i in range(rank - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return desc_t(buf_ptr, buf_ptr, 0,
                  (ctypes.c_int64 * rank)(*shape),
                  (ctypes.c_int64 * rank)(*strides))


def _trampoline_source(name: str, n_args: int) -> str:
    """C trampoline forwarding a void* array to the many-arg MLIR ciface.

    ctypes caps calls at 1024 args; the full model's ciface takes >1100 descriptor
    pointers. C has no such limit, so we unroll the call once (generated)."""
    decl_args = ", ".join(["void*"] * n_args)
    call_args = ", ".join(f"d[{i}]" for i in range(n_args))
    return (f"extern void _mlir_ciface_{name}({decl_args});\n"
            f"void merlin_call_{name}(void **d) {{ "
            f"_mlir_ciface_{name}({call_args}); }}\n")


@dataclass
class HostModel:
    """forward() runner on host: blob pointer + arg table."""

    lib: Any
    fn: Any
    trampoline: Any = None

    @classmethod
    def load(cls, so_path: str, name: str = "forward",
             n_args: int | None = None, rtld_global: bool | None = None) -> "HostModel":
        # RTLD_GLOBAL so a trampoline .so can resolve the ciface symbol; only needed for
        # the trampoline path. Default to LOCAL otherwise, so several model libraries can
        # coexist in one process without their shared `forward`/`memrefCopy` symbols
        # clashing (different models export the same names with different signatures).
        if rtld_global is None:
            rtld_global = n_args is not None
        mode = ctypes.RTLD_GLOBAL if rtld_global else ctypes.RTLD_LOCAL
        lib = ctypes.CDLL(so_path, mode=mode)
        fn = getattr(lib, f"_mlir_ciface_{name}", None)
        if fn is not None:
            fn.restype = None
        model = cls(lib, fn)
        if n_args is not None:
            model._build_trampoline(so_path, name, n_args)
        return model

    def _build_trampoline(self, so_path: str, name: str, n_args: int) -> None:
        import subprocess
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix="merlin_tramp_"))
        src = d / "tramp.c"
        out = d / "tramp.so"
        src.write_text(_trampoline_source(name, n_args), encoding="utf-8")
        subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(src), "-o", str(out)],
                       check=True, capture_output=True)
        self.trampoline = ctypes.CDLL(str(out))
        self._call = getattr(self.trampoline, f"merlin_call_{name}")
        self._call.restype = None
        self._call.argtypes = [ctypes.c_void_p]

    def __call__(self, arg_buffers: list) -> None:
        """arg_buffers: ordered args including outputs (appended last). Each entry is a
        ``(pointer, shape)`` tensor (memref descriptor, by ref) or a :class:`ScalarArg`
        (passed by value)."""
        cargs: list = []
        keep: list = []
        for entry in arg_buffers:
            if isinstance(entry, ScalarArg):
                cargs.append(entry.to_ctype())
            else:
                ptr, shape = entry
                d = descriptor(ptr, shape)
                keep.append(d)
                cargs.append(ctypes.byref(d))
        self._descs = keep  # keep alive
        if self.trampoline is not None:
            if len(keep) != len(arg_buffers):
                raise ValueError("the trampoline path does not support scalar args")
            arr = (ctypes.c_void_p * len(keep))(*[ctypes.addressof(d) for d in keep])
            self._call(ctypes.cast(arr, ctypes.c_void_p))
        else:
            self.fn(*cargs)
