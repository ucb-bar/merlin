"""Host-side execution of the lowered model via ctypes (the verification oracle).

The lowered module exposes ``_mlir_ciface_forward`` (llvm.emit_c_interface): one
pointer per memref argument, each a rank-N descriptor {alloc, aligned, offset,
sizes[N], strides[N]}, result buffers appended last (buffer-results-to-out-params).
One remaining plain scalar result is returned by value, with an explicitly
selected ctypes result type; it is never a rank-zero memref output argument.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from merlin.compiler.lower.c.weights import load_safetensors_header

# ---- abi ---------------------------------------------------------------------------------------

_SCALAR_CTYPE = {
    "i64": ctypes.c_int64,
    "i32": ctypes.c_int32,
    "i16": ctypes.c_int16,
    "i8": ctypes.c_int8,
    "i1": ctypes.c_bool,
    "f64": ctypes.c_double,
    "f32": ctypes.c_float,
}


def scalar_result_ctype(dtype: str | None):
    """The selected plain scalar C return, or the historical void return.

    This supports neither aggregate results nor index/extended floating ABIs.
    The caller must derive the selection from the actual entry function type.
    """
    if dtype is None:
        return None
    if not isinstance(dtype, str) or dtype not in _SCALAR_CTYPE:
        raise ValueError(f"unsupported scalar result dtype {dtype!r}")
    return _SCALAR_CTYPE[dtype]


class ScalarArg:
    """A non-tensor kernel argument passed **by value** through the ciface.

    ``emit_c_interface`` lowers memref args to descriptor pointers but leaves scalar args
    (e.g. a ``cumsum`` accumulator-init ``i64``) passed directly by value — they must not
    be wrapped in a descriptor.
    """

    __slots__ = ("value", "dtype", "_storage")

    def __init__(self, value, dtype: str):
        self.value, self.dtype = value, dtype
        self._storage = None

    @classmethod
    def from_storage(cls, storage, dtype: str):
        """Copy exact native scalar words without a Python floating conversion.

        The raw form keeps ``value=None``; ``to_ctype`` supplies the original
        word, including signaling NaNs, signed zeros and wide integers.
        """
        ct = _SCALAR_CTYPE.get(dtype)
        if ct is None or not isinstance(storage, (bytes, bytearray, memoryview)):
            raise ValueError("native scalar storage needs a supported dtype and bytes")
        raw = bytes(storage)
        if len(raw) != ctypes.sizeof(ct) or dtype == "i1" and raw not in (b"\x00", b"\x01"):
            raise ValueError("native scalar storage differs from its exact ABI width/value")
        result = cls(None, dtype)
        result._storage = raw
        return result

    def to_ctype(self):
        ct = _SCALAR_CTYPE.get(self.dtype)
        if ct is None:
            raise ValueError(f"unsupported scalar arg dtype {self.dtype}")
        if self._storage is not None:
            return ct.from_buffer_copy(self._storage)
        return ct(int(self.value) if self.dtype.startswith("i") else float(self.value))


@dataclass(frozen=True)
class StridedMemRefArg:
    """A ranked buffer with explicit physical strides, staged at the host ABI.

    ``storage_elements`` is the accessible capacity starting at ``pointer``;
    ``offset`` and ``strides`` are element indices, not bytes. The caller owns
    the backing allocation and must keep it alive during the native call. This
    checks the declared footprint, but cannot prove that an arbitrary pointer
    really owns the declared capacity. ``access`` must be ``input``, ``output``
    or ``inout``: inputs are packed before the call; outputs are scattered back
    afterward. Current statically shaped host lowering assumes dense row-major
    memrefs, so this is intentionally *not* a zero-copy pitched descriptor.
    """

    pointer: int
    shape: Sequence[int]
    strides: Sequence[int]
    storage_elements: int
    dtype: str
    access: str
    offset: int = 0

    def __post_init__(self) -> None:
        maximum = (1 << 63) - 1
        if not isinstance(self.pointer, int) or isinstance(self.pointer, bool) or self.pointer <= 0:
            raise ValueError("memref pointer must be a nonzero integer address")
        if self.pointer >= (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)):
            raise ValueError("memref pointer exceeds the host pointer width")
        shape, strides = tuple(self.shape), tuple(self.strides)
        if len(shape) != len(strides):
            raise ValueError("memref shape and strides must have the same rank")
        for label, values, lower in (("shape", shape, 0), ("strides", strides, 1)):
            if any(not isinstance(v, int) or isinstance(v, bool) or not lower <= v <= maximum for v in values):
                raise ValueError(f"memref {label} must contain int64 values >= {lower}")
        for label, value in (("storage_elements", self.storage_elements), ("offset", self.offset)):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
                raise ValueError(f"memref {label} must be a nonnegative int64 value")
        if not isinstance(self.dtype, str) or self.dtype not in _SCALAR_CTYPE:
            raise ValueError(f"unsupported memref element dtype {self.dtype!r}")
        if not isinstance(self.access, str) or self.access not in {"input", "output", "inout"}:
            raise ValueError("memref access must be input, output, or inout")
        address_limit = 1 << (ctypes.sizeof(ctypes.c_void_p) * 8)
        last_byte = self.pointer + self.storage_elements * ctypes.sizeof(_SCALAR_CTYPE[self.dtype]) - 1
        if self.storage_elements and last_byte >= address_limit:
            raise ValueError("memref declared storage exceeds the host pointer address space")
        # A zero-extent memref accesses no element. For a nonempty memref,
        # include the last element of every dimension in the maximum address.
        nonempty = all(shape)
        last = (
            self.offset + sum((size - 1) * stride for size, stride in zip(shape, strides)) if nonempty else self.offset
        )
        if last > maximum or (last >= self.storage_elements if nonempty else self.offset > self.storage_elements):
            raise ValueError("memref logical footprint exceeds declared storage_elements")
        if nonempty and self.access != "input":
            # Sufficient (not necessary) injectivity condition. Every active
            # dimension must step beyond the entire span of smaller strides.
            # Reject ambiguous output scatter instead of silently overwriting
            # several logical results at the same physical element.
            covered_span = 0
            for stride, size in sorted((stride, size) for size, stride in zip(shape, strides) if size > 1):
                if stride <= covered_span:
                    raise ValueError("writable memref strides may alias logical elements")
                covered_span += (size - 1) * stride
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)


def _addressed_bytes(arg: StridedMemRefArg) -> tuple[int, int]:
    """Conservative byte interval spanning all logical elements."""
    item_bytes = ctypes.sizeof(_SCALAR_CTYPE[arg.dtype])
    first = arg.pointer + arg.offset * item_bytes
    if any(size == 0 for size in arg.shape):
        return first, first
    last = arg.offset + sum((size - 1) * stride for size, stride in zip(arg.shape, arg.strides))
    return first, arg.pointer + (last + 1) * item_bytes


def _validate_staged_aliases(arg_buffers: list) -> None:
    """Fail before native execution when staged copyback could alter another arg."""
    staged = [entry for entry in arg_buffers if isinstance(entry, StridedMemRefArg)]
    for index, left in enumerate(staged):
        if left.access == "input":
            continue
        lo, hi = _addressed_bytes(left)
        if lo == hi:
            continue
        for other_index, right in enumerate(staged):
            if other_index == index:
                continue
            other_lo, other_hi = _addressed_bytes(right)
            if lo < other_hi and other_lo < hi:
                raise ValueError("staged memref arguments have overlapping physical storage")
        for entry in arg_buffers:
            if isinstance(entry, (StridedMemRefArg, ScalarArg)):
                continue
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                pointer = entry[0]
                if isinstance(pointer, int) and lo <= pointer < hi:
                    raise ValueError("staged writable memref overlaps a dense argument address")


def make_descriptor(rank: int):
    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRefDescriptor


def descriptor(buf_ptr: int, shape: Sequence[int]):
    """Descriptor for a dense row-major buffer at buf_ptr."""
    rank = len(shape)
    desc_t = make_descriptor(rank)
    strides = [1] * rank
    for i in range(rank - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return desc_t(buf_ptr, buf_ptr, 0, (ctypes.c_int64 * rank)(*shape), (ctypes.c_int64 * rank)(*strides))


def _copy_pitched(arg: StridedMemRefArg, dense, *, to_dense: bool) -> None:
    """Copy logical elements, never padding, between declared storage and dense scratch."""
    if any(extent == 0 for extent in arg.shape):
        return
    item_bytes = ctypes.sizeof(_SCALAR_CTYPE[arg.dtype])
    dense_ptr = ctypes.addressof(dense)
    if not arg.shape:
        pairs = [(0, arg.offset, 1)]
    elif arg.strides[-1] == 1:
        # Preserve contiguous inner rows as one transfer, including on pitched
        # 2-D/ND buffers. The generic branch handles arbitrary positive strides.
        width = arg.shape[-1]
        pairs = (
            (row * width, arg.offset + sum(i * s for i, s in zip(outer, arg.strides)), width)
            for row, outer in enumerate(product(*(range(n) for n in arg.shape[:-1])))
        )
    else:
        pairs = (
            (linear, arg.offset + sum(i * s for i, s in zip(index, arg.strides)), 1)
            for linear, index in enumerate(product(*(range(n) for n in arg.shape)))
        )
    for dense_index, physical_index, count in pairs:
        dense_address = dense_ptr + dense_index * item_bytes
        physical_address = arg.pointer + physical_index * item_bytes
        ctypes.memmove(
            dense_address if to_dense else physical_address,
            physical_address if to_dense else dense_address,
            count * item_bytes,
        )


def _trampoline_source(name: str, n_args: int) -> str:
    """C trampoline forwarding a void* array to the many-arg MLIR ciface.

    ctypes caps calls at 1024 args; the full model's ciface takes >1100 descriptor
    pointers. C has no such limit, so we unroll the call once (generated)."""
    decl_args = ", ".join(["void*"] * n_args)
    call_args = ", ".join(f"d[{i}]" for i in range(n_args))
    return (
        f"typedef void (*Entry)({decl_args});\n"
        f"void merlin_call_{name}(Entry entry, void **d) {{ "
        f"entry({call_args}); }}\n"
    )


@dataclass(frozen=True)
class PrivateHostImagePolicy:
    """Explicit sibling staging in an invocation-owned native build directory.

    This is a trusted build/load choice, not filesystem authorization or
    dependency/build provenance. The owner must permit temporary creation and
    cleanup. Default artifact readers do not acquire this write requirement.
    """

    directory: Path

    def __post_init__(self) -> None:
        if (
            not isinstance(self.directory, Path)
            or not self.directory.is_absolute()
            or ".." in self.directory.parts
            or self.directory.resolve(strict=True) != self.directory
            or not self.directory.is_dir()
        ):
            raise ValueError("private host image requires an exact absolute build directory")


def _load_image(so_path: str, mode: int, policy: PrivateHostImagePolicy) -> tuple[Any, str]:
    """Load a private copy, preserving the source directory's dependency lookup.

    dlopen may return an already loaded image when its pathname is reused after
    recompilation. A private sibling freezes the requested bytes and gives the
    loader a new pathname without changing $ORIGIN. The host build directory
    must permit temporary image creation. Dependencies themselves are not frozen
    by this helper; their ownership belongs to the selected native build service.
    """
    source_path = Path(so_path).absolute()
    if type(policy) is not PrivateHostImagePolicy or source_path.parent != policy.directory:
        raise ValueError("private host image must remain in its selected build directory")
    image = None
    digest = hashlib.sha256()

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    try:
        with source_path.open("rb") as source:
            before = identity(os.fstat(source.fileno()))
            with tempfile.NamedTemporaryFile(
                prefix=".merlin-host-image-",
                suffix=source_path.suffix,
                dir=policy.directory,
                delete=False,
            ) as destination:
                image = Path(destination.name)
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
            if before != identity(os.fstat(source.fileno())) or before != identity(source_path.stat()):
                raise ValueError("native shared image changed while staging its load")
        # Include bytes in the load name as well as the temporary nonce: even a
        # later reused temporary basename cannot alias a different loaded image.
        content_path = image.with_name(image.name + "-" + digest.hexdigest() + source_path.suffix)
        image.rename(content_path)
        image = content_path
        image.chmod(0o400)
        return ctypes.CDLL(str(image), mode=mode), digest.hexdigest()
    finally:
        if image is not None:
            image.unlink(missing_ok=True)


@dataclass
class HostModel:
    """forward() runner on host: blob pointer + arg table."""

    lib: Any
    fn: Any
    trampoline: Any = None
    image_sha256: str | None = None
    scalar_result_dtype: str | None = None
    descriptor_transport: Any = None

    @classmethod
    def load(
        cls,
        so_path: str,
        name: str = "forward",
        n_args: int | None = None,
        rtld_global: bool | None = None,
        *,
        image_policy: PrivateHostImagePolicy | None = None,
        scalar_result_dtype: str | None = None,
        descriptor_transport=None,
    ) -> HostModel:
        """Load an artifact, or explicitly freeze a fresh owned build image.

        Default loading retains native loader caching and read-only compatibility;
        it supplies no current-image digest. The selected policy requires writable
        sibling staging and identifies only the image, not its dependencies.
        """
        if descriptor_transport is not None:
            from merlin.compiler.lower.host.descriptor import HostDescriptorTransport

            if type(descriptor_transport) is not HostDescriptorTransport or image_policy is None:
                raise ValueError("selected host descriptors require observed transport and an exact private image")
            descriptor_transport.require_image(
                so_path, name=name, scalar_result_dtype=scalar_result_dtype, n_args=n_args
            )
        result_type = scalar_result_ctype(scalar_result_dtype)
        # Give the trampoline this library's exact entry address instead of asking the dynamic
        # loader to resolve a process-global symbol. Keep even many-argument models LOCAL: their
        # shared forward/memrefCopy names could otherwise bind a later A/B variant to the first
        # model's implementation.
        if rtld_global is None:
            rtld_global = False
        mode = ctypes.RTLD_GLOBAL if rtld_global else ctypes.RTLD_LOCAL
        if image_policy is None:
            # Preserve artifact loading, including read-only paths and native
            # dependency search. A reused pathname cannot attest current bytes.
            lib, image_sha256 = ctypes.CDLL(so_path, mode=mode), None
        else:
            lib, image_sha256 = _load_image(so_path, mode, image_policy)
        fn = getattr(lib, f"_mlir_ciface_{name}", None)
        if fn is None:
            raise ValueError(f"{so_path}: missing _mlir_ciface_{name}")
        fn.restype = result_type
        model = cls(lib, fn, image_sha256=image_sha256, scalar_result_dtype=scalar_result_dtype)
        if descriptor_transport is not None:
            descriptor_transport.verify()
            expected = next(
                digest for path, digest in descriptor_transport.pins if path == str(descriptor_transport.image)
            )
            if image_sha256 != expected:
                raise ValueError("selected host descriptor image differs from the actually loaded image")
            fn.argtypes = [ctypes.c_void_p] * len(descriptor_transport.selection.source.storage)
            model.descriptor_transport = descriptor_transport
            model._descriptor_transport = descriptor_transport
            model._descriptor_function = fn
        if n_args is not None:
            model._build_trampoline(so_path, name, n_args)
        return model

    def _build_trampoline(self, so_path: str, name: str, n_args: int) -> None:
        if self.scalar_result_dtype is not None:
            raise ValueError("the trampoline path does not support scalar results")
        import subprocess
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix="merlin_tramp_"))
        src = d / "tramp.c"
        out = d / "tramp.so"
        src.write_text(_trampoline_source(name, n_args), encoding="utf-8")
        subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(src), "-o", str(out)], check=True, capture_output=True)
        self.trampoline = ctypes.CDLL(str(out))
        self._call = getattr(self.trampoline, f"merlin_call_{name}")
        self._call.restype = None
        self._call.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    def __call__(self, arg_buffers: list) -> Any:
        """arg_buffers: ordered args including outputs (appended last). Each entry is a
        ``(pointer, shape)`` dense tensor, :class:`StridedMemRefArg` pitched
        tensor (staged through dense storage), or a :class:`ScalarArg`
        (passed by value). Return the explicitly selected plain scalar result,
        or None for a void entry; tensor results remain appended output args.
        """
        if self.descriptor_transport is not None or getattr(self, "_descriptor_transport", None) is not None:
            if (
                self.descriptor_transport is not getattr(self, "_descriptor_transport", None)
                or self.fn is not getattr(self, "_descriptor_function", None)
                or self.trampoline is not None
                or self.scalar_result_dtype is not None
                or self.fn.restype is not None
                or self.fn.argtypes != [ctypes.c_void_p] * len(self.descriptor_transport.selection.source.storage)
            ):
                raise ValueError("selected host descriptor function/ABI changed")
            self._descs = self.descriptor_transport.invoke(self.fn, arg_buffers)
            if (
                self.descriptor_transport is not self._descriptor_transport
                or self.fn is not self._descriptor_function
                or self.trampoline is not None
                or self.scalar_result_dtype is not None
                or self.fn.restype is not None
                or self.fn.argtypes != [ctypes.c_void_p] * len(self.descriptor_transport.selection.source.storage)
            ):
                raise ValueError("selected host descriptor function/ABI changed during execution")
            return None
        _validate_staged_aliases(arg_buffers)
        cargs: list = []
        keep: list = []
        scratch: list = []
        copyback: list = []
        for entry in arg_buffers:
            if isinstance(entry, ScalarArg):
                cargs.append(entry.to_ctype())
            else:
                if isinstance(entry, StridedMemRefArg):
                    element_count = 1
                    for extent in entry.shape:
                        element_count *= extent
                    dense = (_SCALAR_CTYPE[entry.dtype] * element_count)()
                    if entry.access in {"input", "inout"}:
                        _copy_pitched(entry, dense, to_dense=True)
                    scratch.append(dense)
                    if entry.access in {"output", "inout"}:
                        copyback.append((entry, dense))
                    d = descriptor(ctypes.addressof(dense), entry.shape)
                else:
                    ptr, shape = entry
                    d = descriptor(ptr, shape)
                keep.append(d)
                cargs.append(ctypes.byref(d))
        self._descs = keep  # keep alive
        self._scratch = scratch
        result = None
        if self.trampoline is not None:
            if len(keep) != len(arg_buffers):
                raise ValueError("the trampoline path does not support scalar args")
            arr = (ctypes.c_void_p * len(keep))(*[ctypes.addressof(d) for d in keep])
            self._call(ctypes.cast(self.fn, ctypes.c_void_p), ctypes.cast(arr, ctypes.c_void_p))
        else:
            result = self.fn(*cargs)
        for entry, dense in copyback:
            _copy_pitched(entry, dense, to_dense=False)
        return result


# ---- model_runner ------------------------------------------------------------------------------
# Run the whole lowered model on the host (ctypes) — the pre-spike oracle.
#
# Assembles all ``@forward`` arguments in manifest order: 1106 weights as pointers into
# the safetensors payload blob, 4 rotary buffers + 9 runtime inputs from the reference
# capture, output appended last (buffer-results-to-out-params), then drives the 10-step
# denoise loop by varying the timestep input — same loop the bare-metal main runs.

NP_DTYPES = {"F32": "float32", "BF16": "uint16", "I64": "int64", "I32": "int32", "I8": "int8", "BOOL": "bool"}


def parse_forward_signature(mlir_path: str | Path) -> list[tuple[list[int], str]]:
    """[(shape, dtype_str)] for every @forward argument, read from the parsed function type."""
    from merlin.compiler.ir.mlir_query import forward_signature

    inputs, _ = forward_signature(mlir_path)
    return inputs


@dataclass
class ModelIO:
    """Buffers and the argument list for one forward call."""

    arrays: dict[int, object]  # arg_index -> numpy array (kept alive)
    arg_list: list[tuple[int, list[int]]]  # (pointer, shape) per arg + output
    output: object  # 1x50x32 f32


def build_io(
    mlir_path: str | Path,
    manifest_path: str | Path,
    safetensors_path: str | Path,
    reference_npz: str | Path,
    lifted_npz: str | Path,
) -> ModelIO:
    import numpy as np

    sig = parse_forward_signature(mlir_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    header, payload_off = load_safetensors_header(safetensors_path)
    blob = np.fromfile(safetensors_path, dtype=np.uint8)[payload_off:]
    ref = np.load(reference_npz)
    lifted = np.load(lifted_npz)

    arrays: dict[int, np.ndarray] = {}
    args: list[tuple[int, list[int]]] = []
    lifted_names = sorted(lifted.files)
    li = 0
    for idx in range(len(sig)):
        meta = manifest[str(idx)]
        shape, _ = sig[idx]
        if meta["kind"] in ("param", "buffer"):
            name = meta.get("weight") or meta["name"]
            if name in header:
                begin, _ = header[name]["data_offsets"]
                ptr = blob.ctypes.data + begin
                args.append((ptr, shape))
                continue
            key = "buf::" + name[2:].replace("_", ".")
            cands = [
                k for k in ref.files if k.startswith("buf::") and k.replace(".", "_").replace("buf::", "b_") == name
            ]
            arr = np.ascontiguousarray(ref[cands[0]] if cands else ref[key])
        elif meta["name"].startswith("c_lifted_tensor_"):
            arr = np.ascontiguousarray(lifted[lifted_names[li]])
            li += 1
        else:
            arr = np.ascontiguousarray(ref[meta["name"]])
        arrays[idx] = arr
        args.append((arr.ctypes.data, shape))

    out = np.zeros((1, 50, 32), dtype=np.float32)
    arrays[-1] = out
    arrays[-2] = blob  # weight pointers index into this — must stay alive
    args.append((out.ctypes.data, [1, 50, 32]))
    return ModelIO(arrays=arrays, arg_list=args, output=out)


def run_denoise_loop(so_path: str | Path, io: ModelIO, manifest_path: str | Path, steps: int = 10):
    """Euler flow-matching loop: x += dt * v(x, t), t = 1.0 -> 0.1. Returns final x."""
    import numpy as np

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    noise_idx = next(int(k) for k, v in manifest.items() if v.get("name") == "noise")
    t_idx = next(int(k) for k, v in manifest.items() if v.get("name") == "c_lifted_tensor_0")

    model = HostModel.load(str(so_path))
    x_t = io.arrays[noise_idx]
    t_buf = io.arrays[t_idx]
    dt = -1.0 / steps
    per_step = []
    for i in range(steps):
        t_buf[...] = 1.0 + i * dt
        model(io.arg_list)
        x_t[...] = x_t + dt * io.output
        per_step.append(x_t.copy())
    return np.stack(per_step)


# ---- kernel_backend ----------------------------------------------------------------------------
# Per-kernel backend + checker — compile one outlined dispatch in isolation.
#
# Outlining (``xdsl_dialects.lowering.outline``) keeps the view/glue ops (``extract_slice``,
# ``expand_shape``, …) in the *driver*; each kernel func is clean linalg + its cloned
# accumulator init. That has two payoffs this module realizes:
#
# 1. **It routes around the whole-model xDSL printer bug.** The monolithic model cannot be
#    re-emitted through xDSL (rank-reducing ``extract_slice`` mis-prints), so the whole-model
#    path uses textual preprocessing. A single kernel func contains no ``extract_slice``, so
#    it round-trips and compiles through the normal ``lower_model`` path.
# 2. **Each kernel is independently checkable.** ``check_matmul_kernels`` compiles every
#    contraction dispatch and gates it against the analytic numpy reference (``A @ B``) — the
#    numerically critical ops, where the whole-model NaN historically lived. First divergence
#    localizes a bug to one kernel in seconds instead of bisecting the whole model.
#
# Several kernel libraries coexist in one process safely because ``HostModel.load`` opens
# them ``RTLD_LOCAL`` (no trampoline needed at this arity), so their shared
# ``_mlir_ciface_forward``/``memrefCopy`` symbols don't clash.

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


def extract_kernel(module, symbol: str, entry: str = "forward"):
    """A standalone ``builtin.module`` holding just ``symbol``, renamed to ``entry``."""
    from xdsl.dialects.builtin import ModuleOp, StringAttr

    func = next((op for op in module.walk() if op.name == "func.func" and op.sym_name.data == symbol), None)
    if func is None:
        raise KernelBackendError(f"kernel @{symbol} not found")
    clone = func.clone()
    clone.properties["sym_name"] = StringAttr(entry)
    clone.properties.pop("sym_visibility", None)
    return ModuleOp([clone])


def host_scalar_result_dtype(kernel_module, entry: str = "forward") -> str | None:
    """Derive the by-value result left after tensor results become out-params.

    Multiple scalars need an aggregate C return ABI, which this runner does not
    support. Rank-zero tensors are still tensor output descriptors.
    """
    from xdsl.dialects.builtin import TensorType

    function = next((op for op in kernel_module.walk() if op.name == "func.func" and op.sym_name.data == entry), None)
    if function is None:
        return None  # Ordinary lowering retains its missing-entry refusal.
    scalars = [t for t in function.function_type.outputs if not isinstance(t, TensorType)]
    if len(scalars) > 1:
        raise KernelBackendError("host kernel aggregate scalar results are unsupported")
    dtype = str(scalars[0]) if scalars else None
    scalar_result_ctype(dtype)
    return dtype


def compile_host(kernel_module, workdir: str | Path, *, retain_llvm_dialect: bool = False, descriptor_selection=None):
    """Lower one kernel module to a host ``.so`` and load it (RTLD_LOCAL)."""
    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.lower.model.lower import lower_model

    if descriptor_selection is not None:
        from merlin.compiler.lower.host.descriptor import HostDescriptorSelection

        if type(descriptor_selection) is not HostDescriptorSelection:
            raise ValueError("host compilation requires an explicit typed descriptor selection")
        descriptor_selection.require_module(kernel_module)

    # Lowering creates a missing build directory. Resolve its parent aliases
    # first so the selected private sibling retains the compiler's exact origin.
    result_dtype = host_scalar_result_dtype(kernel_module)
    workdir = Path(workdir).resolve()
    res = lower_model(
        # Some custom operation printers omit attribute dictionaries. Preserve
        # the full typed source before preprocessing records its correspondence.
        to_text(kernel_module, generic=True),
        workdir,
        targets=("host",),
        retain_llvm_dialect=retain_llvm_dialect or descriptor_selection is not None,
    )
    transport = None
    if descriptor_selection is not None:
        from merlin.compiler.lower.host.descriptor import observe_host_descriptor_transport

        transport = observe_host_descriptor_transport(selection=descriptor_selection, result=res)
    if transport is None:
        return HostModel.load(
            str(res.host_so),
            image_policy=PrivateHostImagePolicy(workdir.resolve(strict=True)),
            scalar_result_dtype=result_dtype,
        )
    return HostModel.load(
        str(res.host_so),
        name=descriptor_selection.source.entry_symbol,
        image_policy=PrivateHostImagePolicy(workdir.resolve(strict=True)),
        scalar_result_dtype=result_dtype,
        descriptor_transport=transport,
    )


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


# ---- cli ---------------------------------------------------------------------------------------
# Explicit whole-model MLIR lowering and inspection, without capture or deployment.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin lower", description="Lower supplied MLIR through the shared whole-model pipeline; never run it."
    )
    parser.add_argument("input", type=Path, help="existing linalg-on-tensors MLIR file")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory (must not already exist)")
    parser.add_argument(
        "--target",
        action="append",
        choices=("host", "riscv"),
        default=[],
        help="also build host shared library or RISC-V object; repeat for both (default: LLVM IR only)",
    )
    parser.add_argument("--textual", action="store_true", help="use existing text-only preprocessing instead of xDSL")
    parser.add_argument("--feature", action="append", help="explicit registered lowering feature; repeat to combine")
    parser.add_argument(
        "--ir-audit",
        nargs="?",
        const="exact",
        default=False,
        choices=("exact", "compact", "both"),
        help="retain exact stages (may be large), compact inspection-only views, or both",
    )
    parser.add_argument(
        "--audit-sidecar",
        type=Path,
        action="append",
        default=[],
        help="bind an existing weights/manifest file in the audit; repeat for each dependency",
    )
    args = parser.parse_args(argv)
    if args.audit_sidecar and not args.ir_audit:
        parser.error("--audit-sidecar requires --ir-audit")
    for source in (args.input, *args.audit_sidecar):
        if not source.is_file():
            parser.error(f"input is not an existing file: {source}")
    # Reserve a fresh destination before invoking the pipeline. Existing output,
    # including a dangling symlink, is never an implicit overwrite permission.
    output = args.out.absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"output already exists: {output}; choose a fresh --out directory")

    try:
        from merlin.compiler.lower.model.lower import lower_model_file

        output.mkdir(parents=True, exist_ok=False)
        result = lower_model_file(
            args.input,
            output,
            targets=tuple(dict.fromkeys(args.target)),
            textual=args.textual,
            features=frozenset(args.feature) if args.feature else None,
            ir_audit=args.ir_audit,
            audit_sidecars=tuple(args.audit_sidecar),
        )
    except Exception as exc:
        print(f"merlin lower: {type(exc).__name__}: {exc}; output location: {output}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "workdir": str(result.workdir),
                "ll_path": str(result.ll_path),
                "host_so": str(result.host_so) if result.host_so is not None else None,
                "riscv_obj": str(result.riscv_obj) if result.riscv_obj is not None else None,
                "audit_index": str(result.audit_index) if result.audit_index is not None else None,
                "stats": result.stats,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---- ir_inspection -----------------------------------------------------------------------------
# Native torch-MLIR printer inspection, never an executable transformation.
#
# The function's tracked source is embedded in the existing isolated compiler worker;
# the compiler environment need not import Merlin or reparse another dialect's text.


def _inspection_pass_manager(pipeline, context):
    """Preserve native pass execution; enable its own per-pass printer."""
    from pathlib import Path

    from torch_mlir.passmanager import PassManager

    manager = PassManager.parse(pipeline, context)
    root = Path(globals()["_MERLIN_INSPECTION_DIRECTORY"]) / "passes"
    root.mkdir(exist_ok=True)
    # The worker is serial and the audit root is fresh. Number invocations so
    # feature-driven pipeline segments keep both their order and native filenames.
    segment = root / f"segment-{len(list(root.iterdir())):04d}"
    segment.mkdir()
    (segment / "pipeline.txt").write_text(pipeline, encoding="utf-8")
    manager.enable_ir_printing(
        print_before_all=globals()["_MERLIN_PRINT_BEFORE"],
        print_after_all=globals()["_MERLIN_PRINT_AFTER"],
        large_elements_limit=64,
        large_resource_limit=64,
        tree_printing_dir_path=str(segment),
    )
    return manager


def _write_view(operation, directory, name, keep_exact):
    import hashlib
    import json
    from pathlib import Path

    root = Path(directory) / "native"
    root.mkdir(exist_ok=True)
    exact = operation.get_asm().encode("utf-8")
    parent = hashlib.sha256(exact).hexdigest()
    header = "// INSPECTION ONLY: elided tensor/resource contents; NOT EXECUTABLE IR.\n"
    header += "// Exact native get_asm() parent SHA-256: " + parent + "\n"
    compact = operation.get_asm(large_elements_limit=64, large_resource_limit=64)
    payload = (header + compact).encode("utf-8")
    filename = name + ".inspection.mlir.txt"
    (root / filename).write_bytes(payload)
    exact_name = name + ".exact.mlir" if keep_exact else None
    if exact_name:
        (root / exact_name).write_bytes(exact)
    record = {
        "name": name,
        "representation": "inspection-only",
        "executable": False,
        "file": "native/" + filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parent": {
            "serialization": "torch_mlir.Operation.get_asm()",
            "sha256": parent,
            "bytes": len(exact),
            "file": "native/" + exact_name if exact_name else None,
        },
        "limits": {"large_elements_limit": 64, "large_resource_limit": 64},
        "sidecars": "index.json#sidecars",
    }
    (root / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def bind_inspection(
    source: str,
    directory: str | None,
    *,
    keep_exact: bool = False,
    print_before: bool = True,
    print_after: bool = True,
) -> str:
    """Bind declared runner template hooks; absent hooks refuse rather than skip views.

    ``print_before``/``print_after`` choose which side of each pass the native printer dumps (an audit
    keeps both; a compile trace asks only for the side it was told to dump)."""
    hooks = {"# __MERLIN_INSPECT_PARSED__": "00-parsed", "# __MERLIN_INSPECT_LOWERED__": "01-lowered"}
    for token, name in hooks.items():
        if source.count(token) != 1:
            raise ValueError(f"compiler runner must declare one {token} hook")
        replacement = f"_write_view(module.operation, {directory!r}, {name!r}, {keep_exact!r})" if directory else ""
        source = source.replace(token, replacement)
    if not directory:
        return source
    if "PassManager.parse(" not in source:
        raise ValueError("compiler runner must declare a native pass-manager invocation")
    source = source.replace("PassManager.parse(", "_inspection_pass_manager(")
    return (
        f"_MERLIN_INSPECTION_DIRECTORY = {directory!r}\n"
        f"_MERLIN_PRINT_BEFORE = {bool(print_before)!r}\n"
        f"_MERLIN_PRINT_AFTER = {bool(print_after)!r}\n"
        + inspect.getsource(_inspection_pass_manager)
        + "\n"
        + inspect.getsource(_write_view)
        + "\n"
        + source
    )
