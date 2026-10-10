"""Host-side execution of the lowered model via ctypes (the verification oracle).

The lowered module exposes ``_mlir_ciface_forward`` (llvm.emit_c_interface): one
pointer per memref argument, each a rank-N descriptor {alloc, aligned, offset,
sizes[N], strides[N]}, result buffers appended last (buffer-results-to-out-params).
One remaining plain scalar result is returned by value, with an explicitly
selected ctypes result type; it is never a rank-zero memref output argument.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

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

    __slots__ = ("value", "dtype")

    def __init__(self, value, dtype: str):
        self.value, self.dtype = value, dtype

    def to_ctype(self):
        ct = _SCALAR_CTYPE.get(self.dtype)
        if ct is None:
            raise ValueError(f"unsupported scalar arg dtype {self.dtype}")
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
            from .host_descriptor_call import HostDescriptorTransport

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
