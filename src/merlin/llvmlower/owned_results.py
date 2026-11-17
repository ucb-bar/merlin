"""Execute a model once and copy its owned, execution-sized tensor results.

This explicit host transport does not predict result dimensions. It retains
upstream ownership checks and validates descriptors against invocation-local
allocator records before reading payloads. Original tensor/scalar types and
the default out-parameter transport remain unchanged.
"""

import ctypes
import hashlib
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .abi import PrivateHostImagePolicy, ScalarArg, _load_image, descriptor, make_descriptor
from .owned_result_abi import PROFILE, ScalarSlot, TensorSlot

_NUMPY = {
    "i1": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "f16": np.float16,
    "bf16": np.uint16,
    "f32": np.float32,
    "f64": np.float64,
    "f8E4M3FN": np.uint8,
    "f8E4M3FNUZ": np.uint8,
    "f8E5M2": np.uint8,
    "f8E5M2FNUZ": np.uint8,
}


def _pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        return str(path), hashlib.file_digest(stream, "sha256").hexdigest()


def _function(library, name, arguments, result=None):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, result
    return function


def _limits(max_bytes, max_allocations, max_output_bytes):
    if any(
        type(value) is not int or not 0 < value <= (1 << 63) - 1
        for value in (max_bytes, max_allocations, max_output_bytes)
    ):
        raise ValueError("owned result budgets must be positive bounded native sizes")


@dataclass(frozen=True)
class OwnedResultProgram:
    """One source/build selection; invocation values never enter compilation."""

    source: Path
    library: Path
    inputs: tuple[TensorSlot | ScalarSlot, ...]
    outputs: tuple[TensorSlot, ...]
    pins: tuple[tuple[str, str], ...]
    max_bytes: int
    max_allocations: int
    max_output_bytes: int
    entry: str

    @cached_property
    def _image(self):
        return _load_image(str(self.library), ctypes.RTLD_LOCAL, PrivateHostImagePolicy(self.library.parent))

    def verify(self):
        _limits(self.max_bytes, self.max_allocations, self.max_output_bytes)
        if any(_pin(path) != (path, digest) for path, digest in self.pins):
            raise ValueError("owned result source/build/runtime bytes changed")
        from .owned_result_abi import prepare

        _, inputs, outputs = prepare(self.source.read_text(), entry=self.entry)
        if inputs != self.inputs or outputs != self.outputs:
            raise ValueError("owned result original signature changed")

    def execute(self, arguments):
        """Return independent arrays and allocation observations after one call.

        Source assertions and allocation failures abort the native process, as
        in the ordinary host ABI. Run untrusted invocations in a bounded worker.
        """
        self.verify()
        if len(arguments) != len(self.inputs):
            raise ValueError("owned result invocation changes the complete input roster")
        carriers, buffers = [], []
        for argument, slot in zip(arguments, self.inputs, strict=True):
            if isinstance(slot, ScalarSlot):
                if type(argument) is not ScalarArg or argument.dtype != slot.dtype:
                    raise ValueError("owned result scalar needs its exact native by-value type")
                carriers.append(argument.to_ctype())
                continue
            if (
                type(argument) is not np.ndarray
                or argument.dtype != np.dtype(_NUMPY[slot.dtype])
                or not argument.flags.c_contiguous
                or argument.ndim != len(slot.shape)
                or any(want >= 0 and actual != want for actual, want in zip(argument.shape, slot.shape, strict=True))
            ):
                raise ValueError("owned result input storage/rank/static dimensions differ from source")
            buffers.append(argument)
            carriers.append(descriptor(argument.ctypes.data, argument.shape))
        if sum(array.nbytes for array in buffers) > self.max_bytes:
            raise ValueError("owned result inputs exceed the invocation byte budget")
        library, digest = self._image
        if _pin(self.library)[1] != digest or (str(self.library), digest) not in self.pins:
            raise ValueError("owned result loaded image differs from its compiled bytes")
        size = ctypes.c_size_t
        layout = _function(library, "merlin_owned_result_descriptor_bytes", [size], size)
        for slot in (*self.inputs, *self.outputs):
            if isinstance(slot, TensorSlot) and layout(len(slot.shape)) != ctypes.sizeof(
                make_descriptor(len(slot.shape))
            ):
                raise ValueError("owned result descriptor differs from the selected C layout")
        begin = _function(library, "merlin_owned_results_begin", [size, size], ctypes.c_int)
        end = _function(library, "merlin_owned_results_end", [])
        capacity = _function(library, "merlin_owned_results_capacity", [ctypes.c_void_p], size)
        live = _function(library, "merlin_owned_results_live_count", [], size)
        peak = _function(library, "merlin_owned_results_peak_bytes", [], size)
        total = _function(library, "merlin_owned_results_total_count", [], size)
        pointers = ctypes.POINTER(ctypes.c_void_p)
        invoke = _function(library, "merlin_owned_result_invoke", [pointers, pointers])
        results = [make_descriptor(len(slot.shape))() for slot in self.outputs]
        args = (ctypes.c_void_p * len(carriers))(*(ctypes.addressof(value) for value in carriers))
        outs = (ctypes.c_void_p * len(results))(*(ctypes.addressof(value) for value in results))
        self.verify()
        if not begin(self.max_bytes, self.max_allocations):
            raise ValueError("owned result invocation is already active")
        try:
            invoke(args, outs)
            views, owners, logical_bytes = [], set(), 0
            # Validate every descriptor before reading any result payload.
            for result, slot in zip(results, self.outputs, strict=True):
                view, owner, count = _view(result, slot, capacity)
                logical_bytes += count
                if logical_bytes > self.max_output_bytes:
                    raise ValueError("executed result sizes exceed the output byte budget")
                views.append(view)
                owners.add(owner)
            if live() != len(owners):
                raise ValueError("native execution retained allocations outside its returned owners")
            observations = {
                "profile": PROFILE,
                "shapes": [list(v.shape) for v in views],
                "owned_allocations": len(owners),
                "peak_allocation_bytes": peak(),
                "total_allocations": total(),
                "source_calls": 1,
            }
            outputs = tuple(view.copy() for view in views)
            self.verify()
            return outputs, observations
        finally:
            end()


def _view(result, slot, capacity):
    """Validate ownership and the complete signed-stride footprint before reads."""
    shape, strides = tuple(result.sizes), tuple(result.strides)
    if any(size < 0 or want >= 0 and size != want for size, want in zip(shape, slot.shape, strict=True)):
        raise ValueError("executed result has a negative or changed static extent")
    dtype = np.dtype(_NUMPY[slot.dtype])
    base, aligned = result.allocated or 0, result.aligned or 0
    accessible = capacity(base)
    if not base or not accessible:
        raise ValueError("executed result has no live invocation-owned allocation")
    empty = any(size == 0 for size in shape)
    if empty:
        # Empty views perform no access, regardless of inactive offsets/strides.
        return np.empty(shape, dtype=dtype), base, 0
    low = result.offset + sum(min(0, (size - 1) * stride) for size, stride in zip(shape, strides, strict=True))
    high = result.offset + sum(max(0, (size - 1) * stride) for size, stride in zip(shape, strides, strict=True))
    start, stop = aligned + low * dtype.itemsize, aligned + (high + 1) * dtype.itemsize
    logical_start = aligned + result.offset * dtype.itemsize
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    if (
        not aligned
        or aligned % dtype.alignment
        or base + accessible > address_limit
        or not base <= logical_start <= base + accessible
        or not empty
        and not base <= start < stop <= base + accessible
    ):
        raise ValueError("executed result footprint escapes its actual owned allocation")
    storage = (ctypes.c_ubyte * accessible).from_address(base)
    view = np.ndarray(
        shape,
        dtype=dtype,
        buffer=storage,
        offset=logical_start - base,
        strides=tuple(stride * dtype.itemsize for stride in strides),
    )
    return view, base, math.prod(shape) * dtype.itemsize


def compile_owned_results(
    source,
    *,
    workdir,
    entry="forward",
    features=frozenset(),
    timeout=60,
    max_bytes=64 << 20,
    max_allocations=4096,
    max_output_bytes=64 << 20,
    allow_existing_workdir=False,
):
    """Compile actual execution-sized results through normal Merlin ownership."""
    from merlin.common.paths import runtime_dir

    from .codegen import build_host_shared
    from .owned_result_lowering import lower_owned_results
    from .target_data_layout import of as data_layout_of
    from .toolchain import clang

    _limits(max_bytes, max_allocations, max_output_bytes)
    # This transport explicitly selects native 64-bit indices. The bridge's
    # C static assertions verify pointer/intptr width with the selected compiler;
    # valid LLVM layouts may omit their default pointer-width entry entirely.
    layout = data_layout_of(clang(), ("-O2", "-fPIC"))
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ValueError("owned results require the selected native 64-bit host ABI")
    lowered = lower_owned_results(
        source, workdir=workdir, data_layout=layout, index_bits=64, entry=entry,
        features=features, timeout=timeout, max_bytes=max_bytes,
        allow_existing_workdir=allow_existing_workdir,
    )
    work = lowered.source.parent
    library = build_host_shared(lowered.llvm, work / "model.so", owned_result_bridge=lowered.bridge).resolve()
    lowered.verify()
    paths = (
        library,
        Path(__file__),
        runtime_dir() / "abi/owned_results.c",
        runtime_dir() / "abi/owned_result_views.c",
        runtime_dir() / "abi/owned_result_views.h",
    )
    program = OwnedResultProgram(
        lowered.source,
        library,
        lowered.inputs,
        lowered.outputs,
        (*lowered.pins, *(_pin(p) for p in paths)),
        max_bytes,
        max_allocations,
        max_output_bytes,
        entry,
    )
    program.verify()
    return program
