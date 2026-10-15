"""Store each weight once in a whole-model image: device-layout weights live in the weights blob.

A routed group's device call reads its stored weight in the layout the device program holds it
(:func:`merlin.xdsl_dialects.lowering.group_prepack.prepack`). By default the rewrite hands that copy
over as an ``arith.constant`` (:mod:`merlin.llvmlower.group_host_call`), which lands in the model
object's ``.rodata`` -- while the weights blob keeps the capture's original bytes, whose only reader
was the relayout the rewrite just removed. Every routed weight is then in the image twice (TinyLlama:
1.03 GB of ``.rodata`` beside a 1.30 GB blob).

With :class:`WeightArgs` the rewrite binds each laid-out weight to a TRAILING ``@forward`` argument
instead -- the channel ``qinner`` and ``quant_hoist`` already use, so the C runtime appends the bytes to
the blob and adds the argument rows -- and records which original arguments lost their last reader.
:func:`compact` then drops those arguments' bytes from the blob (their descriptors point at the blob's
base and are never read). Numerics are untouched: the device reads the same bytes from another address.

Written beside the prepared module (``prepared_dir``), read by :func:`merlin.llvmlower.c_runtime.generate`:

- ``blob_weights.json`` -- ``{"args": [{key, shape, dtype}...], "dead_args": [original arg index...]}``
- ``blob_weights.npz``  -- the laid-out bytes, keyed by ``key``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PLAN_FILE = "blob_weights.json"
VALUES_FILE = "blob_weights.npz"
ALIGNMENT = 64

#: Whether the enclosing whole-model build asked for blob weights (:func:`requested`). The rewrite
#: runs several calls below the build that decides, inside the shared preparation; a scoped flag lets
#: the image builder choose without threading an option through every preparation entry point.
_REQUESTED: ContextVar[bool] = ContextVar("merlin_blob_weights", default=False)


@contextmanager
def requested(enabled: bool) -> Iterator[None]:
    """Within this scope a device group rewrite binds laid-out weights as blob arguments."""
    if type(enabled) is not bool:
        raise TypeError("blob weights must be requested with an explicit Boolean")
    token = _REQUESTED.set(enabled)
    try:
        yield
    finally:
        _REQUESTED.reset(token)


def is_requested() -> bool:
    return _REQUESTED.get()


_NUMPY = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64, "f32": np.float32, "f64": np.float64}


def _forward(module):
    from .quant_hoist import _forward_func  # noqa: PLC2701 -- one definition of the trailing-argument owner

    return _forward_func(module)


def _used(block) -> set[int]:
    return {index for index, arg in enumerate(block.args) if any(True for _ in arg.uses)}


@dataclass
class WeightArgs:
    """Binds laid-out device weights to trailing ``@forward`` arguments of ``module`` (one per distinct
    tensor); :meth:`finish` seals the function type and the plan."""

    module: Any
    args: list[dict[str, Any]] = field(default_factory=list)
    values: dict[str, np.ndarray] = field(default_factory=dict)
    dead_args: tuple[int, ...] = ()
    _by_content: dict[tuple, Any] = field(default_factory=dict)
    _live_before: set[int] = field(default_factory=set)
    _original: int = 0

    def __post_init__(self) -> None:
        block = _forward(self.module).body.blocks[0]
        self._original = len(block.args)
        self._live_before = _used(block)

    def __call__(self, array: np.ndarray, element_type):
        """The ``@forward`` argument holding ``array`` as a tensor of ``element_type``."""
        from xdsl.dialects.builtin import TensorType

        from merlin.runtime.linalg_numpy import _elem  # noqa: PLC2701 -- the shared element spelling

        dtype = _elem(element_type)
        if dtype not in _NUMPY:
            raise ValueError(f"a blob weight of element type {dtype} has no storage form")
        data = np.ascontiguousarray(np.asarray(array).astype(_NUMPY[dtype]))
        if not np.array_equal(data, np.asarray(array)):
            raise ValueError(f"a laid-out weight does not fit its element type {dtype}")
        content = (dtype, data.shape, hashlib.sha256(data.tobytes()).hexdigest())
        if content in self._by_content:
            return self._by_content[content]
        block = _forward(self.module).body.blocks[0]
        argument = block.insert_arg(TensorType(element_type, list(data.shape)), len(block.args))
        key = f"w{len(self.args)}"
        self.args.append({"key": key, "shape": [int(d) for d in data.shape], "dtype": dtype})
        self.values[key] = data
        self._by_content[content] = argument
        return argument

    def finish(self) -> None:
        """Seal ``@forward``'s function type; record the original arguments nothing reads any more."""
        from xdsl.dialects.builtin import FunctionType

        func = _forward(self.module)
        block = func.body.blocks[0]
        if self.args:
            func.properties["function_type"] = FunctionType.from_lists(
                [a.type for a in block.args], list(func.function_type.outputs.data)
            )
        live = _used(block)
        self.dead_args = tuple(sorted(i for i in self._live_before - live if i < self._original))

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        (directory / PLAN_FILE).write_text(json.dumps({"args": self.args, "dead_args": list(self.dead_args)}) + "\n")
        np.savez(directory / VALUES_FILE, **self.values)


@dataclass(frozen=True)
class Plan:
    """A prepared module's blob weights: the trailing argument rows and the dropped original arguments."""

    args: tuple[dict[str, Any], ...] = ()
    dead_args: frozenset[int] = frozenset()
    directory: Path | None = None

    def arrays(self):
        """``(arg, bytes)`` per trailing argument, in argument order."""
        if not self.args:
            return
        with np.load(Path(self.directory) / VALUES_FILE) as values:
            for arg in self.args:
                data = np.ascontiguousarray(values[arg["key"]])
                if list(data.shape) != list(arg["shape"]) or data.dtype != np.dtype(_NUMPY[arg["dtype"]]):
                    raise ValueError(f"blob weight {arg['key']} bytes disagree with its plan")
                yield arg, data.tobytes()


def read(directory: str | Path) -> Plan:
    """The plan beside a prepared module, or an empty one. Plan without values is an error."""
    directory = Path(directory)
    path = directory / PLAN_FILE
    if not path.is_file():
        return Plan()
    doc = json.loads(path.read_text())
    if doc.get("args") and not (directory / VALUES_FILE).is_file():
        raise FileNotFoundError(f"{path} names blob weights but {VALUES_FILE} is absent")
    return Plan(tuple(doc.get("args") or ()), frozenset(int(i) for i in doc.get("dead_args") or ()), directory)


def compact(header: dict, payload: bytes, manifest: dict, dead_args: frozenset[int]) -> tuple[dict, bytes]:
    """``(header, payload)`` without the tensors only ``dead_args`` read.

    A tensor stays when any live argument names it (a tied weight read by host code keeps its bytes).
    Kept tensors are re-placed ``ALIGNMENT``-aligned in their original order; a dropped tensor's offsets
    become ``(0, 0)``, so its never-read descriptor points at the blob's base."""
    readers: dict[str, set[bool]] = {}
    for key, meta in manifest.items():
        if key.isdigit() and isinstance(meta, dict) and meta.get("weight") in header:
            readers.setdefault(meta["weight"], set()).add(int(key) in dead_args)
    dropped = {name for name, dead in readers.items() if dead == {True}}
    out, blob = {}, bytearray()
    for name, meta in sorted(
        ((n, m) for n, m in header.items() if isinstance(m, dict) and "data_offsets" in m),
        key=lambda item: item[1]["data_offsets"][0],
    ):
        if name in dropped:
            out[name] = {**meta, "data_offsets": [0, 0]}
            continue
        begin, end = meta["data_offsets"]
        blob.extend(b"\0" * ((-len(blob)) % ALIGNMENT))
        out[name] = {**meta, "data_offsets": [len(blob), len(blob) + end - begin]}
        blob.extend(payload[begin:end])
    for name, meta in header.items():
        out.setdefault(name, meta)
    return out, bytes(blob)
