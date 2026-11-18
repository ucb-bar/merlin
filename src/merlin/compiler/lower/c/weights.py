"""Pack safetensors weights into a flat blob + arg-table.

The model's ``@forward`` keeps weights as leading function arguments (manifest kinds
``param``/``buffer``); the runtime passes pointers into one contiguous blob —
``offset(arg) = data_offset within the safetensors payload``. The same table feeds
the host ctypes runner and the generated C table for spike/Zephyr.

Layout: safetensors = 8-byte LE header length + JSON header + payload. The blob IS
the payload (single mmap/copy), so weights are not embedded into C arrays.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from merlin.capture.safetensors import load_header as load_safetensors_header
from merlin.capture.safetensors import read_header

# ---- weights_pack ------------------------------------------------------------------------------

DTYPE_BYTES = {"I8": 1, "U8": 1, "I32": 4, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def weight_files_beside(capture):
    """Select a manifest and its matching payload, never two independent globs.

    A model bundle owns the canonical ``weights.safetensors`` pair. Retained
    source sidecars can legitimately have different bytes after preparation.
    Other capture directories need one unambiguous basename-matched pair.
    """
    capture = Path(capture)
    beside = capture if capture.is_dir() else capture.parent
    canonical = beside / "weights.safetensors.manifest.json", beside / "weights.safetensors"
    if (beside / "model.mlir").is_file() and any(path.exists() for path in canonical):
        if not all(path.is_file() and not path.is_symlink() for path in canonical):
            raise ValueError("model bundle has an incomplete or indirect canonical weight pair")
        return canonical
    pairs = []
    manifests = sorted(beside.glob("*.manifest.json"))
    for manifest in manifests:
        stem = manifest.name.removesuffix(".manifest.json")
        weights = beside / (stem if stem.endswith(".safetensors") else stem + ".safetensors")
        if weights.is_file():
            if manifest.is_symlink() or weights.is_symlink():
                raise ValueError("capture weight pairs require direct regular files")
            pairs.append((manifest, weights))
    if not pairs and len(manifests) == 1:
        # Historical single-pair captures did not require a common basename.
        payloads = list(beside.glob("*.safetensors"))
        if len(payloads) == 1:
            if manifests[0].is_symlink() or payloads[0].is_symlink():
                raise ValueError("capture weight pairs require direct regular files")
            pairs.append((manifests[0], payloads[0]))
    if len(pairs) > 1:
        raise ValueError("capture has ambiguous weight pairs; select a canonical model bundle")
    return pairs[0] if pairs else (None, None)


@dataclass
class WeightEntry:
    arg_index: int
    name: str
    dtype: str
    shape: list[int]
    offset: int  # offset into the blob (safetensors payload)
    nbytes: int


def pack(
    manifest_path: str | Path, safetensors_path: str | Path, out_blob: str | Path | None = None
) -> list[WeightEntry]:
    """Build the arg table; optionally write the payload blob to ``out_blob``."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    header, payload_off = load_safetensors_header(safetensors_path)

    entries: list[WeightEntry] = []
    for arg_idx_s, meta in manifest.items():
        if meta.get("kind") not in ("param", "buffer"):
            continue
        # Params carry "weight" (state-dict key). Non-persistent buffers (e.g. rotary
        # inv_freq) are not serialized; they carry only "name" — the runtime
        # materializes them separately (see missing_buffers()).
        name = meta.get("weight") or meta["name"]
        if name not in header:
            continue
        h = header[name]
        begin, end = h["data_offsets"]
        entries.append(
            WeightEntry(
                arg_index=int(arg_idx_s),
                name=name,
                dtype=h["dtype"],
                shape=list(h["shape"]),
                offset=begin,
                nbytes=end - begin,
            )
        )
    entries.sort(key=lambda e: e.arg_index)

    if out_blob is not None:
        data = Path(safetensors_path).read_bytes()[payload_off:]
        Path(out_blob).write_bytes(data)
    return entries


def missing_buffers(manifest_path: str | Path, safetensors_path: str | Path) -> list[tuple[int, str]]:
    """(arg_index, name) of param/buffer args absent from the safetensors payload.

    These are non-persistent buffers (rotary inv_freq etc.); the runtime supplies
    them from the reference capture or computes them.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    header, _ = load_safetensors_header(safetensors_path)
    out = []
    for arg_idx_s, meta in manifest.items():
        if meta.get("kind") not in ("param", "buffer"):
            continue
        name = meta.get("weight") or meta["name"]
        if name not in header:
            out.append((int(arg_idx_s), name))
    return sorted(out)


def emit_c_table(entries: list[WeightEntry]) -> str:
    """C table mapping forward() weight args to blob offsets (for model_main.c)."""
    lines = [
        "/* Generated by merlin.compiler.lower.c.weights - do not edit. */",
        "#include <stddef.h>",
        "typedef struct { int arg_index; size_t offset; size_t nbytes; } merlin_weight_t;",
        f"#define MERLIN_WEIGHT_COUNT {len(entries)}",
        "static const merlin_weight_t MERLIN_WEIGHTS[MERLIN_WEIGHT_COUNT] = {",
    ]
    lines += [f"  {{{e.arg_index}, {e.offset}, {e.nbytes}}}," for e in entries]
    lines.append("};")
    return "\n".join(lines) + "\n"


def checked_payload(safetensors_path, *, archive_sha256, out_blob=None):
    """Hash the source archive and its payload; optionally publish a fresh blob.

    Metadata and bytes come from the same read stream. Retain bounded header
    and chunk storage, and never replace an existing output. This validates
    byte identity, not tensor numerical domains or target packing.
    """
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(char not in "0123456789abcdef" for char in archive_sha256)
    ):
        raise ValueError("weight archive needs a canonical SHA256 identity")
    target = Path(out_blob) if out_blob is not None else None
    created = False
    try:
        with Path(safetensors_path).open("rb") as source:
            header, offset = read_header(source)
            source.seek(0)
            prefix = source.read(offset)
            if read_header(BytesIO(prefix)) != (header, offset):
                raise ValueError("weight metadata changed during payload intake")
            complete, payload, count = hashlib.sha256(prefix), hashlib.sha256(), 0
            with target.open("xb") if target is not None else nullcontext() as destination:
                created = target is not None
                while chunk := source.read(1024 * 1024):
                    complete.update(chunk)
                    payload.update(chunk)
                    count += len(chunk)
                    if destination is not None:
                        destination.write(chunk)
                if complete.hexdigest() != archive_sha256:
                    raise ValueError("weight archive changed after compiler input validation")
        return {
            "archive_sha256": archive_sha256,
            "payload_sha256": payload.hexdigest(),
            "payload_offset": offset,
            "payload_bytes": count,
        }
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


# ---- weight_payload ----------------------------------------------------------------------------
# Lazy byte spans for the existing generated weight-blob layout.
#
# This is build-time host storage, not a device memory or scheduling contract.
# Readers borrow source files/arrays until consumption; callers still own content
# freshness and immutable invocation inputs. No content identity is memoized.

# Compiler I/O work budget, not hardware geometry or a transfer-cost estimate.
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class _Span:
    size: int
    read: Callable[[int, int, int], Iterator[bytes | memoryview]]
    begin: int = 0


class WeightPayload:
    """Ordered file/array/zero spans; slicing and alignment never copy payloads."""

    def __init__(self):
        self._spans: list[_Span] = []
        self._size = 0

    def __len__(self):
        return self._size

    def __bytes__(self):
        """Explicit legacy materialization; ordinary generation consumes chunks."""
        return b"".join(self.chunks())

    def extend(self, other: WeightPayload):
        self._spans.extend(other._spans)
        self._size += len(other)

    def _add(self, size, reader):
        if type(size) is not int or size < 0:
            raise ValueError("weight byte span needs a nonnegative integer size")
        self._spans.append(_Span(size, reader))
        self._size += size
        return self

    @classmethod
    def zeros(cls, size: int):
        def read(begin, count, chunk):
            block = bytes(min(count, chunk))
            while count:
                length = min(count, len(block))
                yield block[:length]
                count -= length

        return cls()._add(size, read)

    @classmethod
    def file(cls, path: Path, begin: int, size: int):
        path = Path(path).resolve(strict=True)
        selected = path.stat()
        if type(begin) is not int or begin < 0 or type(size) is not int or size < 0 or begin + size > selected.st_size:
            raise ValueError("weight file span exceeds source bytes")

        def identity(stat):
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        def block(stream, index):
            stream.seek(begin + index * CHUNK_BYTES)
            expected = min(CHUNK_BYTES, size - index * CHUNK_BYTES)
            data = stream.read(expected)
            if len(data) != expected:
                raise ValueError("weight file truncated during consumption")
            return data

        # Timestamps are only an additional drift signal. Store bounded block
        # digests so restored/coalesced filesystem times never establish identity.
        with path.open("rb", buffering=0) as stream:
            digests = tuple(
                hashlib.sha256(block(stream, i)).digest() for i in range((size + CHUNK_BYTES - 1) // CHUNK_BYTES)
            )

        def read(offset, count, chunk):
            import os

            original_offset, original_count = offset, count
            with path.open("rb") as stream:
                if identity(os.fstat(stream.fileno())) != identity(selected):
                    raise ValueError("weight file changed before consumption")
                while count:
                    index, within = divmod(offset, CHUNK_BYTES)
                    data = block(stream, index)
                    if hashlib.sha256(data).digest() != digests[index]:
                        raise ValueError("weight file bytes changed before consumption")
                    length = min(count, len(data) - within)
                    view = memoryview(data)
                    for start in range(within, within + length, chunk):
                        yield view[start : min(start + chunk, within + length)]
                    count -= length
                    offset += length
                if identity(os.fstat(stream.fileno())) != identity(selected) or identity(path.stat()) != identity(
                    selected
                ):
                    raise ValueError("weight file changed during consumption")
            # A yielded chunk may outlive a caller's write. Reopen rather than
            # reusing buffered bytes when checking the consumed source blocks.
            with path.open("rb", buffering=0) as stream:
                first = original_offset // CHUNK_BYTES
                last = (original_offset + original_count + CHUNK_BYTES - 1) // CHUNK_BYTES
                for index in range(first, last):
                    if hashlib.sha256(block(stream, index)).digest() != digests[index]:
                        raise ValueError("weight file bytes changed during consumption")

        return cls()._add(size, read)

    @classmethod
    def array(cls, size: int, load: Callable[[], np.ndarray]):
        """Load and validate one caller-owned array only while its span is read."""

        def read(begin, count, chunk):
            array = np.ascontiguousarray(load())
            if array.nbytes != size:
                raise ValueError("weight array disagrees with its declared byte span")
            if not count:
                return
            view = memoryview(array).cast("B")
            for offset in range(begin, begin + count, chunk):
                yield view[offset : min(offset + chunk, begin + count)]

        return cls()._add(size, read)

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("weight byte selection requires a contiguous slice")
        begin = 0 if selection.start is None else selection.start
        end = len(self) if selection.stop is None else selection.stop
        if type(begin) is not int or type(end) is not int or begin < 0 or end < begin or end > len(self):
            raise ValueError("weight byte slice exceeds payload")
        result, cursor = WeightPayload(), 0
        for span in self._spans:
            left, right = max(begin, cursor), min(end, cursor + span.size)
            if left < right:
                result._spans.append(_Span(right - left, span.read, span.begin + left - cursor))
                result._size += right - left
            cursor += span.size
        return result

    def chunks(self, chunk_bytes: int = CHUNK_BYTES):
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ValueError("weight I/O chunk budget must be a positive integer")
        for span in self._spans:
            count = 0
            for data in span.read(span.begin, span.size, chunk_bytes):
                if not isinstance(data, (bytes, memoryview)) or len(data) > chunk_bytes or not len(data):
                    raise ValueError("weight source returned an invalid bounded chunk")
                count += len(data)
                if count > span.size:
                    raise ValueError("weight source exceeded its selected span")
                yield data
            if count != span.size:
                raise ValueError("weight source did not cover its complete span")


# ---- blob_weights ------------------------------------------------------------------------------
# Store each weight once in a whole-model image: device-layout weights live in the weights blob.
#
# A routed group's device call reads its stored weight in the layout the device program holds it
# (:func:`merlin.compiler.partition.groups.weights.prepack`). By default the rewrite hands that copy
# over as an ``arith.constant`` (:mod:`merlin.compiler.lower.device.group_host_call`), which lands in the model
# object's ``.rodata`` -- while the weights blob keeps the capture's original bytes, whose only reader
# was the relayout the rewrite just removed. Every routed weight is then in the image twice (TinyLlama:
# 1.03 GB of ``.rodata`` beside a 1.30 GB blob).
#
# With :class:`WeightArgs` the rewrite binds each laid-out weight to a TRAILING ``@forward`` argument
# instead -- the channel ``qinner`` and ``quant_hoist`` already use, so the C runtime appends the bytes to
# the blob and adds the argument rows -- and records which original arguments lost their last reader.
# :func:`compact` then drops those arguments' bytes from the blob (their descriptors point at the blob's
# base and are never read). Numerics are untouched: the device reads the same bytes from another address.
#
# Written beside the prepared module (``prepared_dir``), read by :func:`merlin.compiler.lower.c.runtime.generate`:
#
# - ``blob_weights.json`` -- ``{"args": [{key, shape, dtype}...], "dead_args": [original arg index...]}``
# - ``blob_weights.npz``  -- the laid-out bytes, keyed by ``key``

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
    from merlin.compiler.lower.options.quant_hoist import _forward_func

    return _forward_func(module)


#: Value-semantic dialects: an op of these with results computes nothing but them, so it matters only
#: when something live reads a result. Every other op (a call, a return, anything on memory) is a root.
_VALUE_DIALECTS = ("arith.", "math.", "tensor.", "linalg.", "quant_ext.")


def _used(block) -> set[int]:
    """Indices of ``block``'s arguments some LIVE op reads (transitively from a root, nested regions too).

    An argument read only by a dead value chain -- a weight's dequantize-and-requantize that fed the
    group the rewrite replaced, left for canonicalization to erase -- is not used."""
    from xdsl.ir import BlockArgument, OpResult

    from merlin.compiler.ir.mlir_query import op_name  # an unregistered op (quant_ext.*) by its real name

    def top(op):
        while op.parent_block() is not block:
            op = op.parent_op()
        return op

    pending = [op for op in block.ops if not op.results or not op_name(op).startswith(_VALUE_DIALECTS)]
    live: set[int] = {id(op) for op in pending}
    args: set[int] = set()
    index = {id(arg): i for i, arg in enumerate(block.args)}
    while pending:
        op = pending.pop()
        for inner in op.walk():
            for value in inner.operands:
                if isinstance(value, BlockArgument) and id(value) in index:
                    args.add(index[id(value)])
                elif isinstance(value, OpResult) and _inside(value.op, block):
                    owner = top(value.op)
                    if id(owner) not in live:
                        live.add(id(owner))
                        pending.append(owner)
    return args


def _inside(op, block) -> bool:
    parent = op.parent_block()
    while parent is not None:
        if parent is block:
            return True
        holder = parent.parent_op()
        parent = holder.parent_block() if holder is not None else None
    return False


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

        from merlin.evaluation.model.linalg_semantics import _elem  # noqa: PLC2701 -- the shared element spelling

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

    @property
    def nbytes(self) -> int:
        """Bytes the trailing arguments add to the blob."""
        return sum(int(np.prod(a["shape"])) * np.dtype(_NUMPY[a["dtype"]]).itemsize for a in self.args)

    def arrays(self):
        """``(arg, bytes)`` per trailing argument, in argument order."""
        if not self.args:
            return
        with np.load(Path(self.directory) / VALUES_FILE) as values:
            for arg in self.args:
                yield arg, self.array(arg, values).tobytes()

    def array(self, arg, values):
        """Read one declared trailing argument without collecting the other weights."""
        if arg not in self.args:
            raise ValueError("blob array is not a selected prepared argument")
        data = np.ascontiguousarray(values[arg["key"]])
        if list(data.shape) != list(arg["shape"]) or data.dtype != np.dtype(_NUMPY[arg["dtype"]]):
            raise ValueError(f"blob weight {arg['key']} bytes disagree with its plan")
        return data


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


def compact(
    header: dict, payload: bytes, manifest: dict, dead_args: frozenset[int], *, appended: int = 0
) -> tuple[dict, bytes]:
    """``(header, payload)`` without the tensors only ``dead_args`` read.

    A tensor stays when any live argument names it (a tied weight read by host code keeps its bytes).
    Kept tensors are re-placed ``ALIGNMENT``-aligned in their original order; a dropped tensor's offsets
    become ``(0, 0)``, so its never-read descriptor points at the blob's base. A tensor larger than the
    final blob (what remains plus the ``appended`` bytes generation adds after it) is kept, so even a dead
    read that survived into the object stays inside the image."""
    readers: dict[str, set[bool]] = {}
    for key, meta in manifest.items():
        if key.isdigit() and isinstance(meta, dict) and meta.get("weight") in header:
            readers.setdefault(meta["weight"], set()).add(int(key) in dead_args)
    dropped = {name for name, dead in readers.items() if dead == {True}}

    def size(name: str) -> int:
        begin, end = header[name]["data_offsets"]
        return end - begin

    remaining = sum(size(n) for n, m in header.items() if isinstance(m, dict) and "data_offsets" in m) - sum(
        size(n) for n in dropped
    )
    dropped = {name for name in dropped if size(name) <= remaining + appended}

    streamed = isinstance(payload, WeightPayload)
    out, blob = {}, WeightPayload() if streamed else bytearray()
    for name, meta in sorted(
        ((n, m) for n, m in header.items() if isinstance(m, dict) and "data_offsets" in m),
        key=lambda item: item[1]["data_offsets"][0],
    ):
        if name in dropped:
            out[name] = {**meta, "data_offsets": [0, 0]}
            continue
        begin, end = meta["data_offsets"]
        padding = (-len(blob)) % ALIGNMENT
        blob.extend(WeightPayload.zeros(padding) if streamed else bytes(padding))
        out[name] = {**meta, "data_offsets": [len(blob), len(blob) + end - begin]}
        blob.extend(payload[begin:end])
    for name, meta in header.items():
        out.setdefault(name, meta)
    return out, blob if streamed else bytes(blob)
