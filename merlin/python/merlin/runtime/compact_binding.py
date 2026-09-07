"""Format-only caller preparation for an explicitly compacted pointer entry.

The command buffer retains its original whole-program tensor ABI. The separate
compiler receipt changes only the called symbol and pointer bindings. Target
alignment/index facts are supplied by the trusted caller, not inferred from the
Python host. No arithmetic, arena reuse, target allocation or invocation occurs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from math import prod
from typing import Any

from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.perf.structural_transitions import _element_bytes
from merlin.runtime.storage_binding import StorageBinding, resolve_storage_bindings
from merlin.runtime.tensor import Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    _require(type(value) is int and value >= int(positive), f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CompactTargetABI:
    """Out-of-band facts, in compact-base and original-argument order.

    Access groups are deliberately homogeneous: read, write or readwrite. A
    caller needing shared/mixed-access or aliased bases needs another protocol.
    Provenance identifies the target data-layout/alignment contract, not timing.
    """
    pointer_index_bits: tuple[int, ...]
    base_alignments: tuple[int, ...]
    argument_alignments: tuple[int, ...]
    base_access: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class _Argument:
    tensor: str
    base_index: int
    byte_offset: int
    byte_extent: int
    storage: StorageBinding


class CompactBinding:
    """Prepared byte buffers; transfer/allocation and entry invocation remain caller-owned."""

    def __init__(self, *, arenas: list[bytearray], arguments: tuple[_Argument, ...],
                 outputs: tuple[str, ...], target_abi: CompactTargetABI, evidence: dict):
        self._arenas = arenas
        self._arguments = arguments
        self._outputs = outputs
        self._target_abi = target_abi
        self._evidence = evidence
        self._readonly_snapshots = tuple(bytes(arena) if access == "read" else None
            for arena, access in zip(arenas, target_abi.base_access, strict=True))

    def base_views(self) -> tuple[memoryview, ...]:
        """Compact argument order. Read-only bases do not expose writable views."""
        return tuple(memoryview(arena).toreadonly() if access == "read" else memoryview(arena)
            for arena, access in zip(self._arenas, self._target_abi.base_access, strict=True))

    def original_argument_views(self) -> tuple[memoryview, ...]:
        bases = self.base_views()
        return tuple(bases[arg.base_index][arg.byte_offset:arg.byte_offset+arg.byte_extent]
                     for arg in self._arguments)

    def validate_runtime_addresses(self, base_addresses: Sequence[int]) -> tuple[int, ...]:
        """Check caller-allocated base addresses; return original argument addresses.

        The caller must separately ensure these are live allocations in the
        entry's address space. Pointer-index widths constrain offsets/extents,
        not the numerical value of a pointer in a potentially different host.
        """
        _require(len(base_addresses) == len(self._arenas), "runtime base address count differs from compact ABI")
        spans = []
        for index, (address, arena, alignment) in enumerate(zip(base_addresses, self._arenas,
                self._target_abi.base_alignments, strict=True)):
            _integer(address, "runtime base address", positive=True)
            _require(address % alignment == 0, "runtime base address violates target alignment")
            spans.append((address, address+len(arena), index))
        spans.sort()
        _require(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])),
                 "runtime base allocations overlap; aliasing/reuse unsupported")
        result = tuple(base_addresses[arg.base_index]+arg.byte_offset for arg in self._arguments)
        _require(all(address % alignment == 0 for address, alignment in
                     zip(result, self._target_abi.argument_alignments, strict=True)),
                 "runtime original argument address violates target alignment")
        return result

    def readback(self, base_payloads: Sequence[bytes] | None = None) -> dict[str, bytes]:
        """Return output logical storage bytes, without decoding any scalar value.

        Explicit returned base payloads support a separate device allocation.
        Immutable/read-only bases are checked even when no output uses them.
        """
        payloads = tuple(bytes(arena) for arena in self._arenas) if base_payloads is None else tuple(base_payloads)
        _require(len(payloads) == len(self._arenas), "readback base count differs from compact ABI")
        for payload, arena, original in zip(payloads, self._arenas, self._readonly_snapshots, strict=True):
            _require(type(payload) is bytes and len(payload) == len(arena), "readback requires exact base byte extents")
            _require(original is None or payload == original, "read-only compact base was modified")
        result = {}
        for arg in self._arguments:
            if arg.tensor not in self._outputs:
                continue
            encoding = arg.storage.encoding
            width = _element_bytes(encoding.dtype)
            source = memoryview(payloads[arg.base_index])[arg.byte_offset:arg.byte_offset+arg.byte_extent]
            logical = bytearray()
            for index in product(*(range(dim) for dim in encoding.logical_shape)):
                offset = (encoding.offset_elements+sum(i*s for i, s in
                    zip(index, encoding.logical_strides_elements, strict=True)))*width
                logical.extend(source[offset:offset+width])
            result[arg.tensor] = bytes(logical)
        return result

    def runtime_address_evidence(self, base_addresses: Sequence[int]) -> dict:
        """A separate address check receipt; format construction is not this proof."""
        arguments = self.validate_runtime_addresses(base_addresses)
        return {"schema": "compact_runtime_addresses_v1", "status": "validated",
            "command_buffer_sha256": self._evidence["command_buffer_sha256"],
            "compact_contract_sha256": self._evidence["compact_contract_sha256"],
            "target_abi_sha256": self._evidence["target_abi_sha256"],
            "base_addresses": list(base_addresses), "original_argument_addresses": list(arguments),
            "alignment_and_nonoverlap_checked": True,
            "live_target_allocation_verified": False, "entry_invoked": False,
            "scope": "supplied addresses obey base+offset/alignment/nonoverlap; allocation validity is caller-owned"}

    def setup_evidence(self) -> dict:
        return json.loads(json.dumps(self._evidence))


def resolve_compact_binding(command_buffer: Mapping[str, Any], compact_contract: Mapping[str, Any],
                            logical_payloads: Mapping[str, bytes], *, target_abi: CompactTargetABI,
                            max_storage_bytes: int,
                            prepack_authorizations: Mapping[str, Any] | None = None) -> CompactBinding:
    """Validate completely before allocation, then relocate exact scalar bytes.

    Every read/readwrite argument needs caller-provided logical bytes. There are
    no deterministic fallback inputs and no option to waive immutable prepack
    authorization. The resource budget bounds base storage AND intermediate
    logical/physical format payload sizes individually, not Python object RSS.
    """
    _integer(max_storage_bytes, "host storage budget", positive=True)
    _require(type(target_abi) is CompactTargetABI, "explicit host CompactTargetABI facts required")
    _require(all(type(value) is tuple for value in (target_abi.pointer_index_bits, target_abi.base_alignments,
             target_abi.argument_alignments, target_abi.base_access)), "target ABI facts require immutable ordered tuples")
    _require(isinstance(command_buffer, Mapping), "command buffer must be a mapping")
    _require(isinstance(target_abi.provenance, str) and bool(target_abi.provenance.strip()), "target ABI provenance required")
    _require(isinstance(compact_contract, Mapping) and compact_contract.get("schema") == "compact_pointer_entry_v1",
             "unsupported compact entry contract")
    for key in ("original_symbol", "compact_symbol", "binding_provenance"):
        _require(isinstance(compact_contract.get(key), str) and bool(compact_contract[key].strip()), f"missing compact {key}")
    _require(compact_contract["original_symbol"] != compact_contract["compact_symbol"], "compact entry must have a distinct symbol")
    _require(compact_contract.get("storage_reused") is False and compact_contract.get("alignment_or_noalias_added") is False
             and compact_contract.get("whole_cfg_preserved_under_pointer_substitution") is True,
             "unsupported compact aliasing/reuse or pointer substitution contract")
    abi = command_buffer.get("kernel_abi")
    _require(isinstance(abi, Mapping) and abi.get("kind") == "whole_program", "original whole_program ABI required")
    args, outputs = abi.get("args"), abi.get("outputs")
    _require(isinstance(args, list) and bool(args) and all(isinstance(arg, Mapping) for arg in args), "complete original argument list required")
    bases, bindings = compact_contract.get("bases"), compact_contract.get("bindings")
    _require(isinstance(bases, list) and bool(bases) and all(isinstance(base, Mapping) for base in bases), "compact bases malformed")
    _require(isinstance(bindings, list) and all(isinstance(binding, Mapping) for binding in bindings), "compact bindings malformed")
    _require(type(compact_contract.get("original_argument_count")) is int and compact_contract["original_argument_count"] == len(args)
             and type(compact_contract.get("compact_argument_count")) is int and compact_contract["compact_argument_count"] == len(bases),
             "compact argument counts disagree with original ABI")
    _require(len(target_abi.pointer_index_bits) == len(target_abi.base_alignments) == len(target_abi.base_access) == len(bases)
             and len(target_abi.argument_alignments) == len(args), "incomplete target ABI facts")
    for alignment in (*target_abi.base_alignments, *target_abi.argument_alignments):
        _integer(alignment, "target alignment", positive=True)
        _require(alignment & (alignment-1) == 0, "unsupported non-power-of-two target alignment")
    names, extents = set(), []
    for i, base in enumerate(bases):
        name = base.get("name")
        _require(isinstance(name, str) and bool(name.strip()) and name not in names, "compact base names must be unique and nonempty")
        names.add(name)
        extent = _integer(base.get("byte_extent"), "base extent", positive=True)
        bits = _integer(base.get("pointer_index_bits"), "pointer index width", positive=True)
        _integer(target_abi.pointer_index_bits[i], "host target pointer index width", positive=True)
        _require(bits == target_abi.pointer_index_bits[i], "candidate pointer index width differs from target facts")
        _require(extent.bit_length() < bits, "base extent exceeds signed pointer index range")
        _require(target_abi.base_access[i] in ("read", "write", "readwrite"), "unsupported base access group")
        extents.append(extent)
    _require(sum(extents) <= max_storage_bytes, "compact bases exceed host allocation budget")
    for binding in bindings:
        for key in ("argument_index", "base_index", "byte_offset", "byte_extent"):
            _integer(binding.get(key), key, positive=key == "byte_extent")
    ordered = sorted(bindings, key=lambda row: row["argument_index"])
    _require([row["argument_index"] for row in ordered] == list(range(len(args))), "bindings must cover every original argument exactly once")
    params = command_buffer.get("params")
    records = params.get("storage_encodings") if isinstance(params, Mapping) else None
    _require(isinstance(records, Mapping), "compact binding requires explicit per-tensor storage encodings")
    _require(isinstance(logical_payloads, Mapping), "logical payloads must be an exact byte mapping")
    access = {arg.get("tensor"): arg.get("access") for arg in args}
    _require(len(access) == len(args) and all(isinstance(name, str) and mode in ("read", "write", "readwrite")
             for name, mode in access.items()), "original arguments must be distinct typed tensor pointers")
    _require(set(logical_payloads) == {name for name, mode in access.items() if mode != "write"},
             "exact logical bytes required for every read/readwrite argument, and no write-only initializer")
    _require(set(records) == set(access), "explicit storage must cover every original argument")
    inputs, spans = {}, [[] for _ in bases]
    total_format_storage = total_logical_storage = 0
    for index, (arg, binding) in enumerate(zip(args, ordered, strict=True)):
        name = arg["tensor"]
        encoding = GroupedAxesStorage.from_dict(records[name])
        width = _element_bytes(encoding.dtype)
        extent = encoding.storage_elements*width
        logical_extent = prod(encoding.logical_shape)*width
        total_format_storage += extent
        total_logical_storage += logical_extent
        _require(max(total_format_storage, total_logical_storage) <= max_storage_bytes,
                 "aggregate tensor storage exceeds host format budget")
        base, offset = binding["base_index"], binding["byte_offset"]
        _require(base < len(bases), "binding references a missing base")
        _require(binding["byte_extent"] == extent, "binding extent differs from exact explicit tensor storage bytes")
        _require(offset+extent <= extents[base], "tensor binding exceeds base bounds")
        _require(target_abi.base_access[base] == arg["access"], "tensor access differs from homogeneous base access group")
        alignment = target_abi.argument_alignments[index]
        _require(target_abi.base_alignments[base] % alignment == 0 and offset % alignment == 0,
                 "base guarantee or byte offset cannot satisfy argument alignment")
        spans[base].append((offset, offset+extent))
        if arg["access"] != "write":
            payload = logical_payloads[name]
            _require(type(payload) is bytes and len(payload) == logical_extent, "logical payload differs from declared dtype/shape byte extent")
            # Little-endian integers are only opaque byte-word containers here;
            # serializing the same words below preserves any target byte order.
            words = [int.from_bytes(payload[i:i+width], "little") for i in range(0, len(payload), width)]
            inputs[name] = Tensor(encoding.logical_shape, words, encoding.dtype)
    for intervals in spans:
        _require(bool(intervals), "unused compact base")
        intervals.sort()
        _require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "tensor bindings overlap; aliasing/reuse unsupported")
    storage = resolve_storage_bindings(command_buffer, inputs, max_storage_bytes=max_storage_bytes,
                                       prepack_authorizations=prepack_authorizations)
    _require(storage is not None, "explicit storage resolution failed")
    packed = {}
    for name, binding in storage.items():
        if binding.access != "write":
            width = _element_bytes(binding.encoding.dtype)
            packed[name] = b"".join(word.to_bytes(width, "little") for word in binding.pack_words(inputs[name].data))
    arenas = [bytearray(extent) for extent in extents]
    arguments = []
    for arg, binding in zip(args, ordered, strict=True):
        name, base, offset, extent = arg["tensor"], binding["base_index"], binding["byte_offset"], binding["byte_extent"]
        if name in packed:
            arenas[base][offset:offset+extent] = packed[name]
        arguments.append(_Argument(name, base, offset, extent, storage[name]))
    evidence = {"schema": "compact_caller_binding_v1", "compact_symbol": compact_contract["compact_symbol"],
        "original_symbol": compact_contract["original_symbol"], "command_buffer_sha256": _digest(command_buffer),
        "compact_contract_sha256": _digest(compact_contract), "target_abi": asdict(target_abi),
        "target_abi_sha256": _digest(asdict(target_abi)), "base_storage_bytes": extents,
        "original_argument_tensors": [arg.tensor for arg in arguments],
        "logical_input_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in logical_payloads.items()},
        "storage_setup": {name: binding.setup_evidence() for name, binding in storage.items()},
        "runtime_addresses_validated": False,
        "packing_inside_compute_roi": False, "storage_reused": False, "numerical_equivalence": "UNPROVEN",
        "scope": "exact byte relocation and caller address contract only; target allocation/invocation not performed"}
    return CompactBinding(arenas=arenas, arguments=tuple(arguments), outputs=tuple(outputs),
                          target_abi=target_abi, evidence=evidence)
