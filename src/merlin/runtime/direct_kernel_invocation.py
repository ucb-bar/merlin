"""Explicit repeated pointer calls and complete original output histories.

The caller derives the original ABI independently and selects this evaluation
plan. Counts and raw histories are observations, not numerical, lifetime,
synchronization or hardware authority. Default harnesses select no plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from merlin.common.quant_formats import get
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi


def _identifier(value):
    from .direct_kernel_harness import _identifier as validate

    return validate(value)


@dataclass(frozen=True)
class InvocationOutputHistory:
    original_name: str
    emitted_tensor: str
    symbol: str
    shape: tuple[int, ...]
    dtype: str
    bytes_per_invocation: int
    invocation_count: int

    @property
    def byte_extent(self):
        return self.bytes_per_invocation * self.invocation_count


@dataclass(frozen=True)
class DirectKernelInvocationObservation:
    """Every original output's raw bytes for every completed selected call."""

    observed_count: int
    output_bytes: tuple[tuple[str, tuple[bytes, ...]], ...]


@dataclass(frozen=True)
class DirectKernelInvocationPlan:
    """Independently selected immutable ABI/count/storage declarations.

    The initial supported ABI has separate read-only inputs and write-only
    outputs, ordered inputs then outputs. Alias/in-place interfaces refuse.
    Input storage persists unchanged between calls; output storage is reused.
    Each completed call is copied into a separate complete output history.
    """

    original_abi: CompileOnlySourceAbi
    count: int
    history_prefix: str
    count_symbol: str

    def record(self):
        if type(self.original_abi) is not CompileOnlySourceAbi:
            raise ValueError("invocation evaluation requires the exact typed original source ABI")
        abi = self.original_abi.record()
        if type(self.count) is not int or not 1 <= self.count < 1 << 64:
            raise ValueError("invocation evaluation needs an explicit positive uint64 call count")
        _identifier(self.history_prefix)
        _identifier(self.count_symbol)
        all_slots = (*self.original_abi.inputs, *self.original_abi.outputs)
        if len({slot.name for slot in all_slots}) != len(all_slots):
            raise ValueError("invocation evaluation does not admit overlapping original input/output names")
        return {
            "schema": "merlin.direct_kernel_invocations.v1",
            "original_abi": abi,
            "count": self.count,
            "history_prefix": self.history_prefix,
            "count_symbol": self.count_symbol,
            "input_lifecycle": "same read-only source storage",
            "output_lifecycle": "same output storage, complete separate per-call snapshots",
            "scope": "explicit software evaluation selection; no effect or runtime authority",
        }

    def bind(self, cb, *, entry_symbol, completion_symbol=None):
        self.record()
        bindings = self.original_abi.bind(cb)["bindings"]
        expected = [
            {"tensor": row["emitted"], "access": "read" if row["role"] == "input" else "write"} for row in bindings
        ]
        outputs = [row for row in bindings if row["role"] == "output"]
        if cb["kernel_abi"]["args"] != expected or cb["kernel_abi"]["outputs"] != [row["emitted"] for row in outputs]:
            raise ValueError("invocation evaluation changes original argument/output order or access")
        _identifier(entry_symbol)
        if completion_symbol is not None:
            _identifier(completion_symbol)
        internal = {"invocation", "invocation_completed", "byte", "console_init", "htif_puts", "htif_exit", "main"}
        if entry_symbol in internal or completion_symbol in internal:
            raise ValueError("invocation entrypoints overlap harness observation identifiers")
        reserved = {entry_symbol, completion_symbol, *internal}
        reserved.update("tensor_" + str(index) for index in range(len(expected)))
        histories = []
        for index, (original, binding) in enumerate(zip(self.original_abi.outputs, outputs, strict=True)):
            symbol = _identifier(self.history_prefix + "_" + str(index))
            dtype = get(original.dtype)
            if (
                not original.shape
                or any(size <= 0 for size in original.shape)
                or dtype.element_bits not in (8, 16, 32, 64)
                or dtype.is_block_scaled
            ):
                raise ValueError("invocation history requires concrete byte-aligned original output types")
            width = dtype.element_bits // 8
            extent = math.prod(original.shape) * width
            if extent * self.count >= 1 << 64:
                raise ValueError("invocation history offset exceeds its explicit uint64 storage")
            if symbol in reserved or symbol == self.count_symbol:
                raise ValueError("invocation observation symbols overlap original storage or entrypoints")
            reserved.add(symbol)
            histories.append(
                InvocationOutputHistory(
                    original.name, binding["emitted"], symbol, original.shape, original.dtype, extent, self.count
                )
            )
        if self.count_symbol in reserved:
            raise ValueError("invocation count symbol overlaps original storage or entrypoints")
        return tuple(histories)

    def decode(self, objects, *, cb, entry_symbol, byte_order, completion_symbol=None):
        """Check the complete observation roster and retain exact raw snapshots.

        The caller independently proves these bytes came from the declared ELF
        objects. Passing a mapping or matching count alone cannot issue proof.
        """
        histories = self.bind(cb, entry_symbol=entry_symbol, completion_symbol=completion_symbol)
        if byte_order not in ("little", "big") or type(objects) is not dict:
            raise ValueError("invocation observations need explicit byte order and a complete raw object mapping")
        if set(objects) != {self.count_symbol, *(row.symbol for row in histories)}:
            raise ValueError("invocation observations have a changed or partial original history roster")
        if type(objects[self.count_symbol]) is not bytes or len(objects[self.count_symbol]) != 8:
            raise ValueError("invocation count does not contain its complete uint64 bytes")
        count = int.from_bytes(objects[self.count_symbol], byte_order)
        if count != self.count:
            raise ValueError("observed completed invocation count differs from the selected original evaluation")
        outputs = []
        for history in histories:
            raw = objects[history.symbol]
            if type(raw) is not bytes or len(raw) != history.byte_extent:
                raise ValueError("invocation history omits original output bytes")
            width = history.bytes_per_invocation
            outputs.append(
                (history.original_name, tuple(raw[start : start + width] for start in range(0, len(raw), width)))
            )
        return DirectKernelInvocationObservation(count, tuple(outputs))
