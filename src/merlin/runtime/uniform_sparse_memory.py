"""Sparse byte storage with an explicit uniform default and cloned tensor reads.

This changes representation only. Callers still execute their original
instruction implementations and select any qualification receipt explicitly.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch


SEMANTICS_ID = "uniform-default-byte-sparse-memory-cloned-reads-v1"
QUALIFICATION_SCHEMA = "merlin.uniform_sparse_byte_memory_qualification.v1"
REQUIRED_CASES = (
    "randomized_cross_page", "zero_writes", "uniform_page_reclamation",
    "bounds_and_errors", "mutation_isolation",
)


def _default_byte(value):
    if type(value) is not int or not 0 <= value <= 255:
        raise ValueError("default_byte must be an integer in [0,255]")
    return value


class UniformSparseByteMemory:
    """Contiguous tensor slice API over lazily materialized uniform pages.

    Reads return independent CPU uint8 tensors. Writes snapshot the detached,
    CPU, uint8, contiguous tensor bytes before mutation. A page is removed
    only when every byte equals the chosen default, preserving explicit zero
    writes when the default is nonzero. Range checks occur before any write.
    """

    PAGE_BYTES = 4096

    def __init__(self, size: int, default_byte: int = 0xCD):
        self.size = size
        self.default_byte = _default_byte(default_byte)
        self._pages: dict[int, bytearray] = {}
        self._default_page = bytes((self.default_byte,))*self.PAGE_BYTES

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Functional memory supports contiguous slices only")
        start = 0 if key.start is None else key.start
        stop = self.size if key.stop is None else key.stop
        if start < 0 or stop < start or stop > self.size:
            raise AssertionError(f"Memory read out of bounds: [{start}, {stop})/{self.size}")
        if start == stop:
            return torch.empty(0, dtype=torch.uint8)
        output = bytearray((self.default_byte,))*(stop-start)
        cursor = start
        while cursor < stop:
            page_index, page_offset = divmod(cursor, self.PAGE_BYTES)
            count = min(stop-cursor, self.PAGE_BYTES-page_offset)
            page = self._pages.get(page_index)
            if page is not None:
                output[cursor-start:cursor-start+count] = page[page_offset:page_offset+count]
            cursor += count
        return torch.frombuffer(output, dtype=torch.uint8).clone()

    def __setitem__(self, key: slice, value: torch.Tensor) -> None:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Functional memory supports contiguous slices only")
        start = 0 if key.start is None else key.start
        data = value.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes()
        stop = start+len(data) if key.stop is None else key.stop
        if start < 0 or stop < start or stop > self.size or stop-start != len(data):
            raise AssertionError(f"Memory write out of bounds: [{start}, {stop})/{self.size}")
        cursor, data_offset = start, 0
        while cursor < stop:
            page_index, page_offset = divmod(cursor, self.PAGE_BYTES)
            count = min(stop-cursor, self.PAGE_BYTES-page_offset)
            chunk = data[data_offset:data_offset+count]
            page = self._pages.get(page_index)
            if page is None and chunk != self._default_page[page_offset:page_offset+count]:
                page = bytearray(self._default_page)
                self._pages[page_index] = page
            if page is not None:
                page[page_offset:page_offset+count] = chunk
                if page == self._default_page:
                    del self._pages[page_index]
            cursor += count
            data_offset += count


def adapter_identity(default_byte=0xCD):
    """Source-bound identity for an explicitly selected storage representation."""
    return {
        "adapter": "merlin.runtime.uniform_sparse_memory.UniformSparseByteMemory",
        "adapter_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "semantics": SEMANTICS_ID,
        "default_byte": _default_byte(default_byte),
        "page_bytes": UniformSparseByteMemory.PAGE_BYTES,
        "torch_version": str(torch.__version__),
        "unmaterialized_reads": "uniform-default-byte",
        "writes": "detached-cpu-uint8-contiguous-snapshot",
        "reads": "independent-cloned-cpu-uint8-tensor",
        "page_reclamation": "all-bytes-equal-default-only",
        "range_and_error_contract": "contiguous-sparse-byte-memory-v1",
        "instruction_arithmetic": "unchanged-by-storage-adapter",
    }


def reference_source_identity(source_path, expected_source_sha256):
    """Refuse a reference other than the caller's explicitly pinned source."""
    actual = sha256(Path(source_path).read_bytes()).hexdigest()
    if actual != expected_source_sha256:
        raise ValueError("byte-memory reference source differs from the selected SHA256")
    return actual


def verify_qualification(receipt, *, reference_source, expected_reference_sha256, default_byte=0xCD):
    """Authenticate a differential receipt; source hashing alone is not a PASS."""
    actual = reference_source_identity(reference_source, expected_reference_sha256)
    expected = adapter_identity(default_byte)
    checks = receipt.get("checks", {})
    if (receipt.get("schema") != QUALIFICATION_SCHEMA or receipt.get("status") != "passed"
            or receipt.get("reference_source_sha256") != actual
            or any(receipt.get(key) != value for key, value in expected.items())
            or any(type(checks.get(key)) is not int or checks[key] <= 0 for key in REQUIRED_CASES)):
        raise ValueError("uniform sparse byte-memory qualification differs from selected sources or semantics")
    return {
        **expected,
        "reference_source_sha256": actual,
        "qualification_sha256": sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "qualification": "source-bound-byte-storage-differential-only",
        "hardware_or_arithmetic_qualification": False,
    }
