"""Bounded byte origins and forwarding epochs for transport proofs; no tensor values."""

from dataclasses import dataclass

import numpy as np
from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.llvm import LLVMArrayType


@dataclass(frozen=True)
class Pointer:
    root: object
    offset: int = 0


@dataclass(frozen=True)
class Copy:
    origins: tuple
    references: tuple = ()


@dataclass(frozen=True)
class ReadReference:
    root: object
    offsets: object
    epochs: object
    active: object = True


class CopyStorage:
    """Exact input-byte offsets, zero padding and initialization/consumption epochs.

    -2 means uninitialized and -1 means zero. Nonnegative entries identify the
    original input, weight or decoded accelerator result byte. No tensor payload
    is loaded by this instrument. Source epochs discharge only into a tracked
    later copy, an accelerator input transfer or unchanged output publication.
    """

    def __init__(self, capacity):
        if sum(capacity.get(root, 0) for root in (0, 1)) >= 1 << 63:
            raise ValueError("input byte-origin storage exceeds finite exact index range")
        self.capacity = capacity
        self.origins = {}
        self.consumed = {}
        self.epochs = {}
        self.next_epoch = 0
        self.output_base = sum(capacity[root] for root in (0, 1)) if all(
            root in capacity for root in (0, 1, 2)
        ) else None
        self.output_written = None
        self.pending_results = set()

    def _allocate(self, root):
        if root not in self.origins:
            if sum(self.capacity[key] for key in self.origins) + self.capacity[root] > 64 << 20:
                raise ValueError("input scratch exceeds the aggregate finite storage proof budget")
            self.origins[root] = np.full(self.capacity[root], -2, dtype=np.int64)
            self.consumed[root] = np.zeros(self.capacity[root], dtype=np.bool_)
            self.epochs[root] = np.zeros(self.capacity[root], dtype=np.int64)

    def __contains__(self, key):
        root, offset = key
        return root in self.origins and self.origins[root][offset] != -2

    def __getitem__(self, key):
        if key not in self:
            raise ValueError("input preparation reads uninitialized scratch")
        root, offset = key
        value = int(self.origins[root][offset])
        if value == -1:
            return "zero"
        if self.output_base is not None and value >= self.output_base:
            return 2, value - self.output_base
        base = self.capacity.get(0)
        return (1, value - base) if base is not None and value >= base else (0, value)

    def __setitem__(self, key, origin):
        root, offset = key
        self._allocate(root)
        if self.origins[root][offset] >= 0 and not self.consumed[root][offset]:
            raise ValueError("input scratch copy is overwritten before accelerator consumption")
        self.origins[root][offset] = -1 if origin == "zero" else origin[1] + (
            self.output_base if origin[0] == 2 else self.capacity.get(0, 0) if origin[0] == 1 else 0
        )
        self.consumed[root][offset] = False
        self.next_epoch += 1
        self.epochs[root][offset] = self.next_epoch

    def store_batch(self, root, offsets, origins, written, read=None):
        self._allocate(root)
        if (np.any(offsets < 0) or np.any(offsets >= self.capacity[root])
                or len(np.unique(offsets)) != len(offsets)):
            raise ValueError("batched copy has out-of-bounds or overlapping writes")
        if root not in written:
            written[root] = np.zeros(self.capacity[root], dtype=np.bool_)
        if np.any(written[root][offsets]):
            raise ValueError("batched copy reorders conflicting scratch writes")
        if read is not None and root in read and np.any(read[root][offsets]):
            raise ValueError("batched copy reorders conflicting scratch reads and writes")
        if np.any((self.origins[root][offsets] >= 0) & ~self.consumed[root][offsets]):
            raise ValueError("input scratch copy is overwritten before accelerator consumption")
        written[root][offsets] = True
        self.origins[root][offsets] = origins
        self.consumed[root][offsets] = False
        self.next_epoch += 1
        self.epochs[root][offsets] = self.next_epoch

    def read_batch(self, root, offsets, written=None, read=None):
        if root in self.pending_results:
            raise ValueError("CPU reads an accelerator result before the command fence")
        if root not in self.origins or np.any(self.origins[root][offsets] == -2):
            raise ValueError("input preparation reads uninitialized scratch")
        if written is not None and root in written and np.any(written[root][offsets]):
            raise ValueError("batched copy reorders conflicting scratch reads and writes")
        if read is not None:
            if root not in read:
                read[root] = np.zeros(self.capacity[root], dtype=np.bool_)
            read[root][offsets] = True
        return self.origins[root][offsets].copy(), ReadReference(
            root, np.asarray(offsets).copy(), self.epochs[root][offsets].copy()
        )

    def forward(self, reference):
        """Discharge a source epoch only after an unchanged tracked destination exists.

        Every destination has its own consumption obligation. Forwarding edges
        point to later writes, so all finite chains must end at a decoded input
        transfer or at publication of accelerator result bytes to the output.
        A dead load, a stale epoch and an overwritten destination prove nothing.
        """
        if np.any(self.epochs[reference.root][reference.offsets] != reference.epochs):
            raise ValueError("scratch copy refers to a changed source write epoch")
        self.consumed[reference.root][reference.offsets] = True

    def publish_output(self, offsets, origins):
        if (self.output_base is None or np.any(origins < self.output_base)
                or np.any(offsets < 0) or np.any(offsets >= self.capacity[2])
                or len(np.unique(offsets)) != len(offsets)):
            raise ValueError("CPU output store is not unchanged bounded accelerator result bytes")
        if self.output_written is None:
            self.output_written = np.zeros(self.capacity[2], dtype=np.bool_)
        if np.any(self.output_written[offsets]):
            raise ValueError("output publication repeats a destination byte")
        self.output_written[offsets] = True

    def result_rows(self, root, offset, rows, width, stride):
        """Record actual decoded result writes; no numerical result is invented."""
        if self.output_base is None or rows * width > 64 << 20:
            raise ValueError("accelerator result exceeds bounded storage proof")
        offsets = (offset + np.arange(rows)[:, None] * stride + np.arange(width)).ravel()
        if root == 2:
            self.publish_output(offsets, np.full(len(offsets), self.output_base, dtype=np.int64))
        else:
            self.store_batch(root, offsets, self.output_base + offsets, {})
            self.pending_results.add(root)

    def consume(self, root, offsets):
        if root not in self.origins or np.any(self.origins[root][offsets] == -2):
            raise ValueError("accelerator consumes uninitialized input scratch")
        if self.output_base is not None and np.any(self.origins[root][offsets] >= self.output_base):
            raise ValueError("accelerator input contains output-result bytes")
        self.consumed[root][offsets] = True

    def consume_rows(self, root, offset, rows, width, stride):
        """Consume the union of bounded row spans, without expanding repeated rows."""
        if root not in self.origins:
            raise ValueError("accelerator consumes uninitialized input scratch")
        if stride <= width:
            end = offset + (rows - 1) * stride + width
            self.consume(root, slice(offset, end))
            return
        for start in range(offset, offset + rows * stride, stride):
            self.consume(root, slice(start, start + width))

    def require_consumed(self):
        for root, origins in self.origins.items():
            if np.any((origins >= 0) & ~self.consumed[root]):
                raise ValueError("input copy scratch never feeds decoded accelerator input")


def scratch_limit(buffers):
    """Finite resource limit from the original logical convolution's patch extent."""
    weights, output = buffers[1]["shape"], buffers[2]["shape"]
    patches = 0
    if (len(weights) == len(output) == 2 and weights[-1] == output[-1]
            and all(type(n) is int and n > 0 for n in [*weights, *output])):
        patches = output[0] * weights[0]
    return min(64 << 20, max(1 << 20, patches))


def byte_gep_stride(typ):
    """Byte and nested byte-array indexing has no target-dependent padding."""
    stride = 1
    while isinstance(typ, LLVMArrayType):
        if not 0 < typ.size.data <= 1 << 24:
            raise ValueError("input preparation has unbounded aggregate pointer layout")
        stride *= typ.size.data
        typ = typ.type
    if typ != IntegerType(8) or stride > 1 << 24:
        raise ValueError("input preparation has unsupported pointer layout")
    return stride
