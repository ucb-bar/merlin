"""Explicit original logical pointer storage, derived without tensor allocation.

The policy is selected independently of compiler output. These typed values are
software ABI declarations, not provenance, actual allocation or physical proof.
Source authorities and emitted-code checkers must establish those separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from merlin.development.targetgen.contract.compile_only import CompileOnlySourceAbi


def _integer_bits(dtype):
    from merlin.common import quant_formats

    bits = quant_formats.machine_bits(dtype)
    if bits is not None and any(
        dtype.startswith(prefix) and dtype[len(prefix) :].isdecimal() for prefix in ("int", "uint", "i", "u")
    ):
        return bits
    raise ValueError("original pointer storage currently needs raw integer element types")


@dataclass(frozen=True)
class PointerStoragePolicy:
    """Closed explicit software choices; missing choices have no inferred default."""

    layout: str
    argument_order: str
    slot_aliasing: str
    extent_derivation: str
    byte_order: str
    tensor_alignment: int

    def record(self):
        if (
            self.layout != "row_major_contiguous"
            or self.argument_order != "inputs_then_outputs"
            or self.slot_aliasing != "disjoint"
            or self.extent_derivation != "static_original_tensor_type"
            or self.byte_order not in {"little", "big"}
            or type(self.tensor_alignment) is not int
            or self.tensor_alignment < 1
            or self.tensor_alignment & (self.tensor_alignment - 1)
        ):
            raise ValueError("original pointer storage requires every explicit supported software ABI choice")
        return dict(self.__dict__)


@dataclass(frozen=True)
class PointerStorageSlot:
    ordinal: int
    role: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    element_bits: int
    element_count: int
    byte_extent: int
    element_strides: tuple[int, ...]

    def record(self):
        return {**self.__dict__, "shape": list(self.shape), "element_strides": list(self.element_strides)}


@dataclass(frozen=True)
class OriginalPointerStorageContract:
    """Original complete logical slots under an independently selected policy.

    Each pointer must denote the complete declared object when executed. The
    contract does not prove physical storage exists, fits a machine, stays alive
    or matches an independently qualified runtime. A checker may prove static
    code obligations under these original software ABI preconditions.
    """

    original_abi: CompileOnlySourceAbi
    policy: PointerStoragePolicy

    @property
    def slots(self):
        if type(self.original_abi) is not CompileOnlySourceAbi or type(self.policy) is not PointerStoragePolicy:
            raise ValueError("original pointer storage needs exact typed original ABI and explicit policy")
        self.original_abi.record()
        self.policy.record()
        slots, names = [], set()
        for role, tensors in (("input", self.original_abi.inputs), ("output", self.original_abi.outputs)):
            for tensor in tensors:
                bits = _integer_bits(tensor.dtype)
                if (
                    bits % 8
                    or bits < 8
                    or any(extent <= 0 for extent in tensor.shape)
                    or not tensor.shape
                    or tensor.name in names
                ):
                    raise ValueError(
                        "original pointer storage currently needs distinct positive byte-sized integer slots"
                    )
                names.add(tensor.name)
                strides, stride = [], 1
                for extent in reversed(tensor.shape):
                    strides.append(stride)
                    stride *= extent
                count = math.prod(tensor.shape)
                slots.append(
                    PointerStorageSlot(
                        len(slots),
                        role,
                        tensor.name,
                        tensor.shape,
                        tensor.dtype,
                        bits,
                        count,
                        count * (bits // 8),
                        tuple(reversed(strides)),
                    )
                )
        return tuple(slots)

    def record(self):
        return {
            "schema": "merlin.original_pointer_storage.v1",
            "policy": self.policy.record(),
            "original_abi": self.original_abi.record(),
            "slots": [slot.record() for slot in self.slots],
            "scope": (
                "original software ABI preconditions; physical allocation, lifetime and runtime equivalence unproved"
            ),
        }

    def bind_candidate(self, cb):
        """Check actual call ordering against the original software declaration.

        Candidate metadata selects neither the policy nor original extents. Its
        pointer roster is checked only as the actual shared harness call binding.
        Emitted memory/index/output semantics still require an independent check.
        """
        slots = self.slots
        binding = self.original_abi.bind(cb)
        expected = [
            {"tensor": row["emitted"], "access": "read" if row["role"] == "input" else "write"}
            for row in binding["bindings"]
        ]
        if cb["kernel_abi"]["args"] != expected or len(expected) != len(slots):
            raise ValueError("candidate pointer order/access differs from the original disjoint slot ABI")
        return binding
