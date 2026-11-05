"""Explicit original tensor storage for a conditional descriptor transport check.

These declarations supply no provenance or physical allocation authority. Every
offset, stride, capacity and alignment is selected by the original caller; none
is inferred from emitted code, current machine defaults or tensor contiguity.
"""

from dataclasses import dataclass
from math import prod
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi


@dataclass(frozen=True)
class DescriptorLimits:
    source_bytes: int
    receipt_bytes: int
    nesting: int
    integer_bits: int
    rank: int
    operations: int
    object_bytes: int

    def record(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("descriptor observation requires explicit positive reader limits")
        return dict(vars(self))


@dataclass(frozen=True)
class TensorStorageDeclaration:
    """One original caller object; accessible bytes and actual lifetime unproved."""

    name: str
    role: str
    offset_elements: int
    element_strides: tuple[int, ...]
    capacity_elements: int
    alignment: int

    def record(self, tensor, role):
        if (
            self.name != tensor.name
            or type(self.name) is not str
            or self.role != role
            or type(self.role) is not str
            or type(self.offset_elements) is not int
            or self.offset_elements < 0
            or type(self.capacity_elements) is not int
            or self.capacity_elements < 0
            or type(self.element_strides) is not tuple
            or len(self.element_strides) != len(tensor.shape)
            or any(type(value) is not int or value <= 0 for value in self.element_strides)
            or type(self.alignment) is not int
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("original descriptor storage needs exact ordered explicit choices")
        nonempty = all(tensor.shape)
        last = self.offset_elements + sum(
            (extent - 1) * stride for extent, stride in zip(tensor.shape, self.element_strides, strict=True)
        )
        if (nonempty and last >= self.capacity_elements) or self.offset_elements > self.capacity_elements:
            raise ValueError("original descriptor footprint exceeds its declared storage")
        if role == "output" and nonempty:
            span = 0
            for stride, extent in sorted(zip(self.element_strides, tensor.shape, strict=True)):
                if extent > 1:
                    if stride <= span:
                        raise ValueError("original writable descriptor lacks supported injective layout proof")
                    span += (extent - 1) * stride
        return {**vars(self), "element_strides": list(self.element_strides), "logical_elements": prod(tensor.shape)}


@dataclass(frozen=True)
class OriginalDescriptorSource:
    """Complete original source/ABI plus explicitly selected ranked memref ABI.

    The ABI choice assigns allocated/aligned/offset/size/stride roles to the
    public MLIR ranked memref expansion. Index field widths still come from
    actual parsed IR; this selection contains no machine width or byte offset.
    """

    path: Path
    sha256: str
    entry_symbol: str
    c_interface_symbol: str
    original_abi: CompileOnlySourceAbi
    descriptor_abi: str
    storage: tuple[TensorStorageDeclaration, ...]
    slot_aliasing: str

    def record(self, limits):
        if type(limits) is not DescriptorLimits:
            raise ValueError("descriptor source requires exact selected reader limits")
        limits.record()
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or not is_sha256(self.sha256)
            or type(self.original_abi) is not CompileOnlySourceAbi
            or self.descriptor_abi != "mlir_ranked_memref_ciface_v1"
            or type(self.descriptor_abi) is not str
            or self.slot_aliasing not in {"disjoint", "unproved"}
            or type(self.slot_aliasing) is not str
            or type(self.storage) is not tuple
            or any(type(slot) is not TensorStorageDeclaration for slot in self.storage)
            or any(
                type(name) is not str or not name.isascii() or not name.isidentifier()
                for name in (self.entry_symbol, self.c_interface_symbol)
            )
            or self.entry_symbol == self.c_interface_symbol
        ):
            raise ValueError("descriptor check requires complete explicit original source/storage/ABI choices")
        abi = self.original_abi.record()
        tensors = (*self.original_abi.inputs, *self.original_abi.outputs)
        roles = ("input",) * len(self.original_abi.inputs) + ("output",) * len(self.original_abi.outputs)
        if len(tensors) != len(self.storage) or any(len(slot.shape) > limits.rank for slot in tensors):
            raise ValueError("descriptor storage omits an original slot or exceeds the selected rank bound")
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "entry_symbol": self.entry_symbol,
            "c_interface_symbol": self.c_interface_symbol,
            "original_abi": abi,
            "descriptor_abi": self.descriptor_abi,
            "slot_aliasing": self.slot_aliasing,
            "storage": [
                slot.record(tensor, role) for slot, tensor, role in zip(self.storage, tensors, roles, strict=True)
            ],
            "scope": "explicit caller preconditions; physical storage, body semantics and alias ownership unproved",
        }
