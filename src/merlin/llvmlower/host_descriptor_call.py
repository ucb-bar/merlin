"""Ordinary host calls through actual observed ranked-descriptor storage.

Only descriptor bytes are constructed here. No pointed-to tensor is read,
converted or staged; the caller owns accessible buffers and their lifetimes.
Transport observations do not qualify bodies, hardware or execution effects.
"""

import ctypes
import hashlib
import sys
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.targetgen.contract.compile_only import CompileOnlyTensor

from .descriptor_contract import TensorStorageDeclaration
from .descriptor_layout import DescriptorLayoutObservation, pack_descriptor, require_descriptor_bytes
from .descriptor_wrapper import read, require

_ISSUED = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class HostDescriptorBuffer:
    """Complete ordered original slot and explicit caller addresses/capacity."""

    tensor: CompileOnlyTensor
    storage: TensorStorageDeclaration
    allocated_address: int
    aligned_address: int


@dataclass(frozen=True, eq=False)
class HostDescriptorTransport:
    selection: object
    layout: DescriptorLayoutObservation
    root: Path
    image: Path
    pins: tuple[tuple[str, str], ...]
    records: tuple[tuple[str, str], ...]
    selection_json: bytes = b""

    def _identity(self):
        return id(self.selection), id(self.layout), self.root, self.image, self.pins, self.records, self.selection_json

    def verify(self):
        from .host_descriptor_compile import _prepared, join_host_image
        from .llvm_dialect_product import verify_llvm_dialect_product

        require(
            _ISSUED.get(self) == self._identity(), "host descriptor transport was not observed by ordinary compilation"
        )
        require(
            canonical_json(self.selection.verify()) == self.selection_json, "host descriptor original selection changed"
        )
        self.layout.verify()
        limits = self.selection.limits
        for path, digest in (*self.pins, *self.records):
            raw = read(path, max(limits.source_bytes, limits.receipt_bytes, limits.object_bytes))
            require(hashlib.sha256(raw).hexdigest() == digest, "host descriptor source/build/image bytes changed")
        product = verify_llvm_dialect_product(self.layout.wrapper.receipt)
        require(
            _prepared(self.selection, product) == self.layout.wrapper.source,
            "host descriptor original source binding changed",
        )
        object_record, records = join_host_image(
            selection=self.selection, build_root=self.root, product=product, image=self.image
        )
        require(
            str(object_record) == self.layout.record()["observation"]["object_record"]
            and tuple(str(path) for path in records) == tuple(path for path, _ in self.records),
            "host descriptor layout/image build membership changed",
        )
        return self.layout.record()["observation"]

    def require_image(self, image, *, name, scalar_result_dtype, n_args):
        self.verify()
        require(
            Path(image).absolute() == self.image and name == self.selection.source.entry_symbol,
            "host descriptor selection belongs to another loaded image/entry",
        )
        require(
            self.selection.source.c_interface_symbol == "_mlir_ciface_" + name,
            "host descriptor wrapper differs from the ordinary selected C-interface entry",
        )
        require(
            scalar_result_dtype is None and n_args is None, "host descriptor scalar/trampoline execution is unsupported"
        )

    def prepare(self, arguments):
        observed = self.verify()
        require(
            observed["pointer_bytes"] == ctypes.sizeof(ctypes.c_void_p) and observed["byte_order"] == sys.byteorder,
            "selected descriptor pointer representation differs from the executing host ABI",
        )
        slots = (*self.selection.source.original_abi.inputs, *self.selection.source.original_abi.outputs)
        require(
            type(arguments) in {tuple, list} and len(arguments) == len(slots),
            "host descriptor call omits/changes complete ordered original slots",
        )
        declarations = self.selection.source.storage
        total = sum(
            max(row["descriptor_alignment"], row["explicit_load_alignment"] or 1) - 1 + row["descriptor_bytes"]
            for row in observed["slots"]
        )
        require(total <= self.selection.limits.object_bytes, "host descriptor carriers exceed the selected byte budget")
        intervals, payloads = [], []
        for ordinal, (argument, slot, declaration, row) in enumerate(
            zip(arguments, slots, declarations, observed["slots"], strict=True)
        ):
            require(
                type(argument) is HostDescriptorBuffer
                and type(argument.tensor) is CompileOnlyTensor
                and type(argument.storage) is TensorStorageDeclaration
                and argument.tensor == slot
                and argument.storage == declaration,
                "host descriptor call changes ordered original dtype/shape/stride/capacity",
            )
            payload = pack_descriptor(
                observation=self.layout,
                ordinal=ordinal,
                allocated_address=argument.allocated_address,
                aligned_address=argument.aligned_address,
            )
            interval = (
                argument.aligned_address,
                argument.aligned_address + declaration.capacity_elements * row["element_bytes"],
            )
            require(
                all(
                    interval[0] == interval[1] or lo == hi or interval[1] <= lo or interval[0] >= hi
                    for lo, hi in intervals
                ),
                "host descriptor caller objects overlap under the original disjoint requirement",
            )
            intervals.append(interval)
            payloads.append(payload)
        carriers, addresses = [], []
        for ordinal, (payload, row, argument) in enumerate(zip(payloads, observed["slots"], arguments, strict=True)):
            alignment = max(row["descriptor_alignment"], row["explicit_load_alignment"] or 1)
            carrier = ctypes.create_string_buffer(len(payload) + alignment - 1)
            address = (ctypes.addressof(carrier) + alignment - 1) // alignment * alignment
            ctypes.memmove(address, payload, len(payload))
            require_descriptor_bytes(
                payload=payload,
                observation=self.layout,
                ordinal=ordinal,
                allocated_address=argument.allocated_address,
                aligned_address=argument.aligned_address,
            )
            carriers.append(carrier)
            addresses.append(ctypes.c_void_p(address))
        self.verify()
        return addresses, carriers, payloads

    def invoke(self, function, arguments):
        addresses, carriers, payloads = self.prepare(arguments)
        owner = self.root / "host-descriptor-calls" / uuid.uuid4().hex
        owner.mkdir(parents=True, mode=0o700)
        paths = tuple(owner / f"descriptor-{ordinal}.bin" for ordinal in range(len(payloads)))
        for path, payload in zip(paths, payloads, strict=True):
            path.write_bytes(payload)
        with I.observe_call(
            owner,
            stage="host_descriptor_execution",
            function=self._invoke,
            arguments={
                "entry": self.selection.source.c_interface_symbol,
                "slots": len(addresses),
                "original_selection": self.selection.verify(),
                "scope": "ranked descriptor transport only; body/runtime/effects unproved",
            },
            inputs=(self.image, self.selection.source.path, self.layout.wrapper.receipt, *paths),
            dependencies=(Path(__file__), *(Path(path) for path, _ in self.records)),
        ) as record:
            self.verify()
            self._invoke(function, addresses)
            self.verify()
            record.returned()
        return carriers

    @staticmethod
    def _invoke(function, addresses):
        function(*addresses)


def _retain_transport(transport):
    _ISSUED[transport] = transport._identity()
