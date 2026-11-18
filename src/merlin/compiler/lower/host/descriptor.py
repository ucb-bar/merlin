"""Explicit original tensor storage for a conditional descriptor transport check.

These declarations supply no provenance or physical allocation authority. Every
offset, stride, capacity and alignment is selected by the original caller; none
is inferred from emitted code, current machine defaults or tensor contiguity.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import sys
import uuid
import weakref
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

from xdsl.dialects import builtin, func, llvm
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.common.digest import canonical_json, is_sha256
from merlin.common.paths import module_source_path
from merlin.common.process import invocation_records as I
from merlin.common.record_formats import loads
from merlin.compiler.dialects.xdsl_context import text
from merlin.compiler.ir.llvm_context import make_llvm_context
from merlin.compiler.lower.host.descriptor_layout import (
    DescriptorLayoutObservation,
    observe_descriptor_layout,
    pack_descriptor,
    require_descriptor_bytes,
)
from merlin.compiler.lower.native.recipe import verify_llvm_dialect_product
from merlin.target.contract.build.services import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.target.contract.interfaces.grammar import admit_mlir_source, type_identity
from merlin.target.contract.interfaces.linalg import _dtype, make_linalg_context

# ---- descriptor_contract -----------------------------------------------------------------------


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


# ---- descriptor_wrapper ------------------------------------------------------------------------
# Bounded ordered ranked-memref C-interface transport observations.
#
# Only the complete transparent load/extract/call/return wrapper is checked.
# Implementation memory/index/numerical semantics, original owner provenance,
# physical storage and native machine equivalence remain separate obligations.

_ISSUED = weakref.WeakKeyDictionary()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path, maximum):
    path = Path(path).absolute()
    require(
        path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
        "descriptor source/products require canonical ordinary files",
    )
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "descriptor source/product exceeds its explicit byte bound")
    return raw


def parse(raw, limits, *, emitted):
    text = raw.decode("utf-8")
    admit_mlir_source(
        text,
        max_source_bytes=limits.source_bytes,
        max_nesting=limits.nesting,
        max_integer_bits=limits.integer_bits,
        allow_dense=False,
        allow_dense_resource=False,
    )
    try:
        module = Parser(make_llvm_context() if emitted else make_linalg_context(), text).parse_module()
        module.verify()
    except (ParseError, VerifyException, RecursionError) as error:
        raise ValueError("descriptor source/product has unsupported or malformed typed IR") from error
    require(sum(1 for _ in module.walk()) <= limits.operations, "descriptor operation roster exceeds its bound")
    return module


def pointer(typ):
    return type(typ) is llvm.LLVMPointerType and type(typ.addr_space) is builtin.NoneAttr


def original_types(source, limits):
    source.record(limits)
    raw = read(source.path, limits.source_bytes)
    require(hashlib.sha256(raw).hexdigest() == source.sha256, "original descriptor source bytes changed")
    module = parse(raw, limits, emitted=False)
    functions = [
        op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == source.entry_symbol
    ]
    require(len(functions) == 1 and len(functions[0].body.blocks) == 1, "original tensor entry is absent or ambiguous")
    function = functions[0]
    types = (*function.function_type.inputs, *function.function_type.outputs)
    slots = (*source.original_abi.inputs, *source.original_abi.outputs)
    require(len(types) == len(slots), "original source omits or changes ordered tensor slots")
    for typ, slot in zip(types, slots, strict=True):
        require(
            type(typ) is builtin.TensorType
            and type(typ.encoding) is builtin.NoneAttr
            and typ.get_shape() == slot.shape
            and type_identity(_dtype(typ)) == type_identity(slot.dtype),
            "original tensor shape/dtype/layout differs from the explicit complete ABI",
        )
    returned = function.body.block.last_op
    require(
        type(returned) is func.ReturnOp
        and tuple(arg.type for arg in returned.arguments) == tuple(function.function_type.outputs),
        "original tensor entry lacks its complete ordered result boundary",
    )
    require(
        tuple(arg.type for arg in function.body.block.args) == tuple(function.function_type.inputs),
        "original entry arguments differ from its declared tensor ABI",
    )
    return types


def fields(typ, rank, limits):
    require(
        type(typ) is llvm.LLVMStructType and not typ.struct_name.data,
        "descriptor must be one transparent unpacked literal aggregate",
    )
    parts = tuple(typ.types)
    require(
        len(parts) == (3 if rank == 0 else 5) and all(pointer(t) for t in parts[:2]),
        "descriptor pointer/offset/array roster differs from the selected ranked memref ABI",
    )
    index = parts[2]
    require(
        type(index) is builtin.IntegerType
        and index.signedness.data is builtin.Signedness.SIGNLESS
        and 0 < index.width.data <= limits.integer_bits,
        "descriptor index type is unsupported",
    )
    roles = [("allocated", (0,), parts[0]), ("aligned", (1,), parts[1]), ("offset", (2,), index)]
    if rank:
        for ordinal, label in ((3, "size"), (4, "stride")):
            array = parts[ordinal]
            require(
                type(array) is llvm.LLVMArrayType and array.size.data == rank and array.type == index,
                "descriptor arrays differ from original rank or index field type",
            )
            roles.extend((f"{label}_{axis}", (ordinal, axis), index) for axis in range(rank))
    return roles


def _plain_function(function):
    require(
        function.body.blocks
        and not function.function_type.is_variadic
        and type(function.function_type.output) is llvm.LLVMVoidType
        and function.CConv.convention.data == "ccc"
        and function.linkage.linkage.data == "external"
        and tuple(value.type for value in function.body.blocks[0].args) == tuple(function.function_type.inputs)
        and not any(a.data for a in function.arg_attrs or ())
        and not any(a.data for a in function.res_attrs or ()),
        "descriptor transport requires defined ordinary void C functions without argument attributes",
    )


def inspect_descriptor_wrapper(*, source, llvm_dialect, limits):
    """Conditional typed transport check; input strings carry no custody authority."""
    require(
        type(source) is OriginalDescriptorSource and type(limits) is DescriptorLimits,
        "descriptor transport needs exact original declarations and limits",
    )
    original = original_types(source, limits)
    module = parse(llvm_dialect, limits, emitted=True)
    functions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp)]
    entries = [f for f in functions if f.sym_name.data == source.c_interface_symbol]
    implementations = [f for f in functions if f.sym_name.data == source.entry_symbol]
    require(len(entries) == len(implementations) == 1, "descriptor functions are missing or ambiguous")
    wrapper, implementation = entries[0], implementations[0]
    _plain_function(wrapper)
    _plain_function(implementation)
    require(len(wrapper.body.blocks) == 1, "descriptor wrapper control flow is unsupported")
    require(
        not wrapper.attributes.keys() - {"llvm.emit_c_interface"}
        and set(wrapper.properties) <= {"CConv", "function_type", "linkage", "sym_name", "visibility_"}
        and wrapper.visibility_.value.data == 0,
        "descriptor wrapper has unsupported function semantics",
    )
    block = wrapper.body.block
    slots = (*source.original_abi.inputs, *source.original_abi.outputs)
    require(
        len(block.args) == len(slots) and all(pointer(arg.type) for arg in block.args),
        "descriptor wrapper changes the complete original argument order/arity",
    )
    loads, extracted, calls, returned = {}, {}, [], []
    for op in block.ops:
        require(
            not op.attributes and not op.regions and not op.successors,
            "descriptor wrapper carries unsupported metadata or control",
        )
        if isinstance(op, llvm.LoadOp):
            alignment = op.alignment.value.data if op.alignment is not None else None
            require(
                op.ptr in block.args
                and op.ptr not in loads
                and set(op.properties) <= {"ordering", "alignment"}
                and op.ordering.value.data == 0,
                "descriptor wrapper has an extra/changed/atomic aggregate load",
            )
            require(
                alignment is None or (alignment > 0 and not alignment & (alignment - 1)),
                "descriptor load has unsupported alignment",
            )
            loads[op.ptr] = op.results[0]
        elif type(op) is llvm.ExtractValueOp:
            require(
                set(op.properties) == {"position"} and op.container in loads.values(),
                "descriptor extraction is disconnected from an original aggregate load",
            )
            key = (op.container, tuple(op.position.iter_values()))
            require(key not in extracted, "descriptor wrapper repeats a field extraction")
            extracted[key] = op.results[0]
        elif type(op) is llvm.CallOp:
            require(
                not op.results
                and not op.op_bundle_operands
                and not tuple(op.op_bundle_sizes.iter_values())
                and op.CConv.convention.data == "ccc"
                and not op.fastmathFlags.data
                and op.TailCallKind.data.value == "none"
                and op.var_callee_type is None
                and op.callee is not None
                and op.callee.root_reference.data == source.entry_symbol
                and not op.callee.nested_references,
                "descriptor wrapper has an unsupported or changed implementation call",
            )
            require(
                set(op.properties)
                <= {"CConv", "TailCallKind", "callee", "fastmathFlags", "op_bundle_sizes", "operandSegmentSizes"},
                "descriptor call carries unsupported semantics",
            )
            calls.append(op)
        elif type(op) is llvm.ReturnOp:
            require(not op.operands and not op.properties, "descriptor wrapper has a changed return")
            returned.append(op)
        else:
            raise ValueError("unsupported descriptor wrapper operation: " + op.name)
    require(
        len(loads) == len(slots)
        and len(calls) == len(returned) == 1
        and block.last_op is returned[0]
        and calls[0].next_op is returned[0],
        "descriptor wrapper does not have one complete load/call/return boundary",
    )
    expected, rows = [], []
    for ordinal, (argument, tensor) in enumerate(zip(block.args, slots, strict=True)):
        loaded = loads[argument]
        selected = fields(loaded.type, len(tensor.shape), limits)
        bits = loaded.type.types.data[2].width.data
        components = ["ptr", "ptr", f"i{bits}"]
        if tensor.shape:
            components.extend([f"[{len(tensor.shape)} x i{bits}]"] * 2)
        element = original[ordinal].element_type
        scalar_types = {builtin.Float16Type: "half", builtin.Float32Type: "float", builtin.Float64Type: "double"}
        element_type = str(element) if type(element) is builtin.IntegerType else scalar_types.get(type(element))
        require(element_type is not None, "original element storage format is unsupported")
        row = {
            "ordinal": ordinal,
            "name": tensor.name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "index_bits": bits,
            "aggregate_type": "{ " + ", ".join(components) + " }",
            "element_type": element_type,
            "explicit_load_alignment": loaded.owner.alignment.value.data
            if loaded.owner.alignment is not None
            else None,
            "fields": [],
        }
        for role, path, typ in selected:
            value = extracted.get((loaded, path))
            require(value is not None and value.type == typ, "descriptor field extraction omits/changes original roles")
            expected.append(value)
            row["fields"].append({"role": role, "path": list(path), "type": "ptr" if pointer(typ) else str(typ)})
        rows.append(row)
    require(
        len(extracted) == len(expected)
        and tuple(calls[0].args) == tuple(expected)
        and tuple(value.type for value in expected) == tuple(implementation.function_type.inputs),
        "descriptor implementation call changes/reorders/omits ordered original fields",
    )
    return {
        "slots": rows,
        "original": source.record(limits),
        "scope": "complete ordered wrapper transport only; implementation storage/index/effects remain unproved",
    }


@dataclass(frozen=True, eq=False)
class DescriptorWrapperObservation:
    source: OriginalDescriptorSource
    limits: DescriptorLimits
    receipt: Path
    pins: tuple[tuple[str, str], ...]
    observation: bytes

    def verify(self):
        require(
            _ISSUED.get(self) == canonical_json(self.record()), "descriptor wrapper needs a live actual observation"
        )
        for path, digest in self.pins:
            maximum = self.limits.receipt_bytes if Path(path) == self.receipt else self.limits.source_bytes
            require(hashlib.sha256(read(path, maximum)).hexdigest() == digest, "descriptor source/product changed")
        actual = observe_descriptor_wrapper(source=self.source, limits=self.limits, llvm_product_receipt=self.receipt)
        require(actual.observation == self.observation, "descriptor transport observation changed")
        return actual.record()

    def record(self):
        import json

        return {
            "schema": "merlin.descriptor_wrapper_observation.v1",
            "observation": json.loads(self.observation),
            "pins": self.pins,
            "receipt": str(self.receipt),
        }


def observe_descriptor_wrapper(*, source, limits, llvm_product_receipt):
    """Reopen actual ordinary source→post-pass LLVM→translation custody."""
    from merlin.common.digest import canonical_json

    source.record(limits)
    receipt = Path(llvm_product_receipt).absolute()
    read(receipt, limits.receipt_bytes)
    product = verify_llvm_dialect_product(receipt)
    require(
        product["source"]["path"] == str(source.path) and product["source"]["sha256"] == source.sha256,
        "descriptor observation is not bound to the exact original source input",
    )
    emitted = read(product["llvm_dialect"]["path"], limits.source_bytes)
    observed = inspect_descriptor_wrapper(source=source, llvm_dialect=emitted, limits=limits)
    paths = (
        source.path,
        receipt,
        Path(product["llvm_dialect"]["path"]),
        Path(__file__),
        module_source_path("merlin.compiler.lower.host.descriptor"),
    )
    pins = tuple(
        (
            str(path),
            hashlib.sha256(read(path, limits.receipt_bytes if path == receipt else limits.source_bytes)).hexdigest(),
        )
        for path in paths
    )
    observation = DescriptorWrapperObservation(source, limits, receipt, pins, canonical_json(observed))
    _ISSUED[observation] = canonical_json(observation.record())
    return observation


# ---- host_descriptor_call ----------------------------------------------------------------------
# Ordinary host calls through actual observed ranked-descriptor storage.
#
# Only descriptor bytes are constructed here. No pointed-to tensor is read,
# converted or staged; the caller owns accessible buffers and their lifetimes.
# Transport observations do not qualify bodies, hardware or execution effects.

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
        from merlin.compiler.lower.native.recipe import verify_llvm_dialect_product

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


# ---- host_descriptor_compile -------------------------------------------------------------------
# Explicit original ranked-descriptor selection for ordinary host compilation.
#
# This joins an unchanged typed source and complete caller storage declarations
# to the actual translation, object and linked image. It establishes transport
# custody only; implementation indexing, numerical and physical effects remain
# unproved. No machine-width, storage or environment default is supplied.


@dataclass(frozen=True)
class HostDescriptorSelection:
    """Caller-owned original choices, not source or runtime admission authority."""

    source: OriginalDescriptorSource
    limits: DescriptorLimits
    environment: tuple[tuple[str, str], ...]
    query_timeout_s: float

    def verify(self):
        require(
            type(self.source) is OriginalDescriptorSource and type(self.limits) is DescriptorLimits,
            "host descriptor selection needs complete original source/storage and limits",
        )
        require(
            type(self.environment) is tuple
            and bool(self.environment)
            and all(
                type(row) is tuple and len(row) == 2 and all(type(part) is str for part in row)
                for row in self.environment
            )
            and len(dict(self.environment)) == len(self.environment),
            "host descriptor selection needs an exact explicit environment mapping",
        )
        require(
            type(self.query_timeout_s) in {int, float}
            and math.isfinite(self.query_timeout_s)
            and 0 < self.query_timeout_s <= 600,
            "host descriptor layout query needs an explicit bounded timeout",
        )
        require(self.source.slot_aliasing == "disjoint", "host descriptor execution requires explicit disjoint slots")
        original_types(self.source, self.limits)
        return {
            "source": self.source.record(self.limits),
            "limits": self.limits.record(),
            "environment": I.environment_identity(dict(self.environment)),
            "query_timeout_s": self.query_timeout_s,
        }

    def require_module(self, module):
        self.verify()
        original = parse(read(self.source.path, self.limits.source_bytes), self.limits, emitted=False)
        # Compare complete generic serialization, including all regions,
        # attributes, properties and ordered returns; names are not proofs.
        candidate = parse(text(module, generic=True).encode(), self.limits, emitted=False)
        require(
            text(original, generic=True) == text(candidate, generic=True),
            "host kernel differs from its original source",
        )


def _prepared(selection, product):
    source = selection.source
    original = parse(read(source.path, selection.limits.source_bytes), selection.limits, emitted=False)
    # Ordinary preprocessing adds only this explicit public calling marker in
    # the supported route. Other preprocessing changes require a separate proof.
    for op in original.body.block.ops:
        if type(op) is func.FuncOp and op.sym_name.data == source.entry_symbol:
            marker = op.attributes.get("llvm.emit_c_interface")
            require(marker is None or type(marker) is builtin.UnitAttr, "original C-interface marker is unsupported")
            op.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    prepared_path = Path(product["source"]["path"])
    raw = read(prepared_path, selection.limits.source_bytes)
    require(hashlib.sha256(raw).hexdigest() == product["source"]["sha256"], "prepared source bytes changed")
    prepared = parse(raw, selection.limits, emitted=False)
    require(
        text(original, generic=True) == text(prepared, generic=True),
        "host source preprocessing changed the original body/ordered ABI; correspondence unavailable",
    )
    return replace(source, path=prepared_path, sha256=product["source"]["sha256"])


def _pin(path, limits):
    path = Path(path).absolute()
    raw = read(path, max(limits.source_bytes, limits.receipt_bytes, limits.object_bytes))
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def join_host_image(*, selection, build_root, product, image):
    """Require actual original LLVM→object and runtime-object→image records."""
    selection.verify()
    environment = dict(selection.environment)
    records = []
    for path in sorted(Path(build_root).rglob("invocation.json")):
        # A query/call record cannot substitute for an original build boundary.
        row = loads(read(path, selection.limits.receipt_bytes))
        if row.get("stage") in {"object", "runtime_object", "link"}:
            records.append((path, I.require_environment(path, environment=environment)))
    objects = [(path, row) for path, row in records if row["stage"] == "object"]
    runtime = [(path, row) for path, row in records if row["stage"] == "runtime_object"]
    require(
        len(objects) == len(runtime) == 1 and len(records) == 3,
        "host descriptor image lacks its exact build boundaries",
    )
    object_path, object_row = objects[0]
    raw_translation = product["translated_llvm_ir"]
    require(
        len(object_row["inputs"]) == len(object_row["outputs"]) == 1
        and object_row["inputs"][0]["sha256"] == raw_translation["sha256"]
        and _pin(raw_translation["path"], selection.limits)["sha256"] == raw_translation["sha256"]
        and read(object_row["inputs"][0]["path"], selection.limits.source_bytes)
        == read(raw_translation["path"], selection.limits.source_bytes),
        "host descriptor object input differs from the complete retained translation",
    )
    runtime_path, runtime_row = runtime[0]
    require(
        len(runtime_row["inputs"]) == len(runtime_row["outputs"]) == 1,
        "host descriptor image lacks its complete runtime-source/object binding",
    )
    expected = sorted((object_row["outputs"][0], runtime_row["outputs"][0]), key=lambda row: row["path"])
    linked = [(path, row) for path, row in records if row["stage"] == "link"]
    require(
        len(linked) == 1
        and linked[0][1]["inputs"] == expected
        and linked[0][1]["outputs"] == [_pin(image, selection.limits)],
        "host descriptor layout object is disconnected from the loaded image",
    )
    require(
        all(row["kind"] == "subprocess" for _, row in records),
        "host descriptor build requires actual subprocess products",
    )
    return object_path, (object_path, runtime_path, linked[0][0])


def observe_host_descriptor_transport(*, selection, result):
    """Produce a live transport for the ordinary caller's own compiled image."""
    from merlin.compiler.lower.native.recipe import verify_llvm_dialect_product

    require(type(selection) is HostDescriptorSelection, "host descriptor selection is unsupported")
    selection.verify()
    root, image = Path(result.workdir).absolute(), Path(result.host_so).absolute()
    require(image.parent == root, "host descriptor image is outside its ordinary compilation owner")
    receipt = Path(result.stats["llvm_dialect_product"]["path"])
    read(receipt, selection.limits.receipt_bytes)
    product = verify_llvm_dialect_product(receipt)
    prepared = _prepared(selection, product)
    object_record, records = join_host_image(selection=selection, build_root=root, product=product, image=image)
    wrapper = observe_descriptor_wrapper(source=prepared, limits=selection.limits, llvm_product_receipt=receipt)
    observation = observe_descriptor_layout(
        wrapper=wrapper,
        object_record=object_record,
        environment=dict(selection.environment),
        output_root=root / "host-descriptor-layout",
        timeout_s=selection.query_timeout_s,
    )
    paths = (selection.source.path, prepared.path, receipt, image, Path(__file__))
    pins = tuple((row["path"], row["sha256"]) for row in (_pin(path, selection.limits) for path in paths))
    selected_records = tuple((row["path"], row["sha256"]) for row in (_pin(path, selection.limits) for path in records))
    transport = HostDescriptorTransport(
        selection, observation, root, image, pins, selected_records, canonical_json(selection.verify())
    )
    _retain_transport(transport)
    transport.verify()
    return transport
