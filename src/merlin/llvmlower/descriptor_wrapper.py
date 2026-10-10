"""Bounded ordered ranked-memref C-interface transport observations.

Only the complete transparent load/extract/call/return wrapper is checked.
Implementation memory/index/numerical semantics, original owner provenance,
physical storage and native machine equivalence remain separate obligations.
"""

import hashlib
import weakref
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects import builtin, func, llvm
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.common.jsonio import canonical_json
from merlin.targetgen.contract.linalg_iface import _dtype, make_linalg_context
from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
from merlin.targetgen.contract.tensor_types import type_identity
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

from .descriptor_contract import DescriptorLimits, OriginalDescriptorSource
from .llvm_dialect_product import verify_llvm_dialect_product

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
    from merlin.common.jsonio import canonical_json

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
        Path(__file__).with_name("descriptor_contract.py"),
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
