"""Compose selected integer products with the ordinary shared host pipeline.

The shared recognizer checks the original scalar recurrence and zero seed.
The caller explicitly selects implementations and supplies synchronous borrowed
effects. Instruction semantics, physical resources and implementation correctness
remain backend obligations. All surrounding source computation stays in Merlin.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, TensorType, UnitAttr
from xdsl.ir import Region
from xdsl.rewriter import Rewriter

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import recognize_signed_integer_matmul

from .borrowed_buffer_calls import BorrowedBufferEffects, from_writers
from .fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers


@dataclass(frozen=True)
class IntegerMatmulCall:
    operation_ordinal: int
    m: int
    k: int
    n: int
    symbol: str
    borrowed_symbol: str
    kernel_symbol: str
    output_dtype: str = "i32"
    readout_identity: str | None = None

    def __post_init__(self):
        if (
            type(self.operation_ordinal) is not int or self.operation_ordinal < 0
            or any(type(d) is not int or not 0 < d < 1 << 63 for d in (self.m, self.k, self.n))
            or max(self.m * self.k, self.k * self.n, 4 * self.m * self.n) >= 1 << 63
            or any(
                type(name) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None
                for name in (self.symbol, self.borrowed_symbol, self.kernel_symbol)
            )
            or len({self.symbol, self.borrowed_symbol, self.kernel_symbol}) != 3
            or self.output_dtype not in ("i32", "i8")
            or (self.output_dtype == "i8") != (self.readout_identity is not None)
            or self.readout_identity is not None and (
                type(self.readout_identity) is not str or re.fullmatch(r"[0-9a-f]{64}", self.readout_identity) is None
            )
        ):
            raise ValueError("bounded integer product geometry and distinct portable symbols required")

    @property
    def shapes(self):
        return ((self.m, self.k), (self.k, self.n), (self.m, self.n))


@dataclass(frozen=True)
class PreparedIntegerMatmulCalls:
    source: str
    original_sha256: str
    calls: tuple[IntegerMatmulCall, ...]
    writers: tuple[FreshTensorWriterContract, ...]
    readout_proofs: tuple = ()

    def borrowed_calls(self, path, *, pins=(), validate_original=lambda: None):
        """Bind the actual prepared file into the existing lifetime checker."""
        path = Path(path)
        if path.read_text() != self.source:
            raise ValueError("integer product preparation changed before lowering")
        return from_writers(path, self.writers, pins, validate_original)


@dataclass(frozen=True)
class ComposedIntegerMatmulProgram:
    """Actual shared-host product and separate backend/ABI compilation inputs."""

    preparation: PreparedIntegerMatmulCalls
    host: object
    kernels: tuple[Path, ...]
    bridges: tuple[Path, ...]
    convolution_preparation: object = None


def lower_integer_matmul_program(
    source,
    workdir,
    *,
    select,
    emit_kernel,
    logical_abi,
    data_layout,
    entry="forward",
    allocation_alignment=64,
    output_readout=None,
    convolution_packing_bytes=None,
    host_features=None,
):
    """Run ordinary Merlin lowering around explicitly selected backend kernels.

    ``emit_kernel(call)`` supplies raw LLVM and immutable ``(Path, sha256)``
    implementation pins. The existing descriptor bridge owns bounded packing
    fallback, and the ordinary lifetime checker consumes supplied borrowed
    effects. The returned files still need ordinary object production, linking
    and complete numerical checks. Closed writers default to tensor-helper
    inlining before bufferization. ``host_features`` supplies an explicit
    alternative; an empty set disables that default. No target is discovered here.
    """
    from .logical_borrowed_abi import LogicalBorrowedABI, dense_buffers, emit_borrowed_dense_translation_unit
    from .lower import lower_model
    from .tensor_helper_inlining import composition_features

    if type(logical_abi) is not LogicalBorrowedABI or not callable(emit_kernel):
        raise ValueError("explicit logical ABI and backend emitter required")
    logical_abi.__post_init__()
    if output_readout is not None and not logical_abi.readout_format_qualification_identity:
        raise ValueError("selected narrow readout requires explicit original format/effect premises")
    packing = None
    if convolution_packing_bytes is not None:
        from .convolution_routes import prepare_integer_convolution_panels

        source, packing = prepare_integer_convolution_panels(source, max_packing_bytes=convolution_packing_bytes)
    prepared = prepare_integer_matmul_calls(
        source, select=select, borrowed_effects=logical_abi.qualified_effects,
        entry=entry, allocation_alignment=allocation_alignment,
        output_readout=output_readout,
    )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prepared.mlir"
    path.write_text(prepared.source)
    kernels, bridges, pins = [], [], []
    for index, call in enumerate(prepared.calls):
        kernel, implementation = emit_kernel(call)
        if not isinstance(kernel, str) or type(implementation) is not tuple:
            raise ValueError("backend emitter must return LLVM text and immutable implementation pins")
        for location, digest in implementation:
            location = Path(location)
            if hashlib.sha256(location.read_bytes()).hexdigest() != digest:
                raise ValueError("backend implementation changed during emission")
            pins.append((location, digest))
        kernel_path, bridge_path = workdir / f"kernel{index}.ll", workdir / f"bridge{index}.c"
        kernel_path.write_text(kernel)
        bridge = emit_borrowed_dense_translation_unit(
            logical_abi, dense_buffers(call, ("i8", "i8", "i32"), output_dtype=call.output_dtype),
            symbol="_mlir_ciface_" + call.borrowed_symbol, kernel=call.kernel_symbol,
        )
        bridge_path.write_text(bridge.text)
        kernels.append(kernel_path)
        bridges.append(bridge_path)
    borrowed = prepared.borrowed_calls(path, pins=tuple(dict.fromkeys(pins))) if prepared.calls else None
    host = lower_model(
        prepared.source, workdir / "host", targets=(), index_bits=64, data_layout=data_layout,
        borrowed_buffer_calls=borrowed,
        hoist_static_allocs=False,
        features=composition_features(host_features, has_writers=bool(prepared.calls)),
    )
    from .adjacent_exp import runtime_c_for as exp_runtime_c_for
    from .guarded_zero_contraction import runtime_c_for

    selected_runtime = runtime_c_for(host.ll_path.read_text()) + exp_runtime_c_for(host.ll_path.read_text())
    if selected_runtime:
        if not bridges:
            raise ValueError(
                "guarded scalar runtime needs a linked bridge; use build_host_shared for a pure host model"
            )
        bridges[0].write_text(bridges[0].read_text() + "\n" + selected_runtime)
    return ComposedIntegerMatmulProgram(prepared, host, tuple(kernels), tuple(bridges), packing)


def _zero(value):
    from .integer_epilogue import _constant

    constant = _constant(value)
    return constant is not None and constant.value.data == 0


def prepare_integer_matmul_calls(
    source: str,
    *,
    select: Callable[[IntegerMatmulCall], bool],
    borrowed_effects: BorrowedBufferEffects,
    entry="forward",
    symbol_prefix="__merlin_integer_dot",
    allocation_alignment=64,
    output_readout=None,
):
    """Outline selected zero-seeded signed i8 x i8 -> i32 rank-two products.

    Unsupported products keep their original source. An explicitly declared
    readout may absorb a proved sole-use epilogue. Other observers, clamping,
    casts and layout operations retain their ordinary source route.
    Named/encoded/batched or nonzero
    recurrences need separate selection support. Selection sees source-derived
    geometry and the proposed output format, without input or golden values.
    """
    if not callable(select) or type(borrowed_effects) is not BorrowedBufferEffects:
        raise ValueError("explicit implementation selection and borrowed effects required")
    borrowed_effects.__post_init__()
    if not isinstance(symbol_prefix, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol_prefix) is None:
        raise ValueError("portable explicit symbol prefix required")
    if (
        type(allocation_alignment) is not int
        or not 0 < allocation_alignment < 1 << 63
        or allocation_alignment & (allocation_alignment - 1)
    ):
        raise ValueError("positive power-of-two allocation alignment required")
    module = parse_mlir_text(source)
    module.verify()
    entries = [o for o in module.body.block.ops if isinstance(o, func.FuncOp) and o.sym_name.data == entry]
    if len(entries) != 1 or len(entries[0].body.blocks) != 1:
        raise ValueError("one single-block selected tensor entry required")
    function = entries[0]
    if any(not isinstance(t, TensorType) for t in (*function.function_type.inputs, *function.function_type.outputs)):
        raise ValueError("selected entry must use tensor arguments and results")
    reserved = {
        attr.data
        for o in module.body.block.ops
        if isinstance(attr := o.properties.get("sym_name", o.attributes.get("sym_name")), StringAttr)
    }
    plans = []
    for ordinal, operation in enumerate(tuple(function.walk())):
        if operation.name != "linalg.generic":
            continue
        detached = operation.clone()
        for nested in detached.walk():
            for key in tuple(nested.attributes):
                if key.startswith("prov."):
                    del nested.attributes[key]
        try:
            recognize_signed_integer_matmul(detached, ("i8", "i8", "i32"))
        except ValueError:
            continue
        finally:
            detached.drop_all_references()
        lhs, rhs, initial = operation.operands
        if not _zero(initial):
            continue
        shapes = tuple(v.type.get_shape() for v in (lhs, rhs, initial))
        if any(len(s) != 2 or any(d <= 0 for d in s) for s in shapes):
            continue
        (m, k), (rk, n), output = shapes
        if rk != k or output != (m, n) or max(m * k, k * n, 4 * m * n) >= 1 << 63:
            continue
        index = len(plans)
        while True:
            symbol = f"{symbol_prefix}_{index}"
            names = (symbol, symbol + "_borrowed", symbol + "_kernel")
            if not reserved.intersection(names):
                break
            index += 1
        call = IntegerMatmulCall(ordinal, m, k, n, *names)
        selected = select(call)
        if type(selected) is not bool:
            raise ValueError("implementation selection must return a Boolean")
        if selected:
            reserved.update(names)
            epilogue = None
            if output_readout is not None:
                from .integer_epilogue import match_integer_epilogue

                proposed = match_integer_epilogue(operation, output_readout)
                if proposed is not None:
                    narrow = replace(call, output_dtype="i8", readout_identity=output_readout.identity)
                    admitted = select(narrow)
                    if type(admitted) is not bool:
                        raise ValueError("implementation selection must return a Boolean")
                    if admitted:
                        call, epilogue = narrow, proposed
            plans.append((operation, call, epilogue))
    writers, proofs = [], []
    for operation, binding, epilogue in plans:
        lhs, rhs = operation.operands[:2]
        typ = (epilogue.operations[-1] if epilogue is not None else operation).results[0].type
        empty = tensor.EmptyOp((), typ)
        call = func.CallOp(binding.symbol, [lhs, rhs, empty.results[0]], [typ])
        call.attributes.update(operation.attributes)
        if epilogue is None:
            Rewriter.replace_op(operation, [empty, call], call.results)
        else:
            from xdsl.rewriter import InsertPoint

            Rewriter.insert_op([empty, call], InsertPoint.before(operation))
            epilogue.operations[-1].results[0].replace_all_uses_with(call.results[0])
            for stage in reversed(epilogue.operations):
                Rewriter.erase_op(stage)
            Rewriter.erase_op(operation)
            proofs.append(epilogue.correspondence)
        declaration = func.FuncOp(
            binding.symbol,
            ([lhs.type, rhs.type, typ], [typ]),
            Region(),
            visibility="private",
            arg_attrs=ArrayAttr([
                DictionaryAttr({"bufferization.access": StringAttr(access)})
                for access in ("read", "read", "write")
            ]),
        )
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        module.body.block.add_op(declaration)
        writers.append(FreshTensorWriterContract(
            binding.symbol, 2, (2,), binding.borrowed_symbol, allocation_alignment,
            borrowed_effects=borrowed_effects,
        ))
    if not plans:
        return PreparedIntegerMatmulCalls(source, hashlib.sha256(source.encode()).hexdigest(), (), ())
    rewrite_fresh_tensor_writers(module, writers)
    module.verify()
    return PreparedIntegerMatmulCalls(
        ir_text(module), hashlib.sha256(source.encode()).hexdigest(),
        tuple(binding for _, binding, _ in plans), tuple(writers), tuple(proofs),
    )
