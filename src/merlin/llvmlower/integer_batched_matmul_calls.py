"""Ordinary source-to-host composition for independent integer batch products.

An explicit backend supplies the same synchronous matrix primitive used for a
single product. The shared bridge owns ranked views and counted head loops;
the ordinary host keeps every surrounding readout and numerical plane sum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects import builtin, func, tensor
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Region
from xdsl.rewriter import Rewriter

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.head_batching import plan_independent_head_batching, read_batched_integer_product

from .borrowed_buffer_calls import from_writers
from .borrowed_effects import BorrowedBufferEffects
from .fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from .logical_borrowed_abi import LogicalBorrowedABI, _symbol
from .logical_borrowed_batch_abi import emit_borrowed_batch_translation_unit


@dataclass(frozen=True)
class IntegerBatchedMatmulCall:
    operation_ordinal: int
    problem: object
    symbol: str
    borrowed_symbol: str
    kernel_symbol: str

    @property
    def m(self):
        return self.problem.m

    @property
    def k(self):
        return self.problem.k

    @property
    def n(self):
        return self.problem.n

    @property
    def shapes(self):
        return self.problem.lhs.shape, self.problem.rhs.shape, self.problem.output_shape


@dataclass(frozen=True)
class PreparedBatchedMatmulCalls:
    source: str
    original_sha256: str
    calls: tuple
    writers: tuple
    batching: object = None

    def borrowed_calls(self, path, *, pins=()):
        path = Path(path)

        def validate():
            if path.read_text() != self.source:
                raise ValueError("batched source changed before ordinary host lowering")
            if self.batching is not None:
                self.batching.revalidate()

        validate()
        return from_writers(path, self.writers, pins, validate)


def prepare_integer_batched_matmul_calls(
    source, *, select, borrowed_effects, entry="forward", symbol_prefix="integer_batch",
    allocation_alignment=64, refold_heads=False,
):
    """Outline actual zero-seeded signed batches; unsupported source stays intact.

    ``refold_heads`` explicitly applies the checked extract/reshape source
    transform first. Selection sees only typed current geometry. No target,
    numerical relaxation, implicit primitive or performance winner is supplied.
    """
    if (not callable(select) or type(borrowed_effects) is not BorrowedBufferEffects
            or type(refold_heads) is not bool or type(allocation_alignment) is not int
            or not 0 < allocation_alignment < 1 << 63 or allocation_alignment & (allocation_alignment-1)):
        raise ValueError("explicit selection, borrowed effects and bounded alignment required")
    borrowed_effects.__post_init__()
    _symbol(symbol_prefix)
    module = parse_mlir_text(source)
    module.verify()
    entries = [op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == entry]
    if len(entries) != 1 or len(entries[0].body.blocks) != 1:
        raise ValueError("one original single-block tensor entry required")
    batching = plan_independent_head_batching(module) if refold_heads else None
    if batching is not None:
        module = batching.candidate.clone()
    reserved = {op.sym_name.data for op in module.body.block.ops if type(op) is func.FuncOp}
    plans = []
    from merlin.compile.analysis.contraction_shapes import zero_initialised

    for ordinal, op in enumerate(tuple(module.walk())):
        if type(op) is not GenericOp:
            continue
        try:
            problem = read_batched_integer_product(op)
        except (ValueError, AttributeError):
            continue
        if not zero_initialised(op):
            continue
        index = len(plans)
        while True:
            symbol = f"{symbol_prefix}_{index}"
            names = symbol, symbol+"_borrowed", symbol+"_kernel"
            if not reserved.intersection(names):
                break
            index += 1
        call = IntegerBatchedMatmulCall(ordinal, problem, *names)
        chosen = select(call)
        if type(chosen) is not bool:
            raise ValueError("batch selection must return an explicit Boolean")
        if chosen:
            plans.append((op, call))
            reserved.update(names)
    writers = []
    for op, binding in plans:
        lhs, rhs = op.inputs
        typ = op.results[0].type
        empty = tensor.EmptyOp((), typ)
        call = func.CallOp(binding.symbol, [lhs, rhs, empty.tensor], [typ])
        call.attributes.update(op.attributes)
        Rewriter.replace_op(op, [empty, call], call.results)
        attrs = [builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("read"),
                  "bufferization.buffer_layout": builtin.StridedLayoutAttr([None]*len(v.type.get_shape()), None)})
                 for v in (lhs, rhs)]
        attrs.append(builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("write")}))
        declaration = func.FuncOp(binding.symbol, ([lhs.type, rhs.type, typ], [typ]), Region(),
                                  visibility="private", arg_attrs=builtin.ArrayAttr(attrs))
        declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
        module.body.block.add_op(declaration)
        writers.append(FreshTensorWriterContract(binding.symbol, 2, (2,), binding.borrowed_symbol,
                                                  allocation_alignment, borrowed_effects=borrowed_effects))
    if not plans:
        return PreparedBatchedMatmulCalls(source, hashlib.sha256(source.encode()).hexdigest(), (), ())
    rewrite_fresh_tensor_writers(module, writers)
    module.verify()
    return PreparedBatchedMatmulCalls(ir_text(module), hashlib.sha256(source.encode()).hexdigest(),
                                     tuple(call for _, call in plans), tuple(writers), batching)


@dataclass(frozen=True)
class ComposedBatchedMatmulProgram:
    preparation: PreparedBatchedMatmulCalls
    host: object
    kernels: tuple
    bridges: tuple


def lower_integer_batched_matmul_program(
    source, workdir, *, select, emit_matrix_kernel, logical_abi, data_layout,
    entry="forward", allocation_alignment=64, refold_heads=False, host_features=None,
):
    """Execute the existing ordinary host pipeline around explicit batch calls.

    ``emit_matrix_kernel(call)`` returns LLVM text plus immutable implementation
    pins. It implements ONE original (M,K,N) matrix, not a guessed flattened
    batch. The shared ranked bridge invokes it for every source head. Build,
    linking and complete numerical qualification remain ordinary caller steps.
    """
    from .lower import lower_model
    from .tensor_helper_inlining import composition_features

    if type(logical_abi) is not LogicalBorrowedABI or not callable(emit_matrix_kernel):
        raise ValueError("explicit selected logical ABI and matrix implementation required")
    logical_abi.__post_init__()
    prepared = prepare_integer_batched_matmul_calls(
        source, select=select, borrowed_effects=logical_abi.qualified_effects, entry=entry,
        allocation_alignment=allocation_alignment, refold_heads=refold_heads,
    )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prepared.mlir"
    path.write_text(prepared.source)
    kernels, bridges, pins = [], [], []
    for index, call in enumerate(prepared.calls):
        llvm, implementation = emit_matrix_kernel(call)
        if not isinstance(llvm, str) or type(implementation) is not tuple:
            raise ValueError("matrix emitter needs LLVM and explicit immutable implementation pins")
        for location, digest in implementation:
            location = Path(location)
            if hashlib.sha256(location.read_bytes()).hexdigest() != digest:
                raise ValueError("matrix implementation changed during batch emission")
            pins.append((location, digest))
        kernel, bridge = workdir/f"kernel{index}.ll", workdir/f"bridge{index}.c"
        kernel.write_text(llvm)
        bridge.write_text(emit_borrowed_batch_translation_unit(
            logical_abi, call.problem, symbol="_mlir_ciface_"+call.borrowed_symbol,
            kernel=call.kernel_symbol,
        ).text)
        kernels.append(kernel)
        bridges.append(bridge)
    borrowed = prepared.borrowed_calls(path, pins=tuple(dict.fromkeys(pins))) if prepared.calls else None
    host = lower_model(prepared.source, workdir/"host", targets=(), index_bits=64, data_layout=data_layout,
                       borrowed_buffer_calls=borrowed, hoist_static_allocs=False,
                       features=composition_features(host_features, has_writers=bool(prepared.calls)))
    return ComposedBatchedMatmulProgram(prepared, host, tuple(kernels), tuple(bridges))
