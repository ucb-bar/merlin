"""Source-native host loops over memory-backed tensor values.

Scalar arithmetic reuses HostLinalg. Shapes and indexing maps determine loops;
no host tensor is expanded into a Python list of emitted scalar instructions.
View operations compose indexing functions without materializing copies.
"""
from math import prod

from xdsl.dialects import llvm
from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, FloatAttr, Float32Type,
                                   Float64Type, IndexType, IntegerAttr, IntegerType,
                                   TensorType, i1, i8, i64)
from xdsl.ir import Block, Operation, SSAValue
from xdsl.ir.affine import AffineBinaryOpExpr, AffineConstantExpr, AffineDimExpr, AffineBinaryOpKind
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def

from .builder import INT_TYPES
from .host_linalg import HostLinalg, _strides, attr_of, elem_name, is_int, row_pitch, tensor_shape
from ..lowering.plan import LoweringDeclined


@irdl_op_definition
class FPTruncOp(IRDLOperation):
    """LLVM's ordinary narrowing floating-point cast.

    The xDSL version pinned by this artifact exposes ``llvm.fpext`` but omits the
    symmetric operation class.  Defining its structural spelling here keeps f64->f32
    conversion in LLVM IR without routing through integer bits or changing rounding.
    """

    name = "llvm.fptrunc"
    value = operand_def()
    result = result_def()

    def __init__(self, value, result_type):
        super().__init__(operands=[value], result_types=[result_type])


class LoopTensor:
    def __init__(self, shape, ety, read, write=None):
        self.shape, self.ety, self.read, self.write = tuple(shape), ety, read, write

    def at(self, index):
        return self.read(tuple(index))


class LoopHostLinalg(HostLinalg):
    """Loop lowering with bounded scalar stack and reusable tensor workspace.

    Tensor storage lives for one host segment.  Values that cross such a boundary are explicit ABI
    spills supplied again by :meth:`run_segment`; therefore every segment may reuse workspace offset
    zero without aliasing a live tensor.  Rank-zero/one-element temporaries remain ordinary bounded
    stack allocations so scalar recurrence retains its existing representation.
    """

    _ALIGNMENT = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace_pointer = None
        self._workspace_cursor = 0
        self._workspace_peak = 0
        self._workspace_requested = 0
        self._stack_frame_upper_bound = 0
        self._segment_number = -1
        self._segment_peaks: list[int] = []
        self._storage_allocations: list[dict] = []
        self._storage_owner = None
        self._source_index_by_op = {}

    @staticmethod
    def _align(value: int) -> int:
        alignment = LoopHostLinalg._ALIGNMENT
        return -(-int(value) // alignment) * alignment

    def _begin_storage_segment(self, operations, source_op_indices=()):
        self._segment_number += 1
        self._workspace_cursor = 0
        self._segment_peaks.append(0)
        self._source_index_by_op = {}
        for operation, source_index in zip(operations, source_op_indices):
            for nested in operation.walk():
                self._source_index_by_op[nested] = int(source_index)

    def storage_receipt(self) -> dict:
        """Cheap storage proof produced while lowering, with no target execution."""
        largest = sorted(
            self._storage_allocations,
            key=lambda row: (-row["bytes"], row["segment"], row["operation"]))[:16]
        return {
            "schema": "host_tensor_storage_v1",
            "lifetime_unit": "ordered_host_segment",
            "workspace_global_bytes": self._align(self._workspace_peak),
            "workspace_requested_bytes": self._workspace_requested,
            "workspace_reuse_bytes": max(
                0, self._workspace_requested - self._align(self._workspace_peak)),
            "bounded_stack_frame_upper_bound_bytes": self._stack_frame_upper_bound,
            "segment_workspace_peaks": list(self._segment_peaks),
            "allocation_count": len(self._storage_allocations),
            "largest_owners": largest,
        }

    def _scalar_results(self, block, args):
        """Lower a scalar region and preserve every value of a variadic yield."""
        if len(block.args) != len(args):
            raise LoweringDeclined(
                "scalar region argument count does not match its operands",
                op="host_lane")
        env = dict(zip(block.args, args))

        def read(value):
            hit = env.get(value)
            return self.get(value) if hit is None else hit

        for operation in block.ops:
            if operation.name == "linalg.yield":
                return [read(value) for value in operation.operands]
            if len(operation.results) != 1:
                raise LoweringDeclined(
                    "scalar region operation has a variadic result", op=operation.name)
            env[operation.results[0]] = self._scalar_op(
                operation, [read(value) for value in operation.operands])
        raise LoweringDeclined("a linalg region did not yield a value", op="host_lane")

    def _scalar_block(self, block, args):
        results = self._scalar_results(block, args)
        if len(results) != 1:
            raise LoweringDeclined(
                "single-result scalar context received a variadic linalg.yield",
                op="host_lane")
        return results[0]

    def _const(self, op):
        """Materialize source constants at their declared computation precision."""
        if op.results and isinstance(op.results[0].type, Float64Type):
            value = attr_of(op, "value")
            if not isinstance(value, FloatAttr):
                raise LoweringDeclined(
                    f"the CPU lane cannot materialise the constant {value}",
                    op="arith.constant")
            return self.fb.prologue(llvm.ConstantOp(
                FloatAttr(float(value.value.data), Float64Type()),
                Float64Type())).results[0]
        return super()._const(op)

    def _structured_block(self, block, args):
        """Lower one structured-control block and return its yielded values.

        SCF regions use ordinary SSA block arguments and may contain the same tensor and
        scalar operations as the enclosing host segment.  Keep one structural dispatcher
        for them instead of duplicating the scalar/tensor operation tables.
        """
        if len(block.args) != len(args):
            raise LoweringDeclined(
                "structured-control block argument count does not match its operands",
                op="host_lane")
        for argument, value in zip(block.args, args):
            self.vals[argument] = value
        for operation in block.ops:
            if operation.name == "scf.yield":
                return [self.get(value) for value in operation.operands]
            self._tensor_op(operation)
        raise LoweringDeclined(
            "structured-control region has no scf.yield terminator", op="host_lane")

    def _condition(self, value):
        """Convert the host integer representation of an i1 value back to LLVM i1."""
        if not isinstance(value, SSAValue):
            raise LoweringDeclined(
                "structured-control condition is not a scalar SSA value", op="host_lane")
        if value.type == i1:
            return value
        if value.type != i64:
            raise LoweringDeclined(
                "structured-control condition is not represented as i1/i64", op="host_lane")
        return self.fb.add(llvm.TruncOp(value, i1)).results[0]

    def _for_range_values(self, count, body, initial):
        """Emit a counted CFG loop carrying any number of scalar SSA values.

        ``FnBuilder.for_range`` intentionally covers the common zero/one recurrence case.
        Source SCF can carry several scalar iter_args, so this local extension preserves the
        same CFG shape while leaving tensors in explicitly materialized loop storage.
        """
        if count <= 0:
            return list(initial)
        scalar_types = [value.type for value in initial]
        header = Block(arg_types=[i64, *scalar_types])
        loop = Block(arg_types=scalar_types)
        after = Block(arg_types=scalar_types)
        for block in (header, loop, after):
            self.fb.region.add_block(block)
        self.fb.add(llvm.BrOp(header, self.fb.const(0), *initial))
        self.fb.blk = header
        condition = self.fb.add(llvm.ICmpOp(
            header.args[0], self.fb.const(count), IntegerAttr(2, i64))).results[0]
        self.fb.add(llvm.CondBrOp(
            condition, loop, header.args[1:], after, header.args[1:]))
        self.fb.blk = loop
        results = list(body(header.args[0], list(loop.args)))
        if (len(results) != len(initial)
                or any(not isinstance(value, SSAValue) or value.type != expected.type
                       for value, expected in zip(results, initial))):
            raise LoweringDeclined(
                "scf.for scalar iter_args changed count or representation type",
                op="scf.for")
        self.fb.add(llvm.BrOp(
            header, self.fb.add_i(header.args[0], self.fb.const(1)), *results))
        self.fb.blk = after
        return list(after.args)

    def _t_scf_for(self, op):
        """Lower a statically bounded SCF loop with scalar and tensor iter_args.

        Tensor SSA recurrence is represented by private mutable storage allocated before
        the CFG loop.  This is equivalent for the admitted functional-update form: each
        yielded tensor must be that same loop-private value.  Scalar recurrences travel as
        LLVM block arguments.  Dynamic rank-one ``tensor.empty`` storage receives the loop
        trip count as a safe capacity bound; an iteration can insert at most one new element.
        """
        if len(op.operands) < 3 or len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            raise LoweringDeclined("scf.for requires one single-block body", op=op.name)
        lower, upper, step = [self.static_int(value) for value in op.operands[:3]]
        if lower is None or upper is None or step is None or step == 0:
            raise LoweringDeclined(
                "scf.for requires compile-time integer bounds and a nonzero step",
                op=op.name)
        trip_count = len(range(lower, upper, step))
        source_initials = list(op.operands[3:])
        if len(source_initials) != len(op.results):
            raise LoweringDeclined("scf.for iter_args/results disagree", op=op.name)

        carried = []
        mutable_tensors = set()
        for source, initial in zip(source_initials, [self.get(v) for v in source_initials]):
            if not isinstance(initial, LoopTensor):
                carried.append(initial)
                continue
            shape = initial.shape
            dynamic = [axis for axis, extent in enumerate(shape) if extent < 0]
            owner = source.owner if isinstance(source.owner, Operation) else None
            is_empty = owner is not None and owner.name == "tensor.empty"
            if dynamic:
                if not is_empty or len(shape) != 1 or dynamic != [0]:
                    raise LoweringDeclined(
                        "scf.for can bound only a rank-one dynamic tensor.empty iter_arg",
                        op=op.name)
                materialized = self.storage((trip_count,), initial.ety)
            elif is_empty:
                materialized = self.storage(shape, initial.ety)
            else:
                materialized = self.materialize(initial)
            carried.append(materialized)
            mutable_tensors.add(materialized)

        body = op.regions[0].blocks[0]
        tensor_positions = [index for index, value in enumerate(carried)
                            if isinstance(value, LoopTensor)]
        scalar_positions = [index for index, value in enumerate(carried)
                            if not isinstance(value, LoopTensor)]
        scalar_initial = [carried[index] for index in scalar_positions]
        prior_mutable = getattr(self, "_scf_mutable_tensors", set())
        self._scf_mutable_tensors = prior_mutable | mutable_tensors

        def loop_body(counter, scalar_values):
            induction = self.fb.add_i(
                self.fb.const(lower), self.fb.mul_i(counter, self.fb.const(step)))
            current = list(carried)
            for index, value in zip(scalar_positions, scalar_values):
                current[index] = value
            yielded = self._structured_block(body, [induction, *current])
            if len(yielded) != len(current):
                raise LoweringDeclined(
                    "scf.for body yield count disagrees with iter_args", op=op.name)
            for index in tensor_positions:
                if yielded[index] is not current[index]:
                    raise LoweringDeclined(
                        "scf.for tensor recurrence is not an in-place functional update",
                        op=op.name)
            return [yielded[index] for index in scalar_positions]

        try:
            scalar_results = self._for_range_values(
                trip_count, loop_body, scalar_initial)
        finally:
            self._scf_mutable_tensors = prior_mutable
        for index, result in enumerate(op.results):
            if index in tensor_positions:
                self.vals[result] = carried[index]
            else:
                self.vals[result] = scalar_results[scalar_positions.index(index)]

    def _t_scf_if(self, op):
        """Lower SCF selection as real LLVM control flow.

        Executing both source branches and selecting their scalar results is not generally
        semantics preserving: an inactive branch may contain an invalid load, division, or
        transcendental domain operation.  Keep tensor recurrence in the same loop-private
        storage, but execute writes and scalar computations only in the selected CFG block.
        Scalar results become block arguments of the merge block.
        """
        if len(op.operands) != 1 or len(op.regions) != 2:
            raise LoweringDeclined("scf.if requires a condition and two regions", op=op.name)
        condition = self._condition(self.get(op.operands[0]))

        scalar_result_positions = []
        scalar_result_types = []
        for index, result in enumerate(op.results):
            source_type = result.type
            if isinstance(source_type, TensorType):
                continue
            scalar_result_positions.append(index)
            if isinstance(source_type, (IndexType, IntegerType)):
                scalar_result_types.append(i64)
            elif isinstance(source_type, Float64Type):
                scalar_result_types.append(Float64Type())
            elif str(source_type) in {"f32", "f16", "bf16"}:
                # Host scalar evaluation uses f32 as the working representation for
                # f16/bf16 and rounds only when a source tensor element is stored.
                scalar_result_types.append(Float32Type())
            else:
                raise LoweringDeclined(
                    f"scf.if cannot represent scalar result type {source_type}", op=op.name)

        true_block = Block()
        false_block = Block()
        after = Block(arg_types=scalar_result_types)
        for block in (true_block, false_block, after):
            self.fb.region.add_block(block)
        self.fb.add(llvm.CondBrOp(condition, true_block, [], false_block, []))

        branches = []
        for region, target in zip(op.regions, (true_block, false_block)):
            if len(region.blocks) != 1:
                raise LoweringDeclined("scf.if branch must be single-block", op=op.name)
            self.fb.blk = target
            yielded = self._structured_block(region.blocks[0], [])
            branches.append(yielded)
            scalars = [yielded[index] for index in scalar_result_positions]
            if (len(scalars) != len(scalar_result_types)
                    or any(not isinstance(value, SSAValue) or value.type != expected
                           for value, expected in zip(scalars, scalar_result_types))):
                raise LoweringDeclined(
                    "scf.if scalar result changed representation type", op=op.name)
            self.fb.add(llvm.BrOp(after, *scalars))
        if any(len(branch) != len(op.results) for branch in branches):
            raise LoweringDeclined("scf.if yield count disagrees with results", op=op.name)
        self.fb.blk = after
        scalar_results = iter(after.args)
        for result, when_true, when_false in zip(op.results, *branches):
            if isinstance(when_true, LoopTensor) or isinstance(when_false, LoopTensor):
                if when_true is not when_false:
                    raise LoweringDeclined(
                        "scf.if tensor branches do not update the same storage",
                        op=op.name)
                self.vals[result] = when_true
            else:
                self.vals[result] = next(scalar_results)

    def _t_tensor_insert(self, op):
        """Apply one functional tensor update to the active SCF private storage."""
        if len(op.operands) < 2 or len(op.results) != 1:
            raise LoweringDeclined("tensor.insert has invalid arity", op=op.name)
        value = self.get(op.operands[0])
        tensor = self.get(op.operands[1])
        if (not isinstance(tensor, LoopTensor) or tensor.write is None
                or tensor not in getattr(self, "_scf_mutable_tensors", set())):
            raise LoweringDeclined(
                "tensor.insert requires loop-private functional-update storage",
                op=op.name)
        index = tuple(self.get(operand) for operand in op.operands[2:])
        tensor.write(index, self._round(value, tensor.ety))
        self.vals[op.results[0]] = tensor

    def _t_arith_constant(self, op):
        if not isinstance(op.results[0].type, TensorType):
            return super()._t_arith_constant(op)
        value = attr_of(op, "value")
        shape = tensor_shape(op.results[0].type)
        if (not isinstance(value, DenseIntOrFPElementsAttr)
                or len(value) != prod(shape) or not hasattr(self, "constant_tensor")):
            raise LoweringDeclined("dense tensor constant requires exact immutable storage")
        ety = elem_name(value.get_element_type())
        tensor = self.storage(shape, ety, self.constant_tensor(value))
        # A source constant is immutable; destination operations must materialize
        # their own writable storage rather than mutate this shared initializer.
        self.vals[op.results[0]] = LoopTensor(shape, ety, tensor.read)

    def _tensor_op(self, op):
        prior = self._storage_owner
        self._storage_owner = op
        try:
            return self._dispatch_tensor_op(op)
        finally:
            self._storage_owner = prior

    def _dispatch_tensor_op(self, op):
        if op in getattr(self, "_commuted_gathers", {}):
            self._emit_commuted_gather(self._commuted_gathers[op])
            return
        if op in getattr(self, "_commuted_source_ops", set()):
            return
        # Whole-model captures can keep scalar producers (rank-zero extracts, casts and
        # comparisons) between tensor operations at function scope.  They use the same scalar
        # semantics as linalg regions; only operations with no nested control flow are admitted.
        if (len(op.results) == 1 and not isinstance(op.results[0].type, TensorType)
                and not op.regions):
            self.vals[op.results[0]] = self._scalar_op(
                op, [self.get(operand) for operand in op.operands])
            return
        if op.name == "builtin.unregistered":
            from xdsl.dialects.builtin import ModuleOp
            from merlin.llvmlower.passes_xdsl import lower_quant_ext
            # Reuse the compiler's existing target-neutral quantization semantics.
            # Normalize only a clone owned by this source operation: the original
            # full-model graph and its operation ordinal remain unchanged.
            normalized = ModuleOp([op.clone()])
            if lower_quant_ext(normalized) != 1:
                name = getattr(getattr(op, "op_name", None), "data", op.name)
                raise LoweringDeclined(f"no loop-backed normalization for {name}", op=name)
            generated = list(normalized.body.block.ops)
            prior_sources = getattr(self, "_normalized_source_values", {})
            self._normalized_source_values = {
                **prior_sources, generated[-1].results[0]: op.results[0]}
            try:
                for operation in generated:
                    self._tensor_op(operation)
            finally:
                self._normalized_source_values = prior_sources
            self.vals[op.results[0]] = self.get(generated[-1].results[0])
            return
        return super()._tensor_op(op)

    def _pointwise_info(self, op):
        """A complete, pure scalar map with statically in-bounds projected indexing."""
        if op.name != "linalg.generic" or len(op.results) != 1:
            return None
        counts = self._static(op, "operandSegmentSizes")
        if len(counts) != 2 or counts[1] != 1:
            return None
        n_in = counts[0]
        shape = tensor_shape(op.results[0].type)
        rank = len(shape)
        if any(size <= 0 for size in shape):
            return None
        maps = [item.data for item in attr_of(op, "indexing_maps")]
        if len(maps) != len(op.operands) or any(m.num_dims != rank or m.num_symbols for m in maps):
            return None
        if [getattr(getattr(item, "data", None), "value", None)
                for item in attr_of(op, "iterator_types")] != ["parallel"] * rank:
            return None
        if len(maps[-1].results) != rank or not all(isinstance(e, AffineDimExpr) for e in maps[-1].results):
            return None
        permutation = [expr.position for expr in maps[-1].results]
        if sorted(permutation) != list(range(rank)):
            return None
        bounds = [shape[permutation.index(dim)] for dim in range(rank)]
        for value, amap in zip(op.operands, maps):
            operand_shape = tensor_shape(value.type)
            if len(amap.results) != len(operand_shape):
                return None
            for expr, size in zip(amap.results, operand_shape):
                if isinstance(expr, AffineDimExpr):
                    if not 0 <= expr.position < rank or bounds[expr.position] > size:
                        return None
                elif not isinstance(expr, AffineConstantExpr) or not 0 <= expr.value < size:
                    return None
        body = op.regions[0].blocks[0]
        if len(body.args) != n_in + 1 or list(body.args[-1].uses):
            return None
        operations = list(body.ops)
        if not operations or operations[-1].name != "linalg.yield":
            return None
        for operation in operations[:-1]:
            # These scalar namespaces are already interpreted by _scalar_op. Unknown
            # scalar operations still decline there; memory reads/calls/regions cannot fuse.
            if (operation.regions or len(operation.results) != 1
                    or operation.name.split(".")[0] not in {"arith", "math"}):
                return None
        return (n_in, shape, maps, permutation, body)

    def _defer_pointwise(self, op, info):
        """Install a scalar pointwise value whose approved sink controls evaluation."""
        n_in, shape, maps, permutation, body = info
        inputs = [self.get(value) for value in op.operands[:n_in]]
        ety = elem_name(op.results[0].type.get_element_type())
        def read(index):
            ivs = tuple(index[permutation.index(dim)] for dim in range(len(shape)))
            prior_ivs = self.ivs
            try:
                self.ivs = ivs
                args = [tensor.at(tuple(self._affine(expr, ivs) for expr in amap.results))
                        for tensor, amap in zip(inputs, maps[:n_in])]
                # Keep every original scalar operation and result-width rounding. This
                # is producer inlining, never algebraic reassociation or transcendental rewrite.
                return self._round(self._scalar_block(body, [*args, self.zero(ety)]), ety)
            finally:
                self.ivs = prior_ivs
        self.vals[op.results[0]] = LoopTensor(shape, ety, read)
        return True

    def _lazy_segment_output(self, op):
        """Write a pure final pointwise map directly to its host/device boundary buffer.

        `run_segment` consumes each declared output exactly once after all segment operations.
        A one-use chain of bijective shape views does not change that evaluation count: the
        boundary traversal maps each destination element back to exactly one pointwise element.
        External users continue to read the same materialized DRAM buffer after the segment
        completes; computation and dtype rounding still occur once per output element.
        """
        info = self._pointwise_info(op)
        if info is None:
            return False
        generated_value = op.results[0]
        value = getattr(self, "_normalized_source_values", {}).get(
            generated_value, generated_value)
        segment_ops = getattr(self, "_segment_ops", set())
        segment_outputs = getattr(self, "_segment_outputs", set())
        views = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape",
                 "linalg.transpose"}
        while value not in segment_outputs:
            uses = list(value.uses)
            if len(uses) != 1:
                return False
            use = uses[0]
            consumer = use.operation
            if (consumer not in segment_ops or consumer.name not in views
                    or use.index != 0 or len(consumer.results) != 1):
                return False
            next_value = consumer.results[0]
            source_shape = tensor_shape(value.type)
            target_shape = tensor_shape(next_value.type)
            if (prod(source_shape) != prod(target_shape)
                    or any(size <= 0 for size in target_shape)
                    or value.type.get_element_type() != next_value.type.get_element_type()):
                return False
            if consumer.name == "linalg.transpose":
                permutation = self._static(consumer, "permutation")
                if (sorted(permutation) != list(range(len(source_shape)))
                        or tuple(source_shape[d] for d in permutation) != tuple(target_shape)):
                    return False
            value = next_value
        if any(use.operation in segment_ops for use in value.uses):
            return False
        return self._defer_pointwise(op, info)

    def _lazy_pointwise(self, op):
        info = self._pointwise_info(op)
        source_value = getattr(self, "_normalized_source_values", {}).get(op.results[0], op.results[0])
        if info is None or source_value in getattr(self, "_segment_outputs", set()):
            return False
        uses = list(source_value.uses)
        if len(uses) != 1:
            return False
        use = uses[0]
        consumer = use.operation
        if consumer not in getattr(self, "_segment_ops", set()):
            return False
        # Follow only element-count-preserving, bijective views.  These operations are
        # themselves emitted as lazy address maps below, so carrying the producer through
        # them neither duplicates scalar evaluation nor changes its rounding point.  This
        # matters for authored pointwise -> reshape/transpose -> pointwise/reduce chains:
        # stopping at the first view needlessly materializes the producer even though the
        # eventual sink still visits every element exactly once.
        view_value = source_value
        bijective_views = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape",
                           "linalg.transpose"}
        while consumer.name in bijective_views:
            if use.index != 0 or len(consumer.results) != 1:
                return False
            next_value = consumer.results[0]
            source_shape = tensor_shape(view_value.type)
            target_shape = tensor_shape(next_value.type)
            if (prod(source_shape) != prod(target_shape)
                    or any(size <= 0 for size in target_shape)
                    or view_value.type.get_element_type() != next_value.type.get_element_type()
                    or next_value in getattr(self, "_segment_outputs", set())):
                return False
            if consumer.name == "linalg.transpose":
                view_permutation = self._static(consumer, "permutation")
                if (sorted(view_permutation) != list(range(len(source_shape)))
                        or tuple(source_shape[d] for d in view_permutation) != tuple(target_shape)):
                    return False
            view_value = next_value
            view_uses = list(view_value.uses)
            if len(view_uses) != 1:
                return False
            use = view_uses[0]
            consumer = use.operation
            if consumer not in getattr(self, "_segment_ops", set()):
                return False
        if consumer.name == "linalg.matmul" and use.index in (0, 1):
            # A contraction ordinarily repeats one operand over an output dimension.
            # Inline only if its projected access map is one-use: A[i,k] repeats N
            # times and B[k,j] repeats M times. Unit repetition preserves the exact
            # number of scalar evaluations while eliminating producer materialization.
            a_shape, b_shape = [tensor_shape(value.type) for value in consumer.operands[:2]]
            if (len(a_shape) != 2 or len(b_shape) != 2
                    or a_shape[1] != b_shape[0]
                    or any(dim <= 0 for dim in (*a_shape, *b_shape))):
                return False
            repetitions = b_shape[1] if use.index == 0 else a_shape[0]
            operand_shape = a_shape if use.index == 0 else b_shape
            if repetitions != 1 or prod(info[1]) != prod(operand_shape):
                return False
            return self._defer_pointwise(op, info)
        if consumer.name == "linalg.reduce":
            # The loop reduction visits every element of every input exactly once.
            # A unique, full-shape producer input can therefore remain lazy without
            # duplicating or dropping any producer scalar evaluation; its dtype
            # rounding still happens before the reduction combines the value.
            n_reduce_inputs = len(consumer.operands) // 2
            if (use.index >= n_reduce_inputs
                    or not consumer.operands
                    or prod(info[1]) != prod(tensor_shape(consumer.operands[0].type))):
                return False
            return self._defer_pointwise(op, info)
        next_info = self._pointwise_info(consumer)
        if next_info is None:
            return False
        _, shape, _, _, _ = info
        next_n_in, next_shape, next_maps, _, _ = next_info
        if prod(next_shape) != prod(shape) or use.index >= next_n_in:
            return False
        # A bijective pointwise-consumer input map evaluates each producer element exactly once.
        # Broadcast consumers and fanout stay materialized.
        consumed = next_maps[use.index].results
        if (len(consumed) != len(shape) or not all(isinstance(e, AffineDimExpr) for e in consumed)
                or sorted(e.position for e in consumed) != list(range(len(shape)))):
            return False
        return self._defer_pointwise(op, info)

    def run(self, func_op):
        self._commuted_gathers, self._commuted_source_ops = {}, set()
        operations = list(func_op.regions[0].blocks[0].ops)
        self._begin_storage_segment(operations)
        self._segment_ops = {op for op in operations if op.name != "func.return"}
        self._segment_outputs = {
            value for op in operations if op.name == "func.return"
            for value in op.operands
        }
        return super().run(func_op)

    def run_segment(self, ops, inputs, outputs, source_op_indices=()):
        self._commuted_gathers, self._commuted_source_ops = {}, set()
        # Tensor values from a completed segment now live in their explicit boundary buffers, not in
        # the workspace range that the next segment is about to reuse.  Scalar SSA remains valid.
        self.vals = {value: lowered for value, lowered in self.vals.items()
                     if not isinstance(lowered, LoopTensor)}
        self._begin_storage_segment(ops, source_op_indices)
        self._segment_ops = set(ops)
        self._segment_outputs = {value for value, _ in outputs}
        return super().run_segment(ops, inputs, outputs)

    def _lazy_copy_map(self, op):
        """Fold a proven pure affine copy into its one same-segment consumer.

        This removes an encoding temporary, not consumer loads or arithmetic. All host
        tensor storage is SSA-owned: consumers allocate new outputs rather than overwriting
        input storage. Do not carry deferred reads across a device/task boundary or fanout.
        """
        if len(op.operands) != 2 or len(op.results) != 1:
            return False
        if self._static(op, "operandSegmentSizes") != [1, 1]:
            return False
        body = op.regions[0].blocks[0]
        body_ops = list(body.ops)
        if (len(body.args) != 2 or len(body_ops) != 1
                or body_ops[0].name != "linalg.yield"
                or tuple(body_ops[0].operands) != (body.args[0],)):
            return False
        source = self.get(op.operands[0])
        shape = tensor_shape(op.results[0].type)
        if op.operands[0].type.get_element_type() != op.results[0].type.get_element_type():
            return False
        maps = [item.data for item in attr_of(op, "indexing_maps")]
        if len(maps) != 2:
            return False
        incoming, outgoing = maps
        if (incoming.num_symbols or outgoing.num_symbols
                or incoming.num_dims != len(shape) or outgoing.num_dims != len(shape)
                or len(incoming.results) != len(source.shape)
                or len(outgoing.results) != len(shape)):
            return False
        if [getattr(getattr(item, "data", None), "value", None)
                for item in attr_of(op, "iterator_types")] != ["parallel"] * len(shape):
            return False
        if not all(isinstance(expr, AffineDimExpr) for expr in outgoing.results):
            return False
        permutation = [expr.position for expr in outgoing.results]
        if sorted(permutation) != list(range(len(shape))) or any(size <= 0 for size in shape):
            return False
        bounds = [shape[permutation.index(dim)] for dim in range(len(shape))]

        def interval(expr):
            if isinstance(expr, AffineDimExpr):
                return (0, bounds[expr.position] - 1)
            if isinstance(expr, AffineConstantExpr):
                return (expr.value, expr.value)
            if not isinstance(expr, AffineBinaryOpExpr):
                return None
            lhs, rhs = interval(expr.lhs), interval(expr.rhs)
            if lhs is None or rhs is None:
                return None
            if expr.kind == AffineBinaryOpKind.Add:
                result = (lhs[0] + rhs[0], lhs[1] + rhs[1])
            elif expr.kind == AffineBinaryOpKind.Mul and (lhs[0] == lhs[1] or rhs[0] == rhs[1]):
                products = [a * b for a in lhs for b in rhs]
                result = (min(products), max(products))
            else:
                return None
            return result if -(1 << 63) <= result[0] <= result[1] < (1 << 63) else None

        for expr, size in zip(incoming.results, source.shape):
            limits = interval(expr)
            if limits is None or limits[0] < 0 or limits[1] >= size:
                return False
        # Restrict view chains to one consumer within this already-ordered host segment.
        value = op.results[0]
        views = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape",
                 "tensor.extract_slice", "linalg.transpose", "linalg.broadcast"}
        while True:
            uses = list(value.uses)
            if len(uses) != 1 or value in getattr(self, "_segment_outputs", set()):
                return False
            consumer = uses[0].operation
            if consumer not in getattr(self, "_segment_ops", set()):
                return False
            if consumer.name not in views:
                break
            if len(consumer.results) != 1:
                return False
            value = consumer.results[0]
        # Generic/reduction consumers are pure tensor operations in this lowering. Avoid
        # deferring loads across unknown side effects or output pointer stores.
        if consumer.name not in {"linalg.generic", "linalg.reduce", "linalg.matmul"}:
            return False
        def read(index):
            ivs = tuple(index[permutation.index(dim)] for dim in range(len(shape)))
            return source.at(tuple(self._affine(expr, ivs) for expr in incoming.results))
        self.vals[op.results[0]] = LoopTensor(shape, source.ety, read)
        return True

    def _lazy_gather_view(self, op):
        """Defer a bounded readonly gather through bijective, one-use encoding views.

        The final pointwise sink consumes every gathered element exactly once. This
        removes only the gathered tensor materialization: indices, source reads and
        original scalar floating-point operations retain their evaluation counts.
        No deferred value escapes the ordered host segment or aliases writable input.
        """
        from .gather_view import readonly_gather
        info = readonly_gather(op, self._static)
        if info is None:
            return False
        value = op.results[0]
        views = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape", "linalg.transpose"}
        while True:
            uses = list(value.uses)
            if len(uses) != 1 or value in getattr(self, "_segment_outputs", set()):
                return False
            use = uses[0]
            consumer = use.operation
            if consumer not in getattr(self, "_segment_ops", set()):
                return False
            if consumer.name not in views:
                break
            if use.index != 0 or len(consumer.results) != 1:
                return False
            source_shape, target_shape = tensor_shape(value.type), tensor_shape(consumer.results[0].type)
            if (prod(source_shape) != prod(target_shape) or any(size <= 0 for size in target_shape)
                    or value.type.get_element_type() != consumer.results[0].type.get_element_type()):
                return False
            if consumer.name == "linalg.transpose":
                permutation = self._static(consumer, "permutation")
                if (sorted(permutation) != list(range(len(source_shape)))
                        or tuple(source_shape[d] for d in permutation) != tuple(target_shape)):
                    return False
            value = consumer.results[0]
        sink = self._pointwise_info(consumer)
        if sink is None or use.index >= sink[0]:
            return False
        source_shape, sink_shape = tensor_shape(value.type), sink[1]
        consumed = sink[2][use.index].results
        if (len(source_shape) != len(sink_shape) or prod(source_shape) != prod(sink_shape)
                or not all(isinstance(e, AffineDimExpr) for e in consumed)
                or sorted(e.position for e in consumed) != list(range(len(sink_shape)))):
            return False
        # The existing scalar emitter includes tensor.extract and linalg.index and
        # restores the enclosing consumer IV state around nested deferred reads.
        return self._defer_pointwise(op, info)

    def _charge(self, n):
        # Count logical work, not expanded code. Loops represent these iterations.
        self.elements += int(n)

    @property
    def fb(self):
        return self.fp.fb

    def value(self, item):
        return item if isinstance(item, SSAValue) else self.fb.const(int(item))

    def flat(self, index, shape):
        # Form a row-major offset as a Horner recurrence.  Compared with the
        # sum-of-products spelling this needs one fewer multiply, does not add
        # an initial zero, and exposes unit extents without emitting arithmetic.
        # Tensor extents and indices are statically bounded by the source-loop
        # construction, so both spellings denote the same non-negative offset.
        result = None
        for item, extent in zip(index, shape):
            item = self.value(item)
            if result is None:
                result = item
                continue
            if extent != 1:
                result = self.fb.mul_i(result, self.fb.const(extent))
            result = self.fb.add_i(result, item)
        return self.fb.const(0) if result is None else result

    def unflat(self, index, shape):
        # Decode the row-major index once as a mixed-radix recurrence.  Computing every
        # coordinate independently as ``(index / stride) % size`` needlessly takes the
        # highest non-unit coordinate modulo its extent even though callers establish
        # ``0 <= index < prod(shape)``.  Peeling dimensions from the minor end preserves
        # every coordinate exactly and lets that highest coordinate be the final quotient.
        values = [None] * len(shape)
        non_unit = [axis for axis, size in enumerate(shape) if size != 1]
        if not non_unit:
            return tuple(self.fb.const(0) for _ in shape)
        highest = non_unit[0]
        remaining = index
        for axis in reversed(range(len(shape))):
            size = shape[axis]
            if size == 1:
                values[axis] = self.fb.const(0)
            elif axis == highest:
                values[axis] = remaining
            else:
                divisor = self.fb.const(size)
                values[axis] = self.fb.urem_i(remaining, divisor)
                remaining = self.fb.udiv_i(remaining, divisor)
        return tuple(values)

    def indices(self, bounds, body):
        def visit(index):
            if len(index) == len(bounds):
                body(tuple(index))
            elif bounds[len(index)] == 1:
                # A unit source dimension has exactly one legal coordinate.  Avoid
                # emitting a one-trip CFG loop (header compare, branch and induction
                # update) while retaining the same row-major visit at index zero.
                visit([*index, self.fb.const(0)])
            else:
                self.fb.for_range(bounds[len(index)], lambda iv: visit([*index, iv]))
        visit([])

    def storage(self, shape, ety, ptr=None, padded=False):
        physical = tuple(shape[:-1]) + (row_pitch(shape[-1]),) if padded and shape else tuple(shape)
        if ptr is None:
            if ety not in INT_TYPES and ety not in {"i1", "bf16", "f32", "f64"}:
                raise LoweringDeclined(f"loop storage format {ety} not implemented", op="host_lane")
            # FpBuilder already defines bf16 memory as packed i16 widened to f32 for
            # computation.  Internal loop storage must use that identical physical format.
            ty = (i1 if ety == "i1" else INT_TYPES["i16"] if ety == "bf16"
                  else INT_TYPES[ety] if is_int(ety) else Float64Type()
                  if ety == "f64" else Float32Type())
            width = (int(ty.width.data) if is_int(ety) else 16 if ety == "bf16"
                     else 64 if ety == "f64" else 32)
            elements = prod(physical)
            nbytes = elements * ((width + 7) // 8)
            owner = self._storage_owner
            record = {
                "operation": getattr(owner, "name", "host_storage"),
                "source_op_index": self._source_index_by_op.get(owner),
                "shape": list(shape), "physical_shape": list(physical),
                "dtype": ety, "bytes": nbytes, "segment": self._segment_number,
            }
            if elements == 1:
                allocation = llvm.AllocaOp(self.fb.const(elements), ty, alignment=self._ALIGNMENT)
                allocation.attributes["merlin.host_storage_bytes"] = IntegerAttr(nbytes, i64)
                self.fb.prologue(allocation)
                ptr = allocation.results[0]
                reserved = self._align(nbytes)
                self._stack_frame_upper_bound += reserved
                record.update({"storage": "bounded_stack_scalar", "offset": None,
                               "reserved_bytes": reserved})
            else:
                if not callable(self.workspace_pointer):
                    raise LoweringDeclined(
                        "tensor storage requires an explicit host workspace", op="host_lane")
                offset = self._align(self._workspace_cursor)
                reserved = self._align(nbytes)
                self._workspace_cursor = offset + reserved
                self._workspace_peak = max(self._workspace_peak, self._workspace_cursor)
                self._workspace_requested += reserved
                self._segment_peaks[-1] = max(
                    self._segment_peaks[-1], self._workspace_cursor)
                ptr = self.workspace_pointer(offset)
                record.update({"storage": "reusable_workspace", "offset": offset,
                               "reserved_bytes": reserved})
            self._storage_allocations.append(record)
        def read(index):
            flat = self.flat(index, physical)
            if ety == "i1":
                value = self.fb.add(llvm.LoadOp(self.fb.gep(ptr, flat, i1), i1)).results[0]
                # Host integer expressions use i64.  An i1 predicate is unsigned: sign-extending
                # true would produce -1 and corrupt comparisons/selects.
                return self.fb.add(llvm.ZExtOp(value, i64)).results[0]
            if is_int(ety):
                return self.fb.load_i64(ptr, flat, ety)
            if ety == "f64":
                return self.fb.add(llvm.LoadOp(
                    self.fb.gep(ptr, flat, Float64Type()), Float64Type())).results[0]
            return self.fp.load(ptr, flat, ety)
        def write(index, value):
            flat = self.flat(index, physical)
            if ety == "i1":
                narrowed = self.fb.add(llvm.TruncOp(value, i1)).results[0]
                self.fb.add(llvm.StoreOp(narrowed, self.fb.gep(ptr, flat, i1)))
            elif is_int(ety):
                self.fb.store_i64(value, ptr, flat, ety)
            elif ety == "f64":
                self.fb.add(llvm.StoreOp(
                    value, self.fb.gep(ptr, flat, Float64Type())))
            else:
                self.fp.store(value, ptr, flat, ety)
        return LoopTensor(shape, ety, read, write)

    def materialize(self, tensor):
        out = self.storage(tensor.shape, tensor.ety)
        self.indices(tensor.shape, lambda idx: out.write(idx, tensor.at(idx)))
        return out

    def load_arg(self, ptr, ty):
        return self.storage(tensor_shape(ty), elem_name(ty.get_element_type()), ptr, padded=True)

    def store_result(self, ptr, val):
        out = self.storage(val.shape, val.ety, ptr, padded=True)
        self.indices(val.shape, lambda idx: out.write(idx, val.at(idx)))

    def _scalar_op(self, op, ins):
        result_type = op.results[0].type if op.results else None
        if op.name == "arith.bitcast":
            if len(ins) != 1:
                raise LoweringDeclined("scalar bitcast requires one operand", op=op.name)
            source_type = op.operands[0].type
            # Host integers use an i64 SSA representation even when their source type is
            # narrower.  A source bitcast is not an integer width conversion: preserve the
            # bits at the source width, then bridge to/from that host representation.
            if isinstance(source_type, Float32Type) and isinstance(result_type, IntegerType):
                if int(result_type.width.data) != 32 or not isinstance(ins[0].type, Float32Type):
                    raise LoweringDeclined(
                        "f32 bitcast requires an i32 result and an f32 value", op=op.name)
                raw = self.fb.add(llvm.BitcastOp(ins[0], IntegerType(32))).results[0]
                return self.fb.add(llvm.SExtOp(raw, i64)).results[0]
            if isinstance(source_type, IntegerType) and isinstance(result_type, Float32Type):
                if int(source_type.width.data) != 32 or ins[0].type != i64:
                    raise LoweringDeclined(
                        "i32 bitcast requires an f32 result and an i64 host integer",
                        op=op.name)
                raw = self.fb.add(llvm.TruncOp(ins[0], IntegerType(32))).results[0]
                return self.fb.add(llvm.BitcastOp(raw, Float32Type())).results[0]
            if isinstance(source_type, Float64Type) and isinstance(result_type, IntegerType):
                if int(result_type.width.data) != 64 or not isinstance(ins[0].type, Float64Type):
                    raise LoweringDeclined(
                        "f64 bitcast requires an i64 result and an f64 value", op=op.name)
                return self.fb.add(llvm.BitcastOp(ins[0], i64)).results[0]
            if isinstance(source_type, IntegerType) and isinstance(result_type, Float64Type):
                if int(source_type.width.data) != 64 or ins[0].type != i64:
                    raise LoweringDeclined(
                        "i64 bitcast requires an f64 result and an i64 host integer",
                        op=op.name)
                return self.fb.add(llvm.BitcastOp(ins[0], Float64Type())).results[0]
            if type(source_type) is type(result_type) and source_type == result_type:
                return ins[0]
            raise LoweringDeclined(
                f"the CPU lane cannot bitcast scalar {source_type} to {result_type}",
                op=op.name)
        if op.name == "math.powf" and isinstance(result_type, Float64Type):
            if len(ins) != 2 or not all(isinstance(value.type, Float64Type) for value in ins):
                raise LoweringDeclined(
                    "f64 power requires two scalar f64 values", op=op.name)
            return self.fb.add(llvm.FPowOp(ins[0], ins[1])).results[0]
        if op.name == "arith.sitofp" and isinstance(result_type, Float64Type):
            return self.fb.add(llvm.SIToFPOp(ins[0], Float64Type())).results[0]
        if op.name == "arith.extf" and isinstance(result_type, Float64Type):
            return self.fb.add(llvm.FPExtOp(ins[0], Float64Type())).results[0]
        if (op.name == "arith.truncf" and isinstance(result_type, Float32Type)
                and ins and isinstance(ins[0].type, Float64Type)):
            return self.fb.add(FPTruncOp(ins[0], Float32Type())).results[0]
        if op.name == "arith.cmpf":
            from xdsl.dialects.arith import CMPF_COMPARISON_OPERATIONS
            if (len(ins) != 2 or ins[0].type != ins[1].type
                    or not isinstance(ins[0].type, (Float32Type, Float64Type))):
                raise LoweringDeclined(
                    "floating comparison requires two scalar values of one supported type",
                    op=op.name)
            predicate = CMPF_COMPARISON_OPERATIONS[attr_of(op, "predicate").value.data]
            result = self.fb.add(llvm.FCmpOp(ins[0], ins[1], predicate)).results[0]
            return self.fb.add(llvm.ZExtOp(result, i64)).results[0]
        if op.name == "arith.cmpi":
            from xdsl.dialects.arith import CMPI_COMPARISON_OPERATIONS
            source_type = op.operands[0].type
            if isinstance(source_type, IndexType):
                width = 64
            elif isinstance(source_type, IntegerType) and 1 <= source_type.width.data <= 64:
                width = int(source_type.width.data)
            else:
                raise LoweringDeclined("integer comparison requires a supported scalar width", op=op.name)
            predicate = CMPI_COMPARISON_OPERATIONS[attr_of(op, "predicate").value.data]
            # Host integers travel in i64. Restore the source bit width before
            # comparison so unsigned predicates do not compare sign-extended values.
            operands = [self.fb.add(llvm.TruncOp(value, IntegerType(width))).results[0]
                        if width != 64 else value for value in ins]
            result = self.fb.add(llvm.ICmpOp(*operands,
                IntegerAttr(llvm.ICmpPredicateFlag(predicate).to_int(), i64))).results[0]
            return self.fb.add(llvm.ZExtOp(result, i64)).results[0]
        if op.name == "arith.select":
            if op.operands[0].type != i1 or ins[1].type != ins[2].type:
                raise LoweringDeclined("scalar select requires an i1 condition and matching values", op=op.name)
            condition = self.fb.add(llvm.TruncOp(ins[0], i1)).results[0]
            return self.fb.add(llvm.SelectOp(condition, ins[1], ins[2])).results[0]
        if op.name in {"math.cos", "math.sin"}:
            if (len(ins) != 1 or op.results[0].type != ins[0].type
                    or not isinstance(ins[0].type, (Float32Type, Float64Type))):
                raise LoweringDeclined(
                    "runtime trigonometry requires one supported scalar float type",
                    op=op.name)
            intrinsic = llvm.FCosOp if op.name == "math.cos" else llvm.FSinOp
            # Preserve the source operation via the standard LLVM intrinsic. The
            # target object pipeline/legal runtime chooses the implementation;
            # do not invent a polynomial, reassociate, or attach fast-math flags.
            return self.fb.add(intrinsic(ins[0])).results[0]
        if op.name == "tensor.extract":
            if not isinstance(ins[0], LoopTensor):
                raise LoweringDeclined("loop gather source is not a tensor", op=op.name)
            return ins[0].at(tuple(ins[1:]))
        if op.name == "linalg.index":
            attr = attr_of(op, "dim")
            return self.value(self.ivs[int(attr.value.data)])
        return super()._scalar_op(op, ins)

    def _affine(self, expr, ivs):
        if isinstance(expr, AffineDimExpr):
            return self.value(ivs[expr.position])
        if isinstance(expr, AffineConstantExpr):
            return self.fb.const(expr.value)
        if isinstance(expr, AffineBinaryOpExpr):
            lhs, rhs = self._affine(expr.lhs, ivs), self._affine(expr.rhs, ivs)
            if expr.kind == AffineBinaryOpKind.Add:
                return self.fb.add_i(lhs, rhs)
            if expr.kind == AffineBinaryOpKind.Mul:
                return self.fb.mul_i(lhs, rhs)
            def floor_div(a, b):
                quotient = self.fb.sdiv_i(a, b)
                remainder = self.fb.srem_i(a, b)
                nonzero = self.fb.ashr_i(self.fb.or_i(remainder, self.fb.sub_i(self.fb.const(0), remainder)), 63)
                correction = self.fb.and_i(self.fb.and_i(self.fb.mask_neg(a), nonzero), self.fb.const(1))
                return self.fb.sub_i(quotient, correction)
            if expr.kind == AffineBinaryOpKind.FloorDiv:
                return floor_div(lhs, rhs)
            if expr.kind == AffineBinaryOpKind.Mod:
                return self.fb.sub_i(lhs, self.fb.mul_i(floor_div(lhs, rhs), rhs))
            if expr.kind == AffineBinaryOpKind.CeilDiv:
                return self.fb.sub_i(self.fb.const(0), floor_div(self.fb.sub_i(self.fb.const(0), lhs), rhs))
        raise LoweringDeclined(f"unsupported loop affine expression {expr}", op="host_lane")

    def _splat(self, op, scalar):
        ty = op.results[0].type
        self.vals[op.results[0]] = LoopTensor(tensor_shape(ty), elem_name(ty.get_element_type()),
                                             lambda index: scalar)

    def _t_tensor_empty(self, op):
        self._splat(op, self.zero(elem_name(op.results[0].type.get_element_type())))

    def _t_tensor_splat(self, op):
        self._splat(op, self.get(op.operands[0]))

    _t_linalg_fill = _t_tensor_splat

    def _reshape(self, op):
        src = self.get(op.operands[0])
        shape = tensor_shape(op.results[0].type)
        if prod(shape) != prod(src.shape):
            raise LoweringDeclined("reshape changes element count", op=op.name)
        self.vals[op.results[0]] = LoopTensor(shape, src.ety,
            lambda idx: src.at(self.unflat(self.flat(idx, shape), src.shape)))

    _t_tensor_expand_shape = _reshape
    _t_tensor_collapse_shape = _reshape
    _t_tensor_reshape = _reshape

    def _t_tensor_insert_slice(self, op):
        from .gather_commute import plan_padded_gather
        plan = plan_padded_gather(self, op)
        if plan is not None:
            if not hasattr(self, "_commuted_gathers"):
                self._commuted_gathers, self._commuted_source_ops = {}, set()
            self._commuted_gathers[plan["gather"]] = plan
            self._commuted_source_ops.update(step[1] for step in plan["steps"])
            # The sole read of this insertion belongs to the fused gather. No float
            # padded tensor is observable; the gather emits its transformed encoding.
            return
        src, dest = [self.get(value) for value in op.operands[:2]]
        out = self.materialize(dest)
        offsets, sizes, strides = [self._static(op, key) for key in
                                   ("static_offsets", "static_sizes", "static_strides")]
        self.indices(sizes, lambda idx: out.write(tuple(self.fb.add_i(self.fb.const(offset),
            self.fb.mul_i(iv, self.fb.const(stride))) for iv, offset, stride in zip(idx, offsets, strides)),
            src.at(idx)))
        self.vals[op.results[0]] = out

    def _emit_commuted_gather(self, plan):
        """Evaluate original Q on each interior value and once on the padding scalar."""
        for constant in plan["constants"]:
            if constant.results[0] not in self.vals:
                self.vals[constant.results[0]] = self._const(constant)
        def transform(value):
            for operation, info, varying, fixed in plan["pointwise"]:
                n_in, _, _, _, body = info
                args = [value if index == varying else self.get(fixed[index]) for index in range(n_in)]
                ety = elem_name(operation.results[0].type.get_element_type())
                value = self._round(self._scalar_block(body, [*args, self.zero(ety)]), ety)
            return value
        source = self.get(plan["source"])
        ety = elem_name(plan["output"].type.get_element_type())
        encoded = self.storage(plan["padded_shape"], ety)
        transformed_padding = transform(self.get(plan["padding"]))
        self.indices(encoded.shape, lambda index: encoded.write(index, transformed_padding))
        self.indices(source.shape, lambda index: encoded.write(tuple(
            self.fb.add_i(iv, self.fb.const(offset)) for iv, offset in zip(index, plan["offsets"])),
            transform(source.at(index))))
        n_in, shape, maps, permutation, body = plan["gather_info"]
        inputs = [self.get(value) for value in plan["gather"].operands[:n_in]]
        inserted = plan["insertion"].results[0]
        def read(index, shape=shape):
            prior_ivs = self.ivs
            prior_value = self.vals.get(inserted)
            try:
                ivs = tuple(index[permutation.index(dim)] for dim in range(len(shape)))
                self.ivs = ivs
                # Scalar index operations are unchanged; only the extracted storage
                # encoding changes after Q has been evaluated with exact source types.
                self.vals[inserted] = encoded
                args = [tensor.at(tuple(self._affine(expr, ivs) for expr in amap.results))
                    for tensor, amap in zip(inputs, maps[:n_in])]
                return self._scalar_block(body, [*args, self.zero(ety)])
            finally:
                self.ivs = prior_ivs
                if prior_value is None:
                    self.vals.pop(inserted, None)
                else:
                    self.vals[inserted] = prior_value
        value = LoopTensor(shape, ety, read)
        for kind, operation, step in plan["steps"]:
            source_value = value
            shape = tensor_shape(operation.results[0].type)
            if kind == "pointwise":
                _, info, varying, _ = step
                amap, output_perm = info[2][varying], info[3]
                def reindex(index, source_value=source_value, amap=amap, output_perm=output_perm):
                    ivs = tuple(index[output_perm.index(dim)] for dim in range(len(output_perm)))
                    return source_value.at(tuple(self._affine(expr, ivs) for expr in amap.results))
            elif operation.name == "linalg.transpose":
                perm = self._static(operation, "permutation")
                def reindex(index, source_value=source_value, perm=perm):
                    return source_value.at(tuple(index[perm.index(dim)] for dim in range(len(perm))))
            else:
                def reindex(index, source_value=source_value, shape=shape):
                    return source_value.at(self.unflat(self.flat(index, shape), source_value.shape))
            value = LoopTensor(shape, ety, reindex)
        self.vals[plan["output"]] = value

    def _t_tensor_extract_slice(self, op):
        src = self.get(op.operands[0])
        shape = tensor_shape(op.results[0].type)
        offsets, sizes, strides = [self._static(op, key) for key in
                                   ("static_offsets", "static_sizes", "static_strides")]
        def read(idx):
            expanded = self.unflat(self.flat(idx, shape), sizes)
            return src.at(tuple(self.fb.add_i(self.fb.const(offset), self.fb.mul_i(iv, self.fb.const(stride)))
                                for iv, offset, stride in zip(expanded, offsets, strides)))
        self.vals[op.results[0]] = LoopTensor(shape, src.ety, read)

    def _t_tensor_concat(self, op):
        sources = [self.get(value) for value in op.operands]
        shape = tensor_shape(op.results[0].type)
        axis = int(attr_of(op, "dim").value.data)
        out = self.storage(shape, sources[0].ety)
        offset = 0
        for src in sources:
            def copy(idx):
                dest = list(idx)
                dest[axis] = self.fb.add_i(dest[axis], self.fb.const(offset))
                out.write(tuple(dest), src.at(idx))
            self.indices(src.shape, copy)
            offset += src.shape[axis]
        self.vals[op.results[0]] = out

    def _t_linalg_transpose(self, op):
        src = self.get(op.operands[0])
        perm = self._static(op, "permutation")
        shape = tensor_shape(op.results[0].type)
        self.vals[op.results[0]] = LoopTensor(shape, src.ety,
            lambda idx: src.at(tuple(idx[perm.index(d)] for d in range(len(perm)))))

    def _t_linalg_broadcast(self, op):
        src = self.get(op.operands[0])
        dims = self._static(op, "dimensions")
        shape = tensor_shape(op.results[0].type)
        self.vals[op.results[0]] = LoopTensor(shape, src.ety,
            lambda idx: src.at(tuple(iv for d, iv in enumerate(idx) if d not in dims)))

    def _t_linalg_generic(self, op):
        if self._lazy_gather_view(op):
            return
        if self._lazy_copy_map(op):
            return
        if self._lazy_segment_output(op):
            return
        if self._lazy_pointwise(op):
            return
        counts = self._static(op, "operandSegmentSizes")
        n_in = counts[0]
        inputs = [self.get(value) for value in op.operands[:n_in]]
        output_values = list(op.operands[n_in:])
        maps = [attr.data for attr in attr_of(op, "indexing_maps")]
        iterator_types = [getattr(getattr(item, "data", None), "value", None)
                          for item in attr_of(op, "iterator_types")]
        initializers = [self.get(value) for value in output_values]
        bounds = [0] * maps[0].num_dims
        for amap, tensor in zip(maps, inputs + initializers):
            for d, expr in enumerate(amap.results):
                if isinstance(expr, AffineDimExpr):
                    bounds[expr.position] = tensor.shape[d]
        if any(bound <= 0 for bound in bounds):
            raise LoweringDeclined("generic loop has unresolved bound", op=op.name)

        # When the source's sole reduction iterator is innermost, its output element is a
        # true scalar loop recurrence.  Carry that f32/iN value through the CFG instead of
        # round-tripping it through the result tensor on every reduction step.  This preserves
        # the initializer, source iteration order, scalar block and per-step dtype rounding.
        output_maps = maps[n_in:]
        reduction_output = (
            len(output_values) >= 1
            and len(bounds) >= 1
            and iterator_types == ["parallel"] * (len(bounds) - 1) + ["reduction"]
            and len(output_maps) == len(output_values)
            and all(output_map.num_symbols == 0
                    and len(output_map.results) == len(bounds) - 1
                    and all(isinstance(expr, AffineDimExpr)
                            for expr in output_map.results)
                    and sorted(expr.position for expr in output_map.results)
                        == list(range(len(bounds) - 1))
                    for output_map in output_maps)
        )
        if reduction_output:
            outputs = [self.storage(initializer.shape, initializer.ety)
                       for initializer in initializers]
            def outer_body(outer_ivs):
                output_indexes = [tuple(self._affine(expr, outer_ivs)
                                        for expr in output_map.results)
                                  for output_map in output_maps]
                initial = [initializer.at(index)
                           for initializer, index in zip(initializers, output_indexes)]
                def reduce_body(reduction_iv, accumulators):
                    ivs = (*outer_ivs, reduction_iv)
                    self.ivs = ivs
                    indexes = [tuple(self._affine(expr, ivs) for expr in amap.results)
                               for amap in maps[:n_in]]
                    args = [tensor.at(index) for tensor, index in zip(inputs, indexes)]
                    values = self._scalar_results(
                        op.regions[0].blocks[0], [*args, *accumulators])
                    if len(values) != len(outputs):
                        raise LoweringDeclined(
                            "generic reduction yield count disagrees with outs",
                            op=op.name)
                    return [self._round(value, output.ety)
                            for value, output in zip(values, outputs)]
                results = self._for_range_values(bounds[-1], reduce_body, initial)
                for output, index, result in zip(outputs, output_indexes, results):
                    output.write(index, result)
            self.indices(bounds[:-1], outer_body)
            self.ivs = ()
            for result, output in zip(op.results, outputs):
                self.vals[result] = output
            return
        # A linalg `outs` operand is an accumulator only when the scalar body reads its
        # corresponding block argument.  For a proven full-domain pointwise map, the output
        # indexing map is a permutation and every iterator is parallel, so every result element
        # is overwritten exactly once.  If fusion was refused above (for example because the
        # result fans out), copying a tensor.empty/fill initializer into the fresh allocation is
        # still dead dynamic work and can be omitted without changing any source computation.
        pointwise = self._pointwise_info(op)
        if pointwise is not None and len(output_values) == 1:
            _, shape, _, _, _ = pointwise
            init = initializers[0]
            outputs = [self.storage(shape, init.ety)]
        else:
            outputs = [self.materialize(value) for value in initializers]
        def body(ivs):
            self.ivs = ivs
            indexes = [tuple(self._affine(expr, ivs) for expr in amap.results) for amap in maps]
            args = [tensor.at(index) for tensor, index in zip(inputs + outputs, indexes)]
            values = self._scalar_results(op.regions[0].blocks[0], args)
            if len(values) != len(outputs):
                raise LoweringDeclined(
                    "generic yield count disagrees with outs", op=op.name)
            for output, index, value in zip(outputs, indexes[n_in:], values):
                output.write(index, self._round(value, output.ety))
        self.indices(bounds, body)
        self.ivs = ()
        for result, output in zip(op.results, outputs):
            self.vals[result] = output

    def _t_linalg_reduce(self, op):
        n_in = len(op.operands) // 2
        inputs = [self.get(value) for value in op.operands[:n_in]]
        initializers = [self.get(value) for value in op.operands[n_in:]]
        if len(initializers) != 1:
            raise LoweringDeclined("loop reduction requires one result", op=op.name)
        dims = self._static(op, "dimensions")

        # A reduction over the final source dimension visits one complete output
        # element at a time in the same row-major order as the generic loop below.
        # Keep that element in an exact typed CFG recurrence: the initializer is
        # read once, every source element and scalar combine remains ordered, and
        # the final value is written once after the innermost reduction completes.
        source_shape = inputs[0].shape
        initializer = initializers[0]
        final_dimension = (
            n_in == 1
            and len(source_shape) >= 1
            and dims == [len(source_shape) - 1]
            and initializer.shape == source_shape[:-1]
            and initializer.ety == inputs[0].ety
        )
        if final_dimension:
            # A unique pointwise consumer with a bijective input map reads every reduced
            # element exactly once. Keep the scalar recurrence lazy in that case: the
            # consumer's parallel loop supplies the outer coordinates and this callback
            # emits the original innermost reduction loop. This removes only the
            # reduction-result temporary while preserving the initializer read, reduction
            # order, and per-step source-width rounding.
            result_value = op.results[0]
            uses = list(result_value.uses)
            lazy_consumer = None
            if (len(uses) == 1
                    and result_value not in getattr(self, "_segment_outputs", set())):
                use = uses[0]
                consumer = use.operation
                info = (self._pointwise_info(consumer)
                        if consumer in getattr(self, "_segment_ops", set()) else None)
                if info is not None:
                    consumer_inputs, _, maps, _, _ = info
                    consumed = maps[use.index].results if use.index < consumer_inputs else ()
                    if (len(consumed) == len(initializer.shape)
                            and all(isinstance(expr, AffineDimExpr) for expr in consumed)
                            and sorted(expr.position for expr in consumed)
                                == list(range(len(initializer.shape)))):
                        lazy_consumer = consumer
            if lazy_consumer is not None:
                def read(outer_ivs):
                    initial = initializer.at(outer_ivs)
                    def reduce_body(reduction_iv, accumulator):
                        ivs = (*outer_ivs, reduction_iv)
                        prior_ivs = self.ivs
                        try:
                            self.ivs = ivs
                            value = self._scalar_block(
                                op.regions[0].blocks[0],
                                [inputs[0].at(ivs), accumulator])
                            return self._round(value, initializer.ety)
                        finally:
                            self.ivs = prior_ivs
                    return self.fb.for_range(
                        source_shape[-1], reduce_body, initial=initial)
                self.vals[result_value] = LoopTensor(
                    initializer.shape, initializer.ety, read)
                return
            output = self.storage(initializer.shape, initializer.ety)
            def outer_body(outer_ivs):
                initial = initializer.at(outer_ivs)
                def reduce_body(reduction_iv, accumulator):
                    ivs = (*outer_ivs, reduction_iv)
                    self.ivs = ivs
                    value = self._scalar_block(
                        op.regions[0].blocks[0],
                        [inputs[0].at(ivs), accumulator])
                    return self._round(value, output.ety)
                result = self.fb.for_range(
                    source_shape[-1], reduce_body, initial=initial)
                output.write(outer_ivs, result)
            self.indices(source_shape[:-1], outer_body)
            self.ivs = ()
            self.vals[op.results[0]] = output
            return

        outputs = [self.materialize(value) for value in initializers]
        def body(ivs):
            self.ivs = ivs
            index = tuple(iv for d, iv in enumerate(ivs) if d not in dims)
            args = [tensor.at(ivs) for tensor in inputs] + [outputs[0].at(index)]
            outputs[0].write(index, self._round(self._scalar_block(op.regions[0].blocks[0], args), outputs[0].ety))
        self.indices(inputs[0].shape, body)
        self.ivs = ()
        self.vals[op.results[0]] = outputs[0]

    def _t_linalg_matmul(self, op):
        a, b, init = [self.get(value) for value in op.operands]
        out = self.materialize(init)
        def body(ivs):
            i, j, k = ivs
            add, mul = (self.fb.add_i, self.fb.mul_i) if is_int(out.ety) else (self.fp.fadd, self.fp.fmul)
            out.write((i,j), self._round(add(out.at((i,j)), mul(a.at((i,k)), b.at((k,j)))), out.ety))
        self.indices((a.shape[0], b.shape[1], a.shape[1]), body)
        self.vals[op.results[0]] = out
