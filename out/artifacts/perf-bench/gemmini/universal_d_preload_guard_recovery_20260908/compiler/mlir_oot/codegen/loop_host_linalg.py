"""Source-native host loops over memory-backed tensor values.

Scalar arithmetic reuses HostLinalg. Shapes and indexing maps determine loops;
no host tensor is expanded into a Python list of emitted scalar instructions.
View operations compose indexing functions without materializing copies.
"""
from math import prod

from xdsl.dialects import llvm
from xdsl.dialects.builtin import Float32Type, IntegerAttr, i16, i64
from xdsl.ir import SSAValue
from xdsl.ir.affine import AffineBinaryOpExpr, AffineConstantExpr, AffineDimExpr, AffineBinaryOpKind

from .builder import INT_TYPES
from .host_linalg import HostLinalg, _strides, attr_of, elem_name, is_int, row_pitch, tensor_shape
from ..lowering.plan import LoweringDeclined


class LoopTensor:
    def __init__(self, shape, ety, read, write=None):
        self.shape, self.ety, self.read, self.write = tuple(shape), ety, read, write

    def at(self, index):
        return self.read(tuple(index))


class LoopHostLinalg(HostLinalg):
    def __init__(self, *args, input_prologue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_prologue = input_prologue
        self._entry_args = ()

    @staticmethod
    def _source_name(op):
        """Recover the parsed name of an allowed unregistered input-dialect op."""
        if op.name != "builtin.unregistered":
            return op.name
        return getattr(getattr(op, "op_name", None), "data", op.name)

    def _tensor_op(self, op):
        name = self._source_name(op)
        if (self.input_prologue and name == "quant_ext.quantize_per_tensor"
                and getattr(op.attributes.get("prov.region_id"), "data", "")
                == self.input_prologue["region_id"]):
            index = int(self.input_prologue["input_arg_index"])
            if op.operands[0] is not self._entry_args[index]:
                raise LoweringDeclined(
                    "externalized quantizer is not rooted at the declared entry argument",
                    op="input_prologue")
            self.vals[op.results[0]] = self.load_arg(self.arg_ptrs[index], op.results[0].type)
            return
        handler = getattr(self, "_t_" + name.replace(".", "_"), None)
        if handler is None:
            raise LoweringDeclined(f"the CPU lane has no rule for `{name}`", op=name)
        handler(op)

    def run(self, func_op):
        block = func_op.regions[0].blocks[0]
        self._entry_args = tuple(block.args)
        if not self.input_prologue:
            return super().run(func_op)
        index = int(self.input_prologue["input_arg_index"])
        if len(block.args) != len(self.arg_ptrs) or not 0 <= index < len(block.args):
            raise LoweringDeclined("invalid external input-prologue argument", op="input_prologue")
        for i, (arg, ptr) in enumerate(zip(block.args, self.arg_ptrs)):
            if i != index:
                self.vals[arg] = self.load_arg(ptr, arg.type)
        results = []
        for op in block.ops:
            if op.name == "func.return":
                results = [self.get(value) for value in op.operands]
                break
            self._tensor_op(op)
        if len(results) != len(self.out_ptrs):
            raise LoweringDeclined("whole model result count disagrees with pointer ABI",
                                   op="host_lane")
        for value, ptr in zip(results, self.out_ptrs):
            self.store_result(ptr, value)
        return results

    def _charge(self, n):
        # Count logical work, not expanded code. Loops represent these iterations.
        self.elements += int(n)

    @property
    def fb(self):
        return self.fp.fb

    def value(self, item):
        return item if isinstance(item, SSAValue) else self.fb.const(int(item))

    def flat(self, index, shape):
        result = self.fb.const(0)
        for item, stride in zip(index, _strides(tuple(shape))):
            result = self.fb.add_i(result, self.fb.mul_i(self.value(item), self.fb.const(stride)))
        return result

    def unflat(self, index, shape):
        return tuple(self.fb.urem_i(self.fb.udiv_i(index, self.fb.const(stride)), self.fb.const(size))
                     for size, stride in zip(shape, _strides(tuple(shape))))

    def indices(self, bounds, body):
        def visit(index):
            if len(index) == len(bounds):
                body(tuple(index))
            else:
                self.fb.for_range(bounds[len(index)], lambda iv: visit([*index, iv]))
        visit([])

    def storage(self, shape, ety, ptr=None, padded=False):
        physical = tuple(shape[:-1]) + (row_pitch(shape[-1]),) if padded and shape else tuple(shape)
        if ptr is None:
            if ety not in INT_TYPES and ety not in ("f32", "bf16"):
                raise LoweringDeclined(f"loop storage format {ety} not implemented", op="host_lane")
            # bf16 values are computed as f32 SSA values but occupy their declared 16-bit
            # container in memory. FpBuilder.load/store performs the lossless widen/round-pack.
            ty = INT_TYPES[ety] if is_int(ety) else (i16 if ety == "bf16" else Float32Type())
            width = int(ty.width.data) if is_int(ety) or ety == "bf16" else 32
            nbytes = prod(physical) * ((width + 7) // 8)
            if self.scratch_alloc is not None:
                ptr = self.scratch_alloc(nbytes)
            else:
                allocation = llvm.AllocaOp(self.fb.const(prod(physical)), ty, alignment=64)
                allocation.attributes["merlin.host_storage_bytes"] = IntegerAttr(nbytes, i64)
                self.fb.prologue(allocation)
                ptr = allocation.results[0]
        def read(index):
            flat = self.flat(index, physical)
            return self.fb.load_i64(ptr, flat, ety) if is_int(ety) else self.fp.load(ptr, flat, ety)
        def write(index, value):
            flat = self.flat(index, physical)
            if is_int(ety):
                self.fb.store_i64(value, ptr, flat, ety)
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
        src, dest = [self.get(value) for value in op.operands[:2]]
        out = self.materialize(dest)
        offsets, sizes, strides = [self._static(op, key) for key in
                                   ("static_offsets", "static_sizes", "static_strides")]
        self.indices(sizes, lambda idx: out.write(tuple(self.fb.add_i(self.fb.const(offset),
            self.fb.mul_i(iv, self.fb.const(stride))) for iv, offset, stride in zip(idx, offsets, strides)),
            src.at(idx)))
        self.vals[op.results[0]] = out

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
        counts = self._static(op, "operandSegmentSizes")
        n_in = counts[0]
        inputs = [self.get(value) for value in op.operands[:n_in]]
        outputs = [self.materialize(self.get(value)) for value in op.operands[n_in:]]
        if len(outputs) != 1:
            raise LoweringDeclined("loop generic requires one result", op=op.name)
        maps = [attr.data for attr in attr_of(op, "indexing_maps")]
        bounds = [0] * maps[0].num_dims
        for amap, tensor in zip(maps, inputs + outputs):
            for d, expr in enumerate(amap.results):
                if isinstance(expr, AffineDimExpr):
                    bounds[expr.position] = tensor.shape[d]
        if any(bound <= 0 for bound in bounds):
            raise LoweringDeclined("generic loop has unresolved bound", op=op.name)
        def body(ivs):
            self.ivs = ivs
            indexes = [tuple(self._affine(expr, ivs) for expr in amap.results) for amap in maps]
            args = [tensor.at(index) for tensor, index in zip(inputs + outputs, indexes)]
            value = self._round(self._scalar_block(op.regions[0].blocks[0], args), outputs[0].ety)
            outputs[0].write(indexes[n_in], value)
        self.indices(bounds, body)
        self.ivs = ()
        self.vals[op.results[0]] = outputs[0]

    def _t_linalg_reduce(self, op):
        n_in = len(op.operands) // 2
        inputs = [self.get(value) for value in op.operands[:n_in]]
        outputs = [self.materialize(self.get(value)) for value in op.operands[n_in:]]
        if len(outputs) != 1:
            raise LoweringDeclined("loop reduction requires one result", op=op.name)
        dims = self._static(op, "dimensions")
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

    @staticmethod
    def _integer_property(op, name):
        attr = attr_of(op, name)
        value = getattr(getattr(attr, "value", attr), "data", None)
        if value is None:
            raise LoweringDeclined(f"quantization op declares no {name}", op="quantize")
        return int(value)

    @staticmethod
    def _scalar(tensor):
        if tensor.shape:
            raise LoweringDeclined("quantization scale/zero point is not scalar", op="quantize")
        return tensor.at(())

    def _dequantize(self, op, axis=None):
        source, scale, zero = [self.get(value) for value in op.operands]
        shape = tensor_shape(op.results[0].type)
        if source.shape != shape:
            raise LoweringDeclined("dequantize shape changes", op="dequantize")
        if axis is not None and (axis < 0 or axis >= len(shape)
                                 or scale.shape != (shape[axis],)
                                 or zero.shape != (shape[axis],)):
            raise LoweringDeclined("per-channel quantization parameters disagree with axis",
                                   op="dequantize")
        def read(index):
            channel = index[axis] if axis is not None else None
            s = scale.at((channel,)) if channel is not None else self._scalar(scale)
            z = zero.at((channel,)) if channel is not None else self._scalar(zero)
            centered = self.fb.sub_i(source.at(index), z)
            return self.fp.fmul(self.fp.sitofp(centered), s)
        self.vals[op.results[0]] = LoopTensor(shape, "f32", read)

    def _quantize(self, op, axis=None):
        source, scale, zero = [self.get(value) for value in op.operands]
        shape = tensor_shape(op.results[0].type)
        ety = elem_name(op.results[0].type.get_element_type())
        if source.shape != shape or not is_int(ety):
            raise LoweringDeclined("quantize input/result shape or dtype is unsupported",
                                   op="quantize")
        if axis is not None and (axis < 0 or axis >= len(shape)
                                 or scale.shape != (shape[axis],)
                                 or zero.shape != (shape[axis],)):
            raise LoweringDeclined("per-channel quantization parameters disagree with axis",
                                   op="quantize")
        lo = self._integer_property(op, "quant_min")
        hi = self._integer_property(op, "quant_max")
        def read(index):
            channel = index[axis] if axis is not None else None
            s = scale.at((channel,)) if channel is not None else self._scalar(scale)
            z = zero.at((channel,)) if channel is not None else self._scalar(zero)
            # TorchAO's PT2E contract is reciprocal-first in f32.  Replacing this with x / s is
            # not algebraically interchangeable at round-to-even boundaries and changed ResNet-50
            # activation codes at multiple graph cuts.
            reciprocal = self.fp.fdiv(self.fp.fconst(1.0), s)
            scaled = self.fp.fmul(source.at(index), reciprocal)
            rounded = self.fp.fptosi(self.fp.round_near_even(scaled))
            return self.fb.clamp(self.fb.add_i(rounded, z), lo, hi)
        self.vals[op.results[0]] = LoopTensor(shape, ety, read)

    def _t_quant_ext_dequantize_per_tensor(self, op):
        self._dequantize(op)

    def _t_quant_ext_dequantize_per_channel(self, op):
        self._dequantize(op, self._integer_property(op, "axis"))

    def _t_quant_ext_quantize_per_tensor(self, op):
        self._quantize(op)

    def _t_quant_ext_quantize_per_channel(self, op):
        self._quantize(op, self._integer_property(op, "axis"))
