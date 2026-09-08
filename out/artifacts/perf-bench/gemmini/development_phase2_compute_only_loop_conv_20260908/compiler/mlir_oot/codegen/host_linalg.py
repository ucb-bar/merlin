"""Compiler-generated CPU lane for a `linalg-on-tensors` region.

This is a LOWERING, not an interpreter of values: it walks the module's real IR and, for every
tensor SSA value, materialises the *SSA values of the emitted kernel* that hold that tensor's
elements.  Ops are handled by their structural definition — a `linalg.generic` by walking the
iteration space its own `indexing_maps` and operand extents imply, a `linalg.reduce` by its
`dimensions`, a `tensor.expand_shape` by its result type — so nothing here is keyed on a
capsule, a shape, an op ordering, or a pattern name.  A construct with no rule DECLINES.

The result is straight-line f32 code in the single block `FnBuilder` owns (see
`fpbuilder.py` for why it is branch-free), reading the kernel's pointer arguments and storing
to the output pointer.  It emits NO accelerator instruction: these regions are placed on the
host lane, and the capsules that carry them forbid the on-mesh lane.
"""
from __future__ import annotations

from typing import Any

from xdsl.ir import Block, Operation, SSAValue
from xdsl.ir.affine import (AffineBinaryOpExpr, AffineConstantExpr, AffineDimExpr,
                            AffineExpr, AffineBinaryOpKind)
from xdsl.dialects.builtin import (BFloat16Type, FloatAttr, Float16Type, Float32Type,
                                   Float64Type, IndexType, IntegerAttr, IntegerType,
                                   TensorType)

from ..lowering.plan import LoweringDeclined
from ..tables import rtl_facts as F
from .fpbuilder import FpBuilder
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

#: Float element types the CPU lane materialises.  A float tensor is carried as f32 SSA values;
#: the declared type decides the memory format and the precision each result is rounded back to.
_ELEM_NAMES = {Float32Type: "f32", BFloat16Type: "bf16", Float16Type: "f16",
               Float64Type: "f64"}

#: An INTEGER tensor is carried in the integer domain (i64 SSA values) rather than as a float:
#: linalg integer arithmetic is exact and modular in the declared width, and re-deriving it
#: through f32 would round every value past 2**24 and would not wrap at all.  The two domains
#: are bridged only where the IR bridges them (`arith.sitofp` / `arith.fptosi`).
INT_WIDTHS = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


def is_int(ety: str) -> bool:
    return ety in INT_WIDTHS

#: Straight-line budget: past this the lowering declines rather than emitting an artifact
#: nobody can assemble (the kernel is single-block, so nothing can be rolled into a loop).
HOST_LINALG_ELEMENT_BUDGET = 400_000


def attr_of(op, key: str):
    """An op's `key`, whether it rides in properties or attributes -- NEVER via `a or b`.

    `IntegerAttr(0)` and an empty `ArrayAttr` are FALSY, so the obvious
    `properties.get(k) or attributes.get(k)` silently misses exactly the values a zero
    accumulator init and an empty static-offset list are made of.
    """
    found = op.properties.get(key)
    return op.attributes.get(key) if found is None else found


def elem_name(ty) -> str:
    if isinstance(ty, IndexType):
        return "i64"
    if isinstance(ty, IntegerType):
        name = f"i{int(ty.width.data)}"
        if name not in INT_WIDTHS:
            raise LoweringDeclined(
                f"the CPU lane has no scalar format for integer width {ty.width.data}",
                op="host_lane")
        return name
    name = _ELEM_NAMES.get(type(ty))
    if name is None:
        raise LoweringDeclined(
            f"the CPU lane has no scalar format for element type {ty}", op="host_lane")
    return name


def tensor_shape(ty) -> tuple[int, ...]:
    if not isinstance(ty, TensorType):
        raise LoweringDeclined(f"expected a tensor type, got {ty}", op="host_lane")
    return tuple(int(d) for d in ty.get_shape())


def row_pitch(cols: int) -> int:
    """DRAM row pitch in elements: the kernel ABI pads every row to a multiple of DIM."""
    return -(-int(cols) // F.DIM) * F.DIM


def _strides(shape: tuple[int, ...]) -> list[int]:
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return out


class TensorVal:
    """One tensor SSA value, as the emitted kernel's own scalar SSA values (row-major)."""

    __slots__ = ("shape", "elems", "ety")

    def __init__(self, shape: tuple[int, ...], elems: list[SSAValue], ety: str):
        self.shape = tuple(shape)
        self.elems = elems
        self.ety = ety

    @property
    def strides(self) -> list[int]:
        return _strides(self.shape)

    def at(self, index: tuple[int, ...]) -> SSAValue:
        flat = 0
        for i, s in zip(index, self.strides):
            flat += i * s
        return self.elems[flat]


def _iter_space(bounds: list[int]):
    if not bounds:
        yield ()
        return
    idx = [0] * len(bounds)
    total = 1
    for b in bounds:
        total *= b
    for _ in range(total):
        yield tuple(idx)
        for d in range(len(bounds) - 1, -1, -1):
            idx[d] += 1
            if idx[d] < bounds[d]:
                break
            idx[d] = 0


def estimate_cost(ops) -> int:
    """A cheap upper-ish bound on the straight-line element evaluations `ops` would need.

    `_charge` already refuses past the budget, but only AFTER emitting up to it -- which on a whole
    model is minutes of work thrown away, and an entrypoint that takes minutes to say "no" is a
    timeout rather than a decline.  This walks the op list once, reading extents off the types.
    """
    total = 0
    for op in ops:
        widest = 0
        for value in list(op.results) + list(op.operands):
            ty = value.type
            if not isinstance(ty, TensorType):
                continue
            n = 1
            for d in ty.get_shape():
                n *= int(d)
            widest = max(widest, n)
        if op.name in ("linalg.matmul", "linalg.batch_matmul"):
            lhs = op.operands[0].type
            if isinstance(lhs, TensorType):
                shape = [int(d) for d in lhs.get_shape()]
                widest *= max(1, shape[-1])
        total += widest
    return total


class HostLinalg:
    """Lower one `func.func` of `linalg-on-tensors` into the CPU lane's straight-line code."""

    def __init__(self, fp: FpBuilder, arg_ptrs: list[SSAValue], out_ptrs: list[SSAValue]):
        self.fp = fp
        self.arg_ptrs = arg_ptrs
        self.out_ptrs = out_ptrs
        self.vals: dict[SSAValue, Any] = {}
        self.elements = 0
        #: the iteration point the region body currently being lowered is at, for `linalg.index`
        self.ivs: tuple[int, ...] = ()

    # -- helpers -----------------------------------------------------------------------------
    def _charge(self, n: int) -> None:
        self.elements += int(n)
        if self.elements > HOST_LINALG_ELEMENT_BUDGET:
            raise LoweringDeclined(
                f"the CPU-lane program for this region needs more than "
                f"{HOST_LINALG_ELEMENT_BUDGET} straight-line element evaluations "
                f"({self.elements} so far) and the emitted kernel must stay single-block",
                op="host_lane")

    def _round(self, value: SSAValue, ety: str) -> SSAValue:
        """Bring a just-computed value back to what its declared element type can hold."""
        if ety == "bf16":
            return self.fp.round_bf16(value)
        if is_int(ety):
            return self.fp.fb.wrap_int(value, INT_WIDTHS[ety])
        return value

    def zero(self, ety: str) -> SSAValue:
        """The additive identity in `ety`'s own domain."""
        return self.fp.fb.const(0) if is_int(ety) else self.fp.fconst(0.0)

    def get(self, value: SSAValue) -> Any:
        hit = self.vals.get(value)
        if hit is None:
            raise LoweringDeclined(
                f"the CPU lane has no value for {value} (its producer was not lowered)",
                op="host_lane")
        return hit

    # -- memory ------------------------------------------------------------------------------
    def load_arg(self, ptr: SSAValue, ty) -> TensorVal:
        shape = tensor_shape(ty)
        ety = elem_name(ty.get_element_type())
        cols = shape[-1] if shape else 1
        pitch = row_pitch(cols)
        rows = 1
        for d in shape[:-1]:
            rows *= d
        self._charge(rows * cols)
        elems: list[SSAValue] = []
        for r in range(rows):
            for c in range(cols):
                index = self.fp.fb.const(r * pitch + c)
                elems.append(self.fp.fb.load_i64(ptr, index, ety) if is_int(ety)
                             else self.fp.load(ptr, index, ety))
        return TensorVal(shape, elems, ety)

    def store_result(self, ptr: SSAValue, val: TensorVal) -> None:
        cols = val.shape[-1] if val.shape else 1
        pitch = row_pitch(cols)
        rows = 1
        for d in val.shape[:-1]:
            rows *= d
        for r in range(rows):
            for c in range(cols):
                value = val.elems[r * cols + c]
                index = self.fp.fb.const(r * pitch + c)
                if is_int(val.ety):
                    self.fp.fb.store_i64(value, ptr, index, val.ety)
                else:
                    self.fp.store(value, ptr, index, val.ety)

    # -- affine ------------------------------------------------------------------------------
    def _affine(self, expr: AffineExpr, ivs: tuple[int, ...]) -> int:
        if isinstance(expr, AffineDimExpr):
            return ivs[expr.position]
        if isinstance(expr, AffineConstantExpr):
            return int(expr.value)
        if isinstance(expr, AffineBinaryOpExpr):
            lhs = self._affine(expr.lhs, ivs)
            rhs = self._affine(expr.rhs, ivs)
            kind = expr.kind
            if kind is AffineBinaryOpKind.Add:
                return lhs + rhs
            if kind is AffineBinaryOpKind.Mul:
                return lhs * rhs
            if kind is AffineBinaryOpKind.FloorDiv:
                return lhs // rhs
            if kind is AffineBinaryOpKind.CeilDiv:
                return -(-lhs // rhs)
            if kind is AffineBinaryOpKind.Mod:
                return lhs % rhs
        raise LoweringDeclined(
            f"the CPU lane cannot evaluate the affine expression {expr}", op="host_lane")

    # -- scalar region -----------------------------------------------------------------------
    def _scalar_block(self, block: Block, args: list[SSAValue]) -> SSAValue:
        env: dict[SSAValue, SSAValue] = dict(zip(block.args, args))

        def read(value: SSAValue) -> SSAValue:
            """A region operand is either a block argument or a value CAPTURED from outside it."""
            hit = env.get(value)
            return self.get(value) if hit is None else hit

        for op in block.ops:
            if op.name == "linalg.yield":
                return read(op.operands[0])
            env[op.results[0]] = self._scalar_op(op, [read(o) for o in op.operands])
        raise LoweringDeclined("a linalg region did not yield a value", op="host_lane")

    def _scalar_op(self, op: Operation, ins: list[SSAValue]) -> SSAValue:
        fp = self.fp
        name = op.name
        ety = elem_name(op.results[0].type) if op.results else "f32"
        simple = {
            "arith.addf": fp.fadd, "arith.subf": fp.fsub, "arith.mulf": fp.fmul,
            "arith.divf": fp.fdiv, "arith.maximumf": fp.fmax, "arith.minimumf": fp.fmin,
            "arith.maxnumf": fp.fmax, "arith.minnumf": fp.fmin,
            "math.powf": fp.powf,
        }
        unary = {
            "arith.negf": fp.fneg, "math.exp": fp.exp, "math.erf": fp.erf,
            "math.rsqrt": fp.rsqrt, "math.sqrt": fp.sqrt, "math.tanh": fp.tanh,
            "math.absf": fp.fabs, "math.log": fp.log,
            "math.roundeven": fp.round_near_even,
        }
        if name in simple:
            return self._round(simple[name](ins[0], ins[1]), ety)
        if name in unary:
            return self._round(unary[name](ins[0]), ety)
        if name == "tensor.extract":
            source = ins[0]
            if not isinstance(source, TensorVal):
                raise LoweringDeclined(
                    "`tensor.extract` reads something that is not a materialised tensor",
                    op=name)
            index = []
            for operand, emitted in zip(op.operands[1:], ins[1:]):
                position = self.static_int(operand)
                if position is None:
                    position = constant_integer(emitted)
                if position is None:
                    raise LoweringDeclined(
                        "`tensor.extract` uses a data-dependent index; the emitted kernel is "
                        "straight-line and select-free, so a runtime gather has no lowering",
                        op=name)
                index.append(position)
            return source.at(tuple(index))
        if name == "linalg.index":
            dim = attr_of(op, "dim")
            position = int(getattr(getattr(dim, "value", dim), "data", 0))
            if position >= len(self.ivs):
                raise LoweringDeclined(
                    f"`linalg.index` names dimension {position} and the enclosing iteration "
                    f"space has {len(self.ivs)}", op=name)
            return fp.fb.const(self.ivs[position])
        if name == "arith.constant":
            return self._const(op)
        if name in ("arith.extf", "arith.truncf"):
            return self._round(ins[0], ety)
        fb = fp.fb
        int_binary = {
            "arith.addi": fb.add_i, "arith.subi": fb.sub_i, "arith.muli": fb.mul_i,
            "arith.andi": fb.and_i, "arith.ori": fb.or_i, "arith.xori": fb.xor_i,
            "arith.divsi": fb.sdiv_i, "arith.divui": fb.udiv_i,
            "arith.remsi": fb.srem_i, "arith.remui": fb.urem_i,
            "arith.shli": fb.shl_v, "arith.shrsi": fb.ashr_v, "arith.shrui": fb.lshr_v,
            "arith.maxsi": fb.smax, "arith.minsi": fb.smin,
        }
        if name in int_binary:
            return self._round(int_binary[name](ins[0], ins[1]), ety)
        if name in ("arith.negi",):
            return self._round(fb.sub_i(fb.const(0), ins[0]), ety)
        if name in ("math.absi", "math.absi.default"):
            mask = fb.ashr_i(ins[0], 63)
            return self._round(fb.sub_i(fb.xor_i(ins[0], mask), mask), ety)
        # -- the two domain bridges, exactly where the IR spells them --
        if name in ("arith.sitofp", "arith.uitofp"):
            return self._round(fp.sitofp(ins[0]), ety)
        if name in ("arith.fptosi", "arith.fptoui"):
            return self._round(fp.fptosi(ins[0]), ety)
        # -- integer width changes: the value is already carried in i64 --
        if name in ("arith.extsi", "arith.trunci", "arith.index_cast", "arith.bitcast"):
            return self._round(ins[0], ety)
        if name == "arith.extui":
            width = INT_WIDTHS[elem_name(op.operands[0].type)]
            if width >= 64:
                return ins[0]
            return fb.and_i(ins[0], fb.const((1 << width) - 1))
        raise LoweringDeclined(
            f"the CPU lane has no rule for the scalar op `{name}`", op=name)

    def static_int(self, value: SSAValue) -> int | None:
        """The integer `value` holds at COMPILE time, or None if it is data-dependent.

        The emitted kernel is straight-line and select-free, so an index has to be resolvable
        while the program is being written.  Loop indices are (the iteration point is known at
        each step) and so is integer arithmetic over them; a value read out of a tensor is not,
        and saying so is a decline rather than a guessed address.
        """
        producer = value.owner if isinstance(value.owner, Operation) else None
        if producer is None:
            return None
        name = producer.name
        if name == "arith.constant":
            attr = attr_of(producer, "value")
            data = getattr(getattr(attr, "value", None), "data", None)
            return None if data is None else int(data)
        if name == "linalg.index":
            dim = attr_of(producer, "dim")
            position = int(getattr(getattr(dim, "value", dim), "data", 0))
            return self.ivs[position] if position < len(self.ivs) else None
        if name in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui",
                    "arith.trunci"):
            return self.static_int(producer.operands[0])
        binary = {"arith.addi": lambda a, b: a + b,
                  "arith.subi": lambda a, b: a - b,
                  "arith.muli": lambda a, b: a * b}
        if name in binary:
            lhs = self.static_int(producer.operands[0])
            rhs = self.static_int(producer.operands[1])
            return None if lhs is None or rhs is None else binary[name](lhs, rhs)
        return None

    def _const(self, op: Operation) -> SSAValue:
        attr = attr_of(op, "value")
        ety = elem_name(op.results[0].type) if op.results else "f32"
        if not isinstance(attr, (FloatAttr, IntegerAttr)):
            raise LoweringDeclined(
                f"the CPU lane cannot materialise the constant {attr}", op="arith.constant")
        value = attr.value.data
        return self.fp.fb.const(int(value)) if is_int(ety) else self.fp.fconst(float(value))

    # -- tensor ops --------------------------------------------------------------------------
    def run(self, func_op: Operation) -> list[TensorVal]:
        block = func_op.regions[0].blocks[0]
        if len(block.args) != len(self.arg_ptrs):
            raise LoweringDeclined(
                f"@{func_op.properties['sym_name'].data} takes {len(block.args)} tensors but the "
                f"kernel ABI resolved {len(self.arg_ptrs)} input pointers", op="host_lane")
        for arg, ptr in zip(block.args, self.arg_ptrs):
            self.vals[arg] = self.load_arg(ptr, arg.type)
        results: list[TensorVal] = []
        for op in block.ops:
            if op.name == "func.return":
                results = [self.get(o) for o in op.operands]
                break
            self._tensor_op(op)
        if len(results) != len(self.out_ptrs):
            raise LoweringDeclined(
                f"the region returns {len(results)} tensors but the kernel ABI resolved "
                f"{len(self.out_ptrs)} output pointers", op="host_lane")
        for val, ptr in zip(results, self.out_ptrs):
            self.store_result(ptr, val)
        return results

    def run_segment(self, ops: list[Operation],
                    inputs: list[tuple[SSAValue, SSAValue]],
                    outputs: list[tuple[SSAValue, SSAValue]]) -> None:
        """Lower ONE host-lane segment of a mixed-lane program.

        `inputs` binds an SSA value to the DRAM pointer holding it (a kernel argument, or the
        buffer a preceding mesh contraction committed to); `outputs` names the values this
        segment has to leave in DRAM for whatever runs next.  The instance is reused across the
        segments of one kernel, so a value computed in an earlier segment is still live here and
        is not reloaded.
        """
        for value, ptr in inputs:
            if value not in self.vals:
                self.vals[value] = self.load_arg(ptr, value.type)
        for op in ops:
            self._tensor_op(op)
        for value, ptr in outputs:
            self.store_result(ptr, self.get(value))

    def _tensor_op(self, op: Operation) -> None:
        handler = getattr(self, "_t_" + op.name.replace(".", "_"), None)
        if handler is None:
            raise LoweringDeclined(
                f"the CPU lane has no rule for `{op.name}`", op=op.name)
        handler(op)

    # scalar producer at tensor level
    def _t_tensor_concat(self, op: Operation) -> None:
        sources = [self.get(value) for value in op.operands]
        ty = op.results[0].type
        shape = tensor_shape(ty)
        axis = int(attr_of(op, "dim").value.data)
        if not 0 <= axis < len(shape) or not sources:
            raise LoweringDeclined("invalid tensor.concat axis or empty operands", op=op.name)
        if any(len(src.shape) != len(shape) or any(
                src.shape[d] != shape[d] for d in range(len(shape)) if d != axis)
               for src in sources) or sum(src.shape[axis] for src in sources) != shape[axis]:
            raise LoweringDeclined("tensor.concat operand extents disagree with result", op=op.name)
        elems = []
        for index in _iter_space(list(shape)):
            offset = index[axis]
            for src in sources:
                if offset < src.shape[axis]:
                    local = list(index)
                    local[axis] = offset
                    elems.append(src.at(tuple(local)))
                    break
                offset -= src.shape[axis]
        self._charge(len(elems))
        self.vals[op.results[0]] = TensorVal(shape, elems, elem_name(ty.get_element_type()))

    def _t_arith_constant(self, op: Operation) -> None:
        self.vals[op.results[0]] = self._const(op)

    def _t_tensor_empty(self, op: Operation) -> None:
        ty = op.results[0].type
        shape = tensor_shape(ty)
        ety = elem_name(ty.get_element_type())
        n = 1
        for d in shape:
            n *= d
        self._charge(n)
        self.vals[op.results[0]] = TensorVal(shape, [self.zero(ety)] * n, ety)

    def _t_tensor_splat(self, op: Operation) -> None:
        ty = op.results[0].type
        shape = tensor_shape(ty)
        ety = elem_name(ty.get_element_type())
        n = 1
        for d in shape:
            n *= d
        self._charge(n)
        self.vals[op.results[0]] = TensorVal(shape, [self.get(op.operands[0])] * n, ety)

    def _reshape(self, op: Operation) -> None:
        src = self.get(op.operands[0])
        ty = op.results[0].type
        self.vals[op.results[0]] = TensorVal(tensor_shape(ty), list(src.elems),
                                             elem_name(ty.get_element_type()))

    _t_tensor_expand_shape = _reshape
    _t_tensor_collapse_shape = _reshape
    _t_tensor_reshape = _reshape
    _t_tensor_bitcast = _reshape

    def _static(self, op: Operation, key: str) -> list[int]:
        attr = attr_of(op, key)
        if attr is None:
            raise LoweringDeclined(f"`{op.name}` declares no {key}", op=op.name)
        return [int(v) for v in attr.get_values()]

    def _t_tensor_insert_slice(self, op: Operation) -> None:
        source = self.get(op.operands[0])
        dest = self.get(op.operands[1])
        offsets = self._static(op, "static_offsets")
        sizes = self._static(op, "static_sizes")
        strides = self._static(op, "static_strides")
        if len(sizes) != len(dest.shape):
            raise LoweringDeclined(
                "tensor.insert_slice with a rank-reducing slice has no CPU-lane rule",
                op=op.name)
        elems = list(dest.elems)
        dstr = dest.strides
        self._charge(len(elems))
        for idx in _iter_space(list(sizes)):
            flat = sum((offsets[d] + idx[d] * strides[d]) * dstr[d] for d in range(len(idx)))
            elems[flat] = source.at(idx)
        ty = op.results[0].type
        self.vals[op.results[0]] = TensorVal(tensor_shape(ty), elems,
                                             elem_name(ty.get_element_type()))

    def _t_tensor_extract_slice(self, op: Operation) -> None:
        source = self.get(op.operands[0])
        offsets = self._static(op, "static_offsets")
        sizes = self._static(op, "static_sizes")
        strides = self._static(op, "static_strides")
        sstr = source.strides
        ty = op.results[0].type
        out_shape = tensor_shape(ty)
        self._charge(len(sizes) and 1 or 1)
        elems: list[SSAValue] = []
        for idx in _iter_space(list(sizes)):
            flat = sum((offsets[d] + idx[d] * strides[d]) * sstr[d] for d in range(len(idx)))
            elems.append(source.elems[flat])
        self._charge(len(elems))
        self.vals[op.results[0]] = TensorVal(out_shape, elems,
                                             elem_name(ty.get_element_type()))

    def _t_linalg_fill(self, op: Operation) -> None:
        scalar = self.get(op.operands[0])
        ty = op.results[0].type
        shape = tensor_shape(ty)
        n = 1
        for d in shape:
            n *= d
        self._charge(n)
        self.vals[op.results[0]] = TensorVal(shape, [scalar] * n,
                                             elem_name(ty.get_element_type()))

    def _t_linalg_transpose(self, op: Operation) -> None:
        src = self.get(op.operands[0])
        perm = self._static(op, "permutation")
        ty = op.results[0].type
        out_shape = tensor_shape(ty)
        self._charge(len(src.elems))
        elems: list[SSAValue] = []
        for idx in _iter_space(list(out_shape)):
            src_idx = [0] * len(perm)
            for d, p in enumerate(perm):
                src_idx[p] = idx[d]
            elems.append(src.at(tuple(src_idx)))
        self.vals[op.results[0]] = TensorVal(out_shape, elems,
                                             elem_name(ty.get_element_type()))

    def _t_linalg_broadcast(self, op: Operation) -> None:
        src = self.get(op.operands[0])
        dims = self._static(op, "dimensions")
        ty = op.results[0].type
        out_shape = tensor_shape(ty)
        keep = [d for d in range(len(out_shape)) if d not in dims]
        self._charge(len(out_shape) and 1)
        elems: list[SSAValue] = []
        for idx in _iter_space(list(out_shape)):
            elems.append(src.at(tuple(idx[d] for d in keep)))
        self._charge(len(elems))
        self.vals[op.results[0]] = TensorVal(out_shape, elems,
                                             elem_name(ty.get_element_type()))

    def _t_linalg_matmul(self, op: Operation) -> None:
        a, b = self.get(op.operands[0]), self.get(op.operands[1])
        c = self.get(op.operands[2])
        m, k = a.shape
        k2, n = b.shape
        if k != k2 or c.shape != (m, n):
            raise LoweringDeclined(
                f"linalg.matmul extents {a.shape} x {b.shape} -> {c.shape} do not contract",
                op=op.name)
        ty = op.results[0].type
        ety = elem_name(ty.get_element_type())
        self._charge(m * n * k)
        add = self.fp.fb.add_i if is_int(ety) else self.fp.fadd
        mul = self.fp.fb.mul_i if is_int(ety) else self.fp.fmul
        elems: list[SSAValue] = []
        for i in range(m):
            for j in range(n):
                acc = c.at((i, j))
                for p in range(k):
                    acc = self._round(add(acc, mul(a.at((i, p)), b.at((p, j)))), ety)
                elems.append(self._round(acc, ety))
        self.vals[op.results[0]] = TensorVal((m, n), elems, ety)

    def _t_linalg_reduce(self, op: Operation) -> None:
        n_in = len(op.operands) // 2
        ins = [self.get(o) for o in op.operands[:n_in]]
        inits = [self.get(o) for o in op.operands[n_in:]]
        dims = set(self._static(op, "dimensions"))
        src = ins[0]
        out_shape = tuple(d for i, d in enumerate(src.shape) if i not in dims)
        ty = op.results[0].type
        ety = elem_name(ty.get_element_type())
        acc = list(inits[0].elems)
        out_strides = _strides(out_shape)
        self._charge(len(src.elems))
        body = op.regions[0].blocks[0]
        for idx in _iter_space(list(src.shape)):
            keep = tuple(v for i, v in enumerate(idx) if i not in dims)
            flat = sum(v * s for v, s in zip(keep, out_strides))
            args = [t.at(idx) for t in ins] + [acc[flat]]
            self.ivs = idx
            acc[flat] = self._round(self._scalar_block(body, args), ety)
        self.ivs = ()
        self.vals[op.results[0]] = TensorVal(out_shape, acc, ety)

    def _t_linalg_map(self, op: Operation) -> None:
        ins = [self.get(o) for o in op.operands[:-1]]
        ty = op.results[0].type
        out_shape = tensor_shape(ty)
        ety = elem_name(ty.get_element_type())
        body = op.regions[0].blocks[0]
        self._charge(len(ins[0].elems))
        elems = [self._round(self._scalar_block(body, [t.elems[i] for t in ins]), ety)
                 for i in range(len(ins[0].elems))]
        self.vals[op.results[0]] = TensorVal(out_shape, elems, ety)

    def _t_linalg_generic(self, op: Operation) -> None:
        seg = op.properties.get("operandSegmentSizes")
        counts = [int(v) for v in seg.get_values()] if seg is not None else None
        n_in = counts[0] if counts else len(op.operands) - len(op.results)
        ins = [self.get(o) for o in op.operands[:n_in]]
        outs = [self.get(o) for o in op.operands[n_in:]]
        maps = [m.data for m in op.properties["indexing_maps"].data]
        iters = op.properties["iterator_types"].data
        n_dims = maps[0].num_dims

        bounds = [0] * n_dims
        for m, t in zip(maps, ins + outs):
            for pos, expr in enumerate(m.results):
                if isinstance(expr, AffineDimExpr):
                    bounds[expr.position] = t.shape[pos]
        for d, b in enumerate(bounds):
            if b == 0:
                raise LoweringDeclined(
                    f"linalg.generic iteration bound d{d} is not implied by any operand extent",
                    op=op.name)
        # A dim that only appears inside a compound expression still has to be bounded; the
        # window dims of an im2col gather are exactly that case, so take the bound from the
        # matching output extent when the output map pins it.
        total = 1
        for b in bounds:
            total *= b
        self._charge(total)

        acc = [list(o.elems) for o in outs]
        out_strides = [_strides(o.shape) for o in outs]
        body = op.regions[0].blocks[0]
        etys = [elem_name(r.type.get_element_type()) for r in op.results]
        for ivs in _iter_space(bounds):
            self.ivs = ivs
            args: list[SSAValue] = []
            for m, t in zip(maps[:n_in], ins):
                args.append(t.at(tuple(self._affine(e, ivs) for e in m.results)))
            flats: list[int] = []
            for k, (m, t) in enumerate(zip(maps[n_in:], outs)):
                idx = tuple(self._affine(e, ivs) for e in m.results)
                flat = sum(v * s for v, s in zip(idx, out_strides[k]))
                flats.append(flat)
                args.append(acc[k][flat])
            yielded = self._scalar_block(body, args)
            acc[0][flats[0]] = self._round(yielded, etys[0] if etys else outs[0].ety)
        self.ivs = ()
        for k, res in enumerate(op.results):
            ty = res.type
            self.vals[res] = TensorVal(tensor_shape(ty), acc[k],
                                       elem_name(ty.get_element_type()))

    _ = _iter_space          # keep the iteration helper reachable from this module's API
