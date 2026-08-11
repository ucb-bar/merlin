"""TTIR -> linalg-on-tensors. The single convergence point (INV-6).

This is where a Triton kernel stops being a Triton kernel and becomes an ordinary Merlin input.
Everything below it — contract inference, scheduling, interface materialization, the generated
target dialect — is code that existed before Triton did and that knows nothing about it. That is the
whole architecture in one sentence, and it is why the bridge lowers to linalg rather than to any
target dialect: going straight to a target would skip the compiler and leave a code emitter.

The translation has two halves. The addressing half (:mod:`merlin.triton.addressing`) re-raises
pointer arithmetic and the SPMD grid back into whole tensors — the hard half. This module is the
easy half: an abstract interpretation of the TTIR body where each value is either an index-space
affine expression, a tile of pointers, a mask, or a materialized tensor, and each op either advances
that state or emits a linalg op.

Three properties are held deliberately.

*Fail closed.* An op this module does not understand is an error naming the op, never a best-effort
translation. Silent approximation in a compiler frontend produces programs that run and are wrong.

*Account for every op.* The capability report tracks ops seen against ops lowered or deliberately
discarded, and a non-empty remainder aborts the translation. Triton emits dead range-check IR that
is genuinely safe to drop; "safe to drop" is therefore a decision that gets recorded, not a gap.

*Stay target-blind.* Nothing here reads a target name, contract or dialect plan. What comes out is
the same linalg-on-tensors a hand-written frontend would produce, and the router in
:mod:`merlin.compile_core` decides where it goes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import source
from .addressing import Affine, Conjunction, PointerTensor, Predicate, whole_tensor_access
from .diagnostics import BridgeError, CapabilityReport
from .spec import TritonKernelSpec

# The core-MLIR boundary. A surviving op outside these dialects means the bridge did not finish, and
# is an error rather than a warning.
CORE_DIALECTS = frozenset({"builtin", "func", "arith", "math", "tensor", "linalg", "scf", "cf"})

# Triton spells element types the way MLIR does, except for the `fp`/`f` prefix on floats.
_DTYPES = {"fp64": "f64", "fp32": "f32", "fp16": "f16", "bf16": "bf16",
           "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8", "i1": "i1"}

# MLIR's arith::CmpIPredicate ordinals, as carried by the `predicate` attribute.
_CMPI = {0: "eq", 1: "ne", 2: "slt", 3: "sle", 4: "sgt", 5: "sge",
         6: "ult", 7: "ule", 8: "ugt", 9: "uge"}

# Ops with no side effect, so an instance whose results are unused computes nothing and may be
# dropped. Triton emits a range check per index expression (extend to i64, compare against the i32
# limits, and) whose result nothing consumes; that is what this list is for. Membership is required
# rather than assumed, so an unrecognized dead op still fails closed.
_PURE = frozenset({
    "arith.constant", "arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast",
    "arith.addi", "arith.muli", "arith.subi", "arith.cmpi", "arith.andi", "arith.ori",
    "arith.addf", "arith.mulf", "arith.subf", "arith.divf",
    "tt.make_range", "tt.splat", "tt.expand_dims", "tt.broadcast", "tt.addptr",
    "tt.get_program_id", "tt.load", "tt.dot",
})

_ELEMENTWISE = {"arith.addf": "AddOp", "arith.addi": "AddOp",
                "arith.mulf": "MulOp", "arith.muli": "MulOp",
                "arith.subf": "SubOp", "arith.subi": "SubOp"}


@dataclass(frozen=True)
class FloatConst:
    """A floating-point constant. Kept apart from :class:`Affine`, which is integer index space."""

    value: float
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class Tensor:
    """A materialized value in the module being built."""

    value: Any
    shape: tuple[int, ...]


@dataclass(frozen=True)
class UnresolvedScalar:
    """A runtime scalar with no declared compile-time value — an error only if something uses it."""

    name: str


@dataclass
class BridgeResult:
    module: Any
    text: str
    report: CapabilityReport
    entry: str
    arg_names: list[str] = field(default_factory=list)
    result_names: list[str] = field(default_factory=list)


def to_linalg(ttir: source.TTIRModule, spec: TritonKernelSpec) -> BridgeResult:
    """Translate ``ttir`` to a linalg-on-tensors module, or fail closed explaining why not."""
    from xdsl.dialects.builtin import ModuleOp

    builder = _Bridge(ttir, spec)
    module: ModuleOp = builder.run()
    _check_core_only(module)
    return BridgeResult(module=module, text=_text(module), report=builder.report,
                        entry=spec.name, arg_names=builder.arg_names,
                        result_names=builder.result_names)


def _text(module) -> str:
    from merlin.xdsl_dialects._common import text
    return text(module)


def _check_core_only(module) -> None:
    offenders: dict[str, int] = {}
    for op in module.walk():
        dialect = op.dialect_name()
        if dialect not in CORE_DIALECTS:
            offenders[op.name] = offenders.get(op.name, 0) + 1
    if offenders:
        raise BridgeError(
            f"the bridge left non-core ops in its output: {sorted(offenders)}",
            hint=f"linalg-on-tensors is the convergence point; allowed dialects are "
                 f"{sorted(CORE_DIALECTS)}")


class _Bridge:
    """One translation. Holds the abstract environment and the ops being emitted."""

    def __init__(self, ttir: source.TTIRModule, spec: TritonKernelSpec) -> None:
        self.ttir = ttir
        self.spec = spec
        self.report = CapabilityReport(kernel_name=spec.name)
        self.env: dict[int, Any] = {}
        self.ops: list[Any] = []
        self.stored: dict[str, Any] = {}
        self.arg_names: list[str] = []
        self.result_names: list[str] = []
        self.grid = spec.grid.resolve(spec.constexprs, spec.assumptions)
        self.report.grid = self.grid
        self.constants = source.constant_table(ttir)

    # ---------------------------------------------------------------- element / tensor types

    def _element_type(self, dtype: str):
        from xdsl.dialects.builtin import BFloat16Type, Float16Type, Float32Type, Float64Type, IntegerType
        mlir = _DTYPES.get(dtype)
        if mlir is None:
            raise BridgeError(f"unsupported element type {dtype!r}",
                              hint=f"known types: {sorted(_DTYPES)}")
        return {"f64": Float64Type(), "f32": Float32Type(), "f16": Float16Type(),
                "bf16": BFloat16Type()}.get(mlir) or IntegerType(int(mlir[1:]))

    def _tensor_type(self, arg):
        from xdsl.dialects.builtin import TensorType
        return TensorType(self._element_type(arg.dtype), list(arg.shape or ()))

    # ---------------------------------------------------------------- driver

    def run(self):
        from xdsl.dialects.builtin import FunctionType, ModuleOp
        from xdsl.dialects.func import FuncOp, ReturnOp
        from xdsl.ir import Block, Region

        inputs = list(self.spec.inputs)
        outputs = list(self.spec.outputs)
        in_types = [self._tensor_type(a) for a in inputs]
        out_types = [self._tensor_type(a) for a in outputs]
        self.arg_names = [a.name for a in inputs]
        self.result_names = [a.name for a in outputs]

        block = Block(arg_types=in_types)
        self.block_values = {a.name: v for a, v in zip(inputs, block.args)}

        self._bind_parameters()
        self._interpret()

        missing = [a.name for a in outputs if a.name not in self.stored]
        if missing:
            raise BridgeError(
                f"nothing is stored to output argument(s) {missing}",
                hint="an argument declared effect='write' must be written by the kernel; either the "
                     "effect declaration is wrong or the store was not recognized")

        self.ops.append(ReturnOp(*[self.stored[a.name] for a in outputs]))
        block.add_ops(self.ops)
        fn = FuncOp(self.spec.name, FunctionType.from_lists(in_types, out_types), Region([block]))
        module = ModuleOp([fn])
        self.report.output_dialects = sorted({op.dialect_name() for op in module.walk()})
        return module

    def _bind_parameters(self) -> None:
        """Map the TTIR entry parameters onto the declared spec arguments, checking they agree."""
        params = source.entry_block_args(self.ttir)
        if len(params) != len(self.spec.args):
            raise BridgeError(
                f"kernel takes {len(params)} parameter(s) but the spec declares "
                f"{len(self.spec.args)}")
        for param, arg in zip(params, self.spec.args):
            ttir_type = str(param.get_type())
            pointee = source.pointee_dtype(ttir_type)
            if arg.kind == "pointer":
                if pointee is None:
                    raise BridgeError(
                        f"argument {arg.name!r} is declared a pointer but the kernel takes "
                        f"{ttir_type} there")
                if pointee != _DTYPES.get(arg.dtype):
                    raise BridgeError(
                        f"argument {arg.name!r} is declared {arg.dtype} but the kernel takes a "
                        f"pointer to {pointee}")
                # A rank-0 offset: a bare pointer is just a tile of one address, so `tt.addptr`
                # before the splat and after it are the same operation.
                self.env[param.id()] = PointerTensor(arg.name, Affine())
            else:
                if pointee is not None:
                    raise BridgeError(
                        f"argument {arg.name!r} is declared a scalar but the kernel takes "
                        f"{ttir_type} there")
                value = self.spec.assumptions.get(arg.name)
                self.env[param.id()] = (Affine(const=int(value)) if value is not None
                                        else UnresolvedScalar(arg.name))

    def _interpret(self) -> None:
        ops = source.walk_ops(self.ttir)
        used: set[int] = set()
        for op in ops:
            for i in range(op.get_num_operands()):
                used.add(op.get_operand(i).id())

        for op in ops:
            name = op.get_name()
            if name in ("builtin.module", "tt.func"):
                continue
            self.report.saw(name)
            results = [op.get_result(i) for i in range(op.get_num_results())]
            if results and all(r.id() not in used for r in results) and name in _PURE:
                # Triton emits a range check per index expression whose result nothing consumes.
                self.report.discarded(name)
                continue
            self._translate(op, name, results)

        unaccounted = self.report.unaccounted
        if unaccounted:
            raise BridgeError(
                f"the bridge did not account for {unaccounted} — translation is incomplete",
                hint="every op must be lowered or explicitly discarded; this is the guard against "
                     "emitting a module that quietly computes something else")

    # ---------------------------------------------------------------- op translation

    def _translate(self, op, name: str, results: list) -> None:
        handler = getattr(self, "_op_" + name.replace(".", "_"), None)
        if handler is None:
            raise BridgeError(
                f"no translation for {name}", op=name,
                hint="the bridge covers pointer arithmetic, masked load/store, tt.dot and "
                     "elementwise arith; anything else must be added deliberately, with a test")
        handler(op, results)
        self.report.lowered(name)

    def _operand(self, op, i: int):
        value = self.env.get(op.get_operand(i).id())
        if value is None:
            raise BridgeError(
                f"operand {i} of {op.get_name()} was produced by an op the bridge skipped",
                op=op.get_name())
        if isinstance(value, UnresolvedScalar):
            raise BridgeError(
                f"runtime scalar {value.name!r} is used in an address or mask but has no "
                "compile-time value",
                op=op.get_name(),
                hint=f"declare it in the spec, e.g. assumptions={{{value.name!r}: <extent>}} — the "
                     "grid and the declared shapes have to be reconcilable at compile time")
        return value

    def _index(self, op, i: int) -> Affine:
        value = self._operand(op, i)
        if not isinstance(value, Affine):
            raise BridgeError(
                f"operand {i} of {op.get_name()} is not an index expression (got "
                f"{type(value).__name__})", op=op.get_name())
        return value

    def _shape(self, value) -> tuple[int, ...]:
        return source.tensor_shape(value.get_type())

    def _bind(self, results: list, value) -> None:
        self.env[results[0].id()] = value

    # -- index space -------------------------------------------------------------------------

    def _op_tt_get_program_id(self, op, results) -> None:
        axis = op.get_int_attr("axis")
        if axis is None or not 0 <= axis <= 2:
            raise BridgeError(f"program_id axis {axis} out of range", op="tt.get_program_id")
        self._bind(results, Affine(pid={axis: 1}))

    def _op_tt_make_range(self, op, results) -> None:
        start, end = op.get_int_attr("start"), op.get_int_attr("end")
        if start is None or end is None:
            raise BridgeError("tt.make_range without static bounds", op="tt.make_range")
        self._bind(results, Affine(shape=(end - start,), const=start, iota={0: 1}))

    def _op_arith_constant(self, op, results) -> None:
        value = self.constants.get(results[0].id())
        if value is None:
            raise BridgeError("constant value could not be read", op="arith.constant")
        shape = self._shape(results[0])
        self._bind(results, FloatConst(float(value), shape) if isinstance(value, float)
                   else Affine(shape=shape, const=int(value)))

    def _op_tt_splat(self, op, results) -> None:
        value = self._operand(op, 0)
        shape = self._shape(results[0])
        if isinstance(value, PointerTensor):
            self._bind(results, PointerTensor(value.base, value.offset.splat_to(shape)))
        elif isinstance(value, Affine):
            self._bind(results, value.splat_to(shape))
        elif isinstance(value, FloatConst):
            self._bind(results, FloatConst(value.value, shape))
        else:
            raise BridgeError(f"cannot splat a {type(value).__name__}", op="tt.splat")

    def _op_tt_expand_dims(self, op, results) -> None:
        axis = op.get_int_attr("axis")
        value = self._operand(op, 0)
        if not isinstance(value, Affine):
            raise BridgeError(f"cannot expand a {type(value).__name__}", op="tt.expand_dims")
        self._bind(results, value.expand_dims(axis))

    def _op_tt_broadcast(self, op, results) -> None:
        value = self._operand(op, 0)
        shape = self._shape(results[0])
        if isinstance(value, PointerTensor):
            self._bind(results, PointerTensor(value.base, value.offset.broadcast_to(shape)))
        elif isinstance(value, Affine):
            self._bind(results, value.broadcast_to(shape))
        else:
            raise BridgeError(f"cannot broadcast a {type(value).__name__}", op="tt.broadcast")

    def _op_arith_extsi(self, op, results) -> None:
        self._bind(results, self._index(op, 0).with_shape(self._shape(results[0])))

    _op_arith_extui = _op_arith_extsi
    _op_arith_trunci = _op_arith_extsi

    def _op_tt_addptr(self, op, results) -> None:
        ptr = self._operand(op, 0)
        if not isinstance(ptr, PointerTensor):
            raise BridgeError(f"tt.addptr base is a {type(ptr).__name__}", op="tt.addptr")
        self._bind(results, PointerTensor(ptr.base, ptr.offset + self._index(op, 1)))

    def _op_arith_cmpi(self, op, results) -> None:
        kind = _CMPI.get(op.get_int_attr("predicate"))
        if kind is None:
            raise BridgeError("unreadable comparison predicate", op="arith.cmpi")
        self._bind(results, Predicate(self._index(op, 0), self._index(op, 1), kind))

    def _op_arith_andi(self, op, results) -> None:
        lhs, rhs = self._operand(op, 0), self._operand(op, 1)
        terms = []
        for value in (lhs, rhs):
            if isinstance(value, Predicate):
                terms.append(value)
            elif isinstance(value, Conjunction):
                terms.extend(value.terms)
            else:
                raise BridgeError(
                    "arith.andi over non-mask values is not translated", op="arith.andi",
                    hint="bitwise integer arithmetic inside an index expression is not affine")
        self._bind(results, Conjunction(tuple(terms)))

    # -- memory ------------------------------------------------------------------------------

    def _pointer_argument(self, ptr: PointerTensor, mask, *, writing: bool):
        arg = self.spec.arg(ptr.base)
        pattern = whole_tensor_access(ptr, shape=arg.shape or (), grid=self.grid, mask=mask)
        previous = self.report.pointer_patterns.get(arg.name)
        if previous and previous != pattern:
            raise BridgeError(
                f"argument {arg.name!r} is accessed two different ways ({previous!r} and "
                f"{pattern!r}) — the bridge re-raises each argument to one tensor value")
        self.report.pointer_patterns[arg.name] = pattern
        if writing and not arg.is_written:
            raise BridgeError(
                f"the kernel stores to {arg.name!r}, which the spec declares effect={arg.effect!r}",
                hint="a mutation the caller believes cannot happen is a miscompile, so the effect "
                     "must be declared, not discovered")
        if not writing and arg.effect == "write":
            raise BridgeError(
                f"the kernel loads from {arg.name!r}, which the spec declares write-only")
        return arg

    def _mask_operand(self, op, i: int):
        if op.get_num_operands() <= i:
            return None
        value = self._operand(op, i)
        if isinstance(value, (Predicate, Conjunction)):
            return value
        raise BridgeError(
            f"operand {i} of {op.get_name()} is a {type(value).__name__}, not a mask",
            op=op.get_name())

    def _op_tt_load(self, op, results) -> None:
        ptr = self._operand(op, 0)
        if not isinstance(ptr, PointerTensor):
            raise BridgeError(f"tt.load through a {type(ptr).__name__}", op="tt.load")
        arg = self._pointer_argument(ptr, self._mask_operand(op, 1), writing=False)
        # The `other` operand (operand 2) fills masked-off lanes. It cannot affect the result: the
        # coverage check above proves the masked-in lanes are exactly the declared tensor, and the
        # store's own coverage check proves masked-off lanes are never written back.
        if op.get_num_operands() > 2:
            self.report.notes.append(
                f"masked load of {arg.name!r}: `other` is unobservable because masked-off lanes are "
                "outside the declared extent and are never stored")
        self._bind(results, Tensor(self.block_values[arg.name], arg.shape or ()))

    def _op_tt_store(self, op, results) -> None:
        ptr = self._operand(op, 0)
        if not isinstance(ptr, PointerTensor):
            raise BridgeError(f"tt.store through a {type(ptr).__name__}", op="tt.store")
        value = self._operand(op, 1)
        if not isinstance(value, Tensor):
            raise BridgeError(
                f"tt.store of a {type(value).__name__} — only a computed tensor can be stored",
                op="tt.store")
        arg = self._pointer_argument(ptr, self._mask_operand(op, 2), writing=True)
        if arg.name in self.stored:
            raise BridgeError(f"argument {arg.name!r} is stored to more than once", op="tt.store")
        self.stored[arg.name] = value.value

    def _op_tt_return(self, op, results) -> None:
        if op.get_num_operands():
            raise BridgeError("a kernel that returns a value is not supported", op="tt.return")

    # -- compute -----------------------------------------------------------------------------

    def _empty_like(self, element_type, shape):
        from xdsl.dialects import tensor as tensor_d
        from xdsl.dialects.builtin import TensorType
        result_type = TensorType(element_type, list(shape))
        empty = tensor_d.EmptyOp((), result_type)
        self.ops.append(empty)
        return empty.tensor, result_type

    def _op_tt_dot(self, op, results) -> None:
        from xdsl.dialects import arith
        from xdsl.dialects.builtin import IntegerAttr, IntegerType
        from xdsl.dialects.linalg import ops as linalg_ops

        if self.grid != (1, 1, 1):
            raise BridgeError(
                f"tt.dot under a grid of {list(self.grid)} programs", op="tt.dot",
                hint="a multi-program contraction needs tiled accumulation re-raised as a reduction; "
                     "the bridge normalizes a grid only when every program's payload is elementwise")
        lhs, rhs = self._operand(op, 0), self._operand(op, 1)
        if not isinstance(lhs, Tensor) or not isinstance(rhs, Tensor):
            raise BridgeError("tt.dot operands are not loaded tensors", op="tt.dot")
        acc = self._operand(op, 2)
        zero = (isinstance(acc, Affine) and acc.is_constant and acc.const == 0
                or isinstance(acc, FloatConst) and acc.value == 0.0)
        if not zero:
            raise BridgeError(
                "tt.dot starts from a non-zero accumulator", op="tt.dot",
                hint="an initial accumulator value would have to become a linalg `outs` operand "
                     "carrying real data, which changes residency analysis; not yet translated")

        element = self._element_type(_result_dtype(results[0]))
        shape = (lhs.shape[0], rhs.shape[1])
        init, result_type = self._empty_like(element, shape)
        zero_const = arith.ConstantOp(_zero_attr(element))
        fill = linalg_ops.FillOp(inputs=(zero_const.result,), outputs=(init,), res=(result_type,))
        self.ops += [zero_const, fill]

        if isinstance(element, IntegerType):
            # The MLIR idiom for i8 x i8 -> i32 accumulation, and exactly what Merlin's own
            # frontend emits, so a Triton matmul and a hand-written one converge bit for bit.
            zp = arith.ConstantOp(IntegerAttr(0, 32))
            self.ops.append(zp)
            mm = linalg_ops.QuantizedMatmulOp(
                inputs=(lhs.value, rhs.value, zp.result, zp.result),
                outputs=(fill.results[0],), res=(result_type,))
        else:
            mm = linalg_ops.MatmulOp(inputs=(lhs.value, rhs.value),
                                     outputs=(fill.results[0],), res=(result_type,))
        self.ops.append(mm)
        self._bind(results, Tensor(mm.results[0], shape))

    def _elementwise(self, op, results, kind: str) -> None:
        from xdsl.dialects.linalg import ops as linalg_ops

        lhs, rhs = self._operand(op, 0), self._operand(op, 1)
        if not isinstance(lhs, Tensor) or not isinstance(rhs, Tensor):
            # Integer adds and multiplies are overwhelmingly index arithmetic, not data.
            if kind in ("AddOp", "MulOp", "SubOp") and op.get_name().endswith("i"):
                return self._index_arithmetic(op, results)
            raise BridgeError(
                f"{op.get_name()} mixes tensors and index expressions", op=op.get_name())
        if lhs.value.type != rhs.value.type:
            raise BridgeError(
                f"{op.get_name()} over differently-typed tensors ({lhs.value.type} and "
                f"{rhs.value.type})", op=op.get_name())
        init, result_type = self._empty_like(lhs.value.type.get_element_type(), lhs.shape)
        emitted = getattr(linalg_ops, kind)(inputs=(lhs.value, rhs.value), outputs=(init,),
                                            res=(result_type,))
        self.ops.append(emitted)
        self._bind(results, Tensor(emitted.results[0], lhs.shape))

    def _index_arithmetic(self, op, results) -> None:
        lhs, rhs = self._index(op, 0), self._index(op, 1)
        name = op.get_name()
        if name == "arith.addi":
            self._bind(results, lhs + rhs)
        elif name == "arith.muli":
            self._bind(results, lhs * rhs)
        elif name == "arith.subi":
            self._bind(results, lhs + rhs.scaled(-1))
        else:  # pragma: no cover - guarded by the caller
            raise BridgeError(f"no index arithmetic for {name}", op=name)


def _result_dtype(result) -> str:
    """The element type of a TTIR tensor result, as a spec dtype string."""
    text = str(result.get_type())
    if not text.startswith("tensor<"):
        raise BridgeError(f"expected a tensor result, got {text}")
    element = text[len("tensor<"):-1].split("x")[-1]
    for spec_name, mlir in _DTYPES.items():
        if mlir == element:
            return spec_name
    raise BridgeError(f"unsupported result element type {element!r}")


def _zero_attr(element_type):
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType
    if isinstance(element_type, IntegerType):
        return IntegerAttr(0, element_type)
    return FloatAttr(0.0, element_type)


def _install_elementwise_handlers() -> None:
    """Bind one handler per elementwise arith op, so the dispatch stays a lookup, not a chain."""
    for op_name, kind in _ELEMENTWISE.items():
        def handler(self, op, results, _kind=kind):
            self._elementwise(op, results, _kind)
        setattr(_Bridge, "_op_" + op_name.replace(".", "_"), handler)


_install_elementwise_handlers()
