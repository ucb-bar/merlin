"""Explicit source-wide binary32 enclosures for a closed integer observer.

These utilities select no workload, source label, numeric policy or target ISA.
The caller supplies an actual typed source endpoint and explicit unobserved FP
effects. Every admitted fixed raw-word cell covers all its binary32 members by
Cartesian interval arithmetic, rather than interpolation or sampled endpoints.
The retained source consumer and original expression are the runtime fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field

from .ordered_fma_groups import _context_snapshot, _snapshot


@dataclass(frozen=True)
class IntervalEffectContract:
    """Admission supplied by the caller; source analysis cannot infer effects."""

    rne: bool = False
    gradual_underflow: bool = False
    nontrapping: bool = False
    flags_unobserved: bool = False
    signed_zero_unobserved: bool = False

    def validate(self):
        if not all(value is True for value in vars(self).values()):
            raise ValueError("explicit RNE/gradual/nontrapping/unobserved-effects contract required")


@dataclass(frozen=True)
class ScalarStep:
    opcode: str
    arguments: tuple[int, ...]
    dtype: str
    literal: int | None = None


_ARITY = {
    "arith.constant": 0,
    "arith.negf": 1,
    "arith.maximumf": 2,
    "arith.minimumf": 2,
    "arith.mulf": 2,
    "arith.addf": 2,
    "arith.subf": 2,
    "math.fma": 3,
    "arith.divf": 2,
    "arith.fptosi": 1,
    "arith.addi": 2,
    "arith.shli": 2,
    "arith.bitcast": 1,
}


@dataclass(frozen=True)
class ScalarExpression:
    """Canonical arithmetic plan; input is value0, steps are value1..N."""

    steps: tuple[ScalarStep, ...]

    def validate(self):
        if not self.steps or self.steps[-1].dtype != "f32":
            raise ValueError("nonempty binary32 endpoint required")
        types = ["f32"]
        for serial, step in enumerate(self.steps, 1):
            if step.opcode not in _ARITY or len(step.arguments) != _ARITY[step.opcode]:
                raise ValueError("unsupported source operation or arity")
            if any(type(index) is not int or not 0 <= index < serial for index in step.arguments):
                raise ValueError("source expression must be an acyclic increasing SSA DAG")
            actual = tuple(types[index] for index in step.arguments)
            if step.opcode == "arith.constant":
                if type(step.literal) is not int or step.dtype not in ("f32", "i32"):
                    raise ValueError("typed scalar literal required")
                if step.dtype == "f32":
                    if not 0 <= step.literal < 2**32 or not math.isfinite(_float(step.literal)):
                        raise ValueError("finite literal binary32 word required")
                elif not -(2**31) <= step.literal < 2**32:
                    raise ValueError("literal must fit its i32 word")
            elif step.literal is not None:
                raise ValueError("nonconstant operation cannot carry a literal")
            elif step.opcode == "arith.fptosi":
                if actual != ("f32",) or step.dtype != "i32":
                    raise ValueError("only f32-to-i32 conversion is supported")
            elif step.opcode in ("arith.addi", "arith.shli"):
                if actual != ("i32", "i32") or step.dtype != "i32":
                    raise ValueError("integer exponent operations require i32")
            elif step.opcode == "arith.bitcast":
                if actual != ("i32",) or step.dtype != "f32":
                    raise ValueError("only i32-to-f32 bitcast is supported")
            elif actual != ("f32",) * len(actual) or step.dtype != "f32":
                raise ValueError("floating operation requires original binary32 operands/result")
            types.append(step.dtype)

    @property
    def canonical_sha256(self):
        self.validate()
        # Only exact typed arithmetic/constants/order choose physical table reuse.
        data = [(s.opcode, s.arguments, s.dtype, s.literal) for s in self.steps]
        return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _numeric_context(operation):
    for key, value in (*operation.attributes.items(), *operation.properties.items()):
        if "strictfp" in str(key) + str(value):
            raise ValueError("strict floating context is unsupported")
        if key == "fastmath" and str(value) != "#arith.fastmath<none>":
            raise ValueError("fast floating arithmetic is unsupported")
        if key == "overflowFlags" and getattr(value, "data", True):
            raise ValueError("integer overflow promises are unsupported")
        # Numerical attributes outside this explicitly understood vocabulary
        # cannot be discarded in the canonical arithmetic fingerprint.
        if key == "passthrough" or any(tag in str(key).lower() for tag in ("round", "fpenv", "denormal", "unsafe")):
            raise ValueError("unsupported source numerical attribute")


def _scalar_attributes(operation, *, observer=False):
    _numeric_context(operation)
    if observer:
        from .bounded_rne_maps import _roundeven_intrinsic

        if _roundeven_intrinsic(operation):
            return
    if any(not str(key).startswith("prov.") for key in operation.attributes):
        raise ValueError("unsupported scalar expression attribute")
    allowed = {"value", "fastmath", "overflowFlags"}
    if observer:
        allowed.add("predicate")
    if any(key not in allowed for key in operation.properties):
        raise ValueError("unsupported scalar expression property")


def extract_scalar_expression(cut, endpoint):
    """Extract supported pure operations without cloning source IR or labels."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i32
    from xdsl.ir import OpResult
    from xdsl.traits import Pure

    if cut.type != f32 or endpoint.type != f32:
        raise ValueError("binary32 scalar cut and endpoint required")
    values = {cut: 0}
    steps, operations = [], []

    def visit(value):
        if value in values:
            return
        if not isinstance(value, OpResult):
            raise ValueError("expression has an additional live input")
        operation = value.owner
        if operation.regions or len(operation.results) != 1 or not operation.has_trait(Pure):
            raise ValueError("source expression requires pure scalar operations")
        _scalar_attributes(operation)
        if operation.name not in _ARITY or value.type not in (f32, i32):
            raise ValueError("unsupported typed source expression")
        for operand in operation.operands:
            visit(operand)
        literal = None
        if isinstance(operation, arith.ConstantOp):
            constant = operation.value
            if isinstance(constant, FloatAttr) and constant.type == f32:
                literal = struct.unpack("<I", struct.pack("<f", constant.value.data))[0]
            elif isinstance(constant, IntegerAttr) and constant.type == i32:
                literal = int(constant.value.data)
            else:
                raise ValueError("unsupported source literal type")
        steps.append(ScalarStep(operation.name, tuple(values[x] for x in operation.operands), str(value.type), literal))
        operations.append(operation)
        values[value] = len(steps)

    visit(endpoint)
    plan = ScalarExpression(tuple(steps))
    plan.validate()
    return plan, tuple(operations)


@dataclass(frozen=True)
class ClosedScalarObserver:
    expression: ScalarExpression
    cut: object
    endpoint: object
    up: object
    quant_factor_bits: int
    integer_result: object
    expression_operations: tuple
    observer_operations: tuple
    _witness: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)


def close_scalar_i8_observer(cut, endpoint, *, effects: IntervalEffectContract):
    """Prove the actual two rounded multiplies and saturated ties-even i8 DAG.

    No changed carrier may escape. Literals can retain unrelated uses; every
    other expression/observer result must remain inside the closed source DAG.
    Tensor coordinates/ownership and actual compilation identity are separate
    caller obligations: this function moves no operation or memory access.
    """
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import FloatAttr, f32, i8
    from xdsl.ir import Block, Region
    from xdsl.traits import Pure

    from .bounded_rne_maps import _roundeven_intrinsic, prove_scalar_bounded_rne

    effects.validate()
    expression, operations = extract_scalar_expression(cut, endpoint)

    def one_user(value):
        uses = tuple(value.uses)
        if len(uses) != 1:
            raise ValueError("floating endpoint or intermediate has an escape")
        return uses[0].operation

    product = one_user(endpoint)
    if product.name != "arith.mulf" or len(product.results) != 1 or product.results[0].type != f32:
        raise ValueError("first unchanged rounded binary32 multiply required")
    _scalar_attributes(product)
    up = next((x for x in product.operands if x is not endpoint), None)
    if up is None or up.type != f32:
        raise ValueError("independent binary32 finishing operand required")
    scaled = one_user(product.results[0])
    if scaled.name != "arith.mulf" or len(scaled.results) != 1 or scaled.results[0].type != f32:
        raise ValueError("second unchanged rounded binary32 multiply required")
    _scalar_attributes(scaled)
    factor = next((x for x in scaled.operands if x is not product.results[0]), None)
    if factor is None or not isinstance(factor.owner, arith.ConstantOp):
        raise ValueError("typed literal finishing factor required")
    constant = factor.owner.value
    if not isinstance(constant, FloatAttr) or constant.type != f32 or not 0 < constant.value.data < math.inf:
        raise ValueError("finite positive nonzero binary32 finishing factor required")
    factor_bits = struct.unpack("<I", struct.pack("<f", constant.value.data))[0]
    block = endpoint.owner.parent
    if block is None or product.parent is not block or scaled.parent is not block:
        raise ValueError("all scalar source operations must share their original block")
    terminator = block.last_op
    publication_witnesses = ()
    if terminator is None:
        raise ValueError("one retained integer source observation required")
    if (
        terminator.name in ("linalg.yield", "func.return")
        and len(terminator.operands) == 1
        and terminator.operands[0].type == i8
    ):
        observation = terminator
    else:
        from .closed_tensor_insert import prove_closed_tensor_insert

        def depends(value, target, seen):
            if value is target:
                return True
            if value in seen:
                return False
            seen.add(value)
            owner = value.owner
            return (
                getattr(owner, "parent", None) is block
                and owner.name != "tensor.insert"
                and any(depends(operand, target, seen) for operand in owner.operands)
            )

        candidates = [
            operation
            for operation in block.ops
            if operation.name == "tensor.insert" and depends(operation.operands[0], scaled.results[0], set())
        ]
        if len(candidates) != 1:
            raise ValueError("one retained integer source observation required")
        observation = candidates[0]
        publication_witnesses = prove_closed_tensor_insert(observation)

    def source_pure(operation):
        if operation.has_trait(Pure) or _roundeven_intrinsic(operation):
            return True
        if publication_witnesses:
            from .closed_tensor_insert import pure_tensor_control

            return pure_tensor_control(operation)
        return False

    if any(operation is not terminator and not source_pure(operation) for operation in block.ops):
        raise ValueError("unknown source effects may observe or alter the floating environment")
    result = observation.operands[0]
    if result.type != i8:
        raise ValueError("signed i8 observation required")
    quant_ops, seen = [], {scaled.results[0]}

    def quant_visit(value):
        if value in seen:
            return
        operation = value.owner
        if (
            (getattr(operation, "parent", None) is not block and not isinstance(operation, arith.ConstantOp))
            or len(operation.results) != 1
            or operation.regions
        ):
            raise ValueError("observer has unknown inputs or control flow")
        for operand in operation.operands:
            quant_visit(operand)
        _scalar_attributes(operation, observer=True)
        quant_ops.append(operation)
        seen.add(value)

    quant_visit(result)
    probe = Block(arg_types=[f32])
    mapping = {scaled.results[0]: probe.args[0]}
    for operation in quant_ops:
        probe.add_op(operation.clone(mapping))
    probe.add_op(func.ReturnOp(mapping[result]))
    # The independent existing numeric theorem proves the full RNE/clamp graph.
    holder = func.FuncOp("observer_proof", ([f32], [i8]), Region(probe))
    proof = prove_scalar_bounded_rne(holder.body.block)
    if proof is None or proof.raw_input is not probe.args[0] or proof.integer_bits != 8 or proof.bounds != (-128, 127):
        raise ValueError("complete saturated ties-even i8 observer proof failed")
    allowed = set((*operations, product, scaled, *quant_ops, observation))
    for operation in allowed:
        _numeric_context(operation)
        if operation.name == "arith.constant" or operation is observation:
            continue
        for value in operation.results:
            if any(use.operation not in allowed for use in value.uses):
                raise ValueError("source expression or observation has an additional live escape")
    observed_ops = tuple(dict.fromkeys((*block.ops, *operations, *quant_ops, factor.owner, *publication_witnesses)))
    contexts, contexts_seen = [], set()
    for operation in observed_ops:
        owner = operation.parent_op()
        while owner is not None:
            if owner not in contexts_seen:
                _numeric_context(owner)
                contexts.append(_context_snapshot(owner))
                contexts_seen.add(owner)
            owner = owner.parent_op()
    return ClosedScalarObserver(
        expression,
        cut,
        endpoint,
        up,
        factor_bits,
        result,
        operations,
        (product, scaled, *quant_ops),
        tuple(_snapshot(op) for op in observed_ops),
        tuple(contexts),
    )


def validate_closed_scalar_observer(proof: ClosedScalarObserver):
    """Refuse changed operations, use lists, precision, attributes or ownership."""
    for witness in proof._witness:
        if _snapshot(witness.operation) != witness:
            raise ValueError("closed source observer changed after analysis")
    for witness in proof._contexts:
        if _context_snapshot(witness.operation) != witness:
            raise ValueError("source observer numerical context or ownership changed")
    proof.expression.validate()


def find_closed_scalar_i8_observers(module, *, effects: IntervalEffectContract):
    """Inventory live division-product endpoints by typed arithmetic and uses.

    Analysis is read-only. Unsupported candidates stay on their original source
    route, with an explicit refusal. Names and provenance do not select policy.
    """
    from xdsl.dialects.builtin import f32

    effects.validate()
    proofs, refusals, visited = [], [], set()
    for operation in module.walk():
        if operation.name != "arith.divf" or len(operation.results) != 1 or operation.results[0].type != f32:
            continue
        quotient = operation.results[0]
        for use in tuple(quotient.uses):
            endpoint = use.operation
            if endpoint in visited or endpoint.name != "arith.mulf" or len(endpoint.results) != 1:
                continue
            visited.add(endpoint)
            cuts = [value for value in endpoint.operands if value is not quotient]
            if len(cuts) != 1:
                continue
            try:
                proofs.append(close_scalar_i8_observer(cuts[0], endpoint.results[0], effects=effects))
            except ValueError as error:
                refusals.append((endpoint, str(error)))
    return tuple(proofs), tuple(refusals)


def evaluate_intervals(expression: ScalarExpression, lower, upper):
    """Enclose every input in the interval under explicit IEEE RNE semantics.

    Products of binary32 endpoints are exact binary64. Adjacent binary64 values
    enclose add/FMA/div rounding before monotone binary32 rounding, including
    double-rounding ties. Integer operations execute only at defined points.
    Every nonfinite/reversed/zero-sensitive endpoint refuses by its valid mask.
    """
    import numpy as np

    expression.validate()
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.dtype != np.float32 or upper.dtype != np.float32 or lower.shape != upper.shape:
        raise ValueError("identically shaped binary32 endpoint arrays required")
    valid = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    values = [(lower, upper)]

    def rounded64(lo, hi):
        return np.nextafter(lo, -np.inf).astype(np.float32), np.nextafter(hi, np.inf).astype(np.float32)

    def product(a, b):
        corners = [x.astype(np.float64) * y.astype(np.float64) for x in a for y in b]
        return np.minimum.reduce(corners), np.maximum.reduce(corners)

    with np.errstate(all="ignore"):
        for step in expression.steps:
            args = [values[i] for i in step.arguments]
            name = step.opcode
            if name == "arith.constant":
                item = (
                    np.array(step.literal, np.uint32).view(np.float32)
                    if step.dtype == "f32"
                    else np.int64(step.literal)
                )
                item = np.broadcast_to(item, lower.shape)
                value = (item, item)
            elif name == "arith.negf":
                value = (-args[0][1], -args[0][0])
            elif name in ("arith.maximumf", "arith.minimumf"):
                op = np.maximum if name == "arith.maximumf" else np.minimum
                value = (op(args[0][0], args[1][0]), op(args[0][1], args[1][1]))
            elif name == "arith.mulf":
                lo, hi = product(*args)
                value = (lo.astype(np.float32), hi.astype(np.float32))
            elif name == "arith.addf":
                value = rounded64(
                    args[0][0].astype(np.float64) + args[1][0], args[0][1].astype(np.float64) + args[1][1]
                )
            elif name == "arith.subf":
                value = rounded64(
                    args[0][0].astype(np.float64) - args[1][1], args[0][1].astype(np.float64) - args[1][0]
                )
            elif name == "math.fma":
                lo, hi = product(args[0], args[1])
                value = rounded64(lo + args[2][0], hi + args[2][1])
            elif name == "arith.divf":
                admitted = (args[1][0] > 0) & np.isfinite(args[1][1])
                valid &= admitted
                den = [np.where(admitted, x, np.float32(1)) for x in args[1]]
                corners = [a.astype(np.float64) / b.astype(np.float64) for a in args[0] for b in den]
                value = rounded64(np.minimum.reduce(corners), np.maximum.reduce(corners))
            elif name == "arith.fptosi":
                admitted = (args[0][0] == args[0][1]) & (args[0][0] >= -(2**31)) & (args[0][0] < 2**31)
                valid &= admitted
                item = np.where(admitted, args[0][0], np.float32(0)).astype(np.int64)
                value = (item, item)
            elif name in ("arith.addi", "arith.shli"):
                admitted = (args[0][0] == args[0][1]) & (args[1][0] == args[1][1])
                if name == "arith.addi":
                    item = args[0][0] + args[1][0]
                else:
                    admitted &= (args[1][0] >= 0) & (args[1][0] < 32)
                    item = np.left_shift(args[0][0], np.where(admitted, args[1][0], np.int64(0)))
                valid &= admitted
                # Original fixed-width integer operations have wrap semantics.
                item = item.astype(np.uint32).view(np.int32).astype(np.int64)
                value = (item, item)
            elif name == "arith.bitcast":
                valid &= args[0][0] == args[0][1]
                item = args[0][0].astype(np.uint32).view(np.float32)
                value = (item, item)
            else:
                raise ValueError("unsupported source interval operation")
            if step.dtype == "f32":
                valid &= np.isfinite(value[0]) & np.isfinite(value[1]) & (value[0] <= value[1])
            values.append(value)
    lo, hi = values[-1]
    valid &= ((lo > 0) & (hi > 0)) | ((lo < 0) & (hi < 0))
    return lo, hi, valid


@dataclass(frozen=True)
class SourceIntervalTable:
    expression_sha256: str
    leading_bits: int
    data: bytes = field(repr=False)
    valid_cells: int

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def build_source_interval_table(
    expression: ScalarExpression, *, effects: IntervalEffectContract, leading_bits, max_table_bytes
):
    """Build a fixed-bit source table within an explicit storage budget.

    Partition width is a caller's compiler/resource choice. The proof is the
    same for every supported width; none receives an inferred performance price.
    """
    import ctypes

    import numpy as np

    effects.validate()
    expression.validate()
    if type(leading_bits) is not int or not 9 <= leading_bits <= 24:
        raise ValueError("fixed raw-word partition width must be in [9,24]")
    if type(max_table_bytes) is not int or max_table_bytes <= 0 or (1 << leading_bits) * 8 > max_table_bytes:
        raise ValueError("source table exceeds the explicit immutable storage budget")
    lib = ctypes.CDLL(None)
    if not hasattr(lib, "fegetround") or lib.fegetround() != 0:
        raise ValueError("table generation requires actual host IEEE RNE")
    if (
        np.float32(2**-149).view(np.uint32) != 1
        or (np.array([2**-126], np.float32) * np.float32(2**-23)).view(np.uint32)[0] != 1
    ):
        raise ValueError("host gradual binary32 underflow is required")
    shift = 32 - leading_bits
    starts = np.arange(1 << leading_bits, dtype=np.uint32) << np.uint32(shift)
    ends = starts + np.uint32((1 << shift) - 1)
    first, last = starts.view(np.float32), ends.view(np.float32)
    exponent = (starts >> np.uint32(23)) & np.uint32(255)
    allowed = (exponent != 0) & (exponent != 255)
    negative = (starts >> np.uint32(31)) != 0
    lo = np.where(allowed, np.where(negative, last, first), np.float32(1))
    hi = np.where(allowed, np.where(negative, first, last), np.float32(1))
    lower, upper, valid = evaluate_intervals(expression, lo, hi)
    valid &= allowed
    data = np.empty((1 << leading_bits, 2), dtype="<f4")
    data[:, 0] = np.where(valid, lower, np.inf)
    data[:, 1] = np.where(valid, upper, -np.inf)
    return SourceIntervalTable(expression.canonical_sha256, leading_bits, data.tobytes(), int(valid.sum()))


def emit_source_interval_lookup(*, table_name, activation_name, quantizer_name, lookup_name, leading_bits):
    """Portable carrier certificate, original scalar source callbacks retained.

    The caller proves the closed observer/context/effects and supplies current
    original callback definitions. No target instruction or ISA is emitted.
    Supported runtime needs finite operands and positive nonzero typed factor;
    every other data/cell path executes the original source expression.
    """
    names = (table_name, activation_name, quantizer_name, lookup_name)
    if any(
        not isinstance(n, str)
        or not n
        or n[0].isdigit()
        or not all(c.isascii() and (c.isalnum() or c == "_") for c in n)
        for n in names
    ):
        raise ValueError("explicit C identifiers required")
    if type(leading_bits) is not int or not 9 <= leading_bits <= 24:
        raise ValueError("explicit fixed-bit width must be in [9,24]")
    nonfinite = "||".join(f"({word}&0x7f800000u)==0x7f800000u" for word in ("w", "bits(up)", "bits(scale)"))
    return f"""#include <stdint.h>
extern const float {table_name}[{1 << leading_bits}][2];
extern float {activation_name}(float);
extern signed char {quantizer_name}(float);
static inline uint32_t bits(float x){{uint32_t w;__builtin_memcpy(&w,&x,4);return w;}}
__attribute__((always_inline)) float {lookup_name}(float x,float up,float scale){{
 uint32_t w=bits(x);
 if({nonfinite})return {activation_name}(x);
 const float*cell={table_name}[w>>{32 - leading_bits}];float lo=cell[0],hi=cell[1];
 if(!(lo<=hi))return {activation_name}(x);
 float low_product=lo*up,high_product=hi*up;
 float low_scaled=low_product*scale,high_scaled=high_product*scale;
 if({quantizer_name}(low_scaled)!={quantizer_name}(high_scaled))return {activation_name}(x);
 return lo;
}}
"""


def emit_source_interval_i8_lookup(
    *,
    table_name,
    activation_name,
    quantizer_name,
    lookup_name,
    leading_bits,
    finite_inputs=(),
    finite_table=None,
    observer_word_cells=(),
    zero_observer_cells=(),
):
    """Return only the already-certified integer observation, explicitly opt-in.

    The typed all-use closure and unobserved/nontrapping floating effect contract
    must authorize removing the redundant floating finish. The original source
    activation and each rounded finishing multiply execute on every refusal.
    Table partition/cells and the original quantizer remain unchanged. A caller
    guard retains original source arithmetic for unsupported floating modes.
    Explicit finite_inputs additionally requires the matching finite_table,
    validated source/LLVM producer bindings and dominating successful immutable scale scans. It removes only
    their redundant finite checks; ordinary emission is byte-identical.
    Explicit observer_word_cells replaces only the second saturated RNE
    quantization with membership in its exact ordered-binary32 preimage. The
    table and complete typed observer/effect witnesses must agree. Every source
    finishing multiply and original refusal continuation remains unchanged.
    Explicit zero_observer_cells instead adds only an exact sufficient zero-bin
    test before the unchanged observers. It consumes no cell storage and keeps
    the original path for every other result. The same typed effect witnesses
    and immutable expression/observer matching are required.
    """
    # Reuse the existing identifier/partition admission, never a second grammar.
    emit_source_interval_lookup(
        table_name=table_name,
        activation_name=activation_name,
        quantizer_name=quantizer_name,
        lookup_name=lookup_name,
        leading_bits=leading_bits,
    )
    finite_inputs = tuple(finite_inputs)
    observer_word_cells = tuple(observer_word_cells)
    zero_observer_cells = tuple(zero_observer_cells)
    if observer_word_cells and zero_observer_cells:
        raise ValueError("choose one explicit source observer representation")
    observer_cells = observer_word_cells or zero_observer_cells
    if finite_inputs:
        from .scaled_integer_finite_llvm import validate_finite_scale_helper

        if (
            not isinstance(finite_table, SourceIntervalTable)
            or finite_table.leading_bits != leading_bits
            or len(finite_table.data) != (1 << leading_bits) * 8
        ):
            raise ValueError("explicit matching immutable source table required for prepared finite inputs")
        for binding in finite_inputs:
            validate_finite_scale_helper(binding)
            for route in binding._routes:
                matches = [
                    observer
                    for observer in binding._observers
                    if observer.quant_factor_bits == route["quant_factor_bits"]
                    and observer.expression.canonical_sha256 == route["source_expression_sha256"]
                ]
                if len(matches) != 1:
                    raise ValueError("selected finite producer has no unique typed observer")
                observer = matches[0]
                validate_closed_scalar_observer(observer)
                if observer.expression.canonical_sha256 != finite_table.expression_sha256:
                    raise ValueError("finite producer observer differs from the actual lookup source expression")
                if not math.isfinite(_float(observer.quant_factor_bits)):
                    raise ValueError("finite original quantization factor required")
    elif finite_table is not None and not observer_cells:
        raise ValueError("finite table supplied without prepared input bindings")
    membership_table = ""
    certify = f" signed char low_word={quantizer_name}(low_scaled),high_word={quantizer_name}(high_scaled);\n if(low_word!=high_word)goto source_fallback;"
    if observer_cells:
        from .bounded_rne_word_cells import validate_bounded_rne_word_cells

        if (
            not isinstance(finite_table, SourceIntervalTable)
            or finite_table.leading_bits != leading_bits
            or len(finite_table.data) != (1 << leading_bits) * 8
        ):
            raise ValueError("explicit matching source interval table required for observer cells")
        for cells in observer_cells:
            validate_bounded_rne_word_cells(cells)
            if cells.expression_sha256 != finite_table.expression_sha256:
                raise ValueError("observer cell source expression differs from the actual lookup table")
        if finite_inputs:
            expected = {
                (route["source_expression_sha256"], route["quant_factor_bits"])
                for binding in finite_inputs
                for route in binding._routes
            }
            supplied = {(cells.expression_sha256, cells.quant_factor_word) for cells in observer_cells}
            if expected != supplied:
                raise ValueError("observer cells do not cover the selected finite source observers")
        ranges = observer_cells[0].ranges
        if any(cells.ranges != ranges for cells in observer_cells):
            raise ValueError("shared observer cells have different numerical semantics")
        if zero_observer_cells:
            # An OR of the two nonnegative magnitude words dominates each.
            # Admission inside the exact q=0 preimage is therefore sufficient;
            # a rare false negative retains the original two observers. This
            # also refuses NaNs/infinities without inspecting FP comparisons.
            zero_upper = ranges[128][1] ^ 0x80000000
            certify = (
                f""" uint32_t endpoint_magnitude=(bits(low_scaled)|bits(high_scaled))&0x7fffffffu;
 if(endpoint_magnitude<={zero_upper}u)return 0;
"""
                + certify
            )
        else:
            rows = ",".join("{" + str(lo) + "u," + str(hi) + "u}" for lo, hi in ranges)
            name = lookup_name + "_rne_cells"
            membership_table = f"static const uint32_t {name}[256][2]={{{rows}}};\n"
            certify = f""" signed char low_word={quantizer_name}(low_scaled);
 uint32_t other_word=bits(high_scaled);
 uint32_t other_key=(other_word>>31)?~other_word:(other_word^0x80000000u);
 const uint32_t*bin={name}[(int)low_word+128];
 if(other_key<bin[0]||other_key>bin[1])goto source_fallback;"""
    nonfinite = "||".join(f"({word}&0x7f800000u)==0x7f800000u" for word in ("w", "bits(up)", "bits(scale)"))
    finite_guard = "" if finite_inputs else f" if({nonfinite})goto source_fallback;\n"
    return f"""#include <stdint.h>
extern const float {table_name}[{1 << leading_bits}][2];
extern float {activation_name}(float);
extern signed char {quantizer_name}(float);
static inline uint32_t bits(float x){{uint32_t w;__builtin_memcpy(&w,&x,4);return w;}}
{membership_table}\
__attribute__((always_inline)) signed char {lookup_name}(float x,float up,float scale){{
 uint32_t w=bits(x);
{finite_guard} const float*cell={table_name}[w>>{32 - leading_bits}];float lo=cell[0],hi=cell[1];
 if(!(lo<=hi))goto source_fallback;
 float low_product=lo*up,high_product=hi*up;
 float low_scaled=low_product*scale,high_scaled=high_product*scale;
{certify}
 return low_word;
source_fallback:;
 float original={activation_name}(x);
 float product=original*up;
 float scaled=product*scale;
 return {quantizer_name}(scaled);
}}
"""


def emit_immutable_bytes_llvm(data: bytes, *, symbol: str, alignment: int = 1):
    """Emit one read-only byte owner inside the normal compiled host module.

    This is raw data, with no target endianness conversion or placement policy.
    A typed consumer/provider must prove its decoding layout and lifetime. No
    separate object or device-catalog relaxation is needed for ordinary data.
    """
    if not isinstance(data, bytes) or not data:
        raise ValueError("nonempty immutable byte data required")
    if (
        not isinstance(symbol, str)
        or not symbol
        or symbol[0].isdigit()
        or not all(c.isascii() and (c.isalnum() or c == "_") for c in symbol)
    ):
        raise ValueError("explicit data symbol required")
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1) or alignment >= 2**32:
        raise ValueError("explicit valid power-of-two alignment required")
    encoded = "".join(f"\\{byte:02X}" for byte in data)
    return f'@{symbol} = constant [{len(data)} x i8] c"{encoded}", align {alignment}\n'
