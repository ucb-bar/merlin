"""Source-exact CPU readout from signed-i32 accumulators to signed-i8.

Positive ordered binary32 scales and nearest-even saturation form monotone step
functions. Full transition proofs permit integer evaluation with a guarded
fixed-point estimate; the generated C uses no accelerator instructions or ABI.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math

from xdsl.dialects import arith, builtin
from xdsl.ir import Operation
from xdsl.traits import get_effects

from merlin.compiler.ir.mlir_query import op_name
from merlin.compiler.lower.quant.requantization import f32, quantized

# ---- integer_readout ---------------------------------------------------------------------------


def derive(scales, lo=-(1 << 31), hi=(1 << 31) - 1, relu=False):
    scales = tuple(f32(s) for s in scales)
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("positive finite f32 source scales required")
    if type(lo) is not int or type(hi) is not int or not -(1 << 31) <= lo <= hi < (1 << 31):
        raise ValueError("invalid i32 accumulator domain")
    low = 0 if relu else -128
    thresholds = []
    for q in range(low + 1, 128):
        left, right = lo, hi + 1
        while left < right:
            mid = (left + right) // 2
            if quantized(mid, scales, relu) >= q:
                right = mid
            else:
                left = mid + 1
        if left <= hi and quantized(left, scales, relu) < q:
            raise AssertionError("invalid upper boundary")
        if left > lo and quantized(left - 1, scales, relu) >= q:
            raise AssertionError("invalid lower boundary")
        thresholds.append(left)
    assert thresholds == sorted(thresholds)
    return dict(
        schema="exact_integer_readout_v1",
        source_scales=list(scales),
        accumulator_min=lo,
        accumulator_max=hi,
        output_min=low,
        output_max=127,
        thresholds=thresholds,
        rounding="source f32 nearest even",
        proof="All monotone output transition thresholds and immediate predecessors verified; positive ordered f32 multiplications preserve monotonicity",
        exact=True,
        max_output_lsb_error=0,
    )


def evaluate(acc, proof):
    if not proof["accumulator_min"] <= acc <= proof["accumulator_max"]:
        raise ValueError("outside proven domain")
    return proof["output_min"] + bisect.bisect_right(proof["thresholds"], acc)


def exact_rational_candidate(scales, *, relu=False):
    """Return a full-domain exact signed-i64 RNE candidate, or None.

    A rational product is not generally equivalent to ordered binary32
    multiplies. Admission checks both ends of every source output interval;
    monotonicity of the positive rational candidate then proves all i32 inputs.
    This grants no physical readout, placement or target execution permission.
    """
    proof = derive(scales, relu=relu)
    numerator, denominator = 1, 1
    for scale in proof["source_scales"]:
        p, q = scale.as_integer_ratio()
        numerator *= p
        denominator *= q
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    shift = denominator.bit_length() - 1
    if denominator != 1 << shift or shift >= 63 or numerator * (1 << 31) > 1 << 63:
        return None

    def rounded(x):
        base, remainder = divmod(x * numerator, denominator)
        twice = 2 * remainder
        base += twice > denominator or (twice == denominator and base % 2 != 0)
        return max(proof["output_min"], min(127, base))

    points = {proof["accumulator_min"], proof["accumulator_max"]}
    for threshold in proof["thresholds"]:
        points.update(
            x for x in (threshold - 1, threshold) if proof["accumulator_min"] <= x <= proof["accumulator_max"]
        )
    if any(rounded(x) != evaluate(x, proof) for x in points):
        return None
    return dict(
        schema="exact_rational_readout_v1",
        multiplier=numerator,
        shift=shift,
        source=proof,
        proof="Monotone signed-i64 rational RNE agrees at both ends of every source interval",
        checked_points=len(points),
        exact=True,
    )


def emit_rational_scalar_ir(scales, *, relu=False):
    """Emit a unary integer scalar graph after full-domain equivalence checking.

    Unsupported scales refuse rather than emitting approximate arithmetic. The
    result can describe a typed readout macro, but its physical implementation
    must independently implement these semantics and the declared storage width.
    """
    candidate = exact_rational_candidate(scales, relu=relu)
    if candidate is None:
        raise ValueError("no exact signed-i64 rational readout for source scales")
    multiplier, shift = candidate["multiplier"], candidate["shift"]
    lines = [
        "module {func.func @readout(%x:i32)->i8 {",
        " %wide=arith.extsi %x:i32 to i64",
        f" %multiplier=arith.constant {multiplier}:i64",
        " %product=arith.muli %wide,%multiplier:i64",
    ]
    result = "%product"
    if shift:
        lines += [
            f" %shift=arith.constant {shift}:i64",
            f" %mask=arith.constant {(1 << shift) - 1}:i64",
            f" %half=arith.constant {1 << (shift - 1)}:i64",
            " %one=arith.constant 1:i64",
            " %zero=arith.constant 0:i64",
            " %base=arith.shrsi %product,%shift:i64",
            " %remainder=arith.andi %product,%mask:i64",
            " %above=arith.cmpi sgt,%remainder,%half:i64",
            " %tie=arith.cmpi eq,%remainder,%half:i64",
            " %parity=arith.andi %base,%one:i64",
            " %odd=arith.cmpi ne,%parity,%zero:i64",
            " %odd_tie=arith.andi %tie,%odd:i1",
            " %increment=arith.ori %above,%odd_tie:i1",
            " %increment_wide=arith.extui %increment:i1 to i64",
            " %rounded=arith.addi %base,%increment_wide:i64",
        ]
        result = "%rounded"
    lines += [
        f" %low=arith.constant {candidate['source']['output_min']}:i64",
        " %high=arith.constant 127:i64",
        f" %positive=arith.maxsi {result},%low:i64",
        " %bounded=arith.minsi %positive,%high:i64",
        " %out=arith.trunci %bounded:i64 to i8",
        " func.return %out:i8",
        "}}",
    ]
    return "\n".join(lines), candidate


def emit_exact_scalar_ir(scales, *, relu=False):
    """Describe the exact ordered-binary32 readout as bounded integer scalar IR.

    Keep the compact rational representation when its full-domain proof holds.
    Otherwise encode the complete monotone i8 transition set. These constants
    follow only from public source scales/rounding, never observed outputs.
    This is a semantic representation, not permission for a physical readout.
    """
    if type(relu) is not bool:
        raise ValueError("explicit Boolean source activation required")
    scales = tuple(scales)
    candidate = exact_rational_candidate(scales, relu=relu)
    if candidate is not None:
        return emit_rational_scalar_ir(scales, relu=relu)
    proof = derive(scales, relu=relu)
    lo, hi = proof["accumulator_min"], proof["accumulator_max"]
    base = proof["output_min"]
    transitions = {}
    for offset, threshold in enumerate(proof["thresholds"], 1):
        value = proof["output_min"] + offset
        if threshold <= lo:
            base = value
        elif threshold <= hi:
            transitions[threshold] = value
    if len(transitions) > 255:
        raise ValueError("i8 transition representation exceeds its finite bound")
    lines = ["module {func.func @readout(%x:i32)->i8 {", f" %initial=arith.constant {base}:i8"]
    result = "%initial"
    for index, (threshold, value) in enumerate(sorted(transitions.items())):
        lines += [
            f" %threshold{index}=arith.constant {threshold}:i32",
            f" %value{index}=arith.constant {value}:i8",
            f" %above{index}=arith.cmpi sge,%x,%threshold{index}:i32",
            f" %selected{index}=arith.select %above{index},%value{index},{result}:i8",
        ]
        result = f"%selected{index}"
    lines += [f" func.return {result}:i8", "}}"]
    text = "\n".join(lines)
    operations = 1 + 4 * len(transitions)
    if operations > 1024 or len(text.encode()) > 65536:
        raise ValueError("exact readout exceeds explicit scalar graph budget")
    return text, dict(
        schema="exact_threshold_readout_v1",
        source=proof,
        transition_count=len(transitions),
        scalar_operations=operations,
        proof="Complete source monotone transitions, with equal thresholds coalesced and out-of-domain thresholds folded; exact for every signed-i32 input",
        exact=True,
        max_output_lsb_error=0,
    )


def partition_column_scales(scales, columns, *, relu=False, max_partitions=128):
    """Partition source columns into equal contiguous scalar readout contracts.

    Each region consumes the existing full-i32-domain scalar proof. Repeated
    nonadjacent scales remain distinct regions; no columns are dropped to meet
    the explicit compilation budget. The result describes numerical semantics,
    not physical subviews, target emission, source provenance or qualification.
    """
    if type(columns) is not int or not 0 < columns < 2**63 or type(relu) is not bool:
        raise ValueError("explicit positive source column count and Boolean activation required")
    if type(scales) not in (list, tuple) or len(scales) != columns:
        raise ValueError("one explicit scalar coefficient per source column required")
    if type(max_partitions) is not int or max_partitions <= 0:
        raise ValueError("explicit positive readout partition budget required")
    if any(type(scale) not in (int, float) for scale in scales):
        raise ValueError("column coefficients must be numeric binary32 source values")
    normalized = tuple(f32(scale) for scale in scales)
    if any(not math.isfinite(scale) or scale <= 0 for scale in normalized):
        raise ValueError("positive finite binary32 column coefficients required")
    ranges = []
    for column, scale in enumerate(normalized):
        if ranges and ranges[-1][2] == scale:
            start, count, old = ranges[-1]
            ranges[-1] = (start, count + 1, old)
        else:
            ranges.append((column, 1, scale))
    if len(ranges) > max_partitions:
        raise ValueError(
            f"column readout partition budget exceeded: {len(ranges)} > {max_partitions}; original route retained"
        )
    # Proof construction is shared for repeated coefficients, but returned
    # records confer no emission authority or reusable mutable IR analysis.
    proofs = {}
    regions = []
    for start, count, scale in ranges:
        if scale not in proofs:
            proofs[scale] = emit_exact_scalar_ir([scale], relu=relu)
        scalar_ir, proof = proofs[scale]
        regions.append(dict(first_column=start, columns=count, scale=scale, scalar_ir=scalar_ir, readout_proof=proof))
    return dict(
        schema="contiguous_column_readout.v1",
        columns=columns,
        source_scales=list(normalized),
        relu=relu,
        regions=regions,
        partition_count=len(regions),
        max_partitions=max_partitions,
        scope="complete ordered column cover and scalar numerical contracts only",
    )


def tiled_readout_parameter_table(readout, *, symbol):
    """Emit immutable source coefficient bits; no tensor values are evaluated.

    The caller keeps the table in the same module as its sole consuming kernel.
    A selected target must bind loaded bits exclusively to scale configuration;
    the ordinary host audit independently checks that closed use slice.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, i32

    from merlin.compiler.schedule.dense.contracts import DenseTiledIntegerReadout

    if type(readout) is not DenseTiledIntegerReadout or type(symbol) is not str or not symbol.isidentifier():
        raise ValueError("parameter table requires an exact tiled source readout and local symbol")
    words = readout.table_words
    return llvm.GlobalOp(
        llvm.LLVMArrayType(len(words), i32),
        symbol,
        "internal",
        constant=True,
        value=DenseIntOrFPElementsAttr.from_list(TensorType(i32, (len(words),)), list(words)),
        alignment=4,
    )


def load_tiled_readout_parameter(builder, table_address, tile_index):
    """Read one source metadata word as unsigned bits for device configuration.

    Bounds and the tie to original output columns belong to the independently
    checked TiledIntegerStore. This does not grant host tensor arithmetic.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import i32, i64

    pointer = builder.gep(table_address, tile_index, i32)
    word = builder.add(llvm.LoadOp(pointer, i32)).results[0]
    return builder.add(llvm.ZExtOp(word, i64)).results[0]


def fixedpoint_candidate(proof):
    """Prove a cheap estimate lies within one output step over the full domain."""
    lo, hi = proof["accumulator_min"], proof["accumulator_max"]
    low = proof["output_min"]
    product = math.prod(proof["source_scales"])
    if not math.isfinite(product):
        return None
    for shift in range(30, 7, -1):
        multiplier = round(product * (1 << shift))
        half = 1 << (shift - 1)
        if multiplier <= 0 or max(abs(lo), abs(hi)) * multiplier + half >= (1 << 63):
            continue

        def estimate(a):
            return max(low, min(127, (a * multiplier + half) // (1 << shift)))

        boundaries = {lo, hi, *[x for x in proof["thresholds"] if x <= hi]}
        for q in range(low + 1, 128):
            numerator = q * (1 << shift) - half
            threshold = -((-numerator) // multiplier)
            if lo <= threshold <= hi:
                boundaries.add(threshold)
        error = max(abs(evaluate(a, proof) - estimate(a)) for a in boundaries)
        if error <= 1:
            return dict(
                multiplier=multiplier,
                shift=shift,
                max_estimate_output_error=error,
                proof="Complete union of source and fixed-point estimate transition points and domain endpoints",
                correction="At most one neighboring source threshold correction",
            )
    return None


def emit_readout(proof, symbol, *, fixedpoint=False, saturation_first=False, packet=1, producer_range=None):
    """Emit exact CPU readout, optionally testing saturation before estimation.

    Packets retain each input load/output store pair in source order. They grant
    no pointer nonoverlap and never hoist all lane loads ahead of the stores.
    The default options preserve the existing generated source bytes.
    A producer_range explicitly changes the input contract: the owner must bind
    its typed integer sum-of-products certificate to this exact storage/call.
    Producer storage must remain unchanged until its last read (in particular,
    overlapping output writes must not invalidate the certified input values).
    Arbitrary public i32 inputs must retain the checked default.
    """
    if type(saturation_first) is not bool:
        raise ValueError("saturation_first requires an explicit boolean")
    if type(packet) is not int or packet not in (1, 2, 4, 8):
        raise ValueError("readout packet must be 1, 2, 4 or 8")
    # Re-derive rather than trusting mutable receipt fields from an external file.
    expected = derive(
        proof["source_scales"], proof["accumulator_min"], proof["accumulator_max"], proof["output_min"] == 0
    )
    if any(proof.get(k) != expected[k] for k in expected):
        raise ValueError("readout proof fields changed")
    domain_check = (
        f"if((int64_t)x<{proof['accumulator_min']}LL || (int64_t)x>{proof['accumulator_max']}LL)__builtin_trap();"
    )
    if producer_range is not None:
        from merlin.compiler.lower.quant.requantization import IntegerSumProductsRange

        if type(producer_range) is not IntegerSumProductsRange:
            raise ValueError("typed source-bound integer producer range required")
        producer_range.require_contained(proof["accumulator_min"], proof["accumulator_max"])
        domain_check = "/* Input domain guaranteed by the bound producer contract. */"
    thresholds = ",".join(str(x) + "LL" for x in proof["thresholds"])
    low = proof["output_min"]
    count = len(proof["thresholds"])
    candidate = fixedpoint_candidate(proof) if fixedpoint else None
    if fixedpoint and candidate is None:
        raise ValueError("no proven single-correction fixed-point estimate")
    if candidate:
        body = f"""int64_t q=((int64_t)x*{candidate["multiplier"]}LL+{1 << (candidate["shift"] - 1)}LL)>>{candidate["shift"]};
 if(q<{low})q={low};if(q>127)q=127;
 if(q<127 && (int64_t)x>={symbol}_thresholds[q-({low})])q++;
 else if(q>{low} && (int64_t)x<{symbol}_thresholds[q-({low})-1])q--;
 return (int8_t)q;"""
    else:
        body = f"""unsigned lo=0,hi={count};
 while(lo<hi){{unsigned mid=(lo+hi)/2;if((int64_t)x>={symbol}_thresholds[mid])lo=mid+1;else hi=mid;}}
 return (int8_t)({low}+(int)lo);"""
    if saturation_first:
        body = f"""if((int64_t)x<{proof["thresholds"][0]}LL)return (int8_t){low};
 if((int64_t)x>={proof["thresholds"][-1]}LL)return (int8_t)127;
 {body}"""
    loop = f"for(size_t i=0;i<count;i++)out[i]={symbol}_scalar(acc[i]);"
    if packet != 1:
        lanes = "\n ".join(f"out[i+{lane}]={symbol}_scalar(acc[i+{lane}]);" for lane in range(packet))
        loop = f"""size_t i=0;
 for(;count-i>={packet};i+={packet}){{
 {lanes}
 }}
 for(;i<count;i++)out[i]={symbol}_scalar(acc[i]);"""
    return f"""#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t {symbol}_thresholds[{count}]={{{thresholds}}};
static inline int8_t {symbol}_scalar(int32_t x){{
 {domain_check}
 {body}
}}
void {symbol}(const int32_t*acc,int8_t*out,size_t count){{
 {loop}
}}
"""


# ---- quant_round_check -------------------------------------------------------------------------
# Independent expression reconstruction for bounded quantization legalization.
#
# Only pure, straight-line typed scalar bodies are interpreted. Outer operations,
# interfaces, effects and dataflow must remain structurally identical. This is a
# conditional arithmetic check at tensor IR, not machine-code or hardware proof.


def bounded_round_input(value, integer_type):
    """Read literal integral clamp bounds for a defined narrow conversion.

    No producer, source label or selected plan supplies these bounds. Unknown
    scalars stop the walk, and both bounds must already have been established.
    """

    if not isinstance(integer_type, builtin.IntegerType) or integer_type.width.data < 2:
        return False
    if _float_width(value.type) is None:
        return False
    lo, hi = -math.inf, math.inf
    current = value
    for _ in range(8):
        owner = current.owner
        if not isinstance(owner, Operation) or owner.name not in {"arith.minimumf", "arith.maximumf"}:
            break
        for constant, other in (owner.operands, tuple(reversed(owner.operands))):
            producer = constant.owner
            attr = producer.properties.get("value") if isinstance(producer, Operation) else None
            if (
                isinstance(producer, Operation)
                and producer.name == "arith.constant"
                and isinstance(attr, builtin.FloatAttr)
            ):
                bound = attr.value.data
                if not math.isfinite(bound) or bound != int(bound):
                    return False
                if owner.name == "arith.minimumf":
                    hi = min(hi, bound)
                else:
                    lo = max(lo, bound)
                current = other
                break
        else:
            break
    imin, imax = _int_range(integer_type)
    limit = _integral_limit(value.type)
    return imin <= lo <= hi <= imax and abs(lo) < limit and abs(hi) < limit


def residual_round_expression(cur, ft, typ):
    """Declarative ties-even identity, independent of the instruction emitter."""

    def floating(number):
        return _term("arith.constant", ft, parameter=str(builtin.FloatAttr(number, ft)))

    def integer(number):
        return _term("arith.constant", typ, parameter=str(builtin.IntegerAttr(number, typ)))

    def node(name, dtype, *args, parameter=None):
        return _term("arith." + name, dtype, args, parameter)

    truncated = node("fptosi", typ, cur)
    residual = node("subf", ft, cur, node("sitofp", ft, truncated))
    magnitude = node("maximumf", ft, residual, node("negf", ft, residual))
    greater = node("cmpf", builtin.i1, magnitude, floating(0.5), parameter=2)
    tie = node("cmpf", builtin.i1, magnitude, floating(0.5), parameter=1)
    odd = node("cmpi", builtin.i1, node("andi", typ, truncated, integer(1)), integer(0), parameter=1)
    bump = node("ori", builtin.i1, greater, node("andi", builtin.i1, tie, odd))
    negative = node("cmpf", builtin.i1, cur, floating(0.0), parameter=4)
    direction = node("select", typ, negative, integer(-1), integer(1))
    return node("addi", typ, truncated, node("select", typ, bump, direction, integer(0)))


def pure_scalar_body(block, *, bounded_intrinsics=False, integer_extensions=False):
    """Canonical value semantics exclude opaque effects and constrained FP."""
    if not block.ops or block.last_op.name != "linalg.yield":
        return False
    for op in block.ops:
        if op.name == "linalg.yield":
            continue
        if bounded_intrinsics and op_name(op) == "llvm.intr.roundeven":
            if (
                len(op.operands) != 1
                or len(op.results) != 1
                or op.operands[0].type != op.results[0].type
                or op.regions
                or op.successors
                or op.properties
                or set(op.attributes) - {"op_name__"}
            ):
                return False
            continue
        # These registered widening operations lack an effect trait in some
        # xDSL versions. The optional source family admits only their exact
        # integer value semantics; unknown or narrowing operations do not gain
        # purity from a name prefix.
        extension = (
            integer_extensions
            and type(op) in (arith.ExtSIOp, arith.ExtUIOp)
            and len(op.operands) == len(op.results) == 1
            and all(type(value.type) is builtin.IntegerType for value in (*op.operands, *op.results))
            and op.operands[0].type.width.data < op.results[0].type.width.data
        )
        if (
            isinstance(op, builtin.UnregisteredOp)
            or op.regions
            or op.successors
            or len(op.results) != 1
            or get_effects(op) != set()
            and not extension
            or not op.name.startswith(("arith.", "math."))
        ):
            return False
        for name, attr in (*op.attributes.items(), *op.properties.items()):
            if name.startswith("prov.") or name in {"value", "predicate"}:
                continue
            if name in {"fastmath", "overflowFlags"} and hasattr(attr, "data") and not attr.data:
                continue
            return False
    return True


def _term(name, typ, operands=(), parameter=None):
    return hashlib.sha256(json.dumps((name, str(typ), operands, parameter), separators=(",", ":")).encode()).hexdigest()


def _outputs(block, external, *, legalize, bounded_intrinsics=False, values=None):

    memo = {arg: _term("argument", arg.type, parameter=i) for i, arg in enumerate(block.args)}

    def expression(value):
        if value in memo:
            return memo[value]
        owner = value.owner
        if not isinstance(owner, Operation) or owner.parent is not block:
            if value not in external:
                raise ValueError("quantization checker encountered an unmapped external value")
            return external[value]
        typ = value.type
        if bounded_intrinsics and owner.name == "arith.fptosi":
            rounded = owner.operands[0].owner
            if isinstance(rounded, Operation) and op_name(rounded) == "llvm.intr.roundeven":
                operand = rounded.operands[0]
                if not bounded_round_input(operand, typ):
                    raise ValueError("intrinsic rounding lacks actual bounded integer observations")
                answer = residual_round_expression(expression(operand), operand.type, typ)
                memo[value] = answer
                return answer
        if legalize and owner.name == "arith.fptosi":
            matched = _match_chain(owner)
            if not isinstance(matched, str):
                rounding, clamps, _lo, _hi = matched
                ft = rounding.operands[0].type
                cur = expression(rounding.operands[0])

                def floating(number):
                    return _term("arith.constant", ft, parameter=str(builtin.FloatAttr(number, ft)))

                # A separate declarative expression, not replay of the emitter.
                # The original matcher supplies only admitted bounds and order.
                for name, bound, _operation in reversed(clamps):
                    if name != "arith.addf":
                        cur = _term(name, ft, (cur, floating(bound)))
                answer = residual_round_expression(cur, ft, typ)
                memo[value] = answer
                return answer
        parameter = None
        if owner.name == "arith.constant":
            parameter = str(owner.properties["value"])
        elif owner.name in {"arith.cmpi", "arith.cmpf"}:
            parameter = owner.properties["predicate"].value.data
        answer = _term(owner.name, typ, tuple(expression(arg) for arg in owner.operands), parameter)
        memo[value] = answer
        return answer

    return tuple(expression(value) for value in (block.last_op.operands if values is None else values))


def verify_quantization_rewrite(before, after, *, bounded_intrinsics=False):
    """Reconstruct actual output expressions and all unchanged outer effects."""
    before.verify()
    after.verify()
    from merlin.compiler.transforms.certificates import ir_text

    if ir_text(before) == ir_text(after):
        return
    correspondence = {}
    external_before = {}
    external_after = {}
    serial = 0

    def bind(left, right):
        nonlocal serial
        if left.type != right.type:
            raise ValueError("quantization legalization changed a value type")
        correspondence[left] = right
        identity = _term("outer", left.type, parameter=serial)
        external_before[left] = identity
        external_after[right] = identity
        serial += 1

    def block(left, right, scalar=False):
        if len(left.args) != len(right.args):
            raise ValueError("quantization legalization changed block arguments")
        for a, b in zip(left.args, right.args, strict=True):
            bind(a, b)
        if scalar and pure_scalar_body(left) and pure_scalar_body(right, bounded_intrinsics=bounded_intrinsics):
            actual = _outputs(right, external_after, legalize=False, bounded_intrinsics=bounded_intrinsics)
            if (
                _outputs(left, external_before, legalize=False) != actual
                and _outputs(left, external_before, legalize=True) != actual
            ):
                raise ValueError("emitted quantization arithmetic differs from the admitted source expression")
            return
        a_ops, b_ops = tuple(left.ops), tuple(right.ops)
        if len(a_ops) != len(b_ops):
            raise ValueError("quantization legalization changed unmodeled operations or effects")
        for a, b in zip(a_ops, b_ops, strict=True):
            operation(a, b)

    def operation(left, right):
        if (
            left.name != right.name
            or left.attributes != right.attributes
            or left.properties != right.properties
            or left.successors
            or right.successors
            or len(left.results) != len(right.results)
            or len(left.regions) != len(right.regions)
            or tuple(correspondence.get(value) for value in left.operands) != tuple(right.operands)
        ):
            raise ValueError("quantization legalization changed outer interfaces, semantics or effects")
        for a, b in zip(left.results, right.results, strict=True):
            bind(a, b)
        for a, b in zip(left.regions, right.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                raise ValueError("quantization legalization changed control flow")
            for x, y in zip(a.blocks, b.blocks, strict=True):
                block(x, y, scalar=left.name == "linalg.generic")

    operation(before, after)


# ---- quant_round_chain -------------------------------------------------------------------------
# Pure matcher and bound helpers of the ``roundeven -> clamp -> fptosi`` quantize chain.
#
# Split out of :mod:`merlin.compiler.lower.options.quant_round` so independent checks (``quant_round_check``) can
# read the chain without importing the rewrite's feature registry. Behaviour is unchanged: the
# rewrite re-exports these names.


#: Largest magnitude a clamp bound may have for the inline round to be exact. At or above 2^23 an f32
#: has no fractional bits left, so `c - trunc(c)` stops being exact and `t` stops being representable
#: in a narrow destination type. NOT a tuning number and NOT a target fact -- it is where f32's
#: mantissa ends, and it is derived from the float type of the value being rewritten rather than
#: written down, so an f16 or f64 chain gets its own bound.
def _integral_limit(float_type) -> float:
    """The magnitude at which ``float_type`` runs out of fractional bits (2 ** mantissa bits)."""
    bits = {16: 11, 32: 24, 64: 53}.get(_float_width(float_type))
    if bits is None:
        return 0.0
    return float(2 ** (bits - 1))


def _float_width(t) -> int | None:
    from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type

    if isinstance(t, Float16Type):
        return 16
    if isinstance(t, Float32Type):
        return 32
    if isinstance(t, Float64Type):
        return 64
    return None


def _const_float(value, remaining: int = 5) -> float | None:
    """Read a float constant through uniform splats and their generic input arguments."""
    from xdsl.dialects.builtin import FloatAttr

    if remaining <= 0:
        return None
    owner = getattr(value, "owner", None)
    name = getattr(owner, "name", None)
    if name == "arith.constant":
        attr = owner.properties.get("value", owner.attributes.get("value"))
        return float(attr.value.data) if isinstance(attr, FloatAttr) else None
    if name == "tensor.splat":
        return _const_float(owner.operands[0], remaining - 1)
    region = getattr(owner, "parent", None)
    consumer = getattr(region, "parent", None)
    index = getattr(value, "index", None)
    if index is not None and getattr(consumer, "name", None) == "linalg.generic" and index < len(consumer.inputs):
        source = consumer.inputs[index]
        if getattr(getattr(source, "owner", None), "name", None) == "tensor.splat":
            return _const_float(source, remaining - 1)
    return None


def _single_use(value) -> bool:
    """``value`` feeds exactly one op. Rewriting a chain whose intermediate is read elsewhere would
    leave the original `math.roundeven` alive AND add the inline round, which is strictly worse than
    doing nothing -- so a shared intermediate is refused rather than duplicated."""
    return len(list(value.uses)) == 1


def _constant_zero(value, remaining: int = 5) -> bool:
    """Prove a zero through casts and uniform splats, including generic input arguments."""
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    if remaining <= 0:
        return False
    owner = getattr(value, "owner", None)
    name = getattr(owner, "name", None)
    if name == "arith.constant":
        attr = owner.properties.get("value", owner.attributes.get("value"))
        return isinstance(attr, (FloatAttr, IntegerAttr)) and attr.value.data == 0
    if name in ("arith.sitofp", "tensor.splat"):
        return len(owner.operands) == 1 and _constant_zero(owner.operands[0], remaining - 1)
    region = getattr(owner, "parent", None)
    consumer = getattr(region, "parent", None)
    index = getattr(value, "index", None)
    if index is not None and getattr(consumer, "name", None) == "linalg.generic" and index < len(consumer.inputs):
        source = consumer.inputs[index]
        if getattr(getattr(source, "owner", None), "name", None) == "tensor.splat":
            return _constant_zero(source, remaining - 1)
    return False


def _guarded_scale_value(value):
    """Return the scalar yielded by a scale tensor feeding this generic block argument.

    Dynamic quantization computes the scale in one ``linalg.generic`` and consumes that tensor in
    another.  The divisor visible beside ``roundeven`` is consequently a block argument, not the
    guarding ``arith.select`` itself.  Follow exactly that one producer edge; do not guess through
    arbitrary tensor computations.
    """
    block = getattr(value, "owner", None)
    index = getattr(value, "index", None)
    region = getattr(block, "parent", None)
    consumer = getattr(region, "parent", None)
    if index is None or getattr(consumer, "name", None) != "linalg.generic" or index >= len(consumer.inputs):
        return None
    producer = getattr(consumer.inputs[index], "owner", None)
    if getattr(producer, "name", None) != "linalg.generic" or not producer.body.blocks:
        return None
    body = producer.body.blocks[0]
    terminator = body.last_op
    if getattr(terminator, "name", None) != "linalg.yield" or len(terminator.operands) != 1:
        return None
    return terminator.operands[0]


def _is_zero_guarded_scale(value) -> bool:
    """Whether ``value`` comes from a proved-positive symmetric scale constructor."""
    selected = _guarded_scale_value(value)
    select = getattr(selected, "owner", None)
    if getattr(select, "name", None) != "arith.select":
        return False
    cond, when_zero, otherwise = select.operands
    fallback = _const_float(when_zero)
    false_owner = getattr(otherwise, "owner", None)
    cmp = getattr(cond, "owner", None)
    if fallback is None or math.isnan(fallback) or fallback <= 0.0 or getattr(cmp, "name", None) != "arith.cmpf":
        return False
    # Most constructors select the raw division. i-GELU retains its pre-existing positive epsilon
    # floor for tiny nonzero rows and selects one only at exactly zero. Accept that exact spelling
    # too; a zero/negative/nonconstant floor proves nothing.
    if getattr(false_owner, "name", None) == "arith.maximumf":
        a, b = false_owner.operands
        floor, otherwise = _const_float(b), a
        if floor is None:
            floor, otherwise = _const_float(a), b
        if floor is None or math.isnan(floor) or floor <= 0.0:
            return False
        false_owner = getattr(otherwise, "owner", None)
    div = false_owner
    if getattr(div, "name", None) != "arith.divf":
        return False
    # xDSL encodes ordered-equal (`oeq`) as predicate 1.  Requiring it matters: unordered equality
    # would select the fallback for NaN too and no longer describe this constructor.
    predicate = cmp.properties.get("predicate")
    if getattr(getattr(predicate, "value", None), "data", None) != 1:
        return False
    numerator, denominator = div.operands
    denom = _const_float(denominator)
    if denom is None or math.isnan(denom) or denom == 0.0:
        return False
    lhs, rhs = cmp.operands
    return (lhs is numerator and _const_float(rhs) == 0.0) or (rhs is numerator and _const_float(lhs) == 0.0)


def _dynamic_scale_of_quant_input(value):
    """Return the scale in ``x / scale`` or TorchAO's ``x * (1 / scale)`` spelling."""
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) == "arith.divf":
        return owner.operands[1]
    if getattr(owner, "name", None) != "arith.mulf":
        return None
    for candidate in owner.operands:
        reciprocal = getattr(candidate, "owner", None)
        if getattr(reciprocal, "name", None) != "arith.divf":
            # The efficient TorchAO form computes the reciprocal in a separate linalg.generic.
            # Follow that one producer edge, then inspect its yielded scalar expression.
            yielded = _guarded_scale_value(candidate)
            reciprocal = getattr(yielded, "owner", None)
        if getattr(reciprocal, "name", None) != "arith.divf":
            continue
        numerator, scale = reciprocal.operands
        if _const_float(numerator) == 1.0:
            return scale
    return None


def _int_range(int_type) -> tuple[int, int]:
    w = int_type.width.data
    return (-(2 ** (w - 1)), 2 ** (w - 1) - 1)


def _match_chain(fptosi_op):
    """``(round_op, clamp_ops, lo, hi)`` for a fusable chain ending at ``fptosi_op``, else a reason.

    Walks BACKWARDS from the convert through a run of `arith.minimumf` / `arith.maximumf` whose other
    operand is an integral float constant, and requires the run to bottom out at `math.roundeven`.
    A single-use addition of a constant signed zero is also traversed. It preserves all defined
    integer results of this chain; the possible zero-sign change is erased by `fptosi`.
    The interval is accumulated in the order the ops apply, so a chain that never bounds one side is
    caught rather than assumed.

    Returns either a 4-tuple or a string naming the refusal.
    """
    from xdsl.dialects.builtin import IntegerType

    res_t = fptosi_op.results[0].type
    if not isinstance(res_t, IntegerType) or res_t.width.data < 2:
        return "dest_not_multibit_int"
    imin, imax = _int_range(res_t)

    cur = fptosi_op.operands[0]
    ftype = cur.type
    if _float_width(ftype) is None:
        return "src_not_float"
    limit = _integral_limit(ftype)
    if limit <= 0.0:
        return "float_type_unhandled"

    lo, hi = float("-inf"), float("inf")
    clamps: list[tuple[str, float, object]] = []
    # Bounded walk: the chain is a handful of ops, and an unbounded loop over a malformed IR graph
    # would hang the build rather than fail it.
    for _ in range(8):
        owner = getattr(cur, "owner", None)
        name = getattr(owner, "name", None)
        if name == "math.roundeven":
            if not _single_use(cur):
                return "round_result_shared"
            # A dynamic quantization scale may be zero.  The real whole-model form is
            # ``roundeven(x / (amax / 127))``; for an all-zero activation ``amax == 0`` and the
            # zero lanes evaluate ``0 / 0``.  Both the old and new LLVM IR then contain
            # ``fptosi(NaN)``, which is poison, but the old scalar libm call happens to inhibit
            # vectorization while the inline form lets LLVM exploit that poison.  On K1 the latter
            # made TinyLlama's complete output canonical NaNs.  Preserve the observed program until
            # Scale construction defines the zero-amax case with a positive floor. Only rewrite
            # when the divisor is a finite nonzero constant or comes from that exact producer.
            rounded = owner.operands[0]
            divisor_value = _dynamic_scale_of_quant_input(rounded)
            if divisor_value is not None:
                divisor = _const_float(divisor_value)
                if (divisor is None or math.isnan(divisor) or divisor == 0.0) and not _is_zero_guarded_scale(
                    divisor_value
                ):
                    return "round_divisor_may_be_zero"
            if lo == float("-inf") or hi == float("inf"):
                # A one-sided clamp leaves the rounded value unbounded on the other side, so the
                # inline round has no range to be exact in. Refuse; do not guess a bound.
                return "clamp_not_two_sided"
            if lo > hi:
                return "clamp_empty"
            if not (abs(lo) < limit and abs(hi) < limit):
                return "clamp_bound_above_mantissa"
            if not (imin <= lo <= imax and imin <= hi <= imax):
                return "clamp_bound_outside_dest_int"
            return (owner, clamps, lo, hi)
        if name not in ("arith.minimumf", "arith.maximumf", "arith.addf"):
            return "chain_root_not_roundeven"
        if not _single_use(cur):
            return "clamp_result_shared"
        a, b = owner.operands
        if name == "arith.addf":
            nxt = b if _constant_zero(a) else a if _constant_zero(b) else None
            if nxt is None:
                return "chain_addend_not_zero"
            clamps.append((name, 0.0, owner))
            cur = nxt
            continue
        cval, nxt = _const_float(b), a
        if cval is None:
            cval, nxt = _const_float(a), b
        if cval is None:
            return "clamp_bound_not_constant"
        if cval != int(cval) if cval == cval and abs(cval) != float("inf") else True:
            # NaN / inf / non-integral bound: Lemma 1 does not hold, so the clamp cannot be moved
            # in front of the rounding. (`cval != cval` is the NaN test; no `math` import needed.)
            return "clamp_bound_not_integral"
        if name == "arith.minimumf":
            hi = min(hi, cval)
        else:
            lo = max(lo, cval)
        clamps.append((name, cval, owner))
        cur = nxt
    return "chain_too_long"
