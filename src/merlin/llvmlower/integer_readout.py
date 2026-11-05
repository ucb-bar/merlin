"""Source-exact CPU readout from signed-i32 accumulators to signed-i8.

Positive ordered binary32 scales and nearest-even saturation form monotone step
functions. Full transition proofs permit integer evaluation with a guarded
fixed-point estimate; the generated C uses no accelerator instructions or ABI.
"""

import bisect
import math

from .requantization import f32, quantized


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
        points.update(x for x in (threshold - 1, threshold)
                      if proof["accumulator_min"] <= x <= proof["accumulator_max"])
    if any(rounded(x) != evaluate(x, proof) for x in points):
        return None
    return dict(schema="exact_rational_readout_v1", multiplier=numerator, shift=shift,
                source=proof, proof="Monotone signed-i64 rational RNE agrees at both ends of every source interval",
                checked_points=len(points), exact=True)


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
    lines = ["module {func.func @readout(%x:i32)->i8 {",
             " %wide=arith.extsi %x:i32 to i64",
             f" %multiplier=arith.constant {multiplier}:i64",
             " %product=arith.muli %wide,%multiplier:i64"]
    result = "%product"
    if shift:
        lines += [f" %shift=arith.constant {shift}:i64",
                  f" %mask=arith.constant {(1 << shift) - 1}:i64",
                  f" %half=arith.constant {1 << (shift - 1)}:i64",
                  " %one=arith.constant 1:i64", " %zero=arith.constant 0:i64",
                  " %base=arith.shrsi %product,%shift:i64",
                  " %remainder=arith.andi %product,%mask:i64",
                  " %above=arith.cmpi sgt,%remainder,%half:i64",
                  " %tie=arith.cmpi eq,%remainder,%half:i64",
                  " %parity=arith.andi %base,%one:i64",
                  " %odd=arith.cmpi ne,%parity,%zero:i64",
                  " %odd_tie=arith.andi %tie,%odd:i1",
                  " %increment=arith.ori %above,%odd_tie:i1",
                  " %increment_wide=arith.extui %increment:i1 to i64",
                  " %rounded=arith.addi %base,%increment_wide:i64"]
        result = "%rounded"
    lines += [f" %low=arith.constant {candidate['source']['output_min']}:i64",
              " %high=arith.constant 127:i64",
              f" %positive=arith.maxsi {result},%low:i64",
              " %bounded=arith.minsi %positive,%high:i64",
              " %out=arith.trunci %bounded:i64 to i8", " func.return %out:i8", "}}"]
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
    lines = ["module {func.func @readout(%x:i32)->i8 {",
             f" %initial=arith.constant {base}:i8"]
    result = "%initial"
    for index, (threshold, value) in enumerate(sorted(transitions.items())):
        lines += [f" %threshold{index}=arith.constant {threshold}:i32",
                  f" %value{index}=arith.constant {value}:i8",
                  f" %above{index}=arith.cmpi sge,%x,%threshold{index}:i32",
                  f" %selected{index}=arith.select %above{index},%value{index},{result}:i8"]
        result = f"%selected{index}"
    lines += [f" func.return {result}:i8", "}}"]
    text = "\n".join(lines)
    operations = 1 + 4 * len(transitions)
    if operations > 1024 or len(text.encode()) > 65536:
        raise ValueError("exact readout exceeds explicit scalar graph budget")
    return text, dict(schema="exact_threshold_readout_v1", source=proof,
        transition_count=len(transitions), scalar_operations=operations,
        proof="Complete source monotone transitions, with equal thresholds coalesced and out-of-domain thresholds folded; exact for every signed-i32 input",
        exact=True, max_output_lsb_error=0)


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
        raise ValueError(f"column readout partition budget exceeded: {len(ranges)} > {max_partitions}; original route retained")
    # Proof construction is shared for repeated coefficients, but returned
    # records confer no emission authority or reusable mutable IR analysis.
    proofs = {}
    regions = []
    for start, count, scale in ranges:
        if scale not in proofs:
            proofs[scale] = emit_exact_scalar_ir([scale], relu=relu)
        scalar_ir, proof = proofs[scale]
        regions.append(dict(first_column=start, columns=count, scale=scale,
                            scalar_ir=scalar_ir, readout_proof=proof))
    return dict(schema="contiguous_column_readout.v1", columns=columns,
                source_scales=list(normalized), relu=relu, regions=regions,
                partition_count=len(regions), max_partitions=max_partitions,
                scope="complete ordered column cover and scalar numerical contracts only")


def tiled_readout_parameter_table(readout, *, symbol):
    """Emit immutable source coefficient bits; no tensor values are evaluated.

    The caller keeps the table in the same module as its sole consuming kernel.
    A selected target must bind loaded bits exclusively to scale configuration;
    the ordinary host audit independently checks that closed use slice.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, i32

    from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout

    if type(readout) is not DenseTiledIntegerReadout or type(symbol) is not str or not symbol.isidentifier():
        raise ValueError("parameter table requires an exact tiled source readout and local symbol")
    words = readout.table_words
    return llvm.GlobalOp(
        llvm.LLVMArrayType(len(words), i32), symbol, "internal", constant=True,
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
        from .integer_producer_range import IntegerSumProductsRange

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
