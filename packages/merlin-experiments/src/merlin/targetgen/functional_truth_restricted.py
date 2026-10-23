"""User-reviewed Phase 0 (functional, all native ops) semantic rules."""

import math

from .functional_truth_types import (
    INT_BITS,
    INT_RANGE,
    UNARY_REF,
    dclass,
    hack,
    integral,
    must,
    nn,
    round_to,
    safe,
    same,
    scalar_val,
    table2,
)


def r_rounding(c):
    if integral(c.x().dtype):
        return must(
            "N6",
            "rounding an integer is the identity",
            c,
            cls=f"{c.ov.replace('aten.', '')} on integer dtypes = identity",
            eq=f"{c.ov.split('.')[1]}(x) == x for integer x",
        )
    return nn("X_rounding", "floor/ceil/round/trunc of non-integral floats (readout-offset rounding is a hack)")


def r_isnan_isinf(c):
    if integral(c.x().dtype):
        return must(
            "N5",
            "integers/bools are never NaN/inf: uniform False fill",
            c,
            cls=f"{c.ov.replace('aten.', '')} on int/bool = False fill",
            eq=f"{c.ov.split('.')[1]}(x) == zeros_like(x, dtype=bool) for integer/bool x",
        )
    return nn("X_comparison", "IEEE class predicate on float/complex data: data-dependent mask")


def r_sign(c):
    if dclass(c.x().dtype) == "bool":
        return must(
            "N6", "sign on bool is the identity", c, cls="sign.default on bool = identity", eq="sign(x) == x for bool x"
        )
    return nn("X_comparison", "sign is a comparator (data-dependent -1/0/+1)")


def r_abs(c):
    if dclass(c.x().dtype) == "bool":
        return must("N6", "abs on bool is the identity", c, cls="abs.default on bool = identity")
    if dclass(c.x().dtype) == "complex":
        return nn("X_transcendental", "complex abs is a magnitude (sqrt of a sum of squares)")
    return nn("X_comparison", "abs needs the sign of x (ReLU identity relu(x)+relu(-x) is a hack)")


def r_unary_trans(c):
    name = c.ov.split(".")[1]
    if dclass(c.x().dtype) == "bool":
        kind, vals = table2(safe(UNARY_REF[name]), c.out().dtype)
        if kind == "identity":
            return must(
                "N6",
                f"{name} on the bool domain maps 0->0 and 1->1: identity (bool->{c.out().dtype} widening)",
                c,
                cls=f"{name}.default on bool = identity widening",
                eq=f"{name}(x) == x.to(result dtype) for bool x ({name}(0)=0, {name}(1)=1)",
            )
        if kind == "uniform":
            return must("N5", f"{name} on the bool domain is constant", c, cls=f"{name}.default on bool = fill")
        return nn(
            "X_bool_table",
            f"{name} on bool gives the nonuniform table {vals} "
            "(selecting literals by data = nonuniform literal table hack)",
        )
    return nn("X_transcendental", f"{name} is transcendental/nonlinear")


def r_elu(c):
    alpha, scale = scalar_val(c.a.get("alpha", 1)), scalar_val(c.a.get("scale", 1))
    if alpha == 0 and scale > 0:
        return must(
            "N3",
            "elu with alpha=0 is scale*relu(x) (negative branch is 0*(exp(.)-1) = -0, NaN "
            "propagates on both): readout ReLU",
            c,
            cls="elu alpha=0 = relu",
            eq="elu(x, alpha=0, scale=s) == s*relu(x) under the quantized contract (output codes equal; the "
            "dequantized zero is +0 on both sides, so eager's -0.0 on negative inputs is not observable)",
        )
    return nn("X_transcendental", "elu negative branch uses exp")


def r_leaky_relu(c):
    s = scalar_val(c.a.get("negative_slope", 0.01))
    if s == 1:
        return must(
            "N6",
            "leaky_relu with slope 1 is the identity",
            c,
            cls="leaky_relu slope=1 = identity",
            eq="leaky_relu(x, 1.0) == x (x*1 == x for every IEEE value)",
        )
    amb = None
    if s == 0:
        amb = (
            "slope 0 equals relu on finite inputs (up to the sign of zero) but leaky_relu(-inf, 0) = "
            "-inf*0 = NaN while relu(-inf) = 0, so it is not an exact relu"
        )
    return nn("X_comparison", "leaky_relu selects a branch by the sign of x (not one readout ReLU)", amb=amb)


def r_clamp_scalar(c):
    """clamp.default / hardtanh: identity or uniform over the whole dtype domain, else not native."""
    hard = "hardtanh" in c.ov
    lo_b = scalar_val(c.a.get("min_val" if hard else "min"))
    hi_b = scalar_val(c.a.get("max_val" if hard else "max"))
    xd, od = c.x().dtype, c.out().dtype

    def f(v):
        r = v
        if lo_b is not None:
            r = max(r, lo_b)
        if hi_b is not None:
            r = min(r, hi_b)
        if integral(od) and dclass(od) != "bool":
            r = int(r)  # computed in the promoted float type, stored with truncation toward zero
        return r

    if integral(xd):
        lo, hi = INT_RANGE[xd]
        glo, ghi = f(lo), f(hi)  # clamp is monotone non-decreasing
        if same(round_to(od, glo), round_to(od, ghi)):
            return must(
                "N5",
                f"{c.ov} maps the whole {xd} range to one value ({glo}): uniform fill",
                c,
                cls=f"{c.ov.replace('aten.', '')} on integers with bounds inside one unit interval = fill",
                eq=f"{c.ov.split('.')[1]}(x, {lo_b}, {hi_b}) == full_like(x, {glo}) for {xd} x",
            )
        if glo == lo and ghi == hi:
            return must(
                "N6",
                f"bounds enclose the {xd} range: identity",
                c,
                cls=f"{c.ov.replace('aten.', '')} with bounds enclosing the dtype range = identity",
            )
        return nn("X_comparison", "clamp/hardtanh with bounds inside the domain is a data-dependent select")
    if dclass(xd) == "float":
        if (lo_b is None or lo_b == -math.inf) and (hi_b is None or hi_b == math.inf):
            return must("N6", "unbounded clamp is the identity", c, cls=f"{c.ov} unbounded = identity")
        return nn("X_comparison", "clamp/hardtanh on floats is a data-dependent select (NaN propagates)")
    return nn("X_comparison", "clamp/hardtanh select")


def r_cmp_scalar(c):
    name = c.ov.split(".")[1]
    cv = scalar_val(c.a.get("other"))
    xd = c.x().dtype
    ops = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "ge": lambda a, b: a >= b,
        "gt": lambda a, b: a > b,
        "le": lambda a, b: a <= b,
        "lt": lambda a, b: a < b,
    }
    op = ops[name]
    if dclass(xd) == "bool":
        kind, vals = table2(lambda v: op(v, cv), "bool")
        if kind == "identity":
            return must(
                "N6",
                f"{name}(x, {cv}) on bool is the identity",
                c,
                cls=f"{name}.Scalar on bool = identity",
                eq=f"{name}(x, {cv}) == x for bool x",
            )
        if kind == "uniform":
            return must(
                "N5",
                f"{name}(x, {cv}) on bool is a uniform fill ({bool(vals[0])})",
                c,
                cls=f"{name}.Scalar on bool = uniform fill",
                eq=f"{name}(x, {cv}) == full_like(x, {bool(vals[0])}) for bool x",
            )
        return nn("X_bool_table", f"{name}(x, {cv}) on bool is logical NOT (Boolean algebra / number-format identity)")
    if dclass(xd) == "int":
        # Eager scalar promotion: an integral Python scalar keeps the tensor's int dtype and is converted
        # (wrapped) into it, so it is always attainable; a float scalar promotes the comparison to the default
        # float dtype (float32): x and c are both converted to float32 and compared there (no wrap).
        lo, hi = INT_RANGE[xd]
        nonint = False
        if isinstance(cv, (bool, int)):
            bits = INT_BITS[xd]
            ce = ((int(cv) + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1)) if xd != "uint8" else int(cv) % 256
            blo, bhi = lo, hi
            out_of_range = False  # the wrapped scalar always lies in [lo, hi]
        else:
            ce = round_to("float32", cv)
            blo, bhi = round_to("float32", lo), round_to("float32", hi)  # range of float32(x)
            nonint = math.isfinite(ce) and ce != math.floor(ce)
            out_of_range = (not math.isnan(ce)) and (ce < blo or ce > bhi)
        uniform = None
        if name in ("eq", "ne"):
            if nonint or out_of_range or (isinstance(ce, float) and math.isnan(ce)):
                uniform = name == "ne"
        else:
            if isinstance(ce, float) and math.isnan(ce):
                uniform = False
            else:
                vlo, vhi = op(blo, ce), op(bhi, ce)
                if vlo == vhi:  # all ordered comparisons are monotone in x
                    uniform = vlo
        if uniform is not None:
            why = (
                "not an integer after float32 promotion"
                if nonint
                else "outside the float32 image of the dtype range (float scalar, no wrap)"
                if out_of_range
                else "the half-space contains the whole range"
            )
            return must(
                "N5",
                f"{name}(x, {cv}) is data-independent for {xd} ({why}): uniform {uniform}",
                c,
                cls=f"{name}.Scalar on integers with an unattainable threshold = uniform fill",
                eq=f"{name}(x, {cv}) == full_like(x, {uniform}, dtype=bool) for every {xd} x",
            )
        return nn(
            "X_comparison",
            f"{name} with an attainable threshold: data-dependent mask (saturation comparator is a hack)",
        )
    if isinstance(cv, float) and math.isnan(cv):
        return must("N5", "comparison with NaN is data-independent", c, cls=f"{name}.Scalar with NaN = fill")
    if dclass(xd) == "float" and ((name == "gt" and cv == math.inf) or (name == "lt" and cv == -math.inf)):
        return must(
            "N5", "strict comparison against an infinite bound is always False", c, cls=f"{name}.Scalar vs inf = fill"
        )
    return nn("X_comparison", f"{name} on {dclass(xd)} data: data-dependent mask")


def r_cmp_tensor(c):
    return nn("X_comparison", "comparison of two runtime tensors: data-dependent mask")


def r_maxmin_elem(c):
    if dclass(c.out().dtype) == "bool":
        if "maximum" in c.ov:
            return hack(
                "H1", "maximum on bool is Boolean OR", cls="maximum on bool (OR)", eq="maximum(x, y) == x OR y on bool"
            )
        return hack(
            "H2",
            "minimum on bool is Boolean AND (runtime Hadamard)",
            cls="minimum on bool (AND)",
            eq="minimum(x, y) == x AND y on bool",
        )
    return nn("X_comparison", "elementwise max/min of runtime tensors is a data-dependent select")


def r_logical(c):
    n = c.ov.split(".")[1]
    if n in ("logical_or", "bitwise_or"):
        return hack(
            "H1",
            f"{n} of runtime tensors: Boolean OR via add + compare (bit-plane OR on integers)",
            cls=f"{c.ov.replace('aten.', '')} (OR)",
        )
    if n in ("logical_and", "bitwise_and"):
        return hack(
            "H2",
            f"{n} of runtime tensors: runtime Hadamard product (bit-plane products on integers)",
            cls=f"{c.ov.replace('aten.', '')} (AND)",
        )
    if n in ("logical_xor", "bitwise_xor"):
        return nn("X_bitwise", f"{n} of runtime tensors (parity: not an affine map or byte selection)")
    return nn("X_bitwise", n)


def r_not(c):
    return nn(
        "X_bitwise",
        "NOT is 1-x on bool / -x-1 in two's complement: a number-format identity "
        "(FINAL_LISTS 4B/4C hack), not a native family",
        amb="equals an affine map (-x-1, or 1-x on bool) by a two's-complement identity; classified "
        "not_native following FINAL_LISTS",
    )


def r_mul_tensor(c):
    return hack(
        "H2",
        "product of two runtime tensors (runtime Hadamard; Boolean AND on bool)",
        cls="mul.Tensor of two runtime tensors",
        eq="mul(x, y) = x*y elementwise, both runtime",
    )


def r_bitwise_scalar(c):
    name = c.ov.split(".")[1]
    cv = scalar_val(c.a.get("other"))
    xd, od = c.x().dtype, c.out().dtype
    ops = {"bitwise_and": lambda a, b: a & b, "bitwise_or": lambda a, b: a | b, "bitwise_xor": lambda a, b: a ^ b}
    op = ops[name]
    if dclass(xd) == "bool":
        if isinstance(cv, bool) or dclass(od) == "bool":

            def f(v):
                return bool(op(bool(v), bool(cv)))
        else:

            def f(v):
                return op(int(v), int(cv))

        kind, vals = table2(f, od)
        if kind == "identity":
            return must(
                "N6",
                f"{name}(x, {cv}) on bool maps 0->0, 1->1: identity ({xd}->{od})",
                c,
                cls=f"{name}.Scalar on bool = identity",
                eq=f"{name}(x, {cv}) == x.to({od}) for bool x",
            )
        if kind == "uniform":
            return must(
                "N5",
                f"{name}(x, {cv}) on bool is the uniform value {vals[0]}",
                c,
                cls=f"{name}.Scalar on bool = uniform fill",
                eq=f"{name}(x, {cv}) == full_like(x, {vals[0]}, dtype={od}) for bool x",
            )
        return nn("X_bool_table", f"{name}(x, {cv}) on bool gives the table {vals} (NOT-like)")
    if dclass(xd) == "int":
        bits = INT_BITS[xd]
        mask = (1 << bits) - 1
        cm = int(cv) & mask
        bytes_ = [(cm >> (8 * i)) & 0xFF for i in range(bits // 8)]
        bytemask = all(b in (0, 0xFF) for b in bytes_)
        if name == "bitwise_and":
            if cm == mask:
                return must("N6", "AND with all-ones is the identity", c, cls="bitwise_and.Scalar all-ones = identity")
            if cm == 0:
                return must("N5", "AND with 0 is a zero fill", c, cls="bitwise_and.Scalar 0 = fill")
            if bytemask:
                return must(
                    "N4",
                    "AND with a byte mask = static byte selection with zero fill",
                    c,
                    cls="bitwise_and.Scalar byte mask = byte selection",
                )
        if name == "bitwise_or":
            if cm == 0:
                return must("N6", "OR with 0 is the identity", c, cls="bitwise_or.Scalar 0 = identity")
            if cm == mask:
                return must("N5", "OR with all-ones is a -1 fill", c, cls="bitwise_or.Scalar all-ones = fill")
            if bytemask:
                return must(
                    "N4",
                    "OR with a byte mask = byte selection with 0xFF fill",
                    c,
                    cls="bitwise_or.Scalar byte mask = byte selection",
                )
        if name == "bitwise_xor" and cm == 0:
            return must("N6", "XOR with 0 is the identity", c, cls="bitwise_xor.Scalar 0 = identity")
        return nn("X_bitwise", f"{name} with constant {cv} acts on sub-byte bits (not a byte selection)")
    return nn("X_bitwise", name)
