"""User-reviewed Phase 0 (functional, all native ops) semantic rules."""

import collections
import math

from .functional_truth_types import (
    CPLX,
    TRIVM,
    V,
    conversion,
    conversions_ok,
    dclass,
    hack,
    integral,
    must,
    nn,
    scalar_val,
    table2,
)


def r_contraction(c):
    if (
        c.ov == "aten.addmm.default"
        and scalar_val(c.a.get("alpha", 1)) == 0
        and any(dclass(t.dtype) == "complex" for t in c.ins)
    ):
        return must(
            "N3",
            "addmm with alpha=0 drops the product term (BLAS alpha=0 contract): out = beta*input, "
            "a runtime x literal pointwise map",
            c,
            cls="addmm complex alpha=0 = beta*input (2x2 linear map)",
            eq="addmm(input, m1, m2, beta=b, alpha=0) == b*input (broadcast); m1, m2 are not read",
        )
    if any(dclass(t.dtype) == "complex" for t in c.ins):
        return V(
            CPLX,
            "C1",
            "complex x complex runtime contraction needs a multi-matmul composition "
            "(real/imag cross terms); strict-native boundary",
            cls="complex runtime contraction",
        )
    return must("N1", "contraction on the systolic array (int8 x int8 -> int32; bias as accumulator seed)", c)


def avg_divisor_is_one(c):
    ov = c.ov
    if "adaptive" in ov:
        # adaptive window sizes: every window has one element iff output spatial == input spatial
        if ov.endswith("_backward.default"):
            g, x = c.ins[0], c.ins[1]
            return g.shape == x.shape
        x, o = c.ins[0], c.outs[0]
        return x.shape == o.shape
    dov = c.a.get("divisor_override")
    if dov is not None:
        return dov == 1
    k = c.a.get("kernel_size")
    k = k if isinstance(k, list) else [k]
    n = 1
    for v in k:
        n *= v
    return n == 1


def r_window_mean(c):
    xd = c.x().dtype
    if integral(xd):
        if avg_divisor_is_one(c):
            return must(
                "N4",
                "integer average over 1-element windows = strided selection",
                c,
                cls="int avg-pool with unit windows",
            )
        return must(
            "N2e",
            "integer average pool: exact device window sum (limbs) followed by the exact host "
            "readout epilogue trunc(sum/divisor) (eager C integer division), the same pattern as requant",
            c,
            cls="integer avg-pool = exact window sum + exact truncating-division epilogue",
            eq="avg_pool(x)[o] == trunc(wrap_dtype(sum_{i in window(o)} x[i]) / divisor(o)) on an integer dtype: the "
            "window sum is an exact device reduction with eager's modular wrap to the dtype, the truncating "
            "division is an exact host readout epilogue (checked on all 16 non-empty int64 forward cases)",
        )
    return must("N2", "static-window sum with 1/|window| folded into the readout scale", c)


def col2im_overlap(c):
    out_size, k, dil, pad, st = (c.a[n] for n in ("output_size", "kernel_size", "dilation", "padding", "stride"))
    mx = 1
    for d in range(2):
        L = (out_size[d] + 2 * pad[d] - dil[d] * (k[d] - 1) - 1) // st[d] + 1
        cnt = collections.Counter()
        for i in range(L):
            for kk in range(k[d]):
                p = i * st[d] - pad[d] + kk * dil[d]
                if 0 <= p < out_size[d]:
                    cnt[p] += 1
        mx *= max(cnt.values()) if cnt else 0
    return mx >= 2


def r_col2im(c):
    if dclass(c.x().dtype) == "bool":
        if col2im_overlap(c):
            return hack(
                "H1",
                "bool col2im accumulates overlapping patches: Boolean OR via add + compare",
                cls="bool col2im with overlapping patches",
                eq="out[p] = OR over patches covering p",
            )
        return must(
            "N4", "non-overlapping col2im = static scatter with zero fill", c, cls="bool col2im without overlap"
        )
    return must("N2", "overlapping patches accumulated into the accumulator (static index sets)", c)


def r_sum(c):
    o = c.out()
    if dclass(o.dtype) == "bool":
        r = c.reduce_size()
        if r == 1:
            return must("N6", "bool sum over 1-element sets = identity", c, cls="bool->bool sum over singleton")
        if r == 0:
            return must("N5", "bool sum over empty sets = False fill", c, cls="bool->bool sum over empty set")
        return hack(
            "H1",
            "sum with a bool result dtype is an OR-reduction (add + compare)",
            cls="bool->bool sum (OR reduction)",
            eq="out = OR_i x[i]",
        )
    bad = conversions_ok(c, [c.x()], o.dtype)
    if bad:
        return hack(("H5" if "H5" in bad[0][2] else "H4"), f"sum with dtype conversion {bad[0]}")
    return must("N2", "additive reduction (accumulate / ones-vector matmul)", c)


def r_mean(c):
    return must("N2", "additive reduction with 1/N readout scale", c)


def bool_result_affine(c, f, what, hk):
    """Affine op whose result dtype is bool (bool input): result must be identity or uniform."""
    kind, vals = table2(f, "bool")
    if kind == "identity":
        return must("N6", f"{what} with bool result is the identity on bool", c, cls=f"{c.ov} bool result identity")
    if kind == "uniform":
        return must("N5", f"{what} with bool result is a uniform fill", c, cls=f"{c.ov} bool result uniform")
    return hack(hk, f"{what} with bool result needs Boolean algebra")


def r_add_scalar(c):
    o = c.out()
    other, alpha = scalar_val(c.a.get("other")), scalar_val(c.a.get("alpha", 1))
    sign = -1 if c.ov.startswith("aten.sub") else 1
    if dclass(o.dtype) == "bool":
        return bool_result_affine(c, lambda x: bool(x + sign * alpha * other), "add/sub scalar", "H1")
    return must("N3", "affine shift by a compile-time scalar (alpha folded)", c)


def r_mul_scalar(c):
    o = c.out()
    other = scalar_val(c.a.get("other"))
    if dclass(o.dtype) == "bool":
        return bool_result_affine(c, lambda x: bool(x * other), "mul by scalar", "H2")
    amb = None
    if isinstance(other, float) and (math.isinf(other) or math.isnan(other)):
        amb = "non-finite compile-time scale"
    return must("N3", "scale by a compile-time scalar", c, amb=amb)


def r_div_scalar(c):
    other = scalar_val(c.a.get("other"))
    if other == 0:
        return DIV0()
    return must("N3", "scale by the compile-time reciprocal 1/c", c)


def DIV0():
    return nn(
        "X_ieee_special",
        "division by a literal 0: IEEE special-value outputs (+-inf, NaN) outside the quantized contract",
        cls="division by literal 0",
    )


def r_neg(c):
    return must("N3", "scale by -1", c)


def r_relu(c):
    return must("N3", "readout ReLU of the native epilogue", c)


def r_add_tensor(c):
    o = c.out()
    if dclass(o.dtype) == "bool":
        alpha = scalar_val(c.a.get("alpha", 1))
        if not alpha:
            return must(
                "N6",
                "bool add with alpha=0: x OR (False AND y) = x (identity)",
                c,
                cls="add.Tensor bool alpha=0 = identity",
                eq="add(x, y, alpha=0) == x for bool x, y",
            )
        return hack(
            "H1",
            "bool add.Tensor is Boolean OR (add + compare)",
            cls="add.Tensor on bool (OR)",
            eq="add(x, y) == x OR y on bool",
        )
    return must("N3", "operand sum of two runtime tensors with compile-time weights (accumulate-on-load)", c)


def r_to_copy(c):
    o = c.out()
    ok, note = conversion(c.x().dtype, o.dtype)
    if not ok:
        return hack(
            "H5" if "H5" in note else "H4",
            f"_to_copy conversion {c.x().dtype}->{o.dtype}: {note}",
            cls=f"_to_copy {dclass(c.x().dtype)}->{dclass(o.dtype)}",
        )
    return must("N4", f"copy/conversion: {note}", c)


def r_movement51(c):
    o = c.outs[0] if c.outs else None
    if o is not None:
        bad = conversions_ok(c, c.ins, o.dtype)
        if bad:
            return hack("H5" if "H5" in bad[0][2] else "H4", f"movement with conversion {bad[0]}")
    return must("N4", "pure movement (static byte selection / view / pad / broadcast)", c)


def r_fill(c):
    return must("N5", "uniform literal fill", c)


# ---------------------------------------------------------------- non-51 handlers
def r_movement(c):
    """Movement ops outside the 51 (flip, copy, expand, alias, upsample_nearest)."""
    o = c.outs[0]
    bad = conversions_ok(c, c.ins, o.dtype)
    if bad:
        return nn("X_conversion", f"movement with a non-native conversion {bad[0]}")
    name = c.ov.split(".")[1]
    eqs = {
        "flip": "flip(x, dims)[i] = x[reflect(i, dims)] (static index map)",
        "copy": "copy(self, src) = src broadcast to self.shape, converted to self.dtype (static byte selection)",
        "expand": "expand(x, size) = stride-0 broadcast view of x",
        "alias": "alias(x) = x (identity view)",
        "upsample_nearest2d": "upsample_nearest2d(x)[n,c,h,w] = x[n,c,floor(h*H/Ho),floor(w*W/Wo)] (static index map)",
    }
    return must(
        "N4", "pure movement: static byte selection", c, cls=f"{c.ov.replace('aten.', '')} = movement", eq=eqs.get(name)
    )


def r_cumsum(c):
    o = c.out()
    bad = conversions_ok(c, [c.x()], o.dtype)
    if bad:
        return nn("X_conversion", f"cumsum with non-native dtype conversion {bad[0]}")
    return must(
        "N1t",
        "prefix sum = one contraction with the constant lower-triangular 0/1 matrix along dim d "
        "(the same native linear-map family as the ones-vector sum); exact in int32/limbs",
        c,
        cls="cumsum = contraction with a constant lower-triangular ones matrix",
        eq="cumsum(x, d) == L @ x along dim d, with L[i, j] = 1 if j <= i else 0 (n = x.shape[d]); entries are "
        "0/1 so the int8 operand is exact and each output is an exact integer sum",
    )


def r_metadata(c):
    return V(
        TRIVM,
        "T1",
        "metadata-only, no work: result is metadata only (allocation / size query / "
        "metadata resize; suite comparison is metadata or a compile-time scalar)",
        cls=f"{c.ov.replace('aten.', '')} metadata-only",
    )


def r_local_scalar(c):
    return nn(
        "X_host_scalar",
        "returns a host Scalar read from tensor data (device->host readback); no device tensor result",
        amb="no-compute op; listed not_native because the result is a host scalar",
    )
