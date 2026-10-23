"""User-reviewed Phase 0 (functional, all native ops) semantic rules."""

import math

from .functional_truth_linear import (
    DIV0,
)
from .functional_truth_types import (
    conversion,
    dclass,
    hack,
    integral,
    is_pow2_recip_int,
    must,
    nn,
    scalar_val,
    table2,
)


def r_div_mode(c):
    cv = scalar_val(c.a.get("other"))
    mode = c.a.get("rounding_mode")
    xd = c.x().dtype
    if cv == 0:
        return DIV0()
    if mode is None:
        return must(
            "N3",
            "div with rounding_mode=None is true division = scale by 1/c",
            c,
            cls="div.Scalar_mode rounding_mode=None = scale",
            eq="div(x, c, rounding_mode=None) == div(x, c) == x * (1/c)",
        )
    if cv == 0 and dclass(c.out().dtype) in ("float", "complex"):
        return must(
            "N3",
            f"x/0 is always +-inf or NaN, so {mode} is a no-op: equals true division by 0",
            c,
            cls="div.Scalar_mode by 0.0 = true division (scale by +inf)",
            eq=f"div(x, 0.0, rounding_mode={mode!r}) == x / 0.0 (floor/trunc fix +-inf and NaN)",
            amb="scale by +inf (see div.Scalar by 0)",
        )
    if integral(xd) and is_pow2_recip_int(cv):
        return must(
            "N3",
            f"integer-valued x divided by c={cv} (1/c a power of two) is an exact integer, so "
            f"{mode} is a no-op: scale by {1 / cv:g}",
            c,
            cls="div.Scalar_mode on int/bool with c = +-2^-k = scale",
            eq=f"div(x, c, rounding_mode={mode!r}) == x * (1/c) for integer x when 1/c is an integer power of two",
        )
    if dclass(xd) == "bool":
        od = c.out().dtype
        fl = math.floor if mode == "floor" else math.trunc

        def f(v):
            q = v / cv
            return fl(q) if not (math.isinf(q) or math.isnan(q)) else q

        kind, vals = table2(f, od)
        if kind == "identity":
            return must("N6", "bool-domain identity", c, cls="div.Scalar_mode on bool = identity")
        if kind == "uniform":
            return must(
                "N5",
                f"{mode}(x/{cv}) is {vals[0]} for both bool values: uniform fill",
                c,
                cls="div.Scalar_mode on bool = uniform fill",
                eq=f"div(x, {cv}, rounding_mode={mode!r}) == full_like(x, {vals[0]}, dtype={od}) for bool x",
            )
        return nn("X_bool_table", f"bool-domain table {vals}")
    if integral(xd):
        return nn("X_rounding", f"integer {mode} division by {cv}: rounding of a non-exact quotient")
    return nn("X_rounding", f"{mode} of a float quotient (rounding of non-integral floats)")


def r_div_tensor(c):
    return nn("X_runtime_division", "division by a runtime tensor")


def r_mod_scalar(c):
    cv = scalar_val(c.a.get("other"))
    name = c.ov.split(".")[1]
    xd, od = c.x().dtype, c.out().dtype
    if cv == 0 and dclass(od) == "float":
        return nn(
            "X_ieee_special",
            f"{name} by a literal 0: IEEE special-value (NaN) outputs outside the quantized contract",
            cls="division by literal 0",
        )
    if dclass(xd) == "bool":

        def f(v):
            r = math.fmod(v, cv)
            if name == "remainder" and r != 0 and ((cv < 0) != (r < 0)):
                r += cv
            return r

        kind, vals = table2(f, od)
        if kind == "identity":
            return must(
                "N6",
                f"{name}(x, {cv}) on bool maps 0->0, 1->1: identity widening",
                c,
                cls=f"{name}.Scalar on bool = identity",
                eq=f"{name}(x, {cv}) == x.to({od}) for bool x",
            )
        if kind == "uniform":
            return must(
                "N5",
                f"{name}(x, {cv}) on bool is {vals[0]}: uniform fill",
                c,
                cls=f"{name}.Scalar on bool = uniform fill",
                eq=f"{name}(x, {cv}) == full_like(x, {vals[0]}) for bool x",
            )
        return nn("X_bool_table", f"{name}(x, {cv}) on bool gives the table {vals}")
    if dclass(xd) == "int" and is_pow2_recip_int(cv):
        return must(
            "N5",
            f"every integer is a multiple of {cv}: {name} is 0 for all x (sign of zero aside)",
            c,
            cls=f"{name}.Scalar on integers with c = +-2^-k = zero fill",
            eq=f"{name}(x, {cv}) == 0 for integer x (float result judged quantized_exact: dequantized zero is +0)",
        )
    if dclass(xd) == "int":
        return nn("X_runtime_division", f"{name} by {cv}: integer remainder needs rounding division")
    return nn("X_runtime_division", f"{name} of floats by {cv}: needs rounding of the quotient")


def r_pow_ts(c):
    e = scalar_val(c.a.get("exponent"))
    xd = c.x().dtype
    if e == 0:
        return must(
            "N5",
            "pow(x, 0) = 1 for every x (ATen fills 1)",
            c,
            cls="pow.Tensor_Scalar exponent 0 = ones fill",
            eq="pow(x, 0) == ones_like(x, dtype=result) (incl. NaN/inf/0 bases)",
        )
    if e == 1:
        return must(
            "N6",
            "pow(x, 1) = x (ATen copies the base)",
            c,
            cls="pow.Tensor_Scalar exponent 1 = identity",
            eq="pow(x, 1) == x.to(result dtype)",
        )
    if dclass(xd) == "bool" and isinstance(e, (int, float)) and e > 0:
        return must(
            "N6",
            f"0^{e}=0 and 1^{e}=1: identity on bool",
            c,
            cls="pow.Tensor_Scalar on bool, exponent > 0 = identity",
            eq="pow(x, e) == x.to(result dtype) for bool x and e > 0",
        )
    if dclass(xd) == "bool":
        return nn("X_bool_table", f"pow(x, {e}) on bool gives the table (inf, 1)")
    if e == 2:
        return hack(
            "H2",
            "pow(x, 2) = x*x: runtime Hadamard product of x with itself",
            cls="pow.Tensor_Scalar exponent 2 (x*x)",
            eq="pow(x, 2) == mul(x, x)",
        )
    return nn("X_transcendental", f"pow with exponent {e}")


def r_pow_scalar(c):
    b = scalar_val(c.a.get("self"))
    if b == 1:
        return must("N5", "pow(1, x) = 1 for every x (ATen fills 1)", c, cls="pow.Scalar base 1 = ones fill")
    if dclass(c.ins[0].dtype) == "bool":
        kind, vals = table2(lambda v: b**v, c.out().dtype)
        if kind in ("identity", "uniform"):
            return must(
                "N6" if kind == "identity" else "N5",
                "bool-domain specialization",
                c,
                cls=f"pow.Scalar on bool = {kind}",
            )
        return nn("X_bool_table", f"pow({b}, x) on bool gives the table {vals}")
    return nn("X_transcendental", f"pow({b}, x) is an exponential")


def r_any(c):
    r = c.reduce_size()
    if r == 1:
        return must(
            "N6",
            "any over 1-element sets is the identity",
            c,
            cls="any.* over singleton sets = identity",
            eq="any(x, dims) == x (reshaped) when every reduced set has one element",
        )
    if r == 0:
        return must("N5", "any over empty sets is False", c, cls="any.* over empty sets = False fill")
    return hack("H1", "any is an OR-reduction (add + compare)", cls="any.* OR reduction", eq="any(x, d) = OR_i x[i]")


def r_amax_amin(c):
    r = c.reduce_size()
    if r == 1:
        return must(
            "N6",
            "max/min over 1-element sets is the identity",
            c,
            cls=f"{c.ov.replace('aten.', '')} over singleton sets = identity",
            eq="amax/amin(x, dims) == x (reshaped) when every reduced set has one element",
        )
    if dclass(c.x().dtype) == "bool":
        if "amax" in c.ov:
            return hack("H1", "amax on bool is an OR-reduction", cls="amax on bool (OR reduction)")
        return hack("H2", "amin on bool is an AND-reduction (runtime products)", cls="amin on bool (AND reduction)")
    return nn("X_select_sort", "max/min reduction (comparator; pooling-unit reduction is a hack)")


def r_argmax(c):
    r = c.reduce_size()
    if r == 1:
        return must(
            "N5",
            "arg-max/min over 1-element sets is index 0: zero fill",
            c,
            cls=f"{c.ov.replace('aten.', '')} over singleton sets = zero fill",
            eq="argmax/argmin(x, d) == zeros (int64) when the reduced extent is 1",
        )
    return nn("X_select_sort", "arg-max/min indices are data-dependent")


def r_max_dim(c):
    r = c.reduce_size(c.ins[0], c.outs[0])
    if r == 1:
        return must(
            "N6",
            "max/min over a size-1 dim: values = identity, indices = zero fill",
            c,
            cls=f"{c.ov.replace('aten.', '')} over a size-1 dim = identity + zero indices",
            eq="max/min(x, d) == (x squeezed/kept, zeros int64) when x.shape[d] == 1",
        )
    return nn("X_select_sort", "max/min with indices: data-dependent selection")


def r_prod(c):
    o = c.out()
    r = c.reduce_size()
    if r == 0:
        return must(
            "N5",
            "product over empty sets is 1: ones fill",
            c,
            cls="prod over empty sets = ones fill",
            eq="prod(x) == 1 when the reduced set is empty",
        )
    if r == 1:
        ok, note = conversion(c.x().dtype, o.dtype)
        if not ok:
            return nn("X_conversion", f"singleton prod is a conversion {c.x().dtype}->{o.dtype}: {note}")
        if dclass(c.x().dtype) == "complex":
            return must(
                "N3",
                "complex product over 1-element sets is eager's (1+0j)*z (full complex formula: "
                "NaN/inf in one component contaminates the other), a runtime x literal product",
                c,
                cls="prod over singleton sets, complex = (1+0j)*z (2x2 linear map)",
                eq="prod(z, d) == (1+0j)*z per element with eager complex multiply: (1*re - 0*im, 1*im + 0*re)",
            )
        return must(
            "N6",
            f"product over 1-element sets is the element ({note})",
            c,
            cls="prod over singleton sets = identity/conversion",
            eq="prod(x, d, dtype=t) == x.to(t) (reshaped) when every reduced set has one element",
        )
    return hack(
        "H2",
        "product reduction = chain of runtime Hadamard products (AND-reduction on bool)",
        cls="prod (product reduction)",
        eq="prod(x, d) = x[0]*x[1]*... (runtime operands)",
    )


def var_correction(c):
    if c.ov.endswith("var.dim"):
        return 1 if c.a.get("unbiased", True) else 0
    corr = c.a.get("correction")
    return 1 if corr is None else corr


def r_var(c):
    r = c.reduce_size()
    corr = var_correction(c)
    if r == 0:
        return must(
            "N5",
            "variance of an empty set is NaN: NaN fill",
            c,
            cls="var over empty sets = NaN fill",
            eq="var(x, d) == nan when the reduced set is empty",
        )
    if r == 1 and corr >= 1:
        return must(
            "N5",
            "variance of one element with correction >= 1 is 0/0 or NaN/0 = NaN: NaN fill",
            c,
            cls="var over singleton sets with correction >= 1 = NaN fill",
            eq="var(x, d, correction>=1) == nan when every reduced set has one element",
        )
    return nn(
        "X_composition",
        "variance needs squares of runtime data (runtime products) plus reductions"
        + ("; singleton with correction 0 is 0 for finite inputs but NaN for inf/NaN" if r == 1 else ""),
    )


def r_sort(c):
    d = c.a.get("dim", -1)
    if c.dim_size(c.x(), d) <= 1:
        return must(
            "N6",
            "sort along a size-1 dim: values = identity, indices = zero fill",
            c,
            cls="sort along a size-1 dim = identity + zero indices",
            eq="sort(x, d) == (x, zeros_like(x, dtype=int64)) when x.shape[d] == 1 (either direction)",
        )
    return nn("X_select_sort", "sort is a data-dependent permutation")


def r_topk(c):
    d = c.a.get("dim", -1)
    k = c.a.get("k")
    if c.dim_size(c.x(), d) == 1 and k == 1:
        return must(
            "N6",
            "topk(k=1) along a size-1 dim = identity + zero indices",
            c,
            cls="topk k=1 along size-1 dim = identity",
        )
    return nn("X_select_sort", "topk is a data-dependent selection")


def r_dropout(c):
    p, train = c.a.get("p"), c.a.get("train")
    if not train:
        return must(
            "N6",
            "native_dropout with train=False: output = input, mask = all True",
            c,
            cls="native_dropout train=False = identity + True mask",
            eq="native_dropout(x, p, False) == (x, ones_like(x, dtype=bool))",
        )
    if p == 0:
        return must(
            "N6",
            "train=True, p=0: mask = bernoulli(1) = all True, scale 1: identity",
            c,
            cls="native_dropout train=True p=0 = identity + True mask",
            eq="native_dropout(x, 0.0, True) == (x, ones_like(x, dtype=bool))",
        )
    if p == 1:
        return must(
            "N3",
            "train=True, p=1: mask = bernoulli(0) = all False, output = x*0*0 (scale by "
            "compile-time 0; NaN for inf/NaN inputs)",
            c,
            cls="native_dropout train=True p=1 = scale by 0 + False mask",
            eq="native_dropout(x, 1.0, True) == (x*0.0, zeros_like(x, dtype=bool))",
        )
    return nn("X_random", "Bernoulli mask with 0 < p < 1 is random")


def r_random(c):
    if c.ov.endswith("randperm.default"):
        n = c.a.get("n")
        if n == 1:
            return must("N5", "randperm(1) = [0]", c, cls="randperm n=1 = zero fill")
    return nn("X_random", "random generation")


def r_arange(c):
    if c.out().numel == 1:
        return must("N5", "one-element arange is the literal start", c, cls="arange with one element = fill")
    return hack(
        "H3",
        "arange is a nonuniform literal result table",
        cls="arange (nonuniform literal)",
        eq="arange(start, end, step) = [start + i*step] (literal table, no runtime input)",
    )


def r_index_family(c):
    ov = c.ov
    src_extent = None
    if ov.endswith("index_select.default"):
        src_extent = c.dim_size(c.ins[0], c.a.get("dim"))
    elif ov.endswith("gather.default"):
        src_extent = c.dim_size(c.ins[0], c.a.get("dim"))
    elif ov.endswith("embedding.default"):
        src_extent = c.ins[0].shape[0] if c.ins[0].shape else None
    if src_extent == 1:
        return must(
            "N4",
            "every valid index into an extent-1 dim is 0: static broadcast",
            c,
            cls=f"{ov.replace('aten.', '')} from an extent-1 source = broadcast",
        )
    return nn("X_index", "addressing by runtime index tensors (one-hot matmul is a hack)")


def r_upsample_bilinear(c):
    x, o = c.x(), c.out()
    if x.shape == o.shape:
        return must(
            "N6",
            "bilinear resampling to the same size is the identity",
            c,
            cls="upsample_bilinear2d same size = identity",
        )
    return nn(
        "X_interp",
        "bilinear interpolation with nonuniform constant weights (two contractions / "
        "constant-matrix emulation, FINAL_LISTS 4A)",
        amb="linear in x with compile-time weights; excluded as constant-matrix emulation with non-dyadic weights",
    )


def r_fft(c):
    dims = c.a.get("dim") or []
    x = c.x()
    if all(c.dim_size(x, d) == 1 for d in dims) and not c.ov.endswith("c2r.default"):
        return must(
            "N4", "length-1 DFT is the identity (real->complex embed)", c, cls="fft over length-1 dims = movement"
        )
    return nn("X_fft", "DFT with n >= 2 (constant complex matrix; FINAL_LISTS 4D)")


def r_nn(xcls, reason):
    def f(c):
        return nn(xcls, reason)

    return f


# ----------------------------------------------------------------------------------------------
# The per-overload functional coverage table
