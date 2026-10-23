"""User-reviewed Phase 0 (functional, all native ops) semantic rules."""

from .functional_truth_linear import (
    r_add_scalar,
    r_add_tensor,
    r_col2im,
    r_contraction,
    r_cumsum,
    r_div_scalar,
    r_fill,
    r_local_scalar,
    r_mean,
    r_metadata,
    r_movement,
    r_movement51,
    r_mul_scalar,
    r_neg,
    r_relu,
    r_sum,
    r_to_copy,
    r_window_mean,
)
from .functional_truth_reductions import (
    r_amax_amin,
    r_any,
    r_arange,
    r_argmax,
    r_div_mode,
    r_div_tensor,
    r_dropout,
    r_fft,
    r_index_family,
    r_max_dim,
    r_mod_scalar,
    r_nn,
    r_pow_scalar,
    r_pow_ts,
    r_prod,
    r_random,
    r_sort,
    r_topk,
    r_upsample_bilinear,
    r_var,
    var_correction,
)
from .functional_truth_restricted import (
    r_abs,
    r_bitwise_scalar,
    r_clamp_scalar,
    r_cmp_scalar,
    r_cmp_tensor,
    r_elu,
    r_isnan_isinf,
    r_leaky_relu,
    r_logical,
    r_maxmin_elem,
    r_mul_tensor,
    r_not,
    r_rounding,
    r_sign,
    r_unary_trans,
)
from .functional_truth_types import (
    MUST,
    TRIV,
    TRIVM,
    UNARY_REF,
    Case,
    V,
    dclass,
    nn,
    scalar_val,
)

T = {}
for o in ["mm.default", "bmm.default", "addmm.default", "convolution.default", "convolution_backward.default"]:
    T[o] = r_contraction
for o in [
    "_adaptive_avg_pool2d.default",
    "_adaptive_avg_pool3d.default",
    "adaptive_avg_pool1d.default",
    "_adaptive_avg_pool2d_backward.default",
    "avg_pool1d.default",
    "avg_pool2d.default",
    "avg_pool3d.default",
    "avg_pool2d_backward.default",
]:
    T[o] = r_window_mean
T["col2im.default"] = r_col2im
T["sum.dim_IntList"] = r_sum
T["mean.dim"] = T["mean.default"] = r_mean
T["add.Scalar"] = T["sub.Scalar"] = r_add_scalar
T["mul.Scalar"] = r_mul_scalar
T["div.Scalar"] = r_div_scalar
T["neg.default"] = r_neg
T["relu.default"] = r_relu
T["add.Tensor"] = T["sub.Tensor"] = r_add_tensor
T["_to_copy.default"] = r_to_copy
for o in [
    "cat.default",
    "clone.default",
    "constant_pad_nd.default",
    "reflection_pad1d.default",
    "reflection_pad2d.default",
    "reflection_pad3d.default",
    "replication_pad2d.default",
    "replication_pad3d.default",
    "repeat.default",
    "select_scatter.default",
    "slice_scatter.default",
    "as_strided.default",
    "diagonal.default",
    "permute.default",
    "select.int",
    "slice.Tensor",
    "split_with_sizes.default",
    "squeeze.dim",
    "squeeze.dims",
    "unsqueeze.default",
    "view.default",
]:
    T[o] = r_movement51
for o in ["fill.Scalar", "full.default", "full_like.default", "scalar_tensor.default"]:
    T[o] = r_fill
# --- beyond the 51
for o in ["flip.default", "copy.default", "expand.default", "alias.default", "upsample_nearest2d.vec"]:
    T[o] = r_movement
T["cumsum.default"] = r_cumsum
for o in [
    "empty.memory_format",
    "empty_strided.default",
    "resize_.default",
    "sym_is_contiguous.default",
    "sym_numel.default",
    "sym_size.int",
    "sym_storage_offset.default",
    "sym_stride.int",
]:
    T[o] = r_metadata
T["_local_scalar_dense.default"] = r_local_scalar
for o in ["ceil.default", "floor.default", "round.default", "trunc.default"]:
    T[o] = r_rounding
T["isnan.default"] = T["isinf.default"] = r_isnan_isinf
T["sign.default"] = r_sign
T["abs.default"] = r_abs
for n in UNARY_REF:
    T[n + ".default"] = r_unary_trans
T["gelu.default"] = r_nn("X_transcendental", "gelu uses erf/tanh")
T["elu.default"] = r_elu
T["leaky_relu.default"] = r_leaky_relu
T["clamp.default"] = T["hardtanh.default"] = r_clamp_scalar
T["clamp.Tensor"] = r_nn("X_select_runtime", "clamp by runtime bound tensors")
T["where.self"] = r_nn("X_select_runtime", "select by a runtime condition (runtime-diagonal construction is a hack)")
for n in ["eq", "ne", "ge", "gt", "le", "lt"]:
    T[n + ".Scalar"] = r_cmp_scalar
    T[n + ".Tensor"] = r_cmp_tensor
T["maximum.default"] = T["minimum.default"] = r_maxmin_elem
for o in [
    "logical_or.default",
    "logical_and.default",
    "logical_xor.default",
    "bitwise_or.Tensor",
    "bitwise_and.Tensor",
    "bitwise_xor.Tensor",
]:
    T[o] = r_logical
T["logical_not.default"] = T["bitwise_not.default"] = r_not
for n in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
    T[n + ".Scalar"] = r_bitwise_scalar
T["mul.Tensor"] = r_mul_tensor
T["div.Scalar_mode"] = r_div_mode
T["div.Tensor"] = r_div_tensor
T["div.Tensor_mode"] = r_div_tensor
T["fmod.Scalar"] = T["remainder.Scalar"] = r_mod_scalar
T["fmod.Tensor"] = T["remainder.Tensor"] = r_nn("X_runtime_division", "modulo by a runtime tensor")
T["pow.Tensor_Scalar"] = r_pow_ts
T["pow.Scalar"] = r_pow_scalar
T["pow.Tensor_Tensor"] = r_nn("X_transcendental", "pow with a runtime exponent tensor")
T["atan2.default"] = T["atan2.out"] = r_nn("X_transcendental", "atan2 is transcendental")
T["any.default"] = T["any.dim"] = T["any.dims"] = r_any
T["amax.default"] = T["amin.default"] = r_amax_amin
T["argmax.default"] = T["argmin.default"] = r_argmax
T["max.dim"] = T["min.dim"] = r_max_dim
T["prod.default"] = T["prod.dim_int"] = r_prod
T["var.correction"] = T["var.dim"] = r_var
T["sort.default"] = r_sort
T["topk.default"] = r_topk
T["native_dropout.default"] = r_dropout
T["rand.default"] = T["randn.default"] = T["randperm.default"] = r_random
T["arange.start_step"] = r_arange
for o in [
    "index_select.default",
    "gather.default",
    "embedding.default",
    "index.Tensor",
    "index_put.default",
    "masked_scatter.default",
    "scatter.src",
    "scatter.value",
    "scatter_add.default",
    "scatter_reduce.two",
    "_embedding_bag.default",
    "embedding_dense_backward.default",
    "max_pool2d_with_indices_backward.default",
]:
    T[o] = r_index_family
T["max_pool2d_with_indices.default"] = T["max_pool3d_with_indices.default"] = r_nn(
    "X_select_sort", "max pooling with indices: comparator selection and data-dependent indices"
)
T["upsample_bilinear2d.vec"] = r_upsample_bilinear
T["grid_sampler_2d.default"] = r_nn("X_interp", "sampling at runtime coordinates")
T["_fft_r2c.default"] = T["_fft_c2r.default"] = r_fft
T["nonzero.default"] = r_nn("X_shape", "output shape depends on data")
for o in [
    "_softmax.default",
    "_log_softmax.default",
    "native_layer_norm.default",
    "native_layer_norm_backward.default",
    "native_group_norm.default",
    "native_group_norm_backward.default",
    "_native_batch_norm_legit.default",
    "_native_batch_norm_legit.no_stats",
    "_native_batch_norm_legit_no_training.default",
]:
    T[o] = r_nn(
        "X_composition",
        "normalization/softmax: exp/rsqrt and runtime products over reductions "
        "(not uniform even on singleton sets, because non-finite inputs give NaN)",
    )
T["_cdist_forward.default"] = T["_pdist_forward.default"] = r_nn(
    "X_composition", "p-norm distances: abs/pow/sqrt/max of runtime differences"
)


# ----------------------------------------------------------------------------------------------
def forced_zero_division(c):
    """Return a description if the case's semantics force a division by zero for EVERY output element
    (0/0 -> NaN or x/0 -> +-inf/NaN) with a float/complex result; else None. General predicate over op
    semantics + arguments + shapes, no values."""
    if not c.outs or c.out_scalar or dclass(c.outs[0].dtype) not in ("float", "complex"):
        return None
    n = c.ov[len("aten.") :]
    if n in ("div.Scalar", "div.Scalar_mode") and scalar_val(c.a.get("other")) == 0:
        return "x / 0 (division by a literal 0)"
    if n in ("fmod.Scalar", "remainder.Scalar") and scalar_val(c.a.get("other")) == 0:
        return "x mod 0 (fmod(x, 0) = NaN)"
    if n in ("mean.default", "mean.dim") and c.reduce_size() == 0:
        return "0/0 (mean over an empty set: sum 0 divided by count 0)"
    if n in ("var.correction", "var.dim"):
        r = c.reduce_size()
        if r == 0:
            return "0/0 (variance over an empty set)"
        if r == 1 and var_correction(c) >= 1:
            return "0/0 or NaN/0 (variance of one element with divisor max(0, N - correction) = 0)"
    return None


def derive(c):
    """Truth for one case."""
    # Empty-tensor rule: every output is a tensor with numel 0.
    # (An op whose output SHAPE depends on data, nonzero, qualifies only when its input is empty.)
    data_shape = c.ov == "aten.nonzero.default" and any(t.numel for t in c.ins)
    if c.outs and not c.out_scalar and all(t.numel == 0 for t in c.outs) and not data_shape:
        return V(TRIV, "T0", "every output has numel 0: nothing to compute on either lane", cls="empty outputs")
    # General IEEE special-value rule (ruling 2026-10-20): if every output element is an IEEE special value
    # forced by 0/0 or x/0, the case is not_native (outside the quantized contract).
    forced = forced_zero_division(c)
    if forced:
        return nn(
            "X_ieee_special",
            f"ieee_special_value: every output element is forced to an IEEE special value by "
            f"{forced}; outside the quantized contract",
            cls="ieee_special_value (forced 0/0 or x/0)",
            eq=f"every output element == NaN or +-inf, forced by {forced}",
        )
    key = c.ov[len("aten.") :]
    fn = T.get(key)
    if fn is None:
        raise KeyError(f"no rule for {c.ov}")
    return fn(c)


# ----------------------------------------------------------------------------------------------
def exact_identity(c):
    """Ruling 2026-10-20 ("exact identities are bit-exact"): a must_accelerate computation that is exactly the
    identity on its input bits is movement, graded "bit_exact_vs_eager" whatever its op family.

    Decided from the schema arguments and dtypes only (never tensor values or case ids). Only computations
    whose eager result equals the input bits for every finite value, infinity, signed zero and NaN qualify:
      * real floating x * 1 and x / 1 (mul.Scalar, div.Scalar, div.Scalar_mode with rounding_mode None);
      * complex addmm with alpha = 0 and beta = 1 (the alpha = 0 contract reads neither matrix, and beta = 1
        does not scale the input).
    Not identities: x + 0.0 (-0 + 0 = +0), eager f16/bf16 x - 0.0 (also returns +0 for -0), a complex scale by
    (1+0j) (eager's full complex product keeps 0*inf terms), and real addmm with alpha = 0 (eager still forms
    0 * (m1 @ m2), so inf/NaN in the matrices propagate)."""
    if not c.ins or not c.outs:
        return False
    x, o = c.ins[0], c.outs[0]
    if x.dtype != o.dtype:
        return False
    other = scalar_val(c.a.get("other"))
    if c.ov in ("aten.mul.Scalar", "aten.div.Scalar") or (
        c.ov == "aten.div.Scalar_mode" and c.a.get("rounding_mode") is None
    ):
        return dclass(x.dtype) == "float" and isinstance(other, (int, float)) and other == 1
    if c.ov == "aten.addmm.default" and dclass(x.dtype) == "complex":
        return scalar_val(c.a.get("alpha", 1)) == 0 and scalar_val(c.a.get("beta", 1)) == 1
    return False


def precision_class(c, v):
    """(b) quantized_exact for float/complex arithmetic; bit_exact_vs_eager for integer/bool arithmetic,
    movement (incl. float movement), literal fills and identity specializations."""
    if v.truth == TRIV:
        return "bit_exact_vs_eager"
    if v.truth == TRIVM:
        return "n/a_metadata"
    if v.cls and "zero fill" in v.cls and ("fmod" in v.cls or "remainder" in v.cls):
        return "quantized_exact"  # float result of float arithmetic on converted integers (ruling 5)
    if v.truth != MUST:
        return "n/a_host"
    if exact_identity(c):
        return "bit_exact_vs_eager"
    fp_in = any(dclass(t.dtype) in ("float", "complex") for t in c.ins)
    if v.family[:2] in ("N1", "N2", "N3"):
        return "quantized_exact" if fp_in else "bit_exact_vs_eager"
    # movement / identity: bit-exact unless a float-to-float conversion of the data is involved (then the
    # quantization rule applies: quantized_exact). Literal fills (N5) read no input data.
    if v.family in ("N4", "N6"):
        for t in c.ins:
            for o in c.outs:
                if (
                    dclass(t.dtype) in ("float", "complex")
                    and dclass(o.dtype) in ("float", "complex")
                    and t.dtype != o.dtype
                ):
                    return "quantized_exact"
    return "bit_exact_vs_eager"


def expectation(raw):
    """Return a deterministic lane commitment from semantics and shape only."""
    case = Case(raw, "phase0")
    decision = derive(case)
    return dict(
        truth=decision.truth,
        family=decision.family,
        reason=decision.reason,
        precision_class=precision_class(case, decision),
        lane=("device" if decision.truth == MUST else "no-work" if decision.truth in {TRIV, TRIVM} else "host-guard"),
    )


def require_placement(commitment, *, device):
    """A correct result on the wrong lane never satisfies a truth commitment."""
    if type(device) is not bool:
        raise ValueError("actual device placement must be a boolean")
    if commitment["lane"] not in {"device", "no-work", "host-guard"}:
        raise ValueError("unknown functional truth lane")
    if device != (commitment["lane"] == "device"):
        raise ValueError(
            "truth mismatch: expected "
            + commitment["lane"]
            + " ("
            + commitment["reason"]
            + "), compiler "
            + ("device" if device else "host")
        )
