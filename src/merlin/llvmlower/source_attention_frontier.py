"""Optional source-ordered attention host certificate with explicit storage.

This emitter consumes an independently validated complete source/consumer
contract. It is not a matcher or a routing policy. Product implementation,
physical target ABI and static directed-arithmetic capability remain external.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from merlin.common.paths import data_path

from .exact_bound_conversion import ExactBoundConversionContract


def _f32(value: float) -> str:
    if type(value) not in (int, float):
        raise ValueError("explicit real source binary32 constant required")
    try:
        value = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("representable source binary32 constant required") from exc
    if not math.isfinite(value):
        raise ValueError("finite source binary32 constant required")
    return value.hex() + "f"


@dataclass(frozen=True)
class SourceAttentionFrontierPlan:
    """Exact admitted operation grammar; dimensions/constants come from IR.

    Two equal online chunks, three ordered PV partials per chunk, a power-of-two
    lane reduction, and finite source polynomial/alpha operations are supported.
    The caller must prove no carrier observations escape the bound row-quant
    consumer and retain original source fallback. These proofs are not inferred
    from shapes or from this dataclass.
    """

    heads: int
    query_rows: int
    depth: int
    chunk: int
    segment: int
    denominator_lanes: int
    score_scale: float
    polynomial_cutoff: float
    polynomial_scale: float
    polynomial_coefficients: tuple[float, float, float, float]
    polynomial_bit_multiplier: float
    polynomial_bit_bias: float
    quant_divisor: float
    quant_epsilon: float
    quant_lower: int
    quant_upper: int

    def validate(self) -> None:
        dims = (self.heads, self.query_rows, self.depth, self.chunk, self.segment, self.denominator_lanes)
        if any(type(value) is not int or value <= 0 for value in dims):
            raise ValueError("positive static dimensions required")
        if self.depth > self.chunk or (self.chunk + self.segment - 1) // self.segment != 3:
            raise ValueError("unsupported three-part source grouping")
        lanes = self.denominator_lanes
        if lanes & (lanes - 1) or self.chunk % lanes:
            raise ValueError("source lane reduction requires a dividing power of two")
        if max(self.depth, self.segment) * (2**21 - 1) ** 2 >= 2**53:
            raise ValueError("weighted original and grouped products must be exact binary64")
        if 3 * max(self.depth, self.segment) * 127**2 > 2**31 - 1:
            raise ValueError("signed-radix group exceeds signed accumulator domain")
        # Every C index and byte count stays representable. This conservative
        # storage bound also refuses impractical code-generation size overflow.
        h, r, d, k = self.heads, self.query_rows, self.depth, self.chunk
        head_bytes = 4 * (22 * r * d + 4 * k * d + 14 * r * k + 6 * r) + 2 * r * k + r + r * d
        product_bytes = (
            4 * (6 * r * k + 2 * k * d) + 8 * (2 * r * k + k * d + r + k) + 3 * r * k + 3 * k * d + 4 * r * k
        )
        conservative_bytes = h * head_bytes + product_bytes + 140 * k + 13 * h * d + r + 1024
        if conservative_bytes > 2**31 - 1:
            raise ValueError("source workspace/index domain exceeds supported bound")
        if len(self.polynomial_coefficients) != 4:
            raise ValueError("cubic source polynomial required")
        values = (
            self.score_scale,
            self.polynomial_cutoff,
            self.polynomial_scale,
            *self.polynomial_coefficients,
            self.polynomial_bit_multiplier,
            self.polynomial_bit_bias,
            self.quant_divisor,
            self.quant_epsilon,
        )
        for value in values:
            _f32(value)
        if not all(float.fromhex(_f32(x)[:-1]) > 0 for x in (self.score_scale, self.quant_divisor, self.quant_epsilon)):
            raise ValueError("positive source scale and quantization constants required")
        if (
            type(self.quant_lower) is not int
            or type(self.quant_upper) is not int
            or not (-128 <= self.quant_lower < self.quant_upper <= 127)
        ):
            raise ValueError("signed-byte source clamp required")


def emit_source_attention_frontier(
    plan: SourceAttentionFrontierPlan,
    *,
    symbol: str,
    prepare_endpoint_rows: bool = False,
    word_interval_enclosure: bool = False,
    prepare_zero_error_blocks: bool = False,
    prepare_product_domain: bool = False,
    prepare_required_norms: bool = False,
    retain_certified_rows: bool = False,
    separable_source_radius: bool = False,
    prepare_softmax_domain: bool = False,
    integer_reconstruction: bool = False,
    prepare_probability_bins: bool = False,
    prepare_encoded_rows: bool = False,
    prepare_softmax_spans: bool = False,
    exact_bound_conversion: ExactBoundConversionContract | None = None,
    polynomial_batch_four: bool = False,
    fuse_encoded_witness: bool = False,
    prepare_probability_points: bool = False,
) -> str:
    """Emit portable C; nonzero result certifies complete output publication.

    On zero result the caller must run the retained original source body. Input,
    output and workspace owners must be disjoint and live for the synchronous
    call. Workspace byte/alignment queries derive from the actual compiled
    structure, and a larger shared caller allocation is accepted.

    Optional integer reconstruction reuses the canonical signed-radix proof:
    every weighted i32 term/prefix is an exact binary64 integer within i64.
    Separate private scratch is fully initialized by the first complete group,
    stays disjoint from the readout/center arrays, and converts once to f64.
    Product callback completeness/range/source binding remains mandatory.
    External scales, certificates, source arithmetic and refusal stay unchanged.

    Optional row preparation shares only immutable private alpha/denominator
    metadata across independent columns. It requires the existing stable RNE,
    nontrapping and unobserved exception-flags contract; source accumulation
    order and each source multiply remain unchanged.

    Optional word-space enclosure uses the prepared global source rounding
    budget and exact source endpoints. It may widen intervals and require more
    original replay. Source consumer proofs and fallback remain mandatory;
    this explicit choice does not infer profitability or enable routing.

    Optional zero-error blocks specialize only after immutable RHS and current
    LHS error admissions prove zero, with no uncertain input positions. The
    original outward sum of zero terms is prepared once; default adjacency and
    directed bound bits are preserved. Source FMA rounding remains checked.

    Optional separable radius shares the source-ordered FMA gamma/L1 product
    across columns only after private representation errors and uncertainty
    are proved zero. A uniform envelope admits every source prefix. Unsupported
    rows retain the original checked bounds; wider intervals may add replay.

    Optional certified-row retention preserves the previously initialized
    endpoint arrays once their complete row observation certificate succeeds.
    Subsequent refinement writes only uncertified rows; endpoint computations
    contain no cross-row reduction. State is private and reset on every call.

    Optional norm requirements retain a distinct L1-only reconstructed-row
    value when all immutable admitted RHS errors are zero. The actual emitted
    zero-error branch bypasses every L2 consumer; the source-domain proof keeps
    the identical upward L1 accumulation. Unknown/nonzero errors keep full norms.

    Optional producer-domain preparation uses current admitted original,
    reconstructed and representation-error rows to prove uniform finite and
    source-overflow bounds. Exact reconstructed products, private immutable
    spans, and monotone outward arithmetic are required producer contracts.
    Original center-versus-absolute comparisons remain checked; unsuccessful
    row admission uses the complete original dynamic checks. No public input
    array or arbitrary callback is admitted by this option alone.
    """
    if type(fuse_encoded_witness) is not bool:
        raise ValueError("fused encoded witness policy must be bool")
    if fuse_encoded_witness and not prepare_encoded_rows:
        raise ValueError("fused witness requires admitted encoded rows")
    if type(polynomial_batch_four) is not bool:
        raise ValueError("polynomial batch selection must be bool")
    if polynomial_batch_four:
        if not word_interval_enclosure or not prepare_softmax_domain:
            raise ValueError("polynomial batching requires prepared word softmax domain")
    if exact_bound_conversion is not None:
        if not isinstance(exact_bound_conversion, ExactBoundConversionContract):
            raise ValueError("typed exact bound-conversion contract required")
        exact_bound_conversion.validate()
    if type(prepare_softmax_spans) is not bool:
        raise ValueError("softmax producer spans policy must be bool")
    if prepare_softmax_spans and not (prepare_softmax_domain and separable_source_radius):
        raise ValueError("softmax producer spans require domain and source-radius contracts")
    if type(prepare_probability_bins) is not bool:
        raise ValueError("probability bin preparation must be bool")
    if type(prepare_encoded_rows) is not bool:
        raise ValueError("encoded row policy must be bool")
    if prepare_encoded_rows and not prepare_required_norms:
        raise ValueError("encoded row proof requires admitted norm requirements")
    if type(integer_reconstruction) is not bool:
        raise ValueError("explicit boolean integer reconstruction policy required")
    if type(prepare_softmax_domain) is not bool:
        raise ValueError("explicit boolean softmax domain policy required")
    if type(separable_source_radius) is not bool:
        raise ValueError("explicit boolean separable source-radius policy required")
    if separable_source_radius and not prepare_product_domain:
        raise ValueError("separable source radius requires exact producer-domain proof")
    if type(prepare_zero_error_blocks) is not bool:
        raise ValueError("explicit boolean zero-error block policy required")

    if type(retain_certified_rows) is not bool:
        raise ValueError("explicit boolean certificate-state retention required")
    if retain_certified_rows and not prepare_endpoint_rows:
        raise ValueError("certificate-state retention requires row-independent endpoint preparation")
    if type(prepare_required_norms) is not bool:
        raise ValueError("explicit boolean norm-requirements policy required")
    if prepare_required_norms and not prepare_product_domain:
        raise ValueError("norm requirements need the typed producer-domain L1 consumer")
    if type(prepare_product_domain) is not bool:
        raise ValueError("explicit boolean producer-domain policy required")
    if type(prepare_endpoint_rows) is not bool:
        raise ValueError("explicit boolean endpoint preparation policy required")
    if type(word_interval_enclosure) is not bool:
        raise ValueError("explicit boolean word interval enclosure policy required")
    if type(prepare_probability_points) is not bool:
        raise ValueError("probability point sharing must be bool")
    if prepare_probability_points and not (
        prepare_probability_bins and prepare_softmax_spans and prepare_encoded_rows
    ):
        raise ValueError("probability points require source bins, producer spans and encoded rows")
    plan.validate()
    if not symbol or symbol[0].isdigit() or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in symbol):
        raise ValueError("explicit valid C symbol required")
    runtime = data_path("runtime", "c")
    text = (runtime / "templates/source_attention_frontier.c.in").read_text()
    if integer_reconstruction:
        from .radix_integer_reconstruct import c_header
        from .radix_product_groups import plan_radix_product_groups

        proof = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=max(plan.depth, plan.segment))
        h, r, d, k = plan.heads, plan.query_rows, plan.depth, plan.chunk
        head_bytes = 4 * (22 * r * d + 4 * k * d + 14 * r * k + 6 * r) + 2 * r * k + r + r * d
        product_bytes = (
            4 * (6 * r * k + 2 * k * d) + 8 * (2 * r * k + k * d + r + k) + 3 * r * k + 3 * k * d + 4 * r * k
        )
        if h * head_bytes + product_bytes + 140 * k + 13 * h * d + r + 1024 + 8 * r * k > 2**31 - 1:
            raise ValueError("integer reconstruction workspace exceeds supported bound")
        original = """ for(int i=0;i<m*n;i++)w->center[i]=0;
 for(int degree=0;degree<5;degree++){
  if(!product(opaque,w->ap,w->bp,w->readout,m,n,k,degree))return 0;
  const double weight=(double)(UINT64_C(1)<<(7*degree));
  for(int i=0;i<m*n;i++)w->center[i]+=(double)w->readout[i]*weight;
 }
"""
        replacement = """ for(int degree=0;degree<5;degree++){
  if(!product(opaque,w->ap,w->bp,w->readout,m,n,k,degree))return 0;
  if(degree==0)merlin_radix_integer_begin_from_first_group_exact_i64(w->integer_center,w->readout,(size_t)m*n);
  else merlin_radix_integer_accumulate_exact_i64(w->integer_center,w->readout,(size_t)m*n,(unsigned)degree);
 }
 merlin_radix_integer_finish_exact_f64(w->center,w->integer_center,(size_t)m*n);
"""
        if text.count(original) != 1 or text.count(" int32_t readout[ROWS*CHUNK];") != 1:
            raise ValueError("source reconstruction ownership template changed")
        text = c_header(proof) + text.replace(original, replacement).replace(
            " int32_t readout[ROWS*CHUNK];", " int32_t readout[ROWS*CHUNK];\n int64_t integer_center[ROWS*CHUNK];"
        )
    if word_interval_enclosure:
        original = "merlin_monotone_bit_polynomial_apply(x,&root_prepared)"
        if text.count(original) != 1:
            raise ValueError("source polynomial enclosure template changed")
        text = text.replace(original, "merlin_monotone_bit_polynomial_apply_words(x,&root_prepared)")

    for placeholder, fragment in [("ZERO_ERROR_PREPARE", "prepare"), ("ZERO_ERROR_LOOP", "loop")]:
        replacement = (
            (runtime / f"templates/source_attention_zero_error_{fragment}.c.in").read_text()
            if prepare_zero_error_blocks
            else ""
        )
        text = text.replace("@" + placeholder + "@\n", replacement)
    fragments = {
        "PRODUCT_DOMAIN_INCLUDE": '#include "prepared_fma_product_bounds.h"\n',
        "PRODUCT_DOMAIN_COLUMNS": " merlin_fma_product_columns product_columns=merlin_fma_product_columns_prepare(bnp,enp,br,n,k);\n",
        "PRODUCT_DOMAIN_ROW": " merlin_fma_product_row product_row=merlin_fma_product_row_prepare(&gamma,&product_columns,&anp,&rnp,&aep,uncertainty,used,center+r*n);\n",
    }
    for key, fragment in fragments.items():
        text = text.replace("@" + key + "@\n", fragment if prepare_product_domain else "")
    checked = "merlin_fma_zero_gamma_batch_apply(&gamma,chunk,&lo[r*n+j],&hi[r*n+j])"
    selected = (
        "(product_row.valid?merlin_fma_product_row_apply(&product_row,j,chunk.absolute_upper,chunk.representation_error_upper,&lo[r*n+j],&hi[r*n+j]):"
        + checked
        + ")"
    )
    text = text.replace("@PRODUCT_DOMAIN_APPLY@", selected if prepare_product_domain else checked)
    if separable_source_radius:
        radius_replacements = (
            (
                '#include "prepared_fma_product_bounds.h"',
                '#include "prepared_fma_product_bounds.h"\n#include "separable_fma_radius.h"',
            ),
            (
                "merlin_fma_product_columns_prepare(bnp,enp,br,n,k);",
                "merlin_fma_product_columns_prepare(bnp,enp,br,n,k);\n merlin_fma_exact_columns exact_columns=merlin_fma_exact_columns_prepare(&product_columns,bnp,enp);",
            ),
            (
                "&anp,&rnp,&aep,uncertainty,used,center+r*n);",
                "&anp,&rnp,&aep,uncertainty,used,center+r*n);\n merlin_fma_separable_radius radius_plan=merlin_fma_separable_radius_prepare(&product_row,&exact_columns,&anp,&aep,used);",
            ),
            (
                "  for(int j=0;j<n;j++){\n   double e=",
                "  for(int j=0;j<n;j++){\n   if(radius_plan.valid){if(!merlin_fma_separable_radius_apply(&radius_plan,j,&lo[r*n+j],&hi[r*n+j]))return 0;continue;}\n   double e=",
            ),
        )
        for before, after in radius_replacements:
            if text.count(before) != 1:
                raise ValueError("source-radius consumer structure changed; refused")
            text = text.replace(before, after)
    if prepare_required_norms:
        # These consumers are part of the emitted source contract: the RHS
        # zero-error branch bypasses every reconstructed-LHS L2 use, and the
        # producer-domain consumer accepts a separate L1-only type.
        replacements = (
            (
                " merlin_fma_product_columns product_columns=",
                " merlin_dot_norm_requirements requirements=merlin_reconstruction_norm_requirements(enp,n);\n merlin_fma_product_columns product_columns=",
            ),
            (
                "  merlin_dot_norms an=",
                "  merlin_l1_norm rnl1=merlin_l1_norm_begin(&eligibility);\n  merlin_dot_norms an=",
            ),
            (
                "merlin_dot_norms_add(&rn,MERLIN_SOURCE_F64_ABS(ar[t]));",
                "if(requirements.require_l2)merlin_dot_norms_add(&rn,MERLIN_SOURCE_F64_ABS(ar[t]));else merlin_l1_norm_add(&rnl1,MERLIN_SOURCE_F64_ABS(ar[t]));",
            ),
            ("merlin_dot_norms_finish(&rn);", "if(requirements.require_l2)merlin_dot_norms_finish(&rn);"),
            (
                "!merlin_dot_norms_admit(&rn,&rnp)",
                "(requirements.require_l2?!merlin_dot_norms_admit(&rn,&rnp):!rnl1.valid)",
            ),
            (
                "merlin_fma_product_row product_row=merlin_fma_product_row_prepare(",
                "merlin_fma_product_row product_row=requirements.require_l2?merlin_fma_product_row_prepare(",
            ),
            (
                "&anp,&rnp,&aep,uncertainty,used,center+r*n);",
                "&anp,&rnp,&aep,uncertainty,used,center+r*n):merlin_fma_product_row_prepare_l1(&gamma,&product_columns,&anp,&rnl1,&aep,uncertainty,used,center+r*n);",
            ),
        )
        for before, after in replacements:
            if text.count(before) != 1:
                raise ValueError("norm consumer structure changed; requirements proof refused")
            text = text.replace(before, after)
    endpoint = "prepared" if prepare_endpoint_rows else "checked"
    text = text.replace(
        "@ENDPOINT_INTERVALS@", (runtime / f"templates/source_attention_endpoint_{endpoint}.c.in").read_text()
    )
    if retain_certified_rows:
        # The private state begins false. Only successful full-row observation
        # certification sets it true. Both refinement writers are guarded by
        # the existing certified-row continue and affect only their row.
        certificate_replacements = (
            ("float*estimate,int rows){", "float*estimate,int rows,const uint8_t*certified){"),
            (
                " for(int r=0;r<rows;r++){\n  float factors[2]=",
                " for(int r=0;r<rows;r++){\n  if(certified[r])continue;\n  float factors[2]=",
            ),
            (
                "h->endpoint_lo,h->endpoint_hi,h->out,ROWS)",
                "h->endpoint_lo,h->endpoint_hi,h->out,ROWS,w->row_certified)",
            ),
        )
        obligations = (
            "if(w->row_certified[row])continue;",
            "if(status<0)return 0;if(status){w->row_certified[row]=1;continue;}",
            "memset(w->row_certified,0,sizeof(w->row_certified));",
        )
        if any(text.count(statement) != 1 for statement in obligations):
            raise ValueError("certificate-state lifecycle changed; retention refused")
        for before, after in certificate_replacements:
            if text.count(before) != 1:
                raise ValueError("row-independent endpoint structure changed; retention refused")
            text = text.replace(before, after)
    if prepare_probability_bins:
        replacements = (
            (
                "    if(mask[off+j] && merlin_interval_bits(merlin_interval_bf16(y.lo)) != merlin_interval_bits(merlin_interval_bf16(y.hi))) {",
                "    merlin_bf16_interval_bins bins=merlin_bf16_interval_prepare(y);\n"
                "    if(mask[off+j] && bins.low_bits != bins.high_bits) {",
            ),
            (
                "y=merlin_interval_point(source_poly(exact*SCORE_SCALE-mx)); counts[2]++;",
                "y=merlin_interval_point(source_poly(exact*SCORE_SCALE-mx)); bins=merlin_bf16_interval_prepare(y); counts[2]++;",
            ),
            (
                "pl[off+j]=merlin_interval_bf16(y.lo);ph[off+j]=merlin_interval_bf16(y.hi);p[off+j]=merlin_interval_bf16((float)(((double)y.lo+y.hi)*.5));",
                "pl[off+j]=bins.lo;ph[off+j]=bins.hi;p[off+j]=merlin_bf16_interval_midpoint(y,bins);",
            ),
        )
        for before, after in replacements:
            if text.count(before) != 1:
                raise ValueError("probability refinement lifecycle changed; refused")
            text = text.replace(before, after)
        text = '#include "prepared_bf16_interval.h"\n' + text
    if prepare_softmax_domain:
        start = text.index("static int soft_details(")
        finish = text.index("\nstatic int endpoint_intervals(", start)
        original = text[start:finish]
        checked = original.replace("static int soft_details(", "static int soft_details_checked(", 1)
        selected = original
        soft_replacements = (
            (
                " if(!(root_prepared.fast_valid))return 0;\n",
                " if(!(root_prepared.fast_valid))return 0;\n"
                " merlin_prepared_softmax_domain domain=merlin_softmax_domain_prepare(&root_env,&root_prepared,SCORE_SCALE,CHUNK,LANES,2);\n"
                " if(!domain.valid||!merlin_softmax_admit_active_spans(&domain,lo,hi,mask,(size_t)rows*KEYS))\n"
                "  return soft_details_checked(q,k,mask,lo,hi,p,pl,ph,dl,dh,alpha,rows,counts,yl,yh,maxima);\n",
            ),
            (
                "merlin_f32_interval lanes[LANES];for(int j=0;j<LANES;j++)lanes[j]=merlin_interval_point(0);",
                "merlin_soft_interval lanes[LANES];for(int j=0;j<LANES;j++)lanes[j]=(merlin_soft_interval){0,0};",
            ),
            (
                "merlin_f32_interval x=merlin_interval_sub(merlin_interval_positive_scale(merlin_interval(lo[off+j],hi[off+j]),SCORE_SCALE),merlin_interval_point(mx));",
                "merlin_soft_interval score=merlin_softmax_score(&domain,lo[off+j],hi[off+j],mx);merlin_f32_interval x={score.lo,score.hi,1};",
            ),
            (
                "lanes[j%LANES]=merlin_interval_add(lanes[j%LANES],y);",
                "if(!y.valid)return 0;lanes[j%LANES]=merlin_softmax_lane_add(lanes[j%LANES],(merlin_soft_interval){y.lo,y.hi});",
            ),
            (
                "lanes[j]=merlin_interval_add(lanes[j],lanes[j+n]);",
                "lanes[j]=merlin_softmax_lane_add(lanes[j],lanes[j+n]);",
            ),
            ("alpha[row*2+tile]=a;", "if(!merlin_softmax_admit_alpha(a))return 0;alpha[row*2+tile]=a;"),
            (
                "merlin_interval_scalar_fma(a,den,lanes[0])",
                "merlin_interval_scalar_fma(a,den,(merlin_f32_interval){lanes[0].lo,lanes[0].hi,1})",
            ),
        )
        for before, after in soft_replacements:
            if selected.count(before) != 1:
                raise ValueError("private softmax source schedule changed; refused")
            selected = selected.replace(before, after)
        if word_interval_enclosure:
            selected = selected.replace("merlin_softmax_domain_prepare(", "merlin_softmax_word_domain_prepare(")
        text = text[:start] + '#include "prepared_softmax_interval.h"\n' + checked + selected + text[finish:]
    if prepare_encoded_rows:
        h, r, d, k = plan.heads, plan.query_rows, plan.depth, plan.chunk
        head_bytes = 4 * (22 * r * d + 4 * k * d + 14 * r * k + 6 * r) + 2 * r * k + r + r * d
        product_bytes = (
            4 * (6 * r * k + 2 * k * d) + 8 * (2 * r * k + k * d + r + k) + 3 * r * k + 3 * k * d + 4 * r * k
        )
        required = h * head_bytes + product_bytes + 140 * k + 13 * h * d + r + 1024 + r + k
        if integer_reconstruction:
            required += 8 * r * k
        if required > 2**31 - 1:
            raise ValueError("encoded row workspace exceeds supported bound")
        from .encoded_row_equality import prepare_encoded_row_equality

        text = prepare_encoded_row_equality(text)
    if fuse_encoded_witness:
        from .fused_encoded_witness import prepare_fused_encoded_witness

        text = prepare_fused_encoded_witness(text)
    if prepare_softmax_spans:
        from .prepared_softmax_spans import prepare_softmax_produced_spans

        text = prepare_softmax_produced_spans(text)
    if polynomial_batch_four:
        from .prepared_polynomial_batch import prepare_polynomial_batch_four

        text = prepare_polynomial_batch_four(text)
    if exact_bound_conversion is not None:
        from .exact_bound_conversion import emit_exact_bound_conversion_permission

        text = emit_exact_bound_conversion_permission(exact_bound_conversion) + text
    if prepare_probability_points:
        from .probability_point_spans import prepare_probability_point_spans

        text = prepare_probability_point_spans(text)
    definitions = {
        "HEADS": plan.heads,
        "ROWS": plan.query_rows,
        "DEPTH": plan.depth,
        "CHUNK": plan.chunk,
        "KEYS": 2 * plan.chunk,
        "SEGMENT": plan.segment,
        "PARTS": 3,
        "LANES": plan.denominator_lanes,
        "SCORE_SCALE": _f32(plan.score_scale),
    }
    polynomial = (
        "{"
        + ",".join(
            [
                _f32(plan.polynomial_cutoff),
                _f32(plan.polynomial_scale),
                "{" + ",".join(_f32(x) for x in plan.polynomial_coefficients) + "}",
                _f32(plan.polynomial_bit_multiplier),
                _f32(plan.polynomial_bit_bias),
            ]
        )
        + "}"
    )
    quant = (
        "{"
        + ",".join([_f32(plan.quant_divisor), _f32(plan.quant_epsilon), str(plan.quant_lower), str(plan.quant_upper)])
        + "}"
    )
    return (
        text.replace("@DEFINES@", "\n".join(f"#define {k} {v}" for k, v in definitions.items()))
        .replace("@POLYNOMIAL@", polynomial)
        .replace("@QUANT@", quant)
        .replace("@SYMBOL@", symbol)
    )
