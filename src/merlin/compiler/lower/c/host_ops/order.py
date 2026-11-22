"""Source operation order used by scalar host emitters.

No reassociation, fused arithmetic, flush-to-zero, or transcendental substitution.
NaN arithmetic payloads require separate source qualification and are refused.
"""
ORDER = {
    'relu': 'ordered x < +0 select; preserve signed zero and NaN bits',
    'batch_norm': 'Refuse non-finite input; per channel fold RN(RN(1/RN(sqrt(RN(var+eps))))*weight) and fused RN(bias-mean*alpha); per element fused RN(x*alpha+beta), then BF16 RNE.',
    'sigmoid': 'Refuse NaN input; widen BF16 exactly; RN(expf(-x)), RN(1+e), RN(1/d); one BF16 RNE.',
    'gelu_tanh': 'Refuse non-finite input; widen BF16 exactly; RN(x*x), RN(*x), RN(0.044715*cube), RN(x+that), RN(sqrt(2/pi)*sum), tanhf, RN(1+t), RN(x*0.5), RN(product); one BF16 RNE.',
    'gelu_none': 'Refuse non-finite input; widen BF16 exactly; RN(x*(1/sqrt(2))), erff, RN(1+e), RN(x*0.5), RN(product); one BF16 RNE.',
    'layer_norm': 'Refuse non-finite input, statistics or results before stores; per row the pinned torch 2.11.0+cpu AVX2/AVX512 kernel: chunked Welford moments over 16-lane BF16 vectors as two 8-lane binary32 halves with 7 fused multiply-adds (inner Welford m1 and m2, AddMomentsVec m2, scalar AddMoments m1 and sum, both output steps) and a rounded product and sum at the AddMomentsVec m1 and tail m2 updates, cascade and lane combination, RN(m2/n), RN(1/RN(sqrt(RN(var+eps)))), fma(fma(x,rstd,RN(-mean*rstd)),weight,bias) with weight 1 and bias 0 when absent; one BF16 RNE.',
    'affine_contraction': 'Refuse non-finite operands and a possible overflow before stores; per output from +0 add RN(f32(x)*f32(w)) in ascending lexicographic reduction-loop order, then RN(+bias); one BF16 RNE.',
    'window_max': 'Refuse NaN input; scan each window in loop order, replace when strictly greater, keep the first equal; skip padding.',
    'f32_add': 'Project operands, exact signed sum, one binary32 RNE rounding.',
    'f32_sub': 'Project operands, flip right sign, exact signed sum, one binary32 RNE rounding.',
    'f32_mul': 'Project operands, exact significand product, one binary32 RNE rounding.',
    'f32_div': 'Project operands, exact significand ratio, one binary32 RNE rounding.',
    'f64_mul': 'Project operands, exact 106-bit significand product, one binary64 RNE rounding.',
    'bf16_add': 'Project operands, exact signed sum after BF16 readout, one BF16 RNE rounding.',
    'bf16_sub': 'Project operands, flip right sign, exact signed sum, one BF16 RNE rounding.',
    'neg': 'Flip only the sign bit, including NaN payloads.',
    'i64_arithmetic': 'Unsigned bit-carrier arithmetic modulo 2^64, reinterpret as signed i64.',
    'cast': 'Decode source exactly, round once to destination with ties to even.',
    'copy': 'Copy each source byte unchanged; preserve all NaN payloads.',
    'reshape': 'Preserve dense row-major element order; copy bytes if materialized.',
}
