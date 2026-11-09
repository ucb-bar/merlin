"""Explicit Atlas domain choices passed to the existing application adapter."""


def add_reduction_policy_argument(parser):
    from atlas_mlir_native.reduction import REDUCTION_POLICIES, REDUCTION_POLICY

    parser.add_argument("--reduction-policy", choices=REDUCTION_POLICIES, default=REDUCTION_POLICY,
                        help="explicit Atlas reduction input domain, recorded in precision and compilation receipts")


def verify_reduction_policy(kind, precision, compilation, selected):
    """Authenticate reused policy choices, including older precision-only receipts."""
    if kind == "attention" or kind.startswith("layer") or kind in {"norm", "head"}:
        if (precision.get("reduction") != selected
                or compilation.get("reduction_policy", precision.get("reduction")) != selected):
            raise ValueError(f"compiled region uses a different selected reduction policy: {kind}")


def build_precision_policy(kind, source_hash, *, reduction_policy, softmax_mask_policy):
    """Select the existing unchanged-source example arithmetic policies."""
    from atlas_mlir_native.already_quantized import POLICY
    from atlas_mlir_native.application_precision import CONTRACTION_POLICY
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.row_softmax import SOFTMAX_POLICY

    policy = dict(schema="atlas.diagnostic_application_precision.v1", source_sha256=source_hash,
                  target_identity=TARGET_IDENTITY, contraction=CONTRACTION_POLICY,
                  row_softmax=SOFTMAX_POLICY, host_support=["transpose", "reshape", "copy"],
                  already_quantized=POLICY)
    if kind == "attention" or kind.startswith("layer") or kind in {"norm", "head"}:
        from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
        from atlas_mlir_native.gelu import GELU_POLICY
        from atlas_mlir_native.pointwise import POINTWISE_POLICY, BF16_POINTWISE_POLICY
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY, GUARDED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        policy.update(schema="atlas.diagnostic_application_precision.v19",
                      contraction=dict(policy=SCALED_CONTRACTION_POLICY, operand_scale_exponents=[0, 0]),
                      host_support=["transpose", "reshape", "copy", "gather", "concat"],
                      epilogues=["relu"], pointwise=POINTWISE_POLICY, reduction=reduction_policy,
                      pointwise_math=MATH_POLICY, sigmoid=SIGMOID_POLICY, masked_softmax=MASKED_SOFTMAX_POLICY,
                      host_predicates=["mask", "select"], host_integer_ops=["i64_add"],
                      guarded_masked_softmax=GUARDED_SOFTMAX_POLICY, gelu=GELU_POLICY,
                      host_float_casts=["f32_to_bf16", "bf16_to_f32"],
                      host_float_arithmetic=["f32_neg"],
                      already_quantized_mask=softmax_mask_policy)
        if kind.startswith("layer") or kind in {"norm", "head"}:
            from atlas_mlir_native.source_exp import POLICY as SOURCE_EXP_POLICY

            policy["bf16_pointwise"] = BF16_POINTWISE_POLICY
            policy["host_float_arithmetic"] += ["bf16_neg", "bf16_add", "bf16_sub"]
            policy["source_exp"] = SOURCE_EXP_POLICY
    if kind == "head":
        from merlin.compile.c_bf16_linear import POLICY as HOST_BF16_LINEAR_POLICY
        policy["host_bf16_linear"] = HOST_BF16_LINEAR_POLICY
    return policy
