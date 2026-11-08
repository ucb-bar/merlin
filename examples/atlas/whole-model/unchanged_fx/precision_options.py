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
