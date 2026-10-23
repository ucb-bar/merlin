"""Exact byte comparison for integer capture; floating diagnostics remain explicit."""


def integerized_agreement(before, after, *, atol: float, rtol: float, output_abi) -> dict:
    """Compare two independently supplied executions on the capture input."""
    import torch

    left, left_abi = output_abi(before)
    right, right_abi = output_abi(after)
    rows = []
    compatible = left_abi == right_abi and bool(left)
    for original, rewritten in zip(left, right):
        lhs = original.detach().to(torch.float64)
        rhs = rewritten.detach().to(torch.float64)
        same_shape = lhs.shape == rhs.shape
        finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()) if same_shape else False
        if same_shape and finite and lhs.numel():
            delta = (lhs - rhs).abs()
            max_abs = float(delta.max())
            max_rel = float((delta / lhs.abs().clamp_min(1e-12)).max())
            if atol == 0 and rtol == 0:
                within = original.dtype == rewritten.dtype and torch.equal(
                    original.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                    rewritten.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                )
            else:
                within = bool(torch.all(delta <= atol + rtol * lhs.abs()))
        elif same_shape and finite:
            max_abs = max_rel = 0.0
            within = True
        else:
            max_abs = max_rel = 0.0
            within = False
        byte_equal = (
            same_shape
            and original.dtype == rewritten.dtype
            and torch.equal(
                original.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                rewritten.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            )
        )
        if atol == 0 and rtol == 0:
            within = byte_equal  # Complete words include any proved IEEE special payload.
        zero_signs = (
            int(((lhs == 0) & (rhs == 0) & (torch.signbit(lhs) != torch.signbit(rhs))).sum()) if same_shape else 0
        )
        rows.append(
            {
                "dtype_before": str(original.dtype),
                "dtype_after": str(rewritten.dtype),
                "byte_equal": byte_equal,
                "signed_zero_differences": zero_signs,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "within_tolerance": within,
                "finite": finite,
                "atol": atol,
                "rtol": rtol,
                "shape": list(original.shape),
            }
        )
    passed = compatible and len(rows) == len(left) and all(row["within_tolerance"] for row in rows)
    return {
        "status": "passed" if passed else "failed",
        "samples": 1,
        "atol": atol,
        "rtol": rtol,
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "max_rel": max((row["max_rel"] for row in rows), default=0.0),
        "outputs": rows,
        "finite": compatible and all(row["finite"] for row in rows),
    }
