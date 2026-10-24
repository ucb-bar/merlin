"""Exact native dtype gate, independent of tolerance-based corpus comparisons."""


def grade_native_bits(record, outputs, *, execution_error=None):
    boundary = record.get("native_eager_boundary")
    if not boundary or boundary.get("authority") != "untransformed_native_cpu_eager":
        return dict(status="ungradable", reason="native dtype execution lacks an independent eager byte boundary")
    if outputs is None or execution_error is not None:
        return dict(status="execution_failed", reason="native dtype execution produced no complete byte boundary")
    indices = record.get(
        "native_eager_indices",
        {"outputs": record["output_indices"], "posts": record.get("semantic_boundary", {}).get("post_indices", [])},
    )
    expected = {}
    for boundary_indices, values in (
        (indices["outputs"], boundary["output_bytes"]),
        (indices["posts"], boundary["post_input_bytes"]),
    ):
        if len(boundary_indices) != len(values):
            return dict(status="ungradable", reason="native eager reference does not cover the exported boundary")
        for index, raw in zip(boundary_indices, values, strict=True):
            if index in expected and expected[index] != raw:
                return dict(status="ungradable", reason="native eager aliases disagree in their storage bits")
            expected[index] = raw
    comparisons = [
        dict(output_index=i, status="pass" if i < len(outputs) and outputs[i].hex() == raw else "mismatch")
        for i, raw in sorted(expected.items())
    ]
    passed = all(row["status"] == "pass" for row in comparisons)
    return dict(
        status="pass" if passed else "mismatch",
        comparison="exact_native_eager_storage_bits",
        signed_zero="distinct",
        nan_payload="preserved",
        results=comparisons,
        reason="" if passed else "native dtype output bits differ from independent CPU eager",
    )
