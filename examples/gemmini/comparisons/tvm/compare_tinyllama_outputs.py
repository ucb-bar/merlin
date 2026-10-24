"""Diagnose one TinyLlama call's logits, active KV cache and positions.

Consumes the out0/out1/out2 NPZ files produced by read_package_session.py.
This numerical diagnostic does not establish execution provenance or EL4 admission.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def statistics(reference, actual):
    a, b = np.asarray(reference), np.asarray(actual)
    if a.shape != b.shape or a.dtype != b.dtype or a.dtype.kind not in "fi":
        raise ValueError("Expected matching floating/integer shapes and dtypes")
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    equal = a == b
    close = (finite & (error <= 2e-6 + 1e-5 * np.abs(a.astype(np.float64)))) if a.dtype.kind == "f" else equal
    mismatch = np.argwhere(~close)
    bits_a = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
    bits_b = np.ascontiguousarray(b).view(np.uint8).reshape(-1, b.dtype.itemsize)
    # INT64 positions must not lose a one-unit error through FP64 conversion.
    maximum = (
        max((abs(int(x) - int(y)) for x, y in zip(a.flat, b.flat)), default=None)
        if a.dtype.kind == "i"
        else float(error[finite].max())
        if finite.any()
        else None
    )
    return {
        "count": a.size,
        "nonfinite": int((~finite).sum()),
        "exact": int(np.all(bits_a == bits_b, axis=-1).sum()),
        "within_tolerance": int(close.sum()),
        "passed": bool(close.all()),
        "max_abs_finite": maximum,
        "first_failure": mismatch[0].tolist() if mismatch.size else None,
    }


def compare_call(reference, actual):
    for tensors in (reference, actual):
        if set(tensors) != {"out0", "out1", "out2"}:
            raise ValueError("Expected exactly out0 logits, out1 cache, out2 positions")
        logits, cache, position = (tensors[f"out{i}"] for i in range(3))
        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.dtype != np.float32
            or cache.ndim != 6
            or cache.shape[0] != 2
            or cache.shape[2] != 1
            or cache.dtype != np.float32
            or position.dtype != np.int64
            or position.shape != (cache.shape[1],)
        ):
            raise ValueError("Invalid TinyLlama FP32 logits/KV and INT64 position ABI")
        if np.any(position < 0) or np.any(position > cache.shape[-2]):
            raise ValueError("Position is outside the cache capacity")
    result = {
        "rtol": 1e-5,
        "atol": 2e-6,
        "integer": "exact",
        "logits": statistics(reference["out0"], actual["out0"]),
        "positions": statistics(reference["out2"], actual["out2"]),
        "cache_all_slots": statistics(reference["out1"], actual["out1"]),
        "cache_by_layer": [],
    }
    for layer, length in enumerate(reference["out2"]):
        record = {
            "layer": layer,
            "reference_used_length": int(length),
            "actual_used_length": int(actual["out2"][layer]),
        }
        for index, name in enumerate(("key", "value")):
            a, b = reference["out1"][index, layer], actual["out1"][index, layer]
            record[name + "_active"] = statistics(a[..., :length, :], b[..., :length, :])
            record[name + "_padding"] = statistics(a[..., length:, :], b[..., length:, :])
            record[name + "_padding"].update(
                {
                    "reference_nonzero": int(np.count_nonzero(a[..., length:, :])),
                    "actual_nonzero": int(np.count_nonzero(b[..., length:, :])),
                }
            )
        result["cache_by_layer"].append(record)
    result["first_active_cache_failure"] = next(
        (
            row["layer"]
            for row in result["cache_by_layer"]
            if not row["key_active"]["passed"] or not row["value_active"]["passed"]
        ),
        None,
    )
    result["passed"] = all(result[name]["passed"] for name in ("logits", "positions", "cache_all_slots"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as reference, np.load(args.actual, allow_pickle=False) as actual:
        result = compare_call(dict(reference), dict(actual))
    for name in ("reference", "actual"):
        path = getattr(args, name).resolve()
        with path.open("rb") as stream:
            result[name] = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
