"""Process and text-console transport for an explicitly selected simulator.

The caller supplies the complete command, timeout, parser and error type.
Capture bytes first: invalid diagnostic UTF-8 must not discard a failed run's
entire console. A printed result or DONE never permits a nonzero process exit.
"""

import subprocess


def run_text(command, *, timeout, error_type, parse_console, label):
    process = subprocess.run([str(c) for c in command], capture_output=True, timeout=timeout)
    console = (process.stdout or b"").decode("utf-8", errors="replace") + (
        process.stderr or b"").decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise error_type(f"{label} exited {process.returncode} even though it may have printed OUT/DONE:\n{console[-2000:]}")
    return parse_console(console)


def compare_prefix(result, reference, elf, *, error_type):
    """Preserve the legacy typed-prefix comparison and completion requirements."""
    import numpy as np

    if reference is not None:
        if result["output_dtype"] == "i1":
            ref = np.asarray(reference).ravel()
            pref = np.asarray(result["prefix"])
            if ref.dtype != np.bool_:
                raise error_type("Boolean output requires an exact bool reference; coercion is forbidden")
            if pref.dtype != np.bool_ or np.any(pref.view(np.uint8) > 1) or np.any(ref.view(np.uint8) > 1):
                raise error_type("Boolean output/reference must retain canonical bool wire bytes")
            complete = pref.size == ref.size
            mismatch = int(np.count_nonzero(pref.view(np.uint8) != ref.view(np.uint8))) if complete else None
            result.update(ok=complete and mismatch == 0, mismatched_elements=mismatch, comparison="exact_i1")
            result["elf"] = str(elf)
            return result
        if result["output_dtype"] == "i64":
            ref = np.asarray(reference).ravel()
            if ref.dtype != np.int64:
                raise error_type("integer output requires an exact int64 reference; float coercion is forbidden")
            pref = np.asarray(result["prefix"])
            if pref.dtype != np.int64:
                raise error_type("integer output prefix must retain its exact int64 wire dtype")
            complete = ref.size == pref.size
            mismatch = int(np.count_nonzero(pref != ref[: pref.size])) if ref.size >= pref.size else None
            result.update(ok=complete and mismatch == 0, mismatched_elements=mismatch, comparison="exact_i64")
            result["elf"] = str(elf)
            return result
        ref = np.asarray(reference, dtype=np.float32).ravel()
        pref = np.asarray(result["prefix"])
        if result["output_dtype"] != "f32" or pref.dtype != np.float32:
            raise error_type("floating output prefix must retain its float32 wire dtype")
        k = len(pref)
        rel = float(np.abs(pref - ref[:k]).max()) / max(1e-9, float(np.abs(ref[:k]).max()))
        cos = float((pref @ ref[:k]) / (np.linalg.norm(pref) * np.linalg.norm(ref[:k]) + 1e-12))
        ok = cos > 0.9999 and rel < 1e-4
        result.update(rel=rel, cos=cos, ok=ok)
        # digest checks for large outputs (LM logits): argmax per row must match torch.
        if result["argmax"] is not None and ref.size % result["argmax"].size == 0:
            last = ref.size // result["argmax"].size
            ref_argmax = ref.reshape(-1, last).argmax(1)
            result["argmax_match"] = bool(np.array_equal(result["argmax"], ref_argmax))
            result["ok"] = ok and result["argmax_match"]
    result["elf"] = str(elf)
    return result
