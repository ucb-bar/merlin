"""Numerical oracle for the microscaling (MX) block-FP datapath.

MX (block-scaled fp8/fp6/fp4 with an E8M0 per-block scale + a 16-deep systolic ACC_E/ACC_M accumulate
schedule) is a FLOAT datapath: the integer reference/simulate engine cannot grade it (see
``capsule_runner`` — L0/L1 are skipped for float datapaths). This module supplies the missing numerical
tier by DELEGATING to the derived MX reference ``mlc.validate.mx_ref`` — which mlc itself validates
bit-exact against the radiance-kernels C++ golden (``lib/golden/mx_golden.cpp``, a mirror of the MX RTL /
``MxRequantizer.scala``). Merlin does NOT re-implement the schedule; it calls the validated reference and
fails closed when mlc is unavailable, so an MX capsule gets a live numerical datapath grade without the
RTL sim. The cycle-accurate RTL cert (a Verilator/VCS build of the gemmini-mx config) stays a separate,
higher tier.

TARGET-AGNOSTIC: the subject here is the MX *format* (a datapath fact mlc derives from the RTL), not a
target name. Any target whose contract declares an MX datapath routes its float matmul grade through here.
"""
from __future__ import annotations

_FMT = {"fp8": 0, "mxfp8": 0, "fp8_e4m3": 0,
        "fp6": 1, "mxfp6": 1, "fp6_e3m2": 1,
        "fp4": 2, "mxfp4": 2, "fp4_e2m1": 2}


def mx_datapath_available() -> bool:
    """True when the derived MX reference (mlc) is importable in this environment."""
    try:
        import mlc.validate.mx_ref  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — any import failure means the oracle is unavailable
        return False


def mx_matmul(a_codes, w_codes, sa_codes, sb_codes, m: int, n: int, k: int,
              fmt: str = "fp8", lut_a=None, lut_b=None, g: int = 0):
    """Run an MX block-scaled ``A @ W`` on the derived datapath reference and return the ``(m, n)`` bf16
    output decoded to float32 — or ``None`` (fail closed) when mlc is unavailable or ``fmt`` is unknown.

    ``a_codes``/``w_codes`` are the raw quantized operand bytes (fp8: one byte/element; fp4/fp6 packed as
    the reference expects); ``sa_codes``/``sb_codes`` are the E8M0 block-scale codes (one per 32-element K
    group, laid out ``[K/32, M]`` and ``[K/32, N]``). ``lut_a``/``lut_b``/``g`` are the per-group LUTs +
    group shift the fp6 (LUT-indexed) format needs (ignored for fp8/fp4). The MX block scale + ACC_E/ACC_M
    schedule are the reference's; this only marshals arrays and decodes the bf16 result."""
    fmt_id = _FMT.get(fmt)
    if fmt_id is None:
        return None
    try:
        import numpy as np

        from mlc.validate import mx_ref as _mx
    except Exception:  # noqa: BLE001 — mlc absent: no MX oracle in this env, fail closed
        return None
    a = np.asarray(a_codes, dtype=np.uint8)
    w = np.asarray(w_codes, dtype=np.uint8)
    sa = np.asarray(sa_codes, dtype=np.int32).reshape(k // 32, m)
    sb = np.asarray(sb_codes, dtype=np.int32).reshape(k // 32, n)
    la = None if lut_a is None else np.asarray(lut_a, dtype=np.uint8)
    lb = None if lut_b is None else np.asarray(lut_b, dtype=np.uint8)
    bits = np.asarray(_mx.mx_matmul(a, w, sa, sb, m, n, k, fmt=fmt_id, lutA=la, lutB=lb, G=g)
                      ).reshape(m, n).astype(np.uint32)
    return (bits << 16).view(np.float32)          # bf16 bits -> float32


def grade_matmul(operand_codes: dict, sa_codes, sb_codes, golden_out) -> dict:
    """Grade an MX matmul capsule BIT-EXACT: re-run the datapath oracle over the golden's RAW operand codes
    (``operand_codes`` = the golden's ``operand_codes`` block: ``A_bytes``/``B_bytes`` + shapes, ``fmt``,
    ``M``/``N``/``K``, ``G``, and the fp6 ``lutA``/``lutB``) and compare to the golden output. Uses the raw
    codes — not the display-rounded ``decoded`` floats — so the comparison is exact. Returns
    ``{status, exact, max_abs_err}`` or ``{status: 'oracle_unavailable'}`` (fail closed)."""
    import numpy as np
    oc = operand_codes
    a = np.array(oc["A_bytes"], dtype=np.uint8).reshape(oc["A_shape"])
    b = np.array(oc["B_bytes"], dtype=np.uint8).reshape(oc["B_shape"])
    out = mx_matmul(a, b, sa_codes, sb_codes, oc["M"], oc["N"], oc["K"], fmt=oc["fmt"],
                    lut_a=oc.get("lutA"), lut_b=oc.get("lutB"), g=oc.get("G", 0))
    if out is None:
        return {"status": "oracle_unavailable"}
    gold = np.asarray(golden_out, dtype=np.float32)
    exact = bool(np.array_equal(out, gold))
    return {"status": "pass" if exact else "fail", "exact": exact,
            "max_abs_err": float(np.max(np.abs(out - gold)))}
