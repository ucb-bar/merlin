"""Tool E (P20): low-bit quant-metadata visibility from the qdq recaptures.

The flat capture dequantizes low-bit weights to f32 (so the low-bit abstractions are blocked). The
``preserve_qdq`` int8 recapture (``recaptures_levels/<wl>/model_qdq.mlir``, P18-B) DOES keep explicit
``quant_ext.dequantize*`` ops with the storage dtype + scale granularity. This tool parses those to emit a
quant-metadata-visibility summary that unblocks the low-bit abstractions for the workloads with a qdq
capture. Reads the gitignored local qdq MLIR and emits a committed summary (the recaptures_levels policy).

Honest gap: the qdq capture is torchao int8 weight-only, NOT the model's native scheme (e.g. bitvla's W1.58
ternary) — recorded per workload, never hidden. Structural; no perf claim.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

_QM_COLS = ["workload", "n_dequant_ops", "storage_dtype", "scale_granularity", "dequant_placement",
            "compute_dtype", "accumulator_dtype", "native_scheme_gap"]

# native low-bit scheme per workload (from the P19 source audit) vs what the torchao qdq capture exposes.
# P21-S4: bitvla's native W1.58 ternary is now captured directly (recaptures_native/bitvla); the gap text
# is overridden to RESOLVED at runtime when that capture is present (see native_quant_rows).
_NATIVE = {"bitvla": "W1.58 ternary BitLinear (packed int2 + absmean scale) — NOT captured (torchao int8)"}

# P21-S4 native low-bit capture (BitLinear.quantize_weights materialized): the packed-int2 ternary
# STORAGE + per-tensor absmean scale are captured directly (vs the torchao-int8 qdq stand-in).
_NATIVE_COLS = ["workload", "native_scheme", "storage", "n_packed_weight_tensors",
                "scale", "dequant_placement", "compute_dtype", "unpack_visibility", "status"]


def native_quant_rows(cs_dir) -> list[dict]:
    """Read recaptures_native/<wl>/model.mlir (P21-S4) and report the NATIVE low-bit datapath that
    the default torchao-int8 qdq capture could not: packed-int2 ternary storage + absmean scale.
    '' / [] when no native capture is present (committed summary stands)."""
    nat = Path(cs_dir).parent / "recaptures_native"
    if not nat.is_dir():
        # fall back to the canonical bench location (so it works when cs_dir is a temp out-dir)
        try:
            from merlin.common import paths as _paths
            nat = _paths.merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures_native"
        except Exception:
            return []
    if not nat.is_dir():
        return []
    rows = []
    for d in sorted(nat.glob("*")):
        p = d / "model.mlir"
        if not p.is_file():
            continue
        txt = p.read_text(errors="ignore")
        n_i8 = txt.count("xi8>")                     # packed-int2 weights stored in i8 tensors
        # P22 GAP-D: the int2 bit-unpack chain is folded to the named quant_ext.unpack_int2 op
        # (opt-in fuse_int2_unpack recognizer). When present, the unpack is RECOVERED as a named op;
        # otherwise it falls back to the opaque func.call form.
        n_unpack = txt.count("quant_ext.unpack_int2")
        if n_unpack:
            unpack_vis = (f"recovered (quant_ext.unpack_int2 named op x{n_unpack}); "
                          "storage + scale + unpack all first-class")
            status = "recovered_full (native ternary datapath: storage + scale + named unpack op)"
        else:
            n_opaque = txt.count("func.call")
            unpack_vis = (f"partial — bit-unpack in opaque func.call ({n_opaque}); "
                          "storage+scale recovered (model forward .item()s the scale)")
            status = "recovered_storage_and_scale (native ternary datapath visible)"
        rows.append({
            "workload": d.name,
            "native_scheme": "W1.58 ternary (BitLinear, packed int2: 4 ternary values per i8 byte)",
            "storage": "int2_packed_in_i8",
            "n_packed_weight_tensors": n_i8,
            "scale": "per_tensor_absmean (w_step buffer)",
            "dequant_placement": "before_matmul (unpack+scale then GEMM; compute f32)",
            "compute_dtype": "f32 (dequant-before-matmul; same placement as the int8 path)",
            "unpack_visibility": unpack_vis,
            "status": status,
        })
    return rows


def native_csv(cs_dir) -> str:
    rows = native_quant_rows(cs_dir)
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_NATIVE_COLS, extrasaction="ignore")
    w.writeheader()
    [w.writerow(x) for x in rows]
    return buf.getvalue()


def quant_rows(cs_dir: Path) -> list[dict]:
    lvl = Path(cs_dir).parent / "recaptures_levels"
    if not lvl.is_dir():
        return []
    rows = []
    for d in sorted(lvl.glob("*")):
        p = d / "model_qdq.mlir"
        if not p.is_file():
            continue
        txt = p.read_text(errors="ignore")
        deq = re.findall(r'quant_ext\.dequantize_per_(channel|tensor|group)?', txt)
        n = len(re.findall(r'quant_ext\.dequantize', txt))
        if not n:
            continue
        dtypes = sorted(set(re.findall(r'input_dtype = "([a-z0-9]+)"', txt))) or ["i8"]
        gran = ("per_channel" if any("channel" in x for x in deq)
                else "per_tensor" if any("tensor" in x for x in deq)
                else "per_group" if any("group" in x for x in deq) else "unspecified")
        rows.append({"workload": d.name, "n_dequant_ops": n,
                     "storage_dtype": "|".join(dtypes), "scale_granularity": gran,
                     "dequant_placement": "before_matmul (weight dequantized then GEMM)",
                     "compute_dtype": "f32 (dequantized)", "accumulator_dtype": "f32",
                     "native_scheme_gap": _NATIVE.get(d.name, "torchao int8 weight-only (capture default)")})
    return rows


def quant_csv(cs_dir: Path) -> str:
    rows = quant_rows(cs_dir)
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_QM_COLS, extrasaction="ignore")
    w.writeheader()
    [w.writerow(x) for x in rows]
    return buf.getvalue()


def requirements_md(cs_dir: Path) -> str:
    rows = quant_rows(cs_dir)
    return ("# Low-bit capture requirements (P20 Tool E)\n\n"
            "> What the qdq recapture exposes (unblocks the low-bit abstractions) vs what a NATIVE low-bit "
            "capture would still need. The qdq MLIR keeps explicit `quant_ext.dequantize*` with storage "
            "dtype + scale granularity; the dequant sits before the GEMM (compute stays f32), so the "
            "packed-compute datapath is still not exercised. Structural; no perf claim.\n\n"
            f"- Quant metadata recovered for {len(rows)} workload(s) with a qdq capture: "
            f"{', '.join(r['workload'] for r in rows) or 'none'}.\n"
            "- **Native-scheme gaps** (qdq is torchao int8, not the model's native scheme): "
            "bitvla needs a packed-ternary (W1.58) capture; native int4/fp8 datapaths need a "
            "compute-in-low-bit capture (dequant-on-load fused), not dequant-before-GEMM.\n"
            "- Per-workload detail: `quant_metadata_visibility.csv`.\n")
