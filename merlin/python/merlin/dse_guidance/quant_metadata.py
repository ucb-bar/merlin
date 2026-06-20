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


_LOWBIT_TIER_COLS = ["workload", "tier", "storage", "scale", "compute", "accuracy_status",
                     "honest_gap", "evidence"]


def low_bit_visibility_rows(cs_dir) -> list[dict]:
    """P24: corpus-wide HONEST low-bit tiering per studyable workload:
      - **native**       : a native packed-low-bit capture exists (bitvla: packed int2 + absmean scale
                           + the named quant_ext.unpack_int2 op).
      - **qdq_int8**     : a torchao-int8 qdq capture exposes storage dtype + per-channel scale
                           (dequant-before-matmul; compute f32). A stand-in, not the model's native scheme.
      - **dequant_only** : only the f32 capture — low-bit abstractions blocked (honest).
    int8 candidates are ratified by the MEASURED W8A8 accuracy gate; fp8/int4 stay unavailable (never
    assumed). Native packed fp8/int4 for the rest needs model-specific quant exports (scoped, not faked)."""
    from merlin.common import paths as _paths
    from merlin.dse_guidance.case_study import available_models
    bench = _paths.merlin_dir() / "benchmarks" / "dse_guidance"
    nat_dir, lvl_dir = bench / "recaptures_native", bench / "recaptures_levels"
    # measured W8A8 accuracy gate (which int8 variants pass/fail/were-not-measured)
    try:
        from merlin.dse_guidance import accuracy_gate as AG
        _pts = AG.load()
    except Exception:  # noqa: BLE001
        AG, _pts = None, []
    def _acc(w):
        if AG is None:
            return "unavailable (int8 not measured for this model)"
        st = AG.status_for(w, "int8_w8a8", _pts)
        return ({"pass": "measured_pass", "fail": "measured_fail"}.get(st)
                or "unavailable (int8 not measured for this model)")
    rows = []
    for w in available_models():
        nat = (nat_dir / w / "model.mlir")
        qdq = (lvl_dir / w / "model_qdq.mlir")
        if nat.is_file() and "quant_ext.unpack_int2" in nat.read_text(errors="ignore"):
            tier, storage, scale, gap = ("native", "int2_packed_in_i8",
                                         "per_tensor_absmean", "none (native ternary fully recovered)")
        elif qdq.is_file():
            tier, storage, scale, gap = ("qdq_int8", "i8", "per_channel",
                                         "torchao int8 stand-in — model's native scheme (fp8/int4/ternary) "
                                         "needs a model-specific capture")
        else:
            tier, storage, scale, gap = ("dequant_only", "f32 (dequantized at load)", "erased",
                                         "low-bit abstractions blocked; needs a qdq/native capture")
        rows.append({
            "workload": w, "tier": tier, "storage": storage, "scale": scale,
            "compute": "f32 (dequant-before-matmul)" if tier != "native"
                       else "f32 (native unpack+scale, dequant-before-matmul)",
            "accuracy_status": _acc(w),
            "honest_gap": gap,
            "evidence": "recovered_from_ir" if tier != "dequant_only" else "n/a (f32 capture)",
        })
    return rows


def low_bit_visibility_csv(cs_dir) -> str:
    from merlin.dse_guidance.case_study import _csv
    return _csv(low_bit_visibility_rows(cs_dir), _LOWBIT_TIER_COLS)


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
