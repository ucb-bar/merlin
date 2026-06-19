"""Tool B (P20): CADOSys-style operand-locality targets from the recovered data-movement facts.

Turns the per-region byte facts (``data_movement_table.csv``) into a per-operand locality map: how big each
operand is, the scope over which it is reused (within_op / across_ops / across_K / across_decode /
across_replan), and whether it is a resident (scratchpad/HAL-object) or streamed (cache) candidate. Plus a
per-workload resident-capacity-by-dtype requirement. Structural only; across-K scopes inherit the
configured-K caveat (K is sidecar/assumed, the loop is unrolled). No perf claim.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

_LOC_COLS = ["workload", "region", "operand", "bytes", "reuse_scope", "resident_candidate",
             "cache_or_scratchpad", "k_dependent", "evidence"]
_CAP_COLS = ["workload", "region", "resident_f32_B", "resident_bf16_B", "resident_int8_B",
             "invocations", "k_basis"]


def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def _rows(path: Path):
    return list(csv.DictReader(io.StringIO(path.read_text()))) if path.is_file() else []


def locality_rows(cs_dir: Path) -> list[dict]:
    """Per (region, operand): bytes + reuse scope + resident/scratchpad-vs-cache candidate."""
    out = []
    for r in _rows(Path(cs_dir) / "data_movement_table.csv"):
        wl, reg = r["workload"], r["region"]
        inv = _int(r.get("invocations")) or 1
        across = reg == "repeated_head" and inv > 1
        wb, ab, ob, kv = (_int(r.get("weight_bytes")), _int(r.get("activation_input_bytes")),
                          _int(r.get("output_bytes")), _int(r.get("kv_bytes")))
        out.append({"workload": wl, "region": reg, "operand": "weight", "bytes": wb,
                    "reuse_scope": "across_K" if across else "across_replan",
                    "resident_candidate": "yes" if wb else "no",
                    "cache_or_scratchpad": "scratchpad/HAL_resident_object" if wb else "n/a",
                    "k_dependent": "yes" if across else "no", "evidence": "data_movement_table.csv"})
        out.append({"workload": wl, "region": reg, "operand": "activation_in", "bytes": ab,
                    "reuse_scope": "within_op", "resident_candidate": "no",
                    "cache_or_scratchpad": "cache/streamed", "k_dependent": "no",
                    "evidence": "data_movement_table.csv"})
        out.append({"workload": wl, "region": reg, "operand": "output", "bytes": ob,
                    "reuse_scope": "within_op", "resident_candidate": "no",
                    "cache_or_scratchpad": "cache/streamed", "k_dependent": "no",
                    "evidence": "data_movement_table.csv"})
        if kv:
            out.append({"workload": wl, "region": reg, "operand": "kv_state", "bytes": kv,
                        "reuse_scope": "across_decode", "resident_candidate": "yes (KV cache)",
                        "cache_or_scratchpad": "scratchpad/HAL_resident_object", "k_dependent": "yes",
                        "evidence": "data_movement_table.csv (kv inputs)"})
    return out


def capacity_rows(cs_dir: Path) -> list[dict]:
    """Per region: resident weight capacity required at f32 / bf16 / int8 (byte-scaled)."""
    out = []
    for r in _rows(Path(cs_dir) / "data_movement_table.csv"):
        wb = _int(r.get("weight_bytes"))
        if not wb:
            continue
        inv = _int(r.get("invocations")) or 1
        out.append({"workload": r["workload"], "region": r["region"], "resident_f32_B": wb,
                    "resident_bf16_B": _int(r.get("resident_bf16_B")) or wb // 2,
                    "resident_int8_B": _int(r.get("resident_int8_B")) or wb // 4,
                    "invocations": inv, "k_basis": "configured/assumed"})
    return out


def locality_csv(cs_dir: Path) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_LOC_COLS, extrasaction="ignore")
    w.writeheader()
    [w.writerow(x) for x in locality_rows(cs_dir)]
    return buf.getvalue()


def capacity_csv(cs_dir: Path) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CAP_COLS, extrasaction="ignore")
    w.writeheader()
    [w.writerow(x) for x in capacity_rows(cs_dir)]
    return buf.getvalue()


def boundary_md(cs_dir: Path) -> str:
    loc = locality_rows(cs_dir)
    resident = [r for r in loc if r["resident_candidate"].startswith("yes")]
    return ("# Cache vs scratchpad boundary (P20 Tool B)\n\n"
            "> Per-operand locality from recovered data movement. Weights (and KV state, where present) are "
            "**resident/scratchpad/HAL-object candidates** — reused across the K-loop / decode loop; "
            "activations + outputs are **streamed/cache** (within-op). Structural; across-K reuse rests on "
            "the configured/assumed K (the loop is unrolled by export).\n\n"
            f"- {len(resident)} resident-candidate operand rows (weight + kv_state) across "
            f"{len({r['workload'] for r in resident})} workloads.\n"
            "- Full per-region resident capacity by dtype: `capacity_requirement_table.csv` "
            "(f32 / bf16 / int8 byte-scaled).\n"
            "- Per-operand detail + reuse scope: `operand_locality_table.csv`.\n")
