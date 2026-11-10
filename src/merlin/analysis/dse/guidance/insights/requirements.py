"""Derive analytical deadline, compute and data-movement requirement envelopes.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows

# =========================================================================== P17 requirements (E/F)
# Timing as an external REQUIREMENT envelope (work / deadline), never a measured hardware result;
# and an honest accounting of the operators this study does NOT count. Every input is tagged
# configured / reference / sweep; no row is a performance prediction.

_DEADLINE_MS = (50, 100, 200, 500)


_K_SWEEP = (4, 8, 16, 32)


_MEASURED_DISPATCH = {"small_llama"}  # dispatch_coupling.csv has a measured per-forward count


_ENV_COLS = [
    "workload",
    "region",
    "K",
    "K_basis",
    "deadline_ms",
    "deadline_basis",
    "required_compute_MAC_per_s",
    "required_weight_B_per_s_nonresident",
    "required_weight_B_per_s_resident",
    "required_command_rate_per_s",
    "command_rate_basis",
    "resident_capacity_bf16_B",
    "resident_capacity_int8_B",
]


def timing_requirement_envelope(cs_dir: Path) -> dict:
    """Part E: cross per-workload structural base facts (macs/step, resident weight bytes,
    dispatches/step, resident capacity by dtype — all already in requirements_table.csv) with
    scenario grids: replan deadlines {50,100,200,500 ms}, a K sweep {4,8,16,32}+configured, and the
    configured H/control-rate deadline. Each row is a REQUIREMENT (work / deadline), explicitly NOT a
    hardware performance prediction. Command rate is proxy-only except where a measured dispatch
    count exists (no silent fill)."""
    from merlin.analysis.dse.guidance.models import MODEL_ARCH, _base_model

    cs_dir = Path(cs_dir)
    req = {
        (r["workload"], r["region"], r["requirement"]): r["value"] for r in _csv_rows(cs_dir / "requirements_table.csv")
    }
    wls = sorted({w for (w, reg, _) in req if reg == "repeated_head"})
    rows = []
    for w in wls:
        arch = MODEL_ARCH.get(_base_model(w))
        Kcfg = arch.loop_count if arch else 1

        def g(name):
            v = req.get((w, "repeated_head", name))
            return float(v) if v not in (None, "") else 0.0

        macs_replan, weight_bytes, disp_replan = (
            g("macs_per_replan"),
            g("resident_capacity_required"),
            g("dispatches_per_replan"),
        )
        cap_bf16, cap_int8 = g("resident_capacity_bf16"), g("resident_capacity_int8")
        if Kcfg <= 0 or macs_replan <= 0:
            continue
        macs_step, disp_step = macs_replan / Kcfg, disp_replan / Kcfg
        cmd_basis = (
            "measured_dispatch_available"
            if _base_model(w) in _MEASURED_DISPATCH
            else "proxy_only (matmul-count, ~12x undercount; NOT a hardware command rate)"
        )
        scen = [(float(dl), "sweep") for dl in _DEADLINE_MS]
        if arch and arch.control_rate_hz and arch.action_horizon:
            scen.append(
                (
                    1000.0 * arch.action_horizon / arch.control_rate_hz,
                    f"derived_H/control_rate (H={arch.action_horizon}, {arch.control_rate_hz}Hz; configured/reference)",
                )
            )
        for K in sorted({Kcfg, *_K_SWEEP}):
            kb = "configured" if K == Kcfg else "sweep"
            for dl, db in scen:
                s = dl / 1000.0
                rows.append(
                    {
                        "workload": w,
                        "region": "repeated_head",
                        "K": K,
                        "K_basis": kb,
                        "deadline_ms": round(dl, 3),
                        "deadline_basis": db,
                        "required_compute_MAC_per_s": round(macs_step * K / s, 1),
                        "required_weight_B_per_s_nonresident": round(weight_bytes * K / s, 1),
                        "required_weight_B_per_s_resident": round(weight_bytes / s, 1),
                        "required_command_rate_per_s": round(disp_step * K / s, 2),
                        "command_rate_basis": cmd_basis,
                        "resident_capacity_bf16_B": int(cap_bf16),
                        "resident_capacity_int8_B": int(cap_int8),
                    }
                )
    return {"workloads": wls, "rows": rows}


def _fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}GB"


def _deadline_sensitivity_md(env, scope) -> str:
    dls = list(_DEADLINE_MS)
    L = [
        f"# Deadline-sensitivity of the requirement envelope ({scope})\n",
        "> Required **compute rate** (= configured-K replan MACs / deadline) as the replan deadline "
        "tightens. A REQUIREMENT, not a hardware prediction — a future accelerator must *exceed* "
        "this to meet the deadline. K is configured/reference; deadlines are a sweep.\n",
        "| workload | K | " + " | ".join(f"{d}ms" for d in dls) + " |",
        "|---|---|" + "|".join(["---"] * len(dls)) + "|",
    ]
    by = {}
    for r in env["rows"]:
        if r["K_basis"] == "configured" and r["deadline_basis"] == "sweep":
            by.setdefault(r["workload"], {})[int(r["deadline_ms"])] = r["required_compute_MAC_per_s"]
            by[r["workload"]]["K"] = r["K"]
    for w in env["workloads"]:
        d = by.get(w, {})
        L.append(f"| {w} | {d.get('K', '?')} | " + " | ".join(f"{d.get(dl, 0) / 1e9:.2f} GMAC/s" for dl in dls) + " |")
    L.append(
        "\nFull K×deadline grid (compute / weight-bandwidth resident-vs-nonresident / command "
        "rate): `timing_requirement_envelope.csv`.\n"
    )
    return "\n".join(L) + "\n"


def _residency_tradeoff_md(env, scope) -> str:
    L = [
        f"# Residency vs deadline tradeoff ({scope})\n",
        "> At configured K and a 100 ms replan deadline: the required **weight bandwidth** if weights "
        "are reloaded every step (non-resident) vs kept resident (loaded once). The ratio is exactly "
        "K — residency removes a K× bandwidth requirement at the cost of the resident capacity shown "
        "(by dtype). Structural requirement only.\n",
        "| workload | K | non-resident B/s | resident B/s | ratio | resident bf16 | resident int8 |",
        "|---|---|---|---|---|---|---|",
    ]
    seen = set()
    for r in env["rows"]:
        if r["K_basis"] != "configured" or int(r["deadline_ms"]) != 100 or r["deadline_basis"] != "sweep":
            continue
        w = r["workload"]
        if w in seen:
            continue
        seen.add(w)
        nr, rs = r["required_weight_B_per_s_nonresident"], r["required_weight_B_per_s_resident"]
        ratio = (nr / rs) if rs else 0
        L.append(
            f"| {w} | {r['K']} | {_fmt_bytes(nr)}/s | {_fmt_bytes(rs)}/s | {ratio:.0f}x | "
            f"{_fmt_bytes(r['resident_capacity_bf16_B'])} | "
            f"{_fmt_bytes(r['resident_capacity_int8_B'])} |"
        )
    L.append(
        "\nResidency is therefore a search axis (capacity to keep weights on-chip vs the K× "
        "reload bandwidth it avoids). Absolute bytes are small/random-init; the K× ratio and the "
        "capacity-by-dtype ordering are the robust signal.\n"
    )
    return "\n".join(L) + "\n"
