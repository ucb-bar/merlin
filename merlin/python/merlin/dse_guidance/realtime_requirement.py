"""Real-time deployment requirements (HW-INDEPENDENT) — P25.

Grounds the deadline machinery in the *named* real-time regimes the deployment literature uses, and
turns each into a hardware-INDEPENDENT requirement (a floor a machine must provide), NOT a performance
claim about any chip:

  * **VLA (action models)**: real-time control needs >= 30 Hz (33 ms/action); high-performance dexterous
    control targets 50-100 Hz. A forward pass produces an **action chunk of H actions**, so the next
    replan must finish within ``H / rate`` seconds — action chunking AMORTIZES the once-prefix over H.
  * **VLM / LLM (reasoning)**: chat needs ~6 tok/s, agentic workflows ~40 tok/s; perception wants
    **TTFT < 500 ms** (the prefill budget).

For each workload we report, per regime, the requirement implied by the structure we recover
(prefix / repeated / K from the loop capture, weight-bytes resident vs reload from ``arithmetic_intensity``):

  required_compute (MAC/s) = work-in-window / window_s
  required_weight_bandwidth (B/s), resident vs reload = weight-bytes-in-window / window_s

The two levers are explicit: **action chunking** divides the compute requirement by H; **residency**
divides the weight-bandwidth requirement by ~K (load once vs reload every step). NOTHING here claims a
chip meets the requirement — it is the minimum a machine MUST provide. Compute/bandwidth are Tier-A/B
(from recovered structure); the regime thresholds are `design_target` (cited literature values).
"""

from __future__ import annotations

from merlin.dse_guidance import models as M

_DB = 2                                   # bf16 bytes (matches arithmetic_intensity default)
_VLA_FAMILIES = {"flow_matching", "diffusion", "autoregressive_vla"}
_VLM_FAMILIES = {"llm"}
# named real-time regimes (design targets from the deployment literature; NOT measurements)
_VLA_HZ = [("VLA 30Hz (real-time baseline)", 30.0), ("VLA 50Hz (high-perf)", 50.0),
           ("VLA 100Hz (dexterous)", 100.0)]
_VLM_TOK = [("VLM 6 tok/s (chat)", 6.0), ("VLM 40 tok/s (agentic)", 40.0)]
_TTFT_MS = 500.0                          # time-to-first-token budget for perception/reasoning

_RT_COLS = ["workload", "family", "regime", "budget_ms", "window", "required_GMAC_per_s",
            "required_weight_GBps_resident", "required_weight_GBps_reload", "amortization",
            "evidence"]


def _ai_index(cs_dir):
    from merlin.dse_guidance.case_study import _csv  # noqa: F401  (ensure pkg import side-effects)
    import csv as _c
    from pathlib import Path
    p = Path(cs_dir) / "arithmetic_intensity.csv"
    if not p.is_file():
        return {}
    return {r["workload"]: r for r in _c.DictReader(p.read_text().splitlines())}


def realtime_rows(cs_dir) -> list[dict]:
    ai = _ai_index(cs_dir)
    rows = []
    for w, r in sorted(ai.items()):
        if w == "small_llama":
            continue
        arch = M.MODEL_ARCH.get(w)
        if arch is None:
            continue
        prefix = float(r["prefix_params"])
        repeated = float(r["repeated_params"])
        K = int(r["K"])
        macs_replan = float(r["macs_per_replan"])             # prefix + repeated*K
        wb_res = float(r["weight_bytes_resident"])            # (prefix+repeated)*db
        wb_non = float(r["weight_bytes_nonresident"])         # (prefix+repeated*K)*db
        if arch.family in _VLA_FAMILIES:
            H = arch.action_horizon or 1                      # actions emitted per replan (chunk)
            for regime, hz in _VLA_HZ:
                window_s = H / hz                             # replan must finish before the chunk plays out
                rows.append({
                    "workload": w, "family": arch.family, "regime": regime,
                    "budget_ms": round(window_s * 1e3, 3), "window": f"replan (H={H} actions / {hz:.0f}Hz)",
                    "required_GMAC_per_s": round(macs_replan / window_s / 1e9, 4),
                    "required_weight_GBps_resident": round(wb_res / window_s / 1e9, 4),
                    "required_weight_GBps_reload": round(wb_non / window_s / 1e9, 4),
                    "amortization": f"chunk/H={H}x compute, residency/~{K}x bandwidth",
                    "evidence": "required_from_recovered_structure (HW-independent); regime=design_target",
                })
        elif arch.family in _VLM_FAMILIES:
            for regime, tok in _VLM_TOK:                      # steady-state decode: one step / token
                window_s = 1.0 / tok
                rows.append({
                    "workload": w, "family": arch.family, "regime": regime,
                    "budget_ms": round(window_s * 1e3, 3), "window": f"per token ({tok:.0f} tok/s)",
                    "required_GMAC_per_s": round(repeated / window_s / 1e9, 4),
                    "required_weight_GBps_resident": 0.0,     # weights resident -> ~0 per-token weight traffic
                    "required_weight_GBps_reload": round(repeated * _DB / window_s / 1e9, 4),
                    "amortization": f"residency removes per-token reload (~{K}x); KV grows with seq",
                    "evidence": "required_from_recovered_structure (HW-independent); regime=design_target",
                })
            # TTFT: the prefill (once-prefix) must finish under the budget
            window_s = _TTFT_MS / 1e3
            rows.append({
                "workload": w, "family": arch.family, "regime": "VLM TTFT < 500ms (prefill)",
                "budget_ms": _TTFT_MS, "window": "prefill (once-prefix)",
                "required_GMAC_per_s": round(prefix / window_s / 1e9, 4),
                "required_weight_GBps_resident": round(wb_res / window_s / 1e9, 4),
                "required_weight_GBps_reload": round(wb_res / window_s / 1e9, 4),
                "amortization": "prefill is one-shot (no chunk/residency amortization)",
                "evidence": "required_from_recovered_structure (HW-independent); regime=design_target",
            })
    return rows


def realtime_csv(cs_dir) -> str:
    from merlin.dse_guidance.case_study import _csv
    return _csv(realtime_rows(cs_dir), _RT_COLS)
