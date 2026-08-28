"""Muon (SIMT) capsule runner — now a thin SHIM over the shared :mod:`capsule_runner`.

The Muon path used to be a hand fork of the Gemmini runner. It is now a config: the shared
``capsule_runner.run_capsule`` serves every target, and this module only supplies Muon's
:class:`RunnerConfig` (SIMT tier ladder, cyclotron/vcs sims, the ``kernel.cpp`` 4th artifact, no RoCC
trace gate, fp-tolerance output matching, the SIMT perf headline) + the cyclotron/VCS oracle adapters.

Its 4th entrypoint emits a **whole-computation SIMT kernel FUNCTION** (endpoint ``external_backend``); the
runner-owned harness wraps it for the fork-free oracle (a full-program artifact still compiles directly).
The RoCC trace gate is absent (``trace_gate=None``) because a SIMT target has no command-ISA analog. The
public ``run_capsule``/``run_suite``/``main`` API is preserved for callers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

# re-exported for callers that used MR.discover_capsules / load_capsule / TierResult
from merlin.targetgen.capsule_common import discover_capsules, load_capsule  # noqa: F401
from merlin.targetgen.capsule_runner import TierResult, OracleUnavailable  # noqa: F401
from .muon_oracles import default_adapters, flops_from_cb
from merlin.targetgen.runner_config import RunnerConfig
from .muon import MuonUnavailable, FP_PEAK_GFLOPS  # noqa: F401

SUITE = "muon-perf-bench"
TARGET = "muon"
CONTRACT_VERSION = "0.1"

# Muon's grading knobs — the only per-target data; everything else is the shared runner.
_MUON_CONFIG = RunnerConfig(
    target=TARGET, suite=SUITE, dtype="f32",
    fourth_output_name="kernel.cpp",                       # SIMT C++ kernel (external_backend endpoint)
    tier_sim={"L2": "cyclotron", "L3": "vcs"}, rtl_tiers=frozenset({"L3"}),
    oracle_tiers=("L2", "L3"),
    perf_fields=("flops", "gflops", "pct_fp_peak",
                 "warp_occupancy", "fp_util", "int_util", "sfu_util", "smem_lane_util",
                 "dma_util", "tensor_util", "smem_conflict_rate"),
    trace_gate=None,                                        # no RoCC trace gate on a SIMT target
    force_match_policy={"compare": "float", "atol": 1e-3},  # fp tolerance (device prints fixed decimals)
)


def _ratio(num, den):
    """num/den, or None when the denominator is absent or zero.

    None means UNMEASURED and must stay distinguishable from a real 0.0: a unit that was never
    exercised and a unit the simulator did not report are different findings, and reporting the
    second as 0% invents an efficiency result.
    """
    try:
        n, d = float(num), float(den)
    except (TypeError, ValueError):
        return None
    return None if d <= 0 else n / d


def _utilization(summary: dict | None) -> dict:
    """Per-unit utilization from the SIMT simulator's own counters.

    Latency alone cannot say WHY a kernel is slow. These ratios do: a kernel with high warp occupancy
    but near-zero FP utilization is issuing the wrong work, while low occupancy with high FP
    utilization is starved rather than inefficient. Both are actionable in a way a cycle count is not,
    and both are what an optimization round needs in its feedback.

    Every value is a fraction of the SAME cycle window the simulator reports, so they are comparable
    to each other and across kernels.
    """
    if not isinstance(summary, dict):
        return {}
    tot = summary.get("total")
    if not isinstance(tot, dict):
        return {}
    sched = tot.get("scheduler") or {}
    ex = tot.get("execute_util") or {}
    smem = tot.get("smem_util") or {}
    dma = tot.get("dma_util") or {}
    tensor = tot.get("tensor_util") or {}
    conf = tot.get("smem_conflicts") or {}

    cycles = ex.get("cycles") or sched.get("cycles")
    # Occupancy is per-warp-slot, so the window is cycles x the warps the scheduler could track.
    warp_slots = _ratio(sched.get("active_warps_sum"), sched.get("cycles"))

    return {
        "warp_occupancy": warp_slots,
        "fp_util": _ratio(ex.get("fp_busy_sum"), cycles),
        "int_util": _ratio((ex.get("int_busy_sum") or 0) + (ex.get("int_mul_busy_sum") or 0)
                           + (ex.get("int_div_busy_sum") or 0), cycles),
        "sfu_util": _ratio(ex.get("sfu_busy_sum"), cycles),
        "smem_lane_util": _ratio(smem.get("lane_busy_sum"),
                                 (smem.get("cycles") or 0) * (smem.get("lane_total") or 0)),
        "dma_util": _ratio(dma.get("busy_sum"), dma.get("cycles")),
        "tensor_util": _ratio(tensor.get("busy_sum"), tensor.get("cycles")),
        "smem_conflict_rate": _ratio(conf.get("conflict_lanes"), conf.get("active_lanes")),
    }


def _muon_perf(cb: dict, res: dict) -> dict:
    """Perf headline for a Muon tier: latency AND where the machine's time actually went.

    The adapters already compute gflops/%FP-peak (flops_from_cb serves any that do not). The
    utilization block is derived from the simulator's own counters, so an optimization round can act
    on the bottleneck rather than guessing from a cycle count.
    """
    out = {"gflops": res.get("gflops"), "pct_fp_peak": res.get("pct_fp_peak")}
    out.update(_utilization(res.get("summary")))
    return out


def _wrap_adapters(adapters: dict[str, Callable]) -> dict[str, Callable]:
    """Translate MuonUnavailable -> OracleUnavailable so the shared runner's honest-unavailable path
    (a single exception type) handles the SIMT oracles without a Muon-specific branch."""
    def wrap(adapter: Callable) -> Callable:
        def run(cb, artifact, workdir, timeout):
            try:
                return adapter(cb, artifact, workdir, timeout)
            except MuonUnavailable as e:
                raise OracleUnavailable(str(e)) from e
        return run
    return {tier: wrap(a) for tier, a in adapters.items()}


def run_capsule(capsule: dict, package_dir: str | Path, *, runs_root: str | Path,
                run_id: str | None = None, contract: str | Path | None = None,
                oracle_adapters: dict[str, Callable] | None = None,
                pkg=None, timeout: int = 600, target: str | None = None) -> dict:
    """Run one Muon capsule via the shared runner with the Muon config.

    ``target`` is accepted for signature-parity with the shared bench driver but is advisory: the Muon
    config (``_MUON_CONFIG.target``) is authoritative, so this runner never mis-targets."""
    from merlin.targetgen import capsule_runner as CR
    adapters = _wrap_adapters(oracle_adapters if oracle_adapters is not None else default_adapters())
    return CR.run_capsule(capsule, package_dir, runs_root=runs_root, run_id=run_id, contract=contract,
                          oracle_adapters=adapters, pkg=pkg, timeout=timeout,
                          config=_MUON_CONFIG, perf_extractor=_muon_perf)


def run_suite(capsules: list[dict], package_dir: str | Path, *, runs_root: str | Path,
              contract: str | Path | None = None,
              oracle_adapters: dict[str, Callable] | None = None, timeout: int = 600,
              target: str | None = None) -> list[dict]:
    from merlin.targetgen import capsule_runner as CR
    adapters = _wrap_adapters(oracle_adapters if oracle_adapters is not None else default_adapters())
    return CR.run_suite(capsules, package_dir, runs_root=runs_root, contract=contract,
                        oracle_adapters=adapters, timeout=timeout,
                        config=_MUON_CONFIG, perf_extractor=_muon_perf)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="muon capsule/perf runner (shim over capsule_runner)")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsule", help="path to a single capsule dir")
    ap.add_argument("--capsules-root", help="run every capsule under this root")
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--runs-root", default="out/runs/muon_perf_bench")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)

    if a.capsule:
        caps = [load_capsule(a.capsule, contract=a.contract)]
    else:
        labels = set(a.labels.split(",")) if a.labels else None
        caps = discover_capsules(a.capsules_root, labels=labels, contract=a.contract)
    results = run_suite(caps, a.package, runs_root=a.runs_root, contract=a.contract, timeout=a.timeout)
    npass = sum(1 for r in results if r["status"] == "pass")
    for r in results:
        l2 = r["tiers"].get("L2", {})
        extra = f"  {l2.get('cycles')}cyc  {l2.get('pct_fp_peak')}%peak" if l2.get("cycles") else ""
        print(f"  [{r['status']:10s}] {r['capsule']}{extra}")
    print(f"\n{npass}/{len(results)} pass")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(argv=None))
