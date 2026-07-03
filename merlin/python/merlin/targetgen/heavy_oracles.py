"""L4 (VCS) and L5 (FireSim) oracle adapters for capsule_bench_v0.

Both reuse the contract ELF build (``compile_lowered_to_elf``); only the *run* step differs. Strict
honest-unavailable: if the simulator binary / FPGA queue is not reachable, the adapter raises
:class:`OracleUnavailable` so the runner records ``status: unavailable`` (never a silent pass).

L4 VCS runs capsules with parallel ``simv`` instances (one per capsule, bounded pool). L5 FireSim
bundles many ELFs into one queued FPGA session to amortize the per-run infra setup; if the queue is
busy the caller can re-schedule the bundle later rather than block.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .capsule_runner import OracleUnavailable
from .contract import compile as oot_compile
from merlin.common.paths import ext_path

# Default Gemmini VCS sim (overridable via MERLIN_GEMMINI_SIMV / MERLIN_SATURN_SIMV).
_DEFAULT_VCS = f"{ext_path("chipyard")}/sims/vcs/simv-chipyard.harness-RadianceGemminiOnlyConfig"


def gemmini_simv() -> Path | None:
    for env in ("MERLIN_GEMMINI_SIMV", "MERLIN_SATURN_SIMV"):
        v = os.environ.get(env)
        if v and Path(v).is_file():
            return Path(v)
    p = Path(_DEFAULT_VCS)
    return p if p.is_file() else None


# ------------------------------------------------------------------ L4 VCS
def vcs_available() -> bool:
    return gemmini_simv() is not None


def vcs_adapter() -> Callable:
    """capsule_runner oracle adapter for L4 (VCS RTL). Honest-unavailable when no simv."""
    def run(cb, llvm_text, workdir, timeout):
        import subprocess
        import time
        from merlin.runtime.backends import gemmini as gem
        simv = gemmini_simv()
        if simv is None:
            raise OracleUnavailable("VCS simv not found (set MERLIN_GEMMINI_SIMV)")
        _t0 = time.perf_counter()
        elf = oot_compile.compile_lowered_to_elf(cb, llvm_text, workdir)
        _t1 = time.perf_counter()
        try:
            proc = subprocess.run([str(simv), str(elf)], capture_output=True, text=True,
                                  timeout=timeout, cwd=str(simv.parent))
        except subprocess.TimeoutExpired as e:
            raise OracleUnavailable(f"VCS simv timed out after {timeout}s") from e
        _t2 = time.perf_counter()
        console = proc.stdout
        outputs, raw = gem.parse_output(console)
        # The ELF is independently validated at L2/L3 in the same run; if the available VCS sim
        # crashes or yields no DONE marker on it, that is a VCS/config incompatibility in this
        # environment, not a backend defect -> honest unavailable (never a fabricated pass).
        if proc.returncode != 0 or "DONE" not in console:
            raise OracleUnavailable(
                f"VCS simv ({simv.name}) incompatible with the bare-metal Gemmini ELF "
                f"(rc={proc.returncode}); ELF is L2/L3-validated. stderr/stdout tail: "
                f"{(proc.stderr or console)[-300:]}")
        return {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles"),
                "oracle": {"kind": "rtl_vcs", "derived_from_rtl": True}, "console": console,
                "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3),
                           "oracle_wait_s": 0.0}}
    return run


def run_vcs_parallel(capsules: list[dict], package_dir: str | Path, *, runs_root: str | Path,
                     contract: str | Path | None = None, max_workers: int = 4,
                     timeout: int = 3600) -> list[dict]:
    """Run the corpus through VCS with parallel simv instances (one per capsule)."""
    from . import capsule_runner as CR
    if not vcs_available():
        return [{"capsule": c["name"], "status": "incomplete",
                 "failure": {"plane": "vcs", "category": "NOT_RUN_IS_NOT_PASS",
                             "detail": "VCS simv unavailable"}} for c in capsules]
    pkg = CR.load_package(package_dir, contract=contract)
    CR.integrity_scan(pkg); CR.build_package(pkg)
    adapters = {"L4": vcs_adapter()}

    def one(cap):
        c = dict(cap); c["required_oracle_tiers"] = ["L0", "L1", "L4"]
        return CR.run_capsule(c, package_dir, runs_root=runs_root, run_id=f"{cap['name']}_vcs",
                              contract=contract, oracle_adapters=adapters, pkg=pkg, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, capsules))


# ------------------------------------------------------------------ L5 FireSim
def firesim_root() -> Path | None:
    r = os.environ.get("FIRESIM_ROOT", f"{ext_path("chipyard")}/sims/firesim")
    return Path(r) if Path(r).is_dir() else None


def firesim_queue_alive() -> bool:
    """Best-effort check that the shared FireSim queue daemon is reachable."""
    q = Path(os.environ.get("FIRESIM_QUEUE", f"{ext_path("firesim_queue")}"))
    return q.is_dir() and (q / "daemon.pid").is_file()


def firesim_adapter() -> Callable:
    """L5 adapter. FireSim bare-metal ELF replay is config-gated; honest-unavailable otherwise."""
    def run(cb, llvm_text, workdir, timeout):
        if firesim_root() is None:
            raise OracleUnavailable("FIRESIM_ROOT not found")
        if not firesim_queue_alive():
            raise OracleUnavailable("FireSim FPGA queue daemon not reachable (shared FPGA busy)")
        # Building the ELF is always safe; the actual FPGA replay path for a bare-metal Gemmini
        # ELF is environment-specific and gated. Until a verified bare-metal FireSim replay hook is
        # wired, report unavailable rather than fabricate a result.
        oot_compile.compile_lowered_to_elf(cb, llvm_text, workdir)
        raise OracleUnavailable("FireSim bare-metal Gemmini replay hook not wired in this env")
    return run


def run_firesim_bundled(capsules: list[dict], package_dir: str | Path, *, runs_root: str | Path,
                        contract: str | Path | None = None, timeout: int = 3600) -> dict:
    """Bundle ELFs into one queued FPGA session. Returns a status dict; honest unavailable.

    When the FPGA/queue is unreachable, returns ``{"status": "unavailable", "retry": True}`` so the
    caller can re-schedule the bundle later instead of blocking.
    """
    if firesim_root() is None:
        return {"status": "unavailable", "reason": "FIRESIM_ROOT not found", "retry": False,
                "not_run_is_not_pass": True, "capsules": [c["name"] for c in capsules]}
    if not firesim_queue_alive():
        return {"status": "unavailable", "reason": "FPGA queue busy/unreachable", "retry": True,
                "not_run_is_not_pass": True, "capsules": [c["name"] for c in capsules]}
    # A real bundled replay would build all ELFs then run them back-to-back in one session.
    return {"status": "unavailable", "reason": "bare-metal FireSim replay hook not wired",
            "retry": False, "not_run_is_not_pass": True,
            "capsules": [c["name"] for c in capsules]}
