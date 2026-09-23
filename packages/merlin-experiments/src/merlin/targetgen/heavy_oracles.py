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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merlin.common.paths import ext_path, target_env_name

from .capsule_runner import OracleUnavailable
from .contract import compile as oot_compile


def simv_env_name(target: str) -> str:
    """The variable that points at ``target``'s prebuilt VCS simv: ``MERLIN_<TARGET>_SIMV``.

    DERIVED from the target name, so each target's existing variable keeps working and a newly registered
    target gets its own with no edit here -- and a target is never handed the simv another one's variable
    names.
    """
    return target_env_name(target, "SIMV")


def vcs_simv(target: str) -> Path | None:
    """``target``'s VCS simv, or None when none is reachable (the caller reports unavailable).

    Resolution: ``MERLIN_<TARGET>_SIMV`` (an explicit binary), then the chipyard VCS build of the design
    the target DECLARES -- its capability manifest's ``runtime.rtl_sim_config`` --
    at ``<chipyard>/sims/vcs/simv-chipyard.harness-<config>``. A target that declares no config and sets
    no variable has no simv; no default SoC config is substituted.
    """
    explicit = os.environ.get(simv_env_name(target))
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    from .runtime_build import rtl_sim_config

    cfg = rtl_sim_config(target)
    if not cfg:
        return None
    try:
        chipyard = ext_path("chipyard")
    except KeyError:  # no chipyard checkout configured: nothing to find, not an error
        return None
    simv = chipyard / "sims" / "vcs" / f"simv-chipyard.harness-{cfg}"
    return simv if simv.is_file() else None


# ------------------------------------------------------------------ L4 VCS
def vcs_available(target: str) -> bool:
    return vcs_simv(target) is not None


def vcs_adapter(target: str) -> Callable:
    """capsule_runner oracle adapter for L4 (VCS RTL). Honest-unavailable when no simv."""

    def run(cb, llvm_text, workdir, timeout):
        import subprocess
        import time

        from merlin.runtime.backends import base as _bk

        simv = vcs_simv(target)
        if simv is None:
            raise OracleUnavailable(
                f"VCS simv not found for {target!r} (set {simv_env_name(target)}, or build the chipyard "
                f"VCS sim of its declared runtime.rtl_sim_config)"
            )
        backend = _bk.get_backend(target)
        _t0 = time.perf_counter()
        elf = oot_compile.compile_lowered_to_elf(cb, llvm_text, workdir, target=target)
        _t1 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(simv), str(elf)], capture_output=True, text=True, timeout=timeout, cwd=str(simv.parent)
            )
        except subprocess.TimeoutExpired as e:
            raise OracleUnavailable(f"VCS simv timed out after {timeout}s") from e
        _t2 = time.perf_counter()
        console = proc.stdout
        outputs, raw = backend.parse_output(console)
        # Same float readback decode the shared oracle path applies: this adapter builds the ELF
        # through `compile_lowered_to_elf` and so gets the same harness, which prints a float
        # destination buffer as its stored bit pattern. Decoding it here keeps this tier's verdict
        # comparable with L2/L3's instead of comparing raw words against a float golden.
        from merlin.runtime.commandbuffer import declared_output_dtypes

        outputs = _bk.decode_float_readback(outputs, declared_output_dtypes(cb))
        # The ELF is independently validated at L2/L3 in the same run; if the available VCS sim
        # crashes or yields no DONE marker on it, that is a VCS/config incompatibility in this
        # environment, not a backend defect -> honest unavailable (never a fabricated pass).
        if proc.returncode != 0 or "DONE" not in console:
            raise OracleUnavailable(
                f"VCS simv ({simv.name}) incompatible with the bare-metal {target} ELF "
                f"(rc={proc.returncode}); ELF is L2/L3-validated. stderr/stdout tail: "
                f"{(proc.stderr or console)[-300:]}"
            )
        return {
            "outputs": outputs,
            "raw_metrics": raw,
            "cycles": raw.get("cycles"),
            "oracle": {"kind": "rtl_vcs", "derived_from_rtl": True},
            "console": console,
            "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3), "oracle_wait_s": 0.0},
        }

    return run


def run_vcs_parallel(
    capsules: list[dict],
    package_dir: str | Path,
    *,
    runs_root: str | Path,
    contract: str | Path | None = None,
    max_workers: int = 4,
    timeout: int = 3600,
    target: str,
) -> list[dict]:
    """Run the corpus through VCS with parallel simv instances (one per capsule).

    ``target`` is required and threaded into each per-capsule run so the grade uses that target's config
    (no silent gemmini default)."""
    from . import capsule_runner as CR

    if not vcs_available(target):
        return [
            {
                "capsule": c["name"],
                "status": "incomplete",
                "failure": {"plane": "vcs", "category": "NOT_RUN_IS_NOT_PASS", "detail": "VCS simv unavailable"},
            }
            for c in capsules
        ]
    pkg = CR.load_package(package_dir, contract=contract)
    CR.integrity_scan(pkg)
    CR.build_package(pkg)
    adapters = {"L4": vcs_adapter(target)}

    def one(cap):
        c = dict(cap)
        c["required_oracle_tiers"] = ["L0", "L1", "L4"]
        return CR.run_capsule(
            c,
            package_dir,
            runs_root=runs_root,
            run_id=f"{cap['name']}_vcs",
            contract=contract,
            oracle_adapters=adapters,
            pkg=pkg,
            timeout=timeout,
            target=target,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, capsules))


# ------------------------------------------------------------------ L5 FireSim
def firesim_root() -> Path | None:
    r = os.environ.get("FIRESIM_ROOT", f"{ext_path('chipyard')}/sims/firesim")
    return Path(r) if Path(r).is_dir() else None


def firesim_queue_alive() -> bool:
    """Best-effort check that the shared FireSim queue daemon is reachable."""
    q = Path(os.environ.get("FIRESIM_QUEUE", f"{ext_path('firesim_queue')}"))
    return q.is_dir() and (q / "daemon.pid").is_file()


def batch_group_id() -> str:
    """The round-group a capsule's exclusive-resource tier accumulates into, or ``UNKNOWN``.

    Read from ``MERLIN_FIRESIM_BATCH_ID``, which the operator sets when a round-group is open.
    ``UNKNOWN`` is returned and SAID rather than a default invented: a window that cannot name the
    batch it belongs to is a cycle count with no denominator, and the reason string has to be able
    to tell "your capsule is queued in group 41" apart from "nothing is collecting these".
    """
    return (os.environ.get("MERLIN_FIRESIM_BATCH_ID") or "").strip() or "UNKNOWN"


def firesim_adapter(target: str, *, batch_id: str | None = None) -> Callable:
    """L5 adapter. Honestly unavailable -- and now says WHAT TO DO about it.

    L5 is a BATCHED tier (the target's own ``target_experiment.yaml`` declares it so): there is one
    FPGA, one queue job runs on it at a time, and scheduling is an operator decision. A per-capsule
    adapter therefore cannot ever succeed, and saying "replay hook not wired" invited exactly the
    wrong follow-up -- somebody wiring a per-capsule replay that the queue policy forbids. The
    refusal names the mechanism that CAN serve this capsule instead.
    """

    def run(cb, llvm_text, workdir, timeout):
        group = batch_id or batch_group_id()
        if firesim_root() is None:
            raise OracleUnavailable("FIRESIM_ROOT not found")
        if not firesim_queue_alive():
            raise OracleUnavailable("FireSim FPGA queue daemon not reachable (shared FPGA busy)")
        # Building the ELF is always safe and is what a batch member contributes, so do it: the
        # artifact is what `run_firesim_bundled` would link into one bootbinary.
        oot_compile.compile_lowered_to_elf(cb, llvm_text, workdir, target=target)
        raise OracleUnavailable(
            f"L5 is a batched tier on {target}; this capsule accumulates into round-group "
            f"{group!r} and is measured as one window of one queue job. A solo FPGA slot is not "
            "available: link the group with merlin.perf.firesim_batch.link_batch and submit it "
            "once (see run_firesim_bundled)"
        )

    return run


def run_firesim_bundled(
    capsules: list[dict],
    package_dir: str | Path,
    *,
    runs_root: str | Path,
    contract: str | Path | None = None,
    timeout: int = 3600,
    batch_id: str | None = None,
    members: list | None = None,
    queue_wall_limit_seconds: float | None = None,
    descriptor: dict | None = None,
) -> dict:
    """PLAN one queued FPGA job carrying every capsule as a measured window. NEVER submits it.

    THE SHAPE, AND WHY IT IS NOT A SESSION.  One queue job runs exactly one ``runworkload``
    (``execution_policy.FIRESIM_LIFECYCLE``, derived from the daemon's own trace), and
    ``FireSimQueuePreflight`` refuses a submission with a nested ``firesim`` command.  So "hold a
    session and replay N ELFs" -- what ``firesim_bundle.sh`` does today -- can never produce a
    sealed receipt.  The admissible bundle is ONE bootbinary with N measured windows, and that is a
    LINKING problem, which :mod:`merlin.perf.firesim_batch` owns.

    THIS FUNCTION DOES NOT RUN ANYTHING.  It returns the linked plan and the argv a submission would
    use, so an operator can schedule it.  Submitting is a person's decision: the queue is shared and
    one job occupies the FPGA at a time.  ``status`` is never ``pass``; every return is
    ``unavailable`` or ``planned``, and ``not_run_is_not_pass`` rides all of them.
    """
    names = [c["name"] for c in capsules]
    if firesim_root() is None:
        return {
            "status": "unavailable",
            "reason": "FIRESIM_ROOT not found",
            "retry": False,
            "not_run_is_not_pass": True,
            "capsules": names,
        }
    if not firesim_queue_alive():
        return {
            "status": "unavailable",
            "reason": "FPGA queue busy/unreachable",
            "retry": True,
            "not_run_is_not_pass": True,
            "capsules": names,
        }
    group = batch_id or batch_group_id()
    if group == "UNKNOWN":
        return {
            "status": "unavailable",
            "reason": "no round-group is open (set MERLIN_FIRESIM_BATCH_ID); an unattributable "
            "window is not a measurement",
            "retry": True,
            "not_run_is_not_pass": True,
            "capsules": names,
        }
    if not members:
        # FAIL CLOSED ON AN UNMEASURED MEMBER. A batch is sized from OBSERVED per-window wall times
        # (`firesim_batch.BatchMember`); without them there is nothing to price the job against and
        # a guess would be a truncated run, not a cheaper one.
        return {
            "status": "unavailable",
            "reason": "no timed batch members were supplied; a batch is sized from observed "
            "per-window wall time, never an estimate",
            "retry": True,
            "not_run_is_not_pass": True,
            "capsules": names,
            "batch_id": group,
        }
    if queue_wall_limit_seconds is None:
        return {
            "status": "unavailable",
            "reason": "the queue wall limit for this tier is UNKNOWN; declare it on the target's "
            "target_experiment.yaml (oracle_resources.<tier>.queue_wall_limit_s)",
            "retry": False,
            "not_run_is_not_pass": True,
            "capsules": names,
            "batch_id": group,
        }
    from merlin.perf import firesim_batch as FB

    try:
        linked = FB.link_batch(
            members, batch_id=group, queue_wall_limit_seconds=queue_wall_limit_seconds, descriptor=descriptor
        )
    except FB.BatchError as exc:
        return {
            "status": "unavailable",
            "reason": f"batch refused: {exc}",
            "retry": False,
            "not_run_is_not_pass": True,
            "capsules": names,
            "batch_id": group,
        }
    return {
        "status": "planned",
        "reason": "the batch links; submitting it is an operator decision because the queue is "
        "shared and one job occupies the FPGA at a time",
        "submitted": False,
        "retry": False,
        "not_run_is_not_pass": True,
        "capsules": names,
        "batch": linked.to_dict(),
    }
