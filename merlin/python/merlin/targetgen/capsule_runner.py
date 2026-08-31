"""capsule_bench_v0 orchestrator: run one capsule through a package via the frozen ABI + tiers.

Reuses the oot_runner primitives (``load_package``/``integrity_scan``/``build_package``/
``run_entrypoint``) and the contract compile/oracle path, then layers the capsule_bench tier gates
on top:

    L0  independent numeric golden   (capsule_golden vs reference(cb))   -- catches a wrong cb
    L1  reference(cb) == simulate(cb)                                    -- cb internal consistency
    trace  rocc_decode(lowered.llvm.mlir) + trace_check(expected, cb)    -- instruction coverage
    L2  spike      oracle == golden == reference == simulate
    L3  verilator  oracle == golden == reference == simulate  (cycle-accurate, RTL)
    L4  VCS        (config-gated adapter; see vcs/firesim adapters)
    L5  FireSim    (config-gated adapter)

The integrity backbone: a **mandatory** tier (one listed in the capsule's ``required_oracle_tiers``)
that is unavailable/skipped makes the capsule ``incomplete`` -- never ``pass``
(``not_run_is_not_pass``). This is enforced here in :func:`run_capsule`, not in any adapter.

The package is invoked ONLY through its CLI entrypoints (subprocess). This module is runner code and
MAY import merlin.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import sys
import threading
import time
import traceback as _traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from aet.core.run_paths import RunPaths

from . import tier_policy as _tier_policy

# Most capsules the suite will run SERIALLY to price the tier ladder before fanning out. Small:
# the loop stops as soon as a capsule prices nothing new, and this only bounds the pathological
# case where each early capsule is refuted by a different tier.
_CALIBRATION_CAP = 3

from . import capsule_golden as CG
from .rocc import decode as RD
from . import trace_check as TCK
from .contract import compile as oot_compile
from .contract import schemas
# shared, target-agnostic capsule I/O (also re-exported: callers use CR.discover_capsules/load_capsule)
from .capsule_common import (NOT_MEASURED_STATUSES, _cat, _flat,  # noqa: F401
                             discover_capsules, load_capsule,
                             make_run_paths, run_entrypoints)
from .oot_runner import (BackendDeclined, CertFailure, Package, build_package, integrity_scan,
                         load_package, run_entrypoint)

SUITE = "gemmini-capsule-bench"
CONTRACT_VERSION = "0.1"

# tier -> simulator name understood by the gemmini backend / adapters
_TIER_SIM = {"L2": "spike", "L3": "verilator", "L4": "vcs", "L5": "firesim"}
_RTL_TIERS = {"L3", "L4", "L5"}


@dataclasses.dataclass
class TierResult:
    tier: str
    status: str                       # pass | fail | skipped | unavailable
    mandatory: bool
    reason: str | None = None
    cycles: int | None = None
    derived_from_rtl: bool = False
    cycle_accurate: bool = False
    evidence: str | None = None
    timing: dict | None = None        # {build_s, sim_active_s, oracle_wait_s} — active vs waiting
    gflops: float | None = None       # perf (SIMT); None -> omitted so systolic output is unchanged
    pct_fp_peak: float | None = None
    timing_observations: list | None = None
                                      # FINER TIMING THE ORACLE COULD ACTUALLY SEE: a list of
                                      # {quantity, value, unit, concurrent, note} entries (per-unit
                                      # busy, per-op latency, per-cycle activity). Every functional
                                      # run already executes on a real oracle, and discarding what it
                                      # timed throws away calibration evidence the run already paid
                                      # for. An adapter with NO timing capability emits nothing here
                                      # — not the key, not a list of zeros, because "not reported"
                                      # and "cost nothing" are different facts. None -> omitted, so
                                      # an adapter that carries none is byte-identical to before.
    counters: dict | None = None      # COUNTS the oracle reported alongside the cycle total — bytes
                                      # moved, reuse hits, commits. Deliberately NOT `utilization`:
                                      # those are fractions of a cycle window and these are integers
                                      # of a different kind, and filing a byte count under a name
                                      # that means "fraction of time" is how a number gets read as
                                      # something it is not. A model that computes these and an
                                      # adapter that drops them are indistinguishable downstream,
                                      # which is what happened: the arc oracle has reported movement
                                      # and residency counts all along and the adapter returned only
                                      # cycles. None -> omitted, so a target reporting none is
                                      # byte-identical to before.
    utilization: dict | None = None   # WHERE THE TIME WENT, as fractions of the oracle's own cycle
                                      # window (warp occupancy, per-unit busy, memory conflicts).
                                      # Latency says a kernel is slow; this says why, which is the
                                      # difference between an actionable result and a number. Shape is
                                      # the target's to define; None -> omitted, so a target whose
                                      # oracle reports no counters is byte-identical to before.
    not_applicable: bool = False      # tier honestly N/A for this capsule's datatype (e.g. the integer
                                      # L0/L1 floor on a float datapath) — a legitimate skip, not a
                                      # not_run_is_not_pass violation (unlike an unavailable RTL oracle)
    budget_deferred: bool = False     # tier APPLIED and was DELIBERATELY NOT PAID FOR: this capsule is
                                      # outside the derived covering set and the certify budget is gone.
                                      # Never a pass, never a fail — the capsule was screened, not
                                      # certified, and it says so by name.
    fidelity: str | None = None       # WHAT THE ORACLE ACTUALLY WAS, in the oracle's own words:
                                      # elaborated_rtl | rtl_derived_model | functional_model. The tier
                                      # NAME cannot carry this — one target's L3 is Verilator and
                                      # another's is a model — so a reader of the record could not tell
                                      # a hardware verdict from a model one.

    concurrency: dict | None = None   # WHAT ELSE THE HOST WAS DOING WHILE THIS WAS MEASURED --
                                      # {workers, nproc, load_avg, serial}, stamped at the moment of
                                      # measurement. CYCLES are concurrency-invariant (verified:
                                      # identical serial and at 16 workers); WALL TIMES are not (the
                                      # same query measured 3.7 s serial and 23.4 s at 16 workers, a
                                      # 6.3x spread). So every ``timing`` block is only interpretable
                                      # next to the concurrency it was taken at, and a block without
                                      # one is not comparable with any other. Absent -> the run
                                      # predates the stamp; see merlin.perf.observations.
                                      # concurrency_of, which marks that rather than assuming it.
    submission: dict | None = None    # WHICH SUBMISSION these numbers are about, stated rather than
                                      # inferred from where the file sits. Cycles are a property of
                                      # the submission, not of the capsule: the SAME capsule has
                                      # measured 1090 / 3078 / 8889 across three submissions of one
                                      # task, an 8.2x spread on identical inputs. A path heuristic
                                      # that mis-keys them pools three different programs into one
                                      # "latency", so the field is explicit and the heuristic becomes
                                      # the fallback it should always have been.
    timing_capability: dict | None = None
                                      # WHAT THE INSTRUMENT COULD AND COULD NOT SEE, beside the
                                      # observations it produced: which units went unread, whether
                                      # its buckets partition the timeline, its alias accounting, and
                                      # everything it refused. An unread signal is UNMEASURED, never
                                      # zero, and this is where that distinction is recorded.

    toolchain: str | None = None      # WHICH PROGRAM was graded, as reported by the adapter. A block-
                                      # scaled MX capsule is graded on the harness's own reference MX
                                      # kernel rather than the submission, so a pass there measures the
                                      # fixture. Recording it keeps a score decomposable instead of
                                      # silently overstating the backend by the size of the MX set.

    def to_dict(self) -> dict:
        d = {"status": self.status, "mandatory": self.mandatory,
             "not_run_is_not_pass": True, "reason": self.reason,
             "cycles": self.cycles, "derived_from_rtl": self.derived_from_rtl,
             "cycle_accurate": self.cycle_accurate, "evidence": self.evidence,
             "timing": self.timing}
        if self.not_applicable:
            d["not_applicable"] = True
        if self.budget_deferred:
            d["budget_deferred"] = True
        if self.fidelity:
            d["fidelity"] = self.fidelity
        # perf fields ride the result ONLY when populated (SIMT) — keeps systolic output byte-identical.
        if self.gflops is not None:
            d["gflops"] = self.gflops
        if self.pct_fp_peak is not None:
            d["pct_fp_peak"] = self.pct_fp_peak
        if self.counters:
            d["counters"] = dict(self.counters)
        if self.utilization:
            d["utilization"] = dict(self.utilization)
        if self.timing_observations:
            d["timing_observations"] = list(self.timing_observations)
        if self.timing_capability:
            d["timing_capability"] = dict(self.timing_capability)
        # Concurrency and submission ride the record ONLY when they were established. An absent
        # field is "unrecorded", which is the honest state for every run taken before the stamp
        # existed; a defaulted one would be a claim.
        if self.concurrency:
            d["concurrency"] = dict(self.concurrency)
        if self.submission:
            d["submission"] = dict(self.submission)
        # unchanged. This is what separates "the submission passed" from "the harness fixture passed":
        # a block-scaled MX capsule is graded on the reference MX kernel, not the submitted backend.
        if self.toolchain:
            d["toolchain"] = self.toolchain
        return d


def concurrency_stamp(workers: int | None = None) -> dict:
    """What else the host was doing at the moment of measurement.

    Taken at measurement time, not at suite start: load average moves, and a stamp that describes the
    machine an hour ago describes nothing. ``workers`` is the fan-out this suite was told to use;
    ``None`` means the caller did not state one (a direct :func:`run_capsule` call), and it stays
    ``None`` rather than being guessed at 1 -- other processes share this host.

    Why this exists at all: cycle counts are concurrency-invariant and wall times are not, by a
    measured factor of 6.3x. Every wall number in a ``timing`` block is uninterpretable without it,
    and 22,845 blocks already on disk have to be read as "concurrency unrecorded" because it did not
    exist when they were written. They are NOT retro-labelled: the concurrency they ran at is gone.
    """
    try:
        load = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):                          # no load average on this platform
        load = None
    return {"workers": None if workers is None else int(workers),
            "nproc": os.cpu_count(),
            "load_avg": load,
            "serial": None if workers is None else bool(int(workers) <= 1),
            "sampled": "at the adapter's return, so the load average covers the measurement itself"}


def submission_identity(package_dir: "str | Path", *, run_id: str | None = None) -> dict:
    """WHICH SUBMISSION a measurement belongs to, stated explicitly.

    The harvest layer has had to infer this from where a file sits (the parent of the nearest
    enclosing ``runs`` directory), and a path heuristic mis-keys as soon as a layout changes -- which
    pools incomparable programs into one series. The package directory is the submission and is known
    here for certain, so it is recorded; the campaign labels (run id, arm, round) are supplied by
    whichever harness is driving, through the environment, and are ``None`` when nothing supplied
    them rather than being reconstructed from a guess.
    """
    pkg = Path(package_dir)
    env = os.environ
    ident = {
        "package": str(pkg.resolve()) if pkg.exists() else str(pkg),
        "run_id": run_id or env.get("MERLIN_SUBMISSION_RUN_ID") or None,
        "arm": env.get("MERLIN_SUBMISSION_ARM") or None,
        "round": env.get("MERLIN_SUBMISSION_ROUND") or None,
    }
    ident["stated"] = sorted(k for k, v in ident.items() if k != "package" and v)
    return ident


# An oracle adapter: (cb, llvm_text, workdir, timeout) -> {outputs, cycles, oracle, console}
# or it raises OracleUnavailable to signal honest unavailability.
class OracleUnavailable(Exception):
    pass


# The failure plane for "the endpoint RAN the submitted artifact and it never reached the ISA's
# halt/terminate instruction inside the budget". It gets a plane of its own because the alternative —
# folding it into ``oracle_unavailable`` — tells an agent its grader is missing when in fact its program
# hangs, and those imply opposite actions ("wait for infra" vs "emit the halt"). Measured cost of the
# collapse: ten rounds of a conformant agent shown "mandatory tier L# did not run (unavailable)" while
# every tier record already carried "program did not halt within N instructions". Target-agnostic — any
# endpoint that executes a program can produce it.
DID_NOT_HALT_PLANE = "program_did_not_halt"


def _clip(msg: str, budget: int) -> str:
    """Clip a long exception to ``budget`` chars KEEPING BOTH ENDS -- the head names the failure, the tail
    usually carries the operand/shape specifics.

    A bare ``msg[-budget:]`` throws the head away, and the head is where the diagnosis lives. Measured:
    the muon operand-binding error is one long sentence whose useful half is first ("could not derive
    harness operands: <which of three cases>") and whose second half is fixed advisory prose. Tail-only
    clipping rendered five radiance capsules as `cyclotron crash: es (outputs: ['Y0'])` -- the real cause
    truncated to the last two letters of "shapes" -- and they read as unexplained infra failures across
    many runs while the message that would have named them was being discarded every time.
    """
    msg = str(msg)
    if len(msg) <= budget:
        return msg
    head = budget * 2 // 3                                   # bias to the head: that is the diagnosis
    tail = budget - head - 5
    return f"{msg[:head]} […] {msg[-tail:]}" if tail > 0 else msg[:budget]


def _did_not_halt_reason(msg: str) -> str:
    """The tier ``reason`` for a program that ran to the cap without halting."""
    return f"did not halt (ran to the cycle cap): {msg[-240:]}"


def _did_not_halt_failure(msg: str) -> "CertFailure":
    """The agent-facing failure for a program that ran to the cap without halting — one construction
    shared by every endpoint, so the verdict reads identically however the endpoint reported it."""
    return CertFailure(DID_NOT_HALT_PLANE, _cat("TIMEOUT"),
                       f"{msg}; the emitted kernel never reached the ISA's halt/terminate "
                       "instruction — emit it as the final instruction on every control path (see the "
                       "program-termination contract). Numerics are never checked until the program "
                       "halts.")


def _spike_verilator_adapter(sim: str, target: str) -> Callable:
    def run(cb, llvm_text, workdir, timeout):
        from ..runtime.backends import base as _backends
        backend = _backends.get_backend(target)
        if not backend.available(sim):
            raise OracleUnavailable(f"{sim} not available")
        return oot_compile.run_on_oracle(cb, llvm_text, simulator=sim, target=target,
                                         workdir=workdir, timeout=timeout)
    return run


def _cb_with_leaf_values(cb: dict) -> dict:
    """A copy of ``cb`` whose leaf tensors carry their operand VALUES inline.

    The command-buffer schema declares a leaf by shape/dtype/role only — values are materialized from
    the declaration, which is how the reference, the simulator and the device harness all end up
    computing on the same stimulus. The mlc arc backend instead indexes ``tensors[name]["data"]``, so
    handing it the raw cb raised ``KeyError: 'data'`` and every capsule failed its RTL tier as a
    ``tool_crash`` (27/28 in both arms of the first saturn_opu agent runs, an infra verdict wearing an
    agent verdict's clothes — the preflight said "mlc arc oracle available", which was true of the
    module and false of the grade).

    Materializing here through the SAME ``materialize_inputs`` the numeric floor uses is the load-bearing
    part: values invented separately for the RTL tier would make it disagree with L0/L1 on stimulus and
    report a mismatch that is really a stimulus difference.
    """
    import copy

    from merlin.runtime.commandbuffer import materialize_inputs

    out = copy.deepcopy(cb)
    leaves = materialize_inputs(cb)
    tensors = out.setdefault("tensors", {})
    for name, spec in tensors.items():
        if name in leaves and "data" not in spec:
            spec["data"] = list(leaves[name].data)
    # A resident handle (RES_PACK dst) is a LAYOUT alias of its source, not a new value — which is
    # exactly how the reference interpreter reads it. The arc backend resolves a matmul's operand names
    # straight against the tensor table, so without the alias entry it raised KeyError on the handle.
    # Only RES_PACK is aliased; any other producer stays absent so the backend fails loudly rather than
    # grading against a value this translation invented.
    for cmd in out.get("commands") or []:
        if cmd.get("opcode") != "RES_PACK":
            continue
        ops = cmd.get("operands") or {}
        src, dst = ops.get("src"), ops.get("dst")
        if src in tensors and dst and dst not in tensors:
            tensors[dst] = {**tensors[src], "resident_of": src}
    return out



def suppressed_tier_result(tier: str, mandatory: bool, failed_tier: str, *, from_rtl: bool = False):
    """The TierResult for a tier the ladder did NOT run because a mandatory tier already failed.

    ``skipped``, never ``fail``. ``not_run_is_not_pass`` treats a recorded ``fail`` as evidence the capsule
    was certified at that tier and found wrong; a tier that never executed has no such evidence, and
    fabricating one would put a cycle-accurate verdict on a capsule no RTL ever saw.
    """
    return TierResult(tier, "skipped", mandatory,
                      reason=(f"not run: mandatory tier {failed_tier} already failed, so this deeper tier "
                              f"could not change the verdict and was not worth its cost"),
                      derived_from_rtl=from_rtl)

def _epilogue_stages_ignored(prepared: dict, res: dict, target: str, mlc_bridge) -> set[str]:
    """Which declared commit-epilogue stages this arc model demonstrably IGNORES (empty set if none).

    Deduced, not declared: run the same buffer again with the epilogue stripped and compare. Identical
    outputs while a stage was declared means the model answered with the pre-epilogue accumulator. Found
    on the first agent runs of a command-buffer OPU target, where the model applies the matmul but not the
    commit stage, so seven epilogue capsules scored numerically "wrong" against a reference that was
    right — a model gap presented as an agent failure. The caller turns this into an UNAVAILABLE tier
    (unknown, not failed); a tier that quietly failed instead would teach the agent to break its epilogue.

    Probing per buffer rather than per target keeps it geometry-free: the shapes are the ones already
    being graded, and only a buffer that actually declares a stage pays the second run.
    """
    import copy

    stages: set[str] = set()
    for cmd in prepared.get("commands") or []:
        if cmd.get("opcode") != "COMMIT":
            continue
        attrs = cmd.get("attributes") or {}
        stages.update(str(x) for x in (attrs.get("epilogue") or []))
    if not stages:
        return set()

    stripped = copy.deepcopy(prepared)
    for cmd in stripped.get("commands") or []:
        if cmd.get("opcode") == "COMMIT":
            (cmd.setdefault("attributes", {}))["epilogue"] = []
    try:
        plain = mlc_bridge.arc_run_command_buffer(stripped, target)
    except Exception:  # noqa: BLE001 — the probe must never turn a working tier into a failure
        return set()
    if (res.get("outputs") or {}) == (plain.get("outputs") or {}):
        return stages
    return set()


def mlc_arc_adapter(target: str) -> Callable:
    """The DEFAULT cross-target RTL oracle: run the command buffer on ``target``'s mlc ARC model (the
    RTL-derived functional model — bit-exact datapath outputs + cycle count from the arc state), for ANY
    target mlc compiled from RTL — NO bespoke sim toolchain needed. This is what makes grading generalize
    across targets. Raises OracleUnavailable if mlc / the arc model is absent for the target."""
    def run(cb, llvm_text, workdir, timeout):
        from .rtl import mlc_bridge
        if not mlc_bridge.arc_available(target):
            raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")
        prepared = _cb_with_leaf_values(cb)
        res = mlc_bridge.arc_run_command_buffer(prepared, target)
        ignored = _epilogue_stages_ignored(prepared, res, target, mlc_bridge)
        if ignored:
            raise OracleUnavailable(
                f"the arc model for {target!r} does not model the commit epilogue "
                f"{sorted(ignored)}: it returned identical outputs with the epilogue declared and "
                f"stripped, so its answer is the pre-epilogue accumulator and cannot grade this "
                f"capsule (the hardware does apply it — this is a gap in the model, not in the RTL)")
        _metrics = dict(res.get("metrics") or {})
        # The model reports movement and residency counts beside the cycle total; keeping only
        # `cycles` threw away the direct feedback for the levers a scheduler actually has on this
        # kind of target (how many bytes moved, how often an operand stayed resident).
        _counters = {k: v for k, v in _metrics.items() if k != "cycles" and v is not None}
        return {"outputs": res.get("outputs"),
                "cycles": _metrics.get("cycles"),
                "counters": _counters or None,
                "oracle": res.get("oracle"), "console": ""}
    return run


def _endpoint_of(target: str) -> tuple[str | None, str | None]:
    """(endpoint_kind, model_ext) from the target's contract, best-effort (None,None if no contract).
    Lets ``oracle_adapters`` self-route without threading manifest fields through every caller."""
    try:
        from .target_experiment import load_capability_manifest
        m = load_capability_manifest(target)
        model_ext = (m.contract.get("runner") or {}).get("model_ext") \
            or (m.contract.get("toolchain") or {}).get("model")
        return m.endpoint_kind, model_ext
    except Exception:  # noqa: BLE001 — no contract / not resolvable -> fall back to the arc default
        return None, None


def _bespoke_sim_via(target: str) -> str:
    """Recover a target's DECLARED bespoke-sim engine from its contract's ``runner.sim_via`` block — the
    sim engine is a declared target fact, NOT inferred from a hardcoded ``{spike,verilator} -> chipyard``
    map. A target that ships the chipyard spike/verilator tiers declares ``sim_via: chipyard``; an arc-only
    target declares none and reads back as ``""``. Lets :func:`oracle_adapters` self-route when a caller
    omits ``sim_via``. The harness passes the descriptor's authoritative ``toolchain.sim_via`` explicitly;
    when it doesn't (a standalone preflight/validator), we fall back to the contract's ``runner.sim_via``
    and then to the descriptor's ``toolchain.sim_via`` — so a SIMT target's ``cyclotron`` engine resolves
    even without a contract, instead of mis-defaulting to the arc path (which would false-green the
    preflight)."""
    try:
        from .target_experiment import load_capability_manifest
        via = str(((load_capability_manifest(target).contract.get("runner") or {})).get("sim_via") or "")
        if via:
            return via
    except Exception:  # noqa: BLE001 — no contract; fall through to the descriptor
        pass
    try:
        from .target_experiment import load_target_experiment
        from merlin.common.paths import merlin_dir
        p = merlin_dir() / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"
        if p.is_file():
            return str(getattr(load_target_experiment(p), "sim_via", "") or "")
    except Exception:  # noqa: BLE001 — no descriptor -> no bespoke sim (arc-default)
        pass
    return ""


def _sim_engine_adapters(sim_via: str, target: str) -> dict[str, Callable]:
    """The concrete oracle adapters a DECLARED sim ENGINE provides (additive registry, mirroring
    ``sandbox.toolchain.SIM_TOOLCHAINS``): ``chipyard`` elaborates spike (L2) + verilator (L3). An
    unknown/absent engine contributes none (the arc RTL tier still carries the grade). A new bespoke sim
    registers one branch here — the engine name is DERIVED from the target's contract, never assumed."""
    if sim_via == "chipyard":
        return {"L2": _spike_verilator_adapter("spike", target),
                "L3": _spike_verilator_adapter("verilator", target)}
    return {}


@dataclass(frozen=True)
class _SimOracle:
    """A DECLARED bespoke simulator's oracle contribution, keyed by the sim-ENGINE name a target's
    contract/descriptor declares (``sim_via``) — the same additive-registry pattern as
    :func:`_sim_engine_adapters` and ``sandbox.toolchain.SIM_TOOLCHAINS``, so the shared dispatch below
    never branches on a literal engine name.

    ``exclusive`` engines grade the emitted kernel ELF DIRECTLY and REPLACE the arc/program-oracle default
    entirely — a self-hosted SIMT core (cyclotron) must not be graded by the arc command-buffer path,
    which would grade the wrong artifact. Non-exclusive engines are ADDITIVE: their higher-fidelity tiers
    layer on top of the arc default (chipyard: spike L2 / verilator L3 over the arc L3). A new bespoke sim
    registers ONE entry here + ships its adapter module; the dispatch is unchanged."""
    adapters: Callable[[str], dict]                 # target -> {tier: adapter}
    available: Callable[[str], tuple[bool, str]]    # target -> (ok, reason) pre-spend probe
    exclusive: bool                                 # replaces (True) vs augments (False) the arc default
    has_memmap: bool = False                        # exposes an SoC memory map (DRAM base derivable from the build)
    is_compile_based: bool = False                  # lowers the kernel via an oracle-side compile toolchain (smoke-testable)


def _chipyard_available(target: str) -> tuple[bool, str]:
    """chipyard (gemmini/mx-gemmini): the loop-tier spike binary carries GO; the mlc arc model is the
    fallback gold tier when spike is absent. Preserves the prior gemmini availability semantics exactly."""
    from .rtl import mlc_bridge
    arc_ok = mlc_bridge.arc_available(target)
    try:
        from ..runtime.backends import base as _bk
        _gem = _bk.get_backend(target)  # resolve THIS target's backend (chipyard spike availability)
        spike_ok = bool(_gem.available("spike"))
    except Exception:  # noqa: BLE001 — an unimportable backend is honestly unavailable
        spike_ok = False
    if spike_ok:
        return True, f"{target!r}: chipyard spike oracle available (loop tier)"
    if arc_ok:
        return True, f"{target!r}: chipyard sim absent but mlc arc oracle available (fallback)"
    return False, f"{target!r}: neither the chipyard spike sim nor the mlc arc oracle is available"


#: DECLARED bespoke-sim oracle registry, keyed by sim ENGINE (``sim_via``) — the seam that keeps oracle
#: routing target-name-free (a new sim engine registers here; the dispatch below is untouched). The
#: ``cyclotron`` (self-hosted SIMT) oracle is NOT hardcoded here: it is DISCOVERED from the muon reference
#: package's ``plugin.sim_oracle`` (merlin/targets/muon/sim_oracle.py), which calls
#: :func:`register_sim_oracle` at import via :func:`_ensure_sim_oracles_discovered` — the same eviction as
#: the muon runtime backend. Radiance grades on that discovered cyclotron oracle.
_SIM_ORACLES: dict[str, _SimOracle] = {
    "chipyard": _SimOracle(lambda t: _sim_engine_adapters("chipyard", t), _chipyard_available,
                           exclusive=False, has_memmap=True, is_compile_based=True),
}


def sim_oracle_caps(sim_via: str | None):
    """The registered :class:`_SimOracle` for a sim engine (its capability flags), or None. The
    contract-routed way for other layers (e.g. runtime_build) to ask 'does this sim expose a memory
    map / a compile toolchain?' without branching on the engine NAME. Runs plugin discovery first so a
    target-contributed engine is visible."""
    _ensure_sim_oracles_discovered()
    return _SIM_ORACLES.get(sim_via or "")


def register_sim_oracle(sim_via: str, *, adapters: Callable[[str], dict],
                        available: Callable[[str], tuple[bool, str]], exclusive: bool,
                        has_memmap: bool = False, is_compile_based: bool = False) -> None:
    """Register a bespoke-sim oracle under its ``sim_via`` engine name (idempotent) — the public seam a
    NEW simulator uses to plug into oracle routing without editing :func:`oracle_adapters` /
    :func:`oracle_available`. ``exclusive=True`` replaces the arc/program default (a self-hosted SIMT
    core graded on its own kernel ELF); ``exclusive=False`` layers additive tiers on top of the arc
    default (a chipyard-style sim). See :class:`_SimOracle`."""
    _SIM_ORACLES[sim_via] = _SimOracle(adapters=adapters, available=available, exclusive=exclusive,
                                       has_memmap=has_memmap, is_compile_based=is_compile_based)


_sim_oracle_env_seen: str | None = None
_sim_oracle_lock = threading.Lock()


def _ensure_sim_oracles_discovered() -> None:
    """Load any bespoke-sim oracle a target contributes via its contract ``plugin.sim_oracle`` — a module
    that calls :func:`register_sim_oracle` at import — through the SAME OOT/reference plugin discovery the
    runtime backends use. This is what WIRES the registry: a new target adds its oracle as DATA (a plugin
    path in its contract), never a core edit to the ``_SIM_ORACLES`` literal. Re-scans only when
    ``MERLIN_TARGET_PATH`` changes; registration is idempotent, so repeated scans are harmless.

    THREAD-SAFE (double-checked locking): the ``_sim_oracle_env_seen`` marker is published only AFTER the
    discovery loop has registered every plugin oracle, under a lock. A grade fans capsules across worker
    threads that each call this; the old code set the marker BEFORE discovering, so a second thread could
    observe "already scanned" and race past with e.g. cyclotron not yet registered — collapsing an
    exclusive-sim target to the external_backend program-oracle path (the spurious 'no runner.model_ext'
    crash). Under the lock the first caller finishes registering before any other proceeds."""
    global _sim_oracle_env_seen
    import os
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _sim_oracle_env_seen:                 # fast path: already discovered for this env
        return
    with _sim_oracle_lock:
        if key == _sim_oracle_env_seen:             # re-check under lock (another thread may have finished)
            return
        try:
            from ..runtime.backends import base as _bk
            for name, path in _bk._oot_plugin_modules("sim_oracle"):
                _bk._load_oot_backend(name, path, ns="merlin._oot_sim_oracles")
        except Exception:  # noqa: BLE001 — discovery is best-effort; a broken plugin must not break routing
            pass
        _sim_oracle_env_seen = key                  # publish ONLY after every plugin oracle is registered


def oracle_adapters(target: str, sim_via: str | None = None) -> dict[str, Callable]:
    """The oracle adapters per tier for a target. The mlc ARC model is the DEFAULT RTL tier (works for
    ANY mlc target, no bespoke sim); a target that DECLARES a bespoke sim (``sim_via``) additionally gets
    its higher-fidelity sim tiers (chipyard -> spike L2 / verilator L3), preserving the gemmini path.

    A self-hosted-ISA target (``endpoint_kind == external_backend``, e.g. atlas) is graded by the generic
    PROGRAM oracle (assemble the emitted `.word`/`.insn` kernel with STOCK LLVM -> its mlc cosim) instead
    of the command_buffer arc path — routed from the contract, no target-name branch.

    ``sim_via=None`` (unspecified) is self-resolved from the target's contract via :func:`_bespoke_sim_via`
    so a bare ``oracle_adapters(target)`` is fully contract-routed — never a silent gemmini default. An
    explicit ``""`` (arc-only, e.g. atlas) is honored as-is and NOT re-resolved.

    Routing order is DELIBERATE: a declared EXCLUSIVE bespoke sim (a self-hosted SIMT core, ``sim_via=
    cyclotron``) takes precedence over the ``external_backend`` program-oracle default. A SIMT core's
    endpoint is ALSO ``external_backend`` (it emits a kernel, not ``.insn``), but its kernel ELF must be
    graded by its own sim — the arc-cosim program oracle grades the wrong artifact for it. So the sim
    engine is resolved first; only when no exclusive sim is declared does the endpoint select the oracle."""
    _ensure_sim_oracles_discovered()                                # wire any target-contributed plugin.sim_oracle
    if sim_via is None:                                              # unspecified -> derive from contract
        sim_via = _bespoke_sim_via(target)
    so = _SIM_ORACLES.get(sim_via)
    if so is not None and so.exclusive:                             # self-hosted SIMT: replaces arc/program default
        return so.adapters(target)
    endpoint_kind, model_ext = _endpoint_of(target)
    if endpoint_kind == "external_backend":
        # Self-hosted-ISA program oracle. ``model_ext`` (the model project that lays out operands + owns
        # the fp8/bf16 dtypes) is REQUIRED from the contract — no ``npu_model`` literal fallback; a target
        # that declares none cannot be program-graded, so we fail closed with an actionable message rather
        # than silently defaulting to one target's model. The DERIVED LADDER here is exactly what mlc
        # provides for an assembled program: a FAST FUNCTIONAL tier (L2 — the model's high-level core, pure
        # Python, no arc .so; the per-round loop tier) and the cycle-exact cosim program-runner (L3 — the
        # gold checkpoint), both resolved by target via mlc inside program_oracle. Each tier guards with
        # its own try/except (OracleUnavailable) so an absent runner reads back as ``unavailable``, never
        # fabricated; the REQUIRED/gold tier stays the cosim (L3) — the functional tier is ADDITIVE.
        if not model_ext:
            raise ValueError(
                f"external_backend target {target!r}: contract declares no runner.model_ext — the program "
                f"oracle needs the model project to lay out operands; set runner.model_ext in the contract")
        from .program_oracle import (program_functional_adapter, program_oracle_adapter,
                                      program_verilator_adapter)
        adapters = {"L2": program_functional_adapter(target, model_ext=model_ext),
                    "L3": program_oracle_adapter(target, model_ext=model_ext)}
        # L4 (ADDITIVE, env-routed): a program-driven Verilator sim of the target's RTL top — the first
        # truly RTL-CERTIFIED tier (arc cosim L3 is the RTL-derived functional gold; this runs the
        # elaborated Verilog). Present only if the target registers a vsim (MERLIN_EXT_<TARGET>_VSIM);
        # otherwise None -> no L4. The REQUIRED/gold tier stays L3 — L4 is additive certification.
        vl = program_verilator_adapter(target, model_ext=model_ext)
        if vl is not None:
            adapters["L4"] = vl
        return adapters
    adapters: dict[str, Callable] = {"L3": mlc_arc_adapter(target)}   # arc default (RTL-derived)
    if so is not None:                                               # optional ADDITIVE bespoke sim (chipyard)
        adapters.update(so.adapters(target))
    return adapters


def oracle_available(target: str, sim_via: str | None = None) -> tuple[bool, str]:
    """Probe whether the target's REQUIRED grading oracle can actually run RIGHT NOW — BEFORE an agent
    run spends anything. Target-agnostic, routed from the contract exactly like :func:`oracle_adapters`:

      * ``external_backend`` (self-hosted-ISA program oracle, e.g. atlas) -> the mlc arc cosim AND the
        model venv must BOTH be present (the emitted program is assembled/laid-out in the model venv and
        run on the arc cosim);
      * ``command_buffer`` / arc-default target -> the mlc arc model must be present;
      * a DECLARED bespoke sim (``sim_via == "chipyard"``, e.g. gemmini) -> its fastest loop-tier sim
        (spike) binary must be available (else the arc tier, if present, still carries the grade).

    Returns ``(ok, reason)``. ``ok is False`` means grading would only ever emit ``oracle_unavailable``:
    a real (gradeable) run must NOT be launched on it — the sanctioned way to proceed without an oracle
    is an EXPLICIT structure-only smoke (``--no-oracle``). This is the preflight the launcher + the
    GO/NO_GO validator share so a run that cannot be graded aborts before spending tokens.

    Routing order mirrors :func:`oracle_adapters` exactly: a declared EXCLUSIVE bespoke sim
    (``sim_via == "cyclotron"``, a self-hosted SIMT core) is probed FIRST and takes precedence over the
    ``external_backend`` program-oracle default (a SIMT core's endpoint is also external_backend, but the
    arc command-buffer path grades the wrong artifact for it, so arc_available must NOT false-green it)."""
    from .rtl import mlc_bridge
    _ensure_sim_oracles_discovered()                  # wire any target-contributed plugin.sim_oracle
    if sim_via is None:
        sim_via = _bespoke_sim_via(target)
    so = _SIM_ORACLES.get(sim_via)
    if so is not None and so.exclusive:               # self-hosted SIMT: its own sim carries the grade
        return so.available(target)
    endpoint_kind, model_ext = _endpoint_of(target)
    if endpoint_kind == "external_backend":
        if not mlc_bridge.arc_available(target):
            return False, (f"external_backend target {target!r}: mlc arc cosim unavailable "
                           f"(set MERLIN_MLC_DIR and build the arc model)")
        if not model_ext:
            return False, (f"external_backend target {target!r}: contract declares no runner.model_ext — "
                           f"cannot resolve the model-venv oracle")
        from .program_oracle import OracleUnavailable as _PU
        from .program_oracle import _model_venv_python
        try:
            _model_venv_python(model_ext)
        except _PU as e:
            return False, f"external_backend target {target!r}: {e}"
        return True, (f"external_backend program oracle available (mlc arc cosim + model venv {model_ext!r})")
    # command_buffer / arc-default target (+ an optional ADDITIVE declared bespoke sim, e.g. chipyard).
    # Dispatch through the sim-oracle REGISTRY rather than naming sims here, so a target contributes its
    # own oracle without editing this function.
    if so is not None:
        return so.available(target)
    if mlc_bridge.arc_available(target):
        # "Present" is not "usable": ANSWER a buffer before promising a grade. A target whose arc model
        # imported fine still failed every capsule as a tool_crash because the adapter's input contract
        # was unmet (leaf values, resident aliases, the target argument) — two full agent runs spent
        # against an oracle that could not grade. The probe closes exactly that gap.
        live_ok, live_why = _arc_answers_a_buffer(target)
        if live_ok:
            return True, f"{target!r}: mlc arc oracle available ({live_why})"
        return False, f"{target!r}: mlc arc model present but cannot grade — {live_why}"
    return False, (f"{target!r}: mlc arc model unavailable (set MERLIN_MLC_DIR and build the arc model)")


def _arc_answers_a_buffer(target: str) -> tuple[bool, str]:
    """Run one minimal command buffer through the arc adapter and check it reproduces the reference.

    The buffer is built from DERIVED geometry (the target's own tile edge) in the same schema an agent
    emits, so the probe exercises the real translation path rather than a special case. A mismatch or a
    raise is reported as "cannot grade" with the reason — never swallowed, since a swallowed probe would
    restore the false green it exists to remove.
    """
    from merlin.runtime.reference import reference_outputs
    from .corpus_spec import _tile_dim
    from .target_experiment import load_capability_manifest
    try:
        contract = load_capability_manifest(target).contract
    except Exception:  # noqa: BLE001 — no manifest: _tile_dim falls back to the RTL facts / mesh default
        contract = {}
    tile = _tile_dim(target, contract)
    cb = {
        "abi_version": "0.1", "target": target,
        "tensors": {
            "probe_w": {"shape": [tile, tile], "dtype": "i8", "role": "weight"},
            "probe_a": {"shape": [tile, tile], "dtype": "i8", "role": "input"},
            "probe_y": {"shape": [tile, tile], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "probe_w", "dst": "probe_w_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "probe_a", "rhs": "probe_w_res", "dst": "probe_acc"}},
            {"opcode": "COMMIT", "operands": {"src": "probe_acc", "dst": "probe_y"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
        ],
    }
    try:
        got = (mlc_arc_adapter(target)(cb, "", None, 120).get("outputs") or {})
    except Exception as e:  # noqa: BLE001 — includes OracleUnavailable: either way it cannot grade
        return False, f"a {tile}x{tile} probe buffer raised {type(e).__name__}: {str(e)[-200:]}"
    want = reference_outputs(cb)
    if not got:
        return False, f"a {tile}x{tile} probe buffer produced no outputs"
    if any(got.get(k) != v for k, v in want.items()):
        return False, (f"a {tile}x{tile} probe buffer disagreed with the reference — the model answers, "
                       "but not with this buffer's result")
    return True, f"answered a {tile}x{tile} probe buffer matching the reference"


def _full_ladder_enabled() -> bool:
    """Whether a mandatory tier failure completes the ladder before failing the capsule.

    ON by default: a GRADE must be able to say what every declared tier thought, and a tier with no
    record is not evidence. The cost is real -- a failing capsule now pays the tiers ordered after the
    one that refuted it -- so a fast iteration loop can turn it off with MERLIN_FULL_LADDER=0. Never
    turn it off for a run whose numbers will be quoted.
    """
    import os
    return (os.environ.get("MERLIN_FULL_LADDER", "1") or "1").strip().lower() not in ("0", "false", "no")


def codegen_smoke(target: str) -> tuple[bool | None, str]:
    """Pre-spend check that the target's OWN compiler backend can emit a RUNNABLE kernel — the codegen
    analogue of :func:`runtime_build.compiler_smoke` (which validates the *oracle's* compile toolchain).
    ``oracle_available`` proves we can GRADE; this proves we can PRODUCE the artifact being graded, so a
    broken emit path is a NO_GO before a paid run tool-crashes on every capsule.

    Target-agnostic, dispatched purely on derived facts: a target whose ISA model is FIXED-FORMAT (a 64-bit
    re-encoded ISA that needs the transcode path) and whose reference sim is cyclotron drives the FULL
    fork-free build the live run uses — stock clang -> derived transcode -> stock assemble -> fork-free link
    (BSP regenerated from source) -> reference sim -> assert the computed result — so the exact pipeline is
    exercised offline and a live run never debugs it. Any other target returns n/a (the oracle-side
    compiler_smoke covers a compile-based backend)."""
    # TRI-STATE, and the middle value is the point. True = the smoke RAN and passed. False = it RAN and
    # FAILED (a NO_GO; the caller refuses to launch). None = it did NOT run — this target's emit path is
    # not covered, or a dependency is absent. None must never be spelled True: a check that could not run
    # reporting success is a recurring failure class here, and it is how `merlincirct_gemarm4_codex3`
    # recorded `codegen_ok: true` on a submission an independent regrade scored 1/23. The caller gates on
    # `is False` so an n/a still launches, and the artifact records null so nothing downstream can read a
    # skipped check as evidence.
    # PREREQUISITE FIRST, AND IT FAILS CLOSED.
    #
    # Everything below early-returns `True, "n/a (…)"` for a target whose emit path this smoke does not
    # cover. That is right for a smoke that does not APPLY — but the fork-free path's ISA model is not a
    # smoke, it is a prerequisite: if the backend cannot build it, every capsule fails to compile and the
    # run grades nothing.
    #
    # The two are resolved by DIFFERENT paths, which is how this hid. `isa_model_for_target` (consulted
    # below) succeeded and reported "not fixed-format" -> n/a -> codegen_ok: true, while the compile path's
    # `_model_for` reads the mlc-derived encoding fact and returned None. Measured: a radiance run spent
    # 101 minutes flat at 6/39 — the six MX fixtures, which need no oracle at all — with every real capsule
    # `incomplete: no derived ISA encoding fact for target 'radiance'`, because MERLIN_MLC_DIR was unset.
    # codegen_smoke said codegen_ok: true for all 101 of those minutes.
    if _bespoke_sim_via(target) == "cyclotron":
        try:
            from ..runtime.backends import base as _bk0
            _bk0.get_backend("muon").muon._model_for(target)
        except Exception as e:  # noqa: BLE001 — the prerequisite is missing; that is a NO_GO, not an n/a
            return False, (f"the fork-free emit path cannot build its ISA model for {target!r}: "
                           f"{str(e)[-160:]}. Every capsule would fail to compile and the run would "
                           f"grade only capsules needing no oracle. Derive the encoding fact / set "
                           f"MERLIN_MLC_DIR before spending.")

    # A target may own a compile-and-run backend that is wholly different from the fixed-format path
    # below (including a command-buffer compiler exposed through an inline-insn endpoint). Let that
    # backend provide the production smoke instead of declaring the check inapplicable. The hook is
    # deliberately optional: an arc-only target has no emitted native artifact to smoke.
    # A hook that exists but raises/returns a malformed result is a BROKEN smoke and therefore fails
    # closed; otherwise a typo here could turn the exact pre-spend proof into another silent skip.
    try:
        from ..runtime.backends import base as _bk_cb
        backend = _bk_cb.get_backend(target)
    except Exception:  # noqa: BLE001 — no target-owned backend hook; the fixed-format path may still apply
        backend = None
    hook = getattr(backend, "preflight_codegen_smoke", None)
    if hook is not None:
        if not callable(hook):
            return False, (f"backend for {target!r} exposes a non-callable "
                           "preflight_codegen_smoke")
        try:
            result = hook(target=target)
        except Exception as e:  # noqa: BLE001 — the production emit path failed its pre-spend probe
            return False, (f"production codegen smoke raised {type(e).__name__}: "
                           f"{str(e)[-200:]}")
        if (not isinstance(result, tuple) or len(result) != 2
                or not isinstance(result[0], bool) or not isinstance(result[1], str)):
            return False, (f"backend for {target!r} returned a malformed "
                           "preflight_codegen_smoke result")
        return result
    try:
        from .isa_model import isa_model_for_target
        if not isa_model_for_target(target).is_fixed_format():
            return None, "n/a (ISA is not fixed-format — no fork-free re-encode smoke for this emit path)"
    except Exception as e:  # noqa: BLE001 — no derived model -> nothing to smoke here
        return None, f"n/a (no fixed-format ISA model: {str(e)[-120:]})"
    if _bespoke_sim_via(target) != "cyclotron":
        return None, "n/a (fixed-format ISA but no cyclotron reference sim declared for the fork-free smoke)"
    try:
        from ..runtime.backends import base as _bk
        _muon = _bk.get_backend("muon")   # the evicted SIMT reference backend, resolved via discovery
    except Exception as e:  # noqa: BLE001
        return None, f"n/a (fork-free backend unimportable: {str(e)[-120:]})"
    if not _muon.available("cyclotron"):
        # cyclotron absence is already reported by oracle_available; not this gate's job to double-block.
        return None, "n/a (reference sim absent — oracle_available reports this separately)"
    # a self-contained, relocation-free kernel: derives its own inputs, prints via inline MMIO, guarded to
    # one thread so the byte stream is clean. C[i]=A[i]+B[i] with A[i]=i+1, B[i]=10(i+1) -> 11,22,...,88.
    kernel = ("#include <stdint.h>\n"
              "static inline uint32_t hid(void){uint32_t r;__asm__ volatile(\"csrr %0,0xF14\":\"=r\"(r));return r;}\n"
              "static inline void pc(char c){*(volatile char*)0xFF080000u=c;}\n"
              "static inline void ph(uint32_t v){for(int i=7;i>=0;--i){uint32_t n=(v>>(i*4))&0xF;"
              "pc(n<10?(char)('0'+n):(char)('a'+n-10));}}\n"
              "int main(void){volatile uint32_t A[8],B[8],C[8];"
              "for(int i=0;i<8;i++){A[i]=(uint32_t)(i+1);B[i]=(uint32_t)(10*(i+1));}"
              "for(int i=0;i<8;i++)C[i]=A[i]+B[i];"
              "if(hid()==0){for(int i=0;i<8;i++){ph(C[i]);pc('\\n');}}return 0;}\n")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        try:
            elf = _muon.compile_kernel_forkfree(kernel, td, target=target)   # BSP regenerated from source
            console, _cyc, _ = _muon.run_elf(str(elf), simulator="cyclotron", timeout=180)
        except Exception as e:  # noqa: BLE001 — a broken emit path is exactly what this gate must catch
            return False, f"fork-free codegen smoke failed: {type(e).__name__}: {str(e)[-200:]}"
    want = ["0000000b", "00000016", "00000021", "0000002c",
            "00000037", "00000042", "0000004d", "00000058"]
    missing = [w for w in want if w not in console]
    if missing:
        return False, (f"fork-free kernel ran but produced the wrong result (missing {missing}); "
                       f"console tail: {console[-200:]!r}")
    return True, f"fork-free codegen emits a runnable kernel with the correct result on the {target!r} reference sim"


def _resolve_oracle_adapters(target: str) -> dict[str, Callable]:
    """Contract-routed adapter set for ``target`` — the ``run_capsule`` default when a caller passes no
    adapters. Calls the module :func:`oracle_adapters` (self-resolving sim_via) so an unrouted grade goes
    to the target's OWN endpoint oracle (external_backend->program_oracle, chipyard->spike/verilator,
    else arc), NEVER the gemmini-hardcoded :func:`default_adapters`. (Named to dodge the ``run_capsule``
    parameter that shadows ``oracle_adapters`` in that function's local scope.)"""
    return oracle_adapters(target)


def default_adapters() -> dict[str, Callable]:
    """Back-compat gemmini-only default (L2/L3 spike/verilator). DO NOT use as an unrouted fallback — a
    non-gemmini target would be mis-graded. New callers use :func:`oracle_adapters` (self-routing) or
    :func:`_resolve_oracle_adapters`. Retained only for the explicitly-gemmini perf-bench script."""
    # target-ok: this function IS the explicitly-single-target back-compat entry point (see the
    # docstring); the name is its subject, not an assumption leaking into a shared path.
    return {"L2": _spike_verilator_adapter("spike", "gemmini"),
            "L3": _spike_verilator_adapter("verilator", "gemmini")}


def qa_loop_adapters(target: str, sim_via: str | None = None, *,
                     declared_tiers: set[str] | None = None) -> dict[str, Callable]:
    """The FAST per-round QA-loop oracle set for ``target`` — resolved from :func:`oracle_adapters`, never
    hardwired. It keeps ONE tier (the fastest) and reserves the slower cycle-accurate tiers for the bounded
    checkpoint (:func:`qa_checkpoint_adapters`). This is a tier-order distinction, not a per-target one; no
    target-name branch — a new accelerator's loop gate falls out of its declared ``sim_via`` with no edit.

    ``declared_tiers`` is the set of tiers the CORPUS BEING GRADED lists in ``required_oracle_tiers``.
    When given, the loop tier is the fastest endpoint tier that the corpus actually declares — so the
    per-round gate always rides a tier the capsule asked for. This matters where an endpoint exposes an
    ADDITIVE cheaper tier below its declared gold tier: picking the merely-fastest tier there makes the
    loop enforce a gate the capsule never declared (and, when that cheap tier's runner misbehaves, every
    capsule fails on it while the declared tier passes). Returns ``{}`` — fail closed, never a
    substitute — when the endpoint reaches none of the declared tiers; the caller reports that.

    Omitting ``declared_tiers`` keeps the legacy "fastest available tier" behavior for callers that have
    no corpus in hand."""
    full = oracle_adapters(target, sim_via)
    if not full:
        return {}
    if declared_tiers:
        cand = sorted(t for t in full if t in declared_tiers)
        if not cand:
            return {}
        return {cand[0]: full[cand[0]]}
    lowest = min(full)                       # tier keys sort lexically (L2 < L3 < L4 …): lowest == fastest
    return {lowest: full[lowest]}


def qa_checkpoint_adapters(target: str, sim_via: str | None = None) -> dict[str, Callable]:
    """The cycle-accurate QA CHECKPOINT oracle set for ``target`` = its full oracle ladder from
    :func:`oracle_adapters` (chipyard: spike L2 + verilator L3; arc/mlc: the RTL-derived arc model). This
    is the higher-fidelity barrier run once the fast loop (:func:`qa_loop_adapters`) has converged — not
    every round. Kept as a named seam so the loop driver never re-hardwires ``{L2: spike, L3: verilator}``."""
    return oracle_adapters(target, sim_via)




def _exact_match(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(_flat(a[k]) == _flat(b[k]) for k in a)


def _match_by_policy(a: dict, b: dict, policy: dict | None) -> bool:
    """Output-equality per the capsule's numeric_policy — exact for integer policies (the systolic
    default), tolerance for float. This replaces the per-runner hardcoded exact-vs-atol fork: an i8
    capsule matches exactly (identical to _exact_match), an fp capsule matches within its declared
    atol/rtol (default atol 1e-3, matching the muon path)."""
    compare = (policy or {}).get("compare", "exact_int")
    if compare in ("exact_int", "exact"):
        return _exact_match(a, b)
    if set(a) != set(b):
        return False
    atol = float((policy or {}).get("atol", 1e-3))
    rtol = float((policy or {}).get("rtol", 0.0))
    for k in a:
        fa, fb = _flat(a[k]), _flat(b[k])
        if len(fa) != len(fb):
            return False
        for x, y in zip(fa, fb):
            if abs(float(x) - float(y)) > atol + rtol * abs(float(y)):
                return False
    return True


def _dtype_has_no_datapath(region, cmap, _el) -> tuple[bool, str]:
    """Is the region's operand dtype absent from EVERY capability this target declares?

    This is the ONLY hard structural fact in an eligibility verdict, and so the only ground on which a
    capsule may be withheld from a suite. If no datapath holds the operand format, no arrangement of the
    program puts it on the device.

    The other two axes of the verdict are NOT hard, and both have cost us a real capability:

    * **family** — families compose. Attention on a systolic array is a contraction plus a transposing
      movement, so "no capability for family X" is not proof the device cannot do X. Measured: a float
      target PASSED two capsules its contract calls ineligible on family grounds.
    * **rank** — rank is the one thing a compiler exists to change. A rank-4 convolution reaches a
      rank-2 mesh through im2col, and this compiler ships that lowering
      (``convolution_im2col_matmul``). Measured: ``RP14_patch_embed_bf16_pt`` was withheld as
      "rank 4 not in contraction legal ranks [2, 3]" while the lowering that turns it into a rank-2
      contraction sat unused in the tree.

    Compares through eligibility's OWN alias-aware check, never string equality: a contract spells the
    format ``int8`` while a capsule region reports ``i8``, and a raw ``!=`` reads a native-dtype capsule
    as having no datapath at all.
    """
    dt = getattr(region, "in_dtype", None)
    all_dtypes = tuple({x for cap in cmap.values() for x in (getattr(cap, "dtypes", ()) or ())})
    if dt is None or not all_dtypes or _el._dtype_ok(dt, all_dtypes):
        return False, ""
    return True, f"operand dtype {dt!r} is in no capability this target declares"


def _split_ineligible(op_caps: list[dict], target: str) -> tuple[list[dict], list[dict]]:
    """Partition op capsules into (to grade, not-gradeable-on-this-target).

    Ineligibility is decided by the target's OWN declared capability, so this adds no new judgement --
    it acts on the verdict :mod:`eligibility` already computes. Fails OPEN in every uncertain case
    (no capability map, an undetermined verdict, any exception): a capsule is only withheld when the
    contract positively says the family/dtype/rank is not supported. Withholding on doubt would let a
    suite shrink itself into a better score, which is the opposite of the point.
    """
    try:
        from . import coverage_report as _cr, eligibility as _el
        cmap = _el.capability_map_for_target(target)
    except Exception:                                     # noqa: BLE001 - never block a grade
        return op_caps, []
    if not cmap:
        return op_caps, []
    keep, withheld = [], []
    for c in op_caps:
        try:
            region = _cr._capsule_region(c)
            v = _el.is_eligible(region, cmap)
        except Exception:                                 # noqa: BLE001
            keep.append(c); continue
        if getattr(v, "undetermined", False) or getattr(v, "eligible", True):
            keep.append(c); continue
        # WITHHOLD ONLY ON A HARD STRUCTURAL FACT: an operand dtype or rank that appears in NO declared
        # capability at all. Then no datapath on the device can hold the operands and no composition can
        # rescue it, whatever family the op belongs to -- which also covers a capsule whose family our
        # taxonomy cannot name (pooling, whole-model), where withholding on the family verdict alone
        # would drop it for OUR ignorance rather than the hardware's limits.
        #
        # A merely UNDECLARED FAMILY is graded. Families compose -- attention on a systolic array is a
        # contraction plus a transposing movement -- so "no capability for family X" is not proof the
        # device cannot do X. Measured: a float target PASSED two capsules its contract calls ineligible
        # on family grounds; withholding those would have hidden a contract under-declaration, which is a
        # finding we want, not noise to suppress.
        # Compare dtypes through eligibility's OWN alias-aware check, never by string equality: the
        # contract spells the format `int8` while a capsule region reports `i8`, so a raw `!=` reads a
        # native-dtype capsule as having no datapath at all and withholds it for a spelling difference.
        # See :func:`_dtype_has_no_datapath` for why the dtype is the only hard fact here, and why
        # family and rank must not withhold. State the fact that ACTUALLY triggered withholding:
        # eligibility's own reason describes its FAMILY verdict, which would send someone to fix the
        # taxonomy when the hardware is the constraint.
        dtype_absent, why = _dtype_has_no_datapath(region, cmap, _el)
        if not dtype_absent:
            keep.append(c); continue
        withheld.append({
            "capsule": c.get("name"), "kind": c.get("kind"), "label": c.get("label"),
            "status": "not_graded", "ineligible": True,
            "failure": {"plane": "capability", "category": "NOT_GRADED",
                        "detail": f"not graded on this target: {why} "
                                  f"(eligibility also reports: {getattr(v, 'reason', '') or 'ineligible'})"},
        })
    return keep, withheld


def _gate_counts(result: dict, capsules: list[dict], target: str) -> bool:
    """Whether a graded op-capsule result belongs in the whole-model gate's denominator.

    True unless the capsule is provably ineligible -- and "provably" means the SAME hard structural fact
    that :func:`_split_ineligible` uses to withhold a capsule from the suite: no declared capability
    holds its operand dtype. Fails OPEN whenever that cannot be decided.

    These two ran different tests and disagreed. This read the raw ``is_eligible`` verdict, so a capsule
    whose FAMILY the taxonomy cannot name was dropped from the gate denominator while the suite graded
    it and counted it in the score. Measured on gemmini: ``C7_attention_qk_i8`` (declared
    ``op: attention_qk``, which the taxonomy maps to the ``attention`` composite) was excluded here,
    lifting the gate fraction from 18/23 = 0.78 to 18/22 = 0.82 and launching a whole-model capstone
    that should have deferred -- a five-and-a-half-hour simulation inside a four-hour round. Meanwhile
    the identical computation spelled ``op: matmul`` (``C5_attention_qk_matmul``) counted normally.
    One test, used by both, or the suite and its gate are scoring different corpora.
    """
    name = result.get("capsule")
    cap = next((c for c in capsules if c.get("name") == name), None)
    if cap is None:
        return True
    try:
        from . import coverage_report as _cr, eligibility as _el
        cmap = _el.capability_map_for_target(target)
        if not cmap:
            return True
        region = _cr._capsule_region(cap)
        verdict = _el.is_eligible(region, cmap)
    except Exception:                                     # noqa: BLE001 - a gate must not crash a grade
        return True
    if getattr(verdict, "undetermined", False) or getattr(verdict, "eligible", True):
        return True
    dtype_absent, _why = _dtype_has_no_datapath(region, cmap, _el)
    return not dtype_absent




def _oracle_kind(oracle_meta) -> str | None:
    """A filesystem-safe name for the engine an adapter reports, or ``None`` when it reports none.

    Adapters return either a rich ``{"kind": ..., "derived_from_rtl": ...}`` dict or a bare provenance
    string. Both carry the engine's identity; the contract's static ``tier_sim`` map does not, because a
    backend may substitute a different RTL-derived engine at runtime. ``None`` means "the oracle did not
    say", and the caller keeps the declared name -- never a guess.
    """
    kind = oracle_meta.get("kind") if isinstance(oracle_meta, dict) else oracle_meta
    if not isinstance(kind, str) or not kind.strip():
        return None
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in kind.strip())
    return safe or None

def lane_report(capsule: dict, routing_plan: dict | None,
                mesh_execution: dict | None = None) -> dict | None:
    """Verify an INTEROP capsule's declared lanes against what the compiler actually DID.

    Returns ``None`` when the capsule declares no ``lanes.require`` (the ordinary case).

    Two things this must not do, both of which it did and both of which produced a hollow pass on the
    first capsule to use it:

    * **A routing plan is not an execution.** ``plan["on_mesh"]`` lists the ops the router ASSIGNED to the
      mesh; whether they ran there is a different fact, recorded in ``mesh_execution``. Measured on the
      same submission: 15 matmuls assigned to the mesh, 15 host fallbacks at run time. So when per-layer
      accounting exists it decides ``on_mesh``, and the plan is only consulted where no accounting does.
    * **Not every key of the plan is a lane.** The plan also carries ``n_mesh_ops``, ``n_scalar_ops``,
      ``mesh_matmul_extents`` and ``note``; scanning it for truthy keys reported those as lanes that
      "carried work". A lane entry is a MAPPING of op -> count, so only dict-valued keys are lanes.
    """
    req = [str(x) for x in ((capsule.get("lanes") or {}).get("require") or [])]
    if not req:
        return None
    plan = routing_plan or {}
    lanes = {k: v for k, v in plan.items() if isinstance(v, dict)}
    carried = {k for k, v in lanes.items() if v}
    me = mesh_execution or {}
    # The mesh lane is the one we have real per-layer accounting for; when it exists, believe it.
    if me:
        on = me.get("matmul_layers_on_mesh")
        if on is not None:
            if int(on) > 0:
                carried.add("on_mesh")
            else:
                carried.discard("on_mesh")
    unexercised = [ln for ln in req if ln not in carried]
    out = {"required": req, "observed": sorted(carried), "unexercised": unexercised,
           "evidence": ("execution" if me.get("matmul_layers_on_mesh") is not None else "routing_plan")}
    if out["evidence"] == "routing_plan":
        out["caveat"] = ("no per-layer execution accounting was recorded, so this reports what the "
                         "router PLANNED, not what ran; a lane can appear here and still have executed "
                         "nothing")
    return out

def _absent_outputs(nrep: dict) -> list[str]:
    """Outputs the kernel DECLARED (present in the interface/golden) but never wrote — or wrote at the
    wrong length — extracted from :func:`capsule_golden.compare`'s ``per_output``. These structural
    failures are the ones that read as ``mismatch_count > 0`` while ``max_abs_error == 0`` and
    ``first_mismatch is None`` (there is no value to diff — the bytes were never produced), a
    self-contradictory signal unless the missing output is named. Reveals no golden VALUE: only the
    identity of a declared output, which the agent already holds in the interface it was handed."""
    po = nrep.get("per_output") or {}
    return [nm for nm, d in po.items()
            if isinstance(d, dict) and d.get("status") == "fail"
            and str(d.get("reason", "")).startswith(("missing", "length"))]


def _unwritten_output_detail(nrep: dict, sim_name: str) -> str | None:
    """A precise detail when a declared output was READ BACK but holds one constant value, else None.

    Sibling of :func:`_absent_output_detail`, for the case that one misses: there the output key is
    absent from the readback; here it is present and uniformly the buffer's untouched fill, so the
    generic "does not compute the declared operation" is technically true and useless. Measured cost of
    not having this: twelve capsules reported ``functional_mismatch`` for six consecutive rounds with
    ``observed: 0.0`` and ``max_rel_error: 1.0``, while the mismatch COUNT was a function of the
    golden's zero distribution rather than of the kernel -- so the number could not move no matter what
    was emitted, and the agent spent six rounds tuning numerics against an unwritable signal.

    Reveals no golden value: only that the observed side is constant, which the agent can compute from
    its own readback."""
    names = nrep.get("outputs_never_written") or []
    if not names:
        return None
    po = nrep.get("per_output") or {}
    got = {n: (po.get(n) or {}).get("observed_constant") for n in names}
    shown = ", ".join(f"{n} (all {got[n]})" for n in names)
    return (f"on {sim_name}, output(s) {shown} came back as a SINGLE CONSTANT value while the expected "
            f"result varies — the store never landed, so what was compared is the buffer's untouched "
            f"fill, not a computed result. This is a WRITEBACK failure, not a numeric one: the mismatch "
            f"count here is set by the reference's own value distribution and will NOT move if you only "
            f"change arithmetic. Decode your emitted artifact and confirm the result is actually stored "
            f"to this output's base address (and that any DMA/commit is awaited before readback).")


def _absent_output_detail(nrep: dict, sim_name: str, expected: dict, observed: dict) -> str | None:
    """A precise failure detail when the kernel dropped a declared output, else None. Turns the baffling
    "1024 mismatches, 0 error" into "you never wrote Y1" — the exact class of silent no-op the generic
    hint tells the agent to hunt for. Target-agnostic; no answer key crosses this boundary."""
    absent = _absent_outputs(nrep)
    if not absent:
        return None
    n_decl = len(expected)
    n_written = sum(1 for k in expected if k in (observed or {}))
    return (f"on {sim_name}, your emitted artifact never wrote output(s) {', '.join(absent)} — "
            f"declared in the interface (a commit op) but ABSENT from the observed DRAM readback "
            f"(your kernel produced {n_written} of {n_decl} declared outputs). This is a DROPPED or "
            f"mis-addressed store, NOT a value error (max_abs_error is 0 because those bytes were never "
            f"produced). Decode your OWN emitted artifact and confirm every declared output gets a store "
            f"to its base address.")


def _default_config(target: str, suite: str, dtype: str):
    """The implicit gemmini/systolic RunnerConfig — the exact constants run_capsule used before it took a
    config, so a call with config=None is byte-identical to the pre-collapse behavior."""
    from .runner_config import RunnerConfig
    return RunnerConfig(target=target, suite=suite, dtype=dtype,
                        fourth_output_name="lowered.llvm.mlir", tier_sim=dict(_TIER_SIM),
                        rtl_tiers=frozenset(_RTL_TIERS), oracle_tiers=("L2", "L3", "L4", "L5"),
                        perf_fields=(), trace_gate="rocc_insn")


def _config_for_target(target: str, suite: str | None, dtype: str):
    """The per-target RunnerConfig, DERIVED from the target's capability manifest (endpoint /
    trace_gate / 4th-output / tiers all come from the contract) — so a non-gemmini target is graded by
    ITS config, never the gemmini default. An atlas (external_backend) run gets trace_gate=None +
    kernel.S, so the RoCC trace gate is correctly skipped. Falls back to the implicit gemmini/systolic
    constants only when no manifest resolves (raw/test targets)."""
    try:
        from .runner_config import runner_config_from_manifest
        from .target_experiment import load_capability_manifest
        return runner_config_from_manifest(load_capability_manifest(target))
    except Exception:  # noqa: BLE001 — no resolvable manifest -> legacy default
        return _default_config(target, suite or f"{target}-capsule-bench", dtype)


def suite_for(target: str, *, dtype: str = "i8xi8_i32") -> str:
    """The suite path segment ``run_capsule`` writes results under for this target (``cfg.suite``).

    Any reader that re-globs the on-disk ``capsule_result.json`` MUST resolve the suite through this,
    NOT the module-level ``SUITE`` literal ('gemmini-capsule-bench'): ``run_capsule`` lays results at
    ``<runs_root>/runs/<cfg.suite>/<capsule>/`` where ``cfg.suite`` is the TARGET's own suite
    (e.g. 'atlas-capsule-bench'). Using the gemmini literal made the atlas self-check glob an empty
    gemmini dir and report ``n_capsules: 0`` on every call — the agent's feedback loop went blind while
    the in-memory grade was correct. Derived from the target's RunnerConfig, so it stays target-agnostic."""
    return _config_for_target(target, None, dtype).suite


def _encoding_divergence_hint(trace_check_res: dict | None, independent_float: bool,
                              cb: dict | None = None, capsule: dict | None = None,
                              trace: dict | None = None) -> str:
    """A mandatory hardware/oracle tier failed while the cheap tiers already passed (numeric fails raise
    earlier), so the defect is in the emitted hardware artifact, NOT the command buffer. Return a
    target-agnostic localization hint so the failure is never an opaque 'oracle != golden' — plus the first
    concrete artifact finding if the decoder produced any (gemmini). For a float/oracle-only target (atlas,
    no cheap trace decode) the generic hint still fires. This is what makes the np12r class localizable on
    EVERY endpoint: it names WHERE to look (the artifact encoding), not just THAT hardware disagreed.

    When the decoded trace + command buffer are available, an ADVISORY, FAIL-CLOSED cross-check
    (:mod:`divergence_localizer`) is run first: it diffs the decoded config fields against the agent's OWN
    command buffer + capsule spec and, if it can derive a concrete divergence, names the FIRST op+field
    (never a golden value, never a fix). The oracle stays the arbiter; a field it cannot anchor is simply
    not flagged, so the generic hint always still fires underneath."""
    concrete = ""
    if trace is not None:
        try:
            from . import divergence_localizer as _DL
            concrete = _DL.format_finding(_DL.localize(cb, capsule, trace))
        except Exception:  # noqa: BLE001 — the localizer is advisory; never let it break the failure path
            concrete = ""
    findings = (trace_check_res or {}).get("violations") or []
    if independent_float:
        hint = (" Decode your OWN emitted artifact (the disassembler / instruction_trace.json) and verify "
                "every op fires with the operands you intend — a config/compute op that silently no-ops "
                "produces wrong output only on hardware, never in a cheap check.")
    else:
        hint = (" The command-buffer tiers (numeric + trace) PASSED, so the divergence is in your "
                "emit_target_artifact hardware encoding — some field the command buffer cannot carry (a "
                "config scale, an accumulate/dataflow bit, a readout dtype, a DRAM address). Decode your OWN "
                "emitted artifact (the disassembler / instruction_trace.json) and check each op's operands "
                "against your intent.")
    if findings:
        hint += f" Your own artifact check also flagged: {findings[0]}"
    # The concrete cross-check (when derivable) is the most actionable line — lead with it.
    return (concrete + hint) if concrete else hint


def _unexercised_note(unexercised: list[str], exercised: dict) -> dict:
    """``{tier: why it did not run}`` for the declared tiers a whole-model grade did not exercise.

    Reported as a bare LIST, this read as "we skipped three of the four tiers this capsule declares",
    which is indistinguishable from a grade that cut corners. The actual reason is structural: a model
    capsule is graded by the whole-model compile+verify path against its golden, plus the mesh oracle --
    it never enters the per-op tier ladder, and the cheap tiers of that ladder (emit/structure,
    command-buffer reference equality, functional sim of a command buffer) have no whole-model analogue
    to run. A tier that is inapplicable and a tier that was skipped deserve different words.
    """
    ran = set(exercised)
    return {t: ("no whole-model analogue: this tier grades a per-op command buffer, which a model "
                "capsule does not produce — the model is graded end to end against its golden")
            for t in unexercised if t not in ran}


def _rtl_tiers_of(target: str | None) -> frozenset[str]:
    """The tiers this target counts as RTL-derived, from its capability manifest. Fails soft to an empty
    set so the caller falls back to the capsule's own declaration rather than guessing a tier name."""
    if not target:
        return frozenset()
    try:
        from .runner_config import runner_config_from_manifest
        from .target_experiment import load_capability_manifest
        return frozenset(runner_config_from_manifest(load_capability_manifest(target)).rtl_tiers)
    except Exception:                                    # noqa: BLE001 — unresolvable manifest
        return frozenset()


def _numeric_when_not_accelerated(st, gate, verify: dict, cos: float, engine: str,
                                  measured_on: str) -> dict:
    """The numeric verdict for a model capsule that will NOT be reported as a pass.

    The whole-model gate runs BEFORE the acceleration checks, so by the time one of those rejects the
    capsule the arithmetic has already been measured -- and was being overwritten with
    ``not_compared``, which reads as "we do not know" when in fact we do. That erased the one thing a
    reader most needs from a failing capstone: whether the compiler got the model RIGHT and merely ran it
    in the wrong place, or got it wrong as well. Those call for completely different work.

    ``measured_on`` is carried explicitly so the number can never be mistaken for an accelerator result;
    the capsule verdict stays ``fail`` in every caller."""
    return {"status": ("pass" if (st == "verified" and gate) else "fail"),
            "engine": engine,
            "gate": verify or None,
            "cos": (cos if cos else None),
            "measured_on": measured_on,
            "note": ("arithmetic only -- this capsule is NOT a pass; see the failure block for what "
                     "disqualified it. A correct number computed on the wrong lane is still not an "
                     "accelerator result."),
            "model_status": st}


def _numeric_not_compared(engine: str, measured_on: str, verify: dict | None,
                          detail: str) -> dict:
    """The numeric block for a verdict where NOTHING WAS COMPARED against the accelerator.

    The single home for this status, deliberately: it is the honest answer when no declared tier ran,
    and the wrong one everywhere a number does exist -- a measured mismatch reported as `not_compared`
    hides a real defect behind an absence. Every path that DID measure reports its verdict through
    :func:`_numeric_when_not_accelerated` instead."""
    return {"status": "not_compared", "engine": engine, "measured_on": measured_on,
            "gate": verify or None, "detail": detail}


def _model_tier_map(declared: list[str], target: str | None, model_exec: dict | None,
                    ) -> "dict[str, TierResult]":
    """The tier block for a WHOLE-MODEL capsule, derived from the model's OWN layer accounting.

    A model capsule used to carry no tier block at all on most paths -- the derivation ran only when
    per-tile verification was present, and every fail-closed branch returned before it. So a capsule
    could report a verdict beside `tiers: {}`, which is how a suite reads N/N while nothing was
    certified. The tests that pin this read `tiers[<rtl tier>]["status"]`, so it must exist on EVERY
    path, including the ones that refuse.

    L0/L1 interpret a COMMAND BUFFER. A whole model has none, so they are honestly not_applicable --
    a legitimate skip, not a tier that failed to run.

    The RTL tier reads the model's own counters, and the four cases are deliberately distinct:

      * layers ran and none fell back  -> pass
      * no layer ran at all            -> skipped   (the target never ran this model)
      * some layer fell back           -> fail      (a hole in the claim, not a rounding error)
      * counters absent                -> unavailable (nobody could tell; absent must never read as 0)
    """
    tiers: dict[str, TierResult] = {}
    for t in declared:
        if t in ("L0", "L1"):
            tiers[t] = TierResult(t, "skipped", True, not_applicable=True,
                                  reason="a whole model has no command buffer to interpret")
    _rtl = [x for x in declared if x in _rtl_tiers_of(target)]
    tier = _rtl[-1] if _rtl else (declared[-1] if declared else None)
    if tier is None or tier in tiers:
        return tiers
    on = (model_exec or {}).get("matmul_layers_on_mesh")
    fb = (model_exec or {}).get("matmul_layers_host_fallback")
    if on is None:
        tiers[tier] = TierResult(tier, "unavailable", True,
                                 reason="no mesh execution counters were reported, so whether this "
                                        "model's layers ran on the accelerator is UNKNOWN")
    elif int(fb or 0) > 0:
        tiers[tier] = TierResult(tier, "fail", True,
                                 reason=f"{int(fb)} matmul layer(s) fell back to the host kernel")
    elif int(on) == 0:
        tiers[tier] = TierResult(tier, "skipped", True,
                                 reason="no matmul layer executed on the accelerator")
    else:
        tiers[tier] = TierResult(tier, "pass", True,
                                 reason=f"{int(on)} matmul layer(s) executed on the accelerator")
    return tiers


def model_budget_seconds() -> float | None:
    """Wall-clock ceiling for ONE whole-model capsule, or ``None`` for unlimited (the default).

    Unlimited is the right default for an operator certification run: a whole model on a
    cycle-accurate oracle legitimately takes hours, and a ceiling that silently truncates it would
    hide coverage rather than buy anything. It is the WRONG default inside an agent loop's per-round
    gate, which is why the loop sets one explicitly.

    The ``timeout`` threaded through this file is a PER-STEP subprocess ceiling. A whole-model grade
    makes many such calls, so their sum is unbounded and a step timeout can never bound the capsule.
    MEASURED (2026-08-29, gemmini arm-4 calibration ``merlincirct_defcal1``): the agent round finished
    in 40 min, the capstone then ran 5 h 30 m past the round's own 4 h timeout, wrote not one byte into
    its run directory, and the round never graded. Its own step timeout was 900 s.
    """
    raw = os.environ.get("MERLIN_MODEL_BUDGET_S", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


#: The self-protection a budgeted child runs BEFORE anything else: ask the kernel to SIGKILL it when
#: its parent goes away. Kept as a string prefix, not a function, because a function would have to be
#: imported -- and importing merlin takes seconds, during which the child is unprotected. A parent
#: killed inside that window orphans it, which is measured, not hypothetical.
_CHILD_GUARD = (
    "import ctypes,os,signal,sys;"
    "ctypes.CDLL('libc.so.6').prctl(1, signal.SIGKILL);"          # PR_SET_PDEATHSIG
    "os.getppid()==1 and os._exit(1);"                            # already orphaned: no signal coming
)

#: Child entry for a budgeted whole-model grade: protect first, then delegate to ``main``.
_CHILD_PREAMBLE = _CHILD_GUARD + (
    "__import__('merlin.targetgen.capsule_runner', fromlist=['main']).main(sys.argv[1:])"
)


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this process when its parent goes away.

    The budgeted whole-model grade runs in its OWN SESSION so the parent can kill the process group
    when the ceiling expires. That protects against a grade overrunning; it does nothing when the
    PARENT is killed, because the parent's cleanup never runs. MEASURED (2026-08-30): stopping a
    regrade left the model-grade child and its Verilator alive on a shared host, reparented to init,
    with nothing left that knew to reap them.

    Set from inside the CHILD rather than through ``preexec_fn``: the parent grades capsules on a
    ThreadPoolExecutor, and ``preexec_fn`` is documented as unsafe in a threaded process. The cost is a
    small race -- the parent could die during interpreter startup -- which the getppid() re-check
    below closes.
    """
    try:
        import ctypes
        import signal as _sig
        ppid = os.getppid()
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, _sig.SIGKILL)
        if os.getppid() != ppid:            # already reparented: the signal will never come
            os._exit(1)
    except Exception:                       # noqa: BLE001 — a best-effort safeguard, never a blocker
        pass


def _descendants(pid: int) -> list[int]:
    """Every live descendant of *pid*, deepest first, read from /proc.

    Must be collected BEFORE any signal is sent: once a process is orphaned its parent link is gone.
    A process GROUP is not enough. MEASURED (2026-08-30): a budgeted whole-model grade's Verilator
    child had called ``setsid`` of its own, so it sat in neither the grade child's group nor its
    session; ``killpg`` reaped the python and left the simulator running with PPID 1, holding a core
    on a shared host.
    """
    kids: dict[int, list[int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            # /proc/<pid>/stat is "pid (comm) state ppid ..."; comm may itself contain spaces and
            # parentheses, so split after the LAST ')' rather than on whitespace from the left.
            ppid = int((entry / "stat").read_text().rpartition(")")[2].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(entry.name))
    out: list[int] = []
    frontier = list(kids.get(pid, ()))
    while frontier:
        out.extend(frontier)
        frontier = [g for p in frontier for g in kids.get(p, ())]
    out.reverse()                                  # deepest generation first
    return out


def _running(pid: int) -> bool:
    """True while *pid* is a live process. A ZOMBIE is not: it has exited and merely awaits a wait(),
    so ``/proc/<pid>`` still exists and a liveness check written that way waits out its whole grace."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _kill_tree(pid: int, *, grace: float = 5.0) -> None:
    """SIGTERM then SIGKILL *pid* and every descendant it had when we looked."""
    import signal as _signal
    victims = _descendants(pid) + [pid]
    for sig in (_signal.SIGTERM, _signal.SIGKILL):
        for victim in victims:
            try:
                os.kill(victim, sig)
            except OSError:                        # already gone, or not ours
                pass
        try:                                       # and the group, for anything spawned since the scan
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(_running(v) for v in victims):
                return
            time.sleep(0.1)


def _grade_model_capsule(capsule: dict, *, target: str | None = None, timeout: int,
                         package_dir: str | Path | None = None,
                         budget_s: float | None = None) -> dict:
    """Grade a whole-model capsule, under a wall-clock budget when one is set.

    ``budget_s`` (default: :func:`model_budget_seconds`) is a ceiling on the CAPSULE, not on a step.
    Exceeding it is reported as ``budget_exhausted`` with the elapsed time: a budget that ran out is
    the absence of evidence, so it is neither a pass nor a fail of the submission and sits in neither
    side of the score (see ``NOT_MEASURED_STATUSES``). With no budget the grade runs inline, exactly
    as before.

    A budgeted grade runs in a CHILD PROCESS in its own session. Two reasons, both load-bearing: an
    in-process call cannot be interrupted (this runs on a ThreadPool worker, so signals are out), and
    the grade spawns oracle subprocesses (verilator, spike) that must die with it -- killing only the
    direct child leaves them competing for the CPU the next capsule needs. A plain subprocess rather
    than ``multiprocessing``: ``spawn`` re-imports the caller's ``__main__`` (the harness entry points
    are scripts, so that re-runs them) and ``fork`` is unsafe from a thread.
    """
    budget = model_budget_seconds() if budget_s is None else budget_s
    if not budget or budget <= 0:
        return _grade_model_capsule_inline(capsule, target=target, timeout=timeout,
                                           package_dir=package_dir)

    import subprocess as _sp
    import tempfile as _tf

    def _stopped(detail: str, status: str = "incomplete", **extra) -> dict:
        return {"capsule": capsule["name"], "kind": "model", "label": capsule.get("label"),
                "contract_version": CONTRACT_VERSION, "status": status, **extra,
                "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS", "detail": detail}}

    with _tf.TemporaryDirectory(prefix="model_grade_") as _td:
        spec_p, out_p = Path(_td) / "spec.json", Path(_td) / "result.json"
        spec_p.write_text(json.dumps({"capsule": capsule, "target": target, "timeout": timeout,
                                      "package_dir": str(package_dir) if package_dir else None}),
                          encoding="utf-8")
        # `-c` rather than `-m`: the preamble sets PR_SET_PDEATHSIG in the first milliseconds, before
        # importing merlin. Importing first leaves a multi-second window in which the child is
        # unprotected, and a parent killed inside that window orphans it -- measured, and the reason
        # this is a preamble and not a call at the top of main().
        cmd = [sys.executable, "-c", _CHILD_PREAMBLE,
               "--model-grade", str(spec_p), "--model-grade-out", str(out_p)]
        started = time.monotonic()
        _child_env = dict(os.environ)
        # A budgeted model grade runs in another interpreter, so the parent's ContextVar cannot carry
        # the capsule attribution into it. The pass recorder already accepts this explicit environment
        # fallback; set it on the child only so concurrent model grades cannot cross-attribute calls.
        if _child_env.get("MERLIN_PASS_LOG"):
            _child_env["MERLIN_PASS_LOG_CAPSULE"] = str(capsule["name"])
            _child_env["MERLIN_PASS_LOG_REQUIREMENTS"] = json.dumps(
                list(capsule.get("pass_requirements") or []))
        proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                         env=_child_env, start_new_session=True)
        try:
            tail = (proc.communicate(timeout=budget)[0] or "")[-2000:]
        except _sp.TimeoutExpired:
            elapsed = time.monotonic() - started
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except _sp.TimeoutExpired:
                pass
            # `budget_exhausted`, not `incomplete`: nothing about the submission was measured, so it
            # belongs in neither bucket. Scoring it as a failure would make all_pass unreachable and
            # cost an agent loop its only early exit -- see NOT_MEASURED_STATUSES.
            return _stopped(
                f"whole-model grade exceeded its {budget:.0f}s budget (ran {elapsed:.0f}s) and was "
                f"stopped. Raise MERLIN_MODEL_BUDGET_S, or clear it for no ceiling, to grade it. "
                f"NOT a verdict on the submission.",
                status="budget_exhausted", model_budget_s=budget, elapsed_s=round(elapsed, 1))
        if not out_p.is_file():
            return _stopped(f"whole-model grade process exited {proc.returncode} without a "
                            f"result: {tail[-400:]}")
        try:
            return json.loads(out_p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _stopped(f"whole-model grade result unreadable: {type(exc).__name__}: {exc}")


def _resolve_model_host_lane(target: str, dtype: str):
    """Return ``(experiment, package_path, identity)`` for targeted whole-model compilation.

    Kept as a small seam because descriptor resolution, package-content validation and datatype
    compatibility are one fail-closed decision; callers must not independently recreate part of it.
    """
    from .corpora import descriptor_path
    from .target_experiment import load_target_experiment

    descriptor = descriptor_path(target)
    if not descriptor.is_file():
        raise ValueError(f"target {target!r} has no experiment descriptor at {descriptor}")
    experiment = load_target_experiment(descriptor)
    if experiment.target != target:
        raise ValueError(
            f"descriptor {descriptor} names target {experiment.target!r}, not {target!r}")
    snapshot_repo, snapshot_identity = _verified_model_host_lane_snapshot()
    package, identity = experiment.resolve_host_lane(root=snapshot_repo)
    if snapshot_identity is not None:
        identity["run_snapshot"] = snapshot_identity

    # An explicit package bypasses compile_rvv's datatype-aware default selection. Reject a known
    # cross-datatype substitution: otherwise an int8 model with an fp32 package sets
    # ``int8_compute=False`` and the numeric outcome is about a different host program. Dtypes such as
    # fp8 deliberately have no scalar strategy; compile_rvv derives their scalar lane from captured IR,
    # so only a strategy it can decide here is gated here.
    from .. import compile_cli as _compile_cli
    expected_strategy = _compile_cli._DTYPE_STRATEGY.get(dtype)
    if expected_strategy is not None and identity["dtype_strategy"] != expected_strategy:
        raise ValueError(
            f"descriptor host package declares dtype_strategy={identity['dtype_strategy']!r}, but "
            f"model compile_dtype={dtype!r} requires {expected_strategy!r}")
    return experiment, package, identity


_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"
_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"


def _verified_model_host_lane_snapshot() -> tuple[Path | None, dict[str, Any] | None]:
    """Return the verified run-snapshot repo root used for host-lane grading.

    A bwrap experiment sets ``*_REQUIRED`` and points ``*_ROOT`` at its host-only
    ``bundle_inputs`` directory. The aggregate was recomputed immediately before export; resolving and
    hashing the declared host package below joins the bytes mounted for the agent to the bytes passed
    into ``compile_model``. Using the live worktree instead would let a post-launch edit change only the
    grader's compiler.

    Callers outside the bwrap experiment remain compatible: with neither variable set, the descriptor
    resolves against the live repository exactly as before. A declared/required snapshot never falls
    back to that path.
    """
    raw_required = os.environ.get(_MODEL_HOST_SNAPSHOT_REQUIRED_ENV, "").strip().lower()
    if raw_required not in ("", "0", "false", "no", "off", "1", "true", "yes", "on"):
        raise ValueError(
            f"{_MODEL_HOST_SNAPSHOT_REQUIRED_ENV} has invalid boolean value {raw_required!r}")
    required = raw_required in ("1", "true", "yes", "on")
    raw_root = os.environ.get(_MODEL_HOST_SNAPSHOT_ROOT_ENV, "").strip()
    if not raw_root:
        if required:
            raise ValueError(
                f"bwrap model grading requires {_MODEL_HOST_SNAPSHOT_ROOT_ENV}; no verified run "
                "snapshot was provided")
        return None, None

    lexical_root = Path(raw_root)
    if not lexical_root.is_absolute() or lexical_root.is_symlink():
        raise ValueError(
            f"{_MODEL_HOST_SNAPSHOT_ROOT_ENV} must name a non-symlink absolute directory")
    try:
        root = lexical_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"model host-lane run snapshot is missing at {lexical_root}") from exc
    marker = root / "snapshot.json"
    if not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise ValueError(f"model host-lane run snapshot is incomplete or unsafe at {root}")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"model host-lane run snapshot marker is unreadable at {marker}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 2:
        raise ValueError("model host-lane run snapshot marker has an unsupported format")

    # Setup/resume has already recomputed this aggregate through verify_bundle_snapshot before it is
    # exported. Preserve that run identity here, while HostLane.resolve below independently hashes the
    # exact (small) compiler package before and after compilation. Re-hashing an entire bundle here would
    # read its LLVM toolchain twice for every capstone and add no host-package evidence.
    digest = manifest.get("content_sha256")
    n_files = manifest.get("n_files")
    n_bytes = manifest.get("n_bytes")
    if (not isinstance(digest, str) or len(digest) != 64 or digest != digest.lower() or
            any(ch not in "0123456789abcdef" for ch in digest) or
            not isinstance(n_files, int) or n_files < 1 or
            not isinstance(n_bytes, int) or n_bytes < 1):
        raise ValueError("model host-lane run snapshot marker has invalid aggregate identity")

    repo = root / "repo"
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError(f"model host-lane run snapshot has no safe repository payload at {repo}")
    resolved_repo = repo.resolve(strict=True)
    if root not in resolved_repo.parents:
        raise ValueError(f"model host-lane run snapshot repository escapes {root}")
    return resolved_repo, {
        "root": str(root),
        "repo_root": str(resolved_repo),
        "content_sha256": digest,
        "n_files": n_files,
        "n_bytes": n_bytes,
        "version": 2,
        "source_repo": manifest.get("repo"),
    }


def _grade_model_capsule_inline(capsule: dict, *, target: str | None = None, timeout: int,
                                package_dir: str | Path | None = None) -> dict:
    """Grade a whole-model (kind == "model") capsule end to end via the target-aware whole-model flow
    (``compile_model``): route each op across the target's compute units (matmul/systolic -> the mesh, the
    rest -> the vector/scalar lane), compile the functional whole model (the scalar/RVV reference,
    numerically correct across every op), and gate its output vs the model's golden — attaching the per-op
    mesh-routing plan for the target. ``MERLIN_MODEL_GRADE_RUN`` selects ``host`` (default; x86 dispatch
    runtime) or ``spike``. Honestly reports ``incomplete`` when the whole-model toolchain (m2m venv /
    clang-23) is absent — never a silent pass."""
    import os
    from pathlib import Path as _P
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    model, dtype = attrs.get("model"), attrs.get("compile_dtype", "fp32")
    # Which lane executes the model. The default was the host dispatch runtime unconditionally, so a
    # capsule that DEMANDS acceleration was graded on a lane that cannot provide it and passed anyway.
    # Derive it instead: a capsule whose semantic block asserts must_accelerate runs on the mesh lane,
    # everything else keeps the host lane. The env var still overrides, for a deliberate diagnostic run.
    _sem = capsule.get("semantic") or {}
    # WHICH LANE THE MODEL RUNS ON. must_accelerate implies the mesh lane. So does an interop capsule
    # that REQUIRES on_mesh: it withholds must_accelerate on purpose (host work is the behaviour under
    # test), but running it on the host lane means the mesh never executes and its own lane requirement
    # becomes unverifiable -- which is exactly how the first such capsule passed while executing nothing
    # on the accelerator.
    _req_lanes = [str(x) for x in ((capsule.get("lanes") or {}).get("require") or [])]
    # THE TARGET'S OWN MESH IS THE ORACLE WHENEVER WE HAVE A TARGET. This used to key on the capsule's
    # must_accelerate / required lanes, which left a capsule declaring neither graded on the HOST even
    # when a target was named -- and merlin's x86 dispatch runtime is OUR compiler, so however well it
    # does it is evidence about us, never about the submission. Keying on the target subsumes both
    # declarations: a capsule demanding acceleration on a target with no mesh has nothing to run on
    # either way. The env var still overrides for a deliberate diagnostic run.
    run_where = os.environ.get("MERLIN_MODEL_GRADE_RUN") or ("mesh" if target else "host")
    _ = _req_lanes  # still read below by the lane report
    # MESH VERIFICATION IS THE CAPSULE'S DEMAND, NOT AN OPERATOR'S OPT-IN. It EXECUTES each mesh-routed
    # matmul as a single systolic tile on the target's real mesh oracle (compile_model mesh_verify),
    # which is the only evidence that the matmul layers run ON the mesh rather than that a routing plan
    # was produced.
    #
    # It used to be off unless an env var said otherwise, because the oracle build/run is heavy. The
    # cost is real; the default was still wrong. A capsule that declares `must_accelerate`, or that
    # names `on_mesh` in `lanes.require`, is asking for exactly this evidence -- and without it the tier
    # ladder below has nothing to record, so the capstone reports a verdict backed by the functional
    # lane alone. That is `capstone-graded-tiles-not-the-model` with the accelerator left out entirely.
    # Derive the default from what the capsule demands; the env var still overrides in BOTH directions,
    # for a deliberate diagnostic run. The wall-clock cost is bounded by MERLIN_MODEL_BUDGET_S.
    _demands_mesh = bool(_sem.get("must_accelerate")) or "on_mesh" in _req_lanes
    _mv_env = os.environ.get("MERLIN_MESH_VERIFY", "").strip().lower()
    mesh_verify = (_mv_env in ("1", "true", "yes", "on")) if _mv_env else _demands_mesh
    result: dict = {"capsule": capsule["name"], "kind": "model", "label": capsule.get("label"),
                    "operation": {"op": "model", "model": model, "dtype": dtype, "run": run_where,
                                  "target": target},
                    "contract_version": CONTRACT_VERSION}
    if not model:
        result.update(status="incomplete",
                      failure={"plane": "model", "category": "RUNNER_CRASH",
                               "detail": "model capsule missing operation.attributes.model"})
        return result

    # The target submission and the scalar/vector host lane are TWO compiler packages. The former is
    # threaded below as ``mesh_package``; the latter is frozen experiment infrastructure and MUST come
    # from this target's descriptor. Passing None used ``default_package`` (the currently certified RVV
    # champion), which could differ from the package granted read-only to the agent. The resulting
    # boundary/interop verdict then depended on a compiler the experiment neither declared nor exposed.
    _host_package = None
    _host_identity = None
    _host_experiment = None
    if target:
        try:
            _host_experiment, _host_package, _host_identity = _resolve_model_host_lane(target, dtype)
            result["host_lane"] = _host_identity
        except Exception as e:  # noqa: BLE001 — an unpinned host compiler cannot produce a verdict
            result.update(status="incomplete",
                          failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": f"frozen host lane unavailable: {type(e).__name__}: "
                                             f"{str(e)[:300]}"})
            return result
    # The captured linalg (visible grounding) drives the per-op mesh routing when present. Read the name
    # the CAPSULE DECLARES -- every model capsule ships `linalg_mlir: capsule.interface.mlir`, while this
    # looked only for `capsule.linalg.mlir`, so it never found one. `linalg_mlir` was therefore always
    # None, `compile_model`'s `if target and linalg_mlir:` never fired, and the routing plan, the coverage
    # certificate AND the mesh verification were all skipped on every whole-model capsule ever graded --
    # silently, because a skipped block leaves no trace in the result.
    linalg_mlir = None
    linalg_path = None
    cdir = capsule.get("__dir__")
    if cdir:
        for _name in (capsule.get("linalg_mlir"), capsule.get("interface_mlir"), "capsule.linalg.mlir"):
            if not _name:
                continue
            lp = _P(cdir) / str(_name)
            if lp.is_file():
                linalg_path = lp
                linalg_mlir = lp.read_text(encoding="utf-8")
                break
    from ..xdsl_dialects.lowering.passes import MODEL_BOUNDARY_CAPSTONE
    _requires_boundary_plane = MODEL_BOUNDARY_CAPSTONE in (
        capsule.get("pass_requirements") or ())
    if _requires_boundary_plane and linalg_path is None:
        result.update(status="incomplete",
                      failure={"plane": "pass_obligations", "category": "NOT_RUN_IS_NOT_PASS",
                               "detail": "model capsule has no readable declared linalg/interface module; "
                                         "the production boundary passes cannot be exercised"})
        return result
    try:
        # This is the production, in-process boundary plane: outline the REAL model, materialize and
        # verify its dispatch DAG, partition it, and attach the C ABI. It is deliberately separate from
        # the OOT package's four subprocess stages, which run below on each mesh tile. A pass-obligation
        # failure is structural and blocks the capsule before expensive oracle work.
        if _requires_boundary_plane:
            from ..frontends.linalg_mlir import parse_mlir_file
            from ..xdsl_dialects.lowering.passes import run_dialect_plane
            _plane = run_dialect_plane(parse_mlir_file(linalg_path), n_harts=1)
            result["pass_obligations"] = _plane.stats

        from ..compile_cli import compile_model
        # `dtype` is the capsule's compile_dtype (an RVV compile mode). The capsule ALSO declares the
        # exact datapath format in operation.attributes.dtype -- pass that for routing so a demand is
        # matched against the unit's declared format rather than against a compile-mode token.
        _attrs = (capsule.get("operation") or {}).get("attributes") or {}
        # THE PACKAGE UNDER TEST MUST REACH THE MESH ORACLE. `run_capsule` is handed the submission
        # being graded, but this function never took it and called `compile_model(package=None)`, so a
        # whole-model capsule was compiled and mesh-verified against the DEFAULT path -- not against the
        # submission. Every whole-model number that resulted (numeric verdict, layers-on-mesh accounting,
        # tile certification) was therefore a statement about the reference flow, not about the backend
        # the capsule was supposed to be judging. Measured: mesh verification reported
        # `n_tiles: 0, reason: "no default OOT backend package for target"` while a perfectly good
        # submission sat in the caller's hand.
        #
        # It goes to `mesh_package` (the OOT ACCELERATOR backend that certifies tiles), while `package`
        # is the independently frozen RVV host lane -- two things that a single name would conflate.
        _pkg = str(package_dir) if package_dir else None
        _host_package_arg = str(_host_package) if _host_package is not None else None
        out = compile_model(model, dtype, target=target, run=run_where, verify=True,
                            package=_host_package_arg,
                            auto_capture=True, timeout=timeout, linalg_mlir=linalg_mlir,
                            mesh_verify=mesh_verify, mesh_package=_pkg,
                            routing_dtype=_attrs.get("dtype"))
    except SystemExit as e:                                   # toolchain/bundle unavailable — honest skip
        result.update(status="incomplete",
                      failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                               "detail": f"whole-model compile/run unavailable: {str(e)[:300]}"})
        return result
    except Exception as e:  # noqa: BLE001
        result.update(status="incomplete",
                      failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                               "detail": f"whole-model grade error: {type(e).__name__}: {str(e)[:300]}"})
        return result
    # Detect drift across the compile, not only before it. For a bwrap run this re-verifies the same
    # immutable snapshot aggregate and resolves the package inside its repo payload; it NEVER switches
    # to the live worktree. Direct/non-bwrap grading retains its live-tree pre/post check.
    if _host_experiment is not None and _host_identity is not None:
        try:
            _snapshot_repo_after, _snapshot_after = _verified_model_host_lane_snapshot()
            _snapshot_before = _host_identity.get("run_snapshot")
            if (_snapshot_before is None) != (_snapshot_after is None):
                raise ValueError("model host-lane run snapshot context changed during grading")
            if _snapshot_before is not None:
                if (_snapshot_after.get("root") != _snapshot_before.get("root") or
                        _snapshot_after.get("content_sha256") !=
                        _snapshot_before.get("content_sha256")):
                    result["host_lane_snapshot_after"] = _snapshot_after
                    raise ValueError("model host-lane run snapshot changed during grading")
            _, _host_after = _host_experiment.resolve_host_lane(root=_snapshot_repo_after)
        except Exception as e:  # noqa: BLE001
            result.update(status="incomplete",
                          failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": f"frozen host lane became unreadable during grading: "
                                             f"{type(e).__name__}: {str(e)[:300]}"})
            return result
        if _host_after["package_sha256"] != _host_identity["package_sha256"]:
            result.update(status="incomplete", host_lane_after=_host_after,
                          failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": "frozen host-lane package changed during grading "
                                             f"({_host_identity['package_sha256'][:12]} -> "
                                             f"{_host_after['package_sha256'][:12]}); refusing a "
                                             "mixed-compiler verdict"})
            return result
    st, gate = out.get("status"), (out.get("verify") or {}).get("gate_ok")
    engine = f"merlin-compile model --target {target} --run {run_where} --verify"
    if out.get("routing_plan") is not None:                   # per-op mesh routing for the target
        result["routing_plan"] = out["routing_plan"]
    if out.get("coverage_certificate") is not None:           # ARR certificate (numerator×independent oracle)
        result["coverage_certificate"] = out["coverage_certificate"]
    if out.get("mesh_execution") is not None:                 # THIS MODEL's layers: on-mesh vs host
        result["mesh_execution"] = out["mesh_execution"]
    if out.get("mesh_tile_verification") is not None:         # synthesized tiles of the routed shapes
        result["mesh_tile_verification"] = out["mesh_tile_verification"]
    # A quantized whole model diverges from the fp32 golden BY DESIGN (int8/fp8 rounding). When the strict
    # gate rejects it only on that expected drop, accept it if the cosine similarity to the golden still
    # clears a quant floor (MERLIN_MODEL_QUANT_COS, default 0.90) — reasonable quantization error is a pass,
    # a gross codegen defect (cosine below the floor / structural mismatch) is still a fail.
    _v = out.get("verify") or {}
    _cos = max(float(_v.get("fp32_cos", 0.0)), float(_v.get("w8a8_cos", 0.0)))
    _quant = dtype in ("int8", "i8", "fp8", "fp8_e4m3")
    _quant_floor = float(os.environ.get("MERLIN_MODEL_QUANT_COS", "0.90") or "0.90")

    # A HOST RUN NEVER PASSES A MODEL CAPSULE, and never fails one either. It is merlin's own x86
    # dispatch runtime -- our compiler -- so however well or badly it does, it is evidence about US and
    # not about the submission. Reporting its mismatch as the capsule's `fail` attributes a defect in
    # our reference to a backend that never executed; reporting its match as `pass` is how a suite reads
    # N/N while the submitted backend never ran. So the diagnostic is RECORDED and the verdict WITHHELD.
    # No provenance stamp rides a withheld verdict: it makes no hardware claim to attribute.
    if run_where == "host":
        result["host_reference"] = {"status": st, "gate": _v, "engine": engine,
                                    "note": "merlin's own dispatch runtime — a diagnostic, never an "
                                            "oracle for a submitted backend"}
        result.update(status="not_gradeable_no_oracle",
                      numeric=_numeric_not_compared(
                          engine, "host_reference", _v,
                          "the whole model ran on merlin's host dispatch runtime; the numeric verdict "
                          "is withheld because that is our compiler, not the target"),
                      failure={"plane": "not_gradeable_no_oracle",
                               "category": "NOT_GRADEABLE_NO_ORACLE",
                               "detail": f"no target mesh ran this model (run={run_where}); a host "
                                         f"reference cannot certify a submitted backend"})
        result.pop("provenance", None)
        return result
    # FAIL CLOSED ON AN UNEXERCISED TIER. A whole-model capsule declares required_oracle_tiers like any
    # other, but this function never entered the tier ladder, so those tiers were dead metadata and a
    # capsule could report `pass` with `tiers == {}` -- a verdict backed by no oracle at all. Worse, the
    # functional gate here is the HOST x86 run unless mesh execution was requested, so "pass" could mean
    # "the CPU computed the model correctly", which is not a statement about the accelerator.
    #
    # Record what actually ran, and refuse to call it a pass when nothing the capsule DECLARED ran. The
    # accelerator evidence is `mesh_tile_verification` (per-tile on-mesh certification); a routing plan is
    # a plan, not an execution, and is never counted as a tier.
    declared = [str(x) for x in (capsule.get("required_oracle_tiers") or [])]
    mesh_exec = out.get("mesh_tile_verification") or {}
    model_exec = out.get("mesh_execution") or {}
    n_tiles = int(mesh_exec.get("n_tiles") or 0) if isinstance(mesh_exec, dict) else 0
    # Set BEFORE the fail-closed branches below, every one of which returns early: a refusal must still
    # say which tier refused it, and this block used to be attached only on the success path.
    _model_tiers = _model_tier_map(declared, target, model_exec)
    result["tiers"] = {k: v.to_dict() for k, v in _model_tiers.items()}
    exercised: dict[str, str] = {}
    if n_tiles:
        # The verdict comes from the per-tile COUNTS. Reading a boolean `ok` key -- which this dict does
        # not carry -- made every on-mesh execution record "fail", including one where all 15 layers
        # passed on the oracle. A tile that was unavailable or unsynthesizable is NOT a pass either, so
        # the tier passes only when every tile is accounted for and every one of them passed.
        _passed = int(mesh_exec.get("n_passed") or 0)
        _failed = int(mesh_exec.get("n_failed") or 0)
        _unavail = int(mesh_exec.get("n_unavailable") or 0)
        _unsynth = int(mesh_exec.get("n_unsynthesizable") or 0)
        _ok = mesh_exec.get("ok")
        _tile_ok = bool(_ok) if _ok is not None else (
            _failed == 0 and _unavail == 0 and _unsynth == 0 and _passed == n_tiles)

        # THE MODEL'S OWN LAYERS DECIDE A MODEL CAPSULE'S TIER, not tiles of its shapes. Certifying a
        # synthesized tile proves the SHAPE is runnable; the capstone is a claim about THIS model, and
        # the two came apart once already -- a run with all 15 layers on the host reported "15 of 15
        # tiles passed". When the model ran the mesh lane, its own per-layer accounting must agree:
        # every routed layer on the accelerator, none fallen back. A host-lane model run has no such
        # accounting and is left to the tile record alone.
        _model_ok = True
        if model_exec:
            _on = model_exec.get("matmul_layers_on_mesh")
            _model_ok = (_on is not None and int(_on) > 0
                         and not int(model_exec.get("matmul_layers_host_fallback") or 0))
        # INTEROP capsules invert one half of that. A capsule may DECLARE that its point is composition
        # across lanes -- part of the model on the accelerator, the rest on the scalar/vector lane the
        # target also owns -- in which case "no host fallback" is the wrong bar: it would fail the exact
        # behaviour under test. Such a capsule names the lanes it requires, and passes only when EVERY
        # named lane actually carried work.
        #
        # The lane names are the routing plan's OWN keys (`on_mesh`, `in_contract_vector_scalar`,
        # `scalar_rvv_lane`), so this asserts against what the compiler reported rather than a vocabulary
        # invented here; a capsule naming a lane the plan does not report fails closed with that lane
        # named, which is the actionable direction.
        _lane_rep = lane_report(capsule, result.get("routing_plan"), model_exec)
        if _lane_rep is not None:
            result["lane_report"] = _lane_rep
            _model_ok = not _lane_rep["unexercised"]

        # WHICH tier the mesh oracle corresponds to, DERIVED from the target's own declared RTL tiers.
        # This was `[x for x in declared if x not in ("L0", "L1")]` with an `"L3"` fallback -- three tier
        # -name literals standing in for a fact the manifest already carries, which names the wrong tier
        # confidently on any target whose ladder differs (atlas grades at L3+L4, not L3).
        _rtl = [t for t in declared if t in _rtl_tiers_of(target)]
        _tier = _rtl[-1] if _rtl else (declared[-1] if declared else None)
        if _tier:
            exercised[_tier] = "pass" if (_tile_ok and _model_ok) else "fail"
    # WHOSE COMPILER PASSED. When the runtime had to discharge a contract obligation the target backend
    # owes -- residency tiling for `capacity_fit` -- the verdict is about runtime+backend together, and
    # saying so is the difference between "this compiler handles a 512x512 layer" and "this layer ran".
    _delegated = (model_exec or {}).get("capacity_fit_delegated_to_runtime") or []
    if _delegated:
        # If ANY delegated layer was tiled from the backend's DECLARED tile rather than from a capacity
        # limit, this verdict rests on the runtime having driven a loop nest the backend does not have.
        # That is a different claim and the score reports it as one (`backend_coverage`).
        _tb = sorted({str(d.get("tiled_by")) for d in _delegated if d.get("tiled_by")})
        result["contract_obligations"] = {
            "capacity_fit": {
                "discharged_by": "merlin runtime (host-side residency tiling)",
                "tiled_by": ("declared_primitive_tile" if "declared_primitive_tile" in _tb
                             else (_tb[0] if _tb else None)),
                "n_layers": len(_delegated), "layers": _delegated[:8],
                "detail": ("the target backend did not satisfy capacity_fit at these extents; the "
                           "runtime split them so the model could run. This verdict is evidence about "
                           "the runtime AND the backend, not about the backend alone."),
            }
        }
    # Overlay the per-tile verdict where one exists; the counter-derived map is the floor.
    for _t, _s in exercised.items():
        base = _model_tiers.get(_t)
        _model_tiers[_t] = TierResult(_t, _s, True, reason=(base.reason if base else None))
    result["tiers"] = {k: v.to_dict() for k, v in _model_tiers.items()}
    # The tile record, kept beside the verdict as the SEPARATE and weaker evidence it is: it speaks about
    # synthesized tiles of this model's shapes, never about the model.
    if mesh_exec:
        result["tile_evidence"] = {"n_tiles": n_tiles, "n_passed": mesh_exec.get("n_passed"),
                                   "note": "synthesized tiles OF THIS MODEL'S SHAPES — evidence that the "
                                           "shapes run, not that this model ran"}
    result["op_coverage"] = {"note": "the op-pass fraction this capsule was gated on is OP COVERAGE, "
                                     "not a verdict on the model"}
    unexercised = [x for x in declared if x not in exercised]

    # ORDER MATTERS: the SPECIFIC cause is reported before the generic consequence. Layers falling back
    # to the host now also drag the tier verdict down (the model, not its tiles, decides it), so leaving
    # the failed-tier branch first turned "2 of 37 layers fell back" into a bare FUNCTIONAL_MISMATCH --
    # true, and useless. The fallback check runs first so the result names what to fix.
    # MUST_ACCELERATE IS ABOUT THIS MODEL, NOT ABOUT TILES OF ITS SHAPES. Certifying a synthesized tile
    # proves the shape is runnable; it says nothing about whether the model's own layer reached the
    # device. Measured: atlas routed 15 matmul layers and the dispatch runtime fell back to the host
    # kernel on all 15, while the tile record read "15 of 15 passed" -- so a run with ZERO layers on the
    # accelerator reported `pass` with `lane: mesh`, which is exactly the CPU-only pass this flag exists
    # to forbid. The two records now live under separate keys and this checks the model one.
    if (capsule.get("semantic") or {}).get("must_accelerate") and model_exec:
        _on = model_exec.get("matmul_layers_on_mesh")
        _fb = int(model_exec.get("matmul_layers_host_fallback") or 0)
        if _on is None:
            result.update(status="incomplete",
                          failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": "capsule declares must_accelerate but the run recorded no "
                                             "per-layer mesh accounting; cannot confirm the model "
                                             "reached the accelerator"})
            return result
        # AN UNMEASURED LAYER IS NOT A FALLEN-BACK LAYER. When the oracle could not tell us whether the
        # mesh runs a layer, the honest verdict is `incomplete` (NOT_RUN_IS_NOT_PASS), not a compiler
        # failure -- the same rule the tier ladder already applies one level up. Measured: a whole model
        # whose every layer the mesh executes correctly reported "15 of 15 fell back" because the
        # simulator timed out at its per-layer budget, and that number failed must_accelerate.
        _unavail = int(model_exec.get("matmul_layers_oracle_unavailable") or 0)
        if int(_on) == 0 and _unavail and not _fb:
            result.update(status="incomplete",
                          numeric=_numeric_when_not_accelerated(
                              st, gate, _v, _cos, engine, measured_on="host_lane_unmeasured"),
                          failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": f"the mesh oracle could not measure {_unavail} of this "
                                             f"model's matmul layer(s) (timed-out or unreachable "
                                             f"simulator), so whether they run on the {target} mesh is "
                                             f"UNKNOWN -- not a fallback, and not evidence about the "
                                             f"backend. Re-run with an oracle that completes."})
            return result
        if int(_on) == 0 and not _fb and not n_tiles:
            # NOTHING RAN AT ALL -- no certified tile and no layer of the model itself. That is not a
            # fallback (nothing was claimed and then missed); it is the absence of a run, and the
            # difference matters: a fallback is evidence about the backend's reach, an empty record is
            # evidence about nothing. Withheld, never failed.
            result.update(status="incomplete",
                          numeric=_numeric_not_compared(engine, run_where, _v, "nothing was compared ON THE ACCELERATOR: no declared tier ran, so the host-side number is not a verdict about this backend. A comparison that did not happen is not a passing comparison."),
                          failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": f"neither a certified tile nor a single matmul layer of "
                                             f"this model executed on the {target} mesh, so there is no "
                                             f"verdict to report either way"})
            return result
        if int(_on) == 0 or _fb:
            result.update(status="fail",
                          numeric=_numeric_when_not_accelerated(
                              st, gate, _v, _cos, engine, measured_on="host_lane_fallback"),
                          failure={"plane": "model", "category": "FALLBACK_ON_ELIGIBLE_REGION",
                                   "detail": f"capsule declares must_accelerate but only {_on} matmul "
                                             f"layer(s) executed on the {target} mesh and {_fb} fell back "
                                             f"to the host kernel; the numeric gate therefore measures "
                                             f"the HOST, not the accelerator"})
            return result

    # A tier that RAN AND FAILED is not a pass, whatever the host-side numeric gate says. The guard below
    # only refuses the case where nothing ran at all, so a failing accelerator tier still reported
    # `status: pass` beside `tiers: {L3: fail}` -- the two halves of the same result contradicting each
    # other, with the flattering half being the one anybody reads.
    # Counter-derived failures gate too. Reading only the per-tile record let a model with three layers
    # fallen back report `pass` beside `tiers: {L2: fail}` -- the two halves of one result contradicting
    # each other, with the flattering half being the one anybody reads.
    _failed_tiers = sorted({t for t, v in exercised.items() if v != "pass"}
                           | {k for k, v in _model_tiers.items() if v.status == "fail"})
    if _failed_tiers:
        result.update(status="fail",
                      numeric=_numeric_when_not_accelerated(
                          st, gate, _v, _cos, engine, measured_on=run_where),
                      failure={"plane": "model", "category": "FUNCTIONAL_MISMATCH",
                               "detail": f"declared oracle tier(s) {_failed_tiers} RAN and did not pass "
                                         f"(on-mesh execution: {mesh_exec.get('n_passed')} of "
                                         f"{n_tiles} tile(s) passed, {mesh_exec.get('n_failed')} failed, "
                                         f"{mesh_exec.get('n_unavailable')} unavailable, "
                                         f"{mesh_exec.get('n_unsynthesizable')} unsynthesizable); a "
                                         f"whole-model verdict cannot be a pass over a failing tier"})
        if unexercised:
            result["tiers_unexercised"] = _unexercised_note(unexercised, exercised)
        return result

    # A tier the model's own counters resolved COUNTS as exercised. Keying this only on the per-tile
    # record made every run without tile verification report `incomplete`, including one whose layers
    # demonstrably ran on the mesh.
    _counter_ran = {k for k, v in _model_tiers.items() if v.status in ("pass", "fail")}
    if declared and not exercised and not _counter_ran:
        # Same reasoning as the branches above: the arithmetic WAS measured before we got here, so report
        # it. The verdict stays `incomplete` -- no declared tier ran, and a number alone is not a tier.
        result.update(status="incomplete",
                      numeric=_numeric_not_compared(engine, run_where, _v, "nothing was compared ON THE ACCELERATOR: no declared tier ran, so the host-side number is not a verdict about this backend. A comparison that did not happen is not a passing comparison."),
                      failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                               "detail": f"declares required oracle tiers {declared} and ran NONE of them "
                                         f"(the functional gate here is the {run_where} reference, not the "
                                         f"accelerator); a whole-model verdict backed by no declared tier "
                                         f"is reported UNKNOWN, never a pass"})
        return result

    if st == "verified" and gate:
        # A whole-model PASS must carry the strength of the gate that produced it. The numeric gate has
        # a per-element ceiling for exactly the failure aggregates hide (a single element 1209% wrong at
        # cos 0.9999986), but its cosine-only tier bypasses that ceiling for regression outputs which
        # cannot meet it. Both are legitimate passes; they are not the same claim, and "certified" read
        # off `status` alone erases the difference.
        _v = out.get("verify") or {}
        _guarded = _v.get("per_element_guarded")
        # A PASS EARNED ON THE TARGET IS A HARDWARE CLAIM, so it must say which tree produced it. A
        # result attributed to the wrong revision is worse than no result, because it gets cited.
        try:
            from ..common import provenance as _prov
            result["provenance"] = _prov.record(extra={"target": target, "run": run_where})
        except Exception as _pe:                            # noqa: BLE001 - never block a verdict
            result["provenance"] = {"status": "UNKNOWN", "detail": f"{type(_pe).__name__}: {_pe}"}
        result.update(status="pass",
                      numeric={"status": "pass", "engine": engine, "gate": _v,
                               "gate_tier": _v.get("tier_ok"),
                               "per_element_guarded": _guarded,
                               "evidence": ("per-element-guarded" if _guarded else
                                            "AGGREGATE ONLY — the cosine-only tier carried this "
                                            "verdict; no per-element bound was applied")})
        if unexercised:
            result["tiers_unexercised"] = _unexercised_note(unexercised, exercised)
    elif _quant and st == "run_mismatch" and _cos >= _quant_floor:
        result.update(status="pass",
                      numeric={"status": "pass", "engine": engine, "gate": _v,
                               "quant_tolerance": {"cos": _cos, "floor": _quant_floor, "dtype": dtype}},
                      note=(f"quantized ({dtype}) whole-model output within quant tolerance of the fp32 "
                            f"golden (cos {_cos:.4f} >= floor {_quant_floor}); the drop vs fp32 is expected "
                            f"quantization error, not a codegen defect."))
    elif st == "not_run":
        result.update(status="incomplete",
                      failure={"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                               "detail": out.get("reason", "whole-model run toolchain unavailable")})
    else:
        result.update(status="fail",
                      numeric={"status": "fail", "engine": engine, "gate": out.get("verify"),
                               "model_status": st},
                      failure={"plane": "model", "category": "FUNCTIONAL_MISMATCH",
                               "detail": f"the whole model did not verify (status={st}, gate_ok={gate})"})
    return result


def _merge_match_policy(force: dict | None, capsule: dict | None) -> dict | None:
    """Combine a target's ``force_match_policy`` with the capsule's declared ``numeric_policy`` so neither's
    tolerance is silently discarded. The compare MODE comes from the force policy (a float/SIMT target
    grades with tolerance even for an integer-policy capsule); ``atol``/``rtol`` take the LOOSER (max) of
    the two present values — a capsule's bf16-appropriate tolerance is honoured, and the tight global
    default still applies where the capsule declares none. Returns the other policy unchanged when one side
    is absent."""
    if not force:
        return capsule
    if not capsule:
        return force
    merged = dict(force)
    for k in ("atol", "rtol"):
        vals = [float(p[k]) for p in (force, capsule) if p.get(k) is not None]
        if vals:
            merged[k] = max(vals)
    return merged


def _finalize_capsule_result(*, name: str, capsule: dict, status: str, failure: dict | None,
                             tiers: dict, trace_check_res: dict, numeric: dict, required,
                             no_oracle: bool, eff_target: str, paths, run_id: str, cfg,
                             contract=None, executability: dict | None = None,
                             declined: dict | None = None, extra: dict | None = None,
                             submission: dict | None = None) -> dict:
    """Turn a graded capsule's parts into its result row, and write it.

    Lifted VERBATIM out of :func:`run_capsule`, whose tail this was, for two reasons.

    The MODEL capsule path finalizes separately, so every rule enforced here -- a mandatory tier that
    did not run cannot yield a pass, a screened capsule is not a pass, a capsule with no runnable
    required tier does not pass on our own command-buffer interpretation alone -- applied to ISA
    capsules and silently did not apply to model ones.

    And the behaviour becomes testable directly, on the row this returns, instead of only through a
    full graded run. Pinning it to a source line inside `run_capsule` is testing where code lives
    rather than what it does, which is how that assertion broke when the block last moved.

    Pure with respect to grading: it decides `status`/`failure` from what it is handed and never
    re-runs an oracle. Its side effects are the artifacts the row implies -- capsule_result.json and
    the run manifest -- plus a self-validation warning."""
    # Imported here, as run_capsule did: this module's import of .provenance is deferred to call time.
    from .provenance import toolchain_shas

    # not_run_is_not_pass: a mandatory tier that did not pass closed (unavailable/skipped/absent) makes
    # the capsule incomplete — never a silent pass. A tier that is honestly N/A for this capsule's
    # datatype (``not_applicable``; the integer L0/L1 floor on a float datapath) is the ONE exception —
    # a legitimate skip like a dropped RoCC gate, not a missing oracle. An unavailable/absent RTL oracle
    # is never not_applicable, so it still fails closed here.
    if status == "pass" and any(getattr(t, "budget_deferred", False) for t in tiers.values()):
        # SCREENED, NOT CERTIFIED. Distinct from `incomplete` (something that should have run did not)
        # and from `pass` (it certified). The capsule cleared the cheap screen and the expensive tier was
        # deliberately not bought, because a capsule in the covering set already certifies the axes this
        # one exercises. Excluded from the pass/fail denominators and reported by name — the one thing it
        # must never do is read as a pass.
        _def = sorted(t for t, r in tiers.items() if getattr(r, "budget_deferred", False))
        status = "screened_only"
        failure = {"plane": "budget", "category": "SCREENED_NOT_CERTIFIED",
                   "tier": _def[0] if _def else None,
                   "detail": (f"passed the screen tier; certify tier(s) {', '.join(_def)} not purchased "
                              f"(outside the derived covering set, certify budget exhausted). This is "
                              f"NOT a verdict on this capsule.")}
    if status == "pass":
        for tier in required:
            tr = tiers.get(tier)
            if tr is not None and getattr(tr, "not_applicable", False):
                continue
            if tr is None or tr.status in ("unavailable", "skipped"):
                if no_oracle:
                    # Explicit --no-oracle STRUCTURE-ONLY smoke: NO numeric oracle was requested this run,
                    # so a mandatory numeric tier that did not run is NOT a fixable failure — it is
                    # honestly NOT GRADEABLE. Record a DISTINCT status/plane so the agent is not handed a
                    # phantom `oracle_unavailable` "fix this" signal it can never satisfy. INTEGRITY: this
                    # is NEVER reported as a pass — the numeric verdict is simply withheld. The
                    # not_run_is_not_pass gate for GRADED (oracle-present) runs is the `else` branch below,
                    # unchanged; keep no_oracle False for every real grade.
                    status = "not_gradeable_no_oracle"
                    if failure is None:
                        failure = {"plane": "not_gradeable_no_oracle",
                                   "category": "NOT_GRADEABLE_NO_ORACLE",
                                   "detail": f"numeric oracle unavailable this run (--no-oracle): "
                                             f"mandatory tier {tier} not graded — structural tiers "
                                             f"(L0/L1/trace) only"}
                else:
                    status = "incomplete"
                    if failure is None:
                        # The tier's OWN reason is the only actionable half of this message, and it was
                        # being dropped. Measured on a self-hosted-ISA target: every capsule carried
                        # "program did not halt within N instructions" while the agent was shown only
                        # "mandatory tier L# did not run (unavailable)". Ten rounds of a fully conformant
                        # model went into feedback naming nothing it could fix; effort decayed each round.
                        # An oracle that is ABSENT and a program that RAN AND HUNG are different events;
                        # when the tier reported a reason, that reason leads.
                        _why = (getattr(tr, "reason", None) or "").strip() if tr else ""
                        _st = tr.status if tr else "absent"
                        failure = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS",
                                   "tier": tier, "tier_status": _st, "tier_reason": _why or None,
                                   "detail": (f"{_why} (mandatory tier {tier}, status {_st})" if _why
                                              else f"mandatory tier {tier} did not run ({_st})")}
                break

    # Fail-open guard (complements the loop above, which never fires for an EMPTY/all-N/A required set):
    # a capsule that declares no runnable oracle requirement must NOT grade 'pass' on the L0/L1 command-
    # buffer interpretation alone — that is our own engine, not an independent oracle. A real grade
    # (not --no-oracle) requires at least one runnable, non-N/A required tier to have certified it.
    if status == "pass" and not no_oracle:
        ran_required = [t for t in required
                        if tiers.get(t) is not None
                        and not getattr(tiers[t], "not_applicable", False)
                        and tiers[t].status == "pass"]
        if not ran_required:
            status = "incomplete"
            if failure is None:
                # When the phase capped a DECLARED tier away, name it: "this phase cannot reach the tier
                # your capsule requires" is a configuration fact the operator can fix, and it is a
                # different statement from "the oracle is missing". The materializer records the dropped
                # declared tiers on the capsule precisely so this message can be specific.
                _unreach = capsule.get("unreachable_required_oracle_tiers") or []
                _ceil = capsule.get("oracle_tier_ceiling")
                failure = {"plane": "declared_tier_unreachable" if _unreach else "oracle_unavailable",
                           "category": "NOT_RUN_IS_NOT_PASS",
                           "unreachable_required_oracle_tiers": sorted(_unreach) or None,
                           "detail": (
                               f"this capsule declares required oracle tier(s) {sorted(_unreach)} that "
                               f"this phase cannot reach (ceiling {_ceil}); the tiers it can reach "
                               f"({sorted(required)}) are not applicable to this capsule's datapath, so "
                               f"nothing certified it. Refusing to substitute a tier the capsule never "
                               f"declared." if _unreach else
                               f"no runnable required oracle tier certified this capsule "
                               f"(required={sorted(required)}) — refusing to pass on the L0/L1 "
                               f"command-buffer interpretation alone")}

    # STAMP THE SUBMISSION ON EVERY TIER RECORD before it is serialised. Stated here, once, rather
    # than left for a reader to infer from the file's path: cycles are a property of the submission,
    # and a path heuristic that mis-keys two submissions of one capsule pools an 8.2x spread into a
    # single "latency". A record that already carries one keeps it.
    if submission:
        for _r in tiers.values():
            if getattr(_r, "submission", None) is None:
                _r.submission = dict(submission)
    result = {
        "capsule": name, "kind": capsule.get("kind"), "label": capsule.get("label"),
        "status": status, "contract_version": CONTRACT_VERSION,
        "tiers": {t: r.to_dict() for t, r in tiers.items()},
        "trace_check": trace_check_res, "numeric": numeric,
        "failure": failure, "toolchain_shas": toolchain_shas(eff_target),
    }
    # Advisory RTL-executability smoke (never a gate): record it as its own field when one ran, so a
    # reader sees the RTL-legality backstop verdict without it ever touching the pass/fail status.
    if executability:
        result["executability"] = executability
    # The refusal rides the result BY NAME AND SHAPE, so the round feedback can quote what was declined
    # rather than reporting a numeric mismatch on a program that was never emitted.
    if declined:
        result["declined"] = declined
    # Caller-supplied evidence (the whole-model path attaches its routing plan and mesh counters).
    # Merged AFTER status and failure are decided, and with setdefault so it can never overwrite an
    # authoritative field: a routing plan is something a reader interprets, never an input to the
    # verdict it is reported alongside.
    for _k, _v in (extra or {}).items():
        result.setdefault(_k, _v)
    (paths.run_path / "capsule_result.json").write_text(json.dumps(result, indent=2),
                                                        encoding="utf-8")
    _write_run_manifest(paths, run_id, name, status, tiers, capsule, target=cfg.target, suite=cfg.suite)
    try:
        schemas.validate(result, "capsule_result", contract=contract)
    except schemas.ContractViolation as e:
        import sys
        sys.stderr.write(f"WARNING: capsule_result self-validation failed: {e}\n")
    return result


def run_capsule(capsule: dict, package_dir: str | Path, *, runs_root: str | Path,
                run_id: str | None = None, contract: str | Path | None = None,
                oracle_adapters: dict[str, Callable] | None = None,
                pkg: Package | None = None, timeout: int = 600,
                target: str | None = None, suite: str | None = None, dtype: str = "i8xi8_i32",
                config=None, perf_extractor: Callable | None = None,
                no_oracle: bool = False, workers: int | None = None) -> dict:
    """Run one capsule through the package; write artifacts; return a capsule_result dict.

    ``config`` (a :class:`runner_config.RunnerConfig`) supplies the per-target grading knobs — the
    4th-artifact name, the sim-tier map + RTL tiers + loop order, the optional trace gate, and the perf
    fields — so ONE runner serves every target. When absent, the implicit gemmini/systolic config is
    built from ``target``/``suite``/``dtype`` (byte-identical to the pre-collapse behavior). Output
    equality uses the capsule's ``numeric_policy`` (exact for integer, tolerance for float). ``perf_extractor``
    (cb -> flops) feeds the SIMT gflops/pct_fp_peak. ``oracle_adapters`` is the per-target oracle set: the
    L0/L1 math floor always runs; RTL tiers grade only if an adapter is present + available (arc or a
    bespoke sim), else honestly ``unavailable`` — arc is never assumed.

    ``workers`` is the fan-out the CALLER is running this capsule inside; it is recorded on every
    measured tier (:func:`concurrency_stamp`). Cycle counts do not move with it, wall times move by
    up to 6.3x, so a ``timing`` block without it cannot be compared with any other run's. ``None`` (a
    direct call) records "not stated" rather than assuming serial -- this host is shared."""
    from ..runtime.reference import reference_outputs
    from ..runtime.simulator import simulate
    from .provenance import toolchain_shas

    # The effective target comes from the config (authoritative — cfg.target drives the run) when one is
    # supplied, else the explicit ``target`` argument. If NEITHER is given we refuse to run rather than
    # silently defaulting to gemmini (the OV2 rule: no core path silently operates on gemmini).
    eff_target = config.target if config is not None else target
    if eff_target is None:
        raise ValueError("run_capsule requires a target (or a config carrying one); "
                         "no default target is assumed")
    cfg = config or _config_for_target(eff_target, suite, dtype)
    # L1/oracle output equality uses the capsule's numeric_policy (integer -> exact). A float/SIMT target
    # ALSO declares a force_match_policy (a global float tolerance) so an integer-policy capsule is still
    # graded with tolerance — but the global force must NOT DISCARD the capsule's OWN declared tolerance:
    # a bf16 / low-precision capsule legitimately declares a looser atol/rtol than the global f32 default,
    # and letting the tight global override it makes a CONFORMANT kernel unpassable (RP3). Merge: keep the
    # force policy's compare MODE, take the LOOSER (max) atol/rtol per field.
    policy = _merge_match_policy(cfg.force_match_policy, capsule.get("numeric_policy"))
    name = capsule["name"]
    run_id = run_id or f"{name}"
    # An unrouted grade (oracle_adapters=None) resolves to the TARGET'S OWN endpoint oracle from the
    # contract — never the gemmini-hardcoded default_adapters (which silently mis-graded atlas as a
    # torch-mlir lowering, run_lowering.py, and crashed). `{}` stays honest no-oracle (L0/L1/trace only).
    adapters = oracle_adapters if oracle_adapters is not None else _resolve_oracle_adapters(eff_target)
    required = set(capsule.get("required_oracle_tiers", []))

    paths = make_run_paths(runs_root, run_id, suite=cfg.suite, target=cfg.target,
                           dtype=cfg.dtype, benchmark=name)

    # Whole-model capsule: graded end to end by compiling the captured model through the merlin
    # whole-model flow (compile_rvv) and gating vs its golden — NOT the per-op tier ladder. Write the
    # same capsule_result.json shape so downstream reporting is uniform.
    if capsule.get("kind") == "model":
        # LEAVE A TRACE BEFORE THE LONG PART. A whole-model grade can legitimately run for hours, and
        # with nothing written until it finishes an operator cannot tell a working run from a wedged
        # one -- which is exactly how a 5h30m stall was read as progress. Written first, so the run
        # directory says what started, when, and under what ceiling.
        _budget = model_budget_seconds()
        paths.run_path.mkdir(parents=True, exist_ok=True)
        (paths.run_path / "model_grade_started.json").write_text(json.dumps({
            "capsule": name, "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "model_budget_s": _budget, "step_timeout_s": timeout,
            "note": "whole-model capsule; absent capsule_result.json means it is still running or "
                    "was killed",
        }, indent=2), encoding="utf-8")
        result = _grade_model_capsule(capsule, target=eff_target, timeout=timeout,
                                      package_dir=package_dir, budget_s=_budget)
        (paths.run_path / "capsule_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Persist the ARR coverage certificate as its own durable artifact (not only inside
        # capsule_result.json) so the report/grader can read it back per compilation.
        cert = result.get("coverage_certificate")
        if cert is not None:
            paths.generated.mkdir(parents=True, exist_ok=True)
            (paths.generated / "coverage_certificate.json").write_text(
                json.dumps(cert, indent=2), encoding="utf-8")
        return result

    tiers: dict[str, TierResult] = {}
    trace_check_res = {"status": "skipped", "violations": []}
    decoded_trace: dict | None = None            # kept for the advisory divergence localizer (D2)
    numeric = {"status": "skipped"}
    executability: dict = {}                      # advisory RTL-executability smoke result(s), by tier
    failure: dict | None = None
    declined: dict | None = None                  # the backend's STATED refusal ({reason, shape, op})
    status = "pass"

    try:
        # shared front half: build + the 4 contract entrypoints (parse/target/cb/artifact), validated.
        pkg, cb, llvm_text = run_entrypoints(pkg, package_dir, capsule, paths, contract=contract,
                                             timeout=timeout, fourth_output_name=cfg.fourth_output_name)

        # --- golden + L0/L1 -----------------------------------------------------------------
        # The golden is the INDEPENDENT oracle's answer. For an integer capsule (gemmini / exact_int /
        # golden_source merlin_tensor_int) it is RECOMPUTED on the Tensor engine (byte-identical). For a
        # float capsule that ships an independent golden.yaml (atlas fp8-e4m3 -> bf16, golden_source
        # specir_refmodel_fp8_bf16), the integer engine cannot reproduce the float datapath, so the golden
        # is READ from golden.yaml — resolved by policy+source, never a target name.
        capsule_dir = capsule.get("__dir__")
        independent_float = CG.is_independent_float_golden(capsule, capsule_dir)
        gsource = CG.golden_source(capsule, capsule_dir)
        gold = CG.golden(capsule, capsule_dir)

        # A linalg-on-tensors reference lowering names its output positionally ("out"); the readback base +
        # golden compare are keyed by the capsule's DECLARED output name (the merlin_iface grammar already
        # names it via the commit op). Rename the sole output leaf to the declared name so preload, kernel,
        # readback, and golden all agree on one name+address. Scoped to linalg_positional cbs (no-op else).
        if cb.get("operand_naming") == "positional" or cb.get("interface") == "linalg_positional":
            _outnm = ((capsule.get("operation") or {}).get("attributes") or {}).get("out")
            _outs = [n for n, s in (cb.get("tensors") or {}).items() if s.get("role") == "output"]
            if _outnm and len(_outs) == 1 and _outs[0] != _outnm:
                _old = _outs[0]
                cb["tensors"][_outnm] = cb["tensors"].pop(_old)
                cb["outputs"] = [_outnm if o == _old else o for o in cb.get("outputs", [])]
                cb["arg_order"] = [_outnm if a == _old else a for a in cb.get("arg_order", [])]
                for _c in cb.get("commands", []):
                    for _k, _v in (_c.get("operands") or {}).items():
                        if _v == _old:
                            _c["operands"][_k] = _outnm

        # A program-oracle (self-hosted-ISA) target must run its emitted kernel on the SAME operands the
        # independent golden used. Attach the capsule's canonical leaf-input bytes (golden.yaml raws) to
        # the cb's leaf tensors so the program oracle preloads them by cb-declared base (AW5). Keyed by
        # name; a no-op for integer capsules (no recorded raws) and for adapters that ignore the field.
        _raws = CG.canonical_input_raws(capsule, capsule_dir)
        if _raws:
            import base64 as _b64
            for _tname, _tspec in (cb.get("tensors") or {}).items():
                if _tspec.get("role") in ("input", "weight", "bias") and _tname in _raws:
                    _tspec["preload_b64"] = _b64.b64encode(_raws[_tname]).decode()

        # Attach the golden's DECODED operand values so a self-hosted kernel harness (the fork-free SIMT
        # path) can embed them and run the emitted kernel on the SAME operands the independent golden used.
        # Generic + additive: empty for integer capsules, ignored by adapters that grade the command buffer.
        # A block-scaled MX capsule's operands (quantized codes + corpus-seeded E8M0 block scales) live only
        # in the golden and cannot be rebuilt from the decoded floats — attach them so the reference MX
        # kernel can bake them. Public-capsule reference path (masked for hidden); a no-op for non-MX goldens.
        _mxops = CG.mx_operands(capsule, capsule_dir)
        if _mxops:
            cb["mx_operands"] = _mxops

        _vals = CG.canonical_input_values(capsule, capsule_dir)
        if _vals:
            cb["canonical_inputs"] = _vals
            # POSITIONAL FALLBACK for interface grammars whose operands are UNNAMED. The merlin_iface
            # grammar names each leaf (``name = "X"``) so the by-name attach above matches; the
            # linalg-on-tensors grammar has positional ``func.func @forward`` args (``%0``, ``%1`` …), so a
            # lowering names its leaves positionally and NONE match the canonical keys (the capsule's
            # arg_order). When no canonical key is a cb tensor name, re-key by POSITION: the capsule's
            # arg_order and the cb's non-output leaves are both in @forward-argument order, so zip them.
            _tensors = cb.get("tensors") or {}
            if not (set(_vals) & set(_tensors)):
                _leaves = [n for n, s in _tensors.items()
                           if s.get("role") in ("input", "weight", "bias")]
                _ordered = list(_vals.values())
                if len(_leaves) == len(_ordered):
                    cb["canonical_inputs"] = dict(zip(_leaves, _ordered))
                    if _raws:
                        import base64 as _b64
                        for _leaf, _rb in zip(_leaves, list(_raws.values())):
                            _tensors[_leaf]["preload_b64"] = _b64.b64encode(_rb).decode()

        # Stamp the HARNESS-owned canonical DRAM layout onto every cb tensor (inputs+output), matched by
        # name. The agent's kernel was told these exact addresses (see the emit contract), so preload,
        # kernel, and readback all agree on one address map — the harness never trusts the agent to have
        # invented a base (the command_buffer schema forbids declaring one), and the output tensor now
        # always has a base to read back from. Target-agnostic: pure shape x dtype (see capsule_dram).
        from . import capsule_dram as _dram
        from .dram_facts import dram_base_for
        # Place any harness-assigned canonical base INSIDE the target's DRAM region [dram_base, +size):
        # a self-hosted-ISA target maps DRAM at a nonzero region base (derived from its facts; 0 for a
        # 0-based target like gemmini, keeping the base at DEFAULT_BASE unchanged), and the L2 oracle
        # relocates by that same base — so a submission that omits a base still grades against a layout
        # the model can index. Bases the agent DECLARED are left untouched (inject_bases only fills gaps).
        _dram.inject_bases(cb, capsule, base=dram_base_for(eff_target) + _dram.DEFAULT_BASE)

        if independent_float:
            # The integer reference/simulate engines cannot execute a float (fp8/bf16) datapath, so the
            # integer L0-reference / L1-sim numeric floor is INAPPLICABLE — skipped honestly (not failed),
            # exactly as a non-RoCC target drops the RoCC trace gate. The real numeric grade rides the RTL
            # program-oracle output vs the independent golden.yaml (tolerance_float), scored in the oracle
            # loop below. These skips are marked not_applicable so not_run_is_not_pass does NOT flip them to
            # incomplete (only an unavailable/absent RTL oracle does).
            ref = sim = None
            numeric = {"status": "skipped", "policy": policy.get("compare"), "golden_source": gsource,
                       "note": "integer reference/simulate N/A for a float datapath; correctness is graded "
                               "by running your artifact on the RTL oracle and checking it computes the "
                               "declared operation within tolerance"}
            tiers["L0"] = TierResult(
                "L0", "skipped", mandatory="L0" in required, not_applicable=True,
                reason="integer reference not applicable to float datapath; graded vs independent "
                       f"golden ({gsource}) at the RTL oracle")
            tiers["L1"] = TierResult(
                "L1", "skipped", mandatory="L1" in required, not_applicable=True,
                reason="integer simulate not applicable to float datapath")
        else:
            # Interpreting the AGENT's command buffer (reference/simulate) can fail if the cb is
            # structurally invalid for the reference interpreter's op/shape model. That is the agent's
            # bug, NOT a harness crash: report it as a gradeable command_buffer failure (so the agent gets
            # feedback and both arms are scored identically) instead of a RUNNER_CRASH. The reason reports
            # the ACTUAL exception rather than asserting one hardcoded cause — the interpreter models a
            # fixed op vocabulary, so an unmodeled op/operand-shape/name is one of several possible causes.
            try:
                ref = reference_outputs(cb)
                sim = simulate(cb)["outputs"]
            except (ValueError, KeyError, IndexError, TypeError) as ce:
                raise CertFailure(
                    "command_buffer", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                    f"command buffer could not be interpreted by reference/simulate "
                    f"({type(ce).__name__}: {ce}); check operand ranks/shapes and that each command's op "
                    f"is one the reference interpreter models (e.g. a windowed op lowered to a 2D matmul)"
                ) from ce
            nrep = CG.compare(gold, ref, capsule["numeric_policy"], golden_source=gsource)
            numeric = {"status": nrep["status"], "policy": nrep["policy"],
                       "max_abs_diff": nrep["max_abs_error"], "max_rel_error": nrep["max_rel_error"],
                       "mismatch_count": nrep["mismatch_count"], "first_mismatch": nrep["first_mismatch"],
                       # same DROPPED-store surfacing as the float path (see _absent_outputs): a missing
                       # declared output is reported distinctly, never as a phantom 0-magnitude mismatch.
                       "per_output": nrep.get("per_output", {}),
                       "missing_outputs": _absent_outputs(nrep)}
            CG.write_numeric_report(paths.generated / "numeric_report.yaml", nrep)
            tiers["L0"] = TierResult("L0", "pass" if nrep["status"] == "pass" else "fail",
                                     mandatory="L0" in required or True,
                                     reason=(None if nrep["status"] == "pass"
                                             else "your command buffer does not compute the declared operation"),
                                     evidence="numeric_report.yaml")
            if nrep["status"] != "pass":
                _absent_cb = (_absent_output_detail(nrep, "command_buffer", gold, ref)
                              or _unwritten_output_detail(nrep, "command_buffer"))
                raise CertFailure("numeric_golden", _cat("FUNCTIONAL_MISMATCH"),
                                  _absent_cb or ("your command buffer does not compute the declared operation "
                                  f"(first divergence at index={(nrep['first_mismatch'] or {}).get('index')})"))

            l1_ok = _match_by_policy(ref, sim, policy)
            tiers["L1"] = TierResult("L1", "pass" if l1_ok else "fail", mandatory=True,
                                     reason=None if l1_ok else "reference(cb) != simulate(cb)")
            if not l1_ok:
                raise CertFailure("command_buffer_reference", _cat("FUNCTIONAL_MISMATCH"),
                                  "reference(cb) != simulate(cb)")

        # --- trace gate (optional; only for targets whose ISA has a decoder+checker plugin) --------
        # A command-ISA target (systolic, trace_gate="rocc_insn") decodes the emitted .insn stream to a
        # RoCC trace and checks instruction coverage. A SIMT/other target (trace_gate=None) has no analog
        # and skips it — the L0/L1 math floor + the oracle tiers carry the verdict either way.
        if cfg.trace_gate == "rocc_insn":
            trace = RD.decode_text(llvm_text, source=str(paths.generated / cfg.fourth_output_name),
                                   target=eff_target)
            decoded_trace = trace                # hand to the advisory localizer on an oracle divergence
            # Validating OUR OWN decoder's output must never crash the capsule before the oracle runs: a
            # schema hiccup (e.g. the manifest maps a funct to a class the enum lacks) is our tooling's
            # limitation, not the backend's defect — record it and continue, not a pre-oracle RUNNER_CRASH.
            try:
                schemas.validate(trace, "instruction_trace", contract=contract)
            except schemas.ContractViolation as _sv:
                import sys as _sys
                _sys.stderr.write(f"WARNING: instruction_trace self-validation failed (advisory): {_sv}\n")
            (paths.generated / "instruction_trace.json").write_text(
                json.dumps(trace, indent=2), encoding="utf-8")
            # Advisory silicon-liveness screen (HW-agnostic): would this emitted stream stall/fault on real
            # silicon? It uses only the target's OWN derived facts (scratchpad/accumulator capacity, address
            # map, ISA legality) + this decoded trace, never a target literal, and — like trace_check — is
            # advisory: it writes liveness_report.json for the author and never gates or crashes the capsule.
            try:
                from merlin.liveness import Program as _LProg, assess as _lassess
                _lname = capsule.get("id") or capsule.get("name") or "kernel"
                _lrep = _lassess(_LProg(name=_lname, trace=trace, address_model="pointer_args"), eff_target)
                (paths.generated / "liveness_report.json").write_text(
                    json.dumps(_lrep.to_dict(), indent=2), encoding="utf-8")
            except Exception as _le:  # noqa: BLE001 — advisory: our screener's limit is not the backend's defect
                import sys as _sys
                _sys.stderr.write(f"WARNING: liveness screen failed (advisory): {_le}\n")
            # trace_check is ADVISORY: instruction-class coverage / ordering / UNKNOWN are diagnostics
            # for the author, NOT the verdict. Correctness is decided by the numeric + L2/L3 RTL oracle
            # below (which EXECUTE the actual emitted stream), and the hidden golden precludes faking an
            # answer — so an instruction our decoder cannot yet classify must never fail a conformant
            # backend (the old gate short-circuited before the RTL oracle even ran).
            # The RoCC/inline-asm bare-metal harness passes each operand buffer as a POINTER argument, so
            # its address model is pointer_args: DRAM addresses must derive from the kernel's args, not be
            # baked. (A fixed-preload oracle — e.g. the program oracle — would pass "fixed_preload"; the
            # model is the harness's, derivable per oracle, not a target literal.)
            trace_check_res = TCK.check(trace, capsule.get("expected", {}), cb=cb,
                                        address_model="pointer_args")
            # Anti-cheese floor (must drive the accelerator) is the ONE trace-derived gate — but ONLY where
            # the RTL oracle will NOT run to prove it. When a required oracle tier executes the emitted
            # stream on real hardware, THAT is the spelling-independent anti-cheese (a host-computed fake
            # produces wrong output on RTL); the decoder-shaped floor only recognizes the `.insn r` form, so
            # gating on it here would false-fail a conformant `.word`-encoded kernel before the oracle.
            if (no_oracle or not required) and not TCK.drives_accelerator(trace):
                raise CertFailure("trace_check", _cat("PROTOCOL_VIOLATION"),
                                  "no accelerator instructions emitted — the kernel did not drive the "
                                  "target's ISA (compute must happen on the device; correctness cannot "
                                  "come from the host).")

        # --- oracle tiers -------------------------------------------------------------------
        # Run every tier the config declares (tier_sim ladder) OR an injected adapter provides — so a
        # target whose RTL tier is supplied by an adapter rather than a static tier_sim (atlas: arc L3,
        # empty tier_sim) still runs, while gemmini's declared ladder is unchanged. The L0/L1 math floor
        # is handled above (reference/simulate), NOT here. Sorted for a stable L2<..<L5 ladder order.
        # program_oracle (the external_backend adapters) raises its OWN OracleUnavailable, a RuntimeError
        # that is a DIFFERENT class from this module's OracleUnavailable(Exception) — unrelated by mro. If
        # we catch only the local one, an honest "oracle unavailable" from the program oracle (missing
        # assembler / model venv / cosim import) falls through to the generic handler and is mislabeled a
        # TOOL_CRASH, so the unavailable->incomplete / not_gradeable_no_oracle path never fires for atlas.
        # Catch BOTH so unavailability is honest for every endpoint.
        from .program_oracle import OracleUnavailable as _POUnavailable
        # An endpoint that RAN the program and watched it never halt reports a VERDICT, not an absent
        # oracle. It is raised as a SUBCLASS of the unavailable exception (so every fail-closed handler
        # still works) and must therefore be caught FIRST, or it collapses back into "the oracle did not
        # run" — which is what left a whole agent run with nothing it could act on.
        from .program_oracle import ProgramDidNotHalt as _PODidNotHalt
        # The muon (SIMT/cyclotron/VCS) adapters raise their OWN MuonUnavailable — a THIRD unrelated class.
        # The VCS RTL difftest is honest-unavailable (WIP upstream: the DPI ELF path is not wired for arbitrary
        # corpus kernels), so it raises MuonUnavailable; without catching it here that honest unavailability
        # falls through to the generic handler and is mislabeled a TOOL_CRASH (an agent FAIL) instead of
        # firing the unavailable -> not_gradeable_no_oracle path — the same bug the program-oracle catch above
        # fixed for atlas. Catch it too so an unavailable RTL oracle is honest for the SIMT endpoint as well.
        # The muon backend was evicted to its own reference package; resolve its MuonUnavailable via the
        # registry (get_backend), and simply skip it if the SIMT backend is not present in this env — an
        # absent backend cannot raise it, so there is nothing to catch.
        try:
            from ..runtime.backends import base as _bk
            _muon_unavail = (_bk.get_backend("muon").MuonUnavailable,)
        except Exception:  # noqa: BLE001 — SIMT backend absent -> no MuonUnavailable to catch
            _muon_unavail = ()
        _ORACLE_UNAVAILABLE = (OracleUnavailable, _POUnavailable) + _muon_unavail
        # Tiers the CAPSULE declares cannot corroborate a result for this target (see
        # corpus_spec._inapplicable_tiers). A tier whose model disagrees with the RTL about the machine
        # itself — not about precision — grades a correct kernel as wrong on every capsule, so running it
        # produces a permanent red the agent can neither fix nor learn from. It is reported the way L0/L1
        # already are for a float datapath: skipped + not_applicable, carrying the declared reason. It is
        # never reported as a pass, and a REQUIRED tier can never be declared inapplicable (that raises at
        # generation time), so this cannot be used to switch off a failing mandatory oracle.
        _inapplicable = capsule.get("inapplicable_oracle_tiers") or {}
        # CHEAPEST MEASURED TIER FIRST, not lexicographic. The ladder aborts on the first mandatory
        # tier that refutes a capsule, so the ORDER decides what a failing capsule costs -- and sorting
        # by name meant one target paid its arc cosim before reaching the cheaper Verilator tier
        # (⚠️ the 24.5 s / 0.29 s pair quoted here and below is a THROUGHPUT figure measured under 16-way
        # parallelism, not a latency. Measured serially over 42 samples 2026-08-29: arc median 3.68 s,
        # Verilator median 0.276 s, both linear in halt cycles (3.63 vs 0.255 ms/cyc) with no build step.
        # The ordering conclusion is unchanged -- 0.276 < 3.68 either way -- but anything PRICED off
        # 24.5 s is wrong by 6.3x. See docs/design/performance_budget_unit.md.)
        # that refutes the same capsules (measured: 12 of 12, identical signature). See tier_policy.
        _tier_seq = _tier_policy.tier_order(str(cfg.target or target or ""),
                                            set(cfg.oracle_tiers) | set(adapters or {}))
        _screen_tier = _tier_seq[0] if _tier_seq else None
        # When set, a mandatory tier failure does not abort the ladder: the remaining tiers still run and
        # record, and the FIRST failure is raised once the loop completes. Costs the later tiers on a
        # failing capsule, which is the price of being able to say what every tier thought.
        _complete_ladder = _full_ladder_enabled()
        _first_cert_failure: "CertFailure | None" = None
        for tier in _tier_seq:
            mand = tier in required
            # THE SCREEN IS ALWAYS PAID FOR; the tiers above it are what a budget can decline. A capsule
            # inside the derived covering set is never declined -- that is what the cover is for.
            if tier != _screen_tier:
                _may, _why = _tier_policy.may_certify(str(cfg.target or target or ""), capsule)
                if not _may:
                    tiers[tier] = TierResult(tier, "skipped", mand, reason=_why,
                                             budget_deferred=True,
                                             derived_from_rtl=tier in cfg.rtl_tiers)
                    continue
            if tier in _inapplicable and not mand:
                tiers[tier] = TierResult(tier, "skipped", mand, not_applicable=True,
                                         reason=str(_inapplicable[tier]),
                                         derived_from_rtl=tier in cfg.rtl_tiers)
                continue
            adapter = (adapters or {}).get(tier)
            if adapter is None:
                if mand:
                    # FAIL CLOSED on the DECLARED tier, and name it. Never substitute another tier and
                    # report its verdict as the declared one — a grade that silently rides a tier the
                    # capsule never asked for is not the grade the capsule specified.
                    tiers[tier] = TierResult(
                        tier, "unavailable", True,
                        reason=(f"this capsule declares required oracle tier {tier} "
                                f"({cfg.tier_sim.get(tier, 'no sim declared')}) but this phase supplies "
                                f"no adapter for it; reachable tiers here: "
                                f"{sorted(adapters or {}) or 'none'}"),
                        derived_from_rtl=tier in cfg.rtl_tiers)
                continue
            import time as _time
            _adapter_t0 = _time.perf_counter()
            try:
                res = adapter(cb, llvm_text, paths.generated, timeout)
            except _PODidNotHalt as e:
                # The oracle RAN and returned a verdict: the program never halted. That is the AGENT's
                # bug, not a missing oracle — record it as a tier FAIL (not "unavailable") so the
                # not_run_is_not_pass path never relabels it, and fail the capsule on the named
                # did-not-halt plane. MUST precede the _ORACLE_UNAVAILABLE clause below: this is a
                # subclass of it, and being caught there is exactly what hid the diagnosis.
                tiers[tier] = TierResult(tier, "fail", mand, reason=_did_not_halt_reason(str(e)),
                                         derived_from_rtl=tier in cfg.rtl_tiers)
                if mand:
                    raise _did_not_halt_failure(str(e)) from e
                continue
            except _ORACLE_UNAVAILABLE as e:
                tiers[tier] = TierResult(tier, "unavailable", mand, reason=str(e),
                                         derived_from_rtl=tier in cfg.rtl_tiers)
                continue
            except Exception as e:  # adapter raised — classify: a non-terminating program is the AGENT's
                # bug (it ran to the cycle cap), a TIMEOUT fail, NOT a tool_crash. Mislabeling it
                # 'tool_crash' read as an infra problem the agent couldn't fix — so it never emitted the
                # ISA's halt instruction and every round failed identically (the atlas flat-0/11 wall).
                _sim = cfg.tier_sim.get(tier, tier)
                _msg = str(e)
                _did_not_halt = "did not halt" in _msg
                # A program that RAN on the sim but FAULTED at runtime (the bare-metal fail/trap banner
                # fired: "*** FAILED ***") is the AGENT's kernel bug — an illegal instruction or an
                # out-of-range memory access (e.g. an MVIN/MVOUT DRAM address of 0, or a baked address that
                # ignores the passed pointers / declared DRAM layout) — NOT an infra tool_crash. Classify it
                # as a functional fault with actionable feedback; mislabeling it tool_crash reads as an
                # unfixable infra problem and wastes every round (the atlas flat-0/11 pattern).
                _trapped = ("*** FAILED ***" in _msg) and not _did_not_halt
                tiers[tier] = TierResult(
                    tier, "fail", mand,
                    reason=(_did_not_halt_reason(_msg) if _did_not_halt
                            else f"kernel faulted at runtime: {_clip(_msg, 260)}" if _trapped
                            else f"{_sim} crash: {_clip(_msg, 300)}"),
                    derived_from_rtl=tier in cfg.rtl_tiers)
                if mand:
                    if _did_not_halt:
                        raise _did_not_halt_failure(_msg) from e
                    if _trapped:
                        raise CertFailure(_sim, _cat("FUNCTIONAL_MISMATCH"),
                                          "the emitted kernel RAN but FAULTED on "
                                          f"{_sim} (a trap — an illegal instruction or an out-of-range "
                                          "memory access — before producing output). Check that every "
                                          "memory-movement instruction uses a valid, in-range DRAM address "
                                          "(derive it from the passed pointer args / the declared DRAM "
                                          f"layout, never a baked 0 or a guessed address): {_clip(_msg, 300)}") from e
                    raise CertFailure(_sim, _cat("TOOL_CRASH"),
                                      f"{_sim} invocation failed: {_clip(_msg, 400)}") from e
                continue
            _adapter_wall = _time.perf_counter() - _adapter_t0
            # Split active (build + sim) vs waiting (queue/FPGA slot). Adapters that route through a
            # queue set oracle_wait_s themselves; for the rest, any wall not accounted by build+sim is
            # treated as wait (≈0 for local spike/verilator).
            _tm = dict(res.get("timing") or {})
            _tm.setdefault("build_s", None)
            _tm.setdefault("sim_active_s", None)
            if _tm.get("oracle_wait_s") in (None, 0.0):
                _acct = (_tm.get("build_s") or 0.0) + (_tm.get("sim_active_s") or 0.0)
                _tm["oracle_wait_s"] = round(max(0.0, _adapter_wall - _acct), 3)
            _tm["adapter_wall_s"] = round(_adapter_wall, 3)
            # STAMPED HERE, at the moment of measurement -- not at suite start. The load average this
            # reads is the one covering the adapter call whose wall time sits beside it.
            _conc = concurrency_stamp(workers)
            _tier_policy.record_cost(str(cfg.target or target or ""), tier, _adapter_wall)
            if tier != _screen_tier:
                _tier_policy.note_spend(str(cfg.target or target or ""), _adapter_wall)
            if res.get("executability") is not None:
                # ADVISORY executability smoke: a bounded-cycle RTL-legality check (does a cyclotron(L2)-
                # passing ELF at least RUN on the real Verilator RTL — boots, no trap, MX PE accepts a
                # command), the RTL-grounding backstop for the non-RTL-certified perf oracle. It does NOT
                # numeric-grade and NEVER blocks: recorded as a NON-mandatory tier + an ``executability``
                # field, and — because it is never in ``required`` — it is not_run_is_not_pass-exempt by
                # construction. A slow/flaky RTL smoke must not fail a capsule whose L2 grade passed, so
                # its verdict (legal/illegal) is informational only; ``mand`` is ignored here on purpose.
                ex = res["executability"]
                executability[tier] = ex
                _sim_name = cfg.tier_sim.get(tier) or tier
                if res.get("console") is not None:
                    (paths.artifacts_dir / f"{_sim_name}_console.log").write_text(
                        res["console"], encoding="utf-8")
                tiers[tier] = TierResult(
                    tier, "pass" if ex.get("legal") else "fail", mandatory=False,
                    reason=ex.get("reason"), cycles=ex.get("cycles"), derived_from_rtl=True,
                    evidence=f"{_sim_name}_console.log", timing=_tm, not_applicable=True,
                    concurrency=_conc)
                continue
            if res.get("completion_only"):
                # An RTL cert that ran the emitted kernel to completion but cannot surface its outputs
                # for an independent numeric check (e.g. the Muon Verilator harness $finish-races the
                # UART flush). It certifies RTL COMPLETION + cycle-accurate cycles; CORRECTNESS is the
                # mandatory functional tier's job. It is NEVER allowed to stand in for a mandatory tier
                # (a required tier must verify output) — there it degrades to honest-unavailable.
                _sim_name = cfg.tier_sim.get(tier) or tier
                if mand:
                    tiers[tier] = TierResult(
                        tier, "unavailable", mand,
                        reason="RTL cert ran to completion but cannot surface outputs for a mandatory "
                               "correctness check (use the functional tier as the required gate)",
                        cycles=res.get("cycles"), derived_from_rtl=tier in cfg.rtl_tiers, timing=_tm,
                        concurrency=_conc)
                    continue
                _cg = _cp = None
                _cu: dict | None = None
                if perf_extractor is not None:
                    _cperf = perf_extractor(cb, res) or {}
                    _cg, _cp = _cperf.get("gflops"), _cperf.get("pct_fp_peak")
                    _cu = {k: v for k, v in _cperf.items()
                           if k not in ("gflops", "pct_fp_peak", "flops")} or None
                if res.get("console") is not None:
                    (paths.artifacts_dir / f"{_sim_name}_console.log").write_text(
                        res["console"], encoding="utf-8")
                tiers[tier] = TierResult(
                    tier, "pass", mand,
                    reason="RTL completion + cycle-accurate perf cert (correctness gated by the "
                           "required functional tier)",
                    cycles=res.get("cycles"), derived_from_rtl=tier in cfg.rtl_tiers,
                    cycle_accurate=tier in cfg.rtl_tiers, evidence=f"{_sim_name}_console.log",
                    timing=_tm, gflops=_cg, pct_fp_peak=_cp, utilization=_cu,
                    timing_observations=res.get("timing_observations"),
                    counters=res.get("counters"),
                    timing_capability=res.get("timing_capability"), concurrency=_conc)
                continue
            if independent_float:
                # Float grade: the RTL program-oracle output vs the INDEPENDENT golden.yaml (tolerance_float).
                # There is no integer reference/simulate to cross-check against — this comparison IS the
                # numeric verdict, recorded as the honest numeric report + evidence.
                onrep = CG.compare(gold, res["outputs"], policy, golden_source=gsource)
                okt = onrep["status"] == "pass"
                # The numeric verdict must ride the MANDATORY/gold tier, not an additive one: otherwise an
                # additive tier later in the ladder (e.g. an approximate or unbuilt model) would overwrite
                # the authoritative numeric. Record from a mandatory tier, or from the first tier to run if
                # none has set it yet.
                if mand or numeric.get("status") == "skipped":
                    numeric = {"status": onrep["status"], "policy": onrep["policy"],
                               "golden_source": gsource, "max_abs_diff": onrep["max_abs_error"],
                               "max_rel_error": onrep["max_rel_error"],
                               "mismatch_count": onrep["mismatch_count"],
                               "first_mismatch": onrep["first_mismatch"],
                               # carry the per-output breakdown + the DROPPED-store list through to
                               # capsule_result.json so the agent (and the self-check, which reads this
                               # dict) sees WHICH output failed and WHY — a missing store otherwise reads
                               # as mismatch_count>0 with max_abs_error==0, a self-contradictory signal.
                               "per_output": onrep.get("per_output", {}),
                               "missing_outputs": _absent_outputs(onrep)}
                    CG.write_numeric_report(paths.generated / "numeric_report.yaml", onrep)
            else:
                okt = _match_by_policy(res["outputs"], gold, policy) \
                    and _match_by_policy(res["outputs"], ref, policy) \
                    and _match_by_policy(res["outputs"], sim, policy)
            # tier_sim is a display LABEL; a tier supplied by an adapter (atlas arc L3) may have no static
            # label, so fall back to the adapter's oracle-provenance string, else the tier name.
            sim_name = cfg.tier_sim.get(tier) or (
                res.get("oracle") if isinstance(res.get("oracle"), str) else None) or tier
            # SIMT perf headline (gflops / % of peak) when a perf extractor is supplied; None (systolic)
            # leaves the fields off the TierResult so the output stays byte-identical.
            _gflops = _pct_peak = None
            _util: dict | None = None
            if perf_extractor is not None:
                _perf = perf_extractor(cb, res) or {}
                _gflops = _perf.get("gflops")
                _pct_peak = _perf.get("pct_fp_peak")
                # Everything the extractor returned beyond the two headline numbers is utilization
                # detail. Taken as the remainder rather than a fixed list so a target can report the
                # counters ITS oracle actually has, without this shared runner naming any of them.
                _util = {k: v for k, v in _perf.items()
                         if k not in ("gflops", "pct_fp_peak", "flops")} or None
            # Engineer framing (no "golden"): the emitted artifact, run on the RTL, does not compute the
            # declared operation. There is no answer key handed to the agent — the reference is the op's
            # own definition, which the agent can reproduce from the declared inputs. The appended
            # _encoding_divergence_hint adds that the cheap tiers agreed, so the defect is in the encoding.
            # A DROPPED declared output (kernel never wrote it) gets a precise, distinct detail so the
            # agent isn't left with the baffling "N mismatches, 0 error" of a store that never fired.
            # A store that never landed is named precisely, in BOTH of its shapes: the output missing
            # from the readback entirely (_absent_output_detail) and the output read back as a single
            # untouched constant (_unwritten_output_detail). Without the second, a writeback failure is
            # reported as a numeric mismatch whose count cannot move -- measured at six wasted rounds.
            _absent_detail = ((_absent_output_detail(onrep, sim_name, gold, res["outputs"])
                               or _unwritten_output_detail(onrep, sim_name))
                              if independent_float else None)
            _mismatch_reason = _absent_detail or (
                f"on {sim_name}, your emitted artifact does not compute the declared operation within tolerance"
                if independent_float
                else f"on {sim_name}, your emitted artifact does not compute the declared operation")
            # ``oracle`` may be a rich dict ({kind, derived_from_rtl, fidelity}) OR a plain provenance
            # string; default to the tier's RTL classification only when it doesn't declare
            # derived_from_rtl. THE ORACLE'S OWN WORD OUTRANKS THE TIER NAME: an RTL-DERIVED model that
            # happens to land on the tier named L3 is not RTL certification, and classifying by name
            # credited it as such on every target whose L3 is not Verilator.
            _oracle_meta = res.get("oracle")
            _derived_from_rtl = (_oracle_meta.get("derived_from_rtl", tier in cfg.rtl_tiers)
                                 if isinstance(_oracle_meta, dict) else (tier in cfg.rtl_tiers))
            _fidelity = _oracle_meta.get("fidelity") if isinstance(_oracle_meta, dict) else None
            # NAME THE EVIDENCE AFTER THE ORACLE THAT PRODUCED IT. ``sim_name`` comes from the contract's
            # static ``tier_sim`` map, which cannot know that a faster RTL-derived engine replaced the
            # declared one at runtime (the muon backend swaps GSIM in for Verilator whenever its emu is
            # configured). The console was then written to ``verilator_console.log`` by an engine that is
            # not Verilator -- a certification filed under the wrong tool's name, which is the same defect
            # as reporting a score without the tier that produced it. Same rule as ``derived_from_rtl``
            # above: the oracle's own word outranks the declared name.
            _ev_name = _oracle_kind(_oracle_meta) or sim_name
            tiers[tier] = TierResult(
                tier, "pass" if okt else "fail", mand,
                reason=None if okt else _mismatch_reason,
                cycles=res.get("cycles"), derived_from_rtl=_derived_from_rtl,
                cycle_accurate=(tier in cfg.rtl_tiers and okt), evidence=f"{_ev_name}_console.log",
                timing=_tm, gflops=_gflops, pct_fp_peak=_pct_peak, utilization=_util,
                timing_observations=res.get("timing_observations"),
                counters=res.get("counters"),
                timing_capability=res.get("timing_capability"), fidelity=_fidelity,
                concurrency=_conc)
            if res.get("console") is not None:
                (paths.artifacts_dir / f"{_ev_name}_console.log").write_text(
                    res["console"], encoding="utf-8")
            # Only a MANDATORY/gold tier mismatch fails the capsule. An ADDITIVE lower-fidelity tier
            # (one not in required_oracle_tiers — e.g. a fast functional model with known approximation
            # gaps) records its fail in the tiers dict but must NOT abort: aborting here would pre-empt the
            # required RTL oracle that follows in the sorted ladder — the same "a cheaper check short-
            # circuits the authoritative oracle" class we fixed on the trace side, here on the oracle side.
            if not okt and mand:
                _cf = CertFailure(sim_name, _cat("FUNCTIONAL_MISMATCH"),
                                  _mismatch_reason + _encoding_divergence_hint(
                                      trace_check_res, independent_float,
                                      cb=cb, capsule=capsule, trace=decoded_trace))
                # COMPLETE THE LADDER, then fail. Raising here aborts the loop, so every tier ordered
                # AFTER the refuting one is left with no record at all -- not "skipped", absent. Measured
                # on atlas: 11 of 26 capsules carried an L4 fail and NO L3 entry, 1 the reverse, because
                # tier_order runs the cheaper Verilator tier before the arc cosim. A tier with no record
                # is not evidence (`not_run_is_not_pass`), and it makes "these 12 failed" unanswerable at
                # the other tier -- which is exactly what you need when the shared defect is fixed.
                # The capsule still fails, on the FIRST refuting plane; only the remaining tiers now run.
                if _first_cert_failure is None:
                    _first_cert_failure = _cf
                if not _complete_ladder:
                    # The opt-out still records: a tier the ladder declined to run says `skipped`,
                    # with the reason, never `fail` and never nothing at all.
                    for _later in _tier_seq[_tier_seq.index(tier) + 1:]:
                        if _later not in tiers:
                            tiers[_later] = suppressed_tier_result(
                                _later, _later in required, tier,
                                from_rtl=_later in cfg.rtl_tiers)
                    raise _cf

        if _first_cert_failure is not None:
            raise _first_cert_failure

    except BackendDeclined as bd:
        # A STATED REFUSAL IS NOT A WRONG ANSWER. It is still not a pass -- the capsule stays in the
        # denominator, uncertified -- but recording it as a numeric mismatch told the agent its
        # arithmetic was wrong about a program it had never emitted, and no feedback could say which
        # shapes it had declined.
        status = "declined"
        declined = bd.to_dict()
        failure = {"plane": "backend_declined", "category": "DECLINED", "detail": bd.reason}
    except CertFailure as cf:
        status = "fail"
        cat = cf.category.value if hasattr(cf.category, "value") else str(cf.category)
        failure = {"plane": cf.plane, "category": cat, "detail": cf.detail}
    except Exception as e:  # internal harness bug
        status = "error"
        failure = {"plane": "runner_internal", "category": "RUNNER_CRASH",
                   "detail": f"{type(e).__name__}: {e}",
                   "traceback": _traceback.format_exc()}

    return _finalize_capsule_result(
        submission=submission_identity(package_dir, run_id=run_id),
        name=name, capsule=capsule, status=status, failure=failure, tiers=tiers,
        trace_check_res=trace_check_res, numeric=numeric, required=required,
        no_oracle=no_oracle, eff_target=eff_target, paths=paths, run_id=run_id,
        cfg=cfg, contract=contract, executability=executability, declined=declined)


def _write_run_manifest(paths: RunPaths, run_id: str, name: str, status: str,
                        tiers: dict, capsule: dict, *, target: str,
                        suite: str | None = None) -> None:
    manifest = {
        "schema_version": "1.0", "project": "merlin", "suite": suite or f"{target}-capsule-bench",
        "method": run_id,
        "run_id": run_id, "target": target, "benchmark": name, "status": status,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "codegen_backend": "oot_package",
        "metadata": {
            "kind": capsule.get("kind"), "label": capsule.get("label"),
            "required_oracle_tiers": capsule.get("required_oracle_tiers", []),
            "tier_status": {t: r.status for t, r in tiers.items()},
            "cycles": {t: r.cycles for t, r in tiers.items() if r.cycles is not None},
        },
    }
    (paths.run_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")








def run_suite(capsules: list[dict], package_dir: str | Path, *, runs_root: str | Path,
              contract: str | Path | None = None,
              oracle_adapters: dict[str, Callable] | None = None,
              timeout: int = 600, max_workers: int = 1,
              target: str | None = None, suite: str | None = None, dtype: str = "i8xi8_i32",
              config=None, perf_extractor: Callable | None = None,
              no_oracle: bool = False) -> list[dict]:
    """Run many capsules through one package (building/integrity-scanning it once).

    ``max_workers > 1`` fans the (independent) per-capsule runs out across a ThreadPoolExecutor — each
    capsule has its own ``run_id``/workdir and builds its own ELF, and ``pkg`` is read-only after the
    one-time build, so concurrent simulator instances (verilator/VCS) don't collide. Mirrors
    :func:`heavy_oracles.run_vcs_parallel`. ``max_workers == 1`` preserves the original sequential order.
    """
    # RESOLVE THE RUNS ROOT HERE, while this is still single-threaded. Resolving it per-capsule is not
    # enough: some capsule paths enter a context that chdirs the process (mlc resolves its arc artifacts
    # relative to its own root), and a relative root resolved inside that window becomes an absolute path
    # under the WRONG tree. Measured: of 26 op capsules run with 8 workers, 18 wrote their entire run
    # directory into the mlc checkout. The suite still scored -- the results are returned in memory --
    # but the trace collection reads from this root, so those 18 capsules contributed no coverage, and a
    # sibling project's tree acquired 18 stray run trees.
    # Same treatment for every OTHER relative path this suite carries into a worker thread. The runs
    # root was only the first one to show: the package directory is re-opened per capsule, and inside a
    # chdir window it raises FileNotFoundError on a directory that is plainly there -- which the runner
    # records as a RUNNER_CRASH, i.e. as though the submission were broken. Measured: the same 18 of 26
    # capsules that misplaced their run dirs also "crashed" this way, and since an errored capsule is in
    # neither the pass nor the fail bucket, they silently left the gate denominator -- the whole-model
    # capstone then cleared its 0.8 gate on 7/8 instead of on 26 capsules.
    runs_root = str(Path(runs_root).resolve())
    package_dir = str(Path(package_dir).resolve())
    if contract is not None and Path(contract).exists():
        contract = str(Path(contract).resolve())
    pkg = load_package(package_dir, contract=contract)
    integrity_scan(pkg)
    build_package(pkg)

    # Pass evidence is opt-in through MERLIN_PASS_LOG. When enabled, install before any capsule work and
    # attribute in-process calls to the capsule that caused them. Package entrypoints remain subprocesses
    # and are certified by the mandatory manifest-command contract; this recorder covers only Merlin's
    # production whole-model boundary plane.
    _pass_recorder = None
    if os.environ.get("MERLIN_PASS_LOG"):
        from ..xdsl_dialects.lowering import passes as _pass_recorder
        _pass_recorder.install_pass_recorder()

    def _one(cap: dict, workers: int = 1) -> dict:
        # ``workers`` is what THIS capsule ran inside, not what the suite was configured for: the
        # calibration head is serial by construction, and recording the suite's max_workers for it
        # would describe contention its numbers never saw.
        if _pass_recorder is None:
            return run_capsule(cap, package_dir, runs_root=runs_root, run_id=cap["name"],
                               contract=contract, oracle_adapters=oracle_adapters,
                               pkg=pkg, timeout=timeout, target=target, suite=suite, dtype=dtype,
                               config=config, perf_extractor=perf_extractor, no_oracle=no_oracle,
                               workers=workers)
        with _pass_recorder.pass_run_context(
                str(cap["name"]), cap.get("pass_requirements") or ()):
            return run_capsule(cap, package_dir, runs_root=runs_root, run_id=cap["name"],
                               contract=contract, oracle_adapters=oracle_adapters,
                               pkg=pkg, timeout=timeout, target=target, suite=suite, dtype=dtype,
                               config=config, perf_extractor=perf_extractor, no_oracle=no_oracle,
                               workers=workers)

    def _run_all(caps: list[dict]) -> list[dict]:
        if max_workers <= 1 or not caps:
            return [_one(c) for c in caps]
        # CALIBRATE BEFORE FANNING OUT. Tier order is learned from observed cost, and a worker that
        # starts before any tier has a price runs the ladder in the arbitrary order -- so a wide fan-out
        # means the ENTIRE first wave pays the expensive tier. Measured with 8 workers: 7 of the 12
        # refutable capsules paid the 24.5 s tier before the 0.29 s one had ever been priced, 171 s of a
        # 614 s suite, against a floor of ~444 s. Running the head serially bounds that to one or two.
        #
        # Self-terminating rather than a fixed count: keep going only while a capsule PRICES A TIER
        # nothing had priced before, and stop the moment one teaches us nothing new. A capsule that is
        # refuted early prices only the tier that refuted it, which is exactly why one capsule is not
        # always enough -- and why a hard cap still bounds the worst case.
        head: list[dict] = []
        seen = set(_tier_policy.priced_tiers(target or ""))
        i = 0
        while i < len(caps) and i < _CALIBRATION_CAP:
            _r = _one(caps[i])
            head.append(_r)
            i += 1
            # A capsule that PASSED ran every mandatory tier, so every tier now has a price and there is
            # nothing left to learn -- stop immediately rather than spending a second serial capsule.
            # This matters on a target whose ladder was ALREADY in cost order: it gains nothing from the
            # reordering and pays the whole serial head, and measured that way the head cost more
            # wall-clock (1074s -> 1195s) than the reordering saved. One passing capsule is enough.
            if _r.get("status") == "pass":
                break
            now = set(_tier_policy.priced_tiers(target or ""))
            if now == seen:
                break
            seen = now
        rest = caps[i:]
        if not rest:
            return head
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            _n = min(max_workers, len(rest))
            return head + list(ex.map(lambda c: _one(c, _n), rest))

    # A whole-model (kind == "model") capsule is the GATED capstone: it is scheduled only after the op
    # suite proves itself (its ``gate.after_op_pass_fraction`` of the graded op capsules passed). Grade the
    # op capsules first; if no model capsule is present this is exactly the original single-pass behavior.
    from .capsule_source import model_gate_satisfied
    op_caps = [c for c in capsules if c.get("kind") != "model"]
    model_caps = [c for c in capsules if c.get("kind") == "model"]

    # DO NOT GRADE what this target provably cannot do. A capsule outside the contract's declared
    # capability can never pass, so grading it (a) spends oracle time on a foregone conclusion, (b)
    # misdirects the agent onto unreachable work, and (c) -- the expensive one -- makes ``all_pass``
    # UNREACHABLE, which disables the loop's early exit and turns every run into a fixed-price purchase
    # of its full round budget. Measured: an int8 target scored 22/22 on everything it declares at round
    # 00, then ran 20 rounds because 15 out-of-scope capsules kept all_pass false; a float target had 9
    # reachable failures and burned the same 20 rounds with 4 out-of-scope capsules doing the same thing.
    #
    # They are NOT dropped -- silently shrinking a suite is how a target scores better than it deserves.
    # Each is reported as ``not_graded`` with the derived reason, counted in neither numerator nor
    # denominator, and visible in the result list.
    op_caps, ungradeable = _split_ineligible(op_caps, target)
    # CERTIFY THE COVERING SET FIRST. The derived cover is the fewest capsules whose declared axes span
    # every axis the eligible corpus declares, so a run that is interrupted, times out, or exhausts a
    # certify budget still has every axis represented rather than a lexicographic prefix. Computed over
    # the ELIGIBLE capsules only -- spending the cover on a capsule the hardware cannot execute buys
    # nothing (measured: including the ineligible ones inflated one target's cover from 6 to 17).
    _cover = set(_tier_policy.covering_set(op_caps))
    for _c in op_caps:
        _c["_covering"] = _c.get("name") in _cover
    op_caps = sorted(op_caps, key=lambda c: (not c.get("_covering"), str(c.get("name"))))
    if _cover:
        print(f"  tier plan: {len(_cover)}/{len(op_caps)} eligible capsule(s) form the derived covering "
              f"set (certified first); budget="
              f"{_tier_policy.budget_seconds() or 'unlimited'}", flush=True, file=sys.stderr)
    op_results = _run_all(op_caps) + ungradeable
    if ungradeable:
        print(f"  {len(ungradeable)} capsule(s) NOT GRADED — outside this target's declared capability: "
              f"{', '.join(r['capsule'] for r in ungradeable[:6])}"
              f"{' ...' if len(ungradeable) > 6 else ''}", flush=True, file=sys.stderr)
    if not model_caps:
        return op_results
    # The gate fraction is over what the HARDWARE CAN DO, not over everything graded. A capsule the
    # target provably cannot accelerate must not be able to block the whole-model capsules behind it.
    # Measured: an int8 systolic target graded 12 bf16 capsules its contract declares no capability for
    # ("input dtype 'bf16' not in contraction formats ['int8']"). They can never pass, so the best
    # reachable fraction was 23/35 = 0.66 against a 0.8 gate -- the whole-model capsules were
    # MATHEMATICALLY unreachable, and nothing said so. Excluding the ineligible makes the gate mean
    # "the ops this device can do are working", which is what it was for.
    # WHAT COUNTS AS GRADED, for the gate. Previously only pass/fail, which quietly excluded every
    # capsule that ERRORED -- and an error is not evidence the op suite works, it is the absence of
    # evidence. Measured: 18 of 26 op capsules died with a runner crash and left the denominator with
    # them, so the whole-model capstone cleared its 0.8 gate on the surviving 7/8 = 0.88 while two
    # thirds of the suite had not run. `not_graded` (the hardware cannot) and `gated` (not yet
    # attempted) stay excluded; those are the two the exclusion was FOR. This matches the definition
    # the score itself already uses for its denominator.
    # `screened_only` joins the two existing exclusions for the same reason they are excluded: the
    # capsule was not measured against the certifying tier, deliberately and by name, so it is neither
    # evidence for nor against. Unlike an ERROR (which used to slip out of here unexplained), this is an
    # opt-in choice whose coverage is guaranteed by the covering set and whose members are listed.
    graded = [r for r in op_results
              if r.get("status") not in NOT_MEASURED_STATUSES]
    eligible = [r for r in graded if _gate_counts(r, capsules, target)]
    denom = eligible or graded          # fall back rather than divide by zero if nothing is classifiable
    frac = (sum(1 for r in denom if r.get("status") == "pass") / len(denom)) if denom else 0.0
    if len(eligible) != len(graded):
        print(f"  model gate: {len(graded) - len(eligible)} graded op capsule(s) excluded from the gate "
              f"denominator as ineligible for this target (the hardware declares no capability for them)",
              flush=True, file=sys.stderr)
    # SAY IT WHEN THE GATE GOT CHEAPER. Under a certify budget the denominator is what was CERTIFIED, not
    # what exists, so a capstone can clear 0.8 on the covering set alone. That is the intended trade --
    # the cover spans every declared axis -- but a gate satisfied by 6 capsules instead of 23 is a
    # different statement, and it has to be printed next to the number rather than inferred.
    _screened = [r for r in op_results if r.get("status") == "screened_only"]
    if _screened:
        print(f"  model gate: denominator is the {len(denom)} CERTIFIED capsule(s); "
              f"{len(_screened)} more passed the screen and were not certified "
              f"(outside the covering set, budget exhausted) — the gate fraction {frac:.2f} is over "
              f"what was certified, not over the whole suite", flush=True, file=sys.stderr)
    model_results = []
    for c in model_caps:
        if model_gate_satisfied(c, frac):
            model_results.append(_one(c))
        else:
            thr = (c.get("gate") or {}).get("after_op_pass_fraction")
            model_results.append({
                "capsule": c["name"], "kind": "model", "label": c.get("label"), "status": "gated",
                "failure": {"plane": "gate", "category": "GATED",
                            "detail": f"whole-model capsule deferred: op pass fraction {frac:.2f} "
                                      f"< gate {thr}"}})
    return op_results + model_results


def main(argv: list[str] | None = None) -> int:
    import argparse
    # The child half of a BUDGETED whole-model grade (see _grade_model_capsule). Parsed before the
    # normal arguments because it shares none of them: the whole invocation is one spec file in and
    # one result file out, so the parent can kill the process group without losing what it knows.
    _av = list(sys.argv[1:] if argv is None else argv)
    if "--model-grade" in _av:
        sub = argparse.ArgumentParser(add_help=False)
        sub.add_argument("--model-grade", required=True)
        sub.add_argument("--model-grade-out", required=True)
        sa, _ = sub.parse_known_args(_av)
        _die_with_parent()
        spec = json.loads(Path(sa.model_grade).read_text(encoding="utf-8"))
        res = _grade_model_capsule_inline(spec["capsule"], target=spec.get("target"),
                                          timeout=int(spec["timeout"]),
                                          package_dir=spec.get("package_dir"))
        Path(sa.model_grade_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        return 0

    ap = argparse.ArgumentParser(description="capsule_bench_v0 runner")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsule", help="path to a single capsule dir")
    ap.add_argument("--capsules-root", help="run every capsule under this root")
    ap.add_argument("--labels", default="public,dev", help="comma-separated label filter")
    ap.add_argument("--runs-root", default="out/runs/capsule_bench")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--target", required=True, help="target to grade (its config/oracle are derived)")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)

    if a.capsule:
        caps = [load_capsule(a.capsule, contract=a.contract)]
    else:
        labels = set(a.labels.split(",")) if a.labels else None
        caps = discover_capsules(a.capsules_root, labels=labels, contract=a.contract)
    results = run_suite(caps, a.package, runs_root=a.runs_root, contract=a.contract,
                        timeout=a.timeout, target=a.target)
    npass = sum(1 for r in results if r["status"] == "pass")
    for r in results:
        print(f"  [{r['status']:10s}] {r['capsule']}")
    print(f"\n{npass}/{len(results)} pass")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
