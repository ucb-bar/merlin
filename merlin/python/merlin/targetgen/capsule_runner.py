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
import sys
import threading
import traceback as _traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from aet.core.run_paths import RunPaths

from ..common.provenance import UNKNOWN as PROV_UNKNOWN
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
from .capsule_common import (_cat, _flat, discover_capsules, load_capsule,  # noqa: F401
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
    toolchain: str | None = None      # WHICH PROGRAM was graded, as reported by the adapter. A block-
                                      # scaled MX capsule is graded on the harness's own reference MX
                                      # kernel rather than the submission, so a pass there measures the
                                      # fixture. Recording it keeps a score decomposable instead of
                                      # silently overstating the backend by the size of the MX set.
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
        # WHICH PROGRAM was graded. Rides only when the adapter reported one, so existing output is
        # unchanged. This is what separates "the submission passed" from "the harness fixture passed":
        # a block-scaled MX capsule is graded on the reference MX kernel, not the submitted backend.
        if self.toolchain:
            d["toolchain"] = self.toolchain
        return d


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
        return {"outputs": res.get("outputs"),
                "cycles": (res.get("metrics") or {}).get("cycles"),
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


def codegen_smoke(target: str) -> tuple[bool, str]:
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
    try:
        from .isa_model import isa_model_for_target
        if not isa_model_for_target(target).is_fixed_format():
            return True, "n/a (ISA is not fixed-format — no fork-free re-encode smoke for this emit path)"
    except Exception as e:  # noqa: BLE001 — no derived model -> nothing to smoke here
        return True, f"n/a (no fixed-format ISA model: {str(e)[-120:]})"
    if _bespoke_sim_via(target) != "cyclotron":
        return True, "n/a (fixed-format ISA but no cyclotron reference sim declared for the fork-free smoke)"
    try:
        from ..runtime.backends import base as _bk
        _muon = _bk.get_backend("muon")   # the evicted SIMT reference backend, resolved via discovery
    except Exception as e:  # noqa: BLE001
        return True, f"n/a (fork-free backend unimportable: {str(e)[-120:]})"
    if not _muon.available("cyclotron"):
        # cyclotron absence is already reported by oracle_available; not this gate's job to double-block.
        return True, "n/a (reference sim absent — oracle_available reports this separately)"
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
        # RANK IS NOT A HARD FACT — it is the one thing a compiler exists to change.
        #
        # The dtype test above IS hard: if no datapath holds the operand format, no arrangement of the
        # program puts it on the device. Rank fails that test by exactly the argument the paragraph above
        # makes for family. Ranks compose: a rank-4 convolution reaches a rank-2 mesh through im2col, and
        # this compiler already SHIPS that lowering (`convolution_im2col_matmul` in linalg_lower, which
        # derives the conv geometry from the operand shapes and emits an (m,k,n) matmul).
        #
        # Measured on radiance: RP14_patch_embed_bf16_pt was withheld as "rank 4 not in contraction legal
        # ranks [2, 3]" and never graded, while the lowering that turns it into a rank-2 contraction sat
        # in the tree unused. Withholding it did not report a hardware limit — it hid a reachable
        # capability behind the SOURCE shape of the op, which is the question a library of kernels asks,
        # not the question a compiler answers.
        #
        # So rank now only DOWNGRADES to graded: if the compiler cannot in fact lower it, the capsule
        # fails honestly and visibly, which is the finding we want. Withholding on doubt is how a suite
        # shrinks itself into a better score.
        dt = getattr(region, "in_dtype", None)
        all_dtypes = tuple({x for cap in cmap.values() for x in (getattr(cap, "dtypes", ()) or ())})
        dtype_absent = bool(dt is not None and all_dtypes and not _el._dtype_ok(dt, all_dtypes))
        if not dtype_absent:
            keep.append(c); continue
        # State the fact that ACTUALLY triggered withholding. eligibility's reason describes its own
        # family verdict, which can read "unrecognized semantic family" for a capsule withheld purely
        # because its dtype has no datapath -- true but misleading about the cause, and it would send
        # someone to fix the taxonomy when the hardware is the constraint.
        # Only the dtype reaches here now; rank is graded (see above), so the reason cannot be about it.
        why = f"operand dtype {dt!r} is in no capability this target declares"
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

    True unless the capsule is provably ineligible for this target -- i.e. the contract declares no
    capability covering its family/dtype, so no compiler could ever make it pass. Fails OPEN (counts the
    capsule) whenever eligibility cannot be decided: an undetermined region must not be silently dropped
    from a gate, which would make the gate easier to satisfy than the evidence warrants.
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
        verdict = _el.is_eligible(_cr._capsule_region(cap), cmap)
    except Exception:                                     # noqa: BLE001 - a gate must not crash a grade
        return True
    if getattr(verdict, "undetermined", False):
        return True
    return bool(getattr(verdict, "eligible", True))


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


def _encoding_divergence_hint(trace_check_res: dict | None, oracle_graded: bool,
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
    if oracle_graded:
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


def _grade_model_capsule(capsule: dict, *, target: str | None = None, timeout: int,
                         package_dir=None, required=(), paths=None, run_id: str = "",
                         cfg=None, contract=None, eff_target: str = "") -> dict:
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
    # The target's OWN mesh is the oracle whenever we have a target -- which subsumes must_accelerate,
    # since a capsule demanding acceleration on a target with no mesh has nothing to run on either way.
    # `host` runs merlin's x86 dispatch runtime, which is OUR compiler: a useful diagnostic, never
    # evidence about the submission, so a host run is recorded advisory and finalized as
    # `not_gradeable_no_oracle` below (never `pass`). The env var still overrides, for a deliberate
    # diagnostic run.
    run_where = os.environ.get("MERLIN_MODEL_GRADE_RUN") or ("mesh" if target else "host")
    # MERLIN_MESH_VERIFY additionally EXECUTES each mesh-routed matmul as a single systolic tile on the
    # target's real mesh oracle (compile_model mesh_verify) — proving the matmul layers run ON the mesh, not
    # just that a routing plan was produced. Off by default (the oracle build/run is heavy); the whole-model
    # functional gate stays compile_rvv either way.
    mesh_verify = os.environ.get("MERLIN_MESH_VERIFY", "").lower() in ("1", "true", "yes", "on")
    result: dict = {"capsule": capsule["name"], "kind": "model", "label": capsule.get("label"),
                    "operation": {"op": "model", "model": model, "dtype": dtype, "run": run_where,
                                  "target": target},
                    "contract_version": CONTRACT_VERSION}
    def _bail(detail: str, category: str = "NOT_RUN_IS_NOT_PASS") -> dict:
        """An honest un-run row: every required tier recorded `unavailable` WITH ITS REASON, then the
        shared finalizer. `tiers: {}` would read as 'no tiers apply here' when it means 'nothing ran'."""
        _req = set(required or capsule.get("required_oracle_tiers") or ())
        _t = {x: TierResult(x, "unavailable", True, reason=detail) for x in sorted(_req)}
        _fail = {"plane": "model", "category": category, "detail": detail}
        _extra = {k: result.pop(k) for k in ("operation",) if k in result}
        if paths is None:
            result.update(status="incomplete", numeric={"status": "not_compared"},
                          failure=_fail, tiers={k: v.to_dict() for k, v in _t.items()}, **_extra)
            return result
        return _finalize_capsule_result(
            name=capsule["name"], capsule=capsule, status="incomplete", failure=_fail, tiers=_t,
            trace_check_res={"status": "skipped", "violations": [],
                             "reason": "the whole model did not run"},
            numeric={"status": "not_compared"}, required=_req, no_oracle=False,
            eff_target=eff_target or (target or ""), paths=paths, run_id=run_id, cfg=cfg,
            contract=contract, extra=_extra)

    if not model:
        return _bail("model capsule missing operation.attributes.model", category="RUNNER_CRASH")
    # The captured linalg (visible grounding) drives the per-op mesh routing when present. Read the name
    # the CAPSULE DECLARES -- every model capsule ships `linalg_mlir: capsule.interface.mlir`, while this
    # looked only for `capsule.linalg.mlir`, so it never found one. `linalg_mlir` was therefore always
    # None, `compile_model`'s `if target and linalg_mlir:` never fired, and the routing plan, the coverage
    # certificate AND the mesh verification were all skipped on every whole-model capsule ever graded --
    # silently, because a skipped block leaves no trace in the result.
    linalg_mlir = None
    cdir = capsule.get("__dir__")
    if cdir:
        for _name in (capsule.get("linalg_mlir"), capsule.get("interface_mlir"), "capsule.linalg.mlir"):
            if not _name:
                continue
            lp = _P(cdir) / str(_name)
            if lp.is_file():
                linalg_mlir = lp.read_text(encoding="utf-8")
                break
    try:
        from ..compile_cli import compile_model
        # `mesh_package` is the SUBMISSION under test. Without it the mesh path resolves
        # `_default_oot_package(target)` -- the promoted package -- and quietly certifies a different
        # compiler than the one being graded.
        # `routing_dtype` is the capsule's declared datapath format: `dtype` here is an RVV compile mode,
        # so routing a demand against it matches a compile-mode token rather than the unit's real format.
        _attrs = (capsule.get("operation") or {}).get("attributes") or {}
        out = compile_model(model, dtype, target=target, run=run_where, verify=True, package=None,
                            auto_capture=True, timeout=timeout, linalg_mlir=linalg_mlir,
                            mesh_verify=mesh_verify, mesh_package=package_dir,
                            routing_dtype=_attrs.get("dtype"))
    except SystemExit as e:                                   # toolchain/bundle unavailable — honest skip
        return _bail(f"whole-model compile/run unavailable: {str(e)[:300]}")
    except Exception as e:  # noqa: BLE001
        return _bail(f"whole-model grade error: {type(e).__name__}: {str(e)[:300]}")
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
    # TWO MESH EVIDENCE SOURCES, AND THEY ARE DIFFERENT CLAIMS. `mesh_execution` records what happened
    # to THIS MODEL's own layers -- how many the dispatch runtime got onto the accelerator and how many
    # fell back to the host kernel. `mesh_tile_verification` records a SYNTHESIZED tile at each routed
    # shape, which proves the SHAPE is runnable and says nothing about this model. They shared one key
    # once and the tile record clobbered the model one, so a run with all 15 layers on the host reported
    # "15 of 15 tiles passed". The model's own accounting decides the tier; the tile record is reported
    # beside it as the weaker, separate evidence it is.
    mx = out.get("mesh_execution") or {}
    ran = mx.get("matmul_layers_on_mesh")
    fell = mx.get("matmul_layers_host_fallback")
    tile = out.get("mesh_tile_verification") or {}
    n_tiles = int(tile.get("n_tiles") or 0) if isinstance(tile, dict) else 0
    # The tile verdict comes from the per-tile COUNTS. Reading a boolean `ok` key -- which this dict does
    # not carry -- made every certification record "fail", including one where every tile passed on the
    # oracle. A tile that was unavailable or unsynthesizable is not a pass either, so it passes only when
    # every tile is accounted for and every one of them passed.
    _tok = tile.get("ok")
    if not n_tiles:
        tiles_ok = False
    elif _tok is not None:
        tiles_ok = bool(_tok)
    else:
        tiles_ok = (int(tile.get("n_failed") or 0) == 0
                    and int(tile.get("n_unavailable") or 0) == 0
                    and int(tile.get("n_unsynthesizable") or 0) == 0
                    and int(tile.get("n_passed") or 0) == n_tiles)
    _declared = [str(x) for x in (capsule.get("required_oracle_tiers") or [])]
    # WHICH tier the mesh oracle corresponds to, DERIVED from the target's own declared RTL tiers rather
    # than assumed: excluding the literals "L0"/"L1" and falling back to a literal names the wrong tier
    # confidently on any target whose ladder differs (atlas grades at L3+L4). The literal path survives
    # only as the last resort for a target whose manifest cannot be resolved.
    _rtl = [x for x in _declared if x in _rtl_tiers_of(target)]
    _mesh_tier = (_rtl[-1] if _rtl
                  else next((x for x in reversed(_declared) if x not in ("L0", "L1")), "L2"))
    numeric: dict
    # WHOSE COMPILER PASSED. When the runtime had to discharge a contract obligation the target backend
    # owes -- residency tiling for `capacity_fit` -- the verdict is about runtime+backend together, and
    # saying so is the difference between "this compiler handles a 512x512 layer" and "this layer ran".
    # `tile_source` on each entry says which tiler chose the extent: a capacity fact read out of the RTL,
    # or a width probed until the backend stopped refusing.
    _delegated = mx.get("capacity_fit_delegated_to_runtime") or []
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
                "tile_sources": sorted({str(d.get("tile_source")) for d in _delegated
                                        if isinstance(d, dict)}),
                "detail": ("the target backend did not satisfy capacity_fit at these extents; the "
                           "runtime split them so the model could run. This verdict is evidence about "
                           "the runtime AND the backend, not about the backend alone."),
            }
        }
    # MUST_ACCELERATE IS ABOUT THIS MODEL, NOT ABOUT TILES OF ITS SHAPES. Certifying a synthesized tile
    # proves the shape is runnable; it says nothing about whether the model's own layer reached the
    # device. Measured: a target routed 15 matmul layers and the dispatch runtime fell back to the host
    # kernel on all 15, while the tile record read "15 of 15 passed" -- so a run with ZERO layers on the
    # accelerator reported `pass` with `lane: mesh`, which is exactly the CPU-only pass this flag exists
    # to forbid. Held aside rather than returned here so the row still exits through the tier ladder and
    # the finalizer: an early return would ship a result with no `tiers` and skip every gate below it.
    # It is applied AFTER the generic failed-tier branch, so the SPECIFIC cause is what the row names --
    # "2 of 37 layers fell back" is actionable where a bare FUNCTIONAL_MISMATCH is not.
    _accel_fail: tuple[str, dict] | None = None
    if (capsule.get("semantic") or {}).get("must_accelerate") and mx:
        if ran is None or ran is PROV_UNKNOWN:
            _accel_fail = ("incomplete",
                           {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                            "detail": "capsule declares must_accelerate but the run recorded no "
                                      "per-layer mesh accounting; cannot confirm the model reached "
                                      "the accelerator"})
        # A FALLBACK, specifically -- at least one layer the mesh was asked for and the host ran instead.
        # `on_mesh == 0` with `host_fallback == 0` is the DIFFERENT case where nothing ran at all: no
        # layer was routed, so nothing fell back, and the honest report is "no evidence" (which the tier
        # ladder below already reaches as `skipped` -> incomplete), not a fallback that never happened.
        elif (fell is not None and fell is not PROV_UNKNOWN and int(fell)):
            _fb = int(fell)
            _accel_fail = ("fail",
                           {"plane": "model", "category": "FALLBACK_ON_ELIGIBLE_REGION",
                            "detail": f"capsule declares must_accelerate but only {int(ran)} matmul "
                                      f"layer(s) executed on the {target} mesh and {_fb} fell back "
                                      f"to the host kernel; the numeric gate therefore measures "
                                      f"the HOST, not the accelerator"})
    if st == "verified" and gate:
        # A whole-model PASS must carry the strength of the gate that produced it. The numeric gate has a
        # per-element ceiling for exactly the failure aggregates hide (a single element 1209% wrong at
        # cos 0.9999986), but its cosine-only tier bypasses that ceiling for regression outputs which
        # cannot meet it. Both are legitimate passes; they are not the same claim, and "certified" read
        # off `status` alone erases the difference.
        _guarded = _v.get("per_element_guarded")
        numeric = {"status": "pass", "engine": engine, "gate": out.get("verify"),
                   "gate_tier": _v.get("tier_ok"),
                   "per_element_guarded": _guarded,
                   "evidence": ("per-element-guarded" if _guarded else
                                "AGGREGATE ONLY — the cosine-only tier carried this verdict; no "
                                "per-element bound was applied")}
        status = "pass"
    elif _quant and st == "run_mismatch" and _cos >= _quant_floor:
        numeric = {"status": "pass", "engine": engine, "gate": _v,
                   "quant_tolerance": {"cos": _cos, "floor": _quant_floor, "dtype": dtype}}
        status = "pass"
        result["note"] = (f"quantized ({dtype}) whole-model output within quant tolerance of the fp32 "
                          f"golden (cos {_cos:.4f} >= floor {_quant_floor}); the drop vs fp32 is expected "
                          f"quantization error, not a codegen defect.")
    elif st == "not_run":
        numeric = {"status": "not_compared", "engine": engine}
        status = "incomplete"
        result["failure"] = {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS",
                             "detail": out.get("reason", "whole-model run toolchain unavailable")}
    else:
        numeric = {"status": "fail", "engine": engine, "gate": out.get("verify"), "model_status": st}
        status = "fail"
        result["failure"] = {"plane": "model", "category": "FUNCTIONAL_MISMATCH",
                             "detail": f"the whole model did not verify (status={st}, gate_ok={gate})"}

    # L0/L1 interpret a COMMAND BUFFER; a whole-model capsule has none, so they are honestly N/A rather
    # than "did not run" -- the same idiom the integer L0/L1 floor uses on a float datapath. That leaves
    # the mesh tier as the only live required tier, which is exactly what should decide a whole-model
    # verdict. The capsule's own `required_oracle_tiers` is the authority on what must certify it.
    # Defaulting to an empty set would make NO tier mandatory, so every fail-closed gate below would be
    # vacuous.
    _required = set(required or capsule.get("required_oracle_tiers") or ())
    tiers: dict = {}
    for _t in ("L0", "L1"):
        tiers[_t] = TierResult(_t, "skipped", _t in _required, not_applicable=True,
                               reason="no command buffer for a whole-model capsule")
    if run_where == "host":
        # Our own engine is not an oracle for the submission: record it, withhold the verdict.
        result["host_reference"] = {"run": "host", "engine": engine, "gate": out.get("verify"),
                                    "status": st}
        numeric = {"status": "not_compared", "engine": engine,
                   "reason": "host dispatch-runtime reference is advisory; not an oracle for the submission"}
        tiers[_mesh_tier] = TierResult(_mesh_tier, "skipped", _mesh_tier in _required,
                                       reason="host reference run: the target mesh did not execute this model")
        status = "pass"                       # the finalizer converts this to not_gradeable_no_oracle
        _no_oracle = True
    else:
        _no_oracle = False
        if ran is not None and ran is not PROV_UNKNOWN:
            # THE MODEL'S OWN LAYERS DECIDE A MODEL CAPSULE'S TIER, and they decide it whenever they were
            # counted -- ahead of any tile record. Consulting the tiles first is how a run that put ZERO
            # layers on the accelerator reported a pass: the tile certification was for the same shapes
            # and had nothing to say about the model.
            if not int(ran):
                tiers[_mesh_tier] = TierResult(_mesh_tier, "skipped", _mesh_tier in _required,
                                     reason="no matmul layer executed on the target mesh (every layer fell "
                                            "back to the host)")
            elif fell:
                tiers[_mesh_tier] = TierResult(_mesh_tier, "fail", _mesh_tier in _required,
                                     reason=f"{fell} mesh-routed layer(s) fell back to the host")
            else:
                # The model ran clean on the mesh. A tile certification, WHEN ONE WAS RUN, can only
                # downgrade that -- a shape the oracle could not certify is evidence against the run, and
                # `status: pass` printed beside a failing tile is the contradiction a reader takes the
                # flattering half of. It can never create a pass: the model-lane conditions above stay
                # necessary.
                _ok = numeric["status"] == "pass" and (tiles_ok or not n_tiles)
                _why = f"{ran} matmul layer(s) executed on the target mesh"
                if n_tiles and not tiles_ok:
                    _why += (f"; {int(tile.get('n_passed') or 0)} of {n_tiles} certified tile(s) passed "
                             f"({tile.get('n_failed')} failed, {tile.get('n_unavailable')} unavailable, "
                             f"{tile.get('n_unsynthesizable')} unsynthesizable)")
                tiers[_mesh_tier] = TierResult(_mesh_tier, "pass" if _ok else "fail",
                                     _mesh_tier in _required, reason=_why, evidence="mesh_execution")
        elif n_tiles:
            # per-tile certification evidence (mesh_verify) and nothing about this model's own layers
            tiers[_mesh_tier] = TierResult(_mesh_tier, "pass" if tiles_ok else "fail",
                                           _mesh_tier in _required,
                                           reason=f"{n_tiles} mesh tile(s) certified at this model's "
                                                  f"shapes; no per-layer accounting for the model itself",
                                           evidence="mesh_tile_verification")
        else:
            tiers[_mesh_tier] = TierResult(_mesh_tier, "unavailable", _mesh_tier in _required,
                                     reason="mesh execution counters unavailable — the target mesh could "
                                            "not be reached (no OOT package, or the oracle failed to run)")

    # The tile record, kept beside the verdict as the SEPARATE and weaker evidence it is: it speaks about
    # synthesized tiles of this model's shapes, never about the model.
    if tile:
        result["tile_evidence"] = {"n_tiles": n_tiles, "n_passed": tile.get("n_passed"),
                                   "note": "synthesized tiles OF THIS MODEL'S SHAPES — evidence that the "
                                           "shapes run, not that this model ran"}
    # The gate fraction is OP COVERAGE, not a model verdict; say so on the row rather than letting a
    # reader infer one from the other. And name the declared tiers that never ran.
    result["op_coverage"] = {"note": "the op-pass fraction this capsule was gated on is OP COVERAGE, "
                                     "not a verdict on the model"}
    _unexercised = [t for t in _declared
                    if tiers.get(t) is None or tiers[t].status not in ("pass", "fail")]
    if _unexercised:
        # WITH THE REASON, not as a bare list. A list read as "we skipped three of the four tiers this
        # capsule declares", which is indistinguishable from a grade that cut corners; the actual reason
        # is structural (those tiers grade a per-op command buffer a model capsule never produces).
        result["tiers_unexercised"] = _unexercised_note(
            _unexercised, {t: r.status for t, r in tiers.items() if r.status in ("pass", "fail")})
    # A tier that RAN AND FAILED is not a pass, whatever the host-side numeric gate says -- and it is not
    # an "incomplete" either: something ran and it was wrong, which is a different report from "nothing
    # ran". The guard below only refuses the nothing-ran case.
    _failed_tiers = sorted(t for t, r in tiers.items() if r is not None and r.status == "fail")
    if status == "pass" and _failed_tiers:
        status = "fail"
        result["failure"] = {"plane": "model", "category": "FUNCTIONAL_MISMATCH",
                             "detail": f"declared oracle tier(s) {_failed_tiers} RAN and did not pass "
                                       f"({'; '.join(tiers[t].reason or '' for t in _failed_tiers)}); a "
                                       f"whole-model verdict cannot be a pass over a failing tier"}
    # The SPECIFIC cause last, so it is the one the row names (see where `_accel_fail` is built).
    if _accel_fail is not None:
        status, result["failure"] = _accel_fail
        numeric = {"status": "not_compared", "engine": engine}
    # If no DECLARED tier certified this model, the numeric verdict is withheld whatever the functional
    # gate said. The gate compared our own reference run against the golden; that is a real comparison,
    # but not of the artifact under test, and reporting it as `pass` beside an `incomplete` status invites
    # exactly the reading this whole change removes.
    if not any(tiers.get(t) is not None and tiers[t].status == "pass" for t in _required):
        numeric = {**numeric, "status": "not_compared"}

    extra = {k: result.pop(k) for k in ("routing_plan", "coverage_certificate", "mesh_execution",
                                        "mesh_tile_verification", "tile_evidence",
                                        "host_reference", "note", "operation", "op_coverage",
                                        "tiers_unexercised")
             if k in result}
    # A verdict that claims the hardware ran something records WHICH tree produced it. Only stamped when
    # the mesh actually executed: a withheld verdict has no hardware claim to attribute. The tier is the
    # one derived above, not a literal -- a target whose ladder does not contain it would never stamp.
    if tiers.get(_mesh_tier) is not None and tiers[_mesh_tier].status == "pass":
        try:
            from ..common import provenance as _PROV
            extra["provenance"] = _PROV.record(
                sources=[str(package_dir)] if package_dir else (),
                extra={"mesh_target": target, "layers_on_mesh": ran})
        except Exception as _e:                     # noqa: BLE001 — never fail a grade on bookkeeping
            extra["provenance"] = {"error": f"{type(_e).__name__}: {_e}"}
    if paths is None:                          # standalone/diagnostic call — no run dir to finalize into
        # Without a run dir we cannot call the finalizer, but we must still apply its gates or this branch
        # re-creates the exact fail-open the rung exists to close. Same two rules, same order.
        if status == "pass":
            _ran_required = [t for t in _required
                             if tiers.get(t) is not None
                             and not tiers[t].not_applicable and tiers[t].status == "pass"]
            if not _ran_required:
                status = "not_gradeable_no_oracle" if _no_oracle else "incomplete"
                result.setdefault("failure", {
                    "plane": "not_gradeable_no_oracle" if _no_oracle else "oracle_unavailable",
                    "category": "NOT_GRADEABLE_NO_ORACLE" if _no_oracle else "NOT_RUN_IS_NOT_PASS",
                    "detail": ("host dispatch-runtime reference only: the target mesh did not execute "
                               "this model, so the numeric verdict is withheld") if _no_oracle else
                              (f"declares required oracle tiers {sorted(_required)} and ran NONE of them "
                               f"(the functional gate here is the {run_where} reference, not the "
                               f"accelerator); a whole-model verdict backed by no declared tier is "
                               f"reported UNKNOWN, never a pass")})
        result.update(status=status, numeric=numeric,
                      tiers={t: r.to_dict() for t, r in tiers.items()}, **extra)
        return result
    return _finalize_capsule_result(
        name=capsule["name"], capsule=capsule, status=status, failure=result.get("failure"),
        tiers=tiers, trace_check_res={"status": "skipped", "violations": [],
                                      "reason": "no per-op instruction trace for a whole-model capsule"},
        numeric=numeric, required=_required, no_oracle=_no_oracle, eff_target=eff_target or (target or ""),
        paths=paths, run_id=run_id, cfg=cfg, contract=contract, extra=extra)


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


def run_capsule(capsule: dict, package_dir: str | Path, *, runs_root: str | Path,
                run_id: str | None = None, contract: str | Path | None = None,
                oracle_adapters: dict[str, Callable] | None = None,
                pkg: Package | None = None, timeout: int = 600,
                target: str | None = None, suite: str | None = None, dtype: str = "i8xi8_i32",
                config=None, perf_extractor: Callable | None = None,
                no_oracle: bool = False) -> dict:
    """Run one capsule through the package; write artifacts; return a capsule_result dict.

    ``config`` (a :class:`runner_config.RunnerConfig`) supplies the per-target grading knobs — the
    4th-artifact name, the sim-tier map + RTL tiers + loop order, the optional trace gate, and the perf
    fields — so ONE runner serves every target. When absent, the implicit gemmini/systolic config is
    built from ``target``/``suite``/``dtype`` (byte-identical to the pre-collapse behavior). Output
    equality uses the capsule's ``numeric_policy`` (exact for integer, tolerance for float). ``perf_extractor``
    (cb -> flops) feeds the SIMT gflops/pct_fp_peak. ``oracle_adapters`` is the per-target oracle set: the
    L0/L1 math floor always runs; RTL tiers grade only if an adapter is present + available (arc or a
    bespoke sim), else honestly ``unavailable`` — arc is never assumed."""
    from ..runtime.reference import UnmodeledOp, reference_outputs
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
        # Hand the SUBMISSION and the run context down: the model grade now builds real TierResults and
        # shares `_finalize_capsule_result`, so it is subject to the same not_run_is_not_pass gate, the
        # same fail-open guard and the same schema validation as every op capsule. It used to return
        # here with `tiers: {}` and a row that did not validate.
        result = _grade_model_capsule(capsule, target=eff_target, timeout=timeout,
                                      package_dir=package_dir, required=required, paths=paths,
                                      run_id=run_id, cfg=cfg, contract=contract, eff_target=eff_target)
        # Persist the ARR coverage certificate as its own durable artifact (not only inside
        # capsule_result.json) so the report/grader can read it back per compilation.
        cert = result.get("coverage_certificate")
        if cert is not None:
            paths.generated.mkdir(parents=True, exist_ok=True)
            (paths.generated / "coverage_certificate.json").write_text(
                json.dumps(cert, indent=2), encoding="utf-8")
        return result

    tiers: dict[str, TierResult] = {}
    # WHICH PROGRAM each tier actually graded, keyed by tier (see TierResult.toolchain). Declared out here
    # so it survives an exception raised inside the tier loop.
    _tier_toolchain: dict[str, str] = {}
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
        if not _vals:
            # No golden.yaml ships operands for this capsule, so its golden is RECOMPUTED on
            # materialize_capsule_leaves — attach that SAME stimulus. Without it a program-oracle target
            # has nothing to build its kernel harness from and fails closed ("could not derive harness
            # operands from the command buffer (no canonical_inputs)"), so the capsule reported
            # `missing from observed` for EVERY submission, correct or not — ungradeable by construction
            # rather than failed on merit. The golden and the device must see one stimulus.
            _vals = CG.materialized_input_values(capsule)
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

        # Does the numeric verdict ride the HARDWARE oracle (tolerance vs this capsule's golden) instead
        # of the cheap integer reference? True for a float datapath with an independent golden, and set
        # below when the cheap engines turn out to have no definition for an opcode in the buffer. Either
        # way there is no usable integer cross-check, so the oracle comparison IS the verdict.
        oracle_graded = independent_float
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
                reason="integer reference not applicable to float datapath; graded vs the "
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
            except UnmodeledOp as ue:
                # THIS ENGINE cannot check the buffer — its op vocabulary has no definition for an opcode
                # the result depends on. That is a limit of the cheap tier, NOT a defect in the submission,
                # so it is skipped honestly (not_applicable) and correctness is graded where the op DOES
                # execute: the hardware oracle, against this capsule's golden. Distinguishing this from a
                # malformed buffer matters — the engine used to drop such an op silently and return an empty
                # output map, which surfaced as "your kernel never wrote its output" and could not be fixed
                # by any submission, correct or not.
                ref = sim = None
                oracle_graded = True
                numeric = {"status": "skipped", "policy": policy.get("compare"), "golden_source": gsource,
                           "note": f"cheap integer tiers N/A ({ue}); correctness is graded by running your "
                                   "artifact on the RTL oracle and checking it computes the declared "
                                   "operation within tolerance"}
                tiers["L0"] = TierResult(
                    "L0", "skipped", mandatory="L0" in required, not_applicable=True,
                    reason=f"integer reference has no definition for {sorted(set(ue.opcodes))}; "
                           f"graded vs the golden ({gsource}) at the RTL oracle")
                tiers["L1"] = TierResult(
                    "L1", "skipped", mandatory="L1" in required, not_applicable=True,
                    reason=f"integer simulate has no definition for {sorted(set(ue.opcodes))}")
            except (ValueError, KeyError, IndexError, TypeError) as ce:
                raise CertFailure(
                    "command_buffer", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                    f"command buffer could not be interpreted by reference/simulate "
                    f"({type(ce).__name__}: {ce}); check operand ranks/shapes and that each command's op "
                    f"is one the reference interpreter models (e.g. a windowed op lowered to a 2D matmul)"
                ) from ce
        if not oracle_graded and ref is not None:
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
        # A MANDATORY tier that already failed makes every DEEPER tier unaffordable and uninformative: the
        # capsule is failed whatever the deeper tier says, and a deeper tier is the expensive one. Measured
        # on this repo's SIMT target, the functional tier costs ~2.5 s per capsule and the cycle-accurate
        # tier costs minutes, so a sweep scoring 18/35 spent RTL time on all 35 -- including the 17 whose
        # numerics were already known wrong. Deeper tiers are recorded as `skipped` naming the tier that
        # failed, never as `fail`: they did not run, and `not_run_is_not_pass` reads a fabricated fail as
        # evidence the capsule was certified at that tier.
        #
        # Scoped to a MANDATORY failure on purpose. An OPTIONAL cert tier that fails must not suppress
        # anything -- it is precisely the "passes the functional oracle, fails RTL" signal this benchmark
        # exists to surface, and a capsule can legitimately pass with an optional tier red.
        # Derived from what was RECORDED, not from a flag each failure path has to remember to set: there
        # are eleven sites in this loop that write a TierResult, and a flag would silently stop
        # short-circuiting the day a twelfth is added.
        def _already_failed_mandatory() -> str | None:
            return next((t for t, r in tiers.items()
                         if getattr(r, "mandatory", False) and getattr(r, "status", None) == "fail"), None)

        # CHEAPEST MEASURED TIER FIRST, not lexicographic -- and the short-circuit above is exactly what
        # makes the order matter. The ladder stops paying once a mandatory tier refutes a capsule, so
        # sorting by name meant one target paid its 24.5 s arc cosim before reaching the 0.29 s Verilator
        # tier that refutes the same capsules (measured: 12 of 12, identical signature). See tier_policy.
        _tier_seq = _tier_policy.tier_order(str(cfg.target or target or ""),
                                            set(cfg.oracle_tiers) | set(adapters or {}))
        _screen_tier = _tier_seq[0] if _tier_seq else None
        for tier in _tier_seq:
            mand = tier in required
            _failed_mandatory = _already_failed_mandatory()
            if _failed_mandatory is not None:
                tiers[tier] = suppressed_tier_result(tier, mand, _failed_mandatory,
                                                     from_rtl=tier in cfg.rtl_tiers)
                continue
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
                # Which program the adapter actually built. Captured here because the branches below each
                # construct their own TierResult and `res` is not in scope after them; applied in one place
                # once the loop is done.
                if isinstance(res, dict) and res.get("toolchain"):
                    _tier_toolchain[tier] = str(res["toolchain"])
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
                    evidence=f"{_sim_name}_console.log", timing=_tm, not_applicable=True)
                continue
            if res.get("completion_only"):
                # An RTL cert that ran the emitted kernel to completion but cannot surface its outputs
                # for an independent numeric check (e.g. the Muon Verilator harness $finish-races the
                # UART flush). It certifies RTL COMPLETION + cycle-accurate cycles; CORRECTNESS is the
                # mandatory functional tier's job. It is NEVER allowed to stand in for a mandatory tier
                # (a required tier must verify output) — there it degrades to honest-unavailable.
                _sim_name = cfg.tier_sim.get(tier) or tier
                if mand:
                    # A mandatory completion-only RTL tier certifies that the emitted kernel RAN to
                    # completion on cycle-accurate RTL, but cannot itself surface outputs for a numeric
                    # check. Correctness is still REQUIRED: it passes only when an independent functional
                    # oracle has ALREADY established a numeric pass on the SAME artifact — L2 (cyclotron)
                    # and this RTL tier build the identical fork-free ELF (``compile_mlir_forkfree``), and
                    # tiers are graded in sorted() order so L2 sets ``numeric`` before this runs. So a
                    # completing RTL run of a numerically-verified ELF is a sound mandatory certificate.
                    # Fail-closed otherwise: if no functional tier established a numeric pass
                    # (``numeric.status`` stays "skipped"/"fail"), it degrades to honest-unavailable. The
                    # agent never sees the golden — correctness is graded against it inside the runner —
                    # so this does not weaken anti-cheat; it makes the mandatory RTL tier meaningful
                    # instead of un-satisfiable. (completion_only is produced only by the Muon adapters;
                    # command-ISA targets surface outputs and never reach this branch.)
                    if numeric.get("status") == "pass":
                        tiers[tier] = TierResult(
                            tier, "pass", mand,
                            reason="RTL completion cert; correctness verified by the functional tier "
                                   "on the same fork-free ELF",
                            cycles=res.get("cycles"), derived_from_rtl=tier in cfg.rtl_tiers,
                            cycle_accurate=tier in cfg.rtl_tiers, timing=_tm)
                        continue
                    tiers[tier] = TierResult(
                        tier, "unavailable", mand,
                        reason="RTL cert ran to completion but cannot surface outputs for a mandatory "
                               "correctness check, and no functional tier established a numeric pass "
                               "(use the functional tier as the required gate)",
                        cycles=res.get("cycles"), derived_from_rtl=tier in cfg.rtl_tiers, timing=_tm)
                    continue
                _cg = _cp = None
                if perf_extractor is not None:
                    _cperf = perf_extractor(cb, res) or {}
                    _cg, _cp = _cperf.get("gflops"), _cperf.get("pct_fp_peak")
                if res.get("console") is not None:
                    (paths.artifacts_dir / f"{_sim_name}_console.log").write_text(
                        res["console"], encoding="utf-8")
                tiers[tier] = TierResult(
                    tier, "pass", mand,
                    reason="RTL completion + cycle-accurate perf cert (correctness gated by the "
                           "required functional tier)",
                    cycles=res.get("cycles"), derived_from_rtl=tier in cfg.rtl_tiers,
                    cycle_accurate=tier in cfg.rtl_tiers, evidence=f"{_sim_name}_console.log",
                    timing=_tm, gflops=_cg, pct_fp_peak=_cp)
                continue
            if oracle_graded:
                # Float grade: the RTL program-oracle output vs the capsule's golden (tolerance_float) —
                # the independent golden.yaml when it ships one, otherwise the recomputed golden, stamped
                # either way by ``gsource``. There is no integer reference/simulate to cross-check against —
                # this comparison IS the numeric verdict, recorded as the honest numeric report + evidence.
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
            if perf_extractor is not None:
                _perf = perf_extractor(cb, res) or {}
                _gflops = _perf.get("gflops")
                _pct_peak = _perf.get("pct_fp_peak")
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
            # Guarded on oracle_graded (a superset of independent_float) so both details also reach a
            # capsule graded against an independent oracle that is not the float path.
            _absent_detail = ((_absent_output_detail(onrep, sim_name, gold, res["outputs"])
                               or _unwritten_output_detail(onrep, sim_name))
                              if oracle_graded else None)
            _mismatch_reason = _absent_detail or (
                f"on {sim_name}, your emitted artifact does not compute the declared operation within tolerance"
                if oracle_graded
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
            tiers[tier] = TierResult(
                tier, "pass" if okt else "fail", mand,
                reason=None if okt else _mismatch_reason,
                cycles=res.get("cycles"), derived_from_rtl=_derived_from_rtl,
                cycle_accurate=(tier in cfg.rtl_tiers and okt), evidence=f"{sim_name}_console.log",
                timing=_tm, gflops=_gflops, pct_fp_peak=_pct_peak, fidelity=_fidelity)
            if res.get("console") is not None:
                (paths.artifacts_dir / f"{sim_name}_console.log").write_text(
                    res["console"], encoding="utf-8")
            # Only a MANDATORY/gold tier mismatch fails the capsule. An ADDITIVE lower-fidelity tier
            # (one not in required_oracle_tiers — e.g. a fast functional model with known approximation
            # gaps) records its fail in the tiers dict but must NOT abort: aborting here would pre-empt the
            # required RTL oracle that follows in the sorted ladder — the same "a cheaper check short-
            # circuits the authoritative oracle" class we fixed on the trace side, here on the oracle side.
            if not okt and mand:
                raise CertFailure(sim_name, _cat("FUNCTIONAL_MISMATCH"),
                                  _mismatch_reason + _encoding_divergence_hint(
                                      trace_check_res, oracle_graded,
                                      cb=cb, capsule=capsule, trace=decoded_trace))

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

    # Stamp each tier with the program the adapter built. Done here rather than at each construction site:
    # the branches above each build their own TierResult and `res` is out of scope after them. A block-
    # scaled MX capsule is graded on the harness's reference MX kernel, so this is what separates "the
    # submission passed" from "the fixture passed" in a finished score.
    from dataclasses import replace as _dc_replace
    for _t, _tc in _tier_toolchain.items():
        if _t in tiers and getattr(tiers[_t], "toolchain", None) is None:
            tiers[_t] = _dc_replace(tiers[_t], toolchain=_tc)

    return _finalize_capsule_result(
        name=name, capsule=capsule, status=status, failure=failure, tiers=tiers,
        trace_check_res=trace_check_res, numeric=numeric, required=required, no_oracle=no_oracle,
        eff_target=eff_target, paths=paths, run_id=run_id, cfg=cfg, contract=contract,
        # A BackendDeclined refusal is raised and recorded in THIS function, but the row is built in the
        # finalizer, so it has to be threaded across. It was not: the finalizer was lifted out of
        # run_capsule on one branch while the decline block was added inline on the other, and the
        # textually-clean merge left `declined` referenced in a scope that never binds it.
        declined=declined,
        executability=executability)


def _finalize_capsule_result(*, name: str, capsule: dict, status: str, failure: dict | None,
                             tiers: dict, trace_check_res: dict, numeric: dict, required, no_oracle: bool,
                             eff_target: str, paths: "RunPaths", run_id: str, cfg, contract,
                             executability: dict | None = None,
                             declined: dict | None = None,
                             extra: dict | None = None) -> dict:
    """The shared tail of every capsule grade: fail-closed gates, the result row, and self-validation.

    Lifted out of :func:`run_capsule` so the whole-model path shares the SAME invariants rather than a
    second copy of them. A model capsule used to return before any of this ran, which is precisely how it
    could report ``pass`` with an empty ``tiers`` block, no ``trace_check``, and a row that does not
    validate against ``capsule_result.schema.json`` -- with nothing to notice, because the validator sits
    on the far side of that early return.

    ``extra`` merges whole-model-only top-level keys (routing plan, coverage certificate, mesh execution,
    op coverage) without giving them a say in the status.
    """
    executability = executability or {}
    # `toolchain_shas` was a function-local import inside run_capsule; lifting this block to module level
    # took it out of scope. Import it here rather than at module level to keep the provenance import lazy,
    # as the original call site did.
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

    result = {
        "capsule": name, "kind": capsule.get("kind"), "label": capsule.get("label"),
        "status": status, "contract_version": CONTRACT_VERSION,
        "tiers": {t: r.to_dict() for t, r in tiers.items()},
        "trace_check": trace_check_res, "numeric": numeric,
        "failure": failure, "toolchain_shas": toolchain_shas(eff_target),
    }
    # Advisory RTL-executability smoke (never a gate): record it as its own field when one ran, so a
    # reader sees the RTL-legality backstop verdict without it ever touching the pass/fail status.
    if extra:
        result.update(extra)
    if executability:
        result["executability"] = executability
    # The refusal rides the result BY NAME AND SHAPE, so the round feedback can quote what was declined
    # rather than reporting a numeric mismatch on a program that was never emitted.
    if declined:
        result["declined"] = declined
    (paths.run_path / "capsule_result.json").write_text(json.dumps(result, indent=2),
                                                        encoding="utf-8")
    _write_run_manifest(paths, run_id, name, status, tiers, capsule, target=cfg.target, suite=cfg.suite)
    try:
        schemas.validate(result, "capsule_result", contract=contract)
    except schemas.ContractViolation as e:
        import sys
        sys.stderr.write(f"WARNING: capsule_result self-validation failed: {e}\n")
    return result


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

    def _one(cap: dict) -> dict:
        return run_capsule(cap, package_dir, runs_root=runs_root, run_id=cap["name"],
                           contract=contract, oracle_adapters=oracle_adapters,
                           pkg=pkg, timeout=timeout, target=target, suite=suite, dtype=dtype,
                           config=config, perf_extractor=perf_extractor, no_oracle=no_oracle)

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
            return head + list(ex.map(_one, rest))

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
    # The gate denominator answers "how much of what this device CAN do is working?" -- three independent
    # ways of getting it wrong, all measured, all guarded here.
    #
    # 1. A CRASHED op capsule (``error``) COUNTS. It produced no passing artifact, which is a failure to
    #    deliver one, not an absence of evidence. Leaving it out let a crashier failure mode UNLOCK the
    #    gate: across three rounds with identical 25/36 op passes, 5 crashes shrank the denominator to 31
    #    (0.806, open) while rounds with 0 and 2 crashes scored 0.694 and 0.735 (closed). The capstone
    #    swung on HOW the submission failed rather than on how much of the suite it passed.
    #    ``not_gradeable_no_oracle`` stays excluded -- there the ORACLE was absent, so the suite genuinely
    #    learned nothing about the submission.
    # 2. An INELIGIBLE op capsule does NOT count. An int8 systolic target graded 12 bf16 capsules its
    #    contract declares no capability for ("input dtype 'bf16' not in contraction formats ['int8']").
    #    They can never pass, so the best reachable fraction was 23/35 = 0.66 against a 0.8 gate: the
    #    whole-model capsules were MATHEMATICALLY unreachable and the only report was a repeated "gated".
    #
    # 3. A `screened_only` capsule does NOT count either, for the reason `not_graded` and `gated` do not:
    #    under a certify budget it was deliberately never measured against the certifying tier, by name
    #    and with its coverage guaranteed by the derived covering set. It is neither evidence for nor
    #    against. Unlike an ERROR -- which used to slip out of this denominator unexplained -- that is an
    #    opt-in choice whose members are listed on the score.
    #
    # These do not cancel: a crashed ELIGIBLE capsule still counts, so the crash loophole stays shut,
    # while a capsule the hardware cannot do, or that was never certified, is out either way. Naming the
    # three that count rather than the ones that do not is deliberate: a status added later must be
    # argued INTO the denominator instead of landing in it silently.
    graded = [r for r in op_results if r.get("status") in ("pass", "fail", "error")]
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
