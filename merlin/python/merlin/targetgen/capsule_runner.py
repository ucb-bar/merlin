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
import traceback as _traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from aet.core.run_paths import RunPaths

from . import capsule_golden as CG
from . import rocc_decode as RD
from . import trace_check as TCK
from .contract import compile as oot_compile
from .contract import schemas
# shared, target-agnostic capsule I/O (also re-exported: callers use CR.discover_capsules/load_capsule)
from .capsule_common import (_cat, _flat, discover_capsules, load_capsule,  # noqa: F401
                             make_run_paths, run_entrypoints)
from .oot_runner import (CertFailure, Package, build_package, integrity_scan,
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
    not_applicable: bool = False      # tier honestly N/A for this capsule's datatype (e.g. the integer
                                      # L0/L1 floor on a float datapath) — a legitimate skip, not a
                                      # not_run_is_not_pass violation (unlike an unavailable RTL oracle)

    def to_dict(self) -> dict:
        d = {"status": self.status, "mandatory": self.mandatory,
             "not_run_is_not_pass": True, "reason": self.reason,
             "cycles": self.cycles, "derived_from_rtl": self.derived_from_rtl,
             "cycle_accurate": self.cycle_accurate, "evidence": self.evidence,
             "timing": self.timing}
        if self.not_applicable:
            d["not_applicable"] = True
        # perf fields ride the result ONLY when populated (SIMT) — keeps systolic output byte-identical.
        if self.gflops is not None:
            d["gflops"] = self.gflops
        if self.pct_fp_peak is not None:
            d["pct_fp_peak"] = self.pct_fp_peak
        return d


# An oracle adapter: (cb, llvm_text, workdir, timeout) -> {outputs, cycles, oracle, console}
# or it raises OracleUnavailable to signal honest unavailability.
class OracleUnavailable(Exception):
    pass


def _spike_verilator_adapter(sim: str, target: str) -> Callable:
    def run(cb, llvm_text, workdir, timeout):
        from ..runtime.backends import base as _bk
        gem = _bk.get_backend(target)  # this target's backend owns the sim-availability probe
        if not gem.available(sim):
            raise OracleUnavailable(f"{sim} not available")
        return oot_compile.run_on_oracle(cb, llvm_text, simulator=sim,
                                         workdir=workdir, timeout=timeout)
    return run


def mlc_arc_adapter(target: str) -> Callable:
    """The DEFAULT cross-target RTL oracle: run the command buffer on ``target``'s mlc ARC model (the
    RTL-derived functional model — bit-exact datapath outputs + cycle count from the arc state), for ANY
    target mlc compiled from RTL — NO bespoke sim toolchain needed. This is what makes grading generalize
    across targets. Raises OracleUnavailable if mlc / the arc model is absent for the target."""
    def run(cb, llvm_text, workdir, timeout):
        from .rtl import mlc_bridge
        if not mlc_bridge.arc_available(target):
            raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")
        res = mlc_bridge.arc_run_command_buffer(cb)
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


def _cyclotron_adapters(target: str) -> dict[str, Callable]:
    """A self-hosted SIMT target graded on its emitted kernel ELF by the bespoke cyclotron/VCS oracle
    (``muon_oracles``) — NOT the arc command-buffer path, which grades the wrong artifact for a SIMT
    kernel. The muon adapters fail closed (MuonUnavailable) when the MERLIN_MUON_* toolchain env is
    unset, so an unwired target degrades honestly, never mis-grades."""
    from . import muon_oracles as _MO
    return _MO.default_adapters()


def _cyclotron_available(target: str) -> tuple[bool, str]:
    """cyclotron (SIMT): require the cyclotron oracle; the mlc arc command-buffer adapter grades the wrong
    artifact for a SIMT kernel, so it is NOT a valid fallback (that was a false-green). Fail closed."""
    try:
        from ..runtime.backends import muon as _muon
        if _muon.available("cyclotron"):
            return True, f"{target!r}: cyclotron SIMT oracle available"
    except Exception:  # noqa: BLE001 — an unimportable backend is honestly unavailable
        pass
    return False, (f"{target!r}: cyclotron SIMT oracle unavailable (set the MERLIN_MUON_* env); the mlc "
                   "arc command-buffer adapter grades the wrong artifact for a SIMT target and is not "
                   "a valid fallback")


#: DECLARED bespoke-sim oracle registry, keyed by sim ENGINE (``sim_via``) — the seam that keeps oracle
#: routing target-name-free (a new sim engine registers here; the dispatch below is untouched).
_SIM_ORACLES: dict[str, _SimOracle] = {
    "chipyard": _SimOracle(lambda t: _sim_engine_adapters("chipyard", t), _chipyard_available,
                           exclusive=False, has_memmap=True, is_compile_based=True),
    "cyclotron": _SimOracle(_cyclotron_adapters, _cyclotron_available, exclusive=True),
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


def _ensure_sim_oracles_discovered() -> None:
    """Load any bespoke-sim oracle a target contributes via its contract ``plugin.sim_oracle`` — a module
    that calls :func:`register_sim_oracle` at import — through the SAME OOT/reference plugin discovery the
    runtime backends use. This is what WIRES the registry: a new target adds its oracle as DATA (a plugin
    path in its contract), never a core edit to the ``_SIM_ORACLES`` literal. Re-scans only when
    ``MERLIN_TARGET_PATH`` changes; registration is idempotent, so repeated scans are harmless."""
    global _sim_oracle_env_seen
    import os
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _sim_oracle_env_seen:
        return
    _sim_oracle_env_seen = key
    try:
        from ..runtime.backends import base as _bk
        for name, path in _bk._oot_plugin_modules("sim_oracle"):
            _bk._load_oot_backend(name, path, ns="merlin._oot_sim_oracles")
    except Exception:  # noqa: BLE001 — discovery is best-effort; a broken plugin must not break routing
        pass


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
    # command_buffer / arc-default target (+ an optional ADDITIVE declared bespoke sim, e.g. chipyard)
    if so is not None:
        return so.available(target)
    if mlc_bridge.arc_available(target):
        return True, f"{target!r}: mlc arc oracle available"
    return False, (f"{target!r}: mlc arc model unavailable (set MERLIN_MLC_DIR and build the arc model)")


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
    try:
        from .isa_model import isa_model_for_target
        if not isa_model_for_target(target).is_fixed_format():
            return True, "n/a (ISA is not fixed-format — no fork-free re-encode smoke for this emit path)"
    except Exception as e:  # noqa: BLE001 — no derived model -> nothing to smoke here
        return True, f"n/a (no fixed-format ISA model: {str(e)[-120:]})"
    if _bespoke_sim_via(target) != "cyclotron":
        return True, "n/a (fixed-format ISA but no cyclotron reference sim declared for the fork-free smoke)"
    try:
        from ..runtime.backends import muon as _muon
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
    return {"L2": _spike_verilator_adapter("spike", "gemmini"),      # target-ok: explicitly gemmini-only back-compat
            "L3": _spike_verilator_adapter("verilator", "gemmini")}  # target-ok: explicitly gemmini-only back-compat


def qa_loop_adapters(target: str, sim_via: str | None = None) -> dict[str, Callable]:
    """The FAST per-round QA-loop oracle set for ``target`` — resolved from :func:`oracle_adapters`, never
    hardwired. It keeps ONLY the lowest (fastest) RTL oracle tier and reserves the slower cycle-accurate
    tiers for the bounded checkpoint (:func:`qa_checkpoint_adapters`). This is a tier-order distinction, not
    a per-target one: a chipyard target's fastest tier is spike (L2) so verilator (L3) is held back; an
    arc/mlc target's single RTL-derived tier (L3) is already fast, so IT is the loop tier. No target-name
    branch — a new accelerator's loop gate falls out of its declared ``sim_via`` with no edit here."""
    full = oracle_adapters(target, sim_via)
    if not full:
        return {}
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


def _grade_model_capsule(capsule: dict, *, target: str | None = None, timeout: int) -> dict:
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
    run_where = os.environ.get("MERLIN_MODEL_GRADE_RUN", "host")
    # MERLIN_MESH_VERIFY additionally EXECUTES each mesh-routed matmul as a single systolic tile on the
    # target's real mesh oracle (compile_model mesh_verify) — proving the matmul layers run ON the mesh, not
    # just that a routing plan was produced. Off by default (the oracle build/run is heavy); the whole-model
    # functional gate stays compile_rvv either way.
    mesh_verify = os.environ.get("MERLIN_MESH_VERIFY", "").lower() in ("1", "true", "yes", "on")
    result: dict = {"capsule": capsule["name"], "kind": "model", "label": capsule.get("label"),
                    "operation": {"op": "model", "model": model, "dtype": dtype, "run": run_where,
                                  "target": target},
                    "contract_version": CONTRACT_VERSION}
    if not model:
        result.update(status="incomplete",
                      failure={"plane": "model", "category": "RUNNER_CRASH",
                               "detail": "model capsule missing operation.attributes.model"})
        return result
    # the captured linalg (visible grounding) drives the per-op mesh routing when present
    linalg_mlir = None
    cdir = capsule.get("__dir__")
    if cdir:
        lp = _P(cdir) / "capsule.linalg.mlir"
        if lp.is_file():
            linalg_mlir = lp.read_text(encoding="utf-8")
    try:
        from ..compile_cli import compile_model
        out = compile_model(model, dtype, target=target, run=run_where, verify=True, package=None,
                            auto_capture=True, timeout=timeout, linalg_mlir=linalg_mlir,
                            mesh_verify=mesh_verify)
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
    st, gate = out.get("status"), (out.get("verify") or {}).get("gate_ok")
    engine = f"merlin-compile model --target {target} --run {run_where} --verify"
    if out.get("routing_plan") is not None:                   # per-op mesh routing for the target
        result["routing_plan"] = out["routing_plan"]
    if out.get("mesh_execution") is not None:                 # per-tile on-mesh EXECUTION for the target
        result["mesh_execution"] = out["mesh_execution"]
    # A quantized whole model diverges from the fp32 golden BY DESIGN (int8/fp8 rounding). When the strict
    # gate rejects it only on that expected drop, accept it if the cosine similarity to the golden still
    # clears a quant floor (MERLIN_MODEL_QUANT_COS, default 0.90) — reasonable quantization error is a pass,
    # a gross codegen defect (cosine below the floor / structural mismatch) is still a fail.
    _v = out.get("verify") or {}
    _cos = max(float(_v.get("fp32_cos", 0.0)), float(_v.get("w8a8_cos", 0.0)))
    _quant = dtype in ("int8", "i8", "fp8", "fp8_e4m3")
    _quant_floor = float(os.environ.get("MERLIN_MODEL_QUANT_COS", "0.90") or "0.90")
    if st == "verified" and gate:
        result.update(status="pass", numeric={"status": "pass", "engine": engine, "gate": out.get("verify")})
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
    from ..runtime.reference import reference_outputs
    from ..runtime.simulator import simulate
    from .eval.gemmini_suite import toolchain_shas

    # The effective target comes from the config (authoritative — cfg.target drives the run) when one is
    # supplied, else the explicit ``target`` argument. If NEITHER is given we refuse to run rather than
    # silently defaulting to gemmini (the OV2 rule: no core path silently operates on gemmini).
    eff_target = config.target if config is not None else target
    if eff_target is None:
        raise ValueError("run_capsule requires a target (or a config carrying one); "
                         "no default target is assumed")
    cfg = config or _config_for_target(eff_target, suite, dtype)
    # L1/oracle output equality uses the capsule's numeric_policy (integer -> exact), unless the config
    # forces one (a float/SIMT target grades its oracle output with tolerance regardless of the capsule).
    policy = cfg.force_match_policy or capsule.get("numeric_policy")
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
        result = _grade_model_capsule(capsule, target=eff_target, timeout=timeout)
        (paths.run_path / "capsule_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    tiers: dict[str, TierResult] = {}
    trace_check_res = {"status": "skipped", "violations": []}
    decoded_trace: dict | None = None            # kept for the advisory divergence localizer (D2)
    numeric = {"status": "skipped"}
    failure: dict | None = None
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
        _vals = CG.canonical_input_values(capsule, capsule_dir)
        if _vals:
            cb["canonical_inputs"] = _vals

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
                _absent_cb = _absent_output_detail(nrep, "command_buffer", gold, ref)
                raise CertFailure("numeric_golden", _cat("FUNCTIONAL_MISMATCH"),
                                  _absent_cb or ("your command buffer does not compute the declared operation "
                                  f"(first divergence at index {(nrep['first_mismatch'] or {}).get('index')})"))

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
        # The muon (SIMT/cyclotron/VCS) adapters raise their OWN MuonUnavailable — a THIRD unrelated class.
        # The VCS RTL difftest is honest-unavailable (WIP upstream: the DPI ELF path is not wired for arbitrary
        # corpus kernels), so it raises MuonUnavailable; without catching it here that honest unavailability
        # falls through to the generic handler and is mislabeled a TOOL_CRASH (an agent FAIL) instead of
        # firing the unavailable -> not_gradeable_no_oracle path — the same bug the program-oracle catch above
        # fixed for atlas. Catch it too so an unavailable RTL oracle is honest for the SIMT endpoint as well.
        from ..runtime.backends.muon import MuonUnavailable as _MuonUnavailable
        _ORACLE_UNAVAILABLE = (OracleUnavailable, _POUnavailable, _MuonUnavailable)
        for tier in sorted(set(cfg.oracle_tiers) | set(adapters or {})):
            mand = tier in required
            adapter = (adapters or {}).get(tier)
            if adapter is None:
                if mand:
                    tiers[tier] = TierResult(tier, "unavailable", True,
                                             reason=f"no adapter for {tier} ({cfg.tier_sim.get(tier, '?')})",
                                             derived_from_rtl=tier in cfg.rtl_tiers)
                continue
            import time as _time
            _adapter_t0 = _time.perf_counter()
            try:
                res = adapter(cb, llvm_text, paths.generated, timeout)
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
                    reason=(f"did not halt (ran to the cycle cap): {_msg[-240:]}" if _did_not_halt
                            else f"kernel faulted at runtime: {_msg[-260:]}" if _trapped
                            else f"{_sim} crash: {_msg[-300:]}"),
                    derived_from_rtl=tier in cfg.rtl_tiers)
                if mand:
                    if _did_not_halt:
                        raise CertFailure(_sim, _cat("TIMEOUT"),
                                          f"{_msg}; the emitted kernel never reached the ISA's "
                                          "halt/terminate instruction — emit it as the final instruction "
                                          "on every control path (see the program-termination contract). "
                                          "Numerics are never checked until the program halts.") from e
                    if _trapped:
                        raise CertFailure(_sim, _cat("FUNCTIONAL_MISMATCH"),
                                          "the emitted kernel RAN but FAULTED on "
                                          f"{_sim} (a trap — an illegal instruction or an out-of-range "
                                          "memory access — before producing output). Check that every "
                                          "memory-movement instruction uses a valid, in-range DRAM address "
                                          "(derive it from the passed pointer args / the declared DRAM "
                                          f"layout, never a baked 0 or a guessed address): {_msg[-300:]}") from e
                    raise CertFailure(_sim, _cat("TOOL_CRASH"),
                                      f"{_sim} invocation failed: {_msg[-400:]}") from e
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
            _absent_detail = (_absent_output_detail(onrep, sim_name, gold, res["outputs"])
                              if independent_float else None)
            _mismatch_reason = _absent_detail or (
                f"on {sim_name}, your emitted artifact does not compute the declared operation within tolerance"
                if independent_float
                else f"on {sim_name}, your emitted artifact does not compute the declared operation")
            # ``oracle`` may be a rich dict (gemmini spike/verilator: {derived_from_rtl, ...}) OR a plain
            # provenance string (the arc / program cosim returns e.g. "atlas-arc-arcilator-cosim"); default
            # to the tier's RTL classification when it doesn't declare derived_from_rtl.
            _oracle_meta = res.get("oracle")
            _derived_from_rtl = (_oracle_meta.get("derived_from_rtl", tier in cfg.rtl_tiers)
                                 if isinstance(_oracle_meta, dict) else (tier in cfg.rtl_tiers))
            tiers[tier] = TierResult(
                tier, "pass" if okt else "fail", mand,
                reason=None if okt else _mismatch_reason,
                cycles=res.get("cycles"), derived_from_rtl=_derived_from_rtl,
                cycle_accurate=(tier in cfg.rtl_tiers and okt), evidence=f"{sim_name}_console.log",
                timing=_tm, gflops=_gflops, pct_fp_peak=_pct_peak)
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
                                      trace_check_res, independent_float,
                                      cb=cb, capsule=capsule, trace=decoded_trace))

    except CertFailure as cf:
        status = "fail"
        cat = cf.category.value if hasattr(cf.category, "value") else str(cf.category)
        failure = {"plane": cf.plane, "category": cat, "detail": cf.detail}
    except Exception as e:  # internal harness bug
        status = "error"
        failure = {"plane": "runner_internal", "category": "RUNNER_CRASH",
                   "detail": f"{type(e).__name__}: {e}",
                   "traceback": _traceback.format_exc()}

    # not_run_is_not_pass: a mandatory tier that did not pass closed (unavailable/skipped/absent) makes
    # the capsule incomplete — never a silent pass. A tier that is honestly N/A for this capsule's
    # datatype (``not_applicable``; the integer L0/L1 floor on a float datapath) is the ONE exception —
    # a legitimate skip like a dropped RoCC gate, not a missing oracle. An unavailable/absent RTL oracle
    # is never not_applicable, so it still fails closed here.
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
                        failure = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS",
                                   "detail": f"mandatory tier {tier} did not run "
                                             f"({tr.status if tr else 'absent'})"}
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
                failure = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS",
                           "detail": f"no runnable required oracle tier certified this capsule "
                                     f"(required={sorted(required)}) — refusing to pass on the L0/L1 "
                                     f"command-buffer interpretation alone"}

    result = {
        "capsule": name, "kind": capsule.get("kind"), "label": capsule.get("label"),
        "status": status, "contract_version": CONTRACT_VERSION,
        "tiers": {t: r.to_dict() for t, r in tiers.items()},
        "trace_check": trace_check_res, "numeric": numeric,
        "failure": failure, "toolchain_shas": toolchain_shas(),
    }
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
    pkg = load_package(package_dir, contract=contract)
    integrity_scan(pkg)
    build_package(pkg)

    def _one(cap: dict) -> dict:
        return run_capsule(cap, package_dir, runs_root=runs_root, run_id=cap["name"],
                           contract=contract, oracle_adapters=oracle_adapters,
                           pkg=pkg, timeout=timeout, target=target, suite=suite, dtype=dtype,
                           config=config, perf_extractor=perf_extractor, no_oracle=no_oracle)

    def _run_all(caps: list[dict]) -> list[dict]:
        if max_workers <= 1:
            return [_one(c) for c in caps]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, caps))

    # A whole-model (kind == "model") capsule is the GATED capstone: it is scheduled only after the op
    # suite proves itself (its ``gate.after_op_pass_fraction`` of the graded op capsules passed). Grade the
    # op capsules first; if no model capsule is present this is exactly the original single-pass behavior.
    from .capsule_source import model_gate_satisfied
    op_caps = [c for c in capsules if c.get("kind") != "model"]
    model_caps = [c for c in capsules if c.get("kind") == "model"]
    op_results = _run_all(op_caps)
    if not model_caps:
        return op_results
    graded = [r for r in op_results if r.get("status") in ("pass", "fail")]
    frac = (sum(1 for r in graded if r.get("status") == "pass") / len(graded)) if graded else 0.0
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
