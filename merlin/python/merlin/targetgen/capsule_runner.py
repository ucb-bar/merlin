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

# tier -> simulator name understood by runtime.backends.gemmini / adapters
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


def _spike_verilator_adapter(sim: str) -> Callable:
    def run(cb, llvm_text, workdir, timeout):
        from ..runtime.backends import gemmini as gem
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


def oracle_adapters(target: str = "gemmini", sim_via: str | None = None) -> dict[str, Callable]:
    """The oracle adapters per tier for a target. The mlc ARC model is the DEFAULT RTL tier (works for
    ANY mlc target, no bespoke sim); a target that DECLARES a bespoke sim (``sim_via``) additionally gets
    its higher-fidelity sim tiers (chipyard -> spike L2 / verilator L3), preserving the gemmini path.

    A self-hosted-ISA target (``endpoint_kind == external_backend``, e.g. atlas) is graded by the generic
    PROGRAM oracle (assemble the emitted kernel via the target's own assembler -> its mlc cosim) instead of
    the command_buffer arc path — routed from the contract, no target-name branch."""
    endpoint_kind, model_ext = _endpoint_of(target)
    if endpoint_kind == "external_backend":
        from .program_oracle import program_oracle_adapter
        return {"L3": program_oracle_adapter(target, model_ext=model_ext or "npu_model")}
    adapters: dict[str, Callable] = {"L3": mlc_arc_adapter(target)}   # arc default (RTL-derived)
    if sim_via == "chipyard":                                         # optional bespoke sim (gemmini)
        adapters["L2"] = _spike_verilator_adapter("spike")
        adapters["L3"] = _spike_verilator_adapter("verilator")
    return adapters


def default_adapters() -> dict[str, Callable]:
    """Back-compat default (gemmini): L2/L3 via the chipyard contract oracle. New callers should use
    :func:`oracle_adapters` with the target's ``sim_via`` (arc default + optional bespoke sim)."""
    return {"L2": _spike_verilator_adapter("spike"),
            "L3": _spike_verilator_adapter("verilator")}


def qa_loop_adapters(target: str = "gemmini", sim_via: str | None = None) -> dict[str, Callable]:
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


def qa_checkpoint_adapters(target: str = "gemmini", sim_via: str | None = None) -> dict[str, Callable]:
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


def run_capsule(capsule: dict, package_dir: str | Path, *, runs_root: str | Path,
                run_id: str | None = None, contract: str | Path | None = None,
                oracle_adapters: dict[str, Callable] | None = None,
                pkg: Package | None = None, timeout: int = 600,
                target: str = "gemmini", suite: str | None = None, dtype: str = "i8xi8_i32",
                config=None, perf_extractor: Callable | None = None) -> dict:
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

    cfg = config or _config_for_target(target, suite, dtype)
    # L1/oracle output equality uses the capsule's numeric_policy (integer -> exact), unless the config
    # forces one (a float/SIMT target grades its oracle output with tolerance regardless of the capsule).
    policy = cfg.force_match_policy or capsule.get("numeric_policy")
    name = capsule["name"]
    run_id = run_id or f"{name}"
    adapters = oracle_adapters if oracle_adapters is not None else default_adapters()
    required = set(capsule.get("required_oracle_tiers", []))

    paths = make_run_paths(runs_root, run_id, suite=cfg.suite, target=cfg.target,
                           dtype=cfg.dtype, benchmark=name)

    tiers: dict[str, TierResult] = {}
    trace_check_res = {"status": "skipped", "violations": []}
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

        if independent_float:
            # The integer reference/simulate engines cannot execute a float (fp8/bf16) datapath, so the
            # integer L0-reference / L1-sim numeric floor is INAPPLICABLE — skipped honestly (not failed),
            # exactly as a non-RoCC target drops the RoCC trace gate. The real numeric grade rides the RTL
            # program-oracle output vs the independent golden.yaml (tolerance_float), scored in the oracle
            # loop below. These skips are marked not_applicable so not_run_is_not_pass does NOT flip them to
            # incomplete (only an unavailable/absent RTL oracle does).
            ref = sim = None
            numeric = {"status": "skipped", "policy": policy.get("compare"), "golden_source": gsource,
                       "note": "integer reference/simulate N/A for float datapath; graded vs independent "
                               "golden at RTL oracle"}
            tiers["L0"] = TierResult(
                "L0", "skipped", mandatory="L0" in required, not_applicable=True,
                reason="integer reference not applicable to float datapath; graded vs independent "
                       f"golden ({gsource}) at the RTL oracle")
            tiers["L1"] = TierResult(
                "L1", "skipped", mandatory="L1" in required, not_applicable=True,
                reason="integer simulate not applicable to float datapath")
        else:
            # Interpreting the AGENT's command buffer (reference/simulate) can fail if the cb is
            # structurally invalid — e.g. a MATMUL operand with rank != 2 because conv2d was not lowered
            # to a 2D im2col matmul. That is the agent's bug, NOT a harness crash: report it as a
            # gradeable command_buffer failure with an actionable reason (so the agent gets feedback and
            # both arms are scored identically) instead of letting it become a RUNNER_CRASH.
            try:
                ref = reference_outputs(cb)
                sim = simulate(cb)["outputs"]
            except (ValueError, KeyError, IndexError, TypeError) as ce:
                raise CertFailure(
                    "command_buffer", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                    f"command buffer could not be interpreted by reference/simulate "
                    f"({type(ce).__name__}: {ce}); check operand ranks/shapes — a MATMUL operand likely "
                    f"has the wrong rank (expected 2D; conv2d must be lowered to a 2D im2col matmul)"
                ) from ce
            nrep = CG.compare(gold, ref, capsule["numeric_policy"], golden_source=gsource)
            numeric = {"status": nrep["status"], "policy": nrep["policy"],
                       "max_abs_diff": nrep["max_abs_error"], "max_rel_error": nrep["max_rel_error"],
                       "mismatch_count": nrep["mismatch_count"], "first_mismatch": nrep["first_mismatch"]}
            CG.write_numeric_report(paths.generated / "numeric_report.yaml", nrep)
            tiers["L0"] = TierResult("L0", "pass" if nrep["status"] == "pass" else "fail",
                                     mandatory="L0" in required or True,
                                     reason=None if nrep["status"] == "pass" else "golden != reference(cb)",
                                     evidence="numeric_report.yaml")
            if nrep["status"] != "pass":
                raise CertFailure("numeric_golden", _cat("FUNCTIONAL_MISMATCH"),
                                  f"golden != reference(cb): {nrep['first_mismatch']}")

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
            trace = RD.decode_text(llvm_text, source=str(paths.generated / cfg.fourth_output_name))
            schemas.validate(trace, "instruction_trace", contract=contract)
            (paths.generated / "instruction_trace.json").write_text(
                json.dumps(trace, indent=2), encoding="utf-8")
            trace_check_res = TCK.check(trace, capsule.get("expected", {}), cb=cb)
            if trace_check_res["status"] != "pass":
                raise CertFailure("trace_check", _cat("PROTOCOL_VIOLATION"),
                                  f"trace_check failed: {trace_check_res['violations']}")

        # --- oracle tiers -------------------------------------------------------------------
        # Run every tier the config declares (tier_sim ladder) OR an injected adapter provides — so a
        # target whose RTL tier is supplied by an adapter rather than a static tier_sim (atlas: arc L3,
        # empty tier_sim) still runs, while gemmini's declared ladder is unchanged. The L0/L1 math floor
        # is handled above (reference/simulate), NOT here. Sorted for a stable L2<..<L5 ladder order.
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
            except OracleUnavailable as e:
                tiers[tier] = TierResult(tier, "unavailable", mand, reason=str(e),
                                         derived_from_rtl=tier in cfg.rtl_tiers)
                continue
            except Exception as e:  # adapter crash is a real failure for that tier
                _sim = cfg.tier_sim.get(tier, tier)
                tiers[tier] = TierResult(tier, "fail", mand,
                                         reason=f"{_sim} crash: {str(e)[-300:]}",
                                         derived_from_rtl=tier in cfg.rtl_tiers)
                if mand:
                    raise CertFailure(_sim, _cat("TOOL_CRASH"),
                                      f"{_sim} invocation failed: {str(e)[-400:]}") from e
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
            if independent_float:
                # Float grade: the RTL program-oracle output vs the INDEPENDENT golden.yaml (tolerance_float).
                # There is no integer reference/simulate to cross-check against — this comparison IS the
                # numeric verdict, recorded as the honest numeric report + evidence.
                onrep = CG.compare(gold, res["outputs"], policy, golden_source=gsource)
                okt = onrep["status"] == "pass"
                numeric = {"status": onrep["status"], "policy": onrep["policy"],
                           "golden_source": gsource, "max_abs_diff": onrep["max_abs_error"],
                           "max_rel_error": onrep["max_rel_error"],
                           "mismatch_count": onrep["mismatch_count"],
                           "first_mismatch": onrep["first_mismatch"]}
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
            _mismatch_reason = (f"{sim_name} oracle != independent golden ({gsource})" if independent_float
                                else f"{sim_name} oracle != golden==reference==simulate")
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
            if not okt:
                raise CertFailure(sim_name, _cat("FUNCTIONAL_MISMATCH"), _mismatch_reason)

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
                status = "incomplete"
                if failure is None:
                    failure = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS",
                               "detail": f"mandatory tier {tier} did not run "
                                         f"({tr.status if tr else 'absent'})"}
                break

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
                        tiers: dict, capsule: dict, *, target: str = "gemmini",
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
              target: str = "gemmini", suite: str | None = None, dtype: str = "i8xi8_i32",
              config=None, perf_extractor: Callable | None = None) -> list[dict]:
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
                           config=config, perf_extractor=perf_extractor)

    if max_workers <= 1:
        return [_one(cap) for cap in capsules]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, capsules))  # order preserved by ex.map


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="capsule_bench_v0 runner")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsule", help="path to a single capsule dir")
    ap.add_argument("--capsules-root", help="run every capsule under this root")
    ap.add_argument("--labels", default="public,dev", help="comma-separated label filter")
    ap.add_argument("--runs-root", default="out/runs/capsule_bench")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)

    if a.capsule:
        caps = [load_capsule(a.capsule, contract=a.contract)]
    else:
        labels = set(a.labels.split(",")) if a.labels else None
        caps = discover_capsules(a.capsules_root, labels=labels, contract=a.contract)
    results = run_suite(caps, a.package, runs_root=a.runs_root, contract=a.contract,
                        timeout=a.timeout)
    npass = sum(1 for r in results if r["status"] == "pass")
    for r in results:
        print(f"  [{r['status']:10s}] {r['capsule']}")
    print(f"\n{npass}/{len(results)} pass")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
