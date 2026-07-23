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

    def to_dict(self) -> dict:
        return {"status": self.status, "mandatory": self.mandatory,
                "not_run_is_not_pass": True, "reason": self.reason,
                "cycles": self.cycles, "derived_from_rtl": self.derived_from_rtl,
                "cycle_accurate": self.cycle_accurate, "evidence": self.evidence,
                "timing": self.timing}


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


def oracle_adapters(target: str = "gemmini", sim_via: str | None = None) -> dict[str, Callable]:
    """The oracle adapters per tier for a target. The mlc ARC model is the DEFAULT RTL tier (works for
    ANY mlc target, no bespoke sim); a target that DECLARES a bespoke sim (``sim_via``) additionally gets
    its higher-fidelity sim tiers (chipyard -> spike L2 / verilator L3), preserving the gemmini path."""
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




def _exact_match(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(_flat(a[k]) == _flat(b[k]) for k in a)


def run_capsule(capsule: dict, package_dir: str | Path, *, runs_root: str | Path,
                run_id: str | None = None, contract: str | Path | None = None,
                oracle_adapters: dict[str, Callable] | None = None,
                pkg: Package | None = None, timeout: int = 600) -> dict:
    """Run one capsule through the package; write artifacts; return a capsule_result dict."""
    from ..runtime.reference import reference_outputs
    from ..runtime.simulator import simulate
    from .eval.gemmini_suite import toolchain_shas

    name = capsule["name"]
    run_id = run_id or f"{name}"
    adapters = oracle_adapters if oracle_adapters is not None else default_adapters()
    required = set(capsule.get("required_oracle_tiers", []))

    paths = make_run_paths(runs_root, run_id, suite=SUITE, target="gemmini",
                           dtype="i8xi8_i32", benchmark=name)

    tiers: dict[str, TierResult] = {}
    trace_check_res = {"status": "skipped", "violations": []}
    numeric = {"status": "skipped"}
    failure: dict | None = None
    status = "pass"

    try:
        # shared front half: build + the 4 contract entrypoints (parse/target/cb/llvm), validated.
        pkg, cb, llvm_text = run_entrypoints(pkg, package_dir, capsule, paths, contract=contract,
                                             timeout=timeout, fourth_output_name="lowered.llvm.mlir")

        # --- golden + L0/L1 -----------------------------------------------------------------
        gold = CG.golden(capsule)
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
        nrep = CG.compare(gold, ref, capsule["numeric_policy"])
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

        l1_ok = _exact_match(ref, sim)
        tiers["L1"] = TierResult("L1", "pass" if l1_ok else "fail", mandatory=True,
                                 reason=None if l1_ok else "reference(cb) != simulate(cb)")
        if not l1_ok:
            raise CertFailure("command_buffer_reference", _cat("FUNCTIONAL_MISMATCH"),
                              "reference(cb) != simulate(cb)")

        # --- trace gate ---------------------------------------------------------------------
        trace = RD.decode_text(llvm_text, source=str(paths.generated / "lowered.llvm.mlir"))
        schemas.validate(trace, "instruction_trace", contract=contract)
        (paths.generated / "instruction_trace.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8")
        trace_check_res = TCK.check(trace, capsule.get("expected", {}), cb=cb)
        if trace_check_res["status"] != "pass":
            raise CertFailure("trace_check", _cat("PROTOCOL_VIOLATION"),
                              f"trace_check failed: {trace_check_res['violations']}")

        # --- oracle tiers L2..L5 ------------------------------------------------------------
        for tier in ("L2", "L3", "L4", "L5"):
            mand = tier in required
            adapter = adapters.get(tier)
            if adapter is None:
                if mand:
                    tiers[tier] = TierResult(tier, "unavailable", True,
                                             reason=f"no adapter for {tier} ({_TIER_SIM[tier]})",
                                             derived_from_rtl=tier in _RTL_TIERS)
                continue
            import time as _time
            _adapter_t0 = _time.perf_counter()
            try:
                res = adapter(cb, llvm_text, paths.generated, timeout)
            except OracleUnavailable as e:
                tiers[tier] = TierResult(tier, "unavailable", mand, reason=str(e),
                                         derived_from_rtl=tier in _RTL_TIERS)
                continue
            except Exception as e:  # adapter crash is a real failure for that tier
                tiers[tier] = TierResult(tier, "fail", mand,
                                         reason=f"{_TIER_SIM[tier]} crash: {str(e)[-300:]}",
                                         derived_from_rtl=tier in _RTL_TIERS)
                if mand:
                    raise CertFailure(_TIER_SIM[tier], _cat("TOOL_CRASH"),
                                      f"{_TIER_SIM[tier]} invocation failed: {str(e)[-400:]}") from e
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
            okt = _exact_match(res["outputs"], gold) and _exact_match(res["outputs"], ref) \
                and _exact_match(res["outputs"], sim)
            sim_name = _TIER_SIM[tier]
            tiers[tier] = TierResult(
                tier, "pass" if okt else "fail", mand,
                reason=None if okt else f"{sim_name} oracle != golden==reference==simulate",
                cycles=res.get("cycles"), derived_from_rtl=res.get("oracle", {}).get(
                    "derived_from_rtl", tier in _RTL_TIERS),
                cycle_accurate=(tier in _RTL_TIERS and okt), evidence=f"{sim_name}_console.log",
                timing=_tm)
            if res.get("console") is not None:
                (paths.artifacts_dir / f"{sim_name}_console.log").write_text(
                    res["console"], encoding="utf-8")
            if not okt:
                raise CertFailure(sim_name, _cat("FUNCTIONAL_MISMATCH"),
                                  f"{sim_name} oracle != golden==reference==simulate")

    except CertFailure as cf:
        status = "fail"
        cat = cf.category.value if hasattr(cf.category, "value") else str(cf.category)
        failure = {"plane": cf.plane, "category": cat, "detail": cf.detail}
    except Exception as e:  # internal harness bug
        status = "error"
        failure = {"plane": "runner_internal", "category": "RUNNER_CRASH",
                   "detail": f"{type(e).__name__}: {e}",
                   "traceback": _traceback.format_exc()}

    # not_run_is_not_pass: a mandatory tier that did not pass closed (unavailable/skipped/absent)
    if status == "pass":
        for tier in required:
            tr = tiers.get(tier)
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
    _write_run_manifest(paths, run_id, name, status, tiers, capsule)
    try:
        schemas.validate(result, "capsule_result", contract=contract)
    except schemas.ContractViolation as e:
        import sys
        sys.stderr.write(f"WARNING: capsule_result self-validation failed: {e}\n")
    return result


def _write_run_manifest(paths: RunPaths, run_id: str, name: str, status: str,
                        tiers: dict, capsule: dict) -> None:
    manifest = {
        "schema_version": "1.0", "project": "merlin", "suite": SUITE, "method": run_id,
        "run_id": run_id, "target": "gemmini", "benchmark": name, "status": status,
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
              timeout: int = 600, max_workers: int = 1) -> list[dict]:
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
                           pkg=pkg, timeout=timeout)

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
