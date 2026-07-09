"""Parallel Muon capsule runner -- the Muon analog of :mod:`capsule_runner`, Gemmini untouched.

This reuses the *generic, target-agnostic* helpers (oot_runner primitives, capsule_golden, schemas,
the 4-entrypoint sequence, the L0/L1 + not_run_is_not_pass invariants) but plugs in the Muon oracle
adapters (cyclotron / VCS) and the Muon tier->sim map. It imports nothing Gemmini-specific
(rocc_decode / trace_check / the gemmini backend / the shared capsule_runner) so the frozen Gemmini
path keeps working byte-for-byte.

Tier ladder (Muon):
    L0  independent numeric golden     capsule_golden vs reference(cb)
    L1  reference(cb) == simulate(cb)   cb internal consistency
    L2  cyclotron --timing             outputs == golden + perf cycles + %FP-peak  (primary)
    L3  VCS RadianceMuonConfig RTL      cycle-exact difftest cert (honest-unavailable while WIP)

Unlike the Gemmini runner, the package's ``lower_target_to_llvm`` entrypoint emits a **Muon SIMT C++
kernel** (not LLVM-dialect MLIR); the oracle adapter compiles it with clang-muon and runs it.
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
from .contract import schemas
# shared, target-agnostic capsule I/O (also re-exported for callers using MR.discover_capsules/load_capsule)
from .capsule_common import (_cat, _flat, discover_capsules, load_capsule,  # noqa: F401
                             make_run_paths, run_entrypoints)
from .muon_oracles import default_adapters
from .oot_runner import (CertFailure, Package, build_package, integrity_scan,
                         load_package, run_entrypoint)
from ..runtime.backends.muon import MuonUnavailable, FP_PEAK_GFLOPS

SUITE = "muon-perf-bench"
CONTRACT_VERSION = "0.1"
TARGET = "muon"

_TIER_SIM = {"L2": "cyclotron", "L3": "vcs"}
_RTL_TIERS = {"L3"}


@dataclasses.dataclass
class TierResult:
    tier: str
    status: str                       # pass | fail | skipped | unavailable
    mandatory: bool
    reason: str | None = None
    cycles: int | None = None
    gflops: float | None = None
    pct_fp_peak: float | None = None
    derived_from_rtl: bool = False
    cycle_accurate: bool = False
    evidence: str | None = None
    timing: dict | None = None

    def to_dict(self) -> dict:
        return {"status": self.status, "mandatory": self.mandatory,
                "not_run_is_not_pass": True, "reason": self.reason,
                "cycles": self.cycles, "gflops": self.gflops, "pct_fp_peak": self.pct_fp_peak,
                "derived_from_rtl": self.derived_from_rtl,
                "cycle_accurate": self.cycle_accurate, "evidence": self.evidence,
                "timing": self.timing}




def _match(a: dict, b: dict, atol: float = 1e-3) -> bool:
    """Output equality with fp tolerance (the device prints fixed-precision decimals)."""
    if set(a) != set(b):
        return False
    for k in a:
        fa, fb = _flat(a[k]), _flat(b[k])
        if len(fa) != len(fb):
            return False
        if any(abs(float(x) - float(y)) > atol for x, y in zip(fa, fb)):
            return False
    return True


def run_capsule(capsule: dict, package_dir: str | Path, *, runs_root: str | Path,
                run_id: str | None = None, contract: str | Path | None = None,
                oracle_adapters: dict[str, Callable] | None = None,
                pkg: Package | None = None, timeout: int = 600) -> dict:
    """Run one Muon capsule through the package; write artifacts; return a capsule_result dict."""
    from ..runtime.reference import reference_outputs
    from ..runtime.simulator import simulate
    from .muon_oracles import flops_from_cb

    name = capsule["name"]
    run_id = run_id or f"{name}"
    adapters = oracle_adapters if oracle_adapters is not None else default_adapters()
    required = set(capsule.get("required_oracle_tiers", []))

    paths = make_run_paths(runs_root, run_id, suite=SUITE, target=TARGET,
                           dtype="f32", benchmark=name)

    tiers: dict[str, TierResult] = {}
    numeric = {"status": "skipped"}
    failure: dict | None = None
    status = "pass"

    try:
        # shared front half: build + the 4 contract entrypoints; muon's 4th emits a SIMT C++ kernel.
        pkg, cb, kernel_src = run_entrypoints(pkg, package_dir, capsule, paths, contract=contract,
                                              timeout=timeout, fourth_output_name="kernel.cpp")

        # --- golden + L0/L1 ----------------------------------------------------------------
        gold = CG.golden(capsule)
        try:
            ref = reference_outputs(cb)
            sim = simulate(cb)["outputs"]
        except (ValueError, KeyError, IndexError, TypeError) as ce:
            raise CertFailure("command_buffer", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                              f"command buffer could not be interpreted ({type(ce).__name__}: {ce})") from ce
        nrep = CG.compare(gold, ref, capsule["numeric_policy"])
        numeric = {"status": nrep["status"], "policy": nrep["policy"],
                   "max_abs_diff": nrep["max_abs_error"], "max_rel_error": nrep["max_rel_error"],
                   "mismatch_count": nrep["mismatch_count"], "first_mismatch": nrep["first_mismatch"]}
        CG.write_numeric_report(paths.generated / "numeric_report.yaml", nrep)
        tiers["L0"] = TierResult("L0", "pass" if nrep["status"] == "pass" else "fail",
                                 mandatory=True, evidence="numeric_report.yaml",
                                 reason=None if nrep["status"] == "pass" else "golden != reference(cb)")
        if nrep["status"] != "pass":
            raise CertFailure("numeric_golden", _cat("FUNCTIONAL_MISMATCH"),
                              f"golden != reference(cb): {nrep['first_mismatch']}")

        l1_ok = _match(ref, sim)
        tiers["L1"] = TierResult("L1", "pass" if l1_ok else "fail", mandatory=True,
                                 reason=None if l1_ok else "reference(cb) != simulate(cb)")
        if not l1_ok:
            raise CertFailure("command_buffer_reference", _cat("FUNCTIONAL_MISMATCH"),
                              "reference(cb) != simulate(cb)")

        # --- oracle tiers L2..L3 -----------------------------------------------------------
        flops = flops_from_cb(cb)
        for tier in ("L2", "L3"):
            mand = tier in required
            adapter = adapters.get(tier)
            if adapter is None:
                if mand:
                    tiers[tier] = TierResult(tier, "unavailable", True,
                                             reason=f"no adapter for {tier} ({_TIER_SIM[tier]})",
                                             derived_from_rtl=tier in _RTL_TIERS)
                continue
            try:
                res = adapter(cb, kernel_src, paths.generated, timeout)
            except MuonUnavailable as e:
                tiers[tier] = TierResult(tier, "unavailable", mand, reason=str(e),
                                         derived_from_rtl=tier in _RTL_TIERS)
                continue
            except Exception as e:
                tiers[tier] = TierResult(tier, "fail", mand,
                                         reason=f"{_TIER_SIM[tier]} crash: {str(e)[-300:]}",
                                         derived_from_rtl=tier in _RTL_TIERS)
                if mand:
                    raise CertFailure(_TIER_SIM[tier], _cat("TOOL_CRASH"),
                                      f"{_TIER_SIM[tier]} invocation failed: {str(e)[-400:]}") from e
                continue
            okt = (_match(res["outputs"], gold) and _match(res["outputs"], ref)
                   and _match(res["outputs"], sim))
            sim_name = _TIER_SIM[tier]
            tiers[tier] = TierResult(
                tier, "pass" if okt else "fail", mand,
                reason=None if okt else f"{sim_name} oracle != golden==reference==simulate",
                cycles=res.get("cycles"), gflops=res.get("gflops"), pct_fp_peak=res.get("pct_fp_peak"),
                derived_from_rtl=res.get("oracle", {}).get("derived_from_rtl", tier in _RTL_TIERS),
                cycle_accurate=(tier in _RTL_TIERS and okt), evidence=f"{sim_name}_console.log",
                timing=res.get("timing"))
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
    except Exception as e:
        status = "error"
        failure = {"plane": "runner_internal", "category": "RUNNER_CRASH",
                   "detail": f"{type(e).__name__}: {e}", "traceback": _traceback.format_exc()}

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
        "status": status, "contract_version": CONTRACT_VERSION, "target": TARGET,
        "fp_peak_gflops": FP_PEAK_GFLOPS,
        "tiers": {t: r.to_dict() for t, r in tiers.items()},
        "numeric": numeric, "failure": failure,
    }
    (paths.run_path / "capsule_result.json").write_text(json.dumps(result, indent=2),
                                                        encoding="utf-8")
    _write_run_manifest(paths, run_id, name, status, tiers, capsule)
    return result


def _write_run_manifest(paths: RunPaths, run_id: str, name: str, status: str,
                        tiers: dict, capsule: dict) -> None:
    manifest = {
        "schema_version": "1.0", "project": "merlin", "suite": SUITE, "method": run_id,
        "run_id": run_id, "target": TARGET, "benchmark": name, "status": status,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "codegen_backend": "oot_package",
        "metadata": {
            "kind": capsule.get("kind"), "label": capsule.get("label"),
            "required_oracle_tiers": capsule.get("required_oracle_tiers", []),
            "tier_status": {t: r.status for t, r in tiers.items()},
            "cycles": {t: r.cycles for t, r in tiers.items() if r.cycles is not None},
            "pct_fp_peak": {t: r.pct_fp_peak for t, r in tiers.items() if r.pct_fp_peak is not None},
        },
    }
    (paths.run_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")








def run_suite(capsules: list[dict], package_dir: str | Path, *, runs_root: str | Path,
              contract: str | Path | None = None,
              oracle_adapters: dict[str, Callable] | None = None, timeout: int = 600) -> list[dict]:
    pkg = load_package(package_dir, contract=contract)
    integrity_scan(pkg)
    build_package(pkg)
    return [run_capsule(cap, package_dir, runs_root=runs_root, run_id=cap["name"],
                        contract=contract, oracle_adapters=oracle_adapters, pkg=pkg, timeout=timeout)
            for cap in capsules]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="muon capsule/perf runner")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsule", help="path to a single capsule dir")
    ap.add_argument("--capsules-root", help="run every capsule under this root")
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--runs-root", default="runs/muon_perf_bench")
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
