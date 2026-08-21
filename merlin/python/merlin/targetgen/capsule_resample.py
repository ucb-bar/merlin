"""Grade a FROZEN package on a HELD-OUT input draw (A.1) — a post-freeze peer to ``capsule_grade --hidden``.

A capsule's ``golden.yaml`` is the answer key for ONE input draw: the public
:func:`vortex_oracle.operand_seed` fill, which anyone who knows the capsule's name can reproduce. So a
kernel that ignores its inputs and writes that (derivable) constant passes L2/L3 — on the PUBLIC
capsules the agent iterates against, which the hidden set does not cover. This re-grades the frozen
package on inputs drawn from a per-run SECRET ``salt`` chosen here, OUTSIDE the agent sandbox and never
written into the tree the agent compiled against, and compares against a reference recomputed by
interpreting each capsule's own interface IR (the authoritative contract) on those same inputs. A
constant-folding / memorised-golden kernel fails; a kernel that actually computes from its inputs passes.

Why this needs no device rebuild and no C++ change: the compiled kernel is INPUT-INDEPENDENT (inputs are
runtime data the host driver fills from the plan's per-operand ``seed``), so this reuses the frozen
package's own emitted kernel — run its contract entrypoints to get the LLVM module, build the image, run
it with a SALTED plan, compare to the resampled reference. No agent, no sandbox; the same freeze/handoff
boundary the hidden grade uses.

Run OUTSIDE the agent sandbox (needs the device sim + readable capsules + goldens). ``--no-oracle``
checks only that the held-out inputs + reference are producible for every capsule and that they actually
differ from the committed draw (so the resample has teeth) — no device, which is what CI can run.

Usage:
    python -m merlin.targetgen.capsule_resample --package <pkg> \
        --capsules merlin/contract/capsules --labels public,dev --tier L2 \
        --runs-root <out> [--salt <hex>] [--score <out.json>]
    python -m merlin.targetgen.capsule_resample --capsules merlin/contract/capsules \
        --labels public --no-oracle           # dry: verify the held-out machinery, no device
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path

import yaml

from . import capsule_golden as CG
from . import vortex_interp as VI
from . import vortex_oracle as VO
from .capsule_common import load_capsule, make_run_paths, run_entrypoints
from .oot_runner import CertFailure

RESAMPLE_GOLDEN_SOURCE = "vortex_resampled_interp"


def _discover_readable(root: str | Path, labels: set[str], contract=None) -> list[dict]:
    """Every capsule under ``root`` matching ``labels``, SKIPPING unreadable dirs.

    Unlike :func:`capsule_common.discover_capsules` (which ``rglob``s and dies on a ``chmod 000`` dir),
    this walks with the unreadable subtrees pruned. That matters here: resample is often run with
    ``--labels public`` while ``hidden/`` is still locked, and reading a public draw must not require
    traversing the locked hidden set. When ``hidden`` IS unlocked (post-freeze), it is included normally.
    """
    caps = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames
                       if os.access(os.path.join(dirpath, d), os.R_OK | os.X_OK)]
        if "capsule.yaml" in filenames:
            cap = load_capsule(dirpath, contract=contract)
            if cap.get("label") in labels:
                caps.append(cap)
    return sorted(caps, key=lambda c: c["name"])


def _committed_outputs(capsule_dir: str | Path) -> dict:
    gy = yaml.safe_load((Path(capsule_dir) / "golden.yaml").read_text())
    return gy.get("outputs", {})


def _fraction_differing(a: dict, b: dict) -> float:
    """Fraction of output elements that differ between two golden dicts (held-out vs committed).

    A value near 0 means this capsule's output barely depends on its inputs, so a held-out draw gives
    A.1 little to bite on — worth surfacing rather than hiding.
    """
    total = diff = 0
    for k, va in a.items():
        vb = b.get(k, [])
        fa = va if not (va and isinstance(va[0], list)) else [x for r in va for x in r]
        fb = vb if not (vb and isinstance(vb[0], list)) else [x for r in vb for x in r]
        for x, y in zip(fa, fb):
            total += 1
            if x != y:
                diff += 1
    return (diff / total) if total else 0.0


def resample_one(cap: dict, *, salt: str, package_dir: str | Path | None = None, pkg=None,
                 cfg=None, runs_root: str | Path | None = None, contract=None,
                 tier: str = "L2", timeout: int = 900, oracle: bool = True) -> tuple[dict, object]:
    """Grade one capsule on the held-out draw; -> (result dict, pkg) (pkg reused across capsules)."""
    name = cap["name"]
    cdir = cap.get("__dir__")
    if not cdir:
        return {"capsule": name, "status": "error", "reason": "capsule has no __dir__"}, pkg

    gold = VI.resampled_reference(cdir, salt)                 # reference for THIS draw, from the IR
    result: dict = {"capsule": name, "label": cap.get("label"), "kind": cap.get("kind"), "salt": salt}

    # Diagnostic: does the held-out draw actually change the answer? (A.1 has no teeth where it doesn't.)
    committed = _committed_outputs(cdir)
    result["reference_differs_frac"] = round(_fraction_differing(gold, committed), 4)

    if not oracle:
        result["status"] = "prepared"
        return result, pkg

    driver = "rtlsim" if tier == "L3" else "simx"
    if not VO.available(tier):
        result["status"] = "unavailable"
        result["reason"] = f"vortex {driver} oracle unavailable"
        return result, pkg

    paths = make_run_paths(runs_root, name, suite=cfg.suite, target=cfg.target,
                           dtype=cfg.dtype, benchmark=name)
    try:
        pkg, _cb, llvm_text = run_entrypoints(
            pkg, package_dir, cap, paths, contract=contract, timeout=timeout,
            fourth_output_name=cfg.fourth_output_name, entrypoints=cfg.entrypoints)
        grid = VO.grid_from_module(llvm_text)
        plan = VO.plan_from_capsule(cap, grid=grid, salt=salt)          # HELD-OUT seeds
        vxbin = VO.build_image(llvm_text, paths.generated, timeout=min(timeout, 600))
        res = VO.run_image(vxbin, plan, paths.generated, driver=driver, timeout=timeout)
    except CertFailure as cf:
        result["status"] = "fail"
        result["failure"] = {"plane": cf.plane, "detail": cf.detail[:300]}
        return result, pkg
    except VO.VortexUnavailable as e:
        result["status"] = "unavailable"
        result["reason"] = str(e)
        return result, pkg

    rep = CG.compare(gold, res["outputs"], cap["numeric_policy"],
                     golden_source=RESAMPLE_GOLDEN_SOURCE)
    result["status"] = "pass" if rep["status"] == "pass" else "fail"
    result["numeric"] = {"policy": rep["policy"], "mismatch_count": rep["mismatch_count"],
                         "max_abs_diff": rep["max_abs_error"], "max_rel_error": rep["max_rel_error"],
                         "first_mismatch": rep["first_mismatch"]}
    result["cycles"] = res.get("cycles")
    return result, pkg


def resample_suite(package_dir: str | Path | None, *, capsules_root: str | Path,
                   labels: set[str], salt: str, tier: str = "L2", runs_root: str | Path | None = None,
                   contract=None, timeout: int = 900, oracle: bool = True) -> dict:
    """Re-grade every capsule under ``labels`` on the held-out ``salt`` draw; return a score dict."""
    caps = _discover_readable(capsules_root, labels, contract=contract)
    cfg = None
    if oracle:
        from .runner_config import runner_config_from_manifest
        from .target_experiment import load_capability_manifest
        cfg = runner_config_from_manifest(load_capability_manifest("vortex"))

    pkg = None
    per: list[dict] = []
    for cap in caps:
        r, pkg = resample_one(cap, salt=salt, package_dir=package_dir, pkg=pkg, cfg=cfg,
                              runs_root=runs_root, contract=contract, tier=tier,
                              timeout=timeout, oracle=oracle)
        per.append(r)

    graded = [r for r in per if r["status"] in ("pass", "fail")]
    toothless = [r["capsule"] for r in per if r.get("reference_differs_frac", 1.0) == 0.0]
    return {
        "task": "vortex-capsule-resample", "salt": salt, "tier": tier,
        "labels_graded": sorted(labels), "oracle": oracle,
        "n_capsules": len(per), "n_graded": len(graded),
        "n_pass": sum(1 for r in graded if r["status"] == "pass"),
        "n_prepared": sum(1 for r in per if r["status"] == "prepared"),
        "n_unavailable": sum(1 for r in per if r["status"] == "unavailable"),
        # capsules whose output does not change with the input draw: A.1 cannot distinguish a real
        # compiler from a constant on these, so flag them rather than let them read as "covered".
        "toothless_capsules": toothless,
        "per_capsule": per,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade a frozen package on a held-out input draw (A.1)")
    ap.add_argument("--package", help="frozen package dir (required unless --no-oracle)")
    # Vortex-specific (uses the linalg-IR interpreter), so point at the vortex corpus root — NOT the
    # shared merlin/contract/capsules parent, which also holds gemmini capsules this cannot interpret.
    ap.add_argument("--capsules", default="merlin/contract/capsules/vortex")
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--salt", default=None,
                    help="held-out salt; default: a fresh secrets.token_hex(8) recorded in the score")
    ap.add_argument("--tier", default="L2", choices=["L2", "L3"])
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--no-oracle", action="store_true",
                    help="verify held-out inputs+reference are producible (and have teeth); no device")
    ap.add_argument("--score", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args(argv)

    salt = a.salt or secrets.token_hex(8)
    oracle = not a.no_oracle
    if oracle and not a.package:
        ap.error("--package is required unless --no-oracle")
    if oracle and not a.runs_root:
        ap.error("--runs-root is required unless --no-oracle")

    score = resample_suite(a.package, capsules_root=a.capsules,
                           labels=set(a.labels.split(",")), salt=salt, tier=a.tier,
                           runs_root=a.runs_root, contract=a.contract, timeout=a.timeout, oracle=oracle)

    out = Path(a.score) if a.score else (Path(a.runs_root) / "score_resample.json" if a.runs_root
                                         else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    tooth = f" toothless={len(score['toothless_capsules'])}" if score["toothless_capsules"] else ""
    print(f"resample salt={salt} tier={a.tier} labels={score['labels_graded']}: "
          f"pass={score['n_pass']}/{score['n_graded']} prepared={score['n_prepared']} "
          f"unavailable={score['n_unavailable']}{tooth}" + (f" -> {out}" if out else ""))
    if not oracle:
        return 0 if score["n_prepared"] == score["n_capsules"] else 1
    return 0 if score["n_graded"] > 0 and score["n_pass"] == score["n_graded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
