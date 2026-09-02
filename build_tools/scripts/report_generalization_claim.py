#!/usr/bin/env python3
"""State the generalization claim for a target, over the roster that target DECLARES.

    Across the declared roster, N of <core opset> PyTorch Core ATen operators appear; on target T, M of
    the classifiable regions route to the accelerator, and those account for P% of the roster's total
    loop-nest work.

:mod:`merlin.targetgen.aten_coverage` could compute this and nothing called it, so the number that
answers "how general is this compiler" was never produced by anything but a unit test.

THE DENOMINATOR IS THE DECLARED ROSTER, not the capture directory. ``workload_spec.models`` is the
target's statement of which workloads it is FOR; computing over whatever captures happen to be on disk
answers a question nobody asked and quietly shrinks when a capture goes missing. A roster member with no
capture is REPORTED, and with ``--require-roster`` it fails the run -- a claim resting on half its roster
must not read like a claim resting on all of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402
from merlin.targetgen import aten_coverage as AC  # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

_TARGETS = "merlin/experiments/capsule_bench/targets"


def _captures() -> dict[str, Path]:
    """Captured bundles available as evidence, keyed by model name.

    Same store and same key normalisation as `check_conformance_coverage`, so the claim and the
    requirement cannot disagree about which capture is which model.
    """
    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    out: dict[str, Path] = {}
    for d in sorted(root.iterdir()):
        m = d / "model.mlir"
        if m.is_file():
            out[d.name.replace("_fp32_consistent", "").replace("_consistent", "")] = m
    return out


def _roster(target: str) -> tuple[list[str], Path | None]:
    p = repo_root() / _TARGETS / target / "target_experiment.yaml"
    if not p.is_file():
        return [], None
    te = load_target_experiment(p)
    ws = dict(getattr(te, "workload_spec", None) or {})
    return [str(m) for m in (ws.get("models") or ())], p


def _for_target(target: str) -> dict:
    roster, descriptor = _roster(target)
    if descriptor is None:
        return {"target": target, "status": "no_descriptor"}
    if not roster:
        return {"target": target, "status": "no_roster",
                "detail": "the descriptor declares no workload_spec.models, so this claim has no "
                          "tracked denominator; add one rather than defaulting to the capture directory"}
    available = _captures()
    # A roster member matches a capture by prefix: one model has several precision variants, and every
    # one of them is evidence about that model.
    used: dict[str, Path] = {}
    missing: list[str] = []
    for model in roster:
        hits = {k: v for k, v in available.items() if k == model or k.startswith(model + "_")}
        if hits:
            used.update(hits)
        else:
            missing.append(model)
    if not used:
        return {"target": target, "status": "no_capture_for_any_roster_model",
                "roster": roster, "roster_without_capture": missing}
    report = AC.coverage(used, target)
    return {"target": target, "status": "ok", "roster": roster,
            "roster_without_capture": missing,
            "captures_used": sorted(used),
            "claim": AC.claim_sentence({**report, "n_models": len(used)}),
            "report": report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--require-roster", action="store_true",
                    help="fail when a declared roster model has no capture")
    a = ap.parse_args(argv)

    targets = a.target or sorted(
        p.name for p in (repo_root() / _TARGETS).iterdir()
        if (p / "target_experiment.yaml").is_file()) if (repo_root() / _TARGETS).is_dir() else []
    if not targets:
        print("no target with a descriptor found", file=sys.stderr)
        return 2

    results = [_for_target(t) for t in targets]
    if a.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for r in results:
            print(f"=== {r['target']}: {r['status']}")
            if r.get("claim"):
                print(f"    {r['claim']}")
            if r.get("roster_without_capture"):
                print(f"    [gap] roster models with no capture: {r['roster_without_capture']}")
            wk = ((r.get("report") or {}).get("work") or {})
            if wk.get("routed_fraction") is not None and not wk.get("exact"):
                print("    [note] work is a LOWER BOUND: at least one iteration nest was only "
                      "partially recovered")
    if a.require_roster and any(r.get("roster_without_capture") for r in results):
        print("\nFAIL: a declared roster model has no capture; the claim would rest on part of the "
              "roster while reading as though it rested on all of it", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
