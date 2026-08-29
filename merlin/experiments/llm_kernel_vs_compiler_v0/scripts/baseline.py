#!/usr/bin/env python3
"""Measure the reference backend on a set of capsules -- the number every arm's cycles are relative to.

WHY THIS HAD TO EXIST. Until it did, the study recorded absolute cycle counts and nothing else. An
absolute count answers no question a reader can check: 526,151 cycles for a 16x16x16 GEMM is neither
good nor bad without knowing what the same capsule costs through the hand-curated lowering. Worse, a
count with no baseline cannot even reveal that the measurement has stopped discriminating -- three
independently written kernels landing within 0.1% of each other reads as agreement rather than as a
resolution limit.

The reference is `out/artifacts/targets/muon/reference_v0`: a hand-curated, correct, UNOPTIMIZED
merlin_iface -> Muon SIMT lowering, and by its own manifest the ceiling an agentic backend has to
re-derive and beat. It is measured through the same runner, the same oracle and the same fidelity as
every arm, because a baseline produced by a different path would be measuring the path.

FIDELITY IS RECORDED, NOT ASSUMED. The functional model is an estimate; only the cycle-accurate tier
may be quoted as a measurement. Both the baseline and the arm it is compared against must come from
the same tier, so the tier is written into the record and `--fidelity` names it explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kvc_eval as KE  # noqa: E402  (same directory; the evaluator owns the fidelity table)

#: Where each pilot capsule lives. Capsules are split across `isa/` (compiler tests written against
#: the ISA) and `model_slices/` (slices lifted out of a real model), and the split is not derivable
#: from the name -- reading it wrong silently evaluates nothing.
DEFAULT_CAPSULES = {
    "R0_gemm_fp32": "isa/R0_gemm_fp32",
    "R4_rmsnorm_fp32": "model_slices/R4_rmsnorm_fp32",
    "R3_attention_qk_fp16": "model_slices/R3_attention_qk_fp16",
}


def measure(capsule_dir: Path, package: Path, runs_root: Path, *, target: str,
            fidelity: str, timeout: int) -> dict:
    """Run one capsule through a package and return its cycles, tier by tier.

    The package is invoked directly rather than through the kernel shim: the reference does its own
    lowering end to end, and substituting a kernel into it would measure the shim.
    """
    code = f'''
import json, pathlib
from merlin.runtime.backends.base import get_backend
MR = get_backend("muon").muon_capsule_runner
d = pathlib.Path({str(capsule_dir)!r})
r = MR.run_capsule(MR.load_capsule(d), {str(package)!r}, runs_root={str(runs_root)!r},
                   target={target!r}, timeout={timeout})
print("<<<BASE>>>" + json.dumps(r, default=str))
'''
    repo = KE.repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "merlin" / "python")
    env.setdefault("TMPDIR", "/scratch/agustin/tmp")
    env.update(KE.FIDELITY_ENV[fidelity])

    started = time.time()
    try:
        proc = subprocess.run([KE._evaluator_python(), "-c", code], capture_output=True, text=True,
                              env=env, cwd=str(repo), timeout=timeout + 300)
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": f"exceeded {timeout + 300}s", "fidelity": fidelity}

    for line in proc.stdout.splitlines():
        if line.startswith("<<<BASE>>>"):
            raw = json.loads(line[len("<<<BASE>>>"):])
            tiers = raw.get("tiers") or {}
            # The cycle count is taken from the certifying tier alone. A tier that only proves the
            # kernel ran cannot supply a latency the study is allowed to quote.
            cycles = None
            cycles_tier = None
            for name in sorted(tiers):
                if name in KE.CERTIFYING_TIERS and tiers[name].get("cycles") is not None:
                    cycles, cycles_tier = tiers[name]["cycles"], name
            return {
                "status": raw.get("status"),
                "cycles": cycles,
                "cycles_tier": cycles_tier,
                "cycles_cycle_accurate": fidelity == "cert",
                "fidelity": fidelity,
                "tier_status": {k: v.get("status") for k, v in tiers.items()},
                "numeric_status": (raw.get("numeric") or {}).get("status"),
                "golden_source": (raw.get("numeric") or {}).get("golden_source"),
                "wall_seconds": round(time.time() - started, 1),
            }
    return {"status": "error", "fidelity": fidelity,
            "reason": (proc.stderr or proc.stdout)[-600:]}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path,
                    default=None, help="package to measure (default: the muon reference backend)")
    ap.add_argument("--label", default="reference_v0",
                    help="what this baseline IS, recorded alongside the numbers")
    ap.add_argument("--runs-root", type=Path, default=Path("/scratch/agustin/tmp/kvc-runs/runs"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/agustin/tmp/kvc-runs/baselines.json"))
    ap.add_argument("--target", default="radiance")
    ap.add_argument("--fidelity", default="fast", choices=sorted(KE.FIDELITY_ENV))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tasks", nargs="*", default=sorted(DEFAULT_CAPSULES))
    args = ap.parse_args(argv)

    repo = KE.repo_root()
    package = args.package or (repo / "out/artifacts/targets/muon/reference_v0")

    missing = KE.missing_requirements(args.fidelity)
    if missing:
        # Fail loudly: silently degrading to the functional model would hand back an estimate that
        # then gets quoted as a cycle-accurate baseline.
        print(f"cannot run at fidelity {args.fidelity!r}; missing: {', '.join(missing)}",
              file=sys.stderr)
        return 2

    caps_root = repo / "merlin" / "contract" / "capsules" / args.target
    record = {
        "label": args.label,
        "package": str(package),
        "target": args.target,
        "fidelity": args.fidelity,
        "measured_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "tasks": {},
    }
    for task in args.tasks:
        rel = DEFAULT_CAPSULES.get(task)
        if rel is None:
            record["tasks"][task] = {"status": "error", "reason": "no capsule path known"}
            print(f"{task:24s} NO CAPSULE PATH", flush=True)
            continue
        result = measure(caps_root / rel, package, args.runs_root,
                         target=args.target, fidelity=args.fidelity, timeout=args.timeout)
        record["tasks"][task] = result
        print(f"{task:24s} {str(result.get('status')):6s} "
              f"cycles={result.get('cycles')} tier={result.get('cycles_tier')} "
              f"({result.get('wall_seconds')}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
