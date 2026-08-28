#!/usr/bin/env python3
"""Run AutoComp's real search on a study capsule, graded by this study's evaluator.

AutoComp is the kernel-generation baseline. Its search -- beam over plan/code candidates, its own
prompt construction, its own model tiering -- is used UNMODIFIED: rewriting it would produce a
strawman opponent and a result about my scaffolding rather than about kernel generation.

Exactly one thing is substituted: the eval backend. Everything the search measures now comes from the
same evaluator the compiler arm uses, at the same fidelity, with the same definition of correct. That
is the difference between comparing two methods and comparing two oracles.

The two arms this drives are the metered ones -- Bedrock and the Google API -- because AutoComp calls
provider APIs. The ChatGPT seat is not an API and cannot be driven this way, which is a real asymmetry
the study records rather than hides.

Run it under AutoComp's own venv (it needs boto3, google-genai and the framework itself):

    /scratch/agustin/projects/autocomp/.venv/bin/python run_autocomp.py --method bedrock_kernel \\
        --capsule merlin/contract/capsules/radiance/isa/R0_gemm_fp32 --run-id ac_qwen_R0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

import kvc_eval  # noqa: E402
from autocomp_bridge import KvcMuonEvalBackend  # noqa: E402

#: AutoComp addresses a model as "<provider>::<id>". Mapping the study's provider names onto that
#: keeps ONE source of truth for which model an arm runs -- the method config -- rather than a second
#: copy here that could drift out of step with the routing gate.
_PROVIDER_PREFIX = {"bedrock": "aws", "google": "gcp"}


def _autocomp_model(cfg: dict) -> str:
    provider = cfg.get("provider", "")
    if provider not in _PROVIDER_PREFIX:
        raise SystemExit(
            f"provider {provider!r} cannot drive AutoComp: it calls provider APIs, and the "
            f"subscription seat is not one. Use the direct-agent runner for that arm."
        )
    model = cfg["model"]
    # opencode writes "google/gemini-x"; AutoComp wants the bare id under its own prefix.
    if "/" in model:
        model = model.split("/", 1)[1]
    return f"{_PROVIDER_PREFIX[provider]}::{model}"


def _task_context(capsule_dir: Path) -> str:
    """What AutoComp tells its planner and coder about the task. No reference answer."""
    iface = (capsule_dir / "capsule.interface.mlir").read_text()
    return f"""Write ONE compute kernel for a SIMT accelerator, as LLVM-dialect MLIR.

The capsule interface below names the operands, their shapes and dtypes, and the order of operations:

```mlir
{iface}
```

ABI: a single `llvm.func` named `radiance_kernel`, parameters are plain pointers in the order
[weight tensors] ++ [inputs in command order] ++ [outputs in command order]. Read with
llvm.getelementptr + llvm.load, write with llvm.store.

**Every operand is 32-bit float in memory, whatever dtype the interface declares.** The harness
decodes a declared f16 / bf16 / fp8 operand to its numeric value and stores it as one `f32` per
element, so a 16x32 f16 tensor arrives as 512 consecutive `f32` values -- NOT as packed halves.
Load `f32`, compute in `f32`, store `f32`. The declared dtype tells you the numeric PRECISION the
values were rounded to; it does not describe the memory layout. Warps, barriers and scheduling come from the
runtime, so emit plain scalar compute. No .insn, no raw .word, no DRAM addresses, no halt.

You are graded by compiling the kernel with a stock toolchain and running it on a simulator, then
comparing its outputs against a reference you cannot see. Correctness first: an incorrect kernel earns
no latency credit at all. Once correct, reduce cycles AND raise utilization -- the feedback reports
per-unit busy fractions, so aim at the unit that is idle rather than guessing.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True)
    ap.add_argument("--capsule", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--target", default="radiance")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--plans", type=int, default=2)
    ap.add_argument("--codes", type=int, default=2)
    ap.add_argument("--fidelity", default="fast", choices=sorted(kvc_eval.FIDELITY_ENV))
    ap.add_argument("--runs-root", type=Path, default=Path("/scratch/agustin/tmp/kvc-runs"))
    a = ap.parse_args(argv)

    import yaml
    cfg = yaml.safe_load((EXP / "methods" / a.method / "method.yaml").read_text())

    # The routing policy is a launch gate, not advice.
    import check_method_models as policy
    bad = policy.violations(cfg, where=a.method)
    if bad:
        for b in bad:
            print(f"  !! {b}")
        raise SystemExit("routing policy violation -- refusing to launch")

    if cfg.get("aws_region"):
        os.environ["AWS_REGION"] = str(cfg["aws_region"])
    if cfg.get("provider") == "google":
        os.environ.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    repo = kvc_eval.repo_root()
    capsule_dir = a.capsule if a.capsule.is_absolute() else repo / a.capsule
    run_dir = a.runs_root / "autocomp" / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shim_pkg = run_dir / "shim"
    shim_pkg.mkdir(parents=True, exist_ok=True)
    for f in ("kernelshim", "manifest.yaml"):
        shutil.copy2(EXP / "shim" / f, shim_pkg / f)

    missing = kvc_eval.missing_requirements(a.fidelity)
    if missing:
        raise SystemExit(f"fidelity {a.fidelity!r} is missing {missing} -- refusing to run rather "
                         f"than silently produce a lower-fidelity number")

    from autocomp.search.prob import Prob
    from autocomp.search.search import BeamSearchStrategy, create_backend_and_agents
    from autocomp.hw_config.muon_config import MuonHardwareConfig

    hw = MuonHardwareConfig()
    prob = Prob("muon", 0, context=_task_context(capsule_dir))
    model = _autocomp_model(cfg)

    # Build AutoComp's own agents, then DISCARD its eval backend for ours. The search, the prompts and
    # the model tiering stay exactly as the framework ships them.
    _their_backend, agent, code_agent = create_backend_and_agents(
        "muon", "built:radiance", hw, prob, [model], [model],
        # AutoComp's own settings for this target, copied from its run_search_muon so the search
        # behaves the way the framework ships it. A custom configuration here would make the baseline
        # something I tuned rather than something AutoComp does.
        menu_strategy="one-shot", fine_grained_isa=True, example_rate=0.25,
        cache_dir=str(run_dir),
    )
    eval_backend = KvcMuonEvalBackend(
        capsule_dir, shim_pkg=shim_pkg, runs_root=a.runs_root,
        target=a.target, fidelity=a.fidelity, hw_config=hw, method=a.method,
    )

    t0 = time.time()
    optimizer = BeamSearchStrategy(
        output_dir=run_dir, eval_backend=eval_backend, agent=agent, code_agent=code_agent,
        orig_code="", prob=prob, metric="latency", simulator="cyclotron",
        num_plan_candidates=a.plans, num_code_candidates=a.codes, beam_size=a.beam,
        # AutoComp's native muon settings, so the search is the framework's and not mine.
        give_score_feedback=1, give_hw_feedback=0,
        # ...with ONE deliberate change: AutoComp ships this OFF for muon because its own backend
        # reported no utilization. Ours does, and an optimizer told a kernel is slow without being
        # told which unit is idle can only guess. Recorded as a deviation rather than passed off as
        # the default.
        give_util_feedback=1,
        include_ancestors=False, plan_icl_examples=False, code_icl_examples=False,
        num_analyses=0, num_pairs_to_combine=0, num_gen_per_combine=0,
        dropout_menu_options=0.25,
        trigger_exhaustive_threshold=1, trigger_exhaustive_iters=20, start_exhaustive_iters=0,
        prevent_duplicate_level=0, reimplement_failed=True,
        translate_iters=0, translate_perf_threshold=15,
        translate_drop_original=True, translate_score=True,
    )
    status = "ok"
    try:
        optimizer.optimize(a.iterations)
    except Exception as e:                       # a search crash must still leave its trajectory
        status = f"{type(e).__name__}: {e}"
        print(f"  !! search stopped: {status}")

    correct = [t for t in eval_backend.trajectory if t.get("correct")]
    best = min((t["cycles"] for t in correct if t.get("cycles")), default=None)
    summary = {
        "run_id": a.run_id, "method": a.method, "driver": "autocomp",
        "provider": cfg.get("provider"), "model": cfg["model"],
        "autocomp_model": model, "billing_mode": cfg.get("billing_mode"),
        "task_id": capsule_dir.name, "target": a.target, "fidelity": a.fidelity,
        "status": status,
        "candidates_evaluated": len(eval_backend.trajectory),
        "candidates_correct": len(correct),
        "best_cycles": best,
        "wall_seconds": round(time.time() - t0, 1),
        "trajectory": eval_backend.trajectory,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  {a.method}/{capsule_dir.name}: {len(correct)}/{len(eval_backend.trajectory)} correct, "
          f"best={best} cycles, wall={summary['wall_seconds']}s")
    print(f"  -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
