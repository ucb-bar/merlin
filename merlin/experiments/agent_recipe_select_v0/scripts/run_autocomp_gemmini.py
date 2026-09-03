"""The AutoComp arm on gemmini: its search, its prompts, its model tiering — our workload and oracle.

WHAT IS HELD CONSTANT ACROSS THE TWO ARMS, so the comparison means something: the workload shape, the
correctness oracle (bit-exact against a host golden over identical deterministic stimulus), the timing
oracle (GSIM on `GemminiGsimSerialClkConfig`), and the planning model (codex `gpt-5.6-sol`, the same
seat the recipe arm uses).

WHAT DELIBERATELY DIFFERS — the ablation itself: the recipe arm chooses among compiler-defined points
and cannot emit a broken candidate; this arm rewrites arbitrary C and can. Every consequence of that
is measured rather than assumed.

FOUR DEVIATIONS FROM STOCK AUTOCOMP, each recorded on the run:

1. **its eval backend is replaced.** AutoComp's own `GemminiEvalBackend` scores `latency_spike_raw`
   — spike cycles — and spike does not model Gemmini timing (counts plateau ~120 from 4K to 2M MACs).
   Optimising against it would be optimising against noise. It also hardcodes a `chipyard-mx`
   checkout. This runner never instantiates it: the agents are constructed directly so that class is
   not imported at all.
2. **`give_hw_feedback=0`.** No per-candidate hardware counters are wired for this arm, and the
   backend returns an EMPTY feedback list rather than an invented one, so the flag must stay off.
3. **models are tiered.** `models=` plans, `code_models=` implements — AutoComp's documented split,
   which the repo's existing muon bridge collapsed by passing the same model twice.
4. **the LLM is a subscription seat**, reached through a monkeypatched `codex` provider. Its own
   deviations (no temperature control; num_samples costs that many fresh sessions; dollars notional)
   are recorded by `autocomp_codex.stats()`.

⚠️ **The two arms do not share a baseline, and a "speedup" must say which.** AutoComp starts from the
canonical Gemmini C library (`tiled_matmul_auto`), measured at 1089 cycles on 32x32x32; the recipe arm
starts from the frozen MLIR compiler's default at 780 on the same shape and oracle. The compiler is
already 1.40x faster than the C library before anything is tuned. So the primary comparison is
ABSOLUTE CYCLES, and each arm's own-baseline speedup is reported separately with its starting point
named.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402
import autocomp_codex as CX                                           # noqa: E402
from autocomp_gemmini_backend import GsimGemminiEvalBackend           # noqa: E402

# NOT `start_run`/`finish_run`: those lazily import `aet`, which AutoComp's venv does not have, and
# installing it there would change an interpreter other sessions share. The run directory is created
# to the same convention here, and `record_spend.py` registers it with aet from merlin's own venv --
# so the aet run is still real, just opened from the side of the seam that owns aet.
from merlin.common.artifacts import utc_stamp, git_sha7                # noqa: E402

#: Shapes shared with the recipe arm, so the arms are compared on identical work. The v0 three are
#: kept as named controls; the model census supplies the rest through `--shape`, because the arms
#: must see the SAME extents and the recipe arm now takes its shapes from ResNet-50 and TinyLlama.
SHAPES = {"w1_small": (32, 32, 32), "w2_medium": (64, 64, 64), "w3_n_heavy": (16, 512, 256)}


def _shape_from_workload_mlir(path) -> "tuple[str, tuple[int, int, int]]":
    """Read (M, K, N) out of a merlin_iface workload, so one file feeds BOTH arms.

    The recipe arm consumes the .mlir directly; AutoComp writes C and needs the extents. Deriving
    them from the same file is what makes "identical work" checkable rather than a convention two
    scripts are each trusted to honour.
    """
    from pathlib import Path as _P                                     # noqa: PLC0415
    text = _P(path).read_text(encoding="utf-8")
    m = k = n = None
    for line in text.splitlines():
        if 'role = "input"' in line:
            core = line[line.rindex("<") + 1:line.rindex(">")].split("x")
            m, k = int(core[0]), int(core[1])
        elif 'role = "weight"' in line:
            core = line[line.rindex("<") + 1:line.rindex(">")].split("x")
            n = int(core[1])
    if None in (m, k, n):
        raise SystemExit(f"could not read M/K/N from {path}")
    return _P(path).stem, (m, k, n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", default="", choices=sorted(SHAPES) + [""],
                    help="a named control shape shared with v0")
    ap.add_argument("--workload-mlir", default="",
                    help="a census workload; its extents are read from the same file the recipe arm "
                         "compiles, so both arms provably see one shape")
    ap.add_argument("--plan-model", default="codex/gpt-5.6-sol:high",
                    help="the EXPENSIVE tier: planning. Same seat model the recipe arm uses.")
    ap.add_argument("--code-model", default="codex/gpt-5.3-codex-spark:low",
                    help="the CHEAP tier: implementation. AutoComp's split is models= plans and "
                         "code_models= implements, and the repo's earlier bridge collapsed both to "
                         "one. This subscription serves seven models (see autocomp_codex."
                         "KNOWN_MODELS); gpt-5.3-codex-spark is the ultra-fast coding one, so the "
                         "tiers differ in MODEL and in effort, not effort alone.")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--plans", type=int, default=4)
    ap.add_argument("--codes", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--budget-candidates", type=int, default=16,
                    help="match the recipe arm's evaluation budget")
    ap.add_argument("--eval-timeout", type=int, default=3600)
    ap.add_argument("--engine", default="gsim")
    ap.add_argument("--method", default="autocomp_codex")
    a = ap.parse_args(argv)
    T.assert_frozen_intact()
    # AutoComp logs to Weights & Biases and its search ABORTS without a wandb key. Telemetry for this
    # study is aet (plus the per-candidate CSV), and an external tracker must not be able to gate a
    # measurement -- nor should this experiment's data be shipped to a third party as a side effect
    # of running it. Disabled explicitly rather than by hoping a key is absent.
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    if a.workload_mlir:
        wl_name, (M, K, N) = _shape_from_workload_mlir(a.workload_mlir)
        a.workload = wl_name
    elif a.workload:
        M, K, N = SHAPES[a.workload]
    else:
        raise SystemExit("one of --workload or --workload-mlir is required")
    # ⚠️ THE RUN ID MUST NAME THE WORKLOAD. Shapes are launched concurrently, so a stamp built from
    # time+method+seed+sha alone COLLIDES: three shapes launched in the same second resolved to one
    # directory, and they were writing each other's generated ELF and results. Measured here, and it
    # is the second time this experiment has hit a run-id collision -- the first collapsed 20 recipe
    # points onto 10 directories while still reporting them as measured.
    wl_tag = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in a.workload)[-48:]
    run_id = f"{utc_stamp()}_{a.method}_{wl_tag}_seed000_{git_sha7()}"
    run_dir = T.RUNS / run_id
    for sub in ("logs", "metrics", "artifacts_dir", "generated", "candidates"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "run_record.json").write_text(json.dumps(
        {"run_id": run_id, "suite": T.SUITE, "target": T.TARGET, "method": a.method, "seed": 0,
         "timestamp": utc_stamp(), "git_sha": git_sha7(), "arm": "autocomp",
         "workload": a.workload, "shape": [M, K, N],
         "tiers": {"plan": {"spec": a.plan_model,
                            "model": CX.split_model(a.plan_model)[0],
                            "effort": CX.split_model(a.plan_model)[1]},
                   "code": {"spec": a.code_model,
                            "model": CX.split_model(a.code_model)[0],
                            "effort": CX.split_model(a.code_model)[1]}},
         "tiering_note": ("the seat serves one model, so the tiers differ on reasoning effort; "
                          "per-tier spend is in the run summary's by_tier block"),
         "plan_model": a.plan_model, "code_model": a.code_model, "engine": a.engine,
         "interpreter": "autocomp venv (the oracle runs under merlin's, across the seam)"},
        indent=1), encoding="utf-8")

    CX.install(home=run_dir / "codex_home", log=run_dir / "codex_calls.jsonl",
               tier_names={a.plan_model: "plan", a.code_model: "code"})

    # Import AFTER the patch so the client class is already taught the provider.
    sys.path.insert(0, str(Path(T.REPO).parent / "autocomp"))
    from autocomp.agents.gemmini.gemmini_agent import GemminiLLMAgent
    from autocomp.agents.llm_ensemble import LLMEnsemble
    from autocomp.hw_config.gemmini_config import GemminiHardwareConfig
    from autocomp.search.prob import Prob
    from autocomp.search.search import BeamSearchStrategy

    # Hardware facts DERIVED, then asserted against what AutoComp will be told, so the agent is never
    # briefed on a machine we are not measuring.
    from merlin.perf.workload_gen import tile_geometry
    from merlin.targetgen.rtl import facts as rtl_facts
    geom = tile_geometry(T.TARGET)
    mems = (rtl_facts.load_facts(T.TARGET).get("facts") or {}).get("memories") or []
    spad_kb = next(m["bytes"] for m in mems if m["name"] == "scratchpad") // 1024
    acc_kb = next(m["bytes"] for m in mems if m["name"] == "accumulator") // 1024
    hw = GemminiHardwareConfig(pe_dim=geom.rows, spad_size_kb=spad_kb, acc_size_kb=acc_kb)
    print(f"hw (derived): pe_dim={geom.rows} spad={spad_kb}KB acc={acc_kb}KB", flush=True)

    backend = GsimGemminiEvalBackend(M=M, K=K, N=N, workdir=run_dir / "candidates",
                                     simulator=a.engine, timeout=a.eval_timeout)
    seed = backend.seed_source()
    (run_dir / "seed_kernel.c").write_text(seed, encoding="utf-8")

    # Measure the seed on OUR oracle before the search: an arm's baseline must come from the same
    # engine in the same run, never from a number recorded elsewhere.
    base_res = backend.evaluate_code(None, [seed])[0]
    if not base_res.get("correct"):
        (run_dir / "FAILED_SEED.txt").write_text(str(base_res.get("stderr")), encoding="utf-8")
        raise SystemExit(f"seed kernel failed: {base_res.get('stderr')[:400]}")
    baseline = base_res["latency"]
    print(f"seed (canonical gemmini C library) = {baseline} cycles, correct", flush=True)

    prob = Prob("gemmini", 0, test_file=run_dir / "seed_kernel.c",
                context=f"C = A({M}x{K}) x B({K}x{N}), int8 operands, int32 accumulate.")
    plan_agent = LLMEnsemble([GemminiLLMAgent(a.plan_model, hw, backend)])
    code_agent = LLMEnsemble([GemminiLLMAgent(a.code_model, hw, backend)]) \
        if a.code_model and a.code_model != a.plan_model else None

    t0 = time.time()
    err = None
    try:
        opt = BeamSearchStrategy(
            output_dir=run_dir, eval_backend=backend, agent=plan_agent, code_agent=code_agent,
            orig_code=seed, prob=prob, metric="latency", simulator=a.engine,
            num_plan_candidates=a.plans, num_code_candidates=a.codes, beam_size=a.beam,
            # Every value below is AutoComp's OWN gemmini setting from
            # `autocomp/search/run_search.py:63-108`, copied verbatim so this arm is the framework's
            # search and not a configuration of mine. Notably `give_util_feedback` and
            # `give_hw_feedback` are 0 in AutoComp's own gemmini config, so the fact that no counters
            # are wired for this arm costs nothing -- it agrees with the framework rather than
            # deviating from it.
            give_score_feedback=1, give_util_feedback=0, give_hw_feedback=0,
            include_ancestors=False, plan_icl_examples=False, code_icl_examples=False,
            num_analyses=0, num_pairs_to_combine=0, num_gen_per_combine=0,
            dropout_menu_options=0.25,
            trigger_exhaustive_threshold=1, trigger_exhaustive_iters=20,
            start_exhaustive_iters=0, prevent_duplicate_level=0,
            reimplement_failed=False,
            translate_iters=0, translate_perf_threshold=15,
            translate_drop_original=True, translate_score=True,
        )
        opt.optimize(a.rounds)   # AutoComp's own entry point (run_search.py:237)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"search raised: {err}", flush=True)
    wall = time.time() - t0

    traj = backend.trajectory
    good = [r for r in traj if r.get("correct") and isinstance(r.get("cycles"), int)]
    best = min((r["cycles"] for r in good), default=None)
    cx = CX.stats()
    summary = {
        "arm": "autocomp", "workload": a.workload, "shape": {"M": M, "K": K, "N": N},
        "plan_model": a.plan_model, "code_model": a.code_model, "engine": a.engine,
        "engine_config": T.GSIM_CONFIG, "engine_note": T.ENGINE_NOTE,
        "baseline_source": "canonical gemmini C library tiled_matmul_auto (NOT the frozen MLIR "
                           "compiler default — the arms start from different code paths)",
        "baseline_cycles": baseline, "best_cycles": best,
        "speedup_vs_own_baseline": round(baseline / best, 4) if best else None,
        # Every candidate the backend saw, including ones the beam discarded: AutoComp's own
        # trajectory records survivors only, and a discarded candidate still cost tokens.
        "candidates_evaluated": len(traj),
        "compile_failures": sum(1 for r in traj if not r.get("compiled")),
        "incorrect": sum(1 for r in traj if r.get("compiled") and not r.get("correct")),
        "invalid_total": sum(1 for r in traj if not r.get("correct")),
        "wall_seconds": round(wall, 1),
        "oracle_seconds": round(sum(r.get("eval_seconds") or 0 for r in traj), 1),
        "llm": cx, "search_error": err, "trajectory": traj,
    }
    # A search that raised produced no arm. Saying so on the record is the difference between "the
    # framework found nothing better" and "the framework never ran" -- the earlier validation run
    # exited 0 with best == seed for exactly this reason.
    summary["arm_completed"] = err is None and len(traj) > 1
    if not summary["arm_completed"]:
        summary["do_not_cite"] = ("the search did not run to completion, so best_cycles is the SEED "
                                 "and this is not an AutoComp result")
    (run_dir / "autocomp_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nseed {baseline} -> best {best} "
          f"({summary['speedup_vs_own_baseline']}x vs its OWN baseline) in "
          f"{len(traj)} candidates ({summary['compile_failures']} did not compile, "
          f"{summary['incorrect']} compiled but wrong)")
    print(f"llm: {cx['calls']} calls, {cx['tokens_total']} tokens, billed none (seat)")
    (run_dir / "run_record.json").write_text(json.dumps(
        {**json.loads((run_dir / "run_record.json").read_text()),
         "status": "ok" if err is None else "error",
         "summary": {"best_cycles": best, "candidates": len(traj),
                     "tokens_total": cx["tokens_total"]}}, indent=1), encoding="utf-8")
    print(f"run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
