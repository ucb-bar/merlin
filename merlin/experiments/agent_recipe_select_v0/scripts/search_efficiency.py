"""Is the agent actually SEARCHING, or is the space just easy? Compare it to random search, exactly.

THE GAP THIS CLOSES. The recipe arm reached the optimum on every workload, but with a 16-evaluation
budget over a 20-point space it explored most of the space — so "it found the optimum" is not yet
evidence of search skill. The honest control is what a uniformly random selection of the same number
of points would have achieved.

WHY THIS IS EXACT AND FREE. The whole space has been measured (`sweep_exhaustive`), so the
distribution of "best of n distinct points drawn at random" has a closed form: for a space of N points
whose cycle counts are ranked, the probability that a random n-subset's best is at rank r or better is
1 - C(N-r, n)/C(N, n). No sampling, no simulator, no tokens — and no seed to pick, which means no
knob that could be turned until the answer flattered the agent.

WHAT WOULD FALSIFY THE AGENT'S VALUE AS A SEARCHER. If its best-at-n sits inside the band random
search reaches, then on this space the agent is not searching better than chance and its contribution
is the mechanism (a compiler-constructed candidate cannot be broken), not the choices. That is a real
result and it is reported as such rather than hidden — the ablation is about where the value comes
from, so finding it in the mechanism rather than the model is informative either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402

from merlin.common.artifacts import artifacts_dir, new_product        # noqa: E402


def space_for(workload: str) -> list[int] | None:
    """Every measured cycle count for the full recipe space of one workload, or None."""
    root = artifacts_dir() / "recipe-select" / T.TARGET
    for f in sorted(root.glob("v*/*/recipe_sweep.json"), key=lambda p: p.parent.name, reverse=True):
        rows = [r for r in json.loads(f.read_text())["rows"]
                if r["workload"] == workload and isinstance(r.get("cycles"), int)
                and r.get("correct")]
        pts = {json.dumps(r["recipe"], sort_keys=True): r["cycles"] for r in rows}
        if len(pts) >= 15:
            return sorted(pts.values())
    return None


def random_best_expectation(sorted_cycles: list[int], n: int) -> tuple[float, int, int]:
    """(expected best, median best, 90th-percentile-good best) for a random n-subset, EXACTLY.

    With the space sorted ascending, the best of a random n-subset is at rank r (1-indexed) with
    probability C(N-r, n-1)/C(N, n): choose the r-th point, and all n-1 others from the N-r worse
    ones. Summing r*P(r) gives the expectation with no sampling error.
    """
    N = len(sorted_cycles)
    n = min(n, N)
    probs = [comb(N - r, n - 1) / comb(N, n) for r in range(1, N - n + 2)]
    exp = sum(c * p for c, p in zip(sorted_cycles, probs))
    cum, med, p90 = 0.0, None, None
    for c, p in zip(sorted_cycles, probs):
        cum += p
        if med is None and cum >= 0.5:
            med = c
        if p90 is None and cum >= 0.9:
            p90 = c
    return exp, med or sorted_cycles[-1], p90 or sorted_cycles[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(argv)
    T.assert_frozen_intact()

    rows = []
    for f in sorted(T.RUNS.glob("*/agent_summary.json"), key=lambda p: p.parent.name):
        d = json.loads(f.read_text())
        wl = Path(d["workload"]).stem.replace(".interface", "")
        space = space_for(wl)
        if not space:
            rows.append({"workload": wl, "verdict": "no exhaustive sweep for this workload — "
                                                     "the control cannot be computed"})
            continue
        opt = space[0]
        base = d["baseline_cycles"]
        agent_curve, best = [], None
        for r in d["history"]:
            if r.get("correct") and isinstance(r.get("cycles"), int):
                best = r["cycles"] if best is None else min(best, r["cycles"])
            agent_curve.append(best)
        per_n = []
        for n in range(1, min(len(d["history"]), len(space)) + 1):
            exp, med, p90 = random_best_expectation(space, n)
            a_best = agent_curve[n - 1]
            per_n.append({"n": n, "agent_best": a_best,
                          "random_expected": round(exp, 1), "random_median": med,
                          "random_p90": p90,
                          "agent_beats_expectation": (a_best is not None and a_best < exp),
                          "agent_beats_median": (a_best is not None and a_best < med)})
        # The decisive number: how many random draws it takes IN EXPECTATION to match what the agent
        # had after n draws. If that is ~n, the agent is no better than chance on this space.
        n_first_opt = next((p["n"] for p in per_n if p["agent_best"] == opt), None)
        rnd_at_that_n = next((p for p in per_n if p["n"] == n_first_opt), None)
        p_opt_in_n = (1 - comb(len(space) - 1, n_first_opt) / comb(len(space), n_first_opt)
                      if n_first_opt and n_first_opt < len(space) else 1.0)
        rows.append({
            "workload": wl, "space_size": len(space), "optimum": opt, "baseline": base,
            "agent_reached_optimum_at_n": n_first_opt,
            "random_expected_best_at_that_n": rnd_at_that_n["random_expected"]
            if rnd_at_that_n else None,
            "prob_random_finds_optimum_in_that_many": round(p_opt_in_n, 4),
            "per_n": per_n,
            "verdict": ("the agent is INSIDE the random band — on this space its value is the "
                        "mechanism (no broken candidates), not the choices"
                        if not any(p["agent_beats_median"] for p in per_n[:max(1, (n_first_opt or 1))])
                        else "the agent beats the median random draw before reaching the optimum"),
        })

    print(f"{'workload':<13}{'|space|':>8}{'opt':>7}{'agent@n':>9}{'rand E[best]@n':>16}"
          f"{'P(rand hits opt)':>18}")
    for r in rows:
        if "space_size" not in r:
            print(f"{r['workload']:<13}  {r['verdict']}")
            continue
        print(f"{r['workload']:<13}{r['space_size']:>8}{r['optimum']:>7}"
              f"{str(r['agent_reached_optimum_at_n']):>9}"
              f"{str(r['random_expected_best_at_that_n']):>16}"
              f"{r['prob_random_finds_optimum_in_that_many']:>18}")
    for r in rows:
        if "verdict" in r and "space_size" in r:
            print(f"\n  {r['workload']}: {r['verdict']}")

    prod = new_product("recipe-select", version=args.version, target=T.TARGET,
                       notes="search-efficiency control: agent vs EXACT random-search expectation")
    out = prod.add_artifact("search_efficiency.json")
    out.write_text(json.dumps({
        "method": ("exact combinatorics over the fully measured space; P(best of a random n-subset "
                   "is at rank r) = C(N-r, n-1)/C(N, n). No sampling, no seed, no tokens."),
        "why": ("with a 16-evaluation budget over a 20-point space the agent explores most of it, so "
                "reaching the optimum is not by itself evidence of search skill"),
        "rows": rows}, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
