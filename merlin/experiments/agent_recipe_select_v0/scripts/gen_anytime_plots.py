"""The headline figure: best performance reached so far against cumulative LLM tokens.

WHY THIS PLOT AND NOT A BAR CHART OF FINAL SPEEDUPS. The question is not "who finds the fastest
kernel" -- with a frozen compiler and a 20-point space, exhaustive search finds the optimum by
definition. The question is **how cheaply each approach gets there**, so the x axis has to be spend
and the curve has to be monotone-best-so-far. No such curve existed anywhere in this repo; every
existing trajectory plots cumulative cost against time, never against achieved quality.

FOUR HONESTY RULES, each of which changes what gets drawn:

* **an incorrect candidate advances the x axis and not the y axis.** It cost its tokens; it earned no
  performance. A curve that skipped failures would make a failing search look efficient.
* **the ground-truth optimum is a horizontal reference line, not an arm.** It came from exhaustively
  measuring all 20 points, which is not something an agent budget can be compared against as a peer.
* **duplicate proposals still consume x.** They are served from the candidate cache and cost no
  simulator time, but the tokens were spent, and that is the real price of revisiting.
* **tokens, not dollars.** The driver is a subscription seat, so `estimated_cost_usd` is `None` by
  construction and any dollar figure is a projection of what the same traffic would have cost
  metered. Notional dollars are reported in the caption, never on an axis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402
sys.path.insert(0, str(T.REPO / "merlin/experiments/gemmini_perf_bench/scripts"))

from merlin.common.artifacts import artifacts_dir, new_product        # noqa: E402


def _agent_runs() -> list[Path]:
    root = T.RUNS
    return sorted((p for p in root.glob("*/agent_summary.json")), key=lambda p: p.parent.name)


def _sweep_optimum(workload_key: str) -> tuple[int | None, int | None, dict | None]:
    """(baseline, optimum, best_recipe) from the exhaustive sweep, if one covers this workload."""
    root = artifacts_dir() / "recipe-select" / T.TARGET
    best: tuple[int | None, int | None, dict | None] = (None, None, None)
    for prod in sorted(root.glob("v*/*/recipe_sweep.json"), key=lambda p: p.parent.name,
                       reverse=True):
        rows = json.loads(prod.read_text())["rows"]
        rows = [r for r in rows if r["workload"] == workload_key
                and isinstance(r.get("cycles"), int) and r.get("correct")]
        if len(rows) < 8:                     # a wave-A-sized sweep is not the full space
            continue
        base = next((r["cycles"] for r in rows if r["is_default"]), None)
        opt = min(rows, key=lambda r: r["cycles"])
        return base, opt["cycles"], opt["recipe"]
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(argv)
    T.assert_frozen_intact()

    runs = _agent_runs()
    if not runs:
        raise SystemExit(f"no agent runs with agent_summary.json under {T.RUNS}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import perf_style as PS                                            # noqa: N812
    PS.use_style()

    summaries = [json.loads(p.read_text()) for p in runs]

    # THREE SPEND AXES, because which one is the fair denominator is not obvious and should not be
    # decided by whoever writes the plot. Tokens are the model's cost; wall-clock is what a person
    # waits; notional dollars are what the same traffic would have cost metered. All three are
    # cumulative and monotone, so the same best-so-far curve is drawn against each.
    AXES = [
        ("cumulative_tokens", "cumulative LLM tokens", 1.0),
        ("cumulative_wall_seconds", "cumulative wall-clock (s): agent + oracle", 1.0),
        ("cumulative_notional_usd", "cumulative notional USD (seat: not billed)", 1.0),
    ]

    def series(s_: dict, key: str) -> list[float]:
        """The cumulative series for one axis, DERIVED when an earlier run did not record it.

        The first agent run predates these fields; its per-round `agent_seconds`, `eval_seconds` and
        `accounting.subscription_notional_usd` are all present, so the cumulative form is recoverable
        rather than lost. Recovered values are identical to recorded ones by construction.
        """
        vals: list[float] = []
        run_t = run_c = 0.0
        for r in s_["history"]:
            if key == "cumulative_tokens":
                vals.append(float(r.get("cumulative_tokens") or 0))
            elif key == "cumulative_wall_seconds":
                if r.get("cumulative_wall_seconds") is not None:
                    vals.append(float(r["cumulative_wall_seconds"]))
                else:
                    run_t += float(r.get("agent_seconds") or 0) + float(r.get("eval_seconds") or 0)
                    vals.append(round(run_t, 2))
            else:
                if r.get("cumulative_notional_usd") is not None:
                    vals.append(float(r["cumulative_notional_usd"]))
                else:
                    run_c += float((r.get("accounting") or {}).get(
                        "subscription_notional_usd") or 0.0)
                    vals.append(round(run_c, 6))
        return vals

    n = len(summaries)
    fig, axes = plt.subplots(n, len(AXES), figsize=(5.6 * len(AXES), 4.3 * n), squeeze=False)

    rows_tbl = []
    for ri, s_ in enumerate(summaries):
        wkey = Path(s_["workload"]).stem.replace(".interface", "")
        base = s_["baseline_cycles"]
        sb, opt, orec = _sweep_optimum(wkey)
        for ci, (key, xlabel, _sc) in enumerate(AXES):
            ax = axes[ri][ci]
            xs = [0.0] + series(s_, key)
            ys = [1.0]
            for r in s_["history"]:
                b = r.get("best_cycles_so_far")
                ys.append(base / b if (b and base) else ys[-1])
            ax.step(xs, ys, where="post", color=PS.COLOR.get("merlin_targetgen", "#E6B84C"),
                    lw=2.4, marker="o", ms=4,
                    label=f"recipe agent ({s_['driver']}/{s_['model']})")
            for r, x in zip(s_["history"], xs[1:]):
                if not r.get("legal") or not r.get("correct"):
                    y = base / r["best_cycles_so_far"] if r.get("best_cycles_so_far") else 1.0
                    ax.plot([x], [y], marker="x", ms=8, mew=2, color="#C0605A", zorder=5)
            if opt and base:
                ax.axhline(base / opt, ls="--", lw=1.6, color=PS.INK, alpha=0.65)
                ax.text(0.99, base / opt, f"exhaustive optimum {base / opt:.3f}x  ", ha="right",
                        va="bottom", transform=ax.get_yaxis_transform(), fontsize=8, color=PS.INK)
            ax.axhline(1.0, ls=":", lw=1.4, color="#8a8175")
            ax.text(0.01, 1.0, "frozen default  ", ha="left", va="bottom", fontsize=8,
                    color="#8a8175", transform=ax.get_yaxis_transform())
            ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel("best speedup vs frozen default")
            sh = s_.get("shape") or {}
            title = f"{wkey}" + (f"  ({sh['M']}x{sh['N']}x{sh['K']})" if sh.get("M") else "")
            ax.set_title(title if ci == 1 else "", fontsize=11)
            if hasattr(PS, "style_ax"):
                PS.style_ax(ax, grid="y")
            else:
                ax.grid(axis="y", color=PS.GRID, lw=0.8)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            if ci == 0:
                ax.legend(loc="center right", fontsize=8, frameon=False)

        t = s_["totals"]
        reached = (base / s_["best_cycles"]) if (s_.get("best_cycles") and base) else None
        # Where the curve actually stopped improving -- the spend that MATTERED, as distinct from the
        # spend the budget happened to allow.
        first_best = next((i for i, r in enumerate(s_["history"])
                           if r.get("best_cycles_so_far") == s_.get("best_cycles")), None)
        rows_tbl.append({
            "workload": wkey, "baseline_cycles": base, "best_cycles": s_["best_cycles"],
            "speedup": round(reached, 4) if reached else None,
            "exhaustive_optimum_cycles": opt, "optimum_recipe": orec,
            "found_optimum": (s_["best_cycles"] == opt) if opt else None,
            "candidates": s_["candidates"], "invalid": s_["invalid_candidates"],
            "duplicates": s_["duplicate_candidates"],
            "candidate_reaching_best": first_best,
            "tokens_to_best": series(s_, "cumulative_tokens")[first_best]
            if first_best is not None else None,
            "wall_s_to_best": series(s_, "cumulative_wall_seconds")[first_best]
            if first_best is not None else None,
            "notional_usd_to_best": series(s_, "cumulative_notional_usd")[first_best]
            if first_best is not None else None,
            "tokens_total": t["tokens_total"], "tokens_in": t["tokens_input"],
            "tokens_out": t["tokens_output"], "tokens_cached": t["tokens_cached"],
            "notional_usd": t["notional_usd"], "billed_usd": t["billed_usd"],
            "agent_seconds": t["agent_seconds"], "eval_seconds": t["eval_seconds"],
            "wall_seconds": round(t["agent_seconds"] + t["eval_seconds"], 1),
        })

    tot_tok = sum(r["tokens_total"] for r in rows_tbl)
    tot_tok = sum(r["tokens_total"] for r in rows_tbl)
    tot_wall = round(sum(r["wall_seconds"] for r in rows_tbl), 1)
    tot_usd = round(sum(r["notional_usd"] for r in rows_tbl), 4)
    hit = sum(1 for r in rows_tbl if r["found_optimum"])
    PS.caption(fig,
               f"Best-so-far speedup against three spend axes: LLM tokens, wall-clock (agent turn + "
               f"oracle), and notional USD. N={sum(r['candidates'] for r in rows_tbl)} candidates "
               f"over {len(rows_tbl)} workload(s); {tot_tok} tokens, {tot_wall}s, ${tot_usd} "
               f"notional; {sum(r['invalid'] for r in rows_tbl)} invalid. Cycles MEASURED on "
               f"elaborated RTL (GSIM, GemminiGsimSerialClkConfig \u2014 NOT Verilator-equivalent). "
               f"x marks a rejected or incorrect candidate: it advances spend and earns no speedup. "
               f"Dashed line = optimum from exhaustively measuring all 20 recipe points \u2014 a "
               f"reference, not a competing arm ({hit}/{len(rows_tbl)} reached it). The driver is a "
               f"subscription seat, so no dollar was billed; the USD axis is what the same traffic "
               f"would have cost metered.")

    prod = new_product("recipe-select", version=args.version, target=T.TARGET,
                       notes="anytime curve: best speedup vs cumulative LLM tokens")
    png = prod.add_artifact("anytime_tokens.png")
    PS.save_fig(fig, png)
    (prod.path / "anytime_summary.json").write_text(json.dumps(rows_tbl, indent=1), encoding="utf-8")
    prod._artifacts.append("anytime_summary.json")
    prod.write_manifest()

    print(f"{'workload':<13}{'base':>6}{'best':>6}{'sp':>8}{'opt':>6}{'hit':>6}{'cand':>5}"
          f"{'inv':>4}{'dup':>4}{'@cand':>6}{'tok@best':>10}{'wall@best':>10}{'$@best':>8}"
          f"{'tok_tot':>9}{'wall_tot':>9}{'$_tot':>7}")
    for r in rows_tbl:
        print(f"{r['workload']:<13}{r['baseline_cycles']:>6}{r['best_cycles']:>6}"
              f"{r['speedup']:>8}{str(r['exhaustive_optimum_cycles']):>6}"
              f"{str(r['found_optimum']):>6}{r['candidates']:>5}{r['invalid']:>4}"
              f"{r['duplicates']:>4}{str(r['candidate_reaching_best']):>6}"
              f"{str(r['tokens_to_best']):>10}{str(r['wall_s_to_best']):>10}"
              f"{str(r['notional_usd_to_best']):>8}{r['tokens_total']:>9}"
              f"{r['wall_seconds']:>9}{r['notional_usd']:>7}")
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
