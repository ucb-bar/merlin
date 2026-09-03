"""Flatten every measurement this track has produced into ONE tidy CSV, so plots are decided later.

The point is to commit to nothing. Each row is one evaluated candidate from any arm -- the frozen
default, an exhaustive sweep point, a held-out arm, or an agent proposal -- with every quantity
recorded beside it: cycles, correctness, the emitted-code shape, tokens in every bucket, wall time
split into agent and oracle, notional cost, the cumulative series for all four spend axes, and the
provenance needed to cite it. Any anytime curve, any cost axis, any grouping is then a column read.

FOUR THINGS THIS DELIBERATELY DOES NOT DO, because each would destroy information a later plot needs:

* it does not average replicates -- every measurement stays its own row;
* it does not drop failed or duplicate candidates -- they cost spend and must appear on any x axis;
* it does not fill an absent value with 0. A field that was never recorded is EMPTY, and the schema
  says which arms record which fields. "Not measured" and "measured zero" are different claims and
  this tree has a recurring bug class that conflates them;
* it does not put notional dollars in a `billed_usd` column. The driver is a subscription seat, so
  billed is *structurally* unavailable, not zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402

from merlin.common.artifacts import artifacts_dir, new_product        # noqa: E402

#: The tidy schema. Ordered for reading; every row carries every key (empty when not applicable).
COLUMNS = [
    # identity
    "arm", "cited", "void_reason", "run_id", "workload", "M", "N", "K", "Mt", "Nt", "Kt", "macs",
    "candidate_index", "recipe_activation_residency", "recipe_config_policy", "recipe_drain",
    "recipe_block_m", "recipe_block_n", "recipe_block_k",
    "is_default", "candidate_id", "artifact_digest",
    # which model this workload came from, and what it is at TRUE size. A census row is a SIZED
    # shape: K is the model's own, M and N are clamped to what the oracle can afford, and a reader
    # who cannot see both cannot tell a model result from a synthetic one.
    "model_id", "layer_fqn", "invocation_count", "mac_share", "sizing",
    "true_M", "true_N", "true_K",
    # the cut the compiler actually used, without which cycles cannot be joined to a schedule
    "block_m", "block_n", "block_k", "n_blocks", "blocks_derived", "blocks_source",
    "fits_without_cutting",
    # emitted code
    "n_instructions", "n_config", "n_mvin", "n_preload", "n_compute", "n_mvout", "n_fence",
    "code_differs_from_default", "counts_differ_from_default",
    # result
    "legal", "correct", "cycles", "speedup_vs_default", "best_cycles_so_far", "speedup_so_far",
    "failure", "duplicate", "served_from_cache",
    # spend: LLM
    "tokens_input", "tokens_cached", "tokens_cache_write", "tokens_output", "tokens_reasoning",
    "tokens_total", "cumulative_tokens", "billing_mode", "billed_usd", "notional_usd",
    "cumulative_notional_usd", "usage_available", "usage_complete", "turns_completed",
    "prompt_chars", "reply_chars",
    # spend: time
    "agent_seconds", "eval_seconds", "cumulative_agent_seconds", "cumulative_eval_seconds",
    "cumulative_wall_seconds", "turn_started_utc", "turn_ended_utc",
    # measurement conditions + provenance
    "engine", "engine_config", "oracle_kind", "derived_from_rtl", "concurrency_declared",
    "sims_running_observed", "loadavg_1m", "driver", "model", "codex_version",
    "gsim_sha256", "product",
    # WHICH MODEL AND EFFORT ANSWERED. AutoComp tiers plan and implement across two models, so a
    # token number that does not name its tier cannot be attributed, and the tiering cannot be
    # evaluated at all. The recipe arm runs one tier by construction and says so.
    "tier", "tier_model", "tier_effort", "plan_model", "code_model",
    # the exact compiler that produced this row: forks are content-addressed and minted per run
    "package_id", "package_source_digest",
]

GSIM_CFG = T.GSIM_CONFIG


def _blank() -> dict:
    return {c: "" for c in COLUMNS}


def _hist(row: dict, h: dict | None) -> None:
    if not h:
        return
    row["n_config"] = h.get("CONFIG", "")
    row["n_mvin"] = h.get("MVIN", "")
    row["n_preload"] = h.get("PRELOAD", "")
    row["n_compute"] = (h.get("COMPUTE_PRELOADED", 0) or 0) + (h.get("COMPUTE_ACCUMULATE", 0) or 0)
    row["n_mvout"] = h.get("MVOUT", "")
    row["n_fence"] = h.get("FENCE", "")


def _recipe(row: dict, rec: dict | None) -> None:
    for k in ("activation_residency", "config_policy", "drain", "block_m", "block_n", "block_k"):
        row[f"recipe_{k}"] = (rec or {}).get(k, "")


_RECIPE_MOD = None


def _recipe_module():
    """The compiler's own recipe module, imported from the fork it is measured against."""
    global _RECIPE_MOD
    if _RECIPE_MOD is None:
        import sys as _sys                                                # noqa: PLC0415
        _sys.path.insert(0, str(T.WORK / "mlir_oot"))
        from lowering import recipe as _r                                 # noqa: PLC0415
        _RECIPE_MOD = _r
    return _RECIPE_MOD


def _blocks_from_compiler(row: dict) -> None:
    """Recover the emitted cut for a row recorded before the runner started storing it.

    DERIVED, not guessed: the block plan is a pure function of (recipe, shape, machine geometry), so
    asking the compiler reproduces exactly what it emitted. Rows written before that field existed
    would otherwise report an empty cut while their cycles plainly belong to one.
    """
    try:
        R = _recipe_module()
        rec = {k: row.get(f"recipe_{k}") or "auto" for k in
               ("activation_residency", "config_policy", "drain", "block_m", "block_n", "block_k")}
        rec = {k: v for k, v in rec.items() if v}
        m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
        import sys as _sys                                                # noqa: PLC0415
        _sys.path.insert(0, str(T.WORK / "mlir_oot"))
        from lowering.isa import ACC_ROWS, DIM, SPAD_ROWS                 # noqa: PLC0415
        pl = R.blocks(R.Recipe(**rec), m=m, n=n, k=k, dim=DIM,
                      spad_rows=SPAD_ROWS, acc_rows=ACC_ROWS)
        if pl.ok:
            row["block_m"], row["block_n"], row["block_k"] = pl.bm, pl.bn, pl.bk
            row["n_blocks"], row["blocks_derived"] = pl.n_blocks, pl.derived
            row["blocks_source"] = "derived_from_compiler"
    except Exception:
        pass          # a row we cannot reconstruct stays EMPTY rather than carrying a guess


def _blocks(row: dict, b: dict | None) -> None:
    """The cut the compiler chose, as distinct from the cut the agent asked for.

    `recipe_block_*` is the REQUEST (often "auto"); these are what was emitted. Recording only the
    request would make every derived-default row look identical while the schedules differed.
    """
    if not b:
        return
    row["block_m"] = b.get("block_m", "")
    row["block_n"] = b.get("block_n", "")
    row["block_k"] = b.get("block_k", "")
    row["n_blocks"] = b.get("n_blocks", "")
    row["blocks_derived"] = b.get("derived", "")
    row["blocks_source"] = "recorded_at_build"


def _census_shape(row: dict, workload: str) -> None:
    """Join a row back to the model layer it came from, via the census CSV.

    The workload FILENAME carries model and layer, but the true extents and the MAC share live in
    the census; without them a reader cannot weight a per-shape result by how much of the model it
    represents, which is the only way a per-layer number becomes a statement about ResNet-50.
    """
    import csv as _csv                                                    # noqa: PLC0415

    global _CENSUS
    if _CENSUS is None:
        _CENSUS = {}
        for prod in sorted((T.REPO / "out/artifacts/recipe-select/gemmini/v2").glob(
                "*/kernel_census.csv")):
            for r in _csv.DictReader(prod.open(encoding="utf-8")):
                if not r.get("M"):
                    continue
                key = (r["eval_M"], r["eval_N"], r["eval_K"])
                _CENSUS[key] = r
    stem = Path(workload).stem
    if "__" not in stem:
        return
    tail = stem.rsplit("__", 1)[-1]
    parts = tail.split("x")
    if len(parts) != 3:
        return
    r = _CENSUS.get(tuple(parts))
    if not r:
        return
    row["model_id"] = r.get("model_id", "")
    row["layer_fqn"] = r.get("layer_fqn", "")
    row["invocation_count"] = r.get("invocation_count", "")
    row["mac_share"] = r.get("mac_share", "")
    row["sizing"] = r.get("sizing", "")
    row["true_M"] = r.get("true_M", "")
    row["true_N"] = r.get("true_N", "")
    row["true_K"] = r.get("true_K", "")
    row["fits_without_cutting"] = r.get("fits_without_cutting", "")


_CENSUS: "dict | None" = None


def from_sweeps() -> list[dict]:
    out = []
    root = artifacts_dir() / "recipe-select" / T.TARGET
    for f in sorted(root.glob("v*/*/recipe_sweep.json")):
        d = json.loads(f.read_text())
        n_pts = len({json.dumps(r["recipe"], sort_keys=True) for r in d["rows"]})
        for r in d["rows"]:
            row = _blank()
            # An exhaustive sweep and a 4-point wave-A sweep are different arms; distinguish them by
            # how much of the space they covered rather than by the order they were run in.
            row["arm"] = "sweep_exhaustive" if n_pts >= 15 else "sweep_wave_a"
            row["cited"] = True
            row.update({
                "run_id": f.parent.name, "workload": r["workload"],
                "M": r["M"], "N": r["N"], "K": r["K"],
                "Mt": r["Mt"], "Nt": r["Nt"], "Kt": r["Kt"],
                "is_default": r["is_default"], "n_instructions": r["n_instr"],
                "artifact_digest": r.get("artifact_digest", ""),
                "code_differs_from_default": r.get("code_differs_from_default", ""),
                "counts_differ_from_default": r.get("instr_counts_differ", ""),
                "legal": True, "correct": r.get("correct", ""), "cycles": r.get("cycles", ""),
                "speedup_vs_default": r.get("speedup_vs_default", ""),
                "failure": r.get("error") or "", "eval_seconds": r.get("wall_s", ""),
                "engine": r.get("engine", d.get("engine", "")), "engine_config": GSIM_CFG,
                "oracle_kind": r.get("oracle_kind", ""),
                "derived_from_rtl": r.get("derived_from_rtl", ""),
                "concurrency_declared": r.get("concurrency", ""),
                "gsim_sha256": d.get("gsim_sha256", ""), "product": str(f.parent),
                # No model was in the loop for a sweep. Left EMPTY rather than 0 so a token axis
                # cannot silently plot a deterministic arm as a zero-cost agent.
            })
            row["macs"] = r["M"] * r["N"] * r["K"]
            _recipe(row, r["recipe"])
            _hist(row, r.get("instr_histogram"))
            out.append(row)
    return out


def from_heldout() -> list[dict]:
    out = []
    root = artifacts_dir() / "recipe-select" / T.TARGET
    for f in sorted(root.glob("v*/*/heldout_rule.json")):
        d = json.loads(f.read_text())
        for r in d["rows"]:
            for arm_value, res in (r.get("arms") or {}).items():
                row = _blank()
                row.update({
                    "arm": "heldout", "cited": True, "run_id": f.parent.name, "workload": r["heldout"],
                    "M": r["M"], "N": r["N"], "K": r["K"], "Mt": r["Mt"], "Nt": r["Nt"],
                    "Kt": r["Kt"], "macs": r["M"] * r["N"] * r["K"],
                    "is_default": arm_value == "per_tile",
                    "legal": res.get("legal", ""), "correct": res.get("correct", ""),
                    "cycles": res.get("cycles", ""), "failure": res.get("reason")
                    or res.get("error") or res.get("status") or "",
                    "eval_seconds": res.get("wall_s", ""),
                    "engine": d.get("engine", ""), "engine_config": GSIM_CFG,
                    "concurrency_declared": r.get("concurrency", ""),
                    "product": str(f.parent),
                })
                _recipe(row, {"activation_residency": arm_value, "config_policy": "per_mvin",
                              "drain": "inline"})
                out.append(row)
    return out


def from_autocomp() -> list[dict]:
    """One row per candidate AutoComp's search produced, including the ones its beam discarded.

    ⚠️ `cited` is False for runs invalidated by defects in THIS harness (an injected `uint64_t` the
    agent's includes could not satisfy, a `solution()` fragment compiled as a whole program, and the
    missing gemmini dialect aliases). Those rows stay in the CSV because the tokens were really spent,
    but they are not evidence about AutoComp and must be filtered out of any claim.
    """
    out = []
    runs = [(f, json.loads(f.read_text())) for f in sorted(T.RUNS.glob("*/autocomp_summary.json"))]
    ok = {}
    for f, d in runs:
        if d.get("arm_completed") and (d.get("invalid_total") or 0) * 2 <= (
                d.get("candidates_evaluated") or 1):
            ok[d.get("workload")] = f.parent.name
    for f, d in runs:
        cited = ok.get(d.get("workload")) == f.parent.name
        void = None if cited else ("search did not run" if not d.get("arm_completed")
                                   else "superseded; candidates rejected by harness defects")
        sh = d.get("shape") or {}
        base = d.get("baseline_cycles")
        best_so_far = None
        for r in d.get("trajectory") or []:
            if r.get("correct") and isinstance(r.get("cycles"), int):
                best_so_far = r["cycles"] if best_so_far is None else min(best_so_far, r["cycles"])
            row = _blank()
            row.update({
                "arm": "autocomp", "cited": cited, "void_reason": void or "",
                "run_id": f.parent.name, "workload": d.get("workload", ""),
                "M": sh.get("M", ""), "K": sh.get("K", ""), "N": sh.get("N", ""),
                "macs": (sh.get("M", 0) or 0) * (sh.get("N", 0) or 0) * (sh.get("K", 0) or 0),
                "candidate_index": r.get("index"), "is_default": r.get("index") == 1,
                "legal": r.get("compiled"), "correct": r.get("correct"),
                "cycles": r.get("cycles") if r.get("cycles") is not None else "",
                "speedup_vs_default": round(base / r["cycles"], 4)
                if (base and isinstance(r.get("cycles"), int) and r.get("correct")) else "",
                "best_cycles_so_far": best_so_far if best_so_far is not None else "",
                "failure": (r.get("stderr") or "")[:200],
                "eval_seconds": r.get("eval_seconds", ""),
                "engine": r.get("engine", d.get("engine", "")),
                "engine_config": r.get("engine_config", GSIM_CFG),
                "driver": "codex", "model": d.get("plan_model", ""),
                # AutoComp spends across TWO tiers. Naming both on every row is what lets the
                # per-tier ledger (codex_calls.jsonl) be joined to per-candidate quality; without it
                # the arm can report what it spent but not which model spent it.
                "plan_model": d.get("plan_model", ""), "code_model": d.get("code_model", ""),
                "tier": "plan+code",
                "tier_model": ((d.get("tiers") or {}).get("plan") or {}).get("model", ""),
                "tier_effort": ((d.get("tiers") or {}).get("plan") or {}).get("effort", ""),
                "package_id": d.get("package_id", ""),
                "product": str(f.parent),
                # AutoComp's per-call usage is aggregated by the codex provider and not attributed to
                # individual candidates, so per-row token columns stay EMPTY rather than 0.
            })
            _census_shape(row, d.get("workload_mlir") or d.get("workload", ""))
            out.append(row)
    return out


def from_agent() -> list[dict]:
    out = []
    for f in sorted(T.RUNS.glob("*/agent_summary.json")):
        d = json.loads(f.read_text())
        sh = d.get("shape") or {}
        tiles = sh.get("tiles") or {}
        base = d.get("baseline_cycles")
        for r in d["history"]:
            a = r.get("accounting") or {}
            row = _blank()
            row.update({
                "arm": "recipe_agent", "cited": True, "run_id": f.parent.name, "workload": Path(d["workload"]).stem,
                "M": sh.get("M", ""), "N": sh.get("N", ""), "K": sh.get("K", ""),
                "Mt": tiles.get("Mt", ""), "Nt": tiles.get("Nt", ""), "Kt": tiles.get("Kt", ""),
                "macs": sh.get("macs", ""),
                "candidate_index": r["candidate"], "is_default": False,
                "candidate_id": r.get("candidate_id", ""),
                "artifact_digest": r.get("artifact_digest", ""),
                "n_instructions": r.get("n_instructions", ""),
                "code_differs_from_default": (r.get("vs_default_code") or {}).get(
                    "identical") is False if r.get("vs_default_code") else "",
                "legal": r.get("legal", ""), "correct": r.get("correct", ""),
                "cycles": r.get("cycles", ""),
                "speedup_vs_default": round(base / r["cycles"], 4)
                if (base and isinstance(r.get("cycles"), int)) else "",
                "best_cycles_so_far": r.get("best_cycles_so_far", ""),
                "speedup_so_far": r.get("speedup_so_far", ""),
                "failure": r.get("failure") or "", "duplicate": r.get("duplicate", ""),
                "served_from_cache": r.get("served_from_cache", ""),
                "tokens_input": a.get("tokens_input", ""), "tokens_cached": a.get("tokens_cached", ""),
                "tokens_cache_write": a.get("tokens_cache_write", ""),
                "tokens_output": a.get("tokens_output", ""),
                "tokens_reasoning": a.get("tokens_reasoning", ""),
                "tokens_total": a.get("tokens_total", ""),
                "cumulative_tokens": r.get("cumulative_tokens", ""),
                "billing_mode": a.get("billing_mode", d.get("billing_mode", "")),
                # billed is structurally unavailable on a seat: EMPTY, never 0.
                "billed_usd": "" if a.get("estimated_cost_usd") is None
                else a.get("estimated_cost_usd"),
                "notional_usd": a.get("subscription_notional_usd", ""),
                "cumulative_notional_usd": r.get("cumulative_notional_usd", ""),
                "usage_available": a.get("available", ""),
                "usage_complete": a.get("usage_complete", ""),
                "turns_completed": a.get("turns_completed", ""),
                "prompt_chars": r.get("prompt_chars", ""), "reply_chars": r.get("reply_chars", ""),
                "agent_seconds": r.get("agent_seconds", ""),
                "eval_seconds": r.get("eval_seconds", ""),
                "cumulative_agent_seconds": r.get("cumulative_agent_seconds", ""),
                "cumulative_eval_seconds": r.get("cumulative_eval_seconds", ""),
                "cumulative_wall_seconds": r.get("cumulative_wall_seconds", ""),
                "turn_started_utc": r.get("turn_started_utc", ""),
                "turn_ended_utc": r.get("turn_ended_utc", ""),
                "engine": d.get("engine", ""), "engine_config": GSIM_CFG,
                "sims_running_observed": r.get("sims_running_observed", ""),
                "loadavg_1m": r.get("loadavg_1m", ""),
                "driver": d.get("driver", ""), "model": d.get("model", ""),
                "codex_version": d.get("codex_version", ""),
                "gsim_sha256": d.get("gsim_sha256", ""), "product": str(f.parent),
            })
            _recipe(row, r.get("recipe"))
            _hist(row, r.get("instr_counts"))
            _blocks(row, r.get("blocks"))
            if not row.get("block_k"):
                _blocks_from_compiler(row)
            _census_shape(row, d["workload"])
            row["tier"] = "select"
            row["tier_model"] = d.get("model", "")
            row["tier_effort"] = d.get("effort", "")
            out.append(row)
        # The baseline is the agent arm's candidate -1: measured by the harness on the same engine in
        # the same run, so a speedup is never against a number from somewhere else.
        row = _blank()
        row.update({"arm": "recipe_agent", "cited": True, "run_id": f.parent.name,
                    "workload": Path(d["workload"]).stem, "candidate_index": -1,
                    "M": sh.get("M", ""), "N": sh.get("N", ""), "K": sh.get("K", ""),
                    "is_default": True, "legal": True, "correct": True, "cycles": base,
                    "speedup_vs_default": 1.0, "best_cycles_so_far": base, "speedup_so_far": 1.0,
                    "cumulative_tokens": 0, "engine": d.get("engine", ""),
                    "engine_config": GSIM_CFG, "product": str(f.parent)})
        _recipe(row, d.get("baseline_recipe"))
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(argv)
    T.assert_frozen_intact()

    rows = from_sweeps() + from_heldout() + from_agent() + from_autocomp()
    if not rows:
        raise SystemExit("no records found")
    rows.sort(key=lambda r: (str(r["arm"]), str(r["workload"]),
                             int(r["candidate_index"]) if str(r["candidate_index"]).lstrip("-")
                             .isdigit() else 0))

    prod = new_product("recipe-select", version=args.version, target=T.TARGET,
                       notes="tidy long-format record of every evaluated candidate, all arms")
    csv_path = prod.add_artifact("candidates.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    schema = prod.add_artifact("candidates_schema.md")
    schema.write_text(
        "# candidates.csv — one row per evaluated candidate\n\n"
        "Empty means NOT RECORDED for that arm, never zero. `billed_usd` is empty for every agent "
        "row because a subscription seat is not billed per token; use `notional_usd` and say so.\n\n"
        "| arm | what it is | records tokens? | records wall? |\n|---|---|---|---|\n"
        "| `sweep_wave_a` | the first 4-point surface | no (no model in the loop) | oracle only |\n"
        "| `sweep_exhaustive` | all 20 points of the space; source of the ground-truth optimum | no | oracle only |\n"
        "| `heldout` | shapes never used to fit the rule | no | oracle only |\n"
        "| `recipe_agent` | one LLM decision per candidate; index -1 is the harness-measured baseline | yes | agent + oracle |\n"
        "| `autocomp` | AutoComp rewrites the kernel; index 1 is its seed. Per-candidate tokens are "
        "NOT attributed (its provider aggregates them), so those columns are empty | no (run-level "
        "only) | oracle |\n\n"
        "⚠️ **Filter on `cited`.** Rows with `cited=false` come from runs invalidated by defects in "
        "this harness, not by the arm; the tokens were really spent, so the rows are kept, but they "
        "are not evidence.\n\n"
        f"Cycles are MEASURED on `{GSIM_CFG}`. {T.ENGINE_NOTE}\n", encoding="utf-8")
    prod.write_manifest()

    from collections import Counter
    c = Counter(r["arm"] for r in rows)
    print(f"{len(rows)} rows, {len(COLUMNS)} columns")
    for arm, n in sorted(c.items()):
        print(f"  {arm:<20} {n}")
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
