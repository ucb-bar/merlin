#!/usr/bin/env python3
"""Grade a frozen agent run through capsule_bench_v0 (post-agent, full-access phase).

Public/dev phase (repairable in a real run) -> freeze -> hidden phase (after freeze only). Writes
score_capsule.json per phase, an iteration_000 snapshot, run_manifest.yaml, and final_report.md.
Runs OUTSIDE the agent sandbox (needs spike/verilator + the hidden goldens).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_grade as CG  # noqa: E402
from merlin.targetgen import capsule_runner as CR  # noqa: E402
import freeze_run  # noqa: E402


def _roots(spec: str) -> list[str]:
    """A comma-separated root list. A target's graded suite spans sibling capsule CATEGORIES, so one
    path cannot name it; the launcher resolves them from the descriptor and passes them through here."""
    return [s for s in (x.strip() for x in str(spec).split(",")) if s]


def _score(pkg, capsules, runs_root, labels, no_oracle):
    # Resolve the TARGET'S OWN oracle ladder from its contract (external_backend->program_oracle,
    # chipyard->spike/verilator, else arc) — never pass None here, which historically fell back to the
    # gemmini spike/verilator MLIR-lowering oracle and mis-graded atlas (torch-mlir run_lowering.py crash).
    # `{}` = honest no-oracle (L0/L1/trace only). sim_via is self-resolved from the contract.
    adapters = {} if no_oracle else CR.oracle_adapters(C.TARGET)
    return CG.grade(pkg, capsules_root=_roots(capsules), runs_root=runs_root, labels=labels,
                    contract=str(C.REPO / "merlin/contract"),
                    oracle_adapters=adapters, timeout=900, target=C.TARGET, no_oracle=no_oracle)



def _cost_phrase(proc: dict) -> str:
    """Render the money field so a notional figure can never be read as a spend."""
    if proc.get("estimated_cost_usd") is not None:
        return f"cost=${proc['estimated_cost_usd']}"
    notional = proc.get("subscription_notional_usd")
    if notional is not None:
        return f"cost=n/a ({proc.get('billing_mode') or 'notional'}: ${notional} notional)"
    return f"cost=n/a ({proc.get('cost_unavailable_reason') or 'no usage metadata'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--capsules", default=str(C.REPO / "merlin/contract" / "capsules"),
                    help="comma-separated capsule roots for the PUBLIC/dev phase. The launcher resolves "
                         "these from the target descriptor (primary corpus + its sibling categories); "
                         "the default is the shared parent and is only right for a target whose corpus "
                         "IS that parent.")
    ap.add_argument("--hidden-capsules", default=None,
                    help="comma-separated capsule roots for the HIDDEN phase (defaults to --capsules). "
                         "The launcher passes the target's own hidden dir so the public-only "
                         "materialized subset does not yield an empty hidden grade.")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--skip-hidden", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    pkg = run_dir / "submission"

    # --- public/dev phase ---
    (run_dir / "grading_public").mkdir(parents=True, exist_ok=True)
    pub = _score(str(pkg), a.capsules, str(run_dir / "grading_public"), {"public", "dev"}, a.no_oracle)
    (run_dir / "grading_public" / "score_capsule.json").write_text(json.dumps(pub, indent=2))

    # --- iteration_000 snapshot (dummy = one-shot; a real repairing agent appends more) ---
    it = run_dir / "iterations" / "iteration_000"
    it.mkdir(parents=True, exist_ok=True)
    (it / "capsule_status_after.yaml").write_text(yaml.safe_dump(
        {c["capsule"]: c["status"] for c in pub.get("per_capsule", [])}, sort_keys=True))
    (it / "first_failure.yaml").write_text(yaml.safe_dump(pub.get("first_failure_planes", {})))
    (it / "notes.md").write_text("iteration_000: initial submission grade (public/dev).\n")

    # --- freeze, then hidden phase (immutability enforced) ---
    frozen = freeze_run.freeze(run_dir)
    hid = {"available": False}
    if not a.skip_hidden:
        # re-hash the submission right before hidden grading; refuse if it changed since freeze
        recheck = C.hash_tree(pkg)["sha256"]
        mutated = recheck != frozen["submission_sha256"]
        import datetime as _dt
        fj = json.loads((run_dir / "freeze.json").read_text())
        fj["hidden_grading_started_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        fj["submission_sha256_recheck"] = recheck
        fj["workspace_mutable_after_freeze"] = bool(mutated)
        (run_dir / "freeze.json").write_text(json.dumps(fj, indent=2))
        if mutated:
            raise SystemExit(f"FREEZE VIOLATION: submission changed after freeze "
                             f"({frozen['submission_sha256'][:12]} -> {recheck[:12]}); refusing hidden grade")
        (run_dir / "grading_hidden").mkdir(parents=True, exist_ok=True)
        hid = _score(str(pkg), (a.hidden_capsules or a.capsules),
                     str(run_dir / "grading_hidden"), {"hidden"}, a.no_oracle)
        (run_dir / "grading_hidden" / "score_capsule.json").write_text(json.dumps(hid, indent=2))

    # --- process metrics (from launcher), env, run_manifest ---
    ctt = {}
    ctt_path = run_dir / "cost_time_toolcalls.yaml"
    if ctt_path.is_file():
        ctt = yaml.safe_load(ctt_path.read_text()) or {}

    manifest = {
        "run_id": run_dir.name, "arm": a.arm, "model": a.model,
        "repo_sha": frozen["repo_sha"], "submission_sha256": frozen["submission_sha256"],
        "frozen_at": frozen["frozen_at"],
        "integrity_status": pub.get("integrity_status"),
        "integrity_exempt": pub.get("integrity_exempt"),
        "public_dev": {"functional_pass": pub.get("functional_pass"),
                       "passed": pub.get("public_passed") or f"{pub.get('n_passed')}/{pub.get('n_capsules')}",
                       "highest_tier": pub.get("highest_tier"),
                       "numeric_all_exact": pub.get("numeric_all_exact"),
                       "trace_all_pass": pub.get("trace_all_pass"),
                       "first_failure_planes": pub.get("first_failure_planes")},
        "hidden": {"passed": hid.get("public_passed") or hid.get("hidden_passed")
                   or (f"{hid.get('n_passed')}/{hid.get('n_capsules')}" if hid.get("available", True)
                       else "not_run"),
                   "functional_pass": hid.get("functional_pass")},
        "cycles_diagnostic": pub.get("cycles_diagnostic", {}),
        "process": {"wall_time_seconds": ctt.get("wall_time_seconds"),
                    "tokens_total": ctt.get("tokens_total"),
                    "tokens_input": ctt.get("tokens_input"),
                    "tokens_output": ctt.get("tokens_output"),
                    "tokens_reasoning": ctt.get("tokens_reasoning"),
                    "estimated_cost_usd": ctt.get("estimated_cost_usd"),
                    # A subscription-seat run has no per-token spend; its dollars are notional and
                    # kept in their own field so nothing sums them into a money budget.
                    "billing_mode": ctt.get("billing_mode"),
                    "subscription_notional_usd": ctt.get("subscription_notional_usd"),
                    "cost_unavailable_reason": ctt.get("cost_unavailable_reason"),
                    "tool_calls": ctt.get("tool_calls"),
                    "metrics_available": ctt.get("available")},
        "iterations": len(list((run_dir / "iterations").glob("iteration_*"))),
        "oracle_mode": "no_oracle(L0/L1/trace)" if a.no_oracle
                       else f"contract-routed({'+'.join(sorted(CR.oracle_adapters(C.TARGET)))})",
        # HONEST gradeability: a --no-oracle run is a structure-only smoke — the numeric verdict is
        # withheld (capsules read back not_gradeable_no_oracle, never a numeric pass), so the run is NOT
        # gradeable. A graded run carries gradeable=true. Sourced from the grade score (gradeable flag).
        "gradeable": bool(pub.get("gradeable", not a.no_oracle)),
        "gradeable_reason": (None if not a.no_oracle else
                             "numeric oracle unavailable — structural (L0/L1/trace) tiers only"),
    }
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    fr = [f"# {run_dir.name} — final report ({a.arm})", "",
          f"- integrity: {manifest['integrity_status']} (exempt={manifest['integrity_exempt']})",
          f"- public/dev: functional_pass={manifest['public_dev']['functional_pass']} "
          f"passed={manifest['public_dev']['passed']} highest_tier={manifest['public_dev']['highest_tier']}",
          f"- hidden: {manifest['hidden']['passed']} (after freeze {frozen['submission_sha256'][:12] if frozen['submission_sha256'] else 'n/a'})",
          f"- process: wall={manifest['process']['wall_time_seconds']}s "
          f"tokens={manifest['process']['tokens_total']} {_cost_phrase(manifest['process'])} "
          f"tool_calls={manifest['process']['tool_calls']} (available={manifest['process']['metrics_available']})",
          f"- oracle_mode: {manifest['oracle_mode']}",
          f"- gradeable: {manifest['gradeable']}"
          + (f" ({manifest['gradeable_reason']})" if manifest.get('gradeable_reason') else ""), ""]
    (run_dir / "final_report.md").write_text("\n".join(fr) + "\n")
    print(f"graded {run_dir.name}: public functional_pass={pub.get('functional_pass')} "
          f"({manifest['public_dev']['passed']}), hidden={manifest['hidden']['passed']}, "
          f"integrity={manifest['integrity_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
