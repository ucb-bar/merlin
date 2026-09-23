#!/usr/bin/env python3
"""Render a sealed, attributable Arm4 performance-measurement report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _pbcommon as PB
import perf_reporting as PR
from merlin_experiments.phase2 import candidate_verification as VERIFY
from merlin_experiments.phase2 import reporting as CURRENT


def _cells(rows: list[dict], simulator: str) -> list[dict]:
    return [row for row in rows if row["identity"]["simulator"] == simulator]


def render_report(run_id: str, campaign: dict, rows: list[dict], counts: dict) -> str:
    """Render gated measurements without turning measurement completion into a claim verdict."""
    verilator = _cells(rows, "verilator")
    spike = _cells(rows, "spike")
    results = campaign["results"]
    decision = campaign["decision_boundary"]
    candidate = campaign["candidate_stage"]
    host_lane = campaign["model_host_lane_snapshot"]
    sentinel = campaign["full_model_admission"]
    formal = campaign["experiment_mode"] == "formal_claim"
    claim_status = campaign["claim_status"]
    if formal:
        headline = f"PK CLAIM {claim_status}"
        claim_summary = (
            "The predeclared PK affine K-to-cycle prediction satisfied every fixed quantitative "
            "bound and is promoted for this frozen cohort."
            if claim_status == "ESTABLISHED"
            else "The completed PK experiment violated one or more predeclared quantitative bounds. "
            "The claim is refuted and is not promoted."
        )
        title = f"# Arm4 PK performance claim ({run_id})"
    else:
        headline = "CLAIM NOT ESTABLISHED"
        claim_summary = (
            "The measurement-only smoke campaign is complete. This is not a completed performance "
            "experiment and it does not establish or promote a performance claim."
        )
        title = f"# Arm4 performance tooling smoke ({run_id})"
    claim_evidence: list[str] = []
    if formal:
        claim = campaign["claim_decision"]
        fit = claim["fit"]
        claim_evidence = [
            "",
            "## PK quantitative decision",
            "",
            f"- Decision: **{claim['status']}**",
            f"- Fit: `{fit['equation']}` over **{fit['n_observations']}** Verilator L3 rows",
            f"- Rate: `{fit['rate_cycles_per_K_element']}` cycles/K; intercept: `{fit['intercept_cycles']}` cycles",
            f"- R²: `{fit['r_squared']}`; RMSE: `{fit['rmse_cycles']}` cycles; maximum absolute "
            f"residual: `{fit['max_absolute_residual_cycles']}` cycles",
            f"- Evidence separation: `{claim['evidence']['l3_positive_cycle_rows_consumed']}` "
            "L3 timing rows consumed; `0` Spike L2 cycles consumed",
        ]
    lines = [
        title,
        "",
        f"## {headline}",
        "",
        claim_summary,
        "",
        f"Decision boundary: `{decision['module']}` via `{decision['identity_bridge']}`; "
        f"promotion integration is `{decision['promotion_integration']}` and the promotion status "
        f"is `{decision['promotion_status']}`. "
        f"Recorded reason: {decision['reason']}",
        "",
        (
            "This is one Arm4 compiler lane, not a cross-approach comparison, speedup claim, or "
            "generalization claim. The formal verdict combines the predeclared fixed-cohort fit with "
            "Verilator L3 measurements; no individual cycle row proves the claim."
            if formal
            else "This is one Arm4 compiler lane, not a cross-approach comparison, speedup claim, or "
            "generalization claim. Verilator L3 cycles below are citable measurements, not proof of "
            "the declared capsule hypothesis."
        ),
        "",
        "## Attribution and enforcement",
        "",
        f"- Functional run ID: `{campaign['functional_run_id']}`",
        f"- Functional submission SHA-256: `{campaign['functional_submission_sha256']}`",
        f"- Sealed performance candidate SHA-256: `{candidate['candidate_sha256']}`",
        f"- Candidate record: `{candidate['record_path']}`",
        f"- Candidate record SHA-256: `{candidate['record_sha256']}`",
        f"- Performance prompt SHA-256: `{candidate['prompt_sha256']}`",
        f"- Prompt facts SHA-256: `{candidate['prompt_facts_sha256']}`",
        f"- Formal replicate identities: `{', '.join(candidate['formal_replicate_identities'])}`",
        f"- Sealed smoke replicate count: `{candidate['smoke_replicates']}`",
        f"- Candidate-stage formal PK preflight: `{candidate['formal_claim']['status']}`",
        f"- Target descriptor SHA-256: `{candidate['target_descriptor_sha256']}`",
        f"- Candidate audit: clean; transcript SHA-256: `{candidate['transcript_sha256']}`",
        f"- Candidate authoring tool evidence: "
        f"**{len(candidate['required_actions'])}** required broker action(s), "
        f"**{len(candidate['tool_evidence']['tool_probe_results'])}** probes passed and rechecked",
        f"- Frozen host lane: `{host_lane['package']}`",
        f"- Frozen host-lane package SHA-256: `{host_lane['package_sha256']}`",
        f"- Functional bundle-input snapshot SHA-256: `{host_lane['run_snapshot']['content_sha256']}`",
        f"- Frozen measurement contract SHA-256: `{campaign['frozen_contract']['sha256']}`",
        f"- Full-model admission sentinel: `{sentinel['capsule']}`; frozen source SHA-256 "
        f"`{sentinel['source_sha256']}`; L2/L3 pass across mesh + scalar RVV lanes",
        "- Full-model sentinel role: correctness admission only; its cycles are not recorded or "
        "included in performance rows",
        f"- Frozen workload manifest SHA-256: `{campaign['workload_manifest_sha256']}`",
        f"- Frozen workload capsule-tree SHA-256: `{campaign['workload_sha256']}`",
        f"- Sealed perf_results.json SHA-256: `{results['sha256']}`",
        f"- perf_results digest-record SHA-256: `{results['digest_record_sha256']}`",
        f"- Functional fork: `{campaign['fork_before']['state']}` before and "
        f"`{campaign['fork_after']['state']}` after measurement",
        "- Sandbox: bwrap with the sealed candidate source read-only, each C++/Python execution "
        "using its own writable build copy, answer surfaces closed, and every declared tool probe "
        "enforced",
        "- Network: available; network isolation is not part of this experiment's validity claim",
        f"- Exact-cell completion: **{counts['reported']}/{counts['expected']}**; "
        f"Spike L2 screens **{counts['screen_passed']}/{counts['screen_expected']}**; "
        f"Verilator L3 citable measurements "
        f"**{counts['citable_passed']}/{counts['citable_expected']}**",
        *claim_evidence,
        "",
        "## Citable timing — Verilator L3 only",
        "",
        "Only Verilator L3 timing is shown. Spike is a correctness screen and its cycles are "
        "omitted "
        "at the sealed-results boundary.",
        "",
        "| family | capsule | replicate | L3 cycles | correct |",
        "|---|---|---:|---:|---:|",
    ]
    for row in verilator:
        identity = row["identity"]
        lines.append(
            f"| {identity['family']} | {identity['capsule']} | {identity['replicate']} | {row['cycles']} | yes |"
        )
    lines += [
        "",
        "## Correctness corroboration — Spike L2",
        "",
        "| family | capsule | replicate | L2 correctness | cycles |",
        "|---|---|---:|---:|---:|",
    ]
    for row in spike:
        identity = row["identity"]
        lines.append(
            f"| {identity['family']} | {identity['capsule']} | {identity['replicate']} | pass | deliberately omitted |"
        )
    lines += [
        "",
        "## Explicit non-claims",
        "",
        "- Measurement GO means every predeclared exact cell completed under its simulator policy.",
        (
            "- Formal PK status applies only to the frozen fixed-M/N K cohort and its predeclared "
            "affine residual bounds."
            if formal
            else "- Measurement smoke does not promote a prediction, recovery, or differential claim."
        ),
        "- It does not compare Arm4 with a golden, hand-tuned, prior-agent, or alternate compiler.",
        "- Spike L2 provides no citable hardware-cycle result.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="historical sealed Verilator campaign")
    source.add_argument("--experiment-manifest", type=Path, help="current parent experiment manifest")
    parser.add_argument("--manifest-sha256", help="expected current parent manifest SHA-256")
    parser.add_argument("--candidate-record", action="append", default=[], metavar="TRIAL=PATH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.experiment_manifest is not None:
        if not args.manifest_sha256 or not args.candidate_record or args.output is None:
            parser.error("current reports require --manifest-sha256, --candidate-record and --output")

        parent = CURRENT.read_immutable_json(
            args.experiment_manifest, args.manifest_sha256, label="parent experiment manifest"
        )
        if not isinstance(parent, dict) or parent.get("schema") != "merlin.agentic-performance-experiment.v1":
            raise CURRENT.ReportingGateError("unsupported current parent experiment schema")
        pins = {row["trial"]: row["agent_evidence_sha256"] for row in parent.get("trials", []) if isinstance(row, dict)}
        handoffs = {}
        for value in args.candidate_record:
            trial, separator, path = value.partition("=")
            if not separator or not trial or not path or trial in handoffs:
                parser.error("candidate records must be unique TRIAL=PATH bindings")
            if trial not in pins:
                raise CURRENT.ReportingGateError("candidate record trial is absent from the parent manifest")
            CURRENT.read_immutable_json(Path(path), pins[trial], label="parent-pinned candidate record")
            handoffs[trial] = VERIFY.verify_candidate_handoff(Path(path), verify_authoring_tools=False)
            if handoffs[trial].record_sha256 != pins[trial]:
                raise CURRENT.ReportingGateError("candidate record changed during verification")
        parent, rows, result = CURRENT.load_current_experiment(
            args.experiment_manifest, expected_sha256=args.manifest_sha256, handoffs=handoffs
        )
        report = render_current_report(parent, rows, result, args.manifest_sha256)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output} ({len(rows)} exact GSIM timing cells; all declared trials)")
        return 0
    if args.manifest_sha256 or args.candidate_record:
        parser.error("parent hashes and candidate bindings require --experiment-manifest")
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise CURRENT.ReportingGateError("performance run ID must be an explicit simple directory name")
    run = PB.RUNS / args.run_id
    campaign, rows, counts = PR.load_reportable_run(run)
    report = render_report(args.run_id, campaign, rows, counts)
    out = args.output or (PB.REPORTS / f"{args.run_id}_arm4_performance.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    claim_label = (
        f"PK CLAIM {campaign['claim_status']}"
        if campaign["experiment_mode"] == "formal_claim"
        else "CLAIM NOT ESTABLISHED"
    )
    print(f"wrote {out} ({counts['expected']} exact Arm4 cells; {claim_label})")
    return 0


def render_current_report(parent: dict, rows: list[dict], result: dict, manifest_sha256: str) -> str:
    """Render recomputed current evidence without translating it into legacy claim fields."""
    aggregate = result["aggregate"]
    lines = [
        "# Paired GSIM performance experiment",
        "",
        f"Parent manifest SHA-256: `{manifest_sha256}`",
        f"Experiment: `{parent['declaration']['experiment_id']}`",
        f"Verified timing cells: {len(rows)}; every predeclared trial and measurement phase included.",
        f"Median paired speedup across independent trials: {aggregate['median_speedup']}",
        "",
        "GSIM supplies RTL timing; Spike is a correctness screen. Verilator is used only for",
        "prelaunch engine qualification. This report preserves the recorded cohort and does not",
        "mint a legacy PK prediction/recovery verdict or select the best agent trial.",
        "",
        "## Recomputed statistics",
        "",
        "```json",
        json.dumps(result, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
