"""``merlin-dse-guidance`` CLI.

Two modes:

  * single workload — build the flat/multi-rate reports, the flat-vs-multi-rate diff, the axis
    triage, and (when a measurement source is supplied) the CPU-coupling result and calibration
    anchor for one workload;
  * ``--study`` — run the exhaustive cross-workload study over every supported workload,
    synthesizing an analytical baseline + temporal view where no measured fixture exists.

Logic lives in :mod:`merlin.dse_guidance`; this is a thin wrapper. It reuses the
``design_pressure`` region loading and pressure-vector machinery (via
:mod:`merlin.dse_guidance.loader` / ``representation``) and never duplicates DSE cost logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from merlin.common import paths
from merlin.dse_guidance import aet_ingest, baseline_cost as BC, loader, synth, temporal as T
from merlin.dse_guidance.pipeline import run_guidance, write_artifacts
from merlin.dse_guidance.study import (discover_model_specs, discover_specs, run_model_study,
                                        run_study)


def _single(args) -> int:
    region = None
    if args.region_yaml or args.workload:
        try:
            region = loader.load_region(args.workload, args.region_yaml, H=args.H)
        except SystemExit:
            region = None  # may be a pure-temporal workload (e.g. smolvla_action_head)

    if args.temporal_metadata:
        temporal = T.load(args.temporal_metadata)
    elif region is not None:
        print("note: no --temporal-metadata; synthesizing from region reuse structure")
        temporal = T.parse(synth.synth_temporal(region, control_rate_hz=args.control_rate_hz))
    else:
        raise SystemExit("need --temporal-metadata (or a resolvable --workload/--region-yaml)")

    baseline = None
    if args.baseline_cost:
        baseline = BC.load(args.baseline_cost)
    elif region is not None and not args.structural_only:
        print("note: no --baseline-cost; synthesizing analytical baseline from region")
        baseline = BC.parse(synth.analytical_baseline_cost(region))
    else:
        print("note: no baseline cost — emitting structural outputs only (topology, capture "
              "fidelity, candidate axes). Quantitative triage needs --baseline-cost.")

    wl = baseline.workload if baseline is not None else temporal.workload
    coupling = aet_ingest.ingest(args.cpu_coupling, args.aet_run, workload=wl)

    result = run_guidance(temporal, baseline, region=region, coupling=coupling)
    out = Path(args.out) if args.out else (
        paths.repo_root() / "output" / "dse_guidance" / wl)
    write_artifacts(result, out)

    for wmsg in result.warnings:
        print(f"warning: {wmsg}")
    print(f"workload={wl}  class={result.topology.workload_class}  "
          f"capture-fidelity severity={result.fidelity.severity}")
    print(f"  structural DSE candidates: {[c.axis for c in result.candidate_axes]}")
    if result.triage_multirate is not None:
        print(f"  (quantitative, uncalibrated) flat top      = {_top_names(result.triage_flat)}")
        print(f"  (quantitative, uncalibrated) multirate top = {_top_names(result.triage_multirate)}")
    print(f"artifacts -> {out}")
    return 0


def _top_names(tr: dict, n: int = 3) -> list[str]:
    return [r["axis"] for r in tr["axes"]
            if r["priority_score"] and r["legality"]][:n]


def _design_envelope(args) -> int:
    from merlin.common.artifacts import Artifact, yaml_artifact
    from merlin.dse_guidance import (attribution as ATTR, design_envelope as DE,
                                     numerical_contract as NC, topology as TOP)
    from merlin.dse_guidance.case_study import analyze, available_models, _recap_dir
    from merlin.common.yaml import load_yaml
    design = load_yaml(args.design_candidate) if args.design_candidate else None
    out = Path(args.out) if args.out else (
        paths.repo_root() / "output" / "dse_guidance" / "design_envelope")

    envs = []
    if args.capture_dir and args.temporal_metadata:
        topo = TOP.load(args.temporal_metadata)
        attr = ATTR.attribute(args.capture_dir, topo)
        dtype = NC.extract_numerical_facts(args.capture_dir).get("compute_dtype", "f32")
        env = DE.from_recovered(topo, attr, captured_dtype=dtype, design=design)
        if env is None:
            raise SystemExit("no repeated_head attributed in the capture — cannot derive envelope")
        envs.append(env)
    else:
        for w in available_models():
            c = analyze(w)
            dtype = NC.extract_numerical_facts(str(_recap_dir(w))).get("compute_dtype", "f32")
            env = DE.from_recovered(c.topo, c.attribution, captured_dtype=dtype, design=design)
            if env:
                envs.append(env)
    if not envs:
        raise SystemExit("no design envelopes derived (no capture / recaptures available)")
    for env in envs:
        yaml_artifact(f"{env.workload}/design_envelope.yaml", DE.to_yaml_obj(env),
                      header=f"design_envelope: {env.workload}").write(out)
        Artifact(f"{env.workload}/design_envelope.md", DE.markdown(env)).write(out)
    Artifact("requirements_table.csv", DE.requirements_csv(envs)).write(out)
    print(f"design envelopes (requirements, not calibration) for {len(envs)} workload(s); "
          f"design candidate={'yes' if design else 'none'}")
    print(f"artifacts -> {out}  (requirements_table.csv, <workload>/design_envelope.{{yaml,md}})")
    return 0


def _study(args) -> int:
    if args.models:
        specs = discover_model_specs()
        default_sub = "study_models"
        kind = "real model captures"
    else:
        specs = discover_specs()
        default_sub = "study"
        kind = "semantic_memory regions"
    if not specs:
        raise SystemExit(f"no workloads discovered for the study ({kind})")
    out = Path(args.out) if args.out else (
        paths.repo_root() / "output" / "dse_guidance" / default_sub)
    summary = run_model_study(out) if args.models else run_study(specs, out)
    print(f"studied {summary['n_workloads']} workloads ({kind}): "
          f"{', '.join(summary['workloads'])}")
    print(f"artifacts -> {out}  (study_summary.csv, study_summary.md, <workload>/...)")
    return 0


def _insight_mining(args) -> int:
    """Meta-analysis runs. The CLI builds each run folder name from scope + UTC timestamp +
    '_dse_analysis' under results/ (regeneratable, not committed). Per-network + combined 'all'."""
    from datetime import datetime, timezone
    from merlin.common import paths
    from merlin.dse_guidance import insight_mining as IM
    from merlin.dse_guidance import presentation_plots as PP
    cs_dir = Path(args.case_study_dir) if args.case_study_dir else (
        paths.merlin_dir() / "benchmarks" / "dse_guidance" / "case_study")
    if not (cs_dir / "dse_contract.json").is_file() and not (cs_dir / "critical_path_table.csv").is_file():
        raise SystemExit(f"no case-study artifacts under {cs_dir} (run --case-study first)")
    base = Path(args.out) if args.out else (paths.repo_root() / "results")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nets = IM._workloads(cs_dir)
    scopes = [args.workload] if args.workload else (nets + ["all"])
    all_ok = True
    for scope in scopes:
        run_dir = base / f"{scope}_{ts}_dse_analysis"
        bundle = IM.mine(cs_dir, scope)
        rendered = PP.render_plots(bundle["plots"], cs_dir, bundle["facts"],
                                   run_dir / "generated_plots")
        IM.emit_run(bundle, run_dir, rendered)
        npass = sum(1 for ok, _ in bundle["consistency_checks"] if ok)
        open_gaps = len(bundle.get("open_avoidable_gaps", []))
        passed = all(ok for ok, _ in bundle["consistency_checks"]) and open_gaps == 0
        all_ok = all_ok and passed
        s = bundle["evidence_strength"]
        print(f"{scope:14s} -> {run_dir}")
        print(f"               facts={s['total_facts']} tiers(A/B/C/D)="
              f"{s['by_tier'].get('A',0)}/{s['by_tier'].get('B',0)}/{s['by_tier'].get('C',0)}/"
              f"{s['by_tier'].get('D',0)} main_findings="
              f"{sum(1 for f in bundle['findings'] if f['presentation_placement']=='main')} "
              f"plots={len(rendered)} consistency={npass}/{len(bundle['consistency_checks'])} "
              f"open_avoidable_gaps={open_gaps} {'PASS' if passed else 'FAIL'}")
    return 0 if all_ok else 1


def _devils_advocate(args) -> int:
    """Run the agent critic over an emitted insight-mining run folder (propose) and keep only
    citation-grounded critiques (dispose). The agent is optional — if `claude` is unavailable we
    report that honestly and exit nonzero, never fabricate a critique."""
    from merlin.dse_guidance.agent import critic
    from merlin.dse_guidance.agent.claude_cli import AgentError
    run_dir = Path(args.devils_advocate)
    if not run_dir.is_dir() or not (run_dir / "DSE_FINDINGS.md").is_file():
        raise SystemExit(f"{run_dir} is not an insight-mining run (no DSE_FINDINGS.md); "
                         "run --insight-mining first")
    try:
        result = critic.run_critic(run_dir)
    except AgentError as e:
        print(f"devil's-advocate agent unavailable: {e}")
        return 2
    out = critic.emit_critique(result, run_dir)
    print(f"critic: {len(result['accepted'])} grounded over-claims kept / "
          f"{result['n_proposed']} proposed ({len(result['rejected'])} rejected ungrounded)")
    print(f"-> {out}")
    return 0


def _query(args) -> int:
    """Consume the case-study contract manifest (dse_contract.json) — the one-object entry point."""
    import json
    from merlin.common import paths
    base = Path(args.out) if args.out else (
        paths.merlin_dir() / "benchmarks" / "dse_guidance" / "case_study")
    manifest_path = base / "dse_contract.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no dse_contract.json under {base}; run --case-study first")
    m = json.loads(manifest_path.read_text())
    topic, _, arg = args.query.partition(":")
    topic = topic.strip().lower()

    if topic == "summary":
        print(f"workloads: {', '.join(m['workloads'])}")
        for w, d in sorted(m["per_workload"].items()):
            print(f"  {w}: class={d['class']} K={d['K']} roles={d['roles']} "
                  f"structural_dse={d['ready_structural_dse']} "
                  f"quantitative_dse={d['ready_quantitative_dse']} acc_int8={d['accuracy_int8']}")
        print(f"\nnot claimed: {m['what_is_not_claimed']}")
    elif topic == "knobs":
        for g in m["search_space_knob_groups"]:
            print(f"  [{g['source_phase']:6s}] {g['group']:28s} enabled={g['enabled']} "
                  f"n_knobs={g['n_knobs']}")
    elif topic == "boundary":
        bp = m["boundary_placement"]
        if arg:
            from merlin.common.yaml import load_yaml
            certs = load_yaml(base / "boundary_candidate_contracts.yaml")[
                "boundary_candidate_contracts"]["certificates"]
            c = next((x for x in certs if x["abstraction"] == arg.strip()), None)
            if c is None:
                raise SystemExit(f"unknown abstraction '{arg}'; see --query boundary")
            print(f"{c['abstraction']}  (pressure/evidence={c['boundary_pressure_score']}, "
                  f"workloads={c['supporting_workloads']})")
            print(f"  compiler proof: {c['required_compiler_proof']} "
                  f"[{c['compiler_proof_status']}]")
            for b in c["boundary_levels"]:
                print(f"  {b['level']:32s} {b['status']:18s} sw='{b['software_manages'][:40]}'")
        else:
            print(f"boundary score = {bp['score_is']}; levels = {', '.join(bp['levels'])}")
            for t in bp["top_by_evidence_breadth"]:
                print(f"  {t['abstraction']:28s} score={t['boundary_pressure_score']} "
                      f"strong@={t['strong_levels']}")
            print("\nquery a placement: --query boundary:<abstraction>")
    elif topic == "missing":
        print("measurements needed before quantitative DSE:")
        for mm in m["measurements_needed_before_quantitative_dse"]:
            print(f"  - {mm}")
    elif topic == "index":
        for k, v in sorted(m["artifacts_index"].items()):
            print(f"  {k:24s} -> {v}")
    else:
        raise SystemExit("query must be one of: summary | knobs | boundary[:<abstraction>] | "
                         "missing | index")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-dse-guidance", description=__doc__)
    ap.add_argument("--workload", default=None,
                    help="synthetic name or semantic_memory benchmark name")
    ap.add_argument("--region-yaml", default=None, help="explicit workload_region YAML path")
    ap.add_argument("--temporal-metadata", default=None,
                    help="temporal_workload_metadata YAML (else synthesized from region)")
    ap.add_argument("--baseline-cost", default=None,
                    help="baseline_cost YAML (else analytical baseline synthesized from region)")
    ap.add_argument("--cpu-coupling", default=None, help="cpu_coupling measurements YAML")
    ap.add_argument("--aet-run", default=None,
                    help="aet run dir with metrics/summary_metrics.json (measured coupling)")
    ap.add_argument("--H", type=int, default=16, help="action horizon (synthetic region loader)")
    ap.add_argument("--control-rate-hz", type=float, default=30.0,
                    help="control rate for synthesized temporal metadata")
    ap.add_argument("--study", action="store_true",
                    help="run the exhaustive cross-workload study")
    ap.add_argument("--models", action="store_true",
                    help="with --study: study the real model zoo (captures under output/) "
                         "instead of the semantic_memory regions")
    ap.add_argument("--design-envelope", action="store_true",
                    help="derive hardware-independent requirements + roofline bounds (NOT "
                         "calibration). Over the recaptures, or a --capture-dir + --temporal-metadata")
    ap.add_argument("--capture-dir", default=None,
                    help="a dir with model.mlir to attribute (for --design-envelope)")
    ap.add_argument("--design-candidate", default=None,
                    help="optional candidate-design YAML for roofline feasibility")
    ap.add_argument("--case-study", action="store_true",
                    help="generate the cross-workload provenance case study from the real "
                         "prov.fqn recaptures under merlin/benchmarks/dse_guidance/recaptures/")
    ap.add_argument("--structural-only", action="store_true",
                    help="emit only the structural front-end (topology, capture fidelity, "
                         "candidate axes); skip the quantitative triage even if a region is given")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--insight-mining", action="store_true",
                    help="meta-analysis over the committed case-study artifacts: mine evidence, "
                         "score DSE-usefulness, extract presentation findings + plots. Writes a "
                         "regeneratable (non-committed) timestamped run under results/ per network "
                         "and a combined 'all' run. Use --workload to scope to one network.")
    ap.add_argument("--case-study-dir", default=None,
                    help="case-study artifact dir to mine (default: committed case_study)")
    ap.add_argument("--query", default=None,
                    help="consume the case-study contract manifest: one of summary | knobs | "
                         "boundary[:<abstraction>] | missing | index (reads dse_contract.json under "
                         "--out or the committed case_study dir)")
    ap.add_argument("--devils-advocate", default=None, metavar="RUN_DIR",
                    help="run the agent devil's-advocate critic over an emitted insight-mining run "
                         "folder (headless `claude -p`); a deterministic citation gate keeps only "
                         "critiques that quote a real artifact. Writes devils_advocate_critique.md")
    args = ap.parse_args(argv)

    if getattr(args, "devils_advocate", None):
        return _devils_advocate(args)
    if getattr(args, "insight_mining", False):
        return _insight_mining(args)
    if args.query:
        return _query(args)
    if args.design_envelope:
        return _design_envelope(args)
    if args.case_study:
        from merlin.dse_guidance.case_study import run_case_study
        out = Path(args.out) if args.out else (
            paths.repo_root() / "output" / "dse_guidance" / "case_study")
        summary = run_case_study(out)
        print(f"cross-workload case study over {len(summary['workloads'])} real captures: "
              f"{', '.join(summary['workloads'])}")
        print(f"artifacts -> {out}  (case_study.md, cross_workload_provenance.csv, <workload>/...)")
        return 0
    if args.study:
        return _study(args)
    return _single(args)


if __name__ == "__main__":
    sys.exit(main())
