#!/usr/bin/env python3
"""Candidate-aware paired Arm-4 performance measurement.

An explicit frozen corpus is labelled ``tuning`` or ``held_out``.  A pure preflight declares an
adjacent/interleaved baseline-candidate schedule before execution. Spike is a fast correctness-only
semantic screen and GSIM is the sole RTL execution and timing backend. Verilator is used only while
producing the prerequisite GSIM equivalence certificates, never in this campaign. Raw executions are
content addressed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import measurement_support as MS
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PM


def main(
    argv: list[str] | None = None,
    *,
    resolve_layout: Callable[[argparse.Namespace, TargetExperiment], dict[str, Path]] | None = None,
    source_root: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=source_root,
        required=source_root is None,
        help="explicit owner of descriptor-relative source and resource paths",
    )
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument(
        "--functional-runs-root",
        type=Path,
        required=resolve_layout is None,
        help="explicit functional run storage",
    )
    parser.add_argument(
        "--waive-functional-gate",
        action="append",
        default=[],
        metavar="PREDICATE",
        help="accept an exact named functional-completeness gap (repeatable); integrity gaps remain unwaivable",
    )
    parser.add_argument("--measurement-root", type=Path, required=resolve_layout is None)
    parser.add_argument("--contract-root", type=Path, required=resolve_layout is None)
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument(
        "--descriptor",
        type=Path,
        required=True,
        help="explicit target descriptor to verify against the sealed candidate",
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--corpus-capsules-sha256", required=True)
    parser.add_argument("--phase", choices=PM.PHASES, required=True)
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=False)
    # DECLARED, NEVER INFERRED -- the same discipline the experiment's phase width uses. Unset means
    # one, the fully serial campaign, so a launch that says nothing behaves exactly as before. The
    # width buys wall time only: cycles do not move with it, and the fan-out actually used is
    # stamped on every measured tier so no row's timing block is comparable to one it should not be.
    parser.add_argument(
        "--sim-workers",
        type=int,
        default=1,
        metavar="N",
        help="how many executions may run at once (default 1 = serial)",
    )
    args = parser.parse_args(argv)
    PM.simple_component(args.run_id, label="run id")
    if args.timeout <= 0:
        raise PC.CampaignGateError("timeout must be positive")
    if not all(
        PM.is_sha256(value)
        for value in (args.corpus_manifest_sha256, args.corpus_capsules_sha256, args.gsim_certificate_sha256)
    ):
        raise PC.CampaignGateError("corpus/certificate identities must be lowercase SHA-256 values")
    target = load_target_experiment(args.descriptor, source_root=args.source_root)
    layout = (
        resolve_layout(args, target)
        if resolve_layout is not None
        else {name: getattr(args, name) for name in ("measurement_root", "functional_runs_root", "contract_root")}
    )
    out_dir = layout["measurement_root"] / args.run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise PC.CampaignGateError(f"run directory must be fresh: {out_dir}")
    inputs = PI.load_paired_inputs(
        args.candidate_record,
        args.functional_run_id,
        args.functional_submission_sha256,
        target,
        corpus_root=args.corpus_root,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        corpus_capsules_sha256=args.corpus_capsules_sha256,
        phase=args.phase,
        corpus_manifest=args.corpus_manifest,
        gsim_certificate=args.gsim_certificate,
        gsim_certificate_sha256=args.gsim_certificate_sha256,
        waive_functional_gate=tuple(args.waive_functional_gate or ()),
        functional_runs_root=layout["functional_runs_root"],
    )
    plan = PM.build_measurement_plan(inputs)
    fanout = PM.schedule_fanout(args.sim_workers, plan, hardware_counters=args.hardware_counters)
    rtl = MS.load_rtl_identity(args.rtl_facts, target.target)
    counter_binding = MS.probe_counter_byte_bindings(rtl, target=target.target) if args.hardware_counters else None
    out_dir.mkdir(parents=True)
    before = PI.identity_guard(inputs)
    fork = PC.functional_fork(inputs.functional)
    fork_before = PC.check_fork(fork, inputs.baseline).to_dict()
    if fork_before.get("ok") is not True:
        raise PC.CampaignGateError("functional fork does not hold")
    manifest: dict[str, Any] = {
        "schema": "paired_arm4_performance_campaign_v2",
        "status": "NO_GO",
        "refusal": "campaign has not completed",
        "phase": args.phase,
        "functional_run_id": inputs.functional.run_id,
        "functional_submission_sha256": inputs.baseline_sha256,
        "candidate_record_sha256": inputs.handoff.record_sha256,
        "candidate_sha256": inputs.candidate_sha256,
        "gsim_certificate": inputs.gsim_certificate.to_dict(),
        "frozen_corpus": {
            "path": str(inputs.corpus.root),
            "manifest_sha256": inputs.corpus.manifest_sha256,
            "capsules_sha256": inputs.corpus.capsules_sha256,
            "visibility": args.phase,
        },
        "measurement_plan": plan.declaration,
        "measurement_plan_sha256": plan.declaration_sha256,
        "simulators": {"spike": "correctness_only_no_timing", "gsim": "sole_rtl_execution_and_timing_backend"},
        "engine_policy": {
            "rtl_execution_backends": ["gsim"],
            "timing_authority": "gsim",
            "verilator": "prelaunch_certificate_qualification_only",
        },
        "execution_fanout": dict(fanout),
        "rtl_identity": rtl,
        "identity_before": before,
        "identity_after": None,
        "fork_before": fork_before,
        "fork_after": None,
        "completion": ME.completion_report([], plan.expected),
        "raw_results": None,
        "roofline_evidence": None,
    }
    PM.write_json(out_dir / "campaign_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    refusal = None
    try:
        rows, roofline_cells = PM.execute_schedule(
            plan,
            out_dir,
            contract_root=layout["contract_root"],
            timeout=args.timeout,
            target_experiment=target,
            rtl_identity=rtl,
            hardware_counters=args.hardware_counters,
            counter_binding=counter_binding,
            fanout=fanout,
        )
        manifest["completion"] = ME.completion_report(rows, plan.expected)
        if not manifest["completion"]["complete"]:
            raise PC.CampaignGateError(f"paired completion failed: {manifest['completion']}")
        PM.write_json(out_dir / "paired_cycles.json", PM.paired_cycle_rows(rows))
        roofline = MS.roofline_auxiliary_requirements(roofline_cells, rtl)
        PM.write_json(out_dir / "roofline_auxiliary_evidence.json", roofline)
        manifest["roofline_evidence"] = roofline
        # HOW MANY MEMBERS' CYCLES HAVE NO COUNTED WORK BEHIND THEM, said on the manifest a reader
        # of the campaign reads. Reported, never gated: a member `work_volume` cannot price is not
        # necessarily defective, but an absence nobody is told about reads as zero, and a zero
        # denominator on a perf bench reads as infinitely fast.
        manifest["compute_axis_coverage"] = MS.compute_axis_coverage(roofline_cells)
        PM.write_json(out_dir / "compute_axis_coverage.json", manifest["compute_axis_coverage"])
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        # Preserve cancellation while recording that this campaign did not complete.
        refusal = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload = PM.canonical_bytes({"schema": "paired_arm4_result_cells_v2", "cells": rows})
        digest = PM.sha256_bytes(payload)
        result_path = out_dir / f"paired_results.{digest}.json"
        result_path.write_bytes(payload)
        result_path.chmod(0o444)
        manifest["raw_results"] = {
            "index": str(out_dir / "raw_results.index.json"),
            "paired_cells": str(result_path),
            "paired_cells_sha256": digest,
            "n_cells": len(rows),
        }
        # WHAT THIS RUN BOUGHT AND WHAT IT CARRIED, on the manifest a reader of the campaign reads.
        # A carried cell is the same number the engine returned for the same bytes, so the campaign
        # is no weaker for it -- but only if it SAYS so. A cell that states neither is a hole in the
        # record and refuses the campaign here rather than being counted as freshly measured.
        manifest["measurement_reuse"] = PM.reuse_report(rows)
        PM.write_json(out_dir / "measurement_reuse.json", manifest["measurement_reuse"])
        if refusal is None and not manifest["measurement_reuse"]["auditable"]:
            refusal = (
                f"CampaignGateError: {manifest['measurement_reuse']['unstated']} cited "
                "cell(s) do not state whether their cycles were measured here or carried "
                "from an earlier measurement of the same bytes on the same pinned engine"
            )
        try:
            after = PI.identity_guard(inputs)
            manifest["identity_after"] = after
            if after != before:
                raise PC.CampaignGateError("input identities changed")
            manifest["fork_after"] = PC.check_fork(fork, inputs.baseline).to_dict()
            if manifest["fork_after"].get("ok") is not True:
                raise PC.CampaignGateError("functional fork changed")
            if MS.load_rtl_identity(args.rtl_facts, target.target) != rtl:
                raise PC.CampaignGateError("RTL identity changed")
        except Exception as exc:
            refusal = f"{type(exc).__name__}: {exc}"
        manifest["refusal"], manifest["status"] = refusal, "GO" if refusal is None else "NO_GO"
        PM.write_json(out_dir / "campaign_manifest.json", manifest)
    _coverage = manifest.get("compute_axis_coverage")
    if isinstance(_coverage, Mapping):
        print(f"compute axis: {_coverage['headline']}")
        for _row in _coverage.get("unattributed", []):
            print(f"  [no compute axis] {_row['kernel']}: {'; '.join(_row['reasons'])}")
    if refusal:
        print(f"NO-GO: {refusal}")
        return 2
    print(f"GO: {manifest['completion']['expected']} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
