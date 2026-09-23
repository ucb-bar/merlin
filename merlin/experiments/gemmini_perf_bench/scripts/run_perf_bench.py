#!/usr/bin/env python3
"""Run the Gemmini performance corpus on one frozen, functionally complete Arm-4 compiler.

The runner deliberately has no "latest submission" discovery and no alternate learned/compiler arm.
The caller supplies the exact functional run ID and submission SHA-256.  The submission is copied into
this campaign, mounted read-only in a credential-free/networkless bwrap, and checked against its
functional fork before and after the corpus.  A campaign is GO only when every expected Arm-4
kernel/simulator cell is correct and reports a positive cycle count.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Mapping
from pathlib import Path

import _pbcommon as PB
import yaml
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import measurement_support as MS

from merlin.benchharness import hash_tree
from merlin.benchharness import runs_root as _runs_root
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.target_experiment import load_target_experiment

_FUNCTIONAL_RUNS = _runs_root(PB.TARGET, "capsule-bench")
_CONTRACT = str(PB.REPO / "merlin/contract")
_DESCRIPTOR = PB.REPO / "merlin/experiments/capsule_bench/targets" / PB.TARGET / "target_experiment.yaml"
_FIXED_PROFILE_FAMILY = "fixed_profile"
_FIXED_PROFILE_REPLICATE = "r000"
_PHYSICAL_BYTE_UNIT = "BYTES"


def _selected_corpus(selection: str, kernels_root: Path = PB.KERNELS) -> list[dict]:
    doc = yaml.safe_load((kernels_root / "kernel_corpus.yaml").read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise PC.CampaignGateError("performance kernel corpus is not a mapping")
    corpus = [
        row
        for section in ("golden_kernels", "model_kernels", "attention_kernels", "conv_kernels", "movement_kernels")
        for row in (doc.get(section) or [])
    ]
    if selection != "all":
        wanted = {value.strip() for value in selection.split(",") if value.strip()}
        known = {str(row.get("id")) for row in corpus}
        missing = sorted(wanted - known)
        if missing:
            raise PC.CampaignGateError(f"unknown performance kernel id(s): {missing}")
        corpus = [row for row in corpus if str(row.get("id")) in wanted]
    if not corpus:
        raise PC.CampaignGateError("performance selection contains zero kernels")
    names = [str(row.get("id") or "") for row in corpus]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise PC.CampaignGateError("performance corpus has missing or duplicate kernel ids")
    return corpus


def _sims_for(kernel: dict, requested: str) -> tuple[str, ...]:
    if requested == "auto":
        return ("spike", "verilator") if kernel.get("sim_hint") == "L2+L3" else ("spike",)
    sims = tuple(value.strip() for value in requested.split(",") if value.strip())
    if not sims or len(sims) != len(set(sims)) or any(s not in ("spike", "verilator") for s in sims):
        raise PC.CampaignGateError("--sims must be auto, spike, or a unique spike,verilator list")
    return sims


def _expected_cells(corpus: list[dict], requested: str) -> tuple[PC.PerfCell, ...]:
    """Expand the fixed profiling corpus into the exact identities its completion gate expects."""
    return tuple(
        PC.PerfCell(_FIXED_PROFILE_FAMILY, str(kernel["id"]), simulator, _FIXED_PROFILE_REPLICATE)
        for kernel in corpus
        for simulator in _sims_for(kernel, requested)
    )


def _completion_rows(capsule: str, arm: dict, sims: tuple[str, ...]) -> list[dict]:
    """Project one legacy profiler record into exact, simulator-specific completion evidence."""
    rows: list[dict] = []
    per_sim = arm.get("per_sim") or {}
    for simulator in sims:
        result = per_sim.get(simulator)
        if not isinstance(result, dict):
            continue
        rows.append(
            {
                "family": _FIXED_PROFILE_FAMILY,
                "capsule": capsule,
                "simulator": simulator,
                "replicate": _FIXED_PROFILE_REPLICATE,
                "correct": result.get("correct"),
                "cycles": None if simulator == "spike" else result.get("cycles"),
                "provenance": result.get("provenance"),
            }
        )
    return rows


def run_arm4(
    package: Path,
    kernel: dict,
    kernel_dir: Path,
    sims: tuple[str, ...],
    capsule_runs: Path,
    timeout: int,
    target: str,
    *,
    measurement_pass: str | None = None,
    expected_package_sha256: str | None = None,
    rtl_identity: Mapping | None = None,
) -> dict:
    """Run one kernel through the frozen Arm-4 package; entrypoints are boxed by the caller."""
    result = {"approach": "arm4", "ok_build": True, "per_sim": {}}
    package_before = hash_tree(package)["sha256"]
    inputs_before = hash_tree(kernel_dir)["sha256"]
    capsule = CR.load_capsule(kernel_dir, contract=_CONTRACT)
    capsule = dict(capsule)
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2"] + (["L3"] if "verilator" in sims else [])
    # This fixed experiment measures Verilator, not the adaptive RTL-engine policy.
    adapters = {"L2": CR.simulator_adapter("spike", target), "L3": CR.simulator_adapter("verilator", target)}
    if "verilator" not in sims:
        adapters = {tier: adapter for tier, adapter in adapters.items() if tier != "L3"}
    try:
        grade = CR.run_capsule(
            capsule,
            str(package),
            runs_root=str(capsule_runs),
            run_id=(f"arm4_{kernel['id']}_{measurement_pass}" if measurement_pass else f"arm4_{kernel['id']}"),
            contract=_CONTRACT,
            oracle_adapters=adapters,
            timeout=timeout,
            target=target,
            workers=1,
        )
    except Exception as exc:  # one failed cell is recorded; the global completion gate still refuses
        result.update(
            {
                "ok_build": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "traceback": traceback.format_exc()[-1600:],
            }
        )
        return result
    result["status"] = grade.get("status")
    numeric = grade.get("numeric")
    result["numeric"] = numeric.get("status") if isinstance(numeric, dict) else numeric
    work_volume = grade.get("work_volume") if isinstance(grade.get("work_volume"), dict) else {}
    result["work_volume"] = work_volume
    command_artifact = grade.get("command_buffer_artifact")
    if isinstance(command_artifact, Mapping):
        result["command_buffer_artifact"] = dict(command_artifact)
    rtl_facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    rtl_facts_sha256 = rtl_facts.get("sha256") if isinstance(rtl_facts, Mapping) else None
    circt_core = rtl_identity.get("circt_core_hw") if isinstance(rtl_identity, Mapping) else None
    if MS.is_sha256(rtl_facts_sha256):
        result["rtl_facts_sha256"] = rtl_facts_sha256
    if isinstance(circt_core, Mapping) and MS.is_sha256(circt_core.get("sha256")):
        result["circt_core_hw"] = dict(circt_core)
    identity, identity_refusals = MS.measurement_identity(
        package_before=package_before,
        package_after=hash_tree(package)["sha256"],
        inputs_before=inputs_before,
        inputs_after=hash_tree(kernel_dir)["sha256"],
        work_volume=work_volume,
        toolchain_shas=grade.get("toolchain_shas"),
        target=target,
        expected_package_sha256=expected_package_sha256,
        rtl_facts_sha256=rtl_facts_sha256,
    )
    result["measurement_identity"] = identity
    result["measurement_identity_refusals"] = identity_refusals
    tiers = grade.get("tiers") or {}
    for sim, tier in (("spike", "L2"), ("verilator", "L3")):
        if sim not in sims:
            continue
        tier_result = tiers.get(tier) or {}
        status = tier_result.get("status") if isinstance(tier_result, dict) else tier_result
        cycles = tier_result.get("cycles") if isinstance(tier_result, dict) else None
        is_rtl_measurement = (
            sim != "spike"
            and isinstance(tier_result, dict)
            and tier_result.get("derived_from_rtl") is True
            and tier_result.get("cycle_accurate") is True
        )
        admitted_cycles = cycles if is_rtl_measurement else None
        exact_macs = work_volume.get("exact_macs")
        achieved = (
            exact_macs / admitted_cycles
            if isinstance(exact_macs, int) and isinstance(admitted_cycles, int) and admitted_cycles > 0
            else None
        )
        result["per_sim"][sim] = {
            "cycles": admitted_cycles,
            "correctness_cycles": cycles if sim == "spike" else None,
            "tier_status": status,
            "correct": status == "pass",
            "achieved_macs_per_cycle": achieved,
            "work_volume": work_volume,
            "provenance": {
                "tier": tier,
                "simulator": sim,
                "derived_from_rtl": tier_result.get("derived_from_rtl") is True,
                "cycle_accurate": tier_result.get("cycle_accurate") is True,
                "evidence": tier_result.get("evidence"),
            }
            if isinstance(tier_result, dict)
            else None,
            "counters": tier_result.get("counters") if isinstance(tier_result, dict) else None,
            "timing_observations": (tier_result.get("timing_observations") if isinstance(tier_result, dict) else None),
            "timing_capability": (tier_result.get("timing_capability") if isinstance(tier_result, dict) else None),
            "measurement_conditions": (
                tier_result.get("measurement_conditions") if isinstance(tier_result, dict) else None
            ),
            "utilization": tier_result.get("utilization") if isinstance(tier_result, dict) else None,
        }
        if sim != "spike" and MS.is_sha256(rtl_facts_sha256):
            result["per_sim"][sim]["rtl_facts_sha256"] = rtl_facts_sha256
    if grade.get("failure"):
        result["failure"] = {key: grade["failure"].get(key) for key in ("plane", "category", "detail")}
    return result


def _write_json(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functional-run-id", required=True, help="exact completed Arm-4 functional run directory name"
    )
    parser.add_argument(
        "--functional-submission-sha256", required=True, help="exact frozen functional submission SHA-256"
    )
    parser.add_argument(
        "--waive-functional-gate",
        action="append",
        default=[],
        metavar="PREDICATE",
        help="accept a NAMED completeness gap in the functional baseline instead of "
        "refusing (repeatable). Integrity predicates -- sandbox, answer mask, "
        "answer-access audit, cohort-admission accounting, public/hidden "
        "identity separation -- cannot be waived and asking is an error. Every "
        "accepted waiver is recorded in the campaign record and every result it "
        "produces is marked functional_gate_clean=false.",
    )
    parser.add_argument("--rtl-facts", help="exact CIRCT-extracted RTL facts JSON for performance provenance")
    parser.add_argument("--kernels", default="all")
    parser.add_argument(
        "--approach",
        choices=("arm4",),
        default="arm4",
        help="only the Arm-4 compiler lane is admitted in this campaign",
    )
    parser.add_argument("--sims", default="auto", help="auto (per-kernel hint), spike, verilator, or spike,verilator")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--run-id", default="perf_0001")
    parser.add_argument(
        "--hardware-counters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="instrument cycle windows with a counter set sized from elaborated RTL",
    )
    parser.add_argument(
        "--counter-unit", help="byte-counter unit family for pass two (default: BYTES from target header)"
    )
    args = parser.parse_args(argv)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise PC.CampaignGateError("performance run id must be a simple directory name")
    if args.timeout <= 0:
        raise PC.CampaignGateError("performance cell timeout must be positive")
    if args.counter_unit is not None:
        unit = str(args.counter_unit).strip()
        if not args.hardware_counters or not unit or any(not (char.isalnum() or char == "_") for char in unit):
            raise PC.CampaignGateError("--counter-unit requires hardware counters and must be one identifier token")
        if unit.upper() != _PHYSICAL_BYTE_UNIT:
            raise PC.CampaignGateError(
                "the linked physical-byte pass requires the BYTES unit declared by the target header"
            )
    physical_unit = str(args.counter_unit).upper() if args.counter_unit else _PHYSICAL_BYTE_UNIT
    if not args.rtl_facts:
        raise PC.CampaignGateError("--rtl-facts is required for content-linked RTL performance")
    rtl_identity = MS.load_rtl_identity(Path(args.rtl_facts), PB.TARGET)
    counter_binding = MS.probe_counter_byte_bindings(rtl_identity, target=PB.TARGET)

    functional = PC.inspect_functional_run(
        _FUNCTIONAL_RUNS,
        args.functional_run_id,
        args.functional_submission_sha256,
        waive=frozenset(args.waive_functional_gate or ()),
    )
    _selected_corpus(args.kernels)  # validate the requested IDs before allocating the fresh run dir
    out_dir = PB.RUNS / args.run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise PC.CampaignGateError(f"performance run directory already exists; choose a fresh --run-id: {out_dir}")

    snapshot = PC.materialize_perf_workspace(functional, out_dir / "_frozen_functional")
    workload_root = out_dir / "_frozen_workload" / "kernels"
    workload_digest = PC.materialize_readonly_tree(PB.KERNELS, workload_root)
    corpus = _selected_corpus(args.kernels, workload_root)
    sims_by_capsule = {str(kernel["id"]): _sims_for(kernel, args.sims) for kernel in corpus}
    expected = _expected_cells(corpus, args.sims)
    fork = PC.functional_fork(functional)
    before = PC.check_fork(fork, snapshot)
    if before.ok is not True:
        raise PC.CampaignGateError(f"functional fork does not hold before performance: {before.reason}")
    fork_record = fork.to_dict()
    fork_record.update(
        {
            "functional_run_id": functional.run_id,
            "functional_submission_sha256": functional.digest,
            "copied_submission": str(snapshot),
        }
    )
    _write_json(out_dir / "functional_fork.json", fork_record)

    target_experiment = load_target_experiment(_DESCRIPTOR)
    probe_workspace = out_dir / "_probe_workspace"
    probe_workspace.mkdir()
    probe_policy = PC.package_sandbox_policy(target_experiment, probe_workspace, snapshot)
    campaign = {
        "status": "NO_GO",
        "approach": args.approach,
        "functional_run_id": functional.run_id,
        "functional_submission_sha256": functional.digest,
        "functional_public_capsules": functional.public_capsules,
        "functional_hidden_capsules": functional.hidden_capsules,
        # A campaign launched over a waived gate is still a real measurement, but it is NOT the same
        # claim as one whose baseline was fully established. Both facts ride in the record so a reader
        # never has to reconstruct which it was: `false` here means the numbers below are conditional
        # on the named gaps, and any write-up must say so.
        "functional_gate_clean": functional.gate_clean,
        "functional_gate_deviations": [d.to_dict() for d in functional.deviations],
        "snapshot": str(snapshot),
        "snapshot_sha256": functional.digest,
        "workload_snapshot": str(workload_root),
        "workload_sha256": workload_digest,
        "instrumentation": {
            "hardware_counters": args.hardware_counters,
            "mode": "linked_multi_pass" if args.hardware_counters else "disabled",
            "applies_to": "verilator_cells" if args.hardware_counters else None,
            "passes": (
                [
                    {"id": "occupancy", "selection": "joint_occupancy"},
                    {
                        "id": "physical_bytes",
                        "unit_family": physical_unit,
                        "semantic_resolution": "raw_named_readings_only",
                    },
                ]
                if args.hardware_counters
                else []
            ),
            "capacity_source": "elaborated CIRCT HW",
            "rtl_identity": rtl_identity,
            "counter_byte_binding": counter_binding,
        },
        "expected_cells": [
            {"family": cell.family, "capsule": cell.capsule, "simulator": cell.simulator, "replicate": cell.replicate}
            for cell in expected
        ],
        "fork_before": before.to_dict(),
        "fork_after": None,
        "sandbox": {
            "engine": "bwrap",
            "network": "unshared",
            "package_read_only": True,
            "answer_surface_coverage_gap": list(probe_policy.coverage_gap),
            "required_tool_probes": [probe.label for probe in probe_policy.required_tools],
            "tool_probe_results": [],
        },
        "completion": PC.completion_report([], expected),
        "refusal": "campaign has not completed",
    }
    _write_json(out_dir / "campaign_manifest.json", campaign)

    results: list[dict] = []
    completion_rows: list[dict] = []
    refusal: str | None = None
    try:
        campaign["sandbox"]["tool_probe_results"] = PC.run_tool_probes(probe_policy)
        _write_json(out_dir / "campaign_manifest.json", campaign)
        cells_root = out_dir / "_cell_workspaces"
        cells_root.mkdir()
        for kernel in corpus:
            name = str(kernel["id"])
            sims = sims_by_capsule[name]
            shape = kernel.get("shape") or (
                f"{kernel.get('M')}x{kernel.get('K')}x{kernel.get('N')}" if kernel.get("M") is not None else "?"
            )
            print(f"\n=== Arm-4 kernel {name} ({shape}, sims={list(sims)}) ===", flush=True)
            # Each pass gets a fresh writable mount. The package cannot inspect oracle/result files
            # from an earlier pass or cell; capsule_runner copies this cell's interface MLIR into
            # generated/ before the first boxed entrypoint and keeps the source corpus outside the mount.
            cell_workspace = cells_root / name
            cell_workspace.mkdir()

            def run_one(pass_name: str) -> dict:
                pass_workspace = cell_workspace / pass_name
                pass_workspace.mkdir()
                capsule_runs = pass_workspace / "capsule_runs"
                capsule_runs.mkdir()
                cell_policy = PC.package_sandbox_policy(target_experiment, pass_workspace, snapshot)
                with PC.boxed_entrypoints(cell_policy):
                    return run_arm4(
                        snapshot,
                        kernel,
                        workload_root / name,
                        sims,
                        capsule_runs,
                        args.timeout,
                        target_experiment.target,
                        measurement_pass=pass_name,
                        expected_package_sha256=functional.digest,
                        rtl_identity=rtl_identity,
                    )

            if args.hardware_counters and "verilator" in sims:
                arm = MS.collect_linked_counter_passes(
                    run_one,
                    physical_unit=physical_unit,
                    counter_binding=counter_binding,
                    rtl_facts_sha256=rtl_identity["rtl_facts"]["sha256"],
                )
            else:
                with MS.counter_environment(enabled=False):
                    arm = run_one("unprofiled")
            cell = {
                "kernel": name,
                "shape": shape,
                "work_volume": arm.get("work_volume"),
                "command_buffer_artifact": arm.get("command_buffer_artifact"),
                "resource_bindings": MS.resource_bindings(arm),
                "output_dtype": kernel.get("output_dtype", ""),
                "source": kernel.get("source"),
                "sim_hint": kernel.get("sim_hint"),
                "approaches": {"arm4": arm},
            }
            results.append(cell)
            completion_rows.extend(_completion_rows(name, arm, sims))
            _write_json(out_dir / f"{name}.json", cell)
            _write_json(out_dir / "completion_cells.json", completion_rows)
            campaign["completion"] = PC.completion_report(completion_rows, expected)
            _write_json(out_dir / "campaign_manifest.json", campaign)
            linked = arm.get("linked_counter_evidence")
            if (
                args.hardware_counters
                and "verilator" in sims
                and (not isinstance(linked, dict) or linked.get("status") != "linked")
            ):
                reasons = linked.get("refusals") if isinstance(linked, dict) else ["missing linkage"]
                raise PC.CampaignGateError(f"{name} counter passes could not be linked: {reasons}")
            summary = {sim: (row.get("cycles"), row.get("correct")) for sim, row in arm.get("per_sim", {}).items()}
            print(f"  [arm4] {summary}", flush=True)
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    finally:
        _write_json(out_dir / "perf_results.json", results)
        auxiliary = MS.roofline_auxiliary_requirements(results, rtl_identity)
        _write_json(out_dir / "roofline_auxiliary_evidence.json", auxiliary)
        campaign["roofline_evidence"] = auxiliary
        coverage = MS.compute_axis_coverage(results)
        _write_json(out_dir / "compute_axis_coverage.json", coverage)
        campaign["compute_axis_coverage"] = coverage
        _write_json(out_dir / "completion_cells.json", completion_rows)
        after = PC.check_fork(fork, snapshot)
        campaign["fork_after"] = after.to_dict()
        if after.ok is not True:
            refusal = f"functional fork changed during performance: {after.reason}"
        try:
            campaign["completion"] = PC.completion_report(completion_rows, expected)
            if refusal is None and not campaign["completion"]["complete"]:
                counts = campaign["completion"]
                refusal = (
                    f"Arm-4 performance reported {counts['reported']} of "
                    f"{counts['expected']} expected cells; {counts['failed']} reported "
                    "cell(s) failed simulator-specific completion evidence"
                )
        except PC.CampaignGateError as exc:
            if refusal is None:
                refusal = str(exc)
        campaign["refusal"] = refusal
        campaign["status"] = "GO" if refusal is None else "NO_GO"
        _write_json(out_dir / "campaign_manifest.json", campaign)

    # NEXT TO THE HEADLINE, not only per row: a reader who skips the cells still sees how many
    # members' cycles have no counted work behind them, because that is the denominator any
    # utilization number quoted off this campaign is missing.
    _coverage = campaign.get("compute_axis_coverage")
    if isinstance(_coverage, Mapping):
        print(f"\ncompute axis: {_coverage['headline']}", flush=True)
        for _row in _coverage.get("unattributed", []):
            print(f"  [no compute axis] {_row['kernel']}: {'; '.join(_row['reasons'])}", flush=True)
    if refusal is not None:
        print(f"\nNO-GO: {refusal}\nmanifest: {out_dir / 'campaign_manifest.json'}", flush=True)
        return 2
    print(
        f"\nGO: completed {campaign['completion']['expected']} Arm-4 cells; "
        f"manifest: {out_dir / 'campaign_manifest.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
