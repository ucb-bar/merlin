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
from pathlib import Path

import yaml

import _pbcommon as PB
import perf_campaign as PC
from merlin.benchharness import runs_root as _runs_root
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.target_experiment import load_target_experiment


_FUNCTIONAL_RUNS = _runs_root(PB.TARGET, "capsule-bench")
_CONTRACT = str(PB.REPO / "merlin/contract")
_DESCRIPTOR = (PB.REPO / "merlin/experiments/capsule_bench/targets" / PB.TARGET
               / "target_experiment.yaml")


def _selected_corpus(selection: str, kernels_root: Path = PB.KERNELS) -> list[dict]:
    doc = yaml.safe_load((kernels_root / "kernel_corpus.yaml").read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise PC.CampaignGateError("performance kernel corpus is not a mapping")
    corpus = [row for section in ("golden_kernels", "model_kernels", "attention_kernels",
                                  "conv_kernels", "movement_kernels")
              for row in (doc.get(section) or [])]
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


def run_arm4(package: Path, kernel: dict, kernel_dir: Path, sims: tuple[str, ...],
             capsule_runs: Path, timeout: int, target: str) -> dict:
    """Run one kernel through the frozen Arm-4 package; entrypoints are boxed by the caller."""
    result = {"approach": "arm4", "ok_build": True, "per_sim": {}}
    capsule = CR.load_capsule(kernel_dir, contract=_CONTRACT)
    capsule = dict(capsule)
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2"] + (
        ["L3"] if "verilator" in sims else [])
    adapters = CR.default_adapters()
    if "verilator" not in sims:
        adapters = {tier: adapter for tier, adapter in adapters.items() if tier != "L3"}
    try:
        grade = CR.run_capsule(
            capsule,
            str(package),
            runs_root=str(capsule_runs),
            run_id=f"arm4_{kernel['id']}",
            contract=_CONTRACT,
            oracle_adapters=adapters,
            timeout=timeout,
            target=target,
            workers=1,
        )
    except Exception as exc:  # one failed cell is recorded; the global completion gate still refuses
        result.update({"ok_build": False, "status": "error",
                       "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                       "traceback": traceback.format_exc()[-1600:]})
        return result
    result["status"] = grade.get("status")
    numeric = grade.get("numeric")
    result["numeric"] = numeric.get("status") if isinstance(numeric, dict) else numeric
    tiers = grade.get("tiers") or {}
    for sim, tier in (("spike", "L2"), ("verilator", "L3")):
        if sim not in sims:
            continue
        tier_result = tiers.get(tier) or {}
        status = tier_result.get("status") if isinstance(tier_result, dict) else tier_result
        cycles = tier_result.get("cycles") if isinstance(tier_result, dict) else None
        result["per_sim"][sim] = {
            "cycles": cycles,
            "tier_status": status,
            "correct": status == "pass",
            "util_pct": PB.utilization_pct(kernel["macs"], cycles),
        }
    if grade.get("failure"):
        result["failure"] = {key: grade["failure"].get(key)
                             for key in ("plane", "category", "detail")}
    return result


def _write_json(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-run-id", required=True,
                        help="exact completed Arm-4 functional run directory name")
    parser.add_argument("--functional-submission-sha256", required=True,
                        help="exact frozen functional submission SHA-256")
    parser.add_argument("--kernels", default="all")
    parser.add_argument("--approach", choices=("arm4",), default="arm4",
                        help="only the Arm-4 compiler lane is admitted in this campaign")
    parser.add_argument("--sims", default="auto",
                        help="auto (per-kernel hint), spike, verilator, or spike,verilator")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--run-id", default="perf_0001")
    args = parser.parse_args(argv)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise PC.CampaignGateError("performance run id must be a simple directory name")
    if args.timeout <= 0:
        raise PC.CampaignGateError("performance cell timeout must be positive")

    functional = PC.inspect_functional_run(
        _FUNCTIONAL_RUNS, args.functional_run_id, args.functional_submission_sha256)
    _selected_corpus(args.kernels)  # validate the requested IDs before allocating the fresh run dir
    out_dir = PB.RUNS / args.run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise PC.CampaignGateError(
            f"performance run directory already exists; choose a fresh --run-id: {out_dir}")

    snapshot = PC.materialize_perf_workspace(functional, out_dir / "_frozen_functional")
    workload_root = out_dir / "_frozen_workload" / "kernels"
    workload_digest = PC.materialize_readonly_tree(PB.KERNELS, workload_root)
    corpus = _selected_corpus(args.kernels, workload_root)
    expected = {str(kernel["id"]): _sims_for(kernel, args.sims) for kernel in corpus}
    fork = PC.functional_fork(functional)
    before = PC.check_fork(fork, snapshot)
    if before.ok is not True:
        raise PC.CampaignGateError(f"functional fork does not hold before performance: {before.reason}")
    fork_record = fork.to_dict()
    fork_record.update({"functional_run_id": functional.run_id,
                        "functional_submission_sha256": functional.digest,
                        "copied_submission": str(snapshot)})
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
        "snapshot": str(snapshot),
        "snapshot_sha256": functional.digest,
        "workload_snapshot": str(workload_root),
        "workload_sha256": workload_digest,
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
    refusal: str | None = None
    try:
        campaign["sandbox"]["tool_probe_results"] = PC.run_tool_probes(probe_policy)
        _write_json(out_dir / "campaign_manifest.json", campaign)
        cells_root = out_dir / "_cell_workspaces"
        cells_root.mkdir()
        for kernel in corpus:
            name = str(kernel["id"])
            sims = expected[name]
            shape = kernel.get("shape") or (
                f"{kernel.get('M')}x{kernel.get('K')}x{kernel.get('N')}"
                if kernel.get("M") is not None else "?")
            print(f"\n=== Arm-4 kernel {name} ({shape}, sims={list(sims)}) ===", flush=True)
            # Each kernel gets a fresh writable mount. The package cannot inspect oracle/result files
            # from an earlier cell; capsule_runner copies this cell's interface MLIR into generated/
            # before the first boxed entrypoint and keeps the source corpus outside the mount.
            cell_workspace = cells_root / name
            cell_workspace.mkdir()
            capsule_runs = cell_workspace / "capsule_runs"
            capsule_runs.mkdir()
            cell_policy = PC.package_sandbox_policy(
                target_experiment, cell_workspace, snapshot)
            with PC.boxed_entrypoints(cell_policy):
                arm = run_arm4(snapshot, kernel, workload_root / name, sims,
                               capsule_runs, args.timeout, target_experiment.target)
            cell = {"kernel": name, "shape": shape, "macs": kernel["macs"],
                    "output_dtype": kernel.get("output_dtype", ""),
                    "source": kernel.get("source"), "sim_hint": kernel.get("sim_hint"),
                    "approaches": {"arm4": arm}}
            results.append(cell)
            _write_json(out_dir / f"{name}.json", cell)
            campaign["completion"] = PC.completion_report(results, expected)
            _write_json(out_dir / "campaign_manifest.json", campaign)
            summary = {sim: (row.get("cycles"), row.get("correct"))
                       for sim, row in arm.get("per_sim", {}).items()}
            print(f"  [arm4] {summary}", flush=True)
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    finally:
        _write_json(out_dir / "perf_results.json", results)
        after = PC.check_fork(fork, snapshot)
        campaign["fork_after"] = after.to_dict()
        if after.ok is not True:
            refusal = f"functional fork changed during performance: {after.reason}"
        try:
            campaign["completion"] = PC.completion_report(results, expected)
            if refusal is None and not campaign["completion"]["complete"]:
                counts = campaign["completion"]
                refusal = (f"Arm-4 performance reported {counts['reported']} of "
                           f"{counts['expected']} expected cells; {counts['failed']} reported "
                           "cell(s) failed correctness or positive-cycle measurement")
        except PC.CampaignGateError as exc:
            if refusal is None:
                refusal = str(exc)
        campaign["refusal"] = refusal
        campaign["status"] = "GO" if refusal is None else "NO_GO"
        _write_json(out_dir / "campaign_manifest.json", campaign)

    if refusal is not None:
        print(f"\nNO-GO: {refusal}\nmanifest: {out_dir / 'campaign_manifest.json'}", flush=True)
        return 2
    print(f"\nGO: completed {campaign['completion']['expected']} Arm-4 cells; "
          f"manifest: {out_dir / 'campaign_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
