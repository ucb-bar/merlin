#!/usr/bin/env python3
"""Build one merged Core ATen MLIR bundle from canonical case captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.targetgen.core_aten_batch import build_core_aten_batch
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch, numeric_policy_counts
from merlin.targetgen.core_aten_device import load_execution_provider
from merlin.targetgen.core_aten_device import run_spike_bundle as _run_spike_bundle


def _passed_by_lane(verdict: dict) -> dict:
    return {
        lane: sum(case["status"] == "pass" and case.get("lane") == lane for case in verdict["cases"].values())
        for lane in ("host", "device")
    }


def _semantic_counts(verdict: dict) -> dict:
    cases = verdict["cases"].values()
    full = sum(case["status"] == "pass" and case.get("semantic_scope") == "full" for case in cases)
    return dict(
        full_semantic_passed_count=full,
        numeric_only_passed_count=verdict["passed_count"] - full,
        **numeric_policy_counts(verdict["cases"].values()),
    )


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _merge_provenance(aggregate: dict, verdict: dict, directory: Path) -> None:
    """Bind aggregate claims to every shard's actual executable and selected pins."""
    shards = dict(aggregate.get("provenance", {}).get("shards", {}))
    shards[str(directory)] = verdict["provenance"]
    aggregate["provenance"] = {**verdict["provenance"], "shards": shards}


def _regrade_existing(output_dir: Path, report: dict) -> dict:
    """Reapply the current comparison policy to retained raw hardware output bytes."""
    aggregate = grade_core_aten_batch(report, execution_error="no recorded hardware execution")
    ledgers = [output_dir / "shard_executions.json"]
    ledgers += sorted(
        output_dir.glob("recovery*_executions.json"),
        key=lambda path: (
            0 if path.stem == "recovery_executions" else int(path.stem.removeprefix("recovery").split("_", 1)[0])
        ),
    )
    for ledger in ledgers:
        if not ledger.is_file():
            continue
        for item in json.loads(ledger.read_text(encoding="utf-8")):
            directory = Path(item["directory"])
            shard = json.loads((directory / "core_aten_batch_map.json").read_text(encoding="utf-8"))
            execution = item.get("execution", {})
            output_path = directory / "spike-output-bytes.json"
            output_bytes = (
                [bytes.fromhex(value) for value in json.loads(output_path.read_text(encoding="utf-8"))]
                if execution.get("status") == "ran" and output_path.is_file()
                else None
            )
            verdict = grade_core_aten_batch(
                shard,
                output_bytes,
                output_shapes=(
                    json.loads((directory / "spike-output-shapes.json").read_text())
                    if (directory / "spike-output-shapes.json").is_file()
                    else None
                ),
                semantic_readback=(
                    json.loads((directory / "spike-semantic-readback.json").read_text())
                    if (directory / "spike-semantic-readback.json").is_file()
                    else None
                ),
                execution_error=execution.get("error") if output_bytes is None else None,
            )
            _write_json(directory / "core_aten_batch_verdict.json", verdict)
            _merge_provenance(aggregate, verdict, directory)
            for overload in item.get("case_ids", item["overloads"]):
                aggregate["cases"][overload] = verdict["cases"][overload]
    counts: dict[str, int] = {}
    for case_verdict in aggregate["cases"].values():
        status = case_verdict["status"]
        counts[status] = counts.get(status, 0) + 1
    aggregate["status_counts"] = dict(sorted(counts.items()))
    aggregate["passed_count"] = counts.get("pass", 0)
    aggregate.update(_semantic_counts(aggregate))
    aggregate["passed_by_lane"] = _passed_by_lane(aggregate)
    _write_json(output_dir / "core_aten_batch_verdict.json", aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-spike", action="store_true", help="build one ELF and run every result through Spike")
    parser.add_argument("--target", help="enable provider-selected device offload with single-case shards")
    parser.add_argument("--device-package", type=Path)
    parser.add_argument("--execution-provider", type=Path, help="explicit target-owned execution provider")
    parser.add_argument("--rtl-facts", type=Path, help="explicit RTL facts pin (also accepts MERLIN_RTL_FACTS)")
    parser.add_argument("--arena-mb", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--max-cases-per-shard",
        type=int,
        default=0,
        help="partition clean captures into deterministic batches of at most N cases",
    )
    parser.add_argument(
        "--recover-failed-shards",
        action="store_true",
        help="split failed shards from an earlier run and merge their verdicts",
    )
    parser.add_argument(
        "--recovery-source",
        type=Path,
        default=None,
        help="execution ledger to split (defaults to shard_executions.json)",
    )
    parser.add_argument(
        "--recovery-label", default="recovery", help="subdirectory and ledger stem for this recovery round"
    )
    parser.add_argument(
        "--regrade-existing", action="store_true", help="reapply current comparison policy to retained raw output bytes"
    )
    parser.add_argument("--numeric-policy", choices=("fp32", "int8_w8a8", "spec_int8"), default="fp32")
    parser.add_argument("--software-spec", type=Path, help="spec_int8: the selected software spec")
    parser.add_argument("--host-capabilities", type=Path, help="spec_int8: the selected host capabilities")
    parser.add_argument(
        "--dram-mb",
        type=int,
        help="pack the image inside this much DRAM at the default base (an RTL simulator's memory) "
        "instead of the functional simulator's sparse high-address map",
    )
    args = parser.parse_args(argv)
    if bool(args.target) != bool(args.device_package):
        parser.error("--target and --device-package must be supplied together")
    if args.numeric_policy in {"int8_w8a8", "spec_int8"} and not args.target:
        parser.error(f"--numeric-policy {args.numeric_policy} requires --target")
    if args.numeric_policy == "spec_int8" and not (args.software_spec and args.host_capabilities):
        parser.error("--numeric-policy spec_int8 requires --software-spec and --host-capabilities")
    if args.dram_mb is not None and args.dram_mb < 1:
        parser.error("--dram-mb must be positive")
    if args.target:
        try:
            load_execution_provider(args.target, args.execution_provider)
        except ValueError as exc:
            parser.error(str(exc))
        if args.max_cases_per_shard not in (0, 1):
            parser.error("device mode requires exactly one case per shard for execution attribution")
        args.max_cases_per_shard = 1
    device_options = dict(
        target=args.target,
        numeric_policy=args.numeric_policy,
        device_package=args.device_package,
        execution_provider=args.execution_provider,
        rtl_facts=args.rtl_facts,
        **(
            dict(software_spec=args.software_spec, host_capabilities=args.host_capabilities)
            if args.numeric_policy == "spec_int8"
            else {}
        ),
        **({"dram_bytes": args.dram_mb * 1024 * 1024} if args.dram_mb is not None else {}),
    )
    corpus_path = args.corpus or artifacts_dir() / "verification" / "core-aten" / "core_aten_test_cases.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    report = build_core_aten_batch(corpus, args.captures, args.output_dir)
    print(
        f"Bundled {report['bundled_count']}/{report['case_count']} cases, "
        f"{report['input_count']} inputs, {report['output_count']} outputs"
    )
    print(args.output_dir / "core_aten_batch_map.json")
    if args.max_cases_per_shard < 0:
        parser.error("--max-cases-per-shard must be nonnegative")
    if args.regrade_existing:
        verdict = _regrade_existing(args.output_dir, report)
        print(verdict["status_counts"])
        return 0 if verdict["passed_count"] == report["bundled_count"] else 1
    if args.recover_failed_shards:
        if not args.max_cases_per_shard or not args.run_spike:
            parser.error("recovery requires --run-spike and --max-cases-per-shard")
        source = args.recovery_source or args.output_dir / "shard_executions.json"
        prior = json.loads(source.read_text(encoding="utf-8"))
        aggregate = json.loads((args.output_dir / "core_aten_batch_verdict.json").read_text(encoding="utf-8"))
        recoveries = []
        for shard_record in prior:
            if shard_record.get("execution", {}).get("status") == "ran":
                continue
            names = shard_record.get("case_ids", shard_record["overloads"])
            for offset in range(0, len(names), args.max_cases_per_shard):
                selected = set(names[offset : offset + args.max_cases_per_shard])
                parent_name = Path(shard_record["directory"]).name
                directory = args.output_dir / args.recovery_label / f"{parent_name}-{offset:03d}"
                shard = build_core_aten_batch(corpus, args.captures, directory, selected_overloads=selected)
                execution, verdict = _run_spike_bundle(
                    directory, shard, arena_mb=args.arena_mb, timeout=args.timeout, **device_options
                )
                recovery = {
                    "directory": str(directory),
                    "case_ids": sorted(selected),
                    "overloads": sorted({case["overload"] for case in shard["cases"] if case["status"] == "bundled"}),
                    "execution": execution,
                    "passed": verdict["passed_count"] == shard["bundled_count"],
                    "provenance": verdict["provenance"],
                }
                recoveries.append(recovery)
                _merge_provenance(aggregate, verdict, directory)
                for overload in selected:
                    aggregate["cases"][overload] = verdict["cases"][overload]
                counts: dict[str, int] = {}
                for case_verdict in aggregate["cases"].values():
                    status = case_verdict["status"]
                    counts[status] = counts.get(status, 0) + 1
                aggregate["status_counts"] = dict(sorted(counts.items()))
                aggregate["passed_count"] = counts.get("pass", 0)
                aggregate.update(_semantic_counts(aggregate))
                aggregate["passed_by_lane"] = _passed_by_lane(aggregate)
                _write_json(args.output_dir / "core_aten_batch_verdict.json", aggregate)
                _write_json(args.output_dir / f"{args.recovery_label}_executions.json", recoveries)
                print(f"recovery {directory.name}: {execution} {verdict['status_counts']}", flush=True)
        return 0 if all(item["execution"]["status"] == "ran" and item["passed"] for item in recoveries) else 1
    if args.max_cases_per_shard:
        names = [item.get("case_id") or item["overload"] for item in report["cases"] if item["status"] == "bundled"]
        aggregate = grade_core_aten_batch(report, execution_error="batch shard has not run")
        execution_records = []
        for offset in range(0, len(names), args.max_cases_per_shard):
            selected = set(names[offset : offset + args.max_cases_per_shard])
            shard_number = offset // args.max_cases_per_shard
            directory = args.output_dir / "shards" / f"shard-{shard_number:03d}"
            shard = build_core_aten_batch(corpus, args.captures, directory, selected_overloads=selected)
            shard_record = {
                "shard": shard_number,
                "directory": str(directory),
                "case_ids": sorted(selected),
                "overloads": sorted({case["overload"] for case in shard["cases"] if case["status"] == "bundled"}),
                "output_count": shard["output_count"],
            }
            if args.run_spike:
                execution, verdict = _run_spike_bundle(
                    directory,
                    shard,
                    arena_mb=args.arena_mb,
                    timeout=args.timeout,
                    **device_options,
                )
                shard_record["execution"] = execution
                shard_record["passed"] = verdict["passed_count"] == shard["bundled_count"]
                shard_record["provenance"] = verdict["provenance"]
                _merge_provenance(aggregate, verdict, directory)
                for overload in selected:
                    aggregate["cases"][overload] = verdict["cases"][overload]
                counts: dict[str, int] = {}
                for case_verdict in aggregate["cases"].values():
                    status = case_verdict["status"]
                    counts[status] = counts.get(status, 0) + 1
                aggregate["status_counts"] = dict(sorted(counts.items()))
                aggregate["passed_count"] = counts.get("pass", 0)
                aggregate.update(_semantic_counts(aggregate))
                aggregate["passed_by_lane"] = _passed_by_lane(aggregate)
                _write_json(args.output_dir / "core_aten_batch_verdict.json", aggregate)
                print(f"shard {shard_number}: {execution} {verdict['status_counts']}", flush=True)
            execution_records.append(shard_record)
            _write_json(args.output_dir / "shard_executions.json", execution_records)
        print(f"Wrote {len(execution_records)} deterministic batch shards", flush=True)
        return (
            0
            if all(
                item.get("execution", {}).get("status") == "ran" and item.get("passed") for item in execution_records
            )
            else (1 if args.run_spike else 0)
        )
    if args.run_spike:
        execution, verdict = _run_spike_bundle(
            args.output_dir, report, arena_mb=args.arena_mb, timeout=args.timeout, **device_options
        )
        print(execution)
        print(verdict["status_counts"])
        return 0 if execution["status"] == "ran" and verdict["passed_count"] == report["bundled_count"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
