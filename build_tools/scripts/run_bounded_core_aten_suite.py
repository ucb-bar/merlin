#!/usr/bin/env python3
"""Capture, execute, and grade every bounded case with resumable process workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from merlin.common import provenance
from merlin.targetgen.capture_refusals import InputStorageUnavailable
from merlin.targetgen.core_aten_batch_grade import numeric_policy_counts
from merlin.targetgen.core_aten_bounded_runner import (
    JsonRecordMapping,
    SavedResponseAdapter,
    atomic_json,
    bounded_loader_source,
    bundle_admission_reason,
    case_digest,
    failure_family,
    grade_portable_outputs,
    semantic_observation,
)
from merlin.targetgen.core_aten_capture import case_capture_name


def git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def stamp(args):
    artifacts = {"spike": Path(os.environ["MERLIN_SPIKE"])}
    if args.extension_library:
        artifacts["extension_library"] = args.extension_library
    sources = [Path(__file__), Path(__import__("merlin.targetgen.core_aten_bounded_runner", fromlist=["x"]).__file__)]
    for relative in (
        "src/merlin/targetgen/_m2m_capture_worker.py",
        "src/merlin/targetgen/_capture_result_contract.py",
        "src/merlin/targetgen/core_aten_capture.py",
        "src/merlin/targetgen/core_aten_input_storage.py",
        "src/merlin/targetgen/core_aten_batch.py",
        "src/merlin/targetgen/core_aten_batch_grade.py",
        "src/merlin/targetgen/core_aten_semantics.py",
        "src/merlin/targetgen/core_aten_eval.py",
        "src/merlin/targetgen/core_aten_numeric.py",
        "src/merlin/targetgen/core_aten_w8a8_reference.py",
        "src/merlin/targetgen/exact_arithmetic.py",
        "src/merlin/runtime/exact_readout.py",
        "src/merlin/llvmlower/exact_quantize.py",
        "src/merlin/llvmlower/passes_xdsl.py",
        "merlin/contract/quant_schema.json",
        "merlin/contract/quant_schema_source.json",
        "src/merlin/targetgen/exact_quantized_reference.py",
        "src/merlin/targetgen/recipe_numerics.py",
        "src/merlin/targetgen/readout_facet.py",
        "src/merlin/xdsl_dialects/lowering/compute_groups.py",
        "src/merlin/xdsl_dialects/lowering/window_sum.py",
        "src/merlin/targetgen/_recipe_integer.py",
        "src/merlin/targetgen/_capture_integer_agreement.py",
        "src/merlin/targetgen/core_aten_spec_int8.py",
        "src/merlin/targetgen/core_aten_admission.py",
        "src/merlin/targetgen/native_unit_policy.py",
        "src/merlin/targetgen/software_spec.py",
        "src/merlin/targetgen/core_aten_readout.py",
        "src/merlin/targetgen/_recipe_quantizer.py",
        "src/merlin/targetgen/_m2m_capture_worker.py",
        "src/merlin/targetgen/quant_recipe.py",
        "src/merlin/targetgen/capture_cache.py",
        "src/merlin/targetgen/capsule_source.py",
        "src/merlin/targetgen/capture_refusals.py",
        "src/merlin/capture/integerization.py",
        "src/merlin/llvmlower/contraction_quantization.py",
        "src/merlin/llvmlower/torchao_affine.py",
        "src/merlin/targetgen/core_aten_device.py",
        "src/merlin/runtime/semantic_readback.py",
        "src/merlin/llvmlower/semantic_io.py",
        "src/merlin/llvmlower/result_layout.py",
        "src/merlin/llvmlower/input_layout.py",
        "src/merlin/llvmlower/pipeline.py",
        "src/merlin/llvmlower/result_buffers.py",
        "src/merlin/llvmlower/c_runtime.py",
        "merlin/runtime/abi/mlir_runtime.c",
        "src/merlin/runtime/backends/spike_model.py",
    ):
        sources.append(args.repo / relative)
    if args.facts:
        sources.append(args.facts)
    if args.execution_provider:
        sources.append(args.execution_provider)
    for path in (args.software_spec, args.host_capabilities):
        if path:
            sources.append(path)
    pins = {name: provenance.verify(name) for name in args.pin}
    head = git(args.repo, "rev-parse", "HEAD")
    export_digest = None
    diff_digest = None
    if head is None:
        # A plain git archive intentionally has no VCS metadata. Bind its
        # complete implementation and contract data instead of inventing a commit or
        # running git diff outside a repository. The campaign's independent
        # archive receipt binds the declared revision separately.
        inventory = {
            path.relative_to(args.repo).as_posix(): provenance.file_digest(path)
            for tree in ("src", "packages", "build_tools")
            for path in sorted((args.repo / tree).rglob("*"))
            if path.suffix in {".py", ".json", ".yaml", ".yml", ".c", ".h"}
            and path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
        }
        inventory_path = getattr(args, "output", args.suite.parent) / "merlin-source-export.json"
        atomic_json(inventory_path, inventory)
        sources.append(inventory_path)
        export_digest = provenance.file_digest(inventory_path)
    else:
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=args.repo)
        untracked = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "packages",
                "build_tools",
                "examples",
                "merlin/contract",
                "merlin/runtime",
            ],
            cwd=args.repo,
            text=True,
        ).splitlines()
        for relative in sorted(untracked):
            candidate = args.repo / relative
            if candidate.is_file() and not {"out", "__pycache__", ".git", ".codex-side.git"}.intersection(
                candidate.relative_to(args.repo).parts
            ):
                diff += subprocess.run(
                    ["git", "diff", "--no-index", "/dev/null", str(candidate)], capture_output=True
                ).stdout
        diff_digest = hashlib.sha256(diff).hexdigest()
    return provenance.record(
        pins=pins,
        sources=sources,
        artifacts=artifacts,
        extra={
            "merlin_commit": head,
            "merlin_diff_sha256": diff_digest,
            "merlin_source_kind": "git_worktree" if head is not None else "plain_source_export",
            "merlin_source_export_sha256": export_digest,
            "model2mlir_commit": git(args.m2m_dir, "rev-parse", "HEAD"),
            "device_package_commit": git(args.device_package, "rev-parse", "HEAD") if args.device_package else None,
            "execution_provider_sha256": provenance.file_digest(args.execution_provider)
            if args.execution_provider
            else None,
            "target": args.target,
            "dram_bytes": getattr(args, "dram_mb", None) * 1024 * 1024
            if getattr(args, "dram_mb", None) is not None
            else None,
            "software_spec_sha256": provenance.file_digest(args.software_spec) if args.software_spec else None,
            "host_capabilities_sha256": provenance.file_digest(args.host_capabilities)
            if args.host_capabilities
            else None,
            "execution_kind": "functional_simulator",
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        },
    )


def capture(args, case, block):
    from merlin.targetgen.capsule_source import PytorchRefSource
    from merlin.targetgen.core_aten_cover import _provenance_ops
    from merlin.targetgen.core_aten_input_storage import apply_native_input_storage, native_input_storage_plan

    directory = args.output / "captures" / case_capture_name(case)
    directory.mkdir(parents=True, exist_ok=True)
    loader = directory / "capsule.pytorch.py"
    source = PytorchRefSource(m2m_dir=args.m2m_dir, python=args.m2m_python, timeout=args.capture_timeout)
    os.environ["PYTHONPATH"] = str(args.m2m_dir)
    try:
        loader.write_text(bounded_loader_source(case))
        artifact = source.capture_loader(loader, "fp32", workdir=directory / "capture-work")
        mlir = directory / "capsule.linalg.mlir"
        mlir.write_text(apply_native_input_storage(artifact.linalg_mlir, case))
        actual = sorted(_provenance_ops(mlir))
        record = {
            "status": "captured_exact" if case["overload"] in actual else "captured_lost_exact_provenance",
            "capture_meta": artifact.meta,
            "input_storage_convention": native_input_storage_plan(case["overload"], case["arguments"]),
            "captured_overloads": actual,
        }
        atomic_json(directory / "inputs.json", artifact.inputs)
        atomic_json(directory / "golden.json", artifact.golden)
    except Exception as exc:
        (directory / "capsule.linalg.mlir").unlink(missing_ok=True)
        record = {"status": "capture_failed", "reason": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, InputStorageUnavailable):
            record.update(refusal_code=exc.code, refusal_details=exc.details)
    record.update(case_id=case["case_id"], overload=case["overload"], case_sha256=case_digest(case), provenance=block)
    atomic_json(directory / "capture.json", record)


def execute(args, case, block):
    from merlin.targetgen.core_aten_batch import build_core_aten_batch
    from merlin.targetgen.core_aten_device import run_spike_bundle

    directory = args.output / args.label / "shards" / case_capture_name(case)
    directory.mkdir(parents=True, exist_ok=True)
    captured = json.loads((args.output / "captures" / case_capture_name(case) / "capture.json").read_text())
    started = time.time()
    if captured["status"] == "capture_failed":
        evidence = {"lane": "unassigned", "capture": captured["status"]}
        if "refusal_code" in captured:
            evidence.update(refusal_code=captured["refusal_code"], refusal_details=captured["refusal_details"])
        observation = {
            "status": "capture_unavailable",
            "reason": captured["reason"],
            "evidence": evidence,
        }
    else:
        try:
            refusal = bundle_admission_reason(
                captured, (args.output / "captures" / case_capture_name(case) / "capsule.linalg.mlir").read_text()
            )
            if refusal:
                raise ValueError(refusal)
            batch = build_core_aten_batch({"cases": [case]}, args.output / "captures", directory)
            execution, grades = run_spike_bundle(
                directory,
                batch,
                arena_mb=args.arena_mb,
                dram_bytes=args.dram_mb * 1024 * 1024 if args.dram_mb is not None else None,
                timeout=args.execution_timeout,
                target=args.target,
                numeric_policy=args.numeric_policy,
                device_package=args.device_package,
                execution_provider=args.execution_provider,
                rtl_facts=args.facts,
                software_spec=args.software_spec,
                host_capabilities=args.host_capabilities,
            )
            grade = grades["cases"][case["case_id"]]
            numeric = {"status": "unavailable"}
            if execution["status"] == "ran":
                outputs = [
                    bytes.fromhex(item) for item in json.loads((directory / "spike-output-bytes.json").read_text())
                ]
                shapes = json.loads((directory / "spike-output-shapes.json").read_text())
                numeric = grade_portable_outputs(batch, outputs, output_shapes=shapes)["cases"][case["case_id"]]
            if execution["status"] == "failed":
                if execution.get("stage") == "numeric_preparation":
                    grade["status"] = "numeric_policy_refused"
                elif execution.get("stage") == "build" and not (directory / "spike-build" / "model.elf").exists():
                    grade["status"] = "compile_lowering_failed"
            evidence = {
                "lane": grade.get("lane", "host"),
                "numeric_verdict": numeric,
                "semantic_verdict": {key: value for key, value in grade.items() if key != "observation"},
                "routing": grade.get("routing", {}),
                "execution": execution,
                "directory": str(directory),
            }
            observation = semantic_observation(grade, evidence)
        except Exception as exc:
            observation = {
                "status": "bundle_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "evidence": {"lane": "unassigned"},
            }
    observation["numeric_policy"] = args.numeric_policy
    observation["scales"] = observation.get("evidence", {}).get("numeric_verdict", {}).get("scales", [])
    # Existing runner writes verdict-bearing documents; stamp them before completion.
    for filename in ("core_aten_batch_map.json", "core_aten_batch_verdict.json", "spike-execution.json"):
        path = directory / filename
        if path.exists():
            doc = json.loads(path.read_text())
            doc["provenance"] = {**block, "native_batch": doc.get("provenance", {})}
            atomic_json(path, doc)
    atomic_json(
        directory / "result.json",
        {
            "case_id": case["case_id"],
            "case_sha256": case_digest(case),
            "observation": observation,
            "seconds": time.time() - started,
            "numeric_policy": args.numeric_policy,
            "provenance": block,
        },
    )


def launch(args, phase, case, block):
    key = case_capture_name(case)
    directory = args.output / "captures" / key if phase == "capture" else args.output / args.label / "shards" / key
    completion = directory / ("capture.json" if phase == "capture" else "result.json")
    if completion.exists():
        saved = json.loads(completion.read_text())
        receipt = saved.get("provenance", {})
        policy_matches = phase == "capture" or saved.get("numeric_policy", "fp32") == args.numeric_policy
        if saved.get("case_sha256") == case_digest(case) and policy_matches and receipt == block:
            return "cached"
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    atomic_json(request, case)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", phase, "--case", str(request)]
    with (directory / "worker.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=(args.capture_timeout + 120) if phase == "capture" else args.worker_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = "timeout"
    if code != 0 or not completion.exists():
        reason = f"{phase} worker {code}; " + (directory / "worker.log").read_text(errors="replace")[-3000:]
        record = {
            "case_id": case["case_id"],
            "case_sha256": case_digest(case),
            "provenance": block,
            "numeric_policy": args.numeric_policy,
            "scales": [],
        }
        if phase == "capture":
            record.update(status="capture_failed", reason=reason)
        else:
            record["observation"] = {"status": "execution_failed", "reason": reason, "evidence": {"lane": "unassigned"}}
        atomic_json(completion, record)
    return "finished"


def response(args, corpus, block):
    paths = {
        case["case_id"]: args.output / args.label / "shards" / case_capture_name(case) / "result.json"
        for case in corpus["selected_cases"]
    }
    results = JsonRecordMapping(
        args.output / args.label,
        paths,
        field="observation",
        missing_value={"status": "not_run", "reason": "execution pending", "evidence": {"lane": "unassigned"}},
    )
    document = {"schema_version": 1, "results": results, "provenance": block}
    if not args.evaluate_only:
        atomic_json(args.output / args.label / "response.json", document)
    return document


def summarize(args, corpus, report, block, timings):
    by_overload = defaultdict(Counter)
    partitions = {axis: defaultdict(Counter) for axis in ("dtype", "layout", "values")}
    lanes = defaultdict(Counter)
    families = defaultdict(list)
    numeric = Counter()
    numeric_overloads = defaultdict(Counter)
    numeric_lanes = defaultdict(Counter)
    policy_verdicts = []
    for case in corpus["selected_cases"]:
        item = report["cases"][case["case_id"]]
        status = item["status"]
        evidence = item.get("evidence", {})
        by_overload[case["overload"]][status] += 1
        lanes[evidence.get("lane", "unassigned")][status] += 1
        grade = evidence.get("numeric_verdict", {})
        policy_verdicts.append(
            {
                **evidence.get("semantic_verdict", {}),
                "numeric_policy": item.get("numeric_policy", "fp32"),
                "numeric_status": grade.get("status", "unavailable"),
                "status": status,
                "lane": evidence.get("lane", "unassigned"),
            }
        )
        numeric[grade.get("status", "unavailable")] += 1
        numeric_overloads[case["overload"]][grade.get("status", "unavailable")] += 1
        numeric_lanes[evidence.get("lane", "unassigned")][grade.get("status", "unavailable")] += 1
        for axis, buckets in partitions.items():
            buckets[case["partition_assignment"].get(axis, "unspecified")][status] += 1
        if status != "pass":
            reason = item.get("reason", "")
            family, owner, cause = failure_family(status, reason, case["overload"])
            families[family].append(
                {"case_id": case["case_id"], "reason": reason, "owner": owner, "suspected_root_cause": cause}
            )
    baseline_counts, baseline_families = Counter(), Counter()
    if args.baseline:
        baseline = json.loads((args.baseline / args.label / "report.json").read_text())
        baseline_counts.update(baseline["status_counts"])
        for identifier, item in baseline["cases"].items():
            if item["status"] != "pass":
                family, _, _ = failure_family(item["status"], item.get("reason", ""), identifier.split("::")[0])
                baseline_families[family] += 1
    status_delta = {
        key: {
            "baseline": baseline_counts[key],
            "current": report["status_counts"].get(key, 0),
            "delta": report["status_counts"].get(key, 0) - baseline_counts[key],
        }
        for key in sorted(set(baseline_counts) | set(report["status_counts"]))
    }
    family_delta = {
        key: {
            "baseline": baseline_families[key],
            "current": len(families.get(key, [])),
            "delta": len(families.get(key, [])) - baseline_families[key],
        }
        for key in sorted(set(baseline_families) | set(families))
    }
    summary = {
        "case_count": report["case_count"],
        "campaign_complete": report.get("campaign_complete", False),
        "status_counts": report["status_counts"],
        "full_semantic_passed_count": sum(
            item["status"] == "pass" and item.get("semantic_scope", "full") == "full"
            for item in report["cases"].values()
        ),
        "observed_semantic_passed_count": sum(
            item["status"] == "pass" and item.get("semantic_scope") == "values+mutation+observed-metadata"
            for item in report["cases"].values()
        ),
        "numeric_only_passed_count": numeric.get("pass", 0),
        "numeric_count_policy": "secondary output-only matches, including full-contract passes",
        **numeric_policy_counts(policy_verdicts),
        "delta_by_status": status_delta,
        "delta_by_family": family_delta,
        "device_lane_evidence": {
            identifier: item.get("evidence", {}).get("routing", {}).get("execution_evidence")
            for identifier, item in report["cases"].items()
            if item.get("evidence", {}).get("lane") == "device"
        },
        "by_lane": dict(lanes),
        "numeric_status_counts": numeric,
        "per_overload": dict(by_overload),
        "partitions": partitions,
        "numeric_per_overload": dict(numeric_overloads),
        "numeric_by_lane": dict(numeric_lanes),
        "failure_families": [
            {
                "family": k,
                "count": len(v),
                "owner": v[0]["owner"],
                "suspected_root_cause": v[0]["suspected_root_cause"],
                "examples": v[:2],
            }
            for k, v in sorted(families.items(), key=lambda pair: -len(pair[1]))
        ],
        "timings": timings,
        "provenance": block,
    }
    atomic_json(args.output / args.label / "summary.json", summary)
    lines = [
        f"# Bounded Core ATen: {args.label}",
        "",
        f"Cases: {report['case_count']}",
        "",
        f"Statuses: {report['status_counts']}",
        "",
        f"Numeric evidence: {dict(numeric)}",
        "",
        "Passes validate values, mutation, aliases and observed metadata; unobserved autograd remains excluded. "
        "Numeric matches are a secondary output-only measurement.",
        "",
        f"Timings (seconds): {timings}",
        "",
        f"Lanes: {dict(lanes)}",
        "",
        f"Numeric results by lane: {dict(numeric_lanes)}",
        "",
        f"Executed numeric results by lane and policy: {summary['numeric_passed_by_lane_and_policy']}",
        "",
        f"Spec-int8 results: {summary['spec_int8_passed_count']}; original-host fallback matches under "
        f"the requested spec-int8 policy: {summary['spec_int8_host_fallback_passed_count']}",
        "",
        "| Overload | Semantic pass | Numeric match | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {counts.get('pass', 0)} | {numeric_overloads[name].get('pass', 0)} | {sum(counts.values())} |"
        for name, counts in sorted(by_overload.items())
    )
    for title, buckets in [("Lane", lanes), *[(axis, buckets) for axis, buckets in partitions.items()]]:
        statuses = sorted(report["status_counts"])
        lines += [
            "",
            f"{title} breakdown:",
            "",
            "| " + title + " | " + " | ".join(statuses) + " |",
            "| --- | " + " | ".join("---:" for _ in statuses) + " |",
        ]
        lines.extend(
            "| " + str(name) + " | " + " | ".join(str(counts.get(status, 0)) for status in statuses) + " |"
            for name, counts in sorted(buckets.items())
        )
    lines += ["", "Failure families:"]
    for title, delta in [("Status", status_delta), ("Family", family_delta)]:
        lines += ["", f"| {title} delta | Baseline | Current | Change |", "| --- | ---: | ---: | ---: |"]
        lines.extend(
            f"| {name} | {row['baseline']} | {row['current']} | {row['delta']:+d} |" for name, row in delta.items()
        )
    for item in summary["failure_families"]:
        lines += ["", f"- {item['count']}: {item['family']} ({item['owner']})", "", item["suspected_root_cause"]]
        for example in item["examples"]:
            lines += ["", f"  `{example['case_id']}`: {example['reason'].replace(chr(10), ' ')[:1500]}"]
    lines += ["", "Provenance:", "```json", json.dumps(block, indent=2), "```"]
    (args.output / args.label / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "output", "repo", "m2m-dir", "m2m-python"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--target")
    for name in ("device-package", "execution-provider", "facts", "extension-library"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--pin", action="append", default=[])
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--capture-timeout", type=int, default=300)
    p.add_argument("--execution-timeout", type=int, default=120)
    p.add_argument("--worker-timeout", type=int, default=900)
    p.add_argument("--arena-mb", type=int, default=256)
    p.add_argument("--dram-mb", type=int, help="Packed executable DRAM extent in MiB")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--worker", choices=["capture", "execute"])
    p.add_argument("--case", type=Path)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--numeric-policy", choices=("fp32", "int8_w8a8", "spec_int8"), default="fp32")
    p.add_argument("--software-spec", type=Path)
    p.add_argument("--host-capabilities", type=Path)
    args = p.parse_args()
    if args.numeric_policy != "fp32" and not args.target:
        p.error("integer numeric policies require --target")
    if args.evaluate_only:
        try:
            __import__("torch")
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise RuntimeError("--evaluate-only requires Torch in the selected interpreter") from exc
    if args.target:
        from merlin.targetgen.core_aten_device import load_execution_provider

        try:
            provider = load_execution_provider(args.target, args.execution_provider)
        except ValueError as exc:
            p.error(str(exc))
        args.execution_provider = Path(provider.__file__)
    if not 1 <= args.workers <= 96:
        p.error("workers must be between 1 and 96")
    block_path = args.output / args.label / "provenance.json"
    if args.worker:
        block = json.loads(block_path.read_text())
        # Supplementary worker diagnostics are separate from the immutable cache stamp.
        worker_sources = provenance.record(
            sources=[
                Path(__file__),
                Path(__import__("merlin.targetgen.core_aten_bounded_runner", fromlist=["x"]).__file__),
            ]
        )
        atomic_json(args.case.parent / "worker-sources.json", worker_sources)
        case = json.loads(args.case.read_text())
        (capture if args.worker == "capture" else execute)(args, case, block)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    block = stamp(args)
    atomic_json(block_path, block)
    corpus = json.loads(args.suite.read_text())
    cases = corpus["selected_cases"]
    if args.limit:
        # Spread smoke selection across overloads, layouts, and dtype partitions.
        cases = [cases[int(i * len(cases) / args.limit)] for i in range(args.limit)]
    timings = {}
    start = time.time()
    if not args.evaluate_only:
        check = subprocess.check_output(
            [str(args.m2m_python), "-c", "import m2m; print(m2m.__file__)"],
            env={**os.environ, "PYTHONPATH": str(args.m2m_dir)},
            text=True,
        )
        if not Path(check.strip()).is_relative_to(args.m2m_dir):
            raise RuntimeError(f"wrong m2m import: {check}")
        print("m2m verified", check.strip(), flush=True)
        for phase in ("capture", "execute"):
            began = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(launch, args, phase, case, block) for case in cases]
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if count % 50 == 0 or count == len(cases):
                        print(phase, count, "/", len(cases), "seconds", round(time.time() - began, 1), flush=True)
            timings[phase] = time.time() - began
    document = response(args, corpus, block)
    if not args.evaluate_only:
        atomic_json(args.output / args.label / "timings.json", timings)
    if args.limit:
        counts = Counter(document["results"][c["case_id"]]["status"] for c in cases)
        atomic_json(
            args.output / args.label / "smoke.json",
            {
                "status_counts": counts,
                "case_ids": [c["case_id"] for c in cases],
                "timings": timings,
                "provenance": block,
            },
        )
        print("smoke", dict(counts), flush=True)
        return
    # The evaluator requires Torch for contract validation; keep capture import pinned.
    if "torch" not in sys.modules:
        command = [str(args.m2m_python), str(Path(__file__).resolve()), *sys.argv[1:], "--evaluate-only"]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(args.m2m_dir), str(args.repo / "src"), *[str(x) for x in (args.repo / "packages").glob("*/src")]]
        )
        subprocess.run(command, cwd=args.repo, env=env, check=True)
        return
    from merlin.targetgen.core_aten_eval import evaluate_bounded_core_aten_suite

    report = evaluate_bounded_core_aten_suite(
        SavedResponseAdapter(document),
        corpus,
        required_execution_kind="simulator",
        validate_live_denominator=False,
        validate_eager_oracles=False,
        validate_solver_certificate=False,
        allow_unobserved_autograd=True,
        verdict_store=JsonRecordMapping(args.output / args.label / "sealed-verdicts"),
    )
    report["provenance"] = block
    report["numeric_policy"] = args.numeric_policy
    for identifier, item in report["cases"].items():
        numeric = item.get("evidence", {}).get("numeric_verdict", {})
        item["numeric_policy"] = args.numeric_policy
        item["scales"] = numeric.get("scales", [])
        if "fp32_error" in numeric:
            item["fp32_error"] = numeric["fp32_error"]
        report["cases"][identifier] = item
    report["campaign_complete"] = len(document["results"]) == len(corpus["selected_cases"]) and not any(
        item["status"] in {"not_run", "missing_result", "adapter_error"} for item in report["cases"].values()
    )
    report["validation_policy"] = (
        "sealed suite, case witnesses/schema/oracle digests checked; "
        "live denominator, eager reruns, solver reruns disabled"
    )
    atomic_json(args.output / args.label / "report.json", report)
    timings = json.loads((args.output / args.label / "timings.json").read_text())
    timings["evaluation"] = time.time() - start
    timings["total"] = sum(timings.values())
    summarize(args, corpus, report, block, timings)
    print(report["status_counts"], flush=True)


if __name__ == "__main__":
    main()
