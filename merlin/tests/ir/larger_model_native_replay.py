"""Native-only observations of ten already-audited complete model compiles.

The previous campaign and its failures stay unchanged. This tool re-audits
current retained proof inputs/products without repeating Merlin's MLIR-to-LLVM
lowering or issuing proof jobs, live completions or numerical-policy approval.
Native object compilation, ordinary runtime linking and execution do run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

import kernel_benchmark_handoff_campaign as leases
import larger_model_campaign as campaign
import larger_model_gate_demo as demo

from merlin.common.owned_process import CleanupUnknown
from merlin.common.owned_process import run as run_owned
from merlin.common.paths import module_source_path, repo_root, runtime_dir

HERE = Path(__file__).resolve()
EXPECTED_CASE_COUNT = 10
NATIVE_HELPER = repo_root() / "merlin/tests/ir/larger_model_native_demo.py"
RUNTIME_MODULES = (
    "merlin.llvmlower.codegen",
    "merlin.llvmlower.toolchain",
    "merlin.llvmlower.adjacent_exp",
    "merlin.llvmlower.guarded_zero_contraction",
    "merlin.common.invocation_record",
    "merlin.common.proc",
    "merlin.common.compile_trace",
    "merlin.common.cpu_pool",
    "merlin.runtime.build_runtime",
    "merlin.common.owned_process",
    "merlin.common.owned_children",
)


def read_pinned(reference):
    """Join the independently read raw bytes to their original pin before JSON."""
    pin = demo.bound(reference)
    if pin["bytes"] > 64 << 20:
        raise ValueError("retained replay JSON exceeds64MiB")
    raw = Path(pin["path"]).read_bytes()
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("retained bytes changed during independent replay read")
    value = json.loads(raw)
    demo.bound(pin)
    return value


def current_tools():
    from merlin.verify.pass_validation import CheckerTools

    return CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), Path(os.environ["MERLIN_COMPILER_PYTHON"]))


def recheck_compile(case, tools):
    previous_case = read_pinned(case["prior_case_report"])
    if previous_case.get("id") != case["id"] or previous_case.get("compile") != case["compile_report"]:
        raise ValueError("the original case does not pin this compile report")
    # The compile-report pin comes from the original CASE-REPORT, before this read.
    compiled = read_pinned(case["compile_report"])
    if compiled.get("status") != "PASS" or compiled.get("phase") != "complete":
        raise ValueError("native-only qualification requires an actually completed audited compile")
    manifest, source, options, refs = demo.load_manifest(case["manifest"]["path"])
    demo.bound(case["manifest"])
    if (
        manifest["id"] != case["id"]
        or compiled.get("id") != case["id"]
        or compiled.get("manifest") != manifest
        or compiled.get("selected_tools")
        != {"sdk_bin": str(tools.sdk_bin), "compiler_python": str(tools.compiler_python)}
        or compiled.get("options", {}).get("formal_transforms") != vars(options)
        or compiled["options"].get("textual") != manifest.get("textual", False)
        or compiled["options"].get("index_bits") != 64
        or compiled.get("checked_matches_ordinary_llvm") is not True
    ):
        raise ValueError("the retained compile differs from original source/options/current selected tools")
    compile_directory = Path(case["compile_report"]["path"]).parent
    expected_source = compile_directory / "original-model.mlir"
    if compiled["source"] != demo.file_ref(expected_source) or expected_source.read_bytes() != source.encode():
        raise ValueError("the actual compile source differs from its pinned original model")
    for name in ("baseline", "checked"):
        if compiled[name + "_llvm"] != demo.file_ref(compile_directory / name / "model.ll"):
            raise ValueError("retained native LLVM differs from the audited compiler product")
    if compiled["baseline_llvm"]["sha256"] != compiled["checked_llvm"]["sha256"]:
        raise ValueError("the audited ordinary/gated complete LLVM bytes differ")
    coverage = read_pinned(compiled["coverage_artifact"])
    if coverage != compiled.get("coverage"):
        raise ValueError("retained journal differs from the completed compiler journal")
    evidence = demo.audit_coverage(coverage, compiled["gate_verdicts"], compile_directory / "pass-validation", tools)
    demo.require_selectors(manifest, coverage)
    # Proof-specific sources, private products and original tools remain exact.
    # The changed native observer is not excused if it occurs in a proof input.
    proof_refs = demo.check_references(compiled)
    for reference in refs:
        demo.bound(reference)
    observation = compiled["host_index_observation"]
    compiler = demo.bound(observation["compiler"])
    if observation.get("status") != "PASS" or observation.get("index_bits") != 64:
        raise ValueError("the compile lacks its original explicit64bit host observation")
    return manifest, compiled, evidence, [*refs, *proof_refs, compiler]


def load_previous(directory, tools):
    directory = Path(directory).resolve(strict=True)
    result_pin = demo.file_ref(directory / "CAMPAIGN-RESULT.json")
    result = read_pinned(result_pin)
    if (
        result.get("schema") != "merlin.larger-model-campaign.results.v1"
        or result.get("status") not in {"PASS", "FAILED"}
        or result.get("final_source_cut_matches") is not True
    ):
        raise ValueError("the previous campaign must be terminal with its original source cut intact")
    cut_pin = demo.bound(result["source_cut"])
    original_cut = read_pinned(cut_pin)
    if original_cut.get("schema") != "merlin.larger-model-campaign.source-cut.v1":
        raise ValueError("the original versioned campaign source cut is required")
    old_native = [row for row in original_cut["files"] if Path(row["path"]) == NATIVE_HELPER.resolve()]
    if len(old_native) != 1:
        raise ValueError("the original native observer must have one exact historical source pin")
    for reference in original_cut["files"]:
        if reference != old_native[0]:
            demo.bound(reference)
    old_helpers = set(original_cut["helper_paths"])
    now_helpers = {str(path.resolve()) for path in campaign.helper_paths()}
    if old_helpers - now_helpers or now_helpers - old_helpers - {str(HERE)}:
        raise ValueError("only the additive reviewed native replay observer may extend the old helper roster")
    # Locate the one original roster by its bound schema, never by model names.
    rosters = []
    for reference in original_cut["files"]:
        if Path(reference["path"]).suffix == ".json" and reference["bytes"] <= 256 << 10:
            value = read_pinned(reference)
            if type(value) is dict and value.get("schema") == "merlin.larger-model-campaign.v1":
                rosters.append((reference, value))
    if len(rosters) != 1:
        raise ValueError("one exact original campaign manifest roster is required")
    cases, original_paths = campaign.load_roster(rosters[0][0]["path"])
    expected = [case["id"] for case in cases]
    rows = result.get("cases")
    if (
        len(expected) != EXPECTED_CASE_COUNT
        or type(rows) is not list
        or len(rows) != EXPECTED_CASE_COUNT
        or len({row.get("case") for row in rows}) != EXPECTED_CASE_COUNT
        or set(expected) != {row.get("case") for row in rows}
        or result.get("cases_requested") != expected
    ):
        raise ValueError("exactly the ten original case IDs and completed record roster are required")
    references = [result_pin, cut_pin, *original_cut["files"]]
    # Exclude only the explicitly superseded native observer, not proof inputs.
    references = [reference for reference in references if reference != old_native[0]]
    prepared = []
    for case in cases:
        row = next(value for value in rows if value["case"] == case["id"])
        cleanup = row.get("owned_process_cleanup", {})
        if cleanup.get("complete") is not True or cleanup.get("guardian_reaped") is not True:
            raise ValueError("an earlier case has unresolved process cleanup; no new native admission")
        expected_case_path = directory / case["id"] / "case/CASE-REPORT.json"
        if row["report"]["path"] != str(expected_case_path):
            raise ValueError("the original campaign points outside its retained case tree")
        previous_case = read_pinned(row["report"])
        compile_pin = previous_case.get("compile")
        if type(compile_pin) is not dict or compile_pin.get("path") != str(
            expected_case_path.parent / "compile/DEMO-REPORT.json"
        ):
            raise ValueError("an original case lacks its own complete pinned compile product")
        current = {**case, "prior_case_report": row["report"], "compile_report": compile_pin}
        _, _, evidence, proof_refs = recheck_compile(current, tools)
        current["prior_accepted_jobs"] = len(evidence)
        current["prior_case_status"] = previous_case.get("status")
        references.extend((row["report"], compile_pin, *proof_refs))
        prepared.append(current)
    predecessor = directory.parent / "native-v2-draft/larger_model_native_demo-v1.py"
    archived = None
    if predecessor.is_file():
        archived = demo.file_ref(predecessor)
        if any(archived[key] != old_native[0][key] for key in ("sha256", "bytes")):
            raise ValueError("archived original native observer differs from the frozen v1 source")
        references.append(archived)
    return (
        prepared,
        references,
        original_paths,
        {
            "original_campaign_result": result_pin,
            "original_campaign_status": result["status"],
            "native_observer_successor": {
                "historical_record_only": {"recorded_" + key: value for key, value in old_native[0].items()},
                "current": demo.file_ref(NATIVE_HELPER),
                "archived_historical_source": archived,
                "scope": "reviewed native observer/runtime-link correction; no changed proof inputs or products",
            },
            "fresh_proofs_executed": False,
            "fresh_model_compiles_executed": False,
        },
    )


def new_cut(references, original_paths, tools):
    from merlin.llvmlower import toolchain

    runtime_paths = [module_source_path(name) for name in RUNTIME_MODULES]
    selected_clang = toolchain.clang()
    runtime_paths.extend(
        (runtime_dir() / "abi/mlir_runtime.c", Path(shutil.which(str(selected_clang)) or str(selected_clang)), HERE)
    )
    selected_llc = toolchain.host_llc()
    if selected_llc is not None:
        runtime_paths.append(selected_llc)
    if selected_cc := os.environ.get("CC", "").strip():
        runtime_paths.append(Path(shutil.which(selected_cc) or selected_cc))
    by_path = {}
    for reference in references:
        previous = by_path.setdefault(reference["path"], reference)
        if previous != reference:
            raise ValueError("one retained path has inconsistent original byte bindings")
        demo.bound(reference)
    return campaign.source_cut([*original_paths, *map(Path, by_path), *runtime_paths], tools)


def require_runtime_link(observed, compiled):
    from merlin.common import invocation_record

    wanted = compiled["host_index_observation"]["compiler"]
    for side in ("baseline", "checked"):
        record = observed[side]
        linkage = record.get("runtime_link")
        if (
            type(linkage) is not dict
            or linkage.get("compiler") != wanted
            or linkage.get("builder") != "merlin.llvmlower.codegen.build_host_shared"
            or linkage.get("cleanup_scope") != "owned_host_shared_build_group"
            or not linkage.get("runtime_sources")
            or not linkage.get("commands")
            or not linkage.get("objects")
            or linkage.get("owned_process_cleanup", {}).get("complete") is not True
            or linkage.get("owned_process_cleanup", {}).get("guardian_reaped") is not True
        ):
            raise ValueError("native runtime link lacks its actual selected compiler/source/owned-case observations")
        documents = [read_pinned(pin) for pin in linkage["commands"]]
        for pin, document in zip(linkage["commands"], documents, strict=True):
            invocation_record.verify(Path(pin["path"]))
            selected = linkage.get("selected_executables", [wanted])
            if not any(
                all(document.get("executable", {}).get(key) == executable[key] for key in ("path", "sha256"))
                for executable in selected
            ) or (
                document.get("stage") in {"runtime_object", "link"}
                and any(document.get("executable", {}).get(key) != wanted[key] for key in ("path", "sha256"))
            ):
                raise ValueError("native compilation selected a compiler different from the original host observation")
        stages = {document.get("stage") for document in documents}
        if not {"object", "runtime_object", "link"} <= stages:
            raise ValueError("native library needs its actual model object, runtime object and link observations")
        model_objects = {
            value["path"]
            for document in documents
            if document.get("stage") == "object"
            and any(row["path"] == record["llvm"]["path"] for row in document.get("inputs", []))
            for value in document.get("outputs", [])
        }
        runtime_objects = {
            value["path"]
            for document in documents
            if document.get("stage") == "runtime_object"
            and any(
                row["path"] in {r["path"] for r in linkage["runtime_sources"]} for row in document.get("inputs", [])
            )
            for value in document.get("outputs", [])
        }
        links = [document for document in documents if document.get("stage") == "link"]
        if len(links) != 1 or "-lm" not in links[0].get("argv", []):
            raise ValueError("one actual runtime/libm link is required")
        if (
            not model_objects
            or not runtime_objects
            or not (model_objects | runtime_objects) <= {value["path"] for value in links[0].get("inputs", [])}
            or record["library"]["path"] not in {value["path"] for value in links[0].get("outputs", [])}
        ):
            raise ValueError("native library does not join the actual model/runtime objects")
        if {reference["path"] for reference in linkage["objects"]} != {value["path"] for value in links[0]["inputs"]}:
            raise ValueError("actual runtime-link object roster differs")
        build = read_pinned(linkage["build_process"])
        if (
            build.get("returncode") != 0
            or build.get("owned_process_cleanup") != linkage["owned_process_cleanup"]
            or build.get("builder") != linkage["builder"]
            or build.get("command", [None])[0] != compiled["selected_tools"]["compiler_python"]
            or type(build.get("stdout")) is not str
            or type(build.get("stderr")) is not str
        ):
            raise ValueError("actual owned runtime build did not complete with the original interpreter")
        demo.check_references(linkage)


def native_case(case, destination, *, cut):
    import larger_model_native_demo as native

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "merlin.larger-model-native-replay.case.v1",
        "id": case["id"],
        "status": "RUNNING",
        "phase": "prior_compile_reaudit",
        "native": [],
        "prior_case_report": case["prior_case_report"],
        "prior_compile_report": case["compile_report"],
        "fresh_proofs_executed": False,
        "fresh_model_compiles_executed": False,
        "whole_model_or_LLVM_proof": False,
        "model_accuracy_claim": False,
    }
    started = perf_counter()
    try:
        campaign.require_cut(cut)
        manifest, compiled, evidence, _ = recheck_compile(case, current_tools())
        report.update(prior_accepted_jobs=len(evidence), reaudited_prior_job_evidence=evidence)
        compile_directory = Path(case["compile_report"]["path"]).parent
        contract = manifest["native_contract"]
        from merlin.llvmlower import toolchain

        selected_clang = toolchain.clang()
        selected_clang = Path(shutil.which(str(selected_clang)) or str(selected_clang))
        if demo.file_ref(selected_clang) != compiled["host_index_observation"]["compiler"]:
            raise ValueError("native compiler changed since the audited original host observation")
        selected_cc = os.environ.get("CC", "").strip()
        if (
            selected_cc
            and demo.file_ref(Path(shutil.which(selected_cc) or selected_cc))
            != compiled["host_index_observation"]["compiler"]
        ):
            raise ValueError("runtime C compiler differs from the audited original host compiler")
        for seed in campaign.SEEDS:
            campaign.require_cut(cut)
            report["phase"] = "native_seed_" + str(seed)
            demo.write(destination / "CASE-REPORT.json", report)
            expected, reference = None, None
            if manifest["source_kind"] == "explicit_model_definition":
                inputs, expected, reference = campaign._mlp_inputs(
                    manifest, contract, seed, destination / "references" / str(seed)
                )
            else:
                inputs = native.generated_inputs(contract, seed=seed)
            campaign._arrays(contract, inputs)
            observed = native.compare(
                compile_directory, destination / "native" / str(seed), contract, inputs, expected=expected
            )
            campaign.native_join(compiled, observed, manifest)
            require_runtime_link(observed, compiled)
            if observed.get("independent_policy_reference") is not (expected is not None):
                raise ValueError("the complete independent reference scope differs")
            report["native"].append(
                {
                    "seed": seed,
                    "comparison": demo.file_ref(destination / "native" / str(seed) / "COMPARISON.json"),
                    "independent_reference": reference,
                    "outputs": observed["checked"]["outputs"],
                }
            )
        demo.check_references(compiled)
        report.update(status="PASS", phase="complete")
    except BaseException as error:
        report.update(status="FAILED_NO_NATIVE_CREDIT", reason=str(error), traceback=traceback.format_exc())
    finally:
        report["independent_references"] = [
            demo.file_ref(path) for path in sorted((destination / "references").glob("*/REFERENCE.json"))
        ]
        try:
            campaign.require_cut(cut)
            demo.check_references(report)
            report["post_source_cut_matches"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.update(status="FAILED_CUSTODY_NO_NATIVE_CREDIT", reason=str(error))
        report["elapsed_seconds"] = perf_counter() - started
        demo.write(destination / "CASE-REPORT.json", report)
    return report


def run_case(case, cpu, *, output, cut, cut_pin, timeout, environment):
    directory = output / case["id"]
    directory.mkdir()
    case_path = directory / "INPUT.json"
    demo.write(case_path, case)
    case_pin = demo.file_ref(case_path)
    argv = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        str(cpu),
        sys.executable,
        str(HERE),
        "--one-case",
        str(case_path),
        "--input-sha256",
        case_pin["sha256"],
        "--case-out",
        str(directory / "case"),
        "--source-cut",
        cut_pin["path"],
        "--source-cut-sha256",
        cut_pin["sha256"],
    ]
    row = {"case": case["id"], "cpu": cpu, "fresh_proofs_executed": False, "fresh_model_compiles_executed": False}
    demo.write(directory / "COMMAND.json", {"argv": argv, "timeout": timeout, "threads": 1})
    started, launched = perf_counter(), False
    try:
        campaign.require_cut(cut)
        demo.bound(cut_pin)
        demo.bound(case_pin)
        launched = True
        process = run_owned(argv, cwd=repo_root(), env=environment, timeout=timeout, capture_output=True, text=True)
        cleanup = getattr(process, "owned_process_cleanup", {})
        row.update(returncode=process.returncode, owned_process_cleanup=cleanup)
        (directory / "process.log").write_text((process.stdout or "") + (process.stderr or ""))
        if cleanup.get("complete") is not True or cleanup.get("guardian_reaped") is not True:
            row.update(status="FAILED_CLEANUP_UNKNOWN", unsafe_to_release_lane=True)
        else:
            report_pin = demo.file_ref(directory / "case/CASE-REPORT.json")
            report = read_pinned(report_pin)
            demo.check_references(report)
            row.update(
                report=report_pin, status="PASS" if process.returncode == 0 and report["status"] == "PASS" else "FAILED"
            )
    except subprocess.TimeoutExpired as error:
        cleanup = getattr(error, "owned_process_cleanup", {})
        row.update(status="FAILED_TIMEOUT", reason=str(error), owned_process_cleanup=cleanup)
        row["unsafe_to_release_lane"] = not (cleanup.get("complete") is True and cleanup.get("guardian_reaped") is True)
        (directory / "process.log").write_text(
            "".join(
                value.decode(errors="replace") if isinstance(value, bytes) else value
                for value in (error.output or "", error.stderr or "")
            )
        )
    except CleanupUnknown as error:
        row.update(status="FAILED_CLEANUP_UNKNOWN", reason=str(error), unsafe_to_release_lane=True)
    except Exception as error:
        row.update(status="FAILED_ORCHESTRATION", reason=str(error))
        if launched and "owned_process_cleanup" not in row:
            row["unsafe_to_release_lane"] = True
    finally:
        try:
            campaign.require_cut(cut)
            demo.bound(cut_pin)
            demo.bound(case_pin)
            row["post_source_cut_matches"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="FAILED_CUSTODY_NO_NATIVE_CREDIT", reason=str(error))
        row["elapsed_seconds"] = perf_counter() - started
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-campaign", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cpus", type=leases.parse_cpus, default=(28, 29, 30, 31))
    parser.add_argument("--case-timeout", type=float, default=1800)
    parser.add_argument("--one-case", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--case-out", type=Path)
    parser.add_argument("--source-cut", type=Path)
    parser.add_argument("--source-cut-sha256")
    args = parser.parse_args(argv)
    if args.one_case is not None:
        case_pin = demo.file_ref(args.one_case)
        cut_pin = demo.file_ref(args.source_cut)
        if case_pin["sha256"] != args.input_sha256 or cut_pin["sha256"] != args.source_cut_sha256:
            raise ValueError("the original native case/cut transport bytes changed")
        report = native_case(read_pinned(case_pin), args.case_out, cut=read_pinned(cut_pin))
        return 0 if report["status"] == "PASS" else 1
    if args.compiled_campaign is None or args.out is None:
        parser.error("--compiled-campaign and --out are required")
    if (
        not 1 <= args.jobs <= 4
        or not args.jobs <= len(args.cpus) <= 4
        or len(set(args.cpus)) != len(args.cpus)
        or not math.isfinite(args.case_timeout)
        or not 0 < args.case_timeout <= 3600
        or not set(args.cpus) <= os.sched_getaffinity(0)
        or os.getpriority(os.PRIO_PROCESS, 0) < 19
    ):
        raise ValueError("at most four available distinct nice19 CPU lanes and a bounded timeout are required")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        tools = current_tools()
        cases, references, original_paths, provenance = load_previous(args.compiled_campaign, tools)
        # No native case starts unless every one of the ten compiles re-audited.
        cut = new_cut(references, original_paths, tools)
        demo.write(
            output / "PRECHECK.json",
            {"status": "PASS", "cases": [case["id"] for case in cases], "fresh_proofs_executed": False},
        )
    except BaseException as error:
        demo.write(
            output / "PRECHECK.json",
            {
                "status": "FAILED_NO_NATIVE_ADMISSION",
                "reason": str(error),
                "traceback": traceback.format_exc(),
                "fresh_proofs_executed": False,
            },
        )
        return 1
    demo.write(output / "SOURCE-CUT.json", cut)
    cut_pin = demo.file_ref(output / "SOURCE-CUT.json")
    environment = dict(os.environ)
    environment.update(
        LEAN_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    by_id = {case["id"]: case for case in cases}

    def execute(case_id, cpu):
        return run_case(
            by_id[case_id],
            cpu,
            output=output,
            cut=cut,
            cut_pin=cut_pin,
            timeout=args.case_timeout,
            environment=environment,
        )

    def completed(row, records):
        (output / row["case"]).mkdir(exist_ok=True)
        demo.write(output / row["case"] / "PROCESS-RESULT.json", row)
        demo.write(
            output / "CAMPAIGN-RESULT.json",
            {
                "schema": "merlin.larger-model-native-replay.results.v1",
                "status": "RUNNING",
                "cases_requested": list(by_id),
                "cases": sorted(records, key=lambda item: list(by_id).index(item["case"])),
                "passed": sum(item["status"] == "PASS" for item in records),
                "failed": sum(item["status"] != "PASS" for item in records),
                "source_cut": cut_pin,
                "prior_compile_campaign": provenance,
                "fresh_proofs_executed": False,
                "fresh_model_compiles_executed": False,
                "jobs": args.jobs,
                "cpus": list(args.cpus),
                "whole_model_or_LLVM_proof": False,
                "model_accuracy_claim": False,
            },
        )
        print("END", row["case"], row["status"], "CPU", row["cpu"], flush=True)

    records = leases.dispatch(tuple(by_id), args.cpus, args.jobs, execute, completed)
    result, _ = demo.read_json(output / "CAMPAIGN-RESULT.json")
    try:
        campaign.require_cut(cut)
        demo.bound(cut_pin)
        complete = len(records) == EXPECTED_CASE_COUNT and {row["case"] for row in records} == set(by_id)
        passed = complete and all(row["status"] == "PASS" for row in records)
        result.update(
            status="PASS" if passed else "FAILED", native_campaign_credit=passed, final_source_cut_matches=True
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        passed = False
        result.update(status="FAILED_FINAL_CUSTODY_NO_NATIVE_CREDIT", reason=str(error), native_campaign_credit=False)
    demo.write(output / "CAMPAIGN-RESULT.json", result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
