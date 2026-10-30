"""Bounded complete-model gates and finite native observations on leased CPUs.

Every child owns one retained case tree. Successful compiler evidence is
audited by larger_model_gate_demo before any native call. Native observations
remain separate from pass proofs, numerical approval and model accuracy.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

import kernel_benchmark_handoff_campaign as leases
import larger_model_gate_demo as demo

from merlin.common.owned_process import CleanupUnknown
from merlin.common.owned_process import run as run_owned
from merlin.common.paths import repo_root

MAX_JOBS = 4
MAX_CASES = 64
MAX_CASE_TIMEOUT_SECONDS = 5400
SEEDS = (20261029, 20261030)
HERE = repo_root() / "merlin/tests/ir/larger_model_campaign.py"
BORROWED = (
    "kernel_benchmark_handoff_campaign.py",
    "test_kernel_benchmark_formal_gates.py",
    "test_compiler_formal_transforms.py",
    "kernel_benchmark_handoff_inputs.py",
    "kernel_benchmark_handoff_oracle.py",
)


def helper_paths():
    directory = repo_root() / "merlin/tests/ir"
    return sorted({*directory.glob("larger_model*.py"), *(directory / name for name in BORROWED)})


def load_roster(path):
    roster, pin = demo.read_json(path, maximum=256 << 10)
    if roster.get("schema") != "merlin.larger-model-campaign.v1":
        raise ValueError("an explicit complete-model campaign roster is required")
    entries = roster.get("cases")
    if type(entries) is not list or not 1 <= len(entries) <= MAX_CASES:
        raise ValueError("campaign requires one through64 explicitly bound manifests")
    cases, paths = [], [Path(pin["path"])]
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"manifest"}:
            raise ValueError("each campaign entry must bind exactly its original manifest")
        if type(entry["manifest"]) is not dict or set(entry["manifest"]) != {"path", "sha256", "bytes"}:
            raise ValueError("manifest references require exact path, byte length and digest")
        manifest_pin = demo.bound(entry["manifest"])
        manifest, _, _, references = demo.load_manifest(manifest_pin["path"])
        if manifest.get("native_contract") is None:
            raise ValueError("this campaign requires a complete descriptive native ABI")
        cases.append({"id": manifest["id"], "manifest": manifest_pin})
        paths.extend(Path(reference["path"]) for reference in references)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("distinct complete-model case IDs are required")
    demo.bound(pin)
    return cases, paths


def source_cut(paths, tools):
    helpers = helper_paths()
    paths = {*paths, *helpers, *(Path(ref["path"]) for ref in demo._source_cut(tools))}
    return {
        "schema": "merlin.larger-model-campaign.source-cut.v1",
        "helper_paths": [str(path.resolve()) for path in helpers],
        "files": [demo.file_ref(path) for path in sorted(paths)],
    }


def require_cut(cut):
    if cut.get("schema") != "merlin.larger-model-campaign.source-cut.v1":
        raise ValueError("the original campaign source cut is required")
    if [str(path.resolve()) for path in helper_paths()] != cut["helper_paths"]:
        raise ValueError("the complete larger-model helper roster changed")
    for reference in cut["files"]:
        demo.bound(reference)


def _arrays(contract, arrays, *, outputs=False):
    import numpy as np

    rows = contract["outputs" if outputs else "arguments"]
    names = [row["name" if outputs else "canonical_input_name"] for row in rows]
    if type(arrays) is not dict or list(arrays) != names:
        raise ValueError("finite model arrays must retain every ordered public buffer")
    for row, array in zip(rows, arrays.values(), strict=True):
        if (
            not isinstance(array, np.ndarray)
            or list(array.shape) != row["shape"]
            or array.dtype != np.dtype(row["dtype"])
            or not array.flags.c_contiguous
        ):
            raise ValueError("finite model array dtype/shape/storage differs before guarded copying")


def _mlp_inputs(manifest, contract, seed, destination):
    import larger_model_mlp as mlp
    import larger_model_native_demo as native
    import numpy as np

    native.require_array_budget(contract)
    fixture = mlp.fixture(**manifest["model_parameters"])
    original = Path(manifest["source"]["path"]).read_bytes()
    if fixture["source"].encode() != original:
        raise ValueError("the independent complete MLP definition differs from the original source")
    for key in ("arguments", "outputs"):
        if len(fixture[key]) != len(contract[key]) or any(
            source["shape"] != target["shape"] or source["dtype"] != target["dtype"]
            for source, target in zip(fixture[key], contract[key], strict=True)
        ):
            raise ValueError("the original MLP fixture and native ABI differ")
    inputs, expected, stages = mlp.inputs_and_reference(fixture, seed=seed)
    inputs = {
        target["canonical_input_name"]: inputs[source["canonical_input_name"]]
        for source, target in zip(fixture["arguments"], contract["arguments"], strict=True)
    }
    expected = {
        target["name"]: expected[source["name"]]
        for source, target in zip(fixture["outputs"], contract["outputs"], strict=True)
    }
    _arrays(contract, inputs)
    _arrays(contract, expected, outputs=True)
    destination.mkdir(parents=True, exist_ok=False)
    references = []
    for name, array in expected.items():
        path = destination / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        references.append({**native.array_record(name, array), "retained_array": demo.file_ref(path)})
    reference = {
        "seed": seed,
        "scope": "independent ordered draft-policy evaluation of the complete explicitly defined MLP",
        "model": fixture["model"],
        "source": demo.bound(manifest["source"]),
        "stages": stages,
        "inputs": [native.array_record(name, array) for name, array in inputs.items()],
        "outputs": references,
        "pretrained_or_model_accuracy_claim": False,
        "original_half_accumulator_equivalence": False,
    }
    demo.write(destination / "REFERENCE.json", reference)
    if len(stages) != fixture["model"]["layers"] or any(
        stage.get("layer") != index
        or type(stage.get("distinct_batch_rows")) is not int
        or stage["distinct_batch_rows"] <= 1
        for index, stage in enumerate(stages)
    ):
        raise ValueError(
            "complete MLP reference must retain distinct batch rows at every layer; REFERENCE.json retained"
        )
    return inputs, expected, demo.file_ref(destination / "REFERENCE.json")


def native_join(compiled, observed, manifest):
    if observed.get("status") != "PASS" or observed.get("all_public_outputs_byte_equal") is not True:
        raise ValueError("the complete finite native comparison did not pass")
    if observed.get("actual_original_source_abi") != compiled["source"]:
        raise ValueError("native source ABI differs from the actual audited compile source")
    if any(compiled["source"][key] != manifest["source"][key] for key in ("sha256", "bytes")):
        raise ValueError("the audited compile source differs from the original manifest")
    for key in ("baseline", "checked"):
        if observed[key]["llvm"] != compiled[key + "_llvm"]:
            raise ValueError("native LLVM differs from the actual audited compiler product")
        demo.bound(observed[key]["llvm"])
    demo.check_references(observed)


def one_case(manifest_path, destination, *, cut, job_timeout=900):
    import larger_model_native_demo as native

    from merlin.verify.pass_validation import CheckerTools

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "merlin.larger-model-campaign.case.v1",
        "status": "RUNNING",
        "phase": "original_source_cut",
        "native": [],
        "whole_model_formal_proof": False,
        "model_accuracy_claim": False,
        "elapsed_scope": "full_case_walltime_not_kernel_performance",
    }
    started = perf_counter()
    try:
        require_cut(cut)
        manifest, _, options, references = demo.load_manifest(manifest_path)
        original_pins = {reference["path"]: reference for reference in cut["files"]}
        if any(original_pins.get(reference["path"]) != reference for reference in references):
            raise ValueError("case manifest/source/provenance differs from the original campaign cut")
        report.update(id=manifest["id"], original_inputs=references)
        contract = manifest["native_contract"]
        if contract is None:
            raise ValueError("finite native observations require the original complete descriptive ABI")
        tools = CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), Path(os.environ["MERLIN_COMPILER_PYTHON"]))
        report["phase"] = "actual_compiler_and_complete_receipt_audit"
        demo.write(destination / "CASE-REPORT.json", report)
        compiled = demo.execute(manifest_path, destination / "compile", tools, job_timeout=job_timeout)
        report["compile"] = demo.file_ref(destination / "compile/DEMO-REPORT.json")
        if compiled.get("status") != "PASS" or compiled.get("phase") != "complete":
            raise ValueError("the actual compile/receipt audit did not pass; native execution was not admitted")
        if compiled["manifest"] != manifest or compiled["options"]["formal_transforms"] != vars(options):
            raise ValueError("audited compile differs from original model/options")
        demo.check_references(compiled)
        report.update(
            coverage=compiled["coverage"],
            verified_boundaries=compiled["verified_boundaries"],
            unverified_boundaries=compiled["unverified_boundaries"],
        )
        for seed in SEEDS:
            require_cut(cut)
            report["phase"] = "finite_native_seed_" + str(seed)
            demo.write(destination / "CASE-REPORT.json", report)
            expected, reference = None, None
            if manifest["source_kind"] == "explicit_model_definition":
                inputs, expected, reference = _mlp_inputs(
                    manifest, contract, seed, destination / "references" / str(seed)
                )
            else:
                inputs = native.generated_inputs(contract, seed=seed)
            _arrays(contract, inputs)
            observed = native.compare(
                destination / "compile", destination / "native" / str(seed), contract, inputs, expected=expected
            )
            native_join(compiled, observed, manifest)
            if observed.get("independent_policy_reference") is not (expected is not None):
                raise ValueError("native reference scope differs from the supplied complete reference")
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
        report.update(
            status="FAILED_NO_CASE_CREDIT",
            failure_type=type(error).__name__,
            reason=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        try:
            report["independent_references"] = [
                demo.file_ref(path) for path in sorted((destination / "references").glob("*/REFERENCE.json"))
            ]
            require_cut(cut)
            demo.check_references(report)
            report["post_source_cut_matches"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.update(status="FAILED_SOURCE_CUT_OR_EVIDENCE_NO_CASE_CREDIT", reason=str(error))
        report["elapsed_seconds"] = perf_counter() - started
        demo.write(destination / "CASE-REPORT.json", report)
    return report


def run_case(case, cpu, *, output, cut, cut_ref, environment, timeout, owned_run=run_owned):
    directory = output / case["id"]
    directory.mkdir()
    report = {"case": case["id"], "cpu": cpu, "case_directory": str(directory)}
    command = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        str(cpu),
        sys.executable,
        str(HERE),
        "--one-case",
        case["manifest"]["path"],
        "--case-out",
        str(directory / "case"),
        "--source-cut",
        cut_ref["path"],
        "--source-cut-sha256",
        cut_ref["sha256"],
        "--job-timeout",
        str(min(900, timeout)),
    ]
    demo.write(directory / "COMMAND.json", {"command": command, "timeout": timeout, "cpu": cpu, "threads": 1})
    started = perf_counter()
    launched = False
    try:
        require_cut(cut)
        demo.bound(case["manifest"])
        launched = True
        result = owned_run(command, cwd=repo_root(), env=environment, timeout=timeout, capture_output=True, text=True)
        report["returncode"] = result.returncode
        cleanup = getattr(result, "owned_process_cleanup", {})
        report["owned_process_cleanup"] = cleanup
        (directory / "process.log").write_text((result.stdout or "") + (result.stderr or ""))
        if cleanup.get("complete") is not True or cleanup.get("guardian_reaped") is not True:
            report.update(status="FAILED_CLEANUP_UNKNOWN_NO_CASE_CREDIT", unsafe_to_release_lane=True)
        else:
            case_report, case_pin = demo.read_json(directory / "case/CASE-REPORT.json")
            report.update(report=case_pin, case_status=case_report["status"], phase=case_report["phase"])
            demo.check_references(case_report)
            report["status"] = "PASS" if result.returncode == 0 and case_report["status"] == "PASS" else "FAILED"
    except subprocess.TimeoutExpired as error:
        cleanup = getattr(error, "owned_process_cleanup", {})
        report.update(status="FAILED_TIMEOUT_NO_CASE_CREDIT", reason=str(error), owned_process_cleanup=cleanup)
        if cleanup.get("complete") is not True or cleanup.get("guardian_reaped") is not True:
            report["unsafe_to_release_lane"] = True
        chunks = [error.output or "", error.stderr or ""]
        (directory / "process.log").write_text(
            "".join(value.decode(errors="replace") if isinstance(value, bytes) else value for value in chunks)
        )
    except CleanupUnknown as error:
        report.update(status="FAILED_CLEANUP_UNKNOWN_NO_CASE_CREDIT", reason=str(error), unsafe_to_release_lane=True)
    except Exception as error:
        report.update(status="FAILED_ORCHESTRATION_NO_CASE_CREDIT", reason=str(error))
        if launched and "owned_process_cleanup" not in report:
            report["unsafe_to_release_lane"] = True
    finally:
        try:
            require_cut(cut)
            demo.bound(cut_ref)
            report["post_source_cut_matches"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.update(status="FAILED_SOURCE_CUT_DRIFT_NO_CASE_CREDIT", reason=str(error))
        report["elapsed_seconds"] = perf_counter() - started
    return report


def campaign(roster_path, destination, *, jobs, cpus, timeout):
    from merlin.verify.pass_validation import CheckerTools

    if not 1 <= jobs <= MAX_JOBS or not jobs <= len(cpus) <= MAX_JOBS or len(set(cpus)) != len(cpus):
        raise ValueError("jobs1..4 must fit at most four distinct selected CPU lanes")
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_CASE_TIMEOUT_SECONDS:
        raise ValueError(f"a finite positive case timeout through{MAX_CASE_TIMEOUT_SECONDS}seconds is required")
    if not set(cpus) <= os.sched_getaffinity(0) or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise ValueError("selected CPUs must be available within a nice19 parent affinity")
    cases, paths = load_roster(roster_path)
    tools = CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), Path(os.environ["MERLIN_COMPILER_PYTHON"]))
    cut = source_cut(paths, tools)
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cut_path = output / "SOURCE-CUT.json"
    demo.write(cut_path, cut)
    cut_ref = demo.file_ref(cut_path)
    environment = dict(os.environ)
    environment.update(
        MERLIN_LEAN_REQUIRED="1",
        MERLIN_VALIDATE_WORKERS="1",
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
            cut_ref=cut_ref,
            environment=environment,
            timeout=timeout,
        )

    def completed(row, records):
        directory = output / row["case"]
        directory.mkdir(exist_ok=True)
        demo.write(directory / "PROCESS-RESULT.json", row)
        aggregate = {
            "schema": "merlin.larger-model-campaign.results.v1",
            "status": "RUNNING",
            "source_cut": cut_ref,
            "jobs": jobs,
            "cpus": list(cpus),
            "seeds": list(SEEDS),
            "passed": sum(value["status"] == "PASS" for value in records),
            "failed": sum(value["status"] != "PASS" for value in records),
            "cases_requested": list(by_id),
            "cases": sorted(records, key=lambda value: list(by_id).index(value["case"])),
            "whole_model_or_LLVM_proof": False,
            "model_accuracy_claim": False,
        }
        demo.write(output / "CAMPAIGN-RESULT.json", aggregate)
        print("END", row["case"], row["status"], "CPU", row["cpu"], flush=True)

    records = leases.dispatch(tuple(by_id), cpus, jobs, execute, completed)
    result, _ = demo.read_json(output / "CAMPAIGN-RESULT.json")
    try:
        require_cut(cut)
        demo.bound(cut_ref)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(
            status="FAILED_FINAL_SOURCE_CUT_NO_CAMPAIGN_CREDIT",
            final_custody_error=str(error),
            campaign_credit=False,
            final_source_cut_matches=False,
        )
        demo.write(output / "CAMPAIGN-RESULT.json", result)
        return 1
    passed = all(row["status"] == "PASS" for row in records)
    result.update(status="PASS" if passed else "FAILED", campaign_credit=passed, final_source_cut_matches=True)
    demo.write(output / "CAMPAIGN-RESULT.json", result)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-list", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cpus", type=leases.parse_cpus, default=(28, 29, 30, 31))
    parser.add_argument("--case-timeout", type=float, default=1800)
    parser.add_argument("--one-case", type=Path)
    parser.add_argument("--case-out", type=Path)
    parser.add_argument("--source-cut", type=Path)
    parser.add_argument("--source-cut-sha256")
    parser.add_argument("--job-timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if args.one_case is not None:
        if args.case_out is None or args.source_cut is None or args.source_cut_sha256 is None:
            parser.error("owned child requires --case-out and the original source-cut pin")
        cut, pin = demo.read_json(args.source_cut)
        if pin["sha256"] != args.source_cut_sha256:
            raise ValueError("the original campaign source-cut bytes changed")
        return (
            0
            if one_case(args.one_case, args.case_out, cut=cut, job_timeout=args.job_timeout)["status"] == "PASS"
            else 1
        )
    if args.manifest_list is None or args.out is None:
        parser.error("campaign requires --manifest-list and --out")
    return campaign(args.manifest_list, args.out, jobs=args.jobs, cpus=args.cpus, timeout=args.case_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
