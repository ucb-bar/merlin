"""Opt-in complete-model compiler demonstrations, without invented native inputs.

This test-only tool records actual bounded pass proofs and ordinary LLVM parity.
Its evidence audit checks current receipts from the live compiler invocation;
it never authorizes compilation or delivery using a saved acceptance status.
An outer owned process should bound the complete invocation and its descendants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import traceback
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

SCHEMA = "merlin.larger-model-gate-demo.v1"
HERE = Path(__file__).resolve()
STD3 = {"propext", "Classical.choice", "Quot.sound"}
KINDS = {"original_capture", "existing_model_capsule", "explicit_model_definition", "derived_compiler_ir"}


def file_ref(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def bound(reference):
    if (
        type(reference) is not dict
        or type(reference.get("path")) is not str
        or not Path(reference["path"]).is_absolute()
        or not re.fullmatch("[0-9a-f]{64}", str(reference.get("sha256", "")))
        or ("bytes" in reference and (type(reference["bytes"]) is not int or reference["bytes"] < 0))
    ):
        raise ValueError("an exact absolute file reference is required")
    actual = file_ref(reference["path"])
    if any(actual[key] != reference[key] for key in ("sha256", "bytes") if key in reference):
        raise ValueError("bound file bytes changed: " + reference["path"])
    return actual


def read_json(path, *, maximum=64 << 20):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > maximum:
        raise ValueError("JSON exceeds the bounded demonstration intake")
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError("JSON changed beyond the bounded demonstration intake")
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def check_references(value):
    """Check explicit file refs; do not interpret plain receipt path strings as proof."""
    refs = []
    if type(value) is dict:
        if "path" in value and "sha256" in value:
            refs.append(bound(value))
        for child in value.values():
            refs.extend(check_references(child))
    elif type(value) in (list, tuple):
        for child in value:
            refs.extend(check_references(child))
    return refs


def load_manifest(path):
    from merlin.verify import pass_validation
    from merlin.verify.compiler_transforms import Options

    manifest, pin = read_json(path, maximum=256 << 10)
    if manifest.get("schema") != SCHEMA or not re.fullmatch("[A-Za-z0-9_-]{1,96}", str(manifest.get("id", ""))):
        raise ValueError("a named complete-model demonstration manifest is required")
    if manifest.get("source_kind") not in KINDS or type(manifest.get("textual", False)) is not bool:
        raise ValueError("explicit original/model-definition/derived source provenance is required")
    native = manifest.get("native_contract")
    if native is not None:
        if (
            type(native) is not dict
            or set(native) != {"entry", "arguments", "outputs", "scope"}
            or native.get("entry") != "forward"
            or native.get("scope") != "finite deterministic baseline-versus-gated execution, no modelaccuracyclaim"
        ):
            raise ValueError("compile-only mode accepts only descriptive native ABI metadata, never invented inputs")
        for key in ("arguments", "outputs"):
            rows = native.get(key)
            if type(rows) is not list or not rows:
                raise ValueError("complete ordered native ABI metadata is required")
            for index, row in enumerate(rows):
                keys = {"name", "shape", "dtype"} | ({"canonical_input_name"} if key == "arguments" else set())
                name = ("arg" if key == "arguments" else "result") + str(index)
                if (
                    type(row) is not dict
                    or set(row) != keys
                    or row.get("name") != name
                    or (key == "arguments" and row["canonical_input_name"] != name)
                    or row.get("dtype") not in {"float16", "bfloat16", "float32", "int8", "int32", "int64", "bool"}
                    or type(row.get("shape")) is not list
                    or any(type(dimension) is not int or dimension < 0 for dimension in row["shape"])
                ):
                    raise ValueError("native ABI metadata must retain ordered actual names, static shapes and dtypes")
    if type(manifest.get("provenance")) is not list or not manifest["provenance"]:
        raise ValueError("at least one original provenance file is required")
    refs = [pin, bound(manifest["source"]), *[bound(ref) for ref in manifest["provenance"]]]
    if manifest["source_kind"] == "derived_compiler_ir":
        steps = manifest.get("derivation")
        if not isinstance(steps, list) or not steps:
            raise ValueError("a derived source must retain every explicit input/operation/output step")
        for step in steps:
            if type(step.get("operation")) is not str or not step["operation"]:
                raise ValueError("derived compiler steps need their actual operation names")
            refs.extend((bound(step["input"]), bound(step["output"])))
        if (
            any(a["output"] != b["input"] for a, b in zip(steps, steps[1:]))
            or steps[-1]["output"] != manifest["source"]
        ):
            raise ValueError("derived source steps must join the exact final source")
    if type(manifest.get("options", {})) is not dict:
        raise ValueError("typed compiler transformation choices are required")
    options = Options(**manifest.get("options", {}))
    for key in ("required_changed_selectors", "required_verified_selectors"):
        selected = manifest.get(key)
        if type(selected) is not list or not selected or len(set(selected)) != len(selected):
            raise ValueError("explicit unique nonempty required selectors are needed")
        if not set(selected) <= set(pass_validation.CHECKS):
            raise ValueError("unknown required compiler selector")
    if not set(manifest["required_verified_selectors"]) <= set(manifest["required_changed_selectors"]):
        raise ValueError("required proof coverage must describe an actual required changed transformation")
    source_pin = refs[1]
    if source_pin["bytes"] > 32 << 20:
        raise ValueError("complete demo source exceeds its explicit32MiB resource policy")
    raw = Path(source_pin["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_pin["sha256"] or len(raw) != source_pin["bytes"]:
        raise ValueError("original model source changed while reading")
    return manifest, raw.decode("utf-8"), options, refs


def _same_bytes(left, right):
    return all(left[key] == right[key] for key in ("sha256", "bytes"))


def _functions(export):
    names = []
    for operation in export["module"]["ops"]:
        if operation["name"] != "func.func" or not any(region["blocks"] for region in operation["regions"]):
            continue
        attrs = {**operation["attrs"], **operation["props"]}
        names.append(attrs["sym_name"]["value"])
    if not names or len(names) != len(set(names)):
        raise ValueError("authoritative export has no unique complete defined-function roster")
    return names


def _receipt(path, expected, *, theorem, verdict="verified"):
    from merlin.verify.lean import check

    value, pin = read_json(path)
    if value.get("schema") != "merlin.lean.receipt.v1" or value.get("verdict") != verdict:
        raise ValueError("missing required completed receipt")
    if (
        value.get("theorem") != theorem
        or value.get("lean") != expected["lean"]
        or value.get("exporter") != expected["exporter"]
    ):
        raise ValueError("receipt obligation/current checker/exporter differs")
    if check._stale(value, "larger-model current evidence") or not set(value.get("axioms", ())) <= STD3:
        raise ValueError("receipt has stale inputs/tools/lock or nonstandard axioms")
    check_references(value)
    for name, export_ref in value.get("exports", {}).items():
        exported, _ = read_json(bound(export_ref)["path"])
        if not _same_bytes(exported["source"], value[name]):
            raise ValueError("receipt authoritative export does not bind its complete source")
    return value, pin


def _tensor_detail(detail, endpoints, expected):
    from merlin.verify import gate_checks
    from merlin.verify.lean import check

    functions = detail.get("functions")
    if (
        type(functions) is not list
        or not functions
        or any(type(name) is not str or not name for name in functions)
        or len(functions) != len(set(functions))
        or list(detail.get("theorems", ())) != list(gate_checks.THEOREMS)
        or type(detail.get("chains")) is not dict
        or set(detail["chains"]) != set(functions)
        or not set(detail.get("axioms", ())) <= STD3
    ):
        raise ValueError("tensor handler lacks complete functions/theorems/strict refinement chains")
    typed, typed_pin = _receipt(detail["receipt"], expected, theorem=gate_checks.THEOREMS[0])
    if typed.get("index_bits") != 64 or typed.get("functions") != functions:
        raise ValueError("typed receipt does not cover the declared full function roster/index64")
    for name, endpoint in zip(("before", "after"), endpoints, strict=True):
        if not _same_bytes(typed[name], endpoint):
            raise ValueError("typed receipt does not bind the actual complete compiler boundary")
        exported, _ = read_json(typed["exports"][name]["path"])
        if set(_functions(exported)) != set(functions):
            raise ValueError("full authoritative function roster differs from handler coverage")
    artifacts = [typed_pin]
    chains = []
    for function in functions:
        support_path = Path(detail["receipt"]).parent.parent / ("support-" + function) / "receipt.json"
        support, support_pin = _receipt(support_path, expected, theorem=gate_checks.THEOREMS[1])
        if (
            support.get("function") != function
            or support.get("index_bits") != 64
            or not _same_bytes(support["subject"], endpoints[0])
        ):
            raise ValueError("required support receipt has a different function/source/domain")
        chain, chain_pin = read_json(detail["chains"][function])
        links = chain.get("links")
        if (
            chain.get("schema") != "merlin.lean.chain.v1"
            or chain.get("function") != function
            or chain.get("index_bits") != 64
            or chain.get("refinement_theorem") != check.TYPED_CHAIN_THEOREM
            or chain.get("refinement_verdict") != "verified"
            or chain.get("chained") is not True
            or chain.get("replayed") is not True
            or type(links) is not list
            or len(links) != 1
            or links[0].get("receipt_sha256") != typed_pin["sha256"]
            or links[0].get("before_sha256") != endpoints[0]["sha256"]
            or links[0].get("after_sha256") != endpoints[1]["sha256"]
            or links[0].get("verdict") != "verified"
            or chain.get("support", {}).get("receipt_sha256") != support_pin["sha256"]
            or chain["support"].get("subject_sha256") != endpoints[0]["sha256"]
            or chain["support"].get("verdict") != "verified"
        ):
            raise ValueError("required per-function strict refinement did not compose the actual boundary")
        artifacts.extend((support_pin, chain_pin))
        chains.append(
            {
                "function": function,
                "source_definedness": chain.get("source_definedness"),
                "overall_verdict": chain.get("verdict"),
            }
        )
    return {
        "functions": functions,
        "artifacts": artifacts,
        "chains": chains,
        "strengthenings": detail.get("strengthenings"),
    }


def _macc_detail(detail, endpoints, expected):
    from merlin.verify import bf16_policy_gate as policy
    from merlin.verify import compiler_macc

    receipt, pin = _receipt(detail["receipt"], expected, theorem=policy.policy.THEOREM, verdict="abstracted")
    for name, endpoint in zip(("before", "after"), endpoints, strict=True):
        if not _same_bytes(receipt[name], endpoint) or not _same_bytes(detail[name], endpoint):
            raise ValueError("draft MACC complete actual boundary differs from its receipt")
    report, report_pin = read_json(receipt["cert"]["path"])
    source, _ = read_json(receipt["exports"]["before"]["path"])
    target, _ = read_json(receipt["exports"]["after"]["path"])
    functions = _functions(source)
    if set(_functions(target)) != set(functions):
        raise ValueError("draft MACC complete source/target function rosters differ")
    sites = policy.validate_report(report, source)
    local = report["pass_boundary"]
    if (
        not sites
        or sites != detail["sites"]
        or local.get("positive_boundary_checked") is not True
        or local.get("positive_boundary_theorem") != compiler_macc.THEOREM
    ):
        raise ValueError("draft MACC positive complete-entry/source-site obligation is missing")
    check_references(detail)
    return {
        "artifacts": [pin, report_pin],
        "sites": sites,
        "functions": functions,
        "policy_reviewed": False,
        "original_source_equivalence": False,
        "scope": "conformance to the source-derived draft accumulation reference",
    }


def audit_coverage(coverage, verdicts, gate_directory, tools):
    """Independently qualify this invocation's records, never issue live authority."""
    from merlin.verify import gate_worker, pass_validation
    from merlin.verify.lean import check

    if (
        coverage.get("finalized") is not True
        or coverage.get("compiler_completed") is not True
        or coverage.get("requested_passes") != sorted(pass_validation.CHECKS)
    ):
        raise ValueError("the compiler did not finish with all ten requested selectors")
    rows = coverage["boundaries"]
    for row in rows:
        status, credit = row.get("status"), row.get("proof_credit")
        if status not in {"verified", "unverified", "unchanged"} or type(credit) is not bool:
            raise ValueError("a completed compiler journal contains an unfinished/unknown boundary")
        if credit != (status == "verified") or (
            status == "unchanged" and row.get("source_sha256") != row.get("final_sha256")
        ):
            raise ValueError("journal status, raw identity and proof credit disagree")
        if status == "verified" and row.get("source_sha256") == row.get("final_sha256"):
            raise ValueError("identity boundaries cannot supply positive transformation proof credit")
    for key, status in (
        ("verified_boundaries", "verified"),
        ("unverified_boundaries", "unverified"),
        ("unchanged_boundaries", "unchanged"),
    ):
        if type(coverage.get(key)) is not int or coverage[key] != sum(row["status"] == status for row in rows):
            raise ValueError("journal aggregate counts differ from its actual boundary rows")
    credited = [row for row in rows if row.get("proof_credit") is True]
    accepted = [verdict for verdict in verdicts if verdict.get("verdict") == "accepted"]
    if len(accepted) != len(verdicts) or len({value["job"] for value in accepted}) != len(accepted):
        raise ValueError("a compiler proof job failed or has duplicate verdict records")
    if (
        len(credited) != len(accepted)
        or len({row.get("job") for row in credited}) != len(credited)
        or {row.get("job") for row in credited} != {value["job"] for value in accepted}
    ):
        raise ValueError("accepted proof jobs and credited complete boundaries differ")
    expected = None
    evidence = []
    for row in credited:
        matches = [value for value in accepted if value["job"] == row["job"]]
        verdict = matches[0]
        directory = Path(gate_directory) / row["job"]
        request, request_pin = read_json(directory / "request.json")
        stored, stored_pin = read_json(directory / "verdict.json")
        endpoints = [file_ref(request[name]) for name in ("before", "after")]
        if (
            request.get("params", {}).get("sdk_bin") != str(tools.sdk_bin)
            or request["params"].get("compiler_python") != str(tools.compiler_python)
            or request["params"].get("index_bits") != 64
        ):
            raise ValueError("accepted proof request used different selected tools or index width")
        if (
            row.get("status") != "verified"
            or request.get("directory") != str(directory)
            or row.get("check") != verdict.get("check")
            or row.get("check") != request.get("check")
            or row.get("label") != verdict.get("label")
            or row.get("label") != request.get("label")
            or row.get("source_sha256") != endpoints[0]["sha256"]
            or row.get("final_sha256") != endpoints[1]["sha256"]
            or row.get("input_refs") != [request_pin, *endpoints]
            or stored.get("verdict") != "accepted"
            or stored.get("detail") != verdict.get("detail")
        ):
            raise ValueError("coverage/request/verdict do not bind one exact actual complete boundary")
        detail = verdict["detail"]
        gate_worker._require_builtin_result(request["check"], detail)
        if expected is None:
            expected = {
                "lean": check._lean_block(),
                "exporter": {**check._exporter_block(tools.compiler_python), "round_trip": "identical"},
            }
        if request["check"] in ("tensor_refinement", "compiler_qdq_refinement"):
            qualified = _tensor_detail(detail, endpoints, expected)
            if request["check"] == "compiler_qdq_refinement":
                if detail.get("source_spec_approved") is not False or detail.get("native_fidelity") is not False:
                    raise ValueError("Q/DQ proposed model cannot grant numerical/source-spec approval")
                if detail.get("bindings", {}).get("endpoints") != endpoints:
                    raise ValueError("Q/DQ complete original endpoints are missing")
                qualified["scope"] = "strict refinement under proposed Q/DQ source semantics"
                check_references(detail)
        elif request["check"] == "compiler_macc_contract":
            qualified = _macc_detail(detail, endpoints, expected)
        else:
            raise ValueError("this demo's independent evidence audit has no qualified route for " + request["check"])
        evidence.append(
            {
                "pass": row["pass"],
                "job": row["job"],
                "request": request_pin,
                "verdict": stored_pin,
                "endpoints": endpoints,
                "evidence": qualified,
            }
        )
    return evidence


def require_selectors(manifest, coverage):
    rows = coverage["boundaries"]
    for name in manifest["required_changed_selectors"]:
        observed = [row for row in rows if name in row["pass"].split("+")]
        if not any(
            row.get("final_sha256") is not None and row["source_sha256"] != row["final_sha256"] for row in observed
        ):
            raise ValueError("required transformation did not actually change complete model IR: " + name)
    for name in manifest["required_verified_selectors"]:
        if not any(
            name in row["pass"].split("+") and row.get("status") == "verified" and row.get("proof_credit") is True
            for row in rows
        ):
            raise ValueError("required bounded proof was not established for " + name)


@contextmanager
def _ordinary_environment():
    from merlin.verify import pass_validation

    old = os.environ.pop(pass_validation.ENV_PASSES, None)
    try:
        yield
    finally:
        if old is not None:
            os.environ[pass_validation.ENV_PASSES] = old


def _source_cut(tools):
    """Explicit compiler/observer/tool custody, without a whole-environment claim."""
    from merlin.common.paths import module_source_path
    from merlin.verify.lean import check

    modules = (
        "merlin.llvmlower.lower",
        "merlin.llvmlower.pipeline",
        "merlin.llvmlower.passes_xdsl",
        "merlin.verify.compiler_transforms",
        "merlin.verify.compiler_coverage",
        "merlin.verify.pass_validation",
        "merlin.verify.speculative_gate",
        "merlin.verify.gate_worker",
        "merlin.verify.gate_checks",
        "merlin.verify.compiler_ir_source_profile",
        "merlin.verify.compiler_qdq",
        "merlin.verify.compiler_macc",
        "merlin.verify.bf16_policy_gate",
        "merlin.verify.lean.check",
        "merlin.verify.lean.mlir_export",
        "merlin.verify.memory_scalar_nan_admission",
        "merlin.frontends.linalg_mlir",
        "merlin.common.owned_process",
    )
    paths = [module_source_path(name) for name in modules]
    paths.extend(
        (
            Path(tools.sdk_bin) / "lean",
            Path(tools.sdk_bin) / "lake",
            Path(tools.compiler_python),
            check.lean_exe(),
            check.lean_root() / "SPEC_LOCK",
        )
    )
    environment_config = Path(tools.compiler_python).parent.parent / "pyvenv.cfg"
    if environment_config.is_file():
        paths.append(environment_config)
    return [file_ref(path) for path in paths]


def execute(manifest_path, destination, tools, *, job_timeout=900):
    import math

    import test_kernel_benchmark_formal_gates as kernel_tools

    from merlin.llvmlower.lower import lower_model
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    if type(job_timeout) not in (int, float) or not math.isfinite(job_timeout) or not 0 < job_timeout <= 1800:
        raise ValueError("one bounded proof worker requires a finite timeout through1800seconds")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / "DEMO-REPORT.json"
    report = {
        "schema": "merlin.larger-model-gate-demo.report.v1",
        "status": "RUNNING",
        "phase": "original_manifest",
        "mode": "compile-only",
        "whole_model_formally_verified": False,
        "LLVM_proof": False,
        "native_evidence": "not_requested; no original input/reference contract supplied",
        "proof_credit_scope": "individual actual bounded transformation claims, including their explicit premises",
    }
    refs = [file_ref(HERE), file_ref(kernel_tools.__file__), *_source_cut(tools)]
    worker = None
    started = perf_counter()
    try:
        manifest, source, options, original_refs = load_manifest(manifest_path)
        refs.extend(original_refs)
        snapshot = destination / "original-model.mlir"
        snapshot.write_bytes(source.encode("utf-8"))
        report.update(id=manifest["id"], manifest=manifest, source=file_ref(snapshot), inputs=refs)
        report["selected_tools"] = {"sdk_bin": str(tools.sdk_bin), "compiler_python": str(tools.compiler_python)}
        report["phase"] = "native_host_index_selection"
        write(report_path, report)
        observation = kernel_tools._host_index_observation(destination / "host-index-observation")
        report["host_index_observation"] = observation
        worker = SpeculativeGate(
            destination / "pass-validation", interrupt=False, job_timeout=job_timeout, workers=1, nice=19
        )
        selected = pass_validation.PassValidation(worker, tools, pass_validation.DEFAULT_CHECKS, index_bits=64)
        arguments = {
            "targets": (),
            "textual": manifest.get("textual", False),
            "data_layout": observation["data_layout"],
            "index_bits": observation["index_bits"],
            "formal_transforms": options,
        }
        report["options"] = {**arguments, "formal_transforms": vars(options)}
        with _ordinary_environment():
            report["phase"] = "default_all_checked_compile"
            write(report_path, report)
            checked = lower_model(source, destination / "checked", pass_validation=selected, **arguments)
            report["checked_llvm"] = file_ref(checked.ll_path)
            report["phase"] = "ordinary_identical_options_baseline"
            write(report_path, report)
            baseline = lower_model(source, destination / "baseline", **arguments)
        report["baseline_llvm"] = file_ref(baseline.ll_path)
        report["checked_matches_ordinary_llvm"] = checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
        if not report["checked_matches_ordinary_llvm"]:
            raise ValueError("gate selection changed the complete ordinary compiler LLVM output")
        coverage, coverage_pin = read_json(worker.directory / "coverage.json")
        report.update(coverage=coverage, coverage_artifact=coverage_pin, gate_verdicts=worker.verdicts)
        report["phase"] = "current_complete_receipt_chain_audit"
        write(report_path, report)
        report["accepted_job_evidence"] = audit_coverage(coverage, worker.verdicts, worker.directory, tools)
        report["phase"] = "required_actual_transformations_and_proofs"
        require_selectors(manifest, coverage)
        report.update(
            status="PASS",
            phase="complete",
            verified_boundaries=coverage["verified_boundaries"],
            unverified_boundaries=coverage["unverified_boundaries"],
        )
    except BaseException as error:
        report.update(
            status="FAILED", failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc()
        )
        if isinstance(error, GateFailure):
            report["gate_failure"] = error.report
        if worker is not None:
            report["gate_verdicts"] = worker.verdicts
            if (worker.directory / "coverage.json").is_file():
                report["coverage"], report["coverage_artifact"] = read_json(worker.directory / "coverage.json")
    finally:
        if worker is not None:
            worker.close()
        try:
            for reference in refs:
                bound(reference)
            # Post-baseline and final-case custody checks include complete
            # receipts, source/target exports, tools and original model files.
            report["current_file_references_checked"] = len(check_references(report))
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.update(status="FAILED", final_custody_error=str(error))
        report["elapsed_seconds"] = perf_counter() - started
        write(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("compile-only",), default="compile-only")
    parser.add_argument("--job-timeout", type=float, default=900)
    args = parser.parse_args(argv)
    from merlin.verify.pass_validation import CheckerTools

    tools = CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), Path(os.environ["MERLIN_COMPILER_PYTHON"]))
    return 0 if execute(args.manifest, args.out, tools, job_timeout=args.job_timeout)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
