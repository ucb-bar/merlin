"""Run independent Phase 0 from explicit source declarations and fresh issuers.

The request contains source selections and construction budgets, never saved
intake authority. Diagnostic execution preserves every mandatory unknown and
does not issue a Phase 1 release or hardware/numerical qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import component_automatic as A
from . import operator_schema_intake as S
from .arithmetic_intake import issue_independent_arithmetic_intake
from .component_execution_budget import validate as validate_execution_budget
from .component_generation import DECLARATION, digest
from .evidence import select_evidence
from .generation import generate_target
from .original_call_sources import validate_budget as validate_source_budget
from .packing_intake import issue_independent_packing_intake
from .rtl_intake import _outside, _plain, issue_independent_hardware_intake
from .software_intake import issue_independent_software_intake

SCHEMA = "merlin.independent_phase0_run.v1"
BRIDGE_SCHEMA = "merlin.independent_phase0_run.v2"
REPORT_SCHEMA = "merlin.independent_phase0_run_report.v1"
_INPUTS = {"descriptor", "hardware_selection", "software_source", "software_review", "semantic_basis"}


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def _pin(value, *, forbidden, runtime=False):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("declared input needs an exact path and SHA256")
    path = Path(value["path"])
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("declared inputs need explicit absolute ordinary paths")
    _outside(path, forbidden)
    if runtime:
        # The caller explicitly selects a venv interpreter alias. The existing
        # native schema issuer separately pins its actual executable and venv.
        actual = path.resolve(strict=True)
        _outside(actual, forbidden)
        _plain(actual)
    else:
        path = _plain(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
        raise ValueError("declared input bytes changed before live issuance")
    return path


def validate(request):
    """Close source and policy declarations before any authority is issued."""
    fields = {"schema", "target", "inputs", "operator_schemas", "circt_opt", "forbidden_roots", "automatic"}
    if (
        not isinstance(request, dict)
        or set(request) != fields
        or request["schema"] not in {SCHEMA, BRIDGE_SCHEMA}
        or not isinstance(request["target"], str)
        or not request["target"]
        or not isinstance(request["inputs"], dict)
        or set(request["inputs"]) != _INPUTS
        or not isinstance(request["forbidden_roots"], list)
        or not request["forbidden_roots"]
        or any(
            not isinstance(root, str) or not Path(root).is_absolute() or ".." in Path(root).parts
            for root in request["forbidden_roots"]
        )
    ):
        raise ValueError("independent Phase 0 needs a closed explicit declared-input request")
    operator = request["operator_schemas"]
    fields = {"schema", "status", "namespace", "python", "canonical_source"}
    tensor = isinstance(operator, dict) and operator.get("schema") in {
        S.TENSOR_SELECTION_SCHEMA,
        S.ZERO_SELECTION_SCHEMA,
    }
    zero = isinstance(operator, dict) and operator.get("schema") == S.ZERO_SELECTION_SCHEMA
    if tensor:
        fields.add("tensor_arguments")
    if zero:
        fields.add("zero_returns")
    versions = {S.SELECTION_SCHEMA}
    if request["schema"] == BRIDGE_SCHEMA:
        versions |= {S.TENSOR_SELECTION_SCHEMA, S.ZERO_SELECTION_SCHEMA}
    if (
        not isinstance(operator, dict)
        or set(operator) != fields
        or operator["schema"] not in versions
        or operator["status"] != "reviewed"
        or not isinstance(operator["canonical_source"], dict)
        or set(operator["canonical_source"]) != {"checkout", "commit", "declarations"}
    ):
        raise ValueError("original schemas require explicit public source/runtime declarations without saved authority")
    if tensor:
        selection = operator["tensor_arguments"]
        if (
            operator["namespace"] != "aten"
            or not isinstance(selection, dict)
            or set(selection) != {"compiler"}
            or not isinstance(selection["compiler"], dict)
            or set(selection["compiler"]) != {"path", "sha256"}
        ):
            raise ValueError("native Tensor arguments require an exact explicit compiler byte selection")
    if zero and operator["zero_returns"] != operator["tensor_arguments"]:
        raise ValueError("native zero returns must select the identical public SDK compiler")
    automatic = request["automatic"]
    if (
        not isinstance(automatic, dict)
        or set(automatic) != {"schema", "status", "budget", "execution_budget", "original_source_budget"}
        or automatic["schema"] not in {A.ORIGINAL_POLICY_SCHEMA, A.LINEAR_POLICY_SCHEMA}
        or automatic["status"] != "reviewed"
        or not isinstance(automatic["budget"], dict)
        or set(automatic["budget"]) != {"max_members", "max_interaction_cells"}
        or any(type(value) is not int or value < 1 for value in automatic["budget"].values())
    ):
        raise ValueError("full original Phase 0 needs explicit supported automatic construction budgets")
    validate_execution_budget(automatic["execution_budget"])
    validate_source_budget(automatic["original_source_budget"])
    return request


def _diagnostic_generation(operation, root):
    """Recognize only the ordinary completed mandatory-incomplete gate refusal."""
    try:
        return operation(), None
    except RuntimeError as error:
        receipt_path = root / "_evidence" / "coverage" / "generation.json"
        coverage_path = root / "_evidence" / "coverage" / "component-coverage.json"
        if not receipt_path.is_file() or not coverage_path.is_file():
            raise
        receipt = json.loads(receipt_path.read_bytes())
        coverage = json.loads(coverage_path.read_bytes())
        expected = [
            {
                "capsule": "component coverage",
                "reason": "mandatory independent obligations unavailable; inspect private coverage report",
            }
        ]
        if (
            receipt.get("schema") != "merlin.phase0_generation.v1"
            or receipt.get("mode") != "diagnostic"
            or receipt.get("qualification") != "not_established"
            or receipt.get("failures") != expected
            or receipt.get("corpus_manifest") != str(root / "MANIFEST.yaml")
            or not (root / "MANIFEST.yaml").is_file()
            or coverage.get("status") != "source_prepared_incomplete"
            or not any(row["mandatory"] and row["state"] == "unavailable" for row in coverage["obligations"])
        ):
            raise
        # This is a diagnostic disposition, never saved intake authority. The
        # caller still replays all original derivation and actual written bytes.
        return [root / row["member"] for row in receipt["capsule_commitments"]], str(error)


def _verify_diagnostic_products(root, coverage, paths, *, target, hardware, software):
    """Reopen complete source membership, written capsules and reference outputs."""
    import yaml

    from merlin.targetgen import golden_store
    from merlin_experiments.phase1.source_inputs import fingerprint

    from .component_coverage import public_summary
    from .component_source_binding import verify_prepared_sources

    if digest({key: value for key, value in coverage.items() if key != "sha256"}) != coverage.get("sha256"):
        raise ValueError("diagnostic component coverage identity changed")
    prepared = verify_prepared_sources(root, coverage, software=software, hardware=hardware)
    receipt = json.loads((root / "_evidence" / "coverage" / "generation.json").read_bytes())
    manifest = yaml.safe_load((root / "MANIFEST.yaml").read_bytes())
    commitments = receipt["capsule_commitments"]
    members = [member for row in coverage["obligations"] for member in row["members"]]
    if (
        receipt["capsules_written"] != len(commitments)
        or receipt["target"] != target
        or receipt.get("component_coverage") != public_summary(coverage)
        or coverage["hardware_intake_sha256"] != hardware.sha256
        or coverage["software_intake_sha256"] != software.sha256
        or prepared["prepared_members"] != len(commitments)
        or len(paths) != len(commitments)
        or {str(path) for path in paths} != {str(root / row["member"]) for row in commitments}
        or {row["member"] for row in members if row["state"] == "source_generated"}
        != {row["member"] for row in commitments}
        or manifest.get("phase0_evidence", {}).get("generation_receipt")
        != str(root / "_evidence" / "coverage" / "generation.json")
    ):
        raise ValueError("diagnostic generation lost complete written source membership")
    seen = set()
    for row in commitments:
        path = Path(row["member"])
        if path.is_absolute() or len(path.parts) != 2 or any(part in {".", ".."} for part in path.parts):
            raise ValueError("diagnostic capsule path is not an ordinary generated member")
        directory = root / path
        if row["member"] in seen or directory.is_symlink() or fingerprint(directory) != row["sha256"]:
            raise ValueError("diagnostic written capsule identity changed")
        seen.add(row["member"])
        selected = [member for member in members if member.get("member") == row["member"]]
        golden = golden_store.load_golden(directory)
        capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        if (
            len(selected) != 1
            or selected[0]["sha256"] != row["sha256"]
            or sorted(golden.get("outputs") or {}) != selected[0]["output_roster"]
            or digest(capsule.get("source_semantics_screen") or {}) != selected[0]["source_semantics_screen_sha256"]
            or digest(capsule.get("software_screen") or {}) != selected[0]["software_screen_sha256"]
        ):
            raise ValueError("diagnostic source lacks complete independent reference outputs")
    return len(commitments)


def run(request_path, *, output):
    """Issue selected public inputs and execute the ordinary complete diagnostic."""
    request_path = _plain(request_path)
    request_bytes = request_path.read_bytes()
    request = validate(json.loads(request_bytes))
    forbidden = tuple(Path(root) for root in request["forbidden_roots"])
    _outside(request_path, forbidden)
    inputs = {name: _pin(pin, forbidden=forbidden) for name, pin in request["inputs"].items()}
    operator = request["operator_schemas"]
    python = _pin(operator["python"], forbidden=forbidden, runtime=True)
    compiler_pin = operator.get("tensor_arguments", {}).get("compiler")
    compiler = _pin(compiler_pin, forbidden=forbidden) if compiler_pin is not None else None
    canonical = operator["canonical_source"]
    checkout = Path(canonical["checkout"])
    _outside(checkout, forbidden)
    _plain(checkout, directory=True)
    declarations = _pin(canonical["declarations"], forbidden=forbidden)
    circt_opt = _pin(request["circt_opt"], forbidden=forbidden)
    output = Path(output).absolute()
    _outside(output, forbidden)
    if output.exists() or ".." in output.parts or any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("independent Phase 0 needs one fresh ordinary run owner")
    selected_paths = [request_path, *inputs.values(), declarations, python, circt_opt, checkout]
    if compiler is not None:
        selected_paths.append(compiler)
    if any(path == output or path.is_relative_to(output) or output.is_relative_to(path) for path in selected_paths):
        raise ValueError("independent Phase 0 owner overlaps its declared inputs")
    output.mkdir(parents=True, mode=0o700)
    _write(output / "request.json", request)
    report = {
        "schema": REPORT_SCHEMA,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "status": "running",
        "steps": [],
        "phases": {},
    }
    report_path = output / "report.json"

    def step(name, operation):
        report["steps"].append({"name": name, "status": "running"})
        _write(report_path, report)
        print("Phase 0: " + name, flush=True)
        value = operation()
        report["steps"][-1]["status"] = "completed"
        _write(report_path, report)
        return value

    try:
        hardware = step(
            "fresh_public_rtl_issuance",
            lambda: issue_independent_hardware_intake(
                target=request["target"],
                descriptor=inputs["descriptor"],
                source_bundle=inputs["hardware_selection"],
                forbidden_roots=forbidden,
                output=output / "hardware",
            ),
        )
        software = step(
            "fresh_original_software_issuance",
            lambda: issue_independent_software_intake(
                hardware=hardware,
                source=inputs["software_source"],
                review=inputs["software_review"],
                forbidden_roots=forbidden,
                output_root=output / "software",
            ),
        )
        record = json.loads(software.receipt_json)
        if record["semantic_basis_sha256"] != request["inputs"]["semantic_basis"]["sha256"]:
            raise ValueError("declared independent graph basis differs from protected original review")
        selection = {key: copy.deepcopy(operator[key]) for key in ("schema", "status", "namespace")}
        selection.update(
            python=str(python),
            software_intake_sha256=software.sha256,
            canonical_source={"checkout": str(checkout), "commit": canonical["commit"], "path": str(declarations)},
        )
        if compiler is not None:
            _pin(compiler_pin, forbidden=forbidden)
            selection["tensor_arguments"] = {"compiler": str(compiler)}
            if operator["schema"] == S.ZERO_SELECTION_SCHEMA:
                selection["zero_returns"] = {"compiler": str(compiler)}
        selection_path = output / "schema-selection.json"
        _write(selection_path, selection)
        schemas = step(
            "fresh_original_public_native_schemas",
            lambda: S.issue_independent_operator_schema_intake(
                software=software,
                selection=selection_path,
                forbidden_roots=forbidden,
                output=output / "schemas",
            ),
        )
        if compiler is not None:
            _pin(compiler_pin, forbidden=forbidden)
        arithmetic = step(
            "fresh_public_rtl_arithmetic",
            lambda: issue_independent_arithmetic_intake(
                hardware=hardware,
                circt_opt=circt_opt,
                forbidden_roots=forbidden,
                output=output / "arithmetic",
            ),
        )
        packing = step(
            "fresh_public_rtl_packing",
            lambda: issue_independent_packing_intake(
                hardware=hardware,
                circt_opt=circt_opt,
                forbidden_roots=forbidden,
                output=output / "packing",
            ),
        )
        facts = output / "hardware" / "facts.json"
        evidence = select_evidence(
            request["target"],
            descriptor=inputs["descriptor"],
            facts_path=facts,
            software_spec=inputs["software_source"],
            hardware_intake=hardware,
            software_intake=software,
            source_components=True,
        )
        identity = {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")}
        policy = copy.deepcopy(request["automatic"])
        policy.update(
            hardware=identity,
            software_spec_sha256=request["inputs"]["software_source"]["sha256"],
            numerical_semantics_sha256=digest(software.public_facts()["numerical_semantics"]),
            semantic_basis_sha256=request["inputs"]["semantic_basis"]["sha256"],
            operator_schema_intake_sha256=schemas.sha256,
            arithmetic_intake_sha256=arithmetic.sha256,
            packing_intake_sha256=packing.sha256,
        )
        policy_path, recipe, template = (
            output / name for name in ("automatic-policy.json", "recipe.json", "template.json")
        )
        _write(policy_path, A._closed_policy(policy))
        _write(
            recipe,
            {
                "semantic_basis": request["inputs"]["semantic_basis"],
                "component_performance": {
                    "schema": DECLARATION,
                    "status": "reviewed",
                    "hardware": identity,
                    "objectives": [],
                },
            },
        )
        _write(template, {"sweeps": []})
        generated = output / "generated"
        paths, gate_refusal = step(
            "ordinary_complete_mandatory_generation",
            lambda: _diagnostic_generation(
                lambda: generate_target(
                    request["target"],
                    descriptor=inputs["descriptor"],
                    recipe=recipe,
                    performance_template=template,
                    software_spec=inputs["software_source"],
                    rtl_facts=facts,
                    evidence_mode="diagnostic",
                    component_only=True,
                    component_coverage=policy_path,
                    hardware_intake=hardware,
                    software_intake=software,
                    operator_schema_intake=schemas,
                    arithmetic_intake=arithmetic,
                    packing_intake=packing,
                    output_root=generated,
                ),
                generated,
            ),
        )
        coverage_path = generated / "_evidence" / "coverage" / "component-coverage.json"
        coverage = json.loads(coverage_path.read_bytes())
        references = _verify_diagnostic_products(
            generated, coverage, paths, target=request["target"], hardware=hardware, software=software
        )
        unknowns = [row for row in coverage["obligations"] if row["mandatory"] and row["state"] == "unavailable"]
        original = coverage["automatic_derivation"]["original_call_sources"]
        source_rows = [member for row in original["members"] for member in row["source_members"]]
        report.update(
            status="diagnostic_executed",
            coverage_report=str(coverage_path),
            coverage_status=coverage["status"],
            mandatory_obligations=sum(row["mandatory"] for row in coverage["obligations"]),
            mandatory_unavailable=len(unknowns),
            mandatory_states=dict(Counter(row["state"] for row in coverage["obligations"] if row["mandatory"])),
            original_calls=sum(len(row["calls"]) for row in original["members"]),
            original_operators=len({call["target"] for row in original["members"] for call in row["calls"]}),
            original_source_members=len(source_rows),
            original_source_states=dict(Counter(row["status"] for row in source_rows)),
            generated_capsules=len(paths),
            independent_reference_capsules=references,
            candidate_comparisons=0,
            original_numerical_admissions=0,
            reference_scope=(
                "complete ordinary writer reference outputs under selected bounded source semantics; "
                "no original floating or candidate acceptance"
            ),
            coverage_gate_refusal=gate_refusal,
            missing_by_class=dict(
                Counter(row["kind"] for row in coverage["automatic_derivation"]["required_unknowns"])
            ),
            unknowns=unknowns,
            required_unknowns=coverage["automatic_derivation"]["required_unknowns"],
            hardware_unknowns=hardware.public_facts()["unknowns"],
            schema_unknowns=schemas.record()["unknowns"],
        )
        report["phases"] = {
            "0": {"status": "diagnostic_executed", "coverage_gate": "incomplete", "unknowns_retained": True},
            "1": {
                "status": "blocked",
                "handoff_accepted": False,
                "reason": "source-only diagnostic is not a qualified consumable Phase 1 release",
            },
            "2": {
                "status": "blocked",
                "reason": "requires the Phase 0 coverage gate, frozen compiler and actual runtime qualification",
            },
        }
        # A changed declaration never inherits successful live issuance.
        if request_path.read_bytes() != request_bytes:
            raise ValueError("declared Phase 0 request changed during execution")
        for pin in request["inputs"].values():
            _pin(pin, forbidden=forbidden)
        _pin(operator["python"], forbidden=forbidden, runtime=True)
        _pin(canonical["declarations"], forbidden=forbidden)
        _pin(request["circt_opt"], forbidden=forbidden)
        if compiler is not None:
            _pin(compiler_pin, forbidden=forbidden)
    except Exception as error:
        details = {"type": type(error).__name__, "message": str(error)}
        stderr = getattr(error, "stderr", None)
        if stderr:
            details["stderr"] = stderr.decode(errors="replace") if isinstance(stderr, bytes) else str(stderr)
        report.update(status="diagnostic_failed", error=details)
        if report["steps"] and report["steps"][-1]["status"] == "running":
            report["steps"][-1]["status"] = "failed"
        report["phases"] = {
            "0": {"status": "diagnostic_failed"},
            "1": {"status": "blocked"},
            "2": {"status": "blocked"},
        }
        _write(report_path, report)
        raise
    _write(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.request, output=args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "coverage_status",
                    "mandatory_obligations",
                    "mandatory_unavailable",
                    "original_calls",
                    "original_operators",
                    "original_source_members",
                    "original_source_states",
                    "generated_capsules",
                    "independent_reference_capsules",
                    "candidate_comparisons",
                    "missing_by_class",
                    "phases",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
