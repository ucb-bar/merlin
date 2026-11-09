"""Private component coverage receipts and independent source/member verification.

Generation establishes finite input coverage, never candidate acceptance. Public
summaries expose only commitments and cohort counts, never transfer selectors.
"""

from __future__ import annotations

import copy
import hashlib
import json
from enum import StrEnum
from pathlib import Path

import yaml

from .component_coverage_plan import COHORTS
from .component_generation import digest

REPORT_SCHEMA = "merlin.phase0.component_coverage.v1"
BUDGETED_REPORT_SCHEMA = "merlin.phase0.component_coverage.v2"


class CoverageState(StrEnum):
    GENERATED = "generated"
    SOURCE_GENERATED = "source_generated"
    VERIFIED_REFUSAL = "verified_refusal"
    UNAVAILABLE = "unavailable"


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def finalize(report, *, root, written, failures, generation_identity, semantic_basis=None):
    """Join obligations to actual independently screened program and golden bytes."""
    from merlin.targetgen import golden_store
    from merlin_experiments.phase1.source_inputs import fingerprint

    from .coverage_commitment import _selected_program

    root = Path(root)
    report = copy.deepcopy(report)
    by_name = {Path(path).name: Path(path) for path in written}
    failed = {name: reason for name, reason in failures}
    report["generation_identity"] = copy.deepcopy(generation_identity)
    if "hardware_intake_sha256" in generation_identity:
        report["hardware_intake_sha256"] = generation_identity["hardware_intake_sha256"]
    if "software_intake_sha256" in generation_identity:
        report["software_intake_sha256"] = generation_identity["software_intake_sha256"]
    if semantic_basis is not None:
        report["semantic_basis_sources"] = semantic_basis.sources()
    report["generation_failures"] = [{"capsule": name, "reason": reason} for name, reason in failures]
    for row in report["obligations"]:
        states = []
        for member in row["members"]:
            directory = by_name.get(member["name"])
            if directory is None:
                member.update(
                    state="unavailable", reason=failed.get(member["name"], "normal writer produced no member")
                )
                states.append("unavailable")
                continue
            capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            screen = capsule.get("software_screen") or {}
            source_screen = capsule.get("source_semantics_screen") or {}
            source_mode = generation_identity.get("source_semantics_admission")
            program = _selected_program(capsule, directory)
            try:
                golden = golden_store.load_golden(directory)
                if not isinstance(golden, dict) or not golden.get("outputs") or program is None:
                    raise ValueError("normal writer omitted program or independent full-output golden")
                from .component_integer_bounds import verify_selected_capsule

                verify_selected_capsule(
                    capsule, semantics_sha256=report["numerical_semantics_sha256"], require_bound=True
                )
                from .sealed_generation import verified_capture_failure

                capture_failure = verified_capture_failure(capsule)
                if capture_failure is not None:
                    raise ValueError(capture_failure)
                stamp = capsule.get("component_coverage") or {}
                if (
                    stamp.get("plan_sha256") != report["plan"]["sha256"]
                    or stamp.get("point_sha256") != member["point_sha256"]
                ):
                    raise ValueError("written member lost its exact obligation/input identity")
                from .component_input_witnesses import NUMERIC_INPUT_EFFECTS, verify

                witness = verify(capsule, golden)
                effect_owners = {owner["id"]: owner["kind"] for owner in report["declaration"]["effects"]}
                required_effects = {effect_owners[name] for name in stamp["effect_owners"]} & NUMERIC_INPUT_EFFECTS
                if not required_effects <= set(witness["effects"]):
                    raise ValueError("selected numerical effect has no actual independent operand witness")
                member["input_palette_witness"] = witness
                from .component_source_witnesses import SOURCE_NUMERIC_EFFECTS
                from .component_source_witnesses import verify as verify_source

                source_witness = verify_source(capsule, golden)
                if not ({effect_owners[name] for name in stamp["effect_owners"]} & SOURCE_NUMERIC_EFFECTS) <= set(
                    source_witness["effects"]
                ):
                    raise ValueError("selected source numerical mechanism has no complete independent witness")
                member["source_mechanism_witness"] = source_witness
                if row["expectation"] == "unsupported_program" and screen.get("status") == "unsupported":
                    state = CoverageState.VERIFIED_REFUSAL.value
                elif row["expectation"] == "admitted_program" and source_mode is not None:
                    if (
                        any(source_screen.get(key) != value for key, value in source_mode.items())
                        or source_screen.get("status") != "source_admitted"
                        or source_screen.get("program_sha256") != hashlib.sha256(program.read_bytes()).hexdigest()
                        or source_screen.get("output_roster")
                        != [item["name"] for item in capsule["component_program"]["outputs"]]
                        or member["name"] in failed
                    ):
                        raise ValueError("source-only member lacks its exact original semantic admission")
                    state = CoverageState.SOURCE_GENERATED.value
                elif (
                    row["expectation"] == "admitted_program"
                    and screen.get("status") == "admitted"
                    and member["name"] not in failed
                ):
                    state = CoverageState.GENERATED.value
                else:
                    raise ValueError(
                        f"written program admission {screen.get('status')} differs from declared expectation"
                    )
                member.update(
                    state=state,
                    member=directory.relative_to(root).as_posix(),
                    sha256=fingerprint(directory),
                    program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),
                    golden_source=golden.get("golden_source"),
                    output_roster=sorted(golden["outputs"]),
                    software_screen_sha256=digest(screen),
                    **({"source_semantics_screen_sha256": digest(source_screen)} if source_mode is not None else {}),
                    reason=source_screen.get("scope") if source_mode is not None else screen.get("reason"),
                )
            except (OSError, ValueError) as exc:
                state = CoverageState.UNAVAILABLE.value
                member.update(state=state, reason=str(exc))
            states.append(state)
        expected = (
            "verified_refusal"
            if row["expectation"] == "unsupported_program"
            else "source_generated"
            if generation_identity.get("source_semantics_admission") is not None
            else "generated"
        )
        row["state"] = (
            expected if states and all(state == expected for state in states) and not row["errors"] else "unavailable"
        )
    from .component_graph_relations import finalize as finalize_relations

    finalize_relations(report, root=root)
    complete = not failures and all(row["state"] != "unavailable" for row in report["obligations"] if row["mandatory"])
    if generation_identity.get("source_semantics_admission") is not None:
        report["status"] = "source_prepared" if complete else "source_prepared_incomplete"
    else:
        report["status"] = "complete" if complete else "incomplete"
    report["sha256"] = digest(report)
    return report


def public_summary(report):
    """The author-visible projection never includes obligations or hidden selectors."""
    return {
        "schema": report["schema"],
        "plan_sha256": report["plan"]["sha256"],
        "report_sha256": report["sha256"],
        "status": report["status"],
        "qualification": report["qualification"],
        "cohorts": {
            cohort: {
                "obligations": sum(row["cohort"] == cohort for row in report["obligations"]),
                "mandatory_unavailable": sum(
                    row["cohort"] == cohort and row["mandatory"] and row["state"] == "unavailable"
                    for row in report["obligations"]
                ),
                "members": sum(len(row["members"]) for row in report["obligations"] if row["cohort"] == cohort),
            }
            for cohort in COHORTS
        },
    }


def write_report(root, report):
    """Private evidence uses the existing run-owned coverage directory."""
    destination = Path(root) / "_evidence" / "coverage" / "component-coverage.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    destination.chmod(0o600)
    return destination


def verify_report(root, report=None, *, verify_sources=True):
    """Recheck actual membership and bytes before downstream freeze/admission.

    Source checking reads only sources explicitly selected in the private receipt.
    A relocated freeze must supply its independently verified source closure and
    pass verify_sources=False; that option never relaxes capsule checks.
    """
    from merlin.targetgen import golden_store
    from merlin_experiments.phase1.source_inputs import fingerprint

    from .coverage_commitment import _selected_program

    root = Path(root)
    if report is None:
        report = json.loads((root / "_evidence/coverage/component-coverage.json").read_bytes())
    report = copy.deepcopy(report)
    stated = report.pop("sha256", None)
    if report.get("schema") not in {REPORT_SCHEMA, BUDGETED_REPORT_SCHEMA} or digest(report) != stated:
        raise ValueError("component coverage report identity changed")
    if report.get("hardware_intake_sha256") != (report.get("generation_identity") or {}).get("hardware_intake_sha256"):
        raise ValueError("component coverage independent hardware binding changed")
    if report.get("software_intake_sha256") != (report.get("generation_identity") or {}).get("software_intake_sha256"):
        raise ValueError("component coverage independent software binding changed")
    report["sha256"] = stated
    if (
        report["generation_identity"].get("source_semantics_admission") is not None
        or any(row["state"] == "source_generated" for row in report["obligations"])
        or any(member["state"] == "source_generated" for row in report["obligations"] for member in row["members"])
    ):
        raise ValueError("source-only preparation is not concrete hardware-admitted component coverage")
    if "automatic_derivation" in report:
        from .component_automatic import verify

        verify(report["automatic_derivation"], report=report, verify_sources=verify_sources)
    elif "automatic_derivation_sha256" in report.get("generation_identity", {}):
        raise ValueError("component coverage lost the independently selected automatic derivation")
    if report["status"] != "complete" or any(
        row["mandatory"] and row["state"] == "unavailable" for row in report["obligations"]
    ):
        raise ValueError("mandatory component coverage is unavailable")
    if verify_sources:
        sources = [
            report["plan"],
            *(report.get("generation_identity") or {}).get("generator_sources", []),
            *(report.get("generation_identity") or {}).get("selected_sources", []),
        ]
        sources += [(report.get("generation_identity") or {})[key] for key in ("recipe", "shared_template")]
        sources += report.get("semantic_basis_sources", [])
        for source in sources:
            if _file(source["path"])["sha256"] != source["sha256"]:
                raise ValueError("selected component generation source changed")
    budgeted = report["schema"] == BUDGETED_REPORT_SCHEMA
    from .component_coverage_plan import BUDGETED_PLAN_SCHEMA

    if (
        budgeted != (report["declaration"].get("schema") == BUDGETED_PLAN_SCHEMA)
        or budgeted != ("execution_budget" in report["declaration"])
        or budgeted != ("execution_admission" in report)
        or budgeted != ("execution_budget_sha256" in report["generation_identity"])
        or budgeted != ("execution_admission_sha256" in report["generation_identity"])
    ):
        raise ValueError("component coverage execution evidence version changed")
    if budgeted:
        from .component_execution_budget import verify

        if verify_sources:
            selected = yaml.safe_load(Path(report["plan"]["path"]).read_bytes())
            if selected != report["declaration"]:
                raise ValueError("component coverage declaration differs from selected plan bytes")
        verify(root, report)
    seen = set()
    for row in report["obligations"]:
        for member in row["members"]:
            if member["state"] == "unavailable":
                continue
            relative = member["member"]
            path = Path(relative)
            if (
                path.is_absolute()
                or len(path.parts) != 2
                or any(part in {".", ".."} for part in path.parts)
                or relative in seen
            ):
                raise ValueError("component coverage member path is invalid or duplicated")
            seen.add(relative)
            directory = root / path
            if (
                directory.is_symlink()
                or directory.resolve().parent.parent != root.resolve()
                or fingerprint(directory) != member["sha256"]
            ):
                raise ValueError("component coverage member bytes changed")
            capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            screen = capsule.get("software_screen") or {}
            if capsule.get("source_semantics_screen") is not None or not (
                (
                    member["state"] == "generated"
                    and row["expectation"] == "admitted_program"
                    and screen.get("status") == "admitted"
                )
                or (
                    member["state"] == "verified_refusal"
                    and row["expectation"] == "unsupported_program"
                    and screen.get("status") == "unsupported"
                )
            ):
                raise ValueError("component coverage concrete admission differs from the original member expectation")
            from .component_integer_bounds import verify_selected_capsule

            verify_selected_capsule(
                capsule, semantics_sha256=report["numerical_semantics_sha256"], require_bound=budgeted
            )
            program = _selected_program(capsule, directory)
            golden = golden_store.load_golden(directory)
            if program is None or hashlib.sha256(program.read_bytes()).hexdigest() != member["program_sha256"]:
                raise ValueError("component coverage program identity changed")
            if (
                sorted(golden.get("outputs") or {}) != member["output_roster"]
                or digest(capsule.get("software_screen") or {}) != member["software_screen_sha256"]
            ):
                raise ValueError("component coverage full outputs or admission changed")
    from .component_graph_relations import verify as verify_relations

    verify_relations(report, root=root)
    return report


def build_guard_link(report):
    """Private Phase 2 guard obligations bind exact finite Phase 1 inputs."""
    return {
        "schema": "merlin.phase0.component_guard_obligation.v2",
        "status": "established" if report["status"] == "complete" else "not_established",
        "plan_sha256": report["plan"]["sha256"],
        "coverage_report_sha256": report["sha256"],
        "guards": [
            copy.deepcopy(member)
            for row in report["obligations"]
            if row["cohort"] == "functional_guard"
            for member in row["members"]
            if member["state"] in {"generated", "verified_refusal"}
        ],
        "qualification": "generated full outputs and declared refusals; candidate numerical acceptance unestablished",
    }
