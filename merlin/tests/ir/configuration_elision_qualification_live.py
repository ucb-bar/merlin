"""Full independent frozen-source configuration gate and separate replay batch."""

import argparse
import json
import os
import sys
from pathlib import Path

from configuration_elision_qualification_classifier import classify
from configuration_elision_qualification_controls import artifacts, protocol
from configuration_elision_qualification_process import run, save
from configuration_elision_qualification_review import audit_decoding

from merlin.common.paths import repo_root
from merlin.verify import configuration_elision_gate as gate
from merlin.verify.configuration_elision_bridge import Unsupported, file_ref
from merlin.verify.configuration_elision_intake import prepare

EXPECTED_ROOTS = {
    "Merlin.GraphState.Configuration.put_idempotent",
    "Merlin.GraphState.Elision.initial_unknown",
    "Merlin.GraphState.Elision.redundant_assign_identity",
    "Merlin.GraphState.Elision.advance_knowledge",
    "Merlin.GraphState.Elision.omitted_identity",
    "Merlin.GraphState.Elision.replay_sound",
    "Merlin.GraphState.Elision.check_sound",
    "Merlin.GraphState.Elision.context_preserved",
    "Merlin.GraphState.Configuration.Entries.checkEntries_sound",
    "Merlin.GraphState.Configuration.Entries.related_invoke",
    "Merlin.GraphState.Configuration.Entries.check_sound",
    "Merlin.GraphState.Configuration.Entries.checked_frame",
    "Merlin.GraphState.Configuration.Entries.checked_observation",
    "ActualConfigurationElision.accepted",
    "ActualConfigurationElision.complete_frame",
    "ActualConfigurationElision.complete_observation",
    "ActualConfigurationElision.complete_entries",
}


def checked(ref):
    path = Path(ref["path"])
    if file_ref(path) != ref:
        raise ValueError("actual independent original/capture changed")
    return path


def source_cut(path):
    for row in json.loads(path.read_text())["sources"]:
        location = Path(row["path"])
        location = location if location.is_absolute() else repo_root() / location
        actual = file_ref(location)
        if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
            raise ValueError("reviewed frozen source cut changed")


def replay(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    incoming = len(gate._LIVE)
    if incoming:
        raise AssertionError("separate replay process had incoming completion authority")
    result = gate.replay(args.replay_report, lean=args.lean, destination=args.objects.resolve())
    if result.status != "checked" or result.completion is None or len(gate._LIVE) != 1:
        raise RuntimeError("fresh separate-process configuration replay unavailable: " + result.reason)
    document = dict(
        status="checked",
        pid=os.getpid(),
        incoming_ledger=incoming,
        newly_issued_completions=1,
        report=file_ref(result.completion.report),
        human_approved=False,
    )
    save(output / "RESULT.json", document)
    print(json.dumps(dict(status="ok", result=document)))


def campaign(args):
    output, objects = args.output.resolve(), args.objects.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if objects.exists():
        raise ValueError("exclusive new private proof root required")
    cuts = [args.q_source_cut.resolve(), args.f_source_cut.resolve()]
    for path in cuts:
        source_cut(path)
    save(output / "SOURCE-CUTS.json", [file_ref(path) for path in cuts])
    cohort = args.cohort.resolve()
    positives = json.loads((cohort / "POSITIVES.json").read_text())
    negative = json.loads((cohort / "NEGATIVES.json").read_text())
    if len(positives) != 14 or len(negative) != 22:
        raise ValueError("independent fixed corpus denominator changed")
    # One baseline first, then expand automatically only if native execution works.
    positives.sort(key=lambda row: row["name"] != "multifield")
    reports, failures, candidates, refused_boundaries = [], [], {}, []
    native_blocked = False

    def request(row):
        return dict(
            reference=str(args.reference.resolve()),
            facts=str(checked(row["schema"]["facts"])),
            software=str(checked(row["schema"]["software"])),
            target="configuration_fixture",
        )

    for row in positives:
        name = row["name"]
        source, final = (checked(ref).read_text() for ref in row["files"])
        try:
            candidate = prepare(source, final, **request(row))
        except Unsupported as error:
            if (
                row["Q_gate_expectation"] == "unsupported_constant_operand_domain"
                and "configuration fields require actual undecorated integer constants" in str(error)
            ):
                refused_boundaries.append(dict(case=name, status="EXPECTED_EXPLICIT_UNSUPPORTED", reason=str(error)))
                continue
            failures.append(dict(case=name, status="UNEXPECTED_HOST_REFUSAL", reason=str(error)))
            continue
        if row["Q_gate_expectation"] != "checked":
            failures.append(dict(case=name, status="EXPECTED_UNSUPPORTED_WAS_NOT_REFUSED"))
            continue
        if set(candidate.roots) != EXPECTED_ROOTS or len(candidate.roots) != 17:
            raise ValueError("complete final/support root roster changed")
        reviewed = audit_decoding(
            json.loads(candidate.projection_json), json.loads(checked(row["producer"]).read_text()), source, final
        )
        folder = output / "positives" / name
        save(folder / "PREPARATION.json", candidate.describe())
        save(folder / "DECODE-AUDIT.json", reviewed)
        save(folder / "PROJECTION.json", json.loads(candidate.projection_json))
        result = gate.check(candidate, lean=args.lean, destination=objects / "positives" / name)
        record = dict(name=name, status=result.status, reason=result.reason)
        if result.status == "checked":
            delivered = []
            gate.deliver(result.completion, candidate, lambda text: delivered.append(text))
            if delivered != [final]:
                raise AssertionError("consumer received different final configuration bytes")
            report = json.loads(Path(result.completion.report).read_text())
            if set(report["statements"]) != EXPECTED_ROOTS or report["audit"]["violations"] != []:
                raise AssertionError("native complete root/audit evidence mismatch")
            if not set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}:
                raise AssertionError("non-std3 axiom")
            record.update(
                consumer_called=True,
                final_bytes_identical=True,
                report=file_ref(result.completion.report),
                roots=len(report["statements"]),
                audited_theorems=report["audit"]["theorems_audited"],
                original_before_after=row["files"],
                actual_producer=row["producer"],
            )
            candidates[name] = candidate
        else:
            failures.append(record)
        reports.append(record)
        save(output / "POSITIVES.json", reports)
        print(json.dumps(dict(case=name, status=result.status)), flush=True)
        if result.status == "unavailable":
            native_blocked = True
            break

    negatives = []
    source_cases = {row["name"]: row for row in positives}
    for row in negative:
        if native_blocked:
            break
        source, final = (checked(ref).read_text() for ref in row["files"])
        try:
            candidate = prepare(source, final, **request(source_cases[row["source_case"]]))
        except Unsupported as error:
            record = dict(case=row["name"], status="HOST_DOMAIN_REFUSAL", reason=str(error))
            if row.get(
                "expected_refusal"
            ) == "HOST_RESERVED_METADATA_ADMISSION" and "reserved raw op_name__ attribute" not in str(error):
                failures.append(dict(case=row["name"], status="WRONG_HOST_REFUSAL_REASON", reason=str(error)))
        else:
            if row.get("expected_refusal") == "HOST_RESERVED_METADATA_ADMISSION":
                failures.append(dict(case=row["name"], status="RESERVED_RAW_METADATA_WAS_ADMITTED"))
                native_blocked = True
                break
            directory = objects / "negatives" / row["name"]
            result = gate.check(candidate, lean=args.lean, destination=directory)
            classification = (
                classify(directory, candidate.query)
                if result.status == "not_established"
                else dict(status="UNAVAILABLE")
            )
            record = dict(
                case=row["name"],
                status=classification["status"],
                gate_status=result.status,
                reason=result.reason,
                classification=classification,
            )
            if (
                result.status != "not_established"
                or classification["status"] != "ACTUAL_EXPECTED_LEAN_OBLIGATION_REFUSAL"
            ):
                failures.append(record)
                native_blocked = True
        negatives.append(record)
        save(output / "NEGATIVES.json", negatives)
        print(json.dumps(record), flush=True)
    native_refusals = sum(row["status"] == "ACTUAL_EXPECTED_LEAN_OBLIGATION_REFUSAL" for row in negatives)
    if not native_refusals:
        failures.append(dict(status="NO_ACTUAL_LEAN_MUTATION_REFUSAL"))
    baseline = candidates.get("multifield")
    controls = {}
    if baseline is not None and not native_blocked:
        controls["forged_requests"] = protocol(baseline, output / "protocol", args.lean)
        controls["artifact_attacks"] = artifacts(
            baseline, output / "artifact-controls", objects / "artifact-controls", args.lean
        )
        before_report = next(row["report"]["path"] for row in reports if row["name"] == "multifield")
        command = [
            sys.executable,
            Path(__file__).resolve(),
            "--replay-report",
            before_report,
            "--output",
            output / "fresh-process-replay",
            "--objects",
            objects / "fresh-process-replay",
            "--lean",
            args.lean,
        ]
        result = run(
            command, output / "fresh-process-driver", environment={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        if result["status"] != "checked" or result["incoming_ledger"] != 0 or result["newly_issued_completions"] != 1:
            raise AssertionError("separate fresh process did not independently recheck")
        controls["fresh_process_replay"] = True
    else:
        failures.append(dict(status="NO_BASELINE_FOR_LIVE_CONTROLS"))
    for path in cuts:
        source_cut(path)
    for row in positives:
        for item in row["files"]:
            checked(item)
    expected_supported = sum(row["Q_gate_expectation"] == "checked" for row in positives)
    result = dict(
        status="CONFIGURATION_ENGINEERING_QUALIFICATION_PASSED" if not failures else "PARTIAL_FAILED_OR_UNAVAILABLE",
        core_changed_positive_denominator=10,
        additional_boundary_denominator=4,
        expected_checked_pairs=expected_supported,
        actual_checked_pairs=len(candidates),
        expected_unsupported_boundaries=1,
        actual_unsupported_boundaries=len(refused_boundaries),
        valid_mutant_pairs=len(negative),
        attempted_mutant_pairs=len(negatives),
        actual_native_refusals=native_refusals,
        host_refusals=sum(row["status"] == "HOST_DOMAIN_REFUSAL" for row in negatives),
        failures=failures,
        native_stopped_on_first_unavailable=native_blocked,
        controls=controls,
        human_approved=False,
        hardware_model_fidelity=False,
        relation="complete conditional entry error and terminal state equality",
        production_enforcement=False,
    )
    save(output / "UNSUPPORTED.json", refused_boundaries)
    save(output / "RESULT.json", result)
    print(json.dumps(result))
    if failures or len(candidates) != expected_supported or len(refused_boundaries) != 1:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--q-source-cut", type=Path)
    parser.add_argument("--f-source-cut", type=Path)
    args = parser.parse_args()
    replay(args) if args.replay_report else campaign(args)


if __name__ == "__main__":
    main()
