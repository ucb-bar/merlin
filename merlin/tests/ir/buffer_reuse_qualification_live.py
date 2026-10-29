"""Fresh actual raw4 gate qualification, never a numerical/native fidelity claim."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from buffer_reuse_qualification import check_cut, write_candidate
from buffer_reuse_qualification_classifier import classify
from buffer_reuse_qualification_protocol import artifact_controls, request_controls
from buffer_reuse_qualification_review import compare as compare_queries
from loop_transform_qualification import save

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain
from merlin.verify.loop_transform_intake import file_ref


def replay_child(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    check_cut(args.q_source_cut)
    if len(gate._LIVE):
        raise ValueError("replay process did not start with an empty live ledger")
    save(output / "INTENT.json", {"incoming_ledger": 0, "pid": os.getpid(), "report": file_ref(args.report)})
    result = gate.replay(args.report, lean=args.lean, destination=args.objects, timeout=120)
    save(
        output / "RESULT.json",
        {
            "status": result.status,
            "reason": result.reason,
            "pid": os.getpid(),
            "incoming_ledger": 0,
            "issued_live_completions": len(gate._LIVE),
            "report": file_ref(Path(result.completion.report)) if result.completion else None,
            "human_approved": False,
        },
    )
    check_cut(args.q_source_cut)
    if result.status != "checked" or result.completion is None or len(gate._LIVE) != 1:
        raise SystemExit(1)


def full(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    check_cut(args.q_source_cut)
    if args.objects.exists():
        raise ValueError("exclusive new private object root required")
    corpus = args.corpus.resolve()
    qcut = json.loads(args.q_source_cut.read_text())
    args.objects = args.objects.resolve()
    source_paths = sorted((repo_root() / "merlin/tests/ir").glob("buffer_reuse_qualification*.py"))
    source_paths += sorted((repo_root() / "merlin/tests/ir").glob("test_buffer_reuse_qualification*.py"))
    source_paths += [
        repo_root() / "merlin/tests/ir/loop_transform_qualification.py",
        repo_root() / "merlin/tests/ir/loop_transform_qualification_classifier.py",
    ]
    source_refs = [file_ref(path) for path in source_paths]
    save(
        output / "SOURCE-CUT.json",
        {
            "Q_cut": file_ref(args.q_source_cut),
            "F_sources": source_refs,
            "lean": file_ref(args.lean),
            "corpus_expected": file_ref(corpus / "EXPECTED-POSITIVES.json"),
            "relation": "equal_possible_successful_public_observations",
            "old_gate_timeout_preserved": "native-first-v1",
            "human_approved": False,
        },
    )
    expected = json.loads((corpus / "EXPECTED-POSITIVES.json").read_text())
    if args.first_case is not None:
        if args.first_case not in {row["name"] for row in expected}:
            raise ValueError("requested first case absent from unchanged full roster")
        expected.sort(key=lambda row: row["name"] != args.first_case)
    verified_rows = json.loads(args.verification.read_text())["result"]["files"]
    verified = {str(Path(row["path"]).resolve()): row for row in verified_rows}
    if len(verified) != len(verified_rows) or any(not row["valid"] for row in verified_rows):
        raise ValueError("complete distinct authoritative verifier roster required")
    producer_refs = {}
    for row in expected:
        folder = corpus / "positives" / row["name"]
        capture = json.loads((folder / "producer/stdout.txt").read_text())["result"]
        if capture["producer_revision"] != "4499eebb959c0f92cefd92cc6d80646ba66474ed":
            raise ValueError("wrong actual producer capture version")
        if tuple(capture["buffers"]) != tuple((folder / f"stage{i}.mlir").read_text() for i in range(4)):
            raise ValueError("raw4 corpus no longer matches actual producer output")
        for ref in capture["producer_sources"]:
            if ref["path"] in producer_refs and producer_refs[ref["path"]] != ref:
                raise ValueError("conflicting producer source pins")
            producer_refs[ref["path"]] = ref
    for ref in producer_refs.values():
        if file_ref(Path(ref["path"])) != ref:
            raise ValueError("pinned producer source changed after capture")
    save(
        output / "EXACT-INPUTS.json",
        {
            "authoritative_verifier": file_ref(args.verification),
            "parsed_bytes": [row["parsed_bytes"] for row in verified_rows],
            "producer_sources": list(producer_refs.values()),
        },
    )

    def buffers_at(folder):
        values = []
        for index in range(4):
            path = (folder / f"stage{index}.mlir").resolve()
            if str(path) not in verified or file_ref(path) != verified[str(path)]["parsed_bytes"]:
                raise ValueError("actual raw buffer differs from authoritative parsed bytes")
            values.append(path.read_text())
        return tuple(values)

    results, candidates, completions = [], {}, {}
    for row in expected:
        name = row["name"]
        folder = output / "positives" / name
        folder.mkdir(parents=True)
        buffers = buffers_at(corpus / "positives" / name)
        command = [sys.executable, str(repo_root() / "merlin/tests/ir/buffer_reuse_qualification_worker.py"), "produce"]
        request = {
            "reference": str(args.reference.resolve()),
            "source": (corpus / "positives" / name / "original-input.mlir").read_text(),
            "tile": row["tile"],
        }
        save(folder / "PRODUCER-INTENT.json", {"command": command, "request": request, "timeout": 120})
        try:
            producer = run_owned(
                command,
                cwd=repo_root(),
                env={**os.environ, "PYTHONPATH": str(args.reference.resolve() / "src")},
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            save(folder / "PRODUCER-FAILURE.json", {"reason": str(error), "cleanup": getattr(error, "__notes__", [])})
            for stream in ("stdout", "stderr"):
                data = getattr(error, stream, "") or ""
                (folder / ("producer." + stream)).write_text(
                    data.decode(errors="replace") if isinstance(data, bytes) else data
                )
            results.append({"name": name, "status": "producer_unavailable", "consumer_called": False})
            save(output / "POSITIVES.json", results)
            continue
        (folder / "producer.stdout").write_text(producer.stdout)
        (folder / "producer.stderr").write_text(producer.stderr)
        save(folder / "PRODUCER-PROCESS.json", {"returncode": producer.returncode})
        if producer.returncode or tuple(json.loads(producer.stdout)["result"]["buffers"]) != buffers:
            results.append({"name": name, "status": "producer_unavailable_or_bytes_differ", "consumer_called": False})
            save(output / "POSITIVES.json", results)
            continue
        candidate = prepare_chain(buffers, allocator=ALLOCATOR)
        previous = corpus / "positives" / name / "CanonicalBufferQuery.lean"
        review = compare_queries(previous.read_text(), candidate.query)
        save(
            folder / "QUERY-PRESERVATION.json",
            {
                "old": file_ref(previous),
                "fresh_query_sha256": candidate.describe()["query_sha256"],
                "comparison": review,
                "scope": (
                    "all definitions and original theorem headers/context exact; "
                    "proof bodies may change under frozen Qcut; actual Lean still required"
                ),
            },
        )
        if tuple(part["module"] for part in json.loads(candidate.query_parts_json)) != tuple(qcut["modules"]):
            raise ValueError("generated module roster differs from frozen successor")
        write_candidate(folder, candidate)
        result = gate.check(candidate, lean=args.lean, destination=args.objects / "positives" / name, timeout=120)
        item = {"name": name, "status": result.status, "reason": result.reason, "roots": len(candidate.roots)}
        if result.status == "checked" and result.completion is not None:
            delivered = []
            gate.deliver(result.completion, candidate, delivered.append)
            if delivered != [buffers[-1]]:
                raise AssertionError("consumer did not receive exact final bytes")
            (folder / "delivered.mlir").write_text(delivered[0])
            item.update(
                {
                    "report": file_ref(Path(result.completion.report)),
                    "consumer_called": True,
                    "final_bytes_identical": True,
                }
            )
            candidates[name], completions[name] = candidate, result.completion
        else:
            item["consumer_called"] = False
        results.append(item)
        save(output / "POSITIVES.json", results)
        print(json.dumps({"positive": name, "status": result.status}), flush=True)
    baseline = prepare_chain(
        tuple((corpus / "positives/retained_two_epochs" / f"stage{i}.mlir").read_text() for i in range(4)),
        allocator=ALLOCATOR,
    )
    request_controls(baseline, output / "request-controls")
    negative_results = []
    for row in json.loads((corpus / "NEGATIVES.json").read_text()):
        name = row["name"]
        buffers = buffers_at(corpus / "negatives" / name)
        try:
            candidate = prepare_chain(buffers, allocator=ALLOCATOR)
        except ValueError as error:
            negative_results.append(
                {"name": name, "status": "HOST_REFUSAL", "reason": str(error), "consumer_called": False}
            )
        else:
            destination = args.objects / "negatives" / name
            result = gate.check(candidate, lean=args.lean, destination=destination, timeout=120)
            if result.status == "checked" or result.completion is not None:
                raise AssertionError("mutant unexpectedly established a live completion: " + name)
            item = {
                "name": name,
                "gate_status": result.status,
                "reason": result.reason,
                "consumer_called": False,
                "status": "UNAVAILABLE",
            }
            for index, part in enumerate(json.loads(candidate.query_parts_json)):
                stem = destination / f"query-{index}-{part['module']}"
                process = Path(str(stem) + ".process.json")
                if process.exists() and json.loads(process.read_text())["returncode"] == 1:
                    source = (destination / (part["module"] + ".lean")).resolve()
                    actual_command = json.loads(process.read_text())["command"]
                    if Path(actual_command[-1]).resolve() != source or file_ref(source)["sha256"] != part["sha256"]:
                        raise ValueError("native failure not bound to the actual prepared source bytes/command")
                    classified = classify(
                        name,
                        1,
                        Path(str(stem) + ".stdout").read_text(),
                        Path(str(stem) + ".stderr").read_text(),
                        source.read_text(),
                        source,
                    )
                    item["native_classification"] = classified
                    if (
                        result.status == "not_established"
                        and classified["status"] == "EXPECTED_COMPONENT_OBLIGATION_REFUSAL"
                    ):
                        item["status"] = "ACTUAL_GATE_EXPECTED_OBLIGATION_REFUSAL"
                    break
            negative_results.append(item)
        save(output / "NEGATIVES.json", negative_results)
    protocol = {"status": "UNAVAILABLE_NO_LIVE_BASELINE"}
    if "retained_two_epochs" in candidates:
        checked = gate.check(
            baseline, lean=args.lean, destination=args.objects / "dedicated-attack-scratch", timeout=120
        )
        if checked.status == "checked" and checked.completion is not None:
            artifact_controls(checked.completion, baseline, output / "artifact-controls")
            protocol = {"status": "checked", "dedicated_attack_report": file_ref(Path(checked.completion.report))}
        else:
            protocol = {"status": checked.status, "reason": checked.reason}
        source_report = Path(completions["retained_two_epochs"].report)
        child = output / "fresh-process-replay"
        command = [
            sys.executable,
            str(Path(__file__)),
            "--phase",
            "replay",
            "--report",
            str(source_report),
            "--output",
            str(child),
            "--objects",
            str(args.objects / "fresh-process-replay"),
            "--q-source-cut",
            str(args.q_source_cut.resolve()),
            "--lean",
            str(args.lean),
        ]
        save(output / "REPLAY-INTENT.json", {"command": command, "timeout": 1800, "owned_process_group": True})
        try:
            process = run_owned(
                command,
                cwd=repo_root(),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            (output / "replay.stdout").write_text(process.stdout)
            (output / "replay.stderr").write_text(process.stderr)
            save(output / "REPLAY-PROCESS.json", {"returncode": process.returncode, "timeout": False})
            protocol["fresh_process_replay"] = process.returncode == 0
        except (OSError, subprocess.SubprocessError) as error:
            for name in ("stdout", "stderr"):
                value = getattr(error, name, "") or ""
                (output / ("replay." + name)).write_text(
                    value.decode(errors="replace") if isinstance(value, bytes) else value
                )
            save(
                output / "REPLAY-PROCESS.json",
                {"returncode": None, "reason": str(error), "cleanup": getattr(error, "__notes__", [])},
            )
            protocol["fresh_process_replay"] = False
    save(output / "PROTOCOL.json", protocol)
    for ref in source_refs:
        if file_ref(Path(ref["path"])) != ref:
            raise ValueError("F source changed during live qualification")
    for ref in [*producer_refs.values(), *[row["parsed_bytes"] for row in verified_rows]]:
        if file_ref(Path(ref["path"])) != ref:
            raise ValueError("captured producer/source/actual verified IR bytes changed during qualification")
    check_cut(args.q_source_cut)
    failed = [row["name"] for row in results if row["status"] != "checked"]
    failed += [row["name"] for row in negative_results if row["status"] == "UNAVAILABLE"]
    if protocol["status"] != "checked" or not protocol.get("fresh_process_replay"):
        failed.append("live_protocol_or_fresh_replay")
    final = {
        "status": "BOUNDED_BUFFER_ENGINEERING_BATCH_PASSED" if not failed else "PARTIAL_UNAVAILABLE",
        "expected_positives": len(expected),
        "actual_gate_positives": sum(row["status"] == "checked" for row in results),
        "valid_mutant_quadruples": len(negative_results),
        "actual_gate_native_refusals": sum(
            row["status"] == "ACTUAL_GATE_EXPECTED_OBLIGATION_REFUSAL" for row in negative_results
        ),
        "host_refusals": sum(row["status"] == "HOST_REFUSAL" for row in negative_results),
        "failed_or_unavailable": failed,
        "protocol": protocol,
        "relation": "equal_possible_successful_public_observations",
        "full_error_world_equivalence": False,
        "human_approved": False,
        "native_allocator_fidelity": False,
        "production_enforcement": False,
    }
    save(output / "RESULT.json", final)
    print(json.dumps(final), flush=True)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("full", "replay"), default="full")
    for name in ("output", "objects", "q-source-cut", "lean"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--first-case", help="execution priority only; never removes cases")
    args = parser.parse_args()
    if args.phase == "replay":
        if args.report is None:
            parser.error("replay requires --report")
        replay_child(args)
    else:
        if args.corpus is None or args.verification is None or args.reference is None:
            parser.error("full requires --corpus --verification --reference")
        full(args)


if __name__ == "__main__":
    main()
