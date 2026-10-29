"""Independent actual-loop qualification; ordinary selected Lean, no mock proofs."""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from merlin.common.paths import repo_root
from merlin.verify import loop_transform_gate as gate
from merlin.verify.loop_transform_intake import Settings, Unavailable, Unsupported, file_ref, prepare, produce


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(command, directory, *, environment=None, input=None, timeout=120):
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "INTENT.json", {"command": list(map(str, command)), "timeout": timeout})
    try:
        process = subprocess.run(
            list(map(str, command)),
            cwd=repo_root(),
            env=environment,
            input=input,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = {"returncode": process.returncode, "timeout": False}
        stdout, stderr = process.stdout, process.stderr
    except subprocess.TimeoutExpired as error:
        result = {"returncode": None, "timeout": True}
        stdout, stderr = error.stdout or b"", error.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    save(directory / "PROCESS.json", result)
    return result, stdout


def verify(files, compiler, output):
    worker = repo_root() / "merlin/tests/ir/loop_transform_qualification_worker.py"
    result, stdout = run(
        [compiler, worker, "verify"], output, input=json.dumps({"files": list(map(str, files))}), timeout=120
    )
    if result["returncode"] != 0:
        raise RuntimeError("authoritative parser unavailable")
    rows = json.loads(stdout)["result"]
    if not all(row["valid"] for row in rows):
        raise ValueError("fixture MLIR verifier refused: " + str(rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--lean", required=True, type=Path)
    parser.add_argument("--compiler-python", required=True, type=Path)
    parser.add_argument("--q-source-cut", type=Path, help="exact reviewed successor Q source manifest")
    parser.add_argument(
        "--protocol-only", action="store_true", help="independent protocol subset; not the full positive roster"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.objects = args.objects.resolve()
    from loop_transform_qualification_cases import POSITIVES, mutations, unsupported_sources

    source_paths = [
        Path(__file__),
        repo_root() / "merlin/tests/ir/loop_transform_qualification_cases.py",
        repo_root() / "merlin/tests/ir/loop_transform_qualification_worker.py",
        repo_root() / "merlin/tests/ir/loop_transform_qualification_classifier.py",
        repo_root() / "merlin/tests/ir/test_loop_transform_qualification_classifier.py",
    ]
    qcut = args.q_source_cut or (
        repo_root()
        / "out/artifacts/verification/merlin-passes/loop-scope-20261028/Q-INTEGRATION/source-cut-v5/SOURCES.json"
    )
    selected = json.loads(qcut.read_text())
    for row in selected["sources"]:
        if file_ref(repo_root() / row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Q frozen source changed")
    source_cut = {
        "Q_cut": file_ref(qcut),
        "F_sources": [file_ref(f) for f in source_paths],
        "lean": file_ref(args.lean),
        "compiler_python": file_ref(args.compiler_python.resolve()),
        "reference_revision": "4499eebb959c0f92cefd92cc6d80646ba66474ed",
    }
    save(output / "SOURCE-CUT.json", source_cut)
    cases = []
    for case in POSITIVES[1:2] if args.protocol_only else POSITIVES:
        settings = Settings(case.factor, case.tile)
        buffers = produce(case.source, reference=args.reference, settings=settings)
        cases.append((case.name, case.source, settings, buffers, dataclasses.asdict(case)))
    historical = (
        repo_root() / "out/artifacts/verification/merlin-passes/loop-scope-20261028/Q-INTEGRATION/first-example"
    )
    buffers = tuple((historical / f"stage{i}.mlir").read_text() for i in range(3))
    if not args.protocol_only:
        cases.append(("witness_example", buffers[0], Settings(), buffers, {"reused_current_Q_example": True}))
    all_files = []
    for name, source, settings, buffers, metadata in cases:
        folder = output / "positives" / name
        folder.mkdir(parents=True)
        (folder / "original-input.mlir").write_text(source)
        for i, text in enumerate(buffers):
            path = folder / f"stage{i}.mlir"
            path.write_text(text)
            all_files.append(path)
        save(folder / "CASE.json", {"settings": dataclasses.asdict(settings), "metadata": metadata})
    verify(all_files, args.compiler_python, output / "positive-mlir-verifier")
    results = []
    live = {}
    for name, source, settings, buffers, metadata in cases:
        prepared = prepare(buffers, reference=args.reference, settings=settings)
        save(output / "positives" / name / "PREPARED.json", prepared.describe())
        token = gate.check(prepared, lean=args.lean, destination=args.objects / "positives" / name)
        delivered = []
        gate.deliver(token, prepared, delivered.append)
        if delivered != [buffers[-1]]:
            raise ValueError("consumer received wrong bytes")
        (output / "positives" / name / "delivered.mlir").write_text(delivered[0])
        report = json.loads(Path(token.report).read_text())
        results.append(
            {
                "name": name,
                "report": file_ref(Path(token.report)),
                "roots": len(prepared.roots),
                "theorems": report["audit"]["theorems_audited"],
                "consumer_bytes_identical": True,
                "short_last_segment": name != "witness_example",
                "status": "KERNEL_CHECKED",
            }
        )
        live[name] = (prepared, token)
        save(output / "POSITIVES.json", results)
    # Every malformed target must first pass the authoritative verifier.
    name = "factor3_carry2_nonaligned"
    prepared, token = live[name]
    negative = mutations(prepared.buffers)
    files = []
    for label, buffers in negative.items():
        folder = output / "negatives" / label
        folder.mkdir(parents=True)
        for i, text in enumerate(buffers):
            path = folder / f"stage{i}.mlir"
            path.write_text(text)
            files.append(path)
    verify(files, args.compiler_python, output / "negative-mlir-verifier")
    negative_results = []
    for label, buffers in negative.items():
        folder = output / "negatives" / label
        called = []
        try:
            request = prepare(buffers, reference=args.reference, settings=prepared.settings)
            completion = gate.check(request, lean=args.lean, destination=args.objects / "unexpected-negative" / label)
            gate.deliver(completion, request, called.append)
        except (Unsupported, Unavailable, ValueError) as error:
            outcome = {"phase": "HOST_REFUSAL", "class": type(error).__name__, "reason": str(error)}
        else:
            raise AssertionError("mutated boundary passed live gate: " + label)
        if called:
            raise AssertionError("mutated output reached consumer")
        # Direct component queries challenge the Lean relation independently of
        # optimizer-attribution refusal. They never create live gate authority.
        request = {
            "reference": str(args.reference.resolve()),
            "projection_sources": str(repo_root() / "src/merlin/verify"),
            "buffers": buffers,
            "factor": 3,
            "tile": 2,
        }
        environment = {**os.environ, "PYTHONPATH": str(args.reference.resolve() / "src")}
        process, stdout = run(
            [sys.executable, repo_root() / "merlin/tests/ir/loop_transform_qualification_worker.py", "project"],
            folder / "projection",
            environment=environment,
            input=json.dumps(request),
        )
        try:
            document = json.loads(stdout)
            if process["returncode"] != 0:
                raise ValueError(document.get("reason", "projection unavailable"))
            from merlin.verify.lean_graph_state_pipeline_emit import emit

            query = emit(document["result"])
            path = folder / "Query.lean"
            path.write_text(query)
        except ValueError as error:
            outcome["component"] = "HOST_PROJECTION_OR_DOMAIN_REFUSAL"
            outcome["component_reason"] = str(error)
        else:
            env = {**os.environ, "LEAN_PATH": str(args.objects / "positives" / name / "lib")}
            result, stdout = run([args.lean, "-j1", path], folder / "lean", environment=env)
            from loop_transform_qualification_classifier import classify

            classified = classify(
                label, result["returncode"], stdout, (folder / "lean" / "stderr.txt").read_text(), query, path
            )
            save(folder / "lean" / "CLASSIFICATION.json", classified)
            if classified["classification"] != "LEAN_COMPONENT_OBLIGATION_REFUSAL":
                outcome["component"] = "PROOF_UNAVAILABLE"
                outcome["component_reason"] = classified
            else:
                outcome["component"] = "LEAN_PROOF_REFUSAL"
                outcome["failed_obligation"] = classified["root"]
        negative_results.append({"name": label, "MLIR_valid": True, "consumer_called": False, **outcome})
        save(output / "NEGATIVES.json", negative_results)
    unsupported = []
    sources = unsupported_sources()
    for label, source in sources.items():
        path = output / "unsupported" / (label + ".mlir")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    verify(list((output / "unsupported").glob("*.mlir")), args.compiler_python, output / "unsupported-mlir-verifier")
    for label, source in sources.items():
        try:
            buffers = produce(source, reference=args.reference)
            prepare(buffers, reference=args.reference)
        except Unsupported as error:
            unsupported.append({"name": label, "outcome": "UNSUPPORTED", "reason": str(error)})
        else:
            raise AssertionError("outside-domain source admitted: " + label)
    for settings in [Settings(1), Settings(5), Settings(tile=1), Settings(tile=17), Settings(index_bits=32)]:
        try:
            prepare(prepared.buffers, reference=args.reference, settings=settings)
        except Unsupported:
            unsupported.append({"settings": dataclasses.asdict(settings), "outcome": "UNSUPPORTED"})
        else:
            raise AssertionError("unsupported settings admitted")
    save(output / "UNSUPPORTED.json", unsupported)
    # Actual source -> producer -> checker -> CLI consumer, same expected bytes.
    source_path = output / "positives" / name / "original-input.mlir"
    delivered = output / "cli-delivered.mlir"
    status, stdout = run(
        [
            sys.executable,
            "-m",
            "merlin.verify.loop_transform_cli",
            "--source",
            source_path,
            "--reference",
            args.reference,
            "--lean",
            args.lean,
            "--destination",
            args.objects / "cli",
            "--factor",
            "3",
            "--tile",
            "2",
            "--output",
            delivered,
        ],
        output / "cli",
    )
    if status["returncode"] != 0 or delivered.read_text() != prepared.buffers[-1]:
        raise AssertionError("real CLI delivery failed")
    replayed = gate.replay(
        token.report, reference=args.reference, lean=args.lean, destination=args.objects / "fresh-replay"
    )
    save(
        output / "REPLAY.json", {"original": file_ref(Path(token.report)), "replayed": file_ref(Path(replayed.report))}
    )
    # Forged request values must stop before launching a native process.
    stale = []
    for field, value in [
        ("query", prepared.query + "\n"),
        ("roots", (prepared.roots[-1],)),
        ("domain", "general_tiling"),
    ]:
        forged = dataclasses.replace(prepared, **{field: value})
        with patch.object(gate, "_run", side_effect=AssertionError("native launched for forged request")):
            try:
                gate.check(forged, lean=args.lean, destination=args.objects / "forged" / field)
            except Unavailable:
                stale.append(field)
            else:
                raise AssertionError("forged preparation accepted")
    for candidate in [None, {"status": "checked"}, gate.Completion(prepared.identity, token.report)]:
        try:
            gate.deliver(candidate, prepared, lambda _: (_ for _ in ()).throw(AssertionError("forged delivery")))
        except Unavailable:
            pass
        else:
            raise AssertionError("saved/constructed completion accepted")
    try:
        gate.deliver(
            token,
            dataclasses.replace(prepared),
            lambda _: (_ for _ in ()).throw(AssertionError("copied request delivered")),
        )
    except Unavailable:
        pass
    else:
        raise AssertionError("copied request accepted")
    save(
        output / "BINDING-REFUSALS.json",
        {"forged_requests": stale, "constructed_completions": 3, "copied_live_request": 1},
    )
    # Each tamper owns a separately checked candidate. Original qualified outputs
    # stay immutable, and exact pre-attack snapshots remain available.
    attacks = []
    for kind in ["query", "object", "report", "middle"]:
        candidate = gate.check(prepared, lean=args.lean, destination=args.objects / "attacks" / kind)
        report = json.loads(Path(candidate.report).read_text())
        original = args.objects / "attacks" / kind
        snapshot = output / "attacks" / kind / "before"
        snapshot.parent.mkdir(parents=True)
        shutil.copytree(original, snapshot)
        target = {
            "query": report["query"]["path"],
            "object": report["object"]["path"],
            "report": candidate.report,
            "middle": report["buffers"][1]["path"],
        }[kind]
        path = Path(target)
        path.write_bytes(path.read_bytes() + b"\nSTALE")
        called = []
        try:
            gate.deliver(candidate, prepared, called.append)
        except Unavailable:
            pass
        else:
            raise AssertionError("stale " + kind + " delivered")
        if called:
            raise AssertionError("stale consumer invoked")
        attacks.append(
            {"kind": kind, "consumer_called": False, "changed": file_ref(path), "before_snapshot": str(snapshot)}
        )
        save(output / "ARTIFACT-ATTACKS.json", attacks)
    for row in selected["sources"]:
        if file_ref(repo_root() / row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Q source changed during batch")
    for ref in [*source_cut["F_sources"], source_cut["Q_cut"], source_cut["lean"], source_cut["compiler_python"]]:
        if file_ref(Path(ref["path"])) != ref:
            raise ValueError("selected qualification source/tool changed during batch")
    unavailable = [x["name"] for x in negative_results if x["component"] == "PROOF_UNAVAILABLE"]
    final = {
        "status": "PARTIAL_UNAVAILABLE"
        if unavailable
        else "PROTOCOL_SUBSET_PASSED_FULL_ROSTER_PENDING"
        if args.protocol_only
        else "SCOPED_LOOP_ENGINEERING_BATCH_PASSED",
        "full_positive_roster_checked": not args.protocol_only,
        "distinct_actual_positives": len(results),
        "MLIR_valid_mutations": len(negative_results),
        "Lean_component_refusals": sum(x["component"] == "LEAN_PROOF_REFUSAL" for x in negative_results),
        "host_component_refusals": sum(x["component"] == "HOST_PROJECTION_OR_DOMAIN_REFUSAL" for x in negative_results),
        "unavailable_mutations": unavailable,
        "explicit_unsupported": len(unsupported),
        "fresh_replay": True,
        "actual_CLI_delivery": True,
        "binding_attacks": attacks,
        "human_approved": False,
        "general_multidimensional_tiling": False,
        "native_fidelity": False,
        "compiler_wide_enforcement": False,
    }
    save(output / "RESULT.json", final)
    print(json.dumps({k: v for k, v in final.items() if k != "binding_attacks"}))
    if unavailable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
