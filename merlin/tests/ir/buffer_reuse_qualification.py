"""Independent scoped buffer corpus, kernel challenges and live gate controls.

Raw4 production is independently attributed to pinned4499. Q's gate checks a
distinct possible-success relation; neither source preparation nor this corpus
claims native allocator fidelity or human approval.
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from loop_transform_qualification import run, save

from merlin.common.paths import repo_root
from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare, prepare_chain
from merlin.verify.loop_transform_intake import file_ref


def check_cut(cut):
    for row in json.loads(cut.read_text())["sources"]:
        actual = file_ref(repo_root() / row["path"])
        if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
            raise ValueError("frozen Q source changed")


def compiler_verify(files, compiler, output):
    worker = repo_root() / "merlin/tests/ir/buffer_reuse_qualification_worker.py"
    status, stdout = run([compiler, worker, "verify"], output, input=json.dumps({"files": list(map(str, files))}))
    if status["returncode"] != 0:
        raise RuntimeError("compiler verifier unavailable; raw first result retained")
    document = json.loads(stdout)["result"]
    if not all(row["valid"] for row in document["files"]):
        raise ValueError("fixture failed authoritative MLIR verification")
    return document


def write_candidate(folder, candidate):
    save(folder / "PREPARATION.json", candidate.describe())
    (folder / "CanonicalBufferQuery.lean").write_text(candidate.query)
    save(folder / "HINT.json", json.loads(candidate.hint_json))
    save(folder / "ROOTS.json", candidate.roots)
    save(folder / "PARTS.json", json.loads(candidate.query_parts_json))
    for part in json.loads(candidate.query_parts_json):
        (folder / (part["module"] + ".lean")).write_text(part["source"])


def make_corpus(args):
    from buffer_reuse_qualification_cases import VARIANTS, mutations

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cut = args.q_source_cut.resolve()
    check_cut(cut)
    worker = repo_root() / "merlin/tests/ir/buffer_reuse_qualification_worker.py"
    save(
        output / "SOURCE-CUT.json",
        {
            "Q_cut": file_ref(cut),
            "F_sources": [
                file_ref(Path(__file__)),
                file_ref(worker),
                file_ref(repo_root() / "merlin/tests/ir/buffer_reuse_qualification_cases.py"),
                file_ref(repo_root() / "merlin/tests/ir/loop_transform_qualification.py"),
            ],
            "status": "SOURCE_PREPARATION_ONLY_NO_KERNEL_AUTHORITY",
        },
    )
    first = (cut.parent / "raw/stage0.mlir").read_text()
    rows = [("retained_two_epochs", first, 2, {"exact_retained": True})]
    rows.extend((case.name, case.source, case.tile, dataclasses.asdict(case)) for case in VARIANTS)
    expected = [{"name": name, "tile": tile, "metadata": metadata} for name, _, tile, metadata in rows]
    save(output / "EXPECTED-POSITIVES.json", expected)
    corpus, files = [], []
    for name, source, tile, metadata in rows:
        folder = output / "positives" / name
        folder.mkdir(parents=True)
        (folder / "original-input.mlir").write_text(source)
        status, stdout = run(
            [sys.executable, worker, "produce"],
            folder / "producer",
            environment={**os.environ, "PYTHONPATH": str(args.reference.resolve() / "src")},
            input=json.dumps({"source": source, "reference": str(args.reference.resolve()), "tile": tile}),
        )
        if status["returncode"] != 0:
            corpus.append({"name": name, "status": "PRODUCER_UNAVAILABLE", "process": status})
            save(output / "POSITIVES.json", corpus)
            continue
        capture = json.loads(stdout)["result"]
        buffers = tuple(capture.pop("buffers"))
        for index, text in enumerate(buffers):
            path = folder / f"stage{index}.mlir"
            path.write_text(text)
            files.append(path)
        save(folder / "PRODUCER.json", capture)
        if not all(capture["changes"].values()):
            corpus.append({"name": name, "status": "NOT_THREE_CHANGED_BOUNDARIES", "changes": capture["changes"]})
            save(output / "POSITIVES.json", corpus)
            continue
        try:
            candidate = prepare_chain(buffers, allocator=ALLOCATOR)
        except ValueError as error:
            corpus.append({"name": name, "status": "HOST_DOMAIN_REFUSAL", "reason": str(error)})
        else:
            write_candidate(folder, candidate)
            corpus.append(
                {
                    "name": name,
                    "status": "PREPARED_UNCOMPILED",
                    "roots": len(candidate.roots),
                    "parts": len(json.loads(candidate.query_parts_json)),
                    "metadata": metadata,
                }
            )
        save(output / "POSITIVES.json", corpus)
    compiler_verify(files, args.compiler_python, output / "positive-compiler-verifier")
    base = output / "positives/retained_two_epochs"
    buffers = tuple((base / f"stage{i}.mlir").read_text() for i in range(4))
    baseline = prepare_chain(buffers, allocator=ALLOCATOR)
    hint = json.loads(baseline.hint_json)
    negative = mutations(buffers)
    results, files = [], []
    for label, changed in negative.items():
        folder = output / "negatives" / label
        folder.mkdir(parents=True)
        changed_stages = [i for i in range(4) if changed[i] != buffers[i]]
        for index, text in enumerate(changed):
            path = folder / f"stage{index}.mlir"
            path.write_text(text)
            files.append(path)
        row = {"name": label, "changed_stages": changed_stages, "status": "PREPARED_UNCOMPILED"}
        try:
            candidate = prepare_chain(changed, allocator=ALLOCATOR)
        except ValueError as error:
            row["raw4_host_refusal"] = str(error)
            if not set(changed_stages) <= {0, 3}:
                row["status"] = "HOST_INTERMEDIATE_JOIN_REFUSAL"
                results.append(row)
                save(output / "NEGATIVES.json", results)
                continue
            try:
                candidate = prepare(changed[0], changed[3], hint, allocator=ALLOCATOR)
                row["component_route"] = "actual_endpoints_with_baseline_UNTRUSTED_certificate"
            except ValueError as error:
                row["status"] = "HOST_ENDPOINT_OR_CERTIFICATE_REFUSAL"
                row["component_host_refusal"] = str(error)
                results.append(row)
                save(output / "NEGATIVES.json", results)
                continue
        else:
            row["component_route"] = "fresh_actual4_reconstruction"
        write_candidate(folder, candidate)
        parts, baseline_parts = json.loads(candidate.query_parts_json), json.loads(baseline.query_parts_json)
        row["changed_parts"] = [
            a["module"] for a, b in zip(parts, baseline_parts, strict=True) if a["source"] != b["source"]
        ]
        if not row["changed_parts"]:
            row["status"] = "NO_CHANGED_COMPONENT_OBLIGATION"
        results.append(row)
        save(output / "NEGATIVES.json", results)
    compiler_verify(files, args.compiler_python, output / "negative-compiler-verifier")
    check_cut(cut)
    save(
        output / "RESULT.json",
        {
            "status": "PREPARED_UNCOMPILED_UNQUALIFIED",
            "expected_positives": len(expected),
            "positive_preparations": corpus,
            "valid_mutant_triples": len(results),
            "mutation_roster": results,
            "kernel_executed": False,
            "human_approved": False,
            "producer_attribution": "independent F actual pinned capture; not supplied by Q intake",
        },
    )
    print(
        json.dumps({"status": "PREPARED_UNCOMPILED_UNQUALIFIED", "positives": len(corpus), "mutations": len(results)})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--compiler-python", required=True, type=Path)
    parser.add_argument("--q-source-cut", required=True, type=Path)
    args = parser.parse_args()
    make_corpus(args)


if __name__ == "__main__":
    main()
