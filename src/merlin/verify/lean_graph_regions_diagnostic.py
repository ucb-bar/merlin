"""Bounded diagnosis of saved/fresh native evidence, without requalification.

Compares actual raw buffers first, then recomputes the production execution
context from each original selected-file location. No saved capsule is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def differences(first, second, path=""):
    if type(first) is not type(second):
        return [dict(field=path, first=first, second=second)]
    if type(first) is dict:
        if set(first) != set(second):
            return [dict(field=path, first_keys=sorted(first), second_keys=sorted(second))]
        return [row for key in sorted(first) for row in differences(first[key], second[key], path + "/" + key)]
    if type(first) is list:
        if len(first) != len(second):
            return [dict(field=path, first_length=len(first), second_length=len(second))]
        return [
            row
            for i, (a, b) in enumerate(zip(first, second, strict=True))
            for row in differences(a, b, path + "/" + str(i))
        ]
    return [] if first == second else [dict(field=path, first=first, second=second)]


def context_worker(original: Path, fixture: Path, destination: Path):
    from merlin.compile.scheduling.fusion_semantics import term
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

    spec = importlib.util.spec_from_file_location("pinned_context_fixture", fixture)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    destination.mkdir(exist_ok=False)
    _, _, options = helper.model(destination, width=16, shape=(3, 5))
    scope = json.loads(original.read_text())
    origin = Path(scope["source"]["path"]).parent / "selected-fixture"
    selected = read_scheduling_contracts(origin / "facts.json", origin / "software.yaml", target="neutral")
    selected.revalidate()
    _, _, _, effective = selected.storage.bind(
        selected.isa,
        options["adapter"],
        banks=None,
        reservations=(),
        external_windows=None,
        private_ownership=True,
        configuration_adapter=None,
    )
    _, _, scalar = selected.fusion.bind(
        effective,
        emitters={},
        qualification_identity=options["composition_law"].qualification_identity,
        max_operations=options["budgets"].max_scalar_operations,
        max_ir_bytes=options["budgets"].max_scalar_ir_bytes,
    )
    law = asdict(options["composition_law"])
    record = selected.record()
    execution = term((law, scalar, term(record)))
    actual = [
        row[side]["context"]["execution"]
        for row in scope["realization"]["source_passes"]
        for side in ("before", "after")
    ]
    if not actual or any(value != execution for value in actual):
        raise ValueError("recomputed original execution context does not match saved production fields")
    return dict(
        selected_record=record,
        law=law,
        scalar_binding=scalar,
        execution=execution,
        all_saved_contexts_match=True,
        semantic_contract_truth_proved=False,
    )


def diagnose(first: Path, second: Path, destination: Path):
    from merlin.verify import lean_graph_regions as base

    destination.mkdir(parents=True, exist_ok=False)
    records = [base.file_ref(path) for path in (first, second)]
    scopes = [json.loads(base.fresh_ref(record).read_text()) for record in records]
    artifact_diffs, comparisons = [], []
    if set(scopes[0]["artifacts"]) != set(scopes[1]["artifacts"]):
        raise ValueError("native artifact domains differ")
    for name in sorted(scopes[0]["artifacts"]):
        left, right = [base.fresh_ref(scope["artifacts"][name]).read_bytes() for scope in scopes]
        same = left == right
        comparisons.append(dict(artifact=name, bytes=len(left), raw_bytes_identical=same))
        if not same:
            offset = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            artifact_diffs.append(
                dict(
                    artifact=name,
                    first_differing_offset=offset,
                    first=left[offset : offset + 32].hex(),
                    second=right[offset : offset + 32].hex(),
                )
            )
    metadata = differences(*(base.content_identity(scope) for scope in scopes))
    contexts = []
    for index, (path, scope) in enumerate(zip((first, second), scopes, strict=True)):
        reference = Path(scope["production_root"])
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(reference / "src"), PYTHONDONTWRITEBYTECODE="1")
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            str(path.resolve()),
            "--fixture",
            str(reference / "merlin/tests/infra/test_multioutput_fusion.py"),
            "--destination",
            str((destination / f"context-{index}").resolve()),
        ]
        process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
        (destination / f"context-{index}-process.json").write_text(
            json.dumps(
                dict(command=command, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr),
                indent=2,
            )
            + "\n"
        )
        process.check_returncode()
        contexts.append(json.loads(process.stdout))
    result = dict(
        originals=records,
        native_raw_buffer_comparisons=comparisons,
        raw_buffer_differences=artifact_diffs,
        first_metadata_divergence=metadata[:1],
        all_metadata_divergences=metadata,
        recomputed_contexts=contexts,
        selected_record_differences=differences(*(row["selected_record"] for row in contexts)),
        qualification_status="UNQUALIFIED: diagnostic only; no gate bypass",
    )
    (destination / "DIAGNOSTIC.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(context_worker(args.worker, args.fixture, args.destination)))
    else:
        result = diagnose(args.first, args.second, args.destination)
        print(
            json.dumps(
                dict(
                    first_divergence=result["first_metadata_divergence"],
                    raw_buffer_differences=result["raw_buffer_differences"],
                )
            )
        )


if __name__ == "__main__":
    main()
