"""Reviewable public call inputs, packaged by the existing retained-call writer.

This development corpus does not alter derived_only release membership. Select its
generated cases and captures explicitly through core_aten_stage for a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


def load_specs(path: Path) -> dict:
    document = yaml.safe_load(path.read_bytes())
    if set(document) != {"schema", "calls"} or document.get("schema") != "merlin.regression_calls.v1":
        raise ValueError("unsupported regression input schema")
    identifiers = set()
    for row in document["calls"]:
        if set(row) != {"case_id", "overload", "arguments", "guards", "boundary", "origin"}:
            raise ValueError("regression calls require inputs, origin, boundary and guards only")
        if row["case_id"] in identifiers or not row["guards"] or not row["boundary"]:
            raise ValueError("regression identifiers must be unique and have obligations")
        identifiers.add(row["case_id"])
    return document


def generate(spec: Path, output: Path, *, m2m_dir: Path, python: Path) -> dict:
    from merlin.common.paths import out_dir, python_import_roots
    from merlin.targetgen.capsule_source import PytorchRefSource
    from merlin.targetgen.core_aten_bounded_runner import bounded_loader_source, bundle_admission_reason
    from merlin.targetgen.core_aten_capsules import write_capsules
    from merlin.targetgen.core_aten_capture import _git_revision, case_capture_name
    from merlin.targetgen.core_aten_cover import _provenance_ops

    output = output.absolute()
    if not output.is_relative_to(out_dir().resolve()) or output != output.resolve() or output.exists():
        raise ValueError("select a fresh ordinary output beneath the configured out root")
    document = load_specs(spec)
    output.mkdir(parents=True)
    output.chmod(0o700)
    (output / "public").mkdir()
    # Torch owns independent answer generation; the controller owns packaging and
    # schema validation. Neither interpreter needs the other's optional stack.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_import_roots())
    observed = subprocess.run(
        [
            str(python),
            "-c",
            """import json,sys
from merlin.targetgen.core_aten_cases import case_document_from_arguments,decode_arguments
cases=[]
for row in json.load(sys.stdin)['calls']:
    args,kwargs=decode_arguments(row['arguments'])
    case=case_document_from_arguments(row['overload'],args,kwargs,rng_seed=0)
    case['case_id']=row['case_id']
    cases.append(case)
print(json.dumps(cases,sort_keys=True))
""",
        ],
        input=json.dumps(document),
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    cases = json.loads(observed.stdout)
    source = PytorchRefSource(m2m_dir=m2m_dir, python=python)
    results, members = [], []
    for row, case in zip(document["calls"], cases, strict=True):
        directory = output / "captures" / case_capture_name(case)
        directory.mkdir(parents=True)
        loader = directory / "capsule.pytorch.py"
        loader.write_text(bounded_loader_source(case))
        result = {"case_id": row["case_id"], "guards": row["guards"]}
        try:
            artifact = source.capture_loader(loader, "fp32", workdir=directory / "capture-work")
            mlir = directory / "capsule.linalg.mlir"
            mlir.write_text(artifact.linalg_mlir)
            capture = {
                "status": "captured_exact",
                "overload": row["overload"],
                "capture_meta": artifact.meta,
                "model2mlir_revision": _git_revision(m2m_dir),
                "captured_overloads": sorted(_provenance_ops(mlir)),
            }
            if row["overload"] not in capture["captured_overloads"]:
                raise ValueError("exact overload provenance lost")
            for filename, value in (
                ("capture.json", capture),
                ("inputs.json", artifact.inputs),
                ("golden.json", artifact.golden),
            ):
                (directory / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            reason = bundle_admission_reason(capture, artifact.linalg_mlir)
            if reason:
                raise ValueError(reason)
            manifest = write_capsules(
                {"cases": [case]}, output / "captures", output / "public", output / "_private" / "public"
            )
            members.extend(manifest["capsules"])
            result.update(status="packaged", capsule=manifest["capsules"][0]["id"])
        except Exception as exc:
            result.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
        results.append(result)
    (output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n")
    (output / "public" / "manifest.json").write_text(
        json.dumps(
            {"schema": "merlin.core_aten_capsules.v1", "label": "public", "count": len(members), "capsules": members},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    receipt = {
        "schema": "merlin.regression_packaging.v1",
        "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "scope": "public development inputs; no numerical or hardware verdict",
        "results": results,
    }
    (output / "packaging.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    from merlin.common.paths import artifacts_dir

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec", type=Path)
    mode.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=artifacts_dir() / "verification" / "regression-calls")
    parser.add_argument("--m2m-dir", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--captures", type=Path)
    args = parser.parse_args()
    if args.selection:
        if not args.cases or not args.captures:
            parser.error("retained selection requires --cases and --captures")
        receipt = generate_selection(args.selection, args.cases, args.captures, args.output)
    else:
        if not args.m2m_dir or not args.python:
            parser.error("fresh capture requires --m2m-dir and --python")
        receipt = generate(args.spec, args.output, m2m_dir=args.m2m_dir, python=args.python)
    print(json.dumps(receipt, indent=2))
    return int(any(row["status"] != "packaged" for row in receipt["results"]))


def generate_selection(spec: Path, cases_path: Path, captures: Path, output: Path) -> dict:
    """Select reviewed calls from an explicitly supplied public retained corpus."""
    from merlin.common.paths import out_dir
    from merlin.targetgen.core_aten_batch import _tensor_documents
    from merlin.targetgen.core_aten_capsules import write_capsules
    from merlin.targetgen.core_aten_capture import case_capture_name

    document = yaml.safe_load(spec.read_bytes())
    if document.get("schema") != "merlin.regression_selection.v1" or document.get("capsule_policy") != "derived_only":
        raise ValueError("retained regression selection must be derived_only")
    corpus = json.loads(cases_path.read_bytes())
    rows = corpus.get("cases", corpus.get("selected_cases", []))
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("retained case identities are ambiguous")
    selected = []
    for row in document["calls"]:
        case = by_id[row["case_id"]]
        inputs = _tensor_documents(case["arguments"])
        metadata = [{key: tensor[key] for key in ("dtype", "shape", "stride", "storage_offset")} for tensor in inputs]
        if metadata != row["inputs"] or case["overload"] != row["overload"]:
            raise ValueError("retained call differs from reviewed input contract")
        if {key: case["expected"][key] for key in ("dtype", "shape")} != row["result"]:
            raise ValueError("retained result type differs from reviewed call contract")
        if case_capture_name(case).replace("-", "_") != row["capsule"]:
            raise ValueError("reviewed capsule identity differs")
        selected.append(case)
    output = output.absolute()
    if output != output.resolve() or not output.is_relative_to(out_dir().resolve()) or output.exists():
        raise ValueError("select a fresh ordinary output beneath the configured out root")
    output.mkdir(parents=True, mode=0o700)
    manifest = write_capsules(
        {**corpus, "cases": selected}, captures, output / "public", output / "_private" / "public"
    )
    # The selected calls guard device compilation, including executed placement.
    # Use the same declaration fields as the receipt-bound cohort stage.
    for member in manifest["capsules"]:
        declaration = output / "public" / member["id"] / "capsule.yaml"
        capsule = yaml.safe_load(declaration.read_bytes())
        capsule.update(lane_expectation="device", semantic={"must_accelerate": True}, lanes={"require": ["on_mesh"]})
        declaration.write_text(yaml.safe_dump(capsule, sort_keys=False))
        member["digests"]["capsule.yaml"] = hashlib.sha256(declaration.read_bytes()).hexdigest()
    (output / "public" / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "merlin.regression_packaging.v1",
        "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "scope": "retained public call packaging; no numerical or hardware verdict",
        "results": [
            {"case_id": row["case_id"], "capsule": row["capsule"], "guards": row["guards"], "status": "packaged"}
            for row in document["calls"]
        ],
        "manifest": manifest,
    }
    (output / "packaging.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
