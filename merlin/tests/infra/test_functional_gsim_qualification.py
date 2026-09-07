"""Offline tests for the resume-safe prelaunch functional GSIM certificate producer."""
from __future__ import annotations

import importlib.util
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir


SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
SOURCE = SCRIPTS / "functional_gsim_qualification.py"
SPEC = importlib.util.spec_from_file_location("functional_gsim_qualification_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
QUAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUAL
SPEC.loader.exec_module(QUAL)


def _capsule(root: Path, name: str, *, m: int, n: int, k: int) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    document = {
        "name": name,
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
            {"name": "X", "role": "input", "shape": [m, k], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {
            "lhs": "X", "weight": "W", "out": "Y0", "epilogue": [],
            "output_dtype": "i32"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }
    path = directory / "capsule.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _functional_capsule(path: Path, *, kind: str = "op"):
    workload = QUAL.PRODUCER.derive_workload(path)
    return QUAL.ORCH.FunctionalCapsule(
        name=path.parent.name, kind=kind, manifest=path.resolve(),
        manifest_sha256=QUAL._sha_file(path),
        workload_sha256=QUAL.GATE.workload_sha256(workload))


def _capture(path: Path, pins: dict[str, dict[str, str]]) -> dict:
    workload = QUAL.PRODUCER.derive_workload(path)
    identity = QUAL.GATE.workload_sha256(workload)
    output = identity[1:] + identity[:1]
    elf = identity[2:] + identity[:2]
    tensors = [{"name": "Y0", "shape": [1], "dtype": "i32", "n_bytes": 4,
                "sha256": output}]
    common = {"ran": True, "verdict": "pass", "elf_sha256": elf,
              "derived_from_rtl": True, "cycle_accurate": True,
              "output_sha256": output, "output_encoding": QUAL.PRODUCER.OUTPUT_ENCODING,
              "output_tensors": tensors}
    return {
        "schema_version": QUAL.PRODUCER.CAPTURE_SCHEMA, "target": "gemmini",
        "capsule": path.parent.name, "capsule_manifest_path": str(path.resolve()),
        "capsule_manifest_sha256": QUAL._sha_file(path), "workload": workload,
        "workload_sha256": identity, "elf_sha256": elf, "agreement": "AGREE",
        "evidence": QUAL.GATE.STRONG_EVIDENCE, "bytes_match": True,
        "reference": {**common, "engine": "verilator",
                      "binary_sha256": pins["verilator_binary"]["sha256"],
                      "firrtl_sha256": pins["verilator_firrtl"]["sha256"]},
        "candidate": {**common, "engine": "gsim",
                      "binary_sha256": pins["gsim_binary"]["sha256"],
                      "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
                      "model_sha256": pins["gsim_model"]["sha256"]},
    }


def _source_certificate(tmp_path: Path, manifests: list[Path]):
    model = tmp_path / "model"
    model.mkdir()
    (model / "TestHarness.h").write_text("generated model\n", encoding="utf-8")
    model_manifest = QUAL.PRODUCER.write_model_manifest(
        model, ["TestHarness.h"], tmp_path / "model-manifest.json")
    files = {}
    for name in ("gsim_firrtl", "verilator_firrtl", "gsim_binary", "verilator_binary",
                 "emitter", "wrapper", "compiler", "harness"):
        files[name] = tmp_path / name
        files[name].write_text(name + "\n", encoding="utf-8")
    artifacts = QUAL.PRODUCER.ArtifactPaths(
        files["gsim_firrtl"], files["verilator_firrtl"], model_manifest,
        files["gsim_binary"], files["verilator_binary"])
    receipt = QUAL.PRODUCER.write_build_receipt(
        output=tmp_path / "build-receipt.json", firrtl=files["gsim_firrtl"],
        model_manifest=model_manifest, binary=files["gsim_binary"], emitter=files["emitter"],
        cxx_wrapper=files["wrapper"], cxx_compiler=files["compiler"],
        inputs=[("harness", files["harness"])], commands=[
            {"stage": "elaborate", "cwd": str(tmp_path), "argv": ["elaborate"]},
            {"stage": "emit", "cwd": str(tmp_path), "argv": ["emit"]},
            {"stage": "compile", "cwd": str(tmp_path), "argv": ["compile"]},
            {"stage": "link", "cwd": str(tmp_path), "argv": ["link"]},
        ])
    pins = artifacts.pinned()
    capture_paths = []
    for index, manifest in enumerate(manifests):
        path = tmp_path / f"source-capture-{index}.json"
        path.write_text(json.dumps(_capture(manifest, pins)), encoding="utf-8")
        capture_paths.append(path)
    document = QUAL.PRODUCER.produce_certificate(
        target="gemmini", captures=capture_paths, artifacts=artifacts, build_receipt=receipt)
    path = tmp_path / "source-certificate.json"
    path.write_text(QUAL.GATE.canonical_json(document) + "\n", encoding="utf-8")
    return path, QUAL._sha_file(path)


def _baseline(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "functional-baseline"
    root.mkdir()
    (root / "compiler.py").write_text("# frozen\n", encoding="utf-8")
    digest = str(hash_tree(root)["sha256"])
    (root / "compiler.py").chmod(0o400)
    root.chmod(0o500)
    return root, digest


def _descriptor(tmp_path: Path) -> Path:
    path = tmp_path / "target.yaml"
    path.write_text("target: gemmini\n", encoding="utf-8")
    return path


def test_cases_fold_duplicate_public_hidden_workloads_deterministically(tmp_path: Path) -> None:
    first = _capsule(tmp_path, "z_public", m=16, n=16, k=17)
    duplicate = _capsule(tmp_path, "a_hidden", m=16, n=16, k=17)
    distinct = _capsule(tmp_path, "b_hidden", m=31, n=17, k=9)
    cohort = QUAL.ORCH.FunctionalGradeCohort(
        public=(_functional_capsule(first),),
        hidden=(_functional_capsule(distinct), _functional_capsule(duplicate)),
        public_source_count=1, hidden_source_count=2)

    cases = QUAL.derive_cases(cohort)

    assert len(cases) == 2
    folded = next(case for case in cases if len(case.capsule_names) == 2)
    assert folded.capsule_names == ("a_hidden", "z_public")
    assert folded.cohorts == ("hidden", "public")
    assert folded.manifest == duplicate.resolve()  # lexical path, not discovery order


def test_cases_leave_whole_models_to_dynamic_gsim_regrade(tmp_path: Path) -> None:
    operation = _capsule(tmp_path, "operation", m=16, n=16, k=16)
    model = _capsule(tmp_path, "whole_model", m=31, n=17, k=9)
    cohort = QUAL.ORCH.FunctionalGradeCohort(
        public=(_functional_capsule(operation), _functional_capsule(model, kind="model")),
        hidden=(), public_source_count=2, hidden_source_count=0)

    cases = QUAL.derive_cases(cohort)

    assert [case.capsule_names for case in cases] == [("operation",)]


def test_exact_certificate_reuses_overlap_filters_extra_and_captures_missing(tmp_path: Path) -> None:
    member = _capsule(tmp_path / "capsules", "member", m=16, n=16, k=16)
    missing = _capsule(tmp_path / "capsules", "missing", m=31, n=17, k=9)
    extra = _capsule(tmp_path / "capsules", "source_extra", m=64, n=64, k=64)
    source, source_sha = _source_certificate(tmp_path, [member, extra])
    baseline, baseline_sha = _baseline(tmp_path)
    descriptor = _descriptor(tmp_path)
    cohort = QUAL.ORCH.FunctionalGradeCohort(
        public=(_functional_capsule(member),), hidden=(_functional_capsule(missing),),
        public_source_count=1, hidden_source_count=1)
    lowered = []

    def lowerer(base, case, output, timeout):
        lowered.append(case.identity)
        output.mkdir()
        (output / "command_buffer.json").write_text("{}\n", encoding="utf-8")
        (output / "lowered.llvm.mlir").write_text("module {}\n", encoding="utf-8")
        return output

    source_record = QUAL.GATE.load_certificate(source, expected_sha256=source_sha)

    def capturer(**kwargs):
        return _capture(Path(kwargs["capsule_manifest"]), source_record.pins)

    certificate, digest = QUAL.produce_functional_certificate(
        descriptor=descriptor, functional_base=baseline, functional_base_sha256=baseline_sha,
        source_certificate=source, source_certificate_sha256=source_sha,
        root=tmp_path / "qualification", timeout=19, workers=2,
        target_experiment=SimpleNamespace(target="gemmini"), cohort=cohort,
        lowerer=lowerer, capturer=capturer, backend=object())

    record = QUAL.GATE.load_certificate(certificate, expected_sha256=digest)
    expected = {capsule.workload_sha256 for capsule in (*cohort.public, *cohort.hidden)}
    assert set(record.members) == expected
    assert lowered == [_functional_capsule(missing).workload_sha256]
    assert _functional_capsule(extra).workload_sha256 not in record.members
    assert len(list((tmp_path / "qualification/captures").glob("seed.*.json"))) == 1


def test_failed_attempt_is_retained_and_resume_uses_fresh_attempt(tmp_path: Path) -> None:
    member = _capsule(tmp_path / "capsules", "member", m=17, n=16, k=15)
    source_member = _capsule(tmp_path / "capsules", "source", m=16, n=16, k=16)
    source, source_sha = _source_certificate(tmp_path, [source_member])
    baseline, baseline_sha = _baseline(tmp_path)
    descriptor = _descriptor(tmp_path)
    capsule = _functional_capsule(member)
    cohort = QUAL.ORCH.FunctionalGradeCohort(
        public=(capsule,), hidden=(), public_source_count=1, hidden_source_count=0)
    source_record = QUAL.GATE.load_certificate(source, expected_sha256=source_sha)
    calls = 0

    def lowerer(base, case, output, timeout):
        output.mkdir()
        (output / "command_buffer.json").write_text("{}\n", encoding="utf-8")
        (output / "lowered.llvm.mlir").write_text("module {}\n", encoding="utf-8")
        return output

    def capturer(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulator interrupted")
        return _capture(Path(kwargs["capsule_manifest"]), source_record.pins)

    arguments = dict(
        descriptor=descriptor, functional_base=baseline, functional_base_sha256=baseline_sha,
        source_certificate=source, source_certificate_sha256=source_sha,
        root=tmp_path / "qualification", timeout=19, workers=1,
        target_experiment=SimpleNamespace(target="gemmini"), cohort=cohort,
        lowerer=lowerer, capturer=capturer, backend=object())
    with pytest.raises(QUAL.FunctionalQualificationError, match="attempts were retained"):
        QUAL.produce_functional_certificate(**arguments)

    declaration = json.loads(next((tmp_path / "qualification").glob(
        "declaration.*.json")).read_text())
    frozen_descriptor = Path(declaration["target_descriptor"]["path"])
    frozen_manifest = Path(declaration["cases"][0]["representative_manifest"])
    assert frozen_descriptor.is_relative_to(tmp_path / "qualification/inputs")
    assert frozen_manifest.is_relative_to(tmp_path / "qualification/inputs/cases")
    assert frozen_manifest.name == "capsule.yaml"
    assert frozen_manifest.parent.name == capsule.workload_sha256

    # The checkout may move between attempts.  Resume is bound to the content-addressed copies in
    # the evidence root and must neither consult nor require the original paths.
    descriptor.unlink()
    member.write_text("name: mutated-after-snapshot\n", encoding="utf-8")
    certificate, digest = QUAL.produce_functional_certificate(**arguments)

    attempts = sorted((tmp_path / "qualification/attempts" / capsule.workload_sha256).iterdir())
    assert [path.name for path in attempts] == ["attempt-000", "attempt-001"]
    assert len(list(attempts[0].glob("failure.*.json"))) == 1
    assert QUAL.GATE.load_certificate(certificate, expected_sha256=digest).members.keys() == {
        capsule.workload_sha256}
    assert not (tmp_path / "qualification").stat().st_mode & 0o222
    assert not frozen_descriptor.stat().st_mode & 0o222
    assert not frozen_manifest.stat().st_mode & 0o222


def _sample_cohort(tmp_path):
    paths = [_capsule(tmp_path / "capsules", name, m=16, n=16, k=k)
             for name, k in (("small", 16), ("large", 256), ("relu", 32))]
    relu = yaml.safe_load(paths[2].read_text())
    relu["operation"]["attributes"]["epilogue"] = ["relu"]
    paths[2].write_text(yaml.safe_dump(relu))
    caps = tuple(_functional_capsule(path) for path in paths)
    return paths, QUAL.ORCH.FunctionalGradeCohort(caps[:2], caps[2:], 2, 1)


def test_stratified_certificate_seals_selection_before_captures_and_resumes(tmp_path):
    paths, cohort = _sample_cohort(tmp_path)
    foreign = _capsule(tmp_path / "capsules", "source", m=1, n=1, k=1)
    source, source_sha = _source_certificate(tmp_path, [foreign])
    baseline, baseline_sha = _baseline(tmp_path)
    descriptor = _descriptor(tmp_path)
    root = tmp_path / "sample"
    source_record = QUAL.GATE.load_certificate(source, expected_sha256=source_sha)
    captured = []

    def lowerer(base, case, output, timeout):
        output.mkdir()
        return output

    def capturer(**kwargs):
        declaration = json.loads(next(root.glob("declaration.*.json")).read_text())
        manifest = Path(kwargs["capsule_manifest"])
        identity = _functional_capsule(manifest).workload_sha256
        assert identity in declaration["functional_coverage"]["selected"]
        captured.append(identity)
        return _capture(manifest, source_record.pins)

    arguments = dict(
        descriptor=descriptor, functional_base=baseline, functional_base_sha256=baseline_sha,
        source_certificate=source, source_certificate_sha256=source_sha, root=root,
        coverage="stratified", target_experiment=SimpleNamespace(target="gemmini"),
        cohort=cohort, lowerer=lowerer, capturer=capturer, backend=object())
    path, digest = QUAL.produce_functional_certificate(**arguments)
    record = QUAL.GATE.load_certificate(path, expected_sha256=digest)
    expected = {_functional_capsule(p).workload_sha256 for p in (paths[0], paths[2])}
    assert set(captured) == expected == set(record.members)
    coverage = QUAL.ORCH._verify_functional_certificate(record, cohort)
    assert coverage["coverage_mode"] == "stratified"
    assert coverage["unsampled_workload_sha256"] == [_functional_capsule(paths[1]).workload_sha256]
    QUAL.ORCH._verify_functional_certificate_provenance(record, source_record, baseline_sha)
    assert QUAL.produce_functional_certificate(**arguments) == (path, digest)
    assert len(captured) == 2
    with pytest.raises(QUAL.FunctionalQualificationError, match="sealed declaration"):
        QUAL.produce_functional_certificate(**{**arguments, "coverage": "exact"})


@pytest.mark.parametrize("mutation", ["missing_stratum", "foreign", "selection", "stratum", "legacy"])
def test_stratified_consumer_refuses_mutated_coverage(tmp_path, mutation):
    paths, cohort = _sample_cohort(tmp_path)
    cases = QUAL.derive_cases(cohort)
    coverage = QUAL.COVERAGE.derive([(case.identity, case.manifest) for case in cases])
    members = dict.fromkeys(coverage["selected"], {})
    document = {"functional_coverage": copy.deepcopy(coverage)}
    if mutation == "missing_stratum":
        members.pop(_functional_capsule(paths[2]).workload_sha256)
    elif mutation == "foreign":
        members["f" * 64] = {}
    elif mutation == "selection":
        document["functional_coverage"]["selected"][0] = "e" * 64
    elif mutation == "stratum":
        document["functional_coverage"]["strata"][0]["semantics"] = {"forged": True}
    else:
        document.clear()  # An old certificate cannot silently gain sampled semantics.
    record = SimpleNamespace(members=members, document=document, pins={})
    with pytest.raises(QUAL.ORCH.ExperimentError):
        QUAL.ORCH._verify_functional_certificate(record, cohort)


def test_sample_cost_ranking_is_pinned_and_partial_strata_use_size(tmp_path):
    paths, cohort = _sample_cohort(tmp_path)
    cases = [(case.identity, case.manifest) for case in QUAL.derive_cases(cohort)]
    small, large, relu = [_functional_capsule(p).workload_sha256 for p in paths]
    pins = {name: {"sha256": "a" * 64} for name in QUAL.GATE.REQUIRED_PINS}
    model = {"engine": "verilator", "pins": {k: v["sha256"] for k, v in pins.items()},
             "seconds": {small: 20, large: 10}}
    sample = QUAL.COVERAGE.derive(cases, reference_cost_model=model, pins=pins)
    assert set(sample["selected"]) == {large, relu}
    assert QUAL.COVERAGE.derive(list(reversed(cases)), reference_cost_model=model, pins=pins) == sample
    del model["seconds"][small]
    assert set(QUAL.COVERAGE.derive(cases, reference_cost_model=model, pins=pins)["selected"]) == {small, relu}
    model["pins"]["gsim_binary"] = "b" * 64
    with pytest.raises(QUAL.COVERAGE.CoverageError, match="pins differ"):
        QUAL.COVERAGE.derive(cases, reference_cost_model=model, pins=pins)
