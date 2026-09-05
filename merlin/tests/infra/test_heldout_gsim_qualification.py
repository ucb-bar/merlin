"""Offline tests for post-seal host-only holdout qualification."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir


_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(_SCRIPTS))
_SOURCE = _SCRIPTS / "heldout_gsim_qualification.py"
_SPEC = importlib.util.spec_from_file_location("heldout_gsim_qualification_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
QUAL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = QUAL
_SPEC.loader.exec_module(QUAL)


def _capsule(root: Path, name: str, *, family: str, cohort: str,
             m: int, n: int, k: int) -> tuple[Path, dict]:
    directory = root / "_perf" / name
    directory.mkdir(parents=True)
    descriptor = {
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
    manifest = directory / "capsule.yaml"
    manifest.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    return manifest, {"name": name, "path": f"_perf/{name}", "family": family,
                      "cohort": cohort, "M": m, "N": n, "K": k}


def _reveal(root: Path) -> tuple[Path, str, str]:
    _, pk = _capsule(root, "PKH00_k17", family="PK", cohort="PK_predictor",
                     m=16, n=16, k=17)
    _, pkg = _capsule(root, "PKG00_m32k31n48", family="PKG",
                      cohort="PK_MNK_generalization", m=32, n=48, k=31)
    manifest = root / "holdout_manifest.json"
    tree = QUAL._tree_without_manifest(root, manifest)
    document = {
        "schema_version": 2, "kind": "generated_performance_holdout_reveal",
        "domain": {"target": "gemmini"},
        "cohorts": {
            "PK_predictor": {"family": "PK", "member_count": 1},
            "PK_MNK_generalization": {"family": "PKG", "member_count": 1},
        },
        "members": [pk, pkg], "corpus": tree,
    }
    manifest.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    manifest_sha = QUAL._sha_file(manifest)
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    return manifest, manifest_sha, tree["sha256"]


def _tuning(tmp_path: Path):
    pins = {}
    for name in sorted(QUAL.GATE.REQUIRED_PINS):
        artifact = tmp_path / name
        artifact.write_text(f"pinned {name}\n", encoding="utf-8")
        pins[name] = {"path": str(artifact.resolve()), "sha256": QUAL._sha_file(artifact)}
    source = tmp_path / "tuning-certificate.json"
    source.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text('{"status":"complete"}\n', encoding="utf-8")
    identity = "a" * 64
    return SimpleNamespace(
        target="gemmini", path=source.resolve(), sha256=QUAL._sha_file(source), pins=pins,
        members={identity: {"workload_sha256": identity}},
        document={"members": [{"workload_sha256": identity}], "unresolved": [],
                  "pins": pins,
                  # Relative on purpose: qualification must rebase it to its validated absolute path.
                  "build_binding": {"path": receipt.name, "sha256": QUAL._sha_file(receipt),
                                    "commands_sha256": "b" * 64}})


def _capture_document(manifest: Path, pins: dict, *, target: str = "gemmini") -> dict:
    workload = QUAL.PRODUCER.derive_workload(manifest)
    identity = QUAL.GATE.workload_sha256(workload)
    output_sha = (identity[1:] + identity[:1])
    output_rows = [{"name": "Y0", "shape": [1], "dtype": "i32", "n_bytes": 4,
                    "sha256": output_sha}]
    elf_sha = (identity[2:] + identity[:2])
    common = {"ran": True, "verdict": "pass", "elf_sha256": elf_sha,
              "derived_from_rtl": True, "cycle_accurate": True,
              "output_sha256": output_sha, "output_encoding": QUAL.PRODUCER.OUTPUT_ENCODING,
              "output_tensors": output_rows}
    return {
        "schema_version": QUAL.PRODUCER.CAPTURE_SCHEMA, "target": target,
        "capsule": manifest.parent.name, "capsule_manifest_path": str(manifest.resolve()),
        "capsule_manifest_sha256": QUAL._sha_file(manifest), "workload": workload,
        "workload_sha256": identity, "elf_sha256": elf_sha, "agreement": "AGREE",
        "evidence": QUAL.GATE.STRONG_EVIDENCE, "bytes_match": True,
        "reference": {**common, "engine": "verilator",
                      "binary_sha256": pins["verilator_binary"]["sha256"],
                      "firrtl_sha256": pins["verilator_firrtl"]["sha256"]},
        "candidate": {**common, "engine": "gsim",
                      "binary_sha256": pins["gsim_binary"]["sha256"],
                      "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
                      "model_sha256": pins["gsim_model"]["sha256"]},
    }


def _valid_tuning(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "ChipTop.h").write_text("generated header\n", encoding="utf-8")
    model_manifest = QUAL.PRODUCER.write_model_manifest(
        model, ["ChipTop.h"], tmp_path / "model-manifest.json")
    artifacts = {}
    for name in ("gsim_firrtl", "verilator_firrtl", "gsim_binary", "verilator_binary"):
        artifacts[name] = tmp_path / name
        artifacts[name].write_text(f"{name}\n", encoding="utf-8")
    paths = QUAL.PRODUCER.ArtifactPaths(
        artifacts["gsim_firrtl"], artifacts["verilator_firrtl"], model_manifest,
        artifacts["gsim_binary"], artifacts["verilator_binary"])
    support = {}
    for name in ("emitter", "wrapper", "compiler", "harness"):
        support[name] = tmp_path / name
        support[name].write_text(f"{name}\n", encoding="utf-8")
    receipt = QUAL.PRODUCER.write_build_receipt(
        output=tmp_path / "build-receipt.json", firrtl=artifacts["gsim_firrtl"],
        model_manifest=model_manifest, binary=artifacts["gsim_binary"],
        emitter=support["emitter"], cxx_wrapper=support["wrapper"],
        cxx_compiler=support["compiler"], inputs=[("harness", support["harness"])],
        commands=[
            {"stage": "elaborate", "cwd": str(tmp_path), "argv": ["elaborate"]},
            {"stage": "emit", "cwd": str(tmp_path), "argv": ["emit"]},
            {"stage": "compile", "cwd": str(tmp_path), "argv": ["compile"]},
            {"stage": "link", "cwd": str(tmp_path), "argv": ["link"]},
        ])
    tuning_root = tmp_path / "tuning"
    manifest, _ = _capsule(tuning_root, "PK00_k16", family="PK", cohort="PK_predictor",
                           m=16, n=16, k=16)
    pins = paths.pinned()
    capture_path = tmp_path / "tuning-capture.json"
    capture_path.write_text(json.dumps(_capture_document(manifest, pins)), encoding="utf-8")
    certificate = QUAL.PRODUCER.produce_certificate(
        target="gemmini", captures=[capture_path], artifacts=paths, build_receipt=receipt)
    certificate_path = tmp_path / "tuning-certificate.valid.json"
    certificate_path.write_text(json.dumps(certificate, sort_keys=True), encoding="utf-8")
    return QUAL.GATE.load_certificate(certificate_path)


def test_reveal_loader_uses_both_manifest_declared_cohorts_and_no_globbed_extra(
        tmp_path: Path) -> None:
    reveal, manifest_sha, corpus_sha = _reveal(tmp_path / "heldout")
    members = QUAL.load_revealed_members(
        reveal, expected_manifest_sha256=manifest_sha,
        expected_corpus_sha256=corpus_sha, expected_target="gemmini")
    assert [(row.family, row.cohort) for row in members] == [
        ("PK", "PK_predictor"), ("PKG", "PK_MNK_generalization")]

    # Even a well-formed capsule is not silently admitted when it is absent from the reveal manifest.
    root = reveal.parent
    root.chmod(0o700)
    (root / "_perf").chmod(0o700)
    _capsule(root, "not_declared", family="PKG", cohort="PK_MNK_generalization",
             m=16, n=16, k=29)
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    with pytest.raises(QUAL.QualificationError, match="committed tree"):
        QUAL.load_revealed_members(reveal, expected_manifest_sha256=manifest_sha)


def test_offline_provider_lowers_frozen_baseline_and_emits_resume_adoptable_extension(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reveal, manifest_sha, corpus_sha = _reveal(tmp_path / "heldout")
    baseline = tmp_path / "functional-baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("# frozen baseline\n", encoding="utf-8")
    baseline_sha = str(hash_tree(baseline)["sha256"])
    (baseline / "compiler.py").chmod(0o400)
    baseline.chmod(0o500)
    tuning = _tuning(tmp_path)
    events = []

    def lowerer(base, member, artifact_dir, timeout):
        assert base == baseline and timeout == 37
        events.append(("lower", member.name, member.family))
        artifact_dir.mkdir()
        (artifact_dir / "command_buffer.json").write_text("{}\n", encoding="utf-8")
        (artifact_dir / "lowered.llvm.mlir").write_text("module {}\n", encoding="utf-8")
        return artifact_dir

    def capturer(**kwargs):
        manifest = Path(kwargs["capsule_manifest"])
        workload = QUAL.PRODUCER.derive_workload(manifest)
        identity = QUAL.GATE.workload_sha256(workload)
        events.append(("capture", manifest.parent.name, kwargs["backend"]))
        return {"workload": workload, "workload_sha256": identity,
                "capsule_manifest_path": str(manifest), "capsule_manifest_sha256": QUAL._sha_file(manifest)}

    monkeypatch.setattr(QUAL.PRODUCER, "validate_capture", lambda *args, **kwargs: {})

    def load_certificate(path, *, expected_sha256):
        assert QUAL._sha_file(Path(path)) == expected_sha256
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        members = {row["workload_sha256"]: row for row in document["members"]}
        return SimpleNamespace(path=Path(path).resolve(), sha256=expected_sha256,
                               target="gemmini", pins=tuning.pins, members=members,
                               document=document)

    monkeypatch.setattr(QUAL.GATE, "load_certificate", load_certificate)
    qualification_root = tmp_path / "host-private" / "qualification"
    qualification_root.parent.mkdir()
    certificate, digest = QUAL.qualify_revealed_holdout(
        reveal, qualification_root, tuning, functional_base=baseline,
        functional_base_sha256=baseline_sha, reveal_manifest_sha256=manifest_sha,
        reveal_corpus_sha256=corpus_sha, timeout=37, lowerer=lowerer,
        capturer=capturer, backend="injected-offline-backend", gsim_max_cycles=100_000_000)

    assert [event[:2] for event in events] == [
        ("lower", "PKH00_k17"), ("capture", "PKH00_k17"),
        ("lower", "PKG00_m32k31n48"), ("capture", "PKG00_m32k31n48")]
    assert certificate.name == f"certificate.{digest}.json"
    assert qualification_root.stat().st_mode & 0o222 == 0
    (qualification_receipt,) = qualification_root.glob("qualification.*.json")
    execution = json.loads(qualification_receipt.read_text(encoding="utf-8"))["execution"]
    assert execution == {"timeout_seconds": 37, "gsim_max_cycles": 100_000_000,
                         "same_elf_engines": ["verilator", "gsim"]}
    adopted = QUAL.load_completed_qualification(
        qualification_root, tuning=tuning, reveal_manifest_sha256=manifest_sha,
        reveal_corpus_sha256=corpus_sha, functional_base_sha256=baseline_sha,
        gsim_max_cycles=100_000_000)
    assert adopted == (certificate, digest)

    with pytest.raises(QUAL.QualificationError, match="differs from this experiment"):
        QUAL.load_completed_qualification(
            qualification_root, tuning=tuning, reveal_manifest_sha256=manifest_sha,
            reveal_corpus_sha256=corpus_sha, functional_base_sha256=baseline_sha,
            gsim_max_cycles=20_000_000)


def test_pinned_runtime_binds_declared_cap_and_restores_ambient_state(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuning = _tuning(tmp_path)

    class Backend:
        def gsim_path(self):
            return tuning.pins["gsim_binary"]["path"]

        def verilator_path(self):
            return tuning.pins["verilator_binary"]["path"]

    from merlin.runtime.backends import base as backends
    monkeypatch.setattr(backends, "get_backend", lambda target: Backend())
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", "/ambient/wrong")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", "17")

    with QUAL._pinned_runtime(tuning, gsim_max_cycles=100_000_000) as selected:
        assert isinstance(selected, Backend)
        assert os.environ["MERLIN_GEMMINI_GSIM_EMU"] \
            == tuning.pins["gsim_binary"]["path"]
        assert os.environ["MERLIN_GEMMINI_GSIM_MAXCYCLES"] == "100000000"

    assert os.environ["MERLIN_GEMMINI_GSIM_EMU"] == "/ambient/wrong"
    assert os.environ["MERLIN_GEMMINI_GSIM_MAXCYCLES"] == "17"

    for invalid in (0, -1, True, "100000000"):
        with pytest.raises(QUAL.QualificationError, match="positive integer or null"):
            with QUAL._pinned_runtime(tuning, gsim_max_cycles=invalid):
                pass


def test_resume_refuses_partial_qualification_root(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    (root / "capture.tmp").write_text("interrupted\n", encoding="utf-8")
    with pytest.raises(QUAL.QualificationError, match="no unique completion receipt"):
        QUAL.load_completed_qualification(
            root, tuning=SimpleNamespace(sha256="a" * 64),
            reveal_manifest_sha256="b" * 64, reveal_corpus_sha256="c" * 64,
            functional_base_sha256="d" * 64, gsim_max_cycles=100_000_000)


def test_extension_merge_is_accepted_by_real_strict_gate(tmp_path: Path) -> None:
    reveal, manifest_sha, corpus_sha = _reveal(tmp_path / "heldout")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("# baseline\n", encoding="utf-8")
    baseline_sha = str(hash_tree(baseline)["sha256"])
    (baseline / "compiler.py").chmod(0o400)
    baseline.chmod(0o500)
    tuning = _valid_tuning(tmp_path)

    def lowerer(_base, _member, artifact_dir, _timeout):
        artifact_dir.mkdir()
        return artifact_dir

    def capturer(**kwargs):
        return _capture_document(Path(kwargs["capsule_manifest"]), tuning.pins)

    qualification_root = tmp_path / "qualification"
    certificate_path, certificate_sha = QUAL.qualify_revealed_holdout(
        reveal, qualification_root, tuning, functional_base=baseline,
        functional_base_sha256=baseline_sha, reveal_manifest_sha256=manifest_sha,
        reveal_corpus_sha256=corpus_sha, timeout=10, lowerer=lowerer,
        capturer=capturer, backend="offline", gsim_max_cycles=100_000_000)
    extension = QUAL.GATE.load_certificate(certificate_path, expected_sha256=certificate_sha)
    revealed = QUAL.load_revealed_members(reveal)
    assert set(extension.members) == set(tuning.members) | {
        member.workload_sha256 for member in revealed}
    assert extension.document["build_binding"]["sha256"] \
        == tuning.document["build_binding"]["sha256"]
