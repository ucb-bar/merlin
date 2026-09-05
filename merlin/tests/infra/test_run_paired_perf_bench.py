"""Offline tests for the candidate-aware paired performance runner."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(_SCRIPTS))
_SOURCE = _SCRIPTS / "run_paired_perf_bench.py"
_SPEC = importlib.util.spec_from_file_location("run_paired_perf_bench_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = RUNNER
_SPEC.loader.exec_module(RUNNER)
import produce_gsim_certificate as PRODUCER  # noqa: E402


def _inputs(tmp_path: Path, members: tuple | None = None, *, phase: str = "held_out"):
    descriptor = {
        "operation": {"op": "matmul", "attributes": {
            "lhs": "X", "weight": "W", "out": "Y", "output_dtype": "i32"}},
        "inputs": [{"name": "X", "shape": [16, 16], "dtype": "i8"},
                   {"name": "W", "shape": [16, 16], "dtype": "i8"}],
        "numeric_policy": {"compare": "exact_int"},
    }
    if members is None:
        members = (
            SimpleNamespace(family="shape", capsule="wide", source_dir=tmp_path / "wide",
                            source_sha256="1" * 64, descriptor=descriptor),
            SimpleNamespace(family="reuse", capsule="deep", source_dir=tmp_path / "deep",
                            source_sha256="2" * 64, descriptor=descriptor),
        )
    for member in members:
        member.source_dir.mkdir(parents=True, exist_ok=True)
        (member.source_dir / "capsule.yaml").write_text(
            yaml.safe_dump(member.descriptor), encoding="utf-8")
    workloads = [RUNNER._gsim_workload(member) for member in members]
    certificate = SimpleNamespace(
        sha256="d" * 64, unresolved={}, pins={},
        members={RUNNER.GATE.workload_sha256(workload): {} for workload in workloads})
    return RUNNER.PairedInputs(
        SimpleNamespace(run_id="functional", digest="a" * 64),
        SimpleNamespace(record_sha256="b" * 64),
        SimpleNamespace(capsules=members), phase,
        tmp_path / "baseline", "a" * 64, tmp_path / "candidate", "c" * 64,
        certificate)


def test_pure_plan_is_exact_interleaved_and_deterministic(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first = RUNNER.build_measurement_plan(inputs)
    second = RUNNER.build_measurement_plan(inputs)

    assert first.declaration == second.declaration
    assert first.declaration_sha256 == second.declaration_sha256
    assert len(first.schedule) == 2 * len(RUNNER.REPLICATES) * 2
    for pair_index in range(2 * len(RUNNER.REPLICATES)):
        pair = first.schedule[pair_index * 2:pair_index * 2 + 2]
        assert len({item.pair_id for item in pair}) == 1
        assert [item.arm for item in pair] == (
            ["baseline", "candidate"] if pair_index % 2 == 0
            else ["candidate", "baseline"])
    assert {item.replicate for item in first.schedule} == set(RUNNER.REPLICATES)
    assert len(first.expected) == len(first.schedule) * 2
    assert {identity.simulator for identity in first.expected} == {"spike", "gsim"}
    assert first.declaration["rtl_execution_backends"] == ["gsim"]
    assert first.declaration["timing_authority"] == "gsim"
    assert "verilator" not in json.dumps(first.declaration).lower()


def test_pure_workload_identity_matches_certificate_producer(tmp_path: Path) -> None:
    descriptor = {
        "name": "case", "operation": {"op": "matmul", "attributes": {
            "lhs": "A", "weight": "B", "out": "Y", "semantic": "ignored-name",
            "output_dtype": "i32", "epilogue": []}},
        "inputs": [{"name": "A", "role": "input", "shape": [7, 11], "dtype": "i8"},
                   {"name": "B", "role": "weight", "shape": [11, 13], "dtype": "i8"}],
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }
    manifest = tmp_path / "capsule.yaml"
    manifest.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    member = SimpleNamespace(family="shape", capsule="case", descriptor=descriptor,
                             source_dir=tmp_path)

    assert RUNNER._gsim_workload(member) == PRODUCER.derive_workload(manifest)


def test_content_addressed_raw_store_is_immutable_and_idempotent(tmp_path: Path) -> None:
    store = RUNNER.ContentAddressedRawStore(tmp_path / "raw")
    first = store.put({"answer": 42})
    second = store.put({"answer": 42})
    path = Path(first["path"])

    assert first == second
    assert path.name == f"{first['sha256']}.json"
    assert json.loads(path.read_text()) == {"answer": 42}
    assert path.stat().st_mode & 0o222 == 0


def test_host_only_holdout_reveal_adapter_verifies_exact_manifest(tmp_path: Path) -> None:
    root = tmp_path / "holdout"
    capsule = root / "_perf" / "PKH00_k17"
    capsule.mkdir(parents=True)
    descriptor = {
        "name": capsule.name, "operation": {"op": "matmul", "attributes": {
            "lhs": "X", "weight": "W", "out": "Y", "epilogue": [],
            "output_dtype": "i32"}},
        "inputs": [{"name": "X", "role": "input", "shape": [16, 17], "dtype": "i8"},
                   {"name": "W", "role": "weight", "shape": [17, 16], "dtype": "i8"}],
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }
    (capsule / "capsule.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    tree = RUNNER._holdout_tree_record(root)
    manifest = root / "holdout_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 2, "kind": "generated_performance_holdout_reveal",
        "domain": {"target": "gemmini"},
        "cohorts": {"PK_predictor": {"family": "PK", "member_count": 1}},
        "members": [{"name": capsule.name, "path": f"_perf/{capsule.name}",
                     "family": "PK", "cohort": "PK_predictor",
                     "M": 16, "N": 16, "K": 17}], "corpus": tree,
    }), encoding="utf-8")
    manifest.chmod(0o444)
    (capsule / "capsule.yaml").chmod(0o444)
    capsule.chmod(0o555)
    (root / "_perf").chmod(0o555)
    root.chmod(0o555)
    try:
        loaded = RUNNER._load_holdout_corpus(
            root, manifest, manifest_sha256=RUNNER._sha256_file(manifest),
            capsules_sha256=tree["sha256"], expected_target="gemmini")
        assert loaded.format == "holdout_reveal_v2"
        assert [(row.family, row.capsule) for row in loaded.capsules] == [
            ("PK", "PKH00_k17")]
    finally:
        root.chmod(0o755)
        (root / "_perf").chmod(0o755)
        capsule.chmod(0o755)
        manifest.chmod(0o644)
        (capsule / "capsule.yaml").chmod(0o644)


def test_l3_adapter_uses_only_pinned_gsim_and_never_resolves_verilator(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"one exact executable")
    def primary(*_args, **_kwargs):
        return {"elf": str(elf), "outputs": {"y": [7]}, "cycles": 123,
                "oracle": {"kind": "rtl_gsim", "derived_from_rtl": True}}

    class Backend:
        @staticmethod
        def gsim_path():
            return gsim_binary

        @staticmethod
        def verilator_path():
            raise AssertionError("final measurement must not resolve Verilator")

    from merlin.runtime.backends import base as backends
    gsim_binary = tmp_path / "gsim"
    gsim_binary.write_bytes(b"gsim")
    certificate = SimpleNamespace(pins={
        "gsim_binary": {"sha256": RUNNER._sha256_file(gsim_binary)},
        "gsim_firrtl": {"sha256": "1" * 64},
        "gsim_model": {"sha256": "3" * 64},
    })
    monkeypatch.setattr(RUNNER.OOT, "run_on_oracle", primary)
    monkeypatch.setattr(backends, "get_backend", lambda _target: Backend())
    evidence = {}
    result = RUNNER._gsim_l3_adapter("gemmini", evidence, certificate)(
        {"tensors": {"y": {"role": "output", "shape": [1], "dtype": "i32"}}},
        "module", tmp_path / "work", 9)

    assert result["cycles"] == 123
    assert set(evidence) == {"gsim"}
    assert evidence["gsim"]["output_encoding"] == RUNNER.CERTPROD.OUTPUT_ENCODING


def _mock_measurement(spec, cycles: int) -> dict:
    digest = str(spec.execution_index % 10) * 64
    per_sim = {
        "spike": {"correct": True, "cycles": None, "provenance": None},
        "gsim": {"correct": True, "cycles": cycles,
                 "provenance": {"tier": "L3", "simulator": "gsim",
                                "oracle_kind": "rtl_gsim", "derived_from_rtl": True,
                                "cycle_accurate": True, "elf_sha256": digest}},
    }
    return {"status": "pass", "numeric": "pass", "per_sim": per_sim,
            "gsim_qualification": {"admitted": True},
            "work_volume": {}, "command_buffer_artifact": None}


def test_mocked_execution_has_fresh_workspaces_primary_gsim_and_exact_denominator(
        tmp_path: Path) -> None:
    member = SimpleNamespace(family="shape", capsule="wide", source_dir=tmp_path / "wide",
                             source_sha256="1" * 64,
                             descriptor={"operation": {"op": "movement", "attributes": {
                                             "src": "X", "out": "Y"}},
                                         "inputs": [{"name": "X", "shape": [1], "dtype": "i8"}],
                                         "numeric_policy": {"compare": "exact_int"}})
    plan = RUNNER.build_measurement_plan(_inputs(tmp_path, (member,)))
    out = tmp_path / "out"
    out.mkdir()
    workspaces = []

    def executor(spec, workspace, *_args, **_kwargs):
        assert not workspace.exists()
        workspace.mkdir(parents=True)
        workspaces.append(workspace)
        cycles = 100 + spec.pair_index
        return {"schema": "mock", "execution": spec.as_dict(),
                "measurement": _mock_measurement(spec, cycles)}

    rows, _ = RUNNER.execute_schedule(
        plan, out, timeout=1, target_experiment=object(), rtl_identity={},
        hardware_counters=False, executor=executor, progress=lambda _line: None)

    # one fresh workspace per (member, replicate, arm) -- derived, so the replicate count can move
    assert len(workspaces) == 2 * len(RUNNER.REPLICATES) and len(set(workspaces)) == len(workspaces)
    # one Spike correctness screen and one GSIM measurement per (member, replicate, arm)
    assert len(rows) == 2 * len(RUNNER.REPLICATES) * 2
    assert {row["simulator"] for row in rows} == {"spike", "gsim"}
    assert all(row["cycles"] is None and not row["citable"]
               for row in rows if row["simulator"] == "spike")
    assert RUNNER.completion_report(rows, plan.expected)["complete"] is True
    pairs = RUNNER.paired_cycle_rows(rows)
    assert (len(pairs) == len(RUNNER.REPLICATES)
            and all(row["simulator"] == "gsim" for row in pairs))
    raw_files = list((out / "raw_results/sha256").glob("*.json"))
    assert len(raw_files) == 2 * len(RUNNER.REPLICATES)
    assert all(path.stem == RUNNER._sha256_file(path) for path in raw_files)


def test_completion_refuses_non_rtl_primary(
        tmp_path: Path) -> None:
    plan = RUNNER.build_measurement_plan(_inputs(tmp_path, (
        SimpleNamespace(family="f", capsule="c", source_dir=tmp_path / "c",
                        source_sha256="1" * 64,
                        descriptor={"operation": {"op": "movement", "attributes": {"src": "X"}},
                                    "inputs": [{"name": "X", "shape": [1], "dtype": "i8"}],
                                    "numeric_policy": {"compare": "exact_int"}}),)))
    rows = []
    for spec in plan.schedule:
        raw = {"execution": spec.as_dict(), "measurement": _mock_measurement(spec, 100)}
        rows.extend(RUNNER.result_rows(raw, {"sha256": "f" * 64, "path": "/raw"}))
    primary = next(row for row in rows if row["simulator"] == "gsim")
    primary["provenance"]["derived_from_rtl"] = False
    assert RUNNER.completion_report(rows, plan.expected)["complete"] is False


def test_final_plan_has_no_verilator_sampling_api(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        RUNNER.build_measurement_plan(_inputs(tmp_path), verilator_sample_per_family=1)
