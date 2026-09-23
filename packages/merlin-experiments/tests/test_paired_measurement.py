"""Installed paired scheduling with synthetic measurements, never native engines."""

import importlib
import importlib.abc
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import paired_measurement as PM


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))


def _plan(root):
    capsule = root / "capsule"
    capsule.mkdir()
    descriptor = {
        "operation": {"op": "movement", "attributes": {"src": "X", "out": "Y"}},
        "inputs": [{"name": "X", "shape": [1], "dtype": "i8"}],
        "numeric_policy": {"compare": "exact_int"},
    }
    (capsule / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
    member = SimpleNamespace(
        family="shape",
        capsule="one",
        source_dir=capsule,
        source_sha256=PM.hash_tree(capsule)["sha256"],
        descriptor=descriptor,
    )
    packages = []
    for name in ("baseline", "candidate"):
        package = root / name
        package.mkdir()
        (package / "source.txt").write_text(name)
        packages.append(package)
    workload = PM.gsim_workload(member)
    certificate = SimpleNamespace(
        sha256="d" * 64, unresolved={}, pins={}, members={PM.GATE.workload_sha256(workload): {}}
    )
    inputs = PM.PairedInputs(
        SimpleNamespace(run_id="functional", digest="a" * 64),
        SimpleNamespace(record_sha256="b" * 64),
        SimpleNamespace(capsules=(member,)),
        "held_out",
        packages[0],
        PM.hash_tree(packages[0])["sha256"],
        packages[1],
        PM.hash_tree(packages[1])["sha256"],
        certificate,
    )
    return PM.build_measurement_plan(inputs)


def test_installed_owner_import_has_no_native_dependencies(monkeypatch):
    forbidden = {
        "perf_agent_stage",
        "run_perf_bench",
        "run_paired_perf_bench",
        "produce_gsim_certificate",
        "heldout_gsim_qualification",
        "_pbcommon",
    }

    class Fence(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                pytest.fail(f"native dependency: {fullname}")

    for name in forbidden:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Fence(), *sys.meta_path])
    # Execute a separate module instance, without replacing canonical dataclass identities.
    name = "merlin_experiments.phase2._paired_measurement_import_test"
    spec = importlib.util.spec_from_file_location(name, PM.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)


def test_serial_parallel_schedule_storage_and_projection_are_byte_identical(tmp_path):
    plan = _plan(tmp_path)
    out = tmp_path / "results"
    out.mkdir()
    completed = []
    release_first = threading.Event()

    def executor(
        spec,
        workspace,
        timeout,
        experiment,
        rtl_identity,
        *,
        hardware_counters,
        counter_binding,
        physical_unit,
        workers,
    ):
        assert timeout == 7 and hardware_counters is False
        assert physical_unit == "BYTES" and not workspace.exists()
        workspace.mkdir(parents=True)
        if workers > 1:
            if spec.execution_index == 0:
                assert release_first.wait(5)
            else:
                completed.append(spec.execution_index)
                release_first.set()
        return {
            "schema": "synthetic",
            "execution": spec.as_dict(),
            "measurement": {
                "status": "pass",
                "numeric": "pass",
                "per_sim": {
                    "spike": {"correct": True, "cycles": 999},
                    "gsim": {
                        "correct": True,
                        "cycles": 100 + spec.pair_index,
                        "provenance": {
                            "tier": "L3",
                            "simulator": "gsim",
                            "oracle_kind": "rtl_gsim",
                            "derived_from_rtl": True,
                            "cycle_accurate": True,
                            "elf_sha256": "a" * 64,
                        },
                    },
                },
                "gsim_qualification": {"admitted": True},
                "work_volume": {},
            },
        }

    kwargs = dict(
        contract_root=tmp_path / "contracts",
        timeout=7,
        target_experiment=object(),
        rtl_identity={},
        hardware_counters=False,
        executor=executor,
        progress=lambda _: None,
    )
    serial = PM.execute_schedule(plan, out, **kwargs)
    saved = tmp_path / "serial"
    out.rename(saved)
    out.mkdir()
    parallel = PM.execute_schedule(plan, out, **kwargs, fanout={"effective": 2})
    assert completed and serial == parallel

    def files(root):
        return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert files(saved) == files(out)
    rows, _ = parallel
    assert PM.ME.completion_report(rows, plan.expected)["complete"] is True
    assert all(row["cycles"] is None and not row["citable"] for row in rows if row["simulator"] == "spike")
    assert len(PM.paired_cycle_rows(rows)) == len(PM.REPLICATES)
    payloads = list((out / "raw_results/sha256").glob("*.json"))
    assert len(payloads) == len(plan.schedule)
    for path in payloads:
        assert path.stem == PM.sha256_file(path)
        assert path.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("changed", ["package", "corpus"])
def test_actual_execution_refuses_changed_inputs_before_engine(tmp_path, monkeypatch, changed):
    spec = _plan(tmp_path).schedule[0]
    root = spec.package if changed == "package" else spec.member.source_dir
    (root / "added.txt").write_text("changed after planning")
    monkeypatch.setattr(PM, "_run_arm4_engines", lambda *a, **k: pytest.fail("engine reached"))
    monkeypatch.setattr(PM.PC, "package_sandbox_policy", lambda *a, **k: pytest.fail("sandbox reached"))
    with pytest.raises(PM.PC.CampaignGateError, match="bytes changed before execution"):
        PM.run_execution(
            spec, tmp_path / "work", 1, object(), {}, contract_root=tmp_path / "contracts", hardware_counters=False
        )
