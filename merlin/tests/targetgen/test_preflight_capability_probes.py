"""Descriptor-driven preflight capability probes for self-hosted-ISA targets.

The infrastructure must not know which target, ISA, or dialect it is checking. A target owns the fixture
and adapter and declares stable operation identities; the generic runner accepts the same observations
from scalar, RVV, and target-dialect checks. The named-program adapter additionally verifies its exact
instructions from the derived ISA before accepting a bit-exact result.
"""
from __future__ import annotations

import base64
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import program_oracle as PO
from merlin.targetgen import preflight_probes as PP
from merlin.targetgen import inline_assembly_probe as IAP
from merlin.targetgen.isa_model import IsaModel
from merlin.targetgen.target_experiment import load_target_experiment


def _descriptor(tmp_path: Path, preflight: str) -> Path:
    path = tmp_path / "target_experiment.yaml"
    path.write_text(
        "target: synthetic_core\n"
        "capsule_corpus: merlin/contract/capsules\n"
        "hardware_spec: {}\n"
        "toolchain: {}\n"
        "preflight:\n"
        f"{preflight}",
        encoding="utf-8",
    )
    return path


def _isa() -> IsaModel:
    return IsaModel(
        target="synthetic_core",
        by_mnemonic={
            "ReadCell": {
                "class": "ScalarMemoryRead",
                "mnemonic": "ReadCell",
                "fixed_mask": 0xFFFFFFFF,
                "fixed_value": 1,
            },
            "WriteCell": {
                "class": "ScalarMemoryWrite",
                "mnemonic": "WriteCell",
                "fixed_mask": 0xFFFFFFFF,
                "fixed_value": 2,
            },
        },
    )


def _bundle(words=(1, 2)) -> dict:
    raw = bytes([0xA5])
    return {
        "words": list(words),
        "inputs": [],
        "output": {"base": 64, "shape": [1], "dtype": "int8"},
        "golden": {
            "b64": base64.b64encode(raw).decode("ascii"),
            "shape": [1],
            "dtype": "int8",
        },
    }


def _operation_contract(*operations: dict) -> dict:
    return {
        "version": 1,
        "operations": [
            {
                **operation,
                "status": "unknown",
                "evidence": [{"kind": "discovery", "detail": "synthetic declaration"}],
                "effects": [],
                "semantics": {"kind": "synthetic"},
            }
            for operation in operations
        ],
    }


def test_loader_exposes_generic_capability_probe(tmp_path):
    te = load_target_experiment(_descriptor(
        tmp_path,
        "  capability_probes:\n"
        "    - capability: scalar_memory.load_store_roundtrip\n"
        "      adapter: synthetic.adapters:run\n"
        "      fixture: {kind: named_program, name: TargetOwnedRoundtrip}\n"
        "      requirements:\n"
        "        operations:\n"
        "          - {domain: instruction, dialect: synthetic.isa, operation: ReadCell}\n"
        "          - {domain: instruction, dialect: synthetic.isa, operation: WriteCell}\n",
    ))

    assert len(te.preflight_capability_probes) == 1
    probe = te.preflight_capability_probes[0]
    assert probe.capability == "scalar_memory.load_store_roundtrip"
    assert probe.adapter == "synthetic.adapters:run"
    assert probe.fixture == {"kind": "named_program", "name": "TargetOwnedRoundtrip"}
    assert probe.requirements["operations"] == [
        {"domain": "instruction", "dialect": "synthetic.isa", "operation": "ReadCell"},
        {"domain": "instruction", "dialect": "synthetic.isa", "operation": "WriteCell"},
    ]


@pytest.mark.parametrize("bad", [
    "  capability_probes: not-a-list\n",
    "  capability_probes:\n    - capability: missing-adapter\n"
    "      fixture: {kind: source, path: probe.c}\n"
    "      requirements: {operations: [{domain: instruction, dialect: riscv.isa, operation: add}]}\n",
    "  capability_probes:\n    - capability: duplicate\n      adapter: one:run\n"
    "      fixture: {kind: source, path: one}\n"
    "      requirements: {operations: [{domain: instruction, dialect: one, operation: op}]}\n"
    "    - capability: duplicate\n      adapter: two:run\n"
    "      fixture: {kind: source, path: two}\n"
    "      requirements: {operations: [{domain: instruction, dialect: two, operation: op}]}\n",
    "  capability_probes:\n    - capability: ungrounded\n      adapter: module:run\n"
    "      fixture: {kind: source, path: probe}\n"
    "      requirements: {operations: []}\n",
])
def test_loader_rejects_malformed_or_ungrounded_capability_probes(tmp_path, bad):
    with pytest.raises(ValueError, match="preflight.capability_probes"):
        load_target_experiment(_descriptor(tmp_path, bad))


def test_program_smoke_requires_declared_classes_in_the_program_it_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(PO, "emit_bundle", lambda **_kwargs: _bundle())
    monkeypatch.setattr(
        PO,
        "run_program_oracle",
        lambda *_args, **_kwargs: {
            "outputs": {"Y0": [-91]},
            "cycles": 9,
            "oracle": {"kind": "synthetic-model"},
        },
    )

    result = PO.run_program_oracle_smoke(
        "synthetic_core",
        model_ext="synthetic_model",
        program="TargetOwnedRoundtrip",
        workdir=tmp_path,
        required_instruction_classes=("ScalarMemoryRead", "ScalarMemoryWrite"),
        isa_model=_isa(),
    )

    assert result["ok"] is True
    assert result["instruction_coverage"] == {
        "required": ["ScalarMemoryRead", "ScalarMemoryWrite"],
        "present": ["ScalarMemoryRead", "ScalarMemoryWrite"],
        "present_mnemonics": ["ReadCell", "WriteCell"],
        "missing": [],
        "n_illegal": 0,
    }


def test_program_smoke_fails_before_execution_when_the_fixture_does_not_exercise_the_capability(
        tmp_path, monkeypatch):
    monkeypatch.setattr(PO, "emit_bundle", lambda **_kwargs: _bundle(words=(1,)))

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a structurally incomplete probe must not reach the expensive oracle")

    monkeypatch.setattr(PO, "run_program_oracle", must_not_run)
    result = PO.run_program_oracle_smoke(
        "synthetic_core",
        model_ext="synthetic_model",
        program="IncompleteRoundtrip",
        workdir=tmp_path,
        required_instruction_classes=("ScalarMemoryRead", "ScalarMemoryWrite"),
        isa_model=_isa(),
    )

    assert result["ok"] is False
    assert result["instruction_coverage"]["missing"] == ["ScalarMemoryWrite"]
    assert "does not exercise" in result["reason"]


def test_declared_probe_runner_reports_each_capability_independently(tmp_path, monkeypatch):
    te = load_target_experiment(_descriptor(
        tmp_path,
        "  capability_probes:\n"
        "    - capability: scalar_memory.load_store_roundtrip\n"
        "      adapter: merlin.targetgen.program_oracle:run_capability_probe\n"
        "      fixture: {kind: named_program, name: TargetOwnedRoundtrip}\n"
        "      requirements:\n"
        "        operations:\n"
        "          - {domain: instruction, dialect: synthetic.isa, operation: ReadCell}\n"
        "          - {domain: instruction, dialect: synthetic.isa, operation: WriteCell}\n",
    ))
    def adapter(*, te, probe, workdir, timeout):
        assert te.target == "synthetic_core"
        assert workdir.is_dir() and timeout == 600
        return {
            "reason": "bit-exact",
            "observations": [
                {**operation, "status": "supported",
                 "evidence": {"kind": "rtl_preflight", "detail": "roundtrip matched"}}
                for operation in probe.requirements["operations"]
            ],
        }

    monkeypatch.setattr(PP, "_load_adapter", lambda _reference: adapter)
    operations = te.preflight_capability_probes[0].requirements["operations"]
    result = PP.run_declared_capability_probes(
        te,
        workdir=tmp_path,
        operation_contract=_operation_contract(*operations),
    )

    assert result["ok"] is True
    assert result["probes"] == [{
        "ok": True,
        "reason": "bit-exact",
        "adapter": "merlin.targetgen.program_oracle:run_capability_probe",
        "fixture": {"kind": "named_program", "name": "TargetOwnedRoundtrip"},
        "capability": "scalar_memory.load_store_roundtrip",
        "observations": [
            {"domain": "instruction", "dialect": "synthetic.isa", "operation": "ReadCell",
             "status": "supported",
             "evidence": {"kind": "rtl_preflight", "detail": "roundtrip matched"}},
            {"domain": "instruction", "dialect": "synthetic.isa", "operation": "WriteCell",
             "status": "supported",
             "evidence": {"kind": "rtl_preflight", "detail": "roundtrip matched"}},
        ],
    }]
    assert [operation["status"] for operation in result["operation_capabilities"]["operations"]] == [
        "supported", "supported",
    ]


def test_one_probe_interface_accepts_scalar_rvv_and_target_dialect_operations(tmp_path, monkeypatch):
    desc = _descriptor(
        tmp_path,
        "  capability_probes:\n"
        "    - capability: scalar.add\n      adapter: adapters:scalar\n"
        "      fixture: {kind: source, path: scalar.c}\n"
        "      requirements: {operations: [{domain: instruction, dialect: riscv.isa, operation: add}]}\n"
        "    - capability: vector.add\n      adapter: adapters:rvv\n"
        "      fixture: {kind: source, path: vector.c}\n"
        "      requirements: {operations: [{domain: instruction, dialect: riscv.v, operation: vfadd}]}\n"
        "    - capability: target.matmul\n      adapter: adapters:dialect\n"
        "      fixture: {kind: mlir, path: matmul.mlir}\n"
        "      requirements: {operations: [{domain: dialect, dialect: vendor_accel, operation: matmul}]}\n",
    )
    te = load_target_experiment(desc)

    def adapter(*, probe, **_kwargs):
        operation = probe.requirements["operations"][0]
        return {"reason": "observed", "observations": [{
            **operation,
            "status": "supported",
            "evidence": {"kind": "preflight", "detail": "executed"},
        }]}

    monkeypatch.setattr(PP, "_load_adapter", lambda _reference: adapter)
    operations = [probe.requirements["operations"][0] for probe in te.preflight_capability_probes]
    result = PP.run_declared_capability_probes(
        te,
        workdir=tmp_path / "out",
        operation_contract=_operation_contract(*operations),
    )

    assert result["ok"] is True
    assert [row["observations"][0]["dialect"] for row in result["probes"]] == [
        "riscv.isa", "riscv.v", "vendor_accel",
    ]


def test_probe_requirement_must_be_declared_by_discovered_operation_contract(tmp_path, monkeypatch):
    te = load_target_experiment(_descriptor(
        tmp_path,
        "  capability_probes:\n"
        "    - capability: scalar.store\n      adapter: adapters:scalar\n"
        "      fixture: {kind: source, path: scalar.c}\n"
        "      requirements: {operations: [{domain: instruction, dialect: riscv.isa, operation: sw}]}\n",
    ))

    def must_not_run(_reference):
        raise AssertionError("an undeclared operation must not execute a probe adapter")

    monkeypatch.setattr(PP, "_load_adapter", must_not_run)
    result = PP.run_declared_capability_probes(
        te,
        workdir=tmp_path / "out",
        operation_contract=_operation_contract(
            {"domain": "instruction", "dialect": "riscv.isa", "operation": "lw"}),
    )

    assert result["ok"] is False
    assert result["probes"][0]["observations"][0]["status"] == "unknown"
    assert "not declared by the operation contract" in result["probes"][0]["reason"]
    assert result["operation_capabilities"]["operations"][0]["status"] == "unknown"


def test_inline_assembly_probe_uses_declared_hooks_and_checks_nonzero_memory(tmp_path, monkeypatch):
    source = tmp_path / "scalar_memory.S"
    source.write_text(
        "# @EXPECT_MEMORY 0x90000020 ddccbbaa\n"
        "SW x10, x8, 0\nLW x9, x8, 0\n",
        encoding="utf-8",
    )
    operations = [
        {"domain": "instruction", "dialect": "synthetic.isa", "operation": "LW"},
        {"domain": "instruction", "dialect": "synthetic.isa", "operation": "SW"},
    ]
    probe = SimpleNamespace(
        fixture={
            "kind": "inline_assembly_memory",
            "source": str(source),
            "assembler": "target_probe.py:assemble",
            "runner": "target_probe.py:run",
            "max_cycles": 123,
        },
        requirements={"operations": operations},
    )
    calls = {}

    def assemble(*, source, **_kwargs):
        calls["source"] = source
        return {"words": [1, 2], "operations": ["LW", "SW"]}

    def run(*, te, words, preload, readback, max_cycles, **_kwargs):
        calls["run"] = (te.target, words, preload, readback, max_cycles)
        return {
            "halted": True,
            "cycles": 17,
            "reads": 1,
            "writes": 1,
            "memory": {0x90000020: bytes.fromhex("ddccbbaa")},
        }

    monkeypatch.setattr(IAP, "load_hook", lambda ref: assemble if ref.endswith(":assemble") else run)
    monkeypatch.setattr(
        IAP, "operation_coverage",
        lambda _target, _words, required: {"required": required, "missing": [], "n_illegal": 0})
    result = IAP.run_capability_probe(
        te=SimpleNamespace(target="synthetic_core"), probe=probe, workdir=tmp_path / "out", timeout=60)

    assert calls["source"].startswith("# @EXPECT_MEMORY")
    assert calls["run"] == (
        "synthetic_core", [1, 2], [], [(0x90000020, 4)], 123,
    )
    assert all(observation["status"] == "supported" for observation in result["observations"])
    assert "0x90000020" in result["reason"]


def test_inline_assembly_probe_reports_expected_memory_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "scalar_memory.S"
    source.write_text("# @EXPECT_MEMORY 0x100 01020304\nSW x1, x2, 0\n", encoding="utf-8")
    probe = SimpleNamespace(
        fixture={
            "kind": "inline_assembly_memory",
            "source": str(source),
            "assembler": "target_probe.py:assemble",
            "runner": "target_probe.py:run",
        },
        requirements={"operations": [
            {"domain": "instruction", "dialect": "synthetic.isa", "operation": "SW"},
        ]},
    )
    monkeypatch.setattr(
        IAP,
        "load_hook",
        lambda ref: ((lambda **_kwargs: {"words": [1], "operations": ["SW"]})
                     if ref.endswith(":assemble")
                     else (lambda **_kwargs: {
                         "halted": True, "cycles": 2, "memory": {0x100: b"\0\0\0\0"}})),
    )
    monkeypatch.setattr(
        IAP, "operation_coverage",
        lambda _target, _words, required: {"required": required, "missing": [], "n_illegal": 0})

    result = IAP.run_capability_probe(
        te=SimpleNamespace(target="synthetic_core"), probe=probe, workdir=tmp_path / "out", timeout=60)

    assert result["observations"][0]["status"] == "unsupported"
    assert "memory mismatch at 0x100" in result["reason"]


def test_named_program_adapter_maps_exact_instruction_operations_to_observations(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import isa_model as IM

    operations = [
        {"domain": "instruction", "dialect": "synthetic.isa", "operation": "ReadCell"},
        {"domain": "instruction", "dialect": "synthetic.isa", "operation": "WriteCell"},
    ]
    probe = SimpleNamespace(
        fixture={"kind": "named_program", "name": "TargetOwnedRoundtrip"},
        requirements={"operations": operations},
    )
    monkeypatch.setattr(CR, "_endpoint_of", lambda _target: ("external_backend", "synthetic_model"))
    monkeypatch.setattr(IM, "isa_model_for_target", lambda _target: _isa())
    called = {}

    def smoke(target, **kwargs):
        called.update({"target": target, **kwargs})
        return {"ok": True, "reason": "bit-exact"}

    monkeypatch.setattr(PO, "run_program_oracle_smoke", smoke)
    result = PO.run_capability_probe(
        te=SimpleNamespace(target="synthetic_core"), probe=probe, workdir=tmp_path)

    assert called["program"] == "TargetOwnedRoundtrip"
    assert called["required_instruction_classes"] == ("ReadCell", "WriteCell")
    assert [observation["status"] for observation in result["observations"]] == [
        "supported", "supported",
    ]


def test_capsule_bench_preflight_invokes_the_generic_probe_runner(tmp_path, monkeypatch):
    harness = merlin_dir() / "experiments" / "capsule_bench" / "harness"
    sys.path.insert(0, str(harness))
    try:
        preflight = importlib.import_module("preflight")
    finally:
        sys.path.remove(str(harness))

    te = SimpleNamespace(
        target="a_new_target",
        preflight_capability_probes=(SimpleNamespace(capability="memory.roundtrip"),),
    )
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import target_experiment as TE

    monkeypatch.setattr(TE, "load_target_experiment", lambda _path: te)
    monkeypatch.setattr(CR, "_endpoint_of", lambda target: ("external_backend", "new_model"))
    called = {}

    def run(probe_te, **kwargs):
        called.update({"target": probe_te.target, **kwargs})
        return {"ok": True, "probes": [{"capability": "memory.roundtrip", "ok": True}],
                "reason": "all passed"}

    monkeypatch.setattr(PP, "run_declared_capability_probes", run)
    result = preflight._capability_smokes(desc=tmp_path / "descriptor.yaml")

    assert result["ok"] is True
    assert called["target"] == "a_new_target"


def test_atlas_declares_nonzero_scalar_memory_roundtrip_through_generic_adapter():
    descriptor = merlin_dir() / "experiments" / "capsule_bench" / "targets" / "atlas" \
        / "target_experiment.yaml"
    te = load_target_experiment(descriptor)
    probe = next(
        probe for probe in te.preflight_capability_probes
        if probe.capability == "scalar_memory.load_store_roundtrip")

    assert probe.adapter == "merlin.targetgen.inline_assembly_probe:run_capability_probe"
    assert {(operation["domain"], operation["dialect"], operation["operation"])
            for operation in probe.requirements["operations"]} >= {
        ("instruction", "atlas", "LW"),
        ("instruction", "atlas", "SW"),
    }
    source = (merlin_dir() / probe.fixture["source"]).read_text(encoding="utf-8")
    assert "LI    x8, 0x00004000" in source
    assert "LI    x6, 0x00001000" in source
    assert "@EXPECT_MEMORY 0x90000000" in source


def test_atlas_declares_observable_taken_branch_through_generic_adapter():
    descriptor = merlin_dir() / "experiments" / "capsule_bench" / "targets" / "atlas" \
        / "target_experiment.yaml"
    te = load_target_experiment(descriptor)
    probe = next(
        probe for probe in te.preflight_capability_probes
        if probe.capability == "control_flow.relative_branch")

    assert probe.adapter == "merlin.targetgen.inline_assembly_probe:run_capability_probe"
    assert probe.requirements["operations"] == [
        {"domain": "instruction", "dialect": "atlas", "operation": "BNE"},
    ]
    source = (merlin_dir() / probe.fixture["source"]).read_text(encoding="utf-8")
    assert "BNE   x16, x0, loop" in source
    assert "@EXPECT_MEMORY 0x90000000 03000000" in source
    assert probe.fixture["max_cycles"] == 1000
