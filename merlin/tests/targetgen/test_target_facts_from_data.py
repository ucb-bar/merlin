"""A new target gets its introspect kind, its encoding-fact file and its elaborated config from DATA only.

Three facts used to be literals in shared library code: an arc-target -> kind table and an encoding-fact
file name in ``rtl.mlc_bridge``, and an elaborated chipyard config in ``rtl.introspect``. Each now comes
from data the target (or its arc model) owns:

  * kind and encoding-fact file: the arc-model registry ``merlin/contract/arc_models.yaml``;
  * elaborated config: ``runtime.rtl_sim_config`` in the target's own contract.

Every target below is synthetic and reaches the code only through data written in ``tmp_path``: a
registry file, an mlc outputs tree, an out-of-tree package on ``MERLIN_TARGET_PATH``. Nothing here names a
shipped target, so a second target that passes these tests needs no library edit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.rtl import introspect
from merlin.targetgen.rtl import mlc_bridge as B

_ARC_KEY = "newchip_arc"            # synthetic mlc arc key
_TARGET = "newchip"                 # synthetic merlin target
_CONFIG = "NewChipHarnessConfig"    # synthetic chipyard harness config


def _registry(tmp_path: Path, models: dict) -> Path:
    p = tmp_path / "arc_models.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 1, "arc_models": models}), encoding="utf-8")
    return p


def _use_registry(monkeypatch, path: Path) -> None:
    real = B.load_arc_models
    monkeypatch.setattr(B, "load_arc_models", lambda p=None: real(path))


def _fake_mlc(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "mlc_checkout"
    (root / "mlc").mkdir(parents=True)          # mlc_dir() requires the package dir to exist
    monkeypatch.setattr(B, "mlc_dir", lambda: root)
    return root


# ------------------------------------------------------------------ kind + encoding file: the registry
def test_arc_model_kind_and_encoding_file_come_from_the_registry(tmp_path, monkeypatch):
    _use_registry(monkeypatch, _registry(tmp_path, {
        _ARC_KEY: {"kind": "spatial", "isa_encoding": "newchip_isa.json"}}))
    mlc = _fake_mlc(tmp_path, monkeypatch)
    fact = {"inst_width": 32, "fields": {"opcode": [6, 0]}, "opcodes": {"op0": 11}}
    outputs = mlc / "runs" / "circt-arc" / _ARC_KEY / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "newchip_isa.json").write_text(json.dumps(fact), encoding="utf-8")

    # no capability manifest exists for this arc-only target, so the registry's kind routes it
    assert B._resolve_kind(_ARC_KEY) == "spatial"
    assert B.compute_unit_kinds(_ARC_KEY) == ("spatial",)
    # the encoding fact is read from the file the registry names, under the model's arc outputs
    assert B.isa_encoding_artifact(_ARC_KEY) == (outputs / "newchip_isa.json", "declared")
    assert B.isa_encoding_for(_ARC_KEY) == fact


def test_an_unregistered_model_fails_closed_with_a_named_reason(tmp_path, monkeypatch):
    _use_registry(monkeypatch, _registry(tmp_path, {}))
    _fake_mlc(tmp_path, monkeypatch)
    assert B._resolve_kind(_ARC_KEY) is None
    assert B.compute_unit_kinds(_ARC_KEY) == ()
    path, why = B.isa_encoding_artifact(_ARC_KEY)
    assert path is None
    assert "isa_encoding" in why and "arc_models.yaml" in why and _ARC_KEY in why
    assert B.isa_encoding_for(_ARC_KEY) is None


def test_no_mlc_checkout_is_a_named_reason_not_a_path(tmp_path, monkeypatch):
    _use_registry(monkeypatch, _registry(tmp_path, {_ARC_KEY: {"isa_encoding": "newchip_isa.json"}}))
    monkeypatch.setattr(B, "mlc_dir", lambda: None)
    path, why = B.isa_encoding_artifact(_ARC_KEY)
    assert path is None and "MERLIN_MLC_DIR" in why


@pytest.mark.parametrize("entry, needle", [
    ({"kind": "not_a_kind"}, "kind"),
    ({"isa_encodings": "x.json"}, "unknown field"),
    ({"isa_encoding": "../escape.json"}, "bare file name"),
])
def test_registry_refuses_what_it_cannot_honour(tmp_path, entry, needle):
    with pytest.raises(B.ArcModelRegistryError, match=needle):
        B.load_arc_models(_registry(tmp_path, {_ARC_KEY: entry}))


def test_registry_refuses_a_missing_file_or_schema(tmp_path):
    with pytest.raises(B.ArcModelRegistryError, match="no arc-model registry"):
        B.load_arc_models(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"schema_version": 99, "arc_models": {}}), encoding="utf-8")
    with pytest.raises(B.ArcModelRegistryError, match="schema_version"):
        B.load_arc_models(bad)


def test_the_shipped_registry_loads():
    models = B.load_arc_models()
    assert models, "the shipped arc-model registry declares no model"
    assert all(set(v) <= {"kind", "isa_encoding"} for v in models.values())


# ------------------------------------------------------------ elaborated config: the target's contract
def _oot_package(root: Path, name: str, runtime: dict) -> Path:
    pkg = root / name
    (pkg / "contracts").mkdir(parents=True)
    contract = {
        "name": name, "version": "0.1", "status": "prototype",
        "runtime": runtime,
        "compute_units": [{
            "name": "mesh", "kind": "systolic", "dtypes": ["fp16"], "ops": ["matmul"],
            "accumulate": [{"in": "fp16", "weight": "fp16", "acc": "f32"}]}],
    }
    (pkg / "contracts" / "target_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    return pkg


def _fake_elaboration(chipyard: Path, config: str) -> Path:
    gen = chipyard / "sims" / "verilator" / "generated-src" / f"chipyard.harness.TestHarness.{config}"
    gen.mkdir(parents=True)
    (gen / f"chipyard.harness.TestHarness.{config}.fir").write_text(
        "FIRRTL version 4.0.0\ncircuit TestHarness :\n  module TestHarness :\n", encoding="utf-8")
    (gen / "top_module_hierarchy.json").write_text(
        json.dumps({"instance_name": "TestHarness", "module_name": "TestHarness", "instances": []}),
        encoding="utf-8")
    return gen


def test_elaborated_config_comes_from_the_target_contract(tmp_path, monkeypatch):
    pkgs = tmp_path / "packages"
    _oot_package(pkgs, _TARGET, {"backends": ["simulator"], "rtl_sim_config": _CONFIG})
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(pkgs))
    chipyard = tmp_path / "chipyard"
    gen = _fake_elaboration(chipyard, _CONFIG)

    assert introspect.sim_config(_TARGET) == _CONFIG
    arts, facts = introspect.sim_elaboration_facts(_TARGET, chipyard)
    assert arts["fir"] == gen / f"chipyard.harness.TestHarness.{_CONFIG}.fir"
    assert arts["hierarchy"] == gen / "top_module_hierarchy.json"
    # the facts record which target and which elaboration they came from, and nothing else is assumed
    assert facts["target"] == _TARGET
    assert facts["source"]["config"] == _CONFIG
    # this reader predates the census and does not scope it to the target
    assert facts["census"]["status"] == "not_run"


def test_a_target_declaring_no_sim_config_is_undeclared_not_defaulted(tmp_path, monkeypatch):
    pkgs = tmp_path / "packages"
    _oot_package(pkgs, _TARGET, {"backends": ["simulator"]})
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(pkgs))
    with pytest.raises(introspect.RtlSourceUndeclared, match="runtime.rtl_sim_config"):
        introspect.sim_config(_TARGET)
    with pytest.raises(introspect.RtlSourceUndeclared):
        introspect.sim_elaboration_facts(_TARGET, tmp_path / "chipyard")


def test_find_artifacts_takes_no_default_config(tmp_path):
    with pytest.raises(introspect.RtlSourceUndeclared):
        introspect.find_artifacts(tmp_path, "")
    arts = introspect.find_artifacts(tmp_path, _CONFIG)
    assert _CONFIG in arts["fir"].name
