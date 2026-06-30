"""Experiment-ABI contract bundle: schemas, the merlin_iface grammar round-trip, golden examples.

These run with no toolchain (K0/K10 surface). The runner smoke/negative tests live in
test_oot_runner_smoke.py / test_oot_runner_negative.py.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import json
from pathlib import Path

import jsonschema
import pytest

from merlin.eval.gemmini_conformance import RUNGS, QUANT_RUNGS, build
from merlin.targetgen.contract import schemas
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

REPO = repo_root()
BC = REPO / "bench_contract"
CORE = ("abi_version", "target", "tensors", "commands")
ALL_RUNGS = list(RUNGS) + list(QUANT_RUNGS)


def test_all_schemas_are_valid_draft2020():
    for p in sorted((BC / "schemas").glob("*.schema.json")):
        jsonschema.Draft202012Validator.check_schema(json.loads(p.read_text()))


@pytest.mark.parametrize("rung", ALL_RUNGS)
def test_interface_grammar_round_trips(rung):
    """K-ladder K0 substrate: emit(cb) -> parse reproduces the interface-core of the cb."""
    cb = build(rung)
    back = parse_interface_mlir(emit_interface_mlir(cb))
    assert back == {k: cb[k] for k in CORE}


def test_golden_examples_parse_and_validate():
    for name in ("g0_matmul", "g1_relu", "g2_acc_scale"):
        text = (BC / "examples" / f"{name}.interface.mlir").read_text()
        cb = parse_interface_mlir(text)
        schemas.validate_command_buffer(cb)            # fail-closed schema check
        assert cb["commands"] and cb["target"] == "gemmini"


def test_expected_g0_command_buffer_matches_emitter():
    golden = json.loads((BC / "examples" / "expected_command_buffer_g0.json").read_text())
    schemas.validate_command_buffer(golden)
    assert parse_interface_mlir(emit_interface_mlir(build("C0"))) == golden


def test_command_buffer_schema_fails_closed():
    golden = json.loads((BC / "examples" / "expected_command_buffer_g0.json").read_text())
    bad = json.loads(json.dumps(golden))
    bad["commands"][1]["opcode"] = "FROBNICATE"
    with pytest.raises(schemas.ContractViolation):
        schemas.validate_command_buffer(bad)


def test_manifest_schema_accepts_real_packages_and_rejects_missing_entrypoint():
    import yaml
    for pkg in ("merlin_native_v0", "hand_smoke_oot"):
        man = yaml.safe_load((REPO / "generated_targets" / "gemmini" / pkg / "manifest.yaml").read_text())
        schemas.validate_manifest(man)
    # drop a required entrypoint -> fail closed
    man = yaml.safe_load((REPO / "generated_targets" / "gemmini" / "merlin_native_v0" / "manifest.yaml").read_text())
    del man["commands"]["emit_command_buffer"]
    with pytest.raises(schemas.ContractViolation):
        schemas.validate_manifest(man)
