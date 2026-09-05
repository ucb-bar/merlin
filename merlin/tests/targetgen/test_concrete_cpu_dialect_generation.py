"""Reviewed CPU dialect plans generate manipulable MLIR and xDSL declarations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from merlin.common.artifacts import write_all
from merlin.common.paths import merlin_dir
from merlin.targetgen.generate import mlir_scaffold, xdsl


def _k1_plan() -> dict:
    return yaml.safe_load(
        (merlin_dir() / "targets/k1_cpu/contracts/dialect_plan.yaml").read_text(
            encoding="utf-8"
        )
    )


def _artifacts(generator, plan: dict) -> dict[str, str]:
    return {artifact.relpath: artifact.content for artifact in generator.generate(plan)}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("generated_rvvhost_dialect", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_k1_mlir_scaffold_declares_all_reviewed_ops_and_types():
    plan = _k1_plan()
    artifacts = _artifacts(mlir_scaffold, plan)
    ops = artifacts[
        "include/MerlinTargetK1Cpu/Dialect/K1Cpu/IR/K1CpuOps.td"
    ]
    types = artifacts[
        "include/MerlinTargetK1Cpu/Dialect/K1Cpu/IR/K1CpuTypes.td"
    ]
    dialect_cpp = artifacts["lib/Dialect/K1Cpu/IR/K1CpuDialect.cpp"]

    expected_ops = {row["name"] for row in plan["ops"]}
    expected_types = {row["name"] for row in plan["types"]}
    assert len(expected_ops) == 9
    assert len(expected_types) == 2
    for name in expected_ops:
        assert f'K1Cpu_Op<"{name}">' in ops
    for name in expected_types:
        assert f'"{name}">' in types

    assert "addOperations<" in dialect_cpp
    assert "GET_OP_LIST" in dialect_cpp
    assert "addTypes<" in dialect_cpp
    assert "GET_TYPEDEF_LIST" in dialect_cpp
    assert "mlir_tablegen(K1CpuOps.h.inc -gen-op-decls)" in artifacts[
        "include/MerlinTargetK1Cpu/Dialect/K1Cpu/IR/CMakeLists.txt"
    ]
    assert "add_mlir_dialect_library(MLIRK1Cpu" in artifacts[
        "lib/Dialect/K1Cpu/IR/CMakeLists.txt"
    ]
    assert "find_package(MLIR REQUIRED CONFIG)" in artifacts["CMakeLists.txt"]
    assert "add_subdirectory(lib/Dialect/K1Cpu)" in artifacts["CMakeLists.txt"]


def test_k1_lowering_pass_contains_every_reviewed_mapping_and_fails_closed():
    plan = _k1_plan()
    artifacts = _artifacts(mlir_scaffold, plan)
    lowering = artifacts["lib/Dialect/K1Cpu/Transforms/LowerInterface.cpp"]
    lit = artifacts["tests/lit/rvvhost/interface_lowering.mlir"]

    assert len(plan["lowering"]) == 6
    for row in plan["lowering"]:
        source, target = row["from"], row["to"]
        assert f'patterns.add<RenameByNamePattern>(context, "{source}", "{target}");' in lowering
        assert f"{source} -> {target}" in lowering
        assert f'"{source}"() : () -> ()' in lit
        assert f'CHECK: "{target}"' in lit

    assert "op->getNumRegions() != 0 || op->getNumSuccessors() != 0" in lowering
    assert "generated lowering did not convert" in lowering
    assert "signalPassFailure()" in lowering
    assert "add_mlir_library(MLIRK1CpuTransforms" in artifacts[
        "lib/Dialect/K1Cpu/Transforms/CMakeLists.txt"
    ]


def test_k1_xdsl_registers_nine_ops_and_two_types(tmp_path):
    plan = _k1_plan()
    staged = tmp_path / "generated-target"
    write_all(mlir_scaffold.generate(plan) + xdsl.generate(plan), staged)
    generated = staged / "xdsl/rvvhost_dialect.py"
    module = _load(generated)

    assert (staged / "lib/Dialect/K1Cpu/Transforms/LowerInterface.cpp").is_file()
    assert (staged / "include/MerlinTargetK1Cpu/Dialect/K1Cpu/IR/K1CpuOps.td").is_file()
    assert module.HAS_XDSL
    dialect = module.get_dialect()
    assert {op.name for op in dialect.operations} == {
        f"rvvhost.{row['name']}" for row in plan["ops"]
    }
    assert {typ.name for typ in dialect.attributes} == {
        f"rvvhost.{row['name']}" for row in plan["types"]
    }
    assert len(module.OP_CLASSES) == 9
    assert len(module.TYPE_CLASSES) == 2
    for operation_class in module.OP_CLASSES:
        operation_class(operands=[[]], result_types=[[]]).verify()
    for type_class in module.TYPE_CLASSES:
        assert type_class().name.startswith("rvvhost.")


def test_reviewed_k1_generators_have_no_placeholder_dialect_claims():
    plan = _k1_plan()
    generated = "\n".join(
        artifact.content
        for generator in (mlir_scaffold, xdsl)
        for artifact in generator.generate(plan)
    ).lower()
    assert "no ops synthesized yet" not in generated
    assert "empty dialect" not in generated
    assert "empty registered dialect" not in generated
