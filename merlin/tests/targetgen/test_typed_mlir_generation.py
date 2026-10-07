"""Typed plans emit buildable ODS with checked operation and type contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.artifacts import write_all
from merlin.targetgen.generate import mlir_scaffold, xdsl


def _plan() -> dict:
    return {
        "target": "synthetic",
        "dialect_name": "synthetic",
        "types": [
            {"name": "state"},
            {
                "name": "slot",
                "parameters": [
                    {"name": "unit", "kind": "unsigned", "min": 0, "max": 1},
                ],
            },
            {
                "name": "pair",
                "parameters": [
                    {
                        "name": "first",
                        "kind": "unsigned",
                        "unit": "register_index",
                        "intervals": [
                            {"min": 0, "max": 6, "step": 2},
                            {"min": 10, "max": 12, "step": 2},
                        ],
                    },
                ],
            },
            {
                "name": "lane",
                "parameters": [
                    {"name": "bank", "kind": "unsigned", "unit": "bank_index", "choices": [0, 2, 5]},
                ],
            },
        ],
        "ops": [
            {
                "name": "step",
                "signature": {
                    "operands": [{"name": "state", "type": "!synthetic.state"}],
                    "results": [{"name": "next", "type": "!synthetic.state"}],
                    "attributes": [
                        {"name": "kind", "type": "string", "choices": ["load", "store"]},
                        {"name": "register_index", "type": "i32", "min": 0, "max": 7},
                        {
                            "name": "offset",
                            "type": "i32",
                            "unit": "byte",
                            "intervals": [
                                {"min": -8, "max": -4, "step": 4},
                                {"min": 0, "max": 16, "step": 8},
                            ],
                        },
                    ],
                    "effects": ["read", "write"],
                    "predicates": [
                        {
                            "id": "even_store",
                            "expr": {
                                "implies": [
                                    {"eq": [{"attr": "kind"}, {"string": "store"}]},
                                    {"eq": [{"mod": [{"attr": "register_index"}, {"integer": 2}]}, {"integer": 0}]},
                                ]
                            },
                        }
                    ],
                },
            },
            {
                "name": "twice",
                "signature": {
                    "operands": [{"name": "input", "type": "i32"}],
                    "results": [{"name": "output", "type": "i32"}],
                    "attributes": [],
                    "effects": [],
                },
            },
            {
                "name": "join_slots",
                "signature": {
                    "operands": [
                        {"name": "left", "type": "!synthetic.slot"},
                        {"name": "right", "type": "!synthetic.slot"},
                    ],
                    "results": [{"name": "joined", "type": "!synthetic.slot"}],
                    "attributes": [],
                    "effects": [],
                    "predicates": [
                        {
                            "id": "same_unit",
                            "expr": {
                                "and": [
                                    {
                                        "eq": [
                                            {"type_param": {"value": "left", "name": "unit"}},
                                            {"type_param": {"value": "right", "name": "unit"}},
                                        ]
                                    },
                                    {
                                        "eq": [
                                            {"type_param": {"value": "left", "name": "unit"}},
                                            {"type_param": {"value": "joined", "name": "unit"}},
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
            },
        ],
        "lowering": [],
    }


def _program(*, kind: str = "load", register_index: int = 3, offset: int = 0, slot: str = "state") -> str:
    state = f"!synthetic.{slot}"
    return f"""module {{
  func.func @test(%arg0: {state}, %arg1: i32) -> ({state}, i32) {{
    %0 = "synthetic.step"(%arg0) {{kind = "{kind}", register_index = {register_index} : i32, offset = {offset} : i32}} : ({state}) -> {state}
    %1 = "synthetic.twice"(%arg1) : (i32) -> i32
    return %0, %1 : {state}, i32
  }}
}}
"""


def _slot_program(right_unit: int) -> str:
    return f"""module {{
  func.func @slots(%a: !synthetic.slot<0>, %b: !synthetic.slot<{right_unit}>) -> !synthetic.slot<0> {{
    %0 = "synthetic.join_slots"(%a, %b) : (!synthetic.slot<0>, !synthetic.slot<{right_unit}>) -> !synthetic.slot<0>
    return %0 : !synthetic.slot<0>
  }}
}}
"""


def _type_program(name: str, value: int) -> str:
    return f"""module {{
  func.func @identity(%arg0: !synthetic.{name}<{value}>) -> !synthetic.{name}<{value}> {{
    return %arg0 : !synthetic.{name}<{value}>
  }}
}}
"""


def test_typed_plan_emits_concrete_signatures_and_effects():
    artifacts = {a.relpath: a.content for a in mlir_scaffold.generate(_plan())}
    ops = artifacts["include/MerlinTargetSynthetic/Dialect/Synthetic/IR/SyntheticOps.td"]
    types = artifacts["include/MerlinTargetSynthetic/Dialect/Synthetic/IR/SyntheticTypes.td"]
    cpp = artifacts["lib/Dialect/Synthetic/IR/SyntheticOps.cpp"]
    type_cpp = artifacts["lib/Dialect/Synthetic/IR/SyntheticDialect.cpp"]
    assert "Synthetic_State:$state" in ops
    assert "Synthetic_State:$next" in ops
    assert "StrAttr:$kind" in ops
    assert 'Synthetic_Op<"step", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>' in ops
    assert 'Synthetic_Op<"twice", [Pure]>' in ops
    assert '"unsigned":$unit' in types
    assert '"unsigned":$first' in types
    assert '"unsigned":$bank' in types
    assert "first outside declared domain" in type_cpp
    assert "bank outside declared domain" in type_cpp
    assert "MemoryEffects::Read::get()" in cpp
    assert "MemoryEffects::Write::get()" in cpp
    assert "kind has invalid choice" in cpp
    assert "register_index above maximum" in cpp
    assert "offset outside declared domain" in cpp
    assert "predicate even_store failed" in cpp
    assert "predicate same_unit failed" in cpp
    assert "getOperand(1).getType()" in cpp
    assert not any("Transforms/LowerInterface.cpp" in path for path in artifacts)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["ops"][0].pop("signature"),
        lambda p: p["ops"][0]["signature"].update(effects=["unknown"]),
        lambda p: p["ops"][0]["signature"].update(effects=[[]]),
        lambda p: p["ops"][0]["signature"]["attributes"][0].update(type="opaque"),
        lambda p: p["ops"][0]["signature"]["operands"][0].update(type=[]),
        lambda p: p["ops"][0]["signature"]["attributes"][1].update(max=2**31),
        lambda p: p["types"][1]["parameters"][0].update(max=2**32),
        lambda p: p["types"][2]["parameters"][0]["intervals"][0].update(step=4),
        lambda p: p["types"][2]["parameters"][0]["intervals"].append({"min": 6, "max": 8, "step": 2}),
        lambda p: p["types"][2]["parameters"][0].update(min=0),
        lambda p: p["types"][3]["parameters"][0].update(choices=[0, 0]),
        lambda p: p["types"][3]["parameters"][0].update(choices=[0, True]),
        lambda p: p["ops"][0]["signature"]["attributes"][2]["intervals"][0].update(step=3),
        lambda p: p["ops"][0]["signature"]["attributes"][2].update(max=16),
        lambda p: p["ops"][0]["signature"]["attributes"][2].update(unit=""),
        lambda p: p["ops"][0]["signature"]["attributes"][0].update(min=0),
        lambda p: p["ops"][0]["signature"].update(
            predicates=[
                {
                    "id": "bad",
                    "expr": {"mod": [{"attr": "register_index"}, {"integer": 0}]},
                }
            ]
        ),
        lambda p: p["ops"][0]["signature"].update(
            predicates=[
                {
                    "id": "bad",
                    "expr": {
                        "eq": [
                            {"mod": [{"attr": "register_index"}, {"integer": -1}]},
                            {"integer": 0},
                        ]
                    },
                }
            ]
        ),
        lambda p: p["ops"][0]["signature"].update(
            predicates=[
                {
                    "id": "bad",
                    "expr": {"eq": [{"attr": "kind"}, {"integer": 1}]},
                }
            ]
        ),
        lambda p: p["ops"][2]["signature"].update(
            predicates=[
                {
                    "id": "bad",
                    "expr": {
                        "eq": [
                            {"type_param": {"value": "left", "name": "missing"}},
                            {"integer": 0},
                        ]
                    },
                }
            ]
        ),
        lambda p: p.update(lowering=[{"from": "synthetic.step", "to": "llvm.call"}]),
    ],
)
def test_typed_plan_rejects_missing_contract_fields_and_name_only_lowering(change):
    plan = _plan()
    change(plan)
    with pytest.raises(ValueError):
        mlir_scaffold.generate(plan)


def test_typed_plan_rejects_variadic_xdsl_emission():
    with pytest.raises(ValueError, match="MLIR/C\\+\\+ plane"):
        xdsl.generate(_plan())


def test_typed_plan_builds_and_verifies_in_mlir(tmp_path: Path):
    mlir_dir = os.environ.get("MERLIN_TEST_MLIR_DIR")
    llvm_dir = os.environ.get("MERLIN_TEST_LLVM_DIR")
    if not mlir_dir or not llvm_dir or not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("set MERLIN_TEST_MLIR_DIR and MERLIN_TEST_LLVM_DIR for pinned MLIR build")
    generated = tmp_path / "generated"
    build = tmp_path / "build"
    write_all(mlir_scaffold.generate(_plan()), generated)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(generated),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DMLIR_DIR={mlir_dir}",
            f"-DLLVM_DIR={llvm_dir}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True, capture_output=True, text=True)
    opt = build / "tools/merlin-synthetic-opt"
    good = subprocess.run([str(opt), "-"], input=_program(), capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert '"synthetic.step"' in good.stdout
    same = subprocess.run([str(opt), "-"], input=_slot_program(0), capture_output=True, text=True)
    assert same.returncode == 0, same.stderr
    for program in (_type_program("pair", 4), _type_program("pair", 12), _type_program("lane", 5)):
        accepted = subprocess.run([str(opt), "-"], input=program, capture_output=True, text=True)
        assert accepted.returncode == 0, accepted.stderr
    for program, expected in (
        (_program(kind="other"), "kind has invalid choice"),
        (_program(register_index=8), "register_index above maximum"),
        (_program(kind="store", register_index=3), "predicate even_store failed"),
        (_program(offset=4), "offset outside declared domain"),
        (_program(offset=-6), "offset outside declared domain"),
        (_program(slot="slot<2>"), "unit above maximum"),
        (_slot_program(1), "predicate same_unit failed"),
        (_type_program("pair", 3), "first outside declared domain"),
        (_type_program("pair", 8), "first outside declared domain"),
        (_type_program("lane", 4), "bank outside declared domain"),
    ):
        bad = subprocess.run([str(opt), "-"], input=program, capture_output=True, text=True)
        assert bad.returncode != 0
        assert expected in bad.stderr
