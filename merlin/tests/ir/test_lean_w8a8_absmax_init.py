"""Actual floating initializers connected to the checked W8A8 abs/max reduction."""

from __future__ import annotations

import json
import math
import subprocess

import pytest
from test_lean_w8a8_body import exported, lowered
from test_lean_w8a8_scalar import parts

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff


def check_initialization(document, index, kind="absmax", kept=()):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_absmax_init.lean"),
            str(document),
            str(index),
            kind,
            ",".join(map(str, kept)) if kept else "-",
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "w8a8_initialized_absmax_components_only"
    assert report["whole_pass_verified"] is False
    assert (proc.returncode == 0) == (report["refused"] == 0)
    return report["refused"] == 0


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_actual_initialized_reductions(tmp_path, lean_tools, shape):
    module, _, _ = lowered(*shape)
    document = exported(module, tmp_path, lean_tools)
    reductions = [part for part in parts(module) if part[0] == "absmax"]
    assert len(reductions) == 2
    for axis, (_, index, _) in enumerate(reductions):
        assert check_initialization(document, index, kept=(axis,))


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("mutation", ["nonzero", "negative_zero", "yield_old", "use_empty"])
def test_valid_mlir_floating_initializer_mutations_refused(tmp_path, lean_tools, side, mutation):
    from xdsl.dialects.builtin import FloatAttr, f32

    module, _, _ = lowered()
    _, index, generic = [part for part in parts(module) if part[0] == "absmax"][side]
    fill = generic.outputs[0].owner
    if mutation in {"nonzero", "negative_zero"}:
        fill.inputs[0].owner.properties["value"] = FloatAttr(1.0 if mutation == "nonzero" else -0.0, f32)
    elif mutation == "yield_old":
        fill.body.block.last_op.operands = [fill.body.block.args[1]]
    else:
        generic.operands = [*generic.inputs, fill.outputs[0]]
    # Preserve the named fill's mutated body through authoritative MLIR parsing.
    document = exported(module, tmp_path, lean_tools, generic=True)
    assert not check_initialization(document, index, kept=(side,))


def fill_module(shape):
    ty = "tensor<" + "".join(f"{d}x" for d in shape) + "f32>"
    return parse_mlir_text(f"""func.func @forward() -> {ty} {{
      %e = tensor.empty() : {ty}
      %z = arith.constant 0.0 : f32
      %f = linalg.fill ins(%z : f32) outs(%e : {ty}) -> {ty}
      return %f : {ty}
    }}""")


@pytest.mark.parametrize("shape", [(), (3,), (0,), (2, 3), (2, 3, 4), (2, 0, 4)])
def test_floating_initializer_static_ranks_and_zero_extents(tmp_path, lean_tools, shape):
    assert check_initialization(exported(fill_module(shape), tmp_path, lean_tools), 3, "fill")


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 0, 4)])
def test_actual_floating_fill_against_native_mlir(tmp_path, fidelity_tools, shape):
    diff.compare(str(fill_module(shape)), [[]], tmp_path, fidelity_tools, function="forward")
    actual = json.loads((tmp_path / "native-results.json").read_text())["results"]
    assert actual == [["0"] * math.prod(shape)]


def test_floating_initializer_ssa_renaming_and_alias_refusal(tmp_path, lean_tools):
    document = exported(fill_module((2, 3, 4)), tmp_path, lean_tools)
    data = json.loads(document.read_text())

    def rename(value):
        if isinstance(value, dict):
            if "id" in value:
                value["id"] += 500
            if "operands" in value:
                value["operands"] = [identifier + 500 for identifier in value["operands"]]
            for item in value.values():
                rename(item)
        elif isinstance(value, list):
            for item in value:
                rename(item)

    rename(data)
    document.write_text(json.dumps(data))
    assert check_initialization(document, 3, "fill")
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    ops[2]["results"][0]["id"] = ops[1]["results"][0]["id"]
    document.write_text(json.dumps(data))
    assert not check_initialization(document, 3, "fill")
