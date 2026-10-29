"""Actual W8A8 initialization, including valid but nonzero or uninitialized mutations."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported, lowered

from merlin.common.paths import repo_root


def check_zero(document, selection):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_zero.lean"),
            str(document),
            str(selection),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "w8a8_zero_initializer_only"
    assert report["whole_pass_verified"] is False
    assert (proc.returncode == 0) == (report["refused"] == 0)
    return report["refused"] == 0


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_actual_zero_initializer(tmp_path, lean_tools, shape):
    module, index, _ = lowered(*shape)
    assert check_zero(exported(module, tmp_path, lean_tools), index)


@pytest.mark.parametrize("mutation", ["nonzero", "yield_old", "use_empty"])
def test_valid_mlir_initializer_mutations_refused(tmp_path, lean_tools, mutation):
    from xdsl.dialects.builtin import IntegerAttr, i32

    module, index, generic = lowered()
    destination = generic.outputs[0]
    fill = destination.owner
    if mutation == "nonzero":
        fill.inputs[0].owner.properties["value"] = IntegerAttr(1, i32)
    elif mutation == "yield_old":
        fill.body.block.last_op.operands = [fill.body.block.args[1]]
    else:
        generic.operands = [*generic.inputs, fill.outputs[0]]
    # Named-op custom assembly reconstructs an implicit default body. Generic printing is
    # necessary to preserve a mutated fill region through the actual compiler parser.
    assert not check_zero(exported(module, tmp_path, lean_tools, generic=True), index)


def test_initializer_renaming_and_malformed_alias_refusal(tmp_path, lean_tools):
    module, index, _ = lowered()
    document = exported(module, tmp_path, lean_tools)
    data = json.loads(document.read_text())
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]

    # Rename every SSA id, including nested arguments and uses. The closed initializer must
    # still establish the same destination value without trusting those ids as literal facts.
    def rename(value):
        if isinstance(value, dict):
            if "id" in value:
                value["id"] += 1000
            if "operands" in value:
                value["operands"] = [identifier + 1000 for identifier in value["operands"]]
            for item in value.values():
                rename(item)
        elif isinstance(value, list):
            for item in value:
                rename(item)

    rename(data)
    document.write_text(json.dumps(data))
    assert check_zero(document, index)
    # This deliberately malformed certificate input is separate from valid MLIR mutations.
    ops[index - 1]["results"][0]["id"] = ops[index - 2]["results"][0]["id"]
    document.write_text(json.dumps(data))
    assert not check_zero(document, index)
