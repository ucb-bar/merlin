"""Actual Merlin scalar regions, including checked captured constants and valid mutations."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported, lowered

from merlin.common.paths import repo_root


def parts(module):
    function = next(op for op in module.walk() if op.name == "func.func")
    found = []
    for index, op in enumerate(function.body.block.ops):
        if op.name != "linalg.generic":
            continue
        names = [inner.name for inner in op.body.block.ops]
        if names[0] == "math.absf":
            kind = "absmax"
        elif "arith.select" in names:
            kind = "scale"
        elif names[:2] == ["arith.constant", "arith.divf"]:
            kind = "reciprocal"
        elif "arith.fptosi" in names:
            kind = "quantize"
        elif names[0] == "arith.sitofp":
            kind = f"requantize{len(op.inputs) - 1}"
        else:
            continue
        found.append((kind, index, op))
    return found


def check(document, index, kind, captures=None):
    if captures is None:
        data = json.loads(document.read_text())
        op = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][index]
        body = op["regions"][0]["blocks"][0]["ops"]
        if kind == "scale":
            captures = [body[0]["operands"][1]]
        elif kind == "quantize":
            captures = [body[2]["operands"][1], body[3]["operands"][1]]
        else:
            captures = []
    driver = repo_root() / "merlin/tests/fixtures/lean_w8a8_scalar.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(driver), str(document), str(index), kind, *map(str, captures)],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    assert proc.stdout.startswith("{"), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "w8a8_scalar_only"
    assert (proc.returncode == 0) == result["matches"]
    return result["matches"]


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (3, 17, 5)])
def test_all_actual_merlin_scalar_regions(tmp_path, lean_tools, shape):
    module, _, _ = lowered(*shape)
    document = exported(module, tmp_path, lean_tools)
    actual = parts(module)
    assert [kind for kind, _, _ in actual] == [
        "absmax",
        "scale",
        "reciprocal",
        "quantize",
        "absmax",
        "scale",
        "reciprocal",
        "quantize",
        "requantize2",
    ]
    for kind, index, _ in actual:
        assert check(document, index, kind), (kind, index)


@pytest.mark.parametrize(
    "mutation,kind",
    [
        ("scale_floor", "scale"),
        ("scale_zero_branch", "scale"),
        ("reciprocal_constant", "reciprocal"),
        ("reciprocal_order", "reciprocal"),
        ("rounding", "quantize"),
        ("upper_bound", "quantize"),
        ("lower_bound", "quantize"),
        ("missing_clamp", "quantize"),
        ("unsigned_conversion", "quantize"),
        ("rescale_order", "requantize2"),
        ("absmax_minimum", "absmax"),
    ],
)
def test_valid_mlir_scalar_mutations_refused(tmp_path, lean_tools, mutation, kind):
    from xdsl.dialects.builtin import FloatAttr, f32

    module, _, _ = lowered()
    _, index, generic = next(part for part in parts(module) if part[0] == kind)
    body = list(generic.body.block.ops)
    # Save the original candidate ids before mutating/removing its operations.
    original = exported(module, tmp_path, lean_tools)
    raw = json.loads(original.read_text())["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][index]
    raw_body = raw["regions"][0]["blocks"][0]["ops"]
    captures = []
    if kind == "scale":
        captures = [raw_body[0]["operands"][1]]
    elif kind == "quantize":
        captures = [raw_body[2]["operands"][1], raw_body[3]["operands"][1]]

    if mutation == "scale_floor":
        body[2].properties["value"] = FloatAttr(1e-4, f32)
    elif mutation == "scale_zero_branch":
        body[-2].operands = [body[4].results[0], body[3].results[0], body[2].results[0]]
    elif mutation == "reciprocal_constant":
        body[0].properties["value"] = FloatAttr(2.0, f32)
    elif mutation == "reciprocal_order":
        body[1].operands = list(reversed(body[1].operands))
    elif mutation == "rounding":
        body[1].name = "math.round"
    elif mutation in ("upper_bound", "lower_bound"):
        clamp = body[2 if mutation == "upper_bound" else 3]
        constant = clamp.operands[1].owner
        constant.properties["value"] = FloatAttr(128.0 if mutation == "upper_bound" else -128.0, f32)
    elif mutation == "missing_clamp":
        body[2].results[0].replace_all_uses_with(body[2].operands[0])
        generic.body.block.erase_op(body[2])
    elif mutation == "unsigned_conversion":
        body[4].name = "arith.fptoui"
    elif mutation == "rescale_order":
        first, second = body[1:3]
        left, right = first.operands[1], second.operands[1]
        first.operands = [first.operands[0], right]
        second.operands = [second.operands[0], left]
    else:
        body[1].name = "arith.minimumf"

    document = exported(module, tmp_path, lean_tools)
    # The captured constant producers precede this region, so their ids survive a body deletion.
    assert not check(document, index, kind, captures)


def test_capture_hints_do_not_supply_constant_facts(tmp_path, lean_tools):
    module, _, _ = lowered()
    _, index, _ = next(part for part in parts(module) if part[0] == "quantize")
    document = exported(module, tmp_path, lean_tools)
    assert not check(document, index, "quantize", [0, 1])


def test_nonconstant_rebinding_invalidates_capture_fact(tmp_path, lean_tools):
    module, _, _ = lowered()
    _, index, _ = next(part for part in parts(module) if part[0] == "quantize")
    document = exported(module, tmp_path, lean_tools)
    data = json.loads(document.read_text())
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    body = ops[index]["regions"][0]["blocks"][0]["ops"]
    upper = body[2]["operands"][1]
    lower = body[3]["operands"][1]
    empty = next(op for op in reversed(ops[:index]) if op["name"] == "tensor.empty")
    empty["results"][0]["id"] = upper
    document.write_text(json.dumps(data))
    # This malformed exported SSA is adversarial checker input, not an MLIR-valid mutation.
    assert not check(document, index, "quantize", [upper, lower])


def test_batch_probe_reports_component_scope(tmp_path, lean_tools):
    module, _, _ = lowered()
    document = exported(module, tmp_path, lean_tools)
    driver = repo_root() / "merlin/tests/fixtures/lean_w8a8_scalar.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(driver), str(document), "inspect"],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "hinted_integer_and_requant_bodies_only"
    assert report["whole_pass_verified"] is False
    assert report["refused"] == 0
    assert [site["role_hint"] for site in report["sites"]] == ["contraction", "requant"]
    assert all(site["body_matches"] for site in report["sites"])
