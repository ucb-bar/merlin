"""A command's operand names are audited against the operands its opcode DECLARES.

THE DEFECT THIS PINS. `command_buffer.schema.json` types an operand map as
`{"type": "object", "additionalProperties": {"type": "string"}}` -- any keys, any arity -- while
`command_buffer_abi.yaml` declares exact operands per opcode. Nothing compared them.

MEASURED: compiling the whole-model capsule `SY_model_smolvla` (8,234 linalg.generic, 812
linalg.reduce, 525 linalg.transpose, 1 linalg.matmul) through a backend produced ONE command --
`DEPTHWISE_CONV2D` with 817 operands named arg0..arg816, i.e. every entry-point argument of the
model bound to an opcode the ABI defines as an NCHW depthwise convolution over {src, weight, dst}.
The schema accepted it. A whole model collapsed into one mislabelled command validated clean.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.command_operands import (
    applies,
    audit,
    declared_operands,
    undeclared_opcodes,
)


def _cmd(opcode="DEPTHWISE_CONV2D", **operands):
    return {"abi_version": "0.1", "target": "t", "backend": "b",
            "tensors": {"a": {"shape": [1], "dtype": "f32", "role": "input"}},
            "commands": [{"opcode": opcode, "operands": dict(operands)}]}


# --- derivation ------------------------------------------------------------------------------

def test_operands_are_derived_from_the_abi_not_listed_in_code():
    d = declared_operands()
    assert d["DEPTHWISE_CONV2D"] == {"src": True, "weight": True, "dst": True}


def test_an_optional_operand_is_read_as_optional():
    """`COMMIT.bias` is spelled "tensor (optional)"; requiring it would reject valid buffers."""
    assert declared_operands()["COMMIT"]["bias"] is False
    assert declared_operands()["COMMIT"]["src"] is True


# --- the degenerate whole-model command ------------------------------------------------------

def test_the_whole_model_as_one_command_is_caught():
    """THE REGRESSION: 817 operands on a depthwise conv must not pass silently."""
    cb = _cmd(**{f"arg{i}": "a" for i in range(817)})
    findings = audit(cb)
    assert findings, "a 817-operand DEPTHWISE_CONV2D produced no finding"
    assert any("do not declare" in f or "does not declare" in f for f in findings)
    assert any("missing required operand" in f for f in findings)


def test_a_correct_command_is_silent():
    assert audit(_cmd(src="a", weight="a", dst="a")) == []


def test_a_command_may_omit_an_optional_operand_but_not_a_required_one():
    assert audit(_cmd("COMMIT", src="a", dst="a")) == []
    assert audit(_cmd("COMMIT", src="a", dst="a", bias="a")) == []
    assert any("missing required" in f for f in audit(_cmd("COMMIT", src="a", bias="a")))


# --- must not report OTHER artifact families as broken ---------------------------------------

@pytest.mark.parametrize("rel", [
    "merlin/contract/capsules/atlas/isa/AS0_matmul_spec/capsule.command_buffer.json",
    "merlin/contract/capsules/radiance/isa/RS0_matmul_spec/capsule.command_buffer.json",
])
def test_a_non_abi_artifact_is_not_audited(rel):
    """An ISA command stream (numeric opcode) and a SIMT warp descriptor (no `commands`) share the
    file name. Describing them as defective would be a false finding, not rigour."""
    cb = json.loads((repo_root() / rel).read_text(encoding="utf-8"))
    assert applies(cb) is False
    assert audit(cb) == []


def test_every_real_command_buffer_in_the_tree_is_clean():
    """Zero false positives is the bar: an audit that fires on the corpus cannot be turned on."""
    import glob

    noisy = []
    paths = ["merlin/contract/examples/expected_command_buffer_g0.json"]
    paths += glob.glob(str(repo_root() / "merlin/contract/capsules/**/capsule.command_buffer.json"),
                       recursive=True)
    for p in sorted(set(paths)):
        cb = json.loads(open(p).read())
        if applies(cb) and audit(cb):
            noisy.append(p)
    assert noisy == []


# --- our own specification gap, reported as ours ---------------------------------------------

def test_an_opcode_with_no_abi_operands_is_named_as_a_contract_gap_not_a_defect():
    cb = _cmd("ATTENTION_FULL", q="a", k="a", v="a", dst="a")
    findings = audit(cb)
    assert len(findings) == 1
    assert "NOT CHECKABLE" in findings[0]
    assert "gap in our contract" in findings[0]


def test_the_size_of_the_specification_gap_is_reported():
    """12 of 25 enumerated opcodes declare no operands. This must be visible, not implicit."""
    gap = undeclared_opcodes()
    assert "ATTENTION_FULL" in gap
    assert "DEPTHWISE_CONV2D" not in gap


def test_audit_never_raises_on_a_malformed_buffer():
    for junk in (None, [], {}, {"commands": "no"}, {"commands": [{"opcode": "MATMUL"}]}):
        audit(junk)
