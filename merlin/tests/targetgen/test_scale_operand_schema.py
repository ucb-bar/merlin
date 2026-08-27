"""A block-scale operand must be expressible in the contract schemas, or the capsule dies at the gate.

Declaring the scales (see test_mx_scale_operands) is only half the contract: both schemas validate what
is declared, and neither knew the word. Measured on a regenerated capsule graded end to end:

    R5_mx_tile_mxfp8: status=fail plane=command_buffer_schema category=protocol_violation
      command_buffer schema violation at tensors/W_scale/role:
      'scale' is not one of ['input', 'weight', 'bias', 'output']

That is strictly worse than the bug it was meant to fix. Before, the 9 microscaling capsules FAILED
(gradeable, wrong answer); after, they were REJECTED at the protocol gate before any grading ran. A
contract change that the contract itself rejects is inert at best and destructive at worst.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import merlin_dir

_CAPSULE = merlin_dir() / "contract/schemas/capsule.schema.json"
_CB = merlin_dir() / "contract/schemas/command_buffer.schema.json"


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _capsule_input_props():
    return _load(_CAPSULE)["properties"]["inputs"]["items"]["properties"]


def _cb_tensor_props():
    return _load(_CB)["properties"]["tensors"]["additionalProperties"]["properties"]


def test_both_schemas_admit_a_scale_role():
    """The two gates a scale operand must pass: the capsule that declares it and the command buffer
    that carries it. Missing from EITHER is a protocol violation, not a grade."""
    assert "scale" in _capsule_input_props()["role"]["enum"]
    assert "scale" in _cb_tensor_props()["role"]["enum"]


def test_the_existing_roles_are_untouched():
    """Additive only: an unscaled target's capsules must validate exactly as before."""
    for props in (_capsule_input_props()["role"], _cb_tensor_props()["role"]):
        assert set(props["enum"]) >= {"input", "weight", "bias", "output"}


def test_a_scale_operand_declares_what_it_scales_and_by_how_much():
    props = _capsule_input_props()
    assert props["scale_of"]["type"] == "string", "the pairing must be explicit, not positional"
    assert props["block"]["type"] == "integer" and props["block"]["minimum"] == 1


def test_a_declared_scale_operand_validates():
    jsonschema = pytest.importorskip("jsonschema")
    sch = _load(_CAPSULE)["properties"]["inputs"]["items"]
    jsonschema.validate({"name": "A0_scale", "role": "scale", "shape": [1, 16], "dtype": "e8m0",
                         "scale_of": "A0", "block": 32}, sch)


def test_the_shipped_corpus_validates_against_the_capsule_schema():
    """The end-to-end guard: every capsule on disk, block-scaled or not, must satisfy the schema. This is
    what would have caught the violation before a live run picked it up."""
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    sch = _load(_CAPSULE)
    bad = []
    for cp in (merlin_dir() / "contract/capsules").rglob("capsule.yaml"):
        try:
            jsonschema.validate(yaml.safe_load(cp.read_text(encoding="utf-8")), sch)
        except jsonschema.ValidationError as e:
            bad.append(f"{cp.parent.name}: {e.message[:90]}")
    assert not bad, "capsules on disk violate the capsule schema:\n  " + "\n  ".join(bad[:10])
