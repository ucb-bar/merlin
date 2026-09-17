"""A rank-0 (scalar) tensor is declarable in a command buffer, and nests to its element.

WHY THIS EXISTS. ``command_buffer_abi.yaml`` requires that every ``kernel_abi`` argument appear in
``tensors`` ("Declare EVERY tensor your kernel touches"), and a whole-model entry point really can
take a scalar argument: ``SY_model_smolvla``'s ``@forward`` takes 816 operands, of which ``%803`` is
``tensor<f32>`` -- an attention-scale broadcast consumed by an elementwise ``mul``. The schema
nevertheless carried ``minItems: 1`` on ``shape`` from its first commit, with no description, no
commit rationale and no test, so declaring that argument was impossible and the whole model was
unrepresentable.

The floor was ALSO inconsistent with the code: ``bundle_pack.physical_nbytes`` and ``bundle_pack.plan``
both special-case rank-0 explicitly -- the packer was written expecting scalars the schema forbade.

Spelling a scalar ``[1]`` instead is NOT the fix and these tests do not accept it: it is a false rank
that ``validate_interface_tensor_dtypes`` (dtype-only) would not catch, and it diverges from the
interface MLIR type any future shape cross-check would compare against.
"""

from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root

jsonschema = pytest.importorskip("jsonschema")

SCHEMAS = (
    "merlin/contract/schemas/command_buffer.schema.json",
    "merlin/python/merlin/_data/contract/schemas/command_buffer.schema.json",
)


def _schema(rel: str) -> dict:
    return json.loads((repo_root() / rel).read_text(encoding="utf-8"))


def _buffer(shape) -> dict:
    return {
        "abi_version": "0.1",
        "target": "atlas",
        "backend": "test",
        "tensors": {
            "S": {"shape": shape, "dtype": "f32", "role": "input"},
            "Y": {"shape": [1, 4], "dtype": "f32", "role": "output"},
        },
        "commands": [{"opcode": "MATMUL", "operands": {"a": "S", "c": "Y"}}],
    }


@pytest.mark.parametrize("rel", SCHEMAS)
def test_a_rank0_tensor_is_accepted(rel):
    """THE REGRESSION. Re-adding ``minItems: 1`` to shape fails exactly here."""
    jsonschema.Draft7Validator(_schema(rel)).validate(_buffer([]))


@pytest.mark.parametrize("rel", SCHEMAS)
def test_the_shape_rule_still_refuses_a_zero_extent(rel):
    """Relaxing the RANK floor must not relax the EXTENT floor: [0] is still a nonsense tensor."""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(_schema(rel)).validate(_buffer([0]))


def test_both_schema_copies_stay_byte_identical():
    """``check_contract_copies.py`` requires it; a one-sided edit here is the way that breaks."""
    a, b = ((repo_root() / rel).read_bytes() for rel in SCHEMAS)
    assert a == b


def test_the_relaxation_is_recorded_where_the_next_reader_will_look():
    """An undocumented floor is what caused this; an undocumented relaxation invites its return."""
    shape = _schema(SCHEMAS[0])["properties"]["tensors"]["additionalProperties"]["properties"]["shape"]
    assert shape["minItems"] == 0
    assert "rank-0" in shape.get("description", "")


def test_a_rank0_tensor_nests_to_its_element_not_a_list():
    """``Tensor.to_list`` walked ``dims[0]`` unguarded and raised IndexError on a scalar."""
    from merlin.runtime.tensor import Tensor

    assert Tensor((), [7]).to_list() == 7


def test_rank0_size_is_one_element_not_zero():
    """Every consumer sizes a tensor as prod(shape); for [] that is 1, so the scalar gets bytes."""
    import math

    assert math.prod([]) == 1
