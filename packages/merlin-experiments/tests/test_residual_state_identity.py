"""A residual-state capsule (main contraction plus ``prelude``) is identified by all its contractions."""

import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus.publication import _excluded_extents, operand_extents, require_public_capsules
from merlin_experiments.spec import SpecError

# Public capsule layers/SY_residual_state_contraction_partial (identity-relevant fields only).
_FIXTURE = yaml.safe_load(
    """
name: SY_residual_state_contraction_partial
kind: layer
label: public
inputs:
- {name: P_W, role: weight, shape: [32, 32], dtype: i8}
- {name: P_A, role: input, shape: [32, 32], dtype: i8}
- {name: W, role: weight, shape: [31, 15], dtype: i8}
- {name: A0, role: input, shape: [16, 31], dtype: i8}
operation:
  op: matmul
  attributes:
    lhs: A0
    weight: W
    out: Y0
    epilogue: []
    output_dtype: i32
    prelude: {lhs: P_A, weight: P_W, out: P_Y, M: 32, K: 32, N: 32, output_dtype: i32}
"""
)


def _fixture():
    return copy.deepcopy(_FIXTURE)


# Public capsule layers/SY_residual_state_conv_k3x3_s1x1_d1x1_pad1x1 (identity-relevant fields only).
_CONV_FIXTURE = yaml.safe_load(
    """
name: SY_residual_state_conv_k3x3_s1x1_d1x1_pad1x1
kind: layer
label: public
inputs:
- {name: P_W, role: weight, shape: [48, 16], dtype: i8}
- {name: P_A, role: input, shape: [16, 48], dtype: i8}
- {name: W, role: weight, shape: [36, 16], dtype: i8}
- {name: IFM, role: input, shape: [1, 4, 4, 4], dtype: i8}
operation:
  op: conv2d
  attributes:
    ifm: IFM
    weight: W
    out: Y0
    ci: 4
    kh: 3
    kw: 3
    stride: [1, 1]
    padding: [1, 1, 1, 1]
    dilation: [1, 1]
    layout: nhwc
    epilogue: []
    output_dtype: i32
    semantic: conv2d
    prelude: {lhs: P_A, weight: P_W, out: P_Y, M: 16, K: 48, N: 16, output_dtype: i32}
"""
)


def _conv_fixture():
    return copy.deepcopy(_CONV_FIXTURE)


def _held_out_capsule(root, a, b, op="matmul", attributes=None):
    """A held-out capsule whose operands are an input of shape ``a`` and a weight of shape ``b``."""
    directory = root / "held" / "cap"
    directory.mkdir(parents=True)
    document = {
        "name": "cap",
        "inputs": [
            {"name": "A", "role": "input", "shape": a},
            {"name": "B", "role": "weight", "shape": b},
        ],
        "operation": {"op": op, "attributes": attributes or {}},
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(document))
    return SimpleNamespace(
        source_root=root,
        held_out_inputs=["held"],
        workload_spec={"models": []},
    )


def _publish(tmp_path, a, b, document=None, **held_out):
    public = tmp_path / "public" / "SY_residual"
    public.mkdir(parents=True)
    (public / "capsule.yaml").write_text(yaml.safe_dump(document or _fixture()))
    require_public_capsules(_held_out_capsule(tmp_path, a, b, **held_out), [tmp_path / "public"])


def test_every_contraction_of_the_capsule_is_identified():
    assert operand_extents(_fixture()) == ((15, 31), (16, 31), (32, 32), (32, 32))


def test_a_residual_state_capsule_stays_unidentified_on_the_held_out_side():
    assert operand_extents(_fixture(), residual_preludes=False) is None
    assert _excluded_extents(yaml.safe_dump(_fixture()).encode()) is None


def test_held_out_core_equal_to_the_prelude_refuses_the_capsule(tmp_path):
    with pytest.raises(SpecError, match="explicitly excluded historical input"):
        _publish(tmp_path, [32, 32], [32, 32])


def test_held_out_core_equal_to_the_main_contraction_refuses_the_capsule(tmp_path):
    with pytest.raises(SpecError, match="explicitly excluded historical input"):
        _publish(tmp_path, [16, 31], [31, 15])


def test_an_unrelated_core_leaves_the_capsule_public(tmp_path):
    _publish(tmp_path, [64, 64], [64, 64])


def test_core_is_a_sub_multiset_test_over_all_contractions():
    extents = Counter(operand_extents(_fixture()))
    assert extents[(32, 32)] == 2


def test_prelude_naming_a_missing_input_stays_unidentified():
    document = _fixture()
    document["operation"]["attributes"]["prelude"]["weight"] = "P_missing"
    assert operand_extents(document) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["operation"]["attributes"].pop("lhs"),
        lambda d: d["operation"]["attributes"].update(weight="A0"),
        lambda d: d["operation"]["attributes"].update(prelude="P"),
        lambda d: d["operation"]["attributes"]["prelude"].update(lhs="P_W"),
        lambda d: d["inputs"].append({"name": "extra", "role": "input", "shape": [4, 4]}),
        lambda d: d["inputs"][2].update(name="P_W"),
        lambda d: d["inputs"][3].update(shape=[16, 30]),
        lambda d: d["inputs"][0].pop("name"),
    ],
)
def test_anything_unresolvable_stays_unidentified(mutate):
    document = _fixture()
    mutate(document)
    assert operand_extents(document) is None


def test_unidentified_residual_capsule_still_refuses_against_held_out_inputs(tmp_path):
    document = _fixture()
    document["operation"]["attributes"]["prelude"]["weight"] = "P_missing"
    public = tmp_path / "public" / "SY_residual"
    public.mkdir(parents=True)
    (public / "capsule.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match="cannot be identified"):
        require_public_capsules(_held_out_capsule(tmp_path, [64, 64], [64, 64]), [tmp_path / "public"])


def test_a_conv_residual_state_capsule_is_identified_by_all_its_contractions():
    assert operand_extents(_conv_fixture()) == ((4, 4, 4), (16, 36), (16, 48), (16, 48))


def test_a_conv_residual_state_capsule_stays_unidentified_on_the_held_out_side():
    assert operand_extents(_conv_fixture(), residual_preludes=False) is None


def test_held_out_core_equal_to_the_conv_prelude_refuses_the_capsule(tmp_path):
    with pytest.raises(SpecError, match="explicitly excluded historical input"):
        _publish(tmp_path, [16, 48], [48, 16], _conv_fixture())


def test_held_out_core_equal_to_the_conv_main_contraction_refuses_the_capsule(tmp_path):
    with pytest.raises(SpecError, match="explicitly excluded historical input"):
        _publish(tmp_path, [1, 4, 4, 4], [36, 16], _conv_fixture(), op="conv2d", attributes={"ci": 4})


def test_an_unrelated_core_leaves_the_conv_capsule_public(tmp_path):
    _publish(tmp_path, [64, 64], [64, 64], _conv_fixture())


def test_conv_without_its_ifm_name_stays_unidentified():
    document = _conv_fixture()
    del document["operation"]["attributes"]["ifm"]
    assert operand_extents(document) is None


def test_conv_naming_lhs_instead_of_ifm_stays_unidentified():
    document = _conv_fixture()
    attributes = document["operation"]["attributes"]
    attributes["lhs"] = attributes.pop("ifm")
    assert operand_extents(document) is None
