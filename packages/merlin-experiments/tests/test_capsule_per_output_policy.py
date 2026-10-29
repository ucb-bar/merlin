"""The existing mixed-output writer must pass the ordinary capsule reader."""

import copy
from types import SimpleNamespace

import pytest
import yaml
from test_public_integer_capture import capture

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_source as S
from merlin.targetgen.contract.schemas import ContractViolation, load_schema, validate_capsule
from merlin.targetgen.corpus_spec import CorpusBinding


def written(tmp_path):
    # Existing synthetic transport fixture; no capture issuer or execution claim.
    selected = capture(tmp_path / "capture")
    artifact = S.materialized_model_artifacts(selected)
    source = SimpleNamespace(m2m_dir=tmp_path, capture_loader=lambda *args, **kwargs: artifact)
    entry = dict(
        name="mixed",
        cat="model",
        kind="model",
        op="model",
        loader=str(tmp_path / "capture/loader.py"),
        capture_quantization="already_materialized",
        output_numeric_policy="captured_per_output_v1",
    )
    binding = CorpusBinding("fixture", None, "int8", "i32", True, ["L0"], "exact_int", atol=0, rtol=0)
    directory = S.write_model_capsule(entry, binding, tmp_path / "capsules", source=source)
    return directory, yaml.safe_load((directory / "capsule.yaml").read_bytes())


def test_ordinary_writer_loads_complete_mixed_integer_outputs(tmp_path):
    directory, original = written(tmp_path)
    loaded = CR.load_capsule(directory)
    assert (
        loaded["numeric_policy"]
        == original["numeric_policy"]
        == {
            "schema": "merlin.numeric_policy.per_output.v1",
            "compare": "per_output",
            "outputs": {
                "Y0": {"compare": "exact_int", "dtype": "i32"},
                "Y1": {"compare": "exact_int", "dtype": "i32"},
                "Y2": {"compare": "exact_int", "dtype": "i8"},
            },
        }
    )
    assert loaded["operation"]["attributes"]["outs"] == ["Y0", "Y1", "Y2"]
    assert loaded["required_oracle_tiers"] == original["required_oracle_tiers"] == ["L0"]


def test_registered_float_leaf_preserves_original_tolerances(tmp_path):
    directory, capsule = written(tmp_path)
    leaf = {"compare": "tolerance_float", "dtype": "bf16", "atol": 0.03125, "rtol": 0.02}
    capsule["numeric_policy"]["outputs"]["Y1"] = leaf
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    assert CR.load_capsule(directory)["numeric_policy"]["outputs"]["Y1"] == leaf


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_version",
        "unknown_mode",
        "missing_version",
        "missing_mode",
        "missing_policies",
        "non_mapping_policies",
        "nested",
        "bounded_leaf",
        "unknown_leaf_mode",
        "leaf_extra",
        "envelope_extra",
        "unregistered_dtype",
        "missing_dtype",
        "missing_leaf",
        "extra_leaf",
        "empty_policies",
        "no_roster",
        "empty_roster",
        "duplicate_roster",
        "wrong_roster",
        "non_string_roster",
        "empty_name",
        "conflicting_first_output",
    ],
)
def test_malformed_per_output_policy_refuses_at_ordinary_load(tmp_path, defect):
    directory, capsule = written(tmp_path)
    policy = capsule["numeric_policy"]
    outputs = policy["outputs"]
    attrs = capsule["operation"]["attributes"]
    if defect == "unknown_version":
        policy["schema"] = "merlin.numeric_policy.per_output.v2"
    elif defect == "unknown_mode":
        policy["compare"] = "per_tensor"
    elif defect == "missing_version":
        del policy["schema"]
    elif defect == "missing_mode":
        del policy["compare"]
    elif defect == "missing_policies":
        del policy["outputs"]
    elif defect == "non_mapping_policies":
        policy["outputs"] = list(outputs.values())
    elif defect == "nested":
        outputs["Y0"] = copy.deepcopy(policy)
    elif defect == "bounded_leaf":
        outputs["Y0"]["compare"] = "bounded_int"
    elif defect == "unknown_leaf_mode":
        outputs["Y0"]["compare"] = "approximately_int"
    elif defect == "leaf_extra":
        outputs["Y0"]["allow_cast"] = True
    elif defect == "envelope_extra":
        policy["dtype"] = "i8"
    elif defect == "unregistered_dtype":
        outputs["Y2"]["dtype"] = "bfl16"
    elif defect == "missing_dtype":
        del outputs["Y1"]["dtype"]
    elif defect == "missing_leaf":
        del outputs["Y2"]
    elif defect == "extra_leaf":
        outputs["extra"] = copy.deepcopy(outputs["Y0"])
    elif defect == "empty_policies":
        outputs.clear()
    elif defect == "no_roster":
        del attrs["outs"]
    elif defect == "empty_roster":
        attrs["outs"] = []
    elif defect == "duplicate_roster":
        attrs["outs"] = ["Y0", "Y0", "Y2"]
    elif defect == "wrong_roster":
        attrs["outs"] = ["Y0", "Y1", "elsewhere"]
    elif defect == "non_string_roster":
        attrs["outs"] = ["Y0", "Y1", 2]
    elif defect == "empty_name":
        attrs["outs"][2] = ""
        outputs[""] = outputs.pop("Y2")
    else:
        attrs["out"] = "Y2"
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    with pytest.raises(ContractViolation):
        CR.load_capsule(directory)


@pytest.mark.parametrize("compare", ["exact_int", "tolerance_float", "bounded_int"])
def test_legacy_flat_policy_does_not_require_a_new_output_roster(tmp_path, compare):
    _, capsule = written(tmp_path)
    capsule["numeric_policy"] = {
        "compare": compare,
        "dtype": "i32",
        "atol": 0.0,
        "rtol": 0.0,
        "acc_scale": 1.0,
        "requant_shift": 3,
    }
    del capsule["operation"]["attributes"]["outs"]
    validate_capsule(capsule)


def test_legacy_flat_policy_stays_closed(tmp_path):
    _, capsule = written(tmp_path)
    capsule["numeric_policy"] = {"compare": "exact_int", "dtype": "i32", "unknown": True}
    with pytest.raises(ContractViolation):
        validate_capsule(capsule)


def test_per_output_branch_reuses_the_unchanged_flat_leaf_definition():
    schema = load_schema("capsule")
    flat, per_output = schema["properties"]["numeric_policy"]["oneOf"]
    assert flat == {"$ref": "#/$defs/numeric_policy_leaf"}
    leaf = per_output["properties"]["outputs"]["additionalProperties"]
    assert leaf["allOf"][0] == flat
    assert leaf["allOf"][1]["properties"]["compare"]["enum"] == ["exact_int", "tolerance_float"]


@pytest.mark.parametrize("name", ["Y0", "Y1", "Y2"])
def test_each_output_leaf_dtype_is_checked_against_the_registry(tmp_path, name):
    directory, capsule = written(tmp_path)
    capsule["numeric_policy"]["outputs"][name]["dtype"] = "not_a_registered_format"
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    with pytest.raises(ContractViolation, match=f"outputs/{name}/dtype"):
        CR.load_capsule(directory)
