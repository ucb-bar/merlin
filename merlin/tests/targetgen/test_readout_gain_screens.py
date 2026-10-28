"""The ``readout_gain_screens`` operation key: authored only where an exhaustive numerical contract exists."""

import copy

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.software_spec import validate_software_spec

SPEC = repo_root() / "examples/gemmini/target/software-spec.yaml"
ROW = "elementwise_map_after_per_operand_sum"


@pytest.fixture
def document():
    return yaml.safe_load(SPEC.read_bytes())


def test_per_operand_row_accepts_both_screens(document):
    document["operations"][ROW]["readout_gain_screens"] = ["tensor", "column_component"]
    validated = validate_software_spec(copy.deepcopy(document))
    row = next(row for row in validated["operations"] if row["id"] == ROW)
    assert row["readout_gain_screens"] == ["tensor", "column_component"]
    # It is a declaration field, never folded into the signature constraints.
    assert "readout_gain_screens" not in row["signature"]


@pytest.mark.parametrize(
    "value",
    [[], "column_component", ["column"], ["tensor", "tensor"], ["tensor", 1], None, {"tensor": True}],
    ids=["empty", "string", "unknown_value", "duplicate", "non_string", "none", "mapping"],
)
def test_bad_screen_lists_are_refused(document, value):
    document["operations"][ROW]["readout_gain_screens"] = value
    with pytest.raises(ValueError, match="readout_gain_screens"):
        validate_software_spec(document)


def test_a_row_without_a_numerical_contract_refuses_the_key(document):
    document["operations"]["movement"]["readout_gain_screens"] = ["tensor"]
    with pytest.raises(ValueError, match="readout_gain_screens requires an exhaustive numerical contract"):
        validate_software_spec(document)
