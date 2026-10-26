"""Device declarations state explicit domains, and admission screens what they declare."""

import copy

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.software_spec import (
    DOMAIN_AXES,
    admit_operation,
    operation_domains,
    validate_software_spec,
)

SPEC = repo_root() / "examples/gemmini/target/software-spec.yaml"


@pytest.fixture
def document():
    return yaml.safe_load(SPEC.read_bytes())


def test_every_device_declaration_states_its_domain(document):
    domains = operation_domains(validate_software_spec(document))
    assert domains and "host_quantize" not in domains
    for identity, record in domains.items():
        assert record["arithmetic"], identity
        for axis in DOMAIN_AXES:
            assert record[axis] is not None, (identity, axis)
            if isinstance(record[axis], dict):
                assert record[axis]["unconstrained"].strip()


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda ops: ops["contraction"].pop("tails"), "undeclared"),
        (lambda ops: ops["movement"].pop("arithmetic"), "undeclared"),
        (lambda ops: ops["movement"].update(unconstrained={"tails": " "}), "written reason"),
        (lambda ops: ops["movement"].update(unconstrained={"tails": "x", "layouts": "y"}), "both constrains"),
        (lambda ops: ops["movement"].update(unconstrained={"dtypes": "x", "tails": "y"}), "not a domain axis"),
        (lambda ops: ops["reduction_standalone"].update(kinds=["max"]), "only sum"),
        (lambda ops: ops["contraction"].update(arithmetic="none"), "arithmetic none"),
        (lambda ops: ops["contraction"].update(arithmetic="integer_sum"), "requires constraints"),
        (lambda ops: ops["contraction"].update(arithmetic="floating"), "must be one of"),
    ],
)
def test_incomplete_or_inconsistent_domains_are_refused(document, edit, message):
    edit(document["operations"])
    with pytest.raises(ValueError, match=message):
        validate_software_spec(document)


def test_domains_are_required_only_when_the_spec_selects_them(document):
    document["operations"]["contraction"].pop("tails")
    with pytest.raises(ValueError):
        validate_software_spec(copy.deepcopy(document))
    document.pop("operation_domains")
    validate_software_spec(document)


def test_an_observed_arithmetic_class_must_match_the_declaration(document):
    spec = validate_software_spec(document)
    signature = dict(
        family="contraction",
        operand_dtype="i8",
        accum_dtype="i32",
        rank=2,
        layout="row_major_contiguous",
        tails="zero_pad_valid_window",
        broadcasting="none",
        aliasing="disjoint_inputs_outputs",
    )
    assert admit_operation(spec, "contraction", signature, "accelerator")["status"] == "admitted"
    refused = admit_operation(spec, "contraction", {**signature, "arithmetic": "none"}, "accelerator")
    assert refused["status"] == "unsupported" and "arithmetic" in refused["reason"]
    outside = admit_operation(spec, "contraction", {**signature, "rank": 5}, "accelerator")
    assert outside["status"] == "unsupported" and "rank" in outside["reason"]
