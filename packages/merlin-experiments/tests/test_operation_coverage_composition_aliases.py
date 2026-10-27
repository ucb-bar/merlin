"""Validated fusion identities trace to authored chains through display aliases."""

import copy
import json

import pytest
import yaml
from merlin_experiments.phase0 import operation_coverage_ledger as L
from merlin_experiments.phase0.component_generation import digest
from merlin_experiments.phase0.composition_generation import coverage


@pytest.fixture
def inputs(tmp_path):
    chain = {
        "id": "authored_chain",
        "producer": "operand_sum",
        "readout_stages": "from_output_path",
        "exits": ["accumulator"],
        "constraints": [{"operand_multiplier_in": "(0, 1]"}, "load_minimum_not_negated"],
    }
    spec = {
        "schema": "merlin.software_spec.v1",
        "target": "toy",
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "int32",
            "readout_dtype": "int8",
            "subnormal_operand_flush": False,
        },
        "operations": [{"id": "views", "ops": ["view"], "placement": "host", "signature": {"ranks": [2]}}],
        "compositions": {
            "schema": "merlin.software_spec.compositions.v1",
            "status": "reviewed",
            "grading": "placement_and_instruction_audit",
            "chains": [chain, dict(chain, id="second_chain"), dict(chain, id="missing_chain")],
        },
    }
    spec_path = tmp_path / "software-spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    record = {"schema": L.SPEC_VALIDATION_SCHEMA, "cases": []}
    record["sha256"] = digest(record)
    record_path = tmp_path / "validation.json"
    record_path.write_text(json.dumps(record))
    return dict(software_spec=spec_path, spec_validation=record_path)


def _entry(*, legacy=False, aliases=()):
    entry = {
        "name": "SY_composition_authored_chain_q_tail" if legacy else f"SY_composition_{'a' * 64}_{'b' * 64}",
        "op": "composition_program",
        "source_role": "derived_sweep",
        "generalization": {
            "composition_chain": {"id": "authored_chain" if legacy else "c" * 64},
            "quantization_scheme": {"id": "q"},
        },
    }
    if not legacy:
        entry["generalization"]["fusion_validation"] = {
            "candidate_id": "a" * 64,
            "case_ids": ["b" * 64],
            "record_sha256": "d" * 64,
            "role": "sampled_regime",
            "display_aliases": list(aliases),
        }
    return entry


@pytest.mark.parametrize("legacy", [False, True])
def test_declared_chain_and_capsule_rows_share_the_same_identity(inputs, legacy):
    entry = _entry(legacy=legacy, aliases=["authored_chain", "second_chain", "undeclared", "authored_chain"])
    original = copy.deepcopy(entry)
    candidate_coverage = coverage([entry])
    document = L.prepare(**inputs, capsules=[entry]).record()
    expected = ["authored_chain"] if legacy else ["authored_chain", "second_chain"]
    rows = {row["composition"]: row for row in document["compositions"]}
    for identity in expected:
        assert rows[identity]["evidence_state"] == "generated"
        assert rows[identity]["test_identity"]["capsules"] == [entry["name"]]
        assert rows[identity]["reference"]["capsules"]
    capsule = document["capsules"][0]
    assert capsule["name"] == entry["name"]
    assert capsule["covers"] == capsule["adds"] == [f"composition:{identity}" for identity in expected]
    assert capsule["justification"]
    assert rows["missing_chain"]["evidence_state"] == "not_evidenced"
    assert set(document["omissions"]["compositions_without_capsule"]) == set(rows) - set(expected)
    assert document["omissions"]["unjustified_capsules"] == []
    assert entry == original
    assert coverage([entry]) == candidate_coverage


@pytest.mark.parametrize("aliases", [[], ["undeclared"]])
def test_validated_entries_do_not_fall_back_to_a_legacy_looking_name(inputs, aliases):
    entry = _entry(aliases=aliases)
    entry["name"] = "SY_composition_authored_chain_q_tail"
    document = L.prepare(**inputs, capsules=[entry]).record()
    assert document["capsules"][0]["covers"] == []
    assert all(row["evidence_state"] == "not_evidenced" for row in document["compositions"])


def test_compositions_off_retains_seal3b_exclusion(inputs):
    document = L.prepare(**inputs, capsules=[], workload_spec={"synthesize_compositions": False}).record()
    assert all(row["evidence_state"] == "out_of_scope" for row in document["compositions"])
    assert document["omissions"]["compositions_without_capsule"] == []
