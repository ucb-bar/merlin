"""The selected instruction description retains exact semantics and unknowns."""

import copy
import hashlib
import pickle

import pytest
import yaml

from merlin.targetgen.instruction_semantics import (
    UNKNOWN,
    load_instruction_semantics,
    normalize_instruction_semantics,
    validate_normalized_instruction_model,
)


def _selected_inputs():
    software = {
        "schema": "merlin.software_spec.v1",
        "target": "fixture_target",
        "status": "reviewed",
        "operations": [
            {"id": "elementwise_add", "placement": "accelerator", "signature": {"dtypes": ["i32"], "ranks": [1]}}
        ],
    }
    facts = {"facts": {"target": "fixture_target", "derived": {"decode": "present"}}}
    return software, facts


def _description():
    return {
        "schema": "merlin.instruction_semantics.v1",
        "target": "fixture_target",
        "memory_spaces": {
            "scratch": {"capacity_bytes": 4096, "alignment_bytes": 16, "banks": 2, "bank_width_bytes": 16}
        },
        "instructions": [
            {
                "id": "add_four",
                "software_operation": "elementwise_add",
                "operands": [
                    {"name": "lhs", "type": "tensor<4xi32>"},
                    {"name": "rhs", "type": "tensor<4xi32>"},
                ],
                "results": [{"name": "out", "type": "tensor<4xi32>"}],
                "computation": {
                    "kind": "linalg.generic",
                    "indexing_maps": ["affine_map<(d0) -> (d0)>"] * 3,
                    "iterator_types": ["parallel"],
                    "scalar_body": {
                        "arguments": ["i32", "i32", "i32"],
                        "operations": [
                            {
                                "op": "arith.addi",
                                "operands": ["arg:0", "arg:1"],
                                "results": ["i32"],
                                "attributes": {},
                                "effects": [],
                                "regions": 0,
                            }
                        ],
                        "yields": ["op:0:0"],
                        "effects": [],
                    },
                },
                "parameters": {"lanes": {"type": "integer", "choices": [4, 8]}},
                "constraints": {
                    "operand_spaces": {"lhs": "scratch", "rhs": "scratch"},
                    "result_spaces": {"out": "scratch"},
                    "distinct_banks": ["lhs", "rhs"],
                },
                "effects": [
                    {
                        "kind": "read",
                        "space": "scratch",
                        "operand": "lhs",
                        "range": {"offset_bytes": 0, "length_bytes": 16},
                    },
                    {
                        "kind": "read",
                        "space": "scratch",
                        "operand": "rhs",
                        "range": {"offset_bytes": 0, "length_bytes": 16},
                    },
                    {
                        "kind": "write",
                        "space": "scratch",
                        "result": "out",
                        "range": {"offset_bytes": 0, "length_bytes": 16},
                    },
                ],
            }
        ],
    }


def test_normalization_is_deterministic_and_preserves_typed_semantics(monkeypatch):
    from merlin.target.spec import computation, instructions, signature
    from merlin.targetgen import instruction_semantics as legacy

    assert legacy is instructions
    assert legacy._computation is computation._computation
    assert legacy._scalar_body is computation._scalar_body
    assert legacy._software_signature_unknowns is signature._software_signature_unknowns
    assert pickle.loads(pickle.dumps(normalize_instruction_semantics)) is instructions.normalize_instruction_semantics
    calls = []
    original_computation = instructions._computation

    def observed_computation(*args, **kwargs):
        calls.append(True)
        return original_computation(*args, **kwargs)

    monkeypatch.setattr(legacy, "_computation", observed_computation)
    software, facts = _selected_inputs()
    description = _description()
    original = copy.deepcopy(description)
    one = normalize_instruction_semantics(description, software_spec=software, rtl_facts=facts, target="fixture_target")
    two = normalize_instruction_semantics(description, software_spec=software, rtl_facts=facts, target="fixture_target")
    assert one == two
    assert calls
    assert description == original
    assert one["status"] == "described"
    assert one["unknowns"] == []
    assert len(one["canonical_sha256"]) == 64
    row = one["instructions"][0]
    assert row["software_operation"] == "elementwise_add"
    assert row["computation"]["scalar_body"]["operations"][0]["operands"] == ["arg:0", "arg:1"]
    assert row["effects"][2]["range"] == {"offset_bytes": 0, "length_bytes": 16}
    assert row["constraints"]["distinct_banks"] == ["lhs", "rhs"]
    assert (
        validate_normalized_instruction_model(
            one,
            expected_target="fixture_target",
            expected_source_sha256=one["source_sha256"],
        )
        == one
    )


def test_exact_selected_bytes_identity(tmp_path):
    software, facts = _selected_inputs()
    sw_bytes = yaml.safe_dump(software).encode()
    fact_bytes = yaml.safe_dump(facts).encode()
    description = yaml.safe_dump(_description()).encode()
    path = tmp_path / "instruction_semantics.yaml"
    path.write_bytes(description)
    model = load_instruction_semantics(path, software_spec=sw_bytes, rtl_facts=fact_bytes, target="fixture_target")
    assert model["source_sha256"] == hashlib.sha256(description).hexdigest()
    assert model["software_spec_sha256"] == hashlib.sha256(sw_bytes).hexdigest()
    assert model["circt_facts_sha256"] == hashlib.sha256(fact_bytes).hexdigest()


def test_raw_selected_software_bytes_match_normalized_spec():
    from merlin.targetgen.software_spec import validate_software_spec

    authored = {
        "schema": "merlin.software_spec.v1",
        "target": "fixture_target",
        "status": "reviewed",
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "i32",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
        },
        "operations": {
            "elementwise_add": {
                "ops": ["aten.add.Tensor"],
                "placement": "accelerator",
                "dtypes": ["i32"],
            }
        },
    }
    raw_software = yaml.safe_dump(authored).encode()
    raw_facts = yaml.safe_dump(_selected_inputs()[1]).encode()
    normalized_software = validate_software_spec(authored, target="fixture_target")
    model = normalize_instruction_semantics(
        _description(),
        software_spec=normalized_software,
        rtl_facts=_selected_inputs()[1],
        target="fixture_target",
        software_source_bytes=raw_software,
        rtl_source_bytes=raw_facts,
    )
    assert model["status"] == UNKNOWN  # the source's ATen selector still needs provenance matching
    assert model["software_spec_sha256"] == hashlib.sha256(raw_software).hexdigest()
    assert model["circt_facts_sha256"] == hashlib.sha256(raw_facts).hexdigest()
    with pytest.raises(ValueError, match="source bytes"):
        normalize_instruction_semantics(
            _description(),
            software_spec=normalized_software,
            rtl_facts=_selected_inputs()[1],
            target="fixture_target",
            rtl_source_bytes=b"facts: {target: foreign}\n",
        )


def test_missing_semantics_and_memory_facts_remain_unknown():
    software, facts = _selected_inputs()
    description = _description()
    description["memory_spaces"]["scratch"].pop("alignment_bytes")
    row = description["instructions"][0]
    row.pop("software_operation")
    row["computation"] = {"status": UNKNOWN, "reason": "numerical effect not established"}
    row.pop("effects")
    model = normalize_instruction_semantics(
        description, software_spec=software, rtl_facts=facts, target="fixture_target"
    )
    assert model["status"] == UNKNOWN
    assert model["memory_spaces"]["scratch"]["alignment_bytes"] == UNKNOWN
    assert "memory_spaces.scratch.alignment_bytes" in model["unknowns"]
    assert model["instructions"][0]["computation"]["status"] == UNKNOWN
    assert "software_operation_unlinked" in model["instructions"][0]["unknowns"]
    assert model["instructions"][0]["effects"] is None


def test_constrained_storage_without_matching_effect_is_unknown():
    software, facts = _selected_inputs()
    description = _description()
    description["instructions"][0]["effects"] = description["instructions"][0]["effects"][:1]
    model = normalize_instruction_semantics(
        description, software_spec=software, rtl_facts=facts, target="fixture_target"
    )
    assert model["status"] == UNKNOWN
    assert "effects.uncovered_read.rhs" in model["instructions"][0]["unknowns"]
    assert "effects.uncovered_write.out" in model["instructions"][0]["unknowns"]
    assert validate_normalized_instruction_model(model) == model


def test_sw_dtype_and_rank_mismatches_cannot_describe_instruction():
    software, facts = _selected_inputs()
    software["operations"][0]["signature"] = {"operand_dtypes": ["int8"], "accumulator_dtype": "i32", "ranks": [2]}
    model = normalize_instruction_semantics(
        _description(), software_spec=software, rtl_facts=facts, target="fixture_target"
    )
    unknowns = model["instructions"][0]["unknowns"]
    assert model["status"] == UNKNOWN
    assert "software_signature_operand_dtype_mismatch.0" in unknowns
    assert "software_signature_operand_rank_mismatch.0" in unknowns
    assert "software_signature_result_rank_mismatch.0" in unknowns
    assert validate_normalized_instruction_model(model) == model


def test_invalid_software_link_and_scalar_reference_refuse():
    software, facts = _selected_inputs()
    description = _description()
    description["instructions"][0]["software_operation"] = "undeclared"
    with pytest.raises(ValueError, match="selected SW operation ID"):
        normalize_instruction_semantics(description, software_spec=software, rtl_facts=facts, target="fixture_target")
    description = _description()
    description["instructions"][0]["computation"]["scalar_body"]["operations"][0]["operands"] = ["op:1:0"]
    with pytest.raises(ValueError, match="preceding scalar values"):
        normalize_instruction_semantics(description, software_spec=software, rtl_facts=facts, target="fixture_target")


def test_captured_scalar_values_use_frontend_reference_shape():
    software, facts = _selected_inputs()
    description = _description()
    body = description["instructions"][0]["computation"]["scalar_body"]
    body["captures"] = [{"source": ("arg", 3), "shape": [], "dtype": "i32", "const_value": 2.0}]
    body["operations"][0]["operands"] = ["arg:0", "capture:0"]
    model = normalize_instruction_semantics(
        description, software_spec=software, rtl_facts=facts, target="fixture_target"
    )
    normalized_body = model["instructions"][0]["computation"]["scalar_body"]
    assert normalized_body["captures"] == [{"source": ["arg", 3], "shape": [], "dtype": "i32", "const_value": 2.0}]
    assert normalized_body["operations"][0]["operands"] == ["arg:0", "capture:0"]


def test_foreign_target_and_missing_fact_bundle_refuse_or_report():
    software, facts = _selected_inputs()
    description = _description()
    description["target"] = "foreign"
    with pytest.raises(ValueError, match="target differs"):
        normalize_instruction_semantics(description, software_spec=software, rtl_facts=facts, target="fixture_target")
    model = normalize_instruction_semantics(
        _description(), software_spec=software, rtl_facts=None, target="fixture_target"
    )
    assert model["status"] == UNKNOWN
    assert model["circt_facts_sha256"] is None
    assert "selected_circt_facts_missing" in model["unknowns"]
    empty_observation = normalize_instruction_semantics(
        _description(), software_spec={}, rtl_facts={}, target="fixture_target"
    )
    assert {"selected_software_spec_missing", "selected_circt_facts_missing"} <= set(empty_observation["unknowns"])


def test_frozen_model_validator_refuses_forged_described_status_and_digest():
    software, facts = _selected_inputs()
    unknown = normalize_instruction_semantics(
        _description(), software_spec=software, rtl_facts=None, target="fixture_target"
    )
    assert validate_normalized_instruction_model(unknown)["status"] == UNKNOWN
    forged = copy.deepcopy(unknown)
    forged["status"] = "described"
    with pytest.raises(ValueError, match="canonical_sha256"):
        validate_normalized_instruction_model(forged)
    known = normalize_instruction_semantics(
        _description(), software_spec=software, rtl_facts=facts, target="fixture_target"
    )
    with pytest.raises(ValueError, match="selected bytes"):
        validate_normalized_instruction_model(known, expected_circt_facts_sha256="0" * 64)
