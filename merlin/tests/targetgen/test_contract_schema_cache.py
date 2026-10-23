"""Repeated contract validation retains source invalidation and error identity."""

import json

import jsonschema
import pytest

from merlin.targetgen.contract.schemas import ContractViolation, validate


def test_changed_schema_bytes_and_malformed_schema_never_use_old_acceptance(tmp_path):
    path = tmp_path / "schemas" / "fixture.schema.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"type": "integer"}))
    validate(7, "fixture", contract=tmp_path)
    with pytest.raises(ContractViolation):
        validate("seven", "fixture", contract=tmp_path)
    path.write_text(json.dumps({"type": "string"}))
    validate("seven", "fixture", contract=tmp_path)
    with pytest.raises(ContractViolation):
        validate(7, "fixture", contract=tmp_path)
    path.write_text(json.dumps({"type": "not_a_json_schema_type"}))
    with pytest.raises(jsonschema.SchemaError):
        validate("seven", "fixture", contract=tmp_path)
    path.write_text(json.dumps({"type": "integer"}))
    validate(7, "fixture", contract=tmp_path)


def test_nested_failure_keeps_jsonschema_best_match_and_location(tmp_path):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}},
    }
    path = tmp_path / "schemas" / "fixture.schema.json"
    path.parent.mkdir()
    path.write_text(json.dumps(schema))
    data = {"value": [None]}
    with pytest.raises(jsonschema.ValidationError) as baseline:
        jsonschema.validate(data, schema)
    expected = baseline.value
    location = "/".join(str(v) for v in expected.absolute_path) or "<root>"
    for _ in range(2):
        with pytest.raises(ContractViolation) as observed:
            validate(data, "fixture", contract=tmp_path)
        assert str(observed.value) == f"fixture schema violation at {location}: {expected.message}"
