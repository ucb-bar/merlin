"""Structural opt-in declaration checks, not execution/effect qualification."""

import copy

import pytest

from merlin.targetgen import shared_pass_contract as S


def manifest():
    requirement = S.shared_pass_requirement()
    features = sorted(set(S.FOUNDATIONS).union(*(set(needs) for needs in S.FAMILY_NEEDS.values())))
    boundaries = {
        name: {
            "ir": f"selected_{name}_mlir",
            "command": f"consume_{name}",
            "contracts": {kind: f"neutral.{kind}.v1" for kind in ("semantics", "effects", "capabilities")},
            "features": features.copy(),
        }
        for name in ("computation", "device", "emission")
    }
    boundaries["device"].update(ir="selected_target_mlir", command="emit_target_bundle")
    return {
        "target": "neutral",
        "target_ir_consumer": {
            "schema": "merlin.target_ir_consumer.v1",
            "command": "emit_target_bundle",
            "input": "selected_target_mlir",
            "output": "command_buffer_and_target_artifact",
        },
        "commands": {
            boundary["command"]: {"argv": ["{tool}", "{input_mlir}", "{output_json}"]}
            for boundary in boundaries.values()
        },
        "shared_pass_contract": {
            "schema": "merlin.shared_pass_contract.v1",
            "boundaries": boundaries,
            "pipeline": requirement["pipeline"],
            "controls": {
                "default_policy": list(S.FAMILY_NEEDS),
                "choices": {name: {"enabled": [False, True], "max_growth_ops": 16} for name in S.FAMILY_NEEDS},
                "max_repairs": 1,
                "retain_original": True,
            },
            "families": [
                {
                    "name": name,
                    "status": "supported",
                    "reason": "Declared neutral interface; qualification is separate",
                    "boundary": "computation",
                    "requirements": list(needs),
                }
                for name, needs in S.FAMILY_NEEDS.items()
            ],
        },
    }


def check(value, requirement=None):
    return S.validate_shared_pass_contract(value, requirement=requirement or S.shared_pass_requirement())


def test_absent_requirement_preserves_legacy_without_inspection():
    class Unreadable:
        def get(self, *args):
            raise AssertionError("legacy must not inspect the package")

    assert S.validate_shared_pass_contract(Unreadable()) is None


def test_valid_declaration_is_detached_and_covers_exact_nine_families():
    value = manifest()
    result = check(value)
    assert len(result["contract"]["families"]) == 9
    assert result["target_consumer"]["declaration"] == value["target_ir_consumer"]
    assert result["commands"]["device"] == value["commands"]["emit_target_bundle"]
    value["shared_pass_contract"]["families"][0]["reason"] = "changed"
    assert result["contract"]["families"][0]["reason"] != "changed"
    requirement = S.shared_pass_requirement()
    requirement["pipeline"]["context"].clear()
    assert len(S.shared_pass_requirement()["pipeline"]["context"]) == 5


@pytest.mark.parametrize("status", ["unknown", "unsupported"])
def test_unknown_integration_is_explicit_but_cannot_satisfy_original_supported_requirement(status):
    value = manifest()
    row = value["shared_pass_contract"]["families"][0]
    row["status"] = status
    controls = value["shared_pass_contract"]["controls"]
    controls["default_policy"].remove(row["name"])
    controls["choices"][row["name"]]["enabled"] = [False]
    assert check(value)["contract"]["families"][0]["status"] == status
    with pytest.raises(ValueError, match="original requirement"):
        check(value, S.shared_pass_requirement(required_supported=[row["name"]]))


@pytest.mark.parametrize("boundary", ["computation", "device", "emission"])
def test_each_named_boundary_and_actual_selected_ir_command_is_required(boundary):
    value = manifest()
    value["shared_pass_contract"]["boundaries"].pop(boundary)
    with pytest.raises(ValueError, match="boundaries"):
        check(value)
    value = manifest()
    command = value["shared_pass_contract"]["boundaries"][boundary]["command"]
    value["commands"][command]["argv"] = ["{tool}", "hardcoded-original.mlir"]
    with pytest.raises(ValueError, match="consume selected IR"):
        check(value)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "schema",
        "boundary_schema",
        "duplicate_ir",
        "missing_contract",
        "unknown_feature",
        "missing_foundations",
        "missing_family",
        "duplicate_family",
        "unknown_family",
        "missing_reason",
        "missing_needs",
        "missing_boundary_needs",
        "unknown_status",
        "unknown_boundary",
        "unknown_control",
        "enabled_unknown",
        "integer_bool",
        "growth",
        "repairs",
        "retain",
        "disabled_default",
        "duplicate_order",
        "reverse_order",
        "omitted_supported",
        "misplaced_foundation",
        "pipeline_context",
        "pipeline_invalidation",
        "pipeline_bool",
        "target_schema",
        "target_output",
    ],
)
def test_malformed_or_unbounded_optin_refuses(fault):
    value = manifest()
    declaration = value["shared_pass_contract"]
    rows, boundaries, controls = (declaration[key] for key in ("families", "boundaries", "controls"))
    if fault == "missing":
        value.pop("shared_pass_contract")
    elif fault == "schema":
        declaration["schema"] = "merlin.shared_pass_contract.v99"
    elif fault == "boundary_schema":
        boundaries["computation"]["unreviewed"] = True
    elif fault == "duplicate_ir":
        boundaries["emission"]["ir"] = boundaries["computation"]["ir"]
    elif fault == "missing_contract":
        boundaries["computation"]["contracts"].pop("effects")
    elif fault == "unknown_feature":
        boundaries["device"]["features"].append("unreviewed_semantics")
    elif fault == "missing_foundations":
        for boundary in boundaries.values():
            boundary["features"].remove("final_program_limits")
    elif fault == "missing_family":
        rows.pop()
    elif fault == "duplicate_family":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "unknown_family":
        rows[0]["name"] = "unreviewed_pass"
    elif fault == "missing_reason":
        rows[0]["reason"] = " "
    elif fault == "missing_needs":
        rows[0]["requirements"].pop()
    elif fault == "missing_boundary_needs":
        boundaries["computation"]["features"].remove("ownership")
    elif fault == "unknown_status":
        rows[0]["status"] = "qualified"
    elif fault == "unknown_boundary":
        rows[0]["boundary"] = "opaque"
    elif fault == "unknown_control":
        controls["force"] = True
    elif fault == "enabled_unknown":
        rows[-1]["status"] = "unknown"
    elif fault == "integer_bool":
        controls["choices"][rows[0]["name"]]["enabled"] = [0, 1]
    elif fault == "growth":
        controls["choices"][rows[0]["name"]]["max_growth_ops"] = 65537
    elif fault == "repairs":
        controls["max_repairs"] = True
    elif fault == "retain":
        controls["retain_original"] = False
    elif fault == "disabled_default":
        controls["choices"][rows[0]["name"]]["enabled"] = [False]
    elif fault == "duplicate_order":
        controls["default_policy"] *= 2
    elif fault == "reverse_order":
        rows[0]["boundary"] = "emission"
        controls["default_policy"] = [rows[0]["name"], rows[1]["name"]]
    elif fault == "omitted_supported":
        controls["default_policy"].pop()
    elif fault == "misplaced_foundation":
        boundaries["device"]["features"].remove("dependency_scheduling")
    elif fault == "pipeline_context":
        declaration["pipeline"]["context"].pop()
    elif fault == "pipeline_invalidation":
        declaration["pipeline"]["requires_and_invalidates"] = False
    elif fault == "pipeline_bool":
        declaration["pipeline"]["source_sha256"] = 1
    elif fault == "target_schema":
        value["target_ir_consumer"]["schema"] = "merlin.target_ir_consumer.v99"
    elif fault == "target_output":
        value["commands"]["emit_target_bundle"]["argv"].pop()
    with pytest.raises(ValueError):
        check(value)


@pytest.mark.parametrize(
    "fault", ["schema", "families", "required_duplicate", "required_unknown", "limits", "raised_limit"]
)
def test_original_requirement_is_versioned_closed_and_cannot_raise_structural_ceiling(fault):
    requirement = S.shared_pass_requirement()
    if fault == "schema":
        requirement["schema"] = "merlin.shared_pass_requirement.v99"
    elif fault == "families":
        requirement["families"].pop("padding_gather")
    elif fault == "required_duplicate":
        requirement["required_supported"] = ["producer_packing"] * 2
    elif fault == "required_unknown":
        requirement["required_supported"] = ["unknown"]
    elif fault == "limits":
        requirement["limits"]["max_repairs"] = -1
    elif fault == "raised_limit":
        requirement["limits"]["max_growth_ops"] += 1
    with pytest.raises(ValueError):
        check(manifest(), requirement)


def test_original_requirement_may_lower_control_ceiling():
    requirement = S.shared_pass_requirement()
    requirement["limits"]["max_growth_ops"] = 8
    with pytest.raises(ValueError, match="max_growth_ops"):
        check(manifest(), requirement)


def test_public_requirement_validator_detaches_and_rejects_unknown_schema():
    original = S.shared_pass_requirement(required_supported=("padding_gather",))
    checked = S.validate_shared_pass_requirement(original)
    original["required_supported"].clear()
    assert checked["required_supported"] == ["padding_gather"]
    original["schema"] = "merlin.shared_pass_requirement.v99"
    with pytest.raises(ValueError, match="requirement schema"):
        S.validate_shared_pass_requirement(original)


@pytest.mark.parametrize("target", ["neutral_a", "different_geometry_and_execution"])
def test_target_identity_is_data_not_a_shared_pass_legality_selector(target):
    value = manifest()
    value["target"] = target
    assert check(value)["target_consumer"]["target"] == target
