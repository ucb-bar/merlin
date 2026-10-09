"""Source performance requests remain closed data with pending later phases."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import test_component_source_performance as source_fixtures
import test_declared_phase0_run as request_fixtures
import yaml
from merlin_experiments.phase0 import declared_run as D

declared = request_fixtures.declared
automatic = source_fixtures.automatic
independent = source_fixtures.independent
selected = source_fixtures.selected


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def performance_request(declared, automatic, tmp_path):
    request, path = declared
    request = copy.deepcopy(request)
    options = source_fixtures.options_for(automatic, tmp_path)
    original = yaml.safe_load(options["recipe"].read_bytes())["component_performance"]
    objective_path = tmp_path / "source-objectives.json"
    objective_path.write_text(json.dumps({key: original[key] for key in ("schema", "status", "objectives")}))
    request.update(
        schema=D.PERFORMANCE_SCHEMA,
        release_purpose="performance_campaign",
        source_performance={
            "schema": source_fixtures.P.SCHEMA,
            "objectives": pin(objective_path),
            "sweeps": pin(options["performance_template"]),
        },
    )
    path.write_text(json.dumps(request))
    return request, path


def test_explicit_source_performance_request_keeps_old_versions_closed(performance_request):
    request, _ = performance_request
    assert D.validate(request) == request
    paths, objectives, sweeps = D._source_performance_inputs(request["source_performance"], forbidden=())
    assert len(objectives) == len(sweeps["sweeps"]) == 1
    assert set(paths) == {"objectives", "sweeps"}
    for old in (D.SCHEMA, D.BRIDGE_SCHEMA, D.REQUIREMENT_SCHEMA):
        request["schema"] = old
        with pytest.raises(ValueError, match="closed explicit"):
            D.validate(request)


@pytest.mark.parametrize(
    "change", ["absent_purpose", "other_purpose", "absent_mode", "wrong_mode", "saved_authority", "extra_factory"]
)
def test_request_cannot_import_performance_authority_or_implicit_modes(performance_request, change):
    request, _ = performance_request
    if change == "absent_purpose":
        del request["release_purpose"]
    elif change == "other_purpose":
        request["release_purpose"] = "source_preparation"
    elif change == "absent_mode":
        del request["source_performance"]
    elif change == "wrong_mode":
        request["source_performance"]["schema"] = "accepted"
    elif change == "saved_authority":
        request["source_performance"]["hardware_guard_link"] = "established"
    else:
        request["source_performance"]["factory"] = "arbitrary.module:callback"
    with pytest.raises(ValueError):
        D.validate(request)


@pytest.mark.parametrize(
    "change",
    [
        "empty_objectives",
        "extra_authority",
        "empty_sweeps",
        "tile_axis",
        "encoding",
        "foreign_family",
        "authored_capsule",
    ],
)
def test_original_source_declaration_refuses_unsupported_or_saved_inputs_before_issuer(
    performance_request, tmp_path, monkeypatch, change
):
    request, request_path = performance_request
    key = "objectives" if change in {"empty_objectives", "extra_authority"} else "sweeps"
    path = Path(request["source_performance"][key]["path"])
    value = yaml.safe_load(path.read_bytes())
    if change == "empty_objectives":
        value["objectives"] = []
    elif change == "extra_authority":
        value["hardware"] = {"status": "accepted"}
    elif change == "empty_sweeps":
        value["sweeps"] = []
    elif change == "tile_axis":
        value["sweeps"][0]["axes"]["R"] = ["tile", "tile+1"]
    elif change == "encoding":
        value["sweeps"][0]["base"]["encoding"] = "supplied-device-encoding"
    elif change == "foreign_family":
        value["sweeps"][0]["id"] = "foreign"
    else:
        value["capsules"] = [{"name": "authored"}]
    path.write_text(json.dumps(value))
    request["source_performance"][key] = pin(path)
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(D, "issue_independent_hardware_intake", lambda **_: pytest.fail("bad source reached issuer"))
    with pytest.raises(ValueError):
        D.run(request_path, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", ["changed", "forbidden", "alias", "parent_component"])
def test_exact_source_pin_membership_refuses_before_issuer(performance_request, tmp_path, monkeypatch, change):
    request, request_path = performance_request
    path = Path(request["source_performance"]["sweeps"]["path"])
    if change == "changed":
        path.write_bytes(b"changed source\n")
    elif change == "forbidden":
        request["forbidden_roots"].append(str(path))
    elif change == "alias":
        alias = tmp_path / "sweep-alias"
        alias.symlink_to(path)
        request["source_performance"]["sweeps"]["path"] = str(alias)
    else:
        request["source_performance"]["sweeps"]["path"] = str(tmp_path / "missing" / ".." / path.name)
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(D, "issue_independent_hardware_intake", lambda **_: pytest.fail("bad pin reached issuer"))
    with pytest.raises(ValueError):
        D.run(request_path, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_complete_generated_source_products_include_development_and_pending_rosters(automatic, tmp_path):
    contracts, inputs = source_fixtures.run(source_fixtures.options_for(automatic, tmp_path))
    root = inputs["root"]
    receipt_path = root / "_evidence/coverage/generation.json"
    receipt = json.loads(receipt_path.read_bytes())
    paths = [root / row["member"] for row in receipt["capsule_commitments"]]
    count, checked, _ = D._verify_source_performance_products(
        root, inputs["coverage"], paths, target="fixture", hardware=inputs["hardware"], software=inputs["software"]
    )
    assert checked == contracts
    assert count == sum(contracts["source_checked_counts"].values())
    assert checked["source_checked_counts"]["development"] == 4
    assert checked["mandatory_missing_ids"] and checked["hardware_guard_link"] == "not_established"
    receipt["capsule_commitments"].pop()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete requested or written membership"):
        D._verify_source_performance_products(
            root, inputs["coverage"], paths, target="fixture", hardware=inputs["hardware"], software=inputs["software"]
        )
