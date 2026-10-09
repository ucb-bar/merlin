"""Actual fresh batch observations and complete original transport refusal controls."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_schema_batch as B
from merlin_experiments.phase0 import original_schema_defaults as D

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen import torch_schema_batch_observer as N

_spec = importlib.util.spec_from_file_location(
    "batch_original_reference_fixtures", Path(__file__).with_name("original_reference_fixtures.py")
)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
live_originals = F.live_originals


@pytest.fixture(scope="module")
def observations(live_originals, tmp_path_factory):
    intake, basis, _ = live_originals
    owner = tmp_path_factory.mktemp("original-schema-batch")
    record = intake.record()
    batched = B.observe_members(schema_record=record, basis=basis, destination=owner / "batch", version=2)
    legacy = D.observe_members(schema_record=record, basis=basis, destination=owner / "legacy", version=2)
    return intake, basis, owner, batched, legacy


def test_actual_batch_matches_every_full_original_schema_and_legacy_default(observations):
    intake, basis, owner, batch, legacy = observations
    assert len(batch) == len(legacy) == len(basis.graph_sources) == 8
    assert [row["graph_path"] for row in batch] == [row["graph_path"] for row in legacy]
    for row, old in zip(batch, legacy, strict=True):
        assert D.verify_member(row, schema_record=intake.record(), version=2, transport=B.TRANSPORT) == D.verify_member(
            old, schema_record=intake.record(), version=2
        )
        assert Path(row["request"]).read_bytes() != b""
        assert Path(row["observation"]).stat().st_mode & 0o077 == 0
    invocations = {row["invocation"] for row in batch}
    assert len(invocations) == 1
    actual = I.require_environment(Path(next(iter(invocations))), environment=B.ENVIRONMENT)
    assert actual["stage"] == "native_original_schema_batch"
    assert actual["returncode"] == 0
    B.verify_members(batch, schema_record=intake.record(), basis=basis, destination=owner / "batch", version=2)
    assert len(list((owner / "batch" / "invocations").glob("*/invocation.json"))) == 1
    assert len(list((owner / "legacy").glob("*/invocations/*/invocation.json"))) == 8


def test_legacy_consumers_cannot_switch_transport_from_saved_batch_metadata(observations):
    intake, _, _, batch, _ = observations
    with pytest.raises(ValueError, match="explicitly selected observation transport"):
        D.verify_member(batch[0], schema_record=intake.record(), version=2)


@pytest.mark.parametrize("defect", ["missing", "reordered", "duplicated", "identity", "request", "schema", "defaults"])
def test_full_actual_native_output_cannot_hide_changed_batch_rows(observations, defect):
    intake, _, _, rows, _ = observations
    request = json.loads(Path(rows[0]["batch_request"]).read_bytes())
    output = json.loads(Path(rows[0]["batch_observation"]).read_bytes())
    if defect == "missing":
        output["rows"].pop()
    elif defect == "reordered":
        output["rows"].reverse()
    elif defect == "duplicated":
        output["rows"][1] = copy.deepcopy(output["rows"][0])
    elif defect == "identity":
        output["rows"][0]["identity"] = request["rows"][1]["identity"]
    elif defect == "request":
        output["rows"][0]["request_sha256"] = "0" * 64
    elif defect == "schema":
        output["rows"][0]["schema_observation"]["rows"].pop()
    else:
        output["rows"][0]["defaults_observation"]["rows"].pop()
    with pytest.raises(ValueError):
        B._check_observation(output, request, schema_record=intake.record())


@pytest.mark.parametrize("defect", ["partial", "order", "transport", "output", "owner", "invocation"])
def test_saved_extracted_rows_cannot_replace_complete_original_transport(observations, defect):
    intake, basis, owner, batch, legacy = observations
    rows = copy.deepcopy(batch)
    if defect == "partial":
        rows.pop()
    elif defect == "order":
        rows.reverse()
    elif defect == "transport":
        rows[0]["transport"] = "per_member"
    elif defect == "output":
        rows[0]["observation"] = legacy[0]["observation"]
    elif defect == "owner":
        rows[0]["batch_request"] = legacy[0]["request"]
    else:
        rows[0]["invocation"] = legacy[0]["invocation"]
    with pytest.raises((ValueError, KeyError)):
        B.verify_members(rows, schema_record=intake.record(), basis=basis, destination=owner / "batch", version=2)


def test_modified_full_native_stdout_and_extracted_output_refuse(observations):
    intake, _, _, rows, _ = observations
    for path in (Path(rows[0]["batch_observation"]), Path(rows[0]["observation"])):
        previous = path.read_bytes()
        try:
            path.write_text("{}")
            with pytest.raises((ValueError, KeyError)):
                B.verify_member(rows[0], schema_record=intake.record(), version=2)
        finally:
            path.write_bytes(previous)
        B.verify_member(rows[0], schema_record=intake.record(), version=2)


def test_extracted_output_alias_cannot_replace_original_private_product(observations, tmp_path):
    intake, _, _, rows, _ = observations
    path = Path(rows[0]["observation"])
    previous = path.read_bytes()
    substitute = tmp_path / "same-default-values.json"
    substitute.write_bytes(previous)
    try:
        path.unlink()
        path.symlink_to(substitute)
        with pytest.raises(ValueError, match="ordinary complete private products"):
            B.verify_member(rows[0], schema_record=intake.record(), version=2)
    finally:
        path.unlink()
        path.write_bytes(previous)
        path.chmod(0o600)
    B.verify_member(rows[0], schema_record=intake.record(), version=2)


@pytest.mark.parametrize("defect", ["duplicate", "missing_identity", "extra", "empty"])
def test_actual_fixed_native_reader_refuses_malformed_requests_before_observation(observations, tmp_path, defect):
    intake, _, _, rows, _ = observations
    request = json.loads(Path(rows[0]["batch_request"]).read_bytes())
    if defect == "duplicate":
        request["rows"][1] = copy.deepcopy(request["rows"][0])
    elif defect == "missing_identity":
        request["rows"][0].pop("identity")
    elif defect == "extra":
        request["cache"] = "previous-observation"
    else:
        request["rows"] = []
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    selected = B._selection(Path(intake.record()["selection_path"]).read_bytes())
    observer = module_source_path("merlin.targetgen.torch_schema_batch_observer")
    result = I.run(
        [selected["python"], "-I", str(observer), str(path), selected["canonical_source"]["path"]],
        directory=tmp_path,
        stage="batch_malformed_request_control",
        inputs=(observer, path, Path(selected["canonical_source"]["path"])),
        env=B.ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert b"ValueError" in result.stderr and not result.stdout


def test_opt_in_reference_roster_uses_one_fresh_batch_and_preserves_all_original_numerical_rows(
    live_originals, tmp_path
):
    intake, basis, spec = live_originals
    selected = F.selection(intake, basis)
    selected.update(schema=P.BATCH_SCHEMA, native_observations=B.TRANSPORT)
    selection = tmp_path / "selection.json"
    F.write(selection, selected)
    before = spec.read_bytes()
    result = R.prepare(schema_intake=intake, basis=basis, selection=selection, destination=tmp_path / "roster")
    record = result.record()
    assert record["schema"] == R.BATCH_SCHEMA
    assert len(record["members"]) == 16
    assert len({row["invocation"] for row in record["defaults"]}) == 1
    checked = [row for row in record["members"] if row["state"] == "reference_checked"]
    assert len(checked) == 12
    assert {row["original_member_id"] for row in record["members"] if row["state"] == "unavailable"} == {
        "matmul_f16",
        "add_nonunit",
    }
    for row in checked:
        compare = json.loads(Path(row["products"]["comparison"]["path"]).read_bytes())
        assert compare["passed"] and compare["mismatches"] == []
        assert row["required_unknowns"] == list(R._UNKNOWN)
    assert spec.read_bytes() == before
    for defect in ("schema", "legacy", "missing", "order"):
        altered = copy.deepcopy(record)
        if defect == "schema":
            altered["schema"] = R.SCHEMA
        elif defect == "legacy":
            altered["defaults"][0].pop("transport")
        elif defect == "missing":
            altered["defaults"].pop()
        else:
            altered["defaults"].reverse()
        with pytest.raises((ValueError, KeyError)):
            R.verify(altered, schema_intake=intake, basis=basis, selection=selection)


def test_original_v1_reference_still_runs_all_originals_without_batch_selection(live_originals, tmp_path):
    intake, basis, _ = live_originals
    selection = tmp_path / "selection.json"
    F.write(selection, F.selection(intake, basis))
    result = R.prepare(schema_intake=intake, basis=basis, selection=selection, destination=tmp_path / "legacy")
    record = result.record()
    assert record["schema"] == R.SCHEMA
    assert len(record["members"]) == 16
    assert sum(row["state"] == "reference_checked" for row in record["members"]) == 12
    assert all("transport" not in row for row in record["defaults"])
    assert len({row["invocation"] for row in record["defaults"]}) == 8


@pytest.mark.parametrize("defect", ["missing", "unknown", "old_extra"])
def test_batch_is_explicit_versioned_selection_without_changing_legacy(live_originals, defect):
    intake, basis, _ = live_originals
    legacy = F.selection(intake, basis)
    assert P.transport(legacy) == "per_member"
    selected = copy.deepcopy(legacy)
    selected.update(schema=P.BATCH_SCHEMA, native_observations=B.TRANSPORT)
    assert P.transport(selected) == "batch.v1"
    if defect == "missing":
        selected.pop("native_observations")
    elif defect == "unknown":
        selected["native_observations"] = "cached"
    else:
        selected["schema"] = P.SCHEMA
    with pytest.raises(ValueError):
        P.validate(selected)


def test_batch_digest_includes_every_original_request_slot():
    request = {"schema": N.REQUEST_SCHEMA, "rows": [{"identity": "one", "schema_request": {}, "defaults_request": {}}]}
    N.validate(request)
    altered = copy.deepcopy(request)
    altered["rows"][0]["defaults_request"]["graph_sha256"] = "different"
    assert N.digest(altered) != N.digest(request)
