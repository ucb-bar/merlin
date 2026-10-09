"""Actual original framework→reference→upstream standard-IR joins, never release."""

import copy
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_roster as R
from merlin_experiments.phase0 import original_reference_standard_ir as S
from merlin_experiments.phase0 import original_standard_ir_plan as P
from merlin_experiments.phase0.original_call_sources import required_source_cohorts


def _fixtures():
    path = Path(__file__).with_name("original_reference_fixtures.py")
    spec = importlib.util.spec_from_file_location("standard_ir_original_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _fixtures()
live_originals = F.live_originals


@pytest.fixture(scope="module")
def references(live_originals, tmp_path_factory):
    intake, basis, _ = live_originals
    owner = tmp_path_factory.mktemp("complete-original-standard-ir-references")
    selected = F.selection(intake, basis)
    selected["cohorts"] = {}
    for cohort, extent in required_source_cohorts():
        selected["cohorts"].setdefault(cohort, []).append(extent)
    path = owner / "selection.json"
    F.write(path, selected)
    return R.prepare(schema_intake=intake, basis=basis, selection=path, destination=owner / "references")


def selection(references):
    if not os.environ.get("MERLIN_TEST_MLIR_OPT"):
        pytest.skip("standard IR joins need the explicitly selected stock verifier")
    return {
        "schema": P.SCHEMA,
        "reference_roster_sha256": references.sha256,
        "capture_checkout": os.environ["MERLIN_TEST_M2M_ROOT"],
        "capture_commit": "7485a829c0195af0ec42820837d609e62e466564",
        "mlir_opt": os.environ["MERLIN_TEST_MLIR_OPT"],
        "budget": {
            "max_members": 100,
            "max_source_bytes": 200000,
            "max_total_source_bytes": 30000000,
            "max_observation_bytes": 2000000,
            "max_nesting": 64,
            "max_integer_bits": 512,
            "max_dense_elements": 1000,
            "max_dense_payload_bytes": 8000,
            "timeout_s": 180,
        },
        "execution_budget": json.loads(references.selection.read_bytes())["execution_budget"],
    }


@pytest.fixture(scope="module")
def observed(references, tmp_path_factory):
    owner = tmp_path_factory.mktemp("actual-original-reference-standard-ir")
    path = owner / "selection.json"
    F.write(path, selection(references))
    return S.prepare(references=references, selection=path, destination=owner / "products")


def test_complete_original_guard_private_roster_actual_upstream_abi_and_every_reference_value(observed):
    record = observed.record()
    assert len(record["members"]) == 24
    checked = [row for row in record["members"] if row["state"] == "source_reference_ir_checked"]
    assert len(checked) == 18, [(row["original"], row["state"], row.get("reason")) for row in record["members"]]
    assert len([row for row in record["members"] if row["state"] == "unavailable"]) == 6
    assert {tuple((row["original"]["cohort"], row["original"]["extent"])) for row in checked} == set(
        required_source_cohorts()
    )
    references = json.loads(observed.references.receipt_json)["members"]
    for row in checked:
        original = references[row["decision"]["index"]]
        metadata = json.loads(Path(original["products"]["metadata"]["path"]).read_bytes())
        assert row["comparison"]["passed"]
        assert [slot["name"] for slot in row["ordered_abi"]["inputs"]] == [slot["name"] for slot in metadata["inputs"]]
        assert [slot["name"] for slot in row["ordered_abi"]["outputs"]] == [
            slot["name"] for slot in metadata["outputs"]
        ]
        assert row["parse_invocation"] and set(row["products"]) == {"source", "trace", "actual", "verified"}
    assert all(set(R._UNKNOWN) | set(S._UNKNOWN) <= set(row["required_unknowns"]) for row in record["members"])
    assert "no compiler, runtime or release authority" in record["scope"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "reordered",
        "cohort",
        "extent",
        "original",
        "abi",
        "comparison",
        "unknown",
        "invocation",
        "source_pins",
        "budget",
        "bool_alias",
        "float_alias",
    ],
)
def test_resigned_data_cannot_drop_original_or_change_real_source_abi_products(observed, defect):
    # The fixture preparation already reopened every actual product. Construct
    # malformed test inputs from its frozen bytes; the tested verifier below
    # still reopens the original source, native products and full comparisons.
    record = json.loads(observed.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "source_reference_ir_checked")
    if defect == "missing":
        record["members"].pop()
    elif defect == "reordered":
        record["members"][0], record["members"][1] = record["members"][1], record["members"][0]
    elif defect in {"cohort", "extent", "original"}:
        field = "original_member_id" if defect == "original" else defect
        row["original"][field] = "other" if field != "extent" else 99
    elif defect == "abi":
        row["ordered_abi"]["inputs"].reverse()
    elif defect == "comparison":
        row["comparison"]["passed"] = False
    elif defect == "unknown":
        row["required_unknowns"] = []
    elif defect == "invocation":
        record["invocation"] = row["parse_invocation"]
    elif defect == "source_pins":
        record["source_pins"].pop()
    elif defect == "bool_alias":
        row["comparison"]["passed"] = 1
    elif defect == "float_alias":
        value = record["totals"]["execution"]["tensor_payload_bytes"]
        record["totals"]["execution"]["tensor_payload_bytes"] = float(value)
    else:
        record["totals"]["execution"]["tensor_payload_bytes"] = 0
    with pytest.raises((ValueError, KeyError)):
        S.verify(record, references=observed.references, selection=observed.selection)


def test_saved_object_cannot_mint_actual_source_observation(observed):
    with pytest.raises(ValueError, match="actual live"):
        replace(observed).verify()


@pytest.mark.parametrize("metric", ["max_members", "max_total_source_bytes", "max_total_tensor_payload_bytes"])
def test_complete_shared_budget_denial_prevents_framework_or_ir_construction(references, tmp_path, monkeypatch, metric):
    selected = selection(references)
    owner = selected["execution_budget"] if metric == "max_total_tensor_payload_bytes" else selected["budget"]
    owner[metric] = 1
    path = tmp_path / "selection.json"
    F.write(path, selected)
    monkeypatch.setattr(
        S, "_native", lambda *args: pytest.fail("budget-denied original roster reached framework construction")
    )
    observed = S.prepare(references=references, selection=path, destination=tmp_path / "denied")
    record = json.loads(observed.receipt_json)
    assert len(record["members"]) == 24
    assert all(row["state"] == "unavailable" for row in record["members"])
    assert record["invocation"] is None


def test_legacy_reference_selection_subset_cannot_replace_required_automatic_membership(live_originals, tmp_path):
    intake, basis, _ = live_originals
    selected = F.selection(intake, basis)
    selected["policies"] = []  # Full original denominator, no tensor execution needed.
    path = tmp_path / "legacy.json"
    F.write(path, selected)
    references = R.prepare(schema_intake=intake, basis=basis, selection=path, destination=tmp_path / "legacy")
    assert len(json.loads(references.receipt_json)["members"]) == 16
    with pytest.raises(ValueError, match="every original guard and private"):
        P.required_members(references)


def test_actual_last_upstream_native_value_is_checked_independently(observed, tmp_path):
    record = json.loads(observed.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "source_reference_ir_checked")
    original = json.loads(observed.references.receipt_json)["members"][row["decision"]["index"]]
    contract = S._contract(original, observed.references)
    inputs = R._tensors(json.loads(Path(original["products"]["inputs"]["path"]).read_bytes()))
    path = Path(row["products"]["actual"]["path"])
    actual = json.loads(path.read_bytes())
    output = actual["outputs"][-1]
    raw = bytearray.fromhex(output["data_hex"])
    raw[-1] ^= 64
    output["data_hex"] = raw.hex()
    changed = tmp_path / "changed-last-value.json"
    F.write(changed, actual)
    assert not R._comparison(contract, inputs, changed)["passed"]


def test_original_ordered_matmul_signature_rejects_swapped_argument_types(observed, tmp_path):
    record = json.loads(observed.receipt_json)
    row = next(
        row
        for row in record["members"]
        if row["state"] == "source_reference_ir_checked" and row["original"]["target"] == "aten.matmul.default"
    )
    original = json.loads(observed.references.receipt_json)["members"][row["decision"]["index"]]
    metadata = json.loads(Path(original["products"]["metadata"]["path"]).read_bytes())
    changed = copy.deepcopy(metadata)
    changed["inputs"].reverse()
    with pytest.raises(ValueError, match="shape or dtype"):
        S.ordered_abi(Path(row["products"]["source"]["path"]), changed, selection(observed.references)["budget"])


def test_actual_standard_source_byte_bound_precedes_parser_allocation(observed, tmp_path):
    record = json.loads(observed.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "source_reference_ir_checked")
    source = Path(row["products"]["source"]["path"])
    original = json.loads(observed.references.receipt_json)["members"][row["decision"]["index"]]
    metadata = json.loads(Path(original["products"]["metadata"]["path"]).read_bytes())
    budget = selection(observed.references)["budget"]
    budget["max_source_bytes"] = 1
    with pytest.raises(ValueError, match="source byte budget"):
        S.ordered_abi(source, metadata, budget)


def test_actual_stock_process_uses_remaining_shared_deadline_and_retains_interruption(observed, tmp_path):
    import subprocess

    from merlin.common.execution_deadline import ExecutionDeadline

    row = next(
        row for row in json.loads(observed.receipt_json)["members"] if row["state"] == "source_reference_ir_checked"
    )
    tool = tmp_path / "slow-verifier"
    tool.write_text("#!/usr/bin/python3\nimport time\ntime.sleep(5)\n")
    tool.chmod(0o700)
    selected = selection(observed.references)
    selected["mlir_opt"] = str(tool)
    paths = {"source": Path(row["products"]["source"]["path"]), "verified": tmp_path / "absent.mlir"}
    deadline = ExecutionDeadline.start(0.1)
    with pytest.raises(subprocess.TimeoutExpired):
        S._stock_verify(paths, selected, tmp_path, deadline)
    records = list((tmp_path / "parse/invocations").glob("*/invocation.json"))
    assert len(records) == 1
    invocation = json.loads(records[0].read_bytes())
    assert invocation["status"] == "interrupted" and "returncode" not in invocation
    assert "outputs" not in invocation and not paths["verified"].exists()
    with pytest.raises(TimeoutError):
        S._stock_verify(paths, selected, tmp_path, deadline)
    assert list((tmp_path / "parse/invocations").glob("*/invocation.json")) == records


def test_completed_parser_refusal_reopens_as_unavailable_not_a_success(observed, tmp_path):
    from merlin.common.execution_deadline import ExecutionDeadline

    row = next(
        row for row in json.loads(observed.receipt_json)["members"] if row["state"] == "source_reference_ir_checked"
    )
    original = json.loads(observed.references.receipt_json)["members"][row["decision"]["index"]]
    for role in ("source", "trace", "actual"):
        (tmp_path / S._paths(tmp_path)[role].name).write_bytes(Path(row["products"][role]["path"]).read_bytes())
    tool = tmp_path / "unavailable-verifier"
    tool.write_text("#!/usr/bin/python3\nraise SystemExit(17)\n")
    tool.chmod(0o700)
    selected = selection(observed.references)
    selected["mlir_opt"] = str(tool)
    first = S._evaluate(
        original, selected, tmp_path, 0, references=observed.references, deadline=ExecutionDeadline.start(5)
    )
    assert first["state"] == "unavailable" and "parse_invocation" in first
    assert first == S._evaluate(original, selected, tmp_path, 0, references=observed.references, run_parse=False)
    invocation = json.loads(Path(first["parse_invocation"]["path"]).read_bytes())
    assert invocation["status"] == "failed" and invocation["returncode"] == 17
    assert set(first["products"]) == {"source", "trace", "actual"}


def test_unavailable_native_products_remain_in_complete_roster(references, tmp_path):
    selected = selection(references)
    selected["budget"]["max_source_bytes"] = 1
    path = tmp_path / "selection.json"
    F.write(path, selected)
    owner = S.prepare(references=references, selection=path, destination=tmp_path / "products")
    record = json.loads(owner.receipt_json)
    assert len(record["members"]) == 24
    assert all(row["state"] == "unavailable" for row in record["members"])
    frame = json.loads(Path(record["observation"]["path"]).read_bytes())
    assert len(frame["rows"]) == 18 and all(row["status"] == "unavailable" for row in frame["rows"])
