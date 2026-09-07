"""Historical timing is a bound-to-bytes reference, never automatic calibration."""
import hashlib
import json
from dataclasses import replace

import pytest

from merlin.perf.harvest import Observation
from merlin.perf.historical_reference import bind_historical_reference, reference_summary
from merlin.perf.work_volume import work_from_command_buffer


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(target="target-a"):
    cb = {"tensors": {"a": {"shape": [2, 3]}, "b": {"shape": [3, 4]}},
          "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}]}
    raw_cb = json.dumps(cb).encode()
    work = work_from_command_buffer(cb).to_dict()
    doc = {"capsule": "w", "label": "public", "target": target,
           "toolchain_shas": {"device_revision": "abc"}, "work_volume": work,
           "tiers": {"L3": {"status": "pass", "cycles": 20, "engine": "engine-a",
                            "sim_provenance": {"sha256": "e" * 64}}}}
    raw = json.dumps(doc).encode()
    artifacts = {"kernel.elf": b"ELF", "harness.c": b"caller", "command_buffer.json": raw_cb}
    observation = Observation("submission", "w", "L3", "console", "rtl",
                              "total_cycles", 20.0, "cycles", status="pass")
    kwargs = dict(target=target, receipt=raw, receipt_sha256=digest(raw),
                  receipt_path="public/w/capsule_result.json", public_workloads={"w"},
                  artifacts=artifacts, artifact_sha256={k: digest(v) for k, v in artifacts.items()},
                  executable_names=("kernel.elf",), harness_name="harness.c",
                  command_buffer_name="command_buffer.json")
    return observation, kwargs


def test_reference_preserves_unknowns_and_cannot_mint_authority():
    obs, kwargs = fixture()
    row = bind_historical_reference(obs, **kwargs)
    assert row["usable_as"] == "historical_engine_relative_proxy"
    assert row["target_cycle_authority"] is False
    assert row["warm_calibration"] is False
    assert row["physical_roofline_complete"] is False
    assert row["window"] == "UNKNOWN"
    assert row["physical_moved_bytes"] is None
    assert "engine_configuration" in row["missing_contracts"]
    assert row["work"]["artifact_sha256"]
    assert row["work"]["exact_macs"] == 24
    assert row["receipt_sha256"] == kwargs["receipt_sha256"]


@pytest.mark.parametrize("target", ["target-a", "target-b"])
def test_target_parameter_is_preserved(target):
    obs, kwargs = fixture(target)
    assert bind_historical_reference(obs, **kwargs)["target"] == target


@pytest.mark.parametrize("field,value", [("workload", "wrong"), ("value", 21.0),
    ("status", "fail"), ("stage", "absent"), ("quantity", "busy"), ("unit", "seconds")])
def test_observation_must_match_raw_receipt(field, value):
    obs, kwargs = fixture()
    with pytest.raises(ValueError):
        bind_historical_reference(replace(obs, **{field: value}), **kwargs)


@pytest.mark.parametrize("mutation", ["receipt", "artifact", "hidden", "target", "work"])
def test_integrity_or_visibility_mismatch_refuses(mutation):
    obs, kwargs = fixture()
    if mutation == "receipt":
        kwargs["receipt"] += b" "
    elif mutation == "artifact":
        kwargs["artifacts"]["kernel.elf"] = b"changed"
    else:
        doc = json.loads(kwargs["receipt"])
        if mutation == "hidden": doc["label"] = "hidden"
        elif mutation == "target": doc["target"] = "different"
        else: doc["work_volume"]["artifact_sha256"] = "0" * 64
        kwargs["receipt"] = json.dumps(doc).encode()
        kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    with pytest.raises(ValueError):
        bind_historical_reference(obs, **kwargs)


def test_unlisted_public_workload_refuses():
    obs, kwargs = fixture()
    kwargs["public_workloads"] = {"another"}
    with pytest.raises(ValueError):
        bind_historical_reference(obs, **kwargs)


def test_no_implicit_pooling_or_speedup_in_summary():
    obs, kwargs = fixture()
    first = bind_historical_reference(obs, **kwargs)
    obs2, kwargs2 = fixture("target-b")
    second = bind_historical_reference(obs2, **kwargs2)
    summary = reference_summary([first, second])
    assert len(summary["engine_groups"]) == 2
    assert summary["target_cycle_authority"] is False
    assert summary["full_model_cycle_ordering"] == "UNKNOWN"
    assert summary["reference_count"] == 2


def test_duplicate_identity_does_not_buy_an_extra_sample():
    obs, kwargs = fixture()
    row = bind_historical_reference(obs, **kwargs)
    assert reference_summary([row, row])["reference_count"] == 1


def test_empty_summary_preserves_unknown():
    report = reference_summary([])
    assert report["reference_count"] == 0
    assert report["full_model_cycle_ordering"] == "UNKNOWN"


@pytest.mark.parametrize("cycles", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_cycle_values_refuse(cycles):
    obs, kwargs = fixture()
    doc = json.loads(kwargs["receipt"])
    doc["tiers"]["L3"]["cycles"] = cycles
    kwargs["receipt"] = json.dumps(doc).encode()
    kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    with pytest.raises(ValueError):
        bind_historical_reference(replace(obs, value=cycles), **kwargs)


def test_unverified_declared_warm_contract_is_not_authority():
    obs, kwargs = fixture()
    doc = json.loads(kwargs["receipt"])
    doc["tiers"]["L3"]["measurement_contract"] = {"warm_runs": 1}
    kwargs["receipt"] = json.dumps(doc).encode()
    kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    row = bind_historical_reference(obs, **kwargs)
    assert row["window"] == "UNKNOWN"
    assert row["warm_calibration"] is False


@pytest.mark.parametrize("field,value", [("cycles", 1), ("target_cycle_authority", True)])
def test_summary_rejects_modified_reference(field, value):
    obs, kwargs = fixture()
    row = bind_historical_reference(obs, **kwargs)
    row[field] = value
    with pytest.raises(ValueError):
        reference_summary([row])


def test_unknown_work_does_not_become_zero_rate():
    obs, kwargs = fixture()
    kwargs["command_buffer_name"] = None
    row = bind_historical_reference(obs, **kwargs)
    group = next(iter(reference_summary([row])["engine_groups"].values()))
    assert group["references"][0]["command_macs_per_observed_cycle"] is None


def test_missing_engine_stays_diagnostic_only():
    obs, kwargs = fixture()
    doc = json.loads(kwargs["receipt"])
    doc["tiers"]["L3"]["sim_provenance"] = {}
    kwargs["receipt"] = json.dumps(doc).encode()
    kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    row = bind_historical_reference(obs, **kwargs)
    assert row["usable_as"] == "diagnostic_only"


def test_duplicate_raw_json_field_refuses():
    obs, kwargs = fixture()
    kwargs["receipt"] = kwargs["receipt"][:-1] + b', "capsule": "w"}'
    kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    with pytest.raises(ValueError):
        bind_historical_reference(obs, **kwargs)


def test_embedded_work_is_bound_to_receipt_not_unbound_generated_file():
    obs, kwargs = fixture()
    doc = json.loads(kwargs["receipt"])
    cb = json.loads(kwargs["artifacts"]["command_buffer.json"])
    doc["command_buffer_artifact"] = {"command_buffer": cb,
                                      "artifact_sha256": doc["work_volume"]["artifact_sha256"]}
    kwargs["receipt"] = json.dumps(doc).encode()
    kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    kwargs["command_buffer_name"] = None
    row = bind_historical_reference(obs, **kwargs)
    assert row["work"]["exact_macs"] == 24
    assert row["work_location"].startswith("receipt#")


def test_embedded_work_hash_mismatch_refuses():
    obs, kwargs = fixture()
    doc = json.loads(kwargs["receipt"])
    doc["command_buffer_artifact"] = {"command_buffer": {}, "artifact_sha256": "0" * 64}
    kwargs["receipt"] = json.dumps(doc).encode()
    kwargs["receipt_sha256"] = digest(kwargs["receipt"])
    kwargs["command_buffer_name"] = None
    with pytest.raises(ValueError):
        bind_historical_reference(obs, **kwargs)
