"""Host protocol controls only: no compiler, SDK proof, native library or model arrays."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import larger_model_campaign as campaign
import pytest


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "helper_paths", lambda: [])
    source = tmp_path / "source"
    source.write_text("original")
    cut = {
        "schema": "merlin.larger-model-campaign.source-cut.v1",
        "helper_paths": [],
        "files": [campaign.demo.file_ref(source)],
    }
    cut_path = tmp_path / "CUT.json"
    campaign.demo.write(cut_path, cut)
    return source, cut, campaign.demo.file_ref(cut_path)


def launch(tmp_path, protocol, owned_run):
    source, cut, pin = protocol
    return campaign.run_case(
        {"id": "host-control", "manifest": campaign.demo.file_ref(source)},
        36,
        output=tmp_path,
        cut=cut,
        cut_ref=pin,
        environment={},
        timeout=3,
        owned_run=owned_run,
    )


def response(command, **kwargs):
    case = Path(command[command.index("--case-out") + 1])
    case.mkdir()
    campaign.demo.write(case / "CASE-REPORT.json", {"status": "PASS", "phase": "complete"})
    return SimpleNamespace(
        returncode=0,
        stdout="host fake only",
        stderr="",
        owned_process_cleanup={"complete": True, "guardian_reaped": True},
    )


def test_positive_protocol_requires_completed_reaped_cleanup(tmp_path, protocol):
    row = launch(tmp_path, protocol, response)
    assert row["status"] == "PASS" and row["post_source_cut_matches"]
    assert row["owned_process_cleanup"] == {"complete": True, "guardian_reaped": True}


def test_pre_source_drift_never_launches_child(tmp_path, protocol):
    protocol[0].write_text("changed")
    row = launch(tmp_path, protocol, lambda *a, **k: pytest.fail("drift launched a process"))
    assert row["status"] == "FAILED_SOURCE_CUT_DRIFT_NO_CASE_CREDIT"


def test_post_source_drift_invalidates_even_successful_child(tmp_path, protocol):
    def mutate(command, **kwargs):
        result = response(command, **kwargs)
        protocol[0].write_text("changed after fake completion")
        return result

    row = launch(tmp_path, protocol, mutate)
    assert row["status"] == "FAILED_SOURCE_CUT_DRIFT_NO_CASE_CREDIT"


@pytest.mark.parametrize("cleanup", [{}, {"complete": True, "guardian_reaped": False}])
def test_unknown_cleanup_halts_lane_admission(tmp_path, protocol, cleanup):
    def unknown(command, **kwargs):
        result = response(command, **kwargs)
        result.owned_process_cleanup = cleanup
        return result

    row = launch(tmp_path, protocol, unknown)
    assert row["unsafe_to_release_lane"] is True
    assert row["status"] == "FAILED_CLEANUP_UNKNOWN_NO_CASE_CREDIT"


def test_timeout_without_cleanup_confirmation_quarantines_lane(tmp_path, protocol):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("host-fake-only", 3)

    row = launch(tmp_path, protocol, timeout)
    assert row["unsafe_to_release_lane"] is True and row["status"] == "FAILED_TIMEOUT_NO_CASE_CREDIT"


def test_unexpected_process_protocol_error_cannot_release_unknown_lane(tmp_path, protocol):
    def broken(*args, **kwargs):
        raise RuntimeError("host injected owned-process protocol failure")

    assert launch(tmp_path, protocol, broken)["unsafe_to_release_lane"] is True


def test_new_helper_membership_is_source_drift(protocol, monkeypatch):
    monkeypatch.setattr(campaign, "helper_paths", lambda: [protocol[0]])
    with pytest.raises(ValueError, match="helper roster"):
        campaign.require_cut(protocol[1])


@pytest.mark.parametrize("mutated", ["source", "baseline", "checked"])
def test_native_observation_must_join_actual_audited_compile(tmp_path, mutated):
    for name in ("source", "baseline", "checked", "different"):
        (tmp_path / name).write_text(name)
    refs = {name: campaign.demo.file_ref(tmp_path / name) for name in ("source", "baseline", "checked", "different")}
    compiled = {"source": refs["source"], "baseline_llvm": refs["baseline"], "checked_llvm": refs["checked"]}
    observed = {
        "status": "PASS",
        "all_public_outputs_byte_equal": True,
        "actual_original_source_abi": refs["source"],
        "baseline": {"llvm": refs["baseline"]},
        "checked": {"llvm": refs["checked"]},
    }
    if mutated == "source":
        observed["actual_original_source_abi"] = refs["different"]
    else:
        observed[mutated]["llvm"] = refs["different"]
    with pytest.raises(ValueError, match="differs"):
        campaign.native_join(compiled, observed, {"source": refs["source"]})


def test_array_type_and_shape_cannot_be_silently_cast_before_guard_copy():
    import numpy as np

    contract = {"arguments": [{"name": "arg0", "canonical_input_name": "arg0", "shape": [2], "dtype": "float32"}]}
    campaign._arrays(contract, {"arg0": np.zeros(2, dtype=np.float32)})
    for value in (np.zeros(2, dtype=np.float64), np.zeros((1, 2), dtype=np.float32)):
        with pytest.raises(ValueError, match="before guarded copying"):
            campaign._arrays(contract, {"arg0": value})


def test_failed_compile_audit_never_admits_native(tmp_path, protocol, monkeypatch):
    import larger_model_native_demo as native

    from merlin.verify import pass_validation

    manifest = {"id": "host-fake", "native_contract": {"entry": "forward"}}
    monkeypatch.setattr(
        campaign.demo, "load_manifest", lambda *a: (manifest, "", SimpleNamespace(), protocol[1]["files"])
    )
    monkeypatch.setattr(pass_validation, "CheckerTools", lambda *a: object())
    monkeypatch.setenv("MERLIN_LEAN_SDK_BIN", str(tmp_path))
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(tmp_path / "unused"))

    def failed_compile(manifest_path, destination, tools, **kwargs):
        destination.mkdir()
        value = {"status": "FAILED", "phase": "injected_host_protocol_only"}
        campaign.demo.write(destination / "DEMO-REPORT.json", value)
        return value

    monkeypatch.setattr(campaign.demo, "execute", failed_compile)
    monkeypatch.setattr(native, "compare", lambda *a, **k: pytest.fail("failed compile reached native"))
    result = campaign.one_case(protocol[0], tmp_path / "case", cut=protocol[1])
    assert result["status"] == "FAILED_NO_CASE_CREDIT" and result["native"] == []


def test_final_campaign_guard_covers_last_case(tmp_path, protocol, monkeypatch):
    from merlin.verify import pass_validation

    source, cut, _ = protocol
    monkeypatch.setattr(campaign, "load_roster", lambda *a: ([{"id": "last", "manifest": {}}], [source]))
    monkeypatch.setattr(campaign, "source_cut", lambda *a: cut)
    monkeypatch.setattr(pass_validation, "CheckerTools", lambda *a: object())
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda *a: {36})
    monkeypatch.setattr(campaign.os, "getpriority", lambda *a: 19)
    monkeypatch.setenv("MERLIN_LEAN_SDK_BIN", str(tmp_path))
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(tmp_path / "unused"))

    def final_mutation(case, cpu, **kwargs):
        source.write_text("changed after final case")
        return {"case": case["id"], "cpu": cpu, "status": "PASS"}

    monkeypatch.setattr(campaign, "run_case", final_mutation)
    result = campaign.campaign(tmp_path / "unused", tmp_path / "campaign", jobs=1, cpus=(36,), timeout=3)
    report = json.loads((tmp_path / "campaign/CAMPAIGN-RESULT.json").read_text())
    assert result == 1 and report["campaign_credit"] is False
    assert report["status"] == "FAILED_FINAL_SOURCE_CUT_NO_CAMPAIGN_CREDIT"


def test_degenerate_mlp_reference_refuses_before_native_and_retains_all_expected_output(tmp_path, monkeypatch):
    import larger_model_mlp as mlp
    import numpy as np

    source = tmp_path / "model.mlir"
    source.write_text("host fake definition, never parsed or compiled")
    rows = [{"name": "x", "canonical_input_name": "x", "shape": [2], "dtype": "float32"}]
    fixture = {
        "source": source.read_text(),
        "arguments": rows,
        "outputs": [{"name": "logits", "shape": [2], "dtype": "float32"}],
        "model": {"layers": 1},
    }
    monkeypatch.setattr(mlp, "fixture", lambda **kwargs: fixture)
    monkeypatch.setattr(
        mlp,
        "inputs_and_reference",
        lambda *a, **k: (
            {"x": np.zeros(2, dtype=np.float32)},
            {"logits": np.zeros(2, dtype=np.float32)},
            [{"layer": 0, "sha256": "host-control", "distinct_batch_rows": 1}],
        ),
    )
    contract = {
        "arguments": [{"name": "arg0", "canonical_input_name": "arg0", "shape": [2], "dtype": "float32"}],
        "outputs": [{"name": "result0", "shape": [2], "dtype": "float32"}],
    }
    with pytest.raises(ValueError, match="distinct batch rows"):
        campaign._mlp_inputs(
            {"source": campaign.demo.file_ref(source), "model_parameters": {}}, contract, 1, tmp_path / "reference"
        )
    record = json.loads((tmp_path / "reference/REFERENCE.json").read_text())
    assert record["stages"][0]["distinct_batch_rows"] == 1
    assert campaign.demo.bound(record["outputs"][0]["retained_array"])


@pytest.mark.parametrize("extra", [0, 1])
def test_case_timeout_maximum_is_inclusive_and_checked_before_roster(tmp_path, monkeypatch, extra):
    class ReachedRoster(Exception):
        pass

    assert campaign.MAX_CASE_TIMEOUT_SECONDS == 5400
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda *a: {36})
    monkeypatch.setattr(campaign.os, "getpriority", lambda *a: 19)

    def stop_before_tools_or_execution(*args):
        raise ReachedRoster

    monkeypatch.setattr(campaign, "load_roster", stop_before_tools_or_execution)
    expected = ValueError if extra else ReachedRoster
    with pytest.raises(expected):
        campaign.campaign(
            tmp_path / "unused",
            tmp_path / "not-created",
            jobs=1,
            cpus=(36,),
            timeout=campaign.MAX_CASE_TIMEOUT_SECONDS + extra,
        )
    assert not (tmp_path / "not-created").exists()


def test_larger_case_budget_preserves_default_timeout_and_parallelism(tmp_path, monkeypatch):
    observed = []

    def capture(*args, **kwargs):
        observed.append((args, kwargs))
        return 0

    monkeypatch.setattr(campaign, "campaign", capture)
    assert campaign.main(["--manifest-list", str(tmp_path / "unused"), "--out", str(tmp_path / "not-created")]) == 0
    assert observed[0][1] == {"jobs": 4, "cpus": (28, 29, 30, 31), "timeout": 1800}
    assert campaign.MAX_JOBS == 4
