"""Closed launch selection controls; data supplies no runtime/timing authority."""

import json
import os
from dataclasses import asdict

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.search_configuration import prepare_search_configuration, read_search_configuration
from merlin_experiments.phase2.search_coordinator import SearchSettings


@pytest.fixture
def selected(tmp_path):
    path = tmp_path / "search.json"
    document = {
        "schema": "merlin.component_search_configuration.v1",
        **{
            field: "a" * 64
            for field in (
                "manifest_sha256",
                "corpus_sha256",
                "baseline_sha256",
                "certificate_sha256",
            )
        },
        "resources": {
            "cpu_ids": list(sorted(os.sched_getaffinity(0)))[:2],
            "ram_bytes": 8 << 30,
            "disk_bytes": 4 << 30,
            "engine_slots": 1,
            "author_cores": 1,
        },
        "job_resources": {"cores": 1, "ram_bytes": 1 << 30, "disk_bytes": 1 << 20},
        "settings": asdict(SearchSettings()),
        "source_bytes": 1 << 20,
        "source_files": 1000,
        "retained_source_bytes": 16 << 20,
        "max_pending": 6,
        "max_jobs": 24,
        "author_agents": 3,
    }
    path.write_text(json.dumps(document))
    return path, document


def test_exact_configuration_refuses_file_or_mutable_document_drift(selected):
    path, document = selected
    configuration = read_search_configuration(path)
    configuration.verify()
    configuration.document["max_jobs"] += 1
    with pytest.raises(StageGateError, match="changed"):
        configuration.verify()
    configuration = read_search_configuration(path)
    path.write_text(json.dumps({**document, "max_jobs": 25}))
    with pytest.raises(StageGateError, match="changed"):
        configuration.verify()


@pytest.mark.parametrize("change", ["callback", "roster", "authors", "beam", "reserve", "jobs"])
def test_search_refuses_authority_payloads_and_unbounded_selection(selected, change):
    path, document = selected
    if change in {"callback", "roster"}:
        document[change] = "unissued"
    elif change == "authors":
        document["author_agents"] = 4
    elif change == "beam":
        document["settings"]["beam_width"] = 4
    elif change == "reserve":
        document["settings"]["final_reserve_seconds"] = 14_400
    else:
        document["max_jobs"] = 1025
    path.write_text(json.dumps(document))
    with pytest.raises(StageGateError):
        read_search_configuration(path)


@pytest.mark.usefixtures("synthetic_component_controller")
def test_prepare_pins_existing_policy_and_refuses_overwriting(tmp_path, monkeypatch):
    from test_component_workflow import _synthetic_rtl_policy

    policy, rtl, _ = _synthetic_rtl_policy(tmp_path, monkeypatch)
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    if len(cpus) < 2:
        pytest.skip("two allocated CPUs required for configuration validation")
    path = tmp_path / "prepared-search.json"
    kwargs = dict(
        resources=AssignedResources(cpus, 8 << 30, 4 << 30, 1, author_cores=1),
        job_resources=JobResources(1, 1 << 30, 1 << 20),
        settings=SearchSettings(),
        source_bytes=1 << 20,
        source_files=1000,
        retained_source_bytes=16 << 20,
        max_pending=6,
        max_jobs=24,
        author_agents=3,
    )
    selection = prepare_search_configuration(policy, path, **kwargs)
    assert selection.document["baseline_sha256"] == policy.baseline_admission.baseline_sha256
    assert selection.document["corpus_sha256"] == policy.component_corpus.capsules_sha256
    assert selection.document["certificate_sha256"] == rtl.certificate.sha256
    with pytest.raises(StageGateError, match="fresh"):
        prepare_search_configuration(policy, path, **kwargs)
