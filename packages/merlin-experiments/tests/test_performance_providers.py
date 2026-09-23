"""Research providers retain import identity and private authority after relocation."""

import importlib
import importlib.abc
import importlib.metadata
import json
import socket
import subprocess
import sys
from pathlib import Path

import merlin_experiments
import pytest

from merlin.common.access import declared_modules

PROVIDERS = (
    "isolated_probe_provider",
    "controlled_context_provider",
    "paired_context_provider",
    "host_region_qualifier",
    "host_physical_transition_qualifier",
    "lane_migration_qualifier",
    "source_contraction_preparation",
    "source_convolution_preparation",
    "source_program_pair",
    "source_initializer_elision",
    "source_program_pair_provider",
)


@pytest.fixture(autouse=True)
def no_execution_or_native_imports(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("provider ownership tests cannot execute processes or bind listeners")

    class NoNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname in {
                "run_global_perf_experiment",
                "launch_global_agent_experiment",
                "run_agentic_perf_experiment",
                "perf_agent_stage",
            }:
                raise AssertionError(f"provider imports native controller: {fullname}")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)
    monkeypatch.setattr(sys, "meta_path", [NoNative(), *sys.meta_path])


@pytest.mark.parametrize("name", PROVIDERS)
def test_provider_has_one_experiments_owner_and_remains_private(name):
    identity = f"merlin.perf.{name}"
    module = importlib.import_module(identity)
    expected = Path(merlin_experiments.__file__).parent.parent / "merlin" / "perf" / f"{name}.py"
    assert Path(module.__file__).resolve() == expected.resolve()
    assert identity in declared_modules("grader")
    # Wheel RECORD proves distribution ownership even when namespace portions share site-packages.
    distribution = importlib.metadata.distribution("merlin-experiments")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    editable = direct_url.get("dir_info", {}).get("editable") is True
    legacy_metadata = expected.parents[2] / "merlin_experiments.egg-info" / "PKG-INFO"
    legacy_develop = (
        distribution.read_text("WHEEL") is None
        and legacy_metadata.is_file()
        and distribution.read_text("PKG-INFO") == legacy_metadata.read_text()
    )
    if not (editable or legacy_develop):
        assert f"merlin/perf/{name}.py" in {str(item) for item in distribution.files or ()}


@pytest.mark.parametrize("status", ["UNKNOWN", "observed_known_class_presence_mismatch", "matched"])
def test_task_route_feedback_never_promotes_presence_to_equivalence(status):
    from merlin.perf.source_contraction_preparation import task_route_feedback

    comparison = {"status": status, "all_static_classes_decoded": True}
    result = task_route_feedback(comparison)
    assert result["status"] == status
    assert result["descriptor_semantic_equivalence"] == "UNKNOWN"
    assert result["emitted_address_equivalence"] == "UNKNOWN"
    assert result["timing_calibration_admissible"] is False
    assert result["global_cost_validated"] is False
    assert comparison == {"status": status, "all_static_classes_decoded": True}
