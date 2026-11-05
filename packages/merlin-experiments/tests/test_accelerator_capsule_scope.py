"""An explicitly scoped release publishes accelerator tests and preserves their source obligations."""

import copy
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import accelerator_scope as scope
from merlin_experiments.corpus.phase_selection import generate_phase_selections, validate_phase_selections
from merlin_experiments.corpus.preparation import assemble
from merlin_experiments.phase0.coverage_commitment import read_inputs, write_inputs
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError


def _screen(**placements):
    return {"status": "admitted", "constraints_status": "matched", "placements": placements}


@pytest.mark.parametrize("placements", [{"accelerator": 1}, {"accelerator": 2, "host": 7}])
def test_accelerator_and_mixed_programs_are_selected(placements):
    assert scope.has_accelerator_work({"software_screen": _screen(**placements)})


@pytest.mark.parametrize("placements", [{}, {"host": 2}])
def test_pure_host_and_support_only_programs_are_not_selected(placements):
    assert not scope.has_accelerator_work({"software_screen": _screen(**placements)})


def test_accelerator_refusal_probe_is_retained_without_admitting_its_unknown_epilogue():
    screen = {
        "status": "unknown",
        "constraints_status": "unknown",
        "placements": {"accelerator": 1},
        "decisions": [
            {"placement": "accelerator", "classification": "accelerator_candidate", "status": "admitted"},
            {"placement": None, "classification": "unsupported", "status": "unknown"},
        ],
    }
    original = copy.deepcopy(screen)
    assert scope.has_accelerator_work({"software_screen": screen})
    assert screen == original and screen["status"] == "unknown"


@pytest.mark.parametrize(
    "screen", [None, {}, {"status": "unknown"}, _screen(accelerator=True), _screen(host=-1), _screen(unknown=1)]
)
def test_unresolved_or_malformed_screens_are_refused(screen):
    with pytest.raises(SpecError, match="resolved software screen"):
        scope.has_accelerator_work({"software_screen": screen})


def test_scoped_assembly_preserves_accelerator_and_mixed_bytes_and_private_boundary(tmp_path):
    generated, destination = tmp_path / "generated", tmp_path / "released"
    definitions = [
        ("isa", "device", {"accelerator": 1}),
        ("model", "mixed", {"accelerator": 2, "host": 3}),
        ("model_slices", "host", {"host": 1}),
        ("hidden", "private_device", {"accelerator": 1}),
        ("hidden", "private_host", {"host": 1}),
    ]
    source_hashes = {}
    for category, name, placements in definitions:
        root = generated / category / name
        root.mkdir(parents=True)
        cap = {
            "name": name,
            "kind": "isa",
            "label": "hidden" if category == "hidden" else "public",
            "software_screen": _screen(**placements),
            "numeric_policy": {"compare": "exact_int"},
            "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        }
        (root / "capsule.yaml").write_text(yaml.safe_dump(cap))
        (root / "capsule.interface.mlir").write_text("selected source bytes")
        source_hashes[category + "/" + name] = fingerprint(root)
    obligation = {"keep": "all original source operations, including host/device edges"}
    inputs = {
        "schema": "merlin.phase0.coverage_inputs.v1",
        "conformance": {"host_lane": {"required": ["float"]}},
        "accounting": obligation,
    }
    inputs_record = write_inputs(generated, inputs)
    public = sorted(key for key in source_hashes if not key.startswith("hidden/"))
    manifest = {
        "generated": public,
        "held_out": {"n_generated": 2},
        "coverage_inputs": inputs_record,
        "phase_corpora": {"fixture-device": generate_phase_selections(public, performance_category="_perf")},
    }
    (generated / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    te = SimpleNamespace(target="fixture-device", capsule_corpus=generated / "isa", workload_spec={})
    receipt = assemble(te, generated, destination, capsule_scope="accelerator")
    kept = {str(p.parent.relative_to(destination)) for p in destination.glob("*/*/capsule.yaml")}
    assert kept == {"isa/device", "model/mixed", "hidden/private_device"}
    assert all(fingerprint(destination / member) == source_hashes[member] for member in kept)
    assert all(fingerprint(generated / member) == digest for member, digest in source_hashes.items())
    promoted = yaml.safe_load((destination / "MANIFEST.yaml").read_text())
    assert promoted["held_out"] == {"n_generated": 1, "n_hand_authored": 0}
    buckets = validate_phase_selections(
        promoted["phase_corpora"][te.target],
        performance_category="_perf",
        generated_members=set(promoted["generated"]),
        complete=True,
    )
    assert set(buckets["phase1"]) == {"isa/device", "model/mixed"}
    assert "private_host" not in (destination / "MANIFEST.yaml").read_text()
    scoped = read_inputs(destination)
    assert scoped["accounting"] == obligation
    assert scoped["conformance"] == inputs["conformance"]
    assert scope.selected_scope(scoped)["source_coverage_inputs_sha256"] == inputs_record["sha256"]
    assert len(receipt["capsule_scope"]["members"]) == 5


def test_scope_projection_changes_only_standalone_host_coverage():
    original = {
        "host_lane": {"required": ["f32"]},
        "host_only": {"families": ["normalization"]},
        "cells": [{"cell": "contraction/i8"}],
        "groups": {"required": ["assembled"]},
    }
    saved = copy.deepcopy(original)
    inputs = {
        "capsule_scope": {
            "schema": scope.SCHEMA,
            "kind": "accelerator",
            "selection_sha256": "a" * 64,
            "source_coverage_inputs_sha256": "b" * 64,
        }
    }
    selected = scope.conformance_for_scope(original, inputs)
    assert selected["host_lane"]["required"] == selected["host_only"]["families"] == []
    assert selected["cells"] == original["cells"] and selected["groups"] == original["groups"]
    assert original == saved
    assert scope.conformance_for_scope(original, {}) is original
    with pytest.raises(ValueError, match="unsupported or unbound"):
        scope.conformance_for_scope(original, {"capsule_scope": {"kind": "accelerator"}})
