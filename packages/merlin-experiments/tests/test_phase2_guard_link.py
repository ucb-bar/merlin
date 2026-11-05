"""Phase 2 keeps performance inputs distinct from byte-bound functional guards."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from merlin_experiments.corpus.phase_selection import generate_phase_selections
from merlin_experiments.phase0.coverage_commitment import INPUT_SCHEMA, _digest
from merlin_experiments.phase0.phase2_guards import build_guard_link, verify_guard_link
from merlin_experiments.phase1.source_inputs import fingerprint


def _fixture(tmp_path):
    corpus = tmp_path / "capsules"
    members = ("model_slices/negative", "_perf/throughput")
    for member in members:
        directory = corpus / member
        directory.mkdir(parents=True)
        negative = member.startswith("model_slices/")
        (directory / "capsule.mlir").write_text("module {}\n", encoding="utf-8")
        (directory / "capsule.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": directory.name,
                    "label": "public" if negative else "dev",
                    "source_role": "derived_sweep",
                    "linalg_mlir": "capsule.mlir",
                    "semantic": {"semantic_family": "movement"},
                    "lanes": {"forbid": ["on_mesh"]} if negative else {},
                    **({"performance": {"family": "throughput", "claim": "RECOVERS"}} if not negative else {}),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    selection = generate_phase_selections(list(members), performance_category="_perf")
    (corpus / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": list(members),
                "phase_corpora": {"fixture": selection},
                "performance_generation": {"fixture": {"phase": {"category": "_perf"}}},
            }
        ),
        encoding="utf-8",
    )
    contract = {"target": "fixture"}
    contract_sha = hashlib.sha256((json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    inputs = {
        "schema": INPUT_SCHEMA,
        "target": "fixture",
        "capability_contract": contract,
        "conformance": {
            "target": "fixture",
            "derivation": {"phase0_execution": {"contract_sha256": contract_sha}},
            "host_lane": {"required": [{"family": "movement", "dtype": "f32"}]},
            "host_only": {"families": []},
        },
    }

    def report(member):
        directory = corpus / member
        rows = [
            {
                "name": directory.name,
                "sha256": fingerprint(directory),
                "program_sha256": hashlib.sha256((directory / "capsule.mlir").read_bytes()).hexdigest(),
            }
        ]
        return {"inputs_sha256": _digest(inputs), "cohort": {"capsules": rows, "sha256": _digest(rows)}}

    return corpus, inputs, report(members[0]), report(members[1])


def test_phase2_guard_link_references_negative_source_without_performance_claim(tmp_path):
    corpus, inputs, phase1, phase2 = _fixture(tmp_path)
    link = build_guard_link(corpus, inputs, phase1, phase2)
    assert link["status"] == "axis_coverage_complete"
    assert [row["member"] for row in link["guards"]] == ["model_slices/negative"]
    assert link["phase2_performance_cohort_sha256"] == phase2["cohort"]["sha256"]
    assert "performance" not in yaml.safe_load((corpus / "model_slices/negative/capsule.yaml").read_text())
    verify_guard_link(link, corpus, inputs, phase1, phase2)


@pytest.mark.parametrize("member", ["model_slices/negative", "_perf/throughput"])
def test_phase2_guard_link_refuses_changed_cohort_bytes(tmp_path, monkeypatch, member):
    from merlin.targetgen import boundary

    corpus, inputs, phase1, phase2 = _fixture(tmp_path)
    monkeypatch.setattr(
        boundary,
        "host_lane_coverage",
        lambda *args, **kwargs: {"status": "ok", "n_required": 1, "n_covered": 1, "uncovered": []},
    )
    monkeypatch.setattr(
        boundary,
        "host_only_coverage",
        lambda *args, **kwargs: {"status": "not_applicable", "n_required": 0, "uncovered": []},
    )
    link = build_guard_link(corpus, inputs, phase1, phase2)
    (corpus / member / "capsule.mlir").write_text("module { changed }\n", encoding="utf-8")
    with pytest.raises(ValueError, match="capsule bytes changed"):
        verify_guard_link(link, corpus, inputs, phase1, phase2)


def test_phase2_guard_link_reports_host_placement_as_merlin_supplied(tmp_path):
    """A host-lane requirement is no public-capsule obligation: host placement is Merlin's codegen."""
    corpus, inputs, phase1, phase2 = _fixture(tmp_path)
    link = build_guard_link(corpus, inputs, phase1, phase2)
    host_lane = link["coverage"]["host_lane"]
    assert host_lane["status"] == "not_applicable"
    assert host_lane["uncovered"] == []
    assert host_lane["merlin_supplied_host"]["n_derived_requirements"] == 1
    assert link["coverage"]["host_only"]["merlin_supplied_host"]["n_derived_requirements"] == 0


def test_fused_stages_and_carried_state_are_byte_bound_functional_guards(tmp_path, monkeypatch):
    from merlin.targetgen import boundary

    corpus, inputs, phase1, phase2 = _fixture(tmp_path)
    stages = ("relu", "acc_scale", "bias_add")
    members = ["model_slices/negative"]
    for stage in stages:
        member = f"layers/fused_{stage}"
        directory = corpus / member
        directory.mkdir(parents=True)
        (directory / "capsule.mlir").write_text("module {}\n")
        (directory / "capsule.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": directory.name,
                    "label": "public",
                    "source_role": "derived_sweep",
                    "linalg_mlir": "capsule.mlir",
                    "semantic": {"generalization_axis": "epilogue"},
                    "operation": {"op": "matmul", "attributes": {"epilogue": [stage]}},
                }
            )
        )
        members.append(member)
    member = "layers/carried_relu"
    directory = corpus / member
    directory.mkdir(parents=True)
    (directory / "capsule.mlir").write_text("module {}\n")
    (directory / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": directory.name,
                "label": "public",
                "source_role": "derived_sweep",
                "linalg_mlir": "capsule.mlir",
                "semantic": {"generalization_axis": "carried_state"},
                "operation": {
                    "op": "resident_reuse",
                    "attributes": {
                        "matmuls": [
                            {"epilogue": ["relu"]},
                            {"epilogue": []},
                        ]
                    },
                },
                "stimulus_range": [-4, 3],
            }
        )
    )
    members.append(member)
    inputs["conformance"]["epilogue"] = {"required": [{"stage": stage} for stage in stages]}
    inputs["conformance"]["carried_state"] = {"required": [{"stage": "relu"}]}
    manifest_path = corpus / "MANIFEST.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["generated"] = members + ["_perf/throughput"]
    manifest["phase_corpora"]["fixture"] = generate_phase_selections(
        manifest["generated"], performance_category="_perf"
    )
    manifest_path.write_text(yaml.safe_dump(manifest))
    phase1_rows = [
        {
            "name": (corpus / member).name,
            "sha256": fingerprint(corpus / member),
            "program_sha256": hashlib.sha256((corpus / member / "capsule.mlir").read_bytes()).hexdigest(),
        }
        for member in members
    ]
    phase1 = {
        "inputs_sha256": _digest(inputs),
        "cohort": {
            "capsules": phase1_rows,
            "sha256": _digest(phase1_rows),
        },
    }
    phase2["inputs_sha256"] = _digest(inputs)
    monkeypatch.setattr(
        boundary,
        "host_lane_coverage",
        lambda *args, **kwargs: {
            "status": "ok",
            "n_required": 1,
            "n_covered": 1,
            "uncovered": [],
        },
    )
    link = build_guard_link(corpus, inputs, phase1, phase2)
    assert link["status"] == "axis_coverage_complete"
    assert link["coverage"]["epilogue"]["n_covered"] == len(stages)
    assert link["coverage"]["carried_state"]["n_covered"] == 1
    assert {row["member"] for row in link["guards"]} == set(members)
    assert all(
        "performance" not in yaml.safe_load((corpus / member / "capsule.yaml").read_text()) for member in members[1:]
    )
    verify_guard_link(link, corpus, inputs, phase1, phase2)
