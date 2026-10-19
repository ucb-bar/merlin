"""Original software selects current-source layouts without granting new effects."""

import json
from dataclasses import replace

import pytest
import yaml
from test_host_graph_optimizations import _source

from merlin.llvmlower.graph_optimizations import optimize_host_graph, recheck_host_graph_artifacts
from merlin.llvmlower.host_graph_policy import read_host_graph_policy
from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA, context_from_software


def software(tmp_path, **premises):
    path = tmp_path / "software.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.software_spec.v1",
                "status": "reviewed",
                "target": "neutral",
                "host_optimization": {
                    "schema": POLICY_SCHEMA,
                    "permitted_stages": ["logical_host"],
                    "permitted_optimizations": ["propagate_layout"],
                    "independent_element_order_unobserved": True,
                    **premises,
                },
            }
        )
    )
    return path


def select(tmp_path, **premises):
    return read_host_graph_policy(software(tmp_path, **premises), target="neutral", entry="kernel")


def retain(directory, policy, source):
    policy.retain(directory)
    selected, report = optimize_host_graph(source, policy)
    (directory / "host-graph-input.mlir").write_text(source)
    (directory / "host-graph-optimized.mlir").write_text(selected)
    (directory / "host-graph-optimization.json").write_text(json.dumps(report))
    return selected, report


def test_current_source_bound_after_preparation_and_full_policy_custody(tmp_path):
    policy = select(tmp_path)
    for shape in ((2, 3), (3, 5)):
        source = _source(*shape)
        selected, report = retain(tmp_path, policy, source)
        assert selected != source
        assert report["automatic_selection"]["status"] == "applied"
        assert report["contract"]["source_sha256"] == report["source_sha256"]
        assert report["layout"]["after_transpose_bytes"] == 0
        assert set(recheck_host_graph_artifacts(tmp_path, policy)) == {
            "input",
            "selected",
            "report",
            "software_spec",
            "policy",
        }


@pytest.mark.parametrize(
    "premises",
    [
        {"independent_element_order_unobserved": False},
        {"permitted_stages": []},
        {"permitted_optimizations": []},
    ],
)
def test_missing_original_permission_keeps_exact_source(tmp_path, premises):
    policy = select(tmp_path, **premises)
    source = _source()
    selected, report = retain(tmp_path, policy, source)
    assert selected == source
    assert "layout" not in report
    assert report["automatic_selection"]["reason"] == "permission_not_admitted"
    recheck_host_graph_artifacts(tmp_path, policy)


def test_review_label_without_host_policy_grants_no_permission(tmp_path):
    path = software(tmp_path)
    document = yaml.safe_load(path.read_text())
    del document["host_optimization"]
    path.write_text(yaml.safe_dump(document))
    policy = read_host_graph_policy(path, target="neutral", entry="kernel")
    selected, report = optimize_host_graph(_source(), policy)
    assert selected == _source() and report["automatic_selection"]["status"] == "unchanged"
    with pytest.raises(ValueError, match="permission schema"):
        context_from_software(document, semantic_sha256="0" * 64, target="neutral", entry="kernel")


def test_no_savings_or_float_permission_falls_back_without_partial_rewrite(tmp_path):
    policy = select(tmp_path)
    for source, chosen, reason in (
        (_source(), replace(policy, minimum_transpose_bytes_saved=1000), "cost contract"),
        (_source().replace("i32", "f32").replace("arith.addi", "arith.addf"), policy, "environment contract"),
    ):
        selected, report = retain(tmp_path, chosen, source)
        assert selected == source and "layout" not in report
        assert reason in report["automatic_selection"]["reason"]
        recheck_host_graph_artifacts(tmp_path, chosen)


@pytest.mark.parametrize("change", ["software", "retained_software", "policy", "input", "selected", "report"])
def test_changed_original_or_realized_artifacts_refuse(tmp_path, change):
    policy = select(tmp_path)
    retain(tmp_path, policy, _source())
    filename = {
        "software": "software.yaml",
        "retained_software": "host-graph-software.yaml",
        "policy": "host-graph-policy.json",
        "input": "host-graph-input.mlir",
        "selected": "host-graph-optimized.mlir",
        "report": "host-graph-optimization.json",
    }[change]
    path = tmp_path / filename
    if path.suffix == ".json":
        record = json.loads(path.read_text())
        record["changed"] = True
        path.write_text(json.dumps(record))
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        recheck_host_graph_artifacts(tmp_path, policy)


@pytest.mark.parametrize("physical", [False, True])
def test_mid_rewrite_software_mutation_is_an_error_not_fallback(tmp_path, monkeypatch, physical):
    from merlin.llvmlower import host_layout_realization, layout_propagation

    policy = replace(select(tmp_path), physical_layout=physical)
    implementation = host_layout_realization if physical else layout_propagation
    original = implementation.rewrite_module

    def mutate(*args, **kwargs):
        report = original(*args, **kwargs)
        policy.software_path.write_bytes(policy.software_bytes + b"\n")
        return report

    monkeypatch.setattr(implementation, "rewrite_module", mutate)
    with pytest.raises(ValueError, match="software specification changed"):
        optimize_host_graph(_source(), policy)


def test_implementation_change_expires_selected_policy(tmp_path, monkeypatch):
    from merlin.llvmlower import graph_optimizations

    policy = select(tmp_path)
    sources = graph_optimizations._implementation_sources()
    monkeypatch.setattr(graph_optimizations, "_implementation_sources", lambda: {**sources, "changed": "0" * 64})
    with pytest.raises(ValueError, match="implementation changed"):
        optimize_host_graph(_source(), policy)


def test_selected_software_target_and_review_state_are_checked(tmp_path):
    path = software(tmp_path)
    with pytest.raises(ValueError, match="target selection"):
        read_host_graph_policy(path, target="different", entry="kernel")
    record = yaml.safe_load(path.read_text())
    record["status"] = "unreviewed"
    path.write_text(yaml.safe_dump(record))
    with pytest.raises(ValueError, match="reviewed"):
        read_host_graph_policy(path, target="neutral", entry="kernel")


def test_conformance_failure_is_never_a_no_gain_fallback(tmp_path, monkeypatch):
    from merlin.llvmlower import graph_optimizations

    policy = select(tmp_path)

    def broken(*args):
        raise ValueError("corrupt scalar expression")

    monkeypatch.setattr(graph_optimizations, "_check_layout_output", broken)
    with pytest.raises(ValueError, match="corrupt scalar expression"):
        optimize_host_graph(_source(), policy)


@pytest.mark.parametrize(
    "premises",
    [
        {"independent_element_order_unobserved": 1},
        {"unknown_permission": True},
        {"schema": "future"},
        {"permitted_stages": ["unknown"]},
    ],
)
def test_unknown_or_nonboolean_software_permissions_refuse(tmp_path, premises):
    with pytest.raises(ValueError):
        select(tmp_path, **premises)
