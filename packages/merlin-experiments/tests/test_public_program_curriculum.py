"""Ordinary public capture packaging: synthetic closure tests, no execution claim."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import freeze
from merlin_experiments.phase0 import public_programs as P
from merlin_experiments.phase0.evidence import _materialize_evidence
from merlin_experiments.phase0.requirements import _materialized_iteration_capsules
from merlin_experiments.phase0.staging import STAGING_PREFIX, write_staged
from merlin_experiments.phase0.writer import _write_capsule
from merlin_experiments.phase2 import corpus as C
from test_operator_private_exclusion import selected
from test_phase0_materialized_capsules import _bundle

from merlin.targetgen.corpus_spec import CorpusBinding
from merlin.targetgen.frontend_trace import _digest

POLICY = {
    "schema": P.POLICY_SCHEMA,
    "selection": "declared_performance_applications",
    "max_members": 1,
    "max_capture_bytes": 1000000,
    "max_input_elements": 2,
    "max_output_elements": 2,
}


def capture(tmp_path):
    application = _bundle(tmp_path / "source")
    source = tmp_path / "source"
    nodes = []
    for ordinal, (identity, op, target, refs) in enumerate(
        [
            ("input", "placeholder", "input", []),
            ("copy", "call_function", "aten.clone.default", ["input"]),
            ("output", "output", "output", ["copy"]),
        ]
    ):
        nodes.append(
            {
                "id": identity,
                "ordinal": ordinal,
                "op": op,
                "target": target,
                "args": [{"node_id": ref, "value_id": ref + ":0"} for ref in refs],
                "kwargs": {},
                "results": [{"id": identity + ":0", "kind": "tensor", "shape": [2], "dtype": "float32"}],
            }
        )
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "nodes": nodes,
        "edges": [
            {
                "producer_node_id": ref["node_id"],
                "producer_value_id": ref["value_id"],
                "consumer_node_id": node["id"],
                "argument_path": "args/" + str(i),
                "dtype": "float32",
                "shape": [2],
                "value_kind": "tensor",
            }
            for node in nodes
            for i, ref in enumerate(node["args"])
        ],
    }
    graph["sha256"] = _digest(graph)
    trace_path = source / "frontend-trace.json"
    trace = json.loads(trace_path.read_bytes())
    trace.update(schema="m2m.frontend_trace.v1", graphs={"original": graph})
    trace_path.write_text(json.dumps(trace))
    profile = source / "profile.json"
    profile.write_text('{"source": "synthetic public fixture"}\n')
    meta_path = source / "meta.json"
    meta = json.loads(meta_path.read_bytes())
    meta["frontend_trace"]["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    meta["loader_provenance"] = {"profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()}
    meta_path.write_text(json.dumps(meta))
    receipt_path = source / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    for member in (trace_path, meta_path):
        receipt["artifacts"][member.name] = {
            "bytes": member.stat().st_size,
            "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
        }
    receipt_path.write_text(json.dumps(receipt))
    application["capture_receipt"]["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    loader = source / "loader.py"
    loader.write_text("# synthetic immutable public loader fixture\n")
    return application, loader


def entries(tmp_path):
    application, loader = capture(tmp_path)
    return _materialized_iteration_capsules(
        {"applications": {"public_source": application}},
        "0" * 64,
        loader_snapshots={"public_source": (loader, hashlib.sha256(loader.read_bytes()).hexdigest())},
        performance_selection=POLICY,
        performance_forms={
            "classes": [
                {
                    "class_id": "fixture",
                    "members": [
                        {
                            "application": "public_source",
                            "entry": {"op": "copy"},
                            "price": {"predicted_cycles": None},
                        }
                    ],
                }
            ]
        },
        facts_sha256="1" * 64,
    )


def write(tmp_path, exclusion):
    rows, outputs = entries(tmp_path)
    installed = tmp_path / "selected"
    _materialize_evidence(installed, outputs)
    entry = rows[0]
    (installed / "synth.yaml").write_text(yaml.safe_dump({"capsules": rows}))
    owned = freeze._materialized_inputs({"inputs": {"synth_profile": str(installed / "synth.yaml")}})
    assert installed / entry["materialized_capture"]["profile_path"] in owned
    for key in ("path", "loader_path", "profile_path"):
        entry["materialized_capture"][key] = str(installed / entry["materialized_capture"][key])
    binding = CorpusBinding("fixture", 2, "f32", "f32", False, ["L0"], "tolerance_float", atol=1e-5, rtol=1e-5)
    root = tmp_path / "corpus"
    result = write_staged(
        lambda stage: _write_capsule(entry, binding, stage),
        root,
        member="_perf/PG_source_public_source",
        operator_exclusion=exclusion,
    )
    return root, result


def test_ordinary_writer_discovery_and_freeze_reopen_complete_public_capture(tmp_path):
    exclusion, _ = selected(tmp_path)
    root, result = write(tmp_path, exclusion)
    capsule = yaml.safe_load((result / "capsule.yaml").read_bytes())
    expected = result / "source-capture-golden.npy"
    assert capsule["reference_closure"] == {
        "schema": "merlin.capsule_reference_closure.v1",
        "members": [{"path": expected.name, "role": "expected_output",
                     "sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
                     "bytes": expected.stat().st_size}],
    }
    observation = P.verify_written(capsule, result)
    assert observation["status"] == "source_prepared_execution_unknown"
    assert capsule["operation"]["op"] == "model"
    assert capsule["performance"]["curriculum"]["evaluation_tiers"]["timing"] == "UNKNOWN"
    assert capsule["performance"]["curriculum"]["observed_families"]["movement"] == ["aten.clone.default"]
    assert capsule["performance"]["curriculum"]["resource_boundary"]["observed_forms"][0]["class"] == "fixture"
    public = root / "public"
    public.mkdir()
    manifest = {
        "generated": ["_perf/PG_source_public_source"],
        "hand_authored": [],
        "performance_generation": {
            "fixture": {
                "errors": [],
                "phase": {"category": "_perf", "label": "dev", "included_in_functional_grade": False},
            }
        },
    }
    (root / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    experiment = SimpleNamespace(target="fixture", capsule_corpus=public, graded_roots=lambda: [public])
    discovered = C.discover_performance_corpus(experiment)
    frozen = C.freeze_performance_corpus(discovered, tmp_path / "frozen", operator_exclusion=exclusion)
    assert C.performance_objectives(frozen) == {}
    assert P.verify_written(frozen.capsules[0].descriptor, frozen.capsules[0].source_dir) == observation
    assert not list(root.glob(STAGING_PREFIX + "*"))


@pytest.mark.parametrize(
    "member",
    [
        "capsule.pytorch.py",
        "profile.json",
        "source-capture-inputs.npz",
        "source-capture-golden.npy",
        "frontend-trace.json",
        "capsule.interface.mlir",
    ],
)
def test_public_complete_source_and_reference_drift_refuses(member, tmp_path):
    exclusion, _ = selected(tmp_path)
    _, result = write(tmp_path, exclusion)
    capsule = yaml.safe_load((result / "capsule.yaml").read_bytes())
    (result / member).write_bytes((result / member).read_bytes() + b"changed")
    with pytest.raises(ValueError):
        P.verify_written(capsule, result)


def test_public_source_exclusion_refuses_before_promotion(tmp_path):
    capture(tmp_path)
    from merlin.targetgen.capsule_source import _portable_weights_reference

    source = (tmp_path / "source/model.mlir").read_text()
    meta = json.loads((tmp_path / "source/meta.json").read_bytes())
    portable, _ = _portable_weights_reference(source, meta["weights"], sidecar=True)
    exclusion, _ = selected(tmp_path, sources=[hashlib.sha256(portable.encode()).hexdigest()])
    # Reuse the selected fixture rather than alter the exclusion after staging.
    case = tmp_path / "case"
    case.mkdir()
    # Both fixtures' only source difference is the relocated weights path.
    with pytest.raises(ValueError, match="public corpus publication eligibility is not established"):
        write(case, exclusion)
    assert not list((case / "corpus").rglob("capsule.yaml"))


def test_selected_budgets_and_complete_roster_refuse_before_materialization(tmp_path):
    application, _ = capture(tmp_path)
    for field in ("max_capture_bytes", "max_input_elements", "max_output_elements"):
        with pytest.raises(ValueError, match="budget"):
            P.check_budget(tmp_path / "source", {**POLICY, field: 1})
    with pytest.raises(ValueError, match="complete preselected public roster"):
        _materialized_iteration_capsules(
            {"applications": {"public_source": application}},
            "0" * 64,
            performance_selection=POLICY,
        )
    rows, _ = _materialized_iteration_capsules({"applications": {"public_source": application}}, "0" * 64)
    assert "schema" not in rows[0]["materialized_capture"]
    assert rows[0]["source_role"] == "materialized_iteration_capture"


def test_inventory_is_observed_by_the_normal_frozen_input_owner_and_refuses_drift(tmp_path):
    inventory = {"applications": {"public_source": {"capture_sha256": "0" * 64}}}
    member = tmp_path / "performance-application-demands.json"
    member.write_text(json.dumps(inventory))
    requirement = {
        "public_program_sources": {
            "schema": P.POLICY_SCHEMA,
            "selection": POLICY,
            "sidecar": member.name,
            "full_inventory_sha256": _digest(inventory),
        }
    }
    observed = []

    def observe(path, role, *, required):
        observed.append((path, role, required))
        return path.read_bytes()

    P.observe_inventory(requirement, tmp_path / "requirements.yaml", observe)
    assert observed == [(member, "public-program-inventory", True)]
    member.write_text('{"applications": {}}')
    with pytest.raises(ValueError, match="requirement's commitment"):
        P.observe_inventory(requirement, tmp_path / "requirements.yaml", observe)
