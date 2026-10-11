"""Small synthetic public graphs/facts exercise the ordinary source producer.

These controls execute tiny host sources, not capture, simulation or load tests.
Synthetic capture records below prove publication routing, never qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from merlin_experiments.phase0 import freeze, profiles
from merlin_experiments.phase0 import public_boundaries as B
from merlin_experiments.phase0 import public_boundary_semantics as S
from merlin_experiments.phase0.evidence import _materialize_evidence
from merlin_experiments.phase0.operator_exclusion import SIGNATURE_SCHEMA, contraction_signature_hashes
from merlin_experiments.phase0.staging import write_staged
from merlin_experiments.phase2 import corpus as C
from test_operator_private_exclusion import selected

from merlin.common.paths import repo_root
from merlin.targetgen import golden_store as GS
from merlin.targetgen.frontend_trace import _digest

POLICY = {
    "schema": B.SCHEMA,
    "max_members": 16,
    "max_nodes": 12,
    "max_logical_elements": 4096,
    "max_extent": 128,
    "max_source_bytes": 16384,
}


def trace(program=None):
    if program is None:
        program = {
            "inputs": [
                {"name": name, "shape": shape, "dtype": "float32"}
                for name, shape in [("x", [2, 3, 4]), ("w", [5, 3]), ("b", [5]), ("r", [2, 3])]
            ],
            "nodes": [
                {
                    "name": "moved",
                    "target": "aten.clone.default",
                    "args": [{"value": "x"}],
                    "kwargs": {},
                    "shape": [2, 3, 4],
                    "dtype": "float32",
                },
                {
                    "name": "mean",
                    "target": "aten.mean.dim",
                    "args": [{"value": "moved"}, [-1]],
                    "kwargs": {},
                    "shape": [2, 3],
                    "dtype": "float32",
                },
                {
                    "name": "sum",
                    "target": "aten.add.Tensor",
                    "args": [{"value": "mean"}, {"value": "r"}],
                    "kwargs": {},
                    "shape": [2, 3],
                    "dtype": "float32",
                },
                {
                    "name": "linear",
                    "target": "aten.linear.default",
                    "args": [{"value": "sum"}, {"value": "w"}, {"value": "b"}],
                    "kwargs": {},
                    "shape": [2, 5],
                    "dtype": "float32",
                },
                {
                    "name": "transposed",
                    "target": "aten.transpose.int",
                    "args": [{"value": "linear"}, 0, 1],
                    "kwargs": {},
                    "shape": [5, 2],
                    "dtype": "float32",
                },
            ],
            "outputs": ["transposed"],
        }
    nodes, values, edges = [], {}, []

    def arg(value):
        if isinstance(value, dict):
            return {"node_id": value["value"], "value_id": value["value"] + ":0"}
        return [arg(v) for v in value] if isinstance(value, list) else value

    for row in [*program["inputs"], *program["nodes"]]:
        result = {"id": row["name"] + ":0", "kind": "tensor", "shape": row["shape"], "dtype": row["dtype"]}
        values[result["id"]] = result
        nodes.append(
            {
                "id": row["name"],
                "ordinal": len(nodes),
                "op": "call_function" if "target" in row else "placeholder",
                "target": row.get("target", row["name"]),
                "args": arg(row.get("args", [])),
                "kwargs": {k: arg(v) for k, v in row.get("kwargs", {}).items()},
                "results": [result],
            }
        )
    nodes.append(
        {
            "id": "output",
            "ordinal": len(nodes),
            "op": "output",
            "target": "output",
            "args": [[arg({"value": name}) for name in program["outputs"]]],
            "kwargs": {},
            "results": [],
        }
    )

    def refs(value, path):
        if isinstance(value, dict):
            if "value_id" in value:
                yield path, value
            else:
                for key, item in value.items():
                    yield from refs(item, path + "/" + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from refs(item, path + "/" + str(i))

    for node in nodes:
        for key in ("args", "kwargs"):
            for path, ref in refs(node[key], key):
                row = values[ref["value_id"]]
                edges.append(
                    {
                        "producer_node_id": ref["node_id"],
                        "producer_value_id": ref["value_id"],
                        "consumer_node_id": node["id"],
                        "argument_path": path,
                        "dtype": row["dtype"],
                        "shape": row["shape"],
                        "value_kind": "tensor",
                    }
                )
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": len(program["nodes"]),
        "nodes": nodes,
        "edges": edges,
        "operator_schemas": {
            row["target"]: row.get("operator_schema", "synthetic schema: " + row["target"]) for row in program["nodes"]
        },
    }
    graph["sha256"] = _digest(graph)
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}


def produce(*, capacity=372, source=None, policy=POLICY):
    raw = json.dumps(source or trace()).encode()
    receipt = json.dumps({"artifacts": {"frontend-trace.json": {"sha256": hashlib.sha256(raw).hexdigest()}}}).encode()
    original = {
        "materialized_capture": {
            "path": "materialized-performance/public/model.mlir",
            "workload_id": "public",
            "capture_sha256": "1" * 64,
            "receipt_sha256": hashlib.sha256(receipt).hexdigest(),
            "loader_sha256": "2" * 64,
            "full_inventory_sha256": "3" * 64,
        }
    }
    facts = {
        "facts": {
            "arrays": [{"rows": 4, "cols": 4}],
            "memories": [
                {"name": "selected_store", "bytes": capacity * 4, "depth": capacity, "row_elems": 4, "elem_bits": 8}
            ],
        }
    }
    evidence = SimpleNamespace(target="fixture", refreshed_facts=facts, raw_facts_sha256=B.digest(facts))
    outputs = {
        "materialized-performance/public/frontend-trace.json": raw,
        "materialized-performance/public/capture_receipt.json": receipt,
    }
    return B.derive([original], outputs, evidence=evidence, policy=policy)


def execute(entry):
    pytest.importorskip("torch")
    namespace = {}
    exec(B.render(entry["public_boundary"]["program"]), namespace)
    model, inputs = namespace["get_model_and_inputs"]()
    return inputs, model(*inputs)


def test_joint_boundaries_construct_changed_typed_graphs_and_real_host_results():
    entries, products, report = produce()
    assert entries and len(entries) == report["constructed"] == sum(name.endswith("loader.py") for name in products)
    assert report["members_sha256"] == B.digest(entries)
    relations, roles = set(), set()
    for entry in entries:
        member = entry["public_boundary"]
        program, point = member["program"], member["point"]
        relations.add("=" if point["total_rows"] == point["capacity_rows"] else "<" if point["fits"] else ">")
        roles.update(point["roles"])
        assert [n["target"] for n in program["nodes"]] == [
            "aten.clone.default",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.linear.default",
            "aten.transpose.int",
        ]
        assert program["inputs"][0]["shape"] != [2, 3, 4]
        assert [program["dimension_values"][d] for d in program["dimensions"][1:]] == [3, 5, 3]
        assert member["pressure"]["hardware_mapping"] == "UNKNOWN"
        assert "logical" in member["pressure"]["scope"]
        assert point["total_rows"] == sum(prod_shape(row["shape"]) for row in [*program["inputs"], *program["nodes"]])
        inputs, outputs = execute(entry)
        x, r, w, b = [v.numpy() for v in inputs]
        expected = [x.copy(), x.mean(axis=-1), x.mean(axis=-1) + r]
        expected += [expected[-1] @ w.T + b]
        expected += [expected[-1].T]
        references = {
            node["name"]: (reference, node) for reference, node in zip(expected, program["nodes"], strict=True)
        }
        assert program["outputs"] == [program["nodes"][-1]["name"]]
        assert len(outputs) == 1
        for actual, name in zip(outputs, program["outputs"], strict=True):
            reference, node = references[name]
            assert list(actual.shape) == node["shape"]
            np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-6, atol=1e-6)
        assert "every original live-out" in entry["performance"]["curriculum"]["purpose"]
    assert relations == {"<", "=", ">"}
    assert {"below", "last_fitting", "first_overflow", "exact_capacity", "last_fitting_tail_plus_1"} <= roles


def prod_shape(shape):
    from math import prod

    return prod(shape)


def test_unreachable_equality_and_budget_failures_remain_explicit():
    entries, _, _ = produce(capacity=373)
    assert entries and entries[0]["public_boundary"]["pressure"]["exact_capacity"]["status"] == "unreachable"
    entries, _, report = produce(policy={**POLICY, "max_logical_elements": 1})
    assert entries == [] and report["requested"] > 0
    assert report["missing"] and all(r["mandatory"] and r["status"] == "UNKNOWN" for r in report["missing"])


def test_live_out_boundary_keeps_external_users_without_publishing_internal_values():
    source = trace()
    graph = source["graphs"]["original"]
    extra = {
        "id": "outside",
        "ordinal": len(graph["nodes"]) - 1,
        "op": "call_function",
        "target": "aten.sigmoid.default",
        "args": [{"node_id": "mean", "value_id": "mean:0"}],
        "kwargs": {},
        "results": [{"id": "outside:0", "kind": "tensor", "shape": [2, 3], "dtype": "float32"}],
    }
    graph["nodes"].insert(-1, extra)
    graph["nodes"][-1]["ordinal"] += 1
    graph["call_count"] += 1
    graph["operator_schemas"][extra["target"]] = "synthetic schema: " + extra["target"]
    graph["edges"].append(
        {
            "producer_node_id": "mean",
            "producer_value_id": "mean:0",
            "consumer_node_id": "outside",
            "argument_path": "args/0",
            "shape": [2, 3],
            "dtype": "float32",
            "value_kind": "tensor",
        }
    )
    graph.pop("sha256")
    graph["sha256"] = _digest(graph)
    cuts, missing = S.cuts(source, max_nodes=12)
    assert missing and len(cuts) == 1
    region = cuts[0]
    assert region["outputs"] == [region["nodes"][1]["name"], region["nodes"][-1]["name"]]
    assert region["nodes"][0]["name"] not in region["outputs"]
    assert region["nodes"][3]["name"] not in region["outputs"]


def test_original_shape_type_semantics_and_receipt_mutations_refuse():
    source = trace()
    source["graphs"]["original"]["nodes"][1]["results"][0]["shape"] = [5, 99]
    for edge in source["graphs"]["original"]["edges"]:
        if edge["producer_node_id"] == "w":
            edge["shape"] = [5, 99]
    source["graphs"]["original"].pop("sha256")
    source["graphs"]["original"]["sha256"] = _digest(source["graphs"]["original"])
    entries, _, report = produce(source=source)
    assert entries == [] and any("shape rule" in r["reason"] for r in report["missing"])
    source = trace()
    source["graphs"]["original"]["nodes"][0]["results"][0]["dtype"] = "int8"
    source["graphs"]["original"].pop("sha256")
    source["graphs"]["original"]["sha256"] = _digest(source["graphs"]["original"])
    with pytest.raises(ValueError):
        produce(source=source)


def test_re_signed_loader_cannot_replace_original_source_calls(tmp_path):
    entries, products, _ = produce()
    _materialize_evidence(tmp_path, products)
    entry = entries[0]
    assert B.selected_loader(entry, tmp_path).is_file()
    original = copy.deepcopy(entry)
    node = entry["public_boundary"]["program"]["nodes"][1]
    node["target"] = "aten.sum.dim_IntList"
    node["operator_schema"] = "synthetic schema: " + node["target"]
    changed = B.render(entry["public_boundary"]["program"]).encode()
    entry["public_boundary"]["loader_sha256"] = hashlib.sha256(changed).hexdigest()
    (tmp_path / entry["loader"]).write_bytes(changed)
    with pytest.raises(ValueError, match="original connected source cut"):
        B.selected_loader(entry, tmp_path)
    (tmp_path / original["loader"]).write_text(B.render(original["public_boundary"]["program"]))
    context = tmp_path / original["public_boundary"]["source_context"]["path"]
    context.write_bytes(context.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="original typed source context changed"):
        B.selected_loader(original, tmp_path)


def synthetic_written(entry, directory):
    """Pinned tiny capture fixture; no captured/compiler or execution authority."""
    directory.mkdir(parents=True)
    program = entry["public_boundary"]["program"]
    source = b"synthetic capture fixture source; no physical or compiler authority\n"
    raw = json.dumps(trace(program)).encode()
    (directory / "capsule.pytorch.py").write_text(B.render(program))
    (directory / "frontend-trace.json").write_bytes(raw)
    for name in ("frontend-source.mlir", "capsule.interface.mlir"):
        (directory / name).write_bytes(source)
    _, values = execute(entry)
    GS.write_golden(
        directory,
        {"golden_source": "host_torch_eager", "outputs": {f"Y{i}": value.tolist() for i, value in enumerate(values)}},
    )
    cap = {
        "name": entry["name"],
        "label": "dev",
        "source_role": "derived_sweep",
        "performance": entry["performance"],
        "public_boundary": entry["public_boundary"],
        "numeric_policy": {"compare": "tolerance_float", "atol": 1e-5, "rtol": 1e-5},
        "frontend_trace": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_mlir_sha256": hashlib.sha256(source).hexdigest(),
        },
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(cap))
    B.verify_written(cap, directory)
    return directory


def test_ordinary_selection_loader_freeze_staged_exclusion_discovery_and_freeze(tmp_path):
    entries, outputs, _ = produce()
    recipe = yaml.safe_load((repo_root() / "examples/gemmini/phase0/recipe-public-programs.yaml").read_bytes())
    assert B.validate(recipe["public_joint_boundaries"])["schema"] == B.SCHEMA
    shared = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_bytes())
    assert shared["public_boundary_factory"] == B.TEMPLATE
    profile = {}
    profiles._merge_shared_perf(
        profile,
        source=tmp_path / "recipe.yaml",
        performance_template=repo_root() / "experiments/templates/phase0/performance.yaml",
    )
    assert profile["_performance_template"]["public_boundary_factory"] == B.TEMPLATE
    _materialize_evidence(tmp_path / "selected", outputs)
    synthesis = tmp_path / "selected/synthesis.yaml"
    synthesis.write_text(yaml.safe_dump({"capsules": entries}))
    owned = freeze._materialized_inputs({"inputs": {"synth_profile": str(synthesis)}})
    assert len(owned) == len(entries) + 1
    exclusion, _ = selected(tmp_path)
    root = tmp_path / "corpus"
    entry = entries[0]
    result = write_staged(
        lambda stage: synthetic_written(entry, stage / "_perf" / entry["name"]),
        root,
        member="_perf/" + entry["name"],
        operator_exclusion=exclusion,
    )
    cap = yaml.safe_load((result / "capsule.yaml").read_bytes())
    signatures = contraction_signature_hashes(cap, directory=result)
    assert len(signatures) == 1 and SIGNATURE_SCHEMA
    (root / "public").mkdir()
    (root / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": ["_perf/" + entry["name"]],
                "hand_authored": [],
                "performance_generation": {
                    "fixture": {
                        "errors": [],
                        "phase": {"category": "_perf", "label": "dev", "included_in_functional_grade": False},
                    }
                },
            }
        )
    )
    experiment = SimpleNamespace(
        target="fixture", capsule_corpus=root / "public", graded_roots=lambda: [root / "public"]
    )
    corpus = C.discover_performance_corpus(experiment)
    frozen = C.freeze_performance_corpus(corpus, tmp_path / "frozen", operator_exclusion=exclusion)
    assert len(frozen.capsules) == 1 and C.performance_objectives(frozen) == {}
    (tmp_path / "private-control").mkdir()
    denied, _ = selected(tmp_path / "private-control", signatures=list(signatures))
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="public corpus publication eligibility is not established"):
        write_staged(
            lambda stage: synthetic_written(entry, stage / "_perf" / entry["name"]),
            rejected,
            member="_perf/" + entry["name"],
            operator_exclusion=denied,
        )
    assert not list(rejected.rglob("capsule.yaml"))


@pytest.mark.parametrize("member", ["capsule.pytorch.py", "frontend-trace.json", "capsule.interface.mlir"])
def test_generated_source_drift_refuses_before_promotion(member, tmp_path):
    entry = produce()[0][0]
    exclusion, _ = selected(tmp_path)

    def altered(stage):
        path = synthetic_written(entry, stage / "_perf" / entry["name"])
        (path / member).write_bytes((path / member).read_bytes() + b"changed")
        return path

    root = tmp_path / "rejected"
    with pytest.raises(ValueError, match="public corpus publication eligibility is not established"):
        write_staged(altered, root, member="_perf/" + entry["name"], operator_exclusion=exclusion)
    assert not list(root.rglob("capsule.yaml"))
