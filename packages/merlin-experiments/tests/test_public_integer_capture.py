"""Complete integer publication plumbing; synthetic references grant no execution."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from merlin_experiments.phase0 import public_programs as P
from merlin_experiments.phase0.operator_exclusion import contraction_signature_hashes
from merlin_experiments.phase0.requirements import _materialized_iteration_capsules
from merlin_experiments.phase0.staging import write_staged
from merlin_experiments.phase2 import agent_workspace, corpus
from test_operator_private_exclusion import selected as exclusion_selection

from merlin.targetgen import capsule_source as S
from merlin.targetgen import materialized_integer_capture as N
from merlin.targetgen.capsule_privacy import capsule_answer_members
from merlin.targetgen.corpus_spec import CorpusBinding
from merlin.targetgen.frontend_trace import _digest
from merlin.targetgen.golden_store import load_golden, write_golden

POLICY = dict(
    schema=P.INTEGER_POLICY_SCHEMA,
    source_mode=N.MODE,
    selection="declared_performance_applications",
    max_members=1,
    max_capture_bytes=1000000,
    max_input_elements=10,
    max_output_elements=10,
)


def capture(root):
    root.mkdir()
    weights = root / "weights.safetensors"
    weights.write_bytes(b"synthetic transport fixture only")
    (root / "weights.safetensors.manifest.json").write_text(
        json.dumps({str(i): dict(kind="input", name=name) for i, name in enumerate(("a", "b", "carried"))})
    )
    program = f'''builtin.module attributes {{prov.weights_file = "{weights}"}} {{
      func.func @forward(%a: tensor<1x2xi8>, %b: tensor<2x1xi8>, %c: tensor<2xi32>)
          -> (tensor<1x1xi32>, tensor<2xi32>, tensor<1x2xi8>) {{
        %z = arith.constant 0 : i32
        %e = tensor.empty() : tensor<1x1xi32>
        %init = linalg.fill ins(%z : i32) outs(%e : tensor<1x1xi32>) -> tensor<1x1xi32>
        %p = linalg.matmul ins(%a, %b : tensor<1x2xi8>, tensor<2x1xi8>)
          outs(%init : tensor<1x1xi32>) -> tensor<1x1xi32>
        func.return %p, %c, %a : tensor<1x1xi32>, tensor<2xi32>, tensor<1x2xi8>
      }} }}'''
    (root / "model.mlir").write_text(program)
    definitions = [
        ("a", "placeholder", "a", [1, 2], "int8", []),
        ("b", "placeholder", "b", [2, 1], "int8", []),
        ("c", "placeholder", "carried", [2], "int32", []),
        ("mm", "call_function", "aten._int_mm.default", [1, 1], "int32", ["a", "b"]),
        ("out", "output", "output", [], "int32", ["mm", "c", "a"]),
    ]
    nodes = []
    for i, (name, op, target, shape, dtype, refs) in enumerate(definitions):
        nodes.append(
            dict(
                id=name,
                ordinal=i,
                op=op,
                target=target,
                args=[dict(node_id=r, value_id=r + ":0") for r in refs],
                kwargs={},
                results=[] if op == "output" else [dict(id=name + ":0", kind="tensor", shape=shape, dtype=dtype)],
            )
        )
    values = {r["id"]: r for node in nodes for r in node["results"]}
    edges = [
        dict(
            producer_node_id=ref["node_id"],
            producer_value_id=ref["value_id"],
            consumer_node_id=node["id"],
            argument_path="args/" + str(i),
            dtype=values[ref["value_id"]]["dtype"],
            shape=values[ref["value_id"]]["shape"],
            value_kind="tensor",
        )
        for node in nodes
        for i, ref in enumerate(node["args"])
    ]
    graph = dict(
        schema="m2m.frontend_graph.v1",
        stage="original",
        status="complete",
        call_count=1,
        nodes=nodes,
        edges=edges,
        operator_schemas={"aten._int_mm.default": dict(schema="aten::_int_mm(Tensor self, Tensor mat2) -> Tensor")},
    )
    graph["sha256"] = _digest(graph)
    trace = dict(
        schema="m2m.frontend_trace.v1",
        graphs=dict(original=graph),
        mlir=dict(sha256=hashlib.sha256(program.encode()).hexdigest(), bytes=len(program.encode())),
    )
    (root / "frontend-trace.json").write_text(json.dumps(trace))
    meta = dict(
        ok=True,
        opaque=0,
        dtype="int8",
        source_mode=N.MODE,
        capture_quantization="already_materialized",
        quantization_stats=dict(applied=False),
        torch_seed=0,
        weights=str(weights),
        input_abi=[dict(shape=[1, 2], dtype="i8"), dict(shape=[2, 1], dtype="i8"), dict(shape=[2], dtype="i32")],
        output_abi=[dict(shape=[1, 1], dtype="i32"), dict(shape=[2], dtype="i32"), dict(shape=[1, 2], dtype="i8")],
        frontend_trace=dict(
            path="frontend-trace.json", sha256=hashlib.sha256((root / "frontend-trace.json").read_bytes()).hexdigest()
        ),
    )
    (root / "meta.json").write_text(json.dumps(meta))
    inputs = [
        np.array([[-2, 3]], dtype=np.int8),
        np.array([[3], [4]], dtype=np.int8),
        np.array([-16777217, 16777217], dtype=np.int32),
    ]
    np.savez(root / "inputs.npz", **{f"in{i}": x for i, x in enumerate(inputs)})
    (root / "input_order.json").write_text(json.dumps(dict(a=0, b=1, carried=2)))
    (root / "golden.json").write_text(json.dumps([[[6]], inputs[2].tolist(), inputs[0].tolist()]))
    np.save(root / "golden.npy", np.array([[6]], dtype=np.float32))
    np.savez(root / "goldens.npz", output0=np.array([[6]], dtype=np.int32), output1=inputs[2], output2=inputs[0])
    (root / "loader.py").write_text("# synthetic public transport fixture; never executed\n")
    receipt = dict(
        schema="m2m.capture-receipt.v1",
        materialized_abi=dict(complete=True),
        source_closure_verified=False,
        source=dict(path="/unavailable/synthetic.py"),
        artifacts={
            p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.iterdir()
            if p.name != "loader.py"
        },
    )
    (root / "capture_receipt.json").write_text(json.dumps(receipt))
    return dict(
        schema=N.SCHEMA,
        source_mode=N.MODE,
        path=str(root / "model.mlir"),
        capture_sha256=hashlib.sha256(program.encode()).hexdigest(),
        receipt_sha256=hashlib.sha256((root / "capture_receipt.json").read_bytes()).hexdigest(),
        full_inventory_sha256="0" * 64,
        workload_id="synthetic_public",
        workload_role="performance_scale",
        coverage_scope="full_capture",
        loader_path=str(root / "loader.py"),
        loader_sha256=hashlib.sha256((root / "loader.py").read_bytes()).hexdigest(),
    )


def write(root, out):
    selection = capture(root)
    art = S.materialized_model_artifacts(selection)
    entry = dict(
        name="integer_program",
        cat="_perf",
        kind="model",
        label="dev",
        materialized_capture=selection,
        capture_quantization="already_materialized",
        output_numeric_policy="captured_per_output_v1",
    )
    entry["performance"] = P.performance(
        entry, trace=json.loads((root / "frontend-trace.json").read_bytes()), forms={}, facts_sha256="1" * 64
    )
    binding = CorpusBinding("fixture", 2, "int8", "i32", True, ["L0"], "exact_int", atol=0, rtol=0)
    # Direct source packaging only: the ordinary Phase0 writer additionally requires a live arithmetic owner.
    return S.write_model_capsule(entry, binding, out, artifact=art)


def test_integer_materialization_is_explicit_and_keeps_values_beyond_fp32(tmp_path):
    selection = capture(tmp_path / "capture")
    art = S.materialized_model_artifacts(selection)
    assert art.golden[1] == [-16777217, 16777217]
    assert art.inputs[2] == [-16777217, 16777217]
    assert P.validate_selection(POLICY) == POLICY
    assert P.capture_schema(POLICY) == N.SCHEMA
    for schema in (P.CAPTURE_SCHEMA, None):
        legacy = {**selection, "schema": schema, "workload_role": "iteration"}
        with pytest.raises(S.M2MUnavailable, match="one losslessly represented floating"):
            S.materialized_model_artifacts(legacy)
    with pytest.raises(ValueError, match="source mode"):
        S.materialized_model_artifacts({**selection, "source_mode": "unselected"})
    bad = {**POLICY, "schema": P.POLICY_SCHEMA}
    with pytest.raises(ValueError, match="selection"):
        P.validate_selection(bad)


@pytest.mark.parametrize("value", [[[6.0]], [[True]], [[2**31]], [[-(2**31) - 1]], [[1], [2]]])
def test_typed_complete_reference_rejects_conversion_overflow_and_shape(value):
    meta = dict(
        ok=True,
        opaque=0,
        source_mode=N.MODE,
        capture_quantization="already_materialized",
        quantization_stats=dict(applied=False),
        input_abi=[dict(shape=[1, 1], dtype="i8")],
        output_abi=[dict(shape=[1, 1], dtype="i32")],
    )
    with pytest.raises(ValueError, match="reference"):
        N.outputs(meta, json.dumps(value))


def test_writer_exclusion_and_frozen_projection_keep_all_three_outputs_private(tmp_path, monkeypatch):
    exclusion, _ = exclusion_selection(tmp_path)
    root = tmp_path / "corpus"
    created = write_staged(
        lambda stage: write(tmp_path / "capture", stage),
        root,
        member="_perf/integer_program",
        operator_exclusion=exclusion,
    )
    cap = yaml.safe_load((created / "capsule.yaml").read_bytes())
    assert P.verify_written(cap, created)["status"] == "source_prepared_execution_unknown"
    assert load_golden(created)["outputs"]["Y1"] == [-16777217, 16777217]
    assert len(contraction_signature_hashes(cap, directory=created)) == 1
    expected = {"source-capture-golden.json", "source-capture-golden.npy", "source-capture-goldens.npz"}
    assert {row["path"] for row in cap["reference_closure"]["members"]} == expected
    stripped = copy.deepcopy(cap)
    stripped.pop("reference_closure")
    stripped.pop("materialized_capture")
    assert expected <= set(capsule_answer_members(created, capsule=stripped))
    (root / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            dict(
                generated=["_perf/integer_program"],
                hand_authored=[],
                performance_generation=dict(
                    fixture=dict(
                        errors=[], phase=dict(category="_perf", label="dev", included_in_functional_grade=False)
                    )
                ),
            )
        )
    )
    experiment = SimpleNamespace(target="fixture", capsule_corpus=root / "isa", graded_roots=lambda: [])
    frozen = corpus.freeze_performance_corpus(
        corpus.discover_performance_corpus(experiment), tmp_path / "frozen", operator_exclusion=exclusion
    )
    monkeypatch.setattr(agent_workspace, "answer_surfaces", lambda _: [])
    public = agent_workspace.build_answer_free_agent_inputs(frozen, experiment, tmp_path / "view")
    agent_workspace.verify_answer_free_agent_inputs(public)
    view = public.root / "_perf/integer_program"
    assert all(not (view / name).exists() for name in expected)
    assert (view / "source-capture-inputs.npz").is_file()
    assert not corpus.performance_objectives(frozen)


@pytest.mark.parametrize("mutation", ["second_output", "exact_policy", "closure", "source_reference"])
def test_original_public_verifier_rejects_complete_reference_drift(tmp_path, mutation):
    created = write(tmp_path / "capture", tmp_path / "corpus")
    cap = yaml.safe_load((created / "capsule.yaml").read_bytes())
    if mutation == "second_output":
        golden = load_golden(created)
        golden["outputs"]["Y1"][0] += 1
        write_golden(created, golden)
    elif mutation == "exact_policy":
        cap["numeric_policy"]["outputs"]["Y1"] = dict(compare="tolerance_float", atol=1, rtol=0)
    elif mutation == "closure":
        cap["reference_closure"]["members"].pop()
    else:
        (created / "source-capture-golden.json").write_text("[]")
    with pytest.raises(ValueError):
        P.verify_written(cap, created)


def test_integer_derivation_requires_real_sealed_attestations_before_source_selection(tmp_path):
    selection = capture(tmp_path / "capture")
    full = dict(
        applications=dict(
            synthetic_public=dict(
                capture_source_path=selection["path"],
                capture_sha256=selection["capture_sha256"],
                capture_receipt=dict(receipt_sha256=selection["receipt_sha256"]),
                n_operations=3,
            )
        )
    )
    with pytest.raises(ValueError, match="every original sealed"):
        _materialized_iteration_capsules(
            full,
            "0" * 64,
            loader_snapshots=dict(synthetic_public=(tmp_path / "capture/loader.py", selection["loader_sha256"])),
            performance_selection=POLICY,
        )


def test_native_contraction_exclusion_survives_output_renaming_and_extra_liveouts(tmp_path):
    directory = write(tmp_path / "capture", tmp_path / "corpus")
    cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    signature = contraction_signature_hashes(cap, directory=directory)
    names = cap["operation"]["attributes"]["outs"]
    renamed = {name: "public_return_" + str(i) for i, name in enumerate(names)}
    cap["operation"]["attributes"]["outs"] = [renamed[name] for name in names]
    cap["operation"]["attributes"]["out"] = renamed[names[0]]
    cap["numeric_policy"]["outputs"] = {
        renamed[name]: policy for name, policy in cap["numeric_policy"]["outputs"].items()
    }
    golden = load_golden(directory)
    golden["outputs"] = {renamed[name]: value for name, value in golden["outputs"].items()}
    write_golden(directory, golden)
    (directory / "capsule.yaml").write_text(yaml.safe_dump(cap))
    assert contraction_signature_hashes(cap, directory=directory) == signature
    excluded, _ = exclusion_selection(tmp_path, signatures=signature)
    with pytest.raises(ValueError, match="publication eligibility"):
        excluded.check_member(directory)


def test_integer_publication_without_original_private_selection_refuses_before_promotion(tmp_path):
    root = tmp_path / "corpus"
    with pytest.raises(ValueError, match="publication eligibility"):
        write_staged(lambda stage: write(tmp_path / "capture", stage), root, member="_perf/integer_program")
    assert not list(root.rglob("capsule.yaml"))


def test_integer_freeze_requires_original_private_selection_before_copy(tmp_path):
    root = tmp_path / "live"
    write(tmp_path / "capture", root)
    (root / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            dict(
                generated=["_perf/integer_program"],
                hand_authored=[],
                performance_generation=dict(
                    fixture=dict(
                        errors=[], phase=dict(category="_perf", label="dev", included_in_functional_grade=False)
                    )
                ),
            )
        )
    )
    experiment = SimpleNamespace(target="fixture", capsule_corpus=root / "isa", graded_roots=lambda: [])
    selected = corpus.discover_performance_corpus(experiment)
    with pytest.raises(ValueError, match="publication eligibility"):
        corpus.freeze_performance_corpus(selected, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()
