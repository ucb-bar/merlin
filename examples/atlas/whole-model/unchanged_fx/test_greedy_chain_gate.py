"""Focused protocol checks for all-output, independent greedy chain comparison."""
import json

import numpy as np
import pytest

import greedy_chain_gate as gate


CACHE_COUNT = 44


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def publish_chain(tmp_path, cache_count=CACHE_COUNT, weight_count=356):
    steps, previous = [], None
    for index, (past, query) in enumerate(((0, 64), (64, 1), (65, 1))):
        application, execution, reference = (tmp_path / f"{index}-{name}" for name in ("app", "run", "ref"))
        for path in (application, execution, reference):
            path.mkdir()
        capture = tmp_path / f"{index}-capture"
        capture.mkdir()
        (capture / "model.mlir").write_text(f"source model {index}")
        (capture / "frontend-trace.json").write_text(f"source trace {index}")
        (capture / "weights.safetensors").write_bytes(b"authenticated source weights")
        seq = past + query
        ids = np.arange(64, dtype=np.int64)[None] if index == 0 else np.array([[index]], dtype=np.int64)
        positions = np.arange(past, seq, dtype=np.int64)[None]
        keep = np.arange(seq)[None, :] <= np.arange(past, seq)[:, None]
        mask = np.where(keep, 0, 0xff7f).astype(np.uint16)[None, None]
        inputs = [ids, positions, mask]
        if previous:
            inputs += [previous["outputs"][f"output_{i + 1}"] for i in range(cache_count)]
        logits = bf16(np.zeros((1, query, 4)))
        logits[0, 0, 0] = bf16(10.0)
        logits[0, -1, index + 1] = bf16(20.0)
        outputs = {"output_0": logits}
        for cache in range(cache_count):
            outputs[f"output_{cache + 1}"] = (np.arange(seq * 2, dtype=np.uint16) + cache * 13).reshape(1, 1, seq, 2)
        compiled = dict(status="COMPILED", capture=str(capture),
            capture_mlir_sha256=gate.digest(capture / "model.mlir"), source_request_digest=f"request-{index}", source_model_sha256="model",
            interface=dict(inputs=[f"input_{i}" for i in range(len(inputs))],
                input_types=[dict(dtype="i64" if i < 2 else "bf16", shape=list(a.shape)) for i, a in enumerate(inputs)],
                outputs=list(outputs), output_types=[dict(dtype="bf16", shape=list(a.shape)) for a in outputs.values()]))
        save(application / "compiled.json", compiled)
        np.savez(execution / "actual.npz", **outputs)
        np.savez(reference / "expected.npz", **outputs)
        generated = dict(status="EXECUTED_UNCOMPARED", immutable_inputs=True,
            source_request_digest=compiled["source_request_digest"], source_model_sha256="model",
            compiled_sha256=gate.digest(application / "compiled.json"),
            inputs=[gate.info(a) for a in inputs], outputs=[gate.info(a) for a in outputs.values()], generated_bindings=[])
        independent = dict(status="EXPECTED", reference_version=3, oracle_mode="independent_whole_cache_source",
            compiler_or_emitted_program_inputs=False, ordinary_framework_kernels_called=False,
            expected_sha256=gate.digest(reference / "expected.npz"),
            source_trace_sha256=gate.digest(capture / "frontend-trace.json"),
            source_weights_sha256=gate.digest(capture / "weights.safetensors"),
            orchestration_sha256=gate.digest(gate.REFERENCE_ROOT / "cache_decode_reference.py"),
            trace_rebuild_sha256=gate.digest(gate.REFERENCE_ROOT / "source_trace.py"),
            adapter_sha256=gate.digest(gate.REFERENCE_ROOT / "whole_reference.py"),
            source_constants={f"weight_{i}": "authenticated" for i in range(weight_count)},
            original_export={"manifest": "authenticated", "authenticated_tensor_count": weight_count},
            selected_policies={"policy": "qualified"}, qualification={"engine": "qualified"},
            inputs={f"input_{i}": gate.info(a) for i, a in enumerate(inputs)},
            outputs={key: {**gate.info(a), "logical_dtype": "bf16"} for key, a in outputs.items()},
            parent_reference_sha256=None, explicit_tokens_sha256=None)
        if previous:
            old_app, old_execution, old_ref = steps[-1]
            old = previous["independent"]
            identities = dict(producer_compiled_sha256=gate.digest(old_app / "compiled.json"),
                producer_execution_sha256=gate.digest(old_execution / "execution.json"),
                producer_actual_sha256=gate.digest(old_execution / "actual.npz"),
                source_request_digest=previous["compiled"]["source_request_digest"], source_model_sha256="model")
            generated["generated_bindings"] = [dict(**identities, input_index=i + 3, output_index=i + 1,
                dtype="bf16", shape=list(previous["outputs"][f"output_{i + 1}"].shape),
                sha256=gate.info(previous["outputs"][f"output_{i + 1}"])["sha256"]) for i in range(cache_count)]
            generated["generated_bindings"].append(dict(**identities, operation="greedy_last_token", input_index=0,
                output_index=0, dtype="i64", shape=[1, 1], sha256=gate.info(ids)["sha256"],
                producer_output_sha256=old["outputs"]["output_0"]["sha256"], source_dtype="bf16",
                source_shape=old["outputs"]["output_0"]["shape"], sequence_index=old["outputs"]["output_0"]["shape"][1] - 1,
                token_id=index, tie_break="lowest_index"))
            independent["parent_reference_sha256"] = gate.digest(old_ref / "reference.json")
            independent["greedy_selection"] = dict(policy=gate.GREEDY_POLICY, source="independent_parent_expected_logits",
                parent_reference_sha256=gate.digest(old_ref / "reference.json"), parent_expected_sha256=old["expected_sha256"],
                parent_logits=old["outputs"]["output_0"], tokens=gate.info(ids), token_ids=ids.tolist())
        save(execution / "execution.json", generated)
        save(reference / "reference.json", independent)
        steps.append((application, execution, reference))
        previous = dict(outputs=outputs, independent=independent, compiled=compiled)
    return steps


@pytest.fixture
def chain(tmp_path):
    return publish_chain(tmp_path)


def test_complete_three_step_chain_compares_every_output(chain):
    result = gate.compare_chain(chain)
    assert result["status"] == "PASS"
    assert [row["outputs_compared"] for row in result["steps"]] == [45, 45, 45]
    assert result["transitions"] == [
        dict(generated_token_ids=[[1]], independent_token_ids=[[1]], cache_bindings=44),
        dict(generated_token_ids=[[2]], independent_token_ids=[[2]], cache_bindings=44)]


def test_explicit_whole_model_scope_requires_22_layers_and_356_weights(chain):
    assert gate.compare_chain(chain, expected_layers=22, expected_weight_tensors=356)["status"] == "PASS"


def test_one_layer_chain_retains_default_gate_behavior_but_cannot_claim_whole_model(tmp_path):
    chain = publish_chain(tmp_path, cache_count=2, weight_count=20)
    assert gate.compare_chain(chain)["status"] == "PASS"
    with pytest.raises(ValueError, match="decoder layer count"):
        gate.compare_chain(chain, expected_layers=22, expected_weight_tensors=356)


@pytest.mark.parametrize("step", range(3))
@pytest.mark.parametrize("field", ("source_constants", "original_export"))
def test_whole_model_scope_checks_both_weight_rosters_at_every_step(chain, step, field):
    reference = chain[step][2]
    receipt = gate.read(reference / "reference.json")
    if field == "source_constants":
        receipt[field].pop("weight_355")
    else:
        receipt[field]["authenticated_tensor_count"] = 355
    save(reference / "reference.json", receipt)
    with pytest.raises(ValueError, match="authenticated weight tensor count"):
        gate.compare_chain(chain, expected_layers=22, expected_weight_tensors=356)


@pytest.mark.parametrize("value", (0, -1, True, "22"))
@pytest.mark.parametrize("name", ("expected_layers", "expected_weight_tensors"))
def test_requested_scope_requires_positive_integer_counts(chain, name, value):
    with pytest.raises(ValueError, match="positive integer"):
        gate.compare_chain(chain, **{name: value})


@pytest.mark.parametrize("step", range(3))
@pytest.mark.parametrize("output", (0, 1, 22, 44))
def test_logits_and_cache_words_are_compared(chain, step, output):
    execution = chain[step][1]
    with np.load(execution / "actual.npz") as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays[f"output_{output}"].flat[-1] ^= np.uint16(1)
    np.savez(execution / "actual.npz", **arrays)
    receipt = gate.read(execution / "execution.json")
    receipt["outputs"][output] = gate.info(arrays[f"output_{output}"])
    save(execution / "execution.json", receipt)
    with pytest.raises(ValueError, match="exact output comparison failed"):
        gate.compare_chain(chain)


@pytest.mark.parametrize("step", (1, 2))
@pytest.mark.parametrize("cache", (0, 21, 43))
def test_skipped_cache_binding_fails_even_with_identical_outputs(chain, step, cache):
    execution = chain[step][1]
    receipt = gate.read(execution / "execution.json")
    del receipt["generated_bindings"][cache]
    save(execution / "execution.json", receipt)
    with pytest.raises(ValueError, match="binding roster"):
        gate.compare_chain(chain)


def test_self_consistent_wrong_greedy_token_fails(chain):
    # Both receipts claim the same wrong token, so simple input/output equality
    # still holds. The gate must derive the true choice from parent logits.
    execution, reference = chain[1][1:]
    generated, independent = gate.read(execution / "execution.json"), gate.read(reference / "reference.json")
    wrong = np.array([[0]], dtype=np.int64)
    generated["inputs"][0] = gate.info(wrong)
    independent["inputs"]["input_0"] = gate.info(wrong)
    generated["generated_bindings"][-1].update(token_id=0, sha256=gate.info(wrong)["sha256"])
    independent["greedy_selection"].update(tokens=gate.info(wrong), token_ids=[[0]])
    save(execution / "execution.json", generated)
    save(reference / "reference.json", independent)
    with pytest.raises(ValueError, match="generated greedy token input differs"):
        gate.compare_chain(chain)


@pytest.mark.parametrize("field", ("producer_compiled_sha256", "producer_execution_sha256", "producer_actual_sha256",
                                   "source_request_digest", "source_model_sha256", "sha256", "output_index"))
def test_cache_binding_provenance_is_authenticated(chain, field):
    execution = chain[1][1]
    receipt = gate.read(execution / "execution.json")
    receipt["generated_bindings"][20][field] = 1 if field == "output_index" else "wrong"
    save(execution / "execution.json", receipt)
    with pytest.raises(ValueError, match="cache binding provenance"):
        gate.compare_chain(chain)


@pytest.mark.parametrize("fault", ("generated_provenance", "reference_provenance", "parent_link", "archive"))
def test_greedy_and_parent_publications_are_authenticated(chain, fault):
    execution, reference = chain[1][1:]
    if fault == "generated_provenance":
        receipt = gate.read(execution / "execution.json")
        receipt["generated_bindings"][-1]["producer_output_sha256"] = "wrong"
        save(execution / "execution.json", receipt)
    elif fault in {"reference_provenance", "parent_link"}:
        receipt = gate.read(reference / "reference.json")
        if fault == "reference_provenance":
            receipt["greedy_selection"]["parent_expected_sha256"] = "wrong"
        else:
            receipt["parent_reference_sha256"] = "wrong"
        save(reference / "reference.json", receipt)
    else:
        with (reference / "expected.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance|parent reference|archive publication"):
        gate.compare_chain(chain)


def test_gate_greedy_rule_uses_last_row_lowest_index_and_rejects_nan():
    words = bf16([[[100, 0, 0], [-2, 3, 3]]])
    np.testing.assert_array_equal(gate.greedy_tokens(words), [[1]])
    np.testing.assert_array_equal(gate.greedy_tokens(np.array([[[0x8000, 0]]], dtype=np.uint16)), [[0]])
    words[0, 0, 0] = 0x7fc1
    with pytest.raises(ValueError, match="NaN"):
        gate.greedy_tokens(words)


def test_chain_must_have_three_steps(chain):
    with pytest.raises(ValueError, match="exactly"):
        gate.compare_chain(chain[:2])


@pytest.mark.parametrize("field", ("source_trace_sha256", "source_weights_sha256"))
def test_reference_must_use_exact_compiled_source_capture(chain, field):
    reference = chain[1][2]
    receipt = gate.read(reference / "reference.json")
    receipt[field] = "another source with identical outputs"
    save(reference / "reference.json", receipt)
    with pytest.raises(ValueError, match="reference/compiled source capture differs"):
        gate.compare_chain(chain)


def test_compiled_capture_mlir_is_authenticated(chain):
    capture = gate.read(chain[1][0] / "compiled.json")["capture"]
    (gate.Path(capture) / "model.mlir").write_text("changed source")
    with pytest.raises(ValueError, match="compiled source capture MLIR differs"):
        gate.compare_chain(chain)
