"""Application edges preserve raw words and refuse changed producer evidence."""

from hashlib import sha256
import json

import numpy as np
import pytest

from application_bindings import load_output_bindings


def fixture(tmp_path):
    producer, execution = tmp_path / "producer", tmp_path / "execution"
    producer.mkdir()
    execution.mkdir()
    words = np.array([0x8000, 0x7fc1, 0xffff, 1], dtype=np.uint16)
    signature = {"dtype": "bf16", "shape": [4]}
    compiled = {"status": "COMPILED", "source_request_digest": "request",
                "source_model_sha256": "model", "interface": {
                    "outputs": ["cache", "other"], "output_types": [signature, signature]}}
    receipt = {"status": "EXECUTED_UNCOMPARED", "source_request_digest": "request",
               "source_model_sha256": "model", "outputs": [
                   {"dtype": words.dtype.str, "shape": [4], "sha256": sha256(words.tobytes()).hexdigest()}
                   for _ in range(2)]}
    consumer = {"interface": {"inputs": ["cache"], "input_types": [signature.copy()]}}
    save(producer / "compiled.json", compiled)
    receipt["compiled_sha256"] = sha256((producer / "compiled.json").read_bytes()).hexdigest()
    save(execution / "execution.json", receipt)
    np.savez(execution / "actual.npz", output_0=words, output_1=words)
    return producer, execution, consumer, words, compiled, receipt


def save(path, value):
    path.write_text(json.dumps(value))


def test_bf16_edges_preserve_unsigned_words_and_authenticate_all_publications(tmp_path):
    producer, execution, consumer, words, _, _ = fixture(tmp_path)
    arrays, rows = load_output_bindings(producer, execution, consumer, {0: 0})
    assert arrays["input_0"].dtype == np.dtype("uint16")
    assert arrays["input_0"].tobytes() == words.tobytes()
    assert arrays["input_0"].flags.owndata and arrays["input_0"].flags.c_contiguous
    assert rows[0]["sha256"] == sha256(words.tobytes()).hexdigest()
    for field, path in (("producer_compiled_sha256", producer / "compiled.json"),
                        ("producer_execution_sha256", execution / "execution.json"),
                        ("producer_actual_sha256", execution / "actual.npz")):
        assert rows[0][field] == sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["status", "identity", "compiled_identity", "missing_compiled_identity",
                                   "receipt_hash", "receipt_dtype", "receipt_shape",
                                   "compiled_shape", "consumer_dtype", "consumer_shape", "unselected",
                                   "extra", "missing", "cast"])
def test_mutated_evidence_and_incompatible_edges_refuse(tmp_path, fault):
    producer, execution, consumer, words, compiled, receipt = fixture(tmp_path)
    if fault == "status":
        receipt["status"] = "RUNNING"
    elif fault == "identity":
        receipt["source_request_digest"] = "another"
    elif fault == "compiled_identity":
        receipt["compiled_sha256"] = "another-build"
    elif fault == "missing_compiled_identity":
        del receipt["compiled_sha256"]
    elif fault == "receipt_hash":
        receipt["outputs"][0]["sha256"] = "changed"
    elif fault == "receipt_dtype":
        receipt["outputs"][0]["dtype"] = "<i2"
    elif fault == "receipt_shape":
        receipt["outputs"][0]["shape"] = [2, 2]
    elif fault == "compiled_shape":
        compiled["interface"]["output_types"][0]["shape"] = [2, 2]
    elif fault == "consumer_dtype":
        consumer["interface"]["input_types"][0]["dtype"] = "i16"
    elif fault == "consumer_shape":
        consumer["interface"]["input_types"][0]["shape"] = [2, 2]
    elif fault == "unselected":
        np.savez(execution / "actual.npz", output_0=words, output_1=words ^ np.uint16(1))
    elif fault == "extra":
        np.savez(execution / "actual.npz", output_0=words, output_1=words, output_2=words)
    elif fault == "missing":
        np.savez(execution / "actual.npz", output_0=words)
    elif fault == "cast":
        np.savez(execution / "actual.npz", output_0=words.astype(np.int16), output_1=words)
    save(producer / "compiled.json", compiled)
    if fault == "compiled_shape":
        receipt["compiled_sha256"] = sha256((producer / "compiled.json").read_bytes()).hexdigest()
    save(execution / "execution.json", receipt)
    with pytest.raises(ValueError):
        load_output_bindings(producer, execution, consumer, {0: 0})


@pytest.mark.parametrize("bindings", [{}, {True: 0}, {0: False}, {-1: 0}, {0: -1}, {1: 0}, {0: 2}, {"0": 0}])
def test_invalid_bindings_refuse(tmp_path, bindings):
    producer, execution, consumer, *_ = fixture(tmp_path)
    with pytest.raises(ValueError):
        load_output_bindings(producer, execution, consumer, bindings)


def test_duplicate_publication_fields_refuse(tmp_path):
    producer, execution, consumer, *_ = fixture(tmp_path)
    path = producer / "compiled.json"
    path.write_text(path.read_text().replace('"status": "COMPILED"', '"status": "COMPILED", "status": "COMPILED"'))
    with pytest.raises(ValueError, match="duplicate"):
        load_output_bindings(producer, execution, consumer, {0: 0})


def test_same_source_recompiled_publication_cannot_reuse_execution(tmp_path):
    producer, execution, consumer, _, compiled, _ = fixture(tmp_path)
    compiled["build_policy"] = "different-build-of-same-source"
    save(producer / "compiled.json", compiled)
    with pytest.raises(ValueError, match="compiled publication differs"):
        load_output_bindings(producer, execution, consumer, {0: 0})


def test_repeated_source_results_remain_addressable_by_output_index(tmp_path):
    producer, execution, consumer, words, compiled, receipt = fixture(tmp_path)
    compiled["interface"]["outputs"] = ["cache", "cache"]
    save(producer / "compiled.json", compiled)
    receipt["compiled_sha256"] = sha256((producer / "compiled.json").read_bytes()).hexdigest()
    save(execution / "execution.json", receipt)
    arrays, rows = load_output_bindings(producer, execution, consumer, {0: 1})
    assert arrays["input_0"].tobytes() == words.tobytes()
    assert rows[0]["output_index"] == 1
