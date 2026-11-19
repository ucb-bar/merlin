"""Example-owned greedy selection consumes authenticated logits and cache words."""

from hashlib import sha256
import json
import shutil
import subprocess

import numpy as np
import pytest

import application_bindings
from application_bindings import greedy_token_from_bf16, load_output_bindings
import execute_graph
from test_execute_bindings import bound_execution


def _bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


@pytest.mark.parametrize("row, expected", [
    ([-3.0, -1.0, -1.0, -2.0], 1),
    ([-0.0, 0.0, -1.0, -2.0], 0),
    ([-np.inf, -np.inf, -np.inf, -np.inf], 0),
    ([0.0, np.inf, np.inf, -np.inf], 1),
])
def test_greedy_selects_last_sequence_row_with_lowest_index_ties(row, expected):
    logits = _bf16([[[100.0, 0.0, 0.0, 0.0], row]])
    token = greedy_token_from_bf16(logits)
    assert token.tolist() == [[expected]]
    assert token.dtype == np.int64 and token.flags.owndata and token.flags.c_contiguous


@pytest.mark.parametrize("word", [0x7f81, 0x7fc0, 0xffc1])
def test_greedy_rejects_nan_even_outside_selected_row(word):
    logits = _bf16([[[0.0, 1.0], [2.0, 0.0]]])
    logits[0, 0, 0] = word
    with pytest.raises(ValueError, match="NaN"):
        greedy_token_from_bf16(logits)


@pytest.mark.parametrize("logits", [
    np.zeros((1, 1, 2), dtype=np.int16), np.zeros((1, 2), dtype=np.uint16),
    np.zeros((2, 1, 2), dtype=np.uint16), np.zeros((1, 0, 2), dtype=np.uint16),
    np.zeros((1, 1, 0), dtype=np.uint16),
])
def test_greedy_rejects_unsupported_logits(logits):
    with pytest.raises(ValueError, match="bf16"):
        greedy_token_from_bf16(logits)


def _save(path, record):
    path.write_text(json.dumps(record))


@pytest.fixture
def greedy_case(bound_execution):
    case = bound_execution
    case.logits = _bf16([[[9.0, 0.0, 0.0, 0.0], [-2.0, -1.0, 3.0, 3.0]]])
    producer = json.loads((case.producer / "compiled.json").read_text())
    producer["interface"]["outputs"].insert(0, "logits")
    producer["interface"]["output_types"].insert(0, {"dtype": "bf16", "shape": [1, 2, 4]})
    _save(case.producer / "compiled.json", producer)
    published = [case.logits, *case.generated]
    receipt = json.loads((case.execution / "execution.json").read_text())
    receipt["compiled_sha256"] = sha256((case.producer / "compiled.json").read_bytes()).hexdigest()
    receipt["outputs"] = [{"dtype": value.dtype.str, "shape": list(value.shape),
                           "sha256": sha256(value.tobytes()).hexdigest()} for value in published]
    _save(case.execution / "execution.json", receipt)
    np.savez(case.execution / "actual.npz", **{f"output_{i}": value for i, value in enumerate(published)})
    consumer = json.loads((case.consumer / "compiled.json").read_text())
    consumer["interface"]["inputs"].insert(0, "token")
    consumer["interface"]["input_types"].insert(0, {"dtype": "i64", "shape": [1, 1]})
    _save(case.consumer / "compiled.json", consumer)
    np.savez(case.consumer / "tensors.npz", input_0=np.array([[99]], dtype=np.int64),
             input_1=case.defaults[0], input_2=case.defaults[1])
    case.descriptor["bindings"] = [{"input_index": i + 1, "output_index": i + 1} for i in range(2)]
    case.descriptor["greedy"] = {"input_index": 0, "output_index": 0}
    _save(case.bindings, case.descriptor)
    case.consumer_record = consumer
    return case


def test_greedy_and_cache_loader_record_derived_and_source_provenance(greedy_case):
    case = greedy_case
    arrays, rows = load_output_bindings(case.producer, case.execution, case.consumer_record,
                                       {1: 1, 2: 2}, greedy=case.descriptor["greedy"])
    np.testing.assert_array_equal(arrays["input_0"], np.array([[2]], dtype=np.int64))
    for i, cache in enumerate(case.generated, 1):
        assert arrays[f"input_{i}"].tobytes() == cache.tobytes()
    row = next(row for row in rows if row.get("operation") == "greedy_last_token")
    assert row["token_id"] == 2 and row["sequence_index"] == 1
    assert row["tie_break"] == "lowest_index"
    assert row["sha256"] == sha256(arrays["input_0"].tobytes()).hexdigest()
    assert row["producer_output_sha256"] == sha256(case.logits.tobytes()).hexdigest()
    assert row["source_dtype"] == "bf16" and row["source_shape"] == [1, 2, 4]
    for key, path in (("producer_compiled_sha256", case.producer / "compiled.json"),
                      ("producer_execution_sha256", case.execution / "execution.json"),
                      ("producer_actual_sha256", case.execution / "actual.npz")):
        assert row[key] == sha256(path.read_bytes()).hexdigest()


def test_greedy_only_needs_no_cache_binding(greedy_case):
    case = greedy_case
    arrays, rows = load_output_bindings(case.producer, case.execution, case.consumer_record, {},
                                       greedy=case.descriptor["greedy"])
    assert set(arrays) == {"input_0"}
    assert arrays["input_0"].tolist() == [[2]]
    assert len(rows) == 1


@pytest.mark.parametrize("changed", [0, 1, 2])
def test_greedy_only_authenticates_logits_and_every_unselected_cache(greedy_case, changed):
    case = greedy_case
    published = [case.logits.copy(), *[value.copy() for value in case.generated]]
    published[changed].flat[0] ^= np.uint16(1)
    np.savez(case.execution / "actual.npz", **{f"output_{i}": value for i, value in enumerate(published)})
    with pytest.raises(ValueError, match=f"output_{changed}"):
        load_output_bindings(case.producer, case.execution, case.consumer_record, {},
                             greedy=case.descriptor["greedy"])


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_greedy_target_requires_exact_i64_single_token(greedy_case, fault):
    case = greedy_case
    signature = case.consumer_record["interface"]["input_types"][0]
    signature[fault] = "i32" if fault == "dtype" else [1]
    with pytest.raises(ValueError, match=r"i64\[1,1\]"):
        load_output_bindings(case.producer, case.execution, case.consumer_record, {},
                             greedy=case.descriptor["greedy"])


@pytest.mark.parametrize("greedy", [None, {}, {"input_index": True, "output_index": 0},
    {"input_index": 0, "output_index": -1}, {"input_index": 0, "output_index": "0"},
    {"input_index": 1, "output_index": 0}, {"input_index": 3, "output_index": 0},
    {"input_index": 0, "output_index": 0, "token_id": 99}])
def test_malformed_or_conflicting_greedy_descriptor_fails_before_execution(greedy_case, greedy):
    case = greedy_case
    case.descriptor["greedy"] = greedy
    _save(case.bindings, case.descriptor)
    with pytest.raises(ValueError, match="greedy"):
        execute_graph.main()
    assert not case.output.exists()


def test_execute_combines_selected_token_with_both_generated_cache_buffers(greedy_case):
    case = greedy_case
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for real pointer dispatch")
    source = case.consumer / "dispatch.c"
    source.write_text("""#include <stdint.h>
int run(const void *const *inputs, void *const *outputs) {
    const int64_t *token = inputs[0];
    const uint16_t *k = inputs[1], *v = inputs[2];
    uint16_t *result = outputs[0];
    for (int i = 0; i < 4; ++i) result[i] = (k[i] ^ v[i]) + token[0];
    return 0;
}
""")
    subprocess.run(["nice", "-n", "10", compiler, "-shared", "-fPIC", str(source),
                    "-o", str(case.consumer / "dispatch.so")], check=True, capture_output=True)
    execute_graph.main()
    expected = np.bitwise_xor(*case.generated) + np.uint16(2)
    with np.load(case.output / "actual.npz", allow_pickle=False) as actual:
        np.testing.assert_array_equal(actual["output_0"], expected)
    receipt = json.loads((case.output / "execution.json").read_text())
    assert len(receipt["generated_bindings"]) == 3
    assert receipt["inputs"][0]["sha256"] == sha256(np.array([[2]], dtype=np.int64).tobytes()).hexdigest()
    assert receipt["inputs"][0]["dtype"] == np.dtype(np.int64).str
    assert next(row for row in receipt["generated_bindings"] if row.get("operation"))["token_id"] == 2


@pytest.mark.parametrize("missing", ["input_0", "input_1", "input_2"])
def test_execution_rejects_skipped_greedy_or_cache_substitution(greedy_case, monkeypatch, missing):
    original = application_bindings.load_output_bindings

    def skipped(*args, **kwargs):
        arrays, rows = original(*args, **kwargs)
        del arrays[missing]
        return arrays, rows

    monkeypatch.setattr(application_bindings, "load_output_bindings", skipped)
    with pytest.raises(ValueError, match="declared input bindings"):
        execute_graph.main()
    assert not greedy_case.output.exists()
