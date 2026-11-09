"""Execution consumes authenticated producer words rather than packaged defaults."""

from hashlib import sha256
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import execute_graph


def _save(path, record):
    path.write_text(json.dumps(record))


@pytest.fixture
def bound_execution(tmp_path, monkeypatch):
    producer, execution, consumer = (tmp_path / name for name in ("producer", "prefill", "decode"))
    for directory in (producer, execution, consumer):
        directory.mkdir()
    # Both generated caches differ from the decode package's reference defaults.
    generated = [np.array([0x8000, 0x7fc1, 0xffff, 1], dtype=np.uint16),
                 np.array([0x3f80, 0x1234, 0x8001, 0xff7f], dtype=np.uint16)]
    defaults = [np.array([1, 2, 3, 4], dtype=np.uint16),
                np.array([5, 6, 7, 8], dtype=np.uint16)]
    signature = {"dtype": "bf16", "shape": [4]}
    producer_record = {"status": "COMPILED", "source_request_digest": "prefill-request",
                       "source_model_sha256": "model", "interface": {
                           "outputs": ["k_cache", "v_cache"],
                           "output_types": [signature, signature]}}
    _save(producer / "compiled.json", producer_record)
    receipt = {"status": "EXECUTED_UNCOMPARED", "source_request_digest": "prefill-request",
               "source_model_sha256": "model",
               "compiled_sha256": sha256((producer / "compiled.json").read_bytes()).hexdigest(),
               "outputs": [{"dtype": value.dtype.str, "shape": list(value.shape),
                            "sha256": sha256(value.tobytes()).hexdigest()} for value in generated]}
    _save(execution / "execution.json", receipt)
    np.savez(execution / "actual.npz", **{f"output_{i}": value for i, value in enumerate(generated)})
    record = {"source_request_digest": "decode-request", "source_model_sha256": "model",
              "logical_source_tensors": {}, "segments": [], "regions": [],
              "host_library": "dispatch.so", "interface": {
                  "inputs": ["k_cache", "v_cache"], "input_types": [signature, signature],
                  "outputs": ["result"], "output_types": [signature]}}
    _save(consumer / "compiled.json", record)
    np.savez(consumer / "tensors.npz", **{f"input_{i}": value for i, value in enumerate(defaults)})
    descriptor = {"schema": "merlin.application_output_bindings.v1",
                  "producer_application": "producer", "producer_execution": "prefill",
                  "bindings": [{"input_index": i, "output_index": i} for i in range(2)]}
    bindings = tmp_path / "bindings.json"
    _save(bindings, descriptor)
    output = tmp_path / "decode-execution"
    monkeypatch.setattr(sys, "argv", ["execute_graph", "--compiled", str(consumer),
        "--model-root", str(tmp_path / "unused-model"), "--input-bindings", str(bindings),
        "--execution-output", str(output)])
    # No device launches occur in this host-only dispatch test. Keep the real
    # loader, pointer construction, C invocation, output archive and receipt.
    class NoDeviceRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def finish(self):
            return {"launches": [], "total_cycles": 0}

    monkeypatch.setattr(execute_graph, "ModelRuntime", NoDeviceRuntime)
    return SimpleNamespace(producer=producer, execution=execution, consumer=consumer,
                           generated=generated, defaults=defaults, descriptor=descriptor,
                           bindings=bindings, output=output)


def test_execute_uses_generated_cache_words_in_actual_output(bound_execution):
    case = bound_execution
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for real pointer dispatch")
    source = case.consumer / "dispatch.c"
    source.write_text("""#include <stdint.h>
int run(const void *const *inputs, void *const *outputs) {
    const uint16_t *k = inputs[0], *v = inputs[1];
    uint16_t *result = outputs[0];
    for (int i = 0; i < 4; ++i) result[i] = k[i] ^ v[i];
    return 0;
}
""")
    subprocess.run(["nice", "-n", "10", compiler, "-shared", "-fPIC", str(source),
                    "-o", str(case.consumer / "dispatch.so")], check=True, capture_output=True)
    execute_graph.main()
    expected = np.bitwise_xor(*case.generated)
    assert not np.array_equal(expected, np.bitwise_xor(*case.defaults))
    with np.load(case.output / "actual.npz", allow_pickle=False) as actual:
        assert actual["output_0"].dtype == np.uint16
        np.testing.assert_array_equal(actual["output_0"], expected)
    receipt = json.loads((case.output / "execution.json").read_text())
    assert receipt["status"] == "EXECUTED_UNCOMPARED"
    assert len(receipt["generated_bindings"]) == 2
    for index, row in enumerate(receipt["generated_bindings"]):
        assert (row["input_index"], row["output_index"]) == (index, index)
        assert row["sha256"] == sha256(case.generated[index].tobytes()).hexdigest()
        for field, path in (("producer_compiled_sha256", case.producer / "compiled.json"),
                            ("producer_execution_sha256", case.execution / "execution.json"),
                            ("producer_actual_sha256", case.execution / "actual.npz")):
            assert row[field] == sha256(path.read_bytes()).hexdigest()


def test_missing_bindings_field_raises_value_error(bound_execution):
    case = bound_execution
    del case.descriptor["bindings"]
    _save(case.bindings, case.descriptor)
    with pytest.raises(ValueError, match="bindings"):
        execute_graph.main()
    assert not case.output.exists()


@pytest.mark.parametrize("outgoing", [True, False, "0", None, [], {}])
def test_outgoing_binding_requires_integer_at_example_layer(bound_execution, monkeypatch, outgoing):
    import application_bindings

    case = bound_execution
    case.descriptor["bindings"][0]["output_index"] = outgoing
    _save(case.bindings, case.descriptor)

    def unexpected_loader(*args, **kwargs):
        pytest.fail("example must reject output index type before loading producer evidence")

    monkeypatch.setattr(application_bindings, "load_output_bindings", unexpected_loader)
    with pytest.raises(ValueError, match="integer"):
        execute_graph.main()
    assert not case.output.exists()


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_bound_array_must_match_packaged_tensor_abi(bound_execution, fault):
    case = bound_execution
    # The declared producer/consumer types agree; tensors.npz is the separate
    # executable ABI authority that must also agree with the bound array.
    incompatible = (case.defaults[0].astype(np.int16) if fault == "dtype"
                    else case.defaults[0].reshape(2, 2))
    np.savez(case.consumer / "tensors.npz", input_0=incompatible, input_1=case.defaults[1])
    with pytest.raises(ValueError, match="ABI"):
        execute_graph.main()
    assert not case.output.exists()
