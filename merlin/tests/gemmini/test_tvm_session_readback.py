"""Whole-session readback preserves actual bits, invocation order and all gates."""
import hashlib
import importlib
import json

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.perf.firesim_receipt import UartValidationPolicy


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/comparisons/tvm"))
    return importlib.import_module("read_package_session")


def frame(name, value, *, signed=None, columns=None):
    data = value.tobytes()
    digest = 0xcbf29ce484222325
    for byte in data:
        digest = ((digest ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    cols = columns or (value.shape[-1] if value.ndim else 1)
    is_signed = value.dtype.kind == "i" if signed is None else signed
    header = f'OUT_BIN_BEGIN v1 {name} {value.size // cols} {cols} {value.itemsize} {"s" if is_signed else "u"} {len(data)}\n'
    return header.encode() + data + f'OUT_BIN_END v1 {digest:016x}\n'.encode()


def example():
    arrays = [np.array([-0., 1.25], dtype="float32"), np.array([-7, 2**40], dtype="int64"),
              np.array([True, False]), np.array(3.5, dtype="float64"),
              np.array([0x3f80, 0xc020], dtype="uint16"), np.array([2**64-1], dtype="uint64")]
    tensors = [{"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes} for value in arrays]
    tensors[4]["dtype"] = "bfloat16"
    session = {"schema": "tvm_merlin_package_session_v1", "readback": {"enabled": True},
               "plan": {"stages": ["prefill", "decode"], "outputs": tensors,
                        "calls": [{"stage": "prefill", "outputs": [0, 1, 2]},
                                  {"stage": "decode", "outputs": [3, 4]},
                                  {"stage": "decode", "outputs": [5]}]}}
    marker = "TVM_SESSION_PASS calls=3 outputs=6 resets=2"
    prefix = ("MERLIN_INVOCATIONS warmup=1 measured=1\nMERLIN_PROFILE warmup begin\n"
              "MERLIN_PROFILE warmup end rc=0\nMERLIN_PROFILE measured begin\n" + marker +
              "\nMETRIC cycles 123\nMERLIN_PROFILE measured end rc=0\n").encode()
    console = prefix + b"".join(frame(f"out{i}", value) for i, value in enumerate(arrays))
    console += b"TVM_SESSION_EXIT rc=0\nDONE\n"
    policy = UartValidationPolicy(policy_id="test", workload="readback", success_markers=(marker,))
    return session, arrays, console, policy


def test_export_preserves_bits_scalar_shapes_and_repeated_stage_order(reader, tmp_path):
    session, expected, console, policy = example()
    source = tmp_path / "session.json"
    source.write_text(json.dumps(session))
    policy = UartValidationPolicy(policy_id="tvm-session-" + hashlib.sha256(source.read_bytes()).hexdigest(),
                                  workload=policy.workload, success_markers=policy.success_markers)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy.to_dict()))
    uart = tmp_path / "uart"
    uart.write_bytes(console)
    report = reader.export_readback(source, uart, policy_path, tmp_path / "outputs")
    assert [(step["program"], step["step"]) for step in report["steps"]] == [("prefill", 0), ("decode", 0), ("decode", 1)]
    for step in report["steps"]:
        with np.load(step["actual"]["path"], allow_pickle=False) as archive:
            for tensor in step["outputs"]:
                actual = archive[f'out{tensor["index"]}']
                wanted = expected[tensor["session_output"]]
                assert actual.shape == wanted.shape and actual.dtype == wanted.dtype
                assert actual.tobytes() == wanted.tobytes()
    assert report["steps"][1]["outputs"][1]["storage_encoding"] == "raw_uint16_bfloat16"
    assert not report["timing_qualified"] and not report["execution_identity_verified"]
    with pytest.raises(FileExistsError):
        reader.export_readback(source, uart, policy_path, tmp_path / "outputs")
    # The same array ABI and counts cannot admit a manifest from another build.
    source.write_text(json.dumps({**session, "build_identity": "another-build"}))
    with pytest.raises(ValueError, match="another session manifest"):
        reader.export_readback(source, uart, policy_path, tmp_path / "stale-policy")
    assert not (tmp_path / "stale-policy").exists()


@pytest.mark.parametrize("fault", ["missing", "extra", "corrupt", "shape", "signed", "bool", "nan", "warm", "exit", "route"])
def test_readback_refuses_incomplete_or_invalid_evidence(reader, fault):
    session, arrays, console, policy = example()
    old = frame("out0", arrays[0])
    if fault == "missing":
        console = console.replace(old, b"")
    elif fault == "extra":
        console = console.replace(b"DONE\n", frame("out6", arrays[0]) + b"DONE\n")
    elif fault == "corrupt":
        console = console.replace(arrays[0].tobytes(), b"\x01" + arrays[0].tobytes()[1:], 1)
    elif fault == "shape":
        console = console.replace(old, frame("out0", arrays[0], columns=1))
    elif fault == "signed":
        console = console.replace(old, frame("out0", arrays[0], signed=True))
    elif fault == "bool":
        console = console.replace(frame("out2", arrays[2]), frame("out2", np.array([2, 0], dtype="uint8")))
    elif fault == "nan":
        console = console.replace(old, frame("out0", np.array([float("nan"), 1.25], dtype="float32")))
    elif fault == "warm":
        console = console.replace(b"warmup end rc=0", b"warmup end rc=22")
    elif fault == "exit":
        console = console.replace(b"TVM_SESSION_EXIT rc=0", b"TVM_SESSION_EXIT rc=21")
    else:
        session["plan"]["calls"][2]["outputs"] = [4]
    with pytest.raises(ValueError):
        reader.decode_session(session, console, policy)
