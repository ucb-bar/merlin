"""Actual linked lazy pointer storage, exact bits and failure custody."""

import hashlib
import json
import shutil
import struct
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.weight_payload import WeightPayload
from merlin.runtime.harness_render import render_harness
from merlin.runtime.tensor_payload import TensorPayload
from merlin.targetgen import bundle_pack, capture_source
from merlin.targetgen.contract.harness_blobs import stage_harness_blobs
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64


def _contract():
    return {"harness_abi": {
        "version": 2, "kind": "logical_pointer", "entry_symbol": "entry", "fence_symbol": "complete",
        "tensor_alignment": 16, "byte_order": "little", "main_convention": "void",
        "readback_transport": FULL_VALUES_B64,
    }}


def _program():
    return {"tensors": {"X": {"shape": [3, 5], "dtype": "i8", "role": "input"},
                        "Y": {"shape": [3, 5], "dtype": "i8", "role": "output"}}}


def _payload(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(bytes(range(128, 143)))
    return path, TensorPayload((3, 5), "i8", WeightPayload.file(path, 0, 15))


def test_real_linked_streamed_and_inline_callers_preserve_complete_signed_outputs(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler unavailable")
    source, payload = _payload(tmp_path)
    control = tmp_path / "control.c"
    control.write_text(
        '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n'
        'void console_init(void){} void htif_puts(const char*s){fputs(s,stdout);}\n'
        'void htif_exit(int c){exit(c);} void complete(void){}\n'
        'void entry(void*x,void*y){for(int i=0;i<15;i++)((signed char*)y)[i]=((signed char*)x)[i]+1;}\n'
        'int harness_main(void);int main(void){return harness_main();}\n'
    )
    results = []
    for mode in ("inline", "streamed"):
        folder = tmp_path / mode
        folder.mkdir()
        (folder / "htif.h").write_text("void console_init(void);void htif_puts(const char*);void htif_exit(int);\n")
        (folder / "out_b64.h").write_bytes((runtime_dir() / "baremetal/out_b64.h").read_bytes())
        blobs = {}
        inputs = {"X": source.read_bytes() if mode == "inline" else payload}
        text = render_harness(_program(), target="synthetic", inputs=inputs, contract=_contract(), blobs=blobs)
        (folder / "harness.c").write_text(text)
        objects = stage_harness_blobs(folder, blobs)
        if mode == "streamed":
            assert 'extern unsigned char tensor_0[15];' in text
            assert '.section .data' in objects[0].read_text()
            assert (folder / "harness_blob_tensor_0.bin").read_bytes() == source.read_bytes()
        subprocess.run([cc, "-Dmain=harness_main", "-c", "harness.c", "-o", "harness.o"], cwd=folder, check=True)
        subprocess.run([cc, "harness.o", str(control), *map(str, objects), "-o", "program"], cwd=folder, check=True)
        results.append(subprocess.run([str(folder / "program")], capture_output=True, check=True).stdout)
    assert results[0] == results[1] and b"OUT_B64_BEGIN v1 Y 3 5 1 s" in results[0]


@pytest.mark.parametrize("mutation", ["shape", "dtype", "extent", "sidecars", "duplicate"])
def test_streamed_input_refuses_ambiguous_or_changed_storage(tmp_path, mutation):
    _, payload = _payload(tmp_path)
    cb, blobs = _program(), {}
    if mutation == "shape":
        payload = TensorPayload((5, 3), "i8", payload.payload)
    elif mutation == "dtype":
        payload = TensorPayload((3, 5), "u8", payload.payload)
    elif mutation == "extent":
        payload = TensorPayload((3, 5), "i8", payload.payload[:14])
    elif mutation == "sidecars":
        blobs = None
    else:
        cb["tensors"]["X"]["data"] = [0] * 15
    with pytest.raises(ValueError):
        render_harness(cb, target="synthetic", inputs={"X": payload}, contract=_contract(), blobs=blobs)


def test_source_drift_keeps_previous_blob_and_span_extent_is_frozen(tmp_path):
    path, payload = _payload(tmp_path)
    blobs = {}
    render_harness(_program(), target="synthetic", inputs={"X": payload}, contract=_contract(), blobs=blobs)
    payload.payload.extend(WeightPayload.zeros(16))
    assert len(blobs["tensor_0"]["bytes"]) == 15
    previous = tmp_path / "harness_blob_tensor_0.bin"
    previous.write_bytes(b"previous")
    path.write_bytes(bytes(15))
    with pytest.raises(ValueError, match="changed"):
        stage_harness_blobs(tmp_path, blobs)
    assert previous.read_bytes() == b"previous"


def test_capture_file_and_runtime_input_are_streamed_without_logical_list_roundtrip(tmp_path):
    raw = bytes(range(15))
    header = json.dumps({"w": {"shape": [3, 5], "dtype": "I8", "data_offsets": [0, 15]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw)
    np.savez(tmp_path / "inputs.npz", in0=np.arange(6, dtype=np.int8).reshape(2, 3))
    (tmp_path / "input_order.json").write_text('{"x":0}')
    cb = {"tensors": {"arg0": {"shape": [3, 5], "dtype": "i8"}, "arg1": {"shape": [2, 3], "dtype": "i8"}},
          "kernel_abi": {"args": [{"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"}]}}
    plan = bundle_pack.plan(cb, row_pitch_elements=16)
    source, report = capture_source.capture_tensor_source(
        tmp_path, plan, weight_manifest={"0": {"weight": "w"}, "1": {"name": "x"}}, streamed=True
    )
    a, b = source("w"), source("x")
    assert type(a) is WeightPayload and bytes(a) == raw
    assert type(b) is WeightPayload and bytes(b) == bytes(range(6))
    assert report.origin == {"arg0": "safetensors", "arg1": "runtime_input"}
    blobs = {"tensor": {"bytes": a, "align": 16, "elems": 15, "section": ".data"}}
    stage_harness_blobs(tmp_path, blobs)
    receipt = json.loads((tmp_path / "harness_blobs.json").read_bytes())
    assert receipt["schema"] == "merlin.harness_blobs.v2"
    assert receipt["blobs"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
