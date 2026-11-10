"""Effect-profile byte/ABI fixtures; no claimed kernel execution evidence."""

import hashlib
import json
from pathlib import Path
import struct

import pytest

from merlin.frontends.compile_inputs import stage_compile_inputs


def record(raw):
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def fixture(root,change=None):
    root.mkdir()
    model = b"module {func.func @check() {return}}"
    ttir = b"source-only TTIR protocol fixture"
    spec = {"schema":"t2m.shared-tensor-abi.v6","entry":"check","buffers":[],"runtime_scalars":[],
        "scalar_constants":[],"grid":[1],"effect_only":True}
    receipt_schema = "t2m.effect-kernel-receipt.v1"
    manifest = {}
    count = {"complete":True,"inputs":0,"outputs":0}
    if change == "old-receipt": receipt_schema = "t2m.capture-receipt.v2"
    elif change == "missing-selection": spec["effect_only"] = False
    elif change == "hidden-write":
        spec["buffers"] = [{"name":"x","shape":[2],"dtype":"f32","effect":"readwrite"}]
        model = b"module {func.func @check(%x:tensor<2xf32>)->tensor<2xf32> {return %x:tensor<2xf32>}}"
        manifest = {"0":{"kind":"input","name":"x"}}
        count.update(inputs=1,outputs=1)
    elif change == "hidden-result": model = b"module {func.func @check()->i32 {%c=arith.constant 1:i32 return %c:i32}}"
    elif change == "incomplete-arity": count["outputs"] = 1
    elif change == "scalar":
        spec["runtime_scalars"] = [{"name":"flag","dtype":"i32","source_arg":0,"bounds":[-2,2]}]
        model = b"module {func.func @check(%flag:i32) {return}}"
        manifest = {"0":{"kind":"input","name":"flag"}}
        count["inputs"] = 1
    trace = {"schema":"t2m.shared-tensor-provenance.v1","spec":spec,"source_operations":[],"emitted_operations":[],
        "source_sha256":record(ttir)["sha256"],"mlir_sha256":record(model)["sha256"]}
    encoded = lambda value:json.dumps(value).encode()
    files = {"model.mlir":model,"source.ttir":ttir,"spec.json":encoded(spec),"provenance.json":encoded(trace),
        "weights.safetensors":struct.pack("<Q",8)+b"{}      ","weights.safetensors.manifest.json":encoded(manifest),
        **{name:b"EVALUATOR "+name.encode() for name in ("inputs.npz","reference.json","execution.json","native/outcome.json","native/stdout.bin")}}
    for name,raw in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    receipt = {"schema":receipt_schema,"profile":"shared-functional-tensor","materialized_abi":count,
        "artifacts":{name:record(raw) for name,raw in files.items()}}
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return receipt


@pytest.mark.parametrize("change", [None,"scalar"])
def test_void_and_scalar_only_entries_exclude_all_evaluator_outcomes(tmp_path,monkeypatch,change):
    source,target = tmp_path / "capture",tmp_path / "compiler"
    receipt = fixture(source,change)
    original = Path.read_bytes
    def read(path):
        if path.name in ("inputs.npz","reference.json","execution.json","outcome.json","stdout.bin"):
            raise AssertionError("compiler read effect evaluator data")
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",read)
    staged = stage_compile_inputs(source,target)
    signature = json.loads((target / "signature.json").read_text())
    assert staged["source_profile"] == "standalone-effect-kernel" and signature["outputs"] == []
    assert len(signature["inputs"]) == (1 if change == "scalar" else 0)
    contents = b"".join((target / name).read_bytes() for name in staged["members"])
    for name in ("inputs.npz","reference.json","execution.json","native/outcome.json","native/stdout.bin"):
        assert receipt["artifacts"][name]["sha256"].encode() not in contents
    assert b"EVALUATOR" not in contents


@pytest.mark.parametrize("change", ["old-receipt","missing-selection","hidden-write","hidden-result","incomplete-arity"])
def test_effect_profile_cannot_discard_results_or_writes(tmp_path,change):
    source,target = tmp_path / "capture",tmp_path / "compiler"
    fixture(source,change)
    with pytest.raises(ValueError):
        stage_compile_inputs(source,target)
    assert not target.exists()
