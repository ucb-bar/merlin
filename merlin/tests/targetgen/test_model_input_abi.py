"""Whole-model input declarations are the captured loader/interface ABI, never value guesses.

JSON erases tensor dtype (the capture intentionally serializes through float64), so an integral-looking
float input and an int tensor can carry identical decoded values.  These tests bind all three records:
the live loader tensors, the captured linalg ``@forward`` runtime arguments, and ``capsule.yaml``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_source as S
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir


ROOT = merlin_dir() / "contract/capsules/model"
MODELS = ("M2_microvit_gemmini", "M3_host_island_seam_gemmini")


def _dir(name: str) -> Path:
    return ROOT / name


def _loader_abi(name: str) -> list[dict]:
    """Read torch tensor leaf types in the interpreter that owns torch (not from decoded values)."""
    python = S._m2m_python()
    if not python.is_file():
        pytest.skip(f"model2MLIR python is unavailable: {python}")
    script = r'''
import importlib.util, json, sys, torch
p = sys.argv[1]
s = importlib.util.spec_from_file_location("_loader_abi_test", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
_model, inputs = m.get_model_and_inputs()
leaves, _ = torch.utils._pytree.tree_flatten(inputs)
spelling = {torch.bool:"i1", torch.int8:"i8", torch.uint8:"ui8", torch.int16:"i16",
            torch.int32:"i32", torch.int64:"i64", torch.float16:"f16", torch.bfloat16:"bf16",
            torch.float32:"f32", torch.float64:"f64"}
print(json.dumps([{"shape":list(x.shape), "dtype":spelling[x.dtype]} for x in leaves]))
'''
    proc = subprocess.run([str(python), "-c", script, str(_dir(name) / "capsule.pytorch.py")],
                          check=True, capture_output=True, text=True, timeout=60)
    return json.loads(proc.stdout.splitlines()[-1])


@pytest.mark.parametrize("name", MODELS)
def test_shipped_model_input_declarations_equal_loader_and_interface_abi(name):
    d = _dir(name)
    cap = yaml.safe_load((d / "capsule.yaml").read_text(encoding="utf-8"))
    loader = _loader_abi(name)
    parsed = parse_linalg_mlir((d / "capsule.interface.mlir").read_text(encoding="utf-8"))
    runtime = parsed["args"][-len(loader):]
    declared = [{"shape": x["shape"], "dtype": x["dtype"]} for x in cap["inputs"]]
    interface = [{"shape": x["shape"], "dtype": x["dtype"]} for x in runtime]
    assert declared == loader == interface


def _artifact(name: str, abi: list[dict] | None = None) -> S.CapsuleArtifacts:
    d = _dir(name)
    golden = yaml.safe_load((d / "golden.yaml").read_text(encoding="utf-8"))
    order = golden["oracle_provenance"]["arg_order"][:-1]
    prov = golden["oracle_provenance"]["inputs"]
    inputs = [np.asarray(prov[n]["decoded"]).reshape(prov[n]["shape"]).tolist() for n in order]
    return S.CapsuleArtifacts(
        op="model", dtype="int8",
        pytorch_src=(d / "capsule.pytorch.py").read_text(encoding="utf-8"),
        linalg_mlir=(d / "capsule.interface.mlir").read_text(encoding="utf-8"),
        inputs=inputs, golden=next(iter(golden["outputs"].values())),
        weights_path=str(d / "capsule.weights.safetensors"),
        meta={"input_abi": abi if abi is not None else _loader_abi(name)},
    )


@pytest.mark.parametrize("name", MODELS)
def test_model_input_abi_refuses_loader_interface_dtype_drift(name):
    art = _artifact(name)
    bad = dataclasses.replace(art, meta={"input_abi": [dict(x) for x in art.meta["input_abi"]]})
    bad.meta["input_abi"][0]["dtype"] = "i64" if bad.meta["input_abi"][0]["dtype"] != "i64" else "f32"
    with pytest.raises(S.M2MUnavailable, match="disagrees with captured @forward runtime arg"):
        S.model_input_abi(bad)


def _binding():
    from merlin.targetgen.corpus_spec import CorpusBinding
    classes = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD",
               "COMPUTE_PRELOADED", "MVOUT"]
    return CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="i32", integer=True,
        tiers=["L0", "L1", "L2", "L3"], compare="exact_int", atol=0.03125, rtol=0.02,
        classes_for=lambda **_: classes)


def _entry(name: str) -> dict:
    cap = yaml.safe_load((_dir(name) / "capsule.yaml").read_text(encoding="utf-8"))
    attrs = cap["operation"]["attributes"]
    return {
        "name": name, "cat": "model", "kind": "model", "op": "model",
        "model": attrs["model"], "loader": str(_dir(name) / "capsule.pytorch.py"),
        "operand_dtype": "int8", "out": attrs["out"], "label": cap["label"],
        "source_reference": cap["source_reference"], "gate": cap["gate"],
        **({"lanes": cap["lanes"]} if cap.get("lanes") else {}),
    }


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("name", MODELS)
def test_model_writer_roundtrip_is_byte_stable_and_preserves_grounded_assets(name, tmp_path):
    """Recapture and write twice: no generated output may drift from its peer or grounded assets."""
    source = S.PytorchRefSource(timeout=900)
    if not source.available():
        pytest.skip("model2MLIR capture environment unavailable")
    out1, out2 = tmp_path / "one", tmp_path / "two"
    d1 = S.write_model_capsule(_entry(name), _binding(), out1, source=source)
    d2 = S.write_model_capsule(_entry(name), _binding(), out2, source=source)
    assert _tree_digest(d1) == _tree_digest(d2)

    shipped = _dir(name)
    for leaf in ("capsule.interface.mlir", "capsule.pytorch.py", "capsule.weights.safetensors",
                 "golden.yaml", "expected_instruction_coverage.yaml"):
        assert (d1 / leaf).read_bytes() == (shipped / leaf).read_bytes(), leaf
    generated = yaml.safe_load((d1 / "capsule.yaml").read_text(encoding="utf-8"))
    current = yaml.safe_load((shipped / "capsule.yaml").read_text(encoding="utf-8"))
    assert generated["inputs"] == current["inputs"]
