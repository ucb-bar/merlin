"""The reference build's caches and worker pool give the bytes of the plain, serial build.

A one-program stage (an int8 contraction against a transposed weight, the weight in the stage's
safetensors) is run as a chain over several inputs: once per case with nothing kept, once with one
:class:`ChainCache` across the cases.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.model_check import inputs as INP
from merlin.model_check.models import Model, Program
from merlin.runtime import dispatch_runtime as DR
from merlin.runtime import linalg_numpy as LN

_CONTRACTION = (
    "{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, "
    'affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}'
)
_PROGRAM = """
module {
  func.func @forward(%x: tensor<2x4xi8>, %w: tensor<3x4xi8>) -> tensor<2x3xi32> {
    %t0 = tensor.empty() : tensor<4x3xi8>
    %wt = linalg.transpose ins(%w : tensor<3x4xi8>) outs(%t0 : tensor<4x3xi8>) permutation = [1, 0]
    %c0 = arith.constant 0 : i32
    %s = tensor.splat %c0 : tensor<2x3xi32>
    %m = linalg.generic CONTRACTION ins(%x, %wt : tensor<2x4xi8>, tensor<4x3xi8>) outs(%s : tensor<2x3xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<2x3xi32>
    return %m : tensor<2x3xi32>
  }
}
""".replace("CONTRACTION", _CONTRACTION)
_WEIGHT = np.arange(-6, 6, dtype=np.int8).reshape(3, 4)


def _stage(root: Path) -> Path:
    stage = root / "stage"
    stage.mkdir(parents=True)
    data = _WEIGHT.tobytes()
    header = json.dumps({"w": {"dtype": "I8", "shape": [3, 4], "data_offsets": [0, len(data)]}}).encode()
    (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
    manifest = {"0": {"kind": "input", "name": "x"}, "1": {"kind": "param", "weight": "w"}}
    (stage / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (stage / "model.mlir").write_text(_PROGRAM)
    np.savez(stage / "inputs.npz", in0=np.zeros((2, 4), np.int8))
    return stage


def _setup(tmp_path: Path, count: int = 4):
    stage = _stage(tmp_path)
    catalog = CAT.ProgramCatalog(
        target="t",
        stage=stage,
        ref_prepared=stage / "model.mlir",
        dev_prepared=stage / "model.mlir",
        routing={},
        sidecar={},
        interfaces={},
    )
    program = Program(name="p", stage="stage", inputs=({"from": "input", "key": "x"},))
    model = Model(
        name="m",
        workload="w",
        chain={"kind": "single"},
        programs=(program,),
        judged={"kind": "logits", "outputs": [{"program": "p", "output": 0}]},
    )
    rng = np.random.default_rng(3)
    cases = []
    for k in range(count):
        here = tmp_path / "bundle" / "m" / "visible" / f"v{k:02d}"
        here.mkdir(parents=True)
        x = rng.integers(-128, 128, size=(2, 4), dtype=np.int8)
        np.savez(here / "inputs.npz", x=x)
        INP._save_npz(here / "fp.npz", logits=(x.astype(np.float32) @ _WEIGHT.T.astype(np.float32)))  # noqa: SLF001
        name = f"visible/v{k:02d}"
        cases.append(
            {
                "split": "visible",
                "ident": name[-3:],
                "draw": 0,
                "inputs": here / "inputs.npz",
                "out": here,
                "name": name,
            }
        )
    return model, catalog, cases


def test_cached_arguments_are_read_once_and_are_the_same_read_only_objects(tmp_path, monkeypatch) -> None:
    stage = _stage(tmp_path)
    plans = [0]
    real = DR._forward_plan  # noqa: SLF001

    def counted(*args, **kwargs):
        plans[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(DR, "_forward_plan", counted)
    cache = DR.ForwardArgsCache()
    first = CAT.capture_arguments(stage, {"in0": np.ones((2, 4), np.int8)}, tmp_path / "a", weights_cache=cache)
    again = CAT.capture_arguments(stage, {"in0": np.full((2, 4), 2, np.int8)}, tmp_path / "b", weights_cache=cache)
    plain = CAT.capture_arguments(stage, {"in0": np.full((2, 4), 2, np.int8)}, tmp_path / "c")
    assert plans[0] == 2  # one cached plan for both shadows of the stage, one uncached call
    assert first[1] is again[1] and cache.is_constant(again[1]) and not again[1].flags.writeable
    assert not cache.is_constant(again[0]) and again[0].tolist() == [[2] * 4] * 2
    assert [a.tobytes() for a in again] == [a.tobytes() for a in plain]


def test_a_cached_chain_gives_the_bytes_of_a_plain_one_and_transposes_once(tmp_path, monkeypatch) -> None:
    model, catalog, cases = _setup(tmp_path)
    arrays = [dict(np.load(c["inputs"])) for c in cases]
    plain = [CH.judged_outputs(model, CH.run_reference(model, {"p": catalog}, a, tmp_path / "s0")) for a in arrays]
    seen = [0]
    real = LN.evaluate_op

    def counted(op, env, **kwargs):
        seen[0] += op.name == "linalg.transpose"
        return real(op, env, **kwargs)

    monkeypatch.setattr(LN, "evaluate_op", counted)
    cache = CH.ChainCache()
    cached = [
        CH.judged_outputs(model, CH.run_reference(model, {"p": catalog}, a, tmp_path / "s1", cache=cache))
        for a in arrays
    ]
    assert seen[0] == 1
    assert [o.tobytes() for r in cached for o in r] == [o.tobytes() for r in plain for o in r]
    assert np.array_equal(cached[1][0], arrays[1]["x"].astype(np.int64) @ _WEIGHT.T.astype(np.int64))
