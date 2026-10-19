"""The reference build's caches and worker pool give the bytes of the plain, serial build.

A one-program stage (an int8 contraction against a transposed weight, the weight in the stage's
safetensors) is run as a chain over several inputs: once per case with nothing kept, once with one
:class:`ChainCache` across the cases, and in a pool of worker processes, each with its own scratch.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.model_check import inputs as INP
from merlin.model_check import references as REF
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
    job = REF.CaseJob(model=model, catalogs={"p": catalog}, blobs=tmp_path / "bundle" / "m" / "blobs", baseline=False)
    return model, catalog, cases, job


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
    model, catalog, cases, _job = _setup(tmp_path)
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


def test_a_pool_gives_the_serial_bytes_in_case_order_with_a_scratch_per_worker(tmp_path) -> None:
    _model, _catalog, cases, job = _setup(tmp_path, count=5)
    serial = [REF.reference_case(job, c, cache=None, scratch=tmp_path / "serial") for c in cases]
    written = [(c["out"] / "int8.npz").read_bytes() for c in cases]
    for c in cases:
        (c["out"] / "int8.npz").unlink()
    lines: list[str] = []
    pooled = REF._pooled_cases(job, cases, 2, tmp_path / "work", lines.append)  # noqa: SLF001
    assert [r["metrics"] for r in pooled] == [r["metrics"] for r in serial]
    assert all(row["int8_numeric_contract"] == LN.numeric_contract() for row in [*serial, *pooled])
    expected_evaluator = hashlib.sha256(Path(LN.__file__).read_bytes()).hexdigest()
    assert all(row["evaluator_sha256"] == expected_evaluator for row in [*serial, *pooled])
    assert [(c["out"] / "int8.npz").read_bytes() for c in cases] == written
    workers = tmp_path / "work" / "workers"
    claimed = {i: int((workers / "claims" / str(i)).read_text()) for i in range(len(cases))}
    assert set(claimed.values()) <= {0, 1}
    for k in set(claimed.values()):  # a worker binds its inputs in its own shadow directory
        assert (workers / str(k) / "chain" / "p" / "inputs.npz").is_file()
        log = (workers / str(k) / "worker.log").read_text().splitlines()
        assert [line.split()[1].rstrip(":") for line in log] == [cases[i]["name"] for i, w in claimed.items() if w == k]


def test_a_pool_refuses_a_job_it_cannot_send_to_a_worker(tmp_path) -> None:
    _model, _catalog, cases, job = _setup(tmp_path, count=2)
    job.keep_call = lambda _symbol, ordinal: ordinal < 1
    with pytest.raises(INP.InputError, match="picklable"):
        REF._pooled_cases(job, cases, 2, tmp_path / "work", print)  # noqa: SLF001


def test_reference_policy_mutation_during_chain_cannot_publish_judged_outputs(tmp_path, monkeypatch):
    _model, _catalog, cases, job = _setup(tmp_path, count=1)
    original = CH.run_reference
    policy = LN.numeric_contract()

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(LN, "numeric_contract", lambda: {**policy, "version": "changed during execution"})
        return result

    monkeypatch.setattr(CH, "run_reference", changed)
    with pytest.raises(INP.InputError, match="source/policy changed"):
        REF.reference_case(job, cases[0], cache=None, scratch=tmp_path / "work")
    assert not (cases[0]["out"] / "int8.npz").exists()


def test_selected_math_case_writes_byte_bound_reference_declaration(tmp_path):
    import shutil

    if shutil.which("cc") is None:
        pytest.skip("needs a native C compiler")
    _model, _catalog, cases, job = _setup(tmp_path, count=1)
    job.host_math_policy = "portable_exp_trig"
    row = REF.reference_case(job, cases[0], cache=None, scratch=tmp_path / "work")
    receipt = json.loads((cases[0]["out"] / "reference_numeric_contract.json").read_text())
    assert receipt["numeric_contract"] == row["int8_numeric_contract"]
    assert receipt["numeric_contract"]["host_math"]["policy"] == "portable_exp_trig"
    assert receipt["outputs_sha256"] == hashlib.sha256((cases[0]["out"] / "int8.npz").read_bytes()).hexdigest()
    # Exercise the real producer -> bundle seal -> direct consumer boundary.
    from merlin.model_check import bundle as B
    from merlin.model_check import full_quantized as FQ

    root = job.blobs.parent.parent
    configuration = root / "configuration" / "models.yaml"
    configuration.parent.mkdir()
    configuration.write_text(json.dumps({"full": {"host_math_policy": "portable_exp_trig"}}))
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in paths))
    (root / "MANIFEST.json").write_text(json.dumps({
        "schema": B.MANIFEST_SCHEMA, "captures": str(tmp_path / "captures"), "models": {"m": {}},
        "seal": {"sums": "SHA256SUMS", "sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest()},
    }))
    frozen, validation = FQ.sealed_reference(B.Bundle(root), job.model, configuration, "visible/v00")
    assert len(frozen) == 1
    assert validation["numeric_contract"]["numeric_contract"] == receipt["numeric_contract"]


def test_reference_configuration_mutation_cannot_publish_judged_outputs(tmp_path, monkeypatch):
    _model, _catalog, cases, job = _setup(tmp_path, count=1)
    configuration = tmp_path / "models.yaml"
    configuration.write_text("selected configuration")
    job.host_math_configuration = (configuration, hashlib.sha256(configuration.read_bytes()).hexdigest())
    original = CH.run_reference

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        configuration.write_text("different configuration")
        return result

    monkeypatch.setattr(CH, "run_reference", changed)
    with pytest.raises(INP.InputError, match="configuration changed"):
        REF.reference_case(job, cases[0], cache=None, scratch=tmp_path / "work")
    assert not (cases[0]["out"] / "int8.npz").exists()


def test_the_worker_count_is_bounded_by_cases_and_memory() -> None:
    gib = 1 << 30
    assert REF.worker_count(4, 64, 26 * gib, available=123 * gib) == 4
    assert REF.worker_count(8, 64, 26 * gib, available=90 * gib) == 3
    assert REF.worker_count(8, 2, gib, available=123 * gib) == 2
    assert REF.worker_count(4, 64, 200 * gib, available=123 * gib) == 1
    assert REF.worker_count(0, 64, gib, available=123 * gib) == 1
    assert REF._shards(list(range(5)), 2) == [[0, 2, 4], [1, 3]]  # noqa: SLF001
