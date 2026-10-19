"""Full recurrent references reuse only byte-corresponding steps and constant work."""
from __future__ import annotations

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.model_check import full as F
from merlin.model_check import full_reference_cache as RC
from merlin.model_check import twins as TW
from merlin.model_check.models import Model, Program
from merlin.runtime import linalg_numpy as LN

_PROGRAM = '''module {
  func.func @forward(%x: tensor<2x4xi8>, %w: tensor<4x4xi8>) -> tensor<2x4xi32> {
    %e = tensor.empty() : tensor<4x4xi8>
    %wt = linalg.transpose ins(%w : tensor<4x4xi8>) outs(%e : tensor<4x4xi8>) permutation = [1, 0]
    %z = arith.constant 0 : i32
    %s = tensor.splat %z : tensor<2x4xi32>
    %m = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>,
         affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>],
         iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%x, %wt : tensor<2x4xi8>, tensor<4x4xi8>) outs(%s : tensor<2x4xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %mul = arith.muli %ae, %be : i32
      %sum = arith.addi %acc, %mul : i32
      linalg.yield %sum : i32
    } -> tensor<2x4xi32>
    return %m : tensor<2x4xi32>
  }
}'''


def _setup(root, steps=4):
    stage = root / "stage"
    stage.mkdir(parents=True)
    w = np.eye(4, dtype=np.int8)
    w[0, 1] = 1
    header = json.dumps({"w": {"dtype": "I8", "shape": [4, 4], "data_offsets": [0, w.nbytes]}}).encode()
    (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + w.tobytes())
    (stage / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "name": "x"}, "1": {"kind": "param", "weight": "w"}}))
    (stage / "model.mlir").write_text(_PROGRAM)
    x = np.arange(8, dtype=np.int8).reshape(2, 4)
    np.savez(stage / "inputs.npz", in0=x)
    catalog = CAT.ProgramCatalog("t", stage, stage / "model.mlir", stage / "model.mlir", {}, {}, {})
    program = Program("step", "stage", ({"from": "input", "key": "x", "carry": 0},), steps)
    model = Model("m", "w", {"kind": "flow", "steps": steps}, (program,),
                  {"kind": "logits", "outputs": [{"program": "step", "output": 0}]})
    bundle_root = root / "bundle"
    bundle_root.mkdir()
    manifest = {"models": {"m": {}}}
    (bundle_root / "MANIFEST.json").write_text(json.dumps(manifest))
    bundle = SimpleNamespace(root=bundle_root, catalog=lambda *_: catalog, manifest=lambda: manifest)
    return model, catalog, bundle, {"x": x}


@pytest.fixture(autouse=True)
def _bounded_evaluator_identity(monkeypatch):
    # Cache tests mutate their own tiny evaluator identity; no dependency-directory scan per assertion.
    monkeypatch.setattr(RC, "evaluator_identity", lambda: {"source": "selected-test-evaluator"})


@pytest.mark.parametrize("steps", [10, 32])
def test_recurrent_references_keep_exact_state_and_transpose_constants_once(tmp_path, monkeypatch, steps):
    model, catalog, bundle, arrays = _setup(tmp_path, steps)
    plain = CH.run_reference(model, {"step": catalog}, arrays, tmp_path / "plain")
    seen = [0]
    real = LN.evaluate_op

    def counted(op, env, **kwargs):
        seen[0] += op.name == "linalg.transpose"
        return real(op, env, **kwargs)

    monkeypatch.setattr(LN, "evaluate_op", counted)
    cached = F.reference_chain(model, bundle, arrays, {"step": steps}, tmp_path / "cache")
    assert seen[0] == 1
    for actual, expected in zip(cached.steps["step"], plain.steps["step"], strict=True):
        assert RC.same_outputs(actual, expected)
    assert not np.array_equal(cached.steps["step"][0][0], cached.steps["step"][-1][0])
    # A second invocation really reads bound, complete step evidence instead of evaluating any op.
    monkeypatch.setattr(LN, "evaluate", lambda *_a, **_k: pytest.fail("corresponding reference should be reused"))
    again = F.reference_chain(model, bundle, arrays, {"step": steps}, tmp_path / "cache")
    assert RC.same_outputs(CH.judged_outputs(model, again), CH.judged_outputs(model, cached))


def test_old_unkeyed_steps_are_ignored_and_partial_counts_do_not_alias(tmp_path):
    model, _catalog, bundle, arrays = _setup(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "step.0.npz", out0=np.full((2, 4), -77, np.int8))
    short = F.reference_chain(model, bundle, arrays, {"step": 2}, cache)
    full = F.reference_chain(model, bundle, arrays, {"step": 4}, cache)
    assert len(short.steps["step"]) == 2 and len(full.steps["step"]) == 4
    assert full.steps["step"][0][0][0, 0] != -77
    assert len(list((cache / "v1").iterdir())) == 2


@pytest.mark.parametrize("mutation", ["input", "weights", "reference", "evaluator", "manifest", "chain"])
def test_each_reference_input_invalidates_a_previously_valid_context(tmp_path, monkeypatch, mutation):
    model, catalog, bundle, arrays = _setup(tmp_path)
    counts = {"step": 4}
    before = RC.material(model, bundle, arrays, counts, {"step": catalog})
    if mutation == "input":
        arrays["x"][0, 0] += 1
    elif mutation == "weights":
        path = catalog.stage / "weights.safetensors"
        value = bytearray(path.read_bytes())
        value[-1] ^= 1
        path.write_bytes(value)
    elif mutation == "reference":
        catalog.ref_prepared.write_text(_PROGRAM + "\n// equivalent source change\n")
    elif mutation == "evaluator":
        monkeypatch.setattr(RC, "evaluator_identity", lambda: {"source": "different"})
    elif mutation == "manifest":
        (bundle.root / "MANIFEST.json").write_text('{"models":{},"revision":2}')
    else:
        binding = {"from": "constant", "value": [[1] * 4] * 2, "dtype": "int8"}
        model = replace(model, programs=(replace(model.programs[0], inputs=(binding,)),))
    after = RC.material(model, bundle, arrays, counts, {"step": catalog})
    assert RC.StepCache(tmp_path, before).key != RC.StepCache(tmp_path, after).key


@pytest.mark.parametrize("damage", ["same_size_npz", "receipt", "arity", "missing"])
def test_corrupted_or_incomplete_steps_are_cache_misses(tmp_path, damage):
    cache = RC.StepCache(tmp_path, {"source": "v1"})
    inputs = {"in0": np.array([1], np.int32)}
    outputs = [np.array([7, 8], np.int32)]
    cache.put("p", 0, inputs, outputs)
    path = cache._path("p", 0)  # noqa: SLF001
    if damage == "same_size_npz":
        value = bytearray(path.read_bytes())
        value[-1] ^= 1
        path.write_bytes(value)
    elif damage == "receipt":
        receipt = path.with_suffix(".json")
        row = json.loads(receipt.read_text())
        row["identity"] = {"source": "elsewhere"}
        receipt.write_text(json.dumps(row))
    elif damage == "missing":
        path.unlink()
    assert cache.get("p", 0, inputs, 2 if damage == "arity" else 1) is None


def test_actual_bound_inputs_are_checked_even_when_the_global_context_matches(tmp_path):
    cache = RC.StepCache(tmp_path, {"source": "v1"})
    cache.put("p", 0, {"x": np.array([1])}, [np.array([2])])
    assert cache.get("p", 0, {"x": np.array([9])}, 1) is None


def test_changes_during_evaluation_publish_no_answers(tmp_path, monkeypatch):
    model, _catalog, bundle, arrays = _setup(tmp_path)
    original = LN.evaluate

    def changed(*args, **kwargs):
        values = original(*args, **kwargs)
        arrays["x"][0, 0] += 1
        return values

    monkeypatch.setattr(LN, "evaluate", changed)
    with pytest.raises(F.FullCheckError, match="changed during the chain"):
        F.reference_chain(model, bundle, arrays, {"step": 4}, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("v1/**/*.npz"))


def test_reference_consistency_never_truncates_outputs_or_changes_dtype():
    values = [np.array([1], np.int32), np.array([2], np.int32)]
    assert not RC.same_outputs(values, values[:1])
    assert not RC.same_outputs(values[:1], values)
    assert not RC.same_outputs(values[:1], [values[0].astype(np.int64)])
    assert RC.same_outputs(values, [v.copy() for v in values])


def test_missing_cached_twins_are_diagnosed_before_build_without_changing_policy(tmp_path):
    model, catalog, bundle, _arrays = _setup(tmp_path)
    catalog.interfaces = {"device0": "a typed interface"}
    model = replace(model, extra={"full": {"differential": "cached"}})
    row = RC.cached_twin_readiness(bundle, model, {"step": 4}, ["visible/v00"])
    assert row["status"] == "pending" and row["pending"] == ["step:visible/v00"]
    assert model.extra["full"]["differential"] == "cached"
    catalog.interfaces = {}
    assert RC.cached_twin_readiness(bundle, model, {"step": 4}, ["visible/v00"])["status"] == "declared"


def test_cached_twin_declared_coverage_requires_a_real_index_and_every_step(tmp_path):
    model, catalog, bundle, _arrays = _setup(tmp_path)
    catalog.interfaces = {"device0": "interface"}
    model = replace(model, extra={"full": {"differential": "cached"}})
    twin = {"elf_sha256": "image", "engine": {"sha256": "simulator"}}
    key = TW.key(twin)
    entry = {"twin_key": key, "twin": twin, "inputs": {"visible/v00": ["a", "b", "c", "d"]}}
    bundle.manifest()["models"]["m"]["twins"] = {"step": entry}
    path = bundle.root / "m/twins/step" / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": TW.SCHEMA, "twin": twin, "steps": {"a": {}, "b": {}, "c": {}}}))
    assert RC.cached_twin_readiness(bundle, model, {"step": 4}, ["visible/v00"])["status"] == "pending"
    blob = bundle.root / "m/twins/blobs" / ("f" * 64 + ".npy")
    blob.parent.mkdir()
    np.save(blob, np.array([1]))
    path.write_text(json.dumps({"schema": TW.SCHEMA, "twin": twin,
                               "steps": {k: {"outputs": ["f" * 64]} for k in "abcd"}}))
    row = RC.cached_twin_readiness(bundle, model, {"step": 4}, ["visible/v00"])
    assert row["status"] == "declared" and "still need validation" in row["detail"]
    blob.unlink()
    assert RC.cached_twin_readiness(bundle, model, {"step": 4}, ["visible/v00"])["status"] == "pending"
    row = RC.cached_twin_readiness(bundle, replace(model, extra={}), {"step": 4}, ["visible/v00"])
    assert row["status"] == "required"


def test_partial_chains_select_frozen_prefixes_after_strict_full_arity_validation():
    programs = (Program("prefix", None, steps=1), Program("decode", None, steps=32))
    model = Model("m", "w", {"kind": "causal"}, programs,
                  {"outputs": [{"program": "prefix", "output": 0}, {"program": "decode", "output": 0}]})
    frozen = [np.array([i]) for i in range(33)]
    chosen = RC.selected_outputs(model, {"prefix": 1, "decode": 2}, frozen)
    assert [value.item() for value in chosen] == [0, 1, 2]
    with pytest.raises(ValueError, match="full chain declares 33"):
        RC.selected_outputs(model, {"prefix": 1, "decode": 2}, frozen[:-1])
    with pytest.raises(ValueError, match="full chain declares 33"):
        RC.selected_outputs(model, {"prefix": 1, "decode": 2}, frozen + [np.array([99])])


def test_reference_resource_symlinks_bind_content_even_without_size_or_time_changes(tmp_path):
    import os

    resources = tmp_path / "resources"
    resources.mkdir()
    payload = resources / "constant.npy"
    np.save(payload, np.array([1, 2], np.int32))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "arrays").symlink_to(resources, target_is_directory=True)
    before = RC.tree_digest(reference)
    info = payload.stat()
    np.save(payload, np.array([3, 4], np.int32))
    os.utime(payload, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert payload.stat().st_size == info.st_size
    assert RC.tree_digest(reference) != before


def test_missing_reference_resources_refuse_a_cache_identity(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    (root / "constant.npy").symlink_to(tmp_path / "missing.npy")
    with pytest.raises(FileNotFoundError):
        RC.tree_digest(root)
