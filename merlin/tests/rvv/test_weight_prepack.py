"""`prepack_weight_layout`: apply the weight layout once at build time, not once per inference.

The lever's payoff is measured elsewhere (interleaved on the live K1, small_llama int8 whole model:
3,548,286/3,574,361/3,561,602 ns stock vs 2,125,388/2,086,712/2,127,671 ns prepacked -- 1.70x against
a 2.6% noise band). What is worth GATING is everything that could make it silently wrong or silently
absent:

* the feature must be REGISTERED, because `wholemodel_proposer._composes` catches the `KeyError` an
  unregistered name raises and returns False -- an unregistered lever is never proposed AND never
  reported as rejected;
* it must carry NO pipeline/schedule/cflags hook, which is what keeps a build that names no feature
  byte-identical to the one that existed before it (verified end-to-end at the object level: with an
  empty feature set `model.o` is sha256 9f02ae4c…2758f both before and after this change);
* the rewrite must be bit-exact and its provenance must name its OWN weights;
* and every way the STORED bytes can defeat the IR analysis must REFUSE the build rather than fall
  back to the stock bundle, which would report the lever as applied while measuring the baseline.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.llvmlower import weight_prepack as wp

MLIR = '''builtin.module attributes {prov.weights_file = "SRC/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x3xi8>, %1: tensor<3xf32>, %2: tensor<2x3xi8>) -> tensor<3x2xi8> {
    %3 = tensor.empty() : tensor<3x2xi8>
    %4 = linalg.transpose ins(%0:tensor<2x3xi8>) outs(%3:tensor<3x2xi8>) permutation = [1, 0]
    func.return %4 : tensor<3x2xi8>
  }
}
'''

W0 = np.arange(6, dtype=np.int8).reshape(2, 3)
S0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)


def _write_safetensors(path: Path, header: dict, payload: bytes) -> None:
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)


def _bundle(root: Path, *, manifest: dict | None = None, header: dict | None = None,
            mlir: str = MLIR, extra_files: dict[str, bytes] | None = None) -> Path:
    """A minimal capture bundle whose arg 0 is a sole-use transposed weight."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.mlir").write_text(mlir.replace("SRC", str(root)))
    (root / "weights.safetensors.manifest.json").write_text(json.dumps(manifest or {
        "0": {"kind": "param", "weight": "w0", "shape": [2, 3], "dtype": "int8"},
        "1": {"kind": "param", "weight": "s0", "shape": [3], "dtype": "float32"},
        "2": {"kind": "param", "weight": "w1", "shape": [2, 3], "dtype": "int8"},
    }, indent=2))
    _write_safetensors(root / "weights.safetensors", header or {
        "w0": {"dtype": "I8", "shape": [2, 3], "data_offsets": [0, 6]},
        "s0": {"dtype": "F32", "shape": [3], "data_offsets": [6, 18]},
        "w1": {"dtype": "I8", "shape": [2, 3], "data_offsets": [18, 24]},
    }, W0.tobytes() + S0.tobytes() + W0.tobytes())
    np.savez(root / "inputs.npz")
    for name, data in (extra_files or {}).items():
        (root / name).write_bytes(data)
    return root


# ---------------------------------------------------------------------------------------------
# registration -- the difference between "rejected" and "silently never proposed"
# ---------------------------------------------------------------------------------------------
def test_feature_is_registered_and_resolves():
    from merlin.llvmlower.impr_features import get, normalize

    assert wp.ensure_registered() == wp.FEATURE
    assert wp.ensure_registered() == wp.FEATURE          # idempotent
    assert normalize([wp.FEATURE]) == frozenset({wp.FEATURE})
    assert get(wp.FEATURE).name == wp.FEATURE


def test_feature_carries_no_compiler_hook_so_the_baseline_is_untouched():
    """It changes the INPUT BUNDLE, not the compiler. A hook here would mean the frozen baseline
    could drift, and would also make the lever schedule-replacing (unable to compose)."""
    from merlin.llvmlower.impr_features import get

    f = get(wp.ensure_registered())
    assert (f.edit_pipeline, f.edit_schedule, f.edit_cflags) == (None, None, None)
    assert f.schedule_replace is False


def test_the_proposer_ranks_it_and_it_composes():
    from merlin.mining import wholemodel_proposer as prop

    assert wp.FEATURE in [n for n, _ in prop.RANKED_LEVERS]
    assert prop._composes([wp.FEATURE])
    # composes on top of the ranked schedule-replacement lever, so it can stack in a beam search
    assert prop._composes([wp.FEATURE, "perop_register_block"])


# ---------------------------------------------------------------------------------------------
# the rewrite itself
# ---------------------------------------------------------------------------------------------
def test_hoist_is_bit_exact_and_names_its_own_weights(tmp_path):
    from merlin.llvmlower.weights_pack import load_safetensors_header

    src = _bundle(tmp_path / "src")
    before = (src / "weights.safetensors").read_bytes()

    plan = wp.plan(src)
    assert (plan.hoistable, plan.blocked, plan.problems) == (1, 0, ())

    dst, effect = wp.prepacked_bundle(src, cache_root=tmp_path / "cache")
    assert effect["weights_pre_transposed"] == 1 and effect["transposes_removed"] == 1

    hdr, off = load_safetensors_header(dst / "weights.safetensors")
    blob = (dst / "weights.safetensors").read_bytes()[off:]
    s, e = hdr["w0"]["data_offsets"]
    stored = np.frombuffer(blob[s:e], np.int8).reshape(hdr["w0"]["shape"])
    assert np.array_equal(stored, W0.T)                  # bit-exact: a transpose moves elements
    assert hdr["w0"]["shape"] == [3, 2]
    assert hdr["s0"]["shape"] == [3]                     # untouched tensors keep their layout

    man = json.loads((dst / "weights.safetensors.manifest.json").read_text())
    assert man["0"]["shape"] == [3, 2] and man["1"]["shape"] == [3]

    text = (dst / "model.mlir").read_text()
    assert "linalg.transpose" not in text                # the run-time re-layout is gone
    assert "%0: tensor<3x2xi8>" in text                  # ... and the argument carries the layout
    # provenance names THIS bundle's blob, not the source's. The `_pretransposed` bundles on disk
    # get this wrong, which is why anyone reading one concludes it is stock.
    assert f'prov.weights_file = "{dst / "weights.safetensors"}"' in text

    # the shared recapture is never mutated -- other sessions read it
    assert (src / "weights.safetensors").read_bytes() == before


def test_transposed_runtime_input_does_not_block_independent_weight_hoist(tmp_path):
    """The IR-only analyzer sees both transposes; the manifest is the authority on storage.

    A runtime input has no safetensors payload to rewrite, so its transpose must remain in the graph
    while the independent stored weight is still hoisted.  Treating the input as a malformed weight
    made mixed-input models such as RDT2 unable to use this otherwise applicable general pass.
    """
    mlir = MLIR.replace(
        "    func.return %4 : tensor<3x2xi8>",
        "    %5 = tensor.empty() : tensor<3x2xi8>\n"
        "    %6 = linalg.transpose ins(%2:tensor<2x3xi8>) "
        "outs(%5:tensor<3x2xi8>) permutation = [1, 0]\n"
        "    func.return %4 : tensor<3x2xi8>",
    )
    src = _bundle(tmp_path / "src", mlir=mlir, manifest={
        "0": {"kind": "param", "weight": "w0", "shape": [2, 3], "dtype": "int8"},
        "1": {"kind": "param", "weight": "s0", "shape": [3], "dtype": "float32"},
        "2": {"kind": "input", "shape": [2, 3], "dtype": "int8"},
    })

    candidate = wp.plan(src)
    assert (candidate.hoistable, candidate.blocked, candidate.problems) == (1, 1, ())
    dst, effect = wp.prepacked_bundle(src, cache_root=tmp_path / "cache")
    text = (dst / "model.mlir").read_text()
    assert text.count("linalg.transpose") == 1
    assert "ins(%2:tensor<2x3xi8>)" in text
    assert effect["weights_pre_transposed"] == 1
    assert effect["non_weight_transposes_not_hoisted"] == 1


def test_cache_is_reused_for_the_same_bundle_and_rewrite(tmp_path):
    src = _bundle(tmp_path / "src")
    cache = tmp_path / "cache"
    first, e1 = wp.prepacked_bundle(src, cache_root=cache)
    second, e2 = wp.prepacked_bundle(src, cache_root=cache)
    assert first == second
    assert e1["cached"] is False and e2["cached"] is True
    assert wp.cache_key(src) in first.name


def test_sidecars_are_carried_and_stale_prepared_ir_is_not(tmp_path):
    """A FIXED carry list dropped `session_contract.yaml` -- the state-carry map `c_runtime.generate`
    reads -- so a rewritten bundle graded as a different, single-shot model. `model.prepared.mlir` is
    the opposite case: it was derived from the PRE-rewrite IR, so carrying it forward would let a
    consumer lower the stale module."""
    src = _bundle(tmp_path / "src", extra_files={
        "session_contract.yaml": b"version: 1\n",
        "session_goldens.npz": b"not-really-an-npz",
        "model.prepared.mlir": b"// derived from the un-hoisted model.mlir\n",
    })
    dst, effect = wp.prepacked_bundle(src, cache_root=tmp_path / "cache")
    assert (dst / "session_contract.yaml").is_file()
    assert (dst / "session_goldens.npz").is_file()
    assert not (dst / "model.prepared.mlir").exists()
    assert effect["sidecars_not_carried"] == ["model.prepared.mlir"]


def test_the_compiler_input_actually_changes(tmp_path):
    """NON-INERT, necessary condition. `beam._emitted_digest` hashes only `model.o`'s objdump
    mnemonic stream, so a lever the digest cannot see reads as `inert=True`, is dropped from the
    survivor set and can never become a parent. Erasing the transposes changes the module the
    compiler is handed; MEASURED on the real K1 build of small_llama int8 it moves the emitted digest
    c020224d64a4efec -> c4d081f28ff03271 (66,421 -> 65,701 decoded instructions)."""
    src = _bundle(tmp_path / "src")
    dst, _ = wp.prepacked_bundle(src, cache_root=tmp_path / "cache")
    before, after = (src / "model.mlir").read_text(), (dst / "model.mlir").read_text()
    assert before.count("linalg.transpose") == 1
    assert after.count("linalg.transpose") == 0
    assert before != after


@pytest.mark.parametrize("backend", ["baremetal", "zephyr"])
@pytest.mark.parametrize("enabled", [False, True])
def test_build_entry_points_share_prepacked_ir_and_runtime_bundle(tmp_path, monkeypatch,
                                                                 backend, enabled):
    """Replay the actual build up to the ABI seam; only tool execution/lowering is substituted."""
    from merlin.runtime.backends import spike_model as sm, zephyr_model as zm
    src = _bundle(tmp_path / "src")
    monkeypatch.setattr(wp, "_default_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(sm._spike, "gcc_path", lambda: tmp_path / "riscv64-unknown-elf-gcc")
    monkeypatch.setattr(sm.toolchain, "clang", lambda: tmp_path / "clang")
    monkeypatch.setattr(zm, "available", lambda: True)
    monkeypatch.setattr(zm, "allocation_bytes", lambda _: (0, False))
    seen = {}

    def prepare(path, work, **kwargs):
        seen["prepare"] = Path(path).parent
        seen["vlen"] = kwargs.get("vlen")
        return path, kwargs["features"]

    def lower(path, work, **kwargs):
        seen["lower"] = Path(path).parent
        return SimpleNamespace(ll_path=tmp_path / "model.ll")

    class ReachedABI(Exception):
        pass

    def generate(bundle, destination, inputs, **kwargs):
        seen["abi"] = Path(bundle)
        seen["inputs"] = Path(inputs)
        seen["prepared_dir"] = Path(kwargs["prepared_dir"])
        raise ReachedABI

    monkeypatch.setattr(zm, "prepare_for_lowering", prepare)
    monkeypatch.setattr(sm, "lower_model_file", lower)
    monkeypatch.setattr(zm, "lower_model_file", lower)
    monkeypatch.setattr(sm, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(zm, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sm.c_runtime, "generate", generate)
    feature_set = frozenset({wp.FEATURE}) if enabled else frozenset()
    work = tmp_path / "work"
    with pytest.raises(ReachedABI):
        (sm.build if backend == "baremetal" else zm.build_app)(
            src, work, features=feature_set, vlen=512)
    assert seen["lower"] == seen["abi"]
    assert seen["inputs"] == seen["abi"] / "inputs.npz"
    assert seen["prepared_dir"] == work
    if enabled:
        assert seen["vlen"] == 512
        assert seen["prepare"] == seen["abi"] != src
        assert "linalg.transpose" not in (seen["abi"] / "model.mlir").read_text()
        assert json.loads((work / "bundle_preparation.json").read_text())["effect"][
            "transposes_removed"] == 1
    else:
        assert seen["abi"] == src
        assert not (work / "bundle_preparation.json").exists()
    assert "linalg.transpose" in (src / "model.mlir").read_text()


# ---------------------------------------------------------------------------------------------
# refusals -- the IR analysis proves sole-USE, which says nothing about how the bytes are STORED
# ---------------------------------------------------------------------------------------------
def _refusal(src: Path, tmp_path: Path) -> str:
    assert wp.plan(src).problems, "plan must see the same problem the build refuses on"
    with pytest.raises(wp.PrepackRefused) as exc:
        wp.prepacked_bundle(src, cache_root=tmp_path / "cache")
    return str(exc.value)


def test_refuses_a_stubbed_weight_with_no_bytes(tmp_path):
    """A quantized-subclass weight is stubbed in the manifest and absent from the blob (real:
    `resnet50_v1_5_int8_w8a8_consistent`, `lstmnetvit_int8_w8a8_consistent`). The blob rewrite walks
    the safetensors HEADER, so such a name is never reached -- while the manifest shape would be
    flipped anyway, describing a transpose nobody performed."""
    src = _bundle(tmp_path / "src", manifest={
        "0": {"kind": "param", "weight": "w0", "shape": [2, 3], "dtype": "int8", "stub": True},
        "1": {"kind": "param", "weight": "s0", "shape": [3], "dtype": "float32"},
        "2": {"kind": "param", "weight": "w1", "shape": [2, 3], "dtype": "int8"},
    }, header={
        "s0": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
        "w1": {"dtype": "I8", "shape": [2, 3], "data_offsets": [12, 18]},
    })
    (src / "weights.safetensors").unlink()
    _write_safetensors(src / "weights.safetensors", {
        "s0": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
        "w1": {"dtype": "I8", "shape": [2, 3], "data_offsets": [12, 18]},
    }, S0.tobytes() + W0.tobytes())
    assert "stub=true" in _refusal(src, tmp_path)


def test_refuses_a_weight_two_arguments_name(tmp_path):
    """`mining/section_build` and the capture manifests key weights by NAME, so two `@forward`
    arguments can name one tensor. Transposing it for the hoisted argument transposes it for the
    other reader too, silently."""
    src = _bundle(tmp_path / "src", manifest={
        "0": {"kind": "param", "weight": "w0", "shape": [2, 3], "dtype": "int8"},
        "1": {"kind": "param", "weight": "s0", "shape": [3], "dtype": "float32"},
        "2": {"kind": "param", "weight": "w0", "shape": [2, 3], "dtype": "int8"},
    })
    assert "also read by arg 2" in _refusal(src, tmp_path)


def test_refuses_an_aliased_byte_range(tmp_path):
    """Two names, one byte range (a tied head, an aliased view): rewriting one rewrites the other's
    data underneath it."""
    src = _bundle(tmp_path / "src", header={
        "w0": {"dtype": "I8", "shape": [2, 3], "data_offsets": [0, 6]},
        "w0_tied": {"dtype": "I8", "shape": [2, 3], "data_offsets": [0, 6]},
        "s0": {"dtype": "F32", "shape": [3], "data_offsets": [6, 18]},
        "w1": {"dtype": "I8", "shape": [2, 3], "data_offsets": [18, 24]},
    })
    assert "shares bytes" in _refusal(src, tmp_path)


def test_refuses_a_non_2d_weight(tmp_path):
    mlir = MLIR.replace("%0: tensor<2x3xi8>", "%0: tensor<1x2x3xi8>") \
               .replace("ins(%0:tensor<2x3xi8>)", "ins(%0:tensor<1x2x3xi8>)") \
               .replace("permutation = [1, 0]", "permutation = [0, 2, 1]") \
               .replace("tensor.empty() : tensor<3x2xi8>", "tensor.empty() : tensor<1x3x2xi8>") \
               .replace("outs(%3:tensor<3x2xi8>)", "outs(%3:tensor<1x3x2xi8>)") \
               .replace("-> tensor<3x2xi8>", "-> tensor<1x3x2xi8>") \
               .replace("func.return %4 : tensor<3x2xi8>", "func.return %4 : tensor<1x3x2xi8>")
    src = _bundle(tmp_path / "src", mlir=mlir, manifest={
        "0": {"kind": "param", "weight": "w0", "shape": [1, 2, 3], "dtype": "int8"},
        "1": {"kind": "param", "weight": "s0", "shape": [3], "dtype": "float32"},
        "2": {"kind": "param", "weight": "w1", "shape": [2, 3], "dtype": "int8"},
    }, header={
        "w0": {"dtype": "I8", "shape": [1, 2, 3], "data_offsets": [0, 6]},
        "s0": {"dtype": "F32", "shape": [3], "data_offsets": [6, 18]},
        "w1": {"dtype": "I8", "shape": [2, 3], "data_offsets": [18, 24]},
    })
    assert "only 2-D" in _refusal(src, tmp_path)


def test_a_bundle_with_nothing_to_hoist_is_refused_not_silently_built(tmp_path):
    """An already-prepacked bundle has no hoistable transpose. Building it stock while reporting the
    lever as applied is the inert-lever failure, but silent: refuse instead."""
    src = _bundle(tmp_path / "src")
    dst, _ = wp.prepacked_bundle(src, cache_root=tmp_path / "cache")
    assert wp.plan(dst).hoistable == 0
    with pytest.raises(wp.PrepackRefused):
        wp.prepacked_bundle(dst, cache_root=tmp_path / "cache2")
