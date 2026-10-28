"""Prepared weights retain original invocation metadata and source custody."""

import json
import os
import subprocess
import sys

import pytest
from test_weight_prepack import _bundle

from merlin.common.digest import sha256_file
from merlin.llvmlower import weight_prepack as wp


def endorse_cache_file(destination, name):
    """Simulate a cache writer replacing both an output and its own hash."""
    path = destination / wp._IDENTITY_FILE
    identity = json.loads(path.read_text())
    changed = destination / name
    identity["outputs"][name] = dict(
        path=str(changed.resolve()), sha256=sha256_file(changed), bytes=changed.stat().st_size
    )
    path.write_text(json.dumps(identity))


def test_repeated_validation_reuses_only_successful_exact_content(tmp_path, monkeypatch):
    from merlin.capture import rewrite

    source = _bundle(tmp_path / "source")
    original = rewrite.verify_hoisted_bundle
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(rewrite, "verify_hoisted_bundle", counted)
    destination, _ = wp.prepacked_bundle(source, cache_root=tmp_path / "cache")
    wp.prepacked_bundle(source, cache_root=tmp_path / "cache")
    assert len(calls) == 1

    # Neither a new cache-owned hash nor a failed check becomes an authority.
    changed = destination / "weights.safetensors"
    contents = changed.read_bytes()
    changed.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
    endorse_cache_file(destination, changed.name)
    for _ in range(2):
        with pytest.raises(wp.PrepackRefused):
            wp.prepacked_bundle(source, cache_root=tmp_path / "cache")
    assert len(calls) == 3


def test_changes_during_semantic_validation_are_not_remembered(tmp_path, monkeypatch):
    from merlin.capture import rewrite

    source = _bundle(tmp_path / "source")
    original = rewrite.verify_hoisted_bundle

    def changed_after_check(src, dst, *args, **kwargs):
        result = original(src, dst, *args, **kwargs)
        path = dst / "weights.safetensors"
        contents = path.read_bytes()
        path.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
        return result

    monkeypatch.setattr(rewrite, "verify_hoisted_bundle", changed_after_check)
    with pytest.raises(wp.PrepackRefused, match="changed during semantic validation"):
        wp.prepacked_bundle(source, cache_root=tmp_path / "cache")


@pytest.mark.parametrize(
    "name", ["session_contract.yaml", "inputs.npz", "extra.npz", "input_order.json", "goldens.npz"]
)
def test_cache_owned_hash_cannot_replace_original_sidecars(tmp_path, name):
    sidecar = b"version: 1\nstates: []\nstreams: []\n" if name == "session_contract.yaml" else b"original sidecar"
    source = _bundle(tmp_path / "source", extra_files={name: sidecar})
    destination, _ = wp.prepacked_bundle(source, cache_root=tmp_path / "cache")
    # Unlink first: unchanged sidecars can be hardlinks to the original bundle.
    changed = destination / name
    changed.unlink()
    changed.write_bytes(b"changed invocation metadata")
    endorse_cache_file(destination, name)
    assert (source / name).read_bytes() == sidecar
    with pytest.raises(wp.PrepackRefused, match="carried sidecar"):
        wp.prepacked_bundle(source, cache_root=tmp_path / "cache")


@pytest.mark.parametrize("mutation", ["extra", "missing", "stale"])
def test_carried_file_roster_is_reconstructed_from_source(tmp_path, mutation):
    source = _bundle(tmp_path / "source")
    destination, _ = wp.prepacked_bundle(source, cache_root=tmp_path / "cache")
    identity_path = destination / wp._IDENTITY_FILE
    identity = json.loads(identity_path.read_text())
    if mutation == "missing":
        (destination / "inputs.npz").unlink()
        del identity["outputs"]["inputs.npz"]
        identity_path.write_text(json.dumps(identity))
    else:
        name = "extra.npz" if mutation == "extra" else "model.prepared.mlir"
        (destination / name).write_bytes(b"unrequested cached sidecar")
        endorse_cache_file(destination, name)
    with pytest.raises(wp.PrepackRefused, match="sidecar roster"):
        wp.prepacked_bundle(source, cache_root=tmp_path / "cache")


@pytest.mark.parametrize("mutation", ["layout", "bytes", "caveat", "history"])
def test_cache_owned_hash_cannot_endorse_rewritten_history(tmp_path, mutation):
    from merlin.capture.rewrite import REWRITES_FILE, RewriteRecord, record_rewrite

    source = _bundle(tmp_path / "source")
    record_rewrite(source, RewriteRecord("original", "capture", "original policy", {"permission": False}))
    destination, _ = wp.prepacked_bundle(source, cache_root=tmp_path / "cache")
    path = destination / REWRITES_FILE
    record = json.loads(path.read_text())
    last = record["rewrites"][-1]
    if mutation == "layout":
        last["effect"]["layouts"][0]["permutation"] = [0, 1]
    elif mutation == "bytes":
        last["effect"]["bytes_moved_per_inference_before"] += 1
    elif mutation == "caveat":
        last["caveats"].append("forged capability")
    else:
        record["rewrites"][0]["effect"]["permission"] = True
    path.write_text(json.dumps(record))
    endorse_cache_file(destination, REWRITES_FILE)
    with pytest.raises(wp.PrepackRefused, match="history or reported effects"):
        wp.prepacked_bundle(source, cache_root=tmp_path / "cache")


def test_same_size_same_timestamp_weight_change_invalidates_cache_key(tmp_path):
    source = _bundle(tmp_path / "source")
    before = wp.cache_key(source)
    path = source / "weights.safetensors"
    timestamp = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    assert path.stat().st_size == len(raw)
    assert path.stat().st_mtime_ns == timestamp.st_mtime_ns
    assert wp.cache_key(source) != before


def test_changed_implementation_invalidates_cache_key(tmp_path, monkeypatch):
    source = _bundle(tmp_path / "source")
    library = tmp_path / "implementation.py"
    library.write_bytes(b"first implementation")
    monkeypatch.setattr(wp, "_implementation_paths", lambda: (library,))
    before = wp.cache_key(source)
    library.write_bytes(b"other implementation")
    assert wp.cache_key(source) != before


@pytest.mark.parametrize("stage", ["completion", "reopen"])
def test_session_added_after_preparation_refuses_final_recipe(tmp_path, monkeypatch, stage):
    from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe

    source = _bundle(tmp_path / "source")
    work = tmp_path / "work"
    work.mkdir()
    recipe = CompilationRecipe(work, producer=source / "model.mlir")
    monkeypatch.setattr(wp, "_default_cache_root", lambda: tmp_path / "cache")
    wp.prepare_build_bundle(source, work, frozenset({wp.ensure_registered()}), compilation=recipe)
    product = work / "model.elf"
    recipe.run(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[3]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())",
            str(source / "model.mlir"),
            "-o",
            str(product),
        ],
        runner=lambda argv: subprocess.run(argv, capture_output=True),
        inputs=[source / "model.mlir"],
        output=product,
    )
    if stage == "reopen":
        recipe.completed(product)
    frozen = sha256_file(recipe.path)
    (source / "session_contract.yaml").write_text("version: 1\nstates: []\nstreams: []\n")
    # An absent file is an ownership premise, even though it has no bound file hash.
    with pytest.raises(wp.PrepackRefused, match="changed"):
        if stage == "completion":
            recipe.completed(product)
        else:
            verify_completed_recipe(recipe.path, executable=product, expected_recipe_sha256=frozen)
