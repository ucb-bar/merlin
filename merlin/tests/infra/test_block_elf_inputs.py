"""External block-fixture admission without compilers or a device provider."""

from merlin.common.digest import sha256_bytes

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from merlin.common.paths import repo_root

SOURCE = repo_root() / "examples/atlas/comparisons"
with patch.object(sys, "path", [str(SOURCE), *sys.path]):
    spec = importlib.util.spec_from_file_location("block_elf_fixture_controls", SOURCE / "build_block_elf.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)


def build(tmp_path):
    return builder.Build(SimpleNamespace(recipe=tmp_path / "recipe.json", out=tmp_path / "out"))


def record(path, relative=False, **kwargs):
    return {
        "path": path.name if relative else str(path),
        "sha256": sha256_bytes(path.read_bytes()),
        **kwargs,
    }


def test_relative_fixture_path_and_input_bookend(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"\x00\xff\x80\x01")
    invocation = build(tmp_path)
    original = record(path, relative=True)
    assert invocation.data(original) == b"\x00\xff\x80\x01"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        invocation.data(original)


def test_array_input_preserves_bf16_signed_zero_and_nan_words(tmp_path):
    path = tmp_path / "words.npz"
    np.savez(path, selected=np.array([[0x8000, 0x7FC1, 0x3F80]], dtype=np.uint16))
    assert build(tmp_path).data(record(path, relative=True, array="selected")) == b"\x00\x80\xc1\x7f\x80\x3f"


def test_unsupported_array_dtype_refused(tmp_path):
    path = tmp_path / "wrong.npy"
    np.save(path, np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="uint8"):
        build(tmp_path).data(record(path, array=None))


def test_parent_paths_refused_before_input_read(tmp_path):
    with patch.object(Path, "open", side_effect=AssertionError("read before admission")):
        with pytest.raises(ValueError):
            build(tmp_path).pin("../outside.bin", "0" * 64)


def test_local_output_cannot_replace_original_pin(tmp_path):
    invocation = build(tmp_path)
    path = invocation.put("nested/source.c", b"original")
    invocation.pin(path)
    path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed"):
        invocation.pin(path)


def test_native_manifest_must_bind_the_actual_npz():
    spec = importlib.util.spec_from_file_location("kernel_manifest_controls", SOURCE / "block_kernel_inputs.py")
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    source = {"sha256": "a" * 64}
    inputs.admit_manifest({"source_dtype": "bfloat16", "npz_sha256": "a" * 64}, source, "bfloat16")
    for changed in (
        {"source_dtype": "float32", "npz_sha256": "a" * 64},
        {"source_dtype": "bfloat16", "npz_sha256": "b" * 64},
    ):
        with pytest.raises(ValueError, match="binding"):
            inputs.admit_manifest(changed, source, "bfloat16")


def test_every_fixed_tiny_array_requires_its_complete_harness_extent():
    sizes = {
        "step0-entry": 524288,
        "step1-entry": 4096,
        "step0-expected": 524288,
        "step1-expected": 4096,
        "input-norm": 4096,
        "post-norm": 4096,
        "cosine": 20480,
        "sine": 20480,
        "step0-keys": 65536,
        "step0-values": 65536,
        "step1-keys": 66048,
        "step1-values": 66048,
    }
    data = {name: bytes(size) for name, size in sizes.items()}
    builder.admit_block_data("tiny", data)
    for name in data:
        for delta in (-1, 1):
            changed = dict(data, **{name: bytes(sizes[name] + delta)})
            with pytest.raises(ValueError, match="extent"):
                builder.admit_block_data("tiny", changed)


def test_smol_rotary_tables_cannot_be_empty_partial_or_unequal():
    sizes = {
        "input": 339840,
        "pad": 177,
        "attention_boundaries": 177,
        "input_norm_weight": 1920,
        "post_norm_weight": 1920,
        "expected_output": 339840,
        "expected_key_cache": 113280,
        "expected_value_cache": 113280,
        "cosine": 128,
        "sine": 128,
    }
    data = {name: bytes(size) for name, size in sizes.items()}
    builder.admit_block_data("smol", data)
    for cosine, sine in ((b"", b""), (b"x", b"x"), (bytes(128), bytes(256))):
        with pytest.raises(ValueError, match="rotary"):
            builder.admit_block_data("smol", dict(data, cosine=cosine, sine=sine))


def test_provider_layout_snapshot_includes_nested_headers_and_pins_bytes(tmp_path):
    runtime = tmp_path / "provider"
    runtime.mkdir()
    layout = runtime / "atlas_host_layout.h"
    nested = runtime / "layout_constants.h"
    layout.write_text('#include "layout_constants.h"\n')
    nested.write_text('#define TILE_BYTES 2048\n')
    invocation = build(tmp_path)
    invocation.args.provider_runtime = runtime
    invocation.provider_sources(["atlas_host_layout.h"], "source/")
    assert (tmp_path / "out/source/atlas_host_layout.h").read_bytes() == layout.read_bytes()
    assert (tmp_path / "out/source/layout_constants.h").read_bytes() == nested.read_bytes()
    assert str(nested) in invocation.pins
    nested.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        invocation.provider_sources(["atlas_host_layout.h"], "source/")


def test_provider_header_parent_escape_refused_before_read(tmp_path):
    runtime = tmp_path / "provider"
    runtime.mkdir()
    (runtime / "atlas_host_layout.h").write_text('#include "../escape.h"\n')
    invocation = build(tmp_path)
    invocation.args.provider_runtime = runtime
    with pytest.raises(ValueError, match="beneath"):
        invocation.provider_sources(["atlas_host_layout.h"], "source/")


def test_provider_adapter_keeps_merlin_header_ownership(tmp_path):
    runtime = tmp_path / "provider"
    runtime.mkdir()
    (runtime / "adapter.h").write_text(
        '#include "smol_vlm_block_prototype.h"\n#include "atlas_host_layout.h"\n')
    layout = runtime / "atlas_host_layout.h"
    layout.write_text('#define TILE_BYTES 2048\n')
    invocation = build(tmp_path)
    invocation.args.provider_runtime = runtime
    invocation.provider_sources(["adapter.h"], "source/", local_headers=["smol_vlm_block_prototype.h"])
    assert (tmp_path / "out/source/atlas_host_layout.h").read_bytes() == layout.read_bytes()
    assert not (tmp_path / "out/source/smol_vlm_block_prototype.h").exists()
