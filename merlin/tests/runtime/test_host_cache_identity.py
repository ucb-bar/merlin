"""Persistent host kernels must be tied to their actual lowering inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from merlin.runtime.host_cache_identity import host_cache_identity


def _synthetic_identity(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    package = tmp_path / "package"
    source.mkdir(parents=True)
    package.mkdir()
    (source / "lower.py").write_text("mean = old\n")
    (package / "dialect.py").write_text("old\n")
    compiler = tmp_path / "clang"
    compiler.write_bytes(b"clang old")
    monkeypatch.setattr("merlin.runtime.host_cache_identity.platform.machine", lambda: "test-machine")

    def identity(passes: str = ""):
        return host_cache_identity(
            source_root=source,
            tool_paths={"clang": compiler},
            package_paths={"xdsl": package},
            environ={"MERLIN_PASSES": passes},
        )

    return identity, source, package, compiler


def test_host_cache_identity_tracks_mean_pass_dirty_source_and_compiler(tmp_path, monkeypatch):
    identity, source, package, compiler = _synthetic_identity(tmp_path, monkeypatch)
    original = identity()
    assert identity()["sha256"] == original["sha256"]
    assert identity("torchao-cpu-avx2-mean")["sha256"] != original["sha256"]
    (source / "lower.py").write_text("mean = fixed\n")
    changed_source = identity()
    assert changed_source["sha256"] != original["sha256"]
    (package / "dialect.py").write_text("fixed\n")
    changed_package = identity()
    assert changed_package["sha256"] != changed_source["sha256"]
    compiler.write_bytes(b"clang fixed")
    assert identity()["sha256"] != changed_package["sha256"]


def test_cache_only_rejects_legacy_kernel_before_execution(tmp_path, monkeypatch):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import kernel_backend, portable_math
    from merlin.runtime import dispatch_runtime
    from merlin.runtime.host_cache_manifest import manifest_path, write_manifest
    from merlin.xdsl_dialects._common import text as to_text
    from merlin.xdsl_dialects.lowering.outline import OutlineResult

    module = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32> {
        %y = func.call @forward$kernel_0(%x) : (tensor<1xf32>) -> tensor<1xf32>
        func.return %y : tensor<1xf32>
      }
      func.func private @forward$kernel_0(%x: tensor<1xf32>) -> tensor<1xf32> {
        func.return %x : tensor<1xf32>
      }
    }""")
    outlined = OutlineResult(module, [])
    ktext = to_text(kernel_backend.extract_kernel(module, "forward$kernel_0"))
    old_salt = "\n// transcendental_math=" + portable_math.cache_token()
    legacy_key = hashlib.sha1((ktext + old_salt).encode()).hexdigest()[:16]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{legacy_key}.so").write_bytes(b"old cached native object")
    identity, *_ = _synthetic_identity(tmp_path / "identity", monkeypatch)
    monkeypatch.setattr("merlin.runtime.host_cache_identity.host_cache_identity", identity)
    monkeypatch.setattr(kernel_backend, "compile_host", lambda *_a, **_k: pytest.fail("compiled on a cache-only miss"))
    with pytest.raises(dispatch_runtime.HostCacheMiss, match="host cache miss_count=1 before execution") as miss:
        dispatch_runtime.execute(
            outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True
        )
    assert miss.value.miss_count == 1
    assert miss.value.identity["sha256"] == identity()["sha256"]

    # A correctly salted entry is loaded by exact path and recorded with its bytes.
    from merlin.llvmlower.abi import HostModel

    new_salt = old_salt + "\n// host_compiler_identity=" + identity()["sha256"]
    key = hashlib.sha1((ktext + new_salt).encode()).hexdigest()[:16]
    (cache / f"{key}.so").write_bytes(b"current cached native object")
    build = tmp_path / "build"
    build.mkdir()
    recipe = {"schema": "merlin.lowering_recipe.v1", "status": "returned", "commands": [{"argv": ["test-lowerer"]}]}
    (build / "lowering_recipe.json").write_text(json.dumps(recipe))
    write_manifest(cache / f"{key}.so", key=key, kernel_text=ktext, identity=identity(), workdir=build)

    class StubHostModel:
        scalar_result_dtype = None

        def __call__(self, _args):
            return None

    monkeypatch.setattr(HostModel, "load", lambda *_a, **_k: StubHostModel())
    counters = {}
    dispatch_runtime.execute(
        outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True, counters=counters
    )
    receipt = counters["host_cache"]
    assert receipt["miss_count"] == 0
    assert receipt["verified_after_execution"] is True
    assert receipt["identity"]["sha256"] == identity()["sha256"]
    assert len(receipt["loaded"]) == 1
    loaded = receipt["loaded"][0]
    assert loaded["key"] == key
    assert loaded["path"] == str((cache / f"{key}.so").resolve())
    assert loaded["bytes"] == len(b"current cached native object")
    assert loaded["sha256"] == hashlib.sha256(b"current cached native object").hexdigest()
    assert loaded["manifest_path"] == str(manifest_path(cache / f"{key}.so"))
    assert loaded["manifest"]["compiler_identity_sha256"] == identity()["sha256"]

    sealed_manifest = manifest_path(cache / f"{key}.so")
    original_manifest = sealed_manifest.read_text()
    sealed_manifest.unlink()
    with pytest.raises(dispatch_runtime.HostCacheIntegrityError, match="host cache integrity refusal"):
        dispatch_runtime.execute(
            outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True
        )
    sealed_manifest.write_text(original_manifest)
    sealed_manifest.write_text("[]")
    with pytest.raises(dispatch_runtime.HostCacheIntegrityError, match="host cache integrity refusal"):
        dispatch_runtime.execute(
            outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True
        )
    sealed_manifest.write_text(original_manifest)
    tampered = json.loads(original_manifest)
    tampered["kernel_text_sha256"] = "0" * 64
    sealed_manifest.write_text(json.dumps(tampered))
    with pytest.raises(dispatch_runtime.HostCacheIntegrityError, match="host cache integrity refusal"):
        dispatch_runtime.execute(
            outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True
        )
    sealed_manifest.write_text(original_manifest)

    class MutatingHostModel(StubHostModel):
        def __call__(self, _args):
            (cache / f"{key}.so").write_bytes(b"changed during execution")

    monkeypatch.setattr(HostModel, "load", lambda *_a, **_k: MutatingHostModel())
    with pytest.raises(dispatch_runtime.DispatchRuntimeError, match="host cache object changed during execution"):
        dispatch_runtime.execute(
            outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True
        )


def test_read_write_population_creates_manifest_then_strict_replay(tmp_path, monkeypatch):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.runtime import dispatch_runtime
    from merlin.xdsl_dialects.lowering.outline import OutlineResult

    module = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32> {
        %y = func.call @forward$kernel_0(%x) : (tensor<1xf32>) -> tensor<1xf32>
        func.return %y : tensor<1xf32>
      }
      func.func private @forward$kernel_0(%x: tensor<1xf32>) -> tensor<1xf32> {
        func.return %x : tensor<1xf32>
      }
    }""")
    outlined = OutlineResult(module, [])
    identity, *_ = _synthetic_identity(tmp_path / "identity", monkeypatch)
    monkeypatch.setattr("merlin.runtime.host_cache_identity.host_cache_identity", identity)
    monkeypatch.setenv("MERLIN_COMPILE_WORKERS", "1")

    compiled = []

    def fake_compile(args):
        _, workdir, so = args
        work = Path(workdir)
        work.mkdir(parents=True)
        (work / "lowering_recipe.json").write_text(
            json.dumps({"schema": "merlin.lowering_recipe.v1", "status": "returned", "commands": [{"argv": ["stub"]}]})
        )
        Path(so).write_bytes(b"synthetic compiled object")
        compiled.append(so)

    class StubHostModel:
        scalar_result_dtype = None

        def __call__(self, _args):
            return None

    monkeypatch.setattr(dispatch_runtime, "_compile_kernel_so", fake_compile)
    monkeypatch.setattr(HostModel, "load", lambda *_a, **_k: StubHostModel())
    cache = tmp_path / "cache"
    first, replay = {}, {}
    dispatch_runtime.execute(outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, counters=first)
    assert len(compiled) == 1
    assert (
        first["host_cache"]["loaded"][0]["manifest"]["object_sha256"]
        == hashlib.sha256(b"synthetic compiled object").hexdigest()
    )
    dispatch_runtime.execute(
        outlined, [np.ones(1, np.float32)], tmp_path / "replay", cache_dir=cache, cache_only=True, counters=replay
    )
    assert len(compiled) == 1
    assert replay["host_cache"]["verified_after_execution"] is True
    assert replay["host_cache"]["loaded"] == first["host_cache"]["loaded"]
