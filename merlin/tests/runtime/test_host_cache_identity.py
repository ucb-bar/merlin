"""Persistent host kernels must be tied to their actual lowering inputs."""

from __future__ import annotations

import hashlib
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
    assert receipt["loaded"] == [
        {
            "key": key,
            "path": str((cache / f"{key}.so").resolve()),
            "bytes": len(b"current cached native object"),
            "sha256": hashlib.sha256(b"current cached native object").hexdigest(),
        }
    ]

    class MutatingHostModel(StubHostModel):
        def __call__(self, _args):
            (cache / f"{key}.so").write_bytes(b"changed during execution")

    monkeypatch.setattr(HostModel, "load", lambda *_a, **_k: MutatingHostModel())
    with pytest.raises(dispatch_runtime.DispatchRuntimeError, match="host cache object changed during execution"):
        dispatch_runtime.execute(
            outlined, [np.ones(1, np.float32)], tmp_path / "work", cache_dir=cache, cache_only=True
        )
