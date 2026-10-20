"""The whole-model build's per-group compiled-object cache.

Compiling a package-answered group's LLVM artifact to a riscv object is the expensive step of a
whole-model build: a full rebuild recompiles every answered group even when an edit changed one
group's kernel. This module lets :mod:`merlin.perf.whole_model_build` reuse a PRIOR object for a
group whose artifact text is byte-identical to one already compiled, under the same target and
toolchain, instead of recompiling it::

    root = object_cache_root()
    fingerprint = object_cache_fingerprint(target)  # everything besides the artifact text and the
    # target name that decides the compiled bytes
    key = object_cache_key(artifact_text, target=target, fingerprint=fingerprint)
    cached = root / key[:2] / f"{key}.o"  # present -> reuse; absent -> compile, then store

A reused object must be byte-identical to what a fresh build would have produced for the same
inputs, so the fingerprint has to cover everything that decides those bytes -- see
:func:`object_cache_fingerprint`. A lookup that cannot be verified (an unreadable toolchain binary, a
target whose recipe declares no stack-frame policy) disables the cache for that build rather than
guessing: fail closed to a fresh compile, never to a cache entry nobody can vouch for.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "OBJECT_CACHE_DEFAULT_MAX_BYTES",
    "OBJECT_CACHE_DISABLE_ENV",
    "OBJECT_CACHE_MAX_BYTES_ENV",
    "OBJECT_CACHE_NAMESPACE",
    "await_inflight",
    "load",
    "object_cache_fingerprint",
    "object_cache_key",
    "object_cache_root",
    "prewarm_async",
    "prune",
    "store",
]

#: The compiled-object cache namespace under ``out/artifacts/cache/`` and its disable switch.
OBJECT_CACHE_NAMESPACE = "whole-model-group-objects"
OBJECT_CACHE_DISABLE_ENV = "MERLIN_WHOLE_MODEL_NO_OBJECT_CACHE"

#: Objects here are tens of KB each, but a long optimization loop tries many distinct candidate
#: kernels per group -- each a new key -- so the cache is bounded least-recently-used, the same
#: pattern the package-reply cache uses (:func:`merlin.perf.whole_model_build.prune_replies`), for the
#: same reason: an unbounded content-addressed cache on a disk-tight host is a slow leak, not a
#: one-time cost.
OBJECT_CACHE_MAX_BYTES_ENV = "MERLIN_WHOLE_MODEL_OBJECT_CACHE_MAX_BYTES"
OBJECT_CACHE_DEFAULT_MAX_BYTES = 2 << 30


def _object_cache_max_bytes() -> int:
    raw = (os.environ.get(OBJECT_CACHE_MAX_BYTES_ENV, "") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else OBJECT_CACHE_DEFAULT_MAX_BYTES


def prune(directory: str | Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Evict cached objects, least recently used (mtime; a hit touches it) first, until under budget.

    Best-effort, mirroring ``whole_model_build.prune_replies``: a cache that cannot be pruned still
    just costs the next caller a recompile, never a wrong answer.
    """
    limit = _object_cache_max_bytes() if max_bytes is None else int(max_bytes)
    entries: list[tuple[float, int, Path]] = []
    try:
        for path in Path(directory).glob("*/*.o"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
    except OSError:
        return {"evicted": 0, "bytes_evicted": 0}
    total = sum(size for _mtime, size, _path in entries)
    evicted = freed = 0
    for _mtime, size, path in sorted(entries):
        if total <= limit:
            break
        try:
            path.unlink()
        except OSError:
            continue
        _digest_sidecar(path).unlink(missing_ok=True)
        total -= size
        freed += size
        evicted += 1
    return {"evicted": evicted, "bytes_evicted": freed, "bytes_kept": total, "limit": limit}


def object_cache_root() -> Path | None:
    """The object cache directory, or ``None`` when disabled or unavailable (fails closed to no
    cache, never to a cache whose entries cannot be trusted)."""
    if (os.environ.get(OBJECT_CACHE_DISABLE_ENV) or "").strip():
        return None
    try:
        from merlin.common.artifacts import cache_dir

        return Path(cache_dir(OBJECT_CACHE_NAMESPACE))
    except Exception:  # noqa: BLE001 -- no cache directory: every group compiles fresh
        return None


def object_cache_fingerprint(target: str) -> str:
    """Everything besides the artifact text and the target name that decides a compiled object's bytes.

    Hashing a hand-picked file or two is exactly the "too-narrow pattern silently drops valid input"
    failure this repo's own regex rule warns about, applied to a cache instead of a parser: a file the
    picker forgot would keep serving stale bytes forever with nothing to notice. So this hashes every
    ``.py`` file the lowering path can reach (:mod:`merlin.llvmlower`, whole package) plus the two
    modules outside it that also decide the outcome (:mod:`merlin.targetgen.contract.compile`, which
    calls the stack-frame preflight, and :mod:`merlin.targetgen.contract.stack_usage`, which measures
    it) -- sorted by path so the digest does not depend on filesystem enumeration order.

    Two RESOLVED FACTS also decide the bytes and are read directly rather than hoped to be implied by
    a source hash, because they can come from data (a target's registry/manifest), not code: the
    target's ``-march=`` (a different ISA string recompiles identically-shaped IR to different
    instructions) and its stack-frame policy's ``max_static_bytes`` -- a budget that merely tightens,
    with no source file touched, can turn on the arena-rewrite repair path
    (:func:`merlin.targetgen.contract.compile._repair_oversized_frame`) and change the emitted bytes
    for the identical input.

    The toolchain binaries actually invoked (``clang``, ``mlir-translate``, the m2m venv's python) are
    hashed by CONTENT, not path/size/mtime, so an in-place binary replacement of the same size cannot
    be missed. NOT covered: an upgrade inside the m2m venv's installed packages (e.g. a torch-mlir
    version bump) that replaces no file this repo owns and no binary path here -- the reference-context
    cache (:func:`merlin.perf.whole_model_group_timing._reference_code_digest`) accepts the identical
    scope limit for the identical reason: chasing every transitive dependency of an out-of-process
    interpreter is unbounded, so this hashes the interpreter binary itself and the facts and code this
    repo controls, and documents the gap rather than hiding it.
    """
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.contract import compile as _compile_mod
    from merlin.targetgen.contract import stack_usage as _stack_usage_mod

    digest = hashlib.sha256()
    llvmlower_root = Path(str(toolchain.__file__)).resolve().parent
    for path in sorted(llvmlower_root.rglob("*.py")):
        digest.update(str(path.relative_to(llvmlower_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for module in (_compile_mod, _stack_usage_mod):
        digest.update(Path(str(module.__file__)).read_bytes())
        digest.update(b"\0")
    recipe = backends.harness_build_recipe(target)
    policy = recipe.require_kernel_stack_frame()
    digest.update(str(recipe.march()).encode())
    digest.update(b"\0")
    digest.update(f"{policy.entry_symbol}|{policy.max_static_bytes}".encode())
    digest.update(b"\0")
    for tool in (toolchain.clang(), toolchain.mlir_translate(), toolchain.m2m_python()):
        digest.update(Path(tool).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def object_cache_key(artifact_text: str, *, target: str, fingerprint: str) -> str:
    digest = hashlib.sha256()
    for part in (artifact_text, target, fingerprint):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _digest_sidecar(cached_path: Path) -> Path:
    return cached_path.with_suffix(cached_path.suffix + ".sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(cached_path: Path) -> Path | None:
    """``cached_path`` if present AND its bytes match the digest :func:`store` recorded when it was
    written, else ``None`` -- dropping a corrupted or incomplete entry so it is never trusted again.

    A reused object must be byte-identical to what a fresh build would produce for the same inputs;
    trusting a cache HIT by file name alone would let a truncated or corrupted entry silently become a
    build's kernel object. This is the same posture :func:`merlin.common.content_store.object_for`
    takes, for the same reason.
    """
    sidecar = _digest_sidecar(cached_path)
    if not cached_path.is_file() or not sidecar.is_file():
        return None
    try:
        want = sidecar.read_text(encoding="utf-8").strip()
        got = _sha256_file(cached_path)
    except OSError:
        return None
    if got != want:
        cached_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        return None
    try:
        os.utime(cached_path)  # recently used: `prune` evicts least-recently-used first
    except OSError:
        pass
    return cached_path


def store(cached_path: Path, compiled: Path) -> None:
    """Record ``compiled``'s bytes under ``cached_path``, with the digest :func:`load` verifies
    against. Best-effort: a cache that cannot be written only costs the next caller a recompile, and a
    writer that loses a race with another writer of the SAME key loses nothing, because a deterministic
    compile makes their bytes identical either way."""
    import shutil

    try:
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        staging = cached_path.with_name(f".{cached_path.name}.{os.getpid()}.tmp")
        shutil.copyfile(compiled, staging)
        os.replace(staging, cached_path)
        digest = _sha256_file(cached_path)
        sidecar_staging = cached_path.with_name(f".{cached_path.name}.{os.getpid()}.sha256.tmp")
        sidecar_staging.write_text(digest, encoding="utf-8")
        os.replace(sidecar_staging, _digest_sidecar(cached_path))
    except OSError:
        return


_FINGERPRINTS: dict[str, str] = {}


def prewarm(artifact_text: str, *, target: str, work: Path) -> str:
    """Compile ``artifact_text`` into the persistent cache now, so the build's object stage reads it.

    ``"hit"`` when a verified entry already exists, ``"miss"`` when it was compiled and stored here,
    ``"off"`` when the cache is disabled or the toolchain cannot be fingerprinted. The entry is the same
    one :func:`merlin.perf.whole_model_build._kernel_objects` would write for the same artifact text, so
    a prewarmed object is byte-for-byte the object that stage would have compiled itself.
    """
    from merlin.targetgen.contract.compile import llvm_mlir_to_object
    from merlin.verify.lean_emission_context import requires_fresh_object

    if requires_fresh_object(artifact_text):
        # The actual consumer must observe this pending traced object; a
        # disposable cache warmup cannot transfer a live completion to it.
        return "off"

    root = object_cache_root()
    if root is None:
        return "off"
    if target not in _FINGERPRINTS:
        try:
            _FINGERPRINTS[target] = object_cache_fingerprint(target)
        except Exception:  # noqa: BLE001 -- no fingerprint: no cache, as in the object stage
            return "off"
    key = object_cache_key(artifact_text, target=target, fingerprint=_FINGERPRINTS[target])
    cached_path = Path(root) / key[:2] / f"{key}.o"
    if load(cached_path) is not None:
        return "hit"
    store(cached_path, llvm_mlir_to_object(artifact_text, Path(work), target=target))
    return "miss"


#: Prewarm compiles still running, by cache key. A compile started the moment a group is answered
#: outlives the statement; the object stage waits for it here instead of compiling the same text again.
_INFLIGHT: dict[str, Any] = {}
_INFLIGHT_LOCK = threading.Lock()
_POOL: list[Any] = []


def _pool(jobs: int):
    from concurrent.futures import ThreadPoolExecutor

    with _INFLIGHT_LOCK:
        if not _POOL:
            _POOL.append(ThreadPoolExecutor(max_workers=max(1, jobs), thread_name_prefix="object-prewarm"))
        return _POOL[0]


def prewarm_async(artifact_text: str, *, target: str, jobs: int) -> Any:
    """Start compiling ``artifact_text`` into the cache in the background; the future, or ``None`` when
    it is already cached (or the cache is off). One compile per key at a time, whoever asks."""
    from contextvars import copy_context

    from merlin.verify.lean_emission_context import requires_fresh_object

    if requires_fresh_object(artifact_text):
        return None
    root = object_cache_root()
    if root is None:
        return None
    if target not in _FINGERPRINTS:
        try:
            _FINGERPRINTS[target] = object_cache_fingerprint(target)
        except Exception:  # noqa: BLE001 -- no fingerprint: no cache, as in the object stage
            return None
    key = object_cache_key(artifact_text, target=target, fingerprint=_FINGERPRINTS[target])
    if load(Path(root) / key[:2] / f"{key}.o") is not None:
        return None

    def compile_once() -> str:
        import shutil
        import tempfile

        work = Path(tempfile.mkdtemp(prefix="merlin-object-prewarm-"))
        try:
            return prewarm(artifact_text, target=target, work=work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            with _INFLIGHT_LOCK:
                _INFLIGHT.pop(key, None)

    with _INFLIGHT_LOCK:
        running = _INFLIGHT.get(key)
        if running is not None:
            return running
    future = _pool(jobs).submit(copy_context().run, compile_once)
    with _INFLIGHT_LOCK:
        _INFLIGHT.setdefault(key, future)
    return future


def await_inflight(key: str) -> None:
    """Wait for a prewarm compile of ``key`` still running in this process; its failure is the object
    stage's to rediscover (it compiles the text itself), never an error here."""
    with _INFLIGHT_LOCK:
        running = _INFLIGHT.get(key)
    if running is not None:
        try:
            running.result()
        except Exception:  # noqa: BLE001 -- the object stage recompiles and reports the real failure
            pass
