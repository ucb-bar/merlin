"""The vendor parameter header a whole-model program is built against: asserted by content for the
machine, copied into the build's harness, and recorded by what the program actually read.

Split out of :mod:`merlin.analysis.whole_model.build.stages` (which re-exports every name here) to keep that
module under the repository's module-size limit.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ---- whole_model_headers -----------------------------------------------------------------------


def _build_error(message: str) -> Exception:
    """The builder's own refusal type (imported at call time: that module imports this one)."""
    from merlin.analysis.whole_model.build.stages import WholeModelBuildError

    return WholeModelBuildError(message)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def machine_header(machine: str, header: str | Path, header_sha256: str | None = None) -> dict[str, Any]:
    """The parameter header a program for ``machine`` is built against, ASSERTED by content, or a refusal.

    A program is compiled for one machine, and which machine is decided by the vendor parameter header
    it includes: two designs of one target can differ in a readout the header either declares or does
    not, and an ELF built against the wrong one runs and reads back zeros. So the header is never
    whatever sits in some checkout. ``machine`` names an entry of the hardware registry
    (``src/merlin/resources/contracts/hardware_pins.yaml``); the ABI header digest it declares -- on itself or on an
    entry it is ``built_from`` -- is what ``header``'s bytes must hash to. A machine the registry
    declares no ABI header for is buildable only when the caller asserts one (``header_sha256``), and
    the record then says the assertion is the caller's, with the registry's own notes beside it.
    """
    import yaml

    from merlin.common.provenance_claims import pins_path

    registry = yaml.safe_load(pins_path().read_text(encoding="utf-8")) or {}
    entries = {**(registry.get("pins") or {}), **(registry.get("artifacts") or {})}
    if machine not in entries:
        raise _build_error(f"{machine!r} is not an entry of the hardware registry; name the machine built for")
    declared, declared_by, seen, frontier = None, None, set(), [machine]
    while frontier and declared is None:
        name = frontier.pop(0)
        if name in seen or name not in entries:
            continue
        seen.add(name)
        entry = entries[name] or {}
        if entry.get("abi_header_sha256"):
            declared, declared_by = str(entry["abi_header_sha256"]), name
        frontier.extend(str(n) for n in entry.get("built_from") or ())
    path = Path(header)
    if not path.is_file():
        raise _build_error(f"the parameter header {path} does not exist")
    actual = _sha256(path)
    if declared is None and not header_sha256:
        raise _build_error(
            f"the registry declares no ABI header for {machine!r} (nor for anything it is built from), so "
            f"which header is correct for it is UNKNOWN; assert one explicitly with its sha256"
        )
    for expected, who in ((declared, f"the registry ({declared_by})"), (header_sha256, "the caller")):
        if expected and actual != expected:
            raise _build_error(
                f"{path} hashes to {actual[:12]}, and {who} declares {expected[:12]} for {machine!r}; "
                f"building against it would describe a different machine"
            )
    record = {"machine": machine, "header": str(path.resolve()), "sha256": actual}
    if declared is not None:
        record.update({"declared_by": declared_by, "status": "registry_declared"})
    else:
        entry = entries[machine] or {}
        record.update(
            {
                "status": "caller_asserted",
                "registry_notes": [str(g) for g in entry.get("gaps") or ()] or entry.get("notes"),
            }
        )
    return record


def _with_header(recipe, header: Path, into: Path, overrides: Sequence[Path] = ()):
    """The target's recipe over a copy of its harness tree in which ``header`` replaces its namesake.

    Only the header moves (and each of ``overrides``, likewise by namesake); everything else in the
    tree is the target's own, byte for byte. Each namesake has to exist exactly once in the tree -- a
    file the tree does not include would change nothing, and one it holds twice would leave which one
    is read to the include order.
    """
    import shutil

    root = Path(os.path.commonpath([str(p) for p in recipe.include_roots]))
    placed = []
    for replacement in (Path(header), *(Path(o) for o in overrides)):
        matches = [p for p in root.rglob(replacement.name) if p.is_file()]
        if len(matches) != 1:
            raise _build_error(
                f"the harness tree {root} holds {len(matches)} file(s) named {replacement.name!r}; "
                f"a replacement must replace exactly one"
            )
        placed.append((replacement, matches[0].relative_to(root)))
    if len({rel for _r, rel in placed}) != len(placed):
        raise _build_error("two replacements name the same harness file")
    if into.exists():
        shutil.rmtree(into)
    shutil.copytree(root, into, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    # The copy keeps the source's modes, and a frozen source snapshot is read-only: its directories
    # are ours, so open them (a read-only copy would block both the replacement and the next build's
    # removal of the copy), and replace each namesake by unlinking it first rather than writing
    # through a file the copy cannot open.
    for directory in (into, *(p for p in into.rglob("*") if p.is_dir() and not p.is_symlink())):
        directory.chmod(directory.stat().st_mode | 0o700)
    for replacement, rel in placed:
        target = into / rel
        target.unlink()
        shutil.copyfile(replacement, target)

    def move(path: Path) -> Path:
        path = Path(path)
        try:
            return into / path.relative_to(root)
        except ValueError:
            return path

    return dataclasses.replace(
        recipe,
        include_roots=tuple(move(p) for p in recipe.include_roots),
        support_sources=tuple(move(p) for p in recipe.support_sources),
        link_script=move(recipe.link_script),
    )


def _headers_read(receipt: Mapping[str, Any], program: Path) -> dict[str, str]:
    """Every file the preprocessor READ to compile the program, with its digest.

    Asked of the compiler (``-M``) rather than assumed from a header's name: which parameter header a
    build saw is whichever one the include path resolved, and a copy of the vendor tree that differs by
    one line is a different machine.
    """
    done = subprocess.run(
        [str(receipt["compiler"]), *receipt["flags"], *receipt.get("includes", ()), "-M", str(program)],
        capture_output=True,
        text=True,
        cwd=str(program.parent),
    )
    if done.returncode != 0:
        return {"UNKNOWN": f"the compiler could not list its inputs: {done.stderr[-300:]}"}
    tokens = done.stdout.replace("\\\n", " ").split()
    files = [t for t in tokens[1:] if t != "\\"]
    return {str(Path(program.parent, f).resolve()): _sha256(Path(program.parent, f)) for f in files}


def machine_harts(machine: str) -> list[dict[str, Any]]:
    """The harts the hardware registry declares for ``machine`` (``[{hart, isa, unit, vlen?}]``), in
    hart order; empty when it declares none. Read on the entry itself -- a hart layout is a property of
    the elaborated design, not of anything it is built from."""
    import yaml

    from merlin.common.provenance_claims import pins_path

    registry = yaml.safe_load(pins_path().read_text(encoding="utf-8")) or {}
    entries = {**(registry.get("pins") or {}), **(registry.get("artifacts") or {})}
    rows = []
    for row in (entries.get(machine) or {}).get("harts") or ():
        if not isinstance(row, Mapping) or not str(row.get("isa") or "") or row.get("hart") is None:
            raise _build_error(f"{machine!r} declares a hart without an index and an ISA: {row!r}")
        rows.append(
            {"hart": int(row["hart"]), "isa": str(row["isa"]), "unit": bool(row.get("unit"))}
            | ({"vlen": int(row["vlen"])} if row.get("vlen") is not None else {})
        )
    return sorted(rows, key=lambda r: r["hart"])


# ---- whole_model_object_cache ------------------------------------------------------------------
# The whole-model build's per-group compiled-object cache.
#
# Compiling a package-answered group's LLVM artifact to a riscv object is the expensive step of a
# whole-model build: a full rebuild recompiles every answered group even when an edit changed one
# group's kernel. This module lets :mod:`merlin.analysis.whole_model.build.stages` reuse a PRIOR object for a
# group whose artifact text is byte-identical to one already compiled, under the same target and
# toolchain, instead of recompiling it::
#
#     root = object_cache_root()
#     fingerprint = object_cache_fingerprint(target)  # everything besides the artifact text and the
#     # target name that decides the compiled bytes
#     key = object_cache_key(artifact_text, target=target, fingerprint=fingerprint)
#     cached = root / key[:2] / f"{key}.o"  # present -> reuse; absent -> compile, then store
#
# A reused object must be byte-identical to what a fresh build would have produced for the same
# inputs, so the fingerprint has to cover everything that decides those bytes -- see
# :func:`object_cache_fingerprint`. A lookup that cannot be verified (an unreadable toolchain binary, a
# target whose recipe declares no stack-frame policy) disables the cache for that build rather than
# guessing: fail closed to a fresh compile, never to a cache entry nobody can vouch for.

#: The compiled-object cache namespace under ``out/artifacts/cache/`` and its disable switch.
OBJECT_CACHE_NAMESPACE = "whole-model-group-objects"
OBJECT_CACHE_DISABLE_ENV = "MERLIN_WHOLE_MODEL_NO_OBJECT_CACHE"

#: Objects here are tens of KB each, but a long optimization loop tries many distinct candidate
#: kernels per group -- each a new key -- so the cache is bounded least-recently-used, the same
#: pattern the package-reply cache uses (:func:`merlin.analysis.whole_model.build.stages.prune_replies`), for the
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
    ``.py`` file the lowering path can reach (:mod:`merlin.compiler.lower.model`, whole package) plus the two
    modules outside it that also decide the outcome (:mod:`merlin.development.targetgen.execution.runner.compile`, which
    calls the stack-frame preflight, and :mod:`merlin.target.contract.build.object_admission`, which measures
    it) -- sorted by path so the digest does not depend on filesystem enumeration order.

    Two RESOLVED FACTS also decide the bytes and are read directly rather than hoped to be implied by
    a source hash, because they can come from data (a target's registry/manifest), not code: the
    target's ``-march=`` (a different ISA string recompiles identically-shaped IR to different
    instructions) and its stack-frame policy's ``max_static_bytes`` -- a budget that merely tightens,
    with no source file touched, can turn on the arena-rewrite repair path
    (:func:`merlin.development.targetgen.execution.runner.compile._repair_oversized_frame`) and change the emitted bytes
    for the identical input.

    The toolchain binaries actually invoked (``clang``, ``mlir-translate``, the m2m venv's python) are
    hashed by CONTENT, not path/size/mtime, so an in-place binary replacement of the same size cannot
    be missed. NOT covered: an upgrade inside the m2m venv's installed packages (e.g. a torch-mlir
    version bump) that replaces no file this repo owns and no binary path here -- the reference-context
    cache (:func:`merlin.analysis.whole_model.grading.group_programs._reference_code_digest`) accepts the identical
    scope limit for the identical reason: chasing every transitive dependency of an out-of-process
    interpreter is unbounded, so this hashes the interpreter binary itself and the facts and code this
    repo controls, and documents the gap rather than hiding it.
    """
    from merlin.compiler.lower.model import toolchain
    from merlin.development.targetgen.execution.runner import compile as _compile_mod
    from merlin.runtime.backends import registry as backends
    from merlin.target.contract.build import object_admission as _stack_usage_mod

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
    build's kernel object. This is the same posture :func:`merlin.common.storage.content_store.object_for`
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
    one :func:`merlin.analysis.whole_model.build.stages._kernel_objects` would write for the same artifact text, so
    a prewarmed object is byte-for-byte the object that stage would have compiled itself.
    """
    from merlin.development.targetgen.execution.runner.compile import llvm_mlir_to_object
    from merlin.verification.lean.emission_custody import requires_fresh_object

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

    from merlin.verification.lean.emission_custody import requires_fresh_object

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


# ---- whole_model_replies -----------------------------------------------------------------------
# A compiler package's replies to a whole model's groups: recorded by content, asked in parallel.
#
# Split out of :mod:`merlin.analysis.whole_model.build.stages` (which re-exports these names). A package is a
# function of its own files and its input, so a reply recorded against both digests is what the package
# would print again (:class:`_ReplyCache`). A model's statement asks one group at a time because each
# splice binds against the buffers the previous ones declared; the ASKING does not depend on that, so
# :class:`Asker` puts every group's interface to the package the moment the statement writes it, in
# parallel, and -- when the target's object cache is on -- compiles each answered group's artifact to its
# object in the same worker, so the slowest group's ask and compile start at once rather than after every
# other group's. Nothing here names a target.

#: The recorded-reply cache is bounded, least-recently-used first: it grew ~0.5 GB per 15 min of a live
#: run with no bound at all. A reply is a pure function of its key, so an evicted one is only asked again.
REPLIES_MAX_BYTES_ENV = "MERLIN_PACKAGE_REPLIES_MAX_BYTES"
REPLIES_DEFAULT_MAX_BYTES = 4 << 30


def _replies_max_bytes() -> int:
    raw = (os.environ.get(REPLIES_MAX_BYTES_ENV, "") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else REPLIES_DEFAULT_MAX_BYTES


#: Replies live under one folder per package (``<cache>/<package digest[:16]>/<key>``), so a finished
#: package's replies can be dropped as a unit.
REPLY_PACKAGE_PREFIX = 16


def replies_folder(directory: Path, package_digest: str) -> Path:
    return Path(directory) / str(package_digest)[:REPLY_PACKAGE_PREFIX]


def drop_package_replies(directory: Path, package_digest: str) -> int:
    """Remove every recorded reply of one package; returns the bytes freed. A reply is a pure function
    of its key, so a later build of the same package only asks again."""
    folder = replies_folder(directory, package_digest)
    if not folder.is_dir():
        return 0
    freed = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    return freed


def _reply_entries(directory: Path):
    """Every recorded reply entry, in the per-package layout (and any left in the flat one)."""
    for child in Path(directory).iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue  # a staging directory belongs to a writer in flight
        if (child / "reply.json").is_file():
            yield child  # the flat layout this cache used before
            continue
        for entry in child.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                yield entry


def prune_replies(directory: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Evict whole reply entries, least recently used (entry mtime; a replay touches it) first, until the
    cache is under ``max_bytes``. Best-effort: a cache that cannot prune still works."""
    limit = _replies_max_bytes() if max_bytes is None else int(max_bytes)
    entries: list[tuple[float, int, Path]] = []
    try:
        for entry in _reply_entries(Path(directory)):
            size = sum(f.stat().st_size for f in entry.iterdir() if f.is_file())
            entries.append((entry.stat().st_mtime, size, entry))
    except OSError:
        return {"evicted": 0, "bytes_evicted": 0}
    total = sum(size for _mtime, size, _entry in entries)
    evicted = freed = 0
    for _mtime, size, entry in sorted(entries):
        if total <= limit:
            break
        shutil.rmtree(entry, ignore_errors=True)
        total -= size
        freed += size
        evicted += 1
    return {"evicted": evicted, "bytes_evicted": freed, "bytes_kept": total, "limit": limit}


class _ReplyCache:
    """A package's replies, keyed by the package's CONTENT, the entrypoint and the interface's bytes.

    A backend is a function of its own files and its input; a reply recorded against both digests is
    that function's value and is what the package would print again. Recorded so a model's seventy
    groups are asked once per package build rather than once per caller, and COUNTED, so a build
    record says how many of its replies were replayed rather than leaving it to be assumed.
    """

    def __init__(self, package_digest: str, directory: Path | None):
        self.package_digest = package_digest
        #: The whole cache (pruned as one); this package's replies live in their own folder under it.
        self.cache_root = directory
        self.directory = replies_folder(directory, package_digest) if directory is not None else None
        self.hits = 0
        self.misses = 0

    def _key(self, name: str, source: Path) -> str:
        digest = hashlib.sha256()
        for part in (self.package_digest, name, Path(source).read_bytes()):
            digest.update(part if isinstance(part, bytes) else str(part).encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def replay(self, name, source, output=None) -> subprocess.CompletedProcess | None:
        """The recorded reply to this question, re-enacted (its output file written), or ``None``."""
        if self.directory is None:
            return None
        entry = self.directory / self._key(name, Path(source))
        record = entry / "reply.json"
        if not record.is_file():
            return None
        reply = json.loads(record.read_text(encoding="utf-8"))
        with contextlib.suppress(OSError):
            os.utime(entry)  # recently used: the bounded cache evicts least-recently-used first
        if output is not None and (entry / "output").is_file():
            Path(output).write_bytes((entry / "output").read_bytes())
        self.hits += 1
        return subprocess.CompletedProcess(reply["argv"], reply["returncode"], reply["stdout"], reply["stderr"])

    def invoke(self, package, name, source, output=None, *, timeout=600):
        from merlin.development.targetgen.packaging import loading as OR

        if self.directory is None:
            return OR.run_entrypoint(package, name, source, output, timeout=timeout)
        replayed = self.replay(name, source, output)
        if replayed is not None:
            return replayed
        entry = self.directory / self._key(name, Path(source))
        done = OR.run_entrypoint(package, name, source, output, timeout=timeout)
        self.misses += 1
        staged = entry.with_name(f".{entry.name}.{os.getpid()}.{id(done)}")
        staged.mkdir(parents=True, exist_ok=True)
        if output is not None and Path(output).is_file():
            (staged / "output").write_bytes(Path(output).read_bytes())
        (staged / "reply.json").write_text(
            json.dumps(
                {
                    "argv": [str(a) for a in (done.args or [])],
                    "returncode": done.returncode,
                    "stdout": done.stdout,
                    "stderr": done.stderr,
                }
            ),
            encoding="utf-8",
        )
        try:
            staged.rename(entry)
        except OSError:  # another caller recorded the same reply first; theirs is the same value
            shutil.rmtree(staged, ignore_errors=True)  # and the half-written copy is ours to remove
        return done


class Asker:
    """Every group interface put to the package as soon as it is written, ``jobs`` at a time.

    ``recorded`` is the first statement pass's invoke: it replays a recorded reply, and for a question
    with no recorded answer it starts asking the package for the WHOLE interface in the background and
    refuses for now. ``invoke`` is the second pass's: it waits for a still-running ask of the same
    interface bytes, then replays its recorded replies (asking only what is still missing). The first
    pass therefore runs concurrently with every ask, and the second is a replay.

    ``compile_artifact(text)`` (optional) runs in the same worker once a group's target artifact is
    known -- a whole model's slowest group is usually slowest to compile as well, and that compile no
    longer waits for every other group to be asked first.
    """

    def __init__(self, package, cache: _ReplyCache, work: Path, *, timeout: int, jobs: int, compile_artifact=None):
        self.package, self.cache, self.work, self.timeout = package, cache, Path(work), timeout
        self.compile_artifact = compile_artifact
        self.pool = ThreadPoolExecutor(max_workers=max(1, jobs))
        self.pending: dict[str, Future] = {}
        self.missing: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _identity(source) -> str:
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()

    def _ask(self, interface: Path, scratch: Path) -> None:
        from merlin.development.targetgen.corpus import capsule_io as CC

        try:
            _buffer, artifact = CC.lower_interface(
                self.package,
                interface,
                scratch,
                contract=None,
                timeout=self.timeout,
                invoke=self.cache.invoke,
                overlap=True,
            )
        except Exception:  # noqa: BLE001 -- a refusal is the statement's to record, in order
            return
        finally:
            with contextlib.suppress(OSError):
                interface.unlink()
        if self.compile_artifact is not None and artifact:
            with contextlib.suppress(Exception):  # noqa: BLE001 -- an optimization; the build compiles it
                self.compile_artifact(artifact)

    def recorded(self, package, name, source, output=None, *, timeout=600):  # noqa: ARG002
        reply = self.cache.replay(name, source, output)
        if reply is not None:
            return reply
        with self._lock:
            self.missing.append(str(name))
            key = self._identity(source)
            if key not in self.pending:
                # A copy of the bytes: the statement's own work tree is rewritten by its next pass.
                scratch = self.work / "prefetch" / key[:16]
                scratch.mkdir(parents=True, exist_ok=True)
                interface = scratch.with_suffix(".iface.mlir")
                shutil.copyfile(source, interface)
                self.pending[key] = self.pool.submit(self._ask, interface, scratch / "generated")
        return subprocess.CompletedProcess([], 1, "", "no recorded reply yet")

    def invoke(self, package, name, source, output=None, *, timeout=600):
        with self._lock:
            waiting = self.pending.get(self._identity(source))
        if waiting is not None:
            waiting.result()
        return self.cache.invoke(package, name, source, output, timeout=timeout)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        # The asking's own work trees are scratch: every answer is recorded in the cache, and the
        # statement replays it into its own tree. Measured on SmolVLA (1551 groups) they held 7.9 GB.
        shutil.rmtree(self.work / "prefetch", ignore_errors=True)


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
