"""Do not rebuild an executable whose inputs have not changed.

A capsule's ELF is a pure function of the program the package emitted, the command buffer it runs
against, and the recipe that turns those into machine code. Nothing in that function is time-varying,
and ``link_elf`` was deliberately made byte-reproducible so it could be treated as one -- yet every
grade rebuilt every capsule from scratch, and the cost was not incidental.

MEASURED, on ``merlincirct_g4p1_biasabi_20260906`` (819 screen-tier executions on disk):

===========  ======  ================  ===================
tier         n       build (median)    simulate (median)
===========  ======  ================  ===================
screen (L2)  819     4.06 s            0.153 s
cert (L3)    99      1.93 s            14.1 s
===========  ======  ================  ===================

The screen tier spent **27x more time building an executable than running it** -- 3,981 s of build in
one run. That inverts the premise :mod:`merlin.targetgen.tier_cache` was written on ("94.7% of a grade
is the cert tier", true only while the cert tier is being re-bought): once certificates carry, what is
left is a screen that is 96% compiler. The agent's own feedback loop paid it hardest, because its
default sweeps the whole corpus and a one-line lowering change leaves every other capsule's emitted
program byte-identical.

**This is a BUILD cache, not a verdict cache, and the distinction is the whole safety argument.** A
restored ELF is still executed, still parsed, still judged; nothing here can carry a verdict, shorten a
ladder, or turn a screen into a certificate. The worst a wrong hit can do is run the wrong program --
so the key covers every input that reaches the executable, and every rule below is written in the
direction of rebuilding.

**The key** (:func:`build_identity`), any part of which failing to resolve means no key and therefore a
normal build:

* the emitted program -- ``lowered_mlir_text`` verbatim;
* the command buffer, and the operands INJECTED into the harness. Both decide harness contents, and
  ``inputs`` is not derivable from ``cb`` alone (a caller may override what ``_recorded_operands``
  would have chosen), so it is hashed as passed;
* the target;
* the build recipe as it would run TODAY: its march, the exact compile and link argument vectors, the
  load address, and the BYTES of the link script and of every support source. A recipe that changes a
  flag or a support source is a different build;
* the toolchain that would answer, by resolved path and reported version. The driver is outside the
  repo and outside every git pin, so nothing else in this repo would notice it moving;
* the bytes of the code that performs the build (:func:`build_path`) -- the lowering pipeline, the
  contract compile path, the command-buffer module, the backend registry, and the target's own backend
  package, derived from the registry rather than named here.

Deliberately absent: merlin's commit (an edit that leaves all of the above identical cannot change the
output), the simulator, the grading path, and the RTL revision. None of them is an input to a compile;
the tier cache keys on them because it carries a VERDICT, and this does not.

**What is stored is the whole build, not just the ELF.** The generated directory is an artifact the
agent reads -- its ``kernel.ll``, its ``harness.c`` -- and restoring only the executable would silently
empty it. :func:`store` keeps every file in the directory, so a hit reproduces what the build would
have produced. Files are verified against their recorded digests on the way out; a store that has been
corrupted or truncated is a miss, never a partial restore.

**Within one grade the ladder builds once.** Every tier's adapter calls the compile path with the same
workdir, so a capsule reaching the cert tier used to compile a byte-identical ELF twice. A key marker
(:data:`KEY_MARKER`) left in the build directory makes the second call free without consulting the
store at all. ``capsule_runner._clear_stale_executable`` removes it along with the executable, so the
marker can never speak for a previous grade.

Switch off with ``MERLIN_ELF_BUILD_CACHE=0``; point it elsewhere by setting that variable to a
directory. The store is a regenerable cache under ``out/artifacts/cache/elf-builds/`` and is safe to
delete at any time; it self-prunes to ``MERLIN_ELF_BUILD_CACHE_ENTRIES`` (default 4000) oldest-first.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

__all__ = ["BUILD_IDENTITY_VERSION", "RECORD_VERSION", "KEY_MARKER", "DEFAULT_MAX_ENTRIES",
           "disabled", "cache_root", "build_path", "recipe_token", "toolchain_token",
           "build_identity", "artifact_identity", "contents", "reuse", "store",
           "metadata_for", "forget"]

#: Version of the payload :func:`build_identity` hashes. Bumping it invalidates every stored build,
#: which is the correct effect of changing what "the same inputs" means.
BUILD_IDENTITY_VERSION = 1

#: Version of the on-disk record shape. A record written by another version is not read.
RECORD_VERSION = 1

#: Written into a build directory, naming the key its contents were produced from. Lets a second tier
#: in the same ladder skip the build without touching the store.
KEY_MARKER = ".merlin_build_key"

#: Store entries kept before oldest-first pruning. A bound on a regenerable cache, not a fact.
DEFAULT_MAX_ENTRIES = 4000

_CACHE_ENV = "MERLIN_ELF_BUILD_CACHE"
_ENTRIES_ENV = "MERLIN_ELF_BUILD_CACHE_ENTRIES"

#: Spellings of "off". A value that is neither one of these nor a usable directory is a PATH.
_OFF = frozenset({"0", "off", "no", "false", "none", "disabled"})

#: Repo-relative sources whose bytes turn the inputs into an executable. Directories are expanded to
#: their ``.py`` files, so a module added to the lowering path is covered without editing this list.
#: The target's own backend package is DERIVED in :func:`build_path`, never named here.
_BUILD_MODULES = (
    "merlin/python/merlin/llvmlower",
    "merlin/python/merlin/targetgen/contract",
    "merlin/python/merlin/targetgen/runtime_build.py",
    "merlin/python/merlin/targetgen/build_cache.py",
    "merlin/python/merlin/runtime/commandbuffer.py",
    "merlin/python/merlin/runtime/backends/base.py",
)


def disabled() -> bool:
    """Whether the operator has switched the cache off."""
    return (os.environ.get(_CACHE_ENV, "") or "").strip().lower() in _OFF


def cache_root() -> "Path | None":
    """The store directory, or ``None`` when the cache is off or its root cannot be created."""
    if disabled():
        return None
    raw = (os.environ.get(_CACHE_ENV, "") or "").strip()
    try:
        if raw:
            d = Path(raw)
            d.mkdir(parents=True, exist_ok=True)
            return d
        from merlin.common.artifacts import cache_dir
        return cache_dir("elf-builds")
    except OSError:                  # an unwritable store is a cache that cannot be used, not a fault
        return None


def _max_entries() -> int:
    raw = (os.environ.get(_ENTRIES_ENV, "") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_MAX_ENTRIES


def _digest_of(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> "str | None":
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------
# WHAT BUILT IT
# ---------------------------------------------------------------------------------------------

def build_path(target: "str | None" = None) -> "tuple[Path, ...] | None":
    """Every file whose bytes decide what the build EMITS, or ``None`` if one cannot be located.

    ``None`` is not "nothing built it" -- it is "the builder cannot be established", and the caller must
    then build normally. A build path silently missing a member would key executables on a partial
    compiler.

    The target's backend contributes its whole package directory, DERIVED from the backend registry
    (it owns the harness renderer and the build recipe), so adding a target never means editing here.
    """
    from merlin.common.paths import repo_root
    root = Path(repo_root())
    files: list[Path] = []
    for rel in _BUILD_MODULES:
        p = root / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            found = [q for q in p.rglob("*.py") if "__pycache__" not in q.parts]
            if not found:
                return None
            files.extend(found)
        else:
            return None
    if target:
        try:
            from merlin.runtime.backends import base as _backends
            mod = _backends.get_backend(str(target))
            home = Path(getattr(mod, "__file__", "") or "").parent
        except Exception:            # noqa: BLE001 -- no resolvable backend: no build identity
            return None
        if not home.is_dir():
            return None
        files.extend(q for q in home.rglob("*.py") if "__pycache__" not in q.parts)
    return tuple(sorted(set(files)))


#: Memoized per compiler path: asking a driver its version is cheap but not free, and a grade asks
#: once per capsule per tier.
_TOOLCHAIN_MEMO: dict = {}


def toolchain_token(compiler: "str | Path") -> "str | None":
    """Identity of the compiler driver that would build today, or ``None``.

    The toolchain lives outside the repo and outside every declared hardware pin, so no other identity
    in this system would notice it being upgraded underneath a stored build. Resolved path plus the
    driver's own reported version and target triple; a driver that cannot be run has no identity and
    the build is not cached.
    """
    key = str(compiler)
    if key in _TOOLCHAIN_MEMO:
        return _TOOLCHAIN_MEMO[key]
    value = None
    try:
        import subprocess
        exe = shutil.which(key) or key
        parts = [str(Path(exe).resolve())]
        for flag in ("--version", "-dumpmachine"):
            proc = subprocess.run([exe, flag], capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                raise OSError(f"{exe} {flag} returned {proc.returncode}")
            parts.append(proc.stdout.strip())
        value = _digest_of(parts)
    except Exception:                # noqa: BLE001 -- unestablished toolchain: fail closed, no key
        value = None
    _TOOLCHAIN_MEMO[key] = value
    return value


def recipe_token(recipe: Any) -> "dict | None":
    """The build recipe as it would run today, as a hashable record, or ``None``.

    Argument vectors are rendered against placeholder file names so the token describes the FLAGS and
    not the temporary paths of one capsule. The link script and every support source contribute their
    bytes: they are compiled into the executable, so a change to either is a different build.
    """
    try:
        placeholder_src = Path("<source>.c")
        placeholder_obj = Path("<object>.o")
        record = {
            "march": recipe.march(),
            "load_address": recipe.load_address,
            "compile": [str(x) for x in recipe.compile_command(source=placeholder_src,
                                                               output=placeholder_obj)],
            "link": [str(x) for x in recipe.link_command(objects=[placeholder_obj],
                                                         output=Path("<elf>"),
                                                         link_script=Path("<script>.ld"))],
        }
        sources = {}
        for src in (Path(recipe.link_script), *(Path(s) for s in recipe.support_sources)):
            sha = _file_sha(src)
            if sha is None:              # a support source we cannot read is a build we cannot key
                return None
            sources[src.name] = sha
        record["sources"] = sources
        return record
    except Exception:                    # noqa: BLE001 -- an unreadable recipe is not a key
        return None


#: Build-path digests, keyed by the stat signature of the files they cover. Reading and hashing ~75
#: source files is cheap once and not cheap per capsule per tier; a stat walk decides whether the answer
#: is still current, and because the signature is part of the KEY there is no stale window.
_BUILD_PATH_MEMO: dict = {}


def _build_path_digest(files) -> str:
    """One digest over the bytes of the code that performs the build, memoized on their stat signature."""
    signature = []
    for p in files:
        try:
            st = p.stat()
            signature.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            signature.append((str(p), -1, -1))
    memo_key = _digest_of(signature)
    if memo_key not in _BUILD_PATH_MEMO:
        from merlin.common.provenance import source_digest
        _BUILD_PATH_MEMO[memo_key] = source_digest([str(p) for p in files])
    return _BUILD_PATH_MEMO[memo_key]


def artifact_identity(*, kind: str, target: str, inputs: "Mapping", source_files,
                      toolchain: "str | None") -> "str | None":
    """Content identity for a build that does NOT go through a declared harness recipe.

    :func:`build_identity` covers the shared operator build path, whose inputs are a recipe and a
    command buffer. A target may bring its own compiler instead -- one that takes the agent's emitted
    artifact and produces its own set of objects -- and such a build is just as much a pure function of
    its inputs, but has no recipe to key on. ``kind`` separates the namespaces so two different builders
    can never collide on one key.

    Same discipline as the recipe path: every component must resolve or there is no key, and no key
    means an ordinary build.
    """
    if disabled() or not isinstance(target, str) or not target.strip() or not kind:
        return None
    if not toolchain:
        return None
    files = tuple(Path(f) for f in (source_files or ()))
    if not files or not all(f.is_file() for f in files):
        return None
    try:
        payload = {
            "version": BUILD_IDENTITY_VERSION,
            "kind": str(kind),
            "target": target,
            "inputs": {str(k): _digest_of(v) for k, v in dict(inputs or {}).items()},
            "toolchain": toolchain,
            "build_path": _build_path_digest(files),
        }
        return _digest_of(payload)
    except Exception:                    # noqa: BLE001 -- an unhashable input is not a key
        return None


def build_identity(*, target: str, lowered_mlir_text: str, cb: Mapping,
                   inputs: "Mapping | None", recipe: Any) -> "str | None":
    """Content identity for exactly what one compile would produce, or ``None`` to build normally.

    Every component must resolve. See the module docstring for what is in the key and what is
    deliberately out of it.
    """
    if disabled() or not isinstance(target, str) or not target.strip():
        return None
    recipe_rec = recipe_token(recipe)
    if recipe_rec is None:
        return None
    compiler = recipe_rec["compile"][0] if recipe_rec["compile"] else ""
    tool = toolchain_token(compiler) if compiler else None
    if tool is None:
        return None
    files = build_path(target)
    if not files:
        return None
    try:
        payload = {
            "version": BUILD_IDENTITY_VERSION,
            "target": target,
            "program_sha256": hashlib.sha256(lowered_mlir_text.encode("utf-8")).hexdigest(),
            "command_buffer": _digest_of(cb),
            "inputs": _digest_of(inputs) if inputs else "",
            "recipe": recipe_rec,
            "toolchain": tool,
            "build_path": _build_path_digest(files),
        }
        return _digest_of(payload)
    except Exception:                    # noqa: BLE001 -- an unhashable input is not a key
        return None


# ---------------------------------------------------------------------------------------------
# THE STORE
# ---------------------------------------------------------------------------------------------

def contents(workdir: "str | Path") -> list:
    """Every file in the build directory, as relative paths, excluding the key marker.

    WHAT IS STORED IS THE WHOLE DIRECTORY, not a diff against its state before the build. An earlier
    version snapshotted first and kept only what changed, which was both slower and less safe: the
    pre-build walk re-read megabytes of emitted IR on every MISS -- measurably worse than having no
    cache at all on a build-once workload -- and a stat-based diff can miss a rewrite that lands on the
    same size inside one mtime tick, which would leave a restored directory holding a stale file beside
    a fresh executable. Storing everything cannot be wrong in that direction, and the extra bytes are
    the few files the emit step put there, whose content the key already pins.
    """
    root = Path(workdir)
    out: list = []
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.name != KEY_MARKER:
                out.append(str(p.relative_to(root)))
        except OSError:
            continue
    return out


def _entry(root: Path, key: str) -> Path:
    return root / key[:2] / key


def _already_there(dst: Path, meta: Mapping) -> bool:
    """Whether ``dst`` already holds exactly the bytes the record says, so the copy can be skipped.

    THE ENTRY HOLDS THE WHOLE DIRECTORY, AND MOST OF IT IS ALREADY IN PLACE ON A HIT. A capsule's
    build directory is written by the emit step before the compile runs, so a restore into a live run
    re-copies megabytes of emitted IR that are already there and already identical. Size is checked
    first because it settles the common negative for free; the digest is what decides, so a file that
    differs in content but not in length is still replaced.
    """
    try:
        if not dst.is_file() or dst.stat().st_size != int(meta.get("size", -1)):
            return False
    except (OSError, TypeError, ValueError):
        return False
    return _file_sha(dst) == meta.get("sha256")


def reuse(workdir: "str | Path", key: "str | None", elf_name: "str | None" = None) -> "Path | None":
    """The executable for ``key`` already present or restorable into ``workdir``, or ``None``.

    Two hits, narrowest first. The MARKER path costs nothing and covers the second tier of one ladder,
    where the previous tier's adapter has already built exactly these bytes into exactly this
    directory. The STORE path restores the whole build, verifying every file against its recorded
    digest first: a corrupted or partial entry is a miss and the caller builds.
    """
    if not key:
        return None
    work = Path(workdir)
    marker = work / KEY_MARKER
    if elf_name:
        # Fast path only when the caller already knows the name; a builder whose output name varies
        # asks the record instead (below), which is the whole reason elf_name is optional.
        try:
            elf = work / elf_name
            if elf.is_file() and marker.is_file() \
                    and marker.read_text(encoding="utf-8").strip() == key:
                return elf
        except OSError:
            pass
    root = cache_root()
    if root is None:
        return None
    entry = _entry(root, key)
    try:
        record = json.loads((entry / "record.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("version") != RECORD_VERSION or record.get("key") != key:
        return None
    files = record.get("files")
    if not isinstance(files, dict) or record.get("elf") not in files:
        return None
    if elf_name and record.get("elf") != elf_name:
        return None                  # a record for a different primary output is not this build
    staged = []
    for rel, meta in files.items():
        src = entry / "files" / rel
        if not src.is_file() or _file_sha(src) != (meta or {}).get("sha256"):
            return None              # corrupt or truncated entry: rebuild rather than restore part
        staged.append((src, rel, meta))
    try:
        work.mkdir(parents=True, exist_ok=True)
        for src, rel, meta in staged:
            dst = work / rel
            if _already_there(dst, meta):
                continue        # the emit step's own output: present by construction, and identical
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            mode = meta.get("mode")
            if isinstance(mode, int):
                os.chmod(dst, mode)
        marker.write_text(key, encoding="utf-8")
    except OSError:                  # a restore we could not finish must not look like a build
        try:
            (work / record["elf"]).unlink()
        except OSError:
            pass
        return None
    try:                             # a hit is a use: keep the entry young for the pruner
        os.utime(entry, None)
    except OSError:
        pass
    return work / record["elf"]


def store(key: "str | None", workdir: "str | Path", elf_name: str,
          metadata: "Mapping | None" = None) -> None:
    """Keep every file this build created or changed, under ``key``. Never raises.

    The entry is assembled in a temporary directory and moved into place, so a concurrent worker
    building the same capsule either wins the rename or discards its copy -- a reader never sees a
    half-written entry.
    """
    if not key:
        return
    work = Path(workdir)
    elf = work / elf_name
    if not elf.is_file():
        return
    root = cache_root()
    if root is None:
        return
    entry = _entry(root, key)
    if (entry / "record.json").is_file():
        try:
            (work / KEY_MARKER).write_text(key, encoding="utf-8")
        except OSError:
            pass
        return
    tmp = None
    try:
        produced = contents(work)
        if elf_name not in produced:
            produced.append(elf_name)
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".staging_", dir=entry.parent))
        files: dict = {}
        for rel in produced:
            src = work / rel
            if not src.is_file():
                continue
            dst = tmp / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            # Digested from the COPY: that is the byte sequence a restore will hand back, so if the
            # source changed under us mid-copy the record describes what was actually stored.
            st = src.stat()
            files[rel] = {"sha256": _file_sha(dst), "mode": st.st_mode & 0o777,
                          "size": dst.stat().st_size}
        if elf_name not in files:
            raise OSError("the executable did not survive staging")
        # METADATA IS PART OF THE BUILD'S RESULT, not decoration. One target stamps WHICH toolchain
        # produced the graded executable, precisely so a fallback can never be silent; restoring the
        # files without that stamp would make a cached build the one case where the stamp went missing.
        (tmp / "record.json").write_text(json.dumps(
            {"version": RECORD_VERSION, "key": key, "elf": elf_name, "files": files,
             "metadata": {str(k): v for k, v in dict(metadata or {}).items()}},
            sort_keys=True, indent=1), encoding="utf-8")
        try:
            os.rename(tmp, entry)
            tmp = None
        except OSError:              # another worker stored the same build first: theirs is identical
            pass
        (work / KEY_MARKER).write_text(key, encoding="utf-8")
        _prune(root)
    except Exception:                # noqa: BLE001 -- a cache that cannot write is not a build failure
        pass
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def metadata_for(key: "str | None") -> "dict | None":
    """The metadata stored beside a build, or ``None``.

    Read separately from :func:`reuse` so a caller that needs no metadata pays nothing, and so a
    caller that DOES need it fails closed: ``None`` here means the stamp is unavailable, which must
    make the caller rebuild rather than proceed with an unstamped result.
    """
    if not key:
        return None
    root = cache_root()
    if root is None:
        return None
    try:
        record = json.loads((_entry(root, key) / "record.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("version") != RECORD_VERSION or record.get("key") != key:
        return None
    meta = record.get("metadata")
    return dict(meta) if isinstance(meta, dict) else {}


def forget(workdir: "str | Path") -> None:
    """Drop the key marker from a build directory, so its contents can never speak for a later grade.

    Called beside the removal of a stale executable: the marker asserts "this directory holds the build
    for that key", and a directory whose executable has just been deleted asserts nothing.
    """
    try:
        (Path(workdir) / KEY_MARKER).unlink()
    except OSError:
        pass


def _prune(root: Path) -> None:
    """Keep the store bounded, oldest-first. Best-effort: a cache that cannot prune still works."""
    try:
        entries = [d for shard in root.iterdir() if shard.is_dir()
                   for d in shard.iterdir() if d.is_dir() and not d.name.startswith(".staging_")]
        excess = len(entries) - _max_entries()
        if excess <= 0:
            return
        entries.sort(key=lambda d: d.stat().st_mtime)
        for d in entries[:excess]:
            shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass
