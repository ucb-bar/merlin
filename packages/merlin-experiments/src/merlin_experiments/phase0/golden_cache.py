"""Host-owned Phase 0 golden_cache implementation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .provenance import _document_digest

#: MEMOIZE THE GOLDEN ENGINES. A golden is DERIVED and fully deterministic -- operands come from a
#: name-salted fill with no RNG -- so identical inputs always produce an identical answer and
#: recomputing them is pure waste. The engines are deliberately slow: `fp_reduce` accumulates in the
#: device's own order, one step at a time, in pure Python, because a numpy dot product would round
#: differently from the hardware. Measured 2026-09-05: an atlas regeneration spends ~90 minutes there
#: and a radiance one ~40, nearly all of it recomputing capsules that nothing changed.
#:
#: ⚠️ THE KEY INCLUDES THE ENGINE'S OWN SOURCE, not just the entry. On this same date a conv2d branch
#: was added to the SIMT engine and an attention statement to the composed micro model. A cache keyed
#: on the entry alone would have served the pre-change goldens straight through both edits -- and a
#: stale golden does not fail loudly, it grades a backend against the wrong answer, which is precisely
#: the failure class this corpus exists to catch. Four things move a golden and all four are in the
#: key: the entry, the binding, the engine, and the operand synthesis.
#:
#: The store is a PURGEABLE cache namespace: deleting it costs time, never correctness.
_GOLDEN_CACHE_DISABLED = os.environ.get("MERLIN_NO_GOLDEN_CACHE", "").strip() not in ("", "0")


def source_files() -> tuple[Path, ...]:
    """The complete source closure formerly carried by the single generator file.

    Ignore interpreter caches, never source indirection. Required owners cannot
    disappear silently, and adding another Python helper extends the identity.
    """
    root = Path(__file__).parent
    required = {
        "__init__.py",
        "__main__.py",
        "profiles.py",
        "numerics.py",
        "golden_cache.py",
        "writer.py",
        "sweeps.py",
        "provenance.py",
        "generation.py",
    }
    sources = []
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise OSError("Phase 0 source closure contains a symlink")
        if path.suffix == ".py":
            if not path.is_file():
                raise OSError("Phase 0 source closure contains a nonregular Python source")
            sources.append(path)
    if not required <= {path.relative_to(root).as_posix() for path in sources}:
        raise OSError("Phase 0 source closure is incomplete")
    return tuple(sources)


def source_digest() -> str:
    """Conservatively invalidate cached answers after any derivation source edit."""
    root = Path(__file__).parent
    try:
        from merlin.common.paths import module_source_path

        record = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files()
        }
        helper = module_source_path("merlin.integrations.specir")
        if helper.is_symlink() or not helper.is_file():
            return "unresolvable"
        record["merlin.integrations.specir"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    except (ImportError, ValueError, OSError):
        return "unresolvable"
    return _document_digest(record)


def _source_digest_of(obj) -> str:
    """Digest of the SOURCE FILE that defines ``obj`` -- the bytes actually on disk.

    A dirty working tree changes this even when the commit does not, which is the property that makes
    an edit-in-progress invalidate its own cached results.
    """
    import inspect

    try:
        return hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
    except (TypeError, OSError):  # no resolvable source -> never cache against an unknown engine
        return "unresolvable"


def _golden_cache_key(fn, entry, binding, facts_sha: str = "") -> str:
    """Digest of everything that determines the answer.

    ``facts_sha`` is the target's RTL-facts digest -- the same one the corpus manifest records. The
    goldens are deliberately INDEPENDENT of the RTL (an oracle derived from the device would be the
    device grading itself), but the RTL still reaches them INDIRECTLY: the binding's tile edge, dtypes
    and subnormal handling are derived from the capability manifest, and an entry's extents come from
    facts like memory capacity and array geometry. Keying on the facts digest invalidates
    conservatively -- more often than strictly required, never less -- so a changed device can never be
    answered from a cache built against the previous one. An empty digest means the caller could not
    establish which device this is, and is carried as its own distinct key rather than treated as
    "no change".
    """
    from merlin.targetgen import corpus_operands as CO

    h = hashlib.sha256()
    h.update(_document_digest(entry).encode("utf-8"))
    h.update(repr(binding).encode("utf-8"))  # frozen dataclass -> stable repr
    h.update(fn.__name__.encode("utf-8"))
    h.update(_source_digest_of(fn).encode("utf-8"))  # the engine that will answer
    h.update(_source_digest_of(CO).encode("utf-8"))  # how its operands are synthesized
    h.update(source_digest().encode("utf-8"))  # all extracted generation helpers, not just the numerical engine
    h.update(str(facts_sha).encode("utf-8"))  # WHICH DEVICE this corpus is about
    return h.hexdigest()


def _golden_cached(fn, entry, binding, facts_sha: str = ""):
    """``fn(entry, binding)``, answered from the cache when every input digest matches."""
    if _GOLDEN_CACHE_DISABLED or _source_digest_of(fn) == "unresolvable" or source_digest() == "unresolvable":
        return fn(entry, binding)
    from merlin.common.artifacts import cache_dir

    key = _golden_cache_key(fn, entry, binding, facts_sha)
    path = Path(cache_dir("capsule_goldens")) / key[:2] / f"{key}.json"
    if path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            return rec["outputs"], rec["prov"]
        except Exception:  # noqa: BLE001 -- a damaged entry is a MISS, never a wrong answer
            pass
    outputs, prov = fn(entry, binding)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"outputs": outputs, "prov": prov}), encoding="utf-8")
        tmp.replace(path)  # atomic: a concurrent reader never sees a half-written entry
    except OSError:  # an unwritable cache must not fail a generation
        pass
    return outputs, prov
