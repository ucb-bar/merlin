"""A content-addressed store for engine cross-validation captures.

A capture is one ELF run on the reference engine and the candidate engine and compared. It costs a
REFERENCE-engine execution, which is the expensive half by more than an order of magnitude: building
one 90-case functional certificate cost over ninety minutes of serial reference simulation, in front
of a run that could not start without it.

That work is perfectly reusable and was being thrown away. The engine outputs are a pure function of
the bytes that were run -- the ELF -- and the engines that ran them. A stored *capture* also says that
those outputs satisfy one canonical workload/oracle binding. Multiple descriptors can lower to the
same ELF while carrying different bindings, so those records must coexist instead of overwriting one
another.

WHAT THE KEY MUST COVER, or a hit is a lie: the ELF digest, every engine pin, the workload identity,
and the semantic-reference document. The same program on a rebuilt simulator is a different
measurement -- that is the entire reason the pins exist -- and the same program presented under a
different oracle is a different correctness assertion.

A HIT IS RE-CHECKED, NOT TRUSTED. The stored document states the ELF and pins it was taken under, and
those are compared against the ones being asked about before it is returned. A cache that hands back a
document it did not verify is worse than no cache: it converts a stale answer into a fresh-looking one.

PURGEABLE. This is a cache under ``out/artifacts/cache/``, never a product: every entry is exactly
reproducible by running the two engines again, and deleting it costs time and no evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import hashlib
import json

__all__ = ["capture_key", "lookup", "store", "store_root", "census"]

#: The pins a capture's validity depends on. A capture is about these engines and no others.
_PIN_NAMES = ("gsim_binary", "gsim_firrtl", "gsim_model", "verilator_binary", "verilator_firrtl")


def store_root(target: str) -> Path:
    """``out/artifacts/cache/gsim_captures/<target>/`` -- created on demand."""
    from merlin.common.artifacts import cache_dir
    root = cache_dir("gsim_captures") / str(target)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pin_shas(pins: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PIN_NAMES:
        entry = pins.get(name) if isinstance(pins, Mapping) else None
        sha = entry.get("sha256") if isinstance(entry, Mapping) else None
        if isinstance(sha, str) and sha:
            out[name] = sha
    return out


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def capture_key(elf_sha256: str, pins: Mapping[str, Any], *,
                workload_sha256: str | None = None,
                semantic_reference: Mapping[str, Any] | None = None) -> str | None:
    """The store key, or ``None`` when the inputs do not fully determine a capture.

    Refuses rather than keying on a partial pin set: an entry filed under an incomplete key would be
    returned for a question it does not answer.
    """
    if not isinstance(elf_sha256, str) or len(elf_sha256) != 64:
        return None
    shas = _pin_shas(pins)
    if set(shas) != set(_PIN_NAMES):
        return None
    if ((workload_sha256 is None) != (semantic_reference is None)
            or (workload_sha256 is not None
                and (not isinstance(workload_sha256, str) or len(workload_sha256) != 64))):
        return None
    digest = hashlib.sha256()
    digest.update(elf_sha256.encode())
    for name in _PIN_NAMES:
        digest.update(b"\0")
        digest.update(f"{name}={shas[name]}".encode())
    if workload_sha256 is not None:
        try:
            reference = _canonical(semantic_reference)
        except (TypeError, ValueError):
            return None
        digest.update(b"\0workload_sha256=")
        digest.update(workload_sha256.encode())
        digest.update(b"\0semantic_reference=")
        digest.update(hashlib.sha256(reference).hexdigest().encode())
    return digest.hexdigest()


def _answers(document: Any, *, elf_sha256: str, pins: Mapping[str, Any],
             workload_sha256: str | None = None,
             semantic_reference: Mapping[str, Any] | None = None) -> bool:
    """Does this stored document answer the question being asked?"""
    if not isinstance(document, Mapping):
        return False
    if document.get("elf_sha256") != elf_sha256:
        return False
    want = _pin_shas(pins)
    for side, binary, firrtl in (("reference", "verilator_binary", "verilator_firrtl"),
                                 ("candidate", "gsim_binary", "gsim_firrtl")):
        arm = document.get(side)
        if not isinstance(arm, Mapping):
            return False
        if arm.get("binary_sha256") != want.get(binary):
            return False
        if arm.get("firrtl_sha256") != want.get(firrtl):
            return False
    candidate = document.get("candidate")
    if (not isinstance(candidate, Mapping)
            or candidate.get("model_sha256") != want.get("gsim_model")):
        return False
    if workload_sha256 is not None and document.get("workload_sha256") != workload_sha256:
        return False
    if semantic_reference is not None and document.get("semantic_reference") != semantic_reference:
        return False
    return True


def lookup(target: str, *, elf_sha256: str, pins: Mapping[str, Any],
           workload_sha256: str | None = None,
           semantic_reference: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """A stored capture for these exact bytes and engines, or ``None``."""
    key = capture_key(elf_sha256, pins, workload_sha256=workload_sha256,
                      semantic_reference=semantic_reference)
    if key is None:
        return None
    root = store_root(target)
    paths = [root / f"{key}.json"]
    # Entries written before workload/oracle scoping used only the ELF and engine pins. Adopt one
    # only after checking the full requested binding, then promote it so another same-ELF workload
    # can no longer evict this result.
    if workload_sha256 is not None:
        legacy = capture_key(elf_sha256, pins)
        if legacy is not None and legacy != key:
            paths.append(root / f"{legacy}.json")
    for path in paths:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _answers(document, elf_sha256=elf_sha256, pins=pins,
                        workload_sha256=workload_sha256,
                        semantic_reference=semantic_reference):
            continue
        if path != paths[0]:
            store(target, elf_sha256=elf_sha256, pins=pins, document=document)
        return dict(document)
    return None


def store(target: str, *, elf_sha256: str, pins: Mapping[str, Any],
          document: Mapping[str, Any]) -> Path | None:
    """File a capture. Returns the path, or ``None`` when it was not storable.

    A document that does not answer for the inputs it is filed under is REFUSED rather than written:
    the whole value of the store is that a hit needs no further checking.
    """
    workload_sha256 = document.get("workload_sha256")
    semantic_reference = document.get("semantic_reference")
    key = capture_key(elf_sha256, pins, workload_sha256=workload_sha256,
                      semantic_reference=semantic_reference)
    if key is None or not _answers(
            document, elf_sha256=elf_sha256, pins=pins,
            workload_sha256=workload_sha256, semantic_reference=semantic_reference):
        return None
    path = store_root(target) / f"{key}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)                      # atomic: a reader never sees a half-written capture
    return path


def census(target: str) -> dict[str, Any]:
    """How much reference-engine time this store is currently holding."""
    root = store_root(target)
    entries = sorted(root.glob("*.json"))
    agree = 0
    for path in entries:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("agreement") == "AGREE":
                agree += 1
        except (OSError, ValueError):
            continue
    return {"root": str(root), "entries": len(entries), "agreeing": agree}
