"""A content-addressed store for engine cross-validation captures.

A capture is one ELF run on the reference engine and the candidate engine and compared. It costs a
REFERENCE-engine execution, which is the expensive half by more than an order of magnitude: building
one 90-case functional certificate cost over ninety minutes of serial reference simulation, in front
of a run that could not start without it.

That work is perfectly reusable and was being thrown away. A capture is a pure function of the bytes
that were run -- the ELF -- and the engines that ran them. Nothing about the campaign, the run id, the
capsule's name or the day it happened changes the answer, so a capture keyed on those inputs answers
for any later run that presents the same ones.

WHAT THE KEY MUST COVER, or a hit is a lie: the ELF digest AND every engine pin. The same program on
a rebuilt simulator is a different measurement -- that is the entire reason the pins exist -- so a
store keyed on the ELF alone would serve a new engine's question with an old engine's answer.

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


def capture_key(elf_sha256: str, pins: Mapping[str, Any]) -> str | None:
    """The store key, or ``None`` when the inputs do not fully determine a capture.

    Refuses rather than keying on a partial pin set: an entry filed under an incomplete key would be
    returned for a question it does not answer.
    """
    if not isinstance(elf_sha256, str) or len(elf_sha256) != 64:
        return None
    shas = _pin_shas(pins)
    if set(shas) != set(_PIN_NAMES):
        return None
    digest = hashlib.sha256()
    digest.update(elf_sha256.encode())
    for name in _PIN_NAMES:
        digest.update(b"\0")
        digest.update(f"{name}={shas[name]}".encode())
    return digest.hexdigest()


def _answers(document: Any, *, elf_sha256: str, pins: Mapping[str, Any]) -> bool:
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
    return True


def lookup(target: str, *, elf_sha256: str, pins: Mapping[str, Any]) -> dict[str, Any] | None:
    """A stored capture for these exact bytes and engines, or ``None``."""
    key = capture_key(elf_sha256, pins)
    if key is None:
        return None
    path = store_root(target) / f"{key}.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(document) if _answers(document, elf_sha256=elf_sha256, pins=pins) else None


def store(target: str, *, elf_sha256: str, pins: Mapping[str, Any],
          document: Mapping[str, Any]) -> Path | None:
    """File a capture. Returns the path, or ``None`` when it was not storable.

    A document that does not answer for the inputs it is filed under is REFUSED rather than written:
    the whole value of the store is that a hit needs no further checking.
    """
    key = capture_key(elf_sha256, pins)
    if key is None or not _answers(document, elf_sha256=elf_sha256, pins=pins):
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
