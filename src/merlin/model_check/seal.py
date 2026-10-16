"""Seal a numerics bundle and verify it (contract §8): a sha256 of every file, checked before use.

``seal`` writes ``SHA256SUMS`` (``<sha256>  <relative path>`` per file, sorted) and records its digest, file
count and size in ``MANIFEST.json`` under ``seal``. ``verify`` recomputes every digest and reports changed,
missing and extra files, plus the bundle's structure: every model the manifest names has its program
catalogs and censuses, and the binding receipt digest matches the manifest's. ``MANIFEST.json`` and
``SHA256SUMS`` are not listed (the second is pinned by the first), nor are dot-named files or directories
(locks and the builders' work directories). A frozen bundle never changes: anything written after the
seal makes ``verify`` fail until the bundle is sealed again.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from merlin.model_check.bundle import Bundle, BundleError

SUMS = "SHA256SUMS"
UNLISTED = frozenset({"MANIFEST.json", SUMS})


def _files(root: Path) -> Iterator[str]:
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        rel = Path(directory).relative_to(root)
        for name in sorted(names):
            if name.startswith(".") or (rel == Path(".") and name in UNLISTED):
                continue
            yield (rel / name).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digests(root: Path, names: list[str], jobs: int) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(zip(names, pool.map(lambda n: _sha256(root / n), names)))


def _read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name:
            sums[name] = digest
    return sums


def seal(bundle: Bundle, *, jobs: int = 8) -> dict[str, Any]:
    """Write ``SHA256SUMS`` for every file of ``bundle`` and pin it in the manifest."""
    import fcntl

    root = bundle.root
    names = list(_files(root))
    digests = _digests(root, names, jobs)
    sums = root / SUMS
    sums.write_text("".join(f"{digests[n]}  {n}\n" for n in names))
    record = {
        "sums": SUMS,
        "sums_sha256": _sha256(sums),
        "files": len(names),
        "bytes": sum((root / n).stat().st_size for n in names),
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (root / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = bundle.manifest()
        manifest["seal"] = record
        (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return record


def _structure(bundle: Bundle, manifest: dict[str, Any]) -> list[str]:
    problems = []
    for model, entry in (manifest.get("models") or {}).items():
        for program in entry.get("programs") or {}:
            directory = bundle.program_dir(model, program)
            for name in ("catalog.json", "census.json"):
                if not (directory / name).is_file():
                    problems.append(f"{model}/{program}: no {name}")
    binding = manifest.get("binding") or {}
    if binding.get("receipt"):
        path = bundle.root / binding["receipt"]
        if not path.is_file():
            problems.append(f"binding receipt {binding['receipt']} is missing")
        else:
            receipt = json.loads(path.read_text()).get("binding") or {}
            if receipt.get("receipt_sha256") != binding.get("receipt_sha256"):
                problems.append("binding receipt digest differs from the manifest's")
    return problems


def verify(bundle: Bundle, *, jobs: int = 8) -> dict[str, Any]:
    """Every file's digest against ``SHA256SUMS``, the sums against the manifest, and the structure."""
    root = bundle.root
    manifest = bundle.manifest()
    manifest_sha256 = _sha256(root / "MANIFEST.json")
    record = manifest.get("seal")
    if not record:
        raise BundleError(f"{root} is not sealed (run bundle-seal)")
    sums = root / record.get("sums", SUMS)
    if not sums.is_file() or _sha256(sums) != record.get("sums_sha256"):
        return {"status": "fail", "manifest_sha256": manifest_sha256, "detail": f"{SUMS} differs from the manifest's"}
    expected = _read_sums(sums)
    present = list(_files(root))
    actual = _digests(root, [n for n in present if n in expected], jobs)
    changed = sorted(n for n, d in actual.items() if d != expected[n])
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    structure = _structure(bundle, manifest)
    ok = not (changed or missing or extra or structure)
    return {
        "status": "pass" if ok else "fail",
        "manifest_sha256": manifest_sha256,
        "files": len(expected),
        "changed": changed[:20],
        "missing": missing[:20],
        "extra": extra[:20],
        "structure": structure,
        "counts": {"changed": len(changed), "missing": len(missing), "extra": len(extra)},
    }
