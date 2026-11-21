"""A numerics bundle on disk: per model, the program catalogs, the census and every input's device calls.

Layout (contract §8)::

    <bundle>/MANIFEST.json
    <bundle>/<model>/blobs/<sha256>.npy                       content-addressed operands and results
    <bundle>/<model>/programs/<program>/catalog.json, ref/, dev/, census.json
    <bundle>/<model>/<split>/<input>/<program>/calls.json      every (kept) device call of that run
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.evaluation.model.bundles import program_catalog as CAT
from merlin.evaluation.model.bundles.models import Model

# ---- bundle ------------------------------------------------------------------------------------

MANIFEST_SCHEMA = "merlin.evaluation.model.bundles.seal.v1"


class BundleError(RuntimeError):
    """The bundle is missing, incomplete, or was built from different inputs."""


def stage_dir(captures: Path, model: Model, program: str) -> Path:
    """``<captures>/<model>/run/capture[/stages/<stage>]``, the sealed capture layout."""
    root = Path(captures) / model.name / "run" / "capture"
    stage = model.program(program).stage
    return root / "stages" / stage if stage else root


class Bundle:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def default(cls, target: str) -> Bundle:
        return cls(Path(os.environ.get("MERLIN_MODEL_CHECK_BUNDLE", f"/work/merlin/private/{target}/numerics/current")))

    def manifest(self) -> dict[str, Any]:
        path = self.root / "MANIFEST.json"
        if not path.is_file():
            raise BundleError(f"no bundle manifest at {path}")
        doc = json.loads(path.read_text())
        if doc.get("schema") != MANIFEST_SCHEMA:
            raise BundleError(f"{path}: not a {MANIFEST_SCHEMA} manifest")
        return doc

    def blobs(self, model: str) -> CAT.BlobStore:
        return CAT.BlobStore(self.root / model / "blobs")

    def program_dir(self, model: str, program: str) -> Path:
        return self.root / model / "programs" / program

    def catalog(self, model: str, program: str) -> CAT.ProgramCatalog:
        directory = self.program_dir(model, program)
        if not (directory / "catalog.json").is_file():
            raise BundleError(f"bundle has no catalog for {model}/{program}")
        return CAT.ProgramCatalog.load(directory)

    def inputs(self, model: str, split: str) -> list[str]:
        splits = ("visible", "sealed") if split == "all" else (split,)
        found = []
        for name in splits:
            directory = self.root / model / name
            if directory.is_dir():
                found += [f"{name}/{p.name}" for p in sorted(directory.iterdir()) if p.is_dir()]
        return found

    def calls(self, model: str, input_id: str, program: str) -> dict[str, Any]:
        return CAT.read_calls(self.root / model / input_id / program / "calls.json")


def _git_commit(root: Path) -> str | None:
    """``root``'s own source revision (export record or own checkout), never an enclosing repository's."""
    from merlin.common.provenance_claims import UNKNOWN
    from merlin.development.targetgen.packaging.records import source_revision

    revision = source_revision(root)
    return None if revision == UNKNOWN else revision


def build(
    bundle: Bundle,
    model: Model,
    *,
    target: str,
    captures: Path,
    programs: Iterable[str] | None = None,
    keep_call: Callable[[str, int], bool] | None = None,
    engine: dict[str, str] | None = None,
    binding: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Catalog, census and the capture's own inputs (``visible/capture``) for every program of ``model``.

    The capture's inputs are one visible input; the frozen bundle adds the selected evaluation inputs.
    A program whose rewritten form does not reproduce its INT8 reference is recorded as inconsistent:
    that is a Merlin defect, and no submission can be judged on that program until it is fixed.
    """
    from merlin.runtime.dispatch.whole_model import resolve_forward_args

    record: dict[str, Any] = {"model": model.name, "programs": {}}
    blobs = bundle.blobs(model.name)
    for program in programs or [p.name for p in model.programs]:
        started = time.monotonic()
        stage = stage_dir(captures, model, program)
        directory = bundle.program_dir(model.name, program)
        log(f"[bundle] {model.name}/{program}: preparing {stage}")
        catalog = CAT.prepare(stage, directory, target=target)
        census = CAT.census(catalog)
        (directory / "census.json").write_text(json.dumps(census, indent=1) + "\n")
        log(f"[bundle] {model.name}/{program}: evaluating the capture's inputs")
        evaluation = CAT.evaluate(catalog, resolve_forward_args(stage), blobs=blobs, keep_call=keep_call)
        consistency = evaluation.consistent()
        CAT.write_calls(
            bundle.root / model.name / "visible" / "capture" / program / "calls.json",
            evaluation.calls,
            model=model.name,
            program=program,
            input="visible/capture",
            consistency=consistency,
            reference_outputs=[blobs.put(v) for v in evaluation.reference],
        )
        record["programs"][program] = {
            "routed": len(catalog.sidecar.get("routed") or ()),
            "symbols": len(catalog.interfaces),
            "calls_recorded": len(evaluation.calls),
            "consistent": all(row["exact"] for row in consistency),
            "census": {k: census[k] for k in ("denominator", "device", "host", "device_mac_fraction")},
            "seconds": round(time.monotonic() - started, 1),
        }
        log(f"[bundle] {model.name}/{program}: {json.dumps(record['programs'][program])}")
    import fcntl

    manifest_path = bundle.root / "MANIFEST.json"
    with (bundle.root / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # two models' builds may finish together
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.is_file()
            else {"schema": MANIFEST_SCHEMA, "models": {}}
        )
        manifest.update({"target": target, "captures": str(captures), "merlin_commit": _git_commit(repo_root())})
        if engine:
            manifest["engine"] = dict(engine)
        if binding:  # the Spike/RTL/FPGA binding receipt this bundle's engine inputs produced
            manifest["binding"] = dict(binding)
        merged = manifest["models"].setdefault(model.name, {"model": model.name, "programs": {}})
        merged["programs"].update(record["programs"])
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return record


# ---- seal --------------------------------------------------------------------------------------
# Seal a numerics bundle and verify it (contract §8): a sha256 of every file, checked before use.
#
# ``seal`` writes ``SHA256SUMS`` (``<sha256>  <relative path>`` per file, sorted) and records its digest, file
# count and size in ``MANIFEST.json`` under ``seal``. ``verify`` recomputes every digest and reports changed,
# missing and extra files, plus the bundle's structure: every model the manifest names has its program
# catalogs and censuses, and the binding receipt digest matches the manifest's. ``MANIFEST.json`` and
# ``SHA256SUMS`` are not listed (the second is pinned by the first), nor are dot-named files or directories
# (locks and the builders' work directories). A frozen bundle never changes: anything written after the
# seal makes ``verify`` fail until the bundle is sealed again.

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
            try:
                receipt = json.loads(path.read_text())
            except (OSError, ValueError):
                receipt = None
                # Older sealed bundles stored a full binding command report here;
                # current bundle-build stores the raw receipt. Preserve verification
                # of both immutable formats.
            if isinstance(receipt, dict) and isinstance(receipt.get("binding"), dict):
                from merlin.evaluation.model.binding import receipt as BD

                nested = receipt["binding"].get("receipt")
                digest = BD.digest(nested) if isinstance(nested, dict) else None
                status = nested.get("status") if isinstance(nested, dict) else None
            else:
                digest = _sha256(path)
                status = receipt.get("status") if isinstance(receipt, dict) else None
            if (
                not isinstance(receipt, dict)
                or digest != binding.get("receipt_sha256")
                or status != binding.get("status")
            ):
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
