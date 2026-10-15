"""A numerics bundle on disk: per model, the program catalogs, the census and every input's device calls.

Layout (contract §8)::

    <bundle>/MANIFEST.json
    <bundle>/<model>/blobs/<sha256>.npy                       content-addressed operands and results
    <bundle>/<model>/programs/<program>/catalog.json, ref/, dev/, census.json
    <bundle>/<model>/<split>/<input>/<program>/calls.json      every (kept) device call of that run
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.model_check import catalog as CAT
from merlin.model_check.models import Model

MANIFEST_SCHEMA = "merlin.model_check.bundle.v1"


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
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return done.stdout.strip() or None


def build(
    bundle: Bundle,
    model: Model,
    *,
    target: str,
    captures: Path,
    programs: Iterable[str] | None = None,
    keep_call: Callable[[str, int], bool] | None = None,
    engine: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Catalog, census and the capture's own inputs (``visible/capture``) for every program of ``model``.

    The capture's inputs are one visible input; the frozen bundle adds the selected evaluation inputs.
    A program whose rewritten form does not reproduce its INT8 reference is recorded as inconsistent:
    that is a Merlin defect, and no submission can be judged on that program until it is fixed.
    """
    from merlin.runtime.dispatch_runtime import resolve_forward_args

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
        merged = manifest["models"].setdefault(model.name, {"model": model.name, "programs": {}})
        merged["programs"].update(record["programs"])
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return record
