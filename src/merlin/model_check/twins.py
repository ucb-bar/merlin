"""The differential's host twins, run once per release: every step's outputs cached in the bundle.

A host twin (:func:`merlin.model_check.full_build.build`, ``offload=False``) depends on no submission
byte, and a submission whose chain is correct feeds every step exactly the inputs the twin's own chain
does. So the twin chain of every bundle input is computed once (``bundle-twins``) and full mode compares a
submission's step outputs with the cached ones, running the twin only on a cache miss.

A cached step is identified by the twin's identity -- its ELF sha256 (a twin build is reproducible, so
full mode rebuilds it to read the hash), the Merlin commit and the simulator -- and the sha256 of the
step's bound inputs (:func:`inputs_sha256`). Layout, below the bundle::

    <model>/twins/<program>/<twin_key>.json   {schema, twin: identity, steps: {inputs_sha256: entry}}
    <model>/twins/blobs/<sha256>.npy           every output, content-addressed
    MANIFEST.json  models.<model>.twins.<program> = {twin_key, twin, index, inputs: {input_id: [sha...]}}

They are ordinary bundle files, so bundle verification and sealing cover them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from merlin.model_check import bundle as B
from merlin.model_check import chain as CH
from merlin.model_check import full_build as FB

SCHEMA = "merlin.model_check.twins.v1"


def _sha_file(path: Path) -> str:
    return FB.sha256(Path(path))


def engine_identity() -> dict[str, str]:
    """The functional simulator twins run on (:func:`merlin.model_check.full_build.run_host`)."""
    from merlin.runtime.backends.spike import spike_path

    path = spike_path()
    return {"simulator": path.name, "sha256": _sha_file(path)}


def identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """What a cached twin step is valid for: the twin image, the Merlin that built it, the simulator."""
    from merlin.common.paths import repo_root

    return {
        "elf_sha256": record["elf_sha256"],
        "build_hash": record["build_hash"],
        "merlin_commit": B._git_commit(repo_root()),  # noqa: SLF001
        "engine": engine_identity(),
    }


def key(twin: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(twin, sort_keys=True).encode()).hexdigest()


def inputs_sha256(stage: Path, inputs: Mapping[str, np.ndarray]) -> str:
    """The step's bound inputs as the image embeds them (each cast to the capture's dtype), hashed."""
    digest = hashlib.sha256()
    with np.load(Path(stage) / "inputs.npz") as expected:
        for name in expected.files:
            value = np.ascontiguousarray(np.asarray(inputs[name]).astype(expected[name].dtype, copy=False))
            digest.update(f"{name}|{value.dtype.str}|{value.shape}|".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


class TwinCache:
    """One program's cached twin steps, for one twin identity."""

    def __init__(self, bundle: B.Bundle, model: str, program: str, twin: Mapping[str, Any]) -> None:
        self.root = bundle.root / model / "twins"
        self.twin = dict(twin)
        self.key = key(twin)
        self.path = self.root / program / f"{self.key}.json"
        self.steps: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            doc = json.loads(self.path.read_text())
            if doc.get("schema") == SCHEMA and doc.get("twin") == self.twin:
                self.steps = dict(doc.get("steps") or {})
        self._lock = threading.Lock()

    def get(self, inputs_sha: str) -> tuple[list[np.ndarray], dict[str, Any]] | None:
        entry = self.steps.get(inputs_sha)
        if entry is None:
            return None
        values = [np.load(self.root / "blobs" / f"{k}.npy", allow_pickle=False) for k in entry["outputs"]]
        return values, entry

    def put(self, inputs_sha: str, values: list[np.ndarray], row: Mapping[str, Any]) -> None:
        keys = [self._store(np.asarray(v)) for v in values]
        with self._lock:
            self.steps[inputs_sha] = {
                "outputs": keys,
                **{k: row.get(k) for k in ("cycles_functional", "seconds", "elf_bytes")},
            }
            self._save()

    def _store(self, array: np.ndarray) -> str:
        array = np.ascontiguousarray(array)
        digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}".encode() + array.tobytes()).hexdigest()
        path = self.root / "blobs" / f"{digest}.npy"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_name(f"{digest}.{os.getpid()}.{threading.get_ident()}.pending.npy")
            np.save(pending, array, allow_pickle=False)
            pending.replace(path)
        return digest

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(".pending")
        pending.write_text(json.dumps({"schema": SCHEMA, "twin": self.twin, "steps": self.steps}, indent=1) + "\n")
        pending.replace(self.path)


def _record_manifest(bundle: B.Bundle, model: str, programs: Mapping[str, dict[str, Any]]) -> None:
    import fcntl

    path = bundle.root / "MANIFEST.json"
    with (bundle.root / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(path.read_text())
        entry = manifest.setdefault("models", {}).setdefault(model, {"model": model})
        twins = entry.setdefault("twins", {})
        for program, row in programs.items():
            old = twins.get(program) or {}
            inputs = dict(old.get("inputs") or {}) if old.get("twin_key") == row["twin_key"] else {}
            inputs.update(row["inputs"])
            twins[program] = {**row, "inputs": inputs}
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(manifest, indent=1) + "\n")
        pending.replace(path)


def command(args, report: dict, *, models, bundle: B.Bundle, log=print) -> dict:
    """``bundle-twins``: build every program's host twin, run each input's whole twin chain, cache it."""
    from merlin.common.paths import build_dir
    from merlin.llvmlower.c_runtime import _out_specs
    from merlin.model_check import full as F

    started_all = time.monotonic()
    deadline = started_all + float(args.timeout_s)
    model = models.model(args.model)
    manifest = bundle.manifest()
    captures = Path(manifest.get("captures") or "")
    inputs = F.judged_inputs(bundle, model.name, args.split, getattr(args, "inputs", None))
    work = Path(args.work or (build_dir() / "model_check" / "twins")) / model.name
    images = F.build_root(work, model.name)
    settings = FB.BuildSettings.of(models, model, args.host_package)
    stages = {p.name: B.stage_dir(captures, model, p.name) for p in model.programs}
    specs = {name: _out_specs(stage / "model.mlir") for name, stage in stages.items()}
    report["bundle"] = {"path": str(bundle.root), "merlin_commit": manifest.get("merlin_commit")}

    records, caches = {}, {}
    for program in model.programs:
        log(f"[twins] {model.name}/{program.name}: building the host twin")
        records[program.name] = FB.build(
            stages[program.name],
            target=models.target,
            label=f"{model.name}:{program.name}",
            submission=Path("."),
            submission_sha256="",
            settings=settings,
            directory=images / "host" / program.name,
            offload=False,
        )
        caches[program.name] = TwinCache(bundle, model.name, program.name, identity(records[program.name]))

    def one(input_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"input": input_id, "steps": [], "chain": {p.name: [] for p in model.programs}}
        started = time.monotonic()
        try:
            arrays = F._arrays(F.derived_input(bundle, model.name, input_id))  # noqa: SLF001
            run = CH.ChainRun()
            for program in model.programs:
                run.steps[program.name] = []
                cache, stage = caches[program.name], stages[program.name]
                for step in range(program.steps):
                    bound = CH.bind(program, model, arrays, step, run)
                    digest = inputs_sha256(stage, bound)
                    hit = cache.get(digest)
                    if hit is None:
                        work_dir = work / "runs" / input_id / f"{program.name}.{step}"
                        done = F.execute(
                            models.target, records[program.name], stage, bound, work_dir, deadline, twin=True
                        )
                        cache.put(digest, done["values"], done["row"])
                        values, row, how = done["values"], done["row"], "computed"
                    else:
                        values, row, how = hit[0], hit[1], "cached"
                    run.steps[program.name].append(F.carried(values, specs[program.name]))
                    result["chain"][program.name].append(digest)
                    result["steps"].append(
                        {
                            "program": program.name,
                            "step": step,
                            "twin": how,
                            "inputs_sha256": digest,
                            **{k: row.get(k) for k in ("cycles_functional", "seconds")},
                        }
                    )
                    log(f"[twins] {model.name} {input_id} {program.name}#{step}: {how}, {row.get('seconds')} s")
        except Exception as error:  # noqa: BLE001 -- one input's failure is reported, the others still cached
            result["error"] = f"{type(error).__name__}: {error}"[:2000]
            log(f"[twins] {model.name} {input_id}: {result['error']}")
        result["seconds"] = round(time.monotonic() - started, 1)
        return result

    jobs = int(args.jobs) if getattr(args, "jobs", None) else F.default_jobs(len(inputs))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, inputs))

    recorded = {
        program.name: {
            "twin_key": caches[program.name].key,
            "twin": caches[program.name].twin,
            "index": str(caches[program.name].path.relative_to(bundle.root)),
            "inputs": {r["input"]: r["chain"][program.name] for r in results if "error" not in r},
        }
        for program in model.programs
    }
    _record_manifest(bundle, model.name, recorded)
    failed = [r for r in results if "error" in r]
    for r in failed:
        report["failures"].append(
            {
                "kind": "infrastructure",
                "gate": None,
                "input": r["input"],
                "split": r["input"].partition("/")[0],
                "detail": r["error"],
            }
        )
    for program in model.programs:
        rows = [dict(s, input=r["input"]) for r in results for s in r["steps"] if s["program"] == program.name]
        report["programs"].append(
            {
                "program": program.name,
                "build": {
                    k: records[program.name][k]
                    for k in ("status", "elf_sha256", "elf_bytes", "build_hash", "seconds", "reused")
                },
                "twin": {"key": caches[program.name].key, **caches[program.name].twin},
                "runs": rows,
            }
        )
    steps = sum(len(r["steps"]) for r in results)
    report["gates"]["twins"] = {
        "status": "pass" if not failed else "fail",
        "detail": f"{len(results) - len(failed)}/{len(results)} input chain(s) cached ({steps} step(s))",
    }
    report["timing"].update(
        {
            "per_input_seconds": {r["input"]: r["seconds"] for r in results},
            "jobs": jobs,
            "twin_build_seconds": round(sum(0 if r["reused"] else r["seconds"] for r in records.values()), 1),
        }
    )
    return report
