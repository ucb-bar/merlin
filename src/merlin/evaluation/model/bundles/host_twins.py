"""The differential's host twins, run once per release: every step's outputs cached in the bundle.

A host twin (:func:`merlin.evaluation.model.whole_model.build.build`, ``offload=False``) depends on no submission
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
import importlib.util
import json
import os
import platform
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from merlin.evaluation.model.bundles import references as CH
from merlin.evaluation.model.bundles import seal as B
from merlin.evaluation.model.whole_model import build as FB

# ---- twins -------------------------------------------------------------------------------------

SCHEMA = "merlin.evaluation.model.bundles.host_twins.v1"


def _sha_file(path: Path) -> str:
    return FB.sha256(Path(path))


def engine_identity() -> dict[str, str]:
    """The functional simulator twins run on (:func:`merlin.evaluation.model.whole_model.build.run_host`)."""
    from merlin.runtime.backends.baremetal.spike_launcher import spike_path

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
    from merlin.compiler.lower.c.runtime import _out_specs
    from merlin.evaluation.model.whole_model import check as F

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


# ---- full_reference_cache ----------------------------------------------------------------------
# Byte-bound per-step references and structural cached-twin readiness for full model checks.

REFERENCE_CACHE_SCHEMA = "merlin.evaluation.model.bundles.host_twins.v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, *, python_only: bool = False) -> str:
    """Hash reachable bytes and names; do not treat a timestamp or a broken traversal as identity."""
    root = Path(root)
    digest = hashlib.sha256()
    seen: set[str] = set()

    def failed(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=True, onerror=failed):
        dirs[:] = sorted(name for name in dirs if name != "__pycache__")
        real = os.path.realpath(directory)
        digest.update(Path(directory).relative_to(root).as_posix().encode() + b"\0")
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        for name in sorted(files):
            if python_only and not name.endswith(".py"):
                continue
            path = Path(directory) / name
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(bytes.fromhex(file_digest(path)))
    if not root.is_dir():
        raise FileNotFoundError(root)
    return digest.hexdigest()


def arrays_digest(arrays) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError("reference input arrays cannot have object dtype")
        digest.update(json.dumps([name, value.dtype.str, list(value.shape)]).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluator_identity() -> dict[str, Any]:
    """Conservative source/dependency closure of the reference parser and evaluator, read afresh."""
    from merlin.evaluation.model.linalg_semantics import numeric_contract

    core = importlib.util.find_spec("merlin.evaluation.model")
    if core is None or not core.origin:
        raise RuntimeError("no core source closure for the reference evaluator")
    roots = [Path(core.origin).parent.parent]
    dependencies = {}
    for name in ("numpy", "xdsl"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise RuntimeError(f"no evaluator dependency source for {name}")
        root = Path(spec.origin).parent
        dependencies[name] = tree_digest(root)
        libraries = root.with_name(name + ".libs")
        if libraries.is_dir():
            dependencies[name + ".libs"] = tree_digest(libraries)
    from merlin.compiler.lower.model.toolchain import compiler_python

    printer = compiler_python()
    bindings = {}
    # A custom-form input may use the selected compiler environment's generic printer.
    # Its bindings include native MLIR libraries, so Python package versions alone are insufficient.
    for library in (printer.parent.parent / "lib", printer.parent.parent / "lib64"):
        for package in library.glob("python*/site-packages/torch_mlir"):
            bindings[str(package)] = tree_digest(package)
    return {
        "numeric_contract": numeric_contract(),
        "sources": {str(root): tree_digest(root, python_only=True) for root in roots},
        "generic_printer": {
            "python": str(printer),
            "sha256": file_digest(printer) if printer.is_file() else None,
            "bindings": bindings,
        },
        "dependencies": dependencies,
        "python": {"version": sys.version, "machine": platform.machine(), "sha256": file_digest(Path(sys.executable))},
        "threads": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def material(model, bundle, arrays, counts, catalogs) -> dict[str, Any]:
    """All chain inputs and reference/weight/evaluator bytes, independently of a device submission."""
    return {
        "schema": REFERENCE_CACHE_SCHEMA,
        "model": asdict(model),
        "counts": dict(counts),
        "inputs_sha256": arrays_digest(arrays),
        "bundle_manifest_sha256": file_digest(bundle.root / "MANIFEST.json"),
        "bundle_sums_sha256": (
            file_digest(bundle.root / "SHA256SUMS") if (bundle.root / "SHA256SUMS").is_file() else None
        ),
        "models_configuration_sha256": (
            file_digest(bundle.root / "configuration" / "models.yaml")
            if (bundle.root / "configuration" / "models.yaml").is_file()
            else None
        ),
        "programs": {
            name: {
                "catalog": catalog.record(),
                "stage_sha256": tree_digest(catalog.stage),
                "reference_sha256": tree_digest(catalog.ref_prepared.parent),
            }
            for name, catalog in catalogs.items()
        },
        "evaluator": evaluator_identity(),
    }


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class StepCache:
    """A reference cache whose entries bind context, actual bound inputs and every output byte."""

    def __init__(self, root: Path, identity: dict[str, Any]):
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.key = _json_digest(self.identity)
        self.root = Path(root) / "v1" / self.key

    def _path(self, program: str, step: int) -> Path:
        # A program name is data, never a cache pathname supplied by the models file.
        name = hashlib.sha256(program.encode()).hexdigest()
        return self.root / name / f"{step}.npz"

    def get(self, program: str, step: int, inputs, arity: int) -> list[np.ndarray] | None:
        path = self._path(program, step)
        try:
            row = json.loads(path.with_suffix(".json").read_text())
            if (
                row.get("schema") != REFERENCE_CACHE_SCHEMA
                or row.get("context_sha256") != self.key
                or row.get("identity") != self.identity
            ):
                return None
            if row.get("inputs_sha256") != arrays_digest(inputs) or len(row.get("outputs", [])) != arity:
                return None
            if row.get("npz_sha256") != file_digest(path):
                return None
            with np.load(path, allow_pickle=False) as data:
                if data.files != [f"out{i}" for i in range(arity)]:
                    return None
                values = [np.asarray(data[f"out{i}"]) for i in range(arity)]
            for value, expected in zip(values, row["outputs"], strict=True):
                if expected != {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": arrays_digest({"output": value}),
                }:
                    return None
            return values
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, program: str, step: int, inputs, values) -> None:
        path = self._path(program, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".{os.getpid()}.{threading.get_ident()}.pending"
        pending = path.with_name(path.stem + suffix + ".npz")
        values = [np.asarray(value) for value in values]
        np.savez(pending, **{f"out{i}": value for i, value in enumerate(values)})
        row = {
            "schema": REFERENCE_CACHE_SCHEMA,
            "context_sha256": self.key,
            "identity": self.identity,
            "inputs_sha256": arrays_digest(inputs),
            "npz_sha256": file_digest(pending),
            "outputs": [
                {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": arrays_digest({"output": value})}
                for value in values
            ],
        }
        receipt = path.with_name(path.stem + suffix + ".json")
        receipt.write_text(json.dumps(row, sort_keys=True) + "\n")
        pending.replace(path)
        receipt.replace(path.with_suffix(".json"))


def same_outputs(actual, expected) -> bool:
    """Require every declared output; a shorter list is never an exact reference match."""
    return len(actual) == len(expected) and all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.asarray(a).shape == np.asarray(b).shape
        and np.array_equal(a, b)
        for a, b in zip(actual, expected, strict=True)
    )


def selected_outputs(model, counts, frozen):
    """Select a requested prefix only after validating the entire frozen judged-output arity."""
    expected = sum(model.program(ref["program"]).steps for ref in model.judged["outputs"])
    if len(frozen) != expected:
        raise ValueError(f"frozen reference has {len(frozen)} judged outputs; full chain declares {expected}")
    selected = []
    offset = 0
    for ref in model.judged["outputs"]:
        program = model.program(ref["program"])
        selected.extend(frozen[offset : offset + counts[program.name]])
        offset += program.steps
    return selected


def cached_twin_readiness(bundle, model, counts, inputs) -> dict[str, Any]:
    """Structural readiness only; actual twin identity and exact bound inputs still require full checking."""
    if (model.extra.get("full") or {}).get("differential", "required") != "cached":
        return {"status": "required", "pending": [], "detail": "host twins run on a cache miss"}

    declared = (bundle.manifest().get("models") or {}).get(model.name, {}).get("twins") or {}
    pending = []
    for program in model.programs:
        if not bundle.catalog(model.name, program.name).interfaces:
            continue
        entry = declared.get(program.name) or {}
        twin = entry.get("twin") or {}
        expected_key = key(twin)
        index = bundle.root / model.name / "twins" / program.name / f"{expected_key}.json"
        try:
            record = json.loads(index.read_text())
        except (OSError, ValueError):
            record = {}
        valid = (
            entry.get("twin_key") == expected_key
            and record.get("schema") == REFERENCE_CACHE_SCHEMA
            and record.get("twin") == twin
        )
        for input_id in inputs:
            hashes = (entry.get("inputs") or {}).get(input_id) or []
            complete = valid and len(hashes) >= counts[program.name]
            if complete:
                for digest in hashes[: counts[program.name]]:
                    outputs = ((record.get("steps") or {}).get(digest) or {}).get("outputs")
                    if not isinstance(outputs, list) or not all(
                        isinstance(key, str)
                        and len(key) == 64
                        and all(c in "0123456789abcdef" for c in key)
                        and (bundle.root / model.name / "twins" / "blobs" / f"{key}.npy").is_file()
                        for key in outputs
                    ):
                        complete = False
                        break
            if not complete:
                pending.append(f"{program.name}:{input_id}")
    return {
        "status": "pending" if pending else "declared",
        "pending": pending,
        "detail": "missing cached host-twin coverage; full differential cannot pass"
        if pending
        else "declared coverage exists; built twin identity and actual step inputs still need validation",
    }
