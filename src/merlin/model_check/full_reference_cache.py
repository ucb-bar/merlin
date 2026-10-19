"""Byte-bound per-step references and structural cached-twin readiness for full model checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "merlin.model_check.full_reference_cache.v1"


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
    core = importlib.util.find_spec("merlin.model_check")
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
    from merlin.llvmlower.toolchain import compiler_python

    printer = compiler_python()
    bindings = {}
    # A custom-form input may use the selected compiler environment's generic printer.
    # Its bindings include native MLIR libraries, so Python package versions alone are insufficient.
    for library in (printer.parent.parent / "lib", printer.parent.parent / "lib64"):
        for package in library.glob("python*/site-packages/torch_mlir"):
            bindings[str(package)] = tree_digest(package)
    return {
        "sources": {str(root): tree_digest(root, python_only=True) for root in roots},
        "generic_printer": {"python": str(printer),
                            "sha256": file_digest(printer) if printer.is_file() else None,
                            "bindings": bindings},
        "dependencies": dependencies,
        "python": {"version": sys.version, "machine": platform.machine(), "sha256": file_digest(Path(sys.executable))},
        "threads": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def material(model, bundle, arrays, counts, catalogs) -> dict[str, Any]:
    """All chain inputs and reference/weight/evaluator bytes, independently of a device submission."""
    return {
        "schema": SCHEMA,
        "model": asdict(model),
        "counts": dict(counts),
        "inputs_sha256": arrays_digest(arrays),
        "bundle_manifest_sha256": file_digest(bundle.root / "MANIFEST.json"),
        "bundle_sums_sha256": (
            file_digest(bundle.root / "SHA256SUMS") if (bundle.root / "SHA256SUMS").is_file() else None
        ),
        "programs": {
            name: {"catalog": catalog.record(), "stage_sha256": tree_digest(catalog.stage),
                   "reference_sha256": tree_digest(catalog.ref_prepared.parent)}
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
                row.get("schema") != SCHEMA or row.get("context_sha256") != self.key
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
                if expected != {"shape": list(value.shape), "dtype": value.dtype.str,
                                "sha256": arrays_digest({"output": value})}:
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
        row = {"schema": SCHEMA, "context_sha256": self.key, "identity": self.identity,
               "inputs_sha256": arrays_digest(inputs), "npz_sha256": file_digest(pending),
               "outputs": [{"shape": list(value.shape), "dtype": value.dtype.str,
                            "sha256": arrays_digest({"output": value})} for value in values]}
        receipt = path.with_name(path.stem + suffix + ".json")
        receipt.write_text(json.dumps(row, sort_keys=True) + "\n")
        pending.replace(path)
        receipt.replace(path.with_suffix(".json"))


def same_outputs(actual, expected) -> bool:
    """Require every declared output; a shorter list is never an exact reference match."""
    return len(actual) == len(expected) and all(
        np.asarray(a).dtype == np.asarray(b).dtype and np.asarray(a).shape == np.asarray(b).shape
        and np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)
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
        selected.extend(frozen[offset:offset + counts[program.name]])
        offset += program.steps
    return selected


def cached_twin_readiness(bundle, model, counts, inputs) -> dict[str, Any]:
    """Structural readiness only; actual twin identity and exact bound inputs still require full checking."""
    if (model.extra.get("full") or {}).get("differential", "required") != "cached":
        return {"status": "required", "pending": [], "detail": "host twins run on a cache miss"}
    from merlin.model_check import twins as TW

    declared = (bundle.manifest().get("models") or {}).get(model.name, {}).get("twins") or {}
    pending = []
    for program in model.programs:
        if not bundle.catalog(model.name, program.name).interfaces:
            continue
        entry = declared.get(program.name) or {}
        twin = entry.get("twin") or {}
        expected_key = TW.key(twin)
        index = bundle.root / model.name / "twins" / program.name / f"{expected_key}.json"
        try:
            record = json.loads(index.read_text())
        except (OSError, ValueError):
            record = {}
        valid = (
            entry.get("twin_key") == expected_key and record.get("schema") == TW.SCHEMA and record.get("twin") == twin
        )
        for input_id in inputs:
            hashes = (entry.get("inputs") or {}).get(input_id) or []
            complete = valid and len(hashes) >= counts[program.name]
            if complete:
                for digest in hashes[:counts[program.name]]:
                    outputs = ((record.get("steps") or {}).get(digest) or {}).get("outputs")
                    if not isinstance(outputs, list) or not all(
                        isinstance(key, str) and len(key) == 64 and all(c in "0123456789abcdef" for c in key)
                        and (bundle.root / model.name / "twins" / "blobs" / f"{key}.npy").is_file()
                        for key in outputs
                    ):
                        complete = False
                        break
            if not complete:
                pending.append(f"{program.name}:{input_id}")
    return {"status": "pending" if pending else "declared", "pending": pending,
            "detail": "missing cached host-twin coverage; full differential cannot pass" if pending
            else "declared coverage exists; built twin identity and actual step inputs still need validation"}
