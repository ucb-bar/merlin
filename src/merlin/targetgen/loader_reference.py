"""A float capsule's reference, captured from its OWN PyTorch loader when its corpus ships no golden.

The integer Tensor engine cannot recompute an f32/bf16 datapath, so a float capsule is graded against
an independent host-eager golden. A Phase 0 corpus writes one beside every capsule; a corpus without
goldens (the historical tree, a hand-staged cohort) left every float capsule that declares
``pytorch_ref.loader`` to crash with ``UnsupportedGoldenFormat`` before any tier ran.

Such a capsule already names the program that defines its answer: ``capsule.pytorch.py`` builds the
model and its deterministic inputs. This module runs that loader on the host (model2MLIR's PyTorch
worker, the same capture Phase 0 uses) and records the result the way Phase 0 does --
``golden_source: host_torch_eager`` with the captured inputs under ``oracle_provenance.inputs``, so
the device is fed exactly the operands the reference was computed on.

The record goes to an operator-side cache keyed by the bytes of ``capsule.yaml`` and the loader,
never into the corpus. ``MERLIN_LOADER_REFERENCE_DIR`` relocates it. Nothing here substitutes for a
golden the corpus does ship, and a capsule whose loader cannot be captured keeps the explicit refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

CACHE_ENV = "MERLIN_LOADER_REFERENCE_DIR"
SCHEMA = "merlin_loader_reference_v1"
_INTEGER_PREFIXES = ("i", "u")


def _is_float(dtype: Any) -> bool:
    token = str(dtype or "i8")
    return not (token.startswith(_INTEGER_PREFIXES) and token[1:].isdigit())


def _capsule(capsule_dir: Path) -> dict | None:
    path = capsule_dir / "capsule.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else None


def loader_of(capsule: dict, capsule_dir: Path) -> Path | None:
    """The capsule's declared PyTorch loader when this module may define its reference, else None.

    Only a single-output, non-model capsule with a floating operand and a loader file inside its own
    directory qualifies; every integer capsule keeps its recomputed golden.
    """
    loader = (capsule.get("pytorch_ref") or {}).get("loader")
    if (
        capsule.get("kind") == "model"
        or not isinstance(loader, str)
        or Path(loader).name != loader
        or not any(_is_float(leaf.get("dtype")) for leaf in capsule.get("inputs") or ())
        or not _output_name(capsule)
    ):
        return None
    path = capsule_dir / loader
    return path if path.is_file() and not path.is_symlink() else None


def _output_name(capsule: dict) -> str | None:
    out = ((capsule.get("operation") or {}).get("attributes") or {}).get("out")
    return out if isinstance(out, str) and out else None


def _cache_root() -> Path:
    configured = os.environ.get(CACHE_ENV, "").strip()
    if configured:
        return Path(configured)
    from merlin.common.paths import build_dir

    return build_dir() / "loader_references"


def _key(capsule_dir: Path, loader: Path) -> str:
    digest = hashlib.sha256(SCHEMA.encode())
    for path in (capsule_dir / "capsule.yaml", loader):
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _flat(value: Any) -> list:
    out: list = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _shape(value: Any) -> list[int]:
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else None
    return shape


def _document(capsule: dict, art: Any, loader: Path) -> dict:
    leaves = list(capsule.get("inputs") or ())
    if len(art.inputs) != len(leaves):
        raise ValueError(f"loader produced {len(art.inputs)} inputs for {len(leaves)} declared leaves")
    inputs = {}
    for leaf, value in zip(leaves, art.inputs, strict=True):
        if _shape(value) != list(leaf.get("shape") or []):
            raise ValueError(
                f"loader input for {leaf.get('name')!r} has shape {_shape(value)}, declared {leaf.get('shape')}"
            )
        inputs[str(leaf["name"])] = {"shape": _shape(value), "decoded": _flat(value)}
    policy = dict(capsule.get("numeric_policy") or {})
    return {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "engine": "PyTorch host eager reference of the capsule's own loader, captured at grade time",
            "note": "INDEPENDENT of the target RTL; the capsule's PyTorch loader + host eval are the reference.",
            "schema": SCHEMA,
            "grade_policy": {key: policy[key] for key in ("compare", "atol", "rtol") if key in policy},
            "pytorch_source": loader.name,
            "path_taken": (art.meta or {}).get("path_taken"),
            "inputs": inputs,
        },
        "outputs": {_output_name(capsule): art.golden},
    }


def captured_reference(capsule_dir: str | Path | None) -> dict | None:
    """The loader-captured golden document for ``capsule_dir``, capturing it once; None when N/A.

    None also when the PyTorch worker is unavailable: the caller's refusal then still names the
    missing independent golden instead of a capture failure it did not ask for.
    """
    if not capsule_dir:
        return None
    capsule_dir = Path(capsule_dir)
    capsule = _capsule(capsule_dir)
    loader = loader_of(capsule, capsule_dir) if capsule is not None else None
    if loader is None:
        return None
    from merlin.targetgen import golden_store

    directory = _cache_root() / _key(capsule_dir, loader)
    cached = golden_store.load_golden(directory)
    if cached is not None:
        return cached
    from merlin.targetgen.capsule_source import M2MUnavailable, PytorchRefSource

    source = PytorchRefSource()
    if not source.available():
        return None
    # Concurrent graders may capture the same capsule: each builds a private directory and the first
    # complete one is published by an atomic rename; a loser discards its copy and reads the winner's.
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        dtype = str((capsule.get("pytorch_ref") or {}).get("dtype") or "f32")
        try:
            art = source.capture_loader(loader, dtype, workdir=staging / "capture")
        except M2MUnavailable:
            return None
        golden_store.write_golden(staging, _document(capsule, art, loader))
        (staging / "capsule_identity.json").write_text(
            json.dumps({"schema": SCHEMA, "capsule_dir": str(capsule_dir.resolve()), "loader": loader.name}),
            encoding="utf-8",
        )
        try:
            os.rename(staging, directory)
        except OSError:
            if golden_store.load_golden(directory) is None:
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return golden_store.load_golden(directory)
