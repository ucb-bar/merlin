"""Evaluation and calibration inputs: deterministic selection from pinned sources, small stored forms.

``sources.yaml`` (``examples/<target>/numerics/``) pins every source by revision and sha256 and states
each split's selection rule. :func:`prepare` applies the rule, runs the model's own preprocessing in its
pinned environment (:mod:`merlin.model_check.input_worker`), and writes:

- the stored form of every VISIBLE input to the public example (small: a uint8 crop or frame, a token list);
- the stored form of every SEALED input, and every input's model-ready arrays, to the private root;
- calibration items, privately, with their ids and hashes committed so the set is reproducible;
- ``MANIFEST.json`` beside ``sources.yaml``: the sha256 of every file, sealed ones included.

A public example never holds a holdout: sealed inputs are committed to by sha256 only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SOURCES_SCHEMA = "merlin.model_check.sources.v1"
MANIFEST_SCHEMA = "merlin.model_check.inputs.v1"
WORKER = Path(__file__).with_name("input_worker.py")


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Byte-stable compressed ``.npz`` (no timestamps), written by the same function the worker uses."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_model_check_input_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save_npz(str(path), **arrays)


#: The committed example stays small: every visible input, the manifests and the calibration ids.
PUBLIC_BUDGET = 10 << 20


class InputError(RuntimeError):
    """A pinned source is missing or differs, or a selection rule cannot be satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if doc.get("schema") != SOURCES_SCHEMA:
        raise InputError(f"{path}: not a {SOURCES_SCHEMA} document")
    return doc


def _verify(path: Path, sha256: str) -> None:
    if not Path(path).is_file():
        raise InputError(f"pinned source is absent: {path}")
    actual = sha256_file(path)
    if actual != sha256:
        raise InputError(f"{path}: sha256 {actual} != pinned {sha256}")


def run_worker(
    python: Path,
    task: str,
    request: dict[str, Any],
    scratch: Path,
    *,
    worker: Path = WORKER,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one worker task in ``python``'s environment (isolated mode, no Merlin)."""
    import os

    scratch.mkdir(parents=True, exist_ok=True)
    req, res = scratch / f"{task}.request.json", scratch / f"{task}.result.json"
    req.write_text(json.dumps({**request, "task": task, "result": str(res)}))
    done = subprocess.run(
        [str(python), "-I", str(worker), str(req)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )
    if done.returncode != 0:
        raise InputError(f"input worker {task} failed: {done.stderr[-2000:]}")
    return json.loads(res.read_text())


def _split_dirs(public: Path, private: Path, model: str, split: str, ident: str) -> tuple[Path, Path]:
    """Stored form (public for visible, private for sealed) and model-ready arrays (always private)."""
    stored = (public / "inputs" if split == "visible" else private / "stored") / model / split / ident
    derived = private / "derived" / model / split / ident
    stored.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    return stored, derived


# ----------------------------------------------------------------------------------------- resnet


def _imagenette_members(archive: Path) -> dict[tuple[str, str], list[str]]:
    by_class: dict[tuple[str, str], list[str]] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if member.isfile() and len(parts) == 4 and parts[3].lower().endswith(".jpeg"):
                by_class.setdefault((parts[1], parts[2]), []).append(member.name)
    for names in by_class.values():
        names.sort()
    return by_class


def _prepare_resnet(spec, data_root, public, private, python, scratch) -> dict[str, Any]:
    archive = Path(data_root) / Path(spec["source"]["url"]).name
    _verify(archive, spec["source"]["sha256"])
    by_class = _imagenette_members(archive)
    synsets = sorted(spec["classes"])
    plan: list[dict[str, Any]] = []
    for split in ("visible", "sealed"):
        rule = spec[split]
        lo, hi = rule["classes"]
        for index, synset in enumerate(synsets[lo:hi]):
            names = by_class[(rule["split"], synset)]
            plan.append(
                {
                    "split": split,
                    "id": f"{split[0]}{index:02d}",
                    "member": names[int(rule["rank"])],
                    "label": int(spec["classes"][synset]),
                    "synset": synset,
                }
            )
    calibration = []
    for synset in synsets:
        calibration += by_class[(spec["calibration"]["split"], synset)][: int(spec["calibration"]["per_class"])]
    with tarfile.open(archive) as tar:
        images = []
        for row in plan:
            stored, derived = _split_dirs(public, private, "resnet50", row["split"], row["id"])
            source = scratch / "resnet" / Path(row["member"]).name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(tar.extractfile(row["member"]).read())
            row["source_sha256"] = sha256_file(source)
            images.append(
                {
                    "id": row["id"],
                    "path": str(source),
                    "stored": str(stored / "image.npz"),
                    "inputs": str(derived / "inputs.npz"),
                }
            )
            row["stored"], row["derived"] = stored, derived
        calibration_rows = [
            {"member": name, "sha256": hashlib.sha256(tar.extractfile(name).read()).hexdigest()} for name in calibration
        ]
    result = run_worker(python, "resnet_crops", {"images": images}, scratch / "resnet")
    transform = result["transform"]
    for row in plan:
        (row["stored"] / "meta.json").write_text(
            json.dumps(
                {
                    "member": row["member"],
                    "synset": row["synset"],
                    "imagenet_label": row["label"],
                    "source_sha256": row["source_sha256"],
                },
                indent=1,
            )
            + "\n"
        )
    return {"transform": transform, "inputs": plan, "calibration": calibration_rows}


# ------------------------------------------------------------------------------------- tiny_llama


def _articles(rows: list[str]) -> list[str]:
    """Article bodies: text between consecutive top-level `` = Title = `` headings, in file order."""
    articles: list[list[str]] = []
    for row in rows:
        stripped = row.strip()
        top_heading = stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= =")
        if top_heading:
            articles.append([])
        elif articles:
            articles[-1].append(row)
    return ["".join(lines).strip() for lines in articles if "".join(lines).strip()]


def _prepare_tiny_llama(spec, data_root, public, private, python, scratch, *, rows_python) -> dict[str, Any]:
    tokenizer = spec["tokenizer_path"]
    count = int(spec["segment_tokens"])
    splits: dict[str, list[list[int]]] = {}
    for split in sorted({spec["visible"]["split"], spec["sealed"]["split"], spec["calibration"]["split"]}):
        pinned = spec["source"]["files"][split]
        path = Path(data_root) / pinned["path"]
        _verify(path, pinned["sha256"])
        out = scratch / "tiny" / f"{split}.rows.json"
        run_worker(rows_python, "parquet_rows", {"path": str(path), "out": str(out)}, scratch / "tiny")
        articles = _articles(json.loads(out.read_text()))
        tokens_out = scratch / "tiny" / f"{split}.tokens.json"
        run_worker(
            python,
            "tokenize_segments",
            {"tokenizer": tokenizer, "texts": articles, "out": str(tokens_out)},
            scratch / "tiny",
        )
        splits[split] = [ids[:count] for ids in json.loads(tokens_out.read_text()) if len(ids) >= count]
    plan = []
    for split in ("visible", "sealed"):
        rule = spec[split]
        lo, hi = rule["articles"]
        for index, ids in enumerate(splits[rule["split"]][lo:hi]):
            ident = f"{split[0]}{index:02d}"
            stored, derived = _split_dirs(public, private, "tiny_llama", split, ident)
            (stored / "tokens.json").write_text(
                json.dumps({"split": rule["split"], "article": lo + index, "tokens": ids}) + "\n"
            )
            _save_npz(derived / "inputs.npz", tokens=np.asarray(ids, np.int64))
            plan.append({"split": split, "id": ident, "article": lo + index})
    lo, hi = spec["calibration"]["articles"]
    calibration = [
        {"article": i, "sha256": hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()}
        for i, ids in enumerate(splits[spec["calibration"]["split"]][lo:hi], start=lo)
    ]
    return {"inputs": plan, "calibration": calibration}


# ---------------------------------------------------------------------------------------- smolvla


def _prepare_smolvla(spec, data_root, public, private, python, scratch) -> dict[str, Any]:
    items, plan = [], []
    for split in ("visible", "sealed"):
        rule = spec[split]
        lo, hi = rule["episodes"]
        for index, episode in enumerate(range(lo, hi)):
            ident = f"{split[0]}{index:02d}"
            stored, derived = _split_dirs(public, private, "smolvla", split, ident)
            seed = int(rule["noise_seed_base"]) + index
            items.append(
                {
                    "id": ident,
                    "episode": episode,
                    "frame": int(rule["frame"]),
                    "noise_seed": seed,
                    "stored": str(stored / "frame.npz"),
                    "arrays": str(derived / "observation.npz"),
                }
            )
            plan.append(
                {
                    "split": split,
                    "id": ident,
                    "episode": episode,
                    "frame": int(rule["frame"]),
                    "noise_seed": seed,
                    "stored": stored,
                }
            )
    calibration_items = []
    lo, hi = spec["calibration"]["episodes"]
    cal_dir = Path(private) / "calibration" / "smolvla"
    cal_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(lo, hi):
        for frame in spec["calibration"]["frames"]:
            ident = f"c{episode:02d}_{frame:03d}"
            calibration_items.append(
                {
                    "id": ident,
                    "episode": episode,
                    "frame": int(frame),
                    "noise_seed": int(spec["calibration"]["noise_seed_base"]) + len(calibration_items),
                    "stored": str(cal_dir / f"{ident}.frame.npz"),
                    "arrays": str(cal_dir / f"{ident}.npz"),
                }
            )
    request = {
        "dataset": spec["source"]["dataset"],
        "revision": spec["source"]["revision"],
        "dataset_root": spec["source"]["local_root"],
        "policy_snapshot": spec["policy_path"],
        "vlm_snapshot": spec["vlm_path"],
        "items": items + calibration_items,
    }
    result = run_worker(python, "smolvla_frames", request, scratch / "smolvla")
    for row in plan:
        meta = result["items"][row["id"]]
        (row["stored"] / "observation.json").write_text(
            json.dumps(
                {
                    "episode": row["episode"],
                    "frame": row["frame"],
                    "noise_seed": row["noise_seed"],
                    "task": meta["task"],
                    "raw_state": meta["raw_state"],
                },
                indent=1,
            )
            + "\n"
        )
    calibration = [
        {"id": c["id"], "episode": c["episode"], "frame": c["frame"], "sha256": sha256_file(Path(c["arrays"]))}
        for c in calibration_items
    ]
    return {"inputs": [{k: v for k, v in r.items() if k != "stored"} for r in plan], "calibration": calibration}


PREPARERS: dict[str, Callable[..., dict[str, Any]]] = {
    "resnet50": _prepare_resnet,
    "tiny_llama": _prepare_tiny_llama,
    "smolvla": _prepare_smolvla,
}


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def prepare(
    model: str,
    sources_path: Path,
    *,
    data_root: Path,
    public: Path,
    private: Path,
    python: Path,
    scratch: Path,
    **extra: Any,
) -> dict[str, Any]:
    """Select, preprocess and store ``model``'s inputs; update both manifests."""
    sources = load_sources(sources_path)
    spec = dict(sources["models"][model])
    spec.update({k: v for k, v in extra.items() if k.endswith("_path")})
    kwargs = {k: v for k, v in extra.items() if not k.endswith("_path")}
    record = PREPARERS[model](spec, data_root, public, private, python, scratch, **kwargs)
    visible_root = public / "inputs" / model / "visible"
    sealed_root = private / "stored" / model / "sealed"
    derived_root = private / "derived" / model
    entry = {
        "visible": _tree_hashes(visible_root) if visible_root.is_dir() else {},
        "sealed": _tree_hashes(sealed_root) if sealed_root.is_dir() else {},
        "derived": _tree_hashes(derived_root) if derived_root.is_dir() else {},
        "selection": [{k: v for k, v in row.items() if k not in ("stored", "derived")} for row in record["inputs"]],
    }
    calibration_path = public / "calibration" / f"{model}.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps({"model": model, "items": record["calibration"]}, indent=1) + "\n")
    manifest_path = public / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"schema": MANIFEST_SCHEMA}
    manifest.setdefault("models", {})[model] = {
        "sources_sha256": sha256_file(sources_path),
        "calibration_sha256": sha256_file(calibration_path),
        **{k: v for k, v in entry.items() if k != "derived"},
        "derived": entry["derived"],
        **({"transform": record["transform"]} if "transform" in record else {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest["models"][model]


def stored_bytes(public: Path) -> int:
    return sum(p.stat().st_size for p in Path(public).rglob("*") if p.is_file())


def read_model_inputs(private: Path, model: str, split: str, ident: str) -> dict[str, np.ndarray]:
    path = Path(private) / "derived" / model / split / ident
    for name in ("inputs.npz", "observation.npz"):
        if (path / name).is_file():
            with np.load(path / name) as data:
                return {k: np.asarray(data[k]) for k in data.files}
    raise InputError(f"no derived inputs at {path}")
