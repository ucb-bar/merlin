#!/usr/bin/env python3
"""Prepare and verify attributed inputs for the frozen K1 paper holdouts.

This tool never invents a paper input.  Callers must supply an attribution label and immutable
source bytes for each corpus.  ``audit`` and ``prepare --dry-run`` are read-only; ``prepare`` writes a
versioned ``paper-inputs`` product under the configured Merlin output root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

import numpy as np

from merlin.common.artifacts import cache_dir, new_product
from merlin.common.paths import bench_dir
from merlin.common.yaml import load_yaml


SPEC = bench_dir() / "rvv_paper" / "input_sources_v1.yaml"
ACTIVE_MODELS = ("gemma2_2b", "tinyllama_1_1b", "smolvla", "resnet50_v1_5", "lstmnetvit")
TOKEN_MODELS = {"gemma2_2b": "M2M_GEMMA", "tinyllama_1_1b": "M2M_LLAMA"}
SMOLVLA_NOISE_SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_repo_dir(cache: Path, repo_id: str) -> Path:
    return cache / ("models--" + repo_id.replace("/", "--"))


def _tree_record(snapshot: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        rel = path.relative_to(snapshot).as_posix()
        file_digest = sha256_file(path)
        size = path.stat().st_size
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(file_digest.encode("ascii") + b"\0")
        records.append({"path": rel, "bytes": size, "sha256": file_digest})
    return digest.hexdigest(), records


def audit_hf_component(cache: Path, component: dict[str, Any]) -> dict[str, Any]:
    repo_id = str(component["repo_id"])
    expected = str(component["revision"])
    root = _hf_repo_dir(cache, repo_id)
    ref_path = root / "refs" / "main"
    actual_ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None
    snapshot = root / "snapshots" / expected
    missing = [name for name in component["required_files"] if not (snapshot / name).is_file()]
    blockers = []
    if not snapshot.is_dir():
        blockers.append(f"pinned HF snapshot is absent: {snapshot}")
    elif missing:
        blockers.append(f"pinned HF snapshot omits {missing}")
    tree_digest, files = _tree_record(snapshot) if snapshot.is_dir() and not missing else (None, [])
    return {
        "kind": "huggingface_snapshot", "repo_id": repo_id, "revision": expected,
        "cache_path": str(snapshot), "observed_main_ref": actual_ref,
        "tree_sha256": tree_digest, "files": files,
        "ready": not blockers, "blockers": blockers,
    }


def audit_checkpoint(model: str, spec: dict[str, Any], *, hf_cache: Path,
                     torch_cache: Path, vitfly_dir: Path, vitfly_checkpoint: Path | None) -> dict[str, Any]:
    cfg = spec["checkpoints"][model]
    kind = cfg["kind"]
    if kind == "huggingface_snapshot":
        return audit_hf_component(hf_cache, cfg)
    if kind == "huggingface_composite":
        components = [audit_hf_component(hf_cache, item) for item in cfg["components"]]
        blockers = [f"{item['repo_id']}: {value}"
                    for item in components for value in item["blockers"]]
        return {"kind": kind, "components": components, "ready": not blockers,
                "blockers": blockers}
    if kind == "torch_hub_file":
        path = torch_cache / str(cfg["filename"])
        actual = sha256_file(path) if path.is_file() else None
        blockers = []
        if not path.is_file():
            blockers.append(f"checkpoint is absent: {path}")
        elif actual != cfg["sha256"]:
            blockers.append(f"checkpoint SHA-256 is {actual}, expected {cfg['sha256']}")
        return {"kind": kind, "path": str(path), "sha256": actual, "url": cfg["url"],
                "weights_enum": cfg["weights_enum"], "ready": not blockers,
                "blockers": blockers}
    if kind == "external_file_and_git":
        blockers = []
        source_file = vitfly_dir / str(cfg["source_file"])
        revision = None
        if (vitfly_dir / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(vitfly_dir), "rev-parse", "HEAD"], capture_output=True,
                text=True, check=False)
            revision = result.stdout.strip() or None
        if revision != cfg["source_revision"]:
            blockers.append(f"VitFly source revision is {revision or 'unavailable'}, expected "
                            f"{cfg['source_revision']}")
        if not source_file.is_file():
            blockers.append(f"VitFly source file is absent: {source_file}")
        checkpoint_digest = None
        checkpoint_validation = None
        if vitfly_checkpoint is None:
            blockers.append("published LSTMNetVIT checkpoint was not supplied with --vitfly-checkpoint")
        elif not vitfly_checkpoint.is_file():
            blockers.append(f"published LSTMNetVIT checkpoint is absent: {vitfly_checkpoint}")
        else:
            checkpoint_digest = sha256_file(vitfly_checkpoint)
            try:
                checkpoint_validation = validate_vitfly_checkpoint(vitfly_dir, vitfly_checkpoint)
            except Exception as error:
                blockers.append(f"published LSTMNetVIT checkpoint fails strict validation: {error}")
        return {
            "kind": kind, "source_repo": cfg["source_repo"], "source_revision": revision,
            "source_path": str(vitfly_dir), "source_file": str(cfg["source_file"]),
            "source_file_sha256": sha256_file(source_file) if source_file.is_file() else None,
            "checkpoint_path": str(vitfly_checkpoint) if vitfly_checkpoint else None,
            "checkpoint_sha256": checkpoint_digest, "checkpoint_url": cfg["checkpoint_url"],
            "checkpoint_validation": checkpoint_validation,
            "ready": not blockers, "blockers": blockers,
        }
    raise ValueError(f"unsupported checkpoint kind {kind!r}")


def _source_required(path: Path | None, label: str | None, description: str) -> list[str]:
    blockers = []
    if path is None:
        blockers.append(f"{description} was not supplied")
    elif not path.is_file():
        blockers.append(f"{description} is absent: {path}")
    if not label:
        blockers.append(f"{description} attribution label was not supplied")
    return blockers


def validate_vitfly_checkpoint(vitfly_dir: Path, checkpoint: Path) -> dict[str, Any]:
    """Load the published state dict into the pinned upstream architecture with ``strict=True``."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "strict VitFly checkpoint validation requires PyTorch; run with the Model2MLIR venv") \
            from error
    model_dir = vitfly_dir / "models"
    model_path = model_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"VitFly model definition is absent: {model_path}")
    old_path = list(sys.path)
    try:
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        module_spec = importlib.util.spec_from_file_location("rvv_paper_vitfly_model", model_path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"cannot import VitFly model definition: {model_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        network = module.LSTMNetVIT()
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        if not isinstance(state, dict):
            raise ValueError("VitFly checkpoint is not a state_dict or a state_dict container")
        network.load_state_dict(state, strict=True)
        parameters = sum(int(value.numel()) for value in network.state_dict().values())
        return {"strict_load": True, "state_entries": len(state), "state_values": parameters}
    finally:
        sys.path[:] = old_path


def validate_smolvla_npz(path: Path) -> dict[str, Any]:
    expected = {
        "image": ((1, 3, 512, 512), "float"), "image_mask": ((1,), "bool"),
        "language_tokens": ((1, 48), "integer"), "language_mask": ((1, 48), "bool"),
        "state": ((1, 32), "float"), "noise": ((1, 50, 32), "float"),
    }
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(expected) - set(data.files))
        if missing:
            raise ValueError(f"SmolVLA input omits {missing}")
        unexpected = sorted(set(data.files) - set(expected))
        if unexpected:
            raise ValueError(f"SmolVLA input has unexpected arrays {unexpected}")
        for name, (shape, kind) in expected.items():
            value = np.asarray(data[name])
            if value.shape != shape:
                raise ValueError(f"SmolVLA {name} has shape {value.shape}, expected {shape}")
            if kind == "bool" and value.dtype.kind != "b":
                raise ValueError(f"SmolVLA {name} must be boolean, got {value.dtype}")
            if kind == "integer" and value.dtype != np.dtype(np.int64):
                raise ValueError(f"SmolVLA {name} must be int64, got {value.dtype}")
            if kind == "float" and value.dtype != np.dtype(np.float32):
                raise ValueError(f"SmolVLA {name} must be float32, got {value.dtype}")
            if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError(f"SmolVLA {name} contains non-finite values")
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return arrays


def validate_vitfly_session(path: Path, steps: int = 256) -> dict[str, Any]:
    expected = {"frames": (steps, 1, 1, 60, 90),
                "desired_velocity": (steps, 1, 1), "quaternions": (steps, 1, 4)}
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(expected) - set(data.files))
        if missing:
            raise ValueError(f"VitFly trajectory omits {missing}")
        unexpected = sorted(set(data.files) - set(expected))
        if unexpected:
            raise ValueError(f"VitFly trajectory has unexpected arrays {unexpected}")
        for name, shape in expected.items():
            value = np.asarray(data[name])
            if value.shape != shape:
                raise ValueError(f"VitFly {name} has shape {value.shape}, expected {shape}")
            if value.dtype != np.dtype(np.float32) or not np.all(np.isfinite(value)):
                raise ValueError(f"VitFly {name} must be finite float32")
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        norms = np.linalg.norm(np.asarray(data["quaternions"]), axis=-1)
        if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=1.0e-4):
            raise ValueError("VitFly quaternions must already be unit-normalized")
    return arrays


def _cache_verified_bytes(path: Path, payload: bytes) -> None:
    """Persist immutable source bytes without silently replacing a different cached object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(payload).hexdigest()
    if path.is_file():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"cached source bytes changed for {path}: {actual} != {expected}")
        return
    path.write_bytes(payload)


def extract_vitfly_trajectory(
        archive: zipfile.ZipFile, output: Path, *, observations: int = 256,
        archive_provenance: dict[str, Any] | None = None,
        raw_output_dir: Path | None = None) -> dict[str, Any]:
    """Range-friendly deterministic extraction of the paper's VitFly trajectory.

    ``archive`` may wrap a local file or a seekable HTTP range reader. Only the selected CSV and
    image members are read: the lexicographically first trajectory containing at least
    ``observations`` PNG frames, followed by its first timestamps in filename order. Image and
    telemetry preprocessing deliberately mirrors ``training/dataloading.py`` from the pinned
    upstream revision.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("VitFly trajectory extraction requires OpenCV") from error

    directories: dict[str, dict[str, Any]] = {}
    for info in archive.infolist():
        if info.is_dir() or "/" not in info.filename:
            continue
        directory, name = info.filename.rsplit("/", 1)
        detail = directories.setdefault(directory, {"images": [], "csv": None})
        if name.lower().endswith(".png"):
            detail["images"].append(info)
        elif name == "data.csv":
            detail["csv"] = info
    candidates = sorted(
        name for name, detail in directories.items()
        if detail["csv"] is not None and len(detail["images"]) >= observations)
    if not candidates:
        raise ValueError(f"VitFly archive has no trajectory with at least {observations} frames")

    rejected_candidates: list[dict[str, Any]] = []
    selected = None
    for trajectory in candidates:
        detail = directories[trajectory]
        image_infos = sorted(detail["images"], key=lambda value: value.filename)
        csv_info = detail["csv"]
        csv_bytes = archive.read(csv_info)
        metadata = np.genfromtxt(io.BytesIO(csv_bytes), delimiter=",", dtype=np.float64)[1:]
        if metadata.ndim != 2 or metadata.shape[1] < 7:
            rejected_candidates.append({
                "trajectory": trajectory, "reason": "invalid_telemetry_shape",
                "telemetry_shape": list(metadata.shape), "image_count": len(image_infos),
            })
            continue
        # This is the exact mismatch rule used by upstream training/dataloading.py. It drops one
        # trailing image only when that image is newer than the final telemetry row, and rejects
        # the directory if the counts still differ.
        original_image_count = len(image_infos)
        if len(image_infos) != metadata.shape[0] and metadata.shape[0] > 0:
            last_timestamp = float(Path(image_infos[-1].filename).stem)
            if last_timestamp > metadata[-1, 1]:
                image_infos = image_infos[:-1]
        if len(image_infos) != metadata.shape[0] or len(image_infos) < observations:
            rejected_candidates.append({
                "trajectory": trajectory, "reason": "upstream_image_telemetry_count_mismatch",
                "image_count": original_image_count,
                "upstream_adjusted_image_count": len(image_infos),
                "telemetry_rows": int(metadata.shape[0]),
            })
            continue
        timestamps_match = all(
            np.isclose(float(Path(info.filename).stem), metadata[index, 1],
                       atol=5.0e-4, rtol=0.0)
            for index, info in enumerate(image_infos[:observations]))
        if not timestamps_match:
            rejected_candidates.append({
                "trajectory": trajectory, "reason": "selected_timestamp_mismatch",
                "image_count": original_image_count, "telemetry_rows": int(metadata.shape[0]),
            })
            continue
        selected = (trajectory, detail, image_infos[:observations], csv_info, csv_bytes, metadata)
        break
    if selected is None:
        raise ValueError(
            f"VitFly archive has no upstream-valid, timestamp-aligned trajectory with "
            f"{observations} frames")
    trajectory, detail, image_infos, csv_info, csv_bytes, metadata = selected
    metadata = metadata[:observations]
    if not np.all(np.isfinite(metadata[:, 1:7])):
        raise ValueError("VitFly selected timestamps, desired velocities, or quaternions are not finite")

    frames = np.empty((observations, 1, 1, 60, 90), dtype=np.float32)
    members: list[dict[str, Any]] = []
    if raw_output_dir is not None:
        _cache_verified_bytes(raw_output_dir / "data.csv", csv_bytes)
    for index, info in enumerate(image_infos):
        payload = archive.read(info)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not decode VitFly image {info.filename}")
        if image.shape != (60, 90):
            image = cv2.resize(image, (90, 60))
        frames[index, 0, 0] = image.astype(np.float32) / np.float32(255.0)
        timestamp = float(Path(info.filename).stem)
        if not np.isclose(timestamp, metadata[index, 1], atol=5.0e-4, rtol=0.0):
            raise ValueError(
                f"VitFly image/telemetry timestamp mismatch at {index}: "
                f"{timestamp} != {metadata[index, 1]}")
        if raw_output_dir is not None:
            _cache_verified_bytes(raw_output_dir / Path(info.filename).name, payload)
        members.append({
            "path": info.filename, "bytes": len(payload),
            "compressed_bytes": int(info.compress_size), "crc32": f"{info.CRC:08x}",
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    desired_velocity = metadata[:, 2].astype(np.float32).reshape(observations, 1, 1)
    quaternions = metadata[:, 3:7].astype(np.float32).reshape(observations, 1, 4)
    norms = np.linalg.norm(quaternions, axis=-1)
    if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=1.0e-4):
        raise ValueError("VitFly source quaternions are not unit-normalized")

    arrays = {"frames": frames, "desired_velocity": desired_velocity,
              "quaternions": quaternions}
    _write_deterministic_npz(output, arrays)
    record = {
        "version": 1, "archive": dict(archive_provenance or {}),
        "selection": "lexicographically_first_trajectory_with_at_least_256_frames_then_first_256_timestamps",
        "trajectory": trajectory, "trajectory_frames": len(detail["images"]),
        "eligibility": "upstream dataloader image/telemetry count rule plus timestamp alignment",
        "rejected_candidates": rejected_candidates,
        "observations": observations,
        "telemetry": {
            "path": csv_info.filename, "bytes": len(csv_bytes),
            "compressed_bytes": int(csv_info.compress_size), "crc32": f"{csv_info.CRC:08x}",
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        },
        "members": members,
        "preprocessing": {
            "images": "cv2.IMREAD_GRAYSCALE; cv2.resize((90,60)); float32 / 255.0",
            "desired_velocity_column": 2, "quaternion_columns": [3, 4, 5, 6],
            "telemetry_dtype": "float64 before selected float32 fields",
        },
        "arrays": validate_vitfly_session(output, observations),
        "output_path": str(output), "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "raw_output_dir": str(raw_output_dir) if raw_output_dir is not None else None,
    }
    provenance = output.with_name(output.stem + ".provenance.json")
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["provenance_path"] = str(provenance)
    return record


def parse_image_manifest(path: Path, observations: int = 256) -> list[dict[str, Any]]:
    entries = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"image manifest line {number} must be PATH<TAB>SHA256")
        source = Path(fields[0])
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        expected_digest = fields[1].strip().lower()
        if len(expected_digest) != 64:
            raise ValueError(f"image manifest line {number} has an invalid SHA-256")
        if not source.is_file():
            raise FileNotFoundError(f"image manifest source is absent: {source}")
        actual = sha256_file(source)
        if actual != expected_digest:
            raise ValueError(f"image digest mismatch for {source}: {actual} != {expected_digest}")
        entries.append({"path": str(source), "sha256": actual})
    if len(entries) != observations:
        raise ValueError(f"image manifest has {len(entries)} images, expected {observations}")
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("image manifest contains duplicate paths")
    return entries


def _input_audit(model: str, args: argparse.Namespace) -> dict[str, Any]:
    if model in TOKEN_MODELS:
        blockers = _source_required(args.language_corpus, args.language_source, "language corpus")
        return {"kind": "token_ids", "source_path": str(args.language_corpus)
                if args.language_corpus else None, "source_label": args.language_source,
                "ready": not blockers, "blockers": blockers}
    if model == "smolvla":
        blockers = _source_required(args.smolvla_input_npz, args.smolvla_input_source,
                                    "SmolVLA preprocessed input")
        arrays = None
        if not blockers:
            try:
                arrays = validate_smolvla_npz(args.smolvla_input_npz)
            except (OSError, ValueError) as error:
                blockers.append(str(error))
        return {"kind": "attributed_preprocessed_npz", "source_path":
                str(args.smolvla_input_npz) if args.smolvla_input_npz else None,
                "source_label": args.smolvla_input_source, "arrays": arrays,
                "ready": not blockers, "blockers": blockers}
    if model == "resnet50_v1_5":
        blockers = _source_required(args.resnet_images_list, args.resnet_input_source,
                                    "ResNet ordered image manifest")
        entries = None
        if not blockers:
            try:
                entries = parse_image_manifest(args.resnet_images_list)
            except (OSError, ValueError) as error:
                blockers.append(str(error))
        return {"kind": "ordered_image_manifest", "source_path":
                str(args.resnet_images_list) if args.resnet_images_list else None,
                "source_label": args.resnet_input_source, "entries": entries,
                "ready": not blockers, "blockers": blockers}
    if model == "lstmnetvit":
        blockers = _source_required(args.vitfly_session_npz, args.vitfly_session_source,
                                    "VitFly trajectory")
        arrays = None
        if not blockers:
            try:
                arrays = validate_vitfly_session(args.vitfly_session_npz)
            except (OSError, ValueError) as error:
                blockers.append(str(error))
        return {"kind": "attributed_trajectory_npz", "source_path":
                str(args.vitfly_session_npz) if args.vitfly_session_npz else None,
                "source_label": args.vitfly_session_source, "arrays": arrays,
                "ready": not blockers, "blockers": blockers}
    raise ValueError(f"unknown paper model {model}")


def audit(spec: dict[str, Any], models: Iterable[str], args: argparse.Namespace) -> dict[str, Any]:
    reports = {}
    for model in models:
        checkpoint = audit_checkpoint(
            model, spec, hf_cache=args.hf_cache, torch_cache=args.torch_cache,
            vitfly_dir=args.vitfly_dir, vitfly_checkpoint=args.vitfly_checkpoint)
        inputs = _input_audit(model, args)
        reports[model] = {"checkpoint": checkpoint, "input": inputs,
                          "ready": checkpoint["ready"] and inputs["ready"],
                          "blockers": checkpoint["blockers"] + inputs["blockers"]}
    return {"version": spec["version"], "models": reports,
            "ready": all(value["ready"] for value in reports.values())}


def _tokenize(snapshot: Path, corpus: Path, count: int = 160) -> np.ndarray:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("token materialization requires transformers") from error
    text = corpus.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokens = tokenizer(
        text, add_special_tokens=False, return_attention_mask=False,
        truncation=True, max_length=count)["input_ids"]
    if len(tokens) < count:
        raise ValueError(f"language corpus produces {len(tokens)} tokens, needs at least {count}")
    return np.asarray(tokens[:count], dtype=np.int64)


def _parquet_text_rows(path: Path) -> tuple[list[str], str]:
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "WikiText parquet extraction requires pyarrow; run with the Model2MLIR venv") from error
    table = parquet.read_table(path, columns=["text"])
    if table.column_names != ["text"]:
        raise ValueError(f"WikiText parquet schema is {table.column_names}, expected ['text']")
    rows = table.column("text").combine_chunks().to_pylist()
    if any(not isinstance(value, str) for value in rows):
        raise ValueError("WikiText parquet contains a null or non-string text row")
    return rows, str(pyarrow.__version__)


def extract_language_parquet(source: Path, config: dict[str, Any], *,
                             output: Path | None = None, row_reader=None) -> dict[str, Any]:
    """Verify the pinned WikiText parquet and deterministically join its stored text rows."""
    if not source.is_file():
        raise FileNotFoundError(f"pinned language parquet is absent: {source}")
    actual = sha256_file(source)
    expected = str(config["file_sha256"])
    if actual != expected:
        raise ValueError(f"language parquet SHA-256 is {actual}, expected {expected}")
    reader = row_reader or _parquet_text_rows
    result = reader(source)
    if (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list)
            and isinstance(result[1], str)):
        rows, pyarrow_version = result
    else:
        rows, pyarrow_version = result, None
    if any(not isinstance(value, str) for value in rows):
        raise ValueError("language parquet reader returned a non-string row")
    destination = output
    if destination is None:
        destination = cache_dir(
            f"paper-inputs/{config['config']}_{config['revision']}") / "corpus.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes("\n".join(rows).encode("utf-8"))
    source_label = (f"{config['dataset']}@{config['revision']}:"
                    f"{config['config']}/{config['split']}")
    record = {
        "version": 1, "dataset": config["dataset"], "revision": config["revision"],
        "config": config["config"], "split": config["split"], "file": config["file"],
        "source_path": str(source), "source_bytes": source.stat().st_size,
        "source_sha256": actual, "rows": len(rows), "extraction": config["extraction"],
        "encoding": "utf-8", "output_path": str(destination),
        "output_bytes": destination.stat().st_size, "output_sha256": sha256_file(destination),
        "pyarrow_version": pyarrow_version, "source_label": source_label,
        "suggested_prepare_args": ["--language-corpus", str(destination),
                                   "--language-source", source_label],
    }
    provenance = destination.with_name("corpus.provenance.json")
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["provenance_path"] = str(provenance)
    return record


def extract_resnet_images(source: Path, config: dict[str, Any], *,
                          output_dir: Path | None = None) -> dict[str, Any]:
    """Verify an attributed image ZIP and materialize the frozen ordered ResNet stream.

    Only the selected members are extracted.  Their bytes, ZIP CRCs, and the source archive are
    hashed, so a later paper-input build never depends on directory enumeration order or mutable
    dataset tooling.
    """
    if not source.is_file():
        raise FileNotFoundError(f"ResNet image archive is absent: {source}")
    expected = str(config.get("archive_sha256", ""))
    actual = sha256_file(source)
    if len(expected) != 64 or actual != expected:
        raise ValueError(f"ResNet archive SHA-256 is {actual}, expected {expected or 'a pinned digest'}")
    observations = int(config.get("observations", 256))
    prefix = str(config.get("member_prefix", ""))
    if not prefix or not prefix.endswith("/"):
        raise ValueError("ResNet source member_prefix must be a non-empty directory prefix")
    destination = output_dir or cache_dir(
        f"paper-inputs/{config['dataset']}_{actual[:16]}")
    destination = Path(destination).resolve()
    selected_dir = destination / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(source) as archive:
        candidates = []
        for info in archive.infolist():
            member = info.filename
            parts = Path(member).parts
            if (info.is_dir() or not member.startswith(prefix)
                    or Path(member).suffix.lower() not in {".jpg", ".jpeg"}):
                continue
            if Path(member).is_absolute() or ".." in parts or len(parts) != 2:
                raise ValueError(f"unsafe/unexpected ResNet archive member: {member}")
            candidates.append(info)
        candidates.sort(key=lambda info: info.filename)
        if len(candidates) < observations:
            raise ValueError(
                f"ResNet archive has {len(candidates)} eligible images, needs {observations}")
        for info in candidates[:observations]:
            payload = archive.read(info)
            path = selected_dir / Path(info.filename).name
            path.write_bytes(payload)
            rows.append({"member": info.filename, "bytes": len(payload), "crc32": info.CRC,
                         "path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("selected ResNet archive members have colliding basenames")
    manifest = destination / "images.tsv"
    manifest.write_text(
        "".join(f"{row['path']}\t{row['sha256']}\n" for row in rows), encoding="utf-8")
    record = {
        "version": 1, "dataset": config["dataset"], "url": config["url"],
        "source_path": str(source), "source_bytes": source.stat().st_size,
        "source_sha256": actual, "selection": config["selection"],
        "member_prefix": prefix, "observations": observations, "members": rows,
        "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
        "source_label": f"{config['dataset']}@sha256:{actual}",
        "suggested_prepare_args": ["--resnet-images-list", str(manifest),
                                   "--resnet-input-source",
                                   f"{config['dataset']}@sha256:{actual}"],
    }
    provenance = destination / "images.provenance.json"
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["provenance_path"] = str(provenance)
    return record


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NumPy arrays as a byte-stable ZIP (np.savez embeds wall-clock timestamps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_smolvla_observation(
        output: Path, arrays: dict[str, np.ndarray], provenance: dict[str, Any]) -> dict[str, Any]:
    """Write one byte-stable SmolVLA observation and its non-circular provenance record."""
    normalized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    _write_deterministic_npz(output, normalized)
    shapes = validate_smolvla_npz(output)
    record = {
        "version": 1,
        "kind": "smolvla_attributed_preprocessed_observation",
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "arrays": {
            name: {**shapes[name], "sha256": _array_sha256(normalized[name])}
            for name in sorted(normalized)
        },
        "provenance": provenance,
    }
    provenance_path = output.with_name(f"{output.stem}.provenance.json")
    provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["provenance_path"] = str(provenance_path)
    return record


def _parse_smolvla_config(config_path: Path):
    """Load the pinned LeRobot config despite the 0.6 subclass/choice parser incompatibility."""
    try:
        import draccus
        import transformers.utils
        # LeRobot 0.6 expects a transformers helper removed from the ExecuTorch venv's pinned
        # transformers pre-release.  It is only a decorator; identity preserves eager preprocessing.
        if not hasattr(transformers.utils, "torch_compilable_check"):
            transformers.utils.torch_compilable_check = lambda function: function
        from lerobot.policies.smolvla import SmolVLAConfig
    except ImportError as error:
        raise RuntimeError(
            "SmolVLA extraction requires LeRobot 0.6 and transformers; use the ExecuTorch venv") \
            from error
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.pop("type", None) != "smolvla":
        raise ValueError(f"pinned policy config is not SmolVLA: {config_path}")
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as stream:
        json.dump(raw, stream)
        stream.flush()
        with draccus.config_type("json"):
            return draccus.parse(SmolVLAConfig, stream.name, args=[])


def extract_smolvla_observation(
        dataset_root: Path, policy_snapshot: Path, vlm_snapshot: Path,
        config: dict[str, Any], *, output: Path) -> dict[str, Any]:
    """Extract episode 0/frame 0 through the pinned LeRobot SmolVLA preprocessing path."""
    raw_cfg = config["inputs"]["smolvla"]["recommended_raw_source"]
    components = {item["repo_id"]: item for item in config["checkpoints"]["smolvla"]["components"]}
    policy_pin = components["lerobot/smolvla_base"]
    vlm_pin = components["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
    if raw_cfg["selection"] != "episode_index_0_frame_index_0":
        raise ValueError("SmolVLA source manifest no longer selects episode 0/frame 0")
    for snapshot, pin in ((policy_snapshot, policy_pin), (vlm_snapshot, vlm_pin)):
        if snapshot.name != pin["revision"] or not snapshot.is_dir():
            raise ValueError(f"snapshot must be the pinned {pin['repo_id']} revision: {pin['revision']}")
    policy_config = policy_snapshot / "config.json"
    if not policy_config.is_file():
        raise FileNotFoundError(f"pinned SmolVLA config is absent: {policy_config}")

    try:
        import av
        import torch
        import transformers.utils
        if not hasattr(transformers.utils, "torch_compilable_check"):
            transformers.utils.torch_compilable_check = lambda function: function
        from lerobot.configs import FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.smolvla import make_smolvla_pre_post_processors, SmolVLAPolicy
        from lerobot.utils.feature_utils import dataset_to_policy_features
    except ImportError as error:
        raise RuntimeError(
            "SmolVLA extraction requires LeRobot 0.6 with the PyAV dataset backend") from error

    dataset = LeRobotDataset(
        raw_cfg["dataset"], root=dataset_root, episodes=[0], revision=raw_cfg["revision"],
        download_videos=True, video_backend="pyav")
    sample = dataset[0]
    episode_index = int(sample["episode_index"].item())
    frame_index = int(sample["frame_index"].item())
    if (episode_index, frame_index) != (0, 0):
        raise ValueError(f"LeRobot selection yielded episode/frame {(episode_index, frame_index)}")
    if sorted(dataset.meta.episodes[0]["tasks"]) != [str(sample["task"])]:
        raise ValueError("sample task differs from the pinned episode metadata")

    raw_image = sample["observation.images.top"].detach().cpu().numpy().astype(np.float32)
    raw_state = sample["observation.state"].detach().cpu().numpy().astype(np.float32)
    task = str(sample["task"])
    policy = _parse_smolvla_config(policy_config)
    policy.device = "cpu"
    # Inference on a new robot dataset replaces feature declarations with its immutable metadata;
    # the neural architecture and every numerical SmolVLA preprocessing knob remain pinned.
    features = dataset_to_policy_features(dataset.meta.features)
    policy.input_features = {name: value for name, value in features.items()
                             if value.type is not FeatureType.ACTION}
    policy.output_features = {name: value for name, value in features.items()
                              if value.type is FeatureType.ACTION}
    original_vlm_id = policy.vlm_model_name
    policy.vlm_model_name = str(vlm_snapshot)
    preprocessor, _ = make_smolvla_pre_post_processors(policy, dataset.meta.stats)
    processed = preprocessor(sample)
    policy_shell = object.__new__(SmolVLAPolicy)
    policy_shell.config = policy
    images, image_masks = SmolVLAPolicy.prepare_images(policy_shell, processed)
    state = SmolVLAPolicy.prepare_state(policy_shell, processed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SMOLVLA_NOISE_SEED)
    noise = torch.randn(
        (1, int(policy.chunk_size), int(policy.max_action_dim)), generator=generator,
        dtype=torch.float32)
    arrays = {
        "image": images[0].detach().cpu().numpy().astype(np.float32),
        "image_mask": image_masks[0].detach().cpu().numpy().astype(np.bool_),
        "language_tokens": processed["observation.language.tokens"].detach().cpu().numpy().astype(np.int64),
        "language_mask": processed["observation.language.attention_mask"].detach().cpu().numpy().astype(np.bool_),
        "state": state.detach().cpu().numpy().astype(np.float32),
        "noise": noise.numpy(),
    }
    raw_tree, raw_files = _tree_record(dataset_root)
    # Hugging Face transfer metadata contains locks and mutable cache bookkeeping, not source data.
    raw_files = [item for item in raw_files if not item["path"].startswith(".cache/")]
    raw_digest = hashlib.sha256()
    for item in raw_files:
        raw_digest.update(item["path"].encode("utf-8") + b"\0")
        raw_digest.update(str(item["bytes"]).encode("ascii") + b"\0")
        raw_digest.update(item["sha256"].encode("ascii") + b"\0")
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "added_tokens.json", "merges.txt", "vocab.json")
    tokenizer_files = [{"path": name, "bytes": (vlm_snapshot / name).stat().st_size,
                        "sha256": sha256_file(vlm_snapshot / name)}
                       for name in tokenizer_names if (vlm_snapshot / name).is_file()]
    provenance = {
        "dataset": {"repo_id": raw_cfg["dataset"], "revision": raw_cfg["revision"],
                    "episode_index": 0, "frame_index": 0, "fps": int(dataset.fps),
                    "root": str(dataset_root), "source_tree_sha256": raw_digest.hexdigest(),
                    "source_files": raw_files, "task": task,
                    "raw_image": {"shape": list(raw_image.shape), "dtype": str(raw_image.dtype),
                                  "sha256": _array_sha256(raw_image)},
                    "raw_state": {"shape": list(raw_state.shape), "dtype": str(raw_state.dtype),
                                  "sha256": _array_sha256(raw_state)}},
        "policy": {"repo_id": policy_pin["repo_id"], "revision": policy_pin["revision"],
                   "config_sha256": sha256_file(policy_config),
                   "feature_adaptation": "dataset_to_policy_features_for_pinned_raw_dataset",
                   "image_preprocessing": "SmolVLAPolicy.prepare_images resize_with_pad then [0,1] to [-1,1]",
                   "state_preprocessing": "LeRobot NormalizerProcessorStep MEAN_STD then SmolVLAPolicy.prepare_state zero-pad",
                   "language_preprocessing": "NewLineTaskProcessorStep then TokenizerProcessorStep right-padded to 48"},
        "tokenizer": {"repo_id": original_vlm_id, "revision": vlm_pin["revision"],
                      "files": tokenizer_files},
        "noise": {"algorithm": "torch.randn CPU float32", "seed": SMOLVLA_NOISE_SEED,
                  "shape": [1, int(policy.chunk_size), int(policy.max_action_dim)]},
        "software": {name: importlib.metadata.version(name) for name in
                     ("lerobot", "torch", "transformers", "numpy", "av")},
        "decoder": {"backend": "pyav", "av_library_versions": dict(av.library_versions)},
    }
    # Silence the intentionally unused digest returned before mutable cache records were filtered.
    del raw_tree
    record = write_smolvla_observation(output, arrays, provenance)
    record["source_label"] = (
        f"{raw_cfg['dataset']}@{raw_cfg['revision']}:episode=0,frame=0;"
        f"{policy_pin['repo_id']}@{policy_pin['revision']};"
        f"{original_vlm_id}@{vlm_pin['revision']};torch_randn_cpu_seed={SMOLVLA_NOISE_SEED}")
    # Persist the source label too; the first write intentionally precedes this non-byte-critical annotation.
    provenance_path = Path(record["provenance_path"])
    persisted = json.loads(provenance_path.read_text(encoding="utf-8"))
    persisted["source_label"] = record["source_label"]
    provenance_path.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _preprocess_resnet(entries: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        from torchvision.models import ResNet50_Weights
    except ImportError as error:
        raise RuntimeError("ResNet materialization requires Pillow and torchvision") from error
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    images = np.empty((len(entries), 1, 3, 224, 224), dtype=np.float32)
    for index, entry in enumerate(entries):
        with Image.open(entry["path"]) as image:
            images[index, 0] = transform(image.convert("RGB")).numpy()
    _write_deterministic_npz(output, {"images": images})
    return {"name": "IMAGENET1K_V2", "resize": 232, "center_crop": 224,
            "interpolation": "bilinear", "antialias": True,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _snapshot_for(report: dict[str, Any], repo_id: str | None = None) -> Path:
    if report["kind"] == "huggingface_snapshot":
        return Path(report["cache_path"])
    components = report["components"]
    selected = components[0] if repo_id is None else next(
        item for item in components if item["repo_id"] == repo_id)
    return Path(selected["cache_path"])


def _shell_env(env: dict[str, str]) -> str:
    lines = ["#!/bin/sh", "set -eu",
             'BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)']
    for key, value in sorted(env.items()):
        rendered = value.replace("{bundle}", "${BUNDLE_DIR}")
        if rendered.startswith("${BUNDLE_DIR}"):
            lines.append(f'export {key}="{rendered}"')
        else:
            lines.append(f"export {key}={shlex.quote(rendered)}")
    lines.append("")
    return "\n".join(lines)


def _copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def materialize(spec: dict[str, Any], selected: list[str], report: dict[str, Any],
                args: argparse.Namespace) -> Path:
    if not report["ready"]:
        blockers = [f"{model}: {value}" for model, detail in report["models"].items()
                    for value in detail["blockers"]]
        raise RuntimeError("paper input preparation is blocked:\n- " + "\n- ".join(blockers))
    product = new_product(
        "paper-inputs", version=int(spec["version"]), target=str(spec["target"]),
        sources=[{"path": str(args.spec), "sha256": sha256_file(args.spec)}],
        notes="Attributed, byte-hashed inputs for the frozen-compiler K1 holdouts.",
        update_latest=False)
    bundle: dict[str, Any] = {"version": spec["version"], "target": spec["target"],
                              "models": {}, "active_holdouts": list(ACTIVE_MODELS)}
    common_env = {"HF_HUB_CACHE": str(args.hf_cache), "HF_HUB_OFFLINE": "1",
                  "TRANSFORMERS_OFFLINE": "1"}
    for model in selected:
        details = report["models"][model]
        input_report = details["input"]
        artifacts: list[dict[str, Any]] = []
        env = dict(common_env)
        provenance: dict[str, Any] = {"checkpoint": details["checkpoint"],
                                     "input_source": input_report["source_label"]}
        if model in TOKEN_MODELS:
            prefix = TOKEN_MODELS[model]
            snapshot = _snapshot_for(details["checkpoint"])
            tokens = _tokenize(snapshot, args.language_corpus)
            destination = product.add_artifact(f"inputs/{model}/token_ids.npy")
            np.save(destination, tokens, allow_pickle=False)
            artifacts.append({"path": destination.relative_to(product.path).as_posix(),
                              "bytes": destination.stat().st_size,
                              "sha256": sha256_file(destination)})
            provenance.update({"source_path": str(args.language_corpus),
                               "source_sha256": sha256_file(args.language_corpus),
                               "tokenizer_revision": details["checkpoint"]["revision"],
                               "tokenization": spec["language_corpus"]["policy"]})
            env.update({f"{prefix}_SESSION": "e2e", f"{prefix}_TOKEN_IDS":
                        f"{{bundle}}/{artifacts[-1]['path']}",
                        f"{prefix}_TOKEN_SOURCE": str(args.language_source),
                        f"{prefix}_PAPER_READY": "1", "M2M_PREFILL_TOKENS": "128",
                        "M2M_DECODE_TOKENS": "32"})
            # The Model2MLIR capture.toml files intentionally default to two-layer smoke models.
            # A sourced paper environment must neutralize any inherited smoke/slice selection so a
            # nominally paper-ready capture cannot silently load a reduced or random-init network.
            env[f"{prefix}_LAYERS"] = ""
            if model == "gemma2_2b":
                env["M2M_GEMMA_SLICE_LAYERS"] = ""
        elif model == "smolvla":
            destination = product.add_artifact("inputs/smolvla/input.npz")
            _copy_artifact(args.smolvla_input_npz, destination)
            artifacts.append({"path": destination.relative_to(product.path).as_posix(),
                              "bytes": destination.stat().st_size,
                              "sha256": sha256_file(destination)})
            provenance.update({"source_path": str(args.smolvla_input_npz),
                               "source_sha256": sha256_file(args.smolvla_input_npz),
                               "arrays": input_report["arrays"]})
            env.update({"M2M_SMOLVLA_SESSION": "e2e", "M2M_SMOLVLA_PRETRAINED": "1",
                        "M2M_SMOLVLA_INPUT_NPZ": f"{{bundle}}/{artifacts[-1]['path']}",
                        "M2M_SMOLVLA_INPUT_SOURCE": str(args.smolvla_input_source),
                        "M2M_SMOLVLA_PAPER_READY": "1", "M2M_SMOLVLA_IMAGE_SIZE": "512"})
        elif model == "resnet50_v1_5":
            destination = product.add_artifact("inputs/resnet50_v1_5/images.npz")
            preprocessing = _preprocess_resnet(input_report["entries"], destination)
            artifacts.append({"path": destination.relative_to(product.path).as_posix(),
                              "bytes": destination.stat().st_size,
                              "sha256": sha256_file(destination)})
            provenance.update({"source_manifest": str(args.resnet_images_list),
                               "source_manifest_sha256": sha256_file(args.resnet_images_list),
                               "ordered_images": input_report["entries"],
                               "preprocessing": preprocessing})
            env.update({"M2M_RESNET_INPUT_NPZ": f"{{bundle}}/{artifacts[-1]['path']}",
                        "M2M_RESNET_INPUT_SOURCE": str(args.resnet_input_source),
                        "M2M_RESNET_PREPROCESSING": "IMAGENET1K_V2",
                        "M2M_RESNET_PRETRAINED": "1", "M2M_RESNET_PAPER_READY": "1",
                        "M2M_SESSION_STEPS": "256"})
        elif model == "lstmnetvit":
            checkpoint_dest = product.add_artifact("checkpoints/lstmnetvit/state_dict.pt")
            session_dest = product.add_artifact("inputs/lstmnetvit/trajectory.npz")
            _copy_artifact(args.vitfly_checkpoint, checkpoint_dest)
            _copy_artifact(args.vitfly_session_npz, session_dest)
            for destination in (checkpoint_dest, session_dest):
                artifacts.append({"path": destination.relative_to(product.path).as_posix(),
                                  "bytes": destination.stat().st_size,
                                  "sha256": sha256_file(destination)})
            provenance.update({"source_path": str(args.vitfly_session_npz),
                               "source_sha256": sha256_file(args.vitfly_session_npz),
                               "arrays": input_report["arrays"]})
            env.update({"VITFLY_DIR": str(args.vitfly_dir),
                        "VITFLY_CKPT": f"{{bundle}}/{artifacts[0]['path']}",
                        "VITFLY_SESSION_NPZ": f"{{bundle}}/{artifacts[1]['path']}",
                        "VITFLY_SESSION_SOURCE": str(args.vitfly_session_source),
                        "VITFLY_PAPER_READY": "1", "M2M_SESSION_STEPS": "256"})
        env_path = product.add_artifact(f"env/{model}.sh")
        env_path.write_text(_shell_env(env), encoding="utf-8")
        env_path.chmod(0o755)
        artifacts.append({"path": env_path.relative_to(product.path).as_posix(),
                          "bytes": env_path.stat().st_size, "sha256": sha256_file(env_path)})
        bundle["models"][model] = {"artifacts": artifacts, "environment": env,
                                   "provenance": provenance}
    record_path = product.add_artifact("paper_inputs.json")
    record_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    product.write_manifest()
    # Publish only a complete product.  A failed transform may leave an inspectable, unreferenced
    # directory, but it can never replace the last valid ``latest`` bundle.
    latest = product.path.parent / "latest"
    temporary = product.path.parent / f".latest.{os.urandom(3).hex()}"
    os.symlink(product.path.name, temporary)
    os.replace(temporary, latest)
    return product.path


def validate_bundle(path: Path) -> list[str]:
    errors = []
    record_path = path / "paper_inputs.json"
    if not record_path.is_file():
        return [f"paper input record is absent: {record_path}"]
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"paper input record is invalid: {error}"]
    if tuple(record.get("active_holdouts", ())) != ACTIVE_MODELS:
        errors.append("active holdout list differs from input_sources_v1.yaml")
    models = record.get("models", {})
    if not models or not set(models) <= set(ACTIVE_MODELS):
        errors.append("bundle model set is empty or contains a non-holdout model")
    for model, detail in models.items():
        checkpoint = detail.get("provenance", {}).get("checkpoint")
        if not isinstance(checkpoint, dict):
            errors.append(f"{model}: checkpoint provenance is absent")
        else:
            errors.extend(f"{model}: {value}" for value in _validate_checkpoint_record(checkpoint))
        for artifact in detail.get("artifacts", []):
            relative = str(artifact.get("path", "")) if isinstance(artifact, dict) else ""
            pure = PurePosixPath(relative)
            root = path.resolve()
            artifact_path = (root / pure).resolve()
            if (not relative or pure.is_absolute() or ".." in pure.parts
                    or relative != pure.as_posix() or not artifact_path.is_relative_to(root)):
                errors.append(f"{model}: unsafe artifact path: {relative!r}")
                continue
            if not artifact_path.is_file():
                errors.append(f"{model}: artifact is absent: {relative}")
                continue
            actual = sha256_file(artifact_path)
            if actual != artifact["sha256"]:
                errors.append(f"{model}: SHA-256 mismatch for {relative}")
            if artifact_path.stat().st_size != artifact["bytes"]:
                errors.append(f"{model}: byte count mismatch for {relative}")
    return errors


def _validate_checkpoint_record(checkpoint: dict[str, Any]) -> list[str]:
    kind = checkpoint.get("kind")
    if kind == "huggingface_composite":
        components = checkpoint.get("components")
        if not isinstance(components, list) or not components:
            return ["composite HF checkpoint has no components"]
        return [f"{component.get('repo_id', 'component')}: {value}"
                for component in components for value in _validate_checkpoint_record(component)]
    if kind == "huggingface_snapshot":
        snapshot = Path(str(checkpoint.get("cache_path", "")))
        expected = checkpoint.get("tree_sha256")
        if not snapshot.is_dir():
            return [f"pinned HF snapshot is absent: {snapshot}"]
        if not expected or not checkpoint.get("files"):
            return ["pinned HF snapshot lacks a recorded byte tree"]
        actual, _files = _tree_record(snapshot)
        return ([] if actual == expected else
                [f"pinned HF snapshot tree SHA-256 is {actual}, expected {expected}"])
    if kind == "torch_hub_file":
        source = Path(str(checkpoint.get("path", "")))
        expected = checkpoint.get("sha256")
        if not source.is_file():
            return [f"checkpoint is absent: {source}"]
        actual = sha256_file(source)
        return [] if expected and actual == expected else [
            f"checkpoint SHA-256 is {actual}, expected {expected or 'a recorded digest'}"]
    if kind == "external_file_and_git":
        errors = []
        source = Path(str(checkpoint.get("source_path", "")))
        source_file = source / str(checkpoint.get("source_file", ""))
        expected_file = checkpoint.get("source_file_sha256")
        if not source_file.is_file():
            errors.append(f"pinned source file is absent: {source_file}")
        elif sha256_file(source_file) != expected_file:
            errors.append(f"pinned source file SHA-256 changed: {source_file}")
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.stdout.strip() != checkpoint.get("source_revision"):
            errors.append("pinned source Git revision changed")
        validation = checkpoint.get("checkpoint_validation")
        if not isinstance(validation, dict) or validation.get("strict_load") is not True:
            errors.append("published checkpoint lacks a successful strict-load record")
        return errors
    return [f"unsupported checkpoint provenance kind {kind!r}"]


def _models(value: str) -> list[str]:
    selected = ACTIVE_MODELS if value == "all" else tuple(item.strip() for item in value.split(","))
    unknown = sorted(set(selected) - set(ACTIVE_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown holdout models: {unknown}")
    if not selected:
        raise argparse.ArgumentTypeError("select at least one model")
    return list(selected)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--spec", type=_path, default=SPEC)
    cli.add_argument("--models", type=_models, default=list(ACTIVE_MODELS),
                     help="comma-separated active holdouts, or all")
    cli.add_argument("--hf-cache", type=_path,
                     default=Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser())
    cli.add_argument("--torch-cache", type=_path,
                     default=Path("~/.cache/torch/hub/checkpoints").expanduser())
    cli.add_argument("--vitfly-dir", type=_path, default=Path("/scratch/agustin/projects/vitfly"))
    cli.add_argument("--vitfly-checkpoint", type=_path)
    cli.add_argument("--vitfly-session-npz", type=_path)
    cli.add_argument("--vitfly-session-source")
    cli.add_argument("--language-corpus", type=_path)
    cli.add_argument("--language-source")
    cli.add_argument("--smolvla-input-npz", type=_path)
    cli.add_argument("--smolvla-input-source")
    cli.add_argument("--resnet-images-list", type=_path)
    cli.add_argument("--resnet-input-source")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="read-only local checkpoint and input audit")
    prepare = sub.add_parser("prepare", help="materialize a versioned paper-input bundle")
    prepare.add_argument("--dry-run", action="store_true")
    extract = sub.add_parser(
        "extract-language", help="verify and deterministically extract the pinned WikiText parquet")
    extract.add_argument("parquet", type=_path)
    extract.add_argument("--output", type=_path,
                         help="destination text path (default: Merlin paper-input cache)")
    extract_resnet = sub.add_parser(
        "extract-resnet", help="verify a pinned image ZIP and extract the ordered ResNet stream")
    extract_resnet.add_argument("archive", type=_path)
    extract_resnet.add_argument("--output-dir", type=_path,
                                help="destination directory (default: Merlin paper-input cache)")
    smolvla = sub.add_parser(
        "extract-smolvla",
        help="download/select the pinned real LeRobot observation and apply SmolVLA preprocessing")
    smolvla.add_argument("--dataset-root", type=_path, required=True,
                         help="local immutable/download destination for the pinned LeRobot episode")
    smolvla.add_argument("--output", type=_path, required=True,
                         help="destination deterministic input.npz")
    validate = sub.add_parser("validate", help="verify every byte in a prepared bundle")
    validate.add_argument("bundle", type=_path)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate":
        errors = validate_bundle(args.bundle)
        print(json.dumps({"bundle": str(args.bundle), "valid": not errors, "errors": errors},
                         indent=2, sort_keys=True))
        return 0 if not errors else 2
    spec = load_yaml(args.spec)
    if tuple(spec.get("holdout_models", ())) != ACTIVE_MODELS:
        raise SystemExit("input source manifest holdouts differ from the frozen active set")
    if args.command == "extract-language":
        try:
            record = extract_language_parquet(
                args.parquet, spec["language_corpus"]["recommended_source"], output=args.output)
        except (OSError, RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.command == "extract-resnet":
        try:
            config = dict(spec["inputs"]["resnet50_v1_5"]["recommended_raw_source"])
            config["observations"] = int(spec["inputs"]["resnet50_v1_5"]["observations"])
            record = extract_resnet_images(args.archive, config, output_dir=args.output_dir)
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.command == "extract-smolvla":
        components = {item["repo_id"]: item
                      for item in spec["checkpoints"]["smolvla"]["components"]}
        policy = components["lerobot/smolvla_base"]
        vlm = components["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
        policy_snapshot = _hf_repo_dir(args.hf_cache, policy["repo_id"]) / "snapshots" / policy["revision"]
        vlm_snapshot = _hf_repo_dir(args.hf_cache, vlm["repo_id"]) / "snapshots" / vlm["revision"]
        try:
            record = extract_smolvla_observation(
                args.dataset_root, policy_snapshot, vlm_snapshot, spec, output=args.output)
        except (OSError, RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    report = audit(spec, args.models, args)
    if args.command == "audit" or args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    try:
        output = materialize(spec, args.models, report, args)
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
