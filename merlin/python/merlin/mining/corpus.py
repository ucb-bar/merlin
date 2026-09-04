"""Deterministic materialization of the generic CPU/RVV development corpus."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .campaign import PartitionPolicy


def _product(mapping: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    names = list(mapping)
    for values in itertools.product(*(mapping[name] for name in names)):
        yield dict(zip(names, values, strict=True))


def _identities(family: dict[str, Any]) -> Iterable[dict[str, Any]]:
    name = str(family["name"])
    operations = family.get("operations", ())
    dtypes = family.get("dtypes", ("not_applicable",))
    if name == "contraction":
        axes = family["axes"]
        for values in _product({"operation": operations, "dtype": dtypes,
                                "M": axes["M"], "N": axes["N"], "K": axes["K"],
                                "layout": family["layouts"]}):
            yield {"family": name, "operation": values.pop("operation"),
                   "dtype": values.pop("dtype"), "shape": values,
                   "layout": values.pop("layout"), "state": "stateless", "core_count": 1}
    elif name in {"elementwise_map", "reduction"}:
        for values in _product({"operation": operations, "dtype": dtypes,
                                "length": family["lengths"]}):
            yield {"family": name, "operation": values["operation"], "dtype": values["dtype"],
                   "shape": {"length": values["length"]}, "layout": "contiguous",
                   "state": "stateless", "core_count": 1}
    elif name == "movement_layout":
        for values in _product({"operation": operations, "dtype": dtypes,
                                "working_set_bytes": family["working_set_bytes"]}):
            yield {"family": name, "operation": values["operation"], "dtype": values["dtype"],
                   "shape": {"working_set_bytes": values["working_set_bytes"]},
                   "layout": "operation_defined", "state": "stateless", "core_count": 1}
    elif name == "fusion_epilogue":
        axes = family["axes"]
        for values in _product({"operation": operations, "dtype": dtypes,
                                "M": axes["M"], "N": axes["N"], "K": axes["K"]}):
            yield {"family": name, "operation": values.pop("operation"),
                   "dtype": values.pop("dtype"), "shape": values, "layout": "row_row",
                   "state": "fused_epilogue", "core_count": 1}
    elif name == "runtime_parallel":
        for values in _product({"operation": operations, "core_count": family["core_counts"],
                                "reuse_count": family["reuse_counts"]}):
            yield {"family": name, "operation": values["operation"], "dtype": "fp32",
                   "shape": {"work_items": 1024}, "layout": "contiguous",
                   "state": {"reuse_count": values["reuse_count"]},
                   "core_count": values["core_count"]}
    else:
        raise ValueError(f"unsupported development-corpus family {name!r}")


def expand_definition(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a frozen axes definition into content-addressed, partitioned capsule descriptors."""
    partition = raw["partition"]
    policy = PartitionPolicy(
        modulus=int(partition["modulus"]), train=tuple(partition["train"]),
        validation=tuple(partition["validation"]), heldout=tuple(partition["heldout"]))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family in raw["families"]:
        for identity in _identities(family):
            canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            capsule_id = f"{identity['family']}-{identity['operation']}-{digest[:16]}"
            if capsule_id in seen:
                raise ValueError(f"duplicate capsule id {capsule_id}")
            seen.add(capsule_id)
            rows.append({"id": capsule_id, "sha256": digest, "split": policy.split(capsule_id),
                         **identity})
    return sorted(rows, key=lambda row: row["id"])


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                    for row in rows)


def derive_materialization(definition_path: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Derive the exact split bytes and content identity from a frozen definition."""
    definition_path = Path(definition_path).resolve()
    definition_bytes = definition_path.read_bytes()
    raw = yaml.safe_load(definition_bytes)
    rows = expand_definition(raw)
    by_split = {name: [row for row in rows if row["split"] == name]
                for name in ("train", "validation", "heldout")}
    payloads = {
        "public/train.jsonl": _jsonl(by_split["train"]),
        "public/validation.jsonl": _jsonl(by_split["validation"]),
        "sealed/heldout.jsonl": _jsonl(by_split["heldout"]),
    }
    payload_digests = {name: hashlib.sha256(payload).hexdigest()
                       for name, payload in payloads.items()}
    aggregate = hashlib.sha256()
    for name in sorted(payloads):
        aggregate.update(name.encode("utf-8") + b"\0" + payloads[name])
    identity = {
        "definition_sha256": hashlib.sha256(definition_bytes).hexdigest(),
        "corpus_sha256": aggregate.hexdigest(),
        "capsule_count": len(rows),
        "split_counts": {name: len(values) for name, values in by_split.items()},
        "files": payload_digests,
    }
    return payloads, identity


def materialize_definition(definition_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Write public train/validation and sealed heldout JSONL plus a content lock."""
    definition_path, output_root = Path(definition_path).resolve(), Path(output_root)
    definition_bytes = definition_path.read_bytes()
    raw = yaml.safe_load(definition_bytes)
    payloads, identity = derive_materialization(definition_path)
    for relpath, payload in payloads.items():
        path = output_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    lock = {
        "version": 1, "status": "materialized", "label": raw["label"],
        "definition": str(definition_path),
        **identity,
        "partition": raw["partition"],
        "paper_model_exclusion": raw["paper_model_exclusion"],
    }
    (output_root / "corpus_lock.yaml").write_text(yaml.safe_dump(lock, sort_keys=False),
                                                   encoding="utf-8")
    return lock
