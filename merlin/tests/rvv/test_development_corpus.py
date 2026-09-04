"""Frozen generic development-corpus materialization and leakage boundaries."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from merlin.common.paths import bench_dir
from merlin.mining.corpus import expand_definition, materialize_definition


DEFINITION = bench_dir() / "rvv_paper" / "development_corpus_v2.yaml"


def test_expansion_is_deterministic_partitioned_and_model_independent():
    raw = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    first, second = expand_definition(raw), expand_definition(raw)
    assert first == second
    assert len(first) == len({row["id"] for row in first})
    assert {row["split"] for row in first} == {"train", "validation", "heldout"}
    encoded = "\n".join(str(row) for row in first)
    assert not any(name in encoded for name in raw["paper_model_exclusion"]["forbidden_workloads"])


def test_each_split_has_generic_movement_layout_vector_tail_cases():
    """The frozen grader must not discover an unwinnable coverage gap after an agent run."""
    raw = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    rows = expand_definition(raw)
    for split in ("train", "validation", "heldout"):
        movement = [
            row for row in rows
            if row["split"] == split and row["family"] == "movement_layout"
        ]
        assert movement
        # The movement descriptor is a byte budget. Both fp32 (8 lanes) and int8
        # (32 lanes) therefore have a tail whenever the output byte count is not
        # divisible by 32. Restrict to copy so the check is operation-independent.
        tails = [
            row for row in movement
            if row["operation"] == "copy"
            and int(row["shape"]["working_set_bytes"]) % 32 != 0
        ]
        assert tails, f"{split} has no generic movement-layout RVV tail"


def test_materialization_separates_public_and_sealed_bytes(tmp_path: Path):
    lock = materialize_definition(DEFINITION, tmp_path)
    assert lock["definition_sha256"] == hashlib.sha256(DEFINITION.read_bytes()).hexdigest()
    assert lock["capsule_count"] == sum(lock["split_counts"].values())
    assert (tmp_path / "public/train.jsonl").is_file()
    assert (tmp_path / "public/validation.jsonl").is_file()
    assert (tmp_path / "sealed/heldout.jsonl").is_file()
    assert not (tmp_path / "public/heldout.jsonl").exists()
