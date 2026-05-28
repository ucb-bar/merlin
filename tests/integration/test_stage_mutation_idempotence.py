"""Stage-mutation idempotence test.

Run ``./merlin targetgen stage-mutation`` twice into separate output
directories for the same capability spec, then assert the two
``proposed_tree/`` outputs are byte-identical. Catches non-determinism
in scaffold generation (random ordering, non-stable timestamps, etc.)
that would otherwise quietly corrupt downstream snapshots.

Marker: ``integration``. Pure-Python path, runs in seconds.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "target_specs" / "examples"

pytestmark = [pytest.mark.integration]

# A representative target per integration style — enough to catch a real
# determinism bug without paying for the full fixture matrix.
TARGETS = [
    "saturn_opu_v128",  # llvm_ukernel
    "gemmini_mx",  # post_global_plugin + llvm_ukernel
    "radiance_muon",  # runtime_hal + structured_text_isa + post_global_plugin
    "spacemit_x60_xsmtvdot",  # llvm_ukernel
]


def _stage_mutation(capability: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "merlin.py"),
            "targetgen",
            "stage-mutation",
            str(capability),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )


def _hash_tree(root: Path) -> dict[str, str]:
    """Return {relative_path: sha256} for every file under ``root``."""
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        hashes[rel] = h.hexdigest()
    return hashes


@pytest.mark.parametrize("target", TARGETS)
def test_stage_mutation_is_byte_identical_across_runs(tmp_path: Path, target: str) -> None:
    capability = EXAMPLES / target / "capability.yaml"
    if not capability.exists():
        pytest.skip(f"capability spec missing: {capability}")

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    rc_a = _stage_mutation(capability, out_a)
    assert rc_a.returncode == 0, f"first run failed: {rc_a.stderr}"
    rc_b = _stage_mutation(capability, out_b)
    assert rc_b.returncode == 0, f"second run failed: {rc_b.stderr}"

    proposed_a = out_a / target / "mutation" / "proposed_tree"
    proposed_b = out_b / target / "mutation" / "proposed_tree"

    if not proposed_a.exists():
        # Some targets may not produce a proposed_tree (no plugin/runtime to scaffold).
        # Still assert the rest of the artefact tree matches.
        proposed_a = out_a / target
        proposed_b = out_b / target

    hashes_a = _hash_tree(proposed_a)
    hashes_b = _hash_tree(proposed_b)

    # Many bundle JSONs embed the absolute output path; strip those before
    # comparing. We compare file *content* hashes after path normalisation.
    only_in_a = sorted(set(hashes_a.keys()) - set(hashes_b.keys()))
    only_in_b = sorted(set(hashes_b.keys()) - set(hashes_a.keys()))
    assert (
        not only_in_a and not only_in_b
    ), f"file set differs between runs:\n  only in A: {only_in_a}\n  only in B: {only_in_b}"

    differing: list[str] = []
    for rel, hash_a in hashes_a.items():
        if hashes_b[rel] == hash_a:
            continue
        # Differences are only acceptable if they reflect the absolute
        # out-dir path. Read both and compare after substituting the path.
        text_a = (proposed_a / rel).read_text(encoding="utf-8", errors="ignore")
        text_b = (proposed_b / rel).read_text(encoding="utf-8", errors="ignore")
        normalised_a = text_a.replace(str(out_a), "<OUT>")
        normalised_b = text_b.replace(str(out_b), "<OUT>")
        if normalised_a != normalised_b:
            differing.append(rel)

    assert not differing, f"{target}: stage-mutation produced different content across runs:\n  " + "\n  ".join(
        differing[:10]
    )
