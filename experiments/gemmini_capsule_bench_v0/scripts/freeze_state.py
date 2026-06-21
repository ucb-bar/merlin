#!/usr/bin/env python3
"""P0 — freeze the capsule_bench_v0 state into a manifest with content hashes.

Records repo SHA / dirty / untracked, toolchain versions, capsule + pass counts, simulator
availability, and a deterministic sha256 over each tracked artifact tree. Re-runnable: same tree
content -> same hashes. Does not mutate anything except the output manifest.

Usage: .venv/bin/python experiments/gemmini_capsule_bench_v0/scripts/freeze_state.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "gemmini" / "capsule_bench_v0_freeze_manifest.yaml"

# artifact trees to hash (relative to repo); build/ and __pycache__ excluded
ARTIFACTS = {
    "bench_contract": "bench_contract",
    "targetgen_modules": "merlin/python/merlin/targetgen",
    "agent_spec_v1": "generated_targets/gemmini/agent_spec_v1_mlir_oot",
    "results_gemmini": "results/gemmini",
    "runs_public": "runs/capsule_bench_v1",
    "runs_hidden": "runs/capsule_bench_v1_hidden",
}
_SKIP_PARTS = {"build", "__pycache__", ".git"}


def _sh(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=str(REPO), capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def hash_tree(root: Path) -> dict:
    """Deterministic sha256 over a directory: sorted relpaths, hash(relpath + content)."""
    if not root.exists():
        return {"present": False, "sha256": None, "n_files": 0}
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _SKIP_PARTS & set(p.parts):
            continue
        rel = p.relative_to(root).as_posix().encode()
        h.update(rel)
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
        n += 1
    return {"present": True, "sha256": h.hexdigest(), "n_files": n}


def _count_pass(runs_root: Path) -> tuple[int, int]:
    total = passed = 0
    if not runs_root.exists():
        return 0, 0
    for cr in runs_root.rglob("capsule_result.json"):
        try:
            r = json.loads(cr.read_text())
        except Exception:
            continue
        total += 1
        if r.get("status") == "pass":
            passed += 1
    return passed, total


def _tool_version(args: list[str]) -> str:
    out = _sh(args)
    return out.splitlines()[0] if out else "unknown"


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "merlin" / "python"))
    try:
        from merlin.targetgen.contract import toolchain as tc
        llvm = f"{tc.LLVM_VERSION}@{tc.LLVM_COMMIT}"
    except Exception:
        llvm = "unknown"
    try:
        from merlin.runtime.backends import gemmini as gem
        spike_ok = gem.available("spike")
        veri_ok = gem.available("verilator")
        gcc = str(gem.gcc_path())
    except Exception:
        spike_ok = veri_ok = False
        gcc = "unknown"

    pub_pass, pub_total = _count_pass(REPO / "runs/capsule_bench_v1")
    hid_pass, hid_total = _count_pass(REPO / "runs/capsule_bench_v1_hidden")
    n_caps = len(list((REPO / "bench_contract/capsules").rglob("capsule.yaml")))
    status = _sh(["git", "status", "--short"])
    untracked = [ln[3:] for ln in status.splitlines() if ln.startswith("??")]

    manifest = {
        "frozen_artifact": "capsule_bench_v0 + agent_spec_v1_mlir_oot",
        "repo_sha": _sh(["git", "rev-parse", "HEAD"]) or "unknown",
        "branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "working_tree_dirty": bool(status),
        "untracked_files_count": len(untracked),
        "untracked_files_sample": untracked[:40],
        "toolchain": {
            "llvm_mlir": llvm,
            "riscv_gcc": gcc,
            "spike": _tool_version([gcc.replace("riscv64-unknown-elf-gcc", "spike"), "--help"])
                     if gcc != "unknown" else "unknown",
            "clang": _tool_version(["clang-23", "--version"]),
        },
        "capsule_count": n_caps,
        "public_dev_passed": f"{pub_pass}/{pub_total}",
        "hidden_passed": f"{hid_pass}/{hid_total}",
        "spike_status": "available" if spike_ok else "unavailable",
        "verilator_status": "available" if veri_ok else "unavailable",
        "vcs_status": "unavailable (simv segfaults on the L2/L3-validated bare-metal ELF)",
        "firesim_status": "unavailable (no verified bare-metal Gemmini replay hook in this env)",
        "artifact_hashes": {name: hash_tree(REPO / rel) for name, rel in ARTIFACTS.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  repo_sha={manifest['repo_sha'][:12]} dirty={manifest['working_tree_dirty']} "
          f"caps={n_caps} public={manifest['public_dev_passed']} hidden={manifest['hidden_passed']}")
    for name, h in manifest["artifact_hashes"].items():
        print(f"  {name}: {(h['sha256'] or 'absent')[:16]} ({h['n_files']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
