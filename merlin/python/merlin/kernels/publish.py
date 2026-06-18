"""Publish a mining run as a DURABLE, VERSIONED artifact — the abstracted kernel knowledge is an
*input* the tuning agent / beam-search iterate on (like the experiment packages), so it must be
persisted with provenance, not left in gitignored ``output/``.

Layout (mirrors generated_targets/<target>/<run_id>/):

    mined_knowledge/<target>/<target>_mined_v{V}_{timestamp}/
        manifest.yaml             # provenance: sources, repo paths, per-source counts, git sha,
                                  #   min_kernels gate, promoted policies + their (kernels, sources)
        policy_rules.yaml         # the promoted policies (the tuning-agent's lever evidence)
        abstraction_candidates.yaml
        kernel_mining_report.md
        *_index.json              # the per-source indexes the run aggregated (reproducibility)

The newest dir under mined_knowledge/<target>/ is the one the tuning agent reads by default.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..common.yaml import load_yaml, write_yaml


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _index_summary(index_path: Path) -> dict:
    d = json.loads(Path(index_path).read_text())
    recs = d.get("records", [])
    src = recs[0].get("source") if recs else Path(index_path).stem.split("_")[0]
    return {"source": src, "kernels": len(recs)}


def publish_mining(index_paths: list[str | Path], target: str, *, version: int,
                   timestamp: str, out_root: str | Path = "mined_knowledge",
                   extract_dir: str | Path = "output/kernels",
                   min_kernels: int = 10) -> Path:
    """Assemble a versioned mined-knowledge artifact from the kernel-extract outputs.

    ``timestamp`` is passed in (callers stamp it; the workflow/runtime forbids Date.now in some
    contexts). Returns the artifact directory.
    """
    extract_dir = Path(extract_dir)
    run_id = f"{target}_mined_v{version}_{timestamp}"
    art = Path(out_root) / target / run_id
    art.mkdir(parents=True, exist_ok=True)

    # copy the extract products + the source indexes
    copied = []
    for name in ("policy_rules.yaml", "abstraction_candidates.yaml", "kernel_mining_report.md"):
        src = extract_dir / name
        if src.is_file():
            shutil.copy2(src, art / name)
            copied.append(name)
    sources = []
    for ip in index_paths:
        ip = Path(ip)
        if ip.is_file():
            shutil.copy2(ip, art / ip.name)
            sources.append(_index_summary(ip))

    # promoted policies + their support (from the copied policy_rules)
    promoted = []
    pr = art / "policy_rules.yaml"
    if pr.is_file():
        d = load_yaml(pr)
        rules = d if isinstance(d, list) else d.get("policy_rules", d.get("policies", []))
        for r in (rules or []):
            if isinstance(r, dict):
                promoted.append(r.get("policy") or r.get("name"))

    manifest = {
        "target": target,
        "run_id": run_id,
        "version": version,
        "timestamp": timestamp,
        "git_sha": _git_sha(),
        "min_kernels_gate": min_kernels,
        "sources": sources,
        "total_kernels": sum(s["kernels"] for s in sources),
        "promoted_policies": [p for p in promoted if p],
        "artifacts": copied,
        "notes": ("Durable, versioned kernel-mining artifact. The newest run_id under "
                  "mined_knowledge/<target>/ is the tuning agent's default evidence input. "
                  "Forks/policies derive from this; it is itself iterable."),
    }
    write_yaml(art / "manifest.yaml", manifest, header="Mined-knowledge artifact (kernels.publish)")
    return art
