#!/usr/bin/env python3
"""Vendor one companion support provider into ``examples/<example>/support``, with exact provenance.

The vendored tree is the companion's provider root at one commit, byte for byte (no normalization), so
``SOURCE.yaml`` can name the companion tree and ``merlin/tests/infra/test_example_support.py`` can
recompute it. The script also upserts the matching entry in ``build_tools/upstreams/target_support.json``.

    python build_tools/scripts/vendor_support.py --repo ../<name>-mlir --example <example> \
        --url https://github.com/<org>/<name>-mlir.git [--commit <rev>] [--provider-root .] [--published]

It refuses a dirty companion (uncommitted bytes have no commit to cite) and a provider root without
``provenance.json`` (a vendored record must point at the file provenance it carries).
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "build_tools/upstreams/target_support.json"
HEADER = """\
# Provenance of support/: a copy of one companion provider root, byte-identical except the files
# listed under `normalized`. The vendored directory's git tree id is vendored_tree; putting each
# normalized file's companion_blob back reproduces source.tree. merlin/tests/infra/test_example_support.py
# recomputes both, and the file count, from the committed bytes. Do not edit support/ in place: a change
# is a new vendoring with its own source commit, recorded here (build_tools/scripts/vendor_support.py).
"""


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode().strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True, help="companion git checkout")
    ap.add_argument("--example", required=True, help="examples/<example>/support receives the provider")
    ap.add_argument("--url", required=True, help="the companion's canonical repository URL")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--provider-root", default=".")
    ap.add_argument("--published", action="store_true", help="the commit is on a pushed remote branch")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    if _git(repo, "status", "--porcelain"):
        print(f"error: {repo} has uncommitted changes; commit them first", file=sys.stderr)
        return 2
    commit = _git(repo, "rev-parse", f"{args.commit}^{{commit}}")
    spec = commit if args.provider_root in (".", "") else f"{commit}:{args.provider_root}"
    tree = _git(repo, "rev-parse", f"{spec}^{{tree}}" if spec == commit else spec)
    date = _git(repo, "show", "-s", "--format=%cI", commit)
    archive = _git(repo, "archive", "--format=tar", spec, binary=True)

    dest = REPO / "examples" / args.example / "support"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = [m for m in tar.getmembers() if m.isfile() or m.issym()]
        tar.extractall(dest, members=members, filter="tar")
    provider = yaml.safe_load((dest / "provider.yaml").read_text(encoding="utf-8"))
    if not (dest / "provenance.json").is_file():
        print(f"error: {spec} has no provenance.json at its provider root", file=sys.stderr)
        return 2

    record = {
        "schema": "merlin.vendored_support.v1",
        "target": provider["target"],
        "provider_id": provider["id"],
        "path": "support",
        "file_count": len(members),
        "vendored": datetime.date.today().isoformat(),
        "source": {
            "repository": args.url,
            "published": bool(args.published),
            "commit": commit,
            "commit_date": date,
            "provider_root": args.provider_root,
            "tree": tree,
        },
        "vendored_tree": tree,
        "normalized": [],
        "pinned": {
            "manifest": MANIFEST.relative_to(REPO).as_posix(),
            "companion_commit": commit,
            "relation": "same_commit",
            "provider_tree": tree,
            "identical_to_source": True,
        },
        "records": {"file_provenance": "support/provenance.json", "migrations": []},
    }
    source_record = REPO / "examples" / args.example / "SOURCE.yaml"
    source_record.write_text(HEADER + yaml.safe_dump(record, sort_keys=False), encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rel_record = source_record.relative_to(REPO).as_posix()
    vendored = {
        "path": dest.relative_to(REPO).as_posix(),
        "source_record": rel_record,
        "commit": commit,
        "tree": tree,
        "vendored_tree": tree,
        "file_count": len(members),
    }
    entries = manifest["companions"]
    existing = next((e for e in entries if (e.get("vendored") or {}).get("source_record") == rel_record), None)
    entry = existing if existing is not None else {"target": provider["target"]}
    entry.update(
        {
            "target": provider["target"],
            "url": args.url,
            "companion_commit": commit,
            "companion_commit_relation": "same_commit",
            "provider_root": args.provider_root,
            "provider_role": provider["role"],
            "vendored": vendored,
            "file_provenance": "provenance.json",
        }
    )
    if existing is None:
        entries.append(entry)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"vendored {provider['target']} @ {commit[:12]} ({len(members)} files, tree {tree[:12]}) -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
