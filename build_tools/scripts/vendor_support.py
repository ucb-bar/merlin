#!/usr/bin/env python3
"""Move a support pin, or vendor a provider that is still carried in-repo, with exact provenance.

Two jobs, one record (``build_tools/upstreams/target_support.json``):

``pin <target> <commit>``: bump a pinned support package (``pinned_support``) to another commit. Chip
packages are not copied into Merlin; editing one is a commit in ``<chip>-mlir`` and this one sha here.
The commit is fetched (by commit, from the pinned repository or ``--url``) into the generated home,
its tree is recorded beside it, and a package whose ``provider.yaml`` ``extends.commit`` is not
Merlin's base pin is refused, since every package layers on the one base:

    python build_tools/scripts/vendor_support.py pin <target> <commit> [--url file:///path/to/<chip>-mlir]

``vendor``: copy a companion provider root into ``examples/<example>/support`` byte for byte, for the
targets still carried in-repo (the base among them). The vendored tree is the companion's provider root at
one commit (no normalization), so ``SOURCE.yaml`` can name the companion tree and
``merlin/tests/infra/test_example_support.py`` can recompute it. It upserts the matching ``companions``
entry and, when the provider is the pinned base, moves the base pin with it (one source of truth); a
pinned package is never vendored.

    python build_tools/scripts/vendor_support.py vendor --repo ../<name>-mlir --example <example> \
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


def pin(argv: list[str]) -> int:
    """Move one pinned support package (or the base, when it keeps no in-repo copy) to ``commit``."""
    from merlin.targetgen import oot_fetch, support_pins
    from merlin.targetgen.providers import read_provider

    ap = argparse.ArgumentParser(prog="vendor_support.py pin", description=pin.__doc__)
    ap.add_argument("target")
    ap.add_argument("commit", help="the full 40-char commit to pin (already published to the pinned repo)")
    ap.add_argument("--url", help="fetch the commit from here instead (a mirror or a local file:// clone)")
    args = ap.parse_args(argv)

    pins = support_pins.load_pins(MANIFEST)
    current = pins.get(args.target)
    if current is None:
        print(f"error: {args.target!r} is not pinned in {MANIFEST}", file=sys.stderr)
        return 2
    if current.in_repo is not None:
        print(f"error: {args.target!r} keeps an in-repo copy at {current.in_repo}; re-vendor it with `vendor`, "
              "which moves the pin with it", file=sys.stderr)  # fmt: skip
        return 2
    moved = support_pins.Pin(**{**current.__dict__, "commit": args.commit, "tree": None})
    try:
        support_pins.parse(_with_pin(json.loads(MANIFEST.read_text(encoding="utf-8")), moved, tree="0" * 40))
        root = oot_fetch.fetch_commit(args.url or current.repo, args.commit, support_pins.pinned_root(moved))
        tree = _git(root, "rev-parse", f"{args.commit}^{{tree}}")
        provider = read_provider(root, materialized=False)
        if provider is None or provider.target != args.target:
            raise support_pins.PinError(f"{args.commit} does not declare a support provider for {args.target!r}")
        support_pins.check_single_base(provider, pins)
    except (oot_fetch.FetchError, support_pins.PinError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from check_wiring import package_core_references  # a sibling script: what the package calls in the core

    references = package_core_references(root) if current.extends else None
    manifest = _with_pin(json.loads(MANIFEST.read_text(encoding="utf-8")), moved, tree=tree, references=references)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"pinned {args.target} @ {args.commit[:12]} (tree {tree[:12]}), was {current.short}; fetched at {root}")
    return 0


def _with_pin(manifest: dict, moved, *, tree: str, references: list[str] | None = None) -> dict:
    block = manifest["pinned_support"]
    entries = [block["base"], *block["packages"]]
    entry = next(e for e in entries if e["target"] == moved.target)
    entry["commit"], entry["tree"] = moved.commit, tree
    if references is not None:
        entry["core_references"] = references
    return manifest


def _move_base_pin(manifest: dict, target: str, commit: str, tree: str, rel_dest: str) -> None:
    """A vendored copy of the pinned base IS the base pin: keep the two equal."""
    block = manifest.get("pinned_support") or {}
    base = block.get("base") or {}
    if base.get("target") != target:
        if any(p.get("target") == target for p in block.get("packages") or []):
            raise SystemExit(f"error: {target!r} is a pinned package; it is fetched, never vendored (use `pin`)")
        return
    if base.get("in_repo") != rel_dest:
        raise SystemExit(f"error: the base pin keeps its in-repo copy at {base.get('in_repo')}, not {rel_dest}")
    base["commit"], base["tree"] = commit, tree


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pin":
        return pin(argv[1:])
    if argv and argv[0] == "vendor":
        argv = argv[1:]
    return vendor(argv)


def vendor(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="vendor_support.py vendor", description=__doc__.split("\n\n")[0])
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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    provider_doc = yaml.safe_load(
        _git(repo, "show", f"{commit}:{'provider.yaml' if spec == commit else args.provider_root + '/provider.yaml'}")
    )
    _move_base_pin(manifest, provider_doc["target"], commit, tree, dest.relative_to(REPO).as_posix())
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
        "records": {
            "file_provenance": "support/provenance.json",
            # Transition records the provider carries (a support package's later moves, each with digests).
            "migrations": sorted(f"support/{p.name}" for p in dest.glob("*_migration.json")),
        },
    }
    source_record = REPO / "examples" / args.example / "SOURCE.yaml"
    if source_record.is_file():
        previous = yaml.safe_load(source_record.read_text(encoding="utf-8")) or {}
        if previous.get("tests"):
            record["tests"] = previous["tests"]  # reviewed test deselections are a property of the vendoring
    source_record.write_text(HEADER + yaml.safe_dump(record, sort_keys=False), encoding="utf-8")

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
