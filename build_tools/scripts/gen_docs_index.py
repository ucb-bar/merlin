#!/usr/bin/env python3
"""Living docs hub: generate docs/README.md from the front-matter of every doc under docs/.

Single source of truth = each doc's own front-matter block (title/kind/status/owner/
last_verified/related). Mirrors gen_cli_docs.py / gen_package_docs.py: generate, and `--check`
fails if the hub is stale. The two auto-generated references (cli.md, module_index.md) are
recognized without front-matter. A doc under docs/ with neither front-matter nor recognition is
listed under "Uncategorized" — it either needs front-matter or belongs in artifacts/ (a report).

Usage:
  gen_docs_index.py            # (re)write docs/README.md
  gen_docs_index.py --check    # exit 1 if docs/README.md is stale
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT = DOCS / "README.md"

# Generated references that carry no hand-authored front-matter.
GENERATED = {
    "reference/cli.md": "CLI reference", "cli.md": "CLI reference",
    "reference/module_index.md": "Package module index", "module_index.md": "Package module index",
    "reference/schemas.md": "Schema reference", "schemas.md": "Schema reference",
}
SKIP_NAMES = {"README.md", "AGENT.md"}
KIND_ORDER = ["reference", "guide", "design"]
KIND_TITLE = {"reference": "Reference", "guide": "Guides", "design": "Design notes"}

PREAMBLE = """# Documentation

**AUTO-GENERATED index** by `build_tools/scripts/gen_docs_index.py` from each doc's front-matter.
Do not edit by hand — run the generator (it's `--check`ed by `check_structure.py`).

## Start here

- **New to the repo?** Read [Architecture](reference/architecture.md) then
  [Repository structure](reference/repo_structure.md).
- **Running something?** [Getting started](guides/getting_started.md) →
  the [CLI reference](reference/cli.md) → the relevant guide below.
- **Writing generated output?** See `CLAUDE.md` "Generated-output convention" and the
  `artifact-layout` skill (three roots only: `runs/`, `artifacts/`, `build/`).
- **Point-in-time reports** (results, findings, status snapshots) do not live here — they live under `artifacts/`.

Each entry shows **status** and **last-verified** date; `⚠` flags a doc whose `last_verified`
predates the newest change to the code it documents (see `check_docs_freshness.py`).
"""


def parse_front_matter(text: str) -> dict | None:
    """Minimal YAML front-matter parser (stdlib only): scalars + `[a, b]` lists."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    fm: dict = {}
    for line in text[4:end].splitlines():
        line = line.rstrip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            fm[key] = [x.strip() for x in val[1:-1].split(",") if x.strip()]
        else:
            fm[key] = val
    return fm


def discover() -> tuple[list[dict], dict[str, str]]:
    """Return (entries, slug->relpath). entry = {rel,title,kind,status,owner,last_verified,related}."""
    entries: list[dict] = []
    slug_to_rel: dict[str, str] = {}
    for p in sorted(DOCS.rglob("*.md")):
        rel = p.relative_to(DOCS).as_posix()
        if p.name in SKIP_NAMES:
            continue
        slug = p.stem
        fm = parse_front_matter(p.read_text(encoding="utf-8"))
        if fm and fm.get("kind"):
            entries.append({
                "rel": rel, "slug": slug, "title": fm.get("title", slug),
                "kind": fm.get("kind"), "status": fm.get("status", "current"),
                "owner": fm.get("owner", "?"), "last_verified": fm.get("last_verified", "?"),
                "related": fm.get("related", []) or [],
            })
            slug_to_rel[slug] = rel
        elif rel in GENERATED:
            entries.append({
                "rel": rel, "slug": slug, "title": GENERATED[rel], "kind": "reference",
                "status": "generated", "owner": "tooling", "last_verified": "—", "related": [],
            })
            slug_to_rel[slug] = rel
        else:
            entries.append({
                "rel": rel, "slug": slug, "title": slug, "kind": "uncategorized",
                "status": "?", "owner": "?", "last_verified": "?", "related": [],
            })
    return entries, slug_to_rel


def render() -> str:
    entries, slug_to_rel = discover()
    out = [PREAMBLE]

    def links(related: list[str]) -> str:
        parts = [f"[{s}]({slug_to_rel[s]})" for s in related if s in slug_to_rel]
        return f" — see also: {', '.join(parts)}" if parts else ""

    for kind in KIND_ORDER:
        rows = sorted([e for e in entries if e["kind"] == kind], key=lambda e: e["title"].lower())
        if not rows:
            continue
        out.append(f"## {KIND_TITLE[kind]}\n")
        for e in rows:
            meta = f"`{e['status']}`" + (f", verified {e['last_verified']}" if e["last_verified"] not in ("—", "?") else "")
            out.append(f"- [{e['title']}]({e['rel']}) — {meta} · owner: {e['owner']}{links(e['related'])}")
        out.append("")

    # By area (owner) quick index.
    owners: dict[str, list[dict]] = {}
    for e in entries:
        if e["kind"] in KIND_ORDER:
            owners.setdefault(e["owner"], []).append(e)
    if owners:
        out.append("## By area\n")
        for owner in sorted(owners):
            titles = ", ".join(f"[{e['title']}]({e['rel']})"
                               for e in sorted(owners[owner], key=lambda e: e["title"].lower()))
            out.append(f"- **{owner}** — {titles}")
        out.append("")

    uncat = [e for e in entries if e["kind"] == "uncategorized"]
    if uncat:
        out.append("## Uncategorized (needs front-matter or relocation to artifacts/)\n")
        for e in sorted(uncat, key=lambda e: e["rel"]):
            out.append(f"- [{e['rel']}]({e['rel']})")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str]) -> int:
    new = render()
    if "--check" in argv:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != new:
            sys.stderr.write("docs/README.md is stale — run: python build_tools/scripts/gen_docs_index.py\n")
            return 1
        print("docs/README.md: up to date")
        return 0
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(render().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
