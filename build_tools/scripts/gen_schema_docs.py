#!/usr/bin/env python3
"""Living schema reference: generate docs/reference/schemas.md from merlin/schemas/*.schema.yaml.

The schemas are the cross-workstream contract; this emits one row per schema -> its title +
purpose (the single source of truth is the schema file's own `title`/`purpose`). Mirrors
gen_cli_docs.py / gen_package_docs.py: generate, and `--check` fails if stale.

Usage:
  gen_schema_docs.py            # (re)write docs/reference/schemas.md
  gen_schema_docs.py --check    # exit 1 if stale vs merlin/schemas/
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "merlin" / "schemas"
OUT = ROOT / "docs" / "reference" / "schemas.md"

HEADER = (
    "# Schema reference\n\n"
    "_Generated from `merlin/schemas/*.schema.yaml` by `build_tools/scripts/gen_schema_docs.py` — "
    "do not edit by hand; run the generator._\n\n"
    "The schemas are the cross-workstream coordination contract (see "
    "[Contracts](contracts.md)). Each is the source of truth for one artifact type.\n\n"
    "| Schema | Title | Purpose |\n|---|---|---|\n"
)


def _title_purpose(path: Path) -> tuple[str, str]:
    """(title, purpose-first-sentence). Uses PyYAML if available, else a minimal line scan."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        d = yaml.safe_load(text) or {}
        title = str(d.get("title", path.stem))
        purpose = " ".join(str(d.get("purpose", "")).split())
    except Exception:
        title, purpose = path.stem, ""
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("title:"):
                title = ln.split(":", 1)[1].strip() or title
            if ln.startswith("purpose:"):
                buf = [ln.split(":", 1)[1].strip().lstrip(">-|").strip()]
                for cont in lines[i + 1:]:
                    if cont[:1] in (" ", "\t"):
                        buf.append(cont.strip())
                    else:
                        break
                purpose = " ".join(x for x in buf if x)
    if len(purpose) > 160:
        purpose = purpose[:157].rstrip() + "…"
    return title.replace("|", "\\|"), purpose.replace("|", "\\|")


def render() -> str:
    rows = ""
    for p in sorted(SCHEMAS.glob("*.schema.yaml")):
        name = p.name[: -len(".schema.yaml")]
        title, purpose = _title_purpose(p)
        rows += f"| `{name}` | {title} | {purpose} |\n"
    return HEADER + rows


def main(argv: list[str]) -> int:
    new = render()
    if "--check" in argv:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != new:
            sys.stderr.write("docs/reference/schemas.md is stale — run: "
                             "python build_tools/scripts/gen_schema_docs.py\n")
            return 1
        print("docs/reference/schemas.md: up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(render().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
