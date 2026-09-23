"""Inventory tracked first-party ownership and imports without traversing generated workspaces."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import subprocess
from pathlib import Path


def inventory(root: Path) -> dict:
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    records = []
    imports = collections.Counter()
    counts = collections.Counter()
    errors = []
    for name in sorted(filter(None, paths)):
        path = root / name
        if not path.is_file() or name.startswith("third_party/"):
            continue
        data = path.read_bytes()
        category = name.split("/", 1)[0]
        counts[category] += 1
        records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if path.suffix == ".py" and (
            name.startswith(("merlin/python/", "src/")) or (name.startswith("packages/") and "/src/" in name)
        ):
            try:
                tree = ast.parse(data, filename=name)
            except SyntaxError as exc:
                errors.append({"path": name, "error": str(exc)})
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports[node.module] += 1
                elif isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
    return {
        "version": 1,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "tracked_counts": dict(sorted(counts.items())),
        "files": records,
        "imports": dict(sorted(imports.items())),
        "syntax_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
    result = inventory(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": len(result["files"]), "errors": result["syntax_errors"]}))
    return bool(result["syntax_errors"])


if __name__ == "__main__":
    raise SystemExit(main())
