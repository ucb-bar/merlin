"""Pending target transfers inventory real bytes without silently moving graded evidence."""

from __future__ import annotations

import hashlib
import json

from merlin.common.paths import repo_root


def test_pending_oot_inventory_matches_source_payload():
    root = repo_root()
    inventory = json.loads((root / "build_tools" / "oot_transfer_manifest.json").read_text())
    assert inventory["status"] == "inventory_only_no_transfer_performed"
    seen = set()
    for group in inventory["groups"]:
        assert group["status"] == "not_moved"
        assert group["destination_repository"] is None
        for row in group["files"]:
            assert row["path"].startswith(group["source_root"] + "/")
            assert row["path"] not in seen
            seen.add(row["path"])
            assert hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
    assert len(seen) == inventory["counts"]["files"]
    assert sum(name.endswith(".py") for name in seen) == inventory["counts"]["python_files"]
