"""The committed evaluation inputs are small, hashed, free of holdouts, and selected by stated rules."""

from __future__ import annotations

import json

from merlin.common.paths import repo_root
from merlin.model_check import inputs as INP


def _root():
    return repo_root() / "examples" / "gemmini" / "numerics"


def test_the_public_example_fits_its_budget():
    assert INP.stored_bytes(_root()) <= INP.PUBLIC_BUDGET


def test_no_sealed_input_is_committed():
    assert not [p for p in (_root() / "inputs").rglob("*") if "sealed" in p.parts]


def test_every_committed_input_matches_its_manifest_hash():
    manifest = json.loads((_root() / "MANIFEST.json").read_text())
    for model, entry in manifest["models"].items():
        visible = _root() / "inputs" / model / "visible"
        on_disk = {str(p.relative_to(visible)): INP.sha256_file(p) for p in visible.rglob("*") if p.is_file()}
        assert on_disk == entry["visible"], model
        assert len(entry["sealed"]) == len(entry["visible"]), model  # committed to by hash only


def test_articles_split_on_top_level_headings_only():
    rows = [" = A = \n", " text a \n", " = = sub = = \n", " more a \n", " = B = \n", " text b \n"]
    assert INP._articles(rows) == ["text a \n = = sub = = \n more a", "text b"]
