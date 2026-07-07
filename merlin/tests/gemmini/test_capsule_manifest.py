"""The capsule provenance MANIFEST must stay complete + consistent with the corpus tree.

Guards that every capsule in merlin/contract/capsules/ is classified exactly once as generated
(emitted by generate_corpus.py) or hand_authored — so an agent can always tell which capsules are
regenerable vs the frozen source-of-record.
"""
from __future__ import annotations

import yaml

from merlin.common.paths import merlin_dir

CAP_ROOT = merlin_dir() / "contract" / "capsules"
MANIFEST = CAP_ROOT / "MANIFEST.yaml"


def test_manifest_covers_every_capsule_exactly_once():
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    generated = set(m.get("generated", []))
    hand = set(m.get("hand_authored", []))
    assert not (generated & hand), f"capsules listed as both generated and hand-authored: {generated & hand}"
    listed = generated | hand
    on_disk = {str(p.parent.relative_to(CAP_ROOT)) for p in CAP_ROOT.rglob("capsule.yaml")}
    assert listed == on_disk, (
        f"MANIFEST out of sync with the tree — missing: {on_disk - listed}; "
        f"stale: {listed - on_disk}. Re-run generate_corpus.py.")


def test_manifest_entries_exist():
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for rel in (m.get("generated", []) + m.get("hand_authored", [])):
        assert (CAP_ROOT / rel / "capsule.yaml").is_file(), f"manifest lists missing capsule: {rel}"
