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
    # This MANIFEST describes THIS (gemmini) corpus — its capsules sit at <category>/<cap> (rel-depth 2).
    # Another target's corpus nested under the same root (e.g. atlas/<category>/<cap>, depth 3) has its own
    # provenance and must not be conflated here.
    #
    # HIDDEN capsules are excluded, and a hidden capsule that appears here is a FAILURE.
    #
    # MANIFEST.yaml is tracked; the hidden capsules are deliberately not (0 of 11 on this tree). The
    # completeness rule and the answer-key rule collide on exactly this set, and secrecy wins: listing
    # them would publish the holdout's composition -- which capsules a submission is measured on that
    # it cannot see -- in a file anyone who clones the repo reads. A submitter who knows the holdout
    # is "matmul, acc_scale, k_accum, movement, conv" can target it, and the set stops measuring
    # generalization. Keyed on the capsule's OWN `label: hidden`, not on the directory name, so a
    # hidden capsule filed anywhere is still excluded.
    #
    # This also explains why the check never fired before: it passes on a machine WITHOUT the hidden
    # set and fails on one that has it, which is backwards for a rule about the hidden set.
    def _label(cap_yaml: Path) -> str:
        try:
            return str((yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}).get("label", ""))
        except Exception:                                  # noqa: BLE001 -- unreadable != hidden
            return ""

    on_disk, hidden_on_disk = set(), set()
    for p in CAP_ROOT.rglob("capsule.yaml"):
        rel = p.parent.relative_to(CAP_ROOT)
        if len(rel.parts) != 2:
            continue
        (hidden_on_disk if _label(p) == "hidden" else on_disk).add(str(rel))

    leaked = listed & hidden_on_disk
    assert not leaked, (
        f"MANIFEST names hidden capsule(s): {sorted(leaked)}. MANIFEST.yaml is tracked and the hidden "
        f"set is not; naming them publishes the holdout's composition.")
    assert listed == on_disk, (
        f"MANIFEST out of sync with the tree — missing: {on_disk - listed}; "
        f"stale: {listed - on_disk}. Re-run generate_corpus.py.")


def test_manifest_entries_exist():
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for rel in (m.get("generated", []) + m.get("hand_authored", [])):
        assert (CAP_ROOT / rel / "capsule.yaml").is_file(), f"manifest lists missing capsule: {rel}"
