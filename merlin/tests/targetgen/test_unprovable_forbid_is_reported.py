"""A synthesized negative-lane capsule whose forbid the target contradicts is dropped, not fatal.

`forbid: [on_mesh]` says the submission must NOT accelerate this program. That is only a legitimate
demand when the program contains nothing the target may legitimately take -- and whether it does is a
property of the WRITTEN interface, which only exists once the capsule is built.

So the two writers of such a capsule need opposite handling, and the distinction is who authored it:

* A HAND-AUTHORED forbid on such a program is a contradiction its author must resolve. Aborting is
  how they find out, and it is cheap at generation time.
* A SYNTHESIZED one is not an authoring error. `corpus_synth.synthesize` is pure -- it derives entries
  from the requirement without building or classifying anything -- so the host-only axis genuinely
  cannot know that `normalization` decomposes into a reduction and an elementwise map that THIS
  target admits. Measured on gemmini: `SY_host_only_normalization` classifies `A`, and aborting on it
  made a clean regeneration of the whole corpus impossible.

Dropping it silently would be the other failure, and the worse one: the family would vanish between
the requirement and the corpus with nothing saying so. Hence the recorded hole.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "contract" / "capsules"))
import generate_corpus as GC  # noqa: E402


def test_the_synth_role_matches_what_the_synthesizer_actually_stamps():
    """The two constants live in different packages and are compared as data. If they drift, every
    synthesized capsule starts being treated as hand-authored and the abort comes back."""
    from merlin.targetgen import corpus_synth as CS

    assert GC.SYNTH_ROLE == CS.SOURCE_ROLE


def test_an_unprovable_forbid_is_its_own_exception_type():
    """A bare `ValueError` cannot be told apart from the dozen other things a writer raises, so the
    handler would have to match on message text -- which is exactly the brittle string-matching this
    repo forbids."""
    assert issubclass(GC.UnprovableForbid, ValueError)


def test_a_hand_authored_capsule_still_aborts(tmp_path):
    """The half that must NOT become lenient. A hand-written forbid on an accelerable program is a
    claim its author made and can fix; swallowing it would let a capsule ship demanding the compiler
    decline work it is entitled to do."""
    cap = {"name": "HAND_forbids", "lanes": {"forbid": ["on_mesh"]},
           "source_role": "handauthored_compiler_test"}
    with pytest.raises(GC.UnprovableForbid):
        GC._verify_a_forbidden_lane_is_provable(tmp_path, cap, "gemmini")


def test_a_capsule_forbidding_nothing_is_not_examined(tmp_path):
    """The check must not classify every capsule -- `profile_capsule` builds and reads the interface,
    and paying that on a capsule with no forbid would be cost for no claim."""
    GC._verify_a_forbidden_lane_is_provable(tmp_path, {"name": "X", "lanes": {}}, "gemmini")
    GC._verify_a_forbidden_lane_is_provable(tmp_path, {"name": "X"}, None)


def test_the_manifest_records_the_dropped_lane_rather_than_omitting_it(tmp_path):
    """The fail-closed half. A dropped capsule must leave a NAMED hole, on the same two-absence rule
    the roster record follows: an empty list means "every forbid this axis derived is provable",
    while `None` means nobody looked -- and the two may not be spelled the same way."""
    import yaml

    man = tmp_path / "MANIFEST.yaml"
    man.write_text("generated_by: test\n", encoding="utf-8")
    dropped = [{"capsule": "SY_host_only_normalization", "family": "layernorm",
                "reason": "classifies as 'A' rather than host-only"}]
    GC.update_provenance_manifest([], cap_root=tmp_path, target="gemmini",
                                  unprovable_forbids=dropped)
    doc = yaml.safe_load(man.read_text(encoding="utf-8")) or {}
    rec = ((doc.get("lane_generation") or {}).get("gemmini") or {}).get("forbid_not_provable")
    assert rec and rec[0]["capsule"] == "SY_host_only_normalization"
    assert "host-only" in rec[0]["reason"]

    # ...and `None` leaves the record untouched rather than claiming nothing is missing.
    GC.update_provenance_manifest([], cap_root=tmp_path, target="gemmini")
    doc2 = yaml.safe_load(man.read_text(encoding="utf-8")) or {}
    assert ((doc2.get("lane_generation") or {}).get("gemmini") or {}).get("forbid_not_provable") == rec
