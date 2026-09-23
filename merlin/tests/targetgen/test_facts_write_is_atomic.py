"""A facts artifact is never observed half-written.

WHY THIS EXISTS, measured 2026-09-19. `write_facts_guarded` already refused a regeneration that would
HOLLOW an artifact — the semantic downgrade, where an extractor ran with a toolchain unset and read
less than it should have. It did not cover the mechanical case. Its last line was a plain
`write_text`, so an interrupted write left a TRUNCATED file; a truncated artifact is not empty, and
`_has_facts` asks only whether the `facts` body is non-empty, so the fragment was accepted as a cache
hit. A derived `target_contract.yaml` then declared an `endpoint_kind` its own provenance string
contradicted, and it cost eleven tests that looked exactly like a regression.

The fix is to make the partial file impossible rather than to teach the reader to recognise one.
"Complete" is a property of each extractor's own output and would need redefining every time a fact is
added; these artifacts are regenerable by construction, so the only thing that must never happen is a
half-written one being mistaken for a whole one.
"""

from __future__ import annotations

import json

import pytest

from merlin.targetgen.rtl import facts as F

pytestmark = pytest.mark.target("gemmini")

WHOLE = {"facts": {"target": "demo", "instruction_classes": ["a", "b"], "arrays": {"mesh": {"rows": 16}}}}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheWriteIsAllOrNothing:
    def test_a_normal_write_lands(self, tmp_path):
        p = tmp_path / "facts.json"
        F.write_facts_guarded(p, WHOLE)
        assert _read(p) == WHOLE

    def test_an_interrupted_write_leaves_the_previous_document_intact(self, tmp_path, monkeypatch):
        """THE MUTATION. Kill the write midway and the reader must still see the OLD facts — not a
        fragment, and not an absence."""
        p = tmp_path / "facts.json"
        F.write_facts_guarded(p, WHOLE)

        class Boom(RuntimeError):
            pass

        real_replace = F.os.replace

        def die(*_args, **_kwargs):
            raise Boom("interrupted between write and publish")

        monkeypatch.setattr(F.os, "replace", die)
        with pytest.raises(Boom):
            F.write_facts_guarded(p, WHOLE | {"facts": dict(WHOLE["facts"], extra="new")}, allow_downgrade=True)
        monkeypatch.setattr(F.os, "replace", real_replace)

        assert _read(p) == WHOLE, "an interrupted regeneration must not disturb the artifact on disk"

    def test_an_interrupted_write_leaves_no_fragment_behind(self, tmp_path, monkeypatch):
        """A surviving temp file would reintroduce exactly the artifact this prevents. Anything that
        does survive must not be readable as a facts document."""
        p = tmp_path / "facts.json"

        def die(*_args, **_kwargs):
            raise RuntimeError("interrupted")

        monkeypatch.setattr(F.os, "replace", die)
        with pytest.raises(RuntimeError):
            F.write_facts_guarded(p, WHOLE)

        assert not p.exists(), "the target must not be created by a failed write"
        leftovers = [q.name for q in tmp_path.iterdir()]
        assert leftovers == [], f"a fragment survived: {leftovers}"

    def test_the_reader_would_have_accepted_a_truncated_body(self, tmp_path):
        """The reason the write has to be atomic, pinned as its own fact rather than argued in prose.

        `_has_facts` is a NON-EMPTINESS test, not a completeness test — deliberately, because
        completeness is per-extractor. So a truncated document with a surviving `facts` key passes it,
        which is precisely why the writer must never be able to produce one.
        """
        truncated = {"facts": {"target": "demo"}}  # lost instruction_classes and arrays mid-write
        assert F._has_facts(truncated) is True
        assert F._has_facts({"facts": {}}) is False, "an empty body is still correctly a miss"


class TestTheSemanticGuardStillHolds:
    def test_a_regeneration_that_hollows_an_artifact_is_still_refused(self, tmp_path):
        """The case that was already covered. Atomicity must not have loosened it."""
        p = tmp_path / "facts.json"
        F.write_facts_guarded(p, WHOLE)
        with pytest.raises(F.FactsDowngrade) as excinfo:
            F.write_facts_guarded(p, {"facts": {"target": "demo", "instruction_classes": []}})
        assert "hollow" in str(excinfo.value).lower()

    def test_allow_downgrade_still_overrides_it(self, tmp_path):
        p = tmp_path / "facts.json"
        F.write_facts_guarded(p, WHOLE)
        thin = {"facts": {"target": "demo", "instruction_classes": []}}
        F.write_facts_guarded(p, thin, allow_downgrade=True)
        assert _read(p) == thin
