"""A fact artifact must be able to say what produced it, and a launch must refuse a stale one.

`ensure_facts` regenerates only when the cache is COLD, so an artifact written by an older extractor is
served indefinitely. That is a provenance defect: a hardware verdict attributed to the wrong elaboration
is worse than no verdict. The comparison was skipped because it was believed to require a live CIRCT
re-extraction — it does not. Extractor identity and provenance shape are both settled from file hashes.

Measured 2026-09-01: gemmini's live cache records
`{target, hw_mlir, hw_sha, fir_sha, isa_sha, extractor_sha}` and is missing `core_hw_mlir`/`core_hw_sha`,
which the current extractor records — so the facts in use did not name the HW dialect actually read.
"""
from __future__ import annotations

import json

from merlin.targetgen.rtl import circt_introspect as CI
from merlin.targetgen.rtl import facts as F


def _artifact(tmp_path, inputs: dict):
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"schema_version": "2.0", "inputs": inputs, "facts": {}}), encoding="utf-8")
    return p


def _current_sha() -> str:
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(CI.__file__).read_bytes()).hexdigest()[:16]


def _full_inputs(**over) -> dict:
    base = {k: f"<{k}>" for k in CI.INPUT_KEYS}
    base["extractor_sha"] = _current_sha()
    base["extractor_module"] = CI.__name__          # the producer is verified by IMPORTING what it names
    base.update(over)
    return base


def test_a_matching_artifact_is_fresh(tmp_path):
    r = F.freshness("t", path=_artifact(tmp_path, _full_inputs()))
    assert r["status"] == F.FRESH, r["reason"]


def test_an_artifact_from_another_extractor_is_stale(tmp_path):
    r = F.freshness("t", path=_artifact(tmp_path, _full_inputs(extractor_sha="0" * 16)))
    assert r["status"] == F.STALE
    assert "different extractor" in r["reason"]


def test_an_artifact_that_cannot_say_what_it_read_is_stale(tmp_path):
    """The gemmini case: hashes present and matching, but a recorded input KEY is absent."""
    inputs = _full_inputs()
    del inputs["core_hw_sha"]
    r = F.freshness("t", path=_artifact(tmp_path, inputs))
    assert r["status"] == F.STALE
    assert "core_hw_sha" in r["reason"]


def test_an_artifact_with_no_extractor_identity_is_stale(tmp_path):
    inputs = _full_inputs()
    del inputs["extractor_sha"]
    r = F.freshness("t", path=_artifact(tmp_path, inputs))
    assert r["status"] == F.STALE
    assert "cannot be identified" in r["reason"]


def test_an_absent_artifact_is_undeterminable_never_fresh(tmp_path):
    r = F.freshness("t", path=tmp_path / "nope.json")
    assert r["status"] == F.UNDETERMINABLE, "a missing artifact must never read as fresh"


def test_an_unreadable_artifact_is_undeterminable(tmp_path):
    p = tmp_path / "facts.json"
    p.write_text("{ not json", encoding="utf-8")
    r = F.freshness("t", path=p)
    assert r["status"] == F.UNDETERMINABLE


def test_input_keys_match_what_the_extractor_actually_records():
    """INPUT_KEYS must track the `inputs` block, or the shape check silently stops checking."""
    from pathlib import Path
    src = Path(CI.__file__).read_text(encoding="utf-8")
    block = src[src.index('"inputs": {'):]
    block = block[:block.index('"facts":')]
    for key in CI.INPUT_KEYS:
        if key in ("core_hw_mlir", "core_hw_sha"):
            assert "_core_hw_input(" in block, "core_hw_* are recorded via _core_hw_input"
            continue
        assert f'"{key}"' in block, f"INPUT_KEYS names {key!r} but the inputs block does not record it"


# --- the artifact names its own producer ------------------------------------------------------------
# A second archetype has a second extractor. Checking every artifact against the systolic one reported a
# SIMT bundle stale for a reason that was about the CHECKER, so the artifact now names the module that
# produced it and the checker verifies THAT module's identity.


def test_an_artifact_naming_an_unimportable_producer_is_undeterminable(tmp_path):
    """Not stale, and certainly not fresh: nobody could check."""
    r = F.freshness("t", path=_artifact(tmp_path, _full_inputs(extractor_module="no.such.module")))
    assert r["status"] == F.UNDETERMINABLE, r
    assert "not importable" in r["reason"]


def test_a_simt_artifact_is_checked_against_the_simt_extractor(tmp_path):
    """The SIMT producer declares its own INPUT_KEYS; the systolic key set does not apply to it."""
    from merlin.targetgen.rtl import mlc_bridge as MB
    import hashlib
    from pathlib import Path as _P
    sha = hashlib.sha256(_P(MB.__file__).read_bytes()).hexdigest()[:16]
    inputs = {k: f"<{k}>" for k in MB.INPUT_KEYS}
    inputs["extractor_module"] = MB.__name__
    inputs["extractor_sha"] = sha
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"schema_version": "simt-facts/v0", "inputs": inputs, "facts": {}}),
                 encoding="utf-8")
    r = F.freshness("t", path=p)
    assert r["status"] == F.FRESH, r
    assert r["expected"]["extractor"] == MB.__name__


def test_a_simt_artifact_checked_with_a_wrong_sha_is_stale(tmp_path):
    from merlin.targetgen.rtl import mlc_bridge as MB
    inputs = {k: f"<{k}>" for k in MB.INPUT_KEYS}
    inputs["extractor_module"] = MB.__name__
    inputs["extractor_sha"] = "0" * 16
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"schema_version": "simt-facts/v0", "inputs": inputs, "facts": {}}),
                 encoding="utf-8")
    assert F.freshness("t", path=p)["status"] == F.STALE


def test_a_legacy_artifact_naming_no_producer_still_checks_against_the_systolic_one(tmp_path):
    """Back-compat: an artifact predating the field is checked the way it always was."""
    inputs = _full_inputs()
    inputs.pop("extractor_module")
    r = F.freshness("t", path=_artifact(tmp_path, inputs))
    assert r["status"] == F.STALE, r
    assert "extractor_module" in r["reason"], "the missing key is what makes it stale"
