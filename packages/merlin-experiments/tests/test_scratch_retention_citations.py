"""Grade scratch retention keeps every file a record outside the grade cites, and fails closed.

Study ledgers and papers quote a grade's files by path (harnesses, responses). Retention reclaims
build products by size and suffix, which says nothing about whether anything still reads them; the
citation registry (merlin.common.cited_evidence) does.
"""

import json

from merlin_experiments.phase1.feedback import scratch_retention

from merlin.common import cited_evidence


def _capsule(scratch, name):
    """A capsule run dir shaped like the grader's: a result plus a generated/ build tree."""
    capsule = scratch / "suite" / name
    generated = capsule / "generated"
    (generated / "compiler-source" / ("a" * 64)).mkdir(parents=True)
    (generated / "compiler-source" / ("a" * 64) / "lower.py").write_text("x = 1\n")
    (generated / "harness.c").write_bytes(b"0," * (scratch_retention.KEEP_FILE_BYTES + 1))
    (generated / "kernel.o").write_bytes(b"obj")
    tiers = {"L2": {"status": "pass", "engine": "spike", "derived_from_rtl": False}}
    (capsule / "capsule_result.json").write_text(json.dumps({"capsule": name, "tiers": tiers}))
    return capsule


def test_slim_keeps_what_a_ledger_outside_the_grade_cites(tmp_path, monkeypatch):
    scratch = tmp_path / "fruns_1"
    capsule = _capsule(scratch, "cited")
    harness = capsule / "generated" / "harness.c"
    ledger = tmp_path / "paper" / "ledger"
    ledger.mkdir(parents=True)
    (ledger / "harvest-receipt.json").write_text(json.dumps({"read": [str(harness)]}))
    # The grade's own working receipt citing its object is not a seal, release or run record of it.
    receipt = {"object": str(capsule / "generated" / "kernel.o")}
    (capsule / "generated" / "build-receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setenv(cited_evidence.ROOTS_ENV, str(tmp_path))
    cited_evidence.clear()
    try:
        assert scratch_retention.slim_tree(scratch) > 0
    finally:
        cited_evidence.clear()
    assert harness.is_file(), "slim deleted a file a ledger outside the grade cites"
    assert not (capsule / "generated" / "kernel.o").exists()
    assert not (capsule / "generated" / "compiler-source").exists()


def test_slim_removes_nothing_when_the_citation_registry_is_unreadable(tmp_path, monkeypatch):
    scratch = tmp_path / "fruns_2"
    capsule = _capsule(scratch, "unknown")
    monkeypatch.setenv(cited_evidence.ROOTS_ENV, str(tmp_path))

    def unreadable(*_args, **_kwargs):
        raise OSError("registry volume unavailable")

    monkeypatch.setattr(cited_evidence, "registry", unreadable)
    assert scratch_retention.slim_tree(scratch) == 0
    assert scratch_retention.slim_once(scratch, scope="test") == 0
    assert (capsule / "generated" / "harness.c").is_file()
    assert not (scratch / scratch_retention.MARKER).exists(), "an unslimmed root was marked done"


def test_a_cited_selfcheck_temporary_is_kept(tmp_path, monkeypatch):
    copy = tmp_path / "selfcheck_iso_x"
    (copy / "submission").mkdir(parents=True)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "study-receipt.json").write_text(json.dumps({"inputs": [str(copy / "submission")]}))
    monkeypatch.setenv(cited_evidence.ROOTS_ENV, str(ledger))
    cited_evidence.clear()

    @scratch_retention.releases_selfcheck_scratch
    def check():
        scratch_retention.temporary(copy)

    try:
        check()
    finally:
        cited_evidence.clear()
    assert (copy / "submission").is_dir(), "a self-check removed a temporary a record cites"
