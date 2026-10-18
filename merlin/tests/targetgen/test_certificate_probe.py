"""A certificate probe decides "changed" exactly where the grade does, and never simulates to do it.

``--capsules changed`` grades every public capsule with each elaborated-RTL engine refused just before
dispatch (``simulation_certificate.probe_observer``). The refusal is safe only while three things hold,
and each is asserted against a state that would otherwise run or carry:

  * a certificate for exactly these bytes is still carried, through the engine's own observer;
  * anything else -- a miss, a changed byte, a changed engine selection, a corrupt record, a request
    that could not be issued -- is refused BEFORE the engine starts, as unavailable, never as a pass;
  * a refusal says "already failed" only for a noted failure under exactly the same key.

The note itself is the second half of this file: a failed exact-byte verdict is noted, never carried,
and answers for exactly the bytes that earned it.
"""

from __future__ import annotations

import json

import pytest
from test_compiler_profile_child import _scope, original  # noqa: F401 -- exact issued original-owner fixture
from test_simulation_certificate import SHAS, _context, selected  # noqa: F401 -- same synthetic engine

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import oracle_readback as ORB
from merlin.targetgen import simulation_certificate as SC
from merlin.targetgen import tier_cache as TC
from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

DIGEST_A = "1" * 64
DIGEST_B = "2" * 64
DIGEST_C = "3" * 64


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(tmp_path / "store"))
    monkeypatch.delenv("MERLIN_TIER_CERT_LEDGER", raising=False)


# ---------------------------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------------------------


def _record(case, context, status):
    CR.record_tier_certificate(
        "synthetic",
        "L3",
        CR.TierResult("L3", status, True, engine="gsim", fidelity="elaborated_rtl", derived_from_rtl=True),
        target=case.target.target,
        generated=case.generated,
        shas=SHAS,
        from_rtl=True,
        simulation_context=context,
    )


def _probe(case, monkeypatch, *, engines="gsim"):
    """Dispatch the package ELF through the ladder's own prepared-reuse observer under a probe."""
    monkeypatch.setenv(SC.PROBE_ENV, engines)
    elf = case.generated / PACKAGE_ELF_NAME
    with CR._prepared_tier_reuse(
        "synthetic",
        "L3",
        True,
        target=case.target.target,
        generated=case.generated,
        shas=lambda: SHAS,
        from_rtl=True,
        simulation_context=_context(case),
    ):
        SC.probe_observer("gsim", CR._PREPARED_EXECUTION_OBSERVER.get())(elf)


@pytest.fixture
def built(selected):  # noqa: F811 -- the shared synthetic-engine fixture
    (selected.generated / "kernel.elf").rename(selected.generated / PACKAGE_ELF_NAME)
    return selected


def test_without_a_probe_every_observer_is_returned_unchanged(monkeypatch):
    def observer(_elf):
        return None

    monkeypatch.delenv(SC.PROBE_ENV, raising=False)
    assert SC.probe_observer("gsim", observer) is observer
    assert SC.probe_observer("gsim", None) is None
    monkeypatch.setenv(SC.PROBE_ENV, "gsim")
    assert SC.probe_observer("spike", observer) is observer, "the functional screen is never refused"


def test_a_probed_engine_without_a_certificate_request_is_refused_before_its_build(monkeypatch):
    monkeypatch.setenv(SC.PROBE_ENV, "verilator, gsim")
    with pytest.raises(CR.OracleUnavailable) as refused:
        SC.probe_observer("gsim", None)
    assert str(refused.value) == SC.PROBE_CHANGED


def test_a_certificate_for_exactly_these_bytes_is_carried_not_refused(built, monkeypatch):
    with _scope(built):
        _record(built, _context(built), "pass")
        with pytest.raises(CR._CarriedPreparedTier) as carried:
            _probe(built, monkeypatch)
    assert carried.value.result.status == "pass" and carried.value.result.carried


@pytest.mark.parametrize("change", ["none", "elf", "engine"])
def test_a_miss_is_refused_unexecuted_as_changed(built, monkeypatch, change):
    with _scope(built):
        if change != "none":
            _record(built, _context(built), "pass")
        if change == "elf":
            (built.generated / PACKAGE_ELF_NAME).write_bytes(b"\x7fELF-one-byte-different-program")
        elif change == "engine":
            built.binary.write_bytes(b"byte-bound different engine selection")
        with pytest.raises(CR.OracleUnavailable) as refused:
            _probe(built, monkeypatch)
    assert str(refused.value) == SC.PROBE_CHANGED


def test_a_corrupt_certificate_is_a_refused_miss_never_a_carry(built, monkeypatch):
    with _scope(built):
        _record(built, _context(built), "pass")
        for record in SC.cache_root().glob("*.json"):
            record.write_text("{ truncated")
        with pytest.raises(CR.OracleUnavailable) as refused:
            _probe(built, monkeypatch)
    assert str(refused.value) == SC.PROBE_CHANGED


def test_only_a_noted_failure_for_the_same_key_reads_as_already_failing(built, monkeypatch):
    with _scope(built):
        _record(built, _context(built), "fail")
        with pytest.raises(CR.OracleUnavailable) as refused:
            _probe(built, monkeypatch)
        assert str(refused.value) == SC.PROBE_FAILING
        (built.generated / PACKAGE_ELF_NAME).write_bytes(b"\x7fELF-the-edited-program")
        with pytest.raises(CR.OracleUnavailable) as refused:
            _probe(built, monkeypatch)
        assert str(refused.value) == SC.PROBE_CHANGED


def test_the_readback_adapter_never_dispatches_a_probed_engine(monkeypatch, tmp_path):
    """The wiring: every simulator adapter run goes through ``run_with_readback``."""
    elf = tmp_path / PACKAGE_ELF_NAME
    elf.write_bytes(b"\x7fELF-program")
    dispatched = []

    def run(cb, llvm_text, *, before_execution=None, **_kwargs):
        if before_execution is not None:
            before_execution(elf)
        dispatched.append(cb)
        return {"outputs": {}}

    monkeypatch.setattr(ORB, "_run", run)
    common = {"sim": "gsim", "target": "fixture", "backend": None, "workdir": tmp_path, "timeout": 1, "policy": None}
    monkeypatch.delenv(SC.PROBE_ENV, raising=False)
    ORB.run_with_readback({"run": 1}, "llvm", **common, before_execution=lambda _elf: None)
    monkeypatch.setenv(SC.PROBE_ENV, "gsim")
    for observer in (None, lambda _elf: None):
        with pytest.raises(CR.OracleUnavailable):
            ORB.run_with_readback({"run": 2}, "llvm", **common, before_execution=observer)
    ORB.run_with_readback({"run": 3}, "llvm", **{**common, "sim": "spike"})
    assert dispatched == [{"run": 1}, {"run": 3}]


# ---------------------------------------------------------------------------------------------
# the note
# ---------------------------------------------------------------------------------------------


def _fail(capsule="C0", tier="L3", identity=DIGEST_A, instrument=DIGEST_B):
    return TC.record(capsule, tier, identity, instrument, status="fail", tier_result={"status": "fail"})


def test_a_compared_failure_is_noted_but_never_stored_or_carried():
    assert _fail() is None
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is True
    assert not list(TC.cache_root().glob("*.json")), "a note must not sit among the carryable records"
    [note] = (TC.cache_root() / TC.NOTES).glob("*.json")
    assert "tier_result" not in json.loads(note.read_text()), "a note keeps the key, not the failure"


@pytest.mark.parametrize(
    "other",
    [
        ("C1", "L3", DIGEST_A, DIGEST_B),
        ("C0", "L4", DIGEST_A, DIGEST_B),
        ("C0", "L3", DIGEST_C, DIGEST_B),
        ("C0", "L3", DIGEST_A, DIGEST_C),
    ],
)
def test_a_note_answers_for_exactly_its_own_key(other):
    _fail()
    assert TC.noted_failure(*other) is False


def test_a_certificate_for_the_same_key_outranks_a_note():
    _fail()
    TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result={"status": "pass"})
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is not None
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False


def test_a_corrupt_or_relabelled_note_is_unknown_not_failing():
    _fail()
    [note] = (TC.cache_root() / TC.NOTES).glob("*.json")
    doc = json.loads(note.read_text())
    note.write_text(json.dumps({**doc, "capsule": "SOMEONE_ELSE"}))
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False
    note.write_text("{ truncated")
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False
    note.unlink()
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False


@pytest.mark.parametrize("status", ["unavailable", "incomplete", "skipped"])
def test_only_a_compared_failure_is_noted(status):
    TC.record("C0", "L3", DIGEST_A, DIGEST_B, status=status, tier_result={"status": status})
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False


def test_no_note_without_a_store_or_valid_digests(monkeypatch):
    assert TC.record("C0", "L3", "not-a-digest", DIGEST_B, status="fail", tier_result={}) is None
    assert TC.noted_failure("C0", "L3", "not-a-digest", DIGEST_B) is False
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", "0")
    _fail()
    assert TC.noted_failure("C0", "L3", DIGEST_A, DIGEST_B) is False
