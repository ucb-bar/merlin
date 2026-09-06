"""A tier already certified for THESE bytes on THIS instrument is carried, not re-bought.

The cert tier is the whole cost of a grade -- measured on
``out/runs/gemmini/capsule-bench/merlin_assisted/merlincirct_g4p1_20260905`` round 5: 2.53 min of screen
over 82 capsules against 44.4 min of cert over 76, so 94.7% of the oracle wall was the cert tier. A
converged submission re-paid that on every post-turn grade, for capsules whose emitted program had not
changed a byte since the certificate was earned -- sometimes minutes earlier, in the same turn.

This is a CACHE, and a wrong hit is worse than the waste it saves, so the tests that matter are the
misses. Every direction is asserted, and each one is a mutation of a state that HITS -- a cache that
cannot miss is the same defect as a check that cannot fail, and only a mutation can tell them apart:

  * identical bytes + a valid certificate -> carried, and REPORTED as carried (never as measured);
  * one byte of the executable changed         -> re-run;
  * the ledger/store absent, or corrupt        -> re-run;
  * a certificate from a different instrument  -> re-run (grading path moved, or another engine);
  * a stored record whose own copy of the key disagrees -> re-run;
  * a stored FAIL                              -> re-run (a failure is what an agent acts on);
  * the FIRST tier the ladder executes         -> always executed (its adapter is what builds the
                                                  program the key is taken from), so nothing can be
                                                  carried until a real measurement has happened.

The end-to-end test is ``test_ladder_carries_the_cert_tier_and_reruns_after_one_byte``: it drives the
real ladder twice and counts adapter invocations. Counting is the point -- a test that only inspected
the emitted record could not tell a skipped tier from one that ran and was overwritten.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import tier_cache as TC
from merlin.targetgen import tier_policy as _TP
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.oracle_schedule import CERT_LEDGER
from merlin.targetgen.runner_config import RunnerConfig
from merlin.common.paths import merlin_dir, repo_root

CAPS = repo_root() / "merlin/contract/capsules"

# A hardware pin, as `toolchain_shas` spells one: a full 40-char revision. Not any target's actual pin --
# what is under test is the SHAPE the identity accepts, and it must accept nothing shorter.
PIN = "a" * 40
SHAS = {"merlin": "b" * 40, "some_rtl": PIN}
DIGEST_A = "1" * 64
DIGEST_B = "2" * 64


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Every test gets its own store and a cleared instrument memo.

    The memo is process-wide by design (the engine probe is not free), so a test that monkeypatches the
    grading path would otherwise be answered from a previous test's digest -- and would pass for the
    wrong reason.
    """
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(tmp_path / "store"))
    monkeypatch.delenv("MERLIN_TIER_CERT_LEDGER", raising=False)
    TC._INSTRUMENT_MEMO.clear()
    yield
    TC._INSTRUMENT_MEMO.clear()


def _elf(tmp_path, body: bytes = b"\x7fELF-program-one"):
    d = tmp_path / "generated"
    d.mkdir(parents=True, exist_ok=True)
    (d / "package_kernel.elf").write_bytes(body)
    return d


# ---------------------------------------------------------------------------------------------
# 1. ONE identity, shared with the promotion recorder
# ---------------------------------------------------------------------------------------------
def test_execution_identity_is_the_one_promotion_binds_a_certificate_to(tmp_path):
    """``tier_promote.execution_digest`` and ``tier_cache.execution_identity`` must be the same number.

    Property 1 of the design: the cache is keyed on the identity ``record_cert`` already binds a
    certificate to, not on a second one invented for the reader. Two implementations would not fail
    loudly -- they would differ in whichever detail drifted first, and the cache would either never hit
    or (far worse) hit for bytes the recorder meant something else by.
    """
    import importlib.util
    import sys
    import yaml

    spec = importlib.util.spec_from_file_location(
        "tier_promote", merlin_dir() / "experiments/capsule_bench/harness/tier_promote.py")
    tp = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("tier_promote", tp)
    spec.loader.exec_module(tp)

    run = tmp_path / "run"
    generated = _elf(run)
    (run / "capsule_result.json").write_text(json.dumps({"toolchain_shas": SHAS}))
    (run / "run_manifest.yaml").write_text(yaml.safe_dump({"target": "t"}))

    from_promotion = tp.execution_digest(run / "capsule_result.json")
    from_cache = TC.execution_identity(target="t",
                                       executable=generated / "package_kernel.elf",
                                       toolchain_shas=SHAS)
    assert from_promotion is not None, "the promotion identity must still be computable"
    assert from_promotion == from_cache, (
        "the cache must key on the SAME identity the promotion ledger binds a certificate to")


def test_execution_identity_moves_with_one_byte_of_the_executable(tmp_path):
    a = TC.execution_identity(target="t", executable=_elf(tmp_path, b"AAAA") / "package_kernel.elf",
                              toolchain_shas=SHAS)
    b = TC.execution_identity(target="t", executable=_elf(tmp_path, b"AAAB") / "package_kernel.elf",
                              toolchain_shas=SHAS)
    assert a and b and a != b, "one changed byte of the program must be a different identity"


def test_execution_identity_moves_with_the_hardware_revision(tmp_path):
    elf = _elf(tmp_path) / "package_kernel.elf"
    a = TC.execution_identity(target="t", executable=elf, toolchain_shas=SHAS)
    b = TC.execution_identity(target="t", executable=elf,
                              toolchain_shas={"merlin": SHAS["merlin"], "some_rtl": "c" * 40})
    assert a and b and a != b, "a certificate is about one device revision; another is not the same"


def test_execution_identity_ignores_merlins_own_commit(tmp_path):
    """An edit that emits a byte-identical program on the same device has not changed the program."""
    elf = _elf(tmp_path) / "package_kernel.elf"
    a = TC.execution_identity(target="t", executable=elf, toolchain_shas=SHAS)
    b = TC.execution_identity(target="t", executable=elf,
                              toolchain_shas={"merlin": "f" * 40, "some_rtl": PIN})
    assert a == b, "keying on merlin's commit would invalidate every certificate on every edit"


@pytest.mark.parametrize("target,shas,write_elf", [
    (None, SHAS, True),                       # no target
    ("", SHAS, True),                         # no target
    ("t", {"merlin": "b" * 40}, True),        # merlin only: no hardware revision at all
    ("t", {"merlin": "b" * 40, "rtl": "UNKNOWN"}, True),   # a pin nobody could resolve
    ("t", {"merlin": "b" * 40, "rtl": "abc123"}, True),    # an abbreviated pin
    ("t", SHAS, False),                       # no executable
    ("t", None, True),                        # no provenance block
])
def test_execution_identity_fails_closed(tmp_path, target, shas, write_elf):
    """Every missing or imprecise input answers ``None``, which makes the tier re-run."""
    if write_elf:
        elf = _elf(tmp_path) / "package_kernel.elf"
    else:
        elf = tmp_path / "generated" / "package_kernel.elf"
    assert TC.execution_identity(target=target, executable=elf, toolchain_shas=shas) is None


# ---------------------------------------------------------------------------------------------
# 2. the instrument: the judge is part of the key
# ---------------------------------------------------------------------------------------------
def _fixed_instrument(monkeypatch, files, engine="engine-x"):
    monkeypatch.setattr(TC, "grading_path", lambda target=None: tuple(files))
    monkeypatch.setattr(TC, "_engine_token", lambda target, tier, rtl_tier: engine)
    TC._INSTRUMENT_MEMO.clear()


def test_instrument_digest_moves_when_the_grading_path_bytes_move(tmp_path, monkeypatch):
    """A verdict is a description of a judge. Edit the judge and the old verdict is about someone else.

    ``source_digest`` hashes the bytes actually READ, so a dirty working tree is already a different
    instrument -- which is the property ``BASELINE.json``'s ``grading_path_digest`` exists to record.
    """
    f = tmp_path / "grader.py"
    f.write_text("def verdict(): return 'pass'\n")
    _fixed_instrument(monkeypatch, [f])
    before = TC.instrument_digest("t", "L3", rtl_tier=True)
    f.write_text("def verdict(): return 'pass'  # one comment\n")
    TC._INSTRUMENT_MEMO.clear()
    after = TC.instrument_digest("t", "L3", rtl_tier=True)
    assert before and after and before != after


def test_instrument_digest_distinguishes_the_engine_that_answers(tmp_path, monkeypatch):
    """A tier is a FIDELITY, not a binary: two engines at one fidelity are two instruments."""
    f = tmp_path / "grader.py"
    f.write_text("x\n")
    _fixed_instrument(monkeypatch, [f], engine="engine-x")
    a = TC.instrument_digest("t", "L3", rtl_tier=True)
    _fixed_instrument(monkeypatch, [f], engine="engine-y")
    b = TC.instrument_digest("t", "L3", rtl_tier=True)
    assert a and b and a != b


def test_instrument_digest_fails_closed_without_a_grading_path_or_an_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "grading_path", lambda target=None: None)
    monkeypatch.setattr(TC, "_engine_token", lambda target, tier, rtl_tier: "e")
    TC._INSTRUMENT_MEMO.clear()
    assert TC.instrument_digest("t", "L3", rtl_tier=True) is None

    f = tmp_path / "g.py"
    f.write_text("x\n")
    monkeypatch.setattr(TC, "grading_path", lambda target=None: (f,))
    monkeypatch.setattr(TC, "_engine_token", lambda target, tier, rtl_tier: None)
    TC._INSTRUMENT_MEMO.clear()
    assert TC.instrument_digest("t", "L3", rtl_tier=True) is None, (
        "an RTL tier whose engine cannot be established has no instrument identity")


def test_every_declared_grading_module_exists():
    """A rename would not fail loudly. ``grading_path`` returns ``None`` when a member has moved, which
    switches the cache off for the whole repo -- the conservative direction, and a silent one. This is
    the check that makes the rename visible instead."""
    missing = [rel for rel in TC._GRADING_MODULES if not (repo_root() / rel).is_file()]
    assert not missing, f"grading-path members have moved: {missing}"
    assert "merlin/python/merlin/targetgen/capsule_runner.py" in TC._GRADING_MODULES, (
        "the ladder decides what a tier verdict means; its bytes must be in the instrument")
    assert "merlin/python/merlin/targetgen/tier_cache.py" in TC._GRADING_MODULES, (
        "this module decides what a HIT means; an edit to it must not be carried across")


def test_grading_path_includes_the_targets_own_backend_and_refuses_an_unknown_one():
    """Derived from the backend registry, never named here -- and ``None`` when it cannot be resolved."""
    got = TC.grading_path("gemmini")            # target-ok: a test may name the target it derives from
    if got is None:
        pytest.skip("this environment cannot resolve a backend")
    assert any("targets" in p.parts for p in got), (
        "the target's own backend decides what a console means; its bytes are part of the instrument")
    assert TC.grading_path("no-such-target-exists") is None, (
        "an unresolvable backend must yield no instrument identity, not a partial one")


# ---------------------------------------------------------------------------------------------
# 3. the store: the mutation matrix
# ---------------------------------------------------------------------------------------------
RESULT = {"status": "pass", "cycles": 116, "mandatory": True, "derived_from_rtl": True,
          "evidence": "engine_console.log", "timing": {"sim_active_s": 33.5},
          "concurrency": {"workers": 16}}


def test_a_stored_pass_is_found_for_the_same_bytes_and_the_same_instrument():
    assert TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result=RESULT) is not None
    hit = TC.lookup("C0", "L3", DIGEST_A, DIGEST_B)
    assert hit is not None and hit["status"] == "pass"
    assert hit["tier_result"]["cycles"] == 116, "a cycle count is a property of the program + device"
    for dropped in TC._NOT_MEASURED_NOW:
        assert dropped not in hit["tier_result"], (
            f"{dropped!r} describes an act of measurement that did not happen now")


@pytest.mark.parametrize("capsule,tier,identity,instrument", [
    ("C0", "L3", DIGEST_A, "3" * 64),         # a different instrument: the judge moved
    ("C0", "L3", "3" * 64, DIGEST_B),         # a different program: one byte changed
    ("C0", "L4", DIGEST_A, DIGEST_B),         # a different tier
    ("C1", "L3", DIGEST_A, DIGEST_B),         # a different capsule
    ("C0", "L3", None, DIGEST_B),             # no identity at all
    ("C0", "L3", DIGEST_A, None),             # no instrument at all
    ("C0", "L3", "not-a-digest", DIGEST_B),   # a malformed identity
])
def test_every_mutation_of_the_key_is_a_miss(capsule, tier, identity, instrument):
    TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result=RESULT)
    assert TC.lookup(capsule, tier, identity, instrument) is None


def test_an_absent_store_is_a_miss():
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None, "nothing recorded: run the tier"


def test_a_corrupt_record_is_a_miss():
    path = TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result=RESULT)
    path.write_text("{ this is not json")
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None


def test_a_record_that_disagrees_with_its_own_key_is_a_miss():
    """Reached by the right path, but its own copy of the key says it belongs to something else.

    A store is a file tree anyone can touch. Re-verifying every key field against the record means a
    file reached by a wrong path -- a hand edit, a half-written swap, a collision -- is a MISS rather
    than a hit for the wrong capsule.
    """
    path = TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result=RESULT)
    doc = json.loads(path.read_text())
    doc["capsule"] = "SOMEONE_ELSE"
    path.write_text(json.dumps(doc))
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None


def test_a_failure_is_never_stored_and_never_carried():
    """A failure is what an agent acts on -- its plane, its category, its first mismatch. A record
    carrying only the word "fail" would replace actionable feedback with an assertion."""
    assert TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="fail",
                     tier_result={"status": "fail"}) is None
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None


def test_switching_the_cache_off_never_hits(monkeypatch):
    TC.record("C0", "L3", DIGEST_A, DIGEST_B, status="pass", tier_result=RESULT)
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is not None
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", "0")
    assert TC.cache_root() is None
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None
    assert TC.record("C0", "L4", DIGEST_A, DIGEST_B, status="pass", tier_result=RESULT) is None


# ---------------------------------------------------------------------------------------------
# 4. the promotion ledger, which nothing used to read
# ---------------------------------------------------------------------------------------------
def _ledger(tmp_path, entry) -> str:
    p = tmp_path / "tier_state.json"
    p.write_text(json.dumps({"C0": {CERT_LEDGER: {"L3": {DIGEST_A: entry}}},
                             "C0_mirror_only": {"L3": entry}}))
    return str(p)


def test_a_promotion_certificate_is_carried(tmp_path, monkeypatch):
    """The measured waste this exists to remove: an async promotion earns a certificate on real RTL and
    the next grade re-buys it. The ledger was written and nothing read it."""
    monkeypatch.setenv("MERLIN_TIER_CERT_LEDGER",
                       _ledger(tmp_path, {"status": "pass", "execution_digest": DIGEST_A,
                                          "instrument": DIGEST_B}))
    hit = TC.lookup("C0", "L3", DIGEST_A, DIGEST_B)
    assert hit is not None and hit["status"] == "pass"
    assert "ledger" in str(hit.get("source")), "a carry must say where the verdict came from"
    assert hit["tier_result"] == {}, (
        "a ledger entry holds no tier record; inventing cycles for it would be a fabricated measurement")


@pytest.mark.parametrize("entry,why", [
    ({"status": "pass", "execution_digest": DIGEST_A},
     "an entry that never recorded its instrument -- every entry written before the reader existed"),
    ({"status": "pass", "execution_digest": DIGEST_A, "instrument": "3" * 64},
     "a certificate earned under a different judge"),
    ({"status": "pending", "execution_digest": DIGEST_A, "instrument": DIGEST_B},
     "an in-flight job is not a verdict"),
    ({"status": "fail", "execution_digest": DIGEST_A, "instrument": DIGEST_B},
     "a failure is re-run for its detail"),
    ({"status": "pass", "execution_digest": "3" * 64, "instrument": DIGEST_B},
     "an entry whose own identity disagrees with the slot it sits in"),
])
def test_every_unusable_ledger_entry_is_a_miss(tmp_path, monkeypatch, entry, why):
    monkeypatch.setenv("MERLIN_TIER_CERT_LEDGER", _ledger(tmp_path, entry))
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None, why


def test_a_corrupt_or_absent_ledger_is_a_miss(tmp_path, monkeypatch):
    bad = tmp_path / "tier_state.json"
    bad.write_text("{ truncated")
    monkeypatch.setenv("MERLIN_TIER_CERT_LEDGER", str(bad))
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None
    monkeypatch.setenv("MERLIN_TIER_CERT_LEDGER", str(tmp_path / "nope.json"))
    assert TC.lookup("C0", "L3", DIGEST_A, DIGEST_B) is None


def test_the_ledger_key_has_one_spelling():
    """``tier_promote`` writes this file and ``tier_cache`` reads it. Two spellings of the reserved key
    would not fail loudly: the reader would find nothing and re-buy every certificate, which looks
    exactly like a run with nothing to reuse."""
    src = (merlin_dir() / "experiments/capsule_bench/harness/tier_promote.py").read_text()
    assert "CERT_LEDGER as _LEDGER" in src, "tier_promote must IMPORT the key, not restate it"
    assert '_LEDGER = "<certs>"' not in src, "a second spelling of the ledger key has reappeared"


# ---------------------------------------------------------------------------------------------
# 5. a carried verdict says so
# ---------------------------------------------------------------------------------------------
def test_a_carried_tier_record_is_never_presented_as_freshly_measured(tmp_path, monkeypatch):
    f = tmp_path / "g.py"
    f.write_text("x\n")
    _fixed_instrument(monkeypatch, [f])
    generated = _elf(tmp_path)
    identity = TC.execution_identity(target="t", executable=generated / "package_kernel.elf",
                                     toolchain_shas=SHAS)
    instrument = TC.instrument_digest("t", "L3", rtl_tier=True)
    TC.record("C0", "L3", identity, instrument, status="pass", tier_result=RESULT, run_id="earlier")

    got = CR.carried_tier_result("C0", "L3", True, target="t", generated=generated, shas=SHAS,
                                 from_rtl=True)
    assert got is not None and got.status == "pass"
    d = got.to_dict()
    assert d["measured_now"] is False
    assert d["carried"]["carried"] is True
    assert d["carried"]["execution_identity"] == identity
    assert d["carried"]["instrument"] == instrument
    assert d["carried"]["earned_by_run"] == "earlier"
    assert "carried" in (d["reason"] or ""), "the record must SAY it was not executed"
    assert d["cycles"] == 116, "cycles are a property of the program and the device"
    assert d.get("timing") is None and d.get("concurrency") is None, (
        "no time was spent measuring now; a copied duration would be a fabricated measurement")
    assert d.get("evidence") is None, "the console file belongs to the run that earned the verdict"
    assert d["carried"]["earned_evidence"]["evidence"] == "engine_console.log", (
        "and it must still be findable, in the block that says where it lives")


def test_an_executed_record_asserts_that_it_was_measured():
    """Both states are asserted, so a reader never has to infer 'measured' from a missing key."""
    assert CR.TierResult("L3", "pass", True).to_dict()["measured_now"] is True


def test_reuse_block_states_both_lists():
    fresh = CR.TierResult("L2", "pass", True).to_dict()
    carried = CR.TierResult("L3", "pass", True, carried={"carried": True}).to_dict()
    block = TC.reuse_block({"L2": fresh, "L3": carried})
    assert block == {"executed": ["L2"], "carried": ["L3"], "note": block["note"]}
    assert TC.reuse_block({"L2": fresh})["carried"] == [], (
        "the accounting is emitted even when nothing was reused, or a cached grade and a fresh one "
        "read the same")


# ---------------------------------------------------------------------------------------------
# 6. end to end: the ladder itself
# ---------------------------------------------------------------------------------------------
def _two_tier_config() -> RunnerConfig:
    """A screen tier and a cert tier. Shape, not identity: the target string is a label here."""
    return RunnerConfig(
        target="cachetest", suite="cachetest-capsule-bench", dtype="fp8_e4m3",
        fourth_output_name="kernel.S", tier_sim={"L2": "screen-sim", "L3": "cert-sim"},
        rtl_tiers=frozenset({"L3"}), oracle_tiers=("L2", "L3"), perf_fields=(), trace_gate=None)


@pytest.fixture
def _ladder(monkeypatch, tmp_path):
    """The ladder with its front half stubbed and a deterministic instrument.

    ``run_entrypoints`` and the numeric comparison are not what is under test; what is under test is
    whether the ladder pays for a tier it already holds a certificate for.
    """
    cb = {"tensors": {"Y0": {"role": "output", "base": 0, "shape": [32, 32], "dtype": "bf16"}}}
    monkeypatch.setattr(CR, "run_entrypoints", lambda *a, **k: (object(), cb, "# kernel.S\n"))
    # PIN THE LADDER ORDER. `tier_policy.tier_order` is cost-driven and puts never-yet-measured tiers
    # FIRST, so once the first grade has priced one tier the second grade legitimately reorders -- which
    # would silently turn this into a test of a different question. The ORDER is not what is under test.
    monkeypatch.setattr(_TP, "tier_order", lambda target, tiers: [t for t in ("L2", "L3") if t in set(tiers)])
    monkeypatch.setattr(CR, "_match_by_policy", lambda *a, **k: True)
    monkeypatch.setattr(CR.CG, "compare", lambda *a, **k: {
        "status": "pass", "policy": "p", "max_abs_error": 0, "max_rel_error": 0,
        "mismatch_count": 0, "first_mismatch": None, "per_output": {}})
    # `run_capsule` imports this at call time from merlin.targetgen.provenance; patch it there.
    from merlin.targetgen import provenance as _PROV
    monkeypatch.setattr(_PROV, "toolchain_shas", lambda *a, **k: dict(SHAS))
    f = tmp_path / "grader.py"
    f.write_text("judge\n")
    _fixed_instrument(monkeypatch, [f])
    return f


def _adapters(program: bytes, calls: dict):
    """A screen adapter that BUILDS the program (as every real one does) and a cert adapter that counts."""
    def screen(cb, llvm_text, workdir, timeout):
        calls["L2"] = calls.get("L2", 0) + 1
        (workdir).mkdir(parents=True, exist_ok=True)
        (workdir / "package_kernel.elf").write_bytes(program)
        return {"outputs": {}, "cycles": 7, "oracle": "screen", "console": "DONE\n"}

    def cert(cb, llvm_text, workdir, timeout):
        calls["L3"] = calls.get("L3", 0) + 1
        return {"outputs": {}, "cycles": 4242, "oracle": "cert", "console": "DONE\n"}

    return {"L2": screen, "L3": cert}


def _capsule():
    cap = dict(load_capsule(CAPS / "atlas/isa/AT2_single_tile_matmul", contract="merlin/contract"))
    cap["required_oracle_tiers"] = ["L2", "L3"]
    return cap


def _grade(tmp_path, run_id, program, calls):
    return CR.run_capsule(_capsule(), "unused-package", runs_root=tmp_path / run_id, run_id=run_id,
                          config=_two_tier_config(), oracle_adapters=_adapters(program, calls))


def test_ladder_carries_the_cert_tier_and_reruns_after_one_byte(tmp_path, _ladder):
    """The whole point, and its falsifier, in one test.

    Grade the same program twice: the second grade must NOT pay for the cert tier, and must say it
    carried it. Then change ONE BYTE of the emitted program and grade again: the cert tier must be
    re-executed. Counting adapter calls is what makes this a real assertion -- a test that inspected
    only the emitted record could not tell a skipped tier from one that ran and was overwritten.
    """
    calls: dict = {}
    first = _grade(tmp_path, "g1", b"\x7fELF-program-one", calls)
    assert first["status"] == "pass", first.get("failure")
    assert calls == {"L2": 1, "L3": 1}, "the first grade pays for both tiers"
    assert first["tier_reuse"] == {"executed": ["L0", "L1", "L2", "L3"], "carried": [],
                                   "note": first["tier_reuse"]["note"]}

    second = _grade(tmp_path, "g2", b"\x7fELF-program-one", calls)
    assert second["status"] == "pass", second.get("failure")
    assert calls == {"L2": 2, "L3": 1}, (
        "the cert tier was re-bought for a program that had not changed a byte")
    assert second["tier_reuse"]["carried"] == ["L3"]
    assert "L3" not in second["tier_reuse"]["executed"]
    assert second["tiers"]["L3"]["measured_now"] is False
    assert second["tiers"]["L3"]["carried"]["carried"] is True
    assert second["tiers"]["L3"]["cycles"] == 4242, "the certified cycle count is carried with it"
    assert second["tiers"]["L2"]["measured_now"] is True, (
        "the first tier the ladder executes is always paid for: its adapter builds the program the "
        "key is taken from, so nothing can be carried until it has run")

    third = _grade(tmp_path, "g3", b"\x7fELF-program-TWO", calls)
    assert calls == {"L2": 3, "L3": 2}, (
        "one changed byte of the emitted program must re-run the cert tier")
    assert third["tier_reuse"]["carried"] == []


def test_ladder_reruns_when_the_instrument_moves(tmp_path, _ladder, monkeypatch):
    """A certificate earned under a different grading path does not describe today's judge."""
    calls: dict = {}
    _grade(tmp_path, "g1", b"\x7fELF-program-one", calls)
    assert calls == {"L2": 1, "L3": 1}

    _ladder.write_text("judge  # the grading path moved\n")
    TC._INSTRUMENT_MEMO.clear()
    _grade(tmp_path, "g2", b"\x7fELF-program-one", calls)
    assert calls == {"L2": 2, "L3": 2}, "the cert tier must be re-run for a different instrument"


def test_ladder_reruns_when_the_store_is_purged(tmp_path, _ladder, monkeypatch):
    calls: dict = {}
    _grade(tmp_path, "g1", b"\x7fELF-program-one", calls)
    import shutil
    shutil.rmtree(TC.cache_root())
    _grade(tmp_path, "g2", b"\x7fELF-program-one", calls)
    assert calls == {"L2": 2, "L3": 2}, "a purged cache re-runs; it never assumes"


def test_ladder_reruns_when_every_stored_record_is_corrupt(tmp_path, _ladder):
    calls: dict = {}
    _grade(tmp_path, "g1", b"\x7fELF-program-one", calls)
    for f in TC.cache_root().glob("*.json"):
        f.write_text("{ not json")
    _grade(tmp_path, "g2", b"\x7fELF-program-one", calls)
    assert calls == {"L2": 2, "L3": 2}, "an unreadable record is a miss, never a hit"


def test_ladder_never_carries_when_the_cache_is_off(tmp_path, _ladder, monkeypatch):
    calls: dict = {}
    _grade(tmp_path, "g1", b"\x7fELF-program-one", calls)
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", "0")
    res = _grade(tmp_path, "g2", b"\x7fELF-program-one", calls)
    assert calls == {"L2": 2, "L3": 2}
    assert res["tier_reuse"]["carried"] == []


def test_a_stale_executable_from_an_earlier_grade_cannot_be_keyed_on(tmp_path, _ladder):
    """The one direction that would be catastrophic: a REUSED run directory still holds the previous
    grade's ELF, so a lookup taken before anything has been built in this run would key on a program the
    current compiler no longer emits. The ladder removes it and asks nothing until a tier has run."""
    calls: dict = {}
    _grade(tmp_path, "same", b"\x7fELF-program-one", calls)
    # Same runs_root and run_id: the previous grade's generated/ directory is still there.
    res = _grade(tmp_path, "same", b"\x7fELF-program-TWO", calls)
    assert calls["L2"] == 2, "the first tier must be paid for even into a reused run directory"
    assert calls["L3"] == 2, "the new program must be certified, not carried on the old one's identity"
    assert res["tier_reuse"]["carried"] == []
