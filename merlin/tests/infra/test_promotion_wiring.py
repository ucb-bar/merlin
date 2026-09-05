"""The broker must actually promote — the policy being right is not the same as the wiring firing.

`oracle_schedule` is tested as pure policy and `cert_capsule_cover` as pure selection, but neither proves
that a real verdict landing in the broker's reap produces a real cert-tier job. That gap is exactly where
this kind of change dies quietly: promotion is wrapped in a `try/except` so it can never gate a run, so a
broken wiring would show up as *nothing happening* — indistinguishable from a corpus where nothing needed
promoting.

These tests drive the broker's own functions against a temp workspace, no oracle and no agent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _broker():
    """Import the shared promotion module by path — it is a harness script, not installed."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("tier_promote", HARNESS / "tier_promote.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"tier_promote not importable here: {type(e).__name__}: {e}")
    return mod


def _ws(tmp_path, files=(("submission/manifest.yaml", "x: 1"),)):
    ws = tmp_path / "ws"
    for rel, body in files:
        f = ws / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    (ws / ".qa_channel").mkdir(parents=True, exist_ok=True)
    return ws


def _verdict(rows):
    return {"per_capsule": [{"capsule": n, "pass": p} for n, p in rows]}


def _execution_verdict(rows):
    """Rows are ``(capsule, passed, execution-byte)``; sha-shaped IDs exercise the real gate."""
    return {"per_capsule": [
        {"capsule": n, "pass": p, "execution_digest": c * 64} for n, p, c in rows
    ]}


def _capsule_result(root: Path, elf: bytes = b"program", *, merlin="1" * 40, rtl="2" * 40):
    run = root / "A"
    (run / "generated").mkdir(parents=True)
    cr = run / "capsule_result.json"
    cr.write_text(json.dumps({
        "capsule": "A", "toolchain_shas": {"merlin": merlin, "target_rtl": rtl}
    }))
    (run / "run_manifest.yaml").write_text("target: test_target\n")
    (run / "generated" / "package_kernel.elf").write_bytes(elf)
    return cr


# ---------------------------------------------------------------------------------------------
def test_a_passing_capsule_produces_a_cert_job(tmp_path):
    """The whole point: an L2 pass enqueues an L3 job, with no agent involvement."""
    B = _broker()
    ws, ch = _ws(tmp_path), None
    ch = ws / ".qa_channel"
    promoted = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert promoted == ["A"]
    reqs = list(ch.glob("simreq_*.json"))
    assert len(reqs) == 1
    r = json.loads(reqs[0].read_text())
    assert r["capsules"] == "A" and r["tiers"] == "L3" and r["promoted"] is True


def test_a_failing_capsule_buys_no_cert_time(tmp_path):
    """A capsule whose numerics are wrong cannot be rescued by RTL, and RTL costs minutes."""
    B = _broker()
    ws = _ws(tmp_path)
    assert B.promote(ws, ws / ".qa_channel", _verdict([("A", False)]), "L2", "L3", None, sys.stderr) == []
    assert list((ws / ".qa_channel").glob("simreq_*.json")) == []


def test_outside_the_cover_is_not_promoted(tmp_path):
    B = _broker()
    ws = _ws(tmp_path)
    got = B.promote(ws, ws / ".qa_channel", _verdict([("A", True), ("B", True)]),
                     "L2", "L3", {"A"}, sys.stderr)
    assert got == ["A"]


def test_the_same_bytes_are_never_certified_twice(tmp_path):
    """Content-addressing is what makes continuous grading affordable: a second identical verdict must
    enqueue nothing, or the loop re-certifies forever."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    first = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    second = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert first == ["A"] and second == []


def test_changed_bytes_re_certify(tmp_path):
    """...and the converse: an edit must invalidate the cert verdict it earned, or a stale RTL pass
    stands for code that no longer exists."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    assert B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]
    (ws / "submission" / "manifest.yaml").write_text("x: 2")          # the submission moved on
    assert B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]


def test_identical_executable_survives_an_unrelated_source_edit(tmp_path):
    """Source bytes may move while the exact program certified by RTL stays byte-identical."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    verdict = _execution_verdict([("A", True, "a")])
    assert B.promote(ws, ch, verdict, "L2", "L3", None, sys.stderr) == ["A"]
    B.record_cert(ws, _execution_verdict([("A", True, "a")]), "L3", sys.stderr)

    (ws / "submission" / "manifest.yaml").write_text("x: 2")
    assert B.promote(ws, ch, verdict, "L2", "L3", None, sys.stderr) == []
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pass"


def test_changed_executable_invalidates_only_its_capsule(tmp_path):
    """Execution identities are per capsule: changing A's ELF must not throw away B's cert."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    first = _execution_verdict([("A", True, "a"), ("B", True, "b")])
    assert B.promote(ws, ch, first, "L2", "L3", None, sys.stderr) == ["A", "B"]
    B.record_cert(ws, first, "L3", sys.stderr)

    (ws / "submission" / "manifest.yaml").write_text("x: 2")
    changed = _execution_verdict([("A", True, "c"), ("B", True, "b")])
    assert B.promote(ws, ch, changed, "L2", "L3", None, sys.stderr) == ["A"]
    assert B._tier_state(ws)["B"]["L3"]["status"] == "pass"


def test_missing_execution_digest_falls_back_to_whole_submission(tmp_path):
    """No artifact identity is not evidence of freshness; preserve the conservative legacy rule."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    verdict = _verdict([("A", True)])
    assert B.promote(ws, ch, verdict, "L2", "L3", None, sys.stderr) == ["A"]
    B.record_cert(ws, _cert_verdict([("A", True)]), "L3", sys.stderr)
    (ws / "submission" / "manifest.yaml").write_text("x: 2")
    assert B.promote(ws, ch, verdict, "L2", "L3", None, sys.stderr) == ["A"]


def test_the_digest_tracks_content_not_time(tmp_path):
    B = _broker()
    ws = _ws(tmp_path)
    d1 = B._submission_digest(ws)
    assert B._submission_digest(ws) == d1                              # stable for identical bytes
    (ws / "submission" / "manifest.yaml").write_text("x: 2")
    assert B._submission_digest(ws) != d1


def test_execution_digest_tracks_elf_and_hardware_but_not_merlin_source(tmp_path):
    B = _broker()
    base = B.execution_digest(_capsule_result(tmp_path / "base"))
    source_edit = B.execution_digest(_capsule_result(tmp_path / "source", merlin="3" * 40))
    elf_edit = B.execution_digest(_capsule_result(tmp_path / "elf", elf=b"changed program"))
    rtl_edit = B.execution_digest(_capsule_result(tmp_path / "rtl", rtl="4" * 40))
    assert base == source_edit
    assert base != elf_edit and base != rtl_edit
    assert isinstance(base, str) and len(base) == 64


def test_execution_digest_fails_closed_without_concrete_hardware_identity(tmp_path):
    B = _broker()
    cr = _capsule_result(tmp_path / "missing", rtl="UNKNOWN")
    assert B.execution_digest(cr) is None


def _harness_module(name: str):
    """Import a harness script by path. They are scripts, not an installed package."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(e).__name__}: {e}")
    return mod


def _runs_root(tmp_path, elf: bytes = b"program"):
    """A runs tree shaped the way `_per_capsule_from_results` globs it: runs/<suite>/<capsule>/."""
    root = tmp_path / "work"
    cr = _capsule_result(root / "runs" / "suite", elf=elf)
    cr.write_text(json.dumps({
        "capsule": "A", "status": "pass",
        "toolchain_shas": {"merlin": "1" * 40, "target_rtl": "2" * 40},
    }))
    return root, cr


def test_the_verdict_reader_computes_a_real_execution_digest(tmp_path):
    """EXECUTE the reader, do not grep it.

    The previous version of this test asserted only that the CALL-SITE TEXT was present in the two
    readers' source. That is a check that cannot fail for the failure it exists to catch: the callee was
    deleted from `tier_promote`, both call sites still read exactly right, the test stayed green, and
    `agent_selfcheck` died with `AttributeError: module 'qa_check' has no attribute
    '_execution_digest_from_result'` on every invocation of the agent's own self-grader. So: build a real
    capsule result, run the reader over it, and demand a real digest out the other end.
    """
    Q, B = _harness_module("qa_check"), _broker()
    root, cr = _runs_root(tmp_path)

    rows = Q._per_capsule_from_results(root)
    assert "A" in rows, "the reader found no capsule result to read"
    digest = rows["A"]["execution_digest"]

    # A REAL digest, not the None the bridge returns when its callee has gone missing.
    assert isinstance(digest, str) and len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert digest == B.execution_digest(cr), "the reader and the broker must agree on the identity"

    # And it is an IDENTITY: different executable bytes are a different capsule to certify.
    other_root, _ = _runs_root(tmp_path / "other", elf=b"different program")
    assert Q._per_capsule_from_results(other_root)["A"]["execution_digest"] != digest


def test_the_self_check_reaches_the_same_live_bridge(tmp_path):
    """`agent_selfcheck` reaches the bridge through its `_qc` alias — the exact expression that raised."""
    A = _harness_module("agent_selfcheck")
    _, cr = _runs_root(tmp_path)
    digest = A._qc._execution_digest_from_result(cr)
    assert isinstance(digest, str) and len(digest) == 64


def test_a_missing_broker_identity_is_not_silently_a_missing_digest(tmp_path):
    """The bridge's `except` is deliberate, but it must not be the reason every row reads `null`.

    Absence has to mean "this capsule has no artifact identity" (no ELF, no concrete hardware pin), never
    "the function the bridge imports is gone". Those two are indistinguishable at the call site, which is
    how the regression survived a green suite.
    """
    B = _broker()
    assert callable(getattr(B, "execution_digest", None)), (
        "tier_promote.execution_digest is the identity both verdict readers import; without it every "
        "row reports execution_digest: null and no certificate can be attributed to the bytes that "
        "earned it")
    # Absence still reports absence, for the honest reason.
    assert B.execution_digest(tmp_path / "nothing" / "capsule_result.json") is None


def test_tier_state_records_both_tiers(tmp_path):
    """The state file is what a continuous loop reads to decide what is left to do; a promotion that does
    not record `pending` would be re-enqueued on every reap."""
    B = _broker()
    ws = _ws(tmp_path)
    B.promote(ws, ws / ".qa_channel", _verdict([("A", True), ("B", False)]), "L2", "L3", None, sys.stderr)
    st = json.loads((ws / "qa" / "tier_state.json").read_text())
    assert st["A"]["L2"]["status"] == "pass"
    assert st["A"]["L3"]["status"] == "pending"
    assert st["B"]["L2"]["status"] == "fail"
    assert "L3" not in st["B"]


def test_promotion_never_gates_a_run(tmp_path):
    """It is an optimisation. A malformed verdict must cost nothing but a log line -- the reap wraps this,
    and this asserts the function itself does not explode on junk."""
    B = _broker()
    ws = _ws(tmp_path)
    for junk in ({}, {"per_capsule": None}, {"per_capsule": [{}]}, {"per_capsule": [{"capsule": None}]}):
        assert B.promote(ws, ws / ".qa_channel", junk, "L2", "L3", None, sys.stderr) == []


# ---------------------------------------------------------------------------------------------
# both brokers must promote — hooking one was the original bug
# ---------------------------------------------------------------------------------------------
def test_both_brokers_call_promotion():
    """Promotion was first wired into the async oracle only. A live run then showed the agent using the
    SYNC self-check 7 times to the async path's 2, so eight verdicts completed and promotion fired zero
    times. Whichever broker produces a verdict must consider promotion, or the feature is dead on the
    path that matters."""
    for name in ("simjob_broker.py", "selfcheck_broker.py", "run_baseline_qa_loop.py"):
        src = (HARNESS / name).read_text(encoding="utf-8")
        assert "tier_promote" in src, f"{name} does not reach the shared promotion module"
        assert "promote" in src, f"{name} never calls promotion"


def test_the_round_grade_promotes_too():
    """A broker only sees a verdict the agent ASKED for, and a converged agent stops asking: measured 24
    self-checks in round 0, then ZERO in rounds 1 and 2 once the corpus ceiling was reached. The round
    grade was then the only verdict produced, and promotion had nothing to fire on -- so the deeper tier
    would only ever be reached while the agent was still struggling, which is backwards. A converged
    submission is the one worth certifying."""
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    i = src.index("def qa_grade(")
    j = src.index("def _write_stage_ledger(")
    assert "tier_promote" in src[i:j], "qa_grade does not promote"


def test_the_policy_is_sourced_not_reimplemented():
    """One policy, one place. The brokers are plumbing; `oracle_schedule` decides."""
    src = (HARNESS / "tier_promote.py").read_text(encoding="utf-8")
    assert "oracle_schedule" in src


# ---------------------------------------------------------------------------------------------
# a rejection must carry a remedy
# ---------------------------------------------------------------------------------------------
def test_a_rejection_names_the_field_and_the_remedy():
    """An agent submitted to the async oracle twice, was rejected twice with "bad sim or capsule
    (constrained runner)", and never used it again -- while the arm that DID reach the async path used it
    98 times in the round its compiler-earned score moved 17 -> 26. The check is load-bearing isolation
    and stays; what changes is that the refusal says which field was wrong and what would be accepted.
    Asserted on the source because the branch needs a live broker loop to reach."""
    src = (HARNESS / "simjob_broker.py").read_text(encoding="utf-8")
    assert "rejected: bad sim or capsule (constrained runner)" not in src, "the remedy-free message is back"
    assert "rejected_field" in src, "a rejection must say WHICH field it refused"
    assert "--tiers" in src, "a neutral-sim target must be told how to choose a tier"


# ---------------------------------------------------------------------------------------------
# The sim NAME, which the tests above never checked
# ---------------------------------------------------------------------------------------------
def _allowed(monkeypatch, sims):
    """Force the broker's allowlist, the way a different target's ladder would."""
    import simjob_broker as SB
    monkeypatch.setattr(SB, "_allowed_sims", lambda: tuple(sims))


def test_the_enqueued_sim_is_one_the_broker_accepts(tmp_path):
    """The invariant the assertions above were missing, and the bug it let through.

    `promote()` wrote the neutral sentinel unconditionally. The broker accepts that ONLY for a target
    whose ladder comes from its contract; a target with a bespoke sim ladder rejects it. So on such a
    target every promotion request was refused while the capsule had already been marked pending,
    stranding it. Measured live: 6 requests, all "rejected: --sim 'contract' is not accepted for this
    target", 2 capsules stuck at L3 pending. Promotion had never once fired on a bespoke-sim target,
    and it looked exactly like "nothing needed promoting".
    """
    B = _broker()
    import simjob_broker as SB
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    reqs = list(ch.glob("simreq_*.json"))
    assert len(reqs) == 1
    sim = json.loads(reqs[0].read_text())["sim"]
    assert sim in SB._allowed_sims(), (
        f"promotion enqueued --sim {sim!r}, which this broker would reject; "
        f"it accepts {SB._allowed_sims()}")


def test_cert_sim_names_the_ladder_sim_when_one_is_declared(tmp_path, monkeypatch):
    """A tier whose contract names a concrete simulator resolves to that simulator.

    REWRITTEN from an assertion that L3 is `verilator`. That was true only while `tier_sim` named a
    BINARY; the contract now names a FIDELITY (`elaborated_rtl`) and the engine is chosen by
    `chipyard_l3_selection`, so hardcoding the old label here would pin a binding the contract no longer
    expresses. The L3 half of the old assertion is covered by the two tests below, against the derived
    engine rather than a literal.
    """
    B = _broker()
    _allowed(monkeypatch, ("spike", "verilator", "vcs"))
    assert B.cert_sim("L2") == "spike"


def _declared_l3_engine():
    """The engine this target's own contract resolves its cert tier to — DERIVED, never a literal."""
    import _common as _C
    from merlin.targetgen.capsule_runner import chipyard_l3_selection
    from merlin.targetgen.target_experiment import load_target_experiment
    te = load_target_experiment(_C.EXP / "target_experiment.yaml")
    return str((chipyard_l3_selection(te.target) or {}).get("engine") or "").strip() or None


def test_cert_sim_resolves_the_fidelity_sentinel_to_a_real_engine(monkeypatch):
    """`tier_sim: {L3: elaborated_rtl}` is a fidelity, not a `--sim` token.

    Left unresolved it matches no token the broker accepts, `cert_sim` returns None, and promotion
    switches off for every unpinned run — announced once as "no --sim serves L3" and thereafter
    indistinguishable from a round with nothing to promote. That is the silent-failure shape this whole
    module is written against, so it gets a falsifier.
    """
    B = _broker()
    engine = _declared_l3_engine()
    if engine is None:
        pytest.skip("this target's contract names no elaborated-RTL engine")
    _allowed(monkeypatch, ("spike", engine))
    assert B.cert_sim("L3") == engine
    assert B.cert_sim("L3") != B._ELABORATED_RTL, "the fidelity sentinel leaked out as a --sim token"


def test_cert_sim_refuses_an_engine_the_broker_would_reject(monkeypatch):
    """Fails closed: the resolved engine still has to be one this broker accepts. Returning it anyway
    would mark a capsule pending for a request the broker refuses — the stranding this file exists for."""
    B = _broker()
    engine = _declared_l3_engine()
    if engine is None:
        pytest.skip("this target's contract names no elaborated-RTL engine")
    _allowed(monkeypatch, tuple(x for x in ("spike", "verilator", "vcs") if x != engine))
    assert B.cert_sim("L3") is None


def test_cert_sim_keeps_the_sentinel_when_that_is_all_that_is_accepted(monkeypatch):
    """An arc-only target grades on its contract-resolved tier; --sim does not apply there."""
    B = _broker()
    _allowed(monkeypatch, (B._NEUTRAL_SIM,))
    assert B.cert_sim("L3") == B._NEUTRAL_SIM


def test_cert_sim_fails_closed_when_nothing_serves_the_tier(monkeypatch):
    B = _broker()
    _allowed(monkeypatch, ("spike", "verilator", "vcs"))
    assert B.cert_sim("L9") is None, "an unknown tier must not resolve to a guessed sim"


def test_no_pending_state_is_written_without_an_enqueued_job(tmp_path, monkeypatch):
    """Marking pending for a job that cannot be enqueued is what stranded the capsules."""
    B = _broker()
    monkeypatch.setattr(B, "cert_sim", lambda tier: None)
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    promoted = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert promoted == []
    assert list(ch.glob("simreq_*.json")) == []
    state = json.dumps(B._tier_state(ws))
    assert "pending" not in state, "a capsule was marked pending for a job that was never enqueued"


# ---------------------------------------------------------------------------------------------
# A completed promotion must be RECORDED, not discarded
# ---------------------------------------------------------------------------------------------
def _cert_verdict(rows):
    return {"per_capsule": [{"capsule": n, "pass": p} for n, p in rows]}


def test_a_completed_promotion_is_recorded_as_a_cert(tmp_path):
    """The whole point of promoting: the cert this PAID FOR on real RTL has to be kept.

    `promote()` marks the capsule `pending` when it enqueues, and the broker's reap skips `_promote`
    for a job that was itself a promotion (correct -- a cert must not re-enqueue itself). But nothing
    wrote the outcome back, so the capsule stayed `pending` forever and the same bytes were
    re-certified on the next loop verdict. Measured live on merlincirct_arm4_func_20260901_v4/_p2:
    21 and 3 promotions COMPLETED, one verified `barrier_tier=L3 barrier_status=pass`, and both tier
    states showed only `L3: pending` -- never once `L3: pass`.
    """
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    assert B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pending"

    B.record_cert(ws, _cert_verdict([("A", True)]), "L3", sys.stderr)
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pass", (
        "a completed promotion left the capsule pending; the certificate was discarded")


def test_a_failed_cert_is_recorded_as_a_failure_not_left_pending(tmp_path):
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    B.record_cert(ws, _cert_verdict([("A", False)]), "L3", sys.stderr)
    assert B._tier_state(ws)["A"]["L3"]["status"] == "fail"


def test_recording_preserves_the_digest_that_earned_the_cert(tmp_path):
    """A cert belongs to the bytes that were pending, not to whatever the agent has edited since."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    before = dict(B._tier_state(ws)["A"]["L3"])
    B.record_cert(ws, _cert_verdict([("A", True)]), "L3", sys.stderr)
    after = B._tier_state(ws)["A"]["L3"]
    assert after["digest"] == before["digest"], "the cert was re-attributed to different bytes"
    assert after.get("components") == before.get("components")


def test_recording_preserves_the_pending_execution_digest(tmp_path):
    """Completion resolves the enqueue-time artifact identity; it never re-hashes edited files."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    verdict = _execution_verdict([("A", True, "a")])
    B.promote(ws, ch, verdict, "L2", "L3", None, sys.stderr)
    before = dict(B._tier_state(ws)["A"]["L3"])
    B.record_cert(ws, _execution_verdict([("A", True, "a")]), "L3", sys.stderr)
    after = B._tier_state(ws)["A"]["L3"]
    assert after["execution_digest"] == before["execution_digest"] == "a" * 64


def test_a_cert_for_different_execution_bytes_does_not_resolve_pending(tmp_path):
    """The async broker may launch after an edit; its result must not be attributed to old bytes."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _execution_verdict([("A", True, "a")]), "L2", "L3", None, sys.stderr)
    resolved = B.record_cert(ws, _execution_verdict([("A", True, "b")]), "L3", sys.stderr)
    assert resolved == []
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pending"


def test_an_unattributable_result_is_not_recorded(tmp_path):
    """No pending entry means we cannot say which bytes earned it -- record nothing, never guess."""
    B = _broker()
    ws = _ws(tmp_path)
    resolved = B.record_cert(ws, _cert_verdict([("NeverPromoted", True)]), "L3", sys.stderr)
    assert resolved == []
    assert "NeverPromoted" not in B._tier_state(ws)


def test_the_broker_reap_actually_calls_the_recorder(tmp_path):
    """Wiring, not just policy: a promoted job's verdict must reach record_cert on reap.

    This file's own docstring warns that promotion is wrapped in a try/except so a broken wiring shows
    up as *nothing happening*. That is precisely how three earlier defects in this path stayed hidden.
    """
    src = (HARNESS / "simjob_broker.py").read_text(encoding="utf-8")
    assert "_TP.record_cert(" in src, "the broker reap never records a completed promotion"
    reap = src[src.index('if not j.get("promoted")'):]
    assert 'elif j.get("promoted")' in reap[:800], (
        "the recorder is not on the promoted-job branch of the reap")
