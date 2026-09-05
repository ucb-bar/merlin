"""A certificate is discarded only when something the capsule DEPENDS ON changed -- and never lost.

Measured on the live gemmini round `merlincirct_arm4_func_20260901_codex1` round 1, from the broker log
and that run's own `.qa_channel` responses:

    [promote] A1_mvin_mvout L3 invalidated by <whole-submission> (changed)
    [promote] A4_acc_scale_i8 L3 invalidated by <whole-submission> (changed)
    [promote] L2 pass -> L3: ['A1_mvin_mvout', 'A4_acc_scale_i8']

repeated from 21:43 to 22:28: 17 cycle-accurate promotion jobs, EVERY one of them on the same two
capsules, each passing 1/1 -- roughly 35 minutes of RTL re-certifying bytes that had already passed --
while every per-capsule cert-tier record read `certified: None`: the tier state never once held a
cert-tier `pass`, only `pending`. (The self-checks' own `n_certified: 0` is a DIFFERENT thing and is not
this defect: those ran on the screen simulator, whose barrier cannot reach the mandatory cert tier, so
their passing capsules are counted as screened. That is `agent_selfcheck`'s definition of a screen and
nothing here changes it.)

Two defects produced the discarded certificates, and the tests here pin both plus the direction that must
never be traded for them:

  * **invalidation was whole-submission** -- any byte anywhere in `submission/` discarded every capsule's
    certificate, so progress was reset faster than promotion could accumulate it;
  * **the record a completed job belonged to was overwritten** by the re-enqueue that the invalidation
    triggered. With an artifact identity the arriving certificate was then refused (dropped); WITHOUT
    one it was accepted onto the newer record, crediting an RTL certification to bytes that never earned
    it. The second is strictly worse than the waste, so it gets its own falsifier below.

The gate that outranks every saving here: `test_a_certificate_never_covers_bytes_that_did_not_earn_it`.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"

# A submission that declares the four entrypoints `manifest.schema.json` requires, and maps each to the
# files that implement it. A test may name the contract it is testing; the library under test DERIVES the
# legal component names from the submission's own manifest and never lists them.
MANIFEST = """\
artifact_type: mlir_oot_target_backend
target: t
language: python
authoring: {mode: hand_curated}
integrity_exempt: false
entrypoints: {tool: mlir_oot/opt}
commands:
  parse: {argv: ["{tool}", "{input_mlir}"]}
  lower_interface_to_target: {argv: ["{tool}", "--to-target", "{input_mlir}"]}
  emit_command_buffer: {argv: ["{tool}", "--cb={output_json}", "{input_mlir}"]}
  lower_target_to_llvm: {argv: ["{tool}", "--artifact", "{input_mlir}"]}
components:
  parse: [mlir_oot/parse.py]
  lower_interface_to_target: [mlir_oot/lowering/]
  emit_command_buffer: [mlir_oot/cmdbuf.py]
  lower_target_to_llvm: [mlir_oot/codegen.py]
"""

FILES = {
    "manifest.yaml": MANIFEST,
    "mlir_oot/opt": "#!/usr/bin/env python3\n",
    "mlir_oot/parse.py": "parse v1\n",
    "mlir_oot/lowering/tile.py": "tile v1\n",
    "mlir_oot/cmdbuf.py": "cb v1\n",
    "mlir_oot/codegen.py": "cg v1\n",
}

# The two capsules the retention tests drive. They declare NOTHING: `depends_on` was retired because a
# truthful per-capsule declaration cannot exist (`run_entrypoints` invokes every command for every
# capsule, and every package maps all four commands onto the same argv). What distinguishes one capsule
# from another is its EXECUTION identity -- the program it actually emits -- not a claim it makes.
CORPUS = ("matmul_tile", "conv_codegen")


def _mod():
    """Import the shared promotion module by path -- it is a harness script, not installed."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("tier_promote", HARNESS / "tier_promote.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"tier_promote not importable here: {type(e).__name__}: {e}")
    return mod


def _ws(tmp_path, files=None, name="ws"):
    ws = tmp_path / name
    for rel, body in (files or FILES).items():
        f = ws / "submission" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    (ws / ".qa_channel").mkdir(parents=True, exist_ok=True)
    return ws


def _plain_ws(tmp_path, name="ws"):
    """A submission nothing can be decomposed FROM -- the genuinely undeterminable case.

    It declares no `components:` block AND no `commands:`, so neither source of a decomposition applies:
    the declared path has no block to read and the derivation has no command surface to trace. That is
    what leaves the whole-submission digest as the only comparison.

    NOTE it used to be the components block alone that was stripped. That is no longer undeterminable:
    the harness DERIVES a decomposition from the submission's own declared surface when no block is
    given, so such a submission gets four components and never reaches the fallback under test.
    """
    return _ws(tmp_path, {"manifest.yaml": "artifact_type: mlir_oot_target_backend\ntarget: t\n",
                          "mlir_oot/opt": "#!/usr/bin/env python3\n"}, name=name)


def _derived_ws(tmp_path, extra=None, name="ws"):
    """A submission with NO `components:` block, so the decomposition is DERIVED from its own surface.

    That is the only path with an `inert` bucket -- bytes the derivation proved no declared command can
    open. A declared block has no such bucket (every file is either a component's or UNATTRIBUTED).
    """
    files = dict(FILES)
    files["manifest.yaml"] = MANIFEST.split("components:")[0]
    files.update(extra or {})
    return _ws(tmp_path, files, name=name)


def _v(rows):
    """Verdict rows the loop tier produced, as ``(capsule, passed)`` -- no artifact identity, which is
    what every gemmini row looked like before the execution digest existed."""
    return {"per_capsule": [{"capsule": n, "pass": p} for n, p in rows]}


def _vx(rows):
    """``(capsule, passed, execution-byte)`` -- sha-shaped artifact identities."""
    return {"per_capsule": [{"capsule": n, "pass": p, "execution_digest": c * 64} for n, p, c in rows]}


def _requests(ch):
    return sorted(json.loads(f.read_text())["capsules"] for f in ch.glob("simreq_*.json"))


def _identity_of(ch, capsule):
    """The ledger key the enqueuer stamped on this capsule's outstanding request."""
    for f in sorted(ch.glob("simreq_*.json")):
        r = json.loads(f.read_text())
        if r["capsules"] == capsule:
            return r.get("identity")
    raise AssertionError(f"no request was enqueued for {capsule}")


def _certify_x(B, ws, ch, verdict, cover=None):
    """:func:`_certify` for rows that CARRY an execution identity -- the result echoes the one it ran.

    A pending record naming an exact artifact may only be resolved by a result naming that same artifact
    (see `test_a_request_identity_never_overrides_the_artifact_the_job_actually_ran`), so a result that
    dropped the identity would be refused here rather than recorded.
    """
    by_name = {r["capsule"]: r.get("execution_digest") for r in verdict["per_capsule"]}
    promoted = B.promote(ws, ch, verdict, "L2", "L3", cover, sys.stderr)
    for capsule in promoted:
        B.record_cert(ws, {"per_capsule": [{"capsule": capsule, "pass": True,
                                            "execution_digest": by_name[capsule]}]},
                      "L3", sys.stderr, identity=_identity_of(ch, capsule))
    for f in ch.glob("simreq_*.json"):
        f.unlink()
    return promoted


def _certify(B, ws, ch, verdict, cover=None):
    """Promote, then hand each promotion's result back the way the broker's reap does."""
    promoted = B.promote(ws, ch, verdict, "L2", "L3", cover, sys.stderr)
    for capsule in promoted:
        B.record_cert(ws, {"per_capsule": [{"capsule": capsule, "pass": True}]}, "L3", sys.stderr,
                      identity=_identity_of(ch, capsule))
    for f in ch.glob("simreq_*.json"):
        f.unlink()
    return promoted


# ---------------------------------------------------------------------------------------------
# 1. an edit that does not change a capsule's PROGRAM keeps its certificate
#
# REWRITTEN. These three tests used to drive a per-capsule `depends_on:` declaration -- capsule X keeps
# its certificate across an edit to a component X did not name. That design was retired (0 of the 509
# corpus capsules ever declared one) because a truthful narrow declaration cannot exist: the shared ABI
# front half `capsule_common.run_entrypoints` invokes EVERY declared command for EVERY capsule, and every
# package maps all four commands onto the same argv, so an edit to any command's files can flip any
# capsule's verdict. A capsule naming a subset would keep a certificate its current bytes did not earn --
# the one direction `oracle_schedule` calls far worse than re-running.
#
# The property those tests were reaching for is real and is preserved here, one layer down, where it IS
# truthful: the EXECUTION identity. Each capsule emits its own program, so "capsule 1 moved and capsule 2
# did not" is a statement about emitted bytes rather than a claim a capsule makes about itself.
# ---------------------------------------------------------------------------------------------
def test_an_edit_that_leaves_a_capsules_program_alone_keeps_its_certificate(tmp_path):
    """The saving the whole scheme is for: an optimization phase edits the compiler continuously, and an
    edit that re-emits only ONE capsule's code must not re-buy the cert tier for the whole corpus."""
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    before = _vx([("matmul_tile", True, "a"), ("conv_codegen", True, "b")])
    assert _certify_x(B, ws, ch, before) == sorted(CORPUS)
    assert B._tier_state(ws)["conv_codegen"]["L3"]["status"] == "pass"

    # a real source edit -- so the whole-submission AND component digests both move -- that changes only
    # what `matmul_tile` emits. `conv_codegen` still emits byte-identical code.
    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("tile v2\n")
    after = _vx([("matmul_tile", True, "c"), ("conv_codegen", True, "b")])
    assert B.promote(ws, ch, after, "L2", "L3", None, sys.stderr) == ["matmul_tile"], (
        "an edit that did not change a capsule's emitted program re-certified it anyway")
    assert _requests(ch) == ["matmul_tile"]
    assert B._tier_state(ws)["conv_codegen"]["L3"]["status"] == "pass", (
        "the unchanged capsule's certificate was discarded")


def test_a_changed_program_invalidates_that_capsules_certificate(tmp_path):
    """The converse, and the reason this is not simply "invalidate less": a certificate standing for a
    program that is no longer emitted is the failure the whole mechanism guards against."""
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    assert _certify_x(B, ws, ch, _vx([("matmul_tile", True, "a"),
                                      ("conv_codegen", True, "b")])) == sorted(CORPUS)

    (ws / "submission" / "mlir_oot" / "codegen.py").write_text("cg v2\n")
    after = _vx([("matmul_tile", True, "a"), ("conv_codegen", True, "d")])
    assert B.promote(ws, ch, after, "L2", "L3", None, sys.stderr) == ["conv_codegen"]
    assert _requests(ch) == ["conv_codegen"]


def test_a_capsule_that_reports_no_program_identity_is_never_spared(tmp_path):
    """FAIL CLOSED. A verdict row that carries no execution identity must fall back to the submission
    comparison, never be read as "unchanged". Absence of evidence is the direction that would let a stale
    certificate stand, and `execution_digest` returns None for exactly the honest reasons (no ELF, no
    target, no concrete hardware revision) -- so this is the common case, not an exotic one."""
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    assert _certify_x(B, ws, ch, _vx([("matmul_tile", True, "a"),
                                      ("conv_codegen", True, "b")])) == sorted(CORPUS)

    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("tile v2\n")
    # the reader could not identify either program this round
    assert B.promote(ws, ch, _v([(n, True) for n in CORPUS]), "L2", "L3",
                     None, sys.stderr) == sorted(CORPUS), (
        "a row with no execution identity was treated as evidence the program had not changed")


def test_the_log_names_the_component_that_invalidated_the_capsule(tmp_path, capsys):
    """That log line is how this defect was diagnosed; it has to stay diagnostic.

    REWRITTEN in its second half. With no execution identity every capsule rides on the same component
    set, so an edit to a component requeues BOTH capsules -- and the log must say so for both rather than
    imply a narrowing that no longer exists. The component is still NAMED, which is the part that made
    the whole-submission-only log undiagnosable.
    """
    B = _mod()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    v = _v([(n, True) for n in CORPUS])
    _certify(B, ws, ch, v)
    capsys.readouterr()
    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("tile v2\n")
    B.promote(ws, ch, v, "L2", "L3", None, sys.stderr)
    err = capsys.readouterr().err
    assert "matmul_tile L3 invalidated by lower_interface_to_target (changed)" in err
    assert "conv_codegen L3 invalidated by lower_interface_to_target (changed)" in err, (
        "the honest ceiling: with no execution identity no capsule may narrow, and the log must show it")
    assert "<whole-submission>" not in err, "a narrower cause was known and the log still said everything"


def test_bytes_no_command_can_read_invalidate_nothing(tmp_path):
    """The saving the DERIVED decomposition delivers, and the reason it is kept alongside the execution
    identity: 17% of a live round's submission-mutating operations touched only the agent's notes, its
    report, and a scratch assembly file no declared command can open. Each one used to wipe every
    recorded verdict."""
    B = _mod()
    ws = _derived_ws(tmp_path, {"REPORT.md": "round 1\n"})
    ch = ws / ".qa_channel"
    assert B.decomposition(ws)["source"] == "derived"
    assert "REPORT.md" in B.decomposition(ws)["inert"], (
        "the fixture no longer exercises the inert bucket, so this test proves nothing")
    v = _v([(n, True) for n in CORPUS])
    assert _certify(B, ws, ch, v) == sorted(CORPUS)

    (ws / "submission" / "REPORT.md").write_text("round 2\n")
    assert B.promote(ws, ch, v, "L2", "L3", None, sys.stderr) == [], (
        "an edit to bytes the derivation proved no command can read discarded every certificate")


# ---------------------------------------------------------------------------------------------
# 2. no narrower identity at all falls back to the whole submission -- and SAYS SO
# ---------------------------------------------------------------------------------------------
def test_no_narrower_identity_falls_back_to_the_whole_submission(tmp_path, capsys):
    """No execution identity and no decomposition, declared or derivable: staleness cannot be decided any
    more narrowly, so the conservative rule stands. Under-invalidating here would credit an RTL certification to bytes that did
    not earn it, which is strictly worse than re-running."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    v = _v([("A", True)])
    assert _certify(B, ws, ch, v) == ["A"]
    capsys.readouterr()

    (ws / "submission" / "mlir_oot" / "opt").write_text("#!/usr/bin/env python3\n# edited\n")
    assert B.promote(ws, ch, v, "L2", "L3", None, sys.stderr) == ["A"]
    err = capsys.readouterr().err
    assert "A L3 invalidated by <whole-submission> (changed)" in err
    # ...and it names every input that was missing, rather than going quiet: 21 identical
    # `<whole-submission> (changed)` lines naming none of them is what made this look like correct
    # conservative behaviour for a whole round.
    assert "no narrower cause:" in err
    assert "no execution identity for this capsule in this verdict" in err
    assert "the submission has no component decomposition, declared or derived" in err
    # The retired clause. `depends_on` no longer exists, so naming it here would report a missing input
    # that cannot be supplied -- an UNKNOWN pointing at nothing, which is worse than silence.
    assert "depends_on" not in err


def test_the_fallback_reason_is_recorded_on_disk_too(tmp_path):
    """A log scrolls away; the state file is what a later reader has. `UNKNOWN`-style honesty: the record
    says why it could only be compared against everything."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    B.promote(ws, ws / ".qa_channel", _v([("A", True)]), "L2", "L3", None, sys.stderr)
    st = json.loads((ws / "qa" / "tier_state.json").read_text())
    assert "no execution identity" in st["A"]["L2"]["fallback_reason"]
    assert "no component decomposition" in st["A"]["L3"]["fallback_reason"]


def test_a_narrow_comparison_records_no_fallback_reason(tmp_path):
    """Guard on the guard: the reason must not be written unconditionally, or it says nothing."""
    B = _mod()
    ws = _ws(tmp_path)
    B.promote(ws, ws / ".qa_channel", _vx([("matmul_tile", True, "a")]), "L2", "L3", None, sys.stderr)
    st = json.loads((ws / "qa" / "tier_state.json").read_text())
    assert "fallback_reason" not in st["matmul_tile"]["L2"]
    assert "fallback_reason" not in st["matmul_tile"]["L3"]


# ---------------------------------------------------------------------------------------------
# 3. a passing promotion is RECORDED, not left outstanding
# ---------------------------------------------------------------------------------------------
def test_a_passing_promotion_is_recorded_as_certified_not_left_pending(tmp_path):
    """The `certified: None` measurement. A cert job takes minutes and the agent edits while it runs, so
    the record it was launched for used to be overwritten by the re-enqueue -- and the certificate the
    RTL had just paid for was dropped on arrival. 17 jobs, all passing, nothing ever certified.
    """
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    assert B.promote(ws, ch, _vx([("A", True, "a")]), "L2", "L3", None, sys.stderr) == ["A"]
    first = _identity_of(ch, "A")

    # the agent edits mid-job: a new program, a second outstanding record
    (ws / "submission" / "mlir_oot" / "opt").write_text("# edited\n")
    assert B.promote(ws, ch, _vx([("A", True, "b")]), "L2", "L3", None, sys.stderr) == ["A"]

    # the FIRST job now finishes and passed on real RTL
    assert B.record_cert(ws, _vx([("A", True, "a")]), "L3", sys.stderr, identity=first) == ["A=pass"]
    ledger = json.loads((ws / "qa" / "tier_state.json").read_text())["A"]["<certs>"]["L3"]
    assert ledger["a" * 64]["status"] == "pass", "the certificate this job paid for was discarded"
    assert ledger["b" * 64]["status"] == "pending", "the in-flight job's record was resolved by another"


def test_a_recorded_certificate_is_reused_when_its_bytes_come_back(tmp_path):
    """Retention is the point: a certificate is not-applicable-right-now, never destroyed. When the
    program it certified is what the submission emits again, it buys no second RTL run."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _vx([("A", True, "a")]), "L2", "L3", None, sys.stderr)
    first = _identity_of(ch, "A")
    (ws / "submission" / "mlir_oot" / "opt").write_text("# edited\n")
    B.promote(ws, ch, _vx([("A", True, "b")]), "L2", "L3", None, sys.stderr)
    B.record_cert(ws, _vx([("A", True, "a")]), "L3", sys.stderr, identity=first)

    assert B.promote(ws, ch, _vx([("A", True, "a")]), "L2", "L3", None, sys.stderr) == [], (
        "the retained certificate for exactly these bytes bought RTL time again")


def test_a_promotion_without_an_artifact_identity_is_still_recorded(tmp_path):
    """The gemmini row shape before the execution digest existed. The identity travels on the REQUEST, so
    the result is still attributable exactly -- and the recorder never has to guess."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _v([("A", True)]), "L2", "L3", None, sys.stderr)
    ident = _identity_of(ch, "A")
    assert ident, "the request carries no identity, so a completed result cannot be attributed"
    assert B.record_cert(ws, _v([("A", True)]), "L3", sys.stderr, identity=ident) == ["A=pass"]
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pass"


# ---------------------------------------------------------------------------------------------
# 4. THE CORRECTNESS GATE -- outranks every saving above
# ---------------------------------------------------------------------------------------------
def test_a_certificate_never_covers_bytes_that_did_not_earn_it(tmp_path):
    """The direction that must never regress, and it was live: with no artifact identity on either side,
    `record_cert` wrote the arriving result onto whatever record was outstanding NOW -- so a certificate
    earned on the pre-edit bytes was recorded against the post-edit ones. Refusing is the only safe
    answer when which bytes earned it cannot be determined.
    """
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _v([("A", True)]), "L2", "L3", None, sys.stderr)
    before = B._tier_state(ws)["A"]["L3"]["digest"]
    (ws / "submission" / "mlir_oot" / "opt").write_text("# edited\n")
    B.promote(ws, ch, _v([("A", True)]), "L2", "L3", None, sys.stderr)
    after = B._tier_state(ws)["A"]["L3"]["digest"]
    assert before != after

    # a result that can name neither its artifact nor its request: unattributable, so record NOTHING
    assert B.record_cert(ws, _v([("A", True)]), "L3", sys.stderr) == []
    ledger = json.loads((ws / "qa" / "tier_state.json").read_text())["A"]["<certs>"]["L3"]
    assert {e["status"] for e in ledger.values()} == {"pending"}, (
        "an RTL certification was attributed to bytes that did not earn it")


def test_a_result_whose_artifact_holds_no_record_is_refused(tmp_path):
    """The async broker may launch after an edit; a job that ran a program nothing is waiting on cannot
    be credited to a program that is."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _vx([("A", True, "a")]), "L2", "L3", None, sys.stderr)
    assert B.record_cert(ws, _vx([("A", True, "c")]), "L3", sys.stderr) == []
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pending"


def test_a_late_result_does_not_move_the_readable_record_onto_its_own_bytes(tmp_path):
    """The ledger is not the only thing a reader sees. ``tier_state[capsule][tier]`` is the READABLE
    per-tier record -- what a report, a round brief, or `_slots_ro`'s fallback reads -- and a cert job
    launched before an edit routinely finishes after it. Resolving that late job must leave the record
    for the CURRENT bytes outstanding: moving the readable record onto the old job's identity publishes
    a cert-tier `pass` for a program the current bytes never emitted, AND discards the pending record
    the in-flight job still has to resolve, so that job's own result then has nothing to land on.

    Mutation-checked 2026-09-05: dropping the ``record_identity(mirror) == key`` guard in `record_cert`
    left all 54 tests in these two files green and only this one red.
    """
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _vx([("A", True, "a")]), "L2", "L3", None, sys.stderr)   # job 1 runs bytes "a"
    B.promote(ws, ch, _vx([("A", True, "b")]), "L2", "L3", None, sys.stderr)   # edit -> job 2 runs "b"
    mirror = B._tier_state(ws)["A"]["L3"]
    assert mirror["status"] == "pending" and mirror["execution_digest"] == "b" * 64

    # job 1 finishes LAST. Its certificate belongs to "a", and to nothing else.
    assert B.record_cert(ws, _vx([("A", True, "a")]), "L3", sys.stderr) == ["A=pass"]
    st = B._tier_state(ws)
    assert st["A"]["<certs>"]["L3"]["a" * 64]["status"] == "pass", "the cert it paid for was dropped"
    assert st["A"]["L3"]["execution_digest"] == "b" * 64, (
        "the readable record was re-attributed to the bytes of an older job")
    assert st["A"]["L3"]["status"] == "pending", (
        "the record the in-flight job still has to resolve was overwritten with a pass it did not earn")


@pytest.mark.parametrize("ran", ["d",     # a well-formed identity for a DIFFERENT program
                                 "z"])    # malformed: the reader could not identify what ran
def test_a_request_identity_never_overrides_the_artifact_the_job_actually_ran(tmp_path, ran):
    """The request says what was ASKED for; the result says what RAN. When the record names an exact
    artifact, only that artifact may be credited -- the agent may have edited between enqueue and launch,
    so the request is not evidence about the program. A malformed identity is the same case: unreadable
    provenance is never evidence that the right artifact ran (this direction was a live hole in the first
    cut of this change, caught by the parametrisation).
    """
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    B.promote(ws, ch, _vx([("A", True, "a")]), "L2", "L3", None, sys.stderr)
    stale_request = _identity_of(ch, "A")
    assert B.record_cert(ws, _vx([("A", True, ran)]), "L3", sys.stderr,
                         identity=stale_request) == []
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pending"


def test_an_unpromoted_capsule_leaves_no_trace(tmp_path):
    """Recording must not invent a record for a capsule nobody promoted -- a fabricated record is a
    certificate with no run behind it."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    assert B.record_cert(ws, _v([("NeverPromoted", True)]), "L3", sys.stderr, identity="anything") == []
    assert "NeverPromoted" not in B._tier_state(ws)


# ---------------------------------------------------------------------------------------------
# 5. the ledger is bounded, and the bound never evicts work in flight
# ---------------------------------------------------------------------------------------------
def test_the_ledger_is_bounded(tmp_path):
    """The state file is re-read and re-written on every verdict; a continuous round produces dozens per
    hour. Retention must not grow the hot file without limit."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    for i in range(B._LEDGER_KEEP + 8):
        (ws / "submission" / "mlir_oot" / "opt").write_text(f"# edit {i}\n")
        B.promote(ws, ch, _vx([("A", True, "abcdef"[i % 6])]), "L2", "L3", None, sys.stderr)
        B.record_cert(ws, _vx([("A", True, "abcdef"[i % 6])]), "L3", sys.stderr)
    led = json.loads((ws / "qa" / "tier_state.json").read_text())["A"]["<certs>"]
    assert len(led["L2"]) <= B._LEDGER_KEEP and len(led["L3"]) <= B._LEDGER_KEEP


def test_the_bound_never_evicts_an_outstanding_record(tmp_path):
    """Evicting a resolved record only costs a re-run. Evicting an OUTSTANDING one makes the in-flight
    job's result unattributable -- which is the certificate-dropping defect this file exists for."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    ch = ws / ".qa_channel"
    first = None
    for i in range(B._LEDGER_KEEP + 8):
        (ws / "submission" / "mlir_oot" / "opt").write_text(f"# edit {i}\n")
        B.promote(ws, ch, _v([("A", True)]), "L2", "L3", None, sys.stderr)
        if first is None:
            first = _identity_of(ch, "A")
    led = json.loads((ws / "qa" / "tier_state.json").read_text())["A"]["<certs>"]["L3"]
    assert first in led, "an in-flight promotion's record was evicted by the retention bound"
    assert all(e["status"] == "pending" for e in led.values())


# ---------------------------------------------------------------------------------------------
# 6. wiring: the identity has to actually reach the recorder
# ---------------------------------------------------------------------------------------------
def test_the_broker_forwards_the_request_identity_to_the_recorder():
    """Asserted on the source because the reap needs a live broker loop and a real simulator. Promotion
    is wrapped in a try/except so it can never gate a run, which is exactly why a broken wiring here
    shows up as nothing happening."""
    src = (HARNESS / "simjob_broker.py").read_text(encoding="utf-8")
    assert 'identity=j.get("identity")' in src, "the reap does not tell the recorder which record it ran"
    assert '"identity": r.get("identity")' in src, "the launch drops the request's identity"


def test_the_state_file_is_replaced_atomically():
    """Two brokers write this file. A truncating write lets the other read a half-written one, and
    `_tier_state` answers an unparseable read with `{}` -- which the next save would then persist,
    deleting every capsule's recorded verdict."""
    src = (HARNESS / "tier_promote.py").read_text(encoding="utf-8")
    i = src.index("def _save_tier_state(")
    body = src[i:i + 1400]
    assert "os.replace(" in body, "the tier state is written non-atomically"
