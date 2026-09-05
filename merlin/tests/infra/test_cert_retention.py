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

# capsule -> the component it declares it rides on.
CORPUS = {"matmul_tile": "lower_interface_to_target", "conv_codegen": "lower_target_to_llvm"}


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
    """A submission that declares NO components -- the undeterminable-decomposition case."""
    return _ws(tmp_path, {"manifest.yaml": MANIFEST.split("components:")[0],
                          "mlir_oot/opt": "#!/usr/bin/env python3\n"}, name=name)


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    for name, dep in CORPUS.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "capsule.yaml").write_text(
            f"name: {name}\nkind: isa\nlabel: public\ndepends_on: [{dep}]\n")
    return root


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
# 1. an edit to a component the capsule does NOT depend on keeps its certificate
# ---------------------------------------------------------------------------------------------
def test_an_edit_to_an_unrelated_component_keeps_the_certificate(tmp_path, monkeypatch):
    B = _mod()
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),))
    ws, ch = _ws(tmp_path), None
    ch = ws / ".qa_channel"
    v = _v([(n, True) for n in CORPUS])
    assert _certify(B, ws, ch, v) == sorted(CORPUS)
    assert B._tier_state(ws)["conv_codegen"]["L3"]["status"] == "pass"

    # touch ONLY the tiling component; `conv_codegen` declares `lower_target_to_llvm`
    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("tile v2\n")
    assert B.promote(ws, ch, v, "L2", "L3", None, sys.stderr) == ["matmul_tile"], (
        "an edit to an unrelated component re-certified a capsule that does not ride on it")
    assert _requests(ch) == ["matmul_tile"]
    assert B._tier_state(ws)["conv_codegen"]["L3"]["status"] == "pass", (
        "the unrelated capsule's certificate was discarded")


def test_an_edit_to_a_declared_dependency_invalidates_the_certificate(tmp_path, monkeypatch):
    """The converse, and the reason this is not simply "invalidate less": a stale certificate standing
    for code that no longer exists is the failure this whole mechanism is guarding against."""
    B = _mod()
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),))
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    v = _v([(n, True) for n in CORPUS])
    assert _certify(B, ws, ch, v) == sorted(CORPUS)

    (ws / "submission" / "mlir_oot" / "codegen.py").write_text("cg v2\n")
    assert B.promote(ws, ch, v, "L2", "L3", None, sys.stderr) == ["conv_codegen"]
    assert _requests(ch) == ["conv_codegen"]


def test_the_log_names_the_component_that_invalidated_the_capsule(tmp_path, monkeypatch, capsys):
    """That log line is how this defect was diagnosed; it has to stay diagnostic."""
    B = _mod()
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),))
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    v = _v([(n, True) for n in CORPUS])
    _certify(B, ws, ch, v)
    capsys.readouterr()
    (ws / "submission" / "mlir_oot" / "lowering" / "tile.py").write_text("tile v2\n")
    B.promote(ws, ch, v, "L2", "L3", None, sys.stderr)
    err = capsys.readouterr().err
    assert "matmul_tile L3 invalidated by lower_interface_to_target (changed)" in err
    assert "<whole-submission>" not in err, "a narrower cause was known and the log still said everything"
    assert "conv_codegen L3 invalidated" not in err


# ---------------------------------------------------------------------------------------------
# 2. an undeterminable dependency set falls back to the whole submission -- and SAYS SO
# ---------------------------------------------------------------------------------------------
def test_an_undeterminable_dependency_set_falls_back_to_the_whole_submission(tmp_path, capsys):
    """No components block and no `depends_on`: staleness cannot be decided any more narrowly, so the
    conservative rule stands. Under-invalidating here would credit an RTL certification to bytes that did
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
    assert "the capsule declares no depends_on" in err
    assert "the submission manifest declares no components" in err


def test_the_fallback_reason_is_recorded_on_disk_too(tmp_path):
    """A log scrolls away; the state file is what a later reader has. `UNKNOWN`-style honesty: the record
    says why it could only be compared against everything."""
    B = _mod()
    ws = _plain_ws(tmp_path)
    B.promote(ws, ws / ".qa_channel", _v([("A", True)]), "L2", "L3", None, sys.stderr)
    st = json.loads((ws / "qa" / "tier_state.json").read_text())
    assert "no execution identity" in st["A"]["L2"]["fallback_reason"]
    assert "declares no components" in st["A"]["L3"]["fallback_reason"]


def test_a_narrow_comparison_records_no_fallback_reason(tmp_path, monkeypatch):
    """Guard on the guard: the reason must not be written unconditionally, or it says nothing."""
    B = _mod()
    monkeypatch.setattr(B, "_graded_roots", lambda: (_corpus(tmp_path),))
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
