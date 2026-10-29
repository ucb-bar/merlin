"""Host-only scoped loop controls; no claim of Lean acceptance from these tests."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify import loop_transform_gate as gate
from merlin.verify.loop_transform_intake import ROOT, Settings, Unavailable, Unsupported, prepare, produce

SOURCE = """builtin.module {
func.func private @step(i32, index) -> i32
func.func private @other(i32, index) -> i32
func.func private @other_watch(i32)
func.func private @watch(i32)
func.func @main(%x:i32, %y:i32) -> i32 {
%lb = arith.constant 3 : index
%ub = arith.constant 27 : index
%st = arith.constant 2 : index
%r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
%a = arith.addi %c, %y : i32
%b = func.call @step(%a,%i) : (i32,index) -> i32
scf.yield %b : i32
}
func.call @watch(%r) : (i32) -> ()
func.return %r : i32
}
func.func @untouched(%x:i32) -> i32 { func.return %x : i32 }
}"""
REFERENCE = repo_root().parent / "merlin-graph-proof-intake"


@pytest.fixture(scope="module")
def actual():
    buffers = produce(SOURCE, reference=REFERENCE)
    return buffers, prepare(buffers, reference=REFERENCE)


def test_changed_child_cleanup_source_invalidates_request(actual, monkeypatch):
    from merlin.verify.loop_transform_intake import revalidate

    _, candidate = actual
    cleanup = (repo_root() / "src/merlin/common/owned_children.py").resolve()
    read_bytes = Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return data + b"\n" if path.resolve() == cleanup else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(Unavailable, match="proof/host/producer source changed"):
        revalidate(candidate)


def test_actual_three_buffers_preserve_complete_roster_and_obligations(actual):
    buffers, prepared = actual
    assert all(a != b for a, b in zip(buffers, buffers[1:]))
    assert prepared.roots[-1] == ROOT
    assert any(r.endswith(".unroll_actual") for r in prepared.roots)
    assert any(r.endswith(".segment_actual") for r in prepared.roots)
    assert len([r for r in prepared.roots if r.endswith(".actual")]) == 2  # complete surrounding entry
    assert prepared.describe()["authority"] is False
    assert prepare(buffers, reference=REFERENCE) == prepared


@pytest.mark.parametrize(
    "settings",
    [Settings(1), Settings(5), Settings(tile=1), Settings(tile=17), Settings(index_bits=32), Settings(factor=True)],
)
def test_explicit_domain_refusal(settings):
    with pytest.raises(Unsupported):
        prepare(("a", "b", "c"), reference=REFERENCE, settings=settings)


@pytest.mark.parametrize(
    "stage,old,new",
    [
        (0, "27", "29"),
        (1, "callee = @step", "callee = @other"),
        (2, "callee = @watch", "callee = @other_watch"),
        (2, "arith.addi", "arith.subi"),
    ],
)
def test_changed_actual_source_middle_body_observer_refuses(actual, stage, old, new):
    buffers, _ = actual
    changed = list(buffers)
    changed[stage] = changed[stage].replace(old, new)
    assert changed[stage] != buffers[stage]
    with pytest.raises((Unsupported, Unavailable)):
        prepare(tuple(changed), reference=REFERENCE)


def test_wrong_factor_tile_require_actual_producer_bindings(actual):
    buffers, _ = actual
    for settings in (Settings(factor=3), Settings(tile=4)):
        with pytest.raises(Unsupported):
            prepare(buffers, reference=REFERENCE, settings=settings)


def test_identity_boundaries_are_not_claimed(actual):
    buffers, _ = actual
    with pytest.raises(Unsupported):
        prepare((buffers[0], buffers[0], buffers[2]), reference=REFERENCE)


@pytest.mark.parametrize("mutation", ["query", "roots", "domain"])
def test_prepared_forgery_refused_before_lean(actual, tmp_path, monkeypatch, mutation):
    _, p = actual
    forged = replace(
        p,
        **{"query": {"query": p.query + "\n"}, "roots": {"roots": (ROOT,)}, "domain": {"domain": "broader"}}[mutation],
    )
    monkeypatch.setattr(gate, "_run", lambda *a: pytest.fail("launched Lean for forged request"))
    with pytest.raises(Unavailable):
        gate.check(forged, lean=Path("/unused"), destination=tmp_path / "check")


@pytest.mark.parametrize("kind", ["none", "dict", "constructed", "copied"])
def test_only_live_completion_may_deliver(actual, kind):
    _, p = actual
    c = gate.Completion(p.identity, "/saved/report")
    candidate = {"none": None, "dict": {"status": "checked"}, "constructed": c, "copied": copy.copy(c)}[kind]
    with pytest.raises(Unavailable, match="live authority"):
        gate.deliver(candidate, p, lambda text: pytest.fail("unauthorized delivery"))


@pytest.mark.parametrize(
    "mutation", ["statement", "missing_import", "duplicate_import", "unknown_import", "missing_axioms", "bad_axioms"]
)
def test_malformed_native_audit_never_grants_authority(mutation):
    audit = {
        "violations": [],
        "assurance": "lean_kernel_std3_no_native_reflection",
        "theorems_audited": 1,
        "axioms_used": ["propext"],
        "imports": [{"module": "LoopTransformQuery", "olean": "/test/object"}],
    }
    statements = {ROOT: "TEST-ONLY TYPE; NOT KERNEL EVIDENCE"}
    if mutation == "statement":
        statements[ROOT] = True
    elif mutation == "missing_import":
        audit["imports"] = []
    elif mutation == "duplicate_import":
        audit["imports"] *= 2
    elif mutation == "unknown_import":
        audit["imports"].append({"module": "Merlin.Unknown", "olean": "/test/extra"})
    elif mutation == "missing_axioms":
        del audit["axioms_used"]
    else:
        audit["axioms_used"] = ["sorryAx"]
    with pytest.raises(Unavailable):
        gate.validate_audits(audit, statements, (ROOT,), {"LoopTransformQuery"})


def test_timeout_retains_partial_output_and_never_delivers(tmp_path, monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial proof output", stderr=b"first diagnostic")

    monkeypatch.setattr(gate, "run_owned", timeout)
    with pytest.raises(Unavailable, match="unavailable"):
        gate._run(["/test/lean"], tmp_path, {}, tmp_path, "proof", 1)
    assert (tmp_path / "proof.stdout").read_text() == "partial proof output"
    assert (tmp_path / "proof.stderr").read_text() == "first diagnostic"


@pytest.mark.parametrize("upper", [3, 5])
def test_known_no_segmentation_output_is_unsupported_not_infrastructure(upper):
    raw = SOURCE.replace("constant 27", "constant " + str(upper))
    with pytest.raises(Unsupported, match="both actual loop boundaries"):
        produce(raw, reference=REFERENCE)


@pytest.mark.parametrize(
    "diagnostic",
    [
        "(deterministic) timeout at `isDefEq`, maximum number of heartbeats (2000000) has been reached",
        "maximum recursion depth has been reached",
    ],
)
def test_proof_resource_exhaustion_is_unavailable(diagnostic):
    import subprocess

    with pytest.raises(Unavailable, match="proof-resource"):
        gate.require_query_success(subprocess.CompletedProcess(["/test/lean"], 1, diagnostic, ""))


def test_ordinary_checked_obligation_refusal_stays_not_established():
    import subprocess

    with pytest.raises(gate.NotEstablished):
        gate.require_query_success(subprocess.CompletedProcess(["/test/lean"], 1, "unsolved goals", ""))
