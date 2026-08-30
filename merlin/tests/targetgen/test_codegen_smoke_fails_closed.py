"""A missing compile PREREQUISITE is a NO_GO, not an "n/a".

codegen_smoke early-returns ``True, "n/a (…)"`` for targets whose emit path it does not cover. That is
correct for a smoke that does not APPLY — and catastrophic for a prerequisite. The fork-free path's ISA
model is the latter: if the backend cannot build it, every capsule fails to compile and the run grades
only capsules that need no oracle.

The two were resolved by DIFFERENT paths, which is how it hid. `isa_model_for_target` succeeded and
reported "not fixed-format" -> n/a -> codegen_ok: true, while the compile path's `_model_for` reads the
mlc-derived encoding fact and returned None. Measured: a radiance run spent 101 minutes flat at 6/39 —
the six MX fixtures, which need no oracle — with every real capsule `incomplete: no derived ISA encoding
fact`, while codegen_smoke reported codegen_ok: true throughout.
"""
from __future__ import annotations

import inspect

from merlin.targetgen import capsule_runner as CR


def test_the_prerequisite_check_runs_before_any_n_a_shortcut():
    """Ordering is the whole fix: the early `n/a` returns must not be reachable before it."""
    src = inspect.getsource(CR.codegen_smoke)
    pre = src.index("_model_for")
    na = src.index('"n/a (ISA is not fixed-format')
    assert pre < na, "the n/a shortcut precedes the prerequisite check — the hole is back"


def test_the_prerequisite_failure_returns_FALSE():
    """It must fail CLOSED. Returning True here is exactly what let a doomed run start."""
    src = inspect.getsource(CR.codegen_smoke)
    # boundary is the START of the next block, not a string inside it — slicing to the n/a literal cuts
    # mid-statement and captures the `return True,` that introduces it
    seg = src[src.index("_model_for"):src.index("from .isa_model import")]
    assert "return False" in seg, "the prerequisite branch does not fail closed"
    assert "return True" not in seg, "the prerequisite branch can still report a pass"


def test_the_reason_names_the_cause_and_the_remedy():
    """A NO_GO that does not say WHY sends the reader hunting; this one cost 101 minutes to diagnose."""
    src = inspect.getsource(CR.codegen_smoke)
    # boundary is the START of the next block, not a string inside it — slicing to the n/a literal cuts
    # mid-statement and captures the `return True,` that introduces it
    seg = src[src.index("_model_for"):src.index("from .isa_model import")]
    assert "MERLIN_MLC_DIR" in seg, "the remedy is not named"
    assert "every capsule" in seg.lower() or "grade only" in seg.lower(), (
        "the consequence is not stated")


def test_it_is_scoped_by_derived_routing_not_a_target_name():
    """Dispatched on the target's declared reference sim, like the rest of the function — a target name
    here would make the guard useless for the next SIMT target."""
    src = inspect.getsource(CR.codegen_smoke)
    seg = src[src.index("PREREQUISITE FIRST"):src.index("from .isa_model import")]
    assert "_bespoke_sim_via(target)" in seg, "the guard is not routed from derived facts"
    assert '"radiance"' not in seg, "the guard hardcodes a target name"


# --- a check that DID NOT RUN is not a pass ------------------------------------------------------
# The prerequisite hole above was one instance of a recurring class: a check that could not run
# reporting success. `merlincirct_gemarm4_codex3` recorded `codegen_ok: true` (reason "n/a (ISA is not
# fixed-format …)") on a submission an independent regrade scored 1/23 — 22 capsules passing the
# command-buffer tiers and failing on verilator. Every run sampled, gemmini and atlas, both arms,
# carried the same true/n-a pair. The verdict is now tri-state: True ran+passed, False ran+failed,
# None did not run.

def test_no_n_a_return_reports_a_pass():
    """The literal hole: `return True, "n/a (…)"`. Any occurrence is the bug coming back."""
    src = inspect.getsource(CR.codegen_smoke)
    lines = [l.strip() for l in src.splitlines()]
    offenders = [l for l in lines if l.startswith("return True") and "n/a" in l]
    assert not offenders, f"a check that did not run reports a pass: {offenders}"


def test_every_did_not_run_path_returns_none():
    """Each n/a branch must be spelled None, so the artifact records null rather than true."""
    src = inspect.getsource(CR.codegen_smoke)
    na = [l.strip() for l in src.splitlines() if '"n/a' in l or "f\"n/a" in l]
    assert na, "no n/a branches found — the function shape changed; re-read this test"
    for l in na:
        if l.startswith("return"):
            assert l.startswith("return None"), f"n/a branch is not None: {l}"


def test_the_signature_admits_the_middle_value():
    """A `-> tuple[bool, str]` annotation cannot express 'did not run', and invites `not ok`."""
    assert "bool | None" in str(inspect.signature(CR.codegen_smoke)) or \
           "bool | None" in inspect.getsource(CR.codegen_smoke).split("\n")[0], \
        "the return type does not admit None"


def test_the_caller_gates_on_is_false_not_falsiness():
    """`if not ok` would refuse every target the smoke does not cover; `is False` is the contract."""
    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "experiments/capsule_bench/harness/run_baseline_qa_loop.py").read_text()
    i = src.index("codegen_smoke(")
    seg = src[i:i + 1200]
    assert "if _cg_ok is False:" in seg, "the launcher no longer gates on `is False`"
    assert "if not _cg_ok:" not in seg, "falsiness gate is back — None would become a NO_GO"
