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
