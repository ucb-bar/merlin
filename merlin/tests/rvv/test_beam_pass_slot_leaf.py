"""The beam's CODEGEN leaf: an escalation the search cannot fork is handed to the pass slot.

The ladder was complete except for its last step. The CCA lifts the loss, the router picks the
cheapest action, the beam forks and measures it, `achieved_residual` says the promise went unmet, and
`route_escalated` returns the next rung -- and when that rung is not forkable the beam recorded a
work-item and stopped. A human then carried it to the slot by hand, which is exactly the manual step
this loop exists to remove.

Off by default: a slot turn costs an agent and a build, so the ladder never enters it implicitly.
"""
import os

import pytest

from merlin.mining import load_rvv_package
from merlin.mining.beam import run_beam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAND_V0 = os.path.join(ROOT, "out/artifacts/targets", "rvv", "hand_v0")


def _certify_with_objdump(ours_objd):
    """Certify that emits an objdump so CCA divergences (and therefore escalations) can surface."""
    from pathlib import Path

    def cert(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
        gen = Path(runs_root) / run_id / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        (gen / "objdump.txt").write_text(ours_objd)
        pkg = load_rvv_package(package_dir)
        n = pkg.op_match[0]["vector"][-2] if pkg.op_match else 8
        return {"correctness": {"gate_ok": True},
                "measurement": [{"target": "k1", "cycle_accurate": False,
                                 "cycles": 4_000_000 // n, "wall_ns": 900_000 // n}]}
    return cert


@pytest.fixture
def ours_objd():
    from merlin.common.paths import merlin_dir
    return (merlin_dir() / "tests" / "data" / "cca_asm" / "ours_baseline_matmul.objdump").read_text()


def _run(tmp_path, ours_objd, **kw):
    """Drive the beam on an axis that actually HAS an unforkable rung above its cheap one.

    Injecting the divergence rather than lifting it: most axes' ladders are exhausted at the PASS
    rung (compute.contraction_form escalates to nothing), so a test that lifts whatever the fixture
    happens to diverge on would silently exercise no escalation at all and pass for the wrong reason.
    compute.activation_vectorization is the real case -- PASS
    (impr_features:vectorized_transcendental_activation) then CODEGEN (pass:llvmlower/act_poly.py),
    and it is the axis this leaf was built for."""
    from merlin.kernels.cca_compare import Divergence
    d = Divergence(axis="compute.activation_vectorization", expert="vectorized_polynomial",
                   ours="scalar_libm_call", backend="rvv")
    return run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "matmul"},
                    runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                    width=2, depth=1, top_k=1, timestamp="t", targets=("k1",),
                    compare_fn=lambda ours: [d],
                    certify_fn=_certify_with_objdump(ours_objd), **kw)


def test_the_slot_is_not_entered_unless_asked_for(tmp_path, ours_objd):
    """A slot turn costs an agent and a build. Nothing may trigger that implicitly."""
    out = _run(tmp_path, ours_objd)
    assert all(not n.get("pass_slot") for n in out["nodes"])


def test_an_unforkable_escalation_reaches_the_slot(tmp_path, ours_objd):
    seen = []

    def slot(action, *, parent_run_id=None):
        seen.append((getattr(action, "divergence_axis", None),
                     getattr(action, "action_class", None), parent_run_id))
        return {"seam": action.target_seam, "actionable": True, "accepted": False}

    out = _run(tmp_path, ours_objd, pass_slot_fn=slot)
    assert seen, "no escalation reached the slot"
    # only rungs the search itself cannot express, and the axis is the one that escalated
    assert all(cls in ("CODEGEN", "PASS", "RUNTIME", "HEURISTIC") for _ax, cls, _p in seen)
    assert any(ax == "compute.activation_vectorization" for ax, _c, _p in seen)
    assert any(n.get("pass_slot") for n in out["nodes"]), "the outcome was not recorded on the node"


def test_the_slot_gets_the_FORK_that_produced_the_residual_not_the_seed(tmp_path, ours_objd):
    """A feature-gated pass is never imported when its feature is off, so gating against the frozen
    baseline would measure a build that never ran the proposal."""
    seen = []
    _run(tmp_path, ours_objd,
         pass_slot_fn=lambda a, *, parent_run_id=None: seen.append(parent_run_id) or {})
    assert seen and all(p and p != "hand_v0__beam" for p in seen), \
        f"the slot was pointed at the seed rather than the fork: {seen}"


def test_a_slot_failure_never_kills_the_search(tmp_path, ours_objd):
    """The search's own measured results must survive a failing leaf -- they cost board time."""
    def boom(action, *, parent_run_id=None):
        raise RuntimeError("agent unreachable")

    out = _run(tmp_path, ours_objd, pass_slot_fn=boom)
    assert out["nodes"], "the beam died on a slot failure"
    recs = [r for n in out["nodes"] for r in (n.get("pass_slot") or ())]
    assert recs and all("agent unreachable" in (r.get("error") or "") for r in recs), \
        "the failure must be RECORDED, not swallowed"


def test_a_slot_that_declines_records_nothing_on_the_node(tmp_path, ours_objd):
    """Returning None is how a slot says 'not mine' without polluting the record."""
    out = _run(tmp_path, ours_objd, pass_slot_fn=lambda a, *, parent_run_id=None: None)
    assert all(not n.get("pass_slot") for n in out["nodes"])


# --------------------------------------------------------------------------- the real slot fn

def test_make_pass_slot_fn_reports_an_unactionable_seam_instead_of_dropping_it(tmp_path):
    """Four of the six blocked seams name no module. "This rung needs a pass that does not exist,
    and here is the declared reason" is the honest outcome -- dropping it would make the ladder look
    complete when it is not."""
    from merlin.mining.pass_slot_wiring import make_pass_slot_fn

    class _A:
        target_seam = "pass:tile-epilogue-store-once (eliminate the rank-generic copy)"
        divergence_axis = "envelope.runtime_calls"
        action_class = "CODEGEN"

    fn = make_pass_slot_fn(frozen_pkg=tmp_path / "seed", model_dir=tmp_path / "wl",
                           runs_root=tmp_path / "runs", targets_root=tmp_path / "targets")
    rec = fn(_A(), parent_run_id="fork_1")
    assert rec["actionable"] is False
    assert "rank-generic" in rec["reason"], "the catalog's declared reason must reach the record"
    assert rec["axis"] == "envelope.runtime_calls"


def test_make_pass_slot_fn_refuses_when_the_fork_package_is_gone(tmp_path):
    """Fail closed: with no package to build, there is nothing to gate a proposal against."""
    from merlin.mining.pass_slot_wiring import make_pass_slot_fn
    from merlin.kernels import action_catalog as ac
    from merlin.kernels.cca_compare import Divergence

    d = Divergence(axis="compute.activation_vectorization", expert="vectorized_polynomial",
                   ours="scalar_libm_call", backend="rvv")
    action = ac.route_escalated(d, ac.route(d).action_class)
    fn = make_pass_slot_fn(frozen_pkg=tmp_path / "seed", model_dir=tmp_path / "wl",
                           runs_root=tmp_path / "runs", targets_root=tmp_path / "targets")
    rec = fn(action, parent_run_id="absent_fork")
    assert rec["actionable"] is False and "not on disk" in rec["reason"]


def test_make_pass_slot_fn_runs_the_turns_and_records_each_one(tmp_path):
    """With an injected proposer and certify, the whole leaf runs with no agent and no toolchain."""
    from merlin.kernels import action_catalog as ac
    from merlin.kernels.cca_compare import Divergence
    from merlin.mining.pass_slot import PassProposal
    from merlin.mining.pass_slot_wiring import make_pass_slot_fn
    from pathlib import Path

    work = tmp_path / "targets" / "fork_1"
    work.mkdir(parents=True)
    d = Divergence(axis="compute.activation_vectorization", expert="vectorized_polynomial",
                   ours="scalar_libm_call", backend="rvv")
    action = ac.route_escalated(d, ac.route(d).action_class)

    calls = {"n": 0}

    def certify(env, *, package_dir, model_dir, runs_root, run_id, targets, timeout):
        calls["n"] += 1
        gen = Path(runs_root) / run_id / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        mnem = "vfmacc.vv v1,v2,v3"
        if run_id == "gate_bitexact_overlay":
            mnem += "\n   4:\t02b7f0d7          \tvfmul.vv\tv4,v5,v6"
        (gen / "objdump.txt").write_text(
            f"0000000000000000 <forward>:\n   0:\t02b7f0d7          \t{mnem}\n")
        return {"status": "ok", "correctness": {"gate_ok": True, "fp32_cos": 1.0},
                "measurement": [], "_imported": ["merlin.llvmlower.act_poly"]}

    fn = make_pass_slot_fn(frozen_pkg=tmp_path / "seed", model_dir=tmp_path / "wl",
                           runs_root=tmp_path / "runs", targets_root=tmp_path / "targets",
                           max_turns=2, certify=certify,
                           propose_fn=lambda a, *, feedback=None: PassProposal(
                               module="merlin.llvmlower.act_poly", source="X = 1\n"))
    rec = fn(action, parent_run_id="fork_1")
    assert rec["actionable"] is True
    assert rec["module"] == "merlin.llvmlower.act_poly"
    assert rec["turns"], "no turn was recorded"
    assert all("stage" in t for t in rec["turns"])
    assert rec["accepted"] is False    # the fake CCA achieves no facet
    assert calls["n"] > 0, "the gate never built anything"
