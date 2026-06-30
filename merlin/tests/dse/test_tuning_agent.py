"""LLM tuning-agent proposer (S5.6): a MOCK llm_fn returns a JSON proposal; assert it yields valid
renderable ForkProposals, drops unknown override keys, degrades gracefully, and plugs into run_beam
as a drop-in for the deterministic gap-router (reusing the mock-certify pattern from
test_rvv_beam.py)."""
import json
import os

from merlin.rvvgen.tuning_agent import propose_forks_llm, build_prompt, prompt_path
from merlin.rvvgen.from_strategy import render_schedule
from merlin.rvvgen.beam import run_beam
from merlin.rvvgen import load_rvv_package
from merlin.kernels.rvv_knobs import ForkProposal

# Reuse the divergences + mock-certify from the beam test (replicated, kept identical).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAND_V0 = os.path.join(ROOT, "generated_targets", "rvv", "hand_v0")

_DIVS = ["lmul_class: expert='m4' vs ours='m2'",
         "fma_form: expert='vf' vs ours=None",
         "vl_strategy: expert='vsetvl_loop' vs ours='vsetivli_fixed'"]


def _mock_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
    pkg = load_rvv_package(package_dir)
    n = pkg.op_match[0]["vector"][-2] if pkg.op_match else 8
    return {"correctness": {"gate_ok": True},
            "measurement": [{"target": "spike", "cycle_accurate": False, "cycles": 4_000_000 // n}],
            "structural_match": min(0.95, 0.45 + 0.02 * n),
            "divergences": _DIVS}


def _mock_llm_good(prompt):
    """A well-behaved agent reply: widen N x2 (renderable), an unknown knob (must be dropped),
    and a non-actionable suggestion (empty overrides -> work-item)."""
    return json.dumps([
        {"overrides": {"op_match": [{"op": "linalg.matmul", "tile": [4, 16, 1],
                                     "vector": [4, 16, 1]}]},
         "rationale": "widen N tile/vector x2 toward higher LMUL", "targets": "lmul_class"},
        {"overrides": {"contraction_strategy": "outerproduct", "frobnicate": 7},
         "rationale": "try outerproduct lowering", "targets": "fma_form"},
        {"overrides": {}, "rationale": "vsetvl-loop needs a scalable-vector lowering path",
         "targets": "vl_strategy"},
    ])


def test_prompt_artifact_exists_and_renders():
    assert prompt_path(1).is_file()
    pkg = load_rvv_package(HAND_V0)
    p = build_prompt(_DIVS, pkg.knobs, context={"policy": "lmul_grouping_policy"})
    assert "op_match" in p and "outerproduct" in p and "lmul_class" in p


def test_llm_proposals_are_valid_and_renderable():
    pkg = load_rvv_package(HAND_V0)
    props = propose_forks_llm(_DIVS, pkg.knobs, llm_fn=_mock_llm_good)
    assert len(props) == 3
    forkable = [p for p in props if p.forkable]
    deferred = [p for p in props if not p.forkable]
    # the two with surviving overrides are forkable knob-levers; the empty one is a work-item.
    assert len(forkable) == 2 and len(deferred) == 1
    assert all(isinstance(p, ForkProposal) and p.lever == "knob" for p in forkable)
    assert deferred[0].targets == "vl_strategy" and deferred[0].lever == "llm_suggestion"
    # every forkable override is renderable by the generator (no exception, real MLIR out).
    for p in forkable:
        knobs = {**pkg.knobs, **p.overrides}
        mlir = render_schedule(knobs)
        assert "transform.named_sequence" in mlir


def test_unknown_override_keys_are_dropped_with_note():
    pkg = load_rvv_package(HAND_V0)
    props = propose_forks_llm(_DIVS, pkg.knobs, llm_fn=_mock_llm_good)
    outer = next(p for p in props if p.targets == "fma_form")
    # the bogus "frobnicate" key is gone; the valid contraction_strategy survives.
    assert "frobnicate" not in outer.overrides
    assert outer.overrides == {"contraction_strategy": "outerproduct"}
    assert "frobnicate" in outer.note  # dropped-key reason recorded


def test_bad_op_match_is_clamped_out():
    pkg = load_rvv_package(HAND_V0)
    bad = json.dumps([{"overrides": {"op_match": [{"op": "linalg.matmul", "tile": [4, 8],
                                                   "vector": [4, 8, 1]}]},
                       "rationale": "mismatched lengths", "targets": "lmul_class"}])
    props = propose_forks_llm(_DIVS, pkg.knobs, llm_fn=lambda _p: bad)
    # the only override is invalid -> dropped -> non-actionable work-item, not a forkable knob.
    assert len(props) == 1 and not props[0].forkable
    assert "op_match" in props[0].note


def test_graceful_on_none_and_garbage():
    pkg = load_rvv_package(HAND_V0)
    assert propose_forks_llm(_DIVS, pkg.knobs, llm_fn=lambda _p: None) == []
    assert propose_forks_llm(_DIVS, pkg.knobs, llm_fn=lambda _p: "sorry, no json here") == []
    # default llm_fn with no API key -> common.llm.complete returns None -> []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        assert propose_forks_llm(_DIVS, pkg.knobs) == []


def test_plugs_into_run_beam(tmp_path):
    """The headline: run_beam(proposer=propose_forks_llm) works unchanged — the LLM proposer is a
    drop-in for the deterministic gap-router. Bind the mock llm_fn via a closure (the beam calls
    proposer(divergences, knobs) with no kwargs)."""
    def llm_proposer(divergences, knobs):
        return propose_forks_llm(divergences, knobs, llm_fn=_mock_llm_good)

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=2, top_k=1, timestamp="t",
                   certify_fn=_mock_certify, proposer=llm_proposer)
    seed = next(n for n in out["nodes"] if n["lever"] == "seed")
    # the widen-N fork the agent proposed should win (higher N -> higher mock structural_match).
    assert out["best"]["structural_match"] >= seed["structural_match"]
    forks = list((tmp_path / "gen" / "rvv").glob("rvv_tuned_v*_d*_*"))
    assert forks, "no fork packages minted by the LLM-proposed beam"
    # the non-actionable vl_strategy suggestion was recorded as a deferred work-item.
    assert any(d["targets"] == "vl_strategy" for d in out["deferred"])
    assert os.path.isfile(out["tree_path"])
