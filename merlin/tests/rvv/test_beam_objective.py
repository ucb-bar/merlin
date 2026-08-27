"""Beam readiness fixes (board-free, mock certify_fn):

P1 — two-phase whole-model objective: explore generations score on a cheap proxy ``certify_fn``, the
     survivor set is RE-CERTIFIED with a full ``validate_fn`` before promotion / best-selection.
P3 — noise-floor margin gate + inert gating: a fork is a win only if it beats its parent by more than
     the board noise floor; inert forks (byte-identical emitted code) are excluded from survivors and
     sort last.

All tests use a crafted mock certify_fn — no board, no toolchain — so ranking/promotion is
deterministic. The board-measured behavior (whole-model campaign) is validated separately.
"""
import os
from pathlib import Path

from merlin.common.paths import merlin_dir, repo_root
from merlin.mining import load_rvv_package
from merlin.mining.beam import (
    _margin_improved,
    _ranked_speedup,
    _resolve_margin,
    run_beam,
)
from merlin.mining.sweep import rank_results

HAND_V0 = str(repo_root() / "out/artifacts/targets/rvv/hand_v0")
_DIVS = ["lmul_class: expert='m4' vs ours='m2'",
         "fma_form: expert='vf' vs ours=None",
         "vl_strategy: expert='vsetvl_loop' vs ours='vsetivli_fixed'"]


def _is_seed(package_dir) -> bool:
    return os.path.abspath(str(package_dir)) == os.path.abspath(HAND_V0)


# --------------------------------------------------------------------------------------------------
# P3 — noise-floor margin helpers (pure, deterministic)
# --------------------------------------------------------------------------------------------------

def test_resolve_margin_default_env_and_param():
    assert _resolve_margin(None) == 0.02                      # measured >=1.9% K1 floor default
    assert _resolve_margin(0.05) == 0.05                      # explicit param wins
    os.environ["MERLIN_BEAM_NOISE_MARGIN"] = "0.1"
    try:
        assert _resolve_margin(None) == 0.1                  # env override when no param
        assert _resolve_margin(0.03) == 0.03                 # param still beats env
    finally:
        del os.environ["MERLIN_BEAM_NOISE_MARGIN"]


def test_margin_improved_only_credits_above_the_noise_floor():
    # +1% over parent with a 2% floor -> NOT a win (board noise); +3% -> a real win.
    assert _margin_improved(1.01, 1.0, 0.02) is False
    assert _margin_improved(1.03, 1.0, 0.02) is True
    # missing either speedup -> never a credited win (fail-closed).
    assert _margin_improved(None, 1.0, 0.02) is False
    assert _margin_improved(1.5, None, 0.02) is False


def test_ranked_speedup_clamps_sub_margin_to_a_tie_but_keeps_wins_and_regressions():
    # sub-margin above parent -> pinned to parent's speed (a tie, no noise-promoted win)
    assert _ranked_speedup(1.01, 1.0, 0.02) == 1.0
    # genuine win -> keep the measured speedup
    assert _ranked_speedup(1.5, 1.0, 0.02) == 1.5
    # genuine regression -> keep the (lower) measured speedup so it sorts below the parent
    assert _ranked_speedup(0.8, 1.0, 0.02) == 0.8
    # no parent speedup -> passthrough
    assert _ranked_speedup(1.5, None, 0.02) == 1.5


# --------------------------------------------------------------------------------------------------
# P3 — rank_results: inert sorts last; ranked_speedup drives when present
# --------------------------------------------------------------------------------------------------

def test_rank_results_inert_sorts_last_on_a_tie():
    nodes = [
        {"run_id": "inert", "gate_ok": True, "speedup": 1.0, "structural_match": 0.9,
         "inert": True},
        {"run_id": "live", "gate_ok": True, "speedup": 1.0, "structural_match": 0.9,
         "inert": False},
    ]
    ranked = [n["run_id"] for n in rank_results(nodes)]
    assert ranked[0] == "live"           # non-inert wins the otherwise-perfect tie
    assert ranked[-1] == "inert"


def test_rank_results_uses_ranked_speedup_when_present():
    # a fork whose raw speedup looks great but whose ranked_speedup was clamped (noise/inert) must
    # NOT out-rank a genuinely-faster fork.
    nodes = [
        {"run_id": "clamped", "gate_ok": True, "speedup": 1.9, "ranked_speedup": 1.0,
         "structural_match": 0.5},
        {"run_id": "real_win", "gate_ok": True, "speedup": 1.4, "ranked_speedup": 1.4,
         "structural_match": 0.5},
    ]
    assert [n["run_id"] for n in rank_results(nodes)][0] == "real_win"


def test_rank_results_backward_compatible_without_new_fields():
    # nodes with neither inert nor ranked_speedup rank exactly on raw speedup (legacy contract).
    nodes = [
        {"run_id": "a", "gate_ok": True, "speedup": 1.2},
        {"run_id": "b", "gate_ok": True, "speedup": 1.5},
    ]
    assert [n["run_id"] for n in rank_results(nodes)][0] == "b"


# --------------------------------------------------------------------------------------------------
# P3 — margin gate inside run_beam (mock certify, no board)
# --------------------------------------------------------------------------------------------------

def _certify_walls(seed_wall: int, fork_wall: int):
    """A mock certify_fn: a fixed K1 wall for the seed vs each fork (identified by package_dir), plus
    the _DIVS so the fingerprint proposer mints forks. No objdump -> forks are never inert here."""
    def _fn(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
        wall = seed_wall if _is_seed(package_dir) else fork_wall
        return {"correctness": {"gate_ok": True},
                "measurement": [{"target": "k1", "wall_ns": wall}],
                "divergences": _DIVS}
    return _fn


def test_margin_gate_treats_sub_margin_fork_as_a_tie(tmp_path):
    # fork is 1% faster than the seed (1000 -> 990) — inside the 2% floor -> a TIE, not a win.
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t",
                   certify_fn=_certify_walls(1000, 990))
    fork = next(n for n in out["nodes"] if n["lever"] != "seed")
    assert fork["speedup"] > 1.0                 # raw measurement shows a tiny gain
    assert fork["margin_improved"] is False      # ...but it is within the noise floor
    assert fork["ranked_speedup"] == 1.0         # pinned to the parent's speed (a tie)


def test_margin_gate_credits_a_real_win(tmp_path):
    # fork is 2x faster (1000 -> 500) — a genuine win well above the floor.
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t",
                   certify_fn=_certify_walls(1000, 500))
    fork = next(n for n in out["nodes"] if n["lever"] != "seed")
    assert fork["margin_improved"] is True
    assert fork["ranked_speedup"] == fork["speedup"] > 1.0
    assert out["best"]["run_id"] == fork["run_id"]   # the real win becomes the best


# --------------------------------------------------------------------------------------------------
# P3 — inert forks excluded from survivors (CCA mode, identical emitted objdump)
# --------------------------------------------------------------------------------------------------

def test_inert_forks_are_excluded_from_survivors(tmp_path):
    from merlin.mining.beam_cli import lift_expert_cca

    ours = (merlin_dir() / "tests" / "data" / "cca_asm" / "ours_baseline_matmul.objdump").read_text()
    expert_objd = merlin_dir() / "tests" / "data" / "cca_asm" / "xnnpack_f32_gemm_rvv.objdump"
    expert_cca = lift_expert_cca(expert_objd, "matmul")

    def mock_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
        # EVERY run emits the identical objdump -> every fork is byte-identical to its parent (inert).
        gen = Path(runs_root) / run_id / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        (gen / "objdump.txt").write_text(ours)
        return {"correctness": {"gate_ok": True},
                "measurement": [{"target": "k1", "wall_ns": 500}]}   # a "faster" (noise) wall

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text=ours, op_key={"op": "matmul"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=2, top_k=2, timestamp="t", certify_fn=mock_certify,
                   expert_cca=expert_cca)
    gen1 = [n for n in out["nodes"] if n["depth"] == 1]
    assert gen1, "no forks minted (CCA divergences did not surface)"
    assert all(n["inert"] for n in gen1)                 # identical emitted code -> all inert
    # inert forks are excluded from the survivor set -> nothing is promoted -> no gen-2 forks.
    assert not [n for n in out["nodes"] if n["depth"] == 2]


# --------------------------------------------------------------------------------------------------
# P1 — two-phase objective: validate_fn re-certifies survivors and overrides the explore ranking
# --------------------------------------------------------------------------------------------------

def _explore_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
    """Cheap explore proxy: the fork looks 2x faster than the seed (1000 -> 500)."""
    wall = 1000 if _is_seed(package_dir) else 500
    return {"correctness": {"gate_ok": True},
            "measurement": [{"target": "k1", "wall_ns": wall}], "divergences": _DIVS}


def _validate_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
    """Full validation: the fork is actually SLOWER than the seed on the whole model (1000 -> 2000)."""
    wall = 1000 if _is_seed(package_dir) else 2000
    return {"correctness": {"gate_ok": True},
            "measurement": [{"target": "k1", "wall_ns": wall}], "divergences": _DIVS}


def test_single_phase_promotes_the_explore_winner(tmp_path):
    # baseline (validate_fn=None): the explore-fast fork wins and nothing is validated.
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t", certify_fn=_explore_certify)
    assert out["best"]["lever"] != "seed"
    assert out["best"]["speedup"] > 1.0
    assert all("validated" not in n for n in out["nodes"])   # default behavior: no validation phase


def test_two_phase_validation_overrides_the_explore_winner(tmp_path):
    # with validate_fn, the survivor is re-certified on the full model and revealed to be a regression
    # -> the SEED wins after validation (the explore proxy was optimistic).
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t",
                   certify_fn=_explore_certify, validate_fn=_validate_certify)
    best = out["best"]
    assert best.get("validated") is True          # best comes from the validated pool
    assert best["lever"] == "seed"                # validation demoted the explore-winner fork
    assert best["speedup"] == 1.0
    # the fork itself was re-certified (validated) and shows its true (regressed) speed.
    fork = next(n for n in out["nodes"] if n["lever"] != "seed")
    assert fork.get("validated") is True
    assert fork["speedup"] < 1.0


def test_two_phase_keeps_a_genuine_win(tmp_path):
    # when validation AGREES the fork is faster, the fork stays the best (validated).
    def validate_agrees(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
        wall = 1000 if _is_seed(package_dir) else 600
        return {"correctness": {"gate_ok": True},
                "measurement": [{"target": "k1", "wall_ns": wall}], "divergences": _DIVS}

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t",
                   certify_fn=_explore_certify, validate_fn=validate_agrees)
    best = out["best"]
    assert best.get("validated") is True
    assert best["lever"] != "seed"
    assert best["speedup"] > 1.0
