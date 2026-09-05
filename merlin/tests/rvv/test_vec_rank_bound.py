"""The rank bound of the bounded-vectorize lever is a MEASURED coverage decision, not a constant.

The non-contraction vectorize arms covered loop ranks 2..4 because that is how many arms had been
written out by hand, and the tagging predicate refused everything else with no counter saying so.
MEASURED on the int8 recaptures (frozen-baseline lowering, per-op attribution of the emitted
``forward`` -- ``build_tools/scripts/scalar_remainder.py``), the ops whose ONLY reason for being
refused is that bound account for **19.2 % of one model's entire scalar remainder** (52 ops, 5,653
scalar instructions), 1.6 % of a second model's and 0.7 % of a third's. The high ranks come from the
shape the frontend expands a convolution into -- a rank-7 ``tensor<64x8x3x3x1x8x12xf32>`` weight
expansion -- so this is a property of the captured graphs, not of any one device.

What is pinned here:

* the DEFAULT is byte-identical. Generating the arms in a loop must reproduce the hand-written
  rank-2..4 text exactly, and a feature set that names nothing must leave the tagger switched off,
  or every existing measurement is invalidated.
* the bound is ONE number driving BOTH halves. The tagger admits a rank only if the schedule carries
  an arm for it; tagging a rank no arm matches leaves the op scalar while the tagger reports it as
  tagged -- the same silent-coverage failure the specialization pass caused for the tags themselves.
* the sub-byte REFUSAL extends to the new arms. The miscompile ``test_vec_subbyte_element`` documents
  is a property of the arm shape, so an arm added at a new rank without the gate would reintroduce
  it at that rank.
* the point is DERIVABLE FROM ITS NAME, so it resolves in the lowering subprocess, which re-imports
  the registry and cannot see a registration the parent made at run time.

Deliberately NOT asserted here: that raising the bound is a speedup. That is a board measurement,
and the emitted-code delta is what this change is entitled to claim.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import impr_features as impr
from merlin.llvmlower import pipeline as P

VEC = impr.VEC_NONCONTRACTION_NAME


# ---------------------------------------------------------------------------------------------
# 1. the default is byte-identical
# ---------------------------------------------------------------------------------------------

def test_the_default_arms_cover_exactly_ranks_two_to_four():
    arms = impr._vec_rank_arms(8, "PFX_")
    for rank in range(impr.VEC_NONCONTRACTION_MIN_RANK, impr.VEC_NONCONTRACTION_MAX_RANK + 1):
        assert f"merlin.vec_r{rank}" in arms
    assert f"merlin.vec_r{impr.VEC_NONCONTRACTION_MAX_RANK + 1}" not in arms


def test_the_generated_arms_reproduce_the_hand_written_tile_shapes():
    """The tile list is ones with the lane count last, and the loop arity is rank+1 handles."""
    arms = impr._vec_rank_arms(8, "PFX_")
    assert "tile_sizes [1, 8]" in arms
    assert "tile_sizes [1, 1, 8]" in arms
    assert "tile_sizes [1, 1, 1, 8]" in arms
    assert "%gt2, %gl2:2 = " in arms and "%gt4, %gl4:4 = " in arms
    # rank 2 tiles two dims -> the tiled op plus two loop handles.
    assert arms.count("(!transform.any_op) -> (!transform.any_op, !transform.any_op, "
                      "!transform.any_op)\n") == 1


def test_the_frozen_baseline_never_reaches_the_tagger():
    """No feature named -> no rank, no lanes, so ``prepare_for_lowering`` never tags anything."""
    assert impr.vec_noncontraction_max_rank(frozenset()) is None
    assert impr.vec_noncontraction_lanes(frozenset()) is None


def test_the_bare_name_still_means_the_old_bound():
    assert impr.vec_noncontraction_max_rank([VEC]) == impr.VEC_NONCONTRACTION_MAX_RANK
    assert impr.vec_noncontraction_lanes([VEC]) == impr.VEC_NONCONTRACTION_LANES


# ---------------------------------------------------------------------------------------------
# 2. the raised bound actually adds arms, and the gate comes with them
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("max_rank", [5, 6, 8])
def test_raising_the_bound_adds_one_gated_arm_per_rank(max_rank):
    arms = impr._vec_rank_arms(8, "PFX_", max_rank)
    for rank in range(impr.VEC_NONCONTRACTION_MIN_RANK, max_rank + 1):
        line = [l for l in arms.splitlines() if f"merlin.vec_r{rank}" in l]
        assert len(line) == 1, (rank, arms)
        # EVERY arm carries the byte-addressable-element gate. An arm added at a new rank without it
        # reintroduces the packed-sub-byte-store miscompile at that rank.
        assert impr.VEC_BYTEWISE_ATTR in line[0], (rank, line)
    assert f"merlin.vec_r{max_rank + 1}" not in arms


def test_the_spliced_schedule_carries_the_raised_arms_and_their_matchers():
    text = impr._splice_vec_rank_arms(P.RVV_TRANSFORM_SCHEDULE, impr.VEC_NONCONTRACTION_LANES, 7)
    assert "merlin.vec_r7" in text
    assert "transform.match.structured.elemental_bitwidth" in text
    # additive: the contraction arms it layers onto are untouched.
    assert 'transform.structured.match ops{["linalg.matmul"]}' in text


def test_splicing_stays_idempotent_at_a_raised_bound():
    once = impr._splice_vec_rank_arms(P.RVV_TRANSFORM_SCHEDULE, 8, 7)
    assert impr._splice_vec_rank_arms(once, 8, 7) == once


# ---------------------------------------------------------------------------------------------
# 3. the point is derivable from its name
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,expect", [
    (VEC, (impr.VEC_NONCONTRACTION_LANES, impr.VEC_NONCONTRACTION_MAX_RANK)),
    (f"{VEC}_l16", (16, impr.VEC_NONCONTRACTION_MAX_RANK)),
    (f"{VEC}_r8", (impr.VEC_NONCONTRACTION_LANES, 8)),
    (f"{VEC}_l16_r6", (16, 6)),
    (f"{VEC}_r6_l32", (32, 6)),
])
def test_a_point_is_read_off_its_name(name, expect):
    assert impr._vec_noncontraction_point(name) == expect


@pytest.mark.parametrize("name", [f"{VEC}_z3", f"{VEC}_l", f"{VEC}_r0", f"{VEC}_lx",
                                  "erase_self_copy", f"{VEC}_l8_"])
def test_an_unparsable_point_is_refused_rather_than_defaulted(name):
    """Reading a mistyped point as the default is how an experiment measures the baseline and
    reports it as the lever."""
    assert impr._vec_noncontraction_point(name) is None


def test_a_named_point_resolves_in_a_fresh_process(tmp_path):
    """The lowering runs in a SUBPROCESS that re-imports this registry, so a point registered on
    demand in the parent is invisible there -- and the build then fails with "unknown impr feature"
    rather than with the lever off. This is that subprocess."""
    script = tmp_path / "resolve_point.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root() / 'merlin' / 'python')!r})\n"
        "from merlin.llvmlower import impr_features as impr\n"
        "from merlin.llvmlower import pipeline as P\n"
        "f = impr.get('vectorize_non_contraction_generics_r7')\n"
        "sched = f.edit_schedule(P.RVV_TRANSFORM_SCHEDULE)\n"
        "print('r7' if 'merlin.vec_r7' in sched else 'MISSING')\n",
        encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "r7", proc.stdout


def test_the_registered_point_keeps_the_hygiene_the_bare_name_implies():
    """A raised bound that lost the self-copy erase would be measured in a realization the bare
    name is not, and the comparison between them would be meaningless."""
    assert impr.get(f"{VEC}_r7").implies == impr.get(VEC).implies


# ---------------------------------------------------------------------------------------------
# 4. the tagger obeys the same number
# ---------------------------------------------------------------------------------------------

#: An all-parallel generic of rank 6 whose innermost extent is a whole multiple of the lane count --
#: i.e. an op the predicate refuses for its RANK ALONE. Shaped like the convolution weight expansion
#: the frontend emits, which is where the refused rank-5..7 ops in the measured models come from.
RANK6 = """
#p = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
func.func @forward(%a: tensor<2x2x2x2x2x16xf32>) -> tensor<2x2x2x2x2x16xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<2x2x2x2x2x16xf32>
  %r = linalg.generic {indexing_maps = [#p, #p],
                       iterator_types = ["parallel", "parallel", "parallel",
                                         "parallel", "parallel", "parallel"]}
      ins(%a : tensor<2x2x2x2x2x16xf32>) outs(%e : tensor<2x2x2x2x2x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.maximumf %in, %z : f32
    linalg.yield %m : f32
  } -> tensor<2x2x2x2x2x16xf32>
  return %r : tensor<2x2x2x2x2x16xf32>
}
"""


def _tag_count(tmp_path, max_rank: int) -> int:
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    src = tmp_path / f"m{max_rank}.mlir"
    src.write_text(RANK6, encoding="utf-8")
    work = tmp_path / f"w{max_rank}"
    work.mkdir(parents=True, exist_ok=True)
    out = _prepare_model_mlir(src, work, tag_vec_ranks=True, vec_max_rank=max_rank)
    return out.read_text(encoding="utf-8").count("merlin.vec_r6")


def test_the_tagger_declines_the_rank_the_schedule_has_no_arm_for(tmp_path):
    assert _tag_count(tmp_path, impr.VEC_NONCONTRACTION_MAX_RANK) == 0


def test_the_tagger_admits_the_rank_once_the_bound_is_raised(tmp_path):
    assert _tag_count(tmp_path, 6) == 1


def test_the_refused_ranks_are_counted_not_silently_dropped(tmp_path, capsys):
    """A predicate that drops work without a counter is how the whole class stayed invisible."""
    _tag_count(tmp_path, impr.VEC_NONCONTRACTION_MAX_RANK)
    assert "outside the rank bound" in capsys.readouterr().out
