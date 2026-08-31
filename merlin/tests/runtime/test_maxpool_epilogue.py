"""The fused max-pool epilogue: one definition, three engines, and no silent skip.

WHY THIS FILE EXISTS. The target's elaborated config sets ``has_max_pool``, so its capability
manifest declares ``{family: reduction, dtypes: [int8], composed_with: [contraction, movement]}`` and
the derived conformance requirement demands ``reduction/i8/aligned`` and ``reduction/i8/partial``.
Both were uncovered, and the reason they could not simply be *declared* covered is the subject of
these tests: at the time, the golden engine, the reference recomputation and the simulator all walked
their epilogue lists with no terminal ``else``. A capsule declaring a stage none of them implemented
had that stage SKIPPED — in silence, in all three — so the golden and the reference agreed on a value
neither had computed, L0 passed, and the cover credited the capsule for a family nobody exercised.

So the contract pinned here is two-sided:

* the three engines must produce the SAME pooled tensor (they grade each other; if each grew its own
  pooling loop they would be free to disagree about window order or the ragged tail); and
* every pooling parameter that is absent must RAISE, naming itself — never be defaulted. A window
  this code picked would be a target fact invented in library code, and the integer gate would then
  enforce arithmetic nobody chose.
"""
from __future__ import annotations

import pytest

from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import SimulationError, simulate
from merlin.runtime.tensor import Tensor, pool_out_dims

TILE = 16


def _commit_cb(*, M, K, N, pool: dict | None, epilogue=("maxpool",)):
    """A RES_PACK / MATMUL / COMMIT buffer — the shape a conv or a contraction is lowered to."""
    attrs: dict = {"epilogue": list(epilogue), "output_dtype": "i32"}
    attrs.update(pool or {})
    return {
        "abi_version": "0.1", "target": "t",
        "tensors": {"W": {"shape": [K, N], "dtype": "i8", "role": "weight"},
                    "A0": {"shape": [M, K], "dtype": "i8", "role": "input"},
                    "Y0": {"shape": [M, N], "dtype": "i32", "role": "output"}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wr"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "Wr", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}, "attributes": attrs},
        ],
    }


_POOL_2x2 = {"pool_in_dims": [4, 4], "pool_size": [2, 2], "pool_stride": [2, 2],
             "pool_padding": [0, 0, 0, 0]}


class TestTheEnginesAgree:
    def test_reference_and_simulate_produce_the_same_pooled_tensor(self):
        cb = _commit_cb(M=TILE, K=TILE, N=TILE, pool=_POOL_2x2)
        assert reference_outputs(cb) == simulate(cb)["outputs"]

    def test_the_pooled_commit_has_the_pooled_extent_not_the_accumulator_shape(self):
        """Pooling is the one epilogue stage that CHANGES the committed extent. A 4x4 plane at 2x2/2
        commits 4 rows, not 16 — an engine that returned the accumulator's shape would look right to
        every elementwise check and be wrong by a factor of four."""
        got = reference_outputs(_commit_cb(M=TILE, K=TILE, N=TILE, pool=_POOL_2x2))["Y0"]
        assert (len(got), len(got[0])) == (4, TILE)

    def test_the_pooled_value_is_the_window_max_computed_independently(self):
        """Recompute the expected value from the Tensor primitives directly, so this asserts the
        arithmetic rather than that two callers of one function agree."""
        cb = _commit_cb(M=TILE, K=TILE, N=TILE, pool=_POOL_2x2)
        acc = (Tensor.deterministic("A0", (TILE, TILE), "i8")
               .matmul(Tensor.deterministic("W", (TILE, TILE), "i8")))
        # window 0 of row-plane 0: rows 0,1 x cols 0,1 of the 4x4 plane -> flat rows 0, 1, 4, 5.
        want_col0 = max(acc.data[r * TILE + 0] for r in (0, 1, 4, 5))
        assert reference_outputs(cb)["Y0"][0][0] == want_col0

    def test_the_ragged_tail_is_dropped_not_folded_in(self):
        """A 5x5 plane at 2x2/2 pools to 2x2: the fifth row and column are covered by NO window. This
        repo has already shipped a taped-out unit that got a partial tile wrong while every functional
        check passed, so the tail is the case worth pinning."""
        assert pool_out_dims(5, 5, (2, 2), (2, 2), (0, 0, 0, 0)) == (2, 2)
        pool = {**_POOL_2x2, "pool_in_dims": [5, 5]}
        cb = _commit_cb(M=25, K=TILE, N=TILE + 1, pool=pool)
        got = reference_outputs(cb)
        assert got == simulate(cb)["outputs"]
        assert (len(got["Y0"]), len(got["Y0"][0])) == (4, TILE + 1)


class TestItFailsClosed:
    @pytest.mark.parametrize("missing", ["pool_in_dims", "pool_size", "pool_stride"])
    def test_an_absent_pool_parameter_raises_naming_itself_in_both_engines(self, missing):
        pool = {k: v for k, v in _POOL_2x2.items() if k != missing}
        cb = _commit_cb(M=TILE, K=TILE, N=TILE, pool=pool)
        with pytest.raises(ValueError, match=missing):
            reference_outputs(cb)
        with pytest.raises(SimulationError, match=missing):
            simulate(cb)

    def test_an_unimplemented_epilogue_stage_is_refused_by_the_reference(self):
        """The bug this whole feature came out of. The reference's epilogue loop had no terminal
        branch, so an unknown stage was dropped silently — and because the golden dropped it too, the
        two AGREED and the capsule passed having proved nothing about the stage it declared."""
        cb = _commit_cb(M=TILE, K=TILE, N=TILE, pool=None, epilogue=("avgpool",))
        with pytest.raises(ValueError, match="avgpool"):
            reference_outputs(cb)

    def test_a_pool_geometry_that_does_not_divide_the_rows_raises(self):
        """16 accumulator rows are not a whole number of 5x5 planes. Rounding down would pool a
        truncated image and still return a well-formed tensor."""
        cb = _commit_cb(M=TILE, K=TILE, N=TILE, pool={**_POOL_2x2, "pool_in_dims": [5, 5]})
        with pytest.raises(ValueError, match="whole multiple"):
            reference_outputs(cb)

    def test_padding_without_a_declared_pad_value_raises(self):
        """The identity element of a max over a padded cell is a datapath property (-inf
        mathematically, commonly 0 in a store path). Choosing one silently would be a full tensor of
        plausible wrong numbers."""
        cb = _commit_cb(M=TILE, K=TILE, N=TILE,
                        pool={**_POOL_2x2, "pool_padding": [1, 1, 1, 1]})
        with pytest.raises(ValueError, match="pad_value"):
            reference_outputs(cb)


class TestTheFusedConvPath:
    """CONV2D pools its own [N*Ho*Wo, Co] product, which is the form the fused conv loop offers."""

    @staticmethod
    def _conv_cb(*, H=8, W=8, ci=4, kh=3, kw=3, co=TILE, pool_in_dims=(6, 6)):
        return {
            "abi_version": "0.1", "target": "t",
            "tensors": {"IFM": {"shape": [1, H, W, ci], "dtype": "i8", "role": "input"},
                        "Wt": {"shape": [kh * kw * ci, co], "dtype": "i8", "role": "weight"},
                        "Y0": {"shape": [9, co], "dtype": "i32", "role": "output"}},
            "commands": [{"opcode": "CONV2D",
                          "operands": {"ifm": "IFM", "weight": "Wt", "dst": "Y0"},
                          "attributes": {"kernel": [kh, kw, ci, co], "stride": [1, 1],
                                         "padding": [0, 0, 0, 0], "dilation": [1, 1],
                                         "layout": "nhwc", "epilogue": ["maxpool"],
                                         "output_dtype": "i32",
                                         "pool_in_dims": list(pool_in_dims), "pool_size": [2, 2],
                                         "pool_stride": [2, 2], "pool_padding": [0, 0, 0, 0]}}],
        }

    def test_the_conv_pools_its_own_output_plane(self):
        got = simulate(self._conv_cb())["outputs"]["Y0"]
        assert (len(got), len(got[0])) == (9, TILE)          # 6x6 -> 3x3 at 2x2/2

    def test_a_pool_extent_that_disagrees_with_the_conv_geometry_is_rejected(self):
        """The golden sees only the flat product and has to trust the declaration; if the two ever
        diverged, the golden and the simulator would pool different images and the mismatch would read
        as a numeric defect rather than a geometry one."""
        with pytest.raises(SimulationError, match="disagrees with the conv"):
            simulate(self._conv_cb(pool_in_dims=(4, 9)))
