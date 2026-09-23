"""The captured model's groups rendered as our schedules, beside the vendor calls they replace.

The group program is the measurement this effort aims at -- 28,084,005 cycles per image against the
compiled program's 1,709,045,472 -- and that number belongs to the VENDOR library. So our schedules are
a second renderer for the same program rather than a replacement: same groups, same buffers, same
harness, and the difference between the two runs is the schedule and nothing else.

The property that matters most here is that a group we cannot express keeps its library call and says
why. A program that silently fell back would report our schedule's cycles for the library's work.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir


@pytest.fixture(scope="module")
def gmsk():
    """The sibling script, loaded by path -- it is a script beside `group_model_program.py`, not a
    package module, and the test must not depend on which directory pytest was started from."""
    path = merlin_dir() / "experiments/gemmini_perf_bench/scripts/group_model_sched_kernels.py"
    spec = importlib.util.spec_from_file_location("group_model_sched_kernels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model() -> dict:
    """One group of every kind the extractor emits, including one that must be refused."""
    return {
        "steps": [
            # expressible: a plain 3x3 stride-1 convolution reading the image
            {
                "kind": "conv2d",
                "group": 1,
                "in": "IMAGE",
                "weight": "W_1",
                "bias": "BIAS_1",
                "out": "B_1",
                "relu": True,
                "scale": 0.03,
                "in_dim": 56,
                "ci": 64,
                "n": 64,
                "out_dim": 56,
                "stride": 1,
                "padding": 1,
                "kernel": 3,
                "pool": {"size": 0, "stride": 0, "padding": 0},
            },
            # expressible: the readout pools, as ResNet-50's stem does
            {
                "kind": "conv2d",
                "group": 2,
                "in": "B_1",
                "weight": "W_2",
                "bias": "BIAS_2",
                "out": "B_2",
                "relu": False,
                "scale": 0.02,
                "in_dim": 112,
                "ci": 64,
                "n": 64,
                "out_dim": 112,
                "stride": 1,
                "padding": 1,
                "kernel": 3,
                "pool": {"size": 3, "stride": 2, "padding": 1},
            },
            {
                "kind": "matmul",
                "group": 3,
                "in": "B_2",
                "weight": "W_3",
                "bias": "BIAS_3",
                "out": "B_3",
                "relu": True,
                "scale": 0.01,
                "m": 64,
                "k": 64,
                "n": 64,
            },
            # expressible: the classifier leaves as an accumulator, at the accumulator's own width
            {
                "kind": "matmul",
                "group": 4,
                "in": "B_3",
                "weight": "W_4",
                "bias": None,
                "out": "B_4",
                "relu": False,
                "scale": None,
                "m": 1,
                "k": 2048,
                "n": 1000,
            },
            # refused, and it must stay refused: the convolution carries no bias, and the bias mvin is
            # the ONLY thing that initialises the accumulator half a descriptor accumulates into.
            {
                "kind": "conv2d",
                "group": 7,
                "in": "B_2",
                "weight": "W_7",
                "bias": None,
                "out": "B_7",
                "relu": True,
                "scale": 0.02,
                "in_dim": 56,
                "ci": 64,
                "n": 64,
                "out_dim": 56,
                "stride": 1,
                "padding": 1,
                "kernel": 3,
                "pool": {"size": 0, "stride": 0, "padding": 0},
            },
            {
                "kind": "sum",
                "group": 5,
                "lhs": "B_1",
                "rhs": "B_3",
                "out": "B_5",
                "rows": 3136,
                "cols": 256,
                "lhs_load": 1.0,
                "rhs_load": 0.5,
                "readout": 0.25,
                "relu": True,
                "bound_lsb": 1,
            },
            {
                "kind": "mean",
                "group": 6,
                "in": "B_5",
                "out": "B_6",
                "rows": 2048,
                "window": 49,
                "multiplier": 0.00042,
            },
        ]
    }


@pytest.fixture(scope="module")
def rendered(gmsk):
    return gmsk.render_kernels(_model(), target="gemmini")


def test_each_expressible_group_gets_one_function_and_one_call(rendered):
    assert sorted(rendered["calls"]) == [1, 2, 3, 4, 5, 6]
    for group in (1, 2, 3, 4, 5, 6):
        assert f"gm_g{group}_" in rendered["calls"][group]
        assert f"static void gm_g{group}_" in rendered["definitions"]


def test_the_two_readouts_that_used_to_fall_back_are_expressed(rendered):
    """A pooled convolution readout and an accumulator-resident matmul readout: the two shapes that
    kept ResNet-50's stem and its classifier on the vendor call while every other group ran ours."""
    by_group = {c["group"]: c for c in rendered["census"]}
    assert by_group[2]["recipe"] == "conv_loop_conv_ws_split_reduction_v3"
    assert by_group[4]["recipe"] == "matmul_ws_full_acc_readout_v1"
    assert "(acc_t *)B_4" in rendered["calls"][4], "the classifier's buffer is the accumulator's width"


def test_a_group_we_cannot_express_keeps_its_vendor_call_and_says_why(rendered):
    """The alternative is a run that reports our cycles for the library's work."""
    refused = {c["group"]: c for c in rendered["census"] if c["on"] != "sched"}
    assert sorted(refused) == [7]
    assert "bias" in refused[7]["why"], refused[7]
    assert 7 not in rendered["calls"]


def test_no_emitted_parameter_is_named_after_a_buffer(rendered):
    """`IMAGE` is a MACRO in the generated program (`((const elem_t *)IMAGE_DATA)`). A parameter named
    after it is macro-expanded inside the parameter list and the file does not compile -- so parameters
    are named by role and the buffer is bound at the call site."""
    for line in rendered["definitions"].splitlines():
        if line.startswith("static void gm_"):
            params = line.split("(", 1)[1].rsplit(")", 1)[0]
            for param in params.split(","):
                name = param.strip().split("*")[-1].strip()
                assert name.startswith("p_"), line
                assert name not in {"IMAGE", "ONES", "B_1", "B_3", "B_5", "W_1", "BIAS_1"}


def test_the_call_site_passes_the_real_buffers(rendered):
    assert "IMAGE" in rendered["calls"][1] and "W_1" in rendered["calls"][1]
    assert "BIAS_1" in rendered["calls"][1] and "B_1" in rendered["calls"][1]
    # the mean reads the ones vector the program already materialises
    assert "ONES" in rendered["calls"][6]


def test_the_mean_reads_the_plane_untransposed(gmsk, rendered):
    """The capture lays the buffer out as [window positions] by [channels], so putting the ones vector
    on the LEFT reads it directly -- the vendor call spells the same reduction with a transposed A."""
    row = next(c for c in rendered["census"] if c["group"] == 6)
    assert row["recipe"] == "window_mean_ones_contraction_v1"


def test_the_summary_reports_the_mix_not_a_headline(gmsk, rendered):
    text = gmsk.summarize(rendered["census"])
    assert "6 of 7 groups on our schedules" in text
    assert "bias" in text


def test_render_substitutes_ours_and_keeps_the_vendor_call_for_the_rest(rendered):
    """The whole point of threading a census through: the C program is a MIXTURE, declared as one."""
    path = merlin_dir() / "experiments/gemmini_perf_bench/scripts/group_model_program.py"
    spec = importlib.util.spec_from_file_location("group_model_program", path)
    gmp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gmp
    spec.loader.exec_module(gmp)

    model = _model()
    model.update(
        {
            "arrays": {"IMAGE_DATA": b"", "GOLDEN": b""},
            "buffers": [{"name": f"B_{i}", "ctype": "elem_t", "elements": 4096} for i in range(1, 8)],
            "classes": 1000,
        }
    )
    model["steps"][-1]["dequantize"] = 0.5
    c = gmp.render(model, sched_kernels=rendered)
    assert "gm_g1_conv2d(" in c and "gm_g2_conv2d(" in c, "our schedule is called, pooled readout and all"
    assert "gm_g4_matmul(" in c, "and the accumulator-resident readout"
    assert "tiled_conv_auto(" in c, "the group with no bias keeps its library call"
    assert "gm_g5_sum(" in c and "tiled_resadd_auto(" not in c, "every sum was expressible"
