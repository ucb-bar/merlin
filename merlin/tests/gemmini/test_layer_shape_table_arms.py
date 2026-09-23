"""The shape-table's three arms share one spec, and the package arm refuses an incomparable one.

The point of the script is that a schedule, a vendor library and an out-of-tree compiler PACKAGE are
measured on one program, one operand blob and one oracle. The two things that can quietly break that
are the ones gated here: the arm roster the CLI accepts, and the bias operand -- the interface GEMM a
package is handed declares none, so a non-zero bias span would have the package computing a different
function from the other two arms while their digests were still being compared.
"""

import sys

import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import layer_shape_table as lst  # noqa: E402
from layer_package_table import READOUTS, iface_module  # noqa: E402

SPEC = {"op": "matmul", "m": 32, "n": 32, "k": 32, "relu": False, "scale": 0.25}


def test_the_three_arms_are_the_declared_roster():
    assert lst.ARMS == ("schedule", "library", "package")
    assert set(lst._ARM_TAG) == set(lst.ARMS)
    assert len(set(lst._ARM_TAG.values())) == len(lst.ARMS)  # a record line names exactly one arm


def test_package_arm_refuses_a_bias_the_interface_cannot_carry():
    with pytest.raises(SystemExit) as raised:
        lst.main(
            [
                "--shape",
                "32x32x32",
                "--arm",
                "package",
                "--package",
                ".",
                "--target",
                "gemmini",
                "--design-pin",
                "x",
            ]
        )
    assert "bias" in str(raised.value)


def test_package_arm_needs_a_package():
    with pytest.raises(SystemExit) as raised:
        lst.main(["--shape", "32x32x32", "--arm", "package", "--target", "gemmini", "--design-pin", "x"])
    assert "--package" in str(raised.value)


def test_spec_carries_the_bias_span_every_arm_shares():
    assert lst.build_spec((4, 8, 16), scale=0.5, relu=False, bias_span=0)["bias_span"] == 0


@pytest.mark.parametrize("readout", READOUTS)
def test_each_readout_declares_its_own_output_width(readout):
    module, binding = iface_module(SPEC, 0.25, "gemmini", readout=readout)
    assert binding == {"A0": "a", "W": "b"}
    if readout == "full_i32":
        # No epilogue at all: the accumulator's own width, which is NOT the function the other arms
        # compute, and the table says so rather than printing the number in a comparable column.
        assert "epilogue = []" in module and 'output_dtype = "i32"' in module
        assert "tensor<32x32xi32>" in module
    else:
        assert 'epilogue = ["acc_scale"]' in module and 'output_dtype = "i8"' in module


def test_full_width_readout_refuses_an_activation_it_cannot_apply():
    with pytest.raises(ValueError):
        iface_module({**SPEC, "relu": True}, 0.25, "gemmini", readout="full_i32")
