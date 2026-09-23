"""The readout a model's final contraction leaves as: the accumulator moved out at its OWN width.

ResNet-50's classifier does not requantize. Its group carries no readout multiplier at all, because the
logits are dequantized on the host, and the matmul recipe could only store a narrow saturating readout
-- so that one group kept its vendor call while the other seventy ran our schedules.

The device already expresses it: ``gemmini_loop_ws`` carries ``full_C``, and ``LoopMatmulStC`` turns it
into the mvout's ``read_full`` bit. What the bit changes is not the width alone. The accumulator's scale
unit produces two rows from every read -- ``data``, which has had the activation, the accumulator scale
and the clip to the operand type applied to it, and ``full_data``, which is the raw accumulator row --
and ``read_full`` picks the second. So a schedule that asks for a full-width store AND for an activation
or a scale is asking for arithmetic this readout does not do, and the checker refuses it by the property
it named rather than storing unscaled bytes under a scaled name.

The port itself is a BUILD fact, not an ISA one: the generator emits ``ACC_READ_FULL_WIDTH`` into the
parameters header exactly when it built the full-width read port, so that is where the capability is
read from and a header without it refuses.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.runtime.backends import base
from merlin.sched.check.static import check_kernel
from merlin.sched.ir import TensorArg, concretize, instances
from merlin.sched.isa import IsaError


@pytest.fixture(scope="module")
def iset():
    return base.get_backend("gemmini").sched_instruction_set()


@pytest.fixture(scope="module")
def instr(iset):
    return iset.instr("loop_ws")


@pytest.fixture(scope="module")
def sched():
    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


def _legal() -> dict:
    """One 16x32 output tile stored at the accumulator's own width, with a repeated bias."""
    return dict(
        I=1,
        J=2,
        K=4,
        pad_I=0,
        pad_J=0,
        pad_K=0,
        A=1,
        B=2,
        D=3,
        C=4,
        A_stride=64,
        B_stride=32,
        D_stride=0,
        C_stride=32,
        A_transpose=0,
        B_transpose=0,
        full_C=1,
        low_D=0,
        ex_accumulate=1,
        act=0,
        a_spad_id=0,
        b_spad_id=0,
        is_resadd=0,
    )


def _state(acc_bytes: int, *, acc_scale: float = 1.0, acc_act: int = 0) -> dict:
    return {
        "config_ld": {0: 64, 1: 32, 2: 0},
        "config_st": {"stride": 32 * acc_bytes, "acc_act": acc_act, "acc_scale": acc_scale},
        "config_ex": {"A_transpose": 0, "B_transpose": 0},
    }


def test_the_target_states_it_built_the_full_width_read_port(iset):
    assert iset.facts["acc_read_full_width"] is True, "read from the parameters header, not assumed"


def test_a_full_width_readout_is_legal(instr, iset):
    assert instr.check(_legal(), _state(iset.facts["acc_bytes"])) == []


def test_the_store_stride_is_the_accumulators_own_width(instr, iset):
    """The mvout's DRAM offset scales the C stride by ``acc_w/8``, so a narrow ``config_st`` stride
    beside a full-width loop walks the output rows at a quarter of their pitch."""
    narrow = _state(iset.facts["elem_bytes"])
    assert any("config_st stride" in e for e in instr.check(_legal(), narrow))


@pytest.mark.parametrize(
    "state_kwargs,wanted",
    [
        ({"acc_act": 1}, "activation"),
        ({"acc_scale": 0.03125}, "acc_scale"),
    ],
)
def test_the_readout_the_full_width_port_does_not_do_is_refused(instr, iset, state_kwargs, wanted):
    """Every mode this readout does not model, refused by name rather than stored under a wrong one."""
    v = _legal() | ({"act": 1} if "acc_act" in state_kwargs else {})
    errs = instr.check(v, _state(iset.facts["acc_bytes"], **state_kwargs))
    assert any("raw accumulator" in e and wanted in e for e in errs), errs


def test_the_activation_is_refused_even_when_the_store_configuration_agrees(instr, iset):
    """Both halves state the ReLU and they agree with each other -- and the readout still does not
    apply it, so agreement is not the property being checked."""
    errs = instr.check(_legal() | {"act": 1}, _state(iset.facts["acc_bytes"], acc_act=1))
    assert any("raw accumulator" in e and "activation" in e for e in errs), errs
    assert instr.check(_legal() | {"act": 1}, _state(iset.facts["elem_bytes"], acc_act=1)) != [], (
        "the same loop without full_C is a different question"
    )


def test_a_narrow_readout_still_takes_its_activation_and_scale(instr, iset):
    """The mutation that must NOT fire: this modelling is about full_C and nothing else."""
    narrow = _legal() | {"full_C": 0, "act": 1}
    assert instr.check(narrow, _state(iset.facts["elem_bytes"], acc_act=1, acc_scale=0.03125)) == []


def test_no_operand_of_a_full_width_store_reports_touching_no_bytes(instr, iset):
    """The footprint's shape of failure: an extent read through an axis that is absent in this mode
    comes back zero, and a footprint that understates a read admits a tile whose operands do not fit."""
    eb, ab = iset.facts["elem_bytes"], iset.facts["acc_bytes"]
    fp = dict((n, b) for n, b, _ in instr.footprint(_legal()))
    assert min(fp.values()) > 0, fp
    narrow = dict((n, b) for n, b, _ in instr.footprint(_legal() | {"full_C": 0}))
    assert fp["C"] == narrow["C"] * (ab // eb), "the store is the accumulator's own width, not the operand's"
    assert fp["A"] == narrow["A"] and fp["B"] == narrow["B"] and fp["D"] == narrow["D"]


def test_a_target_whose_header_states_no_full_width_port_refuses(tmp_path, sched, iset):
    """The capability is read from the parameters header the generator wrote, not assumed.

    ``acc_read_full_width`` is a build parameter: a unit configured without it still DECODES the bit and
    answers the mvout out of the narrow port, so nothing downstream would report the width it got.
    """
    target = importlib.import_module("merlin._oot_backends.gemmini.gemmini")
    header = str(target.rocc_tests_dir() / "include" / "gemmini.h")
    original = target._params_header()
    stripped = tmp_path / "gemmini_params.h"
    stripped.write_text(
        "\n".join(
            line for line in original.read_text(encoding="utf-8").splitlines() if "ACC_READ_FULL_WIDTH" not in line
        ),
        encoding="utf-8",
    )
    lean = sched.instruction_set(header, str(stripped), target="gemmini")
    assert lean.facts["acc_read_full_width"] is False
    errs = lean.instr("loop_ws").check(_legal(), _state(lean.facts["acc_bytes"]))
    assert any("ACC_READ_FULL_WIDTH" in e for e in errs), errs
    with pytest.raises(IsaError, match="ACC_READ_FULL_WIDTH"):
        _matmul(sched, lean.facts, m=1, n=1000, k=2048, scale=None)


# --- the recipe ----------------------------------------------------------------------------------


def _matmul(sched, facts, *, m, n, k, scale, relu=False):
    dtype = "i8" if scale is not None else "i32"
    ops = {
        "a": TensorArg("a", (m, k), "i8", "read"),
        "b": TensorArg("b", (k, n), "i8", "read"),
        "d": TensorArg("d", (n,), "i32", "read"),
        "c": TensorArg("c", (m, n), dtype, "write"),
    }
    return sched.matmul_reference(
        name="cls", m=m, n=n, k=k, operands=ops, relu=relu, scale=scale, facts=facts, tiles=None
    )


def _loops(kernel):
    return [concretize(call, env) for call, env in instances(kernel) if call.instr == "loop_ws"]


def test_the_classifier_shape_passes_the_static_check(sched, iset):
    """ResNet-50's own: 1x1000 out of 2048, bias folded, no readout multiplier."""
    k = _matmul(sched, iset.facts, m=1, n=1000, k=2048, scale=None)
    assert check_kernel(k, iset) == []
    assert dict(k.attrs)["recipe"] == "matmul_ws_full_acc_readout_v1"
    assert dict(k.attrs)["readout"] == "the accumulator at its own width"


def test_the_recipe_configures_the_store_at_the_accumulators_width_and_identity_scale(sched, iset):
    ab = iset.facts["acc_bytes"]
    k = _matmul(sched, iset.facts, m=1, n=1000, k=2048, scale=None)
    config_st = next(s for s in k.body if getattr(s, "instr", None) == "config_st")
    args = dict(config_st.args)
    assert args["stride"].value == 1000 * ab
    assert args["acc_scale"] == 1.0, "the library's ACC_SCALE_IDENTITY: nothing is applied"
    assert args["acc_act"].value == iset.facts["constants"]["NO_ACTIVATION"]


def test_every_stored_tile_carries_the_bit_and_steps_in_accumulator_elements(sched, iset):
    eb, ab = iset.facts["elem_bytes"], iset.facts["acc_bytes"]
    n = 1000
    full = _loops(_matmul(sched, iset.facts, m=1, n=n, k=2048, scale=None))
    narrow = _loops(_matmul(sched, iset.facts, m=1, n=n, k=2048, scale=0.03125))
    assert full and len(full) == len(narrow)
    for mine, theirs in zip(full, narrow):
        assert mine["full_C"] == 1 and theirs["full_C"] == 0
        assert mine["C_stride"] == theirs["C_stride"] == n, "the stride operand is in ELEMENTS either way"
        if mine["C"] is None:
            assert theirs["C"] is None
            continue
        assert mine["C"].offset == theirs["C"].offset * (ab // eb), "the pointer steps in acc_t"


def test_an_activation_asked_for_beside_a_full_width_readout_is_refused_by_the_recipe(sched, iset):
    with pytest.raises(IsaError, match="RAW accumulator"):
        _matmul(sched, iset.facts, m=1, n=1000, k=2048, scale=None, relu=True)
    assert _matmul(sched, iset.facts, m=1, n=1000, k=2048, scale=0.03125, relu=True), "the narrow path keeps its ReLU"


def test_the_narrow_recipe_is_unchanged(sched, iset):
    """The readout this work added a second of; the first must still be exactly what it was."""
    eb = iset.facts["elem_bytes"]
    k = _matmul(sched, iset.facts, m=64, n=64, k=64, scale=0.03125, relu=True)
    config_st = next(s for s in k.body if getattr(s, "instr", None) == "config_st")
    assert dict(config_st.args)["stride"].value == 64 * eb
    assert dict(k.attrs)["recipe"] == "matmul_ws_reference_v1"
    assert check_kernel(k, iset) == []
