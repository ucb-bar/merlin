"""``conv2d`` end to end: declared by the manifest, evidenced by the RTL, and now expressible.

MEASURED, and the reason this file exists: the target manifest declares ``{family: contraction,
dtypes: [int8], ranks: [2, 4]}`` and justifies rank 4 from the RTL's own funct table (a dedicated
``LOOP_CONV_WS`` plus its ``LOOP_CONV_WS_CONFIG_*`` companions), three shipped capsules write
``merlin_iface.conv2d`` — and the frozen interface grammar defined no such op. Before the
fail-closed parse those three capsules parsed "successfully" into ``resident_pack`` + ``evict``
with the CONVOLUTION GONE; after it they raised. Either way nothing could execute them.

The contract this file pins is two-sided:

* the op must reach the command list and then produce the SHIPPED golden bit-for-bit (integer
  equality, never a tolerance — see ``command_buffer_abi.yaml``'s ``correctness_gate``); and
* every conv parameter outside the implemented subset must be REJECTED BY NAME. A conv that
  silently ignores its stride returns a full, plausible-looking tensor of wrong numbers, which is
  precisely the failure the fail-closed grammar was introduced to stop — reintroducing it one
  attribute at a time would be worse than leaving conv2d undefined.

Supported subset: NHWC layout, ``kernel = [kh, kw, ci, co]``, ``stride``, 4-edge ``padding``,
``dilation``, the readout a commit applies (``bias_add``/``bias`` with a NAMED bias tensor,
``acc_scale`` with a DECLARED multiplier, ``requant``, ``relu``, ``maxpool``), and an integer
``output_dtype``. Deliberately NOT supported (and asserted to raise): grouped/depthwise convolution,
a non-nhwc layout, a bias stage that names no tensor, an ``acc_scale`` stage that names no
multiplier, and a float output dtype.

The bias and the multiplier were the measured gap: every quantized CNN layer is a convolution with a
per-channel bias, a requantization and an activation, and the op could state none of it as one
command. Of the 24 distinct device programs a captured ResNet-50 forms, the generator refused 11
for exactly that.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.runtime.commandbuffer import conv_im2col
from merlin.runtime.reference import UnmodeledOp, reference_outputs
from merlin.runtime.simulator import SimulationError, simulate
from merlin.runtime.tensor import Tensor
from merlin.targetgen.contract import interface_emit as IE
from merlin.targetgen.contract.schemas import contract_dir

#: The three shipped capsules whose only compute op is the conv.
_CONV_CAPSULES = ("B3_conv2d_im2col_i8", "B4_conv2d_relu_i8", "GC0_conv2d_i8")

_CONV_IFACE = textwrap.dedent("""\
    module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
      %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x8x8x4xi8>
      %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<36x8xi8>
      %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<36x8xi8>) -> !merlin_iface.resident
      %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 4, 8], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"} : (tensor<1x8x8x4xi8>, !merlin_iface.resident) -> tensor<36x8xi32>
      merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
    }
    """)


def _capsule_dir(name: str):
    return repo_root() / "merlin" / "contract" / "capsules" / "layers" / name


def _conv_cb(
    *,
    H=8,
    W=8,
    ci=4,
    kh=3,
    kw=3,
    co=8,
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    dilation=(1, 1),
    epilogue=(),
    output_dtype="i32",
    extra=None,
):
    """A minimal single-conv command buffer (no residency pack — the weight is read directly)."""
    attrs = {
        "kernel": [kh, kw, ci, co],
        "stride": list(stride),
        "padding": list(padding),
        "dilation": list(dilation),
        "layout": "nhwc",
        "epilogue": list(epilogue),
        "output_dtype": output_dtype,
    }
    attrs.update(extra or {})
    return {
        "abi_version": "0.1",
        "target": "t",
        "tensors": {
            "IFM": {"shape": [1, H, W, ci], "dtype": "i8", "role": "input"},
            "W": {"shape": [kh * kw * ci, co], "dtype": "i8", "role": "weight"},
        },
        "commands": [{"opcode": "CONV2D", "operands": {"ifm": "IFM", "weight": "W", "dst": "Y0"}, "attributes": attrs}],
        "outputs": ["Y0"],
    }


def _im2col_reference(*, H, W, ci, kh, kw, co, stride, padding, dilation):
    """The definition, recomputed independently of the simulator: gather then contract."""
    ifm = Tensor.deterministic("IFM", (1, H, W, ci), "i8")
    w = Tensor.deterministic("W", (kh * kw * ci, co), "i8")
    cols = conv_im2col(ifm, kh=kh, kw=kw, ci=ci, stride=stride, padding=padding, dilation=dilation, layout="nhwc")
    return cols.matmul(w)


class TestTheGrammarDefinesConv2d:
    def test_conv2d_is_a_defined_mnemonic(self):
        assert "conv2d" in IE.defined_mnemonics()
        assert IE.undefined_op_mnemonics(_CONV_IFACE) == []

    def test_the_op_reaches_the_command_list_with_its_operands(self):
        # The measured defect in one assertion: this module used to yield RES_PACK + EVICT with the
        # convolution itself missing, and the parser reported success.
        cb = IE.parse_interface_mlir(_CONV_IFACE)
        assert [c["opcode"] for c in cb["commands"]] == ["RES_PACK", "CONV2D", "EVICT"]
        conv = cb["commands"][1]
        assert conv["operands"] == {"ifm": "IFM", "weight": "W_res", "dst": "Y0"}

    def test_the_geometry_survives_the_parse(self):
        # Attributes are the only place conv geometry lives in this grammar, so an attribute lost here
        # is a parameter the simulator can never honour.
        attrs = IE.parse_interface_mlir(_CONV_IFACE)["commands"][1]["attributes"]
        assert attrs["kernel"] == [3, 3, 4, 8]
        assert attrs["stride"] == [1, 1]
        assert attrs["padding"] == [0, 0, 0, 0]
        assert attrs["dilation"] == [1, 1]
        assert attrs["layout"] == "nhwc"
        assert attrs["output_dtype"] == "i32"

    @pytest.mark.parametrize("name", _CONV_CAPSULES)
    def test_the_shipped_capsules_parse_whole(self, name):
        text = (_capsule_dir(name) / "capsule.interface.mlir").read_text(encoding="utf-8")
        cb = IE.parse_interface_mlir(text)
        ops = [m for m in IE.op_mnemonics(text) if m != "tensor"]
        assert len(cb["commands"]) == len(ops)
        assert any(c["opcode"] == "CONV2D" for c in cb["commands"])


class TestSimulatorSemantics:
    def test_conv_is_im2col_then_contract(self):
        geom = dict(H=8, W=8, ci=4, kh=3, kw=3, co=8, stride=(1, 1), padding=(0, 0, 0, 0), dilation=(1, 1))
        want = _im2col_reference(**geom)
        got = simulate(_conv_cb(**geom))["outputs"]["Y0"]
        assert got == want.to_list()

    @pytest.mark.parametrize(
        "stride,padding,dilation",
        [
            ((2, 2), (0, 0, 0, 0), (1, 1)),
            ((1, 1), (1, 1, 1, 1), (1, 1)),
            ((2, 1), (1, 0, 1, 0), (1, 1)),
            ((1, 1), (0, 0, 0, 0), (2, 2)),
        ],
    )
    def test_stride_padding_and_dilation_are_honoured(self, stride, padding, dilation):
        # These are the parameters the target's own conv loop takes. A simulator that accepted them
        # and ignored them would still return a well-shaped integer tensor, so the check is against
        # the independently recomputed definition, not against "it did not crash".
        geom = dict(H=8, W=8, ci=4, kh=3, kw=3, co=8, stride=stride, padding=padding, dilation=dilation)
        want = _im2col_reference(**geom)
        got = simulate(_conv_cb(**geom))["outputs"]["Y0"]
        assert got == want.to_list()

    def test_a_non_unit_stride_actually_changes_the_answer(self):
        # The falsifier for the test above: if stride were ignored, these two would be equal and every
        # stride assertion in this file would be passing on an implementation that reads none of them.
        s1 = simulate(_conv_cb(stride=(1, 1)))["outputs"]["Y0"]
        s2 = simulate(_conv_cb(stride=(2, 2)))["outputs"]["Y0"]
        assert len(s1) == 36 and len(s2) == 9
        assert s1 != s2

    def test_padding_changes_the_answer_too(self):
        p0 = simulate(_conv_cb(padding=(0, 0, 0, 0)))["outputs"]["Y0"]
        p1 = simulate(_conv_cb(padding=(1, 1, 1, 1)))["outputs"]["Y0"]
        assert len(p0) == 36 and len(p1) == 64
        assert p0 != p1

    def test_relu_epilogue_matches_the_golden_engines_definition(self):
        # Explicit SIGNED stimulus, not the deterministic fill: that fill is 0..3, and with a
        # non-negative activation and a non-negative weight the conv accumulator is never negative, so
        # relu would be indistinguishable from no epilogue at all and this test would prove nothing.
        # Flat row-major (the command buffer's ``inputs`` override flattens rank 2, not rank 4).
        ifm = [-3, 2, 1, -4, 5, -6, 0, 2, -1, 3, -2, 4, 2, -5, 1, -1]  # 1x4x4x1, NHWC
        w = [[1, -2], [-3, 4], [2, 1], [-1, -1]]  # [kh*kw*ci, co] = [4, 2]
        cb = _conv_cb(H=4, W=4, ci=1, kh=2, kw=2, co=2)
        bare = simulate(cb, {"IFM": ifm, "W": w})["outputs"]["Y0"]
        assert any(v < 0 for row in bare for v in row), "stimulus must reach the relu clamp"
        cb_relu = _conv_cb(H=4, W=4, ci=1, kh=2, kw=2, co=2, epilogue=["relu"])
        got = simulate(cb_relu, {"IFM": ifm, "W": w})["outputs"]["Y0"]
        assert got == [[max(v, 0) for v in row] for row in bare]

    def test_a_narrow_output_dtype_saturates_rather_than_wrapping(self):
        want = _im2col_reference(
            H=8, W=8, ci=4, kh=3, kw=3, co=8, stride=(1, 1), padding=(0, 0, 0, 0), dilation=(1, 1)
        ).to_i8()
        got = simulate(_conv_cb(output_dtype="i8"))["outputs"]["Y0"]
        assert got == want.to_list()
        assert max(v for row in got for v in row) <= 127


class TestUnsupportedParametersFailClosed:
    """Anything outside the implemented subset must raise, NAMING what was unsupported."""

    def test_an_unknown_attribute_is_named_not_ignored(self):
        # `groups` is the concrete case: grouped/depthwise convolution is a different contraction, and
        # a conv that read `groups = 2` and computed a dense conv would be silently wrong.
        with pytest.raises(SimulationError, match="groups"):
            simulate(_conv_cb(extra={"groups": 2}))

    def test_a_non_nhwc_layout_is_rejected(self):
        with pytest.raises(SimulationError, match="nchw"):
            simulate(_conv_cb(extra={"layout": "nchw"}))

    def test_a_missing_kernel_attribute_is_rejected(self):
        cb = _conv_cb()
        del cb["commands"][0]["attributes"]["kernel"]
        with pytest.raises(SimulationError, match="kernel"):
            simulate(cb)

    def test_a_two_element_padding_is_rejected_rather_than_reinterpreted(self):
        # padding is [top, left, bottom, right]. Reading a 2-element [h, w] as [top, left] with zero
        # bottom/right silently changes the output geometry.
        cb = _conv_cb()
        cb["commands"][0]["attributes"]["padding"] = [1, 1]
        with pytest.raises(SimulationError, match="padding"):
            simulate(cb)

    def test_a_bias_stage_that_names_no_tensor_is_refused_not_skipped(self):
        # A dropped bias leaves every element off by its channel's bias, which reads as a
        # plausible answer. Both engines refuse the command instead.
        with pytest.raises(SimulationError, match="names no bias tensor"):
            simulate(_conv_cb(epilogue=["bias_add"]))
        with pytest.raises(ValueError, match="names no bias tensor"):
            reference_outputs(_conv_cb(epilogue=["bias_add"]))

    def test_an_acc_scale_stage_that_names_no_multiplier_is_refused_not_applied_at_one(self):
        with pytest.raises(SimulationError, match="no `acc_scale` multiplier"):
            simulate(_conv_cb(epilogue=["acc_scale"], output_dtype="i8"))
        with pytest.raises(ValueError, match="no `acc_scale` multiplier"):
            reference_outputs(_conv_cb(epilogue=["acc_scale"], output_dtype="i8"))

    def test_a_float_output_dtype_is_rejected_by_the_integer_engine(self):
        with pytest.raises(SimulationError, match="bf16"):
            simulate(_conv_cb(output_dtype="bf16"))

    def test_a_channel_count_that_contradicts_the_activation_is_rejected(self):
        cb = _conv_cb(ci=4)
        cb["commands"][0]["attributes"]["kernel"] = [3, 3, 2, 8]
        with pytest.raises(SimulationError, match="channel mismatch"):
            simulate(cb)

    def test_a_weight_that_is_not_im2col_packed_is_rejected(self):
        cb = _conv_cb()
        cb["tensors"]["W"]["shape"] = [8, 36]  # transposed: same element count, wrong packing
        with pytest.raises(SimulationError, match="im2col-packed"):
            simulate(cb)


class TestTheShippedCapsulesRunAndMatchTheirGolden:
    @pytest.mark.parametrize("name", _CONV_CAPSULES)
    def test_parse_then_simulate_reproduces_the_shipped_golden_exactly(self, name):
        # Integer workload => exact equality, never a tolerance (command_buffer_abi correctness_gate).
        d = _capsule_dir(name)
        cb = IE.parse_interface_mlir((d / "capsule.interface.mlir").read_text(encoding="utf-8"))
        got = simulate(cb)["outputs"]
        want = yaml.safe_load((d / "golden.yaml").read_text(encoding="utf-8"))["outputs"]
        assert got == want

    def test_the_relu_capsule_cannot_currently_witness_its_own_relu(self):
        """MEASURED weakness of the shipped corpus, recorded rather than papered over.

        ``B4_conv2d_relu_i8`` is ``B3`` plus a relu, so it looks like the falsifier that would catch a
        dropped epilogue. It is not: both operands come from the deterministic 0..3 fill, so the conv
        accumulator is never negative and the relu is the identity — the two capsules' shipped goldens
        are byte-identical. A backend that ignored ``relu`` entirely would pass both. Relu is therefore
        exercised against signed stimulus in :class:`TestSimulatorSemantics` instead, and this test
        pins the fact so the corpus gap is visible rather than mistaken for coverage.
        """

        def out(name):
            d = _capsule_dir(name)
            return simulate(IE.parse_interface_mlir((d / "capsule.interface.mlir").read_text(encoding="utf-8")))[
                "outputs"
            ]["Y0"]

        assert out("B4_conv2d_relu_i8") == out("B3_conv2d_im2col_i8")
        assert all(v >= 0 for row in out("B3_conv2d_im2col_i8") for v in row)

    def test_the_reference_engine_says_it_cannot_cross_check_a_conv(self):
        # The residency-bypass cross-check does not extend to CONV2D (the reference engine models the
        # matmul/commit path only). That limit is stated, not hidden: an engine that returned an empty
        # output map here would read downstream as "the kernel never wrote its output".
        with pytest.raises(UnmodeledOp, match="CONV2D"):
            reference_outputs(_conv_cb())


class TestTheContractDeclaresConv2d:
    def test_the_command_buffer_abi_declares_the_opcode(self):
        abi = yaml.safe_load((contract_dir() / "command_buffer_abi.yaml").read_text(encoding="utf-8"))
        assert "CONV2D" in abi["opcodes"]
        assert "unsupported" in abi["opcodes"]["CONV2D"], "the refused subset must be written down"

    def test_the_interface_contract_maps_the_mnemonic(self):
        spec = yaml.safe_load((contract_dir() / "interface_dialect_contract.yaml").read_text(encoding="utf-8"))
        mapped = {op["name"]: op.get("maps_to") for op in spec["dialect"]["required_ops"]}
        assert mapped.get("merlin_iface.conv2d") == "CONV2D"


def _binding():
    from merlin.targetgen import corpus_spec as CS

    return CS.CorpusBinding(
        target="t",
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="int32",
        integer=True,
        tiers=["L2", "L3"],
        compare="exact",
        requant_output_dtype="i8",
    )


def _readout_cb(*, signed: bool = True):
    from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE, STIMULUS_RANGE_KEY

    cb = _conv_cb(
        epilogue=["bias_add", "acc_scale", "relu"],
        output_dtype="i8",
        padding=(1, 1, 1, 1),
        extra={"bias": "B", "acc_scale": 0.0123},
    )
    cb["tensors"]["B"] = {"shape": [8], "dtype": "i32", "role": "bias"}
    if signed:
        cb["params"] = {STIMULUS_RANGE_KEY: list(SIGNED_STIMULUS_RANGE)}
    return cb


class TestAConvolutionCarriesTheReadoutACommitDoes:
    def test_both_engines_agree_on_bias_then_scale_then_activation(self):
        cb = _readout_cb()
        simulated, referenced = simulate(cb)["outputs"]["Y0"], reference_outputs(cb)["Y0"]
        assert simulated == referenced
        values = [v for row in simulated for v in row]
        # The stimulus is signed, so the activation and the saturation both have something to do:
        # a result that is never clamped to zero could not tell a backend that skipped the relu.
        assert 0 in values and max(values) > 0 and min(values) == 0

    def test_the_bias_is_load_bearing(self):
        cb, moved = _readout_cb(), _readout_cb()
        moved["commands"][0]["attributes"]["epilogue"] = ["acc_scale", "relu"]
        assert simulate(cb)["outputs"]["Y0"] != simulate(moved)["outputs"]["Y0"]

    def test_the_generator_states_the_bias_and_the_multiplier_and_the_parser_carries_them(self):
        from merlin.targetgen import corpus_spec as CS

        entry = {
            "name": "T", "kind": "layer", "source_role": "derived_sweep", "source_reference": "test",
            "op": "conv2d", "ci": 4, "N": 8, "Himg": 8, "Wimg": 8, "kh": 3, "kw": 3,
            "epilogue": ["bias_add", "acc_scale", "relu"], "acc_scale": 0.0123,
        }  # fmt: skip
        capsule, text = CS.build_conv2d(entry, _binding())
        attrs = capsule["operation"]["attributes"]
        assert (attrs["bias"], attrs["acc_scale"]) == ("B", 0.0123)
        assert [i["role"] for i in capsule["inputs"]] == ["weight", "input", "bias"]
        (command,) = [c for c in IE.parse_interface_mlir(text)["commands"] if c["opcode"] == "CONV2D"]
        assert (command["attributes"]["bias"], command["attributes"]["acc_scale"]) == ("B", 0.0123)

    def test_the_generator_refuses_a_stage_without_its_parameter_and_a_parameter_without_its_stage(self):
        from merlin.targetgen import corpus_spec as CS

        base = {
            "name": "T", "kind": "layer", "source_role": "derived_sweep", "source_reference": "test",
            "op": "conv2d", "ci": 4, "N": 8, "Himg": 8, "Wimg": 8, "kh": 3, "kw": 3,
        }  # fmt: skip
        with pytest.raises(ValueError, match="needs its `acc_scale`"):
            CS.build_conv2d({**base, "epilogue": ["acc_scale"]}, _binding())
        with pytest.raises(ValueError, match="read by nothing"):
            CS.build_conv2d({**base, "epilogue": ["relu"], "acc_scale": 0.5}, _binding())

    def test_a_convolution_with_no_readout_is_built_as_before(self):
        from merlin.targetgen import corpus_spec as CS

        entry = {
            "name": "T", "kind": "layer", "source_role": "derived_sweep", "source_reference": "test",
            "op": "conv2d", "ci": 4, "N": 8, "Himg": 8, "Wimg": 8, "kh": 3, "kw": 3, "epilogue": ["relu"],
        }  # fmt: skip
        capsule, text = CS.build_conv2d(entry, _binding())
        assert "bias" not in capsule["operation"]["attributes"] and "acc_scale" not in text
        assert [i["role"] for i in capsule["inputs"]] == ["weight", "input"]
