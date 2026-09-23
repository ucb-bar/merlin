"""The systolic target's library-layer bench programs and the facts it declares for them."""

import numpy as np
import pytest

from merlin.perf.layer_bench import reference as ref
from merlin.runtime.backends import base


@pytest.fixture(scope="module")
def backend():
    return base.get_backend("gemmini")


CONV = {
    "op": "conv2d",
    "label": "c1",
    "batch": 1,
    "in_dim": 7,
    "in_channels": 16,
    "out_channels": 32,
    "kernel": 3,
    "stride": 1,
    "padding": 1,
    "scale": 0.1,
    "relu": True,
    "seed": 3,
}
MATMUL = {"op": "matmul", "label": "m1", "m": 5, "n": 32, "k": 48, "scale": 0.02, "relu": False}


def _render(backend, spec):
    _, off = ref.pack_operands(spec, accumulator_dtype="i32")
    return backend.render_library_layer(spec, offsets=off)


def test_program_embeds_the_blob_and_uses_protocol_constants(backend):
    src = _render(backend, CONV)
    assert f'.incbin \\"{backend.LIBRARY_LAYER_OPERAND_BLOB}\\"' in src
    for const in (ref.FNV_OFFSET, ref.FNV_PRIME, ref.DIGEST_MASK):
        assert str(const) in src
    assert "LB_RECORD c1 cycles=" in src and "tiled_conv_auto(" in src


def test_no_generator_and_no_large_static_arrays(backend):
    src = _render(backend, CONV)
    assert "lb_next" not in src and "static elem_t" not in src and "static acc_t" not in src


def test_scale_is_an_exact_float32_literal(backend):
    src = _render(backend, {**CONV, "scale": 0.1})
    assert float(np.float32(0.1)).hex() + "f" in src


def test_protocols_and_label_validation(backend):
    warm = _render(backend, CONV)
    cold = _render(backend, {**CONV, "protocol": "cold_single"})
    assert warm.count("tiled_conv_auto(") == 2 and cold.count("tiled_conv_auto(") == 1
    with pytest.raises(ValueError):
        _render(backend, {**CONV, "label": "two words"})
    with pytest.raises(ValueError):
        backend.render_library_layer({**CONV, "op": "softmax"}, offsets={})


CONV_1X1 = {**CONV, "label": "c2", "kernel": 1, "padding": 0}


def test_matmul_route_issues_a_1x1_conv_as_the_nhwc_matmul(backend):
    src = _render(backend, {**CONV_1X1, "route": "matmul"})
    assert "tiled_conv_auto(" not in src
    # I = B*H*W rows, J = CO, K = CI, strides K/J/J/J, bias row repeated: tiled_matmul_nn_auto's call
    assert "tiled_matmul_auto(49, 32, 16, input, weights, bias, output, 16, 32, 32, 32," in src
    assert "0, true," in src
    # the digest covers the same output bytes as the conv route, so one expected digest checks both
    conv_src = _render(backend, CONV_1X1)
    assert "lb_fnv1a_words((const uint8_t *)output, n_out * sizeof(elem_t))" in src
    assert "(size_t)49 * 32" in src and "(size_t)1 * 7 * 7 * 32" in conv_src


@pytest.mark.parametrize("bad", [{"kernel": 3, "padding": 1}, {"stride": 2}, {"padding": 1}])
def test_matmul_route_refuses_convs_it_does_not_compute(backend, bad):
    with pytest.raises(ValueError):
        _render(backend, {**CONV_1X1, **bad, "route": "matmul"})


def test_unknown_route_is_refused(backend):
    with pytest.raises(ValueError):
        _render(backend, {**CONV_1X1, "route": "im2col"})


@pytest.mark.parametrize("spec", [CONV, MATMUL, {**CONV_1X1, "route": "matmul"}])
def test_program_compiles_and_links_with_its_blob(backend, spec, tmp_path):
    """Compile for real through the target's own recipe (skips when the toolchain is absent)."""
    from merlin.perf.layer_bench import build_program, loaded_bytes

    recipe = base.harness_build_recipe("gemmini")
    if not recipe.compiler.is_file():
        pytest.skip("target toolchain not installed")
    blob, off = ref.pack_operands(spec, accumulator_dtype="i32")
    (tmp_path / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (tmp_path / "layer.c").write_text(backend.render_library_layer(spec, offsets=off))
    built = build_program([tmp_path / "layer.c"], tmp_path, target="gemmini", max_loaded_bytes=None)
    assert loaded_bytes(built.elf) >= len(blob)  # the operands really are in the image


PKG_CB = {
    "kernel_abi": {
        "args": [
            {"tensor": "IFM", "access": "read"},
            {"tensor": "W", "access": "read"},
            {"tensor": "Y0", "access": "write"},
            {"tensor": "__im2col_Y0", "access": "write"},
        ],
        "outputs": ["Y0"],
    },
    "tensors": {
        "IFM": {"shape": [1, 7, 7, 16], "dtype": "i8"},
        "W": {"shape": [144, 32], "dtype": "i8"},
        "Y0": {"shape": [49, 32], "dtype": "i8"},
        "__im2col_Y0": {"shape": [49, 144], "dtype": "i8"},
    },
}


def test_package_layer_calls_the_entry_in_abi_order(backend):
    src = backend.render_package_layer(PKG_CB, offsets={"IFM": 0, "W": 832}, label="p1", output="Y0")
    assert "extern void gemmini_kernel(void *, void *, void *, void *);" in src
    assert src.count("gemmini_kernel(lb_arg0, lb_arg1, lb_arg2, lb_arg3);") == 2  # warm + timed
    assert "(lb_operands + 832)" in src and f"lb_fnv1a_words(lb_out, {49 * 32})" in src
    with pytest.raises(ValueError):  # a read argument without an operand
        backend.render_package_layer(PKG_CB, offsets={"IFM": 0}, label="p1", output="Y0")
    with pytest.raises(ValueError):
        backend.render_package_layer(PKG_CB, offsets={"IFM": 0, "W": 832}, label="p1", output="nope")


def test_declared_facts_come_from_the_harness_header(backend):
    facts = backend.readout_facts()
    assert facts["schema"] == "scalar_narrow_readout_contract_v1"
    assert (facts["clamp_min"], facts["clamp_max"]) == (-128, 127)
    assert backend.mac_per_cycle_peak() == 16 * 16  # DIM 16 in the pinned params header
    assert len(backend.library_layer_emitter_digest()) == 64
    assert backend.gsim_backdoor_env()  # the harness's measured load backdoor


# --- a third-party kernel measured on the same operands -----------------------------------------
# A kernel published beside a paper is usually specialised to ONE shape: AutoComp's ResNet-50 bundle
# ships three, of which one is dead code never called, and each carries a fixed scratchpad base and a
# statically sized host padding buffer. Measuring such a kernel on other extents does not measure that
# kernel, so the shape it declares is checked rather than trusted.

_AUTOCOMP_K0 = {"kernel": 3, "in_channels": 64, "out_channels": 64, "stride": 1, "in_dim": 56}


def _external(backend, spec, **kw):
    _, off = ref.pack_operands(spec, accumulator_dtype="i32")
    return backend.render_external_layer(
        spec,
        offsets=off,
        symbol=kw.get("symbol", "autocomp_kernel0_conv_3x3_64to64_s1_56"),
        header=kw.get("header", "resnet50_autocomp_kernels.h"),
        declared_shape=kw.get("declared_shape", _AUTOCOMP_K0),
    )


def test_an_external_kernel_is_called_on_the_librarys_own_operands(backend):
    """Same blob, same output pointer, same digest -- only the call differs, which is the comparison."""
    spec = {**CONV, "label": "ac0", **_AUTOCOMP_K0, "padding": 1}
    src = _external(backend, spec)
    assert '#include "resnet50_autocomp_kernels.h"' in src
    assert "autocomp_kernel0_conv_3x3_64to64_s1_56(1, input, weights, bias, output," in src
    assert "LB_RECORD ac0 cycles=%llu digest=%llu" in src
    assert "lb_fnv1a_words" in src, "a faster-but-wrong kernel must not be able to win"


def test_a_specialised_kernel_on_another_shape_is_refused(backend):
    """The refusal that matters: silently reshaping the comparison would flatter the wrong party."""
    wrong = {**CONV, "label": "ac0", **_AUTOCOMP_K0, "padding": 1, "in_dim": 28}
    with pytest.raises(ValueError, match="not that kernel"):
        _external(backend, wrong)
    wrong_ch = {**CONV, "label": "ac0", **_AUTOCOMP_K0, "padding": 1, "out_channels": 128}
    with pytest.raises(ValueError, match="not that kernel"):
        _external(backend, wrong_ch)


def test_the_external_renderer_declines_ops_it_has_no_surface_for(backend):
    with pytest.raises(ValueError, match="no external kernel renderer"):
        _external(backend, {**MATMUL, "label": "x"}, declared_shape={})


def test_warm_then_measured_is_the_default_for_an_external_kernel_too(backend):
    """Their own log records the same kernel at 2.61M isolated and 5.56M embedded, the delta being a
    per-invocation memset. The warm pass is what keeps the two providers on equal terms."""
    spec = {**CONV, "label": "ac0", **_AUTOCOMP_K0, "padding": 1}
    src = _external(backend, spec)
    assert src.count("autocomp_kernel0_conv_3x3_64to64_s1_56(") == 2, "one warm call, one timed call"


#: A resident-matmul buffer that declares NO kernel_abi -- the shape a package may legitimately emit,
#: where the pointer order is a function of the commands alone.
IMPLIED_CB = {
    "abi_version": "0.1",
    "tensors": {
        "W": {"shape": [48, 32], "dtype": "i8", "role": "weight"},
        "A0": {"shape": [16, 48], "dtype": "i8", "role": "input"},
        "Y0": {"shape": [16, 32], "dtype": "i8", "role": "output"},
    },
    "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
        {
            "opcode": "COMMIT",
            "operands": {"src": "acc0", "dst": "Y0"},
            "attributes": {"epilogue": ["acc_scale"], "output_dtype": "i8", "acc_scale": 0.25},
        },
    ],
}


def test_implied_kernel_abi_follows_the_contract_order(backend):
    """weights ++ lhs ++ outputs, per ``kernel_abi.arg_order_by_command_shape``'s resident_matmul row."""
    abi = backend.kernel_abi_from_commands(IMPLIED_CB)
    assert [a["tensor"] for a in abi["args"]] == ["W", "A0", "Y0"]
    assert [a["access"] for a in abi["args"]] == ["read", "read", "write"]
    assert abi["outputs"] == ["Y0"]
    # A buffer that DOES declare one is returned unchanged: an explicit boundary beats inference.
    assert backend.kernel_abi_from_commands(PKG_CB) is PKG_CB["kernel_abi"]


def test_package_layer_runs_on_an_implied_abi(backend):
    cb = {**IMPLIED_CB, "kernel_abi": backend.kernel_abi_from_commands(IMPLIED_CB)}
    src = backend.render_package_layer(cb, offsets={"W": 0, "A0": 1536}, label="p2", output="Y0")
    assert src.count("gemmini_kernel(lb_arg0, lb_arg1, lb_arg2);") == 2  # warm + timed
    assert f"lb_fnv1a_words(lb_out, {16 * 32})" in src


def test_matmul_accumulator_is_the_exact_pre_readout_value():
    spec = {**MATMUL, "bias_span": 0, "seed": 4}
    ops = dict(ref.operand_arrays(spec))
    acc = ref.matmul_accumulator(ops["a"], ops["b"], ops["d"])
    assert acc.dtype == np.int64
    assert np.array_equal(acc, ops["a"].astype(np.int64) @ ops["b"].astype(np.int64) + ops["d"][None, :])
    assert np.array_equal(ops["d"], np.zeros_like(ops["d"]))  # a zero bias span really is all zeros
