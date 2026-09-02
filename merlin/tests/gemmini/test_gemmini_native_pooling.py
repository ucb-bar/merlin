"""Native Gemmini max-pool lowering and its RTL-derived CONFIG_ST layout."""
from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.runtime import reference_outputs, simulate
from merlin.runtime.backends import base as _bk
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.rocc.decode import decode_text
from merlin.targetgen.rtl.circt_introspect import extract_register_bundle_layouts
from merlin.targetgen.trace_check import check


gm = _bk.get_backend("gemmini").gemmini_codegen_mlir


def _capsule(name: str) -> dict:
    path = merlin_dir() / "contract" / "capsules" / "layers" / name / "capsule.interface.mlir"
    return parse_interface_mlir(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [
    "GP0_matmul_maxpool_i8",
    "GP1_matmul_maxpool_tail_i8",
    "GP2_conv2d_maxpool_i8",
])
def test_pool_capsules_request_the_native_i8_store_datapath(name):
    cb = _capsule(name)
    cmd = next(c for c in cb["commands"] if c["opcode"] in ("COMMIT", "CONV2D"))
    assert cmd["attributes"]["output_dtype"] == "i8"


def test_scala_bundle_extraction_recovers_config_st_pool_fields_without_magic_offsets():
    source = """
      val A_WIDTH = 2
      val GAP_WIDTH = (8 - 2)
      val B_WIDTH = 8
      class ConfigRs1 extends Bundle {
        val b = UInt(B_WIDTH.W)
        val _gap = UInt(GAP_WIDTH.W)
        val a = UInt(A_WIDTH.W)
      }
    """
    layouts = extract_register_bundle_layouts(source)
    assert layouts["ConfigRs1"] == {
        "width": 16,
        "fields": {
            "a": {"offset": 0, "width": 2},
            "b": {"offset": 8, "width": 8},
        },
    }


def test_pool_config_packs_and_round_trips_through_the_derived_rtl_field_map():
    cb = _capsule("GP0_matmul_maxpool_i8")
    normalized = gm._normalize_command_buffer(cb)
    pool = gm._parse_groups(normalized)[0][3][0].pool
    packed = gm._pool_config_rs1(pool, acc_act=0)
    layout = gm._isa().CONFIG_ST_LAYOUT
    fields = layout["fields"]
    expected = {
        "cmd_type": gm._isa().CONFIG_ST_TYPE,
        "activation": 0,
        "pool_stride": 2,
        "pool_size": 2,
        "pool_out_dim": 2,
        "porows": 2,
        "pocols": 2,
        "orows": 4,
        "ocols": 4,
        "upad": 0,
        "lpad": 0,
    }
    for field, value in expected.items():
        spec = fields[field]
        assert (packed >> spec["offset"]) & ((1 << spec["width"]) - 1) == value


def test_ragged_plane_is_retained_then_stored_once_per_channel_tile():
    cb = _capsule("GP1_matmul_maxpool_tail_i8")
    text, _ = gm.emit_kernel_mlir(cb)
    mvout = f"{gm.K_MVOUT}, x0"
    # N=17 is two channel tiles. Pooling must retain both 5x5 accumulator planes and issue one
    # pool-enabled store per channel tile, not store each of the four matrix tiles independently.
    assert sum(mvout in line for line in text.splitlines()) == 2
    harness = gm._harness_c(cb)
    assert "static elem_t T_Y0" in harness
    assert 'printf("OUT Y0 4 17")' in harness


def test_conv_pool_reuses_the_im2col_matmul_backend_and_all_l0_engines_agree():
    cb = _capsule("GP2_conv2d_maxpool_i8")
    normalized = gm._normalize_command_buffer(cb)
    assert [c["opcode"] for c in normalized["commands"]] == [
        "RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"]
    assert normalized["params"]["im2col_recipes"][0]["source"] == "IFM"
    assert reference_outputs(normalized) == simulate(normalized)["outputs"]
    text, args = gm.emit_kernel_mlir(cb)
    assert "llvm.func @gemmini_kernel" in text
    assert any(name.startswith("IFM__im2col") for name in args)


@pytest.mark.parametrize("name", [
    "GP0_matmul_maxpool_i8",
    "GP1_matmul_maxpool_tail_i8",
    "GP2_conv2d_maxpool_i8",
])
def test_native_pooling_trace_satisfies_each_generated_capsule_contract(name):
    directory = merlin_dir() / "contract" / "capsules" / "layers" / name
    cb = parse_interface_mlir((directory / "capsule.interface.mlir").read_text(encoding="utf-8"))
    expected = yaml.safe_load((directory / "capsule.yaml").read_text(encoding="utf-8"))["expected"]
    trace = decode_text(gm.emit_kernel_mlir(cb)[0], target="gemmini")
    assert check(trace, expected, cb, address_model="pointer_args") == {
        "status": "pass", "violations": []}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.update(output_dtype="i32"), "i8 output"),
        (lambda a: a.update(pool_size=[2, 3]), "square pool_size"),
        (lambda a: a.update(pool_stride=[2, 1]), "square pool_stride"),
        (lambda a: a.update(pool_padding=[1, 0, 0, 0]), "zero pool_padding"),
        (lambda a: a.update(pool_in_dims=[2, 4]), "batch 1"),
    ],
)
def test_native_pooling_refuses_unrepresentable_contracts(mutate, message):
    cb = deepcopy(_capsule("GP0_matmul_maxpool_i8"))
    attrs = next(c for c in cb["commands"] if c["opcode"] == "COMMIT")["attributes"]
    mutate(attrs)
    with pytest.raises(gm.CodegenError, match=message):
        gm.emit_kernel_mlir(cb)
