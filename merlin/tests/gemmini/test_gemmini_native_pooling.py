"""Native Gemmini max-pool lowering and its RTL-derived CONFIG_ST layout."""
from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.runtime import reference_outputs, simulate
from merlin.runtime.backends import base as _bk
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.rocc.decode import decode_text
from merlin.targetgen import rtl_check_compiler, rtl_checks
from merlin.targetgen.rtl.circt_introspect import (extract_max_pool_from_firrtl,
                                                   extract_register_bundle_layouts)
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.address_space import derive_address_space
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


@pytest.mark.parametrize(("literal", "supported"), [("0h1", True), ("0h0", False)])
def test_max_pool_capability_is_read_from_elaborated_firrtl_gate(literal, supported):
    firrtl = f"""
      module StoreController :
        node _pooling_is_enabled_T = neq(pool_stride, UInt<1>(0h0))
        node pooling_is_enabled = and(UInt<1>({literal}), _pooling_is_enabled_T)
    """
    assert extract_max_pool_from_firrtl(firrtl) == {
        "value": supported, "line": 4,
        "expression": f"and(UInt<1>({literal}), _pooling_is_enabled_T)",
    }


def test_isa_pool_fields_without_an_elaborated_gate_do_not_claim_capability():
    assert extract_max_pool_from_firrtl(
        "node pool_stride = bits(config, 5, 4)\nnode pool_size = bits(config, 7, 6)\n"
    ) is None


def test_pool_gate_in_a_decoy_module_does_not_claim_store_controller_capability():
    firrtl = """
      module Decoy :
        node pooling_is_enabled = and(UInt<1>(0h1), decoy_pool_enable)
      module StoreController :
        node pool_stride = UInt<2>(0h2)
    """
    assert extract_max_pool_from_firrtl(firrtl) is None


def test_max_pool_capability_comes_from_the_elaborated_rtl_build_configuration():
    facts = load_facts("gemmini")
    feature = next(i for i in facts["facts"]["interfaces"]
                   if i.get("name") == "elaborated_rtl_features")
    assert feature["status"] == "derived"
    assert feature["features"]["max_pool"] is True
    assert len(feature["source_sha256"]) == 64
    assert rtl_checks.load_default_facts("gemmini")["max_pool_supported"] is True
    gm._isa.cache_clear()
    assert gm._isa().MAX_POOL_SUPPORTED is True
    assert gm._isa().POOL_CAPABLE is True


@pytest.mark.parametrize(("supported", "message"), [
    (False, "exact elaborated RTL facts show native max-pool was compiled out"),
    (None, "exact elaborated RTL facts do not establish native max-pool capability"),
])
def test_pool_emitter_preserves_compiled_out_vs_unknown_rtl_evidence(
        supported, message, monkeypatch):
    base = vars(gm._isa()).copy()
    base.update(MAX_POOL_SUPPORTED=supported, POOL_CAPABLE=supported is True)
    monkeypatch.setattr(gm, "_isa", lambda: SimpleNamespace(**base))
    pool = gm.PoolSpec(in_rows=4, in_cols=4, size=2, stride=2, out_rows=2, out_cols=2)
    with pytest.raises(gm.CodegenError, match=message):
        gm._pool_config_rs1(pool, acc_act=0)


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


def test_pool_capacity_uses_total_accumulator_rows_across_all_rtl_banks():
    space = derive_address_space("gemmini")
    acc = space.store("accumulator")
    assert acc is not None and acc.depth is not None and acc.total_rows is not None
    assert acc.total_rows > acc.depth

    cb = deepcopy(_capsule("GP1_matmul_maxpool_tail_i8"))
    # N=17 is two channel tiles. A 20x20 retained plane therefore consumes exactly 800 rows:
    # valid in the RTL-derived 1024-row banked address space, but invalid under the old 512-row
    # per-bank-depth bug.
    side = 20
    witness_rows = 2 * gm._ceil_dim(side * side)
    assert witness_rows == 800
    assert acc.depth < witness_rows <= acc.total_rows
    cb["tensors"]["A0"]["shape"] = [side * side, 16]
    next(c for c in cb["commands"] if c["opcode"] == "COMMIT")["attributes"][
        "pool_in_dims"] = [side, side]
    gm._isa.cache_clear()
    gm.emit_kernel_mlir(cb)  # a per-bank depth check falsely rejected this valid retained plane
    assert gm._isa().ACCUMULATOR_ROWS == acc.total_rows

    too_large = deepcopy(cb)
    overflow_side = math.isqrt(acc.total_rows // 2) + 1
    while 2 * gm._ceil_dim(overflow_side * overflow_side) <= acc.total_rows:
        overflow_side += 1
    too_large["tensors"]["A0"]["shape"] = [overflow_side * overflow_side, 16]
    next(c for c in too_large["commands"] if c["opcode"] == "COMMIT")["attributes"][
        "pool_in_dims"] = [overflow_side, overflow_side]
    with pytest.raises(gm.CodegenError, match=f"target has {acc.total_rows}"):
        gm.emit_kernel_mlir(too_large)


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
    capsule = yaml.safe_load((directory / "capsule.yaml").read_text(encoding="utf-8"))
    expected = capsule["expected"]
    trace = decode_text(gm.emit_kernel_mlir(cb)[0], target="gemmini")
    assert check(trace, expected, cb, address_model="pointer_args") == {
        "status": "pass", "violations": []}

    # The independent RTL screen must understand the StoreController's native pooling loop too: rows=0
    # in MVOUT is intentional because CONFIG_ST porows*pocols supplies the effective output extent.
    report = rtl_checks.screen(trace, capsule, target="gemmini", command_buffer=cb)
    by_id = {c.id: c for c in report.checks}
    for cid in ("T0.pool_config", "T0.output_store_coverage", "T0.extent_tile_legalization"):
        assert by_id[cid].status == "pass", by_id[cid].to_dict()
    output = rtl_checks.declared_outputs(capsule)[0][0]
    assert by_id["T0.output_store_coverage"].evidence[output["name"]]["covered_cells"] == (
        output["rows"] * output["cols"]
    )

    # FileCheck and the Python screen share the same count derivation. GP1's 25x17 pre-pool matrix is
    # two channel stores, not four ordinary matrix tiles; conv deliberately has no exact count here.
    count = rtl_checks.expected_mvout_count(capsule, rtl_checks.load_default_facts("gemmini"))
    checks = rtl_check_compiler.compile_trace_checks(load_facts("gemmini"), capsule)
    if count is not None:
        assert f"MVOUT_COUNT {count[0]}{{{{$}}}}" in checks


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
