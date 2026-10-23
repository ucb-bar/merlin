"""The core-aten int8 pipeline screens elementwise, reduction and movement sites, not only contractions.

WHY THIS EXISTS. ``core_aten_spec_int8.admission_plan`` walked ``observe_contractions`` alone, so on the
core-aten path no relu, pool, cat or cast could ever be a device placement candidate, whatever the
software spec and capability contract said. ``non_contraction_sites`` screens each top-level
non-contraction site through the same authorities: the operand-load and readout transfer contracts,
SW ``admit_operation`` and HW ``is_eligible``. It selects no backend and no lowering.
"""

from __future__ import annotations

import copy

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.core_aten_spec_int8 import admission_plan, non_contraction_sites
from merlin.targetgen.host_capabilities import validate_host_capabilities

pytestmark = pytest.mark.target("gemmini")
_BASE = repo_root() / "examples/gemmini/target"

_PROGRAM = """
module {
  func.func @forward(%x: tensor<2x8xi8>, %y: tensor<2x8xi8>) -> tensor<2x8xi8> {
    %z = arith.constant 0 : i8
    %e = tensor.empty() : tensor<2x8xi8>
    %w = tensor.empty() : tensor<2x8xi32>
    %r = tensor.empty() : tensor<2xi32>
    %add = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
                                            affine_map<(d0, d1) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel"]}
           ins(%x, %y : tensor<2x8xi8>, tensor<2x8xi8>) outs(%e : tensor<2x8xi8>)
           attrs = {prov.aten = "aten.add.Tensor"} {
    ^bb0(%a: i8, %b: i8, %o: i8):
      %s = arith.addi %a, %b : i8
      linalg.yield %s : i8
    } -> tensor<2x8xi8>
    %wide = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                            iterator_types = ["parallel", "parallel"]}
            ins(%x : tensor<2x8xi8>) outs(%w : tensor<2x8xi32>) attrs = {prov.aten = "aten._to_copy.default"} {
    ^bb0(%a: i8, %o: i32):
      %c = arith.extsi %a : i8 to i32
      linalg.yield %c : i32
    } -> tensor<2x8xi32>
    %clone = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel"]}
             ins(%add : tensor<2x8xi8>) outs(%e : tensor<2x8xi8>) attrs = {prov.aten = "aten.clone.default"} {
    ^bb0(%a: i8, %o: i8):
      linalg.yield %a : i8
    } -> tensor<2x8xi8>
    %fill = linalg.fill {prov.aten = "aten.full_like.default"} ins(%z : i8) outs(%e : tensor<2x8xi8>) -> tensor<2x8xi8>
    %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                           iterator_types = ["parallel", "reduction"]}
           ins(%x : tensor<2x8xi8>) outs(%r : tensor<2xi32>) attrs = {prov.aten = "aten.sum.dim_IntList"} {
    ^bb0(%a: i8, %o: i32):
      %v = arith.extsi %a : i8 to i32
      %s = arith.addi %o, %v : i32
      linalg.yield %s : i32
    } -> tensor<2xi32>
    return %clone : tensor<2x8xi8>
  }
}
"""


@pytest.fixture(scope="module")
def selection():
    return {
        "spec": fixture_resolved(_BASE, "gemmini"),
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
        "capability_contract": load_yaml(_BASE / "contracts/target_contract.yaml"),
    }


def _sites(selection):
    with IR_LOCK:
        module = mq.parse(_PROGRAM)
        return {
            site["source_operator"]: site
            for site in non_contraction_sites(module, selection, contract=selection["capability_contract"])
        }


def test_every_non_contraction_site_is_screened(selection):
    sites = _sites(selection)
    assert set(sites) == {
        "aten.add.Tensor",
        "aten._to_copy.default",
        "aten.clone.default",
        "aten.full_like.default",
        "aten.sum.dim_IntList",
    }
    families = {name: site["family"] for name, site in sites.items()}
    assert families["aten.clone.default"] == "movement" and families["aten.sum.dim_IntList"] == "reduction"
    assert sites["aten.clone.default"]["signature"]["form"] == "copy"
    assert sites["aten.full_like.default"]["signature"]["form"] == "zero_fill"
    assert sites["aten.sum.dim_IntList"]["signature"]["kind"] == "sum"


def test_a_site_is_admitted_only_when_every_authority_admits_it(selection):
    sites = _sites(selection)
    # int8 -> int32 widening: standalone elementwise SW row, int32 result format, reviewed i32 readout.
    assert sites["aten._to_copy.default"]["status"] == "admitted", sites["aten._to_copy.default"]["reasons"]
    for name in ("aten.add.Tensor", "aten.clone.default", "aten.full_like.default", "aten.sum.dim_IntList"):
        assert sites[name]["status"] != "refused", (name, sites[name]["reasons"])
        if sites[name]["status"] == "unknown":
            assert sites[name]["reasons"], name


def test_reviewing_the_int8_readout_admits_the_int8_sites(selection):
    reviewed = copy.deepcopy(selection)
    contracts = reviewed["spec"]["transfer_contracts"]
    rows = contracts.get("declarations") or [value for value in contracts.values() if isinstance(value, dict)]
    for row in rows:
        if row.get("id", "") == "int8_readout" or row.get("copy", {}).get("dtype") == "int8":
            row.setdefault("evidence", {})["status"] = "reviewed"
    sites = _sites(reviewed)
    for name in ("aten.add.Tensor", "aten.clone.default", "aten.full_like.default"):
        steps = {row.get("transfer"): row for row in sites[name]["steps"] if "transfer" in row}
        if steps["readout"]["status"] != "admitted":
            pytest.skip("the selected spec declares no int8 readout crossing to review")
        assert sites[name]["status"] == "admitted", (name, sites[name]["reasons"])


def test_a_float_site_is_refused_by_precision(selection):
    source = _PROGRAM.replace("i8", "f32").replace("arith.addi %a, %b : f32", "arith.addf %a, %b : f32")
    with IR_LOCK:
        module = mq.parse(source.split("%wide")[0] + "return %add : tensor<2x8xf32>\n  }\n}\n")
        (site,) = non_contraction_sites(module, selection, contract=selection["capability_contract"])
    assert site["status"] == "refused"


def test_without_a_contract_the_hardware_screen_is_not_claimed(selection):
    with IR_LOCK:
        module = mq.parse(_PROGRAM)
        sites = non_contraction_sites(module, selection)
    assert all(site["status"] != "admitted" for site in sites)
    assert all("no selected capability contract" in " ".join(site["reasons"]) for site in sites)


def test_the_contraction_plan_carries_the_sites(selection):
    with IR_LOCK:
        module = mq.parse(_PROGRAM)
        plan = admission_plan(module, selection, native=True)
    assert plan["sites"] == [] and len(plan["non_contraction_sites"]) == 5


@pytest.mark.parametrize(
    "change, expected",
    [("three_inputs", "tensor_operands"), ("transcendental", "expression"), ("inner", "broadcasting")],
)
def test_primitive_limits_are_enforced(selection, change, expected):
    source = _PROGRAM.split("%wide")[0] + "return %add : tensor<2x8xi8>\n  }\n}\n"
    if change == "three_inputs":
        source = source.replace(
            "ins(%x, %y : tensor<2x8xi8>, tensor<2x8xi8>)",
            "ins(%x, %y, %x : tensor<2x8xi8>, tensor<2x8xi8>, tensor<2x8xi8>)",
        )
        source = source.replace(
            "affine_map<(d0, d1) -> (d0, d1)>],", "affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],"
        )
        source = source.replace("%b: i8, %o: i8", "%b: i8, %c: i8, %o: i8")
    elif change == "transcendental":
        source = source.replace(
            "%s = arith.addi %a, %b : i8",
            (
                "%af = arith.sitofp %a : i8 to f32\n"
                "      %exp = math.exp %af : f32\n"
                "      %s = arith.fptosi %exp : f32 to i8"
            ),
        )
        source = source.replace("aten.add.Tensor", "aten.sigmoid.default")
    else:
        source = source.replace("%y: tensor<2x8xi8>", "%y: tensor<2xi8>")
        source = source.replace(
            "ins(%x, %y : tensor<2x8xi8>, tensor<2x8xi8>)", "ins(%x, %y : tensor<2x8xi8>, tensor<2xi8>)"
        )
        old = (
            "affine_map<(d0, d1) -> (d0, d1)>,\n"
            "                                            affine_map<(d0, d1) -> (d0, d1)>]"
        )
        source = source.replace(
            old,
            (
                "affine_map<(d0, d1) -> (d0)>,\n"
                "                                            affine_map<(d0, d1) -> (d0, d1)>]"
            ),
        )
    with IR_LOCK:
        (site,) = non_contraction_sites(mq.parse(source), selection, contract=selection["capability_contract"])
    software = next(step["software"] for step in site["steps"] if "software" in step)
    assert software["status"] == "unsupported", software
    from merlin.targetgen.software_spec import admit_operation

    standalone = {
        **selection["spec"],
        "operations": [row for row in selection["spec"]["operations"] if row["id"] == "elementwise_map_standalone"],
    }
    decision = admit_operation(standalone, site["source_operator"], site["signature"], "accelerator")
    assert decision["status"] == "unsupported", decision
    assert expected in decision["reason"], decision


def test_standalone_maximum_is_refused_by_sum_only_declaration(selection):
    source = _PROGRAM.replace("arith.addi %o, %v", "arith.maxsi %o, %v").replace(
        "aten.sum.dim_IntList", "aten.amax.default"
    )
    with IR_LOCK:
        sites = non_contraction_sites(mq.parse(source), selection, contract=selection["capability_contract"])
    maximum = next(site for site in sites if site["source_operator"] == "aten.amax.default")
    decision = next(step["software"] for step in maximum["steps"] if "software" in step)
    assert decision["status"] == "unsupported"
    from merlin.targetgen.software_spec import admit_operation

    only_sum = {
        **selection["spec"],
        "operations": [row for row in selection["spec"]["operations"] if row["id"] == "reduction_standalone"],
    }
    decision = admit_operation(only_sum, "aten.amax.default", maximum["signature"], "accelerator")
    assert decision["status"] != "admitted"
    assert "kind" in decision["reason"] or decision["status"] == "unknown"


@pytest.mark.parametrize(
    "declaration,operands,results",
    [
        ("host_quantize_exact", ["f32", "f64", "i64"], ["i8"]),
        ("host_dequantize_per_tensor", ["i8", "f32", "i64"], ["f32"]),
    ],
)
@pytest.mark.parametrize("rank", range(6))
def test_exact_host_quantization_boundaries_admit_rank_five(selection, declaration, operands, results, rank):
    from merlin.targetgen.software_spec import admit_operation

    row = next(row for row in selection["host"]["operations"] if row["id"] == declaration)
    decision = admit_operation(
        {**selection["host"], "operations": [row]},
        row["ops"][0],
        dict(
            family="elementwise_map",
            ordered_operand_dtypes=operands,
            ordered_result_dtypes=results,
            rank=rank,
            quantization_parameters=dict(zero_point=0, quant_min=-128, quant_max=127),
        ),
        "host",
    )
    assert decision["status"] == "admitted", decision
