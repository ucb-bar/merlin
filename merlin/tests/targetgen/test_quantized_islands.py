# ruff: noqa: E501 -- captured MLIR is one op per line
"""A quantized non-contraction op is int8 work, even in the frontend's dequantize -> f32 -> quantize spelling.

WHY THIS EXISTS. PT2E leaves every quantized non-contraction op as ``dequantize -> f32 op -> quantize``.
The core-aten site screen read the f32 element type and refused all of them by precision ("input dtype
'f32' not in elementwise_map formats ['int8']"), so no relu, pad, cat or pool of an int8 capture could
ever be a device placement candidate, although the backend matches exactly that sandwich.
``quantized_islands`` names the f32 ops that compute only between int8 quantization edges (chains
included), and the site screen admits them as the int8 operation they compute.
"""

from __future__ import annotations

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites, quantized_islands
from merlin.targetgen.host_capabilities import validate_host_capabilities

pytestmark = pytest.mark.target("gemmini")
_BASE = repo_root() / "examples/gemmini/target"

_DQ = '"quant_ext.dequantize_per_tensor"({x}, %s, %z) <{{quant_min = -128 : i64, quant_max = 127 : i64}}> : ({t}xi8>, tensor<f32>, tensor<i64>) -> {t}xf32>'
_Q = '"quant_ext.quantize_per_tensor"({x}, %s, %z) <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}> : ({t}xf32>, tensor<f32>, tensor<i64>) -> {t}xi8>'

_NEG = f"""
builtin.module {{
  func.func @forward(%x: tensor<4x8xi8>) -> tensor<4x8xi8> {{
    %c = arith.constant 0.02 : f32
    %s = tensor.splat %c : tensor<f32>
    %c0 = arith.constant 0 : i64
    %z = tensor.splat %c0 : tensor<i64>
    %d = {_DQ.format(x="%x", t="tensor<4x8")}
    %e = tensor.empty() : tensor<4x8xf32>
    %n = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%d : tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) attrs = {{prov.aten = "aten.neg.default"}} {{
    ^bb0(%a: f32, %o: f32):
      %v = arith.negf %a : f32
      linalg.yield %v : f32
    }} -> tensor<4x8xf32>
    %p = tensor.concat dim(1) %n, %n : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
    %r = "tensor.extract_slice"(%p) <{{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 4, 8>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}}> : (tensor<4x16xf32>) -> tensor<4x8xf32>
    %q = {_Q.format(x="%r", t="tensor<4x8")}
    func.return %q : tensor<4x8xi8>
  }}
}}
"""

_FLOAT = """
builtin.module {
  func.func @forward(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %e = tensor.empty() : tensor<4x8xf32>
    %n = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) attrs = {prov.aten = "aten.neg.default"} {
    ^bb0(%a: f32, %o: f32):
      %v = arith.negf %a : f32
      linalg.yield %v : f32
    } -> tensor<4x8xf32>
    func.return %n : tensor<4x8xf32>
  }
}
"""


@pytest.fixture(scope="module")
def selection():
    spec = fixture_resolved(_BASE, "gemmini")
    return {
        "spec": spec,
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
        "capability_contract": load_yaml(_BASE / "contracts/target_contract.yaml"),
    }


def _sites(selection, source):
    with IR_LOCK:
        module = parse_mlir_text(source)
        return module, non_contraction_sites(module, selection, contract=selection["capability_contract"])


def test_the_sandwich_chain_is_one_island(selection):
    module, sites = _sites(selection, _NEG)
    with IR_LOCK:
        islands = quantized_islands(module)
    assert len(islands) == 3  # the neg, the concat and the slice view between the quantize edges
    screened = {site["source_operation"]: site for site in sites}
    assert set(screened) == {"linalg.generic", "tensor.concat"}  # the slice is a view, not a site
    for site in screened.values():
        assert site["quantized_island"] and site["signature"]["operand_dtype"] == "i8"
        assert not any("precision" in reason or "'f32'" in reason for reason in site["reasons"]), site["reasons"]
        assert site["status"] in {"admitted", "unknown"}


def test_a_float_op_outside_any_quantization_edge_stays_refused(selection):
    _module, (site,) = _sites(selection, _FLOAT)
    assert not site["quantized_island"] and site["status"] == "refused"


def test_reviewed_crossings_admit_the_island_sites(selection):
    import copy

    reviewed = copy.deepcopy(selection)
    for row in reviewed["spec"]["transfer_contracts"]["declarations"]:
        row["evidence"]["status"] = "reviewed"
    _module, sites = _sites(reviewed, _NEG)
    assert all(site["status"] == "admitted" for site in sites), [site["reasons"] for site in sites]


def test_a_floating_source_is_screened_prospectively_as_its_int8_program(selection):
    """Before the recapture, a floating source's sites are screened as the int8 work its recipe makes."""
    import copy

    reviewed = copy.deepcopy(selection)
    for row in reviewed["spec"]["transfer_contracts"]["declarations"]:
        row["evidence"]["status"] = "reviewed"
    with IR_LOCK:
        module = parse_mlir_text(_FLOAT)
        (site,) = non_contraction_sites(module, reviewed, contract=reviewed["capability_contract"], prospective=True)
    assert site["quantized_island"] and site["status"] == "admitted", site["reasons"]
