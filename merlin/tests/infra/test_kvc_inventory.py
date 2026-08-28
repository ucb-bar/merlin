"""The workload inventory must not silently shrink a model.

Two defects found while building it, both of which produce a plausible-looking number rather than an
error, and both of which would corrupt cost-weighted task selection downstream:

  * reading provenance without the ``prov.`` prefix returned empty for EVERY op, so every row fell
    back to its MLIR op name and the model appeared to have no semantic families at all;
  * walking every op counted the ops inside a linalg body (``linalg.yield``, ``linalg.index``) as
    ops in their own right, roughly doubling the op count.

A third property is a design requirement rather than a past bug: an opaque ``func.call`` the importer
could not decompose must be priced from its declared signature, or reported UNKNOWN -- never zero.
SmolVLA's single opaque op is its patch embedding at ~1.2 GFLOP.
"""
import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_SCRIPT = (repo_root() / "merlin" / "experiments" / "llm_kernel_vs_compiler_v0"
           / "scripts" / "inventory_models.py")


def _load():
    if not _SCRIPT.is_file():
        pytest.skip(f"inventory script not present at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("kvc_inventory", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MLIR = """module {
  func.func private @aten_convolution_default(tensor<1x3x512x512xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tensor<1x768x32x32xf32>
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %img: tensor<1x3x512x512xf32>, %w: tensor<768x3x16x16xf32>, %bi: tensor<768xf32>) -> tensor<4x16xf32> {
    %e = tensor.empty() : tensor<4x16xf32>
    %m = linalg.matmul {prov.op = "matmul", prov.family = "contraction"}
         ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
         outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
    %c = func.call @aten_convolution_default(%img, %w, %bi) : (tensor<1x3x512x512xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tensor<1x768x32x32xf32>
    return %m : tensor<4x16xf32>
  }
}
"""


@pytest.fixture(scope="module")
def inv(tmp_path_factory):
    mod = _load()
    p = tmp_path_factory.mktemp("kvcinv") / "model.mlir"
    p.write_text(_MLIR)
    return mod.inventory(p)


def test_provenance_families_are_read(inv):
    """The regression: prov.* keys carry their prefix, and dropping it emptied every family."""
    fams = {r["family"] for r in inv["rows"]}
    assert "contraction" in fams, (
        f"no row carried prov.family; got {fams}. Reading provenance without the 'prov.' prefix "
        "yields empty for every op and silently degrades to MLIR op names."
    )
    assert "matmul" in inv["work_by_family"], inv["work_by_family"]


def test_body_ops_are_not_counted_as_ops(inv):
    """linalg.yield / linalg.index are payload, not ops; counting them inflates the op count."""
    names = [r["mlir_op"] for r in inv["rows"]]
    assert "linalg.yield" not in names and "linalg.index" not in names, names
    assert inv["n_linalg_ops"] == 1, f"expected the single matmul, got {names}"


def test_matmul_work_is_in_flops(inv):
    """4x8 @ 8x16 is 512 MACs, and work is priced in FLOPs, so 1024.

    The unit matters because opaque ops are priced by a separate formula: if one side counted MACs
    and the other FLOPs, an op would be weighted by how well the importer handled it.
    """
    assert inv["work_by_family"]["matmul"] == 4 * 8 * 16 * 2


def test_an_opaque_op_is_priced_from_its_signature_not_dropped(inv):
    """An op the importer could not decompose is still typed, so its cost is still knowable."""
    assert inv["n_opaque_ops"] == 1
    op = inv["opaque"][0]
    assert op["callee"] == "aten_convolution_default"
    assert op["arg_shapes"][0] == [1, 3, 512, 512]
    assert op["result_shape"] == [1, 768, 32, 32]
    # |out| x (C_in x kh x kw) MACs, x2 arith ops per MAC -- the same unit work_of uses.
    assert op["work"] == (768 * 32 * 32) * (3 * 16 * 16) * 2 == 1_207_959_552
    assert op["priced"] is True
    assert inv["total_work"] > op["work"], "the opaque op must be included in the total"


def test_an_unpriceable_opaque_op_is_unknown_rather_than_zero():
    """Fail closed: no cost formula must read as UNKNOWN, so a gap cannot masquerade as free."""
    mod = _load()
    rows = mod._opaque_rows(
        "func.func private @aten_mystery_default(tensor<4x4xf32>) -> tensor<4x4xf32>\n"
        "  %0 = func.call @aten_mystery_default(%x) : (tensor<4x4xf32>) -> tensor<4x4xf32>\n"
    )
    assert len(rows) == 1
    assert rows[0]["work"] is None, "an unpriceable op must be UNKNOWN, never 0"
    assert rows[0]["priced"] is False
    assert "UNPRICED" in rows[0]["note"]


def test_family_match_is_token_wise_not_substring():
    """A substring test would let an unrelated callee inherit a cost formula it must not have."""
    mod = _load()
    assert mod._opaque_family("aten_convolution_default") == "convolution"
    assert mod._opaque_family("aten_deconvolutionish_default") == "unknown"
