"""Actual quant-inner propagation changes only the reviewed provenance annotations."""

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.runtime.dispatch_runtime import _propagate_quant_inner
from merlin.verify.lean import stages
from merlin.verify.lean.check import check_typed_stage

SOURCE = """
module {
  func.func @forward() -> f32 {
    %0 = tensor.empty() : tensor<2xf32>
    %c = arith.constant 0 : index
    %1 = "tensor.extract"(%0, %c) {prov.quant_inner_0 = "weight._data"} : (tensor<2xf32>, index) -> f32
    return %1 : f32
  }
}
"""


def test_propagation_only_adds_provenance(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    module = parse_mlir_file(source)
    before = stages.generic_text(module)
    assert _propagate_quant_inner(module) == 1
    empty = next(op for op in module.walk() if op.name == "tensor.empty")
    assert empty.attributes["prov.quant_inner"].data == "weight._data"
    del empty.attributes["prov.quant_inner"]
    assert stages.generic_text(module) == before


def test_existing_checker_abstains_on_changed_inner_annotations(tmp_path, lean_tools):
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    module = parse_mlir_file(source)
    with stages.capture(tmp_path / "capture"):
        stages.run("prepare-propagate-quant-inner", _propagate_quant_inner, module)
    stage, = stages.completed(tmp_path / "capture")
    assert (stage / "before.mlir").read_bytes() != (stage / "after.mlir").read_bytes()
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt
    # Alpha matching currently compares operation attributes literally.
    # A reviewed semantics-free annotation rule is needed before this boundary
    # can verify; runtime binding and initialized-source claims remain separate.
