"""Raw connected regions cannot submit host arithmetic or skip frozen source admission."""
import pytest
import yaml

from merlin.targetgen import capsule_runner as runner
from merlin.targetgen import owned_host_slice_execution as slices


def _capsule():
    return {"kind": "model_slice", "operation": {"op": "model"}, "semantic": {"must_accelerate": True}}


def test_raw_slice_uses_existing_owned_compiler(monkeypatch):
    capsule = _capsule()
    monkeypatch.setattr(runner, "_grade_owned_model_capsule_inline", lambda *a, **k: {"route": "owned"})
    monkeypatch.setattr(runner, "_grade_public_program_capsule_inline", lambda *a, **k: pytest.fail("candidate host route"))
    assert runner._grade_candidate_model_capsule_inline(capsule, target="fixture", timeout=1, package_dir="unused", context={}) == {"route": "owned"}
    with pytest.raises(ValueError, match="Merlin host ownership"):
        runner._grade_candidate_model_capsule_inline(capsule, target="fixture", timeout=1, package_dir="unused", context={"public_program_execution": {}})
    with pytest.raises(ValueError, match="Merlin host ownership"):
        runner._grade_candidate_model_capsule_inline(capsule, target="fixture", timeout=1, package_dir="unused", context={"checked_source_diagnostic": {}})
    assert not slices.selected_source_slice({**capsule, "operation": {"op": "matmul"}})


def test_unsupported_host_seam_is_reported_before_materialization(tmp_path):
    capsule = {**_capsule(), "interface_mlir": "source.mlir", "linalg_mlir": "source.mlir",
               "software_screen": {"status": "unsupported"}}
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    with pytest.raises(ValueError, match="placement is not admitted"):
        slices.materialize_source_slice({**capsule, "__dir__": str(tmp_path)}, target="fixture", output=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("fill_tags", [False, True])
def test_source_join_ignores_only_literal_allocation_scaffolding(tmp_path, monkeypatch, fill_tags):
    from types import SimpleNamespace
    from merlin.common import mlir_query as mq
    from merlin.compile.model_execution_inputs import file_sha256
    from merlin.targetgen import native_model_execution as native
    from merlin.targetgen import owned_host_model_execution as owned

    source = tmp_path / "model.mlir"
    source.write_text('''module {
      func.func @forward(%a: tensor<2x2xi8>, %b: tensor<2x2xi8>) -> tensor<2x2xi32> {
        %empty = tensor.empty() : tensor<2x2xi32>
        %zero = arith.constant 0 : i32
        %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x2xi32>) -> tensor<2x2xi32>
        %r = linalg.matmul {prov.region_id = "dot"} ins(%a, %b : tensor<2x2xi8>, tensor<2x2xi8>) outs(%init : tensor<2x2xi32>) -> tensor<2x2xi32>
        return %r : tensor<2x2xi32>
      }
    }''')
    (tmp_path / "capsule.yaml").write_text("{}")
    module = mq.parse(source.read_text())
    if fill_tags:
        from xdsl.dialects.builtin import StringAttr
        next(op for op in module.walk() if op.name == "arith.constant").attributes["prov.op"] = StringAttr("fill")
        source.write_text(str(module))
    dot = next(op for op in module.walk() if op.name == "linalg.matmul")
    rows = [{"target_eligible": True, "region_id": "dot", "carrier_op": "linalg.matmul",
             "captured_input_format": "int8", "eligibility_input_format": "int8",
             "captured_weight_format": "int8", "eligibility_weight_format": "int8", "semantic_family": "contraction"}]
    certificate = {"source_mlir_sha256": file_sha256(source), "n_unknown_capture_formats": 0,
                   "n_precision_transform_obligations": 0, "precision_transform_verification": {"status": "not_required"},
                   "n_regions": 1, "regions": rows}
    monkeypatch.setattr(native, "independent_frozen_source_eligibility", lambda *a, **k: certificate)
    monkeypatch.setattr(owned, "_frozen_split", lambda *a, **k: {"status": "synthetic-join-only"})
    group = SimpleNamespace(index=0, placement="gemmini", members=[dot])
    result = owned._source_placement({"__dir__": str(tmp_path)}, certificate, module, [group], target="gemmini", source_path=source)
    assert result["eligible_source_op_indices"] == [3]
    # An observed host/layout format gap requires the independently admitted
    # frozen host body; it does not permit guessing the device operand format.
    from merlin.targetgen import frozen_placement
    calls = []
    monkeypatch.setattr(frozen_placement, "read", lambda *args: calls.append(args))
    certificate.update(n_unknown_capture_formats=1, precision_transform_verification={"status": "unknown_capture_format"})
    assert owned._source_placement({"__dir__": str(tmp_path)}, certificate, module, [group], target="gemmini", source_path=source)["eligible_source_op_indices"] == [3]
    assert len(calls) == 1
    rows[0]["captured_input_format"] = None
    with pytest.raises(ValueError, match="observed input-format admission"):
        owned._source_placement({"__dir__": str(tmp_path)}, certificate, module, [group], target="gemmini", source_path=source)
    rows[0]["captured_input_format"] = "int8"
    rows[0].update(target_eligible=False, eligibility_undetermined=True)
    with pytest.raises(ValueError, match="unknown captured precision"):
        owned._source_placement({"__dir__": str(tmp_path)}, certificate, module, [group], target="gemmini", source_path=source)
    rows[0].update(target_eligible=True, eligibility_undetermined=False)
    # Omitting real computation still refuses; only literal/setup nodes are excluded.
    certificate.update(n_regions=0, regions=[])
    with pytest.raises(ValueError, match="complete operation eligibility"):
        owned._source_placement({"__dir__": str(tmp_path)}, certificate, module, [group], target="gemmini", source_path=source)
    from xdsl.dialects.builtin import StringAttr
    dot.attributes["prov.op"] = StringAttr("fill")
    with pytest.raises(ValueError, match="unsupported omitted fill carrier"):
        owned._source_placement({"__dir__": str(tmp_path)}, certificate, module, [group], target="gemmini", source_path=source)
