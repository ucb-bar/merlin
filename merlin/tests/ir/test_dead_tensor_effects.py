"""Unused tensor values never authorize discarding unknown execution effects."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import preprocess_text, prune_dead_pure_tensor_ops


def generic(body, *, extra_args="", returned=False):
    result = " -> tensor<2xi32>" if returned else ""
    terminator = "func.return %r : tensor<2xi32>" if returned else "func.return"
    return parse_mlir_text(
        f"""builtin.module {{
func.func private @opaque()
func.func @forward({extra_args}){result} {{
 %z = arith.constant 0 : i32
 %e = tensor.empty() : tensor<2xi32>
 %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>],
   iterator_types = ["parallel"]}} outs(%e : tensor<2xi32>) {{
 ^bb0(%old : i32):
  {body}
  linalg.yield %z : i32
 }} -> tensor<2xi32>
 {terminator}
}}
}}"""
    )


@pytest.mark.parametrize("ordinary", [False, True])
def test_unknown_nested_call_survives_unused_tensor_result(ordinary):
    module = generic("func.call @opaque() : () -> ()")
    module.verify()
    before = str(module)
    if ordinary:
        text, statistics = preprocess_text(before)
        module = parse_mlir_text(text)
        assert statistics["dead_tensor_ops_pruned"] == 0
    else:
        assert prune_dead_pure_tensor_ops(module) == 0
        assert str(module) == before
    module.verify()
    assert any(op.name == "func.call" for op in module.walk())


def test_nested_store_survives_unused_tensor_result():
    module = generic(
        "%i = arith.constant 0 : index\n  memref.store %z, %buffer[%i] : memref<1xi32>",
        extra_args="%buffer : memref<1xi32>",
    )
    module.verify()
    before = str(module)
    assert prune_dead_pure_tensor_ops(module) == 0
    assert str(module) == before


def test_unknown_nested_operation_without_effect_interface_survives():
    module = generic('"observer.tick"() : () -> ()')
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 0
    assert any(op.name == "builtin.unregistered" for op in module.walk())


@pytest.mark.parametrize("name", ["arith.foreign", "tensor.observe", "quant_ext.effectful"])
def test_dialect_name_does_not_supply_missing_effects(name):
    module = parse_mlir_text(
        f'''builtin.module {{ func.func @forward() {{
 %r = "{name}"() : () -> tensor<2xi32>
 func.return
}} }}'''
    )
    module.verify()
    before = str(module)
    assert prune_dead_pure_tensor_ops(module) == 0
    assert str(module) == before


def test_unknown_metadata_is_a_barrier_even_on_known_arithmetic():
    module = parse_mlir_text(
        """builtin.module { func.func @forward() {
 %z = arith.constant {external.effect} 0 : i32
 func.return
} }"""
    )
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 0


def test_unknown_generic_metadata_is_a_barrier():
    from xdsl.dialects.builtin import UnitAttr

    module = generic("")
    operation = next(op for op in module.walk() if op.name == "linalg.generic")
    operation.attributes["external.effect"] = UnitAttr()
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 0


def test_external_library_call_metadata_is_a_barrier():
    from xdsl.dialects.builtin import StringAttr

    module = generic("")
    operation = next(op for op in module.walk() if op.name == "linalg.generic")
    operation.properties["library_call"] = StringAttr("opaque")
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 0


def test_missing_required_structured_metadata_is_not_erased():
    module = generic("")
    operation = next(op for op in module.walk() if op.name == "linalg.generic")
    operation.properties.pop("indexing_maps")
    before = tuple(module.walk())
    assert prune_dead_pure_tensor_ops(module) == 0
    assert tuple(module.walk()) == before


def test_known_quantization_spelling_without_effect_contract_is_retained():
    module = parse_mlir_text(
        """builtin.module { func.func @forward(%w : tensor<2xi8>, %s : tensor<f32>, %z : tensor<i32>) {
 %r = "quant_ext.dequantize_per_tensor"(%w, %s, %z)
   : (tensor<2xi8>, tensor<f32>, tensor<i32>) -> tensor<2xf32>
 func.return
} }"""
    )
    module.verify()
    before = str(module)
    assert prune_dead_pure_tensor_ops(module) == 0
    assert str(module) == before


def test_non_speculatable_integer_operation_is_retained():
    module = parse_mlir_text(
        """builtin.module { func.func @forward(%x : i32, %y : i32) {
 %r = arith.divsi %x, %y : i32
 func.return
} }"""
    )
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 0


def test_unused_exact_tensor_fill_chain_is_removed():
    module = parse_mlir_text(
        """builtin.module { func.func @forward() {
 %z = arith.constant 7 : i16
 %e = tensor.empty() : tensor<3x5xi16>
 %r = linalg.fill ins(%z : i16) outs(%e : tensor<3x5xi16>) -> tensor<3x5xi16>
 func.return
} }"""
    )
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 3
    module.verify()
    assert [op.name for op in module.walk()] == ["builtin.module", "func.func", "func.return"]


def test_unused_pure_body_can_release_its_nested_scalar_capture():
    module = generic("%sum = arith.addi %old, %z : i32")
    module.verify()
    assert prune_dead_pure_tensor_ops(module) == 3
    module.verify()


def test_live_body_keeps_its_nested_scalar_capture():
    module = generic("%sum = arith.addi %old, %z : i32", returned=True)
    module.verify()
    before = str(module)
    assert prune_dead_pure_tensor_ops(module) == 0
    assert str(module) == before


def test_live_use_in_another_block_keeps_its_definition():
    module = parse_mlir_text(
        """builtin.module { func.func @forward() -> i32 {
 %z = arith.constant 3 : i32
 cf.br ^next
^next:
 func.return %z : i32
} }"""
    )
    module.verify()
    before = str(module)
    assert prune_dead_pure_tensor_ops(module) == 0
    assert str(module) == before
    module.verify()


def test_physical_allocations_and_stores_are_preserved():
    module = parse_mlir_text(
        """builtin.module { func.func @forward() {
 %z = arith.constant 0 : i32
 %i = arith.constant 0 : index
 %a = memref.alloc() : memref<1xi32>
 memref.store %z, %a[%i] : memref<1xi32>
 memref.dealloc %a : memref<1xi32>
 func.return
} }"""
    )
    module.verify()
    before = str(module)
    assert prune_dead_pure_tensor_ops(module) == 0
    assert str(module) == before


def test_ordinary_native_model_observes_side_effect_from_dead_tensor(tmp_path):
    import hashlib
    import json

    import numpy as np

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    if not toolchain.compiler_python().is_file():
        pytest.skip("ordinary native MLIR compiler interpreter is unavailable")
    source = """builtin.module {
"memref.global"() <{sym_name = "counter", sym_visibility = "private", type = memref<1xi32>,
  initial_value = dense<0> : tensor<1xi32>}> : () -> ()
func.func private @observe() {
 %g = memref.get_global @counter : memref<1xi32>
 %i = arith.constant 0 : index
 %one = arith.constant 1 : i32
 %old = memref.load %g[%i] : memref<1xi32>
 %next = arith.addi %old, %one : i32
 memref.store %next, %g[%i] : memref<1xi32>
 func.return
}
func.func @forward() -> tensor<1xi32> {
 %z = arith.constant 0 : i32
 %e = tensor.empty() : tensor<2xi32>
 %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>],
   iterator_types = ["parallel"]} outs(%e : tensor<2xi32>) {
 ^bb0(%old : i32):
  func.call @observe() : () -> ()
  linalg.yield %z : i32
 } -> tensor<2xi32>
 %g = memref.get_global @counter : memref<1xi32>
 %i = arith.constant 0 : index
 %actual = memref.load %g[%i] : memref<1xi32>
 %out = tensor.splat %actual : tensor<1xi32>
 func.return %out : tensor<1xi32>
} }"""
    (tmp_path / "source.mlir").write_text(source)
    result = lower_model(source, tmp_path / "native", targets=("host",), vectorize=False)
    assert result.host_so is not None
    assert result.stats["dead_tensor_ops_pruned"] == 0
    model = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
    actual = np.zeros(1, dtype=np.int32)
    observations = []
    for expected in (2, 4, 6):
        model([(actual.ctypes.data, actual.shape)])
        observations.append(actual.copy())
        assert actual.tolist() == [expected]
    np.savez(tmp_path / "actual.npz", **{f"call{i}": row for i, row in enumerate(observations)})
    artifacts = (tmp_path / "source.mlir", result.ll_path, result.host_so, result.host_so.with_suffix(".o"))
    (tmp_path / "observation.json").write_text(
        json.dumps(
            {
                "scope": "ordinary host compilation and complete three-call counter readback; no target qualification",
                "artifacts": [
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in artifacts
                ],
                "invocations": [str(path) for path in sorted(result.workdir.rglob("invocation.json"))],
                "complete_counter_outputs": [row.tolist() for row in observations],
            },
            indent=2,
        )
        + "\n"
    )
