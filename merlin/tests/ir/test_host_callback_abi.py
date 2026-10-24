"""Host C interfaces must preserve the explicitly selected provider callback ABI."""

import pytest


@pytest.mark.parametrize("private_ciface", [False, True])
def test_host_preprocessing_preserves_provider_callback_abi(private_ciface):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import preprocess_text

    attributes = " attributes {llvm.emit_c_interface}" if private_ciface else ""
    source = (
        """builtin.module {
      func.func @forward(%x: tensor<2xi32>) -> tensor<2xi32> {
        %r = func.call @callback(%x) : (tensor<2xi32>) -> tensor<2xi32>
        func.return %r : tensor<2xi32>
      }
      func.func private @callback(tensor<2xi32>) -> tensor<2xi32>"""
        + attributes
        + "\n}"
    )
    lowered, _ = preprocess_text(source)
    functions = [op for op in parse_mlir_text(lowered).walk() if op.name == "func.func"]
    public, callback = functions
    assert "llvm.emit_c_interface" in public.attributes
    # An external callback's caller-selected ABI must survive host preprocessing.
    # Adding this attribute changes calls into unresolved _mlir_ciface symbols.
    assert ("llvm.emit_c_interface" in callback.attributes) == private_ciface
