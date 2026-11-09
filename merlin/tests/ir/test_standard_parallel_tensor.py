import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


def source(step=1):
    return f'''module {{
      func.func @f(%input: tensor<4x8xf32>, %n: index) -> tensor<4x8xf32> {{
        %out = "scf.forall"(%n, %input) <{{
          operandSegmentSizes = array<i32: 0, 1, 0, 1>,
          staticLowerBound = array<i64: 0, 0>,
          staticUpperBound = array<i64: 4, -9223372036854775808>,
          staticStep = array<i64: 1, {step}>
        }}> ({{
        ^bb0(%i: index, %j: index, %shared: tensor<4x8xf32>):
          "scf.forall.in_parallel"() ({{
          ^update:
          }}) : () -> ()
        }}) : (index, tensor<4x8xf32>) -> tensor<4x8xf32>
        return %out : tensor<4x8xf32>
      }}
    }}'''


def test_parallel_tensor_bindings_roundtrip_multiaxis_dynamic_bounds():
    module = parse_mlir_text(source())
    module.verify()
    assert not any(op.name == "builtin.unregistered" for op in module.walk())
    reloaded = parse_mlir_text(text(module, generic=True))
    reloaded.verify()
    assert [op.name for op in reloaded.walk()] == [op.name for op in module.walk()]


def test_parallel_tensor_binding_rejects_zero_step():
    with pytest.raises(Exception, match="steps must be positive"):
        parse_mlir_text(source(step=0)).verify()
