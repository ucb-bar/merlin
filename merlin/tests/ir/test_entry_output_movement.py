"""Direct user results need device copies; post-input observations stay direct."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.entry_output_movement import expose_argument_results


@pytest.mark.parametrize("typ", ["tensor<f32>", "tensor<4xi1>", "tensor<2x3xf64>", "tensor<4xi64>"])
def test_materialize_only_the_user_output(typ):
    module = parse_mlir_text(f"""module {{func.func @forward(%x:{typ}) -> ({typ},{typ}) {{
      return %x,%x : {typ},{typ}
    }} }}""")
    report = dict(cases=[dict(status="bundled", output_indices=[0], semantic_boundary=dict(post_indices=[1]))])
    assert expose_argument_results(module, report) == 1
    module.verify()
    returned = next(op for op in module.walk() if op.name == "func.return")
    function = next(op for op in module.walk() if op.name == "func.func")
    assert returned.operands[1] is function.body.block.args[0]
    assert returned.operands[0].owner.name == "linalg.generic"
    assert returned.operands[0].owner.inputs[0] is function.body.block.args[0]
    assert expose_argument_results(module, report) == 0
