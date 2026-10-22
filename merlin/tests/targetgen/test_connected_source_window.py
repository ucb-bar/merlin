"""Source extraction must preserve literal proofs and every live output."""
from hashlib import sha256

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.common import mlir_query as mq
from merlin.frontends.connected_region import extract_window, text, verify_window, verify_provenance_restoration
from test_bucketize_host_source_body import _screen, _source


def _module(op):
    while op.parent_op() is not None:
        op = op.parent_op()
    return op


def test_literal_and_nested_scalar_closure_preserves_bucketize_source_screen():
    op = _source()
    op.attributes['prov.source_node_ids'] = ArrayAttr([StringAttr('source.bucketize')])
    module = _module(op)
    block = op.parent
    ordinal = tuple(block.ops).index(op)
    result = extract_window(module, ordinal, ordinal)
    clone = next(new for old, new in result.operation_pairs if old is op)
    assert len(result.inputs) == 1  # Literal boundaries, init and nested zero/one stay internal.
    assert clone.attributes['prov.source_node_ids'] == op.attributes['prov.source_node_ids']
    assert _screen(clone)['status'] == _screen(op)['status'] == 'admitted'
    source, projection = text(module), text(result.module)
    checked = verify_window(source, projection, source_sha256=sha256(source.encode()).hexdigest(),
                            projected_sha256=sha256(projection.encode()).hexdigest(),
                            start=ordinal, stop=ordinal)
    assert len(checked.operation_pairs) == len(result.operation_pairs)
    changed = projection.replace('source.bucketize', 'unrelated.bucketize')
    with pytest.raises(ValueError, match='exact source reconstruction'):
        verify_window(source, changed, source_sha256=sha256(source.encode()).hexdigest(),
                      projected_sha256=sha256(changed.encode()).hexdigest(), start=ordinal, stop=ordinal)


def test_unknown_boundary_tensor_stays_input_and_cannot_gain_literal_proof():
    op = _source(dynamic=True)
    ordinal = tuple(op.parent.ops).index(op)
    result = extract_window(_module(op), ordinal, ordinal)
    clone = next(new for old, new in result.operation_pairs if old is op)
    assert len(result.inputs) == 2
    assert _screen(clone)['status'] == 'unsupported'


def test_all_live_outputs_retained_and_cpu_values_preserved():
    import numpy as np
    from merlin.runtime import linalg_numpy
    module = mq.parse('''builtin.module {
      func.func @original(%x: i64, %y: i64) -> (i64, i64) {
        %a = arith.addi %x, %y : i64
        %b = arith.muli %a, %y : i64
        func.return %a, %b : i64, i64
      }
    }''')
    result = extract_window(module, 0, 1)
    assert len(result.outputs) == 2
    args = [np.asarray(3, dtype=np.int64), np.asarray(-2, dtype=np.int64)]
    expected = linalg_numpy.evaluate(module, args)
    actual = linalg_numpy.evaluate(result.module, args)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))
    with pytest.raises(ValueError, match='return or CFG'):
        extract_window(module, 0, 2)


def test_reduction_root_provenance_survives_and_cannot_be_rebound():
    source = mq.parse('''builtin.module {
      func.func @original(%input: tensor<2xi64>) -> tensor<i64> {
        %zero = arith.constant 0 : i64
        %init = tensor.splat %zero : tensor<i64>
        %out = "linalg.reduce"(%input, %init) <{dimensions = array<i64: 0>}> ({
          ^bb0(%x: i64, %acc: i64):
            %sum = arith.addi %x, %acc : i64
            linalg.yield %sum : i64
        }) {prov.source_node_ids = ["source.sum"], prov.aten = "aten.sum.dim_IntList"}
          : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
        func.return %out : tensor<i64>
      }
    }''')
    original = text(source)
    result = extract_window(source, 2, 2)
    projection = text(result.module)
    reduced = next(mq.walk(mq.parse(projection), 'linalg.reduce'))
    assert reduced.attributes['prov.source_node_ids'] == ArrayAttr([StringAttr('source.sum')])
    changed = projection.replace('source.sum', 'another.sum')
    with pytest.raises(ValueError, match='exact source reconstruction'):
        verify_window(original, changed, source_sha256=sha256(original.encode()).hexdigest(),
                      projected_sha256=sha256(changed.encode()).hexdigest(), start=2, stop=2)


def test_restoration_only_adds_missing_provenance_and_preserves_math():
    original = text(mq.parse('''builtin.module {
      func.func @f(%x: i64, %y: i64) -> i64 {
        %a = arith.addi %x, %y : i64
        func.return %a : i64
      }
    }'''))
    module = mq.parse(original)
    op = next(mq.walk(module, 'arith.addi'))
    op.attributes['prov.source_node_ids'] = ArrayAttr([StringAttr('source.add')])
    restored = text(module)
    def prove(a, b):
        return verify_provenance_restoration(a, b, original_sha256=sha256(a.encode()).hexdigest(),
                                            restored_sha256=sha256(b.encode()).hexdigest())
    assert len(prove(original, restored)['changes']) == 1
    with pytest.raises(ValueError, match='typed operations'):
        prove(original, restored.replace('arith.addi', 'arith.muli'))
    with pytest.raises(ValueError, match='existing attribute'):
        prove(restored, restored.replace('source.add', 'source.other'))
    with pytest.raises(ValueError, match='semantic attribute'):
        prove(original, restored.replace('prov.source_node_ids', 'unreviewed.semantic_option'))
