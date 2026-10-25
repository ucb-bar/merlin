"""Static tensor view reuse retains uses, unrelated effects, and exact provenance."""
import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from merlin.common import mlir_query as mq
from merlin.transforms.tensor_views import canonicalize_tensor_views


def source(*, second_input="%a", second_attribute="", effect=True):
    middle = '"test.observe"(%x) : (tensor<6xi8>) -> ()' if effect else ''
    return f'''module {{
      func.func @f(%a: tensor<2x3xi8>, %b: tensor<2x3xi8>) -> (tensor<6xi8>, tensor<6xi8>) {{
        %x = "tensor.collapse_shape"(%a) <{{reassociation = [[0 : i64, 1 : i64]]}}>
          {{prov.source_node_ids = ["original-x"], prov.region_id = "left"}} : (tensor<2x3xi8>) -> tensor<6xi8>
        {middle}
        %y = "tensor.collapse_shape"({second_input}) <{{reassociation = [[0 : i64, 1 : i64]]}}>
          {{prov.source_node_ids = ["original-y"], prov.region_id = "right"{second_attribute}}} : (tensor<2x3xi8>) -> tensor<6xi8>
        func.return %x, %y : tensor<6xi8>, tensor<6xi8>
      }}
    }}'''


def test_merge_preserves_all_consumers_and_original_metadata():
    module = mq.parse(source())
    fn = next(o for o in module.walk() if o.name == 'func.func')
    before = tuple(fn.body.block.ops)
    witness = canonicalize_tensor_views(module)
    after = tuple(fn.body.block.ops)
    assert len(witness) == 1 and len(before) == len(after) + 1
    retained, observer, returned = after
    assert observer is before[1] and returned is before[-1]
    assert observer.operands[0] is retained.results[0]
    assert returned.operands[0] is returned.operands[1] is retained.results[0]
    assert retained.attributes['prov.source_node_ids'] == ArrayAttr([StringAttr('original-x'), StringAttr('original-y')])
    lineage = retained.attributes['prov.view_cse_sources']
    assert [record.data['prov.region_id'].data for record in lineage] == ['left', 'right']
    assert canonicalize_tensor_views(module) == ()


@pytest.mark.parametrize('kwargs', [dict(second_input='%b'), dict(second_attribute=', unknown_semantics = true')])
def test_distinct_inputs_or_unmodeled_metadata_are_not_merged(kwargs):
    module = mq.parse(source(**kwargs))
    assert canonicalize_tensor_views(module) == ()


def test_never_merge_across_blocks_or_functions():
    first = mq.parse(source(effect=False))
    other = mq.parse(source(effect=False))
    other_fn = next(o for o in other.walk() if o.name == 'func.func')
    other_fn.sym_name = StringAttr('g')
    other.body.block.detach_op(other_fn)
    first.body.block.add_op(other_fn)
    assert len(canonicalize_tensor_views(first)) == 2
    functions = [o for o in first.walk() if o.name == 'func.func']
    assert functions[0].body.block.first_op is not functions[1].body.block.first_op


def test_dynamic_views_remain_untouched():
    module = mq.parse(source(effect=False).replace('tensor<2x3xi8>', 'tensor<?x3xi8>').replace('tensor<6xi8>', 'tensor<?xi8>'))
    assert canonicalize_tensor_views(module) == ()


def test_different_reassociation_and_result_shape_are_preserved():
    text = source(effect=False).replace('tensor<2x3xi8>', 'tensor<2x3x4xi8>')
    text = text.replace('tensor<6xi8>', 'tensor<6x4xi8>')
    text = text.replace('[[0 : i64, 1 : i64]]', '[[0 : i64, 1 : i64], [2 : i64]]')
    text = text.replace('%y = "tensor.collapse_shape"(%a) <{reassociation = [[0 : i64, 1 : i64], [2 : i64]]}>',
                        '%y = "tensor.collapse_shape"(%a) <{reassociation = [[0 : i64], [1 : i64, 2 : i64]]}>')
    text = text.replace('prov.region_id = "right"} : (tensor<2x3x4xi8>) -> tensor<6x4xi8>',
                        'prov.region_id = "right"} : (tensor<2x3x4xi8>) -> tensor<2x12xi8>')
    text = text.replace('-> (tensor<6x4xi8>, tensor<6x4xi8>)', '-> (tensor<6x4xi8>, tensor<2x12xi8>)')
    text = text.replace('func.return %x, %y : tensor<6x4xi8>, tensor<6x4xi8>', 'func.return %x, %y : tensor<6x4xi8>, tensor<2x12xi8>')
    assert canonicalize_tensor_views(mq.parse(text)) == ()
