"""Declared operand roles supply observations, never hardware capability."""
import pytest

from merlin.common import mlir_query as mq
from merlin.common.quant_formats import get
from merlin.targetgen import capsule_source as source
from merlin.targetgen import source_demand_inventory as inventory
from merlin.targetgen.eligibility import RegionDescriptor, is_eligible
from merlin.xdsl_dialects._common import text


def quantizer(name='quant_ext.quantize_per_tensor', *, data='tensor<2x3xf32>',
              result='tensor<2x3xi8>', scale='tensor<f32>', zero='tensor<i64>'):
    return mq.parse(f'''module {{func.func @forward(%x:{data},%s:{scale},%z:{zero}) -> {result} {{
      %r="{name}"(%x,%s,%z) {{prov.op="quantize",prov.region_id="q"}} : ({data},{scale},{zero})->{result}
      return %r:{result}
    }}}}''')


def test_quantizer_data_role_excludes_scale_and_zero_point_metadata():
    demands=source.model_op_demands_checked(text(quantizer(),generic=True),'i8')
    assert len(demands)==1
    demand=demands[0]
    assert demand.captured_input_formats==(get('f32').name,)
    assert demand.admission_input_fmt==get('f32').name
    assert demand.in_fmt=='i8'  # The requested routing format is not evidence.
    verdict=is_eligible(RegionDescriptor(op=demand.op,in_dtype=demand.admission_input_fmt),{})
    assert not verdict.eligible and not verdict.undetermined


def test_dequantizer_observes_its_actual_integer_data_role():
    module=quantizer('quant_ext.dequantize_per_tensor',data='tensor<2x3xi8>',result='tensor<2x3xf32>')
    demand=source.model_op_demands_checked(text(module,generic=True),'f32')[0]
    assert demand.captured_input_formats==(get('i8').name,)
    assert demand.in_fmt=='f32'


@pytest.mark.parametrize('parameters',[
    {'name':'quant_ext.undeclared_conversion'},
    {'data':'tensor<?x3xf32>','result':'tensor<?x3xi8>'},
    {'scale':'tensor<1xf32>'},
    {'zero':'tensor<i32>'},
    {'result':'tensor<6xi8>'},
    {'result':'tensor<2x3xf32>'},
])
def test_undeclared_dynamic_or_malformed_quantizer_stays_unknown(parameters):
    module=quantizer(**parameters)
    operation=next(o for o in module.walk()if mq.attr_str(o,'prov.region_id')=='q')
    assert inventory._declared_data_operand_format(operation) is None
    demand=source.model_op_demands_checked(text(module,generic=True),'i8')[0]
    assert demand.captured_input_formats==()
    assert not demand.source_formats_complete


def test_view_observation_retains_floating_format_instead_of_compile_dtype():
    module=mq.parse('''module {func.func @forward(%x:tensor<2x3xf32>)->tensor<6xf32> {
      %r=tensor.collapse_shape %x [[0,1]] {prov.op="view",prov.region_id="v"}:tensor<2x3xf32> into tensor<6xf32>
      return %r:tensor<6xf32>
    }}''')
    demand=source.model_op_demands_checked(text(module,generic=True),'i8')[0]
    assert demand.captured_input_formats==(get('f32').name,)


def test_dtype_changing_view_cannot_supply_a_format_observation():
    module=mq.parse('''module {func.func @forward(%x:tensor<2x3xf32>)->tensor<2x3xi8> {
      %r="tensor.cast"(%x) {prov.op="view",prov.region_id="v"}:(tensor<2x3xf32>)->tensor<2x3xi8>
      return %r:tensor<2x3xi8>
    }}''')
    operation=next(o for o in module.walk()if mq.attr_str(o,'prov.region_id')=='v')
    assert inventory._declared_data_operand_format(operation) is None
