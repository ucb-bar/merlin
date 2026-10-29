"""Prospective pure-copy geometry is complete, bounded and explicitly selected."""
import pytest
from merlin_experiments.phase0 import connected_integer_bounds as B
from merlin_experiments.phase0 import public_programs as P
from merlin_experiments.phase0.connected_integer_affine_copy import PROTOCOL
from test_connected_integer_bounds import program

from merlin.common import mlir_query as mq


def affine_program(expression='n*2+1'):
    text=program().replace('%x: tensor<2x3xi8>', '%source: tensor<2x7xi8>')
    return text.replace('%z = arith.constant',f'''%gather_init = tensor.empty() : tensor<2x3xi8>
      %x = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,{expression})>,
          affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]}}
          ins(%source : tensor<2x7xi8>) outs(%gather_init : tensor<2x3xi8>) {{
          ^bb9(%source_cell: i8, %unused_cell: i8):
            linalg.yield %source_cell : i8
          }} -> tensor<2x3xi8>
      %z = arith.constant''')


def test_default_refusal_is_preserved_and_prospective_copy_proves_complete_geometry():
    with pytest.raises(ValueError,match='projected permutation'):
        B.derive(mq.parse(affine_program()),[(-3,3)]*2)
    proof=B.derive(mq.parse(affine_program()),[(-3,3)]*2,affine_copy_protocol=PROTOCOL)
    assert proof['schema']=='merlin.connected_integer_source_bounds.v2'
    copy=next(row for row in proof['operations'] if row.get('kind')=='bounded_affine_copy')
    assert copy['input_access_intervals']==[[0,1],[1,5]] and copy['interval']==[-3,3]
    assert proof['output_intervals']==[[-27,27],[-3,3]]
    assert B.derive(mq.parse(program()),[(-3,3)]*2)['schema']==B.SCHEMA


@pytest.mark.parametrize('expression',['n*3+1','n*(-1)+2','n floordiv 2','n mod 2'])
def test_out_of_bounds_or_unselected_affine_operators_refuse(expression):
    with pytest.raises(Exception):
        B.derive(mq.parse(affine_program(expression)),[(-3,3)]*2,affine_copy_protocol=PROTOCOL)


@pytest.mark.parametrize('old,new',[
 ('linalg.yield %source_cell : i8','linalg.yield %unused_cell : i8'),
 ('affine_map<(m,n)->(m,n)>], iterator_types','affine_map<(m,n)->(n,m)>], iterator_types'),
 ('iterator_types = ["parallel","parallel"]','iterator_types = ["parallel","reduction"]'),
])
def test_noncopy_bodies_partial_outputs_and_reductions_cannot_use_copy_protocol(old,new):
    with pytest.raises(Exception):
        B.derive(mq.parse(affine_program().replace(old,new)),[(-3,3)]*2,affine_copy_protocol=PROTOCOL)


def test_unknown_protocol_and_silent_old_policy_expansion_refuse():
    with pytest.raises(ValueError,match='protocol'):
        B.derive(mq.parse(program()),[(-3,3)]*2,affine_copy_protocol='unknown')
    policy=dict(schema=P.AFFINE_INTEGER_POLICY_SCHEMA,selection='declared_performance_applications',
                source_mode='native_integer_v1',source_integer_proof='checked_native_integer_affine_copy_v2',
                max_members=9,max_capture_bytes=512<<20,max_input_elements=16<<20,max_output_elements=16<<20)
    assert P.validate_selection(policy)==policy
    assert P.capture_schema(policy)==P.INTEGER_CAPTURE_SCHEMA
    with pytest.raises(ValueError):
        P.validate_selection(dict(policy,schema=P.INTEGER_POLICY_SCHEMA))
    with pytest.raises(ValueError):
        P.validate_selection(dict(policy,source_integer_proof='unknown'))
