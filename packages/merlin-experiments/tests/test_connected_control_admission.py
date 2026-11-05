"""Exact projected control proofs admit only the scalar roles they establish."""
from copy import deepcopy
from hashlib import sha256

import pytest
from test_private_control_support import _source_scatter
from test_connected_source_context import _selected
from merlin.common import mlir_query as mq
from merlin.frontends.connected_region import extract_window, text
from merlin.frontends import bounded_control_source_body as body
from merlin.targetgen.application_inventory import operation_structure
from merlin_experiments.phase0.connected_source_context import ConnectedSourceContext


def _fixture(tmp_path, source_text=None):
    raw = source_text or _source_scatter()
    module = mq.parse(raw)
    function = next(op for op in mq.walk(module) if op.name == 'func.func')
    stop = len(tuple(function.body.block.ops)) - 2
    source, cut = tmp_path/'source.mlir', tmp_path/'cut.mlir'
    source.write_text(raw)
    cut.write_text(text(extract_window(module, 0, stop).module))
    context = ConnectedSourceContext(source, cut, source_sha256=sha256(source.read_bytes()).hexdigest(),
        projected_sha256=sha256(cut.read_bytes()).hexdigest(), start=0, stop=stop,
        selected_index_observation=_selected())
    proof = context.control_admission_context(cut, target='fixture')
    parsed = tuple(mq.walk(mq.parse(cut.read_text())))
    return source, cut, context, proof, parsed


def _screen(proof, parsed, item):
    ordinal, role = item['ordinal'], item['role']
    op = parsed[ordinal]
    structure = operation_structure(op)
    row = dict(mlir_operation=op.name, ordinals=[ordinal], count=1,
               ordered_operand_types=structure['ordered_operand_types'],
               ordered_result_types=structure['ordered_result_types'])
    declaration = dict(id='fixture', source_body=dict(schema=body.SCHEMA, operation=role))
    signature = dict(ordered_operand_dtypes=body.ROLES[role][1], ordered_result_dtypes=body.ROLES[role][2])
    return body.screen(declaration, row, signature, (op,), proof)


def test_actual_bounded_projection_admits_four_roles(tmp_path):
    _, _, _, proof, parsed = _fixture(tmp_path)
    roles = proof['verified_bounded_control_source']['roles']
    assert {r['role'] for r in roles} == set(body.ROLES)
    assert all(_screen(proof, parsed, r)['status'] == 'admitted' for r in roles)


def test_changed_source_and_foreign_or_missing_role_refuse(tmp_path):
    source, cut, context, proof, parsed = _fixture(tmp_path)
    role = proof['verified_bounded_control_source']['roles'][0]
    for key in ['parsed_module_sha256', 'index_bits']:
        changed = deepcopy(proof)
        changed['verified_bounded_control_source'][key] = '0'*64 if key.endswith('sha256') else 32
        assert _screen(changed, parsed, role)['status'] == 'unsupported'
    changed = deepcopy(proof)
    changed['verified_bounded_control_source']['roles'] = []
    assert _screen(changed, parsed, role)['status'] == 'unsupported'
    assert _screen({}, parsed, role)['status'] == 'unknown'
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError, match='bytes changed'):
        context.control_admission_context(cut, target='fixture')


def test_false_guard_cannot_issue_control_roles(tmp_path):
    # The source proof must reject a changed bound; a provenance tag is insufficient.
    source = _source_scatter().replace('value = 4 : index', 'value = 3 : index')
    _, _, _, proof, _ = _fixture(tmp_path, source)
    assert proof['verified_bounded_control_source']['roles'] == []
