"""Projected index roles with freshly re-proved capture-root and source joins.

The existing closed index recognizers retain their arithmetic and width checks.
This wrapper keeps projection provenance distinct from a full-capture record.
"""
from hashlib import sha256
from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json
from merlin.xdsl_dialects._common import text
from merlin.frontends import prepared_index_source_body as prepared

SCHEMA = 'merlin.connected_index_source_body.v1'


def validate(body):
    if type(body) is not dict or body.get('schema') != SCHEMA:
        raise ValueError('connected index needs an explicit projected source role')
    return prepared.validate_declaration(dict(body, schema=prepared.SCHEMA))


def screen(declaration, row, signature, source_operations, source_context):
    if type(source_context) is not dict or set(source_context) != {
            'selected_index_observation', 'verified_connected_index_source'}:
        return {'status':'unknown','reason':'connected index requires freshly joined source/capture context'}
    record = source_context['verified_connected_index_source']
    if (type(record) is not dict or record.get('schema') != 'merlin.connected_index_source_observation.v1'
            or record.get('status') != 'source_proved_pending_capture_preparation_join_host_and_build'
            or any(not is_sha256(record.get(k)) for k in ('source_sha256','projected_sha256',
                'normalized_projected_sha256','parsed_module_sha256','capture_model_sha256',
                'frontend_trace_sha256','capture_receipt_sha256'))
            or not isinstance(source_operations, tuple) or not source_operations
            or type(record.get('capture_root_joins')) is not list):
        return {'status':'unsupported','reason':'connected index source/capture join is incomplete'}
    module = source_operations[0]
    while module.parent_op() is not None:
        module = module.parent_op()
    parsed = tuple(mq.walk(module))
    ordinals = row.get('ordinals')
    if (module.name != 'builtin.module' or len(parsed) != record.get('n_source_operations')
            or sha256(text(module,generic=True).encode()).hexdigest() != record['parsed_module_sha256']
            or type(ordinals) is not list or len(ordinals) != len(source_operations)
            or any(type(i) is not int or not 0 <= i < len(parsed) or parsed[i] is not op
                   for i,op in zip(ordinals,source_operations,strict=True))):
        return {'status':'unsupported','reason':'connected index parsed module or exact occurrence differs'}
    for ordinal in ordinals:
        joins = [r for r in record['capture_root_joins'] if isinstance(r,dict) and r.get('projected_ordinal') == ordinal]
        if (len(joins) != 1 or not is_sha256(joins[0].get('typed_body_sha256'))
                or type(joins[0].get('capture_ordinal')) is not int or joins[0]['capture_ordinal'] < 0
                or prepared._node_ids(parsed[ordinal]) != (joins[0].get('source_node_id'),)):
            return {'status':'unsupported','reason':'connected index occurrence lacks one captured typed root'}
    # Reuse only the strict structural role verifier, not a saved admission.
    adapted = dict(schema=prepared.PROOF_SCHEMA, status=prepared.PROOF_STATUS, scope=prepared.PROOF_SCOPE,
        index_bits_premise=prepared._selected_width(source_context['selected_index_observation']),
        raw_source_sha256=record['projected_sha256'], normalized_source_sha256=record['normalized_projected_sha256'],
        frontend_trace_sha256=record['frontend_trace_sha256'], capture_receipt_sha256=record['capture_receipt_sha256'],
        n_source_operations=len(parsed), original_to_prepared_equivalence='not_proved', records=record['records'])
    result = prepared.screen_source_body(dict(declaration,source_body=dict(declaration['source_body'],schema=prepared.SCHEMA)),
        row,signature,source_operations,dict(selected_index_observation=source_context['selected_index_observation'],
                                           verified_index_source=adapted))
    if result['status'] == 'admitted':
        result['proof'].update(schema=SCHEMA, source_record_sha256=sha256(canonical_json(record)).hexdigest(),
            scope='reconstructed cut and exact captured typed roots; linked host, complete outputs and execution remain required')
    return result
