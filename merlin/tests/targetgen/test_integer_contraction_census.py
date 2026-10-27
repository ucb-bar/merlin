"""A complete contraction union is proved by provenance, not matching totals."""
import copy

import pytest

from merlin.common import mlir_query as query
from merlin.targetgen.integerization_export import (
    admit_integerization_census,
    census_export_complete,
    contraction_census,
    receipt_export_complete,
)


def _fixture(*, seeded=True, generic=False):
    # Static attention and seeded integerizer calls share the same MLIR form.
    ops = []
    nodes = []
    for index, (target, transform) in enumerate([
        ('aten._int_mm.default', 'pt2e_integerization'),
        ('m2m_q.int_mm_seeded.default', 'pt2e_integerization'),
        ('aten._int_mm.default', 'static_attention'),
    ]):
        if not seeded and target == 'm2m_q.int_mm_seeded.default':
            target = 'aten._int_mm.default'
        identity = f'g:prepared:root:n{index}'
        ops.append(f'''%z{index} = tensor.empty() : tensor<2x2xi32>
        %r{index} = linalg.matmul {{prov.source_node_ids = ["{identity}"]}}
          ins(%a, %b : tensor<2x4xi8>, tensor<4x2xi8>)
          outs(%z{index} : tensor<2x2xi32>) -> tensor<2x2xi32>''')
        if generic:
            ops[-1] = f'''%z{index} = tensor.empty() : tensor<2x2xi32>
            %r{index} = linalg.generic {{indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel", "reduction"]}}
              ins(%a, %b : tensor<2x4xi8>, tensor<4x2xi8>)
              outs(%z{index} : tensor<2x2xi32>)
              attrs = {{prov.op = "int_matmul", prov.source_node_ids = ["{identity}"]}} {{
              ^bb0(%x: i8, %w: i8, %acc: i32):
                %xx = arith.extsi %x : i8 to i32
                %ww = arith.extsi %w : i8 to i32
                %p = arith.muli %xx, %ww : i32
                %s = arith.addi %acc, %p : i32
                linalg.yield %s : i32
              }} -> tensor<2x2xi32>'''
        nodes.append({'id': identity, 'target': target, 'transformation': transform,
                      'results': [{'dtype': 'int32', 'shape': [2, 2]}],
                      'origin_node_ids': [f'g:original:root:n{index}']})
    program = '''builtin.module {
      func.func @forward(%a: tensor<2x4xi8>, %b: tensor<4x2xi8>) -> tensor<2x2xi32> {
      ''' + '\n'.join(ops) + '''
      func.return %r2 : tensor<2x2xi32>
    }}'''
    module = query.parse(program)
    receipt = {'integer_mm_emitted': 2, 'integer_bmm_emitted': 0,
               'exported_integer_mm_count': 3, 'exported_integer_bmm_count': 0,
               'accumulator_bound_checked': True, 'golden_agreement': {'status': 'passed'}}
    prepared = {'nodes': nodes}
    receipt['integer_contraction_census'] = contraction_census(module, prepared, 1)
    return module, prepared, receipt


def _evidence():
    """A valid saved observation for the fixture's single static-attention call."""
    return {
        'program_sha256': 'a' * 64,
        'dynamic_evidence': {
            'schema': 'merlin.dynamic_integerization.v1',
            'recipe_sha256': 'b' * 64, 'program_sha256': 'a' * 64,
            'trace': {'sha256': 'c' * 64, 'bytes': 1},
            'families': ['activation_contraction'], 'samples': 1,
            'site_agreement': {'schema': 'dynamic_site_agreement_v1', 'cosine_min': 0.99, 'rel_rmse_max': 0.1},
            'weight_contractions': 2,
            'dynamic_owners': [{'lineage': ['g:original:root:n2'], 'calls': 1}],
            'contractions': {'rewritten': 1, 'kept_in_float': [], 'device_calls': 1, 'sites': [{
                'status': 'rewritten', 'family': 'activation_contraction', 'device_calls': 1,
                'shapes': [[2, 4], [4, 2]], 'lineage': ['g:original:root:n2'],
                'verification': {'schema': 'm2m.dynamic_site_verification.v1', 'samples': 1,
                                 'finite': True, 'cosine_min': 1.0, 'cosine_mean': 1.0,
                                 'max_abs_error': 0.0, 'rel_rmse_max': 0.0},
            }]},
        },
    }


def test_integerizer_attention_and_seeded_union_reconciles():
    module, _, receipt = _fixture()
    assert census_export_complete(receipt, module, **_evidence())
    assert receipt_export_complete(receipt, {}, **_evidence())
    rows = receipt['integer_contraction_census']['calls']
    assert [r['provenance'] for r in rows] == ['pt2e_integerization', 'seeded', 'static_attention']


@pytest.mark.parametrize('damage', ['extra', 'missing', 'duplicate', 'wrong_owner', 'no_origin',
                                   'wrong_rank', 'static_count', 'emitted_count', 'export_count',
                                   'seeded_double_count'])
def test_incomplete_or_misattributed_union_refused(damage):
    module, prepared, receipt = _fixture()
    census = receipt['integer_contraction_census']
    if damage == 'extra':
        census['calls'].pop()
    elif damage == 'missing':
        census['export_bindings'].pop()
    elif damage == 'duplicate':
        census['export_bindings'][1] = copy.deepcopy(census['export_bindings'][0])
    elif damage == 'wrong_owner':
        census['calls'][0]['provenance'] = 'unknown'
    elif damage == 'no_origin':
        census['calls'][0]['origin_node_ids'] = []
    elif damage == 'wrong_rank':
        census['calls'][0]['rank'] = 3
    elif damage == 'static_count':
        census['static_attention_device_calls'] += 1
    elif damage == 'emitted_count':
        receipt['integer_mm_emitted'] -= 1
    elif damage == 'export_count':
        receipt['exported_integer_mm_count'] += 1
    else:
        receipt['integer_mm_emitted'] += 1
    assert not census_export_complete(receipt, **_evidence())
    assert not receipt_export_complete(receipt, {}, **_evidence())


def test_matching_totals_cannot_substitute_source_identity():
    module, _, receipt = _fixture()
    op = next(o for o in module.walk() if 'prov.source_node_ids' in o.attributes)
    from xdsl.dialects.builtin import ArrayAttr, StringAttr
    op.attributes['prov.source_node_ids'] = ArrayAttr([StringAttr('unaccounted')])
    assert not census_export_complete(receipt, module, **_evidence())


def test_unknown_prepared_producer_refused():
    module, prepared, receipt = _fixture()
    prepared['nodes'][0]['transformation'] = 'unknown'
    receipt['integer_contraction_census'] = contraction_census(module, prepared, 1)
    assert not census_export_complete(receipt, **_evidence())


def test_malformed_census_refused():
    _, _, receipt = _fixture()
    del receipt['integer_contraction_census']['calls'][0]['source_node_id']
    assert not census_export_complete(receipt, **_evidence())


@pytest.mark.parametrize('origins', [
    None, [], [None], [''], 'bogus', ['bogus'], ['g:original:root:n0', None],
    ['g:original:root'], ['g::root:n0'], ['g:original::n0'],
    ['g:original:root:n-1'], ['g:original:root:n1:v0'],
    ['g:original:root:n01'], ['g:original:root:n١'],
])
def test_malformed_origin_identifiers_refused(origins):
    module, _, receipt = _fixture()
    receipt['integer_contraction_census']['calls'][0]['origin_node_ids'] = origins
    assert not census_export_complete(receipt, **_evidence())
    assert not census_export_complete(receipt, module, **_evidence())
    assert not receipt_export_complete(receipt, {}, **_evidence())


def test_nested_and_intermediate_stage_origins_are_valid():
    module, _, receipt = _fixture()
    receipt['integer_contraction_census']['calls'][0]['origin_node_ids'] = [
        'g:original:submod_1.submod_2:n123', 'g:quantization_input:root:n0',
    ]
    assert census_export_complete(receipt, module, **_evidence())


def _inventory_fixture(tmp_path, *, seeded=True, generic=True):
    """A byte-bound capture of ordinary, seeded and static-attention contractions."""
    import hashlib
    import json

    from merlin.targetgen.application_inventory import verify_capture_receipt
    from merlin.targetgen.quant_recipe import digest

    module, prepared, receipt = _fixture(seeded=seeded, generic=generic)
    program = str(module).encode()
    trace = {
        'graphs': {'prepared': prepared},
        'mlir': {
            'sha256': hashlib.sha256(program).hexdigest(),
            'operations': [
                {'operation': query.op_name(op),
                 'operand_types': [str(value.type) for value in op.operands],
                 'result_types': [str(value.type) for value in op.results],
                 'source_node_ids': [node['id']]}
                for op, node in zip(
                    query.walk(module, 'linalg.matmul', 'linalg.generic'), prepared['nodes'], strict=True
                )
            ],
        },
    }
    bar = {'schema': 'dynamic_site_agreement_v1', 'cosine_min': 0.99, 'rel_rmse_max': 0.1}
    recipe = {
        'schema': 'quant_recipe_v1', 'status': 'derived',
        'dynamic': {'schema': 'dynamic_contractions_v1', 'families': ['activation_contraction'],
                    'scales': 'static_heads', 'site_agreement': bar},
    }
    recipe['recipe_sha256'] = digest(recipe)
    receipt.update(
        schema='m2m.pt2e-integerize.v1', quantized_contractions_seen=2,
        quantized_contractions_integerized=2, quantized_contractions_remaining=0,
        refusals=[], max_reduction_k=4,
        quantized_by_kind={kind: {'seen': 2 if kind == 'linear' else 0,
                                  'integerized': 2 if kind == 'linear' else 0, 'remaining': 0}
                           for kind in ('linear', 'conv2d', 'matmul', 'unsupported')},
    )
    receipt['golden_agreement'].update(
        finite=True, samples=1, max_abs=0.0, max_rel=0.0, atol=0.0, rtol=0.0,
        outputs=[{'finite': True, 'within_tolerance': True,
                  'max_abs': 0.0, 'max_rel': 0.0, 'atol': 0.0, 'rtol': 0.0}],
    )
    metadata = {
        'scheme': 'int8_static_act_int8_weight', 'integerization_receipt': receipt,
        'recipe': recipe, 'recipe_sha256': recipe['recipe_sha256'],
        'recipe_agreement': {'samples': 1},
        'quantization_stats': {'annotated_contractions': 2, 'dynamic_contractions': {
            'rewritten': 1, 'kept_in_float': [], 'device_calls': 1,
            'sites': [{
                'status': 'rewritten', 'family': 'activation_contraction', 'device_calls': 1,
                'shapes': [[2, 4], [4, 2]], 'lineage': prepared['nodes'][2]['origin_node_ids'],
                'verification': {'schema': 'm2m.dynamic_site_verification.v1', 'samples': 1,
                                 'finite': True, 'cosine_min': 1.0, 'cosine_mean': 1.0,
                                 'max_abs_error': 0.0, 'rel_rmse_max': 0.0},
            }],
        }},
    }
    (tmp_path / 'model.mlir').write_bytes(program)
    (tmp_path / 'frontend-trace.json').write_text(json.dumps(trace))
    metadata['frontend_trace'] = {'sha256': hashlib.sha256((tmp_path / 'frontend-trace.json').read_bytes()).hexdigest()}
    (tmp_path / 'meta.json').write_text(json.dumps(metadata))
    (tmp_path / 'weights.safetensors').write_bytes(b'fixture weights')
    (tmp_path / 'weights.safetensors.manifest.json').write_text('{}')
    artifacts = {
        path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        for path in tmp_path.iterdir()
    }
    (tmp_path / 'capture_receipt.json').write_text(json.dumps({
        'schema': 'm2m.capture-receipt.v1', 'materialized_abi': {'complete': True}, 'artifacts': artifacts,
    }))
    return verify_capture_receipt(tmp_path / 'model.mlir'), trace


@pytest.mark.parametrize('generic', [False, True])
def test_inventory_admits_ordinary_seeded_and_static_attention_union(tmp_path, generic):
    from merlin.targetgen.application_inventory import verified_static_integerization

    observation, _ = _inventory_fixture(tmp_path, generic=generic)
    assert observation['status'] == 'verified_materialized', observation
    assert observation['source_closure_verified'] is False
    projection = observation['capture_integerization']
    assert verified_static_integerization(projection)
    assert projection['dynamic_integerization']['weight_contractions'] == 2
    assert projection['dynamic_integerization']['dynamic_owners'] == [
        {'lineage': ['g:original:root:n2'], 'calls': 1},
    ]


@pytest.mark.parametrize('dynamic_stats', ['absent', 'null'])
def test_inventory_refuses_attention_without_dynamic_evidence(tmp_path, dynamic_stats):
    import hashlib
    import json

    from merlin.targetgen.application_inventory import (
        verified_static_integerization,
        verify_capture_receipt,
    )

    observation, _ = _inventory_fixture(tmp_path)
    assert verified_static_integerization(observation['capture_integerization'])
    metadata_path = tmp_path / 'meta.json'
    metadata = json.loads(metadata_path.read_text())
    if dynamic_stats == 'absent':
        del metadata['quantization_stats']['dynamic_contractions']
    else:
        metadata['quantization_stats']['dynamic_contractions'] = None
    metadata_path.write_text(json.dumps(metadata))
    receipt_path = tmp_path / 'capture_receipt.json'
    capture_receipt = json.loads(receipt_path.read_text())
    raw = metadata_path.read_bytes()
    capture_receipt['artifacts']['meta.json'] = {
        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
    }
    receipt_path.write_text(json.dumps(capture_receipt))

    observation = verify_capture_receipt(tmp_path / 'model.mlir')
    assert observation['status'] == 'verified_materialized', observation
    projection = observation['capture_integerization']
    receipt = projection['integerization_receipt']
    assert 'dynamic_integerization' not in projection
    assert receipt['integer_mm_emitted'] == 2
    assert receipt['exported_integer_mm_count'] == 3
    verdict = admit_integerization_census(receipt)
    assert verdict.structure_valid and verdict.counts_agree and verdict.dynamic_required
    assert not verdict.accepted
    assert not census_export_complete(receipt)
    assert not verified_static_integerization(projection)


@pytest.mark.parametrize('damage', ['absent', 'null', 'empty', 'wrong_program', 'failed_site'])
def test_saved_attention_projection_requires_valid_dynamic_evidence(tmp_path, damage):
    from merlin.targetgen.application_inventory import verified_static_integerization

    observation, _ = _inventory_fixture(tmp_path)
    projection = observation['capture_integerization']
    assert verified_static_integerization(projection)
    if damage == 'absent':
        del projection['dynamic_integerization']
    elif damage == 'null':
        projection['dynamic_integerization'] = None
    elif damage == 'empty':
        projection['dynamic_integerization'] = {}
    elif damage == 'wrong_program':
        projection['dynamic_integerization']['program_sha256'] = '0' * 64
    else:
        projection['dynamic_integerization']['contractions']['sites'][0]['verification']['finite'] = False
    assert not verified_static_integerization(projection)


@pytest.mark.parametrize('with_census', [False, True])
def test_inventory_admits_equal_integerizer_exports_without_dynamic_evidence(tmp_path, with_census):
    from merlin.targetgen.application_inventory import verified_static_integerization

    observation, _ = _inventory_fixture(tmp_path)
    projection = observation['capture_integerization']
    del projection['dynamic_integerization']
    receipt = projection['integerization_receipt']
    receipt.update(integer_mm_emitted=3, quantized_contractions_seen=3,
                   quantized_contractions_integerized=3)
    census = receipt['integer_contraction_census']
    census['calls'][2]['provenance'] = 'pt2e_integerization'
    census['static_attention_device_calls'] = 0
    if not with_census:
        del receipt['integer_contraction_census']
    assert census_export_complete(receipt)
    assert verified_static_integerization(projection)
    receipt['exported_integer_mm_count'] += 1
    assert not verified_static_integerization(projection)


@pytest.mark.parametrize('damage', [
    'duplicate_ordinal', 'null_ordinal', 'negative_ordinal', 'bool_ordinal',
    'float_ordinal', 'string_ordinal', 'missing_ordinal',
    'string_source_ids', 'null_source_ids', 'nonstring_source_id',
])
def test_saved_binding_structure_refused_without_module(tmp_path, damage):
    from merlin.targetgen.application_inventory import verified_static_integerization

    observation, _ = _inventory_fixture(tmp_path)
    projection = observation['capture_integerization']
    receipt = projection['integerization_receipt']
    assert census_export_complete(receipt, **_evidence())
    assert verified_static_integerization(projection)
    bindings = receipt['integer_contraction_census']['export_bindings']
    binding = bindings[0]
    if damage == 'duplicate_ordinal':
        binding['ordinal'] = bindings[1]['ordinal']
    elif damage == 'missing_ordinal':
        del binding['ordinal']
    elif damage.endswith('_ordinal'):
        binding['ordinal'] = {
            'null_ordinal': None, 'negative_ordinal': -1, 'bool_ordinal': True,
            'float_ordinal': float(binding['ordinal']), 'string_ordinal': str(binding['ordinal']),
        }[damage]
    elif damage == 'string_source_ids':
        binding['source_node_ids'] = binding['source_node_ids'][0]
    elif damage == 'null_source_ids':
        binding['source_node_ids'] = None
    else:
        binding['source_node_ids'].append(None)
    assert not census_export_complete(receipt, **_evidence())
    assert not verified_static_integerization(projection)


@pytest.mark.parametrize('damage', ['missing_call', 'no_origins', 'wrong_rank', 'static_count', 'null_census'])
def test_dynamic_admission_independently_validates_present_census(tmp_path, damage):
    from merlin.capture.integerization import dynamic_integerization_count
    from merlin.targetgen.application_inventory import verified_static_integerization

    observation, _ = _inventory_fixture(tmp_path, seeded=False)
    projection = observation['capture_integerization']
    assert verified_static_integerization(projection)
    receipt = projection['integerization_receipt']
    census = receipt['integer_contraction_census']
    if damage == 'missing_call':
        census['calls'].pop()
    elif damage == 'no_origins':
        census['calls'][0]['origin_node_ids'] = []
    elif damage == 'wrong_rank':
        census['calls'][0]['rank'] += 1
    elif damage == 'static_count':
        census['static_attention_device_calls'] += 1
    else:
        receipt['integer_contraction_census'] = None
    assert dynamic_integerization_count(
        projection['dynamic_integerization'], emitted=2, exported=3,
        program_sha256=projection['capture']['sha256'],
    ) == 1
    assert not census_export_complete(receipt, **_evidence())
    assert not verified_static_integerization(projection)


@pytest.mark.parametrize('damage', ['missing', 'duplicate_id', 'duplicate_export', 'unknown_transform', 'wrong_owner'])
def test_seeded_traced_owners_still_fail_closed(tmp_path, damage):
    from merlin.capture.integerization import traced_integer_owners

    _, trace = _inventory_fixture(tmp_path)
    nodes = trace['graphs']['prepared']['nodes']
    operations = trace['mlir']['operations']
    if damage == 'missing':
        operations.pop(1)
    elif damage == 'duplicate_id':
        nodes[1]['id'] = nodes[0]['id']
    elif damage == 'duplicate_export':
        operations.append(copy.deepcopy(operations[1]))
    elif damage == 'unknown_transform':
        nodes[1]['transformation'] = 'unknown'
    else:
        operations[1]['source_node_ids'] = [nodes[0]['id']]
    with pytest.raises(ValueError):
        traced_integer_owners(trace, trace['mlir']['sha256'])


def test_custom_entry_name_is_used_for_worker_counts_and_census():
    from xdsl.dialects.builtin import StringAttr

    from merlin.targetgen._m2m_capture_worker import _exported_integer_mm_count
    from merlin.targetgen.integerization_export import exported_integer_counts

    module, prepared, receipt = _fixture()
    entry = next(query.walk(module, 'func.func'))
    entry.properties['sym_name'] = StringAttr('custom')
    assert _exported_integer_mm_count(module, func_name='custom') == 3
    assert _exported_integer_mm_count(module, batched=True, func_name='custom') == 0
    assert exported_integer_counts(str(module), func_name='custom') == (3, 0)
    receipt['integer_contraction_census'] = contraction_census(module, prepared, 1, func_name='custom')
    assert census_export_complete(receipt, module, func_name='custom', **_evidence())
    assert receipt_export_complete(receipt, {}, **_evidence())
    assert not census_export_complete(receipt, module, func_name='missing', **_evidence())


def test_present_null_census_does_not_use_legacy_count_equality():
    receipt = {'integer_mm_emitted': 1, 'exported_integer_mm_count': 1}
    assert census_export_complete(receipt)
    receipt['integer_contraction_census'] = None
    assert not census_export_complete(receipt)


def _refresh_capture(tmp_path, metadata, trace, program):
    """Refresh every byte identity, so refusals test evidence rather than stale digests."""
    import hashlib
    import json

    (tmp_path / 'model.mlir').write_text(program)
    trace['mlir']['sha256'] = hashlib.sha256(program.encode()).hexdigest()
    raw_trace = json.dumps(trace).encode()
    (tmp_path / 'frontend-trace.json').write_bytes(raw_trace)
    metadata['frontend_trace'] = {'sha256': hashlib.sha256(raw_trace).hexdigest()}
    (tmp_path / 'meta.json').write_text(json.dumps(metadata))
    path = tmp_path / 'capture_receipt.json'
    receipt = json.loads(path.read_text())
    for name in receipt['artifacts']:
        raw = (tmp_path / name).read_bytes()
        receipt['artifacts'][name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    path.write_text(json.dumps(receipt))
    return raw_trace, trace['mlir']['sha256']


def _matrix_damage(damage, metadata, trace):
    receipt = metadata['integerization_receipt']
    census = receipt['integer_contraction_census']
    bindings = census['export_bindings']
    binding = bindings[0]
    if damage in ('contraction_free_zero', 'contraction_free_empty_dynamic'):
        from merlin.capture.integerization import NON_CONTRACTION_ANNOTATIONS

        receipt.update(integer_mm_emitted=0, exported_integer_mm_count=0,
                       quantized_contractions_seen=0, quantized_contractions_integerized=0,
                       integer_program='non_contraction', non_contraction_annotations={
                           key: int(key == 'annotated_affine_maps') for key in NON_CONTRACTION_ANNOTATIONS})
        receipt['quantized_by_kind']['linear'].update(seen=0, integerized=0)
        metadata['quantization_stats'].update(annotated_contractions=0, **receipt['non_contraction_annotations'])
        census.update(calls=[], export_bindings=[], static_attention_device_calls=0)
        trace['graphs']['prepared']['nodes'] = []
        trace['mlir']['operations'] = []
        if damage == 'contraction_free_zero':
            metadata['quantization_stats'].pop('dynamic_contractions')
        else:
            metadata['quantization_stats']['dynamic_contractions'].update(
                rewritten=0, device_calls=0, sites=[], host_elements=0, int8_macs=0)
    elif damage in ('integerizer_only', 'legacy_equal', 'legacy_no_batch'):
        receipt.update(integer_mm_emitted=3, quantized_contractions_seen=3, quantized_contractions_integerized=3)
        receipt['quantized_by_kind']['linear'].update(seen=3, integerized=3)
        metadata['quantization_stats'].update(annotated_contractions=3)
        metadata['quantization_stats'].pop('dynamic_contractions')
        census['static_attention_device_calls'] = 0
        for call, node in zip(census['calls'], trace['graphs']['prepared']['nodes'], strict=True):
            call['provenance'] = 'pt2e_integerization'
            node['transformation'] = 'pt2e_integerization'
        if damage.startswith('legacy_'):
            receipt.pop('integer_contraction_census')
        if damage == 'legacy_no_batch':
            receipt.pop('integer_bmm_emitted')
            receipt.pop('exported_integer_bmm_count')
    elif damage.startswith('static_only_'):
        from merlin.capture.integerization import NON_CONTRACTION_ANNOTATIONS

        receipt.update(integer_mm_emitted=0, quantized_contractions_seen=0, quantized_contractions_integerized=0,
                       integer_program='non_contraction', non_contraction_annotations={
                           key: int(key == 'annotated_affine_maps') for key in NON_CONTRACTION_ANNOTATIONS})
        receipt['quantized_by_kind']['linear'].update(seen=0, integerized=0)
        metadata['quantization_stats'].update(annotated_contractions=0, **receipt['non_contraction_annotations'])
        census['static_attention_device_calls'] = 3
        sites = []
        for call, node in zip(census['calls'], trace['graphs']['prepared']['nodes'], strict=True):
            call['provenance'] = node['transformation'] = 'static_attention'
            site = copy.deepcopy(metadata['quantization_stats']['dynamic_contractions']['sites'][0])
            site['lineage'] = call['origin_node_ids']
            sites.append(site)
        metadata['quantization_stats']['dynamic_contractions'].update(rewritten=3, device_calls=3, sites=sites)
        if damage == 'static_only_underexport':
            receipt['exported_integer_mm_count'] = 2
            bindings.pop()
        elif damage == 'static_only_no_dynamic':
            metadata['quantization_stats'].pop('dynamic_contractions')
        elif damage == 'static_only_null_census':
            receipt['integer_contraction_census'] = None
    elif damage == 'legacy_surplus':
        receipt.pop('integer_contraction_census')
    elif damage == 'null_census':
        receipt['integer_contraction_census'] = None
    elif damage == 'list_census':
        receipt['integer_contraction_census'] = []
    elif damage == 'missing_dynamic':
        metadata['quantization_stats'].pop('dynamic_contractions')
    elif damage == 'null_dynamic':
        metadata['quantization_stats']['dynamic_contractions'] = None
    elif damage == 'failed_dynamic':
        metadata['quantization_stats']['dynamic_contractions']['sites'][0]['verification']['finite'] = False
    elif damage == 'duplicate_ordinal':
        binding['ordinal'] = bindings[1]['ordinal']
    elif damage == 'missing_ordinal':
        binding.pop('ordinal')
    elif damage.endswith('_ordinal'):
        binding['ordinal'] = {'null_ordinal': None, 'negative_ordinal': -1, 'bool_ordinal': True,
                              'float_ordinal': float(binding['ordinal']),
                              'string_ordinal': str(binding['ordinal'])}[damage]
    elif damage == 'string_sources':
        binding['source_node_ids'] = binding['source_node_ids'][0]
    elif damage == 'null_sources':
        binding['source_node_ids'] = None
    elif damage == 'nonstring_source':
        binding['source_node_ids'].append(None)
    elif damage == 'another_call_source':
        binding['source_node_ids'].append(bindings[1]['source_node_ids'][0])
    elif damage == 'forged_intersection':
        binding['contraction_source_ids'] = bindings[1]['contraction_source_ids']
    elif damage == 'bool_rank':
        binding['rank'] = True
    elif damage == 'duplicate_call':
        census['calls'][1] = copy.deepcopy(census['calls'][0])
    elif damage == 'null_calls':
        census['calls'] = None
    elif damage == 'null_bindings':
        census['export_bindings'] = None
    elif damage == 'bool_emitted':
        receipt['integer_mm_emitted'] = True
    elif damage == 'surplus_export':
        receipt['exported_integer_mm_count'] += 1
    elif damage == 'underexport':
        receipt['exported_integer_mm_count'] -= 1
    elif damage == 'wrong_dynamic_owner':
        census['calls'][2]['origin_node_ids'] = ['g:original:root:n999']
    elif damage != 'valid_union':
        raise AssertionError(damage)


_MATRIX_CASES = [
    ('valid_union', True), ('integerizer_only', True), ('legacy_equal', True), ('legacy_no_batch', True),
    ('contraction_free_zero', True), ('contraction_free_empty_dynamic', True),
    ('legacy_surplus', False), ('null_census', False), ('list_census', False),
    ('missing_dynamic', False), ('null_dynamic', False), ('failed_dynamic', False),
    ('duplicate_ordinal', False), ('missing_ordinal', False), ('null_ordinal', False),
    ('negative_ordinal', False), ('bool_ordinal', False), ('float_ordinal', False), ('string_ordinal', False),
    ('string_sources', False), ('null_sources', False), ('nonstring_source', False),
    ('another_call_source', False), ('forged_intersection', False), ('bool_rank', False),
    ('duplicate_call', False), ('null_calls', False), ('null_bindings', False), ('bool_emitted', False),
    ('surplus_export', False), ('underexport', False), ('wrong_dynamic_owner', False),
    ('static_only_valid', True), ('static_only_underexport', False), ('static_only_no_dynamic', False),
    ('static_only_null_census', False),
]


@pytest.fixture(scope='module')
def sealed_census_admission():
    """Exercise the optional issuer's adapter from this copied checkout."""
    import importlib.util

    from merlin.common.paths import repo_root

    path = repo_root() / 'packages/merlin-experiments/src/merlin_experiments/capture_execution/sealed_m2m_validation.py'
    spec = importlib.util.spec_from_file_location('sealed_census_matrix', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._integerization_census_admission


@pytest.mark.parametrize('owner_path', ['ordinary', 'seeded', 'custom_entry'])
@pytest.mark.parametrize('damage,expected', _MATRIX_CASES)
def test_admission_consistency_matrix(tmp_path, owner_path, damage, expected, sealed_census_admission):
    """The identical refreshed receipt must get the same verdict at every admission boundary."""
    import json
    from types import SimpleNamespace

    from merlin.targetgen.application_inventory import verified_static_integerization, verify_capture_receipt
    from merlin.targetgen.capsule_source import M2MUnavailable, _require_pt2e_integerization_receipt
    from merlin.targetgen.integerization_export import capture_dynamic_evidence, worker_export_complete

    _, trace = _inventory_fixture(tmp_path, seeded=owner_path != 'ordinary')
    metadata = json.loads((tmp_path / 'meta.json').read_text())
    program = (tmp_path / 'model.mlir').read_text()
    func_name = 'custom' if owner_path == 'custom_entry' else 'forward'
    if func_name != 'forward':
        program = program.replace('@forward', '@custom')
    metadata['func_name'] = func_name
    _matrix_damage(damage, metadata, trace)
    if damage.startswith('contraction_free_'):
        program = f"""builtin.module {{
            func.func @{func_name}(%a: tensor<2x4xi8>, %b: tensor<4x2xi8>) -> tensor<2x2xi32> {{
                %zero = tensor.empty() : tensor<2x2xi32>
                func.return %zero : tensor<2x2xi32>
            }}
        }}"""
    raw_trace, program_sha256 = _refresh_capture(tmp_path, metadata, trace, program)
    receipt = metadata['integerization_receipt']
    module = query.parse(program)
    options = {'func_name': func_name, 'program_sha256': program_sha256,
               'dynamic_evidence': lambda: capture_dynamic_evidence(metadata, raw_trace, program_sha256=program_sha256)}
    observation = verify_capture_receipt(tmp_path / 'model.mlir')
    results = {
        'authoritative_saved': admit_integerization_census(receipt, **options).accepted,
        'authoritative_module': admit_integerization_census(receipt, module, **options).accepted,
        'census_saved': census_export_complete(receipt, **options),
        'census_module': census_export_complete(receipt, module, **options),
        'worker_compatibility': receipt_export_complete(receipt, {}, **options),
        'inventory': verified_static_integerization(observation.get('capture_integerization')),
        'sealed_stage': sealed_census_admission(metadata, tmp_path).accepted,
    }
    # New workers always emit a census. Legacy compatibility is exercised above;
    # requiring it for a new export must refuse its omission, even at exact equality.
    results['new_worker'] = worker_export_complete(
        SimpleNamespace(module=module, mlir_text=program, capture_trace=trace), {}, metadata, func_name=func_name)
    try:
        _require_pt2e_integerization_receipt(program, metadata, agreement_tolerance=(0.0, 0.0), capture_root=tmp_path)
        results['capsule_fresh_and_cached'] = True
    except M2MUnavailable:
        results['capsule_fresh_and_cached'] = False
    expectations = {entry: expected for entry in results}
    if damage.startswith('legacy_'):
        expectations['new_worker'] = False
    assert results == expectations, (results, observation)
    # Byte custody alone does not endorse producer arithmetic.
    if damage == 'another_call_source':
        assert observation['status'] == 'verified_materialized'
