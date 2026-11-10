from copy import deepcopy
from hashlib import sha256
import json
import pytest
from merlin.compile.region_loop import analyze_region_loop, expand_region_loop, structural_signature


def request(name, op='copy', immediate=1):
    return dict(nodes=[dict(id=name, inputs=[], attrs={'value': immediate}, op=op, effect='pure', type={'dtype':'i32','shape':[1]})],
                outputs=[name], constants=[], input_storages={}, source_identity=name)


def test_loop_expands_exact_wiring_and_parameters():
    flat=[request('prefix','start')]+[request(str(i),op,i) for i in range(4) for op in ('a','b')]+[request('suffix','end')]
    record=analyze_region_loop(flat, parameter_attributes=('value',))
    assert (record['offset'],record['period'],record['trip_count'])==(1,2,4)
    assert expand_region_loop(record)==flat
    changed=deepcopy(record); changed['iterations'][1][0]['outputs']=['wrong']
    with pytest.raises(ValueError,match='expansion'): expand_region_loop(changed)
    changed=deepcopy(record);changed['trip_count']+=1
    with pytest.raises(ValueError,match='bounds'):expand_region_loop(changed)


def test_signatures_retain_semantics_unless_explicitly_parameterized():
    a=request('x');b=request('y',immediate=2)
    assert structural_signature(a)!=structural_signature(b)
    assert structural_signature(a,parameter_attributes=('value',))==structural_signature(b,parameter_attributes=('value',))
    b['nodes'][0]['type']['shape']=[2]
    assert structural_signature(a,parameter_attributes=('value',))!=structural_signature(b,parameter_attributes=('value',))
    assert analyze_region_loop([request('x','a'),request('y','b')]) is None


def boundary_request(left='left', right='right', output='out'):
    """A complete valid request with declared ordered input boundaries."""
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
    tensor = TensorType((2,), 'bf16', 'raw-bit-pattern')
    return KernelRequest(
        nodes=(SemanticNode(left, 'input', (), tensor, effect='input'),
               SemanticNode(right, 'input', (), tensor, effect='input'),
               SemanticNode(output, 'bf16_add', (left, right), tensor,
                            attrs=(('numeric_semantics', 'exact'),), index_maps=pointwise_identity_maps(1, 2))),
        outputs=(output,), output_storages=('external',),
        input_storages=((left, 'external'), (right, 'external')), target_identity='test-target',
    ).record()


def test_v2_and_v5_empty_optional_fields_have_same_analysis_signature():
    # Reordering the declared boundary can make record() select v5 even when
    # all physical metadata is empty. Analysis canonicalizes by operand use;
    # exact expansion still retains the original ABI and versioned records.
    old = boundary_request()
    ordered = boundary_request('zleft', 'aright', 'result')
    assert old['schema'] == 'merlin.semantic_kernel.v2'
    assert ordered['schema'] == 'merlin.semantic_kernel.v5'
    assert ordered['physical_extents'] == {}
    assert ordered['boundary_layouts'] == {}
    assert ordered['output_layouts'] == []
    assert structural_signature(old) == structural_signature(ordered)
    flat = [deepcopy(old), deepcopy(ordered), deepcopy(old), deepcopy(ordered)]
    loop = analyze_region_loop(flat)
    assert (loop['offset'], loop['period'], loop['trip_count']) == (0, 1, 4)
    assert expand_region_loop(loop) == flat


def test_constants_keep_bytes_encoding_and_binding_association():
    source = boundary_request()
    source['nodes'][0].update(op='constant', effect='constant')
    source['input_storages'].pop('left')
    source['constants'] = [dict(node_id='left', encoding='bf16-le-raw', data_hex='803f0040')]
    renamed = deepcopy(source)
    renamed['nodes'][0]['id'] = 'literal'
    renamed['nodes'][-1]['inputs'][0] = 'literal'
    renamed['constants'][0]['node_id'] = 'literal'
    assert structural_signature(source) == structural_signature(renamed)
    for field, value in [('data_hex', '803f0140'), ('encoding', 'i32-le'), ('node_id', 'right')]:
        changed = deepcopy(source)
        changed['constants'][0][field] = value
        assert structural_signature(source) != structural_signature(changed)


def test_index_maps_types_and_physical_layouts_remain_significant():
    source = boundary_request()
    changed = deepcopy(source)
    changed['nodes'][-1]['index_maps'][0]['offsets'][0] = 1
    assert structural_signature(source) != structural_signature(changed)
    for field, value in [('shape', [1, 2]), ('dtype', 'f32'), ('numerical_policy', 'different')]:
        changed = deepcopy(source)
        changed['nodes'][-1]['type'][field] = value
        assert structural_signature(source) != structural_signature(changed)
    for field, value in [('physical_extents', {'left': 2, 'right': 2}),
                         ('boundary_layouts', {'left': 'contiguous', 'right': 'contiguous'}),
                         ('output_layouts', ['contiguous'])]:
        changed = deepcopy(source)
        changed[field] = value
        assert structural_signature(source) != structural_signature(changed)


def test_rehashed_changed_iteration_is_rejected_by_structure_check():
    loop = analyze_region_loop([boundary_request(), boundary_request('x', 'y', 'z')])
    changed = deepcopy(loop)
    changed['iterations'][1][0]['nodes'][-1]['op'] = 'bf16_sub'
    flat = changed['prefix'] + [item for row in changed['iterations'] for item in row] + changed['suffix']
    changed['flat_sha256'] = sha256(json.dumps(flat, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match='body structure'):
        expand_region_loop(changed)


def test_application_default_skips_loop_analysis_and_metadata(monkeypatch):
    from merlin.compile.semantic_application import CompiledCRegion, _render
    from merlin.compile.semantic_regions import outline_semantic_regions
    from merlin.tests.ir.test_semantic_regions import graph, plan_for
    source = graph()
    plan = plan_for(source, ((0,), (1,), (2,), (3,)))

    class Adapter:
        def plan(self, source):
            return plan

    adapter = Adapter()
    outlined = outline_semantic_regions(source, entry='forward', plan=plan)
    compiled = []
    for i, (_, region) in enumerate(outlined.regions):
        name = f'region_{i}'
        header = name + '.h'
        compiled.append(CompiledCRegion(name, header, tuple(k for k, _ in region.input_storages),
            region.outputs, {header: f'int {name}(void *,const void *const *,void *const *);\n'.encode()},
            (), (), {}))
    baseline = _render(source, adapter, 'forward', 4096, compiled)
    adapter.region_loop_parameters = ()

    def unexpected_analysis(*args, **kwargs):
        pytest.fail('default application must not analyze region loops')

    monkeypatch.setattr('merlin.compile.region_loop.analyze_region_loop', unexpected_analysis)
    analyzed = _render(source, adapter, 'forward', 4096, compiled)
    assert 'application-loop-plan.json' not in analyzed[-1]
    assert analyzed[1:] == baseline[1:]
