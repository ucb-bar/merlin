"""Connected source composition through real native checked stage compilation."""
from dataclasses import replace
import struct

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.compile.connected_regions import plan_connected_region, compile_connected_pointwise
from merlin.compile.scheduling.program_limits import AddressWindow, Continuation
from merlin.sched.ir import Const, Kernel
from test_multioutput_fusion import model, WORD, SPACES


def fixture(tmp_path):
    module, operations, selected = model(tmp_path)
    region = plan_connected_region(module, operations[:2])
    count = 15
    windows = (AddressWindow('input0', 0, count * 2), AddressWindow('input1', 0, count * 2),
               AddressWindow('output1', 0, count * 2), AddressWindow('output0', 0, count))
    runtime = Continuation('neutral-sequential-executor', SPACES[2:], True, True, 8)
    def options(region, stage, bindings):
        ordinal = region.stages.index(stage)
        call = selected['kernel'].body[ordinal]
        changes = {'dst': 5, 'w': 0} if ordinal == 0 else {'src0': 5, 'r0': 0}
        call = replace(call, args=tuple((k, Const(changes.get(k, v.value))) for k, v in call.args))
        return {k:v for k,v in selected.items() if k not in
                {'bindings', 'selected_contracts', 'composition_law', 'kernel', 'objects'}} | {
                'kernel': Kernel('stage', (), (call,)), 'objects': (),
                'fusion_qualification_identity': 'neutral-byte-machine-scalar-v1'}
    settings = dict(windows=windows, byte_orders=('little',) * 4,
                    selected_contracts=selected['selected_contracts'], continuation=runtime,
                    stage_options=options)
    return region, settings


def execute_launches(images):
    """Independent byte machine keeps runtime buffers across separate launches."""
    left = np.array([-32768,32767,2,-4,100,200,-20,127,128,-128,0,1,2,3,4],dtype='<i2')
    right = np.array([-1,1,5,-8,300,-500,-300,1,0,-2,0,32767,-32768,10,5],dtype='<i2')
    memory = [bytearray([193] * 512) for _ in SPACES]
    memory[2][:30], memory[3][:30] = left.tobytes(), right.tobytes()
    for launch in images:
        for image in launch:
            for opcode, src0, src1, dst, r0, r1, out, elements in WORD.iter_unpack(image):
                for i in range(elements):
                    value = int.from_bytes(memory[src0][r0+i*2:r0+(i+1)*2], 'little', signed=True)
                    if opcode == 0:
                        b = int.from_bytes(memory[src1][r1+i*2:r1+(i+1)*2], 'little', signed=True)
                        value = (value+b+32768) % 65536 - 32768
                    elif opcode == 1:
                        value = min(127, max(-128, value))
                    else:
                        raise AssertionError('unexpected instruction')
                    size = 1 if opcode == 1 else 2
                    memory[dst][out+i*size:out+(i+1)*size] = value.to_bytes(size,'little',signed=True)
    expected = (left.astype(np.int32) + right.astype(np.int32)).astype('<i2')
    assert memory[2][:30] == left.tobytes() and memory[3][:30] == right.tobytes()
    assert memory[5][:30] == expected.tobytes()  # Retained for external fanout.
    assert memory[4][:15] == expected.clip(-128,127).astype('i1').tobytes()
    assert memory[5][30:] == bytes([193] * (512-30))
    assert memory[4][15:] == bytes([193] * (512-15))


def test_connected_numerical_output_preserves_external_fanout(tmp_path):
    region, settings = fixture(tmp_path)
    assert region.outputs == (2,3)  # Producer remains observable by third source op.
    assert region.lifetimes[2][1] == 2
    compiled = compile_connected_pointwise(region, **settings)
    assert compiled.report['actual_encoded_bytes'] == 2 * WORD.size
    assert compiled.report['stage_launches'] == [1,1]
    assert [p.source_passes[-1].name for p in compiled.products] == ['source-conformance'] * 2
    execute_launches(compiled.images())
    region.source.attributes['change'] = StringAttr('source')
    with pytest.raises(ValueError, match='source changed'):
        compiled.images()


@pytest.mark.parametrize('failure', ['alias', 'extent', 'preserve', 'budget', 'override', 'binding_mutation'])
def test_connected_boundary_refusals(tmp_path, failure):
    region, settings = fixture(tmp_path)
    if failure == 'alias':
        settings['windows'] = (*settings['windows'][:2], settings['windows'][0], settings['windows'][3])
    if failure == 'extent':
        settings['windows'] = (*settings['windows'][:3], AddressWindow('output0',0,14))
    if failure == 'preserve':
        settings['continuation'] = replace(settings['continuation'], preserved_spaces=('input0','input1'))
    if failure == 'budget':
        settings['continuation'] = replace(settings['continuation'], max_segments=1)
    original = settings['stage_options']
    if failure == 'override':
        settings['stage_options'] = lambda r,s,b: dict(original(r,s,b), bindings=b)
    if failure == 'binding_mutation':
        def mutate(r,s,b):
            object.__setattr__(b.output, 'start', 20)
            return original(r,s,b)
        settings['stage_options'] = mutate
    with pytest.raises(ValueError):
        compile_connected_pointwise(region, **settings)


def test_wrong_consumer_input_is_rejected_by_native_source_checker(tmp_path):
    region, settings = fixture(tmp_path)
    original = settings['stage_options']
    def wrong(region, stage, bindings):
        result = original(region, stage, bindings)
        if stage is region.stages[1]:
            call, = result['kernel'].body
            call = replace(call, args=tuple((k, Const(2) if k == 'src0' else v) for k,v in call.args))
            result['kernel'] = replace(result['kernel'], body=(call,))
        return result
    with pytest.raises(ValueError, match='undeclared external read'):
        compile_connected_pointwise(region, **dict(settings, stage_options=wrong))


@pytest.mark.parametrize('field', ['inputs', 'output', 'byte_orders'])
def test_stage_selector_cannot_reassign_source_owned_binding_fields(tmp_path, field):
    region, settings = fixture(tmp_path)
    original = settings['stage_options']

    def replace_field(region, stage, bindings):
        result = original(region, stage, bindings)
        replacement = {'inputs': (), 'output': bindings.inputs[0],
                       'byte_orders': tuple('big' for _ in bindings.byte_orders)}[field]
        object.__setattr__(bindings, field, replacement)
        return result

    with pytest.raises(ValueError, match='tensor bindings changed'):
        compile_connected_pointwise(region, **dict(settings, stage_options=replace_field))


def test_discovered_connected_candidate_consumed_by_normal_emission(tmp_path):
    from merlin.compile import fusion_workflow as F
    region, settings = fixture(tmp_path)
    contract = dict(capability_contract={}, numerical_policy_sha256='a'*64)
    page = F.discover(region.source, **contract)
    ordinals = [s.ordinal for s in region.stages]
    row = next(row for row in page['candidates'] if row['operation_ordinals'] == ordinals)
    def realize(private):
        return F.realize_candidate(private, row, connected=settings, **contract)
    def emit(product, session):
        return product.artifact.images() if product.receipt['status'] == 'applied' else str(product.artifact)
    results = list(F.evaluate_fusions(region.source, (F.FusionAlternative('connected', realize),), emit))
    assert [r.status for r in results] == ['compiled','compiled'], [r.reason for r in results]
    assert results[-1].realization.receipt['physical_savings_claimed'] is False
    execute_launches(results[-1].evaluation.result.artifact)


def test_typed_mlp_plan_reports_counted_launch_gap_before_target_callback(tmp_path):
    import importlib.util
    from merlin.common.paths import repo_root
    from merlin.frontends.linalg_mlir import parse_mlir_text
    spec = importlib.util.spec_from_file_location('public_fusion_example',
        repo_root() / 'examples/compiler/fusion_exploration.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    module = parse_mlir_text(dict(example.patterns())['simple_integer_mlp'])
    operations = tuple(op for op in module.walk() if op.name == 'linalg.generic')
    region = plan_connected_region(module, operations)
    assert [s.kind for s in region.stages] == ['contraction','elementwise_map','contraction']
    called = []
    with pytest.raises(ValueError, match='multi-stage pointer/launch ABI'):
        compile_connected_pointwise(region, windows=(), byte_orders=(), selected_contracts=None,
            continuation=None, stage_options=lambda *args: called.append(args))
    assert not called


def test_plan_refuses_intervening_observer_and_runtime_mutation(tmp_path):
    from xdsl.dialects import func
    region, settings = fixture(tmp_path)
    compiled = compile_connected_pointwise(region, **settings)
    object.__setattr__(settings['continuation'], 'preserved_spaces', ())
    with pytest.raises(ValueError, match='continuation changed'):
        compiled.images()
    first, second = (s.operation for s in region.stages)
    observer = func.CallOp('observe', [first.results[0]], [])
    second.parent.insert_op_before(observer, second)
    with pytest.raises(ValueError, match='unmodeled operation or observer'):
        plan_connected_region(region.source, (first, second))
