"""Public, target-neutral source exploration for offline and agent callers.

These are small mechanism patterns, not ResNet or TinyLlama coverage.
Run: PYTHONPATH=src python examples/compiler/fusion_exploration.py --output <dir>
This starts from public source IR and selected synthetic resource declarations,
not an authored backend. Outputs are source IR and checked logical schedules;
they are explicitly not target binaries or execution/performance evidence.
Replace emit() with the selected compiler's ordinary target adapter/emitter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from merlin.compile import fusion_workflow as F
from merlin.compile.connected_regions import ConnectedCompilation, plan_connected_region
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.shared_input_schedule import SharedInputSchedule
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.dense_planning import DenseResources
from merlin.transforms.pipeline_state import PipelineContext
from merlin.verify.lean_contract import document_digest

IDENTITY = 'affine_map<(m,n)->(m,n)>'


def elementwise(name, inputs, output, shape, dtype, output_dtype, body):
    maps = ','.join([IDENTITY] * (len(inputs) + 1))
    in_types = ','.join([f'tensor<{shape}x{dtype}>'] * len(inputs))
    args = ','.join(f'%x{i}:{dtype}' for i in range(len(inputs)))
    return f'''%{name} = linalg.generic {{indexing_maps=[{maps}], iterator_types=["parallel","parallel"]}}
      ins({','.join(inputs)}:{in_types}) outs({output}:tensor<{shape}x{output_dtype}>) {{
      ^bb0({args},%unused:{output_dtype}):
        {body}
        linalg.yield %value:{output_dtype}
      }} -> tensor<{shape}x{output_dtype}>'''


def contraction(name, lhs, rhs, init, m, k, n):
    return f'''%{name}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
      iterator_types=["parallel","parallel","reduction"]}}
      ins({lhs},{rhs}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs({init}:tensor<{m}x{n}xi32>) {{
      ^bb0(%a:i8,%b:i8,%acc:i32):
        %aa=arith.extsi %a:i8 to i32
        %bb=arith.extsi %b:i8 to i32
        %prod=arith.muli %aa,%bb:i32
        %value=arith.addi %acc,%prod:i32
        linalg.yield %value:i32
      }} -> tensor<{m}x{n}xi32>'''


def patterns():
    residual = elementwise('sum', ('%x', '%skip'), '%init', '3x5', 'i8', 'i8',
                           '%value=arith.addi %x0,%x1:i8')
    residual += elementwise('out', ('%sum',), '%init', '3x5', 'i8', 'i8',
                            '%zero=arith.constant 0:i8\n%value=arith.maxsi %x0,%zero:i8')
    yield 'simple_integer_residual', f'''module {{func.func @residual(%x:tensor<3x5xi8>,%skip:tensor<3x5xi8>)
      ->tensor<3x5xi8> {{%init=tensor.empty():tensor<3x5xi8>
      {residual}
      return %out:tensor<3x5xi8>}}}}'''

    widths = (4, 7, 2)
    args = ','.join(f'%w{i}:tensor<5x{n}xi8>' for i, n in enumerate(widths))
    bodies = []
    for i, n in enumerate(widths):
        bodies.append(f'''%zero{i}=arith.constant 0:i32
          %init{i}="tensor.splat"(%zero{i}):(i32)->tensor<3x{n}xi32>
          %out{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
            affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
            iterator_types=["parallel","parallel","reduction"]}}
          ins(%x,%w{i}:tensor<3x5xi8>,tensor<5x{n}xi8>) outs(%init{i}:tensor<3x{n}xi32>) {{
          ^bb0(%a:i8,%b:i8,%acc:i32):
            %aa=arith.extsi %a:i8 to i32
            %bb=arith.extsi %b:i8 to i32
            %prod=arith.muli %aa,%bb:i32
            %value=arith.addi %acc,%prod:i32
            linalg.yield %value:i32
          }} -> tensor<3x{n}xi32>''')
    types = ','.join(f'tensor<3x{n}xi32>' for n in widths)
    yield 'shared_projections', f'''module {{func.func @projections(%x:tensor<3x5xi8>,{args})
      ->({types}) {{ {' '.join(bodies)} return %out0,%out1,%out2:{types} }} }}'''

    epilogue = elementwise('activation', ('%hidden',), '%a_init', '3x7', 'i32', 'i8',
        '%lo=arith.constant 0:i32\n%hi=arith.constant 127:i32\n'
        '%positive=arith.maxsi %x0,%lo:i32\n%clipped=arith.minsi %positive,%hi:i32\n'
        '%value=arith.trunci %clipped:i32 to i8')
    yield 'simple_integer_mlp', f'''module {{func.func @mlp(%x:tensor<3x5xi8>,%up:tensor<5x7xi8>,%down:tensor<7x2xi8>)
      ->tensor<3x2xi32> {{
      %zero=arith.constant 0:i32
      %h_init="tensor.splat"(%zero):(i32)->tensor<3x7xi32>
      %a_init=tensor.empty():tensor<3x7xi8>
      %o_init="tensor.splat"(%zero):(i32)->tensor<3x2xi32>
      {contraction('hidden', '%x', '%up', '%h_init', 3, 5, 7)}
      {epilogue}
      {contraction('out', '%activation', '%down', '%o_init', 3, 7, 2)}
      return %out:tensor<3x2xi32> }} }}'''


def run(output, *, connected_selector=None):
    """Optional selected target callback receives a typed ConnectedRegion plan.

    It returns compile_connected_pointwise options; no target implementation is
    supplied by this example. The default run only emits generic mechanism IR.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capability = {'compute_units': [{'name': 'synthetic_array', 'semantic_capabilities': [
        {'family': 'contraction', 'dtypes': ['int8', 'int32']}]}]}
    numeric = document_digest({'products': 'signed_i8', 'accumulator': 'wrapping_i32',
                               'epilogues': 'exact_source_order'})
    discovery = dict(capability_contract=capability, numerical_policy_sha256=numeric)
    context = PipelineContext(hardware=document_digest(capability), numerical_policy=numeric)
    storage = DenseLoopStorage(DenseMemoryView('lhs', 'lhs', 3, 1, 64),
        DenseMemoryView('rhs', 'rhs', 4, 1, 64), DenseMemoryView('acc', 'acc', 16, 1, 64))
    resources = DenseResources(448, 1024, max_m_blocks=2, max_k_blocks=2, max_n_blocks=3,
                               partial_output_tiles=True, partial_output_blocks=True)
    summary = {'qualification': 'public mechanism study; no target executable or measured speedup', 'patterns': []}
    for name, text in patterns():
        module = parse_mlir_text(text)
        module.verify()
        folder = output / name
        folder.mkdir(exist_ok=True)
        cursor, rounds, emitted = None, [], []
        def emit(product, session):
            path = folder / f'artifact-{len(emitted)}'
            if isinstance(product.artifact, ConnectedCompilation):
                files = []
                for stage, images in enumerate(product.artifact.images()):
                    launch = []
                    for segment, data in enumerate(images):
                        target = folder / f'{path.name}-stage-{stage}-{segment}.bin'
                        target.write_bytes(data)
                        launch.append(str(target))
                    files.append(launch)
                record = {'launches': files, 'kind': 'native_checked_images_runtime_unqualified'}
                emitted.append(record)
                return record
            if isinstance(product.artifact, SharedInputSchedule):
                path = path.with_suffix('.schedule.json')
                path.write_text(json.dumps(asdict(product.artifact), indent=2) + '\n')
                kind = 'checked_logical_schedule_pending_target_adapter'
            else:
                path = path.with_suffix('.mlir')
                path.write_text(str(product.artifact) + '\n')
                kind = 'unchanged_source_pending_target_compiler'
            record = {'path': str(path), 'kind': kind}
            emitted.append(record)
            return record
        while True:
            page = F.discover(module, max_candidates=15, cursor=cursor, **discovery)
            alternatives, plans = [], []
            for row in page['candidates']:
                if row['route'] in {'adjacent_elementwise', 'connected_residency', 'epilogue'}:
                    try:
                        members = tuple(module.walk())
                        region = plan_connected_region(module, tuple(members[i] for i in row['operation_ordinals']))
                        plans.append({'candidate': row['candidate_index'], 'plan': region.record()})
                    except ValueError as error:
                        plans.append({'candidate': row['candidate_index'], 'reason': str(error)})
                def realize(private, row=row):
                    selection = dict(primitive_tile=(2, 3, 4), storage=storage, resources=resources,
                                     selected_block=(1, 1, 1)) if row['route'] == 'shared_input' else {}
                    if connected_selector is not None and row['route'] in {
                            'adjacent_elementwise', 'connected_residency', 'epilogue'}:
                        members = tuple(private.walk())
                        plan = plan_connected_region(private, tuple(members[i] for i in row['operation_ordinals']))
                        selection['connected'] = connected_selector(plan)
                    return F.realize_candidate(private, row, **discovery, **selection)
                alternatives.append(F.FusionAlternative(f"candidate-{row['candidate_index']}", realize))
            results = list(F.evaluate_fusions(module, alternatives, emit, context=context))
            rounds.append({'page': page, 'connected_plans': plans, 'results': [{'name': r.name, 'status': r.status,
                'reason': r.reason, 'realization': r.realization.receipt if r.realization else None,
                'pipeline': r.evaluation.result.receipt if r.evaluation.result else None} for r in results]})
            cursor = page['next_cursor']
            if cursor is None:
                break
        (folder / 'exploration.json').write_text(json.dumps(rounds, indent=2) + '\n')
        summary['patterns'].append({'name': name, 'rounds': len(rounds), 'artifacts': emitted,
            'statuses': [r['status'] for turn in rounds for r in turn['results']]})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(run(parser.parse_args().output), indent=2))
