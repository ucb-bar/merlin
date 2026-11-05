"""Inventory existing public sources and deferred policy without claiming admission."""
from __future__ import annotations

import collections
import json
from math import prod
from pathlib import Path
import sys

import common as C

C.configure()
import yaml
from merlin.targetgen.corpus_spec import CorpusBinding, build
from merlin.perf.derived_bound import machine_from_facts
from merlin.perf.movement_peak import movement_peak
from merlin.perf.staged_roofline import Contraction, staged_roofline

sys.path.insert(0, str(C.BASE))
from frontend import parse
from emitter import make_c


def binding():
    return CorpusBinding(target='gemmini', tile_dim=16, operand_dtype='int8', accum_dtype='i32',
                         integer=True, tiers=['L2', 'L3'], compare='exact_int',
                         requant_output_dtype='i8', requant_shift=4)


def contractions(cb):
    tensors = cb['tensors']
    residents, accumulators, work = {}, {}, []
    for cmd in cb['commands']:
        op, o, a = cmd['opcode'], cmd.get('operands', {}), cmd.get('attributes', {})
        if op == 'RES_PACK':
            residents[o['dst']] = o['src']
        elif op in {'MATMUL_RESIDENT', 'MATMUL'}:
            lhs, rhs = tensors[o['lhs']]['shape'], tensors[residents.get(o['rhs'], o['rhs'])]['shape']
            accumulators[o['dst']] = {'M': prod(lhs[:-1]), 'K': lhs[-1], 'N': rhs[-1]}
        elif op == 'COMMIT':
            work.append(dict(accumulators[o['src']], readout=a, output=o['dst']))
        elif op == 'BATCHED_MATMUL':
            lhs, rhs = tensors[o['a']]['shape'], tensors[o['w']]['shape']
            work.append({'M': lhs[-2], 'K': lhs[-1], 'N': rhs[-1], 'batch': prod(lhs[:-2]), 'readout': a})
        elif op in {'ATTENTION_QK', 'ATTENTION_PV'}:
            lhs = tensors[o['q'] if op == 'ATTENTION_QK' else o['p']]['shape']
            rhs = tensors[o['k'] if op == 'ATTENTION_QK' else o['v']]['shape']
            work.append({'M': lhs[-2], 'K': lhs[-1], 'N': rhs[-2] if op == 'ATTENTION_QK' else rhs[-1],
                         'batch': prod(lhs[:-2]), 'readout': a, 'layout': op})
        elif op == 'CONV2D':
            kh, kw, ci, co = a['kernel']
            out = tensors[o['dst']]['shape']
            work.append({'M': prod(out[:-1]), 'K': kh * kw * ci, 'N': co,
                         'readout': a, 'conv_kernel': a['kernel'], 'stride': a.get('stride'),
                         'padding': a.get('padding'), 'source_layout': tensors[o['ifm']]['shape']})
    return work


def inventory_row(identity, source, descriptor, cohort, machine, ports):
    row = {'identity': identity, 'cohort': cohort, 'source': str(source),
           'source_sha256': C.file_hash(source), 'descriptor_sha256': C.digest(descriptor),
           'performance': descriptor.get('performance'), 'formal_admission': False,
           'purpose': descriptor.get('source_reference', descriptor.get('semantic', descriptor.get('description'))),
           'baseline': {'status': 'not_executed', 'reason': 'inventory is not a passing baseline'},
           'expected_evaluation_cost': {'compile_s': None, 'spike_s': None, 'gsim_s': None,
                                        'basis': 'UNKNOWN until identity-matched measurement'},
           'numerical_contract': {'policy': 'selected i8/i32 and explicit binary32 readouts; no tolerance changes',
                                  'comparison': 'complete output exact integer; full models retain Phase1 selected gates'}}
    try:
        text = source.read_text()
        mixed = None
        if 'merlin_iface.' in text:
            cb = parse(text)
        else:
            from mixed_codegen import emit_mixed
            cb, mixed, _ = emit_mixed(text)
        row['required_accelerator_operations'] = sorted({c['opcode'] for c in cb['commands']})
        row['tensors'] = cb['tensors']
        row['derived_shapes'] = contractions(cb)
        if any('requant' in c.get('attributes', {}).get('epilogue', []) for c in cb['commands']):
            raise ValueError('selected compiler refuses integer requant; no declared target readout route')
        if mixed is None:
            make_c(cb)
        row['baseline']['emission'] = 'C schedule generated; LLVM/link/runtime qualification pending'
        save_path = C.OUT / 'inventory-command-buffers' / (identity.replace('/', '__') + '.json')
        C.save(save_path, cb)
        row['command_buffer_path'] = str(save_path)
        row['command_buffer_sha256'] = C.digest(cb)
        shapes = row['derived_shapes']
        row['resource_pressure'] = []
        for w in shapes:
            m, k, n = (w[x] for x in ('M', 'K', 'N'))
            operand = m * k + k * n
            accum = m * n * 4
            row['resource_pressure'].append({
                'operand_bytes': operand, 'accumulator_bytes': accum,
                'rhs_scratchpad_rows': ((k + 15)//16)*((n + 15)//16)*16,
                'padded_macs': ((m+15)//16)*((k+15)//16)*((n+15)//16)*4096,
                'useful_macs': m*k*n,
                'note': 'logical whole-operation pressure; emitted tile lifetimes need separate proof'})
        if shapes and not any('conv_kernel' in x or x.get('batch', 1) != 1 for x in shapes):
            widths = {1 if cb['tensors'][x['output']]['dtype']=='i8' else 4 for x in shapes if x.get('output')}
            if len(widths)!=1:
                row['roofline'] = {'status':'UNKNOWN','reason':'mixed readout widths need per-stage accounting'}
            else:
                row['roofline'] = staged_roofline([Contraction(x['M'], x['K'], x['N']) for x in shapes],
                                                 machine, ports, result_bytes=widths.pop())
                row['roofline']['scope_limit'] = 'partial lower bound; host issue/layout unresolved, no convergence claim'
        else:
            row['roofline'] = {'status': 'UNKNOWN', 'reason': 'need layout/batch/convolution physical transfer derivation'}
    except Exception as error:
        row['baseline'] = {'status': 'refused_or_unmaterialized', 'reason': str(error)}
        row.setdefault('derived_shapes', [])
        row.setdefault('required_accelerator_operations', [])
        row.setdefault('resource_pressure', [])
        row['roofline'] = {'status': 'UNKNOWN', 'reason': 'baseline emission refused or semantics unavailable'}
    return row


def model_inventory():
    """Read source/group metadata only; never weights, tensor outputs or calibration labels."""
    handoff = C.P1 / '.tmp/handoff/functional-phase1-baseline-v1'
    selected = []
    for name in ('tiny32-prefill-compile-v2', 'tiny32-decode-compile-v2',
                 'smolvla-prefix_encode-compile-v2', 'smolvla-flow_denoise-compile-v2',
                 'smolvla-action_decode-compile-v2', 'resnet-retained-spike-v1'):
        p = handoff / name / 'status.json'
        if p.exists():
            obj = json.loads(p.read_text())
            selected.append({'selection': name, 'path': str(p), 'sha256': C.file_hash(p), 'metadata': obj})
    groups = []
    roots = [C.P1 / 'out/build/phase1-corrected-input-compile-20261018-v1']
    # Follow only compilation product directories explicitly recorded by the selected owner statuses.
    def paths(value):
        if isinstance(value, dict):
            for val in value.values():
                yield from paths(val)
        elif isinstance(value, list):
            for val in value:
                yield from paths(val)
        elif isinstance(value, str) and value.startswith(str(C.P1)):
            yield Path(value)
    files = set()
    for selection in selected:
        for p in paths(selection['metadata']):
            if p.is_file() and p.name in {'compilation_recipe.json', 'device_signatures.json'}:
                files.add(p.parent / 'device_signatures.json')
            elif p.is_dir():
                for rel in ('device_signatures.json', 'build/device_signatures.json'):
                    files.add(p / rel)
    for root in roots:
        files.update(root.glob('*/*/build/device_signatures.json'))
    for p in sorted(files):
        if not p.exists():
            continue
        v = json.loads(p.read_text())
        source = p.parent / 'model.prepared.mlir'
        groups.append({'path': str(p), 'sha256': C.file_hash(p),
                       'prepared_source_sha256': C.file_hash(source) if source.exists() else None,
                       'granularity': v.get('granularity'), 'routed': v.get('routed', []),
                       'signatures': v.get('signatures', {}),
                       'qualification': 'structural relevance only; private model evidence excluded from calibration'})
    # Retain every source-derived group identity, but avoid duplicating the owner's
    # large compilation/status payloads in this task's compact audit artifact.
    compact_groups = []
    for group in groups:
        group['routed'] = [{k:v for k,v in entry.items() if k in
                            {'group','parallel','reduction','dtypes','source_region_id','source_node_ids','symbol'}}
                           for entry in group['routed']]
        compact_groups.append(group)
    return {'selected_statuses': [{k:v for k,v in item.items() if k!='metadata'} for item in selected], 'groups': compact_groups,
            'required_workloads': {
                'resnet50': {'batch': 1, 'image': [224,224], 'contractions': 54, 'selected_readout': 'Phase1 dependency'},
                'tiny_llama': {'layers': 22, 'batch': 1, 'prefill': 32, 'decode': 32, 'kv_reuse': True, 'layout': 'actual GQA required'},
                'smolvla': {'batch': 1, 'camera': [1,512,512], 'action_chunk': 50, 'denoise_steps': 10, 'prefix_reuse': True}},
            'no_model_reference_values_read': True, 'formal_qualification': False}


def main():
    with C.stage('capsule_audit'):
        facts = json.loads(Path(C.environment()['MERLIN_RTL_FACTS']).read_text())
        machine = machine_from_facts('gemmini', facts=facts, measure_fill=False)
        ports = movement_peak(facts)
        rows = []
        guards = json.loads((C.CORPUS / 'coverage/phase2-functional-guards.json').read_text())
        guard_ids = {x['member'] for x in guards['guards']}
        for family in ('isa', 'layers', 'model_slices'):
            for source in sorted((C.CORPUS / 'capsules' / family).glob('*/capsule.interface.mlir')):
                identity = source.parent.relative_to(C.CORPUS / 'capsules').as_posix()
                desc = yaml.safe_load((source.parent / 'capsule.yaml').read_text())
                rows.append(inventory_row(identity, source, desc, 'functional_guard' if identity in guard_ids else 'public_exploratory_witness', machine, ports))
        deferred_path = C.CORPUS / 'capsules/phase2-deferred.json'
        deferred = json.loads(deferred_path.read_text())
        for entry in deferred['entries']:
            identity = '_perf/' + entry['name']
            try:
                desc, text = build(entry, binding())
                desc['performance'] = entry['performance']
                desc['source_reference'] = entry.get('source_reference')
                source = C.OUT / 'policy-materializations' / entry['name'] / 'capsule.interface.mlir'
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(text)
                row = inventory_row(identity, source, desc, 'deferred_policy_exploratory_materialization', machine, ports)
                row['declaration'] = entry
                row['source_declaration_sha256'] = C.digest(entry)
            except Exception as error:
                row = {'identity': identity, 'cohort': 'deferred_unmaterialized', 'declaration': entry,
                       'source_declaration_sha256': C.digest(entry), 'source_sha256': None,
                       'family': entry['performance']['family'], 'purpose': entry.get('source_reference'),
                       'derived_shapes': {k:entry[k] for k in ('M','K','N') if k in entry},
                       'baseline': {'status':'refused_or_unmaterialized', 'reason':str(error)},
                       'roofline': {'status':'UNKNOWN'}, 'formal_admission': False,
                       'expected_evaluation_cost': None, 'required_accelerator_operations': [entry['op']],
                       'numerical_contract': 'selected declaration; not executed', 'resource_pressure': None}
            rows.append(row)
        manifest = {'schema':'phase2.exploratory.capsule_manifest.v1', 'formal_admission':False,
                    'policy_sha256':C.file_hash(deferred_path), 'facts_sha256':C.file_hash(C.environment()['MERLIN_RTL_FACTS']),
                    'frozen_objective': 'all deferred declarations retained; no aggregate promotion before admission',
                    'members':rows}
        manifest['members_sha256'] = C.digest(rows)
        C.save(C.EVIDENCE / 'capsule-manifest.json', manifest)
        model_groups = model_inventory()
        C.save(C.EVIDENCE / 'full-model-groups.json', model_groups)
        axes = {
            'capacity_knees': ['knee'], 'deep_k_panels':['deep_k'], 'decode_bandwidth':['short_m'],
            'independent_tails':['tails'], 'convolution_geometry':['conv'], 'readout':['readout'],
            'connected_residency':['chain'], 'queue_pressure':['queue'], 'dataflow':['dataflow'],
        }
        coverage = []
        for axis, families in axes.items():
            matches = [x for x in rows if x.get('declaration',{}).get('performance',{}).get('capacity',{}).get('family') in families]
            coverage.append({'axis':axis, 'members':[x['identity'] for x in matches],
                             'status':'declared_not_qualified', 'gaps':['no admitted performance cohort', 'no capacity-scale timing qualification']})
        for axis, reason in {
            'attention_prefill_decode_actual_GQA':'public witnesses are not source-faithful full TinyLlama GQA/prefix/decode connected groups',
            'grouped16_or_column_readout':'preserved compiler lacks varying column gains; Phase1 owns selected recipe/readout closure',
            'configuration_fences_overlap_banks':'queue declarations do not establish legal overlap or bank-conflict coverage',
            'full_model_connected_groups':'source metadata inventory is not materialized public connected capsules for all three full models',
        }.items():
            coverage.append({'axis':axis,'members':[], 'status':'gap', 'gaps':[reason]})
        shape_witnesses = collections.defaultdict(list)
        for member in rows:
            for shape in member.get('derived_shapes', []) if isinstance(member.get('derived_shapes'), list) else []:
                if all(k in shape for k in ('M','K','N')):
                    shape_witnesses[(shape['M'],shape['K'],shape['N'])].append(member['identity'])
        model_coverage = []
        for source in model_groups['groups']:
            by_shape = collections.defaultdict(list)
            for group in source['routed']:
                p, r = group.get('parallel',[]), group.get('reduction',[])
                if len(p)==2 and len(r)==1:
                    by_shape[(p[0],r[0],p[1])].append(group['source_region_id'])
            model_coverage.append({'source':source['path'],'source_sha256':source['sha256'],
                'shapes':[{'M':key[0],'K':key[1],'N':key[2], 'source_groups':ids,
                           'exact_shape_witnesses':shape_witnesses.get(key,[]),
                           'connected_layout_contract_covered':False,
                           'gap':'shape equality does not establish layout/readout/connected-boundary equivalence'}
                          for key,ids in sorted(by_shape.items())]})
        C.save(C.EVIDENCE / 'coverage-matrix.json', {'schema':'phase2.exploratory.coverage.v1',
               'manifest_sha256':C.file_hash(C.EVIDENCE/'capsule-manifest.json'), 'rows':coverage,
               'functional_guards':guards, 'full_model_shape_coverage':model_coverage,
               'source_basis_limit':'Phase0 performance-basis covers five small derivation applications, not full intended workloads',
               'counts':dict(collections.Counter(x['cohort'] for x in rows))})
        print(json.dumps({'members':len(rows), 'counts':dict(collections.Counter(x['baseline']['status'] for x in rows)),
                          'materializations':str(C.OUT/'policy-materializations')}, indent=2))


if __name__ == '__main__':
    main()
