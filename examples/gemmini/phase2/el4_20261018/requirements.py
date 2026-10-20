"""Exact full-model device-interface requirements, joined to all source groups.

This reads structural metadata and interface IR, never weights or reference
values. Requirements are separate from the frozen public performance objective.
"""
import collections
import json
from pathlib import Path

import common as C
C.configure()
import audit
from merlin.perf.derived_bound import machine_from_facts
from merlin.perf.movement_peak import movement_peak


def main():
    with C.stage('full_model_requirements_audit'):
        facts=json.loads(Path(C.environment()['MERLIN_RTL_FACTS']).read_text())
        machine=machine_from_facts('gemmini',facts=facts,measure_fill=False)
        ports=movement_peak(facts)
        public=[]
        for name in ('capsule-manifest.json','stress-manifest.json'):
            public.extend(json.loads((C.EVIDENCE/name).read_text())['members'])
        by_source=collections.defaultdict(list)
        for member in public:
            if member.get('source_sha256'):
                by_source[member['source_sha256']].append(member['identity'])
        rows=[]
        selections={
            'tiny_llama/prefill':('tiny32-prefill',1),
            'tiny_llama/decode':('tiny32-decode',32),
            'smolvla/prefix_encode':('smolvla-prefix_encode',1),
            'smolvla/flow_denoise':('smolvla-flow_denoise',10),
            'smolvla/action_decode':('smolvla-action_decode',1),
        }
        program_rows=[]
        for program,(build_name,steps) in selections.items():
            build=C.P1/'out/build/functional-phase1-baseline-v1'/build_name/'build'
            signature_path=build/'device_signatures.json'
            if not signature_path.exists():
                program_rows.append({'program':program,'steps':steps,'accelerator_groups':0,
                    'purpose':'host action decoding remains part of full model, not a CPU-only capsule'})
                continue
            signatures=json.loads(signature_path.read_text())
            status_path=C.P1/f'.tmp/handoff/functional-phase1-baseline-v1/{build_name}-compile-v2/status.json'
            status=json.loads(status_path.read_text())
            assert status['candidate_tree']['sha256']==C.initialize()['baseline_sha256']
            program_rows.append({'program':program,'steps':steps,'accelerator_groups':len(signatures['routed']),
                'device_signature_sha256':C.file_hash(signature_path),
                'compile_status_sha256':C.file_hash(status_path),
                'signature_package_sha256':signatures.get('package_sha256'),
                'package_binding_limit':'original owner status pins compiler bytes; signature package tag is not a formal certification',
                'prepared_source_sha256':C.file_hash(build/'model.prepared.mlir'),
                'unique_interfaces':len(signatures['signatures'])})
            for symbol in sorted(signatures['signatures']):
                source=build/'device'/f'{symbol}.lowering/input.interface.mlir'
                identity=f'model-requirement/{program}/{symbol}'
                row=audit.inventory_row(identity,source,
                    {'source_reference':'source-derived selected full-model device interface; independent signed public execution required'},
                    'full_model_requirement_not_public_admission',machine,ports)
                assert row.get('required_accelerator_operations'),identity
                source_groups=[{k:v for k,v in route.items() if k in {'group','source_region_id','source_node_ids','parallel','reduction','dtypes'}}
                    for route in signatures['routed'] if route['symbol']==symbol]
                row.update(program=program,model_steps=steps,source_groups=source_groups,
                    calls_per_program_step=len(source_groups),calls_in_requested_chain=len(source_groups)*steps,
                    exact_public_source_witnesses=by_source.get(row['source_sha256'],[]),
                    connected_contract={'status':'gap','reason':'single device interface does not capture host packing/layout, GQA/KV or prefix reuse and connected dispatcher boundaries'},
                    baseline_chain={'status':'retained_execution_available_provisional',
                        'path':str(C.P1/f'.tmp/handoff/functional-phase1-baseline-v1/{program.split("/")[0]}-full-readback-spike-v1/status.json'),
                        'reference_equivalence':False},
                    public_measurement='UNKNOWN unless joined to an identity-matched fresh public result')
                rows.append(row)
        # The selected ResNet requirement is explicit even though its varying
        # readout contract is not executable through this frozen compiler.
        gaps=[{'program':'resnet50/forward','batch':1,'image':[224,224],
            'source_groups_required':54,'status':'blocked_phase1_readout',
            'reason':'selected fused path needs block-dependent scalar or grouped16 readout; historical 1000-output recipe is a different boundary',
            'next_owner':'Phase1 selected readout/recipe admission, no recapture in this campaign'},
            {'program':'tiny_llama/full_chain','status':'connected_capsule_gap',
             'reason':'all22 layers, 32-token prefill +32 decode, actual GQA/KV layout must remain a connected validation boundary'},
            {'program':'smolvla/full_chain','status':'connected_capsule_gap',
             'reason':'one512 camera, action50, prefix reuse and ten denoising steps need connected validation; approximation is not equivalence'}]
        result={'schema':'phase2.full_model_capsule_requirements.v1',
            'frozen_performance_objective_changed':False,'formal_admission':False,
            'public_manifest_sha256':C.file_hash(C.EVIDENCE/'capsule-manifest.json'),
            'programs':program_rows,'members':rows,'gaps':gaps,
            'counts':{'unique_selected_device_interfaces':len(rows),
                'routed_source_groups':sum(len(x['source_groups']) for x in rows),
                'requested_chain_accelerator_calls':sum(x['calls_in_requested_chain'] for x in rows),
                'interfaces_with_exact_public_source_witness':sum(bool(x['exact_public_source_witnesses']) for x in rows)}}
        C.save(C.EVIDENCE/'full-model-capsule-requirements.json',result)
        print(json.dumps(result['counts']))


if __name__=='__main__':main()
