"""Use the original v3 intake; retain honest UNKNOWN results for incomplete timing."""
from __future__ import annotations

import collections
import json
from pathlib import Path
import tarfile

import common as C
C.configure()
from merlin_timing import archive as A, contracts as T, fitting as F, validation as V
from merlin_experiments.phase1.annotated_spike_evidence import review_elf
from evaluate import OBJDUMP

ARCHIVE = C.P1/'merlin/out/runs/gemmini/phase1/el4-fire-aligned-phase1-20261017-v3/capsule_timing_training_d4c730b33e12.tar'
ARCHIVE_SHA = 'f815a0d2f4d80415b1e89e6ccdd62a0eb7af039287f539c91bd235451f8cfd5a'


def main():
    with C.stage('calibration_archive_intake'):
        inventory = A.inspect_archive(ARCHIVE, ARCHIVE_SHA)
        root = C.OUT/'calibration'
        C.save(root/'archive-inventory.json', inventory)
        report_status, unresolved = collections.Counter(), collections.Counter()
        intake_gaps = []
        with tarfile.open(ARCHIVE) as archive:
            def artifact(row, role):
                reference = row['artifacts'].get(role)
                return json.load(archive.extractfile('run/'+reference['path'])) if reference else None
            rows = [x['row'] for x in inventory['records']]
            first_report = next(artifact(r,'annotated_spike_report') for r in rows if r['artifacts'].get('annotated_spike_report'))
            selection = rows[0]['timing_selection']
            pins = selection['counter_semantics']['source_pins']
            # The original selected CSR.scala implementation pins accessor, mcycle semantics and clock.
            # Do not substitute another ISA document or simulator identity.
            source_digest = pins[0]['sha256']
            assert C.file_hash(pins[0]['path']) == source_digest
            review = {
                'timing_selection': selection, 'cycle_scope':rows[0]['cycle_scope'],
                'sim_provenance_sha256':T.digest(rows[0]['sim_provenance']), 'cache_protocol':None,
                'counter':{'unit':'cycles','width_bits':64,'wrap':'modulo',
                           'clock':selection['counter_semantics']['clock'],
                           'max_delta_cycles':selection['max_delta_cycles'],
                           'sources':{role:source_digest for role in ('accessor','isa','clock')}}}
            by_case = {}
            reports_by_result = {}
            for row in rows:
                source_sha = row['artifacts']['input_interface']['sha256']
                group = source_sha  # All exact source variants/repeats remain together.
                split = ('train','train','train','calibration','test')[int(group[:8],16)%5]
                cb = artifact(row,'command_buffer')
                ops = {x['opcode'] for x in cb.get('commands',[])}
                family = ('conv' if 'CONV2D' in ops else 'attention' if ops & {'ATTENTION_QK','ATTENTION_PV','BATCHED_MATMUL'}
                          else 'residual' if 'RESIDUAL_ADD' in ops else 'movement' if ops == {'MOVEMENT'}
                          else 'resident_chain' if sum(x['opcode']=='COMMIT' for x in cb.get('commands',[]))>1 else 'matmul')
                case = {'case_id':T.digest([row['elf_sha256'],row['harness_sha256'],source_sha]),
                        'source_group_sha256':group, 'mechanism_family':family,'split':split,
                        'source':{'visibility':'public','phase':'phase1','manifest_sha256':inventory['index_sha256'], 'source_sha256':source_sha},
                        'elf_sha256':row['elf_sha256'],
                        'input_sha256':T.digest({'schema':'merlin.embedded_input_context.v1','harness_sha256':row['harness_sha256']}),
                        'scope_sha256':first_report['domain']['scope_sha256'], 'device_sha256':first_report['domain']['device_sha256']}
                case['workload_sha256']=T.workload(case)
                native = artifact(row,'annotated_spike_report')
                if native:
                    reports_by_result[row['result']]={'sha256':T.digest(native),'report':native}
                    report_status['complete' if native.get('timing_complete') else 'incomplete']+=1
                    unresolved.update(native.get('unresolved',[]))
                else:
                    report_status['missing']+=1
                counter = artifact(row,'annotated_spike_counter_review')
                if counter is None:
                    elf = C.TASK/'.tmp'/('archive-'+row['elf_sha256']+'.elf')
                    if not elf.exists():
                        elf.write_bytes(archive.extractfile('run/'+row['elf']).read())
                    assert C.file_hash(elf)==row['elf_sha256']
                    try:
                        counter = review_elf(elf,objdump=OBJDUMP,counter_asm='csrr %0, mcycle',entry_symbol='kernel_entry',completion_symbol='kernel_complete')
                    except Exception as error:
                        intake_gaps.append({'result':row['result'],'elf_sha256':row['elf_sha256'],
                                            'reason':'original counter window review unavailable: '+str(error)})
                        continue
                scope = {k:counter[k] for k in ('begin_pc','end_pc','occurrence')}
                scope['evidence_sha256']=T.digest(counter)
                key = case['case_id']
                bound = by_case.setdefault(key,{'case':case,'results':[], 'candidate_sha256':[],
                                               'shape_sha256':T.digest(row['shape']),'counter_scope':scope})
                assert bound['case']==case
                if bound['counter_scope']!=scope:
                    intake_gaps.append({'result':row['result'],'elf_sha256':row['elf_sha256'],
                                        'reason':'same execution has distinct original static-review receipt; replay required before attachment'})
                bound['results'].append(row['result'])
                bound['candidate_sha256'].append(row['candidate_sha256'])
            for row in by_case.values():
                row['candidate_sha256']=sorted(set(row['candidate_sha256']))
            binding = {'schema':A.BINDING,'archive_sha256':ARCHIVE_SHA,'domain':first_report['domain'],
                       'feature_names':['cpu.integer','cpu.load','cpu.store','cpu.branch','target.commands','target.requested_read_bytes'],
                       'review':review,'cases':list(by_case.values())}
            C.save(root/'reviewed-binding.json',binding)
            try:
                plan = A.prepare(inventory,binding)
            except Exception as error:
                summary={'schema':'phase2.exploratory.calibration_status.v1','archive_sha256':ARCHIVE_SHA,
                         'inventory':{k:v for k,v in inventory.items() if k!='records'},
                         'source_groups':len({x['case']['source_group_sha256'] for x in by_case.values()}),
                         'cases':len(by_case),'reports':dict(report_status),'unresolved':dict(unresolved),
                         'intake_gaps':intake_gaps,'archive_prepare':{'status':'refused','reason':str(error)},
                         'archive_attach':{'status':'not_applicable','reason':'no complete observation plan admitted'},
                         'prediction_authority':'NONE', 'empirical_validation':{'status':'UNKNOWN'},
                         'native_validation':{'status':'UNKNOWN'},
                         'capacity_extrapolation':'UNKNOWN; original reports have no event predictions and timing model has unbound resources/prices',
                         'artifacts':{str(p.relative_to(root)):C.file_hash(p) for p in root.glob('*.json')},
                         'dependency':'original owner must supply all reviewed counter windows and complete event resource/price domain; no subsets or invented bindings'}
                C.save(C.EVIDENCE/'calibration-status.json',summary)
                print(json.dumps({k:summary[k] for k in ('reports','archive_prepare','prediction_authority')},indent=2))
                return
            C.save(root/'observation-plan.json',plan)
            attachment = {'schema':'merlin.phase1.spike_observations.v1','plan_sha256':T.digest(plan),
                          'reports':{x['record_id']:reports_by_result[x['result']] for x in plan['observations'] if x['result'] in reports_by_result}}
            C.save(root/'original-observations.json',attachment)
            dataset = A.attach(plan,attachment)
            C.save(root/'dataset.json',dataset)
            native_validation = V.validate(dataset,prediction_kind='event_timing')
            C.save(root/'event-validation.json',native_validation)
            try:
                screen = F.fit(dataset)
                C.save(root/'screen.json',screen)
                empirical = V.validate(dataset,screen)
                C.save(root/'screen-validation.json',empirical)
            except Exception as error:
                empirical = {'status':'UNKNOWN','reason':str(error)}
            summary={'schema':'phase2.exploratory.calibration_status.v1', 'archive_sha256':ARCHIVE_SHA,
                     'inventory':{k:v for k,v in inventory.items() if k!='records'},
                     'source_groups':len({x['case']['source_group_sha256'] for x in by_case.values()}),
                     'cases':len(by_case),'reports':dict(report_status), 'unresolved':dict(unresolved),
                     'native_validation':native_validation,'empirical_validation':empirical,
                     'prediction_authority':'NONE unless all original gates actually pass',
                     'capacity_extrapolation':'UNKNOWN: original event model has unbound resources/prices; no qualified native predictions',
                     'artifacts':{str(p.relative_to(root)):C.file_hash(p) for p in root.glob('*.json')},
                     'cost_policy':'host-only intake; no copied verdict relabeled as a fresh simulation'}
            C.save(C.EVIDENCE/'calibration-status.json',summary)
            print(json.dumps({k:summary[k] for k in ('cases','source_groups','reports','empirical_validation','prediction_authority')},indent=2))


if __name__=='__main__':
    main()
