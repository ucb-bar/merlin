"""Emit the same production driver for continuous and admitted single phases."""
import hashlib
import json
from smol_full_phases import extra_live, segments


def emit(source, parameters, references, vocabulary, reference, read, asset, *, profile, fusion_set='none'):
    evidence = json.loads(read(reference))
    if evidence['status'] != 'PASS_NATIVE_PHASE_REFERENCE' or evidence['mode'] != 'phased':
        raise ValueError('Require actual phased native reference preparation')
    inventory = extra_live(profile)
    inventory.update((r['actual_symbol'], r['bytes']) for r in references)
    boundaries = [evidence['initial_boundary'], *evidence['boundaries']]
    if len(boundaries) != 201: raise ValueError('Complete boundary set required')
    data = [source, '#include "smol_full_driver.h"']
    data += ['const smol_full_phase_descriptor smol_full_phase_plan[200]={']
    for p in segments(profile):
        values = [{'vision':0,'prefix':1,'expert':2}[p['kind']], *[p[k] for k in
            ('ordinal','layer','launch_start','launch_end','call_start','call_end')], 0]
        data.append('{'+','.join(map(str, values))+'},')
    data += ['};', 'const smol_full_phase_region smol_full_phase_regions[56]={']
    data += [f'{{(unsigned char*)&{name},{size},"{name}"}},' for name,size in inventory.items()]
    data.append('};')
    labels, rows, bindings = {}, [], []
    for boundary in boundaries:
        if set(boundary['regions']) != set(inventory): raise ValueError('Reference live inventory differs')
        row, bound = [], {}
        for name,size in inventory.items():
            item = boundary['regions'][name]
            raw = read(reference.parent/item['file'], item['sha256'])
            if len(raw) != size or item['bytes'] != size: raise ValueError('Reference extent differs')
            digest = hashlib.sha256(raw).hexdigest()
            relative = 'segments/blobs/'+digest+'.bin'
            if digest not in labels:
                label = 'smol_phase_blob_'+digest
                labels[digest] = label
                asset(raw, label, relative)
                data.append('extern const unsigned char '+label+'[];')
            row.append(labels[digest]); bound[name]=relative
        rows.append('{'+','.join(row)+'},'); bindings.append(bound)
    data += ['const unsigned char *const smol_full_phase_expected[201][56]={', *rows, '};']
    def p(name, kind): return '(const '+kind+'*)'+parameters['observation.'+name]['symbol']
    data += ['const smol_full_driver_inputs smol_full_driver_observation={'+','.join([
        p('images','float'),p('state','float'),p('noise','float'),p('image_masks','uint8_t'),
        p('language_mask','uint8_t'),p('language_tokens','int64_t'),str(vocabulary)])+'};',
        'static unsigned segment_mode,segment_start;',
        'volatile uint64_t smol_full_segment_self_restore=0;',
        '#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING',
        'static uint64_t smol_timing_start,smol_timing_start_instret;static inline uint64_t smol_timing_rdcycle(void){uint64_t ticks;__asm__ volatile("csrr %0,0xc00":"=r"(ticks)::"memory");return ticks;}',
        'static inline uint64_t smol_timing_rdinstret(void){uint64_t ticks;__asm__ volatile("csrr %0,0xc02":"=r"(ticks)::"memory");return ticks;}',
        'static void smol_timing_total(void){uint64_t atlas=0;for(unsigned i=0;i<'+str(profile.launch_count)+';++i)atlas+=(segment_mode?smol_full_segment_observations[i]:smol_full_cycles[i]);atlas_result_text(&smol_result,"ATLAS_SMOL_TOTAL launches=");atlas_result_number(&smol_result,smol_full_host.invocation_sequence-segment_start);atlas_result_text(&smol_result," atlas_cycles=");atlas_result_number(&smol_result,atlas);atlas_result_text(&smol_result," host_rdcycle=");uint64_t instret=smol_timing_rdinstret()-smol_timing_start_instret,ticks=smol_timing_rdcycle()-smol_timing_start;atlas_result_number(&smol_result,ticks);atlas_result_text(&smol_result," host_instret=");atlas_result_number(&smol_result,instret);atlas_result_text(&smol_result,"\\n");}',
        'void smol_full_report_launch_ticks(size_t index,uint64_t ticks){atlas_result_text(&smol_result,"ATLAS_SMOL_HOST_LAUNCH index=");atlas_result_number(&smol_result,index);atlas_result_text(&smol_result," host_rdcycle=");atlas_result_number(&smol_result,ticks);atlas_result_text(&smol_result,"\\n");}',
        '#endif',
        ('int smol_full_driver_vision_launch(void *w,unsigned layer,unsigned role,const uint16_t '
         '*input,size_t rows,size_t k,uint16_t *output,size_t n){int code=smol_policy_raw_vision(w,lay'
         'er,role,input,rows,k,output,n);if(!code&&!segment_mode&&(role==6||role==7))atlas_result_prog'
         'ress(&smol_result,smol_full_raw.panel_cursor+smol_full_raw.attention_cursor,role==7?"connector":"vision-layer");return '
         'code;}'),
        ('int smol_full_driver_policy_launch(void *w,unsigned site,unsigned layer,unsigned '
         'role,unsigned dtype,const void *input,size_t rows,size_t k,unsigned output_dtype,void '
         '*output,size_t n){int code=smol_policy_raw_graph(w,site,layer,role,dtype,input,rows,k,output'
         '_dtype,output,n);if(!code&&!segment_mode&&(((site==5||site==6)&&role==6)||site==4))atlas_res'
         'ult_progress(&smol_result,smol_full_raw.panel_cursor+smol_full_raw.attention_cursor,site==4?"step":site==5?"vlm-layer":"exp'
         'ert-layer");return code;}'),
        ('static void report_init(unsigned expected){atlas_result_uart_init(&smol_result_uart);atlas_r'
         'esult_begin(&smol_result,atlas_result_uart_putchar,&smol_result_uart,segment_mode?"smol-segm'
         'ent":"smol-full",expected);}'),
        ('static void fail(unsigned code){if(!smol_result.putchar)report_init('+str(profile.launch_count)+');smol_result.comple'
         'ted=smol_full_raw.panel_cursor+smol_full_raw.attention_cursor-segment_start;atlas_result_fail(&smol_result,smol_full_segmen'
         't_error_region?smol_full_segment_error_region:"host-check",code,0);atlas_workload_mailbox[1]'
         '=3;atlas_workload_mailbox[8]=code;__asm__ volatile("fence '
         'rw,rw":::"memory");tohost=((uint64_t)code<<1)|1;smol_full_segment_failed();}'),
        ('void atlas_block_trap_report(uint64_t cause,uint64_t pc,uint64_t value){atlas_workload_mailb'
         'ox[12]=cause;atlas_workload_mailbox[13]=pc;atlas_workload_mailbox[14]=value;fail(255);}'),
        'void atlas_workload_main(void){',
        '#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING',
        'smol_timing_start_instret=smol_timing_rdinstret();smol_timing_start=smol_timing_rdcycle();',
        '#endif',
        'memset(&smol_result,0,sizeof(smol_result));segment_mode=0;segment_start=0;smol_full_segment_error_region=0;',
        ('memset((void*)atlas_workload_mailbox,0,sizeof(atlas_workload_mailbox));atlas_workload_mailbo'
         'x[0]=UINT64_C(0x534d4f4c46554c4c);tohost=0;fromhost=0;'),
        'memset(smol_full_segment_observations,0,sizeof(smol_full_segment_observations));',
        ('smol_full_raw.attention_cursor=0;smol_full_raw.panel_cursor=0;smol_full_raw.call_cursor=0;smol_full_raw.failed=0;smol_full_ra'
         'w.result=0;memset(&smol_full_raw.active_buffer,0,sizeof(smol_full_raw.active_buffer));'),
        'atlas_host_map map={(uintptr_t)ATLAS_IMEM_BASE,(uintptr_t)ATLAS_CSR_BASE,ATLAS_IMEM_BYTES};',
        'atlas_host_mmio mmio={atlas_host_direct_read32,atlas_host_direct_write32,0};atlas_host_memory memory={1,0,0};',
        'if(atlas_host_init(&smol_full_host,map,mmio,memory)!=ATLAS_HOST_OK){fail(1);return;}',
        ('smol_attention_configure(smol_policy_raw_attention,&smol_full_raw);smol_full_segment_ready();' if profile.attention_count else 'smol_attention_configure(0,0);smol_full_segment_ready();'),
        'uint64_t selected=smol_full_segment_request;unsigned first=0,end=200,expected='+str(profile.launch_count)+',calls=1421;',
        ('if(selected!=UINT64_MAX){segment_mode=1;if(selected>=200){fail(10);return;}first=(unsigned)s'
         'elected;end=first+1;const smol_full_phase_descriptor *p=&smol_full_phase_plan[first];segment'
         '_start=p->launch_start;expected=p->launch_end-p->launch_start;calls=p->call_end-p->call_star'
         't;smol_full_raw.attention_cursor='+('first' if profile.attention_count else '0')+';smol_full_raw.panel_cursor=p->launch_start'+('-first' if profile.attention_count else '')+';smol_full_raw.call_cursor=p->call_start;smol_fu'
         'll_host.invocation_sequence=p->launch_start;}'),
        'report_init(expected);atlas_workload_mailbox[1]=1;',
        'if(*(const volatile uint64_t*)smol_full_compiled_phase_layout!=1){fail(12);return;}',
        ('for(unsigned i=0;i<48;++i)if(smol_full_driver_observation.language_tokens[i]<0||(uint64_t)sm'
         'ol_full_driver_observation.language_tokens[i]>=smol_full_driver_observation.vocabulary){fail'
         '(7);return;}'),
        'if(segment_mode&&smol_full_segment_self_restore==1)for(unsigned i=0;i<56;++i)memcpy(smol_full_phase_regions[i].actual,smol_full_phase_expected[first][i],smol_full_phase_regions[i].bytes);',
        'if(smol_full_driver_input(first)){fail(11);return;}',
        ('for(unsigned phase=first;phase<end;++phase){int '
         'code=smol_full_driver_phase(phase,segment_mode);if(code){fail(16+(unsigned)code);return;}}'),
        'if(segment_mode){atlas_result_progress(&smol_result,expected,"layer");}else{']
    for i,r in enumerate(references):
        data.append(f'if(smol_full_driver_differ({r["actual_symbol"]},{r["expected_symbol"]},{r["bytes"]}))'
                    f'{{smol_full_segment_error_region="{r["name"]}";fail({32+i});return;}}')
    data += ['for(unsigned i=0;i<'+str(profile.launch_count)+';++i)if(!smol_full_cycles[i]){fail(6);return;}', '}',
        'if(!smol_full_driver_drained()){fail(5);return;}',
        ('atlas_workload_mailbox[2]=expected;atlas_workload_mailbox[3]=calls;atlas_workload_mailbox[4]'
         '=smol_full_graph_state.prefix_ready;atlas_workload_mailbox[5]=smol_full_graph_state.steps;at'
         'las_workload_mailbox[6]=1;atlas_workload_mailbox[9]=smol_full_raw.panel_cursor;atlas_workloa'
         'd_mailbox[10]=smol_full_raw.call_cursor;atlas_workload_mailbox[11]=smol_full_host.invocation'
         '_sequence;atlas_workload_mailbox[1]=2;'),
        '#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING', 'smol_timing_total();', '#endif',
        'if(!atlas_result_pass(&smol_result)){fail(8);return;}__asm__ volatile("fence rw,rw":::"memory");tohost=1;smol_full_segment_done();', '}']
    if fusion_set != 'none':
        from smol_fusion_assets import SETS
        if fusion_set not in SETS:raise ValueError('Unsupported driver fusion profile')
        marker='if(atlas_host_init(&smol_full_host,map,mmio,memory)!=ATLAS_HOST_OK){fail(1);return;}'
        data.insert(data.index(marker)+1,'if((smol_full_segment_request==UINT64_MAX||smol_full_segment_request<24)&&smol_fusion_configure(&smol_full_raw,smol_full_fusion_panels,2715)){fail(17);return;}')
    return '\n'.join(data)+'\n', dict(reference_sha256=hashlib.sha256(read(reference)).hexdigest(),
        bindings=bindings, inventory=inventory)


def write_pending_chain(out, args, image, contract, elf_digest, contract_digest, products):
    """Publish only unqualified preparation until the exact new driver is gated."""
    data = args.phase_data
    profile = args.execution_profile
    def write(name, value):
        raw = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
        (out/name).parent.mkdir(parents=True, exist_ok=True)
        (out/name).write_bytes(raw)
        products[name] = hashlib.sha256(raw).hexdigest()
        return products[name]
    receipt_digest = write('segments/receipt.json', dict(
        schema='atlas.smol_boundary_preparation.v1', status='PENDING_NATIVE_DRIVER_QUALIFICATION',
        reference_only=True, guest_execution=False, device_execution=False,
        source_reference_sha256=contract['source_reference_sha256'], segments=200, boundaries=201,
        projection_calls=1421, physical_panels=2715,attention_launches=profile.attention_count,device_events=profile.launch_count,
        semantic_regions=data['inventory']))
    plan = segments(profile)
    boundaries = [dict(index=i, launch_cursor=0 if i==0 else plan[i-1]['launch_end'],regions=regions)
                  for i,regions in enumerate(data['bindings'])]
    write('segments.json', dict(schema='atlas.segment_chain.v1', workload='smol-full',
        qualification='REFERENCE_BOUNDARIES_ONLY', elf='host/host.elf',elf_sha256=elf_digest,
        contract='host/smol-full-contract.json',contract_sha256=contract_digest,
        capture_receipt='segments/receipt.json',capture_receipt_sha256=receipt_digest,
        kernel_count=profile.launch_count,placement=args.physical,
        model_profile=args.model_profile,
        live_regions=[dict(name=name,address=image.symbols[name]['address'],bytes=size)
                      for name,size in data['inventory'].items()],boundaries=boundaries,
        final_outputs=[r['actual_symbol'] for r in contract['reference_buffers']]))
