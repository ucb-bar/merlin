#!/usr/bin/env python3
"""aggregate193.py LABEL PKGID : runs193/LABEL + canon193.json -> RESULT-193-LABEL.md, result-193-LABEL.csv (HEADLINE=1 also RESULT-193.md/.csv)"""
import json,csv,sys,os,re,collections
H=os.environ.get('H193_H', os.environ.get('H', '/scratch2/loren/backend-verify-author')); R=os.environ.get('R', '/scratch/loren/backend-verify-author'); L=sys.argv[1]; PID=sys.argv[2]; RD=f'{H}/runs193/{L}'
S=json.load(open(H+'/canon193.json'))
def load(name):
    p=f'{RD}/{name}/result.json'
    return json.load(open(p)) if os.path.exists(p) else None
def cls(j):
    """-> (result, dev, reason) for one run_case result: ok_dev / ok_host / declined / wrong / crashed / missing"""
    if j is None: return 'missing',None,'no result.json'
    oc=j['outcome']; dev=j.get('device_insns_retired'); static=j.get('static_rocc_insns') or 0; msg=j.get('cause') or ''
    if oc=='ok': return ('ok_dev' if dev else ('ok_host' if not static else 'ok_host_static_dev')),dev,''
    if oc=='declined': return 'declined',dev,msg
    if oc=='wrong': return 'wrong',dev,msg
    if 'gemmini compiler:' in msg: return 'declined',dev,f"{j.get('stage')}: {msg}"
    return 'crashed',dev,f"{j.get('stage')}: {msg}"
def vocab(m):
    m=re.sub(r'^rc=N ','',m)
    k=re.search(r'((?:tensor|linalg|scf|ml_program|arith|math|func|cf)\.[a-z_0-9]+):',m)
    if 'multi-result' in m: return 'multi-result '+(k.group(1) if k else 'op')
    if 'outside proved exact affine integer form' in m: return 'linalg.generic scalar body / indexing map outside the proved integer-affine form'
    if 'reduction result shape does not preserve unreduced' in m: return 'window_sum reduction result shape'
    if 'Unsupported movement container dtype' in m: return 'movement container dtype outside i8/i32'
    if k: return k.group(1)+' (no exact command-buffer lowering)'
    return re.sub(r'\d+','N',m)[:110]
FULL=bool(os.environ.get('FULL'))
def load_full(name):
    p=f'{RD}/{name}/full.json'
    return json.load(open(p)) if os.path.exists(p) else None
rows=[]
for s in S:
    c=cls(load('c_'+s['n'])); f=cls(load('f_'+s['n'])) if s['form_testable'] else None
    r=dict(n=s['n'],overload=s['overload'].replace('aten.',''),klass=s['klass'],expected='device' if s['klass']=='DEVICE57' else 'host',
           input_used='',result='',ok='',device_insns='',reason='',f32_capture_result=c[0],f32_capture_reason=c[2][:200],int_form=s['form_slug'] if s['form_testable'] else '',elf='')
    if s['klass']=='DEVICE57':
        if f: res,dev,why=f; r['input_used']={'SPEC_INT8':'spec_int8 form (f32 capture, exact vs the declared quantized model)','SPEC_FORM':'spec form (corpus builder at the declared form, exact)'}.get(s['form_lane'],'int8 form'); src='f_'+s['n']
        else: res,dev,why=c; r['input_used']='f32 capture (no integer form: '+s['form_lane']+')'; src='c_'+s['n']
        r['result']={'ok_dev':'lowered-on-device','ok_host':'lowered-on-host (device expected)','ok_host_static_dev':'lowered-on-host (device expected)'}.get(res,res)
        r['ok']='PASS' if res=='ok_dev' else 'FAIL'
    else:
        res,dev,why=c; r['input_used']='f32 capture'; src='c_'+s['n']
        if res=='ok_host': r['result']='lowered-on-host'; r['ok']='PASS'
        elif res in('ok_dev','ok_host_static_dev'): r['result']='wrong'; why='HOST-ONLY overload emitted accelerator opcodes (retired %s, static %s)'%(dev,(load(src) or {}).get('static_rocc_insns')); r['ok']='FAIL'
        else: r['result']=res; r['ok']='FAIL'
    if s['klass']=='DEVICE57' and r['result'].startswith('lowered-on-host'):
        hj=load(src) or {}; why='compiler ran it all on the host: %s, 0 RoCC retired, output exact (did not lower on device)'%('empty command buffer (whole-program host ELF)' if hj.get('host_only_program') else 'no retired RoCC')
    r['device_insns']='' if dev is None else dev; r['reason']=why[:220]
    if os.path.exists(f'{RD}/{src}/result.json'): r['elf']=f'compute-1:{R}/runs193/{L}/{src}/prog.elf'
    r.update(meets_full_declared_host_ok='',numerical='',host_compute='',host_ops_per_element='',host_value_ops_per_element='',host_payload_ratio='',host_compute_why='',trace_check='',trace_requirement='',trace_reason='',meets_full='')
    if FULL:
        fj=load_full(src)
        if s['klass']=='DEVICE57':
            r['numerical']='pass' if res=='ok_dev' else ('fail' if res in('ok_host','ok_host_static_dev','wrong') else 'fail (no program: %s)'%res)
            if fj is None: r['host_compute']=r['trace_check']='n/a (no emitted program)'; r['meets_full']='NO'
            else:
                hc=fj.get('host_compute') or {}; tc=fj.get('trace') or {}; v=hc.get('verdict')
                r['host_compute']={'clean':'pass','host_compute_in_accelerator_group':'fail'}.get(v,'unknown ('+str(v)+')')
                if v=='declared_host': r['host_compute']='not judged (declared_host: grader census outcome %s; the veto does not fire)'%hc.get('program_outcome')
                silent=hc.get('program_outcome')=='silent'
                # Operator decision 2026-11-04: FULL requires a census outcome other than `silent`, whatever the audit verdict.
                if silent: r['host_compute']='fail (census outcome silent: the persisted command buffer declares no command, no lane_placement and no decline; undeclared unit work earns no device credit)'
                r['host_ops_per_element']='' if hc.get('arithmetic_per_element') is None else hc['arithmetic_per_element']
                r['host_value_ops_per_element']='' if hc.get('value_arithmetic_per_element') is None else hc['value_arithmetic_per_element']
                r['host_payload_ratio']='' if hc.get('payload_ratio') is None else hc['payload_ratio']
                r['host_compute_why']=(hc.get('why') or '')[:200]
                trs=tc.get('trace_requirement_status'); st=tc.get('status')
                r['trace_check']='pass' if (st=='pass' and trs in(None,'pass')) else ('unavailable_fields' if trs=='unavailable_fields' else ('unavailable_protocol' if trs=='unavailable_protocol' else 'fail'))
                r['trace_requirement']=str(trs); r['trace_reason']='; '.join(tc.get('violations') or [])[:200]
                r['meets_full']='YES' if (res=='ok_dev' and r['host_compute']=='pass' and r['trace_check']=='pass') else 'NO'
                r['meets_full_declared_host_ok']='YES' if (res=='ok_dev' and not silent and (r['host_compute']=='pass' or v=='declared_host') and r['trace_check']=='pass') else 'NO'
        else:
            r['numerical']='pass (zero accelerator opcodes, correct output)' if r['ok']=='PASS' else 'fail'; r['host_compute']=r['trace_check']='n/a (host group)'; r['meets_full']='n/a'
    rows.append(r)
C=lambda k: collections.Counter(r['result'] for r in rows if r['klass']==k)
d=C('DEVICE57'); h=C('HOST136')
out=[f"# RESULT-193 ({L}): authored backend vs all 193 core-ATen overloads, canonical case; package file-set id {PID}\n",
f"**DEVICE57 (30 exact movement + 27 int8-quantized, ALL57 of tooling/exercised/exercised.py): {d['lowered-on-device']} / 57 lowered on device (accelerator instructions retired, exact output).  HOST136 (100 host-only + 36 declared host-only): {h['lowered-on-host']} / 136 ran on the host path correctly with zero accelerator opcodes.**\n",
"Verification instrument (never an author input). Route per case: parse -> lower_interface_to_target -> emit_target_bundle -> kernel ABI v2 -> clang-23 -> Spike with -l RoCC (0x7b) retirement count (tools/run_case.py). Inputs: DEVICE57 are handed the op-capsule group's int8/integer form (forms93, from Merlin's core_aten_op_capsules registry, exact integer reference) (operator decision 2026-11-04: every DEVICE57 op is graded on a spec-declared form; 47 through the op-capsule registry's builder lanes, cumsum/diagonal/select/split_with_sizes/convolution_backward/upsample_nearest2d/full/full_like/scalar_tensor through the spec form table (tools/gen_spec_forms.py, lane SPEC_FORM), and div.Scalar through its spec_int8 form: the f32 capture compared EXACTLY with the software spec's declared quantized model, tools/gen_spec_int8_form.py); HOST136 are handed Merlin's own corpus_193 canonical f32 capture (f32 rounding tolerance for floats, exact for ints). Column f32_capture_result in the CSV gives the outcome of the corpus_193 f32 capture for every overload (what Merlin's own backend consumes). Classes: lowered-on-device / lowered-on-host / declined / wrong / crashed. A 'wrong' on HOST136 includes any accelerator opcode.\n",
"## Summary\n\n| class | n | lowered-on-device | lowered-on-host | declined | wrong | crashed | other |\n|---|---|---|---|---|---|---|---|"]
for k,cn in (('DEVICE57',d),('HOST136',h)):
    oth=sum(v for kk,v in cn.items() if kk not in('lowered-on-device','lowered-on-host','declined','wrong','crashed'))
    out.append(f"| {k} | {sum(cn.values())} | {cn['lowered-on-device']} | {cn['lowered-on-host']} | {cn['declined']} | {cn['wrong']} | {cn['crashed']} | {oth} |")
out.append("\n(DEVICE57 'lowered-on-host (device expected)' is counted under other.)\n\n## Declines / failures by reason\n\n| class | result | reason | n | overloads |\n|---|---|---|---|---|")
cl=collections.defaultdict(list)
for r in rows:
    if r['ok']=='FAIL': cl[(r['klass'],r['result'],vocab(r['reason']))].append(r['overload'])
for (k,res,v),ov in sorted(cl.items(),key=lambda kv:(kv[0][0],-len(kv[1]))): out.append(f"| {k} | {res} | {v.replace('|','/')} | {len(ov)} | {', '.join(ov)} |")
out.append("\n## Per overload\n\n| # | overload | class | input | result | device insns | reason |\n|---|---|---|---|---|---|---|")
for r in sorted(rows,key=lambda r:(r['klass'],r['ok']!='PASS',r['overload'])): out.append(f"| {r['n']} | {r['overload']} | {r['klass']} | {r['input_used'][:40]} | {r['result']} | {r['device_insns']} | {r['reason'][:120].replace('|','/')} |")
if FULL:
    dv=[r for r in rows if r['klass']=='DEVICE57']; Y=lambda f: sum(1 for r in dv if f(r))
    weak=Y(lambda r:r['ok']=='PASS'); full=Y(lambda r:r['meets_full']=='YES')
    out.append(f"\n## FULL definition ('supported' = numerical AND host-compute AND instruction-trace)\n\n**DEVICE57: {weak}/57 weak pass (RoCC retired + exact output) -> {full}/57 meets the full definition.**  HOST136: n/a (kept as zero accelerator opcodes + correct output: {h['lowered-on-host']}/136); host-compute and trace checks do not apply.\n")
    out.append("Checks (the grader's own functions, run unmodified in the FIN6yn neutral release namespace: operand-selection sidecar e5f2486b, observer active): host-compute = offload_census._source_host_compute -> host_compute_audit.audit_group, budget 1.0 host arithmetic ops/element (also 0.25 float/convert ops/element, 0.5 scalar-payload/element-byte), category host_compute_in_accelerator_group; instruction-trace = rocc.decode.decode_module(lowered LLVM) -> trace_check.check(trace, capsule.expected, cb, address_model=pointer_args, subsumes, target, lowered_module) with the op-capsule registry's own `expected` modes (the grader's trace is the static decode of the emitted .insn stream, not the Spike log). Numerical = this harness's exact (device57) comparison on Spike.\n")
    out.append("| step | n |\n|---|---|\n"+"\n".join(f"| {k} | {v} |" for k,v in (
        ("DEVICE57 total",len(dv)),("weak pass (RoCC insns retired + exact output)",weak),
        ("  of which host_compute pass",Y(lambda r:r['ok']=='PASS' and r['host_compute']=='pass')),("  of which host_compute FAIL (incl. census silent)",Y(lambda r:r['ok']=='PASS' and r['host_compute'].startswith('fail'))),("  of which host_compute unknown",Y(lambda r:r['ok']=='PASS' and (r['host_compute'].startswith('unknown') or r['host_compute'].startswith('not judged')))),
        ("  of which trace pass",Y(lambda r:r['ok']=='PASS' and r['trace_check']=='pass')),("  of which trace unavailable_fields/protocol",Y(lambda r:r['ok']=='PASS' and r['trace_check'].startswith('unavailable'))),("  of which trace fail",Y(lambda r:r['ok']=='PASS' and r['trace_check']=='fail')),
        ("meets full definition (strict: host_compute must be 'pass')",full),("meets full definition if declared_host (not judged, veto does not fire) is accepted",Y(lambda r:r['meets_full_declared_host_ok']=='YES')))))
    out.append("\n### DEVICE57 per overload (full definition)\n\n| # | overload | numerical | host_compute | host ops/elem (budget 1.0) | float ops/elem | trace | trace reason | meets full definition |\n|---|---|---|---|---|---|---|---|---|")
    for r in sorted(dv,key=lambda r:(r['meets_full']!='YES',r['overload'])):
        out.append(f"| {r['n']} | {r['overload']} | {r['numerical']} | {r['host_compute']} | {r['host_ops_per_element']} | {r['host_value_ops_per_element']} | {r['trace_check']} | {(r['trace_reason'] or (r['host_compute_why'] if r['host_compute'].startswith('fail') else '')).replace('|','/')[:110]} | {r['meets_full']} |")
ct={c:(load('ctl_'+c) or {}) for c in ('ctl-movement','ctl-movement-neg')}
out.append("\n## Harness controls\n\n* ctl-movement (correct i8 movement program, correct reference): expect outcome ok with device insns > 0: got %s / %s.\n* ctl-movement-neg (same, one reference element corrupted): expect wrong: got %s."%(ct['ctl-movement'].get('outcome'),ct['ctl-movement'].get('device_insns_retired'),ct['ctl-movement-neg'].get('outcome')))
txt='\n'.join(out)+'\n'
open(f'{H}/RESULT-193-{L}.md','w').write(txt)
w=csv.DictWriter(open(f'{H}/result-193-{L}.csv','w'),list(rows[0])); w.writeheader(); w.writerows(rows)
if os.environ.get('HEADLINE'):
    open(f'{H}/RESULT-193.md','w').write(txt); w=csv.DictWriter(open(f'{H}/RESULT-193.csv','w'),list(rows[0])); w.writeheader(); w.writerows(rows)
print('\n'.join(out[:2])); print('\n'.join(out[3:12]))
