"""Neutral custody fixtures; no accelerator or physical simulator is executed."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from merlin.common import invocation_record as I
from merlin.common.tree_hash import hash_tree
from merlin.runtime import _native_files as F
from merlin.runtime.backends.base import parse_console
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.readback_policy import canonical_sha256
from merlin_experiments.phase1 import joint_calibration as C
from merlin_experiments.phase1 import joint_calibration_reuse as R
from merlin_experiments.phase1 import joint_calibration_admission as A
from merlin_experiments.phase1 import joint_control
from test_candidate_execution_products import SOURCE, LOWERED
from test_joint_calibration_eligibility import valid_grade, capsule


def pin(p):
    return F.pin(p, max_bytes=64 << 20)


def write(p, doc):
    p.write_text(json.dumps(doc));return pin(p)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    """Real host producer receipts over neutral bytes, independent verdict fixture."""
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    root=tmp_path/'generated';root.mkdir()
    for name,data in {'package_kernel.elf':b'opaque neutral ELF fixture', 'harness.c':b'neutral caller',
                      'kernel.o':b'neutral object'}.items():(root/name).write_bytes(data)
    (root/'input.interface.mlir').write_text(SOURCE);(root/'lowered.llvm.mlir').write_text(LOWERED)
    cb=parse_interface_mlir(SOURCE);effective=deepcopy(cb)
    effective['tensors']['y']=dict(shape=[2,4],dtype='i32',role='output')
    (root/'command_buffer.json').write_text(json.dumps(cb))
    build=dict(status='completed',elf_sha256=pin(root/'package_kernel.elf')['sha256'],
        harness_sha256=pin(root/'harness.c')['sha256'],kernel_object_name='kernel.o',
        kernel_object_sha256=pin(root/'kernel.o')['sha256'],command_buffer_sha256=canonical_sha256(effective))
    build['build_identity_sha256']=canonical_sha256(build)
    # This prints a fixture frame, never executes the opaque ELF or hardware.
    console='OUT y 2 4 1 2 3 4 5 6 7 8\nMETRIC cycles 100\nDONE\n'
    def observe(folder, *, stage="execution", inputs=None, dependencies=()):
        I.run([sys.executable,'-I','-B','-c','print('+repr(console)+',end="")'],directory=folder,
            stage=stage,inputs=inputs or (root/'package_kernel.elf',),dependencies=dependencies,capture_output=True,env={})
        path=next((folder/'invocations').glob('*/invocation.json'))
        return path
    invocation=observe(root)
    facts=write(tmp_path/'facts.json',{'facts':{'configuration':'neutral fixture'}})
    receipt=write(tmp_path/'build.json',{'build':'neutral fixture, no hardware qualification'})
    binary=pin(Path(sys.executable).resolve())
    oracle={'provenance':{'sha256':binary['sha256']},'memory_engine':{
        'rtl_facts':{**facts,'firrtl_sha256':'f'*64},'engine':{'receipt':{
            'binary_sha256':binary['sha256'],'firrtl_sha256':'f'*64,
            'receipt_path':receipt['path'],'receipt_sha256':receipt['sha256']}}}}
    E.retain(dict(elf=str(root/'package_kernel.elf'),readback_build=build,console=console.encode(),
        oracle=oracle,execution_invocation=pin(invocation)),sim='gsim',workdir=root,
        compiler_artifacts=dict(command_buffer=effective,lowered_mlir=LOWERED))
    compiler=tmp_path/'compiler';compiler.mkdir();(compiler/'source.py').write_text('pass\n')
    digest=hash_tree(compiler)['sha256'];shutil.copytree(compiler,root/'compiler-source'/digest)
    grade=valid_grade(source=digest)
    grade['compiler_source'].update(package=str(compiler),snapshot='compiler-source/'+digest,
        artifacts={name:pin(root/name)['sha256']for name in ['input.interface.mlir','lowered.llvm.mlir','command_buffer.json']},
        executed_programs=E.collect(root))
    grade['tiers']['L3'].update(engine='gsim',measured_now=True,cycle_accurate=True,derived_from_rtl=True,
        cycles=100,sim_provenance=dict(engine='gsim',sha256=binary['sha256']))
    cap=capsule();cap_pin=write(tmp_path/'capsule.json',cap)
    grade_pin=write(tmp_path/'grade.json',grade)
    profile=write(tmp_path/'profile.json',{'members':{'public':{'source_group_sha256':'g'*64}}})
    model={'domain':{'label_engine_sha256':binary['sha256'],'scope_sha256':'s'*64}};model_pin=write(tmp_path/'model.json',model)
    runtime_pin=write(tmp_path/'runtime.json',{'model':model_pin['path'],'model_sha256':model_pin['sha256']})
    selection=write(tmp_path/'selection.json',{'facts':facts,'members':{
        'public':{'capsule_sha256':R.PERF.capsule_identity(cap)}}})
    case=dict(id='case',split='train',source_group_sha256='g'*64,capsule=cap_pin,result=grade_pin,
        compiler=str(compiler),compiler_sha256=digest,generated_root=str(root))
    plan=dict(schema=C.SCHEMA,gates=dict(C.GATES),cases=[case],selection=selection,runtime=runtime_pin)
    scope=dict(elf_sha256=pin(root/'package_kernel.elf')['sha256'],input_sha256=pin(root/'package_kernel.elf')['sha256'],
        scope_sha256='s'*64,begin_pc=10,end_pc=20,occurrence=0)
    timing=root/'joint_timing';timing.mkdir()
    counter={k:v for k,v in scope.items()if k in ['elf_sha256','begin_pc','end_pc','occurrence']}
    write(timing/'counter_review.json',counter)
    scope['evidence_sha256']=C.document_sha256(counter)
    scope_pin=write(timing/'scope.json',scope)
    report=write(timing/'report.json',dict(timing_complete=True,unresolved=[],elf_sha256=scope['elf_sha256'],
        model_sha256=model_pin['sha256'],scope_sha256=scope['scope_sha256'],scope_binding=scope))
    processes=[]
    for kind in ['functional','annotated']:
        folder=tmp_path/kind;folder.mkdir()
        reference=observe(folder,stage='archive_screen_count',inputs=(root/'package_kernel.elf',Path(scope_pin['path'])),
            dependencies=(Path(model_pin['path']),))
        ref=I.verify(reference)
        processes.append(dict(path=str(reference),sha256=pin(reference)['sha256'],stdout_sha256=ref['stdout']['sha256']))
    record=dict(elf=str(root/'package_kernel.elf'),observation=dict(elf_sha256=scope['elf_sha256'],report=report,scope_binding=scope,
        functional_identical=True,complete_counts=True,processes=processes))
    class Backend:
        def _engine_identity(self, engine):return {'binary':binary,'receipt':'neutral'}
        def _engine(self, engine):return {'receipt':receipt,'environment':{}}
        def parse_output(self, data):return parse_console(data.decode()if isinstance(data,bytes)else data)
    backend=Backend()
    monkeypatch.setattr(A,'assert_allowed',lambda p:None)
    monkeypatch.setattr(joint_control,'selected',lambda:{'policy':{'native_calibration':profile,'gsim':{'witness_capsules':['public']}}})
    monkeypatch.setattr(R,'_selected_runtime',lambda b,e:(object(),{}))
    # Runtime-component parser has independent mutation tests below. Here only
    # this instrument boundary is replaced; native producer/source joins are real.
    reopen_process=R._process_record
    monkeypatch.setattr(R,'_process_record',lambda *a,**k:{'kind':'neutral instrument fixture'})
    return SimpleNamespace(root=root,case=case,plan=plan,backend=backend,model=model,record=record,
        grade=grade,console=console,receipt=receipt,reference=reference,tmp=tmp_path,reopen_process=reopen_process)


def review(f):return R.inspect(f.case,record=f.record,plan=f.plan,backend=f.backend,model=f.model)


def test_exact_native_custody_ready_and_label_reopens(retained):
    f=retained;r=review(f);assert r['status']=='ready',r
    plan=write(f.tmp/'plan.json',f.plan);receipt=R.measurement(r,plan_pin=plan)
    assert receipt['label_origin']=='retained_correctness_gsim'and receipt['cycles']==100
    assert C._reopen_physical(receipt,f.record,f.backend)==100
    # The executed caller is the retained content-addressed member; the root
    # harness is only a mutable convenience copy and is not selected evidence.
    execution=f.grade['compiler_source']['executed_programs'][0]
    (f.root/execution['harness']['path']).write_text('changed')
    with pytest.raises(ValueError):C._reopen_physical(receipt,f.record,f.backend)


@pytest.mark.parametrize('damage',['compiler','capsule','profile_group','grade','elf','console','invocation','scope','model','build'])
def test_exact_custody_mutations_are_refused(retained,damage):
    f=retained
    if damage=='compiler':(Path(f.case['compiler'])/'source.py').write_text('changed')
    elif damage=='capsule':Path(f.case['capsule']['path']).write_text('{}')
    elif damage=='profile_group':f.case['source_group_sha256']='x'*64
    elif damage=='grade':Path(f.case['result']['path']).write_text('{}')
    elif damage=='elf':(f.root/'package_kernel.elf').write_bytes(b'changed')
    elif damage=='console':
        execution=f.grade['compiler_source']['executed_programs'][0]
        (f.root/execution['compiler_artifacts']['members']['console']['path']).write_bytes(b'changed')
    elif damage=='invocation':Path(f.reference).write_text('{}')
    elif damage=='scope':f.record['observation']['scope_binding']['end_pc']+=4
    elif damage=='model':f.model['domain']['label_engine_sha256']='x'*64
    else:Path(f.receipt['path']).write_text('{}')
    r=review(f);assert r['status']=='denied'and not r['fresh_observation_required'],r


def test_runtime_missing_is_explicit_not_invented(retained,monkeypatch):
    f=retained
    def missing(*a,**k):raise R._Unavailable('original closure unavailable')
    monkeypatch.setattr(R,'_process_record',missing)
    r=review(f);assert r['status']=='denied'and r['fresh_observation_required']
    assert r['reason']=='original closure unavailable'


def test_fitting_denial_precedes_any_evidence_read(retained,monkeypatch):
    def deny(plan):raise ValueError('blocked_structural_model_admission: unresolved')
    monkeypatch.setattr(A,'assert_allowed',deny)
    r=review(retained);assert r['status']=='denied'and'blocked_structural'in r['reason']


def test_ready_reuse_avoids_authorization_and_physical_execution(retained,monkeypatch):
    f=retained;round=f.tmp/'round';(round/'probe-0').mkdir(parents=True)
    write(round/'plan.json',f.plan);write(round/'probe-0/case.json',f.record)
    def no_measure(*a,**k):pytest.fail('exact ready evidence must not run new physical work')
    monkeypatch.setattr(C,'_measure',no_measure);monkeypatch.setattr(C,'_authorize_if_needed',no_measure)
    labels={};C._collect_labels([f.case],root=round,plan=f.plan,backend=f.backend,base_model=f.model,labels=labels)
    assert labels=={'case':100}
    assert json.loads((round/'reuse/case.json').read_text())['status']=='ready'


@pytest.fixture
def components(tmp_path):
    """Native namespace argument fixture, not a native execution observation."""
    image=tmp_path/'image';image.mkdir();(image/'engine').write_bytes(b'neutral engine')
    elf=tmp_path/'kernel.elf';elf.write_bytes(b'neutral opaque ELF')
    source=tmp_path/'owner.py';source.write_text('pass\n')
    bwrap=tmp_path/'bwrap';bwrap.write_bytes(b'neutral launcher')
    expected={'cwd':'/run','guest_argv':['/engine','/run/merlin/input.elf'],
              'environment':I.environment_identity({}),'unlimited_stack':False,'text_mode':'utf8'}
    identity={'runtime_view_sha256':F.inventory(image,max_files=16,max_bytes=4096)['sha256'],
              'selection_sha256':'s'*64}
    runtime=SimpleNamespace(bwrap=bwrap,image=image,max_files=16,max_bytes=4096,
        verify=lambda backend:identity,observation_sources=lambda:(source,))
    backend=SimpleNamespace(_engine=lambda engine:{'environment':{}})
    script='printf READY >&7; read -r native_go <&7; [[ $native_go == GO ]] || exit 126; exec 7>&-; exec /usr/bin/env -- "$@"'
    argv=[str(bwrap),'--unshare-all','--unshare-user','--as-pid-1','--cap-drop','ALL',
          '--disable-userns','--assert-userns-disabled','--die-with-parent','--new-session','--clearenv',
          '--ro-bind',str(image),'/', '--ro-bind',str(elf),'/run/merlin/input.elf',
          '--bind',str(tmp_path/'work'),'/run/work','--bind',str(tmp_path/'tmp'),'/tmp',
          '--chdir','/run','--','/bin/bash','--noprofile','--norc','-c',script,'native-runtime',
          *expected['guest_argv']]
    process=dict(kind='subprocess',returncode=0,stage='native_runtime_process',argv=argv,
        environment=I.environment_identity({}),inputs=[pin(elf)],dependencies=[pin(source)])
    return SimpleNamespace(runtime=runtime,backend=backend,elf=elf,expected=expected,
        process=process,image=image,source=source)


def test_component_equivalence_never_invents_issued_owner(components):
    f=components;result=R._runtime_components(f.process,runtime=f.runtime,backend=f.backend,
        elf=f.elf,expected=f.expected)
    assert result['kind']=='legacy_component_equivalence'
    assert 'signature'not in result and result['reset_seed_closure']=='UNKNOWN'


@pytest.mark.parametrize('damage',['image','source','argv','cwd','environment','input','script','mount','unclosed'])
def test_component_runtime_mutations_refused(components,damage):
    f=components
    if damage=='image':(f.image/'engine').write_bytes(b'changed')
    elif damage=='source':f.source.write_text('changed')
    elif damage=='argv':f.process['argv'].append('+unreviewed-memory-option')
    elif damage=='cwd':f.expected['cwd']='/elsewhere'
    elif damage=='environment':f.process['environment']=I.environment_identity({'MEMORY':'changed'})
    elif damage=='input':f.process['inputs']=[]
    elif damage=='script':f.process['argv'][30]='echo invented'
    elif damage=='mount':f.process['argv'][16]='/run/other.elf'
    else:f.process['stage']='execution'
    with pytest.raises(ValueError):R._runtime_components(f.process,runtime=f.runtime,
        backend=f.backend,elf=f.elf,expected=f.expected)


@pytest.mark.parametrize('damage',['receipt_cycles','console_substitution','case_id'])
def test_reused_receipt_cannot_substitute_a_scalar_or_console(retained,damage):
    f=retained;r=review(f);assert r['status']=='ready',r
    receipt=R.measurement(r,plan_pin=write(f.tmp/'plan.json',f.plan))
    if damage=='receipt_cycles':receipt['cycles']=101
    elif damage=='case_id':receipt['case_id']='another_case'
    else:
        other=f.tmp/'unrelated_console.bin';other.write_text('METRIC cycles 101\nDONE\n')
        receipt.update(console=pin(other),cycles=101)
    with pytest.raises(ValueError,match='reused label receipt differs'):
        C._reopen_physical(receipt,f.record,f.backend)


def test_full_physical_outputs_are_required(retained,monkeypatch):
    f=retained;original=f.backend.parse_output;calls=0
    def missing(data):
        nonlocal calls
        calls+=1
        return ({},{'cycles':100})if calls==1 else original(data)
    monkeypatch.setattr(f.backend,'parse_output',missing)
    r=review(f)
    assert r['status']=='denied'and r['fresh_observation_required']
    assert 'full-value'in r['reason']


def test_both_original_native_processes_must_reopen(retained):
    f=retained
    Path(f.record['observation']['processes'][0]['path']).write_text('{}')
    r=review(f)
    assert r['status']=='denied'and not r['fresh_observation_required']


def test_native_probe_model_copy_may_differ_only_in_json_formatting(retained):
    f=retained
    model_path=Path(f.plan['runtime']['path'])
    runtime=json.loads(model_path.read_text())
    selected_model=f.tmp/'equivalent_selected_model.json'
    selected_model.write_text(json.dumps(f.model,indent=4))
    runtime.update(model=str(selected_model),model_sha256=pin(selected_model)['sha256'])
    f.plan['runtime']=write(model_path,runtime)
    assert review(f)['status']=='ready'


@pytest.mark.parametrize('field,value',[('unlimited_stack',True),('text_mode','locale-dependent')])
def test_legacy_equivalence_refuses_uncovered_process_mode(components,field,value):
    f=components;f.expected[field]=value
    with pytest.raises(ValueError,match='process mode'):
        R._runtime_components(f.process,runtime=f.runtime,backend=f.backend,elf=f.elf,expected=f.expected)



def test_unclosed_original_process_is_not_legacy_native_closure(retained):
    f=retained
    execution=f.grade['compiler_source']['executed_programs'][0]
    with pytest.raises(R._Unavailable,match='no unique original physical process'):
        f.reopen_process(f.root,execution,console=f.console.encode(),runtime=object(),backend=f.backend,
            elf=f.root/'package_kernel.elf',expected={})


def test_live_original_signed_owner_path_is_preserved(retained):
    f=retained;execution=deepcopy(f.grade['compiler_source']['executed_programs'][0])
    expected={'runtime':{'selection':'exact'},'capture_bytes':True}
    body={'selection':{**expected,'capture_bytes':False},'record':{'path':'original-owner-record'}}
    native={'body':body,'signature':'original-only'}
    execution['compiler_artifacts']['native_observation']=native
    reopened=[]
    runtime=SimpleNamespace(reopen_observation=lambda *a,**k:reopened.append((a,k)))
    result=f.reopen_process(f.root,execution,console=f.console.encode(),runtime=runtime,backend=f.backend,
        elf=f.root/'package_kernel.elf',expected=expected)
    assert result['kind']=='same_issued_native_owner'and result['observation']==native
    assert reopened[0][0]==(body,'original-only')


def test_other_owner_signature_does_not_create_a_new_issued_observation(retained):
    f=retained;execution=deepcopy(f.grade['compiler_source']['executed_programs'][0])
    execution['compiler_artifacts']['native_observation']={'body':{},'signature':'previous-owner-only'}
    def reject(*a,**k):raise ValueError('not issued by this owner')
    runtime=SimpleNamespace(reopen_observation=reject)
    with pytest.raises(R._Unavailable):
        f.reopen_process(f.root,execution,console=f.console.encode(),runtime=runtime,backend=f.backend,
            elf=f.root/'package_kernel.elf',expected={})



def test_equivalent_separately_materialized_guest_image_reopens_original_dependency(components,tmp_path):
    f=components
    old_engine=f.image/'engine'
    f.process['dependencies'].append(pin(old_engine))
    new_image=tmp_path/'new_image';shutil.copytree(f.image,new_image)
    new_engine=new_image/'engine'
    f.runtime.image=new_image
    f.runtime.observation_sources=lambda:(f.source,new_engine)
    proof=R._runtime_components(f.process,runtime=f.runtime,backend=f.backend,elf=f.elf,expected=f.expected)
    assert proof['kind']=='legacy_component_equivalence'
    # Keeping the new owner's dependency does not substitute for the original.
    f.process['dependencies']=[pin(f.source),pin(new_engine)]
    with pytest.raises(ValueError,match='source components differ'):
        R._runtime_components(f.process,runtime=f.runtime,backend=f.backend,elf=f.elf,expected=f.expected)



class DumpReader:
    """Decode actual neutral producer bytes; no ELF/RTL semantics supplied."""
    def __init__(self,elf,payload,regions):self.elf,self.payload,self.regions=elf,payload,regions
    def decode(self,console):
        import struct,hashlib
        assert console=='METRIC cycles 100\nDONE\n'
        values=list(struct.unpack('bb',self.payload.read_bytes()))
        return {'y':[values]},dict(status='complete',source='coherent',original_elf=pin(self.elf),
            payload=pin(self.payload),regions=pin(self.regions),artifact_sha256=pin(self.payload)['sha256'],
            console_sha256=hashlib.sha256(console.encode()).hexdigest())


def dispatch_dump_fixture(root,elf,payload,regions):
    # A real neutral host process writes the requested bytes. It does not run
    # the opaque ELF, an ISA emulator or an accelerator model.
    return I.run([sys.executable,'-I','-B','-c',
        'import pathlib,sys; p=sys.argv[-2].split("=",1)[1]; pathlib.Path(p).write_bytes(bytes([1,254])); print("METRIC cycles 100\\nDONE")',
        str(elf),'+dump-regions='+str(regions),'+dump-out='+str(payload),'+dump-mode=coherent'],
        directory=root,stage='selected_rocc_process',inputs=(elf,regions),outputs=(payload,),
        capture_output=True,env={})


@pytest.fixture
def memory_products(tmp_path):
    from merlin.targetgen.contract.compile import _observed_memory_decode
    from merlin.targetgen.contract.readback_policy import ReadbackPolicy,COHERENT_DUMP_V1,BUILD_RECEIPT
    root=tmp_path/'generated';root.mkdir()
    elf=root/'opaque.elf';elf.write_bytes(b'opaque fixture, no executable semantics')
    regions=root/'regions.txt';regions.write_text('0x8000 2\n')
    payload=root/'output.dump'
    with I.observe_call(root,stage='execution',function=dispatch_dump_fixture,arguments={},inputs=(elf,))as outer:
        process=dispatch_dump_fixture(root,elf,payload,regions)
        process.check_returncode();outer.returned(stdout=process.stdout)
    console=process.stdout.decode()
    (root/'oracle_console.log').write_text(console);(root/BUILD_RECEIPT).write_text('{}')
    cb={'kernel_abi':{'kind':'whole_program','outputs':['y']},'tensors':{'y':{'shape':[1,2],'dtype':'i8','role':'output'}}}
    values,memory,observation=_observed_memory_decode(DumpReader(elf,payload,regions),console,
        cb=cb,elf=elf,workdir=root,policy=ReadbackPolicy(COHERENT_DUMP_V1),retain_in_invocation=True,inputs=(regions,))
    execution={'elf':pin(elf),'compiler_artifacts':dict(readback_memory=memory,readback_observation=observation,
        original_execution_invocation=pin(outer.path))}
    engine={'binary':pin(Path(sys.executable).resolve())}
    return SimpleNamespace(root=root,execution=execution,console=console.encode(),engine=engine,
        elf=elf,payload=payload,regions=regions,observation=observation,values=values)


def test_original_dump_and_decoder_join_never_issues_runtime_qualification(memory_products):
    f=memory_products
    values,custody=R._retained_memory_outputs(f.root,f.execution,console=f.console,engine=f.engine)
    assert values=={'y':[[1,-2]]}
    assert custody['runtime_qualification']=='not_issued_by_byte_join'
    # A direct recorded process cannot be relabeled a closed NativeRuntimeView.
    with pytest.raises(R._Unavailable):
        R._process_record(f.root,f.execution,console=f.console,runtime=object(),backend=object(),elf=f.elf,expected={})


@pytest.mark.parametrize('defect',['payload','regions','decoder','product','producer_output','console'])
def test_full_value_memory_join_reopens_actual_original_products(memory_products,defect):
    f=memory_products
    if defect=='payload':f.payload.write_bytes(b'changed')
    elif defect=='regions':f.regions.write_text('0x8008 2\n')
    elif defect=='decoder':Path(f.observation['record']['path']).write_text('{}')
    elif defect=='product':Path(f.observation['product']['path']).write_text('{}')
    elif defect=='console':f.console=b'METRIC cycles 101\nDONE\n'
    else:
        p=next(p for p in(f.root/'invocations').glob('*/invocation.json')
            if json.loads(p.read_text()).get('stage')=='selected_rocc_process')
        row=json.loads(p.read_text());row['outputs']=[];p.write_text(json.dumps(row))
    with pytest.raises(ValueError):R._retained_memory_outputs(f.root,f.execution,console=f.console,engine=f.engine)



def test_native_scope_join_does_not_parse_an_observed_symbol_elf_as_json(retained):
    f=retained;extra=f.tmp/'output-symbols.elf';extra.write_bytes(b'\x7fELF neutral metadata fixture')
    scope=f.root/'joint_timing/scope.json';model=Path(json.loads(Path(f.plan['runtime']['path']).read_text())['model'])
    processes=[]
    for i in range(2):
        folder=f.tmp/f'extra-reference-{i}';folder.mkdir()
        I.run([sys.executable,'-I','-B','-c','print('+repr(f.console)+',end="")'],directory=folder,
            stage='archive_screen_count',inputs=(f.root/'package_kernel.elf',scope,extra),dependencies=(model,),
            env={},capture_output=True)
        path=next((folder/'invocations').glob('*/invocation.json'));row=I.verify(path)
        processes.append(dict(path=str(path),sha256=pin(path)['sha256'],stdout_sha256=row['stdout']['sha256']))
    f.record['observation']['processes']=processes
    assert review(f)['status']=='ready'
