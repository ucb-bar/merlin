"""Portable native admission controls using bounded synthetic receipt fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'tools'))
from smol_capture.admission import AdmissionRegistry,UnadmittedKind,program_kind,ORIGINAL_PIN,NATIVE_PIN


class CatalogFixture:
    def __init__(self,base):
        self.base=base;self.data=base/'data';self.data.mkdir();(self.data/'blobs').mkdir()
        self.atlas=base/'atlas';self.original=base/'original';self.native=base/'native'
        self.models={'original':{},'native':{}};self.tools={}
        required=['npu_model/configs/numerics.py','npu_model/configs/rtl_math.py',
                  'npu_model/configs/isa_definition.py','npu_model/hardware/arch_state.py']
        for role,root in [('original',self.original),('native',self.native)]:
            for name in required:
                self.models[role][name]=self.file(root/name,b'fixture model source\n')
        for name in ['atlas_program.py','atlas_epilogue.py','atlas_program_checks.py',
                     'atlas_program_reuse_common.py','smol_fusion_program.py']:
            self.tools['tools/'+name]=self.file(self.atlas/'tools'/name,b'fixture tool source\n')
        native_c=self.file(self.native/'npu_model/configs/native_systolic.c',b'fixture native source\n')
        library=base/'native.so';library.write_bytes(b'fixture library')
        cache=base/'native.json'
        self.backend=dict(backend='rtl-native-systolic-integer',base_commit=ORIGINAL_PIN,
                          sources={'native_systolic.c':native_c['sha256']},library_path=str(library),
                          library_sha256=self.sha(library.read_bytes()),compiler_version='fixture compiler',
                          qualification='fixture only',environment={'NPU_MODEL_RTL_NATIVE_MXU0':'1'},build_record=str(cache))
        cache.write_text(json.dumps({k:v for k,v in self.backend.items() if k!='build_record'}))
        self.catalog=dict(schema='atlas.smol_native_admission_catalog.v1',gate_of_record=False,
                          all_launch_original_gate_required=True,model_sources=self.models,atlas_sources=self.tools,
                          native_backend=self.json_blob(self.backend),kinds={})
        self.add(False);self.add(True);self.add(True,fp8=True)

    @staticmethod
    def sha(raw):return hashlib.sha256(raw).hexdigest()

    def blob(self,raw):
        digest=self.sha(raw);(self.data/'blobs'/digest).write_bytes(raw)
        return dict(file='blobs/'+digest,bytes=len(raw),sha256=digest)

    def json_blob(self,value):return self.blob(json.dumps(value,sort_keys=True).encode())

    def file(self,path,raw):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return self.blob(raw)

    def object(self,record):return json.loads((self.data/record['file']).read_bytes())

    def add(self,fused,fp8=False):
        source='vli.all m4, 16256\nnop\necall\n';binary=struct.pack('<3I',(127<<23)|0x25f,0x13,0x73)
        size=1024 if fp8 else 2048
        arena_size=2304+size+256+(2048+256 if fused else 0)
        desc=dict(physical_mkn=[32,32,32],recipe='traffic-reuse-v1' if fused else 'baseline-v1',
                  output_offset=2304,output_bytes=size,output_rescale_exponent=0,
                  operations=['bias','gelu_tanh'] if fp8 else ['bias'] if fused else [],
                  output_dtype='fp8' if fp8 else 'bf16',output_layout='tiles',dram_base=0x90000000,dram_bytes=arena_size,
                  total_dram_bytes=arena_size,extra_offset=2304+size+256,extra_bytes=2048 if fused else 0,
                  scratch_registers=[10,11] if fused else None,input_regions=[])
        if fp8:desc['output_scale_exponent']=-5
        key,kind=program_kind(binary,source,desc)
        initial=b'A'*2048+b'\xa5'*256+b'\xcd'*size+b'Z'*256+(b'B'*2048+b'Z'*256 if fused else b'')
        output=b'\0'*size;arena=initial[:2304]+output+initial[2304+size:]
        requested=dict(load_bytes=4096 if fused else 2048,store_bytes=size)
        sample=dict(source=self.blob(source.encode()),binary=self.blob(binary),initial=self.blob(initial),
                    descriptor=desc,output_offset=2304,output_bytes=size,requested_dma=requested)
        entry=dict(kind_key=key,kind=kind,proof_type='full-producer' if fused else 'legacy',sample=sample,source_archive={})
        request=dict(kind_key=key,pins={name:dict(path='/historical/'+name,bytes=sample[name]['bytes'],sha256=sample[name]['sha256'])
                                      for name in ['source','binary','initial']})
        request_record=self.json_blob(request);request_pin=dict(path='/historical/request.json',
                                                               bytes=request_record['bytes'],sha256=request_record['sha256'])
        if fused:entry['request']=request_record
        report=dict(status='ADMITTED_NATIVE_SMOL_FUSION_KIND' if fused else 'ADMITTED_NATIVE_MXU0',kind_key=key,kind=kind,
                    full_arena_exact=True,gate_of_record=False,all_launch_original_gate_required=True,
                    initial_sha256=self.sha(initial),expected_full_arena_sha256=self.sha(arena),native_backend=self.backend)
        if fused:report['request']=request_pin
        runner=self.blob(b'fixture historical runner')
        entry['source_archive']['/historical/runner.py']=runner
        for role,pin,root in [('original',ORIGINAL_PIN,self.original),('native',NATIVE_PIN,self.native)]:
            execution=dict(program_sha256=self.sha(binary),program_words=3,halt_pc=2,instructions=3,observed=requested)
            receipt=dict(model=dict(commit=pin,numerics='rtl',root=str(root)),execution=execution)
            if fused:
                receipt.update(status='PASS_FULL_PRODUCER',request=request_pin,original_model=role=='original',math_override=False,
                               full_arena_exact=True,output_exact=True,input_and_guards_preserved=True,source_bookends_equal=True,
                               source_bookends_before={'/historical/runner.py':runner['sha256']},
                               source_bookends_after={'/historical/runner.py':runner['sha256']},data_pins=[],
                               initial_sha256=self.sha(initial),program_sha256=self.sha(binary),source_sha256=self.sha(source.encode()),
                               expected_arena_sha256=self.sha(arena),actual_arena_sha256=self.sha(arena),
                               expected_output_sha256=self.sha(output),actual_output_sha256=self.sha(output),
                               native_backend=None if role=='original' else self.backend)
            elif role=='native':receipt['expected_exact']=True
            record=self.json_blob(receipt)
            entry[role]=dict(receipt=record,arena=self.blob(arena),output=self.blob(output))
            if fused:
                historical=dict(path='/historical/'+role+'/receipt.json',bytes=record['bytes'],sha256=record['sha256'])
                entry[role].update(historical_pin=historical,completion=self.json_blob(dict(returncode=0,request=request_pin,receipt=historical)))
                report[role+'_receipt']=historical
            else:report[role]=receipt
        if not fused:
            command=['python','/historical/harness.py','--source','/historical/program.S','--arena','/historical/initial.bin']
            report['commands']=[dict(returncode=0,command=command),dict(returncode=0,command=command)]
            entry['harness']=runner;entry['historical_harness_path']='/historical/harness.py'
            entry['historical_sample']=self.json_blob(dict(physical_mkn=[32,32,32],recipe='baseline-v1',
                output_offset=2304,output_bytes=2048,program_source='/historical/program.S',initial_arena='/historical/initial.bin'))
        entry['admission']=self.json_blob(report);self.catalog['kinds'][key]=entry
        return key

    def git(self,args,text=False):
        root=Path(args[2])
        if 'rev-parse' in args:value=ORIGINAL_PIN if root==self.original else NATIVE_PIN;return value if text else value.encode()
        return (root/args[-1].removeprefix('HEAD:')).read_bytes()

    def registry(self):
        raw=json.dumps(self.catalog,sort_keys=True).encode();(self.data/'catalog.json').write_bytes(raw)
        return AdmissionRegistry(self.data,self.sha(raw),atlas_root=self.atlas,original_model_root=self.original,
                                 native_model_root=self.native,native_backend_metadata=self.backend)

    def validate(self,registry,entry,initial=None):
        return registry.validate((self.data/entry['sample']['binary']['file']).read_bytes(),
                                 (self.data/entry['sample']['source']['file']).read_text(),entry['sample']['descriptor'],initial)


class NativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.f=CatalogFixture(Path(self.temp.name))
        self.mock=patch('smol_capture.admission.subprocess.check_output',side_effect=self.f.git);self.mock.start();self.addCleanup(self.mock.stop)

    def test_portable_legacy_and_fused_proofs_admit_new_input_bytes(self):
        registry=self.f.registry()
        for entry in self.f.catalog['kinds'].values():
            result=self.f.validate(registry,entry,b'Q'*entry['sample']['initial']['bytes'])
            self.assertFalse(result['gate_of_record']);self.assertTrue(result['all_launch_original_gate_required'])

    def test_only_unique_m4_literal_normalizes_and_controls_remain_distinct(self):
        entry=next(iter(self.f.catalog['kinds'].values()));sample=entry['sample'];desc=sample['descriptor']
        raw=(self.f.data/sample['binary']['file']).read_bytes();source=(self.f.data/sample['source']['file']).read_text()
        changed=struct.pack('<3I',(128<<23)|0x25f,0x13,0x73)
        self.assertEqual(program_kind(raw,source,desc)[0],program_kind(changed,source.replace('16256','16384'),{**desc,'output_rescale_exponent':1})[0])
        with self.assertRaises(ValueError):program_kind(raw,source+source.splitlines()[0]+'\n',desc)
        registry=self.f.registry();words=struct.pack('<3I',(127<<23)|0x25f,0x100013,0x73)
        with self.assertRaises(UnadmittedKind):registry.validate(words,source,desc)
        with self.assertRaises(ValueError):registry.validate(raw,source.replace('nop','addi x1, x0, 0'),desc)

    def test_fp8_kind_preserves_output_scale_and_rejects_wrong_view_extent(self):
        entry=next(e for e in self.f.catalog['kinds'].values() if e['sample']['descriptor']['output_dtype']=='fp8')
        sample=entry['sample'];desc=sample['descriptor']
        raw=(self.f.data/sample['binary']['file']).read_bytes();source=(self.f.data/sample['source']['file']).read_text()
        key,kind=program_kind(raw,source,desc)
        self.assertEqual(kind['schema'],'atlas.smol_fp8_fusion_program_kind.v1')
        self.assertEqual(kind['output_scale_exponent'],-5)
        self.assertNotEqual(key,program_kind(raw,source,{**desc,'output_scale_exponent':-6})[0])
        for change in [dict(output_bytes=2048),dict(output_scale_exponent=True),dict(operations=['bias'])]:
            with self.assertRaises(ValueError):program_kind(raw,source,{**desc,**change})
        self.f.validate(self.f.registry(),entry)

    def fused(self):return next(e for e in self.f.catalog['kinds'].values() if e['proof_type']=='full-producer')

    def test_failed_or_boolean_completion_is_rejected(self):
        entry=self.fused()
        for code in (1,False):
            completion=self.f.object(entry['native']['completion']);completion['returncode']=code
            entry['native']['completion']=self.f.json_blob(completion)
            with self.assertRaises(ValueError):self.f.validate(self.f.registry(),entry)

    def test_full_arena_corruption_and_source_bookend_mismatch_are_rejected(self):
        entry=self.fused();old=entry['native']['arena'];raw=(self.f.data/old['file']).read_bytes()
        entry['native']['arena']=self.f.blob(b'X'+raw[1:])
        with self.assertRaises(ValueError):self.f.validate(self.f.registry(),entry)
        entry['native']['arena']=old
        receipt=self.f.object(entry['original']['receipt']);receipt['source_bookends_after']={}
        record=self.f.json_blob(receipt);entry['original']['receipt']=record
        # Keep the process digest link valid so the bookend check must reject it.
        historical=entry['original']['historical_pin'];historical.update(bytes=record['bytes'],sha256=record['sha256'])
        completion=self.f.object(entry['original']['completion']);completion['receipt']=historical
        entry['original']['completion']=self.f.json_blob(completion)
        admission=self.f.object(entry['admission']);admission['original_receipt']=historical
        entry['admission']=self.f.json_blob(admission)
        with self.assertRaises(ValueError):self.f.validate(self.f.registry(),entry)

    def test_current_model_generator_and_backend_bytes_are_checked(self):
        model_path=self.f.original/'npu_model/configs/numerics.py';model_path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.f.registry()
        model_path.write_bytes(b'fixture model source\n')
        tool=self.f.atlas/'tools/atlas_program.py';tool.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.f.registry()
        tool.write_bytes(b'fixture tool source\n')
        Path(self.f.backend['library_path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.f.registry()

    def test_bad_catalog_digest_and_unsafe_payload_member_are_rejected(self):
        registry=self.f.registry();(self.f.data/'catalog.json').write_text('{}')
        with self.assertRaises(ValueError):self.f.validate(registry,self.fused())
        registry=self.f.registry()
        with self.assertRaises(ValueError):registry._blob(dict(file='../blocked',bytes=0,sha256='a'*64))


if __name__=='__main__':unittest.main()
