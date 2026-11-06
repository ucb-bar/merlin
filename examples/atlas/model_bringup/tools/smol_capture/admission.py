"""Portable, opt-in native MXU0 kind admission; original47 remains the final gate.

The caller supplies a hash-pinned data catalog and current source/model/backend
bindings. Historical job helpers are archived data and are never imported.
"""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked

ORIGINAL_PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
NATIVE_PIN = '70e2e7ceb81dac86210c8b3cca54024909c57d14'


class UnadmittedKind(LookupError):
    """An absent exact kind must use original47 or acquire a new admission."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _template(source, exponent):
    _require(type(source) is str and type(exponent) is int and -126 <= exponent <= 127, 'invalid source or exponent')
    literal = re.findall(r'^vli\.all m4, ([0-9]+)$', source, re.M)
    _require(literal == [str((exponent+127)<<7)], 'source requires one validated m4 rescale literal')
    return re.sub(r'^vli\.all m4, [0-9]+$', 'vli.all m4, 0', source, flags=re.M)


def program_kind(binary, source, descriptor):
    """Normalize only the unique validated m4 literal, preserving both kind schemas."""
    exponent = descriptor['output_rescale_exponent']
    _template(source, exponent)
    _require(type(binary) is bytes and binary and len(binary)%4 == 0, 'invalid program bytes')
    dims = descriptor['physical_mkn']
    _require(len(dims)==3 and all(type(v) is int and v>0 and v%32==0 for v in dims), 'invalid physical M/K/N')
    recipe = descriptor['recipe']
    _require(recipe in ('baseline-v1', 'traffic-reuse-v1'), 'unknown projection recipe')
    operations = tuple(descriptor.get('operations', ()))
    _require(operations in ((), ('bias',), ('bias','gelu_tanh'), ('connector_scale',)), 'unsupported fusion operations')
    dtype = descriptor.get('output_dtype', 'bf16')
    _require(dtype in ('bf16','fp8') and descriptor.get('output_layout', 'tiles')=='tiles',
             'native admission requires typed tile output')
    words = list(struct.unpack('<'+'I'*(len(binary)//4), binary))
    matches = [i for i,word in enumerate(words) if word == ((exponent+127)<<23)|0x25f]
    _require(len(matches)==1 and words[-1]==0x73 and 0x73 not in words[:-1], 'invalid rescale position or ECALL')
    position = matches[0]
    words[position] &= 0x7fffff
    normalized = _sha(struct.pack('<'+'I'*len(words), *words))
    if dtype=='fp8':
        output_exponent = descriptor.get('output_scale_exponent')
        _require(recipe=='traffic-reuse-v1' and operations==('bias','gelu_tanh')
                 and descriptor['output_bytes']==dims[0]*dims[2], 'invalid FP8 producer recipe or extent')
        _require(type(output_exponent) is int and -126<=output_exponent<=119, 'invalid FP8 output scale')
        kind = dict(schema='atlas.smol_fp8_fusion_program_kind.v1', physical_mkn=list(dims), recipe=recipe,
                    operations=list(operations), output_dtype='fp8', output_layout='tiles',
                    output_bytes=descriptor['output_bytes'], output_scale_exponent=output_exponent,
                    rescale_word=position, normalized_program_sha256=normalized)
    elif operations:
        _require(recipe=='traffic-reuse-v1' and descriptor['output_bytes']==2*dims[0]*dims[2], 'invalid fused extent')
        kind = dict(schema='atlas.smol_fusion_program_kind.v1', physical_mkn=list(dims), recipe=recipe,
                    operations=list(operations), output_dtype='bf16', output_layout='tiles',
                    output_bytes=descriptor['output_bytes'], rescale_word=position, normalized_program_sha256=normalized)
    else:
        kind = dict(schema='atlas.projection_program_kind.v1', physical_mkn=list(dims), recipe=recipe,
                    rescale_word=position, normalized_program_sha256=normalized)
    return _sha(json.dumps(kind, sort_keys=True).encode()), kind


class AdmissionRegistry:
    def __init__(self, data_dir, catalog_sha256, *, atlas_root, original_model_root, native_model_root,
                 native_backend_metadata):
        self.data_dir = checked(Path(data_dir).absolute())
        self.atlas_root = checked(Path(atlas_root).absolute())
        self.original_root = checked(Path(original_model_root).absolute())
        self.native_root = checked(Path(native_model_root).absolute())
        self.backend = native_backend_metadata
        self.catalog_sha256 = catalog_sha256
        self._git_checked = False
        self.catalog = self._catalog()
        _require(self.catalog.get('schema')=='atlas.smol_native_admission_catalog.v1', 'unknown native catalog schema')
        _require(self.catalog.get('gate_of_record') is False and self.catalog.get('all_launch_original_gate_required') is True,
                 'native catalog cannot replace the original-model gate')
        self._bindings()

    def _catalog(self):
        raw = checked(self.data_dir/'catalog.json').read_bytes()
        _require(_sha(raw)==self.catalog_sha256, 'native catalog changed or has the wrong digest')
        return json.loads(raw)

    def _blob(self, record):
        _require(type(record.get('bytes')) is int and record['bytes']>=0
                 and re.fullmatch('[0-9a-f]{64}', record.get('sha256','')) is not None, 'invalid catalog payload record')
        _require(record.get('file')=='blobs/'+record['sha256'], 'catalog payload must use its content-addressed member')
        raw = checked(self.data_dir/record['file']).read_bytes()
        _require(len(raw)==record['bytes'] and _sha(raw)==record['sha256'], 'catalog payload changed')
        return raw

    def _json(self, record):
        return json.loads(self._blob(record))

    @staticmethod
    def _relative(name):
        path = Path(name)
        _require(not path.is_absolute() and '..' not in path.parts, 'unsafe source member')
        return path

    def _bindings(self):
        _require(self._catalog()==self.catalog, 'native catalog changed during use')
        required_model = {'npu_model/configs/numerics.py', 'npu_model/configs/rtl_math.py',
                          'npu_model/configs/isa_definition.py', 'npu_model/hardware/arch_state.py'}
        for role, root, revision in (('original',self.original_root,ORIGINAL_PIN), ('native',self.native_root,NATIVE_PIN)):
            _require(required_model <= set(self.catalog['model_sources'][role]), 'incomplete model source closure')
            actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
            _require(actual==revision, 'current model revision differs from admission')
            for name,record in self.catalog['model_sources'][role].items():
                path = checked(root/self._relative(name))
                raw = path.read_bytes()
                _require(raw==self._blob(record), 'current model source/data differs from admitted content')
                if not self._git_checked:
                    committed = subprocess.check_output(['git','-C',str(root),'show','HEAD:'+str(self._relative(name))])
                    _require(committed==raw, 'current model source/data is not the pinned Git content')
        required_tools = {'tools/atlas_program.py', 'tools/atlas_epilogue.py', 'tools/atlas_program_checks.py',
                          'tools/atlas_program_reuse_common.py', 'tools/smol_fusion_program.py'}
        _require(required_tools <= set(self.catalog['atlas_sources']), 'incomplete generator source closure')
        for name,record in self.catalog['atlas_sources'].items():
            # Catalogs name generators by their tools/ path; kernels now live in atlas-kernels.
            path = self.atlas_root/self._relative(name)
            if path.parent == self.atlas_root/'tools' and not path.exists():
                path = self.atlas_root/kernel_library.relative_path(path.name)
            _require(checked(path).read_bytes()==self._blob(record),
                     'current generator/harness source differs from producer admission')
        expected = self._json(self.catalog['native_backend'])
        for field in ('backend','base_commit','sources','library_sha256','compiler_version','qualification'):
            _require(self.backend.get(field)==expected[field], 'current native backend differs: '+field)
        _require(self.backend.get('environment',{}).get('NPU_MODEL_RTL_NATIVE_MXU0')=='1', 'native backend was not selected')
        for name,digest in expected['sources'].items():
            _require(Path(name).name==name, 'unsafe backend source name')
            _require(_sha(checked(self.native_root/'npu_model/configs'/name).read_bytes())==digest, 'backend source changed')
        library = checked(Path(self.backend['library_path']).absolute())
        _require(_sha(library.read_bytes())==expected['library_sha256'], 'current native library changed')
        cached = json.loads(checked(Path(self.backend['build_record']).absolute()).read_bytes())
        _require(all(cached.get(k)==v for k,v in self.backend.items() if k!='build_record'), 'native cache metadata differs')
        self._git_checked = True

    @staticmethod
    def key(binary, source, descriptor):
        return program_kind(binary, source, descriptor)[0]

    def _worker(self, entry, role, request):
        proof = entry[role]
        receipt = self._json(proof['receipt'])
        completion = self._json(proof['completion']) if entry['proof_type']=='full-producer' else None
        original = role=='original'
        model = receipt['model']
        _require(model['commit']==(ORIGINAL_PIN if original else NATIVE_PIN) and model['numerics']=='rtl', 'receipt model differs')
        execution = receipt['execution']
        binary = self._blob(entry['sample']['binary'])
        initial = self._blob(entry['sample']['initial'])
        arena = self._blob(proof['arena'])
        output = self._blob(proof['output'])
        offset,size = entry['sample']['output_offset'],entry['sample']['output_bytes']
        _require(0<=offset<=len(initial)-size and len(output)==size, 'sample output extent differs')
        _require(arena==initial[:offset]+output+initial[offset+size:], 'sample complete arena preservation failed')
        _require(execution['program_sha256']==_sha(binary) and execution['program_words']*4==len(binary)
                 and execution['halt_pc']==execution['program_words']-1 and execution['instructions']>0, 'sample program execution differs')
        _require(execution['observed']==entry['sample']['requested_dma'], 'sample requested DMA differs')
        if completion is not None:
            _require(type(completion.get('returncode')) is int and completion['returncode']==0, 'worker subprocess failed')
            _require(proof['historical_pin']['sha256']==proof['receipt']['sha256']
                     and proof['historical_pin']['bytes']==proof['receipt']['bytes'], 'worker receipt digest link differs')
            _require(completion['request']==request['historical_pin'] and completion['receipt']==proof['historical_pin'],
                     'worker completion is not bound to request and receipt')
            _require(receipt['request']==request['historical_pin'] and receipt['status']=='PASS_FULL_PRODUCER'
                     and receipt['original_model'] is original and receipt['math_override'] is False
                     and receipt['full_arena_exact'] is True and receipt['output_exact'] is True
                     and receipt['input_and_guards_preserved'] is True and receipt['source_bookends_equal'] is True,
                     'full producer receipt did not establish exact original/native execution')
            _require(receipt['source_bookends_before']==receipt['source_bookends_after'], 'producer source bookends differ')
            for historical, digest in receipt['source_bookends_after'].items():
                _require(historical in entry['source_archive'] and
                         _sha(self._blob(entry['source_archive'][historical]))==digest, 'archived source pin differs')
            for record in receipt['data_pins']:
                _require(record['path'] in entry['source_archive'] and
                         _sha(self._blob(entry['source_archive'][record['path']]))==record['sha256'], 'archived model data differs')
            _require(receipt['initial_sha256']==_sha(initial) and receipt['program_sha256']==_sha(binary)
                     and receipt['source_sha256']==_sha(self._blob(entry['sample']['source']))
                     and receipt['expected_arena_sha256']==receipt['actual_arena_sha256']==_sha(arena)
                     and receipt['expected_output_sha256']==receipt['actual_output_sha256']==_sha(output), 'worker byte identities differ')
            if original:
                _require(receipt['native_backend'] is None, 'original execution used a native backend')
            else:
                _require(receipt['native_backend']==self._json(self.catalog['native_backend']), 'native receipt backend differs')
        return receipt, arena, output

    def validate(self, binary, source, descriptor, initial=None):
        """Validate an admitted kind for new input data, plus its complete executed proof."""
        key, kind = program_kind(binary, source, descriptor)
        self._bindings()
        entry = self.catalog['kinds'].get(key)
        if entry is None:
            raise UnadmittedKind(key)
        _require(entry['kind']==kind and entry['kind_key']==key, 'catalog kind descriptor differs')
        sample = entry['sample']
        sample_source = self._blob(sample['source']).decode()
        sample_binary = self._blob(sample['binary'])
        _require(program_kind(sample_binary,sample_source,sample['descriptor'])==(key,kind), 'sample kind does not match execution')
        _require(_template(source,descriptor['output_rescale_exponent'])==
                 _template(sample_source,sample['descriptor']['output_rescale_exponent']), 'current source controls differ')
        for field in ('output_offset','output_bytes'):
            _require(descriptor[field]==sample[field], 'current output geometry differs')
        if tuple(descriptor.get('operations', ())):
            for field in ('dram_base','dram_bytes','total_dram_bytes','extra_offset','extra_bytes','scratch_registers'):
                _require(descriptor[field]==sample['descriptor'][field], 'current fused descriptor geometry differs')
            fields = ('name','offset','size_bytes','dtype','layout','scale_exponent')
            current_regions = [{k:r.get(k) for k in fields} for r in descriptor['input_regions']]
            sampled_regions = [{k:r.get(k) for k in fields} for r in sample['descriptor']['input_regions']]
            _require(current_regions==sampled_regions, 'current fused input regions differ')
        if initial is not None:
            _require(type(initial) is bytes and len(initial)==sample['initial']['bytes'], 'current arena extent differs')
            _require(descriptor.get('dram_bytes',len(initial))==len(initial), 'current descriptor arena extent differs')
        admission = self._json(entry['admission'])
        _require(admission['kind_key']==key and admission['kind']==kind and admission['full_arena_exact'] is True
                 and admission['gate_of_record'] is False, 'admission report differs')
        request = None
        if entry['proof_type']=='full-producer':
            request = dict(value=self._json(entry['request']),historical_pin=admission['request'])
            _require(admission['request']['sha256']==entry['request']['sha256']
                     and admission['request']['bytes']==entry['request']['bytes'], 'request digest link differs')
            for field in ('source','binary','initial'):
                original_pin = request['value']['pins'][field]
                _require(original_pin['sha256']==sample[field]['sha256'] and original_pin['bytes']==sample[field]['bytes'],
                         'request does not bind sample bytes')
            _require(admission['status']=='ADMITTED_NATIVE_SMOL_FUSION_KIND' and
                     admission['all_launch_original_gate_required'] is True and request['value']['kind_key']==key,
                     'new fusion kind was not fully admitted')
            for role in ('original','native'):
                _require(admission[role+'_receipt']==entry[role]['historical_pin'], 'admission worker receipt pin differs')
        else:
            _require(entry['proof_type']=='legacy' and admission['status']=='ADMITTED_NATIVE_MXU0', 'unknown proof type')
            _require(len(admission['commands'])==2 and all(type(c.get('returncode')) is int and c['returncode']==0
                     for c in admission['commands']), 'legacy worker commands did not succeed')
            self._blob(entry['harness'])
            historical = self._json(entry['historical_sample'])
            _require(historical['physical_mkn']==kind['physical_mkn'] and historical['recipe']==kind['recipe']
                     and historical['output_offset']==sample['output_offset']
                     and historical['output_bytes']==sample['output_bytes'], 'legacy sample descriptor differs')
            for command in admission['commands']:
                args = command['command']
                _require(len(args)>1 and args[1]==entry['historical_harness_path'], 'legacy command used another harness')
                for flag,field in (('--source','program_source'),('--arena','initial_arena')):
                    _require(args.count(flag)==1 and args[args.index(flag)+1]==historical[field], 'legacy command input differs')
        original, original_arena, original_output = self._worker(entry,'original',request)
        native, native_arena, native_output = self._worker(entry,'native',request)
        _require(original_arena==native_arena and original_output==native_output, 'original/native complete arenas differ')
        _require(admission['expected_full_arena_sha256']==_sha(original_arena)
                 and admission['initial_sha256']==_sha(self._blob(sample['initial'])), 'admission arena identities differ')
        _require(admission['native_backend']==self._json(self.catalog['native_backend']), 'admission backend differs')
        if entry['proof_type']=='legacy':
            _require(admission['original']==original and admission['native']==native
                     and native.get('expected_exact') is True, 'legacy result payload differs')
        return dict(kind_key=key,kind=kind,gate_of_record=False,all_launch_original_gate_required=True,
                    admission_sha256=entry['admission']['sha256'],sample_initial_sha256=sample['initial']['sha256'])
