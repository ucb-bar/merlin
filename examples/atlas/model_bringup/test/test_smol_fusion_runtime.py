"""Actual raw staging/ownership with injected mock-device outputs, not numerical execution."""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked as safe
from test_smol_policy_raw import Panel, Call, Context, Buffer

# runtime/smol_policy_raw.c before the fusion runtime, addressed by content so it survives history rewrites.
UNFUSED_RAW_BLOB = '6b464bc7780bff612570efd7f26e6c9cdbf31e6d'


def unfused_raw_source():
    if os.environ.get('SMOL_RAW_BASELINE'):
        return safe(os.environ['SMOL_RAW_BASELINE']).read_bytes()
    result = subprocess.run(['git', '-C', str(ROOT), 'cat-file', 'blob', UNFUSED_RAW_BLOB], capture_output=True)
    return result.stdout if result.returncode == 0 else None


class FusionPanel(ct.Structure):
    _fields_ = [('extra', ct.c_void_p), ('extra_offset', ct.c_size_t), ('extra_bytes', ct.c_size_t),
        ('operations', ct.c_uint), ('output_scale', ct.c_int)]


MOCK = r'''
#include "smol_policy_raw.h"
#include "smol_fusion.h"
#include <string.h>
static atlas_host host;
static unsigned char *dram;
static const unsigned char **initials, **outputs, **programs;
static size_t *sizes, *offsets, *output_bytes, *program_bytes, count, cursor, mutation_at, mutation_panel;
static unsigned mutation;
static size_t scale_calls, pack_calls;
static uint32_t ticks;
size_t unit_context_bytes(void) { return sizeof(smol_policy_raw_context); }
size_t unit_panel_bytes(void) { return sizeof(smol_policy_panel); }
size_t unit_fusion_bytes(void) { return sizeof(smol_fusion_panel); }
size_t unit_scale_calls(void) { return scale_calls; }
size_t unit_pack_calls(void) { return pack_calls; }
int __real_host_matrix_scale_bf16(const uint16_t *,size_t,size_t,int *);
int __wrap_host_matrix_scale_bf16(const uint16_t *p,size_t m,size_t n,int *s) {
    ++scale_calls; return __real_host_matrix_scale_bf16(p,m,n,s);
}
int __real_host_matrix_pack_bf16_at_scale(const uint16_t *,size_t,size_t,size_t,size_t,size_t,size_t,int,uint8_t *,size_t);
int __wrap_host_matrix_pack_bf16_at_scale(const uint16_t *p,size_t m,size_t n,size_t pm,size_t pn,
    size_t tm,size_t tn,int s,uint8_t *o,size_t capacity) {
    ++pack_calls; return __real_host_matrix_pack_bf16_at_scale(p,m,n,pm,pn,tm,tn,s,o,capacity);
}
size_t unit_completed(void) { return cursor; }
static uint32_t read32(void *p, uintptr_t a) { (void)p; (void)a; return ticks; }
void *unit_config(unsigned char *d, const unsigned char **i, const unsigned char **o, const unsigned char **p,
    size_t *s, size_t *f, size_t *z, size_t *b, size_t n) {
    memset(&host,0,sizeof(host)); host.mmio.read32=read32; host.map.csr_base=0x40000;
    dram=d; initials=i; outputs=o; programs=p; sizes=s; offsets=f; output_bytes=z; program_bytes=b;
    count=n; cursor=0; ticks=100; mutation=0; scale_calls=pack_calls=0; return &host;
}
void unit_mutate(size_t panel, size_t at, unsigned mode) { mutation_panel=panel; mutation_at=at; mutation=mode; }
void unit_sequence(void) { ++host.invocation_sequence; }
atlas_host_result atlas_host_load_raw(atlas_host *h,const uint32_t *p,size_t n) {
    if(h!=&host || cursor>=count || n*4!=program_bytes[cursor] || memcmp(p,programs[cursor],n*4)) return ATLAS_HOST_INVALID;
    h->program_loaded=1; return ATLAS_HOST_OK;
}
atlas_host_result atlas_host_start(atlas_host *h,const atlas_host_buffer *b,size_t n) {
    if(h!=&host || cursor>=count || n!=1 || b->address!=(uintptr_t)dram || b->size_bytes!=sizes[cursor]
        || memcmp(dram,initials[cursor],sizes[cursor])) return ATLAS_HOST_INVALID;
    h->active=1; ++h->invocation_sequence; return ATLAS_HOST_OK;
}
atlas_host_result atlas_host_wait(atlas_host *h,size_t polls) {
    if(h!=&host || !polls || !h->active) return ATLAS_HOST_INVALID;
    memcpy(dram+offsets[cursor],outputs[cursor],output_bytes[cursor]);
    if(mutation && mutation_panel==cursor) {
        if(mutation==1) dram[mutation_at]^=1;
        if(mutation==2) { dram[mutation_at]=0x80; dram[mutation_at+1]=0x7f; }
        if(mutation==3) { dram[mutation_at]=0; dram[mutation_at+1]=0x80; }
        if(mutation==4) dram[mutation_at]=0x7f;
        if(mutation==5) dram[mutation_at]=0xff;
    }
    h->active=0; ticks+=100; ++cursor; return ATLAS_HOST_OK;
}
'''


class FusionRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merlin = safe(os.environ.get('SMOL_MERLIN', str(ROOT.parent / 'merlin/examples/atlas/comparisons')))
        if not (cls.merlin / 'host_matrix_fp8pack.c').is_file():
            raise unittest.SkipTest('set SMOL_MERLIN to a Merlin examples/atlas/comparisons checkout')
        temporary = safe(os.environ.get('TMPDIR', str(ROOT.parent / 'tmp')))
        temporary.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix='fusion-runtime-', dir=temporary)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = safe(cls.temporary.name)
        mock = cls.work / 'mock.c'
        mock.write_text(MOCK)
        library = cls.work / 'runtime.so'
        cls.command = ['/usr/bin/gcc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
            '-DSMOL_FUSION_RUNTIME=1', '-ffp-contract=off', '-fno-fast-math', '-I'+str(ROOT / 'runtime'),
            '-I'+str(cls.merlin), str(ROOT / 'runtime/smol_policy_raw.c'), str(ROOT / 'runtime/smol_fusion.c'),
            str(cls.merlin / 'host_matrix_fp8pack.c'), str(mock), '-lm', '-o', str(library)]
        cls.command += ['-Wl,--wrap=host_matrix_scale_bf16,--wrap=host_matrix_pack_bf16_at_scale']
        subprocess.run(cls.command, check=True)
        cls.lib = ct.CDLL(str(library))
        cls.lib.unit_config.argtypes = [ct.c_void_p]*8 + [ct.c_size_t]
        cls.lib.unit_config.restype = ct.c_void_p
        cls.lib.unit_completed.restype = ct.c_size_t
        cls.lib.unit_scale_calls.restype = cls.lib.unit_pack_calls.restype = ct.c_size_t
        cls.lib.unit_mutate.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_uint]
        cls.lib.smol_fusion_configure.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        cls.lib.smol_fusion_input_view.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p]
        cls.lib.smol_policy_raw_vision.argtypes = [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
            ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t]
        cls.lib.host_matrix_scale_bf16.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p]
        cls.lib.host_matrix_pack_bf16_at_scale.argtypes = [ct.c_void_p] + [ct.c_size_t]*6 + [ct.c_int, ct.c_void_p, ct.c_size_t]
        for name in ('smol_fusion_vision_gelu_completed', 'smol_fusion_vision_connector_scale_completed'):
            getattr(cls.lib, name).argtypes = [ct.c_uint, ct.c_void_p, ct.c_size_t, ct.c_size_t]
        for name, expected in [('unit_context_bytes', ct.sizeof(Context)), ('unit_panel_bytes', ct.sizeof(Panel)),
            ('unit_fusion_bytes', ct.sizeof(FusionPanel))]:
            getattr(cls.lib, name).restype = ct.c_size_t
            if getattr(cls.lib, name)() != expected: raise ValueError('changed raw ABI')

    def setUp(self):
        self.assertEqual(self.lib.smol_fusion_configure(None, None, 0), 0)

    def tearDown(self):
        self.lib.smol_fusion_configure(None, None, 0)

    def _transport(self, f, calls, call_count, initials, outputs, programs, activation, conversion):
        count = len(f.panels)
        f.dram=np.full(max(p.dram_bytes for p in f.panels),0x93,np.uint8)
        f.activation=np.zeros(activation,np.uint8);f.conversion=np.zeros(conversion,np.uint16)
        f.cycles=np.zeros(count,np.uint32)
        pointer=lambda xs:(ct.c_void_p*len(xs))(*[x.ctypes.data for x in xs])
        ip,op,pp=pointer(initials),pointer(outputs),pointer(programs)
        arrays=[(ct.c_size_t*count)(*[getattr(p,name) for p in f.panels])
            for name in ('dram_bytes','output_offset','store_bytes')]
        sizes=(ct.c_size_t*count)(*[program.nbytes for program in programs])
        f.keep.extend([ip,op,pp,*arrays,sizes])
        host=self.lib.unit_config(f.dram.ctypes.data,ip,op,pp,*arrays,sizes,count)
        f.context=Context(host,ct.addressof(f.panels),ct.addressof(calls),count,call_count,0,0,f.dram.ctypes.data,
            f.activation.ctypes.data,f.conversion.ctypes.data,f.dram.size,f.activation.size,f.conversion.size,1000,
            f.cycles.ctypes.data,Buffer(),f.image,0,0)

    def fixture(self, operations=3, role=5, rows=256, k=32, n=32):
        if operations == 4: role, rows, k, n = 7, 64, 12288, 960
        f = SimpleNamespace(rows=rows, k=k, n=n, role=role, layer=12 if role==7 else 2, image=1, keep=[])
        f.input = np.full((rows,k), 0x3f80, np.uint16)
        f.output = np.full((rows,n), 0x3555, np.uint16)
        f.bias = np.full(n, 0x3f80, np.uint16)
        scale = ct.c_int()
        self.assertEqual(self.lib.host_matrix_scale_bf16(f.input.ctypes.data, rows, k, ct.byref(scale)), 0)
        pk, pn = (k+31)//32*32, (n+31)//32*32
        f.program = np.array([((scale.value+127)<<23)|0x25f, 0x73], np.uint32)
        f.weights = np.zeros(pk*pn, np.uint8)
        f.extra = np.tile(f.bias.reshape(-1,2,1,16), (1,1,32,1)).reshape(-1).view(np.uint8) if operations&1 else np.array([],np.uint8)
        count = (rows+127)//128
        f.panels, f.metadata = (Panel*count)(), (FusionPanel*count)()
        initials, outputs, programs = [], [], []
        for index, start in enumerate(range(0,rows,128)):
            logical = min(128, rows-start); m=(logical+31)//32*32
            activation=np.zeros(m*pk,np.uint8)
            self.assertEqual(self.lib.host_matrix_pack_bf16_at_scale(f.input[start:].ctypes.data,logical,k,m,pk,32,32,
                scale.value,activation.ctypes.data,activation.size),0)
            offset=m*pk+f.weights.size+256; old_end=offset+2*m*pn+256
            size=old_end+(f.extra.size+256 if operations&1 else 0)
            physical=np.zeros((m,pn),np.uint16);physical[:logical,:n]=0x41f8 if operations==4 else 0x4000
            out=physical.reshape(m//32,32,pn//32,2,16).transpose(0,2,3,1,4).copy().reshape(-1).view(np.uint8)
            initial=np.concatenate([activation,f.weights,np.full(256,0xa5,np.uint8),np.full(2*m*pn,0xcd,np.uint8),
                np.full(256,0x5a,np.uint8),f.extra,np.full(256,0x5a,np.uint8) if operations&1 else np.array([],np.uint8)])
            self.assertEqual(initial.size,size)
            f.panels[index]=Panel(f.program.ctypes.data,f.weights.ctypes.data,2,0,m,pk,pn,start,logical,k,n,0,
                offset,size,f.weights.size,m*pk+f.weights.size+f.extra.size,2*m*pn,scale.value,0)
            f.metadata[index]=FusionPanel(f.extra.ctypes.data if f.extra.size else None,old_end if f.extra.size else 0,f.extra.size,operations)
            initials.append(initial);outputs.append(out);programs.append(f.program);f.keep.extend([activation,initial,out])
        f.call=Call(rows,k,n,0,count,f.bias.ctypes.data if role!=7 else None,7,f.layer,role,f.image,0,0,0,scale.value)
        self._transport(f,f.call,1,initials,outputs,programs,128*pk,rows*k)
        f.initials,f.outputs,f.programs=initials,outputs,programs
        return f

    def configure(self, f):
        return self.lib.smol_fusion_configure(ct.byref(f.context), f.metadata, len(f.metadata))

    def invoke(self, f):
        return self.lib.smol_policy_raw_vision(ct.byref(f.context),f.layer,f.role,f.input.ctypes.data,f.rows,f.k,f.output.ctypes.data,f.n)

    def query(self, f, **changes):
        args=dict(layer=f.layer,pointer=f.output.ctypes.data,rows=f.rows,width=f.n);args.update(changes)
        method=self.lib.smol_fusion_vision_connector_scale_completed if f.role==7 else self.lib.smol_fusion_vision_gelu_completed
        return method(args['layer'],args['pointer'],args['rows'],args['width'])

    def fp8_pair(self, rows=1024, k=768, hidden=3072, width=768, scale=-5):
        a=self.fixture(operations=3,role=5,rows=rows,k=k,n=hidden)
        b=self.fixture(operations=0,role=6,rows=rows,k=hidden,n=width)
        b.call.activation_scale=scale
        b.program[0]=((scale+127)<<23)|0x25f
        for i,p in enumerate(a.panels):
            m,pn=p.physical_rows,p.physical_n
            p.store_bytes=m*pn
            old_end=p.output_offset+p.store_bytes+256
            p.dram_bytes=old_end+a.extra.size+256
            a.metadata[i]=FusionPanel(a.extra.ctypes.data,old_end,a.extra.size,11,scale)
            logical=np.zeros((m,pn),np.uint8)
            values=(np.arange(p.logical_rows*hidden,dtype=np.uint32)+p.row_start*hidden)%254
            values=np.where(values>=127,values+1,values).astype(np.uint8)
            logical[:p.logical_rows,:hidden]=values.reshape(p.logical_rows,hidden)
            a.outputs[i]=logical.reshape(m//32,32,pn//32,32).transpose(0,2,1,3).copy().reshape(-1)
            a.initials[i]=np.concatenate([a.initials[i][:p.output_offset],np.full(p.store_bytes,0xcd,np.uint8),
                np.full(256,0x5a,np.uint8),a.extra,np.full(256,0x5a,np.uint8)])
            b.panels[i].call=1;b.panels[i].activation_scale=scale
            b.initials[i][:m*pn]=a.outputs[i]
        b.call.first_panel=len(a.panels)
        f=SimpleNamespace(producer=a,consumer=b,rows=rows,k=k,n=hidden,role=5,layer=a.layer,image=a.image,
            input=a.input,output=a.output,scale=scale,keep=[a,b])
        f.panels=(Panel*(len(a.panels)+len(b.panels)))(*a.panels,*b.panels)
        f.metadata=(FusionPanel*len(f.panels))(*a.metadata,*b.metadata)
        f.calls=(Call*2)(a.call,b.call)
        initials=a.initials+b.initials;outputs=a.outputs+b.outputs;programs=a.programs+b.programs
        f.keep.extend([initials,outputs,programs])
        self._transport(f,f.calls,2,initials,outputs,programs,128*hidden,rows*max(k,hidden))
        return f

    def input_view(self, f, **changes):
        args=dict(raw=ct.byref(f.context),pointer=f.output.ctypes.data,rows=f.rows,width=f.n);args.update(changes)
        scale=ct.c_int(123)
        result=self.lib.smol_fusion_input_view(args['raw'],args['pointer'],args['rows'],args['width'],ct.byref(scale))
        return result,scale.value

    def invoke_consumer(self, f):
        b=f.consumer
        return self.lib.smol_policy_raw_vision(ct.byref(f.context),b.layer,b.role,f.output.ctypes.data,
            b.rows,b.k,b.output.ctypes.data,b.n)

    def test_fp8_eight_panel_consumer_bypasses_conversion_scan_and_pack(self):
        f=self.fp8_pair()
        self.assertEqual(ct.sizeof(FusionPanel),32)
        self.assertEqual(self.configure(f),0);self.assertEqual(self.input_view(f),(0,123))
        self.assertEqual(self.invoke(f),0);self.assertEqual(self.query(f),1)
        expected=np.concatenate(f.producer.outputs)
        np.testing.assert_array_equal(f.output.view(np.uint8).reshape(-1)[:expected.size],expected)
        self.assertTrue(np.all(f.output.view(np.uint8).reshape(-1)[expected.size:]==np.tile([0x55,0x35],expected.size//2)))
        self.assertEqual(self.input_view(f),(1,-5));self.assertEqual(self.input_view(f),(1,-5))
        self.assertEqual((self.lib.unit_scale_calls(),self.lib.unit_pack_calls()),(1,8))
        f.conversion[:]=0x5a5a
        self.assertEqual(self.invoke_consumer(f),0)
        self.assertEqual((f.context.call_cursor,f.context.panel_cursor,self.lib.unit_completed()),(2,16,16))
        self.assertEqual((self.lib.unit_scale_calls(),self.lib.unit_pack_calls()),(1,8))
        self.assertTrue(np.all(f.conversion==0x5a5a));self.assertTrue(np.all(f.consumer.output==0x4040))
        self.assertEqual(self.input_view(f),(0,123))

    def test_fp8_view_mismatch_refusals(self):
        for kind in ('pointer','rows','width','role','layer','image','call_image','scale','panel_scale','panel_cursor','sequence','attention'):
            with self.subTest(kind=kind):
                f=self.fp8_pair(rows=32,k=32,hidden=32,width=32)
                self.assertEqual(self.configure(f),0);self.assertEqual(self.invoke(f),0)
                changes={}
                if kind=='pointer':changes['pointer']=f.output.ctypes.data+1
                if kind=='rows':changes['rows']=f.rows-1
                if kind=='width':changes['width']=f.n-1
                if kind=='role':f.calls[1].role=4
                if kind=='layer':f.calls[1].layer+=1
                if kind=='image':f.context.image+=1
                if kind=='call_image':f.calls[1].image+=1
                if kind=='scale':f.calls[1].activation_scale+=1
                if kind=='panel_scale':f.panels[1].activation_scale+=1
                if kind=='panel_cursor':f.context.panel_cursor+=1
                if kind=='sequence':self.lib.unit_sequence()
                if kind=='attention':f.context.attention_cursor+=1
                self.assertLess(self.input_view(f,**changes)[0],0)
                if kind in ('pointer','rows','width'):self.assertEqual(self.input_view(f),(1,-5))

    def test_fp8_expected_view_requires_actual_producer_success(self):
        f=self.fp8_pair(rows=32,k=32,hidden=32,width=32)
        self.assertEqual(self.configure(f),0)
        f.context.call_cursor=1;f.context.panel_cursor=1;self.lib.unit_sequence()
        self.assertLess(self.input_view(f)[0],0)

    def test_fp8_consumer_refuses_wrong_view_before_host_math(self):
        for kind in ('pointer','scale','role','layer','image','rows'):
            with self.subTest(kind=kind):
                f=self.fp8_pair(rows=32,k=32,hidden=32,width=32)
                self.assertEqual(self.configure(f),0);self.assertEqual(self.invoke(f),0)
                pointer,rows=f.output.ctypes.data,f.rows
                if kind=='pointer':pointer+=1
                if kind=='scale':f.calls[1].activation_scale+=1
                if kind=='role':f.calls[1].role=4
                if kind=='layer':f.calls[1].layer+=1
                if kind=='image':f.context.image+=1
                if kind=='rows':rows-=1
                f.conversion[:]=0x5a5a
                self.assertEqual(self.lib.smol_policy_raw_vision(ct.byref(f.context),f.layer,6,pointer,
                    rows,f.n,f.consumer.output.ctypes.data,f.consumer.n),1)
                self.assertEqual((self.lib.unit_scale_calls(),self.lib.unit_pack_calls(),self.lib.unit_completed()),(1,1,1))
                self.assertTrue(np.all(f.conversion==0x5a5a));self.assertTrue(np.all(f.consumer.output==0x3555))

    def test_fp8_consumer_input_and_weights_still_preserved(self):
        for region in ('activation','weight'):
            with self.subTest(region=region):
                f=self.fp8_pair(rows=32,k=32,hidden=32,width=32)
                self.assertEqual(self.configure(f),0);self.assertEqual(self.invoke(f),0)
                p=f.panels[1]
                self.lib.unit_mutate(1,0 if region=='activation' else p.physical_rows*p.physical_k,1)
                self.assertEqual(self.invoke_consumer(f),5);self.assertEqual(f.context.call_cursor,1)
                self.assertTrue(np.all(f.consumer.output==0x3555));self.assertLess(self.input_view(f)[0],0)

    def test_fp8_metadata_pair_refusals(self):
        for kind in ('without_gelu','wrong_producer','scale','nonfp8_scale','consumer_layer','consumer_role','partition'):
            with self.subTest(kind=kind):
                f=self.fp8_pair(rows=32,k=32,hidden=32,width=32)
                if kind=='without_gelu':f.metadata[0].operations=9
                if kind=='wrong_producer':f.calls[0].role=4
                if kind=='scale':f.metadata[0].output_scale+=1
                if kind=='nonfp8_scale':f.metadata[1].output_scale=1
                if kind=='consumer_layer':f.calls[1].layer+=1
                if kind=='consumer_role':f.calls[1].role=4
                if kind=='partition':f.panels[1].row_start=32
                self.assertLess(self.configure(f),0)

    def test_fp8_output_nan_and_padding_refusals(self):
        for mode in (1,4,5):
            with self.subTest(mode=mode):
                f=self.fp8_pair(rows=17,k=32,hidden=32,width=32)
                self.assertEqual(self.configure(f),0)
                offset=f.panels[0].output_offset+(17*32 if mode==1 else 0)
                self.lib.unit_mutate(0,offset,mode)
                self.assertEqual(self.invoke(f),6);self.assertEqual(f.context.call_cursor,0)
                self.assertTrue(np.all(f.output==0x3555));self.assertLess(self.input_view(f)[0],0)

    def test_full_eight_panel_fc1_staging(self):
        f=self.fixture(rows=1024,k=768,n=3072)
        self.assertEqual(self.configure(f),0);self.assertEqual(self.invoke(f),0)
        self.assertEqual((f.context.call_cursor,f.context.panel_cursor,self.lib.unit_completed()),(1,8,8))
        self.assertTrue(np.all(f.output==0x4000));self.assertTrue(np.all(f.cycles==100));self.assertEqual(self.query(f),1)

    def test_bias_only_and_unfused_bias_ownership(self):
        for operations in (0,1,3):
            with self.subTest(operations=operations):
                f=self.fixture(operations=operations)
                self.assertEqual(self.configure(f),0);self.assertLess(self.query(f),0);self.assertEqual(self.invoke(f),0)
                self.assertTrue(np.all(f.output==(0x4000 if operations else 0x4040)))
                self.assertEqual(self.query(f),int(bool(operations&2)))

    def test_unconfigured_and_connector(self):
        f=self.fixture(operations=0)
        self.assertEqual(self.query(f),0);self.assertEqual(self.invoke(f),0);self.assertEqual(self.query(f),0)
        f=self.fixture(operations=4)
        self.assertEqual(self.configure(f),0);self.assertEqual(self.invoke(f),0)
        self.assertTrue(np.all(f.output==0x41f8));self.assertEqual(self.query(f),1)

    def test_metadata_refusals(self):
        mutations={
            'count':lambda f:setattr(f.context,'panel_count',len(f.panels)+1),
            'unknown':lambda f:setattr(f.metadata[0],'operations',8),
            'inconsistent':lambda f:setattr(f.metadata[1],'operations',1),
            'gelu_without_bias':lambda f:setattr(f.metadata[0],'operations',2),
            'wrong_site':lambda f:setattr(f.call,'site',5),
            'wrong_role':lambda f:setattr(f.call,'role',4),
            'fp32_bias':lambda f:setattr(f.call,'bias_dtype',1),
            'fp32_input':lambda f:setattr(f.call,'input_dtype',1),
            'fp32_output':lambda f:setattr(f.call,'output_dtype',1),
            'missing_bias':lambda f:setattr(f.call,'bias',None),
            'wrong_offset':lambda f:setattr(f.metadata[0],'extra_offset',f.metadata[0].extra_offset+32),
            'wrong_extent':lambda f:setattr(f.metadata[0],'extra_bytes',f.metadata[0].extra_bytes-32),
            'wrong_arena':lambda f:setattr(f.panels[0],'dram_bytes',f.panels[0].dram_bytes-256),
            'wrong_bias_tile':lambda f:f.extra.__setitem__(0,1),
            'extra_alias':lambda f:setattr(f.metadata[0],'extra',f.dram.ctypes.data),
            'panel_owner':lambda f:setattr(f.panels[0],'call',1),
            'call_coverage':lambda f:setattr(f.call,'first_panel',1),
            'overflow_rows':lambda f:setattr(f.panels[0],'physical_rows',ct.c_size_t(-1).value),
            'overflow_offset':lambda f:setattr(f.panels[0],'output_offset',ct.c_size_t(-1).value),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                f=self.fixture();mutate(f)
                self.assertLess(self.configure(f),0);self.assertLess(self.query(f),0)
                self.assertEqual(self.lib.unit_completed(),0)

    def test_connector_constraints(self):
        for field,value in (('role',6),('layer',11),('rows',32),('output_width',928),('bias_dtype',1)):
            with self.subTest(field=field):
                f=self.fixture(operations=4)
                if field=='bias_dtype':f.call.bias=f.bias.ctypes.data
                else:setattr(f.call,field,value)
                self.assertLess(self.configure(f),0)

    def test_runtime_metadata_mutation_rejected(self):
        f=self.fixture();self.assertEqual(self.configure(f),0)
        f.metadata[1].operations=1
        self.assertEqual(self.invoke(f),1);self.assertEqual(f.context.failed,1)
        self.assertEqual(self.lib.unit_completed(),0);self.assertLess(self.query(f),0)

    def test_all_staged_regions_preserved(self):
        for region in ('activation','weight','before_output','old_guard','extra','tail'):
            with self.subTest(region=region):
                f=self.fixture();p=f.panels[0];meta=f.metadata[0]
                offsets=dict(activation=0,weight=p.physical_rows*p.physical_k,before_output=p.output_offset-1,
                    old_guard=p.output_offset+p.store_bytes,extra=meta.extra_offset,tail=p.dram_bytes-1)
                self.assertEqual(self.configure(f),0);self.lib.unit_mutate(0,offsets[region],1)
                self.assertNotEqual(self.invoke(f),0);self.assertEqual(f.context.failed,1)
                self.assertEqual(f.context.call_cursor,0);self.assertTrue(np.all(f.output==0x3555));self.assertLess(self.query(f),0)

    def test_nonfinite_and_padding_rejected(self):
        for mode in (2,3):
            with self.subTest(mode=mode):
                f=self.fixture(operations=0,rows=1)
                self.assertEqual(self.configure(f),0)
                at=f.panels[0].output_offset+(32 if mode==3 else 0)
                self.lib.unit_mutate(0,at,mode)
                self.assertEqual(self.invoke(f),6);self.assertTrue(np.all(f.output==0x3555));self.assertLess(self.query(f),0)

    def test_success_identity_refusals(self):
        for kind in ('layer','pointer','rows','width','image','sequence','cursor','attention','failed'):
            with self.subTest(kind=kind):
                f=self.fixture();self.assertEqual(self.configure(f),0);self.assertEqual(self.invoke(f),0)
                changes={}
                if kind=='layer':changes['layer']=f.layer+1
                if kind=='pointer':changes['pointer']=f.output.ctypes.data+2
                if kind=='rows':changes['rows']=f.rows-1
                if kind=='width':changes['width']=f.n-1
                if kind=='image':f.context.image+=1
                if kind=='sequence':self.lib.unit_sequence()
                if kind=='cursor':f.context.call_cursor+=1
                if kind=='attention':f.context.attention_cursor+=1
                if kind=='failed':f.context.failed=1
                self.assertLess(self.query(f,**changes),0)

    def test_macro_disabled_object_identity(self):
        raw=unfused_raw_source()
        if raw is None:self.skipTest('set SMOL_RAW_BASELINE or run in a git checkout that contains blob '+UNFUSED_RAW_BLOB)
        baseline=self.work / 'unfused/smol_policy_raw.c'
        baseline.parent.mkdir(exist_ok=True)
        baseline.write_bytes(raw)
        objects=[]
        for index,(source,flags) in enumerate(((baseline,[]),(ROOT / 'runtime/smol_policy_raw.c',[]),
            (ROOT / 'runtime/smol_policy_raw.c',['-DSMOL_FUSION_RUNTIME=0']))):
            target=self.work / f'disabled-{index}.o'
            subprocess.run(['/usr/bin/gcc','-std=c11','-O2','-ffp-contract=off','-fno-fast-math',
                *flags,'-I'+str(ROOT / 'runtime'),'-I'+str(self.merlin),'-c',str(source),'-o',str(target)],check=True)
            objects.append(target.read_bytes())
        self.assertEqual(objects[0],objects[1])
        self.assertEqual(objects[0],objects[2])
        print(json.dumps(dict(macro_disabled_raw_object_sha256=hashlib.sha256(objects[0]).hexdigest(),
            ctypes_context_bytes=ct.sizeof(Context),ctypes_panel_bytes=ct.sizeof(Panel))))


if __name__=='__main__':unittest.main()
