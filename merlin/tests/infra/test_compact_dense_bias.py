"""Explicit biased accumulator initialization, actual operands and native output."""
from dataclasses import replace
import ctypes
import subprocess
import numpy as np
import pytest
from test_compact_dense import encode as plain_encode, project as plain_project, fixture, mutate
from merlin.compile.scheduling.block_schedule import InitializeAccumulator
from merlin.compile.scheduling.compact_dense import DenseBiasSource, DenseLoopAdapter, schedule_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.sched.ir import Call, as_expr, instances
from merlin.sched.isa import InstrDef, InstructionSet, Operand


def encode(op):
    if not isinstance(op,InitializeAccumulator):return plain_encode(op)
    names=('m','n','src','rows','cols','dst','row_stride','col_stride')
    values=(*op.block,op.dram_col,op.rows,op.cols,op.row,op.source_row_stride,op.source_col_stride)
    return Call('bias_init',tuple(zip(names,map(as_expr,values))))


def project(call):
    if call.instr!='bias_init':return plain_project(call)
    a=dict(call.args)
    return InitializeAccumulator((a['m'],a['n']),a['src'],a['rows'],a['cols'],a['dst'],a['row_stride'],a['col_stride'])


ADAPTER=DenseLoopAdapter(encode,project,'neutral-bias','signed-modular-add',True,True,64)


def build(shape=(5,14,13)):
    p,r,s=fixture(shape=shape)
    p=replace(p,bias=True);bias=DenseBiasSource(p.n,'i32')
    result=schedule_compact_dense(p,r,s,ADAPTER,name='biased',dtypes=('i8','i8','i32'),
        static_budget=80,external_buffers_disjoint=True,verification_timeout_ms=20000,bias_source=bias)
    return p,r,s,bias,result


@pytest.mark.parametrize('shape',[(5,14,13),(1,7,3)])
def test_emitted_counted_c_consumes_runtime_bias_and_preserves_wrap_tails(tmp_path_factory,shape):
    # Retain each loaded library: per-test cleanup can recycle its inode while
    # dlopen still caches the previous image, corrupting the next shape ABI.
    tmp_path=tmp_path_factory.mktemp("bias-native")
    p,r,s,bias,result=build(shape)
    definitions={}
    for call,_ in instances(result.kernel):
        suffix={'lhs_load':',lhs','weight_load':',rhs','store':',output','bias_init':',bias'}.get(call.instr,'')
        definitions[call.instr]=InstrDef(call.instr,tuple(Operand(k,'int')for k,_ in call.args),
            render_c=lambda a,n=call.instr,end=suffix:f"do_{n}({', '.join(a)}{end});")
    iset=InstructionSet('neutral-bias',definitions,{'i8':'int8_t','i32':'int32_t'},'')
    source=f"""#include <stdint.h>
#include <string.h>
static int32_t left[128][16],right[128][16];
static uint32_t acc[128][16];
static int64_t wr,ar,ncols,continuing;
static void do_lhs_load(int64_t a,int64_t b,int64_t row,int64_t col,int64_t rows,int64_t cols,int64_t dst,int64_t step,const int8_t*x){{for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)left[dst+i][j]=x[(row+i)*{p.k}+col+j];}}
static void do_weight_load(int64_t a,int64_t b,int64_t row,int64_t col,int64_t rows,int64_t cols,int64_t dst,int64_t step,const int8_t*x){{for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)right[dst+i][j]=x[(row+i)*{p.n}+col+j];}}
static void do_bias_init(int64_t m,int64_t n,int64_t col,int64_t rows,int64_t cols,int64_t dst,int64_t rs,int64_t cs,const int32_t*x){{for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)acc[dst+i][j]=(uint32_t)x[col+i*rs+j*cs];}}
static void do_preload(int64_t weight,int64_t k,int64_t n,int64_t dest,int64_t cont,int64_t rows,int64_t cols,int64_t depth){{wr=weight;ar=dest;continuing=cont;ncols=cols;}}
static void do_compute(int64_t lhs,int64_t m,int64_t k,int64_t rows,int64_t depth){{for(int i=0;i<rows;i++)for(int j=0;j<ncols;j++){{uint32_t v=continuing?acc[ar+i][j]:0;for(int k=0;k<depth;k++)v+=(uint32_t)(left[lhs+i][k]*right[wr+k][j]);acc[ar+i][j]=v;}}}}
static void do_store(int64_t row,int64_t col,int64_t rows,int64_t cols,int64_t src,int32_t*out){{for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)memcpy(out+(row+i)*{p.n}+col+j,&acc[src+i][j],4);}}
"""+result.emit_c(iset,symbol='biased')
    source += "void run(int8_t*a,int8_t*b,int32_t*bias,int32_t*out){biased(a,b,bias,out);}"
    assert 'for (int64_t' in source
    c=tmp_path/'bias.c';c.write_text(source);so=tmp_path/'bias.so'
    subprocess.run(['cc','-shared','-fPIC','-O2',str(c),'-o',str(so)],check=True,capture_output=True)
    fn=ctypes.CDLL(str(so)).run;fn.argtypes=[ctypes.c_void_p]*4;fn.restype=None
    a=((np.arange(p.m*p.k)%13)-6).astype(np.int8).reshape(p.m,p.k)
    b=((np.arange(p.k*p.n)%11)-5).astype(np.int8).reshape(p.k,p.n)
    for bias_values in [np.arange(p.n,dtype=np.int32)-7,np.resize(np.array([2**31-1,-2**31,0,127,-128],dtype=np.int32),p.n)]:
        out=np.full((p.m,p.n),-111,dtype=np.int32)
        fn(a.ctypes.data,b.ctypes.data,bias_values.ctypes.data,out.ctypes.data)
        expected=(a.astype(np.int64)@b.astype(np.int64)+bias_values).astype(np.int32)
        np.testing.assert_array_equal(out,expected)
    assert result.realization['bias_initializations']==p.tiles[0]*p.tiles[2]
    assert result.realization['requested_bias_load_bytes']==p.tiles[0]*p.n*4


@pytest.fixture(scope='module')
def prepared():return build()


@pytest.mark.parametrize('instruction,operand,value,omit',[
 ('bias_init',None,None,True),('bias_init','src',1,False),('bias_init','row_stride',1,False),
 ('bias_init','col_stride',0,False),('bias_init','dst',100,False),('preload','continue',0,False)])
def test_actual_bad_initialization_refused(prepared,instruction,operand,value,omit):
    p,r,s,bias,result=prepared
    kernel=mutate(result.kernel,instruction,operand,value,omit=omit)
    with pytest.raises(ValueError):verify_compact_dense(kernel,p,r,s,ADAPTER,static_budget=80,timeout_ms=20000,bias_source=bias)


def test_bias_contract_cannot_change_width_or_erase_source(prepared):
    p,r,s,bias,result=prepared
    for source in (None,DenseBiasSource(p.n,'i8'),DenseBiasSource(1,'i32'),DenseBiasSource(p.n,'i32','saturating-add')):
        with pytest.raises(ValueError):verify_compact_dense(result.kernel,p,r,s,ADAPTER,static_budget=80,bias_source=source)
