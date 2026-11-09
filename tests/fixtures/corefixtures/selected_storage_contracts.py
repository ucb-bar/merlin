import ctypes
import shutil
import subprocess

import pytest

STORAGE_EXECUTOR = "\n#include <stdint.h>\n#include <string.h>\nstatic uint64_t number(const unsigned char *p,int width,int big) {\n uint64_t x=0; for(int i=0;i<width;i++) x|=(uint64_t)p[i] << (8*(big?width-1-i:i)); return x;\n}\nint run(const unsigned char *code,int bytes,int width,int big,int separate,int faulty_spill,\n        unsigned char *first,unsigned char *second,unsigned char *input,unsigned char *output,\n        int capacity,int external,uint64_t *counts) {\n unsigned char *memory[4]={first,separate?second:first,input,output};\n int limits[4]={capacity,capacity,external,external};\n int pending=0,pr=0,pw=0; uint64_t ps=0,pd=0,pn=0;\n memset(counts,0,5*sizeof(*counts));\n int word=3+3*width;\n for(int pc=0;pc<bytes;pc+=word) {\n  int op=code[pc],r=code[pc+1],w=code[pc+2];\n  uint64_t s=number(code+pc+3,width,big), d=number(code+pc+3+width,width,big);\n  uint64_t n=number(code+pc+3+2*width,width,big);\n  if(r>3 || w>3 || pc+word>bytes) return 1;\n  counts[0]++;\n  if(op==3) {\n   if(!pending || n!=pn) return 2;\n   if(faulty_spill!=2) memmove(memory[pw]+pd,memory[pr]+ps,pn);\n   pending=0; continue;\n  }\n  if(n==0 || s+n>(uint64_t)limits[r] || d+n>(uint64_t)limits[w]) return 3;\n  counts[1]+=n; counts[2]+=n;\n  if(r<2 && s+n>counts[3+(r&&separate)]) counts[3+(r&&separate)]=s+n;\n  if(w<2 && d+n>counts[3+(w&&separate)]) counts[3+(w&&separate)]=d+n;\n  if(op==2) { if(pending) return 4; pending=1; pr=r; pw=w; ps=s; pd=d; pn=n; }\n  else if(op==0 || op==4) {\n   memmove(memory[w]+d,memory[r]+s,n);\n   if(faulty_spill==1 && r==0 && w==1) memory[w][d]++;\n  }\n  else if(op==1) { for(uint64_t i=0;i<n;i++) memory[w][d+i]=(unsigned char)(memory[r][s+i]+1); }\n  else return 5;\n }\n return pending?6:0;\n}\n"


@pytest.fixture
def native(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    source = tmp_path / "machine.c"
    source.write_text(STORAGE_EXECUTOR)
    library = tmp_path / "machine.so"
    subprocess.run(
        [cc, "-std=c11", "-O1", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True
    )
    machine = ctypes.CDLL(str(library))
    bytep = ctypes.POINTER(ctypes.c_ubyte)
    machine.run.argtypes = [
        bytep,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        bytep,
        bytep,
        bytep,
        bytep,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]

    def execute(
        image,
        first,
        second,
        data,
        output,
        width=8,
        big=False,
        separate=False,
        faulty_spill=False,
        faulty_completion=False,
    ):

        def buffer(array):
            return (ctypes.c_ubyte * len(array)).from_buffer(array)

        counts = (ctypes.c_uint64 * 5)()
        code = bytearray(image)
        fault = 2 if faulty_completion else int(faulty_spill)
        status = machine.run(
            buffer(code),
            len(code),
            width,
            big,
            separate,
            fault,
            buffer(first),
            buffer(second),
            buffer(data),
            buffer(output),
            len(first),
            len(data),
            counts,
        )
        assert status == 0, status
        return tuple(counts)

    return execute
