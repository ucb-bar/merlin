"""A generated C call observes actual result shapes and copies owned payloads."""

import hashlib
import subprocess

import numpy as np
import pytest
from test_owned_result_execution import COMPACT

from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_result_abi import ScalarSlot, TensorSlot, bridge
from merlin.llvmlower.owned_result_program import load, save
from merlin.llvmlower.owned_results import compile_owned_results

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")

DRIVER = r'''
#include "owned_result_views.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern int merlin_owned_result_invoke_checked(void **,MerlinOwnedResultView *,size_t);
typedef struct {void *allocated,*aligned;intptr_t offset,sizes[1],strides[1];} D1;
int main(int argc,char **argv) {
  if(argc>65) return 1;
  int32_t words[64], original[64];
  for(int i=1;i<argc;i++) words[i-1]=(int32_t)strtoll(argv[i],NULL,10);
  memcpy(original,words,(argc-1)*sizeof(int32_t));
  D1 input={words,words,0,{argc-1},{1}};
  void *arguments[]={&input};
  for(int repeat=0;repeat<2;repeat++) {
    if(!merlin_owned_results_begin(64u<<20,4096)) return 2;
    MerlinOwnedResultView views[3];
    int status=merlin_owned_result_invoke_checked(arguments,views,64u<<20);
    if(status) {fputs(merlin_owned_result_error(status),stderr);return 3;}
    void *outputs[3];
    for(int i=0;i<3;i++) {
      if(views[i].rank!=1 || views[i].element_bytes!=sizeof(int32_t)) return 4;
      outputs[i]=malloc(views[i].bytes ? views[i].bytes : 1);
      if(!outputs[i]) return 5;
      status=merlin_owned_result_copy(&views[i],outputs[i],views[i].bytes);
      if(status) return 6;
    }
    merlin_owned_results_end();
    /* Results are independent copies, still valid after all source owners die. */
    for(int i=0;i<3;i++) {
      uint64_t size=views[i].count;
      if(fwrite(&size,sizeof(size),1,stdout)!=1 ||
         fwrite(outputs[i],1,views[i].bytes,stdout)!=views[i].bytes) return 7;
      free(outputs[i]);
    }
    if(memcmp(original,words,(argc-1)*sizeof(int32_t))) return 8;
  }
  return 0;
}
'''


@pytest.fixture(scope="module")
def c_program(tmp_path_factory):
    root = tmp_path_factory.mktemp("owned-c-program")
    program = compile_owned_results(COMPACT, workdir=root / "compiler", timeout=30)
    record = save(program)
    source, executable = root / "driver.c", root / "driver"
    source.write_text(DRIVER)
    done = subprocess.run(
        [str(toolchain.clang()), "-std=c11", "-Wall", "-Wextra", "-O2",
         "-I", str(runtime_dir() / "abi"), str(source), str(program.library),
         "-Wl,-rpath," + str(program.library.parent), "-o", str(executable)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    return executable, record, hashlib.sha256(program.library.read_bytes()).hexdigest()


@pytest.mark.parametrize("words", [[], [0], [-1], [7], [0, -1, 4, 0, 9, -3],
                                  [1, 2, 3, 4], [-(1 << 31), (1 << 31) - 1]])
def test_c_only_invocation_keeps_dynamic_empty_duplicate_and_borrowed_results(c_program, words):
    executable, record, library_digest = c_program
    program = load(record)
    x = np.asarray(words, np.int32)
    expected = (x[x > 0], x[x > 0], x)
    done = subprocess.run([str(executable), *map(str, words)], capture_output=True, timeout=5)
    assert done.returncode == 0, (done.returncode, done.stderr)
    raw, cursor = done.stdout, 0
    for _repeat in range(2):
        for reference in expected:
            size = int(np.frombuffer(raw[cursor:cursor + 8], dtype=np.uint64)[0])
            cursor += 8
            actual = np.frombuffer(raw[cursor:cursor + size * 4], dtype=np.int32)
            cursor += size * 4
            np.testing.assert_array_equal(actual, reference)
    assert cursor == len(raw)
    # The independently checked original Python consumer sees the same model.
    outputs, observation = program.execute((x,))
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    assert observation["source_calls"] == 1
    assert hashlib.sha256(program.library.read_bytes()).hexdigest() == library_digest


def test_generated_aggregate_call_executes_once_and_keeps_native_scalar_words(tmp_path):
    # This is a C calling-contract fixture, not a source numerical oracle.
    source = tmp_path / "aggregate.c"
    inputs = (TensorSlot("i32", (-1,)), ScalarSlot("i64"), ScalarSlot("f64"), ScalarSlot("i1"))
    outputs = (TensorSlot("i32", (-1,)), TensorSlot("i64", ()), TensorSlot("i8", (0,)))
    source.write_text(bridge(inputs, outputs) + r'''
#include <stdlib.h>
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern void *_mlir_memref_to_llvm_alloc(size_t);
static int calls;
void _mlir_ciface_forward(Results *r,D1 *x,int64_t word,double scale,_Bool flag) {
  ++calls;
  if(word!=INT64_C(1152921504606846979) || scale!=1.5 || !flag) abort();
  int32_t *out=_mlir_memref_to_llvm_alloc(12);
  memcpy(out,x->aligned,12);
  int64_t *scalar=_mlir_memref_to_llvm_alloc(8); *scalar=word;
  r->r0=(D1){out,out,2,{3},{-1}};
  r->r1=(D0){scalar,scalar,0};
  r->r2=(D1){out,NULL,INTPTR_MIN,{0},{INTPTR_MIN}};
}
int main(void) {
  int32_t x[]={7,-3,11}, copied[3];
  D1 input={x,x,0,{3},{1}};
  int64_t word=INT64_C(1152921504606846979), actual;
  double scale=1.5; _Bool flag=1;
  void *args[]={&input,&word,&scale,&flag};
  MerlinOwnedResultView views[3];
  if(merlin_owned_result_invoke_checked(args,NULL,64)!=MERLIN_OWNED_RESULT_TYPE || calls) return 1;
  if(!merlin_owned_results_begin(8192,8)) return 2;
  if(merlin_owned_result_invoke_checked(args,views,64) || calls!=1) return 3;
  if(views[1].rank || views[2].bytes) return 4;
  if(merlin_owned_result_copy(&views[0],copied,sizeof(copied)) ||
     merlin_owned_result_copy(&views[1],&actual,sizeof(actual)) ||
     merlin_owned_result_copy(&views[2],NULL,0)) return 5;
  merlin_owned_results_end();
  if(copied[0]!=11 || copied[1]!=-3 || copied[2]!=7 || actual!=word) return 6;
  if(!merlin_owned_results_begin(8192,8)) return 7;
  if(merlin_owned_result_invoke_checked(args,views,0)!=MERLIN_OWNED_RESULT_BUDGET || calls!=2) return 8;
  unsigned char zero[sizeof(views)]={0};
  if(memcmp(views,zero,sizeof(views))) return 9;
  merlin_owned_results_end();
  return 0;
}
''')
    runtime, executable = runtime_dir() / "abi", tmp_path / "aggregate"
    done = subprocess.run(
        [str(toolchain.clang()), "-std=c11", "-Wall", "-Wextra", "-O2",
         "-I", str(runtime), str(source), str(runtime / "owned_results.c"),
         str(runtime / "owned_result_views.c"), "-o", str(executable)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    done = subprocess.run([str(executable)], capture_output=True, timeout=5)
    assert done.returncode == 0, (done.returncode, done.stderr)
