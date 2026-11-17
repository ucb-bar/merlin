"""The generic C driver cleans every failure without ending another owner."""

import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.toolchain import clang

DRIVER = r'''
#include "owned_result_driver.h"
#include <stdlib.h>
#include <string.h>
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern size_t merlin_owned_results_live_count(void);
extern void *_mlir_memref_to_llvm_alloc(size_t);
void *__real_malloc(size_t);
static int fail_next,mode,calls,observations,reject;
void *__wrap_malloc(size_t bytes) {
  if(fail_next) {fail_next=0;return NULL;} return __real_malloc(bytes);
}
static int invoke(void **arguments,MerlinOwnedResultView *views,size_t budget) {
  (void)arguments;++calls;
  int32_t *data=_mlir_memref_to_llvm_alloc(8);data[0]=11;data[1]=22;
  intptr_t shape[]={2},strides[]={1};
  MerlinOwnedResultDescriptor d={data,data,mode==2 ? INTPTR_MAX : 0,1,4,4,shape,strides,shape};
  int status=merlin_owned_result_validate_all(&d,1,views,budget);
  if(mode==3) fail_next=1;
  return status;
}
static int observe(void *context,int kind,size_t index,const char *dtype,size_t rank,
                    const intptr_t *shape,const void *payload,size_t bytes) {
  (void)context;++observations;
  if(merlin_owned_results_live_count() || kind!=MERLIN_OWNED_OUTPUT || index ||
     strcmp(dtype,"i32") || rank!=1 || shape[0]!=2 || bytes!=8 ||
     ((const int32_t *)payload)[0]!=11 || ((const int32_t *)payload)[1]!=22) abort();
  return reject;
}
int main(void) {
  const char *types[]={"i32"};
  MerlinOwnedPlan plan={0,1,NULL,types,invoke,8192,16,8192};
  reject=1;
  if(merlin_owned_driver_execute(&plan,NULL,observe,NULL)!=MERLIN_OWNED_DRIVER_OBSERVER ||
     calls!=1 || observations!=1 || merlin_owned_results_live_count()) return 1;
  reject=0;
  if(merlin_owned_driver_execute(&plan,NULL,observe,NULL) || calls!=2 || observations!=2) return 2;
  mode=2;
  if(merlin_owned_driver_execute(&plan,NULL,observe,NULL)!=MERLIN_OWNED_RESULT_FOOTPRINT ||
     calls!=3 || observations!=2 || merlin_owned_results_live_count()) return 3;
  mode=3;
  if(merlin_owned_driver_execute(&plan,NULL,observe,NULL)!=MERLIN_OWNED_DRIVER_ALLOCATION ||
     calls!=4 || observations!=2 || merlin_owned_results_live_count()) return 4;
  mode=0;
  if(!merlin_owned_results_begin(8192,16)) return 5;
  _mlir_memref_to_llvm_alloc(1);
  if(merlin_owned_driver_execute(&plan,NULL,observe,NULL)!=MERLIN_OWNED_DRIVER_BUSY ||
     calls!=4 || observations!=2 || merlin_owned_results_live_count()!=1) return 6;
  merlin_owned_results_end();
  if(merlin_owned_driver_execute(&plan,NULL,observe,NULL) || calls!=5 || observations!=3) return 7;
  return 0;
}
'''


@pytest.mark.parametrize("serial", [False, True])
def test_rejected_observer_copy_failure_and_busy_context_preserve_cleanup(tmp_path, serial):
    source, executable = tmp_path / "control.c", tmp_path / "control"
    source.write_text(DRIVER)
    abi, runtime = runtime_dir() / "abi", runtime_dir() / "c"
    flags = ["-DMERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1"] if serial else []
    done = subprocess.run(
        [str(clang()), "-std=c11", "-Wall", "-Wextra", "-O2", *flags,
         "-I", str(abi), "-I", str(runtime), str(source), str(runtime / "owned_result_driver.c"),
         str(abi / "owned_result_views.c"), str(abi / "owned_results.c"),
         "-Wl,--wrap=malloc", "-o", str(executable)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    done = subprocess.run([str(executable)], capture_output=True, timeout=5)
    assert done.returncode == 0, (done.returncode, done.stderr)
