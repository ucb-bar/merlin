"""C result checks precede payload reads, including corrupt native descriptors."""

import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.toolchain import clang

DRIVER = r'''
#include "owned_result_views.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
extern void *_mlir_memref_to_llvm_alloc(size_t);
extern void _mlir_memref_to_llvm_free(void *);
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern size_t merlin_owned_results_live_count(void);

int main(int argc,char **argv) {
  struct rlimit no_core={0,0}; setrlimit(RLIMIT_CORE,&no_core);
  int mode=argc>1 ? atoi(argv[1]) : 0, expected=0;
  if(!merlin_owned_results_begin(8192,8)) return 1;
  int32_t *owner=_mlir_memref_to_llvm_alloc(16*sizeof(int32_t));
  for(int i=0;i<16;i++) owner[i]=11*i;
  intptr_t sizes[]={2,3,0}, strides[]={4,1,0}, shape[]={-1,3,-1};
  intptr_t empty_sizes[]={0}, empty_strides[]={INTPTR_MIN};
  MerlinOwnedResultDescriptor d[2]={
    {owner,owner,1,2,sizeof(int32_t),_Alignof(int32_t),sizes,strides,shape},
    {owner,NULL,INTPTR_MIN,1,sizeof(int32_t),_Alignof(int32_t),empty_sizes,empty_strides,NULL}
  };
  size_t count=2, budget=64;
  int32_t expected_values[]={11,22,33,55,66,77}, result[6]={0};
  switch(mode) {
  case 0: break;
  case 1:
    d[0].offset=5; strides[0]=-4;
    for(int i=0;i<3;i++) {expected_values[i]=55+11*i; expected_values[i+3]=11+11*i;}
    break;
  case 2:
    d[0].offset=3; d[0].rank=1; sizes[0]=4; strides[0]=0;
    for(int i=0;i<4;i++) expected_values[i]=33;
    break;
  case 3: d[0].rank=0; d[0].offset=7; expected_values[0]=77; break;
  case 4:
    d[0].rank=3; sizes[0]=INTPTR_MAX; sizes[1]=0; sizes[2]=INTPTR_MAX;
    d[0].static_sizes=NULL; d[0].aligned=NULL; d[0].offset=INTPTR_MIN;
    break;
  case 5:
    d[0].aligned=owner+1; d[0].offset=-1;
    for(int i=0;i<3;i++) {expected_values[i]=11*i; expected_values[i+3]=44+11*i;}
    break;
  case 6:
    d[0].rank=1; sizes[0]=1; strides[0]=INTPTR_MIN; expected_values[0]=11;
    break;
  case 10: sizes[0]=-1; expected=MERLIN_OWNED_RESULT_EXTENT; break;
  case 11: shape[1]=4; expected=MERLIN_OWNED_RESULT_EXTENT; break;
  case 12: d[0].allocated=(char *)owner+1; expected=MERLIN_OWNED_RESULT_OWNER; break;
  case 13: d[0].allocated=NULL; expected=MERLIN_OWNED_RESULT_OWNER; break;
  case 14: d[0].aligned=(char *)owner+1; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 15: d[0].aligned=NULL; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 16: d[0].offset=INTPTR_MAX; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 17: d[0].offset=INTPTR_MIN; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 18: strides[1]=INTPTR_MIN; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 19: strides[0]=16; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 20: strides[0]=-4; expected=MERLIN_OWNED_RESULT_FOOTPRINT; break;
  case 21:
    sizes[0]=INTPTR_MAX; sizes[1]=INTPTR_MAX; d[0].static_sizes=NULL;
    expected=MERLIN_OWNED_RESULT_BUDGET; break;
  case 22: budget=23; expected=MERLIN_OWNED_RESULT_BUDGET; break;
  case 23: d[1]=d[0]; budget=47; expected=MERLIN_OWNED_RESULT_BUDGET; break;
  case 24: _mlir_memref_to_llvm_alloc(1); expected=MERLIN_OWNED_RESULT_UNRETURNED; break;
  case 25: d[0].element_alignment=3; expected=MERLIN_OWNED_RESULT_TYPE; break;
  case 26: d[0].rank=MERLIN_OWNED_RESULT_MAX_RANK+1; expected=MERLIN_OWNED_RESULT_TYPE; break;
  case 27: sizes[0]=0; sizes[1]=-1; expected=MERLIN_OWNED_RESULT_EXTENT; break;
  case 28: d[0].strides=NULL; expected=MERLIN_OWNED_RESULT_TYPE; break;
  case 29: case 30: case 31: case 32: break;
  case 33: d[0].element_bytes=0; expected=MERLIN_OWNED_RESULT_TYPE; break;
  case 34: d[0].element_alignment=0; expected=MERLIN_OWNED_RESULT_TYPE; break;
  default: return 2;
  }
  MerlinOwnedResultView views[2]; memset(views,0xa5,sizeof(views));
  int status=merlin_owned_result_validate_all(d,count,views,budget);
  if(status!=expected) {
    fprintf(stderr,"%d expected %d: %s\n",status,expected,merlin_owned_result_error(status)); return 3;
  }
  if(status) {
    unsigned char zero[sizeof(views)]={0};
    if(memcmp(views,zero,sizeof(views))) return 4;
  } else {
    if(mode==29) {
      _mlir_memref_to_llvm_free(owner);
      status=merlin_owned_result_copy(&views[0],result,sizeof(result));
      if(status!=MERLIN_OWNED_RESULT_OWNER) return 5;
    } else if(mode==30) {
      if(merlin_owned_result_copy(&views[0],result,23)!=MERLIN_OWNED_RESULT_BUDGET) return 6;
    } else if(mode==31 || mode==32) {
      if(mode==31) views[0].bytes=0; else views[0].count=0;
      if(merlin_owned_result_copy(&views[0],result,sizeof(result))!=MERLIN_OWNED_RESULT_COPY) return 7;
    } else {
      if(merlin_owned_result_copy(&views[0],result,sizeof(result))) return 8;
      if(memcmp(result,expected_values,views[0].bytes)) return 9;
      if(merlin_owned_result_copy(&views[1],NULL,0)) return 10;
      if(views[0].bytes &&
         merlin_owned_result_copy(&views[0],owner,views[0].bytes)!=MERLIN_OWNED_RESULT_COPY) return 11;
    }
  }
  merlin_owned_results_end();
  if(merlin_owned_results_live_count()) return 12;
  if(mode<7 && memcmp(result,expected_values,mode==4 ? 0 : (mode==3 || mode==6 ? 4 : (mode==2 ? 16 : 24)))) return 13;
  return 0;
}
'''


@pytest.fixture(scope="module", params=[False, True], ids=["optimized", "undefined-traps"])
def checked_views(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("owned-result-views")
    source, output = root / "driver.c", root / "driver"
    source.write_text(DRIVER)
    runtime = runtime_dir() / "abi"
    flags = ["-fsanitize=undefined", "-fsanitize-trap=undefined"] if request.param else []
    done = subprocess.run(
        [str(clang()), "-std=c11", "-Wall", "-Wextra", "-O2", *flags,
         "-I", str(runtime), str(source), str(runtime / "owned_results.c"),
         str(runtime / "owned_result_views.c"), "-o", str(output)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    return output


@pytest.mark.parametrize("mode", [*range(7), *range(10, 35)])
def test_checked_descriptors_copy_only_valid_owned_payloads(checked_views, mode):
    done = subprocess.run([str(checked_views), str(mode)], capture_output=True, text=True, timeout=5)
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
