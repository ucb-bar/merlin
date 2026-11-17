"""Exercise the selected generic allocator, including actual abort paths."""

import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.toolchain import clang


@pytest.fixture(scope="module", params=["thread-local", "serial", "serial-external"])
def allocator(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("owned-result-allocator")
    source = root / "driver.c"
    source.write_text(r"""
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/resource.h>
extern void *_mlir_memref_to_llvm_alloc(size_t);
extern void *_mlir_memref_to_llvm_aligned_alloc(size_t,size_t);
extern void _mlir_memref_to_llvm_free(void *);
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern size_t merlin_owned_results_capacity(void *);
extern size_t merlin_owned_results_live_count(void);
extern size_t merlin_owned_results_peak_bytes(void);
extern size_t merlin_owned_results_total_count(void);
#ifdef MERLIN_OWNED_RESULTS_EXTERNAL_FAILURE
_Noreturn void merlin_owned_results_failure(const char *reason) {
  fputs("explicit failure hook: ",stderr); fputs(reason,stderr); fputc('\n',stderr); abort();
}
#endif
int main(int argc,char **argv) {
  struct rlimit no_core={0,0}; setrlimit(RLIMIT_CORE,&no_core);
  int mode=argc>1 ? atoi(argv[1]) : 0;
  if(mode==1) { _mlir_memref_to_llvm_alloc(1); return 1; }
  if(!merlin_owned_results_begin(8192,4)) return 2;
  if(merlin_owned_results_begin(8192,4)) return 3;
  void *a=_mlir_memref_to_llvm_alloc(0);
  void *b=_mlir_memref_to_llvm_aligned_alloc(256,256);
  if(!a || !b || a==b || (uintptr_t)b%256) return 4;
  if(merlin_owned_results_capacity(a)!=1 || merlin_owned_results_capacity(b)!=256) return 5;
  if(merlin_owned_results_live_count()!=2 || merlin_owned_results_total_count()!=2) return 6;
  if(merlin_owned_results_capacity((char *)b+1)) return 7;
  if(mode==2) { _mlir_memref_to_llvm_alloc(8192); return 8; }
  if(mode==3) { _mlir_memref_to_llvm_aligned_alloc(0,1); return 8; }
  if(mode==4) { _mlir_memref_to_llvm_alloc(SIZE_MAX); return 8; }
  _mlir_memref_to_llvm_free(a);
  if(merlin_owned_results_capacity(a) || merlin_owned_results_live_count()!=1) return 9;
  if(mode==5) { _mlir_memref_to_llvm_free(a); return 8; }
  if(mode==6) { _mlir_memref_to_llvm_free((char *)b+1); return 8; }
  if(mode==7) {
    for(int i=0;i<4;i++) { void *p=_mlir_memref_to_llvm_alloc(1); _mlir_memref_to_llvm_free(p); }
    return 8;
  }
  if(merlin_owned_results_peak_bytes()>8192) return 10;
  merlin_owned_results_end();
  if(merlin_owned_results_live_count() || merlin_owned_results_capacity(b)) return 11;
  if(!merlin_owned_results_begin(8192,4)) return 12;
  if(merlin_owned_results_total_count()) return 13;
  _mlir_memref_to_llvm_free(0);
  merlin_owned_results_end();
  return 0;
}
""")
    executable = root / "allocator"
    runtime = runtime_dir() / "abi/owned_results.c"
    flags = []
    if request.param != "thread-local":
        flags.append("-DMERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1")
    if request.param == "serial-external":
        flags.append("-DMERLIN_OWNED_RESULTS_EXTERNAL_FAILURE=1")
    done = subprocess.run(
        [
            str(clang()),
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-O2",
            *flags,
            str(source),
            str(runtime),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    return executable


def test_alignment_empty_owners_accounting_and_repeated_invocations(allocator):
    done = subprocess.run([str(allocator)], capture_output=True, text=True, timeout=5)
    assert done.returncode == 0, (done.stdout, done.stderr)


@pytest.mark.parametrize(
    "mode,message",
    (
        (1, "outside its invocation"),
        (2, "allocation budget"),
        (3, "alignment"),
        (4, "size overflow"),
        (5, "already released"),
        (6, "unknown"),
        (7, "allocation budget"),
    ),
)
def test_allocator_refuses_invalid_ownership_and_resource_requests(allocator, mode, message):
    done = subprocess.run([str(allocator), str(mode)], capture_output=True, text=True, timeout=5)
    assert done.returncode < 0
    assert message in done.stderr
