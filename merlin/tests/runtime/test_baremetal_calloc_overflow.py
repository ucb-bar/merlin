"""Overflow must fail before a wrapped calloc request can write the arena."""

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_calloc_overflow_and_zero_size(tmp_path, mode):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler required")
    runtime = runtime_dir() / "baremetal/spike"
    shutil.copy2(runtime / "merlin_malloc.c", tmp_path / "allocator.c")
    shutil.copy2(runtime / "htif.h", tmp_path / "htif.h")
    source = tmp_path / "check.c"
    source.write_text(
        r"""
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static unsigned char storage[2048] __attribute__((aligned(64)));
#define MERLIN_ARENA_BASE_ADDR ((uintptr_t)storage)
#define MERLIN_ARENA_SIZE_BYTES sizeof(storage)
#define malloc checked_malloc
#define calloc checked_calloc
#define aligned_alloc checked_aligned_alloc
#define free checked_free
#include "allocator.c"
#undef malloc
#undef calloc
#undef aligned_alloc
#undef free
static jmp_buf failed;
static char diagnostic[2048];
static size_t used;
void htif_putc(char c) { assert(used+1<sizeof(diagnostic)); diagnostic[used++]=c; }
void htif_puts(const char *s) { while(*s)htif_putc(*s++); }
void htif_putd(long v) { (void)v; }
void htif_puthex(unsigned long long v) { (void)v; }
__attribute__((noreturn)) void htif_exit(int code) { assert(code==0x900); longjmp(failed,1); }
int main(int argc,char **argv) {
 assert(argc==2);int mode=atoi(argv[1]);memset(storage,0x3c,sizeof(storage));
 merlin_arena_reset();unsigned char before[2048];memcpy(before,storage,sizeof(storage));
 if(mode==0){
  unsigned char *p=checked_calloc(2,7);
  assert((uintptr_t)p >= (uintptr_t)storage && (uintptr_t)p+14 <= (uintptr_t)storage+sizeof(storage));
  assert(((uintptr_t)p & 63)==0);
  for(size_t i=0;i<14;i++)assert(p[i]==0);
  unsigned char *z0=checked_calloc(0,SIZE_MAX), *z1=checked_calloc(SIZE_MAX,0);
  assert(z0 && z1 && z0!=z1 && z0!=p && z1!=p);
  for(size_t i=0;i<14;i++)assert(p[i]==0);
  return 0;
 }
 volatile size_t count=mode==1?SIZE_MAX/2+1:SIZE_MAX;
 volatile size_t width=mode==3?SIZE_MAX:2;
 if(!setjmp(failed)){checked_calloc(count,width);return 7;}
 assert(strstr(diagnostic,"calloc size multiplication overflow"));
 assert(!memcmp(before,storage,sizeof(storage)));
 assert(checked_malloc(1)==storage+64);return 0;
}
"""
    )
    exe = tmp_path / "check"
    subprocess.run(
        [
            cc,
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(source),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(exe), str(mode)], check=True, capture_output=True)
