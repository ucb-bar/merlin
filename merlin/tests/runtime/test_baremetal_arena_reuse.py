"""Dead activation chunks must serve a later large tensor within the declared arena."""

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir


def test_coalesced_space_serves_larger_tensor_without_touching_live_data(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler required")
    runtime = runtime_dir() / "baremetal/spike"
    shutil.copy2(runtime / "merlin_malloc.c", tmp_path / "allocator.c")
    shutil.copy2(runtime / "htif.h", tmp_path / "htif.h")
    source = tmp_path / "reuse.c"
    source.write_text(
        r"""
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
static unsigned char storage[2048] __attribute__((aligned(128)));
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
void htif_putc(char c) { (void)c; }
void htif_puts(const char *s) { (void)s; }
void htif_putd(long v) { (void)v; }
void htif_puthex(unsigned long long v) { (void)v; }
__attribute__((noreturn)) void htif_exit(int code) { (void)code; abort(); }
int main(void) {
  merlin_arena_reset();
  unsigned char *a=checked_malloc(400), *b=checked_malloc(400), *live=checked_malloc(400);
  assert(a && b && live);
  memset(live,0x5a,400);
  checked_free(a);checked_free(b);
  /* No single old chunk or unallocated tail can fit 800 bytes. */
  unsigned char *large=checked_malloc(800);
  assert(large==a);
  memset(large,0xa5,800);
  for(size_t i=0;i<400;i++)assert(live[i]==0x5a);
  checked_free(large);checked_free(live);
  unsigned char *again=checked_malloc(1600);
  assert(again && (uintptr_t)again+1600<=(uintptr_t)storage+sizeof(storage));
  checked_free(again);
  unsigned char *aligned=checked_aligned_alloc(128,128);
  assert(aligned && ((uintptr_t)aligned&127)==0);
  checked_free(aligned);
  return 0;
}
""",
        encoding="utf-8",
    )
    exe = tmp_path / "reuse"
    subprocess.run(
        [cc, "-O2", "-ffreestanding", "-fno-builtin", "-fsanitize=undefined", "-fno-sanitize-recover=all",
         str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True)
