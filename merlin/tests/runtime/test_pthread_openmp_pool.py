"""Native contract tests for Merlin's hosted persistent OpenMP worker pool."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir


CC = shutil.which("cc") or shutil.which("gcc")


_REPEATED_REGIONS = r"""
#include "libomp_pthread.h"
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>

typedef struct { _Atomic int hits[37]; } context_t;
typedef struct ident ident_t;
extern void __kmpc_push_num_threads(ident_t *, int32_t, int32_t);
extern void __kmpc_fork_call(ident_t *, int32_t, void *, ...);
extern void __kmpc_for_static_init_8(ident_t *, int32_t, int32_t, int32_t *,
                                     int64_t *, int64_t *, int64_t *, int64_t, int64_t);
extern void __kmpc_for_static_fini(ident_t *, int32_t);

static _Atomic int creates;
static _Atomic int joins;
extern int __real_pthread_create(pthread_t *, const pthread_attr_t *,
                                 void *(*)(void *), void *);
extern int __real_pthread_join(pthread_t, void **);
int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a,
                          void *(*fn)(void *), void *arg) {
  atomic_fetch_add(&creates, 1);
  return __real_pthread_create(t, a, fn, arg);
}
int __wrap_pthread_join(pthread_t t, void **out) {
  atomic_fetch_add(&joins, 1);
  return __real_pthread_join(t, out);
}

static void region(int32_t *gtid, int32_t *bound, void *opaque) {
  (void)bound;
  context_t *ctx = (context_t *)opaque;
  int32_t last = 0;
  int64_t lo = 0, hi = 36, stride = 0;
  __kmpc_for_static_init_8(0, *gtid, 34, &last, &lo, &hi, &stride, 1, 1);
  for (int64_t i = lo; i <= hi; ++i) atomic_fetch_add(&ctx->hits[i], 1);
  __kmpc_for_static_fini(0, *gtid);
}

int main(void) {
  context_t ctx = {0};
  if (merlin_omp_init(4) != 4) return 2;
  if (atomic_load(&creates) != 3 || atomic_load(&joins) != 0) return 3;
  for (int repeat = 0; repeat < 40; ++repeat) {
    __kmpc_push_num_threads(0, 0, 4);
    __kmpc_fork_call(0, 1, (void *)region, &ctx);
  }
  if (atomic_load(&creates) != 3 || atomic_load(&joins) != 0) return 4;
  for (int i = 0; i < 37; ++i) if (atomic_load(&ctx.hits[i]) != 40) return 5;

  /* A one-thread region remains serial without changing or destroying the pool. */
  __kmpc_push_num_threads(0, 0, 1);
  __kmpc_fork_call(0, 1, (void *)region, &ctx);
  if (atomic_load(&creates) != 3 || atomic_load(&joins) != 0) return 6;
  for (int i = 0; i < 37; ++i) if (atomic_load(&ctx.hits[i]) != 41) return 7;
  return 0;
}
"""


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_repeated_exact_cover_regions_reuse_one_process_lifetime_pool(tmp_path):
    harness = tmp_path / "repeated.c"
    harness.write_text(_REPEATED_REGIONS, encoding="utf-8")
    binary = tmp_path / "repeated"
    proc = subprocess.run([
        CC, "-std=c11", "-O2", "-pthread", f"-I{runtime_dir() / 'c'}",
        str(harness), str(runtime_dir() / "c/libomp_pthread.c"),
        "-Wl,--wrap=pthread_create", "-Wl,--wrap=pthread_join", "-o", str(binary),
    ], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
