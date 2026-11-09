"""Explicit staged roles and the actual concurrent C publication protocol."""

import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.staged_dispatch import (
    StagedBindings,
    StagedExecutionContract,
    StagedRoles,
    render_zephyr_staged,
)


def roles():
    return StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))


def contract():
    return StagedExecutionContract(
        "test shared coherent storage",
        "test synchronous callback",
        "test pinned workers",
        "test interrupt context preserved",
        "test one owner retains all buffers until ACK",
    )


def bindings():
    return StagedBindings(
        "context", "initialize", "prepare", "compute", "finish", "hart", "ticks", "cycles", "retired", "completed"
    )


def test_role_map_is_explicit_and_profile_excludes_controller():
    r = replace(roles(), cpu_hart_ids=(2, 0, 1))
    assert r.to_dict()["controller_cpu"] == 0
    assert r.to_dict()["worker_cpus"] == [1, 2]
    r.validate_compute_capabilities((0, 1))
    with pytest.raises(ValueError, match="capability"):
        r.validate_compute_capabilities((0,))


@pytest.mark.parametrize(
    "change",
    [
        {"compute_harts": (0, 2), "profile_harts": (0, 2)},
        {"profile_harts": (0, 1, 2)},
        {"compute_harts": (0, 0)},
        {"pack_hart": 0},
        {"model_init_hart": False},
        {"finish_hart": 8},
        {"cpu_hart_ids": (0, 0, 2)},
        {"compute_harts": ()},
    ],
)
def test_unsupported_roles_never_render(change):
    with pytest.raises(ValueError):
        replace(roles(), **change)


def test_target_obligations_and_c_identifiers_are_explicit():
    with pytest.raises(ValueError, match="evidence"):
        replace(contract(), synchronous_compute_completion="")
    with pytest.raises(ValueError, match="identifiers"):
        replace(bindings(), compute="compute(); injected()")


def test_generated_harness_pins_before_start_and_has_no_implicit_counter_isa():
    src, metadata = render_zephyr_staged(
        roles(), contract(), bindings(), compute_capable_harts=(0, 1), sequence=1, max_wait_polls=10000
    )
    assert "const int cpus[2] = {0, 1}" in src
    assert "k_thread_cpu_pin(thread, 2)" in src
    assert src.index("stage_status = k_thread_cpu_pin(thread, cpus[i])") < src.index(
        "k_thread_start(&stage_workers[i])"
    )
    assert "asm" not in src and "__kmpc" not in src
    assert metadata["controller_counter_reads"] is False
    assert metadata["full_model_segmentation_proved"] is False
    assert metadata["contract"]["status"] == "caller_declared_not_independently_proved"


def compiler():
    value = shutil.which("cc")
    if not value:
        pytest.skip("C compiler unavailable")
    return value


_CONCURRENT = r"""
#include "merlin_staged.h"
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
static _Thread_local unsigned self;
static _Thread_local uint64_t local_c, local_i;
static merlin_staged_team team;
static atomic_uint phase, global_reads, cycle_reads, retire_reads, computes;
static atomic_uint_fast64_t clock_ticks;
static int failure;
static int window_mode;
static unsigned window_reads[2];
static unsigned hart(void *x) { (void)x; return self; }
static uint64_t ticks(void *x) {
  (void)x; assert(self == 0);
  unsigned call = atomic_fetch_add(&global_reads, 1);
  assert((call == 0 && atomic_load(&phase) == 0) ||
         (call == 1 && atomic_load(&phase) == 4));
  return atomic_load(&clock_ticks);
}
static uint64_t cycles(void *x) {
  (void)x; assert(self < 2); atomic_fetch_add(&cycle_reads, 1); return local_c;
}
static uint64_t retired(void *x) {
  (void)x; assert(self < 2); atomic_fetch_add(&retire_reads, 1); return local_i;
}
static void relax(void *x) { (void)x; sched_yield(); }
static int shared_ticks(void *x, uint64_t *value) {
  (void)x; assert(self < 2);
  unsigned call=window_reads[self]++; assert(call < 4);
  assert(atomic_load(&phase)==(call==0?0:call==3?4:2));
  const uint64_t samples[2][4]={{100,110,140,180},{105,115,145,185}};
  if(window_mode==2 && self==1 && call==1) return 42;
  if(window_mode==6 && self==0 && call==0) return -9;
  if(window_mode==7) { *value=atomic_fetch_add(&clock_ticks,1); return 0; }
  *value=samples[self][call];
  if(window_mode==3 && self==1 && call==2) *value=90;
  if(window_mode==4) *value=105; /* a genuinely shared unresolved clock tick */
  if(window_mode==5 && self==1 && call==0) *value=UINT64_MAX;
  return 0;
}
/* Independent source operands and known complete result, plus exterior guards. */
static const int8_t A[6] = {1, -2, 127, -128, 4, -5};
static const int8_t B[9] = {1,2,-3, 4,-5,6, -7,8,9};
static const int32_t EXPECTED[6] = {-896,1028,1128, -77,-316,363};
static struct { uint32_t before; int8_t data[6]; uint32_t after; } At;
static struct { uint32_t before; int8_t data[9]; uint32_t after; } Bp;
static struct { uint32_t before; int32_t data[6]; uint32_t after; } Cp, C;
static int initialize(void *x) {
  (void)x; assert(self == 2 && atomic_load(&team.armed) == 3);
  assert(atomic_load(&global_reads) == (window_mode?0:1));
  if(window_mode) assert(window_reads[0]==1 && window_reads[1]==1);
  At.before=At.after=Bp.before=Bp.after=Cp.before=Cp.after=C.before=C.after=0x12345678;
  atomic_store(&phase, 1); atomic_fetch_add(&clock_ticks, 5); return 0;
}
static int prepare(void *x) {
  (void)x; assert(self == 2 && atomic_load(&phase) == 1);
  if (failure == 1) return 77;
  for(unsigned k=0;k<3;k++) for(unsigned m=0;m<2;m++) At.data[k*2+m]=A[m*3+k];
  memcpy(Bp.data, B, sizeof B); memset(Cp.data, 0, sizeof Cp.data);
  atomic_fetch_add(&clock_ticks,17); atomic_store(&phase,2); return 0;
}
static int compute(void *x,unsigned slot) {
  (void)x; assert(self == slot && self < 2 && atomic_load(&phase) == 2);
  atomic_fetch_add(&computes,1);
  if (failure == 2 && slot == 1) return 88;
  for(unsigned j=slot;j<3;j+=2) for(unsigned m=0;m<2;m++) {
    uint32_t sum=0;
    for(unsigned k=0;k<3;k++) sum+=(uint32_t)((int32_t)At.data[k*2+m]*(int32_t)Bp.data[k*3+j]);
    Cp.data[m*3+j]=(int32_t)sum;
  }
  local_c+=100+slot; local_i+=50+slot; atomic_fetch_add(&clock_ticks,100+slot); return 0;
}
static int finish(void *x) {
  (void)x; assert(self==2 && atomic_load(&team.done)==3);
  if (failure == 3) return 99;
  memcpy(C.data,Cp.data,sizeof C.data);
  assert(memcmp(C.data,EXPECTED,sizeof EXPECTED)==0);
  assert(At.before==0x12345678 && At.after==0x12345678 && Bp.before==0x12345678 && Bp.after==0x12345678);
  assert(Cp.before==0x12345678 && Cp.after==0x12345678 && C.before==0x12345678 && C.after==0x12345678);
  atomic_fetch_add(&clock_ticks,11); atomic_store(&phase,4); return 0;
}
static void *worker(void *v) {
  self=(unsigned)(uintptr_t)v; local_c=1000;local_i=2000;
  int rc=merlin_staged_worker(&team,self);
  assert(rc==0 || rc==MERLIN_STAGED_ABORTED); return NULL;
}
static void publish(void) {
  const merlin_staged_report *r=&team.report;
  printf("{\"abi_version\":2,\"status\":%d,\"quiescent\":%u,\"sequence\":%" PRIu64
         ",\"global_clock_hart\":%u,\"global_begin\":%" PRIu64 ",\"global_end\":%" PRIu64
         ",\"global_interval_valid\":%u,\"worker_clock_windows_enabled\":%u,"
         "\"worker_clock_windows_valid\":%u,\"workers\":[",r->status,r->quiescent,r->sequence,
         r->global_clock_hart,r->global_begin,r->global_end,r->global_interval_valid,
         r->worker_clock_windows_enabled,r->worker_clock_windows_valid);
  for(unsigned i=0;i<2;i++) {
    const merlin_staged_worker_report *w=&r->workers[i];
    printf("%s{\"hart\":%u,\"status\":%d,\"wait_status\":%d,\"begin_cycle\":%" PRIu64
           ",\"end_cycle\":%" PRIu64 ",\"begin_instret\":%" PRIu64 ",\"end_instret\":%" PRIu64
           ",\"arm_ready_tick\":%" PRIu64 ",\"compute_begin_tick\":%" PRIu64
           ",\"compute_end_tick\":%" PRIu64 ",\"final_ack_tick\":%" PRIu64
           ",\"clock_valid_mask\":%u,\"clock_status\":%u,\"clock_reader_status\":%d}",
           i?",":"",w->hart,w->status,w->wait_status,w->begin_cycle,w->end_cycle,
           w->begin_instret,w->end_instret,w->arm_ready_tick,w->compute_begin_tick,
           w->compute_end_tick,w->final_ack_tick,w->clock_valid_mask,w->clock_status,w->clock_reader_status);
  }
  puts("]}");
}
int main(int argc,char **argv) {
  failure=argc>1?atoi(argv[1]):0; self=2;
  window_mode=argc>2?atoi(argv[2]):0;
  merlin_staged_config cfg={.controller_hart=2,.worker_count=2,.worker_harts={0,1},
                           .sequence=17,.max_wait_polls=10000000,.enable_worker_windows=window_mode!=0};
  merlin_staged_hooks h={.current_hart=hart,.global_ticks=ticks,.worker_cycles=cycles,
                          .worker_instret=retired,.relax=relax,.shared_window_ticks=shared_ticks};
  merlin_staged_callbacks cb={initialize,prepare,compute,finish};
  self=0; assert(merlin_staged_team_init(&team,&cfg,&h,&cb,NULL)==MERLIN_STAGED_ROLE); self=2;
  merlin_staged_config bad=cfg;bad.worker_harts[1]=0;
  assert(merlin_staged_team_init(&team,&bad,&h,&cb,NULL)==MERLIN_STAGED_INVALID);
  bad=cfg;bad.enable_worker_windows=2;
  assert(merlin_staged_team_init(&team,&bad,&h,&cb,NULL)==MERLIN_STAGED_INVALID);
  if(window_mode) {
    h.shared_window_ticks=NULL;
    assert(merlin_staged_team_init(&team,&cfg,&h,&cb,NULL)==MERLIN_STAGED_INVALID);
    h.shared_window_ticks=shared_ticks;
  }
  assert(merlin_staged_team_init(&team,&cfg,&h,&cb,NULL)==0);
  assert(merlin_staged_worker(&team,0)==MERLIN_STAGED_ROLE);
  atomic_store(&clock_ticks,1000);
  if (failure==4) {
    team.config.max_wait_polls=10;
    assert(merlin_staged_controller(&team)==MERLIN_STAGED_UNQUIESCED);
    assert(team.report.status==MERLIN_STAGED_TIMEOUT && !team.report.quiescent);
    assert(!atomic_load(&global_reads) && !atomic_load(&computes) && !atomic_load(&phase));
    /* Late workers can acknowledge ABORT; recorded nonquiescence stays honest. */
    self=0; assert(merlin_staged_worker(&team,0)==MERLIN_STAGED_ABORTED);
    self=1; assert(merlin_staged_worker(&team,1)==MERLIN_STAGED_ABORTED);
    assert(atomic_load(&team.exited)==3 && !team.report.quiescent); return 0;
  }
  pthread_t p[2]; assert(!pthread_create(&p[0],NULL,worker,(void*)0));
  assert(!pthread_create(&p[1],NULL,worker,(void*)1));
  int rc=merlin_staged_controller(&team);
  assert(!pthread_join(p[0],NULL)); assert(!pthread_join(p[1],NULL));
  assert(team.report.quiescent==1 && team.report.sequence==17);
  if(failure) {
    assert(rc==(failure==1?77:failure==2?88:99));
    assert(team.report.global_interval_valid==0 && atomic_load(&global_reads)==(window_mode?0:1));
    assert(atomic_load(&computes)==(failure==1?0:2));
  } else {
    assert(rc==0);
    if(window_mode) {
      assert(atomic_load(&global_reads)==0 && window_reads[0]==4 && window_reads[1]==4);
      assert(team.report.global_interval_valid==(window_mode!=6));
      assert(team.report.worker_clock_windows_valid==(window_mode==1 || window_mode==4 || window_mode==7));
      if(window_mode==7) assert(team.report.global_begin>=1000 &&
                               team.report.global_end>team.report.global_begin);
      else if(window_mode==4) assert(team.report.global_begin==105 && team.report.global_end==105);
      else assert(team.report.global_begin==(window_mode==6?0:100) && team.report.global_end==180);
    } else {
      assert(team.report.global_interval_valid==1);
      assert(team.report.global_begin==1000 && team.report.global_end==1234);
      assert(atomic_load(&global_reads)==2);
    }
    assert(atomic_load(&cycle_reads)==4 && atomic_load(&retire_reads)==4);
    for(unsigned i=0;i<2;i++) {
      assert(team.report.workers[i].hart==i && team.report.workers[i].status==0);
      assert(team.report.workers[i].end_cycle-team.report.workers[i].begin_cycle==100+i);
      assert(team.report.workers[i].end_instret-team.report.workers[i].begin_instret==50+i);
    }
  }
  assert(merlin_staged_controller(&team)==MERLIN_STAGED_USED);
  publish();
  return 0;
}
"""


@pytest.fixture
def staged_c_executable(tmp_path):
    source = tmp_path / "concurrent.c"
    source.write_text(_CONCURRENT)
    executable = tmp_path / "concurrent"
    runtime = runtime_dir() / "c"
    subprocess.run(
        [
            compiler(),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I" + str(runtime),
            str(source),
            str(runtime / "merlin_staged.c"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return executable


def test_real_concurrent_runtime_full_outputs_guards_ordering_and_failures(staged_c_executable):
    for mode in range(5):
        executable = staged_c_executable
        subprocess.run([str(executable), str(mode)], check=True, capture_output=True, timeout=20)


@pytest.mark.parametrize("window_mode", range(1, 8))
def test_actual_worker_windows_and_clock_failure_status(staged_c_executable, window_mode):
    from merlin.runtime.staged_windows import analyze_worker_windows

    result = subprocess.run(
        [str(staged_c_executable), "0", str(window_mode)], check=True, capture_output=True, text=True, timeout=20
    )
    record = json.loads(result.stdout)
    report, trace = analyze_worker_windows(
        record,
        roles(),
        expected_sequence=17,
        clock_domain="controlled test shared ticks",
        counter_semantics="controlled test counters",
    )
    assert report["pass"] == (window_mode in (1, 4, 7))
    assert {event["tid"] for event in trace["traceEvents"]} <= {0, 1}
    assert report["clock_hz"] is None and report["physical_unit_busy"] is None
    if window_mode == 1:
        assert report["inclusive_stage_ticks"] == 80
        assert report["workers"][0]["pre_compute_wait_ticks"] == 10
        assert report["workers"][1]["post_compute_wait_ticks"] == 40
    else:
        invalid = [worker for worker in report["workers"] if not worker["window_valid"]]
        for worker in invalid:
            assert worker["compute_window_ticks"] is None


def test_actual_prepare_failure_preserves_incomplete_windows(staged_c_executable):
    from merlin.runtime.staged_windows import analyze_worker_windows

    result = subprocess.run(
        [str(staged_c_executable), "1", "1"], check=True, capture_output=True, text=True, timeout=20
    )
    report, trace = analyze_worker_windows(
        json.loads(result.stdout),
        roles(),
        expected_sequence=17,
        clock_domain="controlled test shared ticks",
        counter_semantics="controlled test counters",
    )
    assert report["pass"] is False and trace["traceEvents"] == []
    assert all(worker["clock_status"] == "incomplete" for worker in report["workers"])
    assert all(worker["compute_cycles"] is None for worker in report["workers"])


def test_actual_generated_zephyr_entry_is_valid_c(tmp_path):
    src, _ = render_zephyr_staged(
        roles(), contract(), bindings(), compute_capable_harts=(0, 1), sequence=1, max_wait_polls=10000
    )
    header = tmp_path / "zephyr" / "kernel.h"
    header.parent.mkdir()
    header.write_text("""#include <stddef.h>
struct k_thread { int dummy; }; struct k_sem { int dummy; }; typedef struct k_thread *k_tid_t;
#define K_THREAD_STACK_ARRAY_DEFINE(n,c,s) static char n[c][s]
#define K_THREAD_STACK_DEFINE(n,s) static char n[s]
#define K_THREAD_STACK_SIZEOF(n) sizeof(n)
#define K_FOREVER -1
#define K_PRIO_COOP(n) (n)
void k_sem_init(struct k_sem*,unsigned,unsigned); void k_sem_give(struct k_sem*);
int k_sem_take(struct k_sem*,int); int k_thread_cpu_pin(k_tid_t,int); void k_thread_start(k_tid_t);
k_tid_t k_thread_create(struct k_thread*,char*,size_t,void(*)(void*,void*,void*),void*,void*,void*,int,int,int);
""")
    source = tmp_path / "entry.c"
    source.write_text(src)
    subprocess.run(
        [
            compiler(),
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I" + str(tmp_path),
            "-I" + str(runtime_dir() / "c"),
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
