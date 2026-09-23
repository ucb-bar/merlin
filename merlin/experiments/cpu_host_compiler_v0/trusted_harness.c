/* Trusted executable authority for CPU-host compiler capsules.
 *
 * This file is compiled by the grader after an untrusted submission has emitted kernel.c.  Shape,
 * type, layout, and semantic operation are supplied as compiler definitions generated from the
 * sealed capsule.  Inputs are generated from a seed selected after code generation.  Every buffer
 * has guard zones, every input is snapshotted, and the native L1 build enables ASan and UBSan.
 */
#ifndef MERLIN_FREESTANDING
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif
#include <math.h>
#include <stdint.h>
#ifndef MERLIN_FREESTANDING
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef MERLIN_K1_LINUX
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <time.h>
#endif
#endif
#include <string.h>
#ifdef MERLIN_FREESTANDING
#include "htif.h"
int memcmp(const void *left, const void *right, size_t count) {
  const unsigned char *a=left, *b=right;
  while (count--) { if (*a != *b) return *a < *b ? -1 : 1; ++a; ++b; }
  return 0;
}
#endif

typedef struct {
  uint32_t version;
  uint32_t family;
  uint32_t operation;
  uint32_t dtype;
  uint32_t layout;
  uint32_t harts;
  uint32_t vlen_bits;
  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  uint64_t state0;
} merlin_capsule_params_t;

extern int merlin_capsule_run(const merlin_capsule_params_t *, const void *, const void *,
                              const void *, void *);

#define KIND_NONE 0
#define KIND_F32 1
#define KIND_I8 2
#define KIND_I32 3
#define FAMILY_CONTRACTION 1
#define FAMILY_ELEMENTWISE 2
#define FAMILY_REDUCTION 3
#define FAMILY_MOVEMENT 4
#define FAMILY_FUSION 5
#define FAMILY_RUNTIME 6
enum { OP_MATMUL = 1, OP_ADD = 2, OP_MULTIPLY = 3, OP_RELU = 4, OP_SILU = 5,
       OP_GELU = 6, OP_CLAMP = 7, OP_REQUANT = 8, OP_SUM = 9, OP_MAX = 10,
       OP_SOFTMAX_COMPONENTS = 11, OP_LAYERNORM_COMPONENTS = 12, OP_COPY = 13,
       OP_TRANSPOSE2D = 14, OP_PACK_RHS = 15, OP_UNPACK = 16,
       OP_STRIDED_SLICE = 17, OP_CONCATENATE = 18, OP_MATMUL_BIAS = 19,
       OP_MATMUL_BIAS_RELU = 20, OP_MATMUL_REQUANT = 21, OP_RESIDUAL_NORM = 22,
       OP_RUNTIME_AFFINE = 23 };
#define LAYOUT_CONTIGUOUS 1
#define LAYOUT_ROW_ROW 2
#define LAYOUT_ROW_PACKED_RHS 3
#define LAYOUT_TRANSPOSED_RHS 4
#define LAYOUT_OPERATION_DEFINED 5

#ifndef MERLIN_FAMILY
#error "MERLIN_FAMILY is required"
#endif

#define GUARD_BYTES 64u
#define GUARD_VALUE 0xa5u

#ifndef MERLIN_RECEIPT_NONCE
#error "MERLIN_RECEIPT_NONCE is required"
#endif

typedef struct {
  unsigned char *raw;
  unsigned char *data;
  unsigned char *snapshot;
  size_t bytes;
} guarded_buffer_t;

#ifdef MERLIN_FREESTANDING
#define STORAGE_BYTES(count, kind) ((count) * ((kind) == KIND_I8 ? 1 : 4) + 2 * GUARD_BYTES)
static unsigned char storage0[STORAGE_BYTES(MERLIN_INPUT0_COUNT, MERLIN_INPUT0_KIND)];
static unsigned char storage1[STORAGE_BYTES(MERLIN_INPUT1_COUNT, MERLIN_INPUT1_KIND)];
static unsigned char storage2[STORAGE_BYTES(MERLIN_INPUT2_COUNT, MERLIN_INPUT2_KIND)];
static unsigned char storage3[STORAGE_BYTES(MERLIN_OUTPUT_COUNT, MERLIN_OUTPUT_KIND)];
static unsigned char snapshot0[MERLIN_INPUT0_COUNT *
                               (MERLIN_INPUT0_KIND == KIND_I8 ? 1 : 4) + 1];
static unsigned char snapshot1[MERLIN_INPUT1_COUNT *
                               (MERLIN_INPUT1_KIND == KIND_I8 ? 1 : 4) + 1];
static unsigned char snapshot2[MERLIN_INPUT2_COUNT *
                               (MERLIN_INPUT2_KIND == KIND_I8 ? 1 : 4) + 1];
static unsigned char snapshot3[MERLIN_OUTPUT_COUNT *
                               (MERLIN_OUTPUT_KIND == KIND_I8 ? 1 : 4) + 1];
static unsigned char expected_storage[MERLIN_OUTPUT_COUNT *
                                      (MERLIN_OUTPUT_KIND == KIND_I8 ? 1 : 4) + 1];
static unsigned allocation_index;
#endif

static uint64_t rng_state;
static size_t kind_bytes(int kind);

#ifdef MERLIN_K1_LINUX
/* Trusted link-time interposition records the thread/affinity APIs used by an emitted kernel.
 * This is necessary for sub-jiffy workers: /proc polling alone cannot reliably observe them, and
 * accepting the process-wide 0-N affinity mask would not prove per-hart dispatch. */
static volatile uint64_t observed_pinned_hart_mask;
static volatile uint64_t completed_worker_hart_mask;
static volatile uint64_t productive_worker_hart_mask;
static volatile uint64_t pthread_create_attempts;
static volatile uint64_t pthread_create_successes;
static volatile uint64_t pthread_create_failures;
static volatile uint64_t pthread_worker_completions;
static volatile uint64_t pthread_affinity_attempts;
static volatile uint64_t pthread_affinity_successes;
static volatile uint64_t pthread_affinity_failures;
static volatile uint64_t worker_cpu_ns_by_hart[64];
static volatile uint64_t counterfactual_create_attempts;
static volatile uint64_t counterfactual_create_successes;
static volatile uint64_t counterfactual_create_failures;
static volatile uint64_t counterfactual_suppressed_starts;
static volatile uint64_t audit_serialized_callbacks;
static volatile uint64_t audit_output_coverage;
static volatile uint64_t audit_owner_min_elements;
static volatile uint64_t audit_owner_max_elements;
static volatile uint64_t audit_ownership_violations;
static unsigned char *audit_output_data;
static const unsigned char *audit_expected_data;
static unsigned char *audit_output_owners;
static uint64_t audit_owner_counts[64];
static size_t audit_output_elements;
static size_t audit_output_element_bytes;
/* 0: transparent, 1: secret measured audit, 2: untimed worker-suppression challenge. */
static volatile int trusted_thread_instrumentation_mode;

extern int __real_pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t *);
extern int __real_pthread_create(pthread_t *, const pthread_attr_t *,
                                 void *(*)(void *), void *);
extern int __real_pthread_join(pthread_t, void **);

typedef struct {
  void *(*start)(void *);
  void *argument;
  unsigned mode;
} trusted_pthread_start_t;

typedef struct {
  void *(*start)(void *);
  void *argument;
  uintptr_t token;
  unsigned used;
  unsigned joined;
  unsigned target_hart;
  unsigned target_valid;
} trusted_serial_worker_t;

static trusted_serial_worker_t trusted_serial_workers[64];
static int trusted_current_serial_worker=-1;
#define TRUSTED_THREAD_TOKEN_BASE UINT64_C(0x4d45524c00000100)

static uintptr_t trusted_pthread_bits(pthread_t thread) {
  uintptr_t bits=0;
  size_t count=sizeof(thread)<sizeof(bits)?sizeof(thread):sizeof(bits);
  memcpy(&bits,&thread,count);
  return bits;
}

static void trusted_set_pthread_bits(pthread_t *thread, uintptr_t bits) {
  memset(thread,0,sizeof(*thread));
  size_t count=sizeof(*thread)<sizeof(bits)?sizeof(*thread):sizeof(bits);
  memcpy(thread,&bits,count);
}

static int trusted_serial_slot(pthread_t thread) {
  uintptr_t bits=trusted_pthread_bits(thread);
  if(bits<TRUSTED_THREAD_TOKEN_BASE||bits>=TRUSTED_THREAD_TOKEN_BASE+64)return -1;
  unsigned slot=(unsigned)(bits-TRUSTED_THREAD_TOKEN_BASE);
  return trusted_serial_workers[slot].used?(int)slot:-1;
}

static int trusted_singleton_hart(const cpu_set_t *set, unsigned *hart) {
  unsigned count=0,last=0;
  for(unsigned cpu=0;cpu<MERLIN_HARTS&&cpu<64;++cpu)
    if(CPU_ISSET((int)cpu,set)){++count;last=cpu;}
  if(count!=1)return 0;
  *hart=last;return 1;
}

static int trusted_audit_element_correct(size_t index) {
  const unsigned char *actual=audit_output_data+index*audit_output_element_bytes;
  const unsigned char *expected=audit_expected_data+index*audit_output_element_bytes;
  if(MERLIN_OUTPUT_KIND!=KIND_F32)
    return memcmp(actual,expected,audit_output_element_bytes)==0;
  float a,e;memcpy(&a,actual,sizeof(a));memcpy(&e,expected,sizeof(e));
  return isfinite(a)&&fabs((double)a-(double)e)<=2.0e-4*(1.0+fabs((double)e));
}

static void trusted_audit_assign_correct(unsigned owner) {
  for(size_t index=0;index<audit_output_elements;++index){
    int correct=trusted_audit_element_correct(index);
    if(audit_output_owners[index]==UINT8_MAX){
      if(correct){audit_output_owners[index]=(unsigned char)owner;++audit_owner_counts[owner];}
    } else if(!correct)++audit_ownership_violations;
  }
}

static void trusted_audit_poison_unowned(void) {
  for(size_t index=0;index<audit_output_elements;++index)
    if(audit_output_owners[index]==UINT8_MAX)
      memset(audit_output_data+index*audit_output_element_bytes,0xcc,
             audit_output_element_bytes);
}

static int trusted_expected_has_poison_element(const void *expected) {
  const unsigned char *bytes=(const unsigned char *)expected;
  size_t width=kind_bytes(MERLIN_OUTPUT_KIND);
  for(size_t index=0;index<MERLIN_OUTPUT_COUNT;++index){
    int poison=1;
    for(size_t byte=0;byte<width;++byte)
      if(bytes[index*width+byte]!=0xcc){poison=0;break;}
    if(poison)return 1;
  }
  return 0;
}

int __wrap_pthread_setaffinity_np(pthread_t thread, size_t size, const cpu_set_t *set) {
  int audit=trusted_thread_instrumentation_mode==1;
  unsigned last=0;
  int singleton=trusted_singleton_hart(set,&last);
  /* A treatment may restore the caller's original multi-hart affinity after its
   * exact singleton-pinned shards join.  The restore is real but earns no shard
   * attribution and therefore must not perturb the exact H affinity counters. */
  if(audit&&!singleton)return __real_pthread_setaffinity_np(thread,size,set);
  if(audit)__atomic_fetch_add(&pthread_affinity_attempts,UINT64_C(1),__ATOMIC_RELAXED);
  int slot=audit?trusted_serial_slot(thread):-1;
  if(audit&&slot<0&&trusted_current_serial_worker>=0)slot=trusted_current_serial_worker;
  int rc=(audit&&slot>=0)?0:__real_pthread_setaffinity_np(thread,size,set);
  if(audit&&rc==0){
    __atomic_fetch_add(&pthread_affinity_successes,UINT64_C(1),__ATOMIC_RELAXED);
    if(singleton){
      __atomic_fetch_or(&observed_pinned_hart_mask,UINT64_C(1)<<last,__ATOMIC_RELAXED);
      if(slot>=0){trusted_serial_workers[slot].target_hart=last;
                  trusted_serial_workers[slot].target_valid=1;}
    }
  } else if(audit) __atomic_fetch_add(&pthread_affinity_failures,UINT64_C(1),__ATOMIC_RELAXED);
  return rc;
}

static void *trusted_pthread_start(void *opaque) {
  trusted_pthread_start_t *context=(trusted_pthread_start_t*)opaque;
  void *(*start)(void*)=context->start; void *argument=context->argument;
  unsigned mode=context->mode;
  free(context);
  if(mode==2){
    __atomic_fetch_add(&counterfactual_suppressed_starts,UINT64_C(1),__ATOMIC_RELAXED);
    return NULL;
  }
  struct timespec before,after;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID,&before);
  void *result=start(argument);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID,&after);
  cpu_set_t set;CPU_ZERO(&set);
  if(pthread_getaffinity_np(pthread_self(),sizeof(set),&set)==0){
    unsigned count=0,last=0;
    for(unsigned cpu=0;cpu<MERLIN_HARTS&&cpu<64;++cpu)
      if(CPU_ISSET((int)cpu,&set)){++count;last=cpu;}
    uint64_t before_ns=(uint64_t)before.tv_sec*UINT64_C(1000000000)+(uint64_t)before.tv_nsec;
    uint64_t after_ns=(uint64_t)after.tv_sec*UINT64_C(1000000000)+(uint64_t)after.tv_nsec;
    uint64_t delta=after_ns>before_ns?after_ns-before_ns:0;
    if(count==1&&delta>0) {
      __atomic_fetch_or(&completed_worker_hart_mask,UINT64_C(1)<<last,__ATOMIC_RELAXED);
      __atomic_fetch_add(&worker_cpu_ns_by_hart[last],delta,__ATOMIC_RELAXED);
      /* A worker must spend measurable CPU time inside the submitted start routine. This rejects
       * empty/sleep-only workers; the audited call's independently poisoned output is checked as
       * a whole immediately after all exact worker completions. */
      if(delta>=UINT64_C(100))
        __atomic_fetch_or(&productive_worker_hart_mask,UINT64_C(1)<<last,__ATOMIC_RELAXED);
    }
  }
  __atomic_fetch_add(&pthread_worker_completions,UINT64_C(1),__ATOMIC_RELAXED);
  return result;
}

int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attributes,
                          void *(*start)(void *), void *argument) {
  int mode=trusted_thread_instrumentation_mode;
  if(!mode)
    return __real_pthread_create(thread,attributes,start,argument);
  if(mode==1){
    uint64_t attempt=__atomic_fetch_add(
      &pthread_create_attempts,UINT64_C(1),__ATOMIC_RELAXED);
    if(attempt>=63){
      __atomic_fetch_add(&pthread_create_failures,UINT64_C(1),__ATOMIC_RELAXED);
      return EAGAIN;
    }
    unsigned slot=(unsigned)attempt;
    trusted_serial_workers[slot]=(trusted_serial_worker_t){
      start,argument,TRUSTED_THREAD_TOKEN_BASE+slot,1,0,0,0};
    trusted_set_pthread_bits(thread,trusted_serial_workers[slot].token);
    __atomic_fetch_add(&pthread_create_successes,UINT64_C(1),__ATOMIC_RELAXED);
    return 0;
  }
  else __atomic_fetch_add(&counterfactual_create_attempts,UINT64_C(1),__ATOMIC_RELAXED);
  trusted_pthread_start_t *context=(trusted_pthread_start_t*)malloc(sizeof(*context));
  if(!context){
    if(mode==1)__atomic_fetch_add(&pthread_create_failures,UINT64_C(1),__ATOMIC_RELAXED);
    else __atomic_fetch_add(&counterfactual_create_failures,UINT64_C(1),__ATOMIC_RELAXED);
    return ENOMEM;
  }
  context->start=start;context->argument=argument;context->mode=(unsigned)mode;
  int rc=__real_pthread_create(thread,attributes,trusted_pthread_start,context);
  if(rc!=0){
    free(context);
    if(mode==1)__atomic_fetch_add(&pthread_create_failures,UINT64_C(1),__ATOMIC_RELAXED);
    else __atomic_fetch_add(&counterfactual_create_failures,UINT64_C(1),__ATOMIC_RELAXED);
    return rc;
  }
  if(mode==1)__atomic_fetch_add(&pthread_create_successes,UINT64_C(1),__ATOMIC_RELAXED);
  else __atomic_fetch_add(&counterfactual_create_successes,UINT64_C(1),__ATOMIC_RELAXED);
  return 0;
}

int __wrap_pthread_join(pthread_t thread, void **result) {
  if(trusted_thread_instrumentation_mode!=1)return __real_pthread_join(thread,result);
  int slot=trusted_serial_slot(thread);
  if(slot<0||trusted_serial_workers[slot].joined){
    ++audit_ownership_violations;
    return ESRCH;
  }
  trusted_serial_worker_t *worker=&trusted_serial_workers[slot];
  /* Anything already correct belongs to main-thread work performed before this join. */
  trusted_audit_assign_correct(0);
  /* Reset every unowned element immediately before the exact submitted callback.  Therefore a
   * worker earns ownership only through its own poison-to-correct transition.  Algorithms that
   * require reductions through shared partial output, deferred callbacks, or persistent pools are
   * intentionally unsupported by this proof and fail closed. */
  trusted_audit_poison_unowned();
  struct timespec before,after;
  trusted_current_serial_worker=slot;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID,&before);
  void *worker_result=worker->start(worker->argument);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID,&after);
  trusted_current_serial_worker=-1;
  worker->joined=1;
  ++audit_serialized_callbacks;
  ++pthread_worker_completions;
  uint64_t before_ns=(uint64_t)before.tv_sec*UINT64_C(1000000000)+(uint64_t)before.tv_nsec;
  uint64_t after_ns=(uint64_t)after.tv_sec*UINT64_C(1000000000)+(uint64_t)after.tv_nsec;
  uint64_t delta=after_ns>before_ns?after_ns-before_ns:0;
  if(worker->target_valid&&worker->target_hart<64){
    unsigned hart=worker->target_hart;
    if(delta>0){completed_worker_hart_mask|=UINT64_C(1)<<hart;
                worker_cpu_ns_by_hart[hart]+=delta;}
    if(delta>=UINT64_C(100))productive_worker_hart_mask|=UINT64_C(1)<<hart;
    trusted_audit_assign_correct(hart);
  } else ++audit_ownership_violations;
  if(result)*result=worker_result;
  return 0;
}

static uint64_t wall_ns(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}
static uint64_t read_time(void) {
  uint64_t value; __asm__ volatile("rdtime %0" : "=r"(value)); return value;
}
static uint64_t read_vlenb(void) {
  uint64_t value; __asm__ volatile("csrr %0, vlenb" : "=r"(value)); return value;
}
#elif defined(MERLIN_FREESTANDING)
static uint64_t read_cycle(void) {
  uint64_t value; __asm__ volatile("csrr %0, mcycle" : "=r"(value)); return value;
}
#endif

static uint32_t next_random(void) {
  uint64_t x = rng_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  rng_state = x;
  return (uint32_t)((x * UINT64_C(2685821657736338717)) >> 32);
}

static size_t kind_bytes(int kind) {
  if (kind == KIND_I8) return 1;
  if (kind == KIND_F32 || kind == KIND_I32) return 4;
  return 1;
}

static guarded_buffer_t allocate_guarded(size_t elements, int kind) {
  guarded_buffer_t buffer;
  buffer.bytes = elements * kind_bytes(kind);
#ifdef MERLIN_FREESTANDING
  unsigned char *storages[] = {storage0, storage1, storage2, storage3};
  unsigned char *snapshots[] = {snapshot0, snapshot1, snapshot2, snapshot3};
  if (allocation_index >= 4) htif_exit(90);
  buffer.raw = storages[allocation_index];
  buffer.snapshot = snapshots[allocation_index++];
  memset(buffer.raw, GUARD_VALUE, buffer.bytes + 2 * GUARD_BYTES);
#else
  size_t allocation = buffer.bytes + 2 * GUARD_BYTES;
  buffer.raw = (unsigned char *)mmap(NULL, allocation ? allocation : 1,
                                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  buffer.snapshot = (unsigned char *)malloc(buffer.bytes ? buffer.bytes : 1);
  if (buffer.raw == MAP_FAILED || !buffer.snapshot) {
    fprintf(stderr, "allocation failure\n");
    exit(90);
  }
  memset(buffer.raw, GUARD_VALUE, allocation);
#endif
  buffer.data = buffer.raw + GUARD_BYTES;
  return buffer;
}

static int guards_ok(const guarded_buffer_t *buffer) {
  for (size_t i = 0; i < GUARD_BYTES; ++i) {
    if (buffer->raw[i] != GUARD_VALUE ||
        buffer->raw[GUARD_BYTES + buffer->bytes + i] != GUARD_VALUE) return 0;
  }
  return 1;
}

static double load_number(const void *data, int kind, size_t index) {
  if (kind == KIND_F32) return ((const float *)data)[index];
  if (kind == KIND_I8) return ((const int8_t *)data)[index];
  return ((const int32_t *)data)[index];
}

static int64_t load_integer(const void *data, int kind, size_t index) {
  if (kind == KIND_I8) return ((const int8_t *)data)[index];
  if (kind == KIND_I32) return ((const int32_t *)data)[index];
  return (int64_t)((const float *)data)[index];
}

static void store_number(void *data, int kind, size_t index, double value) {
  if (kind == KIND_F32) ((float *)data)[index] = (float)value;
  else if (kind == KIND_I8) ((int8_t *)data)[index] = (int8_t)value;
  else ((int32_t *)data)[index] = (int32_t)value;
}

static void fill_input(guarded_buffer_t *buffer, int kind, size_t elements) {
  for (size_t i = 0; i < elements; ++i) {
    uint32_t value = next_random();
    if (kind == KIND_F32) ((float *)buffer->data)[i] = ((int)(value % 1025) - 512) / 256.0f;
    else if (kind == KIND_I8) ((int8_t *)buffer->data)[i] = (int8_t)((int)(value % 17) - 8);
    else ((int32_t *)buffer->data)[i] = (int32_t)((int)(value % 129) - 64);
  }
  memcpy(buffer->snapshot, buffer->data, buffer->bytes);
}

static size_t rhs_index(size_t k, size_t j) {
#if MERLIN_LAYOUT == LAYOUT_TRANSPOSED_RHS
  return j * MERLIN_DIM2 + k;
#elif MERLIN_LAYOUT == LAYOUT_ROW_PACKED_RHS
  return (j / 8) * MERLIN_DIM2 * 8 + k * 8 + (j % 8);
#else
  return k * MERLIN_DIM1 + j;
#endif
}

static void reference_contraction(const guarded_buffer_t *a, const guarded_buffer_t *b,
                                  void *expected) {
  for (size_t i = 0; i < MERLIN_DIM0; ++i) {
    for (size_t j = 0; j < MERLIN_DIM1; ++j) {
      if (MERLIN_OUTPUT_KIND == KIND_F32) {
        float acc = 0.0f;
        for (size_t k = 0; k < MERLIN_DIM2; ++k)
          acc += (float)load_number(a->data, MERLIN_INPUT0_KIND, i * MERLIN_DIM2 + k) *
                 (float)load_number(b->data, MERLIN_INPUT1_KIND, rhs_index(k, j));
        ((float *)expected)[i * MERLIN_DIM1 + j] = acc;
      } else {
        int32_t acc = 0;
        for (size_t k = 0; k < MERLIN_DIM2; ++k)
          acc += (int32_t)(load_integer(a->data, MERLIN_INPUT0_KIND, i * MERLIN_DIM2 + k) *
                           load_integer(b->data, MERLIN_INPUT1_KIND, rhs_index(k, j)));
        ((int32_t *)expected)[i * MERLIN_DIM1 + j] = acc;
      }
    }
  }
}

static void reference_elementwise(const guarded_buffer_t *a, const guarded_buffer_t *b,
                                  void *expected) {
  for (size_t i = 0; i < MERLIN_DIM0; ++i) {
    if (MERLIN_OUTPUT_KIND == KIND_F32) {
      float x = (float)load_number(a->data, MERLIN_INPUT0_KIND, i);
      float y = (MERLIN_SEMANTIC_OP == OP_ADD || MERLIN_SEMANTIC_OP == OP_MULTIPLY)
                    ? (float)load_number(b->data, MERLIN_INPUT1_KIND, i) : 0.0f;
      float z = x;
      if (MERLIN_SEMANTIC_OP == OP_ADD) z = x + y;
      else if (MERLIN_SEMANTIC_OP == OP_MULTIPLY) z = x * y;
      else if (MERLIN_SEMANTIC_OP == OP_RELU) z = x > 0.0f ? x : 0.0f;
      else if (MERLIN_SEMANTIC_OP == OP_SILU) z = x / (1.0f + expf(-x));
      else if (MERLIN_SEMANTIC_OP == OP_GELU)
        z = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f*x*x*x)));
      else if (MERLIN_SEMANTIC_OP == OP_CLAMP) z = fmaxf(-1.0f, fminf(1.0f, x));
      else if (MERLIN_SEMANTIC_OP == OP_REQUANT) z = x * 0.25f;
      ((float *)expected)[i] = z;
    } else {
      int64_t x = load_integer(a->data, MERLIN_INPUT0_KIND, i);
      int64_t y = (MERLIN_SEMANTIC_OP == OP_ADD || MERLIN_SEMANTIC_OP == OP_MULTIPLY)
                      ? load_integer(b->data, MERLIN_INPUT1_KIND, i) : 0;
      int64_t z = x;
      if (MERLIN_SEMANTIC_OP == OP_ADD) z = x + y;
      else if (MERLIN_SEMANTIC_OP == OP_MULTIPLY) z = x * y;
      else if (MERLIN_SEMANTIC_OP == OP_RELU || MERLIN_SEMANTIC_OP == OP_SILU ||
               MERLIN_SEMANTIC_OP == OP_GELU) z = x > 0 ? x : 0;
      else if (MERLIN_SEMANTIC_OP == OP_CLAMP) z = x < -8 ? -8 : (x > 8 ? 8 : x);
      else if (MERLIN_SEMANTIC_OP == OP_REQUANT) z = x / 4;
      store_number(expected, MERLIN_OUTPUT_KIND, i, (double)z);
    }
  }
}

static void reference_reduction(const guarded_buffer_t *a, void *expected) {
  if (MERLIN_OUTPUT_KIND == KIND_F32) {
    float sum = 0.0f, sumsq = 0.0f, maximum = -INFINITY;
    for (size_t i = 0; i < MERLIN_DIM0; ++i) {
      float x = (float)load_number(a->data, MERLIN_INPUT0_KIND, i);
      sum += x; sumsq += x*x; if (x > maximum) maximum = x;
    }
    if (MERLIN_SEMANTIC_OP == OP_SUM) ((float *)expected)[0] = sum;
    else if (MERLIN_SEMANTIC_OP == OP_MAX) ((float *)expected)[0] = maximum;
    else if (MERLIN_SEMANTIC_OP == OP_SOFTMAX_COMPONENTS) {
      float exp_sum = 0.0f;
      for (size_t i = 0; i < MERLIN_DIM0; ++i)
        exp_sum += expf((float)load_number(a->data, MERLIN_INPUT0_KIND, i) - maximum);
      ((float *)expected)[0] = maximum; ((float *)expected)[1] = exp_sum;
    } else {
      ((float *)expected)[0] = sum; ((float *)expected)[1] = sumsq;
    }
  } else {
    int32_t sum = 0, sumsq = 0, maximum = INT32_MIN;
    for (size_t i = 0; i < MERLIN_DIM0; ++i) {
      int32_t x = (int32_t)load_integer(a->data, MERLIN_INPUT0_KIND, i);
      sum += x; sumsq += x*x; if (x > maximum) maximum = x;
    }
    if (MERLIN_SEMANTIC_OP == OP_MAX) ((int32_t *)expected)[0] = maximum;
    else if (MERLIN_SEMANTIC_OP == OP_SOFTMAX_COMPONENTS) {
      int32_t exp_sum = 0;
      for (size_t i = 0; i < MERLIN_DIM0; ++i) {
        int32_t delta = (int32_t)load_integer(a->data, MERLIN_INPUT0_KIND, i) - maximum;
        exp_sum += delta <= -8 ? 1 : (1 << (8 + delta));
      }
      ((int32_t *)expected)[0] = maximum; ((int32_t *)expected)[1] = exp_sum;
    }
    else if (MERLIN_SEMANTIC_OP == OP_LAYERNORM_COMPONENTS) {
      ((int32_t *)expected)[0] = sum; ((int32_t *)expected)[1] = sumsq;
    } else ((int32_t *)expected)[0] = sum;
  }
}

static void reference_movement(const guarded_buffer_t *a, const guarded_buffer_t *b,
                               void *expected) {
  size_t elem_bytes = kind_bytes(MERLIN_OUTPUT_KIND);
  if (MERLIN_SEMANTIC_OP == OP_COPY) memcpy(expected, a->data, MERLIN_OUTPUT_COUNT * elem_bytes);
  else if (MERLIN_SEMANTIC_OP == OP_TRANSPOSE2D) {
    for (size_t i = 0; i < MERLIN_DIM1; ++i)
      for (size_t j = 0; j < MERLIN_DIM2; ++j)
        store_number(expected, MERLIN_OUTPUT_KIND, j * MERLIN_DIM1 + i,
                     load_number(a->data, MERLIN_INPUT0_KIND, i * MERLIN_DIM2 + j));
  } else if (MERLIN_SEMANTIC_OP == OP_PACK_RHS) {
    memset(expected, 0, MERLIN_OUTPUT_COUNT * elem_bytes);
    for (size_t k = 0; k < MERLIN_DIM1; ++k)
      for (size_t j = 0; j < MERLIN_DIM2; ++j)
        store_number(expected, MERLIN_OUTPUT_KIND, (j/8)*MERLIN_DIM1*8 + k*8 + j%8,
                     load_number(a->data, MERLIN_INPUT0_KIND, k*MERLIN_DIM2+j));
  } else if (MERLIN_SEMANTIC_OP == OP_UNPACK) {
    for (size_t k = 0; k < MERLIN_DIM1; ++k)
      for (size_t j = 0; j < MERLIN_DIM2; ++j)
        store_number(expected, MERLIN_OUTPUT_KIND, k*MERLIN_DIM2+j,
                     load_number(a->data, MERLIN_INPUT0_KIND,
                                 (j/8)*MERLIN_DIM1*8 + k*8 + j%8));
  } else if (MERLIN_SEMANTIC_OP == OP_STRIDED_SLICE) {
    for (size_t i = 0; i < MERLIN_OUTPUT_COUNT; ++i)
      store_number(expected, MERLIN_OUTPUT_KIND, i,
                   load_number(a->data, MERLIN_INPUT0_KIND, i*2));
  } else {
    memcpy(expected, a->data, MERLIN_INPUT0_COUNT * elem_bytes);
    memcpy((unsigned char *)expected + MERLIN_INPUT0_COUNT * elem_bytes, b->data,
           MERLIN_INPUT1_COUNT * elem_bytes);
  }
}

static void reference_fusion(const guarded_buffer_t *a, const guarded_buffer_t *b,
                             const guarded_buffer_t *c, void *expected) {
  reference_contraction(a, b, expected);
  if (MERLIN_OUTPUT_KIND == KIND_F32) {
    for (size_t i = 0; i < MERLIN_DIM0; ++i) {
      if (MERLIN_SEMANTIC_OP == OP_RESIDUAL_NORM) {
        float mean = 0.0f, variance = 0.0f;
        for (size_t j = 0; j < MERLIN_DIM1; ++j) {
          size_t q = i*MERLIN_DIM1+j;
          ((float *)expected)[q] += ((const float *)c->data)[q];
          mean += ((float *)expected)[q];
        }
        mean /= (float)MERLIN_DIM1;
        for (size_t j = 0; j < MERLIN_DIM1; ++j) {
          float d = ((float *)expected)[i*MERLIN_DIM1+j] - mean; variance += d*d;
        }
        float scale = 1.0f/sqrtf(variance/(float)MERLIN_DIM1 + 1.0e-5f);
        for (size_t j = 0; j < MERLIN_DIM1; ++j) {
          size_t q=i*MERLIN_DIM1+j; ((float *)expected)[q]=(((float *)expected)[q]-mean)*scale;
        }
      } else for (size_t j = 0; j < MERLIN_DIM1; ++j) {
        size_t q=i*MERLIN_DIM1+j; float z=((float *)expected)[q];
        if (MERLIN_SEMANTIC_OP == OP_MATMUL_BIAS ||
            MERLIN_SEMANTIC_OP == OP_MATMUL_BIAS_RELU) z += ((const float *)c->data)[j];
        if (MERLIN_SEMANTIC_OP == OP_MATMUL_BIAS_RELU && z < 0.0f) z=0.0f;
        if (MERLIN_SEMANTIC_OP == OP_MATMUL_REQUANT) z *= 0.25f;
        ((float *)expected)[q]=z;
      }
    }
  } else {
    for (size_t i=0;i<MERLIN_DIM0;++i) {
      int32_t mean = 0;
      if (MERLIN_SEMANTIC_OP == OP_RESIDUAL_NORM) {
        for (size_t j=0;j<MERLIN_DIM1;++j) {
          size_t q=i*MERLIN_DIM1+j;
          ((int32_t *)expected)[q] += ((const int32_t *)c->data)[q];
          mean += ((int32_t *)expected)[q];
        }
        if (MERLIN_DIM1) mean /= (int32_t)MERLIN_DIM1;
      }
      for (size_t j=0;j<MERLIN_DIM1;++j) {
        size_t q=i*MERLIN_DIM1+j; int32_t z=((int32_t *)expected)[q];
        if (MERLIN_SEMANTIC_OP == OP_MATMUL_BIAS ||
            MERLIN_SEMANTIC_OP == OP_MATMUL_BIAS_RELU) z += ((const int32_t *)c->data)[j];
        if (MERLIN_SEMANTIC_OP == OP_MATMUL_BIAS_RELU && z < 0) z=0;
        if (MERLIN_SEMANTIC_OP == OP_MATMUL_REQUANT) z /= 4;
        if (MERLIN_SEMANTIC_OP == OP_RESIDUAL_NORM) z -= mean;
        ((int32_t *)expected)[q]=z;
      }
    }
  }
}

static void reference_runtime(const guarded_buffer_t *a, const guarded_buffer_t *b, void *expected) {
  for (size_t i=0;i<MERLIN_DIM0;++i)
    ((float *)expected)[i]=((const float *)a->data)[i]+((const float *)b->data)[i];
}

static void compute_reference(const guarded_buffer_t *a, const guarded_buffer_t *b,
                              const guarded_buffer_t *c, void *expected) {
  memset(expected,0,MERLIN_OUTPUT_COUNT*kind_bytes(MERLIN_OUTPUT_KIND));
#if MERLIN_FAMILY == FAMILY_CONTRACTION
  reference_contraction(a,b,expected);
#elif MERLIN_FAMILY == FAMILY_ELEMENTWISE
  reference_elementwise(a,b,expected);
#elif MERLIN_FAMILY == FAMILY_REDUCTION
  reference_reduction(a,expected);
#elif MERLIN_FAMILY == FAMILY_MOVEMENT
  reference_movement(a,b,expected);
#elif MERLIN_FAMILY == FAMILY_FUSION
  reference_fusion(a,b,c,expected);
#else
  reference_runtime(a,b,expected);
#endif
}

static int buffers_unchanged(const guarded_buffer_t *a, const guarded_buffer_t *b,
                             const guarded_buffer_t *c, const guarded_buffer_t *out) {
  return guards_ok(a)&&guards_ok(b)&&guards_ok(c)&&guards_ok(out)&&
    memcmp(a->data,a->snapshot,a->bytes)==0&&
    memcmp(b->data,b->snapshot,b->bytes)==0&&
    memcmp(c->data,c->snapshot,c->bytes)==0;
}

/* Refresh every submitted input after code generation and before each retained K1 invocation.
 * fill_input also snapshots the trusted mutation, so a kernel cannot modify an input unnoticed. */
static void refresh_measured_inputs(guarded_buffer_t *a, guarded_buffer_t *b,
                                    guarded_buffer_t *c) {
  fill_input(a,MERLIN_INPUT0_KIND,MERLIN_INPUT0_COUNT);
  fill_input(b,MERLIN_INPUT1_KIND,MERLIN_INPUT1_COUNT);
  fill_input(c,MERLIN_INPUT2_KIND,MERLIN_INPUT2_COUNT);
}

static int outputs_match(const void *actual, const void *expected, double *max_abs) {
  *max_abs = 0.0;
  if (MERLIN_OUTPUT_KIND != KIND_F32)
    return memcmp(actual, expected, MERLIN_OUTPUT_COUNT * kind_bytes(MERLIN_OUTPUT_KIND)) == 0;
  for (size_t i=0;i<MERLIN_OUTPUT_COUNT;++i) {
    double a=((const float *)actual)[i], e=((const float *)expected)[i];
    double error=fabs(a-e); if (error>*max_abs) *max_abs=error;
    if (!isfinite(a) || error > 2.0e-4*(1.0+fabs(e))) return 0;
  }
  return 1;
}

static int output_is_poison(const void *data) {
  const unsigned char *bytes=(const unsigned char *)data;
  size_t count=MERLIN_OUTPUT_COUNT*kind_bytes(MERLIN_OUTPUT_KIND);
  for(size_t i=0;i<count;++i)if(bytes[i]!=0xcc)return 0;
  return 1;
}

#ifdef MERLIN_FREESTANDING
int main(uint64_t hartid) {
  if (hartid != 0) for (;;) { __asm__ volatile("wfi"); }
  console_init();
  uint64_t observed_vlenb;
  __asm__ volatile("csrr %0, vlenb" : "=r"(observed_vlenb));
  if (observed_vlenb * 8 != MERLIN_VLEN_BITS) {
    htif_puts("FAIL vlenb="); htif_putd((long)observed_vlenb); htif_putc('\n'); htif_exit(1);
  }
  rng_state = MERLIN_SEED;
#else
int main(int argc, char **argv) {
  if (argc != 2) return 91;
  rng_state = strtoull(argv[1], NULL, 10);
#endif
  if (!rng_state) rng_state = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t receipt_seed = rng_state;
  guarded_buffer_t a=allocate_guarded(MERLIN_INPUT0_COUNT, MERLIN_INPUT0_KIND);
  guarded_buffer_t b=allocate_guarded(MERLIN_INPUT1_COUNT, MERLIN_INPUT1_KIND);
  guarded_buffer_t c=allocate_guarded(MERLIN_INPUT2_COUNT, MERLIN_INPUT2_KIND);
  guarded_buffer_t out=allocate_guarded(MERLIN_OUTPUT_COUNT, MERLIN_OUTPUT_KIND);
  fill_input(&a,MERLIN_INPUT0_KIND,MERLIN_INPUT0_COUNT);
  fill_input(&b,MERLIN_INPUT1_KIND,MERLIN_INPUT1_COUNT);
  fill_input(&c,MERLIN_INPUT2_KIND,MERLIN_INPUT2_COUNT);
  memset(out.data,0xcc,out.bytes); memcpy(out.snapshot,out.data,out.bytes);
#ifdef MERLIN_FREESTANDING
  void *expected=expected_storage; memset(expected,0,sizeof(expected_storage));
#else
  void *expected=calloc(MERLIN_OUTPUT_COUNT,kind_bytes(MERLIN_OUTPUT_KIND));
  if (!expected) return 90;
#endif
  compute_reference(&a,&b,&c,expected);
  merlin_capsule_params_t params={1,MERLIN_FAMILY,MERLIN_OPERATION_CODE,MERLIN_DTYPE_CODE,
    MERLIN_LAYOUT,MERLIN_HARTS,MERLIN_VLEN_BITS,MERLIN_DIM0,MERLIN_DIM1,MERLIN_DIM2,MERLIN_STATE0};
#ifdef MERLIN_K1_LINUX
  typedef struct {
    volatile uint64_t completed, elapsed, time_ticks, calls, vlenb;
    volatile uint64_t audit_call, audit_wall_ns, audit_time_ticks, correctness_checks;
    volatile uint64_t pinned_hart_mask, worker_hart_mask, productive_worker_hart_mask;
    volatile uint64_t pthread_create_attempts, pthread_creates, pthread_create_failures;
    volatile uint64_t pthread_completions, pthread_affinity_attempts;
    volatile uint64_t pthread_affinity_successes, pthread_affinity_failures;
    volatile uint64_t minimum_worker_cpu_ns;
    volatile uint64_t counterfactual_create_attempts, counterfactual_creates;
    volatile uint64_t counterfactual_create_failures, counterfactual_suppressed_starts;
    volatile uint64_t audit_serialized_callbacks, audit_output_elements;
    volatile uint64_t audit_output_coverage, audit_owner_min_elements;
    volatile uint64_t audit_owner_max_elements, audit_ownership_violations;
    volatile int rc, memory_ok, numeric_ok, audit_output_changed;
    volatile int counterfactual_worker_dependence, audit_balanced_shards;
  } child_report_t;
  child_report_t *child_report=(child_report_t *)mmap(
    NULL,sizeof(*child_report),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
  if (child_report == MAP_FAILED) return 90;
  memset(child_report,0,sizeof(*child_report));
  pid_t child=fork();
  if (child < 0) return 90;
  if (child == 0) {
    int null_fd=open("/dev/null",O_WRONLY);
    if (null_fd >= 0) { dup2(null_fd,STDOUT_FILENO); close(null_fd); }
    int child_rc=0,child_memory_ok=1,child_numeric_ok=1,audit_output_changed=0;
    int audit_balanced_shards=0;
    double child_max_abs=0.0;
    trusted_thread_instrumentation_mode=0;
    for (unsigned i=0;i<3;++i) {
      memset(out.data,0xcc,out.bytes);
      child_rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data);
      child_memory_ok=buffers_unchanged(&a,&b,&c,&out);
      child_numeric_ok=outputs_match(out.data,expected,&child_max_abs);
      if(child_rc||!child_memory_ok||!child_numeric_ok)break;
    }
    /* The audit ordinal is selected from the post-input trusted PRNG after emitted code is frozen.
     * Exactly that measured invocation enables link wrappers. Its instrumentation-heavy duration
     * and invocation are excluded from the reported latency numerator and denominator. */
    const uint64_t audit_iteration=(uint64_t)(next_random()%20u);
    uint64_t child_calls=0,total_iterations=0,elapsed=0,time_ticks=0,correctness_checks=0;
    uint64_t audit_wall=0,audit_ticks=0;
    while (!child_rc && child_memory_ok && child_numeric_ok &&
           (child_calls < 20 || elapsed < UINT64_C(500000000))) {
      int audit=total_iterations==audit_iteration;
      refresh_measured_inputs(&a,&b,&c);
      memset(out.data,0xcc,out.bytes);
      compute_reference(&a,&b,&c,expected);
      /* A correct value equal to the byte poison is ambiguous.  The trusted PRNG may refresh only
       * this public audit input until every element has an unambiguous poison->correct transition. */
      if(audit){
        unsigned retries=0;
        while(trusted_expected_has_poison_element(expected)&&retries++<32){
          refresh_measured_inputs(&a,&b,&c);
          memset(out.data,0xcc,out.bytes);
          compute_reference(&a,&b,&c,expected);
        }
        if(trusted_expected_has_poison_element(expected)){child_rc=98;break;}
        audit_output_owners=(unsigned char*)malloc(MERLIN_OUTPUT_COUNT?MERLIN_OUTPUT_COUNT:1);
        if(!audit_output_owners){child_rc=90;break;}
        memset(audit_output_owners,UINT8_MAX,MERLIN_OUTPUT_COUNT);
        memset(audit_owner_counts,0,sizeof(audit_owner_counts));
        memset(trusted_serial_workers,0,sizeof(trusted_serial_workers));
        audit_output_data=out.data;
        audit_expected_data=(const unsigned char*)expected;
        audit_output_elements=MERLIN_OUTPUT_COUNT;
        audit_output_element_bytes=kind_bytes(MERLIN_OUTPUT_KIND);
      } else if(output_is_poison(expected)){child_rc=98;break;}
      trusted_thread_instrumentation_mode=audit;
      uint64_t begin_wall=wall_ns(),begin_time=read_time();
      child_rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data);
      uint64_t end_time=read_time(),end_wall=wall_ns();
      trusted_thread_instrumentation_mode=0;
      child_memory_ok=buffers_unchanged(&a,&b,&c,&out);
      child_numeric_ok=outputs_match(out.data,expected,&child_max_abs);
      ++correctness_checks;
      if(audit){
        /* Credit work after the last join to the controller/main shard, then require exact full,
         * disjoint and balanced coverage.  In particular, flag-only workers receive zero elements
         * and cannot be used to attribute main-thread computation to multiple harts. */
        trusted_audit_assign_correct(0);
        uint64_t coverage=0,minimum=UINT64_MAX,maximum=0;
        for(unsigned owner=0;owner<MERLIN_HARTS&&owner<64;++owner){
          uint64_t count=audit_owner_counts[owner];coverage+=count;
          if(count<minimum)minimum=count;if(count>maximum)maximum=count;
        }
        audit_output_coverage=coverage;
        audit_owner_min_elements=minimum==UINT64_MAX?0:minimum;
        audit_owner_max_elements=maximum;
        uint64_t floor_count=MERLIN_OUTPUT_COUNT/MERLIN_HARTS;
        uint64_t ceil_count=(MERLIN_OUTPUT_COUNT+MERLIN_HARTS-1)/MERLIN_HARTS;
        audit_balanced_shards=(
          MERLIN_HARTS<=64&&floor_count>0&&
          audit_serialized_callbacks==(MERLIN_HARTS>0?MERLIN_HARTS-1:0)&&
          coverage==MERLIN_OUTPUT_COUNT&&audit_ownership_violations==0&&
          audit_owner_min_elements>=floor_count&&audit_owner_max_elements<=ceil_count);
        audit_wall=end_wall-begin_wall;audit_ticks=end_time-begin_time;
        audit_output_changed=!output_is_poison(out.data);
        audit_output_data=NULL;audit_expected_data=NULL;
        free(audit_output_owners);audit_output_owners=NULL;
      } else {
        elapsed+=end_wall-begin_wall;time_ticks+=end_time-begin_time;++child_calls;
      }
      ++total_iterations;
    }
    /* Untimed causal challenge: rerun a fresh input while the trusted wrapper suppresses every
     * worker start routine but still makes pthread_create/join appear successful. A scalar main
     * thread with decorative busy workers remains correct and is rejected; a genuinely partitioned
     * implementation must lose correctness (or report failure) when its worker shards are removed. */
    int counterfactual_worker_dependence=MERLIN_HARTS<=1;
    if(MERLIN_HARTS>1&&!child_rc&&child_memory_ok&&child_numeric_ok){
      refresh_measured_inputs(&a,&b,&c);
      memset(out.data,0xcc,out.bytes);
      compute_reference(&a,&b,&c,expected);
      int counterfactual_rc=0,counterfactual_numeric_ok=0;
      if(output_is_poison(expected))child_rc=98;
      else {
        trusted_thread_instrumentation_mode=2;
        counterfactual_rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data);
        trusted_thread_instrumentation_mode=0;
        child_memory_ok=child_memory_ok&&buffers_unchanged(&a,&b,&c,&out);
        counterfactual_numeric_ok=outputs_match(out.data,expected,&child_max_abs);
        counterfactual_worker_dependence=(
          counterfactual_create_attempts==MERLIN_HARTS-1&&
          counterfactual_create_successes==MERLIN_HARTS-1&&
          counterfactual_create_failures==0&&
          counterfactual_suppressed_starts==MERLIN_HARTS-1&&
          (counterfactual_rc!=0||!counterfactual_numeric_ok));
      }
      /* Restore a fresh, normally executed output for the independent parent-side final check. */
      if(!child_rc&&child_memory_ok){
        refresh_measured_inputs(&a,&b,&c);
        memset(out.data,0xcc,out.bytes);
        compute_reference(&a,&b,&c,expected);
        child_rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data);
        child_memory_ok=child_memory_ok&&buffers_unchanged(&a,&b,&c,&out);
        child_numeric_ok=child_numeric_ok&&counterfactual_worker_dependence&&
          outputs_match(out.data,expected,&child_max_abs);
      }
    }
    uint64_t minimum_worker_cpu_ns=UINT64_MAX;
    if(MERLIN_HARTS<=1)minimum_worker_cpu_ns=0;
    else for(unsigned hart=1;hart<MERLIN_HARTS&&hart<64;++hart)
      if(worker_cpu_ns_by_hart[hart]<minimum_worker_cpu_ns)
        minimum_worker_cpu_ns=worker_cpu_ns_by_hart[hart];
    child_report->elapsed=elapsed;
    child_report->time_ticks=time_ticks;
    child_report->calls=child_calls;
    child_report->audit_call=audit_iteration+1;
    child_report->audit_wall_ns=audit_wall;
    child_report->audit_time_ticks=audit_ticks;
    child_report->correctness_checks=correctness_checks;
    child_report->vlenb=read_vlenb();
    child_report->pinned_hart_mask=observed_pinned_hart_mask;
    child_report->worker_hart_mask=completed_worker_hart_mask;
    child_report->productive_worker_hart_mask=productive_worker_hart_mask;
    child_report->pthread_create_attempts=pthread_create_attempts;
    child_report->pthread_creates=pthread_create_successes;
    child_report->pthread_create_failures=pthread_create_failures;
    child_report->pthread_completions=pthread_worker_completions;
    child_report->pthread_affinity_attempts=pthread_affinity_attempts;
    child_report->pthread_affinity_successes=pthread_affinity_successes;
    child_report->pthread_affinity_failures=pthread_affinity_failures;
    child_report->minimum_worker_cpu_ns=minimum_worker_cpu_ns;
    child_report->counterfactual_create_attempts=counterfactual_create_attempts;
    child_report->counterfactual_creates=counterfactual_create_successes;
    child_report->counterfactual_create_failures=counterfactual_create_failures;
    child_report->counterfactual_suppressed_starts=counterfactual_suppressed_starts;
    child_report->audit_serialized_callbacks=audit_serialized_callbacks;
    child_report->audit_output_elements=MERLIN_OUTPUT_COUNT;
    child_report->audit_output_coverage=audit_output_coverage;
    child_report->audit_owner_min_elements=audit_owner_min_elements;
    child_report->audit_owner_max_elements=audit_owner_max_elements;
    child_report->audit_ownership_violations=audit_ownership_violations;
    child_report->memory_ok=child_memory_ok;
    child_report->numeric_ok=child_numeric_ok;
    child_report->audit_output_changed=audit_output_changed;
    child_report->counterfactual_worker_dependence=counterfactual_worker_dependence;
    child_report->audit_balanced_shards=audit_balanced_shards;
    child_report->rc=child_rc;
    __sync_synchronize(); child_report->completed=UINT64_C(0x4d45524c494e4f4b);
    _exit(0);
  }
  int child_status=0;
  if (waitpid(child,&child_status,0) != child || !WIFEXITED(child_status) ||
      WEXITSTATUS(child_status) != 0 || child_report->completed != UINT64_C(0x4d45524c494e4f4b))
    child_report->rc=95;
  int rc=child_report->rc;
  uint64_t elapsed=child_report->elapsed, end_time=child_report->time_ticks;
  uint64_t calls=child_report->calls, vlenb=child_report->vlenb;
  uint64_t audit_call=child_report->audit_call;
  uint64_t audit_wall_ns=child_report->audit_wall_ns;
  uint64_t audit_time_ticks=child_report->audit_time_ticks;
  uint64_t correctness_checks=child_report->correctness_checks;
  uint64_t pinned_hart_mask=child_report->pinned_hart_mask;
  uint64_t worker_hart_mask=child_report->worker_hart_mask;
  uint64_t productive_worker_mask=child_report->productive_worker_hart_mask;
  uint64_t pthread_attempts=child_report->pthread_create_attempts;
  uint64_t pthread_creates=child_report->pthread_creates;
  uint64_t pthread_create_failures=child_report->pthread_create_failures;
  uint64_t pthread_completions=child_report->pthread_completions;
  uint64_t affinity_attempts=child_report->pthread_affinity_attempts;
  uint64_t affinity_successes=child_report->pthread_affinity_successes;
  uint64_t affinity_failures=child_report->pthread_affinity_failures;
  uint64_t minimum_worker_cpu_ns=child_report->minimum_worker_cpu_ns;
  uint64_t counterfactual_attempts=child_report->counterfactual_create_attempts;
  uint64_t counterfactual_creates=child_report->counterfactual_creates;
  uint64_t counterfactual_failures=child_report->counterfactual_create_failures;
  uint64_t counterfactual_suppressed=child_report->counterfactual_suppressed_starts;
  uint64_t audit_serialized=child_report->audit_serialized_callbacks;
  uint64_t audit_elements=child_report->audit_output_elements;
  uint64_t audit_coverage=child_report->audit_output_coverage;
  uint64_t audit_owner_min=child_report->audit_owner_min_elements;
  uint64_t audit_owner_max=child_report->audit_owner_max_elements;
  uint64_t audit_owner_violations=child_report->audit_ownership_violations;
  int timed_memory_ok=child_report->memory_ok;
  int timed_numeric_ok=child_report->numeric_ok;
  int audit_output_changed=child_report->audit_output_changed;
  int worker_dependence=child_report->counterfactual_worker_dependence;
  int audit_balanced_shards=child_report->audit_balanced_shards;
#elif defined(MERLIN_FREESTANDING)
  int rc=0;
  for (unsigned i=0;i<3;++i)
    if ((rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data))) break;
  uint64_t begin_cycle=read_cycle();
  const uint64_t spike_calls=20;
  for (uint64_t i=0;!rc && i<spike_calls;++i)
    rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data);
  uint64_t spike_cycles=read_cycle()-begin_cycle;
#else
  typedef struct { volatile uint64_t completed; volatile int rc; } child_report_t;
  child_report_t *child_report=(child_report_t *)mmap(
    NULL,sizeof(*child_report),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
  if (child_report == MAP_FAILED) return 90;
  memset(child_report,0,sizeof(*child_report));
  pid_t child=fork();
  if (child < 0) return 90;
  if (child == 0) {
    int null_fd=open("/dev/null",O_WRONLY);
    if (null_fd >= 0) { dup2(null_fd,STDOUT_FILENO); close(null_fd); }
    child_report->rc=merlin_capsule_run(&params,a.data,b.data,c.data,out.data);
    __sync_synchronize(); child_report->completed=UINT64_C(0x4d45524c494e4f4b);
    _exit(0);
  }
  int child_status=0;
  if (waitpid(child,&child_status,0) != child || !WIFEXITED(child_status) ||
      WEXITSTATUS(child_status) != 0 || child_report->completed != UINT64_C(0x4d45524c494e4f4b))
    child_report->rc=95;
  int rc=child_report->rc;
#endif
  int memory_ok=guards_ok(&a)&&guards_ok(&b)&&guards_ok(&c)&&guards_ok(&out);
#ifdef MERLIN_K1_LINUX
  memory_ok=memory_ok&&timed_memory_ok;
  compute_reference(&a,&b,&c,expected);
#else
  memory_ok=memory_ok&&memcmp(a.data,a.snapshot,a.bytes)==0&&
    memcmp(b.data,b.snapshot,b.bytes)==0&&memcmp(c.data,c.snapshot,c.bytes)==0;
#endif
  double max_abs=0.0; int numeric_ok=outputs_match(out.data,expected,&max_abs);
#ifdef MERLIN_K1_LINUX
  numeric_ok=numeric_ok&&timed_numeric_ok&&audit_output_changed&&worker_dependence&&
    audit_balanced_shards;
#endif
  /* K1 emits its complete trusted audit counters even on rejection; the receipt remains success
   * only.  Other targets retain the compact fail-fast path. */
#ifndef MERLIN_K1_LINUX
  if (rc || !memory_ok || !numeric_ok) {
#ifdef MERLIN_FREESTANDING
    htif_puts("FAIL rc="); htif_putd(rc); htif_puts(" memory="); htif_putd(memory_ok);
    htif_puts(" numeric="); htif_putd(numeric_ok); htif_putc('\n'); htif_exit(1);
#else
    fprintf(stderr,"FAIL rc=%d memory=%d numeric=%d max_abs=%.9g\n",rc,memory_ok,numeric_ok,max_abs);
    return rc ? 92 : (!memory_ok ? 93 : 94);
#endif
  }
#endif
#ifdef MERLIN_FREESTANDING
  htif_puts("MERLIN_TRUSTED_RESULT version=1 seed="); htif_putd((long)receipt_seed);
  htif_puts(" nonce="); htif_putd((long)MERLIN_RECEIPT_NONCE); htif_puts(" vlenb=");
  htif_putd((long)observed_vlenb); htif_puts(" cycles="); htif_putd((long)spike_cycles);
  htif_puts(" calls="); htif_putd((long)spike_calls); htif_putc('\n'); htif_exit(0);
#else
  #ifdef MERLIN_K1_LINUX
  cpu_set_t affinity; CPU_ZERO(&affinity); sched_getaffinity(0,sizeof(affinity),&affinity);
  struct rusage usage; getrusage(RUSAGE_SELF,&usage);
  printf("K1_METRIC vlenb %llu\n",(unsigned long long)vlenb);
  printf("K1_METRIC affinity_count %d\n",CPU_COUNT(&affinity));
  printf("K1_METRIC wall_ns %llu\n",(unsigned long long)elapsed);
  printf("K1_METRIC time_ticks %llu\n",(unsigned long long)end_time);
  printf("K1_METRIC calls %llu\n",(unsigned long long)calls);
  printf("K1_METRIC audit_call %llu\n",(unsigned long long)audit_call);
  printf("K1_METRIC audit_wall_ns %llu\n",(unsigned long long)audit_wall_ns);
  printf("K1_METRIC audit_time_ticks %llu\n",(unsigned long long)audit_time_ticks);
  printf("K1_METRIC correctness_checks %llu\n",(unsigned long long)correctness_checks);
  printf("K1_METRIC pinned_hart_mask %llu\n",(unsigned long long)pinned_hart_mask);
  printf("K1_METRIC worker_hart_mask %llu\n",(unsigned long long)worker_hart_mask);
  printf("K1_METRIC productive_worker_hart_mask %llu\n",(unsigned long long)productive_worker_mask);
  printf("K1_METRIC pthread_create_attempts %llu\n",(unsigned long long)pthread_attempts);
  printf("K1_METRIC pthread_creates %llu\n",(unsigned long long)pthread_creates);
  printf("K1_METRIC pthread_create_failures %llu\n",(unsigned long long)pthread_create_failures);
  printf("K1_METRIC pthread_completions %llu\n",(unsigned long long)pthread_completions);
  printf("K1_METRIC pthread_affinity_attempts %llu\n",(unsigned long long)affinity_attempts);
  printf("K1_METRIC pthread_affinity_successes %llu\n",(unsigned long long)affinity_successes);
  printf("K1_METRIC pthread_affinity_failures %llu\n",(unsigned long long)affinity_failures);
  printf("K1_METRIC minimum_worker_cpu_ns %llu\n",(unsigned long long)minimum_worker_cpu_ns);
  printf("K1_METRIC counterfactual_create_attempts %llu\n",(unsigned long long)counterfactual_attempts);
  printf("K1_METRIC counterfactual_creates %llu\n",(unsigned long long)counterfactual_creates);
  printf("K1_METRIC counterfactual_create_failures %llu\n",(unsigned long long)counterfactual_failures);
  printf("K1_METRIC counterfactual_suppressed_starts %llu\n",(unsigned long long)counterfactual_suppressed);
  printf("K1_METRIC counterfactual_worker_dependence %d\n",worker_dependence);
  printf("K1_METRIC audit_serialized_callbacks %llu\n",(unsigned long long)audit_serialized);
  printf("K1_METRIC audit_output_elements %llu\n",(unsigned long long)audit_elements);
  printf("K1_METRIC audit_output_coverage %llu\n",(unsigned long long)audit_coverage);
  printf("K1_METRIC audit_owner_min_elements %llu\n",(unsigned long long)audit_owner_min);
  printf("K1_METRIC audit_owner_max_elements %llu\n",(unsigned long long)audit_owner_max);
  printf("K1_METRIC audit_ownership_violations %llu\n",(unsigned long long)audit_owner_violations);
  printf("K1_METRIC audit_balanced_shards %d\n",audit_balanced_shards);
  printf("K1_METRIC peak_rss_kb %ld\n",usage.ru_maxrss);
  if (rc || !memory_ok || !numeric_ok) {
    fprintf(stderr,"FAIL rc=%d memory=%d numeric=%d worker_dependence=%d audit_shards=%d max_abs=%.9g\n",
            rc,memory_ok,numeric_ok,worker_dependence,audit_balanced_shards,max_abs);
    return rc ? 92 : (!memory_ok ? 93 : 94);
  }
  #endif
  printf("MERLIN_TRUSTED_RESULT version=1 seed=%llu nonce=%llu memory=1 numeric=1\n",
         (unsigned long long)receipt_seed,(unsigned long long)MERLIN_RECEIPT_NONCE);
  return 0;
#endif
}
