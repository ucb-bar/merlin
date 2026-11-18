"""Generated C harnesses for whole-model runs on a remote Linux board.

:func:`main_linux_c` and :func:`main_linux_session_c` emit the OUT/ARGMAX/SUM/METRIC/DONE console
protocol the bare-metal harness uses, so one console parser grades every substrate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from merlin.runtime.backends.linux.remote_session import Profile, binary_name, profile

# ---- linux_harness_source ----------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.runtime.boards.catalog import Board


# ---- generated Linux harness --------------------------------------------------------


def main_linux_c(
    board: Board | Profile,
    dump_cap: int | None = 4096,
    mmap_weights: bool = False,
    dispatch_timing: bool = False,
    op_profile: bool = False,
    persistent_openmp_pool: int | None = None,
) -> str:
    """Generate the board's Linux ``main()``: build the memref descriptors from the generated arg
    table, call ``merlin_run`` once, time it with the ``time`` CSR, and print the SAME
    OUT/ARGMAX/SUM/METRIC/DONE markers the spike harness uses so ``zephyr_execution._parse_console``
    consumes the console unchanged. Glibc-hosted: stdio, malloc, no HTIF/medany.

    ``mmap_weights=True``: instead of embedding the weight blob in the binary (``ld -r -b
    binary`` -> ``_binary_weights_bin_start``), ``mmap`` the blob read-only from a file path
    (``MERLIN_WEIGHTS`` env or argv[1]). The OS then demand-pages the weights, so resident RAM
    is the working set, not the whole blob — required for the big VLAs (groot 1.8G, pi05 ~5G)
    whose embedded binary won't fit a RAM-backed scratch directory and whose all-resident weights
    can exceed the board's memory.

    ``dump_cap`` bounds how many output elements the console prints. The host gate compares the
    printed PREFIX against the leading elements of the reference, so this number is the ceiling on
    how much of the answer any accuracy verdict can actually cover: at the historical 4096 a
    256,000-element LM head was graded on 1.6% of its output, and the resulting cos read exactly
    like a whole-tensor score. ``dump_cap=None`` removes the ceiling by deriving it from the
    model's OWN generated ``MERLIN_OUT_ELEMS`` rather than a host-side literal, so the console
    carries the whole tensor and ``_gate`` reports ``comparison_complete``. The default stays 4096
    so an unchanged caller emits byte-identical C."""
    brd = profile(board).board
    timebase_hz, cpu_hz, stack_bytes, banner = brd.timebase_hz, brd.chip_freq_hz, brd.stack_bytes, binary_name(brd)
    title = brd.name.upper()
    weights_decl = (
        "/* weights mmap'd from a file at run time (see main). */\nstatic const void *MERLIN_WEIGHTS_PTR;\n"
        if mmap_weights
        else "/* weights.bin linked in as a binary blob (ld -r -b binary): glibc Linux is\n"
        " * small-code-model and fully relocatable, so address the blob via its symbol. */\n"
        "extern const unsigned char _binary_weights_bin_start[];\n"
        "static const void *MERLIN_WEIGHTS_PTR;\n"
    )
    mmap_includes = (
        "#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n#include <stdlib.h>\n"
        if mmap_weights
        else ""
    )
    if mmap_weights:
        weights_init = """  /* mmap the weight blob read-only: the loader demand-pages it, so resident RAM is the
   * working set, not the whole blob (the big-VLA fit lever). */
  const char *wpath = getenv("MERLIN_WEIGHTS");
  if (!wpath && argc > 1) wpath = argv[1];
  if (!wpath) {{ fprintf(stderr, "FAIL no MERLIN_WEIGHTS path\\n"); return 2; }}
  int wfd = open(wpath, O_RDONLY);
  if (wfd < 0) {{ fprintf(stderr, "FAIL open weights %s\\n", wpath); return 2; }}
  struct stat wst; fstat(wfd, &wst);
  void *wmap = mmap(NULL, (size_t)wst.st_size, PROT_READ, MAP_PRIVATE, wfd, 0);
  if (wmap == MAP_FAILED) {{ fprintf(stderr, "FAIL mmap weights\\n"); return 2; }}
  MERLIN_WEIGHTS_PTR = wmap;"""
    else:
        weights_init = "  MERLIN_WEIGHTS_PTR = (const void *)_binary_weights_bin_start;"
    # Per-dispatch matmul-bucket timing (default-OFF). Only emitted when the routed-matmul shim
    # was built with -DMERLIN_DISPATCH_TIMING; then the harness reads the accumulated GEMM-ukernel
    # ticks/calls and prints them as extra METRIC lines. With dispatch_timing=False the harness is
    # byte-identical to the un-instrumented path.
    dtiming_decl = (
        "extern unsigned long long merlin_matmul_ticks(void);\nextern unsigned long long merlin_matmul_calls(void);\n"
        if dispatch_timing
        else ""
    )
    dtiming_print = (
        '  printf("METRIC matmul_ticks %llu\\n", (unsigned long long)merlin_matmul_ticks());\n'
        '  printf("METRIC matmul_calls %llu\\n", (unsigned long long)merlin_matmul_calls());\n'
        if dispatch_timing
        else ""
    )
    # Per-op whole-model profile (default-OFF). Emitted only when the IR was instrumented with
    # `@merlin_prof_mark` (llvmlower.op_profile) and the shim object is linked; with
    # op_profile=False the harness is byte-identical to the un-instrumented path.
    oprof_decl = "extern void merlin_prof_dump(void);\n" if op_profile else ""
    oprof_print = "  merlin_prof_dump();\n" if op_profile else ""
    if persistent_openmp_pool is not None and int(persistent_openmp_pool) < 1:
        raise ValueError("persistent_openmp_pool must be positive when provided")
    openmp_include = '#include "libomp_pthread.h"\n' if persistent_openmp_pool is not None else ""
    openmp_census = (
        f"""  int omp_threads = merlin_omp_init_from_env({int(persistent_openmp_pool)});
  printf("METRIC omp_threads %d\\n", omp_threads);
  printf("METRIC omp_worker_cpus");
  for (int tid = 0; tid < omp_threads; ++tid)
    printf(" %d", merlin_omp_worker_cpu(tid));
  printf("\\n");
"""
        if persistent_openmp_pool is not None
        else ""
    )
    # The cap is a TOKEN, not a number: `None` renders the model's own generated element count, so
    # "print everything" is derived from the target's header instead of a host-side guess that a
    # bigger model would silently outgrow. A non-positive cap would print an empty OUT line that
    # parses fine and grades as zero coverage -- refuse it rather than emit it.
    if dump_cap is None:
        dump_cap_token = "MERLIN_OUT_ELEMS"
    else:
        if int(dump_cap) < 1:
            raise ValueError("dump_cap must be positive, or None for the model's full output")
        dump_cap_token = str(int(dump_cap))
    return f"""/* Generated by merlin.runtime.backends.linux.remote_session — {title} Linux RVV runtime harness. Do not edit. */
#define _GNU_SOURCE
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
{mmap_includes}
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"
{openmp_include}

{weights_decl}
{dtiming_decl}{oprof_decl}#define MERLIN_DUMP_CAP {dump_cap_token}
/* Timebase->CPU frequency ratio for the estimated core-cycle count (see module doc): this
 * kernel traps userspace `rdcycle`, so we time with the delegated `rdtime` counter. */
#define MERLIN_TIMEBASE_HZ {timebase_hz}ULL
#define MERLIN_CPU_HZ {cpu_hz}ULL

static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

static inline uint64_t rd_time(void) {{
  uint64_t t;
  __asm__ volatile("rdtime %0" : "=r"(t));
  return t;
}}

static inline uint64_t rd_vlenb(void) {{
  uint64_t v;
  __asm__ volatile("csrr %0, vlenb" : "=r"(v));
  return v;
}}

static uint64_t wall_ns(void) {{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}}

static int merlin_one_step(long step, int validate_session) {{
  merlin_prepare_step(step);
  merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
                   MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
  if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                          MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                          MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0)
    return -1;
#endif
  if (validate_session) merlin_validate_step(step);
  return 0;
}}

static int merlin_one_session(int validate_session) {{
  merlin_reset_session();
  for (long step = 0; step < MERLIN_SESSION_STEPS; step++)
    if (merlin_one_step(step, validate_session) != 0) return -1;
  return 0;
}}

static void *worker(void *arg) {{
  (void)arg;
  printf("=== {banner} vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {{
    fprintf(stderr, "FAIL sched_getaffinity\\n"); return NULL;
  }}
  printf("METRIC affinity_cpus %d\\n", CPU_COUNT(&allowed));
{openmp_census}

  /* Diagnostic sustained mode uses MERLIN_WARMUP + MERLIN_ITERS for individual transitions.
   * Paper mode uses MERLIN_SESSION_WARMUPS + MERLIN_SESSION_REPEATS: every timing sample then
   * contains every semantic observation from reset state. Observation count and statistical
   * repeat count are deliberately distinct.
   *
   * Transition passes use the SAME buffers -- which is the point: a per-iteration cost that creeps
   * (arena growth,
   * allocator churn) is invisible in a single run, and a single run's cold caches make the one
   * number you quote either optimistic or pessimistic depending on which you pick. Defaults
   * (1 timed, 0 warmup) reproduce the historical single-shot console byte-for-byte. Each timed
   * pass emits `METRIC iter_cycles <i> <cycles>`, the same protocol the Zephyr harness uses, so
   * _parse_console reads both without a special case. */
  long n_iters = 1, n_warmup = 0, n_session_repeats = 0, n_session_warmups = 0;
  int validate_session = 0;
  {{
    const char *e = getenv("MERLIN_ITERS");
    if (e && *e) {{ n_iters = strtol(e, NULL, 10); if (n_iters < 1) n_iters = 1; }}
    e = getenv("MERLIN_WARMUP");
    if (e && *e) {{ n_warmup = strtol(e, NULL, 10); if (n_warmup < 0) n_warmup = 0; }}
    e = getenv("MERLIN_SESSION_REPEATS");
    if (e && *e) {{
      n_session_repeats = strtol(e, NULL, 10);
      if (n_session_repeats < 1) n_session_repeats = 1;
    }}
    e = getenv("MERLIN_SESSION_WARMUPS");
    if (e && *e) {{
      n_session_warmups = strtol(e, NULL, 10);
      if (n_session_warmups < 0) n_session_warmups = 0;
    }}
    e = getenv("MERLIN_VALIDATE_SESSION");
    if (e && *e && strcmp(e, "0") != 0) validate_session = 1;
  }}
  uint64_t w0, w1, t0, t1;
  if (n_session_repeats > 0) {{
    for (long repeat = 0; repeat < n_session_warmups; repeat++)
      if (merlin_one_session(0) != 0) return NULL;
    w0 = wall_ns(); t0 = rd_time();
    for (long repeat = 0; repeat < n_session_repeats; repeat++) {{
      uint64_t iw0 = wall_ns(), it0 = rd_time();
      if (merlin_one_session(validate_session && repeat == 0) != 0) return NULL;
      uint64_t it1 = rd_time(), iw1 = wall_ns();
      printf("METRIC iter_cycles %ld %llu\\n", repeat,
             (unsigned long long)((it1 - it0) * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
      printf("METRIC iter_wall_ns %ld %llu\\n", repeat, (unsigned long long)(iw1 - iw0));
    }}
    t1 = rd_time(); w1 = wall_ns();
    t1 = t0 + (t1 - t0) / (uint64_t)n_session_repeats;
    w1 = w0 + (w1 - w0) / (uint64_t)n_session_repeats;
    goto session_complete;
  }}
  merlin_reset_session();
  for (long i = 0; i < n_warmup; i++) {{
    merlin_prepare_step(i);
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
    if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                            MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                            MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {{
      fprintf(stderr, "FAIL state ABI mismatch during warmup\n"); return NULL;
    }}
#endif
  }}

  /* Warmup may advance KV/recurrent state. Reset before the measured trajectory so cache warming
   * never changes the semantic input to the reported session. */
  merlin_reset_session();
  w0 = wall_ns();
  t0 = rd_time();
  for (long i = 0; i < n_iters; i++) {{
    uint64_t iw0 = wall_ns();
    uint64_t it0 = rd_time();
    merlin_prepare_step(i);
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
    if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                            MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                            MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {{
      fprintf(stderr, "FAIL state ABI mismatch during timed session\n"); return NULL;
    }}
#endif
    if (validate_session) merlin_validate_step(i);
    uint64_t it1 = rd_time();
    uint64_t iw1 = wall_ns();
    if (n_iters > 1) {{
      printf("METRIC iter_cycles %ld %llu\\n", i,
             (unsigned long long)((it1 - it0) * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
      printf("METRIC iter_wall_ns %ld %llu\\n", i, (unsigned long long)(iw1 - iw0));
    }}
  }}
  t1 = rd_time();
  w1 = wall_ns();
  if (n_iters > 1) {{                 /* report the PER-ITERATION cost, not the batch total */
    t1 = t0 + (t1 - t0) / (uint64_t)n_iters;
    w1 = w0 + (w1 - w0) / (uint64_t)n_iters;
  }}

session_complete: ;
  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  const char *output_path = getenv("MERLIN_OUTPUT_FILE");
  if (output_path && *output_path) {{
    FILE *output_file = fopen(output_path, "wb");
    if (!output_file) {{ fprintf(stderr, "FAIL open output file %s\\n", output_path); return NULL; }}
    size_t written = fwrite(MERLIN_OUTPUT_PTR[0], sizeof(float), MERLIN_OUT_ELEMS, output_file);
    if (fclose(output_file) != 0 || written != MERLIN_OUT_ELEMS) {{
      fprintf(stderr, "FAIL write output file %s\\n", output_path); return NULL;
    }}
    /* Preserve the parser framing without streaming a multi-megabyte line through SSH. */
    printf("OUT 0\\n");
    printf("METRIC output_file_elems %d\\n", MERLIN_OUT_ELEMS);
  }} else {{
    printf("OUT %d", k);
    for (int i = 0; i < k; i++) {{
      uint32_t bits;
      memcpy(&bits, &((float*)MERLIN_OUTPUT_PTR[0])[i], 4);
      printf(" %u", (unsigned)bits);
    }}
    printf("\\n");
  }}

  if ((!output_path || !*output_path) && MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {{
    int rows = MERLIN_OUT_ELEMS / MERLIN_OUT_LASTDIM;
    printf("ARGMAX %d", rows);
    for (int r = 0; r < rows; r++) {{
      const float *row = &((float*)MERLIN_OUTPUT_PTR[0])[(long)r * MERLIN_OUT_LASTDIM];
      int best = 0; float bv = row[0];
      for (int j = 1; j < MERLIN_OUT_LASTDIM; j++)
        if (row[j] > bv) {{ bv = row[j]; best = j; }}
      printf(" %d", best);
    }}
    printf("\\n");
    float s = 0.0f;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++) s += ((float*)MERLIN_OUTPUT_PTR[0])[i];
    uint32_t sb; memcpy(&sb, &s, 4);
    printf("SUM %u\\n", (unsigned)sb);
  }}
  uint64_t ticks = t1 - t0;
  /* estimated core cycles = timebase ticks * (cpu_hz / timebase_hz). The METRIC cycles line is
   * what _parse_console reads; the raw timebase ticks + wall ns are emitted alongside as the
   * ground truth (the cycles value is an estimate — see MERLIN_CPU_HZ/MERLIN_TIMEBASE_HZ). */
  uint64_t est_cycles = ticks * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ);
  printf("METRIC cycles %llu\\n", (unsigned long long)est_cycles);
  printf("METRIC time_ticks %llu\\n", (unsigned long long)ticks);
  printf("METRIC wall_ns %llu\\n", (unsigned long long)(w1 - w0));
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
  printf("METRIC peak_rss_kb %llu\\n", (unsigned long long)usage.ru_maxrss);
#if MERLIN_HAS_SESSION_CORRECTNESS
  if (validate_session) {{
    printf("METRIC correctness_trajectory_steps %ld\\n", merlin_correctness_steps());
    printf("METRIC correctness_trajectory_min_cos_ppm %ld\\n", merlin_correctness_min_cos_ppm());
    printf("METRIC correctness_trajectory_max_rel_ppm %ld\\n", merlin_correctness_max_rel_ppm());
    printf("METRIC correctness_trajectory_top1_matches %ld\\n", merlin_correctness_top1());
  }}
#endif
#if MERLIN_HAS_SESSION_QUALITY
  if (validate_session) {{
    printf("METRIC trajectory_steps %ld\\n", merlin_quality_steps());
    printf("METRIC trajectory_min_cos_ppm %ld\\n", merlin_quality_min_cos_ppm());
    printf("METRIC trajectory_max_rel_ppm %ld\\n", merlin_quality_max_rel_ppm());
    printf("METRIC trajectory_top1_matches %ld\\n", merlin_quality_top1());
  }}
#endif
{dtiming_print}{oprof_print}  printf("DONE\\n");
  fflush(stdout);
  return NULL;
}}

int main(int argc, char **argv) {{
  (void)argc; (void)argv;
{weights_init}
  /* The lowered model alloca's large intermediate buffers; the default 8MB stack overflows
   * (SIGSEGV) for VLAs. Run merlin_run on a pthread with an explicitly-mmap'd {stack_bytes}-byte
   * stack (one contiguous region — robust vs main-thread growth, which collides with lower mmaps).
   * Also raise RLIMIT_STACK as a fallback if pthread creation fails. */
  struct rlimit rl = {{ {stack_bytes}ULL, {stack_bytes}ULL }};
  setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr;
  pthread_t th;
  if (pthread_attr_init(&attr) == 0 &&
      pthread_attr_setstacksize(&attr, (size_t){stack_bytes}ULL) == 0 &&
      pthread_create(&th, &attr, worker, NULL) == 0) {{
    pthread_join(th, NULL);
  }} else {{
    worker(NULL);   /* fallback: run inline (relies on the raised RLIMIT) */
  }}
  return 0;
}}
"""


def main_linux_session_c(
    board: Board | Profile, dump_cap: int | None = 4096, persistent_openmp_pool: int | None = None
) -> str:
    """Board harness for a generated multi-program session scheduler.

    Every ``iter_wall_ns`` sample covers the full ordered stage graph from reset state. Per-stage
    totals are emitted as diagnostic attribution and are never substituted for that headline total.

    ``dump_cap`` bounds the printed quality output exactly as in :func:`main_linux_c`; ``None``
    prints all of it. The session's element count is a RUN-TIME value here (the scheduler reports
    it), not a compile-time macro, so the uncapped form drops the ceiling from the comparison
    rather than defining a macro over a local variable.
    """
    brd = profile(board).board
    timebase_hz, cpu_hz, stack_bytes, banner = brd.timebase_hz, brd.chip_freq_hz, brd.stack_bytes, binary_name(brd)
    title = brd.name.upper()
    if dump_cap is None:
        cap_define, k_expr = "", "(int)elems"
    else:
        if int(dump_cap) < 1:
            raise ValueError("dump_cap must be positive, or None for the full session output")
        cap_define = f"#define MERLIN_DUMP_CAP {int(dump_cap)}\n"
        k_expr = "elems < MERLIN_DUMP_CAP ? (int)elems : MERLIN_DUMP_CAP"
    if persistent_openmp_pool is not None and int(persistent_openmp_pool) < 1:
        raise ValueError("persistent_openmp_pool must be positive when provided")
    openmp_include = '#include "libomp_pthread.h"\n' if persistent_openmp_pool is not None else ""
    openmp_census = (
        f"""  int omp_threads = merlin_omp_init_from_env({int(persistent_openmp_pool)});
  printf("METRIC omp_threads %d\\n", omp_threads);
  printf("METRIC omp_worker_cpus");
  for (int tid = 0; tid < omp_threads; ++tid)
    printf(" %d", merlin_omp_worker_cpu(tid));
  printf("\\n");
"""
        if persistent_openmp_pool is not None
        else ""
    )
    return f"""/* Generated {title} multi-program continuous-session harness. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "merlin_session.h"
{openmp_include}
{cap_define}#define MERLIN_TIMEBASE_HZ {timebase_hz}ULL
#define MERLIN_CPU_HZ {cpu_hz}ULL
static const void *WEIGHTS[MERLIN_SESSION_N_PROGRAMS];
static inline uint64_t rd_time(void) {{ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }}
static inline uint64_t rd_vlenb(void) {{ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }}
static uint64_t wall_ns(void) {{
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}}
static int run_session(int validate, uint64_t *stage_ns) {{
  merlin_session_reset();
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; program++) {{
    if (merlin_session_prepare_program(program) != 0) return -1;
    uint64_t start = wall_ns();
    long steps = merlin_session_program_steps(program);
    for (long step = 0; step < steps; step++)
      if (merlin_session_run_step(program, WEIGHTS[program], step, validate) != 0) return -1;
    if (stage_ns) stage_ns[program] = wall_ns() - start;
  }}
  return 0;
}}
static void *worker(void *unused) {{
  (void)unused;
  printf("=== {banner}_session vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {{
    fprintf(stderr, "FAIL sched_getaffinity\\n"); return 0;
  }}
  printf("METRIC affinity_cpus %d\\n", CPU_COUNT(&allowed));
{openmp_census}
  long repeats = 1, warmups = 0; int validate = 0;
  const char *e = getenv("MERLIN_SESSION_REPEATS");
  if (e && *e) {{ repeats = strtol(e, 0, 10); if (repeats < 1) repeats = 1; }}
  e = getenv("MERLIN_SESSION_WARMUPS");
  if (e && *e) {{ warmups = strtol(e, 0, 10); if (warmups < 0) warmups = 0; }}
  e = getenv("MERLIN_VALIDATE_SESSION");
  if (e && *e && strcmp(e, "0") != 0) validate = 1;
  for (long repeat = 0; repeat < warmups; repeat++) if (run_session(0, 0) != 0) return 0;
  uint64_t all_w0 = wall_ns(), all_t0 = rd_time();
  for (long repeat = 0; repeat < repeats; repeat++) {{
    uint64_t stage_ns[MERLIN_SESSION_N_PROGRAMS];
    uint64_t w0 = wall_ns(), t0 = rd_time();
    if (run_session(validate && repeat == 0, stage_ns) != 0) return 0;
    uint64_t t1 = rd_time(), w1 = wall_ns();
    printf("METRIC iter_cycles %ld %llu\\n", repeat,
           (unsigned long long)((t1 - t0) * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
    printf("METRIC iter_wall_ns %ld %llu\\n", repeat, (unsigned long long)(w1 - w0));
    for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; program++)
      printf("STAGE %ld %s %llu\\n", repeat, merlin_session_program_name(program),
             (unsigned long long)stage_ns[program]);
  }}
  uint64_t all_t1 = rd_time(), all_w1 = wall_ns();
  uint64_t ticks = (all_t1 - all_t0) / (uint64_t)repeats;
  uint64_t elapsed = (all_w1 - all_w0) / (uint64_t)repeats;
  size_t elems = merlin_session_quality_output_elems();
  const float *output = (const float *)merlin_session_quality_output();
  int k = {k_expr};
  printf("OUT %d", k);
  for (int i = 0; i < k; i++) {{ uint32_t bits; memcpy(&bits, &output[i], 4); printf(" %u", bits); }}
  printf("\\n");
  printf("METRIC cycles %llu\\n", (unsigned long long)(ticks * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
  printf("METRIC time_ticks %llu\\n", (unsigned long long)ticks);
  printf("METRIC wall_ns %llu\\n", (unsigned long long)elapsed);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("METRIC peak_rss_kb %llu\\n", (unsigned long long)usage.ru_maxrss);
  if (validate) {{
    printf("METRIC correctness_trajectory_steps %ld\\n", merlin_session_correctness_steps());
    printf("METRIC correctness_trajectory_min_cos_ppm %ld\\n",
           merlin_session_correctness_min_cos_ppm());
    printf("METRIC correctness_trajectory_max_rel_ppm %ld\\n",
           merlin_session_correctness_max_rel_ppm());
    printf("METRIC correctness_trajectory_top1_matches %ld\\n",
           merlin_session_correctness_top1());
    printf("METRIC trajectory_steps %ld\\n", merlin_session_quality_steps());
    printf("METRIC trajectory_min_cos_ppm %ld\\n", merlin_session_quality_min_cos_ppm());
    printf("METRIC trajectory_max_rel_ppm %ld\\n", merlin_session_quality_max_rel_ppm());
    printf("METRIC trajectory_top1_matches %ld\\n", merlin_session_quality_top1());
  }}
  printf("DONE\\n"); fflush(stdout); return 0;
}}
int main(void) {{
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; program++) {{
    char key[64]; snprintf(key, sizeof(key), "MERLIN_WEIGHTS_%d", program);
    const char *path = getenv(key); if (!path) {{ fprintf(stderr, "FAIL %s unset\\n", key); return 2; }}
    int fd = open(path, O_RDONLY); if (fd < 0) return 2;
    struct stat st; if (fstat(fd, &st) != 0) return 2;
    if (st.st_size == 0) {{ close(fd); WEIGHTS[program] = (const void *)1; continue; }}
    void *mapped = mmap(0, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (mapped == MAP_FAILED) return 2; WEIGHTS[program] = mapped;
  }}
  struct rlimit rl = {{ {stack_bytes}ULL, {stack_bytes}ULL }}; setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr; pthread_t thread;
  if (pthread_attr_init(&attr) == 0 &&
      pthread_attr_setstacksize(&attr, (size_t){stack_bytes}ULL) == 0 &&
      pthread_create(&thread, &attr, worker, 0) == 0) pthread_join(thread, 0);
  else worker(0);
  return 0;
}}
"""
