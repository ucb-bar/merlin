/* board-support TinyLlama v2 host. Projection and attention launches share one
 * ordered launch plan, one raw-loader invocation sequence, and whole-arena exact
 * checks. TINYLLAMA_STOP_AFTER provides the existing launch-prefix stop mode. */
#include "tiny_full_model.h"

#include "tiny_full_projection.h"
#include "tiny_full_assets.h"
#include "atlas_host.h"
#include "tiny_attention.h"
#include "chip_config.h"
#ifndef ATLAS_FAST_MODE
#define ATLAS_FAST_MODE 0
#endif
#if ATLAS_FAST_MODE
#include "fast-assets.h"
static atlas_host_buffer fast_buffers[2];
#endif
#if (defined(TINY_FULL_FUSIONS) && TINY_FULL_FUSIONS) || (defined(TINY_FULL_ATTENTION_V2) && TINY_FULL_ATTENTION_V2)
#define TINY_FULL_EXTENT_CHECK 1
#else
#define TINY_FULL_EXTENT_CHECK 0
#endif


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TINYLLAMA_STOP_AFTER
#define TINYLLAMA_STOP_AFTER 0
#endif
#ifndef TINYLLAMA_QUIET
#define TINYLLAMA_QUIET 0
#endif
#ifndef TINYLLAMA_MAX_POLLS
#define TINYLLAMA_MAX_POLLS 200000000u
#endif

__attribute__((aligned(64))) volatile uint64_t tiny_full_mailbox[16];
atlas_host tiny_full_host;
static uint64_t workspace[24 * 1024 * 1024 / 8];
static uint8_t before[TINY_FULL_DRAM_BYTES];
#if TINY_FULL_LAYER_COUNT == TINY_FULL_LAYERS
static uint16_t all_logits[128 * 32000];
#endif

#ifndef PENGUIN26_UART_BAUDRATE
#define PENGUIN26_UART_BAUDRATE 115200
#endif

extern tiny_full_projection_context tiny_full_projection_state;

static const char *const role_names[8] = {"q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj", "lm_head"};

typedef struct {
  uint64_t model, copy, load, wait, check, instret;
  uint64_t atlas_cycles, atlas_instret;
} launch_cycles;

static launch_cycles total;
static uint64_t mark_cycle, mark_instret, wall_begin;
static int failed;

static inline uint64_t rdcycle(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
static inline uint64_t rdinstret(void) { uint64_t c; __asm__ volatile("rdinstret %0" : "=r"(c)); return c; }
static inline uint32_t atlas_csr(uint32_t offset) {
  return atlas_host_direct_read32(0, (uintptr_t)ATLAS_CSR_BASE + offset);
}

static void finish(int pass, const char *check, uint64_t got, uint64_t want) {
  uint64_t launches = tiny_full_mailbox[2];
  if (!pass)
    printf("TINYLLAMA FAIL check=%s got=%lu expected=%lu launches_completed=%lu\n", check,
           (unsigned long)got, (unsigned long)want, (unsigned long)launches);
  printf("TOTAL model=tinyllama launches=%lu failed=%u bad=%u host_cycles=%lu load_cycles=%lu wait_cycles=%lu "
         "atlas_cycles=%lu atlas_instret=%lu image_loads=%lu wall_cycles=%lu model_cycles=%lu copy_cycles=%lu "
         "check_cycles=%lu host_instret=%lu\n",
         (unsigned long)launches, pass ? 0u : 1u, failed ? 1u : 0u,
         (unsigned long)(total.copy + total.load + total.wait + total.check), (unsigned long)total.load,
         (unsigned long)total.wait, (unsigned long)total.atlas_cycles, (unsigned long)total.atlas_instret,
         (unsigned long)launches, (unsigned long)(rdcycle() - wall_begin), (unsigned long)total.model,
         (unsigned long)total.copy, (unsigned long)total.check, (unsigned long)total.instret);
  printf(pass ? "*** PASSED ***\n" : "*** FAILED ***\n");
  exit(pass ? 0 : 1);
}

/* Byte offset of the first difference, or SIZE_MAX. memcmp is the fast path. */
static size_t first_difference(const volatile uint8_t *actual, const uint8_t *wanted, size_t bytes) {
  if (memcmp((const void *)actual, wanted, bytes) == 0) return SIZE_MAX;
  for (size_t i = 0; i < bytes; ++i)
    if (actual[i] != wanted[i]) return i;
  return SIZE_MAX;
}

/* One line per launch. host_cycles = copy + load + wait + check (this launch's device interaction);
 * model_cycles = host model code and operand staging since the previous launch. */
static void report(ptrdiff_t index, const tiny_panel *panel, unsigned kind, int status, unsigned bad, const launch_cycles *c) {
  size_t call = tiny_full_projection_state.calls, site = call % 155;
  unsigned step = (unsigned)(call / 155), layer = site == 154 ? 22u : (unsigned)(site / 7);
  printf("LAUNCH model=tinyllama k=%ld name=s%u.l%u.%s.c%lu status=%d bad=%u loaded=1 host_cycles=%lu "
         "load_cycles=%lu wait_cycles=%lu atlas_cycles=%lu atlas_instret=%lu model_cycles=%lu copy_cycles=%lu "
         "check_cycles=%lu host_instret=%lu shape=%lux%lux%lu words=%lu\n",
         (long)index, step, layer, kind == 2 ? (panel->logical ? "post_o_norm" : "pre_attention_norm") : kind == 1 ? "attention" : role_names[panel->logical & 7], (unsigned long)panel->column, status, bad,
         (unsigned long)(c->copy + c->load + c->wait + c->check), (unsigned long)c->load, (unsigned long)c->wait,
         (unsigned long)c->atlas_cycles, (unsigned long)c->atlas_instret, (unsigned long)c->model,
         (unsigned long)c->copy, (unsigned long)c->check, (unsigned long)c->instret, (unsigned long)panel->rows,
         (unsigned long)panel->k, (unsigned long)panel->n,
#if ATLAS_FAST_MODE
         (unsigned long)(kind ? panel->program_words : tiny_fast_panels[tiny_full_launch_plan[index].index].program_words));
#else
         (unsigned long)panel->program_words);
#endif
}

static int transfer(const tiny_panel *panel, const uint8_t *expected, size_t table_index, unsigned kind,
                    uint8_t *dram, size_t bytes) {
  size_t index = tiny_full_mailbox[2];
#if ATLAS_FAST_MODE
  if (tiny_full_host.active || kind > 2) return 1;
#endif
#if TINY_FULL_EXTENT_CHECK
  const tiny_fusion_panel *fusion = !kind && tiny_full_projection_state.fusions
      ? tiny_full_projection_state.fusions + table_index : NULL;
#if defined(TINY_FULL_ATTENTION_V2) && TINY_FULL_ATTENTION_V2
  const tiny_attention_v2_panel *attention_v2 = kind == 1 && tiny_full_projection_state.attention_v2
      ? tiny_full_projection_state.attention_v2 + table_index : NULL;
#else
  const tiny_attention_v2_panel *attention_v2 = NULL;
#endif
  size_t extent = attention_v2 ? attention_v2->output_extent_bytes
      : fusion ? fusion->output_extent_bytes : panel ? panel->store_bytes : 0;
  if (!panel || !expected || !dram || panel->output_offset > bytes || extent > bytes - panel->output_offset ||
#else
  if (!panel || !expected || !dram || panel->output_offset > bytes || panel->store_bytes > bytes - panel->output_offset ||
#endif
      index >= TINY_FULL_LAUNCHES || bytes > sizeof(before) ||
      tiny_full_launch_plan[index].kind != kind || tiny_full_launch_plan[index].index != table_index) return 1;
  launch_cycles c = {0};
  uint64_t t0 = rdcycle(), i0 = rdinstret();
  c.model = t0 - mark_cycle;
  tiny_full_mailbox[3] = (uint64_t)index;
#if !ATLAS_FAST_MODE
  memcpy(before, dram, bytes);
#endif
  uint64_t t1 = rdcycle();
  int status = 0;
#if ATLAS_FAST_MODE
  const tiny_fast_panel *fast = kind ? NULL : tiny_fast_panels + table_index;
  if (fast && (fast->encoded_weight_base != (uintptr_t)panel->weights ||
      ((uintptr_t)panel->weights >> 32) != ((uintptr_t)dram >> 32) ||
      ((uintptr_t)panel->weights & UINT32_MAX) + (uint64_t)panel->weight_bytes > UINT64_C(0x100000000))) status = 2;
  if (!status && atlas_host_load_raw(&tiny_full_host, fast ? fast->program : panel->program,
      fast ? fast->program_words : panel->program_words) != ATLAS_HOST_OK) status = 2;
  fast_buffers[0] = (atlas_host_buffer){(uintptr_t)dram, bytes, ATLAS_HOST_DEVICE_READ_WRITE};
  fast_buffers[1] = fast ? (atlas_host_buffer){(uintptr_t)panel->weights, panel->weight_bytes, ATLAS_HOST_DEVICE_READ} : (atlas_host_buffer){0};
#else
  if (atlas_host_load_raw(&tiny_full_host, panel->program, panel->program_words) != ATLAS_HOST_OK) status = 2;
  atlas_host_buffer buffer = {(uintptr_t)dram, bytes, ATLAS_HOST_DEVICE_READ_WRITE};
#endif
  uint64_t t2 = rdcycle();
  uint32_t a0 = atlas_csr(ATLAS_CSR_CYCLES), r0 = atlas_csr(ATLAS_CSR_RETIRED);
#if ATLAS_FAST_MODE
  if (!status && atlas_host_start(&tiny_full_host, fast_buffers, kind ? 1 : 2) != ATLAS_HOST_OK) status = 3;
#else
  if (!status && atlas_host_start(&tiny_full_host, &buffer, 1) != ATLAS_HOST_OK) status = 3;
#endif
  if (!status && atlas_host_wait(&tiny_full_host, TINYLLAMA_MAX_POLLS) != ATLAS_HOST_OK) status = 4;
  uint32_t a1 = atlas_csr(ATLAS_CSR_CYCLES), r1 = atlas_csr(ATLAS_CSR_RETIRED);
  uint64_t t3 = rdcycle();
  if (!status && (tiny_full_host.invocation_sequence != (uint32_t)index + 1 || tiny_full_host.active)) status = 5;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
#if ATLAS_FAST_MODE
  /* Expected blobs cover the complete extent, including every untouched hole. */
  const volatile uint8_t *actual = dram;
  size_t out = panel->output_offset, where = SIZE_MAX;
  if (!status) {
    where = first_difference(actual + out, expected, extent);
    if (where != SIZE_MAX) where += out;
  }
#else
  /* Whole-arena check: output extent against the reference, everything else unchanged. */
  const volatile uint8_t *actual = dram;
#if TINY_FULL_EXTENT_CHECK
  size_t out = panel->output_offset;
  size_t where = SIZE_MAX;
  if (!status) {
    if (attention_v2) {
      const tiny_attention_panel *attention = tiny_full_projection_state.attention_panels + table_index;
      if (tiny_full_attention_overlay_expected(attention, attention_v2, expected, before, bytes)) status = 7;
    } else if (fusion && fusion->role == 5) {
      for (size_t tr = 0; tr < panel->physical_rows / 32; ++tr) {
        size_t at = tr * 32 * panel->width + panel->column * 32;
        memcpy(before + out + at, expected + at, panel->n * 32);
      }
    } else memcpy(before + out, expected, extent);
    if (!status) where = first_difference(actual, before, bytes);
  }
#else
  size_t out = panel->output_offset, end = out + panel->store_bytes, bad;
  size_t where = SIZE_MAX;
  if (!status) {
    if ((bad = first_difference(actual, before, out)) != SIZE_MAX) where = bad;
    else if ((bad = first_difference(actual + out, expected, panel->store_bytes)) != SIZE_MAX) where = out + bad;
    else if ((bad = first_difference(actual + end, before + end, bytes - end)) != SIZE_MAX) where = end + bad;
  }
#endif
#endif
  uint64_t t4 = rdcycle();
  c.copy = t1 - t0;
  c.load = t2 - t1;
  c.wait = t3 - t2;
  c.check = t4 - t3;
  c.atlas_cycles = (uint32_t)(a1 - a0);
  c.atlas_instret = (uint32_t)(r1 - r0);
  c.instret = (rdinstret() - i0) + (i0 - mark_instret);
  if (status || where != SIZE_MAX) {
    failed = 1;
    report(index, panel, kind, status, where != SIZE_MAX, &c);
    if (status == 7) {
      printf("LAUNCH k=%ld output-metadata error 7\n", (long)index);
      return status;
    }
    if (status) {
      printf("LAUNCH k=%ld device error %d (2 load, 3 start, 4 wait/timeout or bad halt, 5 launch marker)\n",
             (long)index, status);
      return status;
    }
#if TINY_FULL_EXTENT_CHECK
#if ATLAS_FAST_MODE
    uint8_t wanted = expected[where - out];
#else
    uint8_t wanted = before[where];
#endif
#else
    uint8_t wanted = where >= out && where < end ? expected[where - out] : before[where];
#endif
    tiny_full_mailbox[5] = where;
    printf("LAUNCH k=%ld mismatch at arena byte %lu (0x%lx): got 0x%02x expected 0x%02x (output extent %lu+%lu)\n",
           (long)index, (unsigned long)where, (unsigned long)((uintptr_t)dram + where), (unsigned)actual[where],
           (unsigned)wanted, (unsigned long)out,
#if TINY_FULL_EXTENT_CHECK
           (unsigned long)extent);
#else
           (unsigned long)panel->store_bytes);
#endif
    return 6;
  }
  tiny_full_mailbox[2] = (uint64_t)index + 1;
  total.model += c.model; total.copy += c.copy; total.load += c.load; total.wait += c.wait;
  total.check += c.check; total.instret += c.instret;
  total.atlas_cycles += c.atlas_cycles; total.atlas_instret += c.atlas_instret;
  if (!TINYLLAMA_QUIET) report(index, panel, kind, 0, 0, &c);
  if (TINYLLAMA_STOP_AFTER && (size_t)index + 1 == (size_t)TINYLLAMA_STOP_AFTER) {
    printf("TINYLLAMA PARTIAL stop_after=%u (every completed launch matched its reference)\n",
           (unsigned)TINYLLAMA_STOP_AFTER);
    finish(1, "", 0, 0);
  }
  mark_cycle = rdcycle();
  mark_instret = rdinstret();
  return 0;
}

static int transport(void *context, const tiny_panel *panel, uint8_t *dram, size_t bytes) {
  (void)context;
  ptrdiff_t index = panel - tiny_full_panels;
  if (index < 0 || (size_t)index >= TINY_FULL_KERNELS) return 1;
  return transfer(panel, tiny_full_panel_expected[index], index, 0, dram, bytes);
}

static int attention_transport(void *context, const tiny_attention_panel *panel, uint8_t *dram, size_t bytes) {
  (void)context;
  ptrdiff_t index = panel - tiny_full_attention_panels;
  if (index < 0 || (size_t)index >= TINY_FULL_ATTENTION_KERNELS) return 1;
  tiny_panel launch = {
    .rows = panel->query_rows, .physical_rows = (panel->query_rows + 31) / 32 * 32,
    .k = 64, .n = panel->query_heads * 64,
    .program = panel->program, .program_words = panel->program_words,
    .output_offset = panel->output_offset, .total_bytes = panel->total_bytes,
    .load_bytes = panel->load_bytes, .store_bytes = panel->store_bytes
  };
  return transfer(&launch, panel->expected, index, 1, dram, bytes);
}

#if defined(TINY_FULL_POST_O_NORM) && TINY_FULL_POST_O_NORM
static int norm_transport(void *context, const tiny_norm_panel *panel, uint8_t *dram, size_t bytes) {
  (void)context;
  ptrdiff_t index = panel - tiny_full_norm_panels;
  if (index < 0 || (size_t)index >= TINY_FULL_NORM_KERNELS) return 1;
  /* logical names the first consumer: q for the pre-attention norm, gate for post-o. */
  tiny_panel launch = {
    .rows = panel->rows, .physical_rows = panel->physical_rows, .k = 2048, .n = 2048,
    .logical = panel->flags & TINY_NORM_PRE_ATTENTION ? 0 : 4,
    .program = panel->program, .program_words = panel->program_words,
    .output_offset = panel->output_offset, .total_bytes = panel->total_bytes,
    .load_bytes = panel->load_bytes, .store_bytes = panel->store_bytes
  };
  return transfer(&launch, panel->expected, index, 2, dram, bytes);
}
#endif

tiny_full_projection_context tiny_full_projection_state = {
    .panels = {tiny_full_panels, TINY_FULL_KERNELS, 0, (uint8_t *)(uintptr_t)TINY_FULL_ARENA_BASE,
               TINY_FULL_DRAM_BYTES, transport, 0},
    .calls = 0, .step_count = TINY_FULL_STEPS,
    .attention_panels = tiny_full_attention_panels, .attention_count = TINY_FULL_ATTENTION_KERNELS,
    .attention_cursor = 0, .attention_transport = attention_transport, .attention_transport_context = 0,
#if defined(TINY_FULL_FUSIONS) && TINY_FULL_FUSIONS
    .fusions = tiny_full_fusion_panels,
#endif
#if defined(TINY_FULL_ATTENTION_V2) && TINY_FULL_ATTENTION_V2
    .attention_v2 = tiny_full_attention_v2_panels,
#endif
#if defined(TINY_FULL_POST_O_NORM) && TINY_FULL_POST_O_NORM
    .norm_panels = tiny_full_norm_panels, .norm_count = TINY_FULL_NORM_KERNELS,
    .norm_transport = norm_transport, .layer_count = TINY_FULL_LAYER_COUNT,
#endif
#if defined(TINY_FULL_PREFILL_ROPE) && TINY_FULL_PREFILL_ROPE
    .rope_tables = tiny_full_rope_tables,
#endif
};

void app_init(void) {
#if defined(TERMINAL_DEVICE_UART0)
  UART_InitType UART_init_config;
  UART_init_config.baudrate = PENGUIN26_UART_BAUDRATE;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART_init_config);
#endif
}

int app_main(void) {
  tiny_full_parameters parameters = tiny_full_parameters_global;
#if TINY_FULL_LAYER_COUNT == TINY_FULL_LAYERS
  if ((parameters.capabilities & (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL))
      == (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL))
    parameters.capabilities |= TINY_FULL_CAP_TILED_HIDDEN;
#ifdef TINY_FULL_LOGITS_LAYOUT_CHECK
  parameters.capabilities |= TINY_FULL_CAP_TILED_LOGITS;
#endif
#endif
  printf("TINYLLAMA BEGIN launches=%u steps=%u arena=0x%lx+%u stop_after=%u\n", (unsigned)TINY_FULL_LAUNCHES,
         (unsigned)TINY_FULL_STEPS, (unsigned long)TINY_FULL_ARENA_BASE, (unsigned)TINY_FULL_DRAM_BYTES,
         (unsigned)TINYLLAMA_STOP_AFTER);
  tiny_full_mailbox[0] = UINT64_C(0x54494e5946554c4c);
  tiny_full_mailbox[1] = 1;
  if (tiny_full_workspace_bytes() > sizeof(workspace)) finish(0, "workspace-bytes", tiny_full_workspace_bytes(), sizeof(workspace));
  atlas_host_map map = {ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_SIZE};
  atlas_host_mmio mmio = {atlas_host_direct_read32, atlas_host_direct_write32, 0};
  atlas_host_memory memory = {1, 0, 0}; /* coherent: Atlas DMA is behind the coherence manager */
  if (atlas_host_init(&tiny_full_host, map, mmio, memory) != ATLAS_HOST_OK) finish(0, "atlas-host-init", 0, 0);
  if (tiny_full_projection_configure(&tiny_full_projection_state, parameters.capabilities))
    finish(0, "norm-preflight", parameters.capabilities, 0);
  mark_cycle = wall_begin = rdcycle();
  mark_instret = rdinstret();
#if TINY_FULL_LAYER_COUNT < TINY_FULL_LAYERS
  if (TINY_FULL_STEPS != 1 || TINY_FULL_LAYER_COUNT < 1)
    finish(0, "block-scope", TINY_FULL_STEPS, 1);
  uint32_t tokens[128];
  for (size_t i = 0; i < 128; ++i) tokens[i] = (uint32_t)tiny_full_prefill_tokens[i];
  tiny_full_phase phase = {0};
  int status = tiny_full_model_begin(&parameters, &tiny_full_state_global, &phase,
                                    tokens, 128, workspace, sizeof(workspace));
  if (status) finish(0, "block-begin", (uint64_t)(100 - status), 0);
  for (unsigned layer = 0; layer < TINY_FULL_LAYER_COUNT; ++layer) {
    status = tiny_full_model_layer(&parameters, &tiny_full_state_global, &phase, layer,
                                  tiny_full_project, &tiny_full_projection_state, workspace, sizeof(workspace));
    if (status) finish(0, failed ? "launch-output" : "block-layer", (uint64_t)(100 - status), 0);
  }
  if (phase.active != 1 || phase.next_layer != TINY_FULL_LAYER_COUNT)
    finish(0, "block-continuation", phase.next_layer, TINY_FULL_LAYER_COUNT);
  int check = tiny_full_save_check_block(TINY_FULL_LAYER_COUNT, workspace, sizeof(workspace));
  printf("BLOCK layers=%u rows=128 hidden_cache_check=%s\n", (unsigned)TINY_FULL_LAYER_COUNT,
         check ? "FAIL" : "ok");
  if (check) finish(0, "block-hidden-cache", (uint64_t)check, 0);
#else
  for (unsigned step = 0; step < TINY_FULL_STEPS; ++step) {
    uint32_t tokens[128];
    size_t rows = step ? 1 : 128;
    if (step) {
      tokens[0] = tiny_full_state_global.next_token;
      tiny_full_decode_inputs[step - 1] = tokens[0];
    } else {
      for (size_t i = 0; i < 128; ++i) tokens[i] = (uint32_t)tiny_full_prefill_tokens[i];
    }
    tiny_full_mailbox[6] = step ? 1 : 0;
    int status = tiny_full_model_step(&parameters, &tiny_full_state_global, tokens, rows,
                                      tiny_full_project, &tiny_full_projection_state, workspace, sizeof(workspace),
                                      all_logits, sizeof(all_logits) / sizeof(all_logits[0]));
    if (status) finish(0, failed ? "launch-output" : "model-step", (uint64_t)(100 - status), 0);
    tiny_full_greedy_tokens[step] = tiny_full_state_global.next_token;
#ifdef TINY_FULL_LOGITS_LAYOUT_CHECK
    int check = tiny_full_save_check_step_layout(step, all_logits, rows - 1, parameters.capabilities);
#else
    int check = tiny_full_save_check_step(step, all_logits + (rows - 1) * 32000);
#endif
    printf("STEP %u token=%lu kv_logits_token_check=%s\n", step, (unsigned long)tiny_full_greedy_tokens[step],
           check ? "FAIL" : "ok");
    if (check) finish(0, "step-cache-logits-tokens", step, 0);
  }
#endif
  if (tiny_full_projection_state.panels.cursor != TINY_FULL_KERNELS)
    finish(0, "panel-count", tiny_full_projection_state.panels.cursor, TINY_FULL_KERNELS);
  size_t projection_calls = TINY_FULL_LAYER_COUNT < TINY_FULL_LAYERS ? TINY_FULL_LAYER_COUNT * 7 : TINY_FULL_STEPS * 155;
  if (tiny_full_projection_state.calls != projection_calls)
    finish(0, "projection-count", tiny_full_projection_state.calls, projection_calls);
  if (tiny_full_projection_state.attention_cursor != TINY_FULL_ATTENTION_KERNELS)
    finish(0, "attention-count", tiny_full_projection_state.attention_cursor, TINY_FULL_ATTENTION_KERNELS);
  if (tiny_full_projection_state.norm_cursor != tiny_full_projection_state.norm_count)
    finish(0, "norm-count", tiny_full_projection_state.norm_cursor, tiny_full_projection_state.norm_count);
  if (tiny_full_mailbox[2] != TINY_FULL_LAUNCHES)
    finish(0, "completed-count", tiny_full_mailbox[2], TINY_FULL_LAUNCHES);
  tiny_full_mailbox[7] = TINY_FULL_STEPS - 1;
#if TINY_FULL_LAYER_COUNT < TINY_FULL_LAYERS
  /* A checked block has no LM-head token and does not commit all 22 caches. */
  tiny_full_mailbox[8] = 128;
  tiny_full_mailbox[9] = 0;
  printf("TINYLLAMA BLOCK PASSED layers=%u rows=128 position=128 scope=block\n", (unsigned)TINY_FULL_LAYER_COUNT);
#else
  tiny_full_mailbox[8] = tiny_full_state_global.position;
  tiny_full_mailbox[9] = tiny_full_state_global.next_token;
  printf("TINYLLAMA PASS tokens=");
  for (unsigned step = 0; step < TINY_FULL_STEPS; ++step)
    printf(step ? ",%lu" : "%lu", (unsigned long)tiny_full_greedy_tokens[step]);
  printf(" position=%lu\n", (unsigned long)tiny_full_state_global.position);
#endif
  finish(1, "", 0, 0);
  return 0;
}

int main(void) {
  app_init();
  return app_main();
}
