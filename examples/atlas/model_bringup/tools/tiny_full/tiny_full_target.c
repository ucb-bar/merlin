/* Complete22-layer prefill and actual greedy decode, genuine Atlas panels. */
#include "tiny_full_model.h"
#include "tiny_full_projection.h"
#include "tiny_full_assets.h"
#include "atlas_host.h"
#include "atlas_result_transport.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

__attribute__((aligned(64))) volatile uint64_t tiny_full_mailbox[16];
atlas_host tiny_full_host;
static uint64_t workspace[24 * 1024 * 1024 / 8];
static uint8_t before[TINY_FULL_DRAM_BYTES];
static uint16_t all_logits[128 * 32000];
static atlas_result result;
static atlas_result_uart result_uart;

static int transport(void *context, const tiny_panel *panel, uint8_t *dram, size_t bytes) {
    (void)context;
    ptrdiff_t index = panel - tiny_full_panels;
    if (index < 0 || (size_t)index >= TINY_FULL_KERNELS || bytes > sizeof(before)) return 1;
    tiny_full_mailbox[3] = (uint64_t)index;
    memcpy(before, dram, bytes);
    if (atlas_host_load_raw(&tiny_full_host, panel->program, panel->program_words) != ATLAS_HOST_OK) return 2;
    atlas_host_buffer buffer = {(uintptr_t)dram, bytes, ATLAS_HOST_DEVICE_READ_WRITE};
    if (atlas_host_start(&tiny_full_host, &buffer, 1) != ATLAS_HOST_OK) return 3;
    if (atlas_host_wait(&tiny_full_host, 200000000) != ATLAS_HOST_OK) return 4;
    if (tiny_full_host.invocation_sequence != (uint32_t)index + 1 || tiny_full_host.active) return 5;
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    const volatile uint8_t *actual = dram;
    for (size_t i = 0; i < bytes; ++i) {
        uint8_t wanted = i >= panel->output_offset && i - panel->output_offset < panel->store_bytes
            ? tiny_full_panel_expected[index][i - panel->output_offset] : before[i];
        if (actual[i] != wanted) {
            tiny_full_mailbox[5] = i;
            atlas_result_fail(&result,"physical-dram-offset",i,panel->output_offset);
            atlas_result_fail(&result,"physical-dram-byte",actual[i],wanted);
            return 6;
        }
    }
    tiny_full_mailbox[2] = (uint64_t)index + 1;
    if ((size_t)index+1 == TINY_FULL_KERNELS
        || ((panel->logical == 6 || panel->logical == 7)
            && tiny_full_panels[index+1].logical != panel->logical))
        atlas_result_progress(&result,(uint32_t)index+1,panel->logical == 7 ? "lm-head" : "layer");
    return 0;
}

tiny_full_projection_context tiny_full_projection_state = {
    {tiny_full_panels, TINY_FULL_KERNELS, 0, (uint8_t *)(uintptr_t)TINY_FULL_ARENA_BASE,
     TINY_FULL_DRAM_BYTES, transport, 0}, 0, TINY_FULL_STEPS, 0, 0, 0, 0, 0
};

static void complete(uint64_t state, uint64_t code) {
    if (state == 2 && !atlas_result_pass(&result)) { state = 3; code = 301; }
    if (state != 2) atlas_result_fail(&result,"host-status",code,0);
    tiny_full_mailbox[1] = state;
    tiny_full_mailbox[4] = code;
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    tohost = state == 2 ? 1 : 3;
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    for (;;) __asm__ volatile("wfi");
}

void tiny_full_trap_report(uint64_t cause, uint64_t pc, uint64_t value) {
    if (result.putchar != atlas_result_uart_putchar) {
        atlas_result_uart_init(&result_uart);
        atlas_result_begin(&result,atlas_result_uart_putchar,&result_uart,"tiny-full",TINY_FULL_KERNELS);
    }
    atlas_result_fail(&result,"trap-cause",cause,0);
    atlas_result_fail(&result,"trap-pc",pc,0);
    atlas_result_fail(&result,"trap-value",value,0);
}

int main(void) {
    atlas_result_uart_init(&result_uart);
    atlas_result_begin(&result,atlas_result_uart_putchar,&result_uart,"tiny-full",TINY_FULL_KERNELS);
    tiny_full_mailbox[0] = UINT64_C(0x54494e5946554c4c);
    tiny_full_mailbox[1] = 1;
    if (tiny_full_workspace_bytes() > sizeof(workspace)) complete(3, 1);
    atlas_host_map map = {ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES};
    atlas_host_mmio mmio = {atlas_host_direct_read32, atlas_host_direct_write32, 0};
    atlas_host_memory memory = {1, 0, 0};
    if (atlas_host_init(&tiny_full_host, map, mmio, memory) != ATLAS_HOST_OK) complete(3, 2);
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
        int status = tiny_full_model_step(&tiny_full_parameters_global, &tiny_full_state_global,
            tokens, rows, tiny_full_project, &tiny_full_projection_state, workspace, sizeof(workspace),
            all_logits, sizeof(all_logits) / sizeof(all_logits[0]));
        if (status) complete(3, (uint64_t)(100 - status));
        tiny_full_greedy_tokens[step] = tiny_full_state_global.next_token;
        int final_check = tiny_full_save_check_step(step, all_logits + (rows - 1) * 32000);
        if (!atlas_result_check(&result,"step-cache-logits-tokens",(uint64_t)final_check,0)) complete(3,200+step);
    }
    if (!atlas_result_check(&result,"panel-count",tiny_full_projection_state.panels.cursor,TINY_FULL_KERNELS)
        || !atlas_result_check(&result,"projection-count",tiny_full_projection_state.calls,TINY_FULL_STEPS*155)
        || !atlas_result_check(&result,"completed-count",tiny_full_mailbox[2],TINY_FULL_KERNELS)) complete(3,300);
    tiny_full_mailbox[7] = TINY_FULL_STEPS - 1;
    tiny_full_mailbox[8] = tiny_full_state_global.position;
    tiny_full_mailbox[9] = tiny_full_state_global.next_token;
    complete(2, 0);
    return 0;
}
