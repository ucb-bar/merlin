/* Common segment dispatcher; all projection transport and host math are shared. */
#include "atlas_result_transport.h"
static uint32_t segment_start;
static void segment_progress(atlas_result *, uint32_t, const char *);
#define atlas_result_progress segment_progress
#define workspace tiny_full_segment_workspace
#define main tiny_full_unsegmented_main
#include "tiny_full_target.c"
#undef main
#undef workspace
#undef atlas_result_progress

tiny_full_phase tiny_full_segment_phase;
uint64_t tiny_full_segment_control[8];
uint64_t tiny_full_segment_request;

static void segment_progress(atlas_result *s, uint32_t completed, const char *boundary) {
    atlas_result_progress(s, completed - segment_start, boundary);
}

__attribute__((noinline)) void tiny_full_segment_ready(void) {
    __asm__ volatile("fence iorw,iorw\n wfi" ::: "memory");
}

static void publish_control(unsigned step, unsigned layer) {
    uint64_t *c = tiny_full_segment_control;
    c[0] = step; c[1] = layer; c[2] = tiny_full_state_global.position;
    c[3] = tiny_full_state_global.bank; c[4] = tiny_full_state_global.next_token;
    c[5] = step ? 1 : 128; c[6] = tiny_full_projection_state.calls;
    c[7] = tiny_full_projection_state.panels.cursor;
}

int main(void) {
    atlas_result_uart_init(&result_uart);
    tiny_full_mailbox[0] = UINT64_C(0x54494e5946554c4c);
    tiny_full_mailbox[1] = 1;
    atlas_host_map map = {ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES};
    atlas_host_mmio mmio = {atlas_host_direct_read32, atlas_host_direct_write32, 0};
    atlas_host_memory memory = {1, 0, 0};
    if (atlas_host_init(&tiny_full_host, map, mmio, memory) != ATLAS_HOST_OK) complete(3, 401);
    tiny_full_segment_ready();
    uint64_t request = tiny_full_segment_request;
    unsigned step = (unsigned)(request / 23), layer = (unsigned)(request % 23);
    segment_start = step * 280 + (layer < 22 ? layer * 12 : 264);
    unsigned end = segment_start + (layer < 22 ? 12 : 16);
    atlas_result_begin(&result, atlas_result_uart_putchar, &result_uart, "tiny-segment", end - segment_start);
    uint64_t *c = tiny_full_segment_control;
    if (request >= 69 || c[0] != step || c[1] != layer || c[2] != (step ? 127u + step : 0)
        || c[3] != step % 2 || c[5] != (step ? 1u : 128u)
        || c[6] != step * 155 + layer * 7 || c[7] != segment_start) complete(3, 402);
    tiny_full_state_global.position = c[2];
    tiny_full_state_global.bank = (unsigned)c[3];
    tiny_full_state_global.next_token = (uint32_t)c[4];
    tiny_full_projection_state.panels.cursor = segment_start;
    tiny_full_projection_state.calls = c[6];
    tiny_full_host.invocation_sequence = segment_start;
    tiny_full_mailbox[2] = segment_start;
    tiny_full_mailbox[6] = step ? 1 : 0;
    size_t rows = step ? 1 : 128;
    if (layer == 0) {
        uint32_t tokens[128];
        if (step) {
            tokens[0] = tiny_full_state_global.next_token;
            tiny_full_decode_inputs[step - 1] = tokens[0];
        } else for (size_t i = 0; i < 128; ++i) tokens[i] = (uint32_t)tiny_full_prefill_tokens[i];
        if (tiny_full_model_begin(&tiny_full_parameters_global, &tiny_full_state_global, &tiny_full_segment_phase,
                tokens, rows, tiny_full_segment_workspace, sizeof(tiny_full_segment_workspace))) complete(3, 403);
    }
    int status;
    if (layer < 22) {
        status = tiny_full_model_layer(&tiny_full_parameters_global, &tiny_full_state_global,
            &tiny_full_segment_phase, layer, tiny_full_project, &tiny_full_projection_state,
            tiny_full_segment_workspace, sizeof(tiny_full_segment_workspace));
        if (status) complete(3, (uint64_t)(410 - status));
        publish_control(step, layer + 1);
    } else {
        status = tiny_full_model_finish(&tiny_full_parameters_global, &tiny_full_state_global,
            &tiny_full_segment_phase, tiny_full_project, &tiny_full_projection_state,
            tiny_full_segment_workspace, sizeof(tiny_full_segment_workspace), all_logits,
            sizeof(all_logits) / sizeof(all_logits[0]));
        if (status) complete(3, (uint64_t)(420 - status));
        tiny_full_greedy_tokens[step] = tiny_full_state_global.next_token;
        int check = tiny_full_save_check_step(step, all_logits + (rows - 1) * 32000);
        if (!atlas_result_check(&result, "step-cache-logits-tokens", (uint64_t)check, 0)) complete(3, 430);
        publish_control(step + 1, 0);
    }
    if (!atlas_result_check(&result, "panel-count", tiny_full_projection_state.panels.cursor, end)
        || !atlas_result_check(&result, "completed-count", tiny_full_mailbox[2], end)
        || !atlas_result_check(&result, "host-sequence", tiny_full_host.invocation_sequence, end)
        || tiny_full_host.active) complete(3, 440);
    tiny_full_mailbox[7] = request + 1;
    tiny_full_mailbox[8] = tiny_full_state_global.position;
    tiny_full_mailbox[9] = tiny_full_state_global.next_token;
    complete(2, 0);
    return 0;
}
