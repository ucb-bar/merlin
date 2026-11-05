/* Bounded Tiny first-block prefill + first decode. Platform execution pending.
 * Uses the unchanged raw Atlas host runtime and caller-established coherence.
 */
#include "tiny_block.h"
#include "tiny_projection.h"
#include "tiny_bundle_assets.h"
#include "atlas_host.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

__attribute__((section(".tohost"), aligned(64))) volatile uint64_t tohost;
__attribute__((aligned(64))) volatile uint64_t atlas_test_mailbox[16];
static uint64_t workspace[24 * 1024 * 1024 / 8];
static uint8_t before[6750720];
static uint16_t prefill_keys[4 * 128 * 64], prefill_values[4 * 128 * 64];
static uint16_t decode_keys[4 * 129 * 64], decode_values[4 * 129 * 64];
static uint16_t prefill_output[128 * 2048], decode_output[2048];
static atlas_host device;

static int transport(void *context, const tiny_panel *panel, uint8_t *dram, size_t bytes) {
    (void) context;
    ptrdiff_t index = panel - tiny_panels;
    if (index < 0 || index >= 24 || bytes > sizeof(before)) return 1;
    atlas_test_mailbox[3] = (uint64_t) index;
    memcpy(before, dram, bytes);
    if (atlas_host_load_raw(&device, panel->program, panel->program_words) != ATLAS_HOST_OK) return 2;
    atlas_host_buffer buffer = {(uintptr_t) dram, bytes, ATLAS_HOST_DEVICE_READ_WRITE};
    if (atlas_host_start(&device, &buffer, 1) != ATLAS_HOST_OK) return 3;
    if (atlas_host_wait(&device, 200000000) != ATLAS_HOST_OK) return 4;
    if (device.invocation_sequence != (uint32_t) index + 1 || device.active) return 5;
    __asm__ volatile("fence iorw,iorw" : : : "memory");
    const volatile uint8_t *actual = dram;
    for (size_t i = 0; i < bytes; ++i) {
        uint8_t wanted = i >= panel->output_offset && i - panel->output_offset < panel->store_bytes
            ? tiny_panel_expected[index][i - panel->output_offset] : before[i];
        if (actual[i] != wanted) {
            atlas_test_mailbox[5] = i;
            return 6;
        }
    }
    atlas_test_mailbox[2] = (uint64_t) index + 1;
    return 0;
}

static void complete(uint64_t state, uint64_t code) {
    atlas_test_mailbox[1] = state;
    atlas_test_mailbox[4] = code;
    __asm__ volatile("fence iorw,iorw" : : : "memory");
    tohost = state == 2 ? 1 : 3;
    __asm__ volatile("fence iorw,iorw" : : : "memory");
    for (;;)__asm__ volatile("wfi");
}

int main(void) {
    atlas_test_mailbox[0] = UINT64_C(0x54494e59424c4b31);
    atlas_test_mailbox[1] = 1;
    if (tiny_block_workspace_bytes() > sizeof(workspace)) complete(3, 1);
    atlas_host_map map = {0x20000, 0x40000, 128 * 1024};
    atlas_host_mmio mmio = {atlas_host_direct_read32, atlas_host_direct_write32, 0};
    atlas_host_memory memory = {1, 0, 0};
    if (atlas_host_init(&device, map, mmio, memory) != ATLAS_HOST_OK) complete(3, 2);
    tiny_projection_context c = {tiny_panels, 24, 0, (uint8_t *)(uintptr_t) UINT64_C(0x100000000),
                                6750720, transport, 0};
    for (size_t phase = 0; phase < 2; ++phase) {
        atlas_test_mailbox[6] = phase;
        size_t rows = phase ? 1 : 128;
        uint16_t *out = phase ? decode_output : prefill_output;
        int code = tiny_block_run(rows, tiny_entry[phase], tiny_input_norm, tiny_post_norm,
            tiny_epsilon, tiny_cosine, tiny_sine, phase ? prefill_keys : 0, phase ? prefill_values : 0,
            tiny_project, &c, workspace, sizeof(workspace), phase ? decode_keys : prefill_keys,
            phase ? decode_values : prefill_values, out);
        if (code) complete(3, (uint64_t)(100 - code));
        for (size_t i = 0; i < rows * 2048; ++i)
            if (out[i] != tiny_expected_output[phase][i]) {
                atlas_test_mailbox[5] = i;
                complete(3, 200 + phase);
            }
        const uint16_t *keys = phase ? decode_keys : prefill_keys;
        const uint16_t *values = phase ? decode_values : prefill_values;
        for (size_t i = 0; i < 4 * (phase ? 129u : 128u) * 64; ++i) {
            if (keys[i] != tiny_expected_keys[phase][i]) {
                atlas_test_mailbox[5] = i;
                complete(3, 210 + phase);
            }
            if (values[i] != tiny_expected_values[phase][i]) {
                atlas_test_mailbox[5] = i;
                complete(3, 220 + phase);
            }
        }
    }
    if (c.cursor != 24 || atlas_test_mailbox[2] != 24) complete(3, 300);
    complete(2, 0);
    return 0;
}
