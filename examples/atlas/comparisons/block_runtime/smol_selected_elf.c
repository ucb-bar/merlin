#include "smol_selected_workload.h"
#include <stddef.h>
#include <stdint.h>
__attribute__((section(".tohost"), aligned(64))) volatile uint64_t tohost, fromhost;
typedef struct {
 uint32_t magic, version, status, completed_linears, failing_role, runtime_result;
 uint32_t sequence[7], device_cycles[7], output_mismatches, key_mismatches, value_mismatches;
 uint64_t trap_mcause, trap_mepc, trap_mtval;
} smol_mailbox;
__attribute__((section(".handoff"), aligned(64))) volatile smol_mailbox smol_workload_mailbox;
static atlas_host host;
static smol_raw_context context;
__attribute__((aligned(64))) static uint8_t activation_copy[192 * 2560];
static void finish(uint32_t result) {
 smol_workload_mailbox.status = result;
 __asm__ volatile("fence iorw,iorw" : : : "memory");
 tohost = ((uint64_t) result << 1) | 1;
 __asm__ volatile("fence iorw,iorw" : : : "memory");
 for (;;)__asm__ volatile("wfi");
}
void atlas_block_trap_report(uint64_t cause, uint64_t pc, uint64_t value) {
 smol_workload_mailbox.magic = 0x534d4f4c;
 smol_workload_mailbox.trap_mcause = cause;
 smol_workload_mailbox.trap_mepc = pc;
 smol_workload_mailbox.trap_mtval = value;
 finish(0xff);
}
void atlas_workload_main(void) {
 smol_workload_mailbox.magic = 0x534d4f4c;
 smol_workload_mailbox.version = 1;
 smol_workload_mailbox.status = UINT32_MAX;
 atlas_host_result result = atlas_host_init(&host, (atlas_host_map) {
     0x20000, 0x40000, 128 * 1024},
  (atlas_host_mmio) {
      atlas_host_direct_read32, atlas_host_direct_write32, 0}, (atlas_host_memory) {
      1, 0, 0});
 if (result != ATLAS_HOST_OK) {
     smol_workload_mailbox.runtime_result = result;
     finish(0xfe);
 }
 context = (smol_raw_context) {
     .host = &host, .kernels = smol_kernels,
  .dram = (uint8_t *)(uintptr_t) UINT64_C(0x100000000), .dram_capacity = 3625472,
  .activation_copy = activation_copy, .activation_capacity = sizeof(activation_copy), .max_polls = 12000000};
 int code = smol_selected_run(smol_block_raw_linear, &context);
 smol_workload_mailbox.completed_linears = context.completed;
 smol_workload_mailbox.failing_role = context.failing_index;
 smol_workload_mailbox.runtime_result = context.runtime_result;
 for (unsigned i = 0; i < 7; ++i) {
     smol_workload_mailbox.sequence[i] = context.sequence[i];
     smol_workload_mailbox.device_cycles[i] = context.cycle_delta[i];
 }
 smol_workload_mailbox.output_mismatches = smol_output_mismatches;
 smol_workload_mailbox.key_mismatches = smol_key_mismatches;
 smol_workload_mailbox.value_mismatches = smol_value_mismatches;
 if (!code && (context.completed != 7 || context.failed || host.active || !smol_selected_state.complete ||
     !smol_selected_state.cache_valid)) code = 6;
 finish((uint32_t) code);
}
