/* Live selected Smol layer: retained nonlinear C, seven raw projections. */
#include "smol_selected_workload.h"
#include "smol_block_capture.h"
#include <stddef.h>
#include <stdint.h>

__attribute__((section(".tohost"), aligned(64))) volatile uint64_t tohost, fromhost;
typedef struct {
  uint32_t magic, version, status, completed_linears, failing_role, runtime_result;
  uint32_t sequence[7], device_cycles[7], output_mismatches, key_mismatches, value_mismatches;
  uint64_t trap_mcause, trap_mepc, trap_mtval;
} smol_mailbox;
__attribute__((section(".handoff"), aligned(64))) volatile smol_mailbox smol_workload_mailbox;
smol_capture_result smol_capture_results[7];
static atlas_host host;
static smol_raw_context context;
__attribute__((aligned(64))) static uint8_t activation_copy[192 * 2560];
static const char *const names[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
static unsigned launch_writes;
static uint64_t total_device_cycles;
static inline uint64_t cycles(void) {
  uint64_t value; __asm__ volatile("rdcycle %0" : "=r"(value)); return value;
}
static uint32_t csr(unsigned offset) { return atlas_host_direct_read32(0, ATLAS_CSR_BASE + offset); }
static void write32(void *opaque, uintptr_t address, uint32_t value) {
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL && value == 1) ++launch_writes;
  atlas_host_direct_write32(opaque, address, value);
}

/* Standard HTIF write syscall; diagnostic QEMU intercepts this exact function.
 * The same ELF reaches the real HTIF frontend in EE290 simulation. */
__attribute__((noinline)) void atlas_layer_write(const char *text, size_t bytes) {
  static volatile uint64_t syscall[8] __attribute__((aligned(64)));
  syscall[0] = 64; syscall[1] = 1; syscall[2] = (uintptr_t)text; syscall[3] = bytes;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost = (uintptr_t)syscall;
  while (!fromhost) { }
  fromhost = 0;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
}
static char line[768];
static size_t used;
static void text(const char *s) { while (*s && used < sizeof(line) - 2) line[used++] = *s++; }
static void number(uint64_t value) {
  char digits[20]; unsigned n = 0;
  do { digits[n++] = (char)('0' + value % 10); value /= 10; } while (value);
  while (n && used < sizeof(line) - 2) line[used++] = digits[--n];
}
static void field(const char *key, uint64_t value) { text(key); number(value); }
static void emit(void) { line[used++] = '\n'; atlas_layer_write(line, used); used = 0; }
static void named(const char *prefix, unsigned role) { text(prefix); text(" name="); text(names[role]); }

static int launch(void *opaque, unsigned role, const uint16_t *input, size_t rows,
                  size_t input_width, uint16_t *output, size_t output_width) {
  if (role >= 7) return SMOL_RAW_INVALID;
  const smol_raw_kernel *k = &smol_kernels[role];
  uint64_t begin = cycles();
  named("ATLAS_KERNEL_BEGIN", role); field(" words=", k->program_words);
  field(" dram_bytes=", k->dram_bytes); field(" output_bytes=", k->output_bytes); emit();
  int result = smol_block_raw_linear(opaque, role, input, rows, input_width, output, output_width);
  total_device_cycles += context.cycle_delta[role];
  named("ATLAS_KERNEL_PHASE", role); field(" device_cycles=", context.cycle_delta[role]);
  field(" result=", result); field(" active=", host.active); field(" sequence=", host.invocation_sequence);
  field(" status=", csr(ATLAS_CSR_STATUS)); emit();
  named("ATLAS_KERNEL_CHECK", role); field(" output_byte_mismatches=", smol_capture_results[role].output_mismatches);
  field(" preserved_byte_mismatches=", smol_capture_results[role].preserved_mismatches);
  field(" check_cpu_cycles=", smol_capture_results[role].check_cycles);
  field(" dbg0=", csr(ATLAS_CSR_DBG0)); field(" dbg1=", csr(ATLAS_CSR_DBG1));
  field(" total_cpu_cycles=", cycles() - begin); emit();
  if (!result && (host.active || host.invocation_sequence != role + 1 || launch_writes != role + 1 ||
                 csr(ATLAS_CSR_DBG0) != role + 1 || csr(ATLAS_CSR_DBG1) != role + 1)) result = SMOL_RAW_RUNTIME;
  named(result ? "ATLAS_KERNEL_FAIL" : "ATLAS_KERNEL_PASS", role);
  field(" device_cycles=", context.cycle_delta[role]); emit();
  return result;
}

static void finish(uint32_t result) {
  smol_workload_mailbox.status = result;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost = ((uint64_t)result << 1) | 1;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  for (;;) __asm__ volatile("wfi");
}
void atlas_block_trap_report(uint64_t cause, uint64_t pc, uint64_t value) {
  smol_workload_mailbox.magic = 0x534d4f4c;
  smol_workload_mailbox.trap_mcause = cause; smol_workload_mailbox.trap_mepc = pc;
  smol_workload_mailbox.trap_mtval = value; finish(0xff);
}
void atlas_workload_main(void) {
  uint64_t begin = cycles();
  smol_workload_mailbox.magic = 0x534d4f4c; smol_workload_mailbox.version = 1;
  smol_workload_mailbox.status = UINT32_MAX;
  atlas_host_result result = atlas_host_init(&host, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES},
    (atlas_host_mmio){atlas_host_direct_read32, write32, 0}, (atlas_host_memory){1, 0, 0});
  if (result != ATLAS_HOST_OK) { smol_workload_mailbox.runtime_result = result; finish(0xfe); }
  context = (smol_raw_context){.host=&host, .kernels=smol_kernels,
    .dram=(uint8_t *)(uintptr_t)ATLAS_DRAM_BASE, .dram_capacity=3625472,
    .activation_copy=activation_copy, .activation_capacity=sizeof(activation_copy), .max_polls=12000000};
  int code = smol_selected_run(launch, &context);
  smol_workload_mailbox.completed_linears = context.completed;
  smol_workload_mailbox.failing_role = context.failing_index;
  smol_workload_mailbox.runtime_result = context.runtime_result;
  for (unsigned i=0; i<7; ++i) { smol_workload_mailbox.sequence[i]=context.sequence[i];
    smol_workload_mailbox.device_cycles[i]=context.cycle_delta[i]; }
  smol_workload_mailbox.output_mismatches=smol_output_mismatches;
  smol_workload_mailbox.key_mismatches=smol_key_mismatches;
  smol_workload_mailbox.value_mismatches=smol_value_mismatches;
  if (!code && (context.completed != 7 || context.failed || host.active || launch_writes != 7 ||
                !smol_selected_state.complete || !smol_selected_state.cache_valid)) code=6;
  text(code ? "ATLAS_LAYER_FAIL" : "ATLAS_LAYER_PASS"); field(" launches=", context.completed);
  field(" device_cycles=", total_device_cycles); field(" total_cpu_cycles=", cycles()-begin);
  field(" output_mismatches=", smol_output_mismatches); field(" key_mismatches=", smol_key_mismatches);
  field(" value_mismatches=", smol_value_mismatches); emit();
  finish((uint32_t)code);
}
