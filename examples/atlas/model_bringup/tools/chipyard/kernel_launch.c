/* One ResNet kernel launch on the EE290 SoC. LOADMEM places the initial DRAM image at ATLAS_DRAM_BASE. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "atlas_host.h"
#include "payload.h"

static atlas_host host;
static atlas_host_buffer buffer;
static inline uint64_t cpu_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}
static uint32_t csr(unsigned offset) { return atlas_host_direct_read32(0, ATLAS_CSR_BASE + offset); }

/* Count mismatching 64-bit words between DRAM and an expected image, or against a repeated byte. */
static uint64_t compare(const volatile uint64_t *dram, const uint64_t *expected, uint64_t fill, size_t words) {
  uint64_t bad = 0;
  for (size_t i = 0; i < words; ++i) bad += dram[i] != (expected ? expected[i] : fill);
  return bad;
}

int main(void) {
  const uint64_t begin = cpu_cycles();
  printf("ATLAS_KERNEL_BEGIN name=%s words=%u output_offset=%u output_bytes=%u\n", PAYLOAD_NAME,
         PAYLOAD_PROGRAM_WORDS, PAYLOAD_OUTPUT_OFFSET, PAYLOAD_OUTPUT_BYTES);
  atlas_host_result result = atlas_host_init(
      &host, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES},
      (atlas_host_mmio){atlas_host_direct_read32, atlas_host_direct_write32, 0}, (atlas_host_memory){1, 0, 0});
  if (result == ATLAS_HOST_OK) result = atlas_host_load_raw(&host, atlas_program, PAYLOAD_PROGRAM_WORDS);
  if (result != ATLAS_HOST_OK) {
    printf("ATLAS_KERNEL_FAIL phase=init-or-load result=%u\n", (unsigned)result);
    return 1;
  }
  buffer = (atlas_host_buffer){ATLAS_DRAM_BASE, PAYLOAD_DRAM_BYTES, ATLAS_HOST_DEVICE_READ_WRITE};
  const uint32_t device_begin = csr(ATLAS_CSR_CYCLES);
  const uint64_t start = cpu_cycles();
  result = atlas_host_start(&host, &buffer, 1);
  if (result == ATLAS_HOST_OK) result = atlas_host_wait(&host, PAYLOAD_MAX_POLLS);
  const uint64_t finished = cpu_cycles();
  const uint32_t device_cycles = csr(ATLAS_CSR_CYCLES) - device_begin;
  printf("ATLAS_KERNEL_PHASE setup_cpu_cycles=%" PRIu64 " launch_wait_cpu_cycles=%" PRIu64 " device_cycles=%" PRIu32
         " result=%u active=%u sequence=%" PRIu32 " status=0x%08" PRIx32 "\n",
         start - begin, finished - start, device_cycles, (unsigned)result, (unsigned)host.active,
         host.invocation_sequence, csr(ATLAS_CSR_STATUS));
  if (result != ATLAS_HOST_OK || host.active || host.invocation_sequence != 1) {
    printf("ATLAS_KERNEL_FAIL phase=launch-or-wait\n");
    return 2;
  }
  const volatile uint64_t *dram = (const volatile uint64_t *)(uintptr_t)(ATLAS_DRAM_BASE + PAYLOAD_OUTPUT_OFFSET);
  const uint64_t output_bad = compare(dram, (const uint64_t *)atlas_expected, 0, PAYLOAD_OUTPUT_BYTES / 8);
  const uint64_t guard_bad = compare(dram - 32, 0, 0xa5a5a5a5a5a5a5a5ull, 32) +
                             compare(dram + PAYLOAD_OUTPUT_BYTES / 8, 0, 0x5a5a5a5a5a5a5a5aull, 32);
  printf("ATLAS_KERNEL_CHECK output_word_mismatches=%" PRIu64 " guard_word_mismatches=%" PRIu64
         " dbg0=%" PRIu32 " dbg1=%" PRIu32 " total_cpu_cycles=%" PRIu64 "\n",
         output_bad, guard_bad, csr(ATLAS_CSR_DBG0), csr(ATLAS_CSR_DBG1), cpu_cycles() - begin);
  if (output_bad || guard_bad || csr(ATLAS_CSR_DBG0) != 1 || csr(ATLAS_CSR_DBG1) != 1) {
    printf("ATLAS_KERNEL_FAIL phase=reference-or-markers\n");
    return 3;
  }
  printf("ATLAS_KERNEL_PASS name=%s device_cycles=%" PRIu32 "\n", PAYLOAD_NAME, device_cycles);
  return 0;
}
