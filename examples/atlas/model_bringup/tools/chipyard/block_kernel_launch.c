/* Independent raw block kernel: LOADMEM image, sequenced mailbox, full-image checks. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "atlas_host.h"
#include "payload.h"
#ifndef ATLAS_FAST_CHECK
#define ATLAS_FAST_CHECK 0
#endif
#if ATLAS_FAST_CHECK
#include "atlas_host_check.h"
static atlas_host_check_result first_failure;
static size_t first_failure_base;

static inline uint64_t check_cycles(void) {
  uint64_t value;
  __asm__ volatile("fence rw,rw\nrdcycle %0" : "=r"(value) : : "memory");
  return value;
}
#endif

static atlas_host host;
static atlas_host_buffer buffer;
static unsigned launch_writes;
static void write32(void *context, uintptr_t address, uint32_t value) {
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL && value == 1) ++launch_writes;
  atlas_host_direct_write32(context, address, value);
}
static inline uint64_t cpu_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}
static uint32_t csr(unsigned offset) { return atlas_host_direct_read32(0, ATLAS_CSR_BASE + offset); }

/* Word comparisons cover every byte, with a byte tail for arbitrary images. */
static uint64_t compare(size_t offset, size_t bytes) {
  const volatile uint8_t *dram = (const volatile uint8_t *)(uintptr_t)(PAYLOAD_DRAM_BASE + offset);
  const uint8_t *expected = atlas_expected + offset;
#if ATLAS_FAST_CHECK
  atlas_host_check_result result = atlas_host_check_exact(dram, expected, bytes);
  if (result.mismatch_bytes && !first_failure.mismatch_bytes) {
    first_failure = result;
    first_failure_base = offset;
  }
  return result.mismatch_bytes;
#else
  uint64_t bad = 0;
  while (bytes && ((uintptr_t)dram & 7u)) {
    bad += *dram++ != *expected++; --bytes;
  }
  while (bytes >= 8) {
    uint64_t want = 0;
    for (unsigned i = 0; i < 8; ++i) want |= (uint64_t)expected[i] << (8 * i);
    bad += *(const volatile uint64_t *)dram != want;
    dram += 8; expected += 8; bytes -= 8;
  }
  while (bytes--) bad += *dram++ != *expected++;
  return bad;
#endif
}

int main(void) {
  const uint64_t begin = cpu_cycles();
  printf("ATLAS_KERNEL_BEGIN name=%s words=%u dram_bytes=%u output_bytes=%u\n", PAYLOAD_NAME,
         PAYLOAD_PROGRAM_WORDS, PAYLOAD_DRAM_BYTES, PAYLOAD_OUTPUT_BYTES);
  atlas_host_result result = atlas_host_init(
      &host, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, ATLAS_IMEM_BYTES},
      (atlas_host_mmio){atlas_host_direct_read32, write32, 0}, (atlas_host_memory){1, 0, 0});
  if (result == ATLAS_HOST_OK) result = atlas_host_load_raw(&host, atlas_program, PAYLOAD_PROGRAM_WORDS);
  if (result != ATLAS_HOST_OK) {
    printf("ATLAS_KERNEL_FAIL phase=init-or-load result=%u\n", (unsigned)result); return 1;
  }
  buffer = (atlas_host_buffer){PAYLOAD_DRAM_BASE, PAYLOAD_DRAM_BYTES, ATLAS_HOST_DEVICE_READ_WRITE};
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
  if (result != ATLAS_HOST_OK || host.active || host.invocation_sequence != 1 || launch_writes != 1) {
    printf("ATLAS_KERNEL_FAIL phase=launch-or-wait\n"); return 2;
  }
  uint64_t output_bad = 0, preserved_bad = 0, cursor = 0;
#if ATLAS_FAST_CHECK
  const uint64_t check_begin = check_cycles();
#endif
  for (unsigned i = 0; i < PAYLOAD_OUTPUT_RANGES; ++i) {
    preserved_bad += compare(cursor, atlas_output_ranges[i][0] - cursor);
    output_bad += compare(atlas_output_ranges[i][0], atlas_output_ranges[i][1]);
    cursor = atlas_output_ranges[i][0] + atlas_output_ranges[i][1];
  }
  preserved_bad += compare(cursor, PAYLOAD_DRAM_BYTES - cursor);
#if ATLAS_FAST_CHECK
  const uint64_t check_elapsed = check_cycles() - check_begin;
  printf("ATLAS_KERNEL_CHECK_TIMING fast=%u check_cpu_cycles=%" PRIu64 "\n", ATLAS_FAST_CHECK, check_elapsed);
  if (first_failure.mismatch_bytes) {
    if (first_failure.first_offset == SIZE_MAX)
      printf("ATLAS_KERNEL_BYTE_FAILURE unstable=1 offset=unknown\n");
    else
      printf("ATLAS_KERNEL_BYTE_FAILURE offset=%" PRIu64 " actual=%u expected=%u\n",
             (uint64_t)(first_failure_base + first_failure.first_offset),
             (unsigned)first_failure.actual, (unsigned)first_failure.expected);
  }
  printf("ATLAS_KERNEL_CHECK output_byte_mismatches=%" PRIu64 " preserved_byte_mismatches=%" PRIu64
#else
  printf("ATLAS_KERNEL_CHECK output_unit_mismatches=%" PRIu64 " preserved_unit_mismatches=%" PRIu64
#endif
         " dbg0=%" PRIu32 " dbg1=%" PRIu32 " total_cpu_cycles=%" PRIu64 "\n",
         output_bad, preserved_bad, csr(ATLAS_CSR_DBG0), csr(ATLAS_CSR_DBG1), cpu_cycles() - begin);
  if (output_bad || preserved_bad || csr(ATLAS_CSR_DBG0) != 1 || csr(ATLAS_CSR_DBG1) != 1) {
    printf("ATLAS_KERNEL_FAIL phase=reference-or-markers\n"); return 3;
  }
  printf("ATLAS_KERNEL_PASS name=%s device_cycles=%" PRIu32 "\n", PAYLOAD_NAME, device_cycles);
  return 0;
}
