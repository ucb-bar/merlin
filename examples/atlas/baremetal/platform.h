#ifndef MERLIN_EXAMPLE_ATLAS_BAREMETAL_PLATFORM_H
#define MERLIN_EXAMPLE_ATLAS_BAREMETAL_PLATFORM_H

#include "atlas_host.h"
#include "ee290_host_map.h"
/* Select the generated application at compile time. */
#ifndef HARNESS_APPLICATION_HEADER
#define HARNESS_APPLICATION_HEADER "combined/capsule_combined.h"
#endif
#ifndef HARNESS_APPLICATION_NAMESPACE
#define HARNESS_APPLICATION_NAMESPACE capsule
#endif
#include HARNESS_APPLICATION_HEADER
#define HARNESS_JOIN_(a, b) a##b
#define HARNESS_JOIN(a, b) HARNESS_JOIN_(a, b)
#define HARNESS_APP(suffix) HARNESS_JOIN(HARNESS_APPLICATION_NAMESPACE, suffix)

static atlas_host host;
static int harness_platform_bind(HARNESS_APP(_context) *context, size_t polls) {
  atlas_host_map map = {
      .imem_base = EE290_IMEM_BASE,
      .csr_base = EE290_CSR_BASE,
      .imem_capacity_bytes = EE290_IMEM_CAPACITY_BYTES};
  atlas_host_mmio mmio = {
      .read32 = atlas_host_direct_read32, .write32 = atlas_host_direct_write32};
  atlas_tile_memory_io io = {
      .write = atlas_host_direct_memory_write, .read = atlas_host_direct_memory_read};
  atlas_host_memory memory = {.coherent = 1};
  if (atlas_host_init(&host, map, mmio, memory)) return 1;
  return HARNESS_APP(_bind)(context, &host, io, polls);
}
#endif
