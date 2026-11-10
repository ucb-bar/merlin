/* Qualification-only composition: generic Merlin policy and Atlas primitives. */
#include "program_session.h"
#include <stdlib.h>

typedef struct {
  atlas_host host;
  merlin_atlas_program_session session;
  atlas_program_slot *slots;
  merlin_resident_program *entries;
} residency_bridge;

void *residency_open(atlas_host_read32 read32, atlas_host_write32 write32,
                     size_t capacity) {
  if (!capacity || capacity > 4096) return NULL;
  residency_bridge *bridge = calloc(1, sizeof(*bridge));
  if (!bridge) return NULL;
  bridge->slots = calloc(capacity, sizeof(*bridge->slots));
  bridge->entries = calloc(capacity, sizeof(*bridge->entries));
  int result = !bridge->slots || !bridge->entries;
  if (!result)
    result = atlas_host_init(&bridge->host,
        (atlas_host_map){0x20000, 0x40000, 0x20000},
        (atlas_host_mmio){read32, write32, NULL},
        (atlas_host_memory){.coherent = 1});
  if (!result)
    result = merlin_atlas_program_session_open(&bridge->session, &bridge->host,
        bridge->slots, bridge->entries, capacity);
  if (result) {
    free(bridge->slots); free(bridge->entries); free(bridge);
    return NULL;
  }
  return bridge;
}

int residency_prepare(void *value, const uint32_t *words, size_t count) {
  residency_bridge *bridge = value;
  return merlin_atlas_prepare(&bridge->session, words, count);
}

uint64_t residency_counter(void *value, unsigned index) {
  residency_bridge *bridge = value;
  switch (index) {
  case 0: return bridge->session.session.loads;
  case 1: return bridge->session.session.hits;
  case 2: return bridge->session.session.evictions;
  case 3: return bridge->session.session.fallbacks;
  case 4: return bridge->host.imem_words_written;
  default: return UINT64_MAX;
  }
}

size_t residency_slot_count(void *value) {
  return ((residency_bridge *)value)->session.slots.count;
}
size_t residency_slot_value(void *value, size_t index, unsigned field) {
  residency_bridge *bridge = value;
  if (index >= bridge->session.slots.count) return SIZE_MAX;
  return field ? bridge->slots[index].bytes : bridge->slots[index].offset;
}
int residency_close(void *value) {
  residency_bridge *bridge = value;
  int result = merlin_atlas_program_session_close(&bridge->session);
  if (!result) {
    free(bridge->slots); free(bridge->entries); free(bridge);
  }
  return result;
}
