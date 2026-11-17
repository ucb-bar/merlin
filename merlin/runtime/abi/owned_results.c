/* Invocation-owned allocation for explicitly selected returned MLIR results.
 * Generic MemRefToLLVM callbacks use the upstream malloc/aligned_alloc/free ABI.
 * This host runtime is selected by the caller; ordinary model linking is unchanged.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Allocation {
  struct Allocation *next;
  void *raw;
  void *payload;
  size_t bytes;
  size_t charged;
} Allocation;

typedef struct {
  Allocation *head;
  size_t live_bytes, peak_bytes, live_count, total_count;
  size_t byte_limit, count_limit;
  int active;
} Invocation;

/* Each native caller thread owns its complete synchronous invocation. */
static _Thread_local Invocation current;

static _Noreturn void refuse(const char *reason) {
  fputs("Merlin owned results: ", stderr);
  fputs(reason, stderr);
  fputc('\n', stderr);
  abort();
}

int merlin_owned_results_begin(size_t byte_limit, size_t count_limit) {
  if (current.active || !byte_limit || !count_limit)
    return 0;
  current = (Invocation){.byte_limit = byte_limit,
                         .count_limit = count_limit,
                         .active = 1};
  return 1;
}

static void *allocate(size_t alignment, size_t bytes) {
  if (!current.active)
    refuse("allocation outside its invocation");
  if (!alignment || (alignment & (alignment - 1)))
    refuse("invalid allocation alignment");
  if (alignment < _Alignof(max_align_t))
    alignment = _Alignof(max_align_t);
  /* Empty memrefs still receive an owned, uniquely identifiable byte. */
  size_t payload_bytes = bytes ? bytes : 1;
  if (payload_bytes > SIZE_MAX - sizeof(Allocation) ||
      payload_bytes + sizeof(Allocation) > SIZE_MAX - (alignment - 1))
    refuse("allocation size overflow");
  size_t charged = payload_bytes + sizeof(Allocation) + alignment - 1;
  if (current.total_count >= current.count_limit ||
      charged > current.byte_limit - current.live_bytes)
    refuse("invocation allocation budget exceeded");
  void *raw = malloc(charged);
  if (!raw)
    refuse("allocation failed");
  Allocation *item = raw;
  uintptr_t start = (uintptr_t)raw + sizeof(Allocation);
  if (start > UINTPTR_MAX - (alignment - 1))
    refuse("allocation address overflow");
  void *payload = (void *)((start + alignment - 1) & ~(uintptr_t)(alignment - 1));
  *item = (Allocation){current.head, raw, payload, payload_bytes, charged};
  current.head = item;
  current.live_bytes += charged;
  if (current.live_bytes > current.peak_bytes)
    current.peak_bytes = current.live_bytes;
  ++current.live_count;
  ++current.total_count;
  return payload;
}

/* Hidden definitions prevent another loaded model from interposing its arena. */
__attribute__((visibility("hidden")))
void *_mlir_memref_to_llvm_alloc(size_t bytes) {
  return allocate(_Alignof(max_align_t), bytes);
}

__attribute__((visibility("hidden")))
void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t bytes) {
  return allocate(alignment, bytes);
}

__attribute__((visibility("hidden")))
void _mlir_memref_to_llvm_free(void *pointer) {
  if (!current.active)
    refuse("deallocation outside its invocation");
  if (!pointer)
    return;
  Allocation **slot = &current.head;
  while (*slot && (*slot)->payload != pointer)
    slot = &(*slot)->next;
  if (!*slot)
    refuse("unknown or already released allocation");
  Allocation *item = *slot;
  *slot = item->next;
  current.live_bytes -= item->charged;
  --current.live_count;
  free(item->raw);
}

size_t merlin_owned_results_capacity(void *pointer) {
  if (!current.active)
    return 0;
  for (Allocation *item = current.head; item; item = item->next)
    if (item->payload == pointer)
      return item->bytes;
  return 0;
}

size_t merlin_owned_results_live_count(void) { return current.live_count; }
size_t merlin_owned_results_peak_bytes(void) { return current.peak_bytes; }
size_t merlin_owned_results_total_count(void) { return current.total_count; }

/* Free tracked owners, never a pointer supplied by an output descriptor. */
void merlin_owned_results_end(void) {
  if (!current.active)
    refuse("release outside its invocation");
  while (current.head) {
    Allocation *item = current.head;
    current.head = item->next;
    free(item->raw);
  }
  current = (Invocation){0};
}
