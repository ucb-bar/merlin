/* Bounded, target-independent memory sink for complete binary output frames.
 *
 * A caller appends the exact frame bytes (including arbitrary binary payload)
 * and publishes the used length only after every declared frame is complete.
 * finish() does not parse frames or prove their source/value roster: those are
 * caller and host-reader obligations. Failed calls revoke publication.
 * This sink does not select outputs, inspect reference values, or change the
 * existing serial output codecs. The owner of the arena must establish that
 * its guest-to-host readback path observes the completed bytes coherently.
 */
#ifndef MERLIN_OUT_BIN_MEMORY_H
#define MERLIN_OUT_BIN_MEMORY_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  unsigned char *arena;
  size_t capacity;
  size_t used;
  volatile uint64_t *published_used;
  int valid;
  int finished;
} merlin_out_bin_memory;

static inline void merlin_out_bin_memory_invalidate(merlin_out_bin_memory *sink) {
  sink->valid = 0;
  if (sink->published_used)
    *sink->published_used = 0;
}

static inline void merlin_out_bin_memory_init(merlin_out_bin_memory *sink,
                                              unsigned char *arena, size_t capacity,
                                              volatile uint64_t *published_used) {
  if (!sink)
    return;
  sink->arena = arena;
  sink->capacity = capacity;
  sink->used = 0;
  sink->published_used = published_used;
  sink->finished = 0;
  sink->valid = arena != 0 && published_used != 0 && capacity != 0 &&
                (uint64_t)capacity == capacity;
  if (published_used)
    *published_used = 0;
}

static inline int merlin_out_bin_memory_append(merlin_out_bin_memory *sink,
                                               const void *data, size_t length) {
  if (!sink || !sink->valid || sink->finished ||
      sink->used > sink->capacity || (length && !data) ||
      length > sink->capacity - sink->used) {
    if (sink)
      merlin_out_bin_memory_invalidate(sink);
    return 0;
  }
  const unsigned char *source = (const unsigned char *)data;
  const size_t used = sink->used;
  unsigned char *destination = sink->arena + used;
  /* A source alias into the destination would need memmove semantics. Refuse
   * it instead: callers stage model values separately from this arena. */
  if (length) {
    uintptr_t source_address = (uintptr_t)source;
    uintptr_t destination_address = (uintptr_t)destination;
    uintptr_t control_address = (uintptr_t)sink;
    uintptr_t publication_address = (uintptr_t)sink->published_used;
    /* Hoisted cursor/state must remain stable while copying this span. Refuse
     * storage that would overwrite the sink or its publication word. */
    if ((destination_address <= control_address &&
         control_address - destination_address < length) ||
        (control_address < destination_address &&
         destination_address - control_address < sizeof(*sink)) ||
        (sink->published_used &&
         ((destination_address <= publication_address &&
           publication_address - destination_address < length) ||
          (publication_address < destination_address &&
           destination_address - publication_address < sizeof(*sink->published_used))))) {
      merlin_out_bin_memory_invalidate(sink);
      return 0;
    }
    if ((source_address <= destination_address &&
         destination_address - source_address < length) ||
        (destination_address < source_address &&
         source_address - destination_address < length)) {
      merlin_out_bin_memory_invalidate(sink);
      return 0;
    }
  }
  for (size_t i = 0; i < length; ++i)
    destination[i] = source[i];
  sink->used = used + length;
  return 1;
}

static inline int merlin_out_bin_memory_cstr(merlin_out_bin_memory *sink,
                                             const char *text) {
  size_t length = 0;
  if (!sink || !sink->valid || sink->finished || !text ||
      sink->used > sink->capacity) {
    if (sink)
      merlin_out_bin_memory_invalidate(sink);
    return 0;
  }
  /* A frame token cannot exceed the remaining arena. Avoid an unbounded
   * strlen on malformed caller input. */
  while (length <= sink->capacity - sink->used && text[length])
    ++length;
  if (length > sink->capacity - sink->used) {
    merlin_out_bin_memory_invalidate(sink);
    return 0;
  }
  return merlin_out_bin_memory_append(sink, text, length);
}

static inline int merlin_out_bin_memory_decimal(merlin_out_bin_memory *sink,
                                                uint64_t value) {
  char reversed[20];
  char digits[20];
  size_t length = 0;
  do {
    reversed[length++] = (char)('0' + value % 10u);
    value /= 10u;
  } while (value);
  for (size_t i = 0; i < length; ++i)
    digits[i] = reversed[length - i - 1];
  return merlin_out_bin_memory_append(sink, digits, length);
}

static inline int merlin_out_bin_memory_hex16(merlin_out_bin_memory *sink,
                                              uint64_t value) {
  static const char hex[] = "0123456789abcdef";
  char digits[16];
  for (unsigned i = 0; i < 16; ++i)
    digits[i] = hex[(value >> (60u - 4u * i)) & 15u];
  return merlin_out_bin_memory_append(sink, digits, sizeof(digits));
}

static inline int merlin_out_bin_memory_finish(merlin_out_bin_memory *sink) {
  if (!sink || !sink->valid || sink->finished || sink->used == 0 ||
      sink->used > sink->capacity || !sink->published_used ||
      (uint64_t)sink->used != sink->used) {
    if (sink)
      merlin_out_bin_memory_invalidate(sink);
    return 0;
  }
  sink->finished = 1;
  /* Keep the symbol-visible length store observable; the provider supplies
   * any stronger compiler/cache fence its selected transport requires. */
  *sink->published_used = (uint64_t)sink->used;
  return 1;
}

#endif
