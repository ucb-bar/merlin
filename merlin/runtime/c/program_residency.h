#ifndef MERLIN_PROGRAM_RESIDENCY_H
#define MERLIN_PROGRAM_RESIDENCY_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Policy uses byte sizes/offsets; target supplies all layout restrictions.
 * Immutable payload arrays remain alive until session invalidation/closure.
 * One session exclusively owns one endpoint. All operations are synchronous.
 * Negative statuses are generic refusal; positive statuses are target errors.
 * Payload identity only controls physical image reuse. The caller must still
 * authenticate every request/manifest and retain its original bindings.
 * Targets keep layout stable within a generation and refuse mutations while
 * active. Failed load/select must leave no selected image launchable. */
typedef struct {
  size_t first, capacity, alignment, max_entries;
  uint64_t generation;
} merlin_program_layout;
typedef struct {
  int (*query)(void *, merlin_program_layout *);
  int (*eligible)(void *, const void *, size_t);
  int (*load)(void *, size_t, const void *, size_t, size_t *);
  int (*select)(void *, size_t);
  int (*discard)(void *);
  int (*fallback)(void *, const void *, size_t);
  void *context;
} merlin_program_target;
typedef struct {
  const unsigned char *payload;
  size_t bytes, handle;
} merlin_resident_program;
typedef struct {
  merlin_program_target target;
  merlin_resident_program *entries;
  size_t entry_capacity, count, next;
  uint64_t generation;
  uint64_t loads, hits, evictions, fallbacks;
  int closed;
} merlin_program_session;

static inline int merlin_program_session_init(merlin_program_session *s,
    merlin_program_target target, merlin_resident_program *entries, size_t capacity) {
  if (!s || !entries || !capacity || capacity > SIZE_MAX / sizeof(*entries) || !target.query || !target.eligible ||
      !target.load || !target.select || !target.discard || !target.fallback) return -1;
  *s=(merlin_program_session){0};
  s->target=target; s->entries=entries; s->entry_capacity=capacity;
  return 0;
}
static inline int merlin_program_layout_valid(const merlin_program_layout *layout) {
  return layout->alignment && !(layout->alignment & (layout->alignment - 1)) &&
         layout->first <= layout->capacity && layout->max_entries;
}
static inline void merlin_program_forget(merlin_program_session *s,
                                        const merlin_program_layout *layout) {
  s->count=0; s->next=layout->first; s->generation=layout->generation;
}
static inline int merlin_program_prepare(merlin_program_session *s,
                                         const void *payload, size_t bytes) {
  if (!s || s->closed || !payload || !bytes) return -1;
  merlin_program_layout layout;
  int result=s->target.query(s->target.context,&layout);
  if (result) return result; /* Active/nonhalted endpoints cannot mutate policy. */
  if (!merlin_program_layout_valid(&layout)) return -1;
  if (s->generation!=layout.generation || !s->count) {
    result=s->target.discard(s->target.context);
    if (result) return result;
    result=s->target.query(s->target.context,&layout);
    if (result) return result;
    if (!merlin_program_layout_valid(&layout)) { s->count=0; return -1; }
    merlin_program_forget(s,&layout);
  }
  int eligible=s->target.eligible(s->target.context,payload,bytes);
  if (eligible<0) return eligible;
  if (!eligible || bytes>layout.capacity-layout.first) {
    result=s->target.discard(s->target.context);
    if (result) return result;
    s->count=0;
    ++s->fallbacks;
    return s->target.fallback(s->target.context,payload,bytes);
  }
  for (size_t i=0;i<s->count;++i) {
    if (s->entries[i].bytes==bytes && !memcmp(s->entries[i].payload,payload,bytes)) {
      result=s->target.select(s->target.context,s->entries[i].handle);
      if (!result) ++s->hits;
      else s->count=0;
      return result;
    }
  }
  if (s->next>SIZE_MAX-(layout.alignment-1)) return -1;
  size_t offset=(s->next+layout.alignment-1)&~(layout.alignment-1);
  if (s->count==s->entry_capacity || s->count==layout.max_entries ||
      offset>layout.capacity || bytes>layout.capacity-offset) {
    result=s->target.discard(s->target.context);
    if (result) return result;
    ++s->evictions;
    result=s->target.query(s->target.context,&layout);
    if (result) { s->count=0; return result; }
    if (!merlin_program_layout_valid(&layout)) { s->count=0; return -1; }
    merlin_program_forget(s,&layout);
    if (s->next>SIZE_MAX-(layout.alignment-1)) return -1;
    offset=(s->next+layout.alignment-1)&~(layout.alignment-1);
    if (offset>layout.capacity || bytes>layout.capacity-offset) return -1;
  }
  size_t handle=0;
  result=s->target.load(s->target.context,offset,payload,bytes,&handle);
  if (result) { s->count=0; return result; }
  result=s->target.select(s->target.context,handle);
  if (result) { s->count=0; return result; }
  s->entries[s->count++]=(merlin_resident_program){payload,bytes,handle};
  s->next=offset+bytes; ++s->loads;
  return 0;
}
static inline int merlin_program_session_close(merlin_program_session *s) {
  if (!s || s->closed) return -1;
  int result=s->target.discard(s->target.context);
  if (!result) { s->count=0; s->next=0; s->closed=1; }
  return result;
}
#endif
