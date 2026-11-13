#ifndef MERLIN_ATLAS_PROGRAM_SESSION_H
#define MERLIN_ATLAS_PROGRAM_SESSION_H
/* Qualification-only composition adapter, used by the documented residency recipe.
 * Generic policy is never in Atlas primitives. */
#include "program_residency.h"
#include "atlas_program_slots.h"
typedef struct {
  atlas_program_slots slots;
  merlin_program_session session;
} merlin_atlas_program_session;
static int merlin_atlas_query(void *p, merlin_program_layout *layout) {
  merlin_atlas_program_session *s=p;
  layout->alignment=4;
  return atlas_slots_query(&s->slots,&layout->first,&layout->capacity,
                            &layout->max_entries,&layout->generation);
}
static int merlin_atlas_eligible(void *p,const void *bytes,size_t count) {
  (void)p; return atlas_slots_eligible(bytes,count);
}
static int merlin_atlas_load(void *p,size_t offset,const void *bytes,size_t count,size_t *handle) {
  return atlas_slots_load(&((merlin_atlas_program_session *)p)->slots,offset,bytes,count,handle);
}
static int merlin_atlas_select(void *p,size_t handle) {
  return atlas_slots_select(&((merlin_atlas_program_session *)p)->slots,handle);
}
static int merlin_atlas_discard(void *p) {
  return atlas_slots_discard(&((merlin_atlas_program_session *)p)->slots);
}
static int merlin_atlas_fallback(void *p,const void *bytes,size_t count) {
  if (count&3u || (uintptr_t)bytes&3u) return ATLAS_HOST_INVALID;
  return atlas_host_load(((merlin_atlas_program_session *)p)->slots.host,bytes,count/4);
}
static atlas_host_result merlin_atlas_prepare(void *p,const uint32_t *words,size_t count) {
  if (count>SIZE_MAX/4) return ATLAS_HOST_INVALID;
  int r=merlin_program_prepare(&((merlin_atlas_program_session *)p)->session,words,count*4);
  return r<0 ? ATLAS_HOST_INVALID : (atlas_host_result)r;
}
/* Caller exclusively owns the endpoint and explicitly invokes merlin_atlas_prepare.
 * Platform resets must use atlas_slots_reset_completed(&s->slots).
 * Opt in only after independent relocation/execution gates; this adapter
 * tests/control-writes primitives and confers no execution proof. */
static inline int merlin_atlas_program_session_open(merlin_atlas_program_session *s,
    atlas_host *host,atlas_program_slot *slots,merlin_resident_program *entries,size_t capacity) {
  if (!s || !host || host->active) return -1;
  int r=atlas_slots_init(&s->slots,host,slots,capacity);
  if (r) return r;
  merlin_program_target target={merlin_atlas_query,merlin_atlas_eligible,merlin_atlas_load,
    merlin_atlas_select,merlin_atlas_discard,merlin_atlas_fallback,s};
  return merlin_program_session_init(&s->session,target,entries,capacity);
}
static inline int merlin_atlas_program_session_close(merlin_atlas_program_session *s) {
  if (!s) return -1;
  return merlin_program_session_close(&s->session);
}
#endif
