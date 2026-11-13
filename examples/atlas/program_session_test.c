#include "program_session.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>
typedef struct { uint32_t words[32]; size_t writes; int halted; } device;
static uint32_t read32(void *p,uintptr_t address) {
  device *d=p; (void)address; return d->halted ? 5u : 0u;
}
static void write32(void *p,uintptr_t address,uint32_t word) {
  device *d=p;
  assert(address>=0x20000 && address<0x20080);
  d->words[(address-0x20000)/4]=word; ++d->writes;
}
int main(void) {
  const uint32_t a[]={0x00100093u,0x00000073u};
  const uint32_t same_a[]={0x00100093u,0x00000073u};
  const uint32_t b[]={0x00200093u,0x00000073u};
  const uint32_t c[]={0x00300093u,0x00000073u};
  const uint32_t absolute[]={0x00000017u,0x00000073u};
  const uint32_t cfg_body[]={0x0040006fu,0x00000013u,0x00000073u,0x00000013u};
  const uint32_t two_halts[]={0x00000073u,0x00000073u};
  assert(atlas_slots_eligible(cfg_body,sizeof cfg_body));
  assert(!atlas_slots_eligible(two_halts,sizeof two_halts));
  device d={.halted=1}; atlas_host host;
  assert(atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,128},
     (atlas_host_mmio){read32,write32,&d},(atlas_host_memory){.coherent=1})==0);
  atlas_program_slot slots[2]; merlin_resident_program entries[2];
  merlin_atlas_program_session session;
  assert(!merlin_atlas_program_session_open(&session,&host,slots,entries,2));
  /* Only this explicit qualification entry invokes the generic policy. */
  assert(!merlin_atlas_prepare(&session,a,2));
  assert(!merlin_atlas_prepare(&session,b,2));
  assert(!merlin_atlas_prepare(&session,same_a,2));
  assert(session.session.loads==2 && session.session.hits==1 && slots[0].offset==8 && slots[1].offset==16);
  assert(d.words[2]==a[0] && d.words[4]==b[0]);
  size_t writes=d.writes, count=session.session.count;
  host.active=1;
  assert(merlin_atlas_prepare(&session,c,2)==ATLAS_HOST_BUSY);
  assert(d.writes==writes && session.session.count==count);
  assert(merlin_atlas_program_session_close(&session)==ATLAS_HOST_BUSY);
  host.active=0; d.halted=0;
  assert(merlin_atlas_prepare(&session,c,2)==ATLAS_HOST_NOT_READY);
  assert(d.writes==writes); d.halted=1;
  size_t stale=entries[0].handle;
  assert(!merlin_atlas_prepare(&session,c,2));
  assert(session.session.evictions==1 && session.session.loads==3);
  assert(atlas_slots_select(&session.slots,stale)==ATLAS_HOST_INVALID);
  assert(!host.program_loaded);
  assert(!merlin_atlas_prepare(&session,c,2));
  assert(!atlas_slots_reset_completed(&session.slots));
  uint64_t loads=session.session.loads;
  writes=d.writes;
  assert(!merlin_atlas_prepare(&session,c,2));
  assert(d.writes>writes && session.session.loads==loads+1);
  assert(!merlin_atlas_prepare(&session,absolute,2));
  assert(session.session.fallbacks==1 && d.words[0]==absolute[0]);
  assert(!merlin_atlas_prepare(&session,a,2));
  assert(session.session.loads==loads+2);
  /* Failed physical replacement clears launchability; next generic call recovers. */
  size_t handle;
  assert(atlas_slots_load(&session.slots,8,b,sizeof b,&handle)==ATLAS_HOST_INVALID);
  assert(!host.program_loaded);
  assert(!merlin_atlas_prepare(&session,a,2));
  /* Baseline host loading stays independent and invalidates retained handles. */
  loads=session.session.loads;
  assert(!atlas_host_ensure_loaded(&host,b,2));
  assert(d.words[0]==b[0] && session.session.loads==loads);
  assert(!merlin_atlas_prepare(&session,a,2));
  assert(session.session.loads==loads+1);
  assert(!merlin_atlas_program_session_close(&session));
  assert(!host.program_loaded);
  puts("Atlas primitive + generic session tests passed; no instruction execution modeled");
  return 0;
}
