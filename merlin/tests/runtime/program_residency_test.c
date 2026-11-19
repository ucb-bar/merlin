#include "program_residency.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
typedef struct { uint64_t epoch; size_t loads, selects, discards; int busy, fail_load, fail_select; size_t capacity, alignment; } endpoint;
static int query(void *p,merlin_program_layout *l) {
  endpoint *e=p; if(e->busy)return 9;
  *l=(merlin_program_layout){4,e->capacity?e->capacity:12,e->alignment?e->alignment:4,2,e->epoch}; return 0;
}
static int eligible(void *p,const void *b,size_t n) {(void)p;(void)b;return n==4;}
static int load(void *p,size_t offset,const void *b,size_t n,size_t *h) {
  endpoint *e=p; (void)b; assert(n==4 && (offset==4 || offset==8));
  ++e->loads; if(e->fail_load)return 8; *h=offset; return 0;
}
static int select_image(void *p,size_t h) {
  endpoint *e=p; assert(h==4 || h==8); ++e->selects;return e->fail_select?7:0;
}
static int discard(void *p) {endpoint *e=p;if(e->busy)return 9; ++e->epoch;++e->discards;return 0;}
static int fallback(void *p,const void *b,size_t n){(void)p;(void)b;(void)n;return 0;}
static void init(merlin_program_session *s,endpoint *e,merlin_resident_program *rows) {
  assert(!merlin_program_session_init(s,(merlin_program_target){query,eligible,load,select_image,discard,fallback,e},rows,2));
}
int main(void) {
  endpoint a={0},b={0}; merlin_program_session x,y; merlin_resident_program xr[2],yr[2];
  const unsigned char p[4]={1},equal[4]={1},different[4]={2},third[4]={3};
  init(&x,&a,xr);init(&y,&b,yr);
  assert(!merlin_program_prepare(&x,p,4));
  assert(!merlin_program_prepare(&x,different,4));
  assert(!merlin_program_prepare(&x,equal,4));
  assert(a.loads==2 && x.hits==1 && b.loads==0);
  assert(!merlin_program_prepare(&y,p,4) && b.loads==1);
  a.busy=1;size_t count=x.count;
  assert(merlin_program_prepare(&x,third,4)==9 && x.count==count);
  assert(merlin_program_session_close(&x)==9);a.busy=0;
  assert(!merlin_program_prepare(&x,third,4) && x.evictions==1);
  ++a.epoch;
  assert(!merlin_program_prepare(&x,third,4) && x.loads==4);
  a.fail_select=1;
  assert(merlin_program_prepare(&x,third,4)==7 && !x.count);
  a.fail_select=0;
  assert(!merlin_program_prepare(&x,third,4));
  a.fail_load=1;
  assert(merlin_program_prepare(&x,p,4)==8 && !x.count);
  a.fail_load=0;
  assert(!merlin_program_prepare(&x,p,4));
  assert(!merlin_program_prepare(&x,p,3) && x.fallbacks==1);
  assert(!merlin_program_prepare(&x,p,4));
  assert(!merlin_program_session_close(&x) && !x.count);
  assert(merlin_program_prepare(&x,p,4)==-1);
  b.capacity=4;
  ++b.epoch;
  assert(!merlin_program_prepare(&y,p,4) && y.fallbacks==1);
  b.capacity=12; b.alignment=3;
  assert(merlin_program_prepare(&y,p,4)==-1);
  b.alignment=4;
  assert(!merlin_program_session_close(&y));
  puts("Generic fake-target tests passed");
  return 0;
}
