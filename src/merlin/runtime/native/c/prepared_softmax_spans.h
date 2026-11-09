#ifndef MERLIN_PREPARED_SOFTMAX_SPANS_H
#define MERLIN_PREPARED_SOFTMAX_SPANS_H
#include <stddef.h>
#include <stdint.h>
/* Private producer evidence, not a public array admission API. The emitter
 * proves mask gathering validates each byte and every successful bound writer
 * produces finite ordered f32 endpoints. Each tile is recorded AFTER its full
 * row-strided copy. A fresh local epoch and exact addresses bind this evidence
 * to one synchronous head evaluation; no persistent cache exists. */
typedef struct {
 const float *lo,*hi;
 const unsigned char *mask;
 const void *epoch;
 size_t rows,chunk,next_tile;
 int valid;
} merlin_softmax_produced_spans;
static inline merlin_softmax_produced_spans merlin_softmax_spans_begin(
 const float *lo,const float *hi,const unsigned char *mask,
 size_t rows,size_t chunk,const void *epoch) {
 if(!lo||!hi||!mask||!epoch||!rows||!chunk||chunk>SIZE_MAX/2||rows>SIZE_MAX/(2*chunk))
  return (merlin_softmax_produced_spans){0};
 return (merlin_softmax_produced_spans){lo,hi,mask,epoch,rows,chunk,0,1};
}
/* Called only on a compiler-proved successful bound-producer edge, after the
 * corresponding complete private copy. This operation cannot inspect or admit
 * arbitrary data. The source-bound emitter is its sole trusted constructor. */
static inline int merlin_softmax_spans_record_tile(
 merlin_softmax_produced_spans *p,const float *lo,const float *hi,
 const unsigned char *mask,size_t rows,size_t chunk,size_t tile,const void *epoch) {
 if(!p||!p->valid)return 0;
 if(p->lo!=lo||p->hi!=hi||p->mask!=mask||p->epoch!=epoch||
    p->rows!=rows||p->chunk!=chunk||tile!=p->next_tile||tile>=2){p->valid=0;return 0;}
 ++p->next_tile;return 1;
}
static inline int merlin_softmax_spans_consume(
 merlin_softmax_produced_spans *p,const float *lo,const float *hi,
 const unsigned char *mask,size_t count,const void *epoch) {
 if(!p)return 0;
 int valid=p->valid&&p->next_tile==2&&p->lo==lo&&p->hi==hi&&p->mask==mask&&
   p->epoch==epoch&&p->rows&&p->chunk&&p->chunk<=SIZE_MAX/2&&
   p->rows<=SIZE_MAX/(2*p->chunk)&&count==p->rows*2*p->chunk;
 p->valid=0;return valid;
}
#endif
