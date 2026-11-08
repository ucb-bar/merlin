/* Table-driven seeded anchor reduction. Integer arithmetic only. */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

static unsigned leading_bit(uint32_t value) {
  unsigned lead=0;
  while (value>>1) { value>>=1; ++lead; }
  return lead;
}

/* Standard BF16 bit fields, and the explicitly selected anchor profile:
   product exponent bias 13, headroom 7, fraction alignment 24, wrap 32. */
static uint16_t seeded_dot(const uint32_t *table,const uint8_t *a,const uint8_t *b,
                           size_t columns,size_t length,uint16_t incoming) {
  const unsigned seed_exponent=(incoming>>7)&255u;
  int anchor=(incoming&32767u)?(int)seed_exponent-127:-1024;
  for (size_t lane=0;lane<length;++lane) {
    const uint32_t term=table[((unsigned)a[lane]<<8)|b[lane*columns]];
    const unsigned exponent=(term>>8)&255u;
    const int candidate=exponent?(int)exponent-13:-1024;
    if (candidate>anchor) anchor=candidate;
  }
  anchor+=7;
  uint32_t wrapped=0;
  for (size_t lane=0;lane<length;++lane) {
    const uint32_t term=table[((unsigned)a[lane]<<8)|b[lane*columns]];
    const unsigned exponent=(term>>8)&255u;
    if (exponent) {
      const unsigned shift=(unsigned)(anchor-((int)exponent-13));
      const uint32_t magnitude=shift>=32u?0u:(uint32_t)(((uint64_t)(128u|(term&127u))<<24)>>shift);
      /* Modular addition is equivalent to the source's final signed wrap. */
      if ((term>>16)&1u) wrapped-=magnitude; else wrapped+=magnitude;
    }
  }
  const int seed_shift=anchor-((int)seed_exponent-127)-24;
  const uint32_t seed_significand=(seed_exponent?128u:0u)|(incoming&127u);
  uint32_t seed_magnitude=0;
  if (seed_shift>=0 && seed_shift<32) seed_magnitude=seed_significand>>(unsigned)seed_shift;
  else if (seed_shift<0 && seed_shift>=-31)
    seed_magnitude=(uint32_t)(((uint64_t)seed_significand<<(unsigned)-seed_shift)&UINT32_MAX);
  if (incoming&32768u) wrapped-=seed_magnitude; else wrapped+=seed_magnitude;
  const int64_t total=wrapped<=INT32_MAX?(int64_t)wrapped:(int64_t)wrapped-INT64_C(4294967296);
  if (!total) return 0;
  const unsigned sign=total<0;
  const uint32_t magnitude=(uint32_t)(sign?-total:total);
  unsigned lead=leading_bit(magnitude);
  const unsigned discard=lead>7u?lead-7u:0u;
  uint32_t significand=magnitude>>discard;
  if (discard) {
    const uint32_t residue=magnitude&((UINT32_C(1)<<discard)-1u);
    const uint32_t half=UINT32_C(1)<<(discard-1u);
    if (residue>half || (residue==half && (significand&1u))) ++significand;
  }
  if (significand>=256u) { significand>>=1; ++lead; }
  const int output_exponent=(int)lead+anchor-31+127;
  if (output_exponent<=0) return 0;
  if (output_exponent>=255) return (uint16_t)((sign<<15)|0x7f7fu);
  return (uint16_t)((sign<<15)|((unsigned)output_exponent<<7)|(significand&127u));
}

int merlin_seeded_anchor_dot(const uint32_t *table,const uint8_t *a,const uint8_t *b,
                             const uint16_t *seed,uint16_t *output,
                             size_t rows,size_t length,size_t columns) {
  if (!table || !a || !b || !seed || !output || !rows || !length || !columns
      || rows>SIZE_MAX/columns || rows>SIZE_MAX/length || length>SIZE_MAX/columns) return -1;
  for (size_t row=0;row<rows;++row) for (size_t column=0;column<columns;++column)
    output[row*columns+column]=seeded_dot(table,a+row*length,b+column,columns,length,seed[row*columns+column]);
  return 0;
}

int merlin_product_table_lookup(const uint32_t *table,const uint8_t *a,const uint8_t *b,
                                uint32_t *output,size_t count) {
  if (!table || !a || !b || !output) return -1;
  for (size_t i=0;i<count;++i) output[i]=table[((unsigned)a[i]<<8)|b[i]];
  return 0;
}
