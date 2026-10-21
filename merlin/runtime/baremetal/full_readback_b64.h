/* Native-width whole-model frames; actual bytes only, no range narrowing. */
#ifndef MERLIN_FULL_READBACK_B64_H
#define MERLIN_FULL_READBACK_B64_H
#include "out_b64.h"

static char *merlin_readback_decimal(char *p, uint64_t n) {
  char digits[20]; unsigned count=0;
  do { digits[count++]=(char)('0'+n%10u); n/=10u; } while(n);
  while(count) *p++=digits[--count];
  return p;
}

static int merlin_readback_b64(const char *name, uint64_t rows, uint64_t cols,
    unsigned width, int signed_words, const unsigned char *data,
    void (*write_text)(const char *)) {
  merlin_out_b64 out;
  if(!name || !data || !write_text || !rows || !cols || rows>UINT64_MAX/cols ||
      !merlin_out_b64_width_valid(width) || (signed_words!=0 && signed_words!=1)) return 0;
  uint64_t count=rows*cols;
  if(count>UINT64_MAX/width) return 0;
  char *p=out.line;
  const char *prefix="OUT_B64_BEGIN v1 ";
  while(*prefix) *p++=*prefix++;
  unsigned length=0;
  while(*name) {
    unsigned char c=(unsigned char)*name++;
    if(++length>64 || !((c>='a'&&c<='z')||(c>='A'&&c<='Z')||
        (c>='0'&&c<='9')||c=='_')) return 0;
    *p++=(char)c;
  }
  if(!length) return 0;
  *p++=' '; p=merlin_readback_decimal(p,rows);
  *p++=' '; p=merlin_readback_decimal(p,cols);
  *p++=' '; p=merlin_readback_decimal(p,width);
  *p++=' '; *p++=signed_words?'s':'u'; *p++='\n'; *p=0;
  write_text(out.line);
  merlin_out_b64_init(&out,count,width,signed_words,write_text);
  for(uint64_t i=0;i<count;i++) {
    uint64_t word=0;
    for(unsigned b=0;b<width;b++) word|=(uint64_t)data[i*width+b]<<(8u*b);
    if(signed_words && width<8u && (word&(UINT64_C(1)<<(8u*width-1u))))
      word|=UINT64_MAX<<(8u*width);
    if(!merlin_out_b64_word(&out,word)) return 0;
  }
  if(!merlin_out_b64_finish(&out)) return 0;
  write_text("OUT_B64_END\n");
  return 1;
}
#endif
