#include "owned_result_readback.h"
#include <string.h>

static int bytes(MerlinOwnedReadback *sink, const void *p, size_t n) {
  return n ? sink->write(sink->context, p, n) : 0;
}
static int text(MerlinOwnedReadback *sink, const char *value) {
  return bytes(sink, value, strlen(value));
}
static int number(MerlinOwnedReadback *sink, uint64_t value) {
  char data[20]; size_t n=0;
  do { data[n++]=(char)('0'+value%10); value/=10; } while(value);
  for(size_t i=0;i<n/2;i++) {char c=data[i];data[i]=data[n-i-1];data[n-i-1]=c;}
  return bytes(sink,data,n);
}
static int frame(MerlinOwnedReadback *sink, int kind, size_t index, int shape,
                 size_t width, int signed_words, const void *data, size_t n) {
  if(text(sink,"OUT_BIN_BEGIN v1 ") || text(sink,kind==MERLIN_OWNED_INPUT ? "in" : "out") ||
     number(sink,index) || (shape && text(sink,"_shape")) || text(sink," 1 ") ||
     number(sink,n/width) || text(sink," ") || number(sink,width) ||
     text(sink,signed_words ? " s " : " u ") || number(sink,n) || text(sink,"\n")) return 1;
  if(bytes(sink,data,n)) return 1;
  uint64_t hash=UINT64_C(0xcbf29ce484222325);
  for(size_t i=0;i<n;i++) hash=(hash^((const unsigned char *)data)[i])*UINT64_C(0x100000001b3);
  char checksum[16];
  for(size_t i=0;i<16;i++) checksum[i]="0123456789abcdef"[(hash>>(60-4*i))&15];
  return text(sink,"OUT_BIN_END v1 ") || bytes(sink,checksum,16) || text(sink,"\n");
}

int merlin_owned_readback_observe(void *context, int kind, size_t index, const char *dtype,
                                 size_t rank, const intptr_t *shape, const void *data, size_t n) {
  MerlinOwnedReadback *sink=context;
  if(!sink || !sink->write || !dtype || rank>MERLIN_OWNED_RESULT_MAX_RANK ||
     (rank && !shape) || (n && !data) || (kind!=MERLIN_OWNED_INPUT && kind!=MERLIN_OWNED_OUTPUT)) return 1;
  size_t width=0; int signed_words=0;
  if(!strcmp(dtype,"i1")) width=1;
  else if(!strcmp(dtype,"i8")) {width=1;signed_words=1;}
  else if(!strcmp(dtype,"i16")) {width=2;signed_words=1;}
  else if(!strcmp(dtype,"i32")) {width=4;signed_words=1;}
  else if(!strcmp(dtype,"i64")) {width=8;signed_words=1;}
  else if(!strcmp(dtype,"f16") || !strcmp(dtype,"bf16")) width=2;
  else if(!strcmp(dtype,"f32")) width=4;
  else if(!strcmp(dtype,"f64")) width=8;
  else if(!strcmp(dtype,"f8E4M3FN") || !strcmp(dtype,"f8E4M3FNUZ") ||
          !strcmp(dtype,"f8E5M2") || !strcmp(dtype,"f8E5M2FNUZ")) width=1;
  if(!width || n%width) return 1;
  uint64_t dimensions[MERLIN_OWNED_RESULT_MAX_RANK];
  size_t count=1; int empty=0;
  for(size_t axis=0;axis<rank;axis++) {
    if(shape[axis]<0) return 1;
    dimensions[axis]=(uint64_t)shape[axis]; empty|=shape[axis]==0;
  }
  if(empty) count=0;
  else for(size_t axis=0;axis<rank;axis++) {
    if((size_t)shape[axis]>SIZE_MAX/count) return 1;
    count*=(size_t)shape[axis];
  }
  if(count!=n/width) return 1;
  return frame(sink,kind,index,1,8,0,dimensions,rank*8) ||
         frame(sink,kind,index,0,width,signed_words,data,n);
}
