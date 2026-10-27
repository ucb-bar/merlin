
#include <stdint.h>
#include <stddef.h>
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)));
static void puts_htif(const char* text) {
  static volatile uint64_t request[4] __attribute__((aligned(64)));
  size_t n=0;while(text[n])++n;
  request[0]=64;request[1]=1;request[2]=(uintptr_t)text;request[3]=n;
  asm volatile("fence rw,rw":::"memory");tohost=(uintptr_t)request;
  while(!fromhost){}fromhost=0;asm volatile("fence rw,rw":::"memory");
}
__attribute__((weak)) void *memcpy(void *d,const void *s,size_t n){uint8_t *a=d;const uint8_t *b=s;for(size_t i=0;i<n;++i)a[i]=b[i];return d;}
__attribute__((weak)) void *memmove(void *d,const void *s,size_t n){uint8_t *a=d;const uint8_t *b=s;if((uintptr_t)a<(uintptr_t)b){for(size_t i=0;i<n;++i)a[i]=b[i];}else{while(n){--n;a[n]=b[n];}}return d;}
__attribute__((weak)) void *memset(void *d,int x,size_t n){uint8_t *a=d;for(size_t i=0;i<n;++i)a[i]=(uint8_t)x;return d;}
__attribute__((weak)) int memcmp(const void *a,const void *b,size_t n){const uint8_t *x=a,*y=b;for(size_t i=0;i<n;++i)if(x[i]!=y[i])return (int)x[i]-(int)y[i];return 0;}
static int guest_errno;
int *__errno(void){return &guest_errno;}
int printf(const char *format,...){(void)format;puts_htif("VENDOR_DIAGNOSTIC\n");return 0;}
__attribute__((noreturn)) void exit(int code){tohost=((uint64_t)(code?code:99)<<1)|1;for(;;){}}
static uint64_t hash(const void* p,size_t n) {
  const unsigned char* b=p;uint64_t h=1469598103934665603ULL;
  for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ULL;return h;
}
static inline uint64_t cycles(void){uint64_t v;asm volatile("fence rw,rw; csrr %0,cycle":"=r"(v)::"memory");return v;}
static char* put_u64(char* out,uint64_t value){
  char text[24];unsigned used=0;do{text[used++]=(char)('0'+value%10);value/=10;}while(value);
  while(used)*out++=text[--used];return out;
}
extern int32_t model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes);
extern const uint8_t dse_inputs[];
extern const uint8_t dse_expected[];
#define GUARD 64
static uint8_t outputs_area[144256] __attribute__((section(".noinit"),aligned(64)));
static uint8_t workspace_area[144128] __attribute__((section(".noinit"),aligned(64)));
static const uint64_t input_offsets[]={0};
static const uint64_t input_bytes[]={1600};
static const uint64_t input_hashes[]={UINT64_C(17905341960060996592)};
static const uint64_t output_offsets[]={64};
static const uint64_t output_bytes[]={144000};
static const uint64_t expected_offsets[]={0};
static const uint8_t output_kinds[]={1};  /* 0 integer bytes, 1 float32 */
static int compare(unsigned index){
  const uint8_t* actual=outputs_area+output_offsets[index];
  const uint8_t* expected=dse_expected+expected_offsets[index];
  uint64_t n=output_bytes[index];
  if(output_kinds[index]==0){
    for(uint64_t i=0;i<n;++i)if(actual[i]!=(uint8_t)(expected[i]^(i==0?BAD_ORACLE:0)))return 1;
    return 0;
  }
  for(uint64_t i=0;i<n/4;++i){
    union{uint32_t u;float f;}a,e;
    a.u=((const uint32_t*)actual)[i];e.u=((const uint32_t*)expected)[i]^(BAD_ORACLE?UINT32_C(0x40000000):0);
    /* Reject nonfinite values before tolerance bounds can overflow. */
    if((a.u&UINT32_C(0x7f800000))==UINT32_C(0x7f800000) ||
       (e.u&UINT32_C(0x7f800000))==UINT32_C(0x7f800000))return 1;
    /* BAD_ORACLE has already changed e.u. Valid configured tolerances are
       nonnegative; preserve the old fallback for other configurations. */
    if(a.u==e.u && 0x1.0c6f7a0b5ed8dp-19f>=0 && 0x1.4f8b588e368f1p-17f>=0)continue;
    float tolerance=0x1.0c6f7a0b5ed8dp-19f+0x1.4f8b588e368f1p-17f*(e.f<0?-e.f:e.f);
    if(!(a.f>=e.f-tolerance && a.f<=e.f+tolerance))return 1;
  }
  return 0;
}
int main(void){
  uint64_t arch;asm volatile("csrr %0,marchid":"=r"(arch));
  int rtl=arch==1;
  const void* inputs[1+1];void* outputs[1+1];
  for(unsigned i=0;i<1;++i)inputs[i]=dse_inputs+input_offsets[i];
  for(unsigned i=0;i<1;++i)outputs[i]=outputs_area+output_offsets[i];
  if(!rtl){memset(outputs_area,0x6d,sizeof(outputs_area));memset(workspace_area,0x6d,sizeof(workspace_area));}
  uint64_t begin=cycles();
  if(model_run(inputs,outputs,workspace_area+GUARD,144000))return 20;
  uint64_t first=cycles()-begin,second=0;
  if(first<UINT64_MAX){begin=cycles();if(model_run(inputs,outputs,workspace_area+GUARD,144000))return 20;second=cycles()-begin;}
  char line[96];char* out=line;const char* text="DSE_CYCLES first=";
  while(*text)*out++=*text++;out=put_u64(out,first);text=" second=";
  while(*text)*out++=*text++;out=put_u64(out,second);*out++='\n';*out=0;puts_htif(line);
  for(unsigned i=0;i<1;++i)if(compare(i)){puts_htif("TENSORIR_NUMERICAL_FAILURE\n");return 22;}
  if(rtl){puts_htif("DSE_PASS\n");return 0;}
  for(unsigned i=0;i<GUARD;++i)if(workspace_area[i]!=0x6d||workspace_area[sizeof(workspace_area)-1-i]!=0x6d)return 23;
  for(unsigned i=0;i<1;++i){
    const uint8_t* p=outputs_area+output_offsets[i];
    for(unsigned g=1;g<=GUARD;++g)if(p[-(int)g]!=0x6d||p[output_bytes[i]+g-1]!=0x6d)return 23;
  }
  for(unsigned i=0;i<1;++i)if(hash(dse_inputs+input_offsets[i],input_bytes[i])!=input_hashes[i])return 24;
  puts_htif("DSE_PASS\n");
  return 0;
}
