/* Public queue-depth/controlled-history validation. No compiler/reference kernel.
 * Full16x16 i8 WS, raw i32 output. First-use means this arm's first execution
 * after preparation; it does NOT mean an independently flushed CPU cache. */
typedef unsigned long u64;
typedef signed char i8;
typedef int i32;
extern volatile u64 tohost;
#define D 16
#define JOBS 11
#define ARMS 8
#define REPEATS 3
#define COMMAND(f,x,y) __asm__ volatile(".insn r 0x7b, 3, %c2, x0, %0, %1"::"r"((u64)(x)),"r"((u64)(y)),"i"(f):"memory")
static volatile i8 a[JOBS][D*D] __attribute__((aligned(64)));
static volatile i8 b[D*D] __attribute__((aligned(64)));
static struct { i32 before[16]; i32 values[JOBS*D*D]; i32 after[16]; } output __attribute__((aligned(64)));
struct result { u64 cycles; unsigned errors, checked; };
static struct result results[ARMS][REPEATS], warmup[ARMS];
static const unsigned depths[4]={1,3,7,11};
static u64 tile(unsigned row) { return row | ((u64)D<<32) | ((u64)D<<48); }
static void complete(void) { __asm__ volatile("fence":::"memory"); }
static void put(char c) {tohost=0x0101000000000000ul|(unsigned char)c;while(tohost){}}
static void text(const char *s) {while(*s)put(*s++);}
static void decimal(u64 n) {char buf[24];unsigned i=0;do{buf[i++]=(char)('0'+n%10);n/=10;}while(n);while(i)put(buf[--i]);}
__attribute__((noinline)) static void execute(unsigned jobs, unsigned fenced) {
 for(unsigned n=0;n<jobs;n++) {
  COMMAND(6,tile(4096),tile(0x80000000u+n*D));
  COMMAND(4,tile(n*D),tile(0xffffffffu));
  if(fenced)complete();
 }
 complete();
 for(unsigned n=0;n<jobs;n++)COMMAND(3,output.values+n*D*D,tile(0xa0000000u+n*D));
 complete();
}
static void reset_outputs(void) {
 for(unsigned i=0;i<JOBS*D*D;i++)output.values[i]=0x35353535;
 for(unsigned i=0;i<16;i++){output.before[i]=0x12345678;output.after[i]=0x76543210;}
 complete();
}
static void prepare(void) {
 const u64 one=0x3f800000ul;
 for(unsigned n=0;n<JOBS;n++)for(unsigned i=0;i<D*D;i++)a[n][i]=(i8)((int)((i*3+n*7+1)%9)-4);
 for(unsigned i=0;i<D*D;i++)b[i]=(i8)((int)((i*11+2)%7)-3);
 COMMAND(0,6,(one<<32)|64);COMMAND(0,2,(one<<32)|64);
 COMMAND(0,1ul|(1ul<<8)|(D<<16)|(one<<32),D);
 for(unsigned n=0;n<JOBS;n++)COMMAND(2,a[n],tile(n*D));
 COMMAND(2,b,tile(4096));
 COMMAND(0,4ul|(1ul<<16)|(one<<32),1ul<<48);
 complete();
}
static void check(unsigned jobs, struct result *r) {
 r->errors=0;r->checked=0;
 for(unsigned n=0;n<jobs;n++)for(unsigned i=0;i<D;i++)for(unsigned j=0;j<D;j++) {
  i32 expected=0;for(unsigned k=0;k<D;k++)expected+=(i32)a[n][i*D+k]*(i32)b[k*D+j];
  r->errors+=output.values[n*D*D+i*D+j]!=expected;r->checked++;
 }
 for(unsigned i=jobs*D*D;i<JOBS*D*D;i++)r->errors+=output.values[i]!=0x35353535;
 for(unsigned i=0;i<16;i++){r->errors+=output.before[i]!=0x12345678;r->errors+=output.after[i]!=0x76543210;}
}
__attribute__((noinline)) static void measure(unsigned jobs, unsigned fenced, struct result *r) {
 u64 begin,end;
 reset_outputs();
 __asm__ volatile(".globl capacity_begin\ncapacity_begin:\ncsrr %0,mcycle":"=r"(begin)::"memory");
 execute(jobs,fenced);
 complete();
 __asm__ volatile(".globl capacity_end\ncapacity_end:\ncsrr %0,mcycle":"=r"(end)::"memory");
 r->cycles=end-begin;check(jobs,r);
}
int main(void) {
 unsigned errors=0;
 for(unsigned arm=0;arm<ARMS;arm++) {
  unsigned jobs=depths[arm/2],fenced=arm%2;
  prepare();
  measure(jobs,fenced,&results[arm][0]);
  /* Exact unmeasured work plus all-output checking before EACH warm sample.
   * This controls the history sequence, not physical cache replacement state. */
  for(unsigned rep=1;rep<REPEATS;rep++) {
   reset_outputs();execute(jobs,fenced);check(jobs,&warmup[arm]);errors+=warmup[arm].errors;
   measure(jobs,fenced,&results[arm][rep]);
  }
  for(unsigned rep=0;rep<REPEATS;rep++)errors+=results[arm][rep].errors;
 }
 for(unsigned arm=0;arm<ARMS;arm++)for(unsigned rep=0;rep<REPEATS;rep++) {
  struct result *r=&results[arm][rep];text("CAPACITY arm=");decimal(arm);text(" rep=");decimal(rep);
  text(" jobs=");decimal(depths[arm/2]);text(" fenced=");decimal(arm%2);
  text(" cycles=");decimal(r->cycles);text(" checked=");decimal(r->checked);text(" errors=");decimal(r->errors);text(" guards=32\n");
 }
 text("CAPACITY_TOTAL errors=");decimal(errors);text(" warmup_runs=16\n");
 return errors?1:0;
}
