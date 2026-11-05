/* Independent public validation protocol. No compiler or reference kernel.
 * Selected platform ISA: DIM16, i8 operands, i32 accumulator/full readout.
 * NEW source family: three distinct A tiles, one B tile, complete output checks.
 * All printing and reference computation are outside measured windows. */
typedef unsigned long u64;
typedef signed char i8;
typedef int i32;
extern volatile u64 tohost;
#define D 16
#define JOBS 3
#define REPEATS 3
#define COMMAND(f,x,y) __asm__ volatile(".insn r 0x7b, 3, %c2, x0, %0, %1"::"r"((u64)(x)),"r"((u64)(y)),"i"(f):"memory")
static volatile i8 a[JOBS][D*D] __attribute__((aligned(64)));
static volatile i8 b[D*D] __attribute__((aligned(64)));
static struct { i32 before[16]; i32 values[JOBS*D*D]; i32 after[16]; } output __attribute__((aligned(64)));
struct result { u64 cycles, checksum; unsigned errors, checked; };
static struct result results[REPEATS][7];
static volatile u64 preparation_checksum;
static u64 tile(unsigned row) { return row | ((u64)D<<32) | ((u64)D<<48); }
static void complete(void) { __asm__ volatile("fence":::"memory"); }
static void put(char c) {tohost=0x0101000000000000ul|(unsigned char)c;while(tohost){}}
static void text(const char *s) {while(*s)put(*s++);}
static void decimal(u64 n) {char buf[24];unsigned i=0;do{buf[i++]=(char)('0'+n%10);n/=10;}while(n);while(i)put(buf[--i]);}
__attribute__((noinline)) static u64 integer_delay(u64 x) {
 __asm__ volatile(".rept 128\naddi %0, %0, 1\n.endr":"+r"(x)::"memory"); return x;
}
static void compute(unsigned jobs, unsigned fenced, unsigned delayed, u64 *checksum) {
 for(unsigned n=0;n<jobs;n++) {
  if(delayed)*checksum=integer_delay(*checksum);
  COMMAND(6,tile(4096),tile(0x80000000u+n*D));
  COMMAND(4,tile(n*D),tile(0xffffffffu));
  if(fenced)complete();
 }
}
static void readout(unsigned jobs) {
 for(unsigned n=0;n<jobs;n++)COMMAND(3,output.values+n*D*D,tile(0xa0000000u+n*D));
 complete();
}
static void prepare(void) {
 const u64 one=0x3f800000ul;
 for(unsigned n=0;n<JOBS;n++)for(unsigned i=0;i<D*D;i++)a[n][i]=(i8)((int)((i*5+n*11+2)%7)-3);
 for(unsigned i=0;i<D*D;i++)b[i]=(i8)((int)((i*7+3)%5)-2);
 for(unsigned i=0;i<JOBS*D*D;i++)output.values[i]=0x35353535;
 for(unsigned i=0;i<16;i++){output.before[i]=0x12345678;output.after[i]=0x76543210;}
 COMMAND(0,6,(one<<32)|64);COMMAND(0,2,(one<<32)|64);
 COMMAND(0,1ul|(1ul<<8)|(D<<16)|(one<<32),D);
 for(unsigned n=0;n<JOBS;n++)COMMAND(2,a[n],tile(n*D));
 COMMAND(2,b,tile(4096));
 COMMAND(0,4ul|(1ul<<16)|(one<<32),1ul<<48);
 complete();
}
__attribute__((noinline)) static void measure(unsigned arm, struct result *r) {
 prepare();
 u64 checksum=7, begin, end;
 unsigned jobs=arm==0?1:JOBS;
 if(arm==4 || arm==5) {
  compute(JOBS,0,0,&checksum);complete();readout(JOBS);
  u64 sum=0;for(unsigned i=0;i<JOBS*D*D;i++)sum+=(u64)output.values[i];preparation_checksum=sum;
  if(arm==5)for(unsigned i=0;i<JOBS*D*D;i++)output.values[i]=0x35353535;
  complete();
 }
 __asm__ volatile(".globl independent_begin\nindependent_begin:\ncsrr %0,mcycle":"=r"(begin)::"memory");
 if(arm==6) {
  for(unsigned n=0;n<JOBS;n++)checksum=integer_delay(checksum);
 } else {
  if(arm<4){compute(jobs,arm==2,arm==3,&checksum);complete();}
  readout(jobs);
 }
 complete();
 __asm__ volatile(".globl independent_end\nindependent_end:\ncsrr %0,mcycle":"=r"(end)::"memory");
 r->cycles=end-begin;r->checksum=checksum;r->errors=0;r->checked=0;
 if(arm!=6)for(unsigned n=0;n<jobs;n++)for(unsigned i=0;i<D;i++)for(unsigned j=0;j<D;j++) {
  i32 expected=0;for(unsigned k=0;k<D;k++)expected+=(i32)a[n][i*D+k]*(i32)b[k*D+j];
  r->errors+=output.values[n*D*D+i*D+j]!=expected;r->checked++;
 }
 /* Also require untouched inactive outputs in the single-job and CPU arms. */
 for(unsigned i=arm==6?0:jobs*D*D;i<JOBS*D*D;i++)r->errors+=output.values[i]!=0x35353535;
 for(unsigned i=0;i<16;i++){r->errors+=output.before[i]!=0x12345678;r->errors+=output.after[i]!=0x76543210;}
 r->errors+=checksum!=(arm==3||arm==6?391ul:7ul);
}
int main(void) {
 unsigned errors=0;
 for(unsigned rep=0;rep<REPEATS;rep++)for(unsigned arm=0;arm<7;arm++){measure(arm,&results[rep][arm]);errors+=results[rep][arm].errors;}
 for(unsigned rep=0;rep<REPEATS;rep++)for(unsigned arm=0;arm<7;arm++) {
  struct result *r=&results[rep][arm];text("INDEPENDENT rep=");decimal(rep);text(" arm=");decimal(arm);
  text(" cycles=");decimal(r->cycles);text(" checked=");decimal(r->checked);text(" errors=");decimal(r->errors);
  text(" checksum=");decimal(r->checksum);text(" guards=32\n");
 }
 return errors?1:0;
}
