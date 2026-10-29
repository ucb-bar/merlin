/* Public frontend-layout diagnostic; original selected ISA, no reference kernel.
 * Exact CPU reference once. All matrix outputs and guards checked on every call. */
typedef unsigned long u64;
typedef signed char i8;
typedef int i32;
extern volatile u64 tohost;
extern u64 layout0(u64,u64,u64,u64,u64,u64);
extern u64 layout44(u64,u64,u64,u64,u64,u64);
#define D 16
#define COMMAND(f,x,y) __asm__ volatile(".insn r 0x7b, 3, %c2, x0, %0, %1"::"r"((u64)(x)),"r"((u64)(y)),"i"(f):"memory")
static volatile i8 a[D*D] __attribute__((aligned(64)));
static volatile i8 b[D*D] __attribute__((aligned(64)));
static i32 expected[D*D];
static struct { i32 before[16]; i32 values[D*D]; i32 after[16]; } output __attribute__((aligned(64)));
struct result {u64 cycles;unsigned errors;};
static struct result result[2][2][3];
static u64 tile(unsigned row){return row|((u64)D<<32)|((u64)D<<48);}
static void complete(void){__asm__ volatile("fence":::"memory");}
static void invalidate(void){__asm__ volatile("fence.i":::"memory");}
static void put(char c){tohost=0x0101000000000000ul|(unsigned char)c;while(tohost){}}
static void text(const char*s){while(*s)put(*s++);}
static void decimal(u64 n){char buf[24];unsigned i=0;do{buf[i++]=(char)('0'+n%10);n/=10;}while(n);while(i)put(buf[--i]);}
static void reset(void){for(unsigned i=0;i<D*D;i++)output.values[i]=0x35353535;for(unsigned i=0;i<16;i++){output.before[i]=0x12345678;output.after[i]=0x76543210;}complete();}
static unsigned check(void){unsigned e=0;for(unsigned i=0;i<D*D;i++)e+=output.values[i]!=expected[i];for(unsigned i=0;i<16;i++){e+=output.before[i]!=0x12345678;e+=output.after[i]!=0x76543210;}return e;}
static void prepare(void){const u64 one=0x3f800000ul;
 for(unsigned i=0;i<D*D;i++){a[i]=(i8)((int)((i*5+3)%9)-4);b[i]=(i8)((int)((i*11+1)%7)-3);}
 for(unsigned i=0;i<D;i++)for(unsigned j=0;j<D;j++){i32 v=0;for(unsigned k=0;k<D;k++)v+=(i32)a[i*D+k]*(i32)b[k*D+j];expected[i*D+j]=v;}
 COMMAND(0,6,(one<<32)|64);COMMAND(0,2,(one<<32)|64);
 COMMAND(0,1ul|(1ul<<8)|(D<<16)|(one<<32),D);
 COMMAND(2,a,tile(0));COMMAND(2,b,tile(4096));
 COMMAND(0,4ul|(1ul<<16)|(one<<32),1ul<<48);complete();
}
__attribute__((noinline)) static u64 invoke(unsigned layout){
 u64 p0=tile(4096),p1=tile(0x80000000u),p2=tile(0),p3=tile(0xffffffffu),p4=(u64)output.values,p5=tile(0xa0000000u);
 if(layout)return layout44(p0,p1,p2,p3,p4,p5);
 return layout0(p0,p1,p2,p3,p4,p5);
}
int main(void){unsigned errors=0,warmups=0;prepare();
 for(unsigned layout=0;layout<2;layout++)for(unsigned rep=0;rep<3;rep++){
  reset();invalidate();result[layout][0][rep].cycles=invoke(layout);result[layout][0][rep].errors=check();errors+=result[layout][0][rep].errors;
  reset();(void)invoke(layout);errors+=check();warmups++;
  reset();result[layout][1][rep].cycles=invoke(layout);result[layout][1][rep].errors=check();errors+=result[layout][1][rep].errors;
 }
 for(unsigned layout=0;layout<2;layout++)for(unsigned rep=0;rep<3;rep++)for(unsigned state=0;state<2;state++){
  struct result*r=&result[layout][state][rep];text("FRONTEND layout=");decimal(layout?44:0);text(" state=");decimal(state);text(" rep=");decimal(rep);text(" cycles=");decimal(r->cycles);text(" checked=256 guards=32 errors=");decimal(r->errors);text("\n");
 }
 text("FRONTEND_TOTAL errors=");decimal(errors);text(" warmup_runs=");decimal(warmups);text("\n");return errors?1:0;
}
