#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "host.h"
#include "assets.h"
#include "atlas_numeric.h"
#ifndef M2_VERIFY
#define M2_VERIFY 1
#endif

__attribute__((aligned(64))) float logits[M2_LOGITS];
__attribute__((aligned(64))) volatile struct {
  uint32_t magic, version, status, completed, failing_index, runtime_result;
  uint64_t bad_offset, mismatches;
  uint64_t trap_cause, trap_pc, trap_value;
} atlas_workload_mailbox;
__attribute__((aligned(64))) volatile struct {
  uint32_t status, completed, bad_launch;
  uint64_t mismatches;
} smoke_result;
static atlas_host host;
#ifdef M2_FAST
/* Ownership can remain active after a timeout, fault, or cache-sync failure. */
static atlas_host_buffer fast_buffers[2];
#endif
static uint64_t mark_cycle, mark_instret;
static uint64_t total_host_instret, total_launch_instret;
static uint64_t total_atlas_load_clocks, total_atlas_start_wait_clocks;

static inline uint64_t cycle(void) { uint64_t x; __asm__ volatile("rdcycle %0" : "=r"(x)); return x; }
static inline uint64_t instret(void) { uint64_t x; __asm__ volatile("rdinstret %0" : "=r"(x)); return x; }
static float power2(int exponent) { union { uint32_t u; float f; } x = {.u=(uint32_t)(exponent+127)<<23}; return x.f; }
static float bf16_float(uint16_t bits) { union { uint32_t u; float f; } x={.u=(uint32_t)bits<<16}; return x.f; }
static float unpack_fp8(uint8_t bits, int scale) {
  unsigned exp=(bits>>3)&15, fraction=bits&7;
  if (!exp || (exp==15 && fraction==7)) return bits&128 ? -0.0f : 0.0f;
  uint16_t bf16=(uint16_t)(((bits&128)<<8) | ((exp+120)<<7) | (fraction<<4));
  return bf16_float(bf16)*power2(scale);
}
__attribute__((noipa)) static size_t different(const volatile uint8_t *actual, const uint8_t *expected, size_t bytes, size_t offset) {
  size_t bad=0;
  while (bytes && (((uintptr_t)actual | (uintptr_t)expected)&7)) {
    if (*actual++ != *expected++) { if (!bad) atlas_workload_mailbox.bad_offset=offset; ++bad; }
    --bytes; ++offset;
  }
  while (bytes>=8) {
    uint64_t delta=*(const volatile uint64_t *)actual ^ *(const uint64_t *)expected;
    if (delta) for (unsigned i=0;i<8;++i) {
      if (delta&255) { if (!bad) atlas_workload_mailbox.bad_offset=offset+i; ++bad; }
      delta>>=8;
    }
    actual+=8; expected+=8; bytes-=8; offset+=8;
  }
  while (bytes--) { if (*actual++ != *expected++) { if (!bad) atlas_workload_mailbox.bad_offset=offset; ++bad; } ++offset; }
  return bad;
}
__attribute__((noipa)) static void copy_bytes(volatile uint8_t *dest, const uint8_t *src, size_t bytes) {
  while (bytes && (((uintptr_t)dest | (uintptr_t)src)&7)) { *dest++=*src++; --bytes; }
  while (bytes>=8) { *(volatile uint64_t *)dest=*(const uint64_t *)src; dest+=8; src+=8; bytes-=8; }
  while (bytes--) *dest++=*src++;
}
/* Quantize the original image exactly once; every convolution patch then
 * gathers FP8 bytes, including repeated references to the same image pixel. */
static uint8_t quantized_input[M2_INPUT_FP8_BYTES] __attribute__((aligned(64)));
__attribute__((noipa)) static int quantize_input(const m2_tensor *input, int scale) {
  if (input->dtype!=M2_F32_NCHW || !input->h || !input->w || !input->channels ||
      input->h>M2_INPUT_FP8_BYTES || input->w>M2_INPUT_FP8_BYTES/input->h ||
      input->channels>M2_INPUT_FP8_BYTES/input->h/input->w) return 1;
  size_t spatial=(size_t)input->h*input->w;
  if (input->bytes!=spatial*input->channels*sizeof(float)) return 1;
  const float *source=(const float *)input->data;
  for (size_t row=0;row<spatial;++row) for (unsigned channel=0;channel<input->channels;++channel) {
    uint16_t bits;
    if (f32_bf16(source[(size_t)channel*spatial+row],&bits)) return 1;
    quantized_input[row*input->channels+channel]=pack_bf16(bits,scale);
  }
  return 0;
}
/* K order is channel, kernel row, kernel column, matching captured weights. */
__attribute__((noipa)) static int pack_a(const m2_launch *launch, const m2_tensor *input, volatile uint8_t *dest) {
  m2_tensor quantized;
  if (input->dtype==M2_F32_NCHW) {
    if (quantize_input(input,launch->input_scale)) return 1;
    quantized=*input;
    quantized.data=quantized_input;
    quantized.bytes=(size_t)input->h*input->w*input->channels;
    quantized.padded_channels=input->channels;
    quantized.dtype=M2_FP8_NHWC;
    input=&quantized;
  }
  unsigned oh=(input->h+2*launch->padding-launch->kernel)/launch->stride+1;
  unsigned ow=(input->w+2*launch->padding-launch->kernel)/launch->stride+1;
  if ((size_t)launch->m*launch->k != launch->a_bytes || launch->logical_m != oh*ow) return 1;
  size_t cursor=0;
  for (unsigned mt=0;mt<launch->m;mt+=32)
    for (unsigned kt=0;kt<launch->k;kt+=32)
      for (unsigned r=0;r<32;++r)
        for (unsigned c=0;c<32;++c) {
          unsigned row=mt+r, inner=kt+c;
          uint8_t value=0;
          if (row<launch->logical_m && inner<launch->logical_k) {
            unsigned channel=inner/(launch->kernel*launch->kernel);
            int ih=(int)(row/ow*launch->stride+(inner/launch->kernel)%launch->kernel)-(int)launch->padding;
            int iw=(int)(row%ow*launch->stride+inner%launch->kernel)-(int)launch->padding;
            if (ih>=0 && iw>=0 && ih<(int)input->h && iw<(int)input->w) {
              if (input->dtype==M2_FP8_NHWC) {
                value=input->data[((size_t)ih*input->w+(unsigned)iw)*input->padded_channels+channel];
              } else {
                const float *f=(const float *)input->data;
                if (input->dtype!=M2_F32_VECTOR) return 1;
                float x=f[channel];
                uint16_t bits;
                if (f32_bf16(x,&bits)) return 1;
                value=pack_bf16(bits,launch->input_scale);
              }
            }
          }
          dest[cursor++]=value;
        }
  return 0;
}
static int run_launch(unsigned index) {
#ifdef M2_FAST
  if (host.active) return 7;
#endif
  const m2_launch *launch=&m2_launches[index];
  const m2_tensor *input=&m2_tensors[launch->input];
  m2_tensor *output=&m2_tensors[launch->output];
  volatile uint8_t *arena=(volatile uint8_t *)(uintptr_t)M2_ARENA_BASE;
  uint64_t c0=cycle(),i0=instret();
  uint64_t host_cycles=c0-mark_cycle,host_instret=i0-mark_instret;
  atlas_workload_mailbox.failing_index=index;
#ifdef M2_FAST
  const m2_fast_launch *fast=&m2_fast_launches[index];
  size_t b_end=fast->b_offset+fast->b_bytes;
  if (fast->b_offset>launch->output_offset || b_end>launch->output_offset ||
      fast->b_bytes!=(size_t)launch->k*launch->n || fast->weights!=launch->initial+fast->b_offset || fast->source_base!=(uintptr_t)fast->weights) return 7;
  /* Preserve every auxiliary input, zero line, guard, and output poison byte.
   * Only immutable B is read directly from its embedded read-only blob. */
  copy_bytes(arena,launch->initial,fast->b_offset);
  copy_bytes(arena+b_end,launch->initial+b_end,launch->arena_bytes-b_end);
#else
  copy_bytes(arena,launch->initial,launch->arena_bytes);
#endif
  if (launch->direct) {
    if (input->dtype!=M2_FP8_NHWC || launch->source_offset+input->bytes>launch->arena_bytes) return 1;
    copy_bytes(arena+launch->source_offset,input->data,input->bytes);
  } else if (pack_a(launch,input,arena+launch->a_offset)) return 2;
  for (size_t i=0;i<launch->dynamic_count;++i) {
    const m2_dynamic *region=&launch->dynamic[i];
    const m2_tensor *tensor=&m2_tensors[region->tensor];
    if (region->offset+tensor->bytes>launch->arena_bytes) return 3;
    copy_bytes(arena+region->offset,tensor->data,tensor->bytes);
  }
#if M2_VERIFY && !defined(M2_FAST)
  size_t bad=different(arena,launch->initial,launch->arena_bytes,0);
  if (bad) { atlas_workload_mailbox.mismatches=bad; return 4; }
#endif
  /* The free-running Atlas clock includes halted time. Keep IMEM loading
   * separate from start through observed completion, which includes host
   * start/poll latency and is not a hardware-latched kernel cycle count. */
  uint32_t load_before=atlas_host_direct_read32(0,ATLAS_CSR_BASE+ATLAS_CSR_CYCLES);
  atlas_host_result status=atlas_host_load_raw(&host,launch->program,launch->program_bytes/4);
  uint32_t after_load=atlas_host_direct_read32(0,ATLAS_CSR_BASE+ATLAS_CSR_CYCLES);
#ifdef M2_FAST
  fast_buffers[0]=(atlas_host_buffer){(uintptr_t)arena,launch->arena_bytes,ATLAS_HOST_DEVICE_READ_WRITE};
  fast_buffers[1]=(atlas_host_buffer){(uintptr_t)fast->weights,fast->b_bytes,ATLAS_HOST_DEVICE_READ};
  /* Weightless launches (the device max pool) own only their arena. */
  if (status==ATLAS_HOST_OK) status=atlas_host_start(&host,fast_buffers,fast->b_bytes ? 2 : 1);
#else
  atlas_host_buffer buffer={(uintptr_t)arena,launch->arena_bytes,ATLAS_HOST_DEVICE_READ_WRITE};
  if (status==ATLAS_HOST_OK) status=atlas_host_start(&host,&buffer,1);
#endif
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(&host,100000000);
  atlas_workload_mailbox.runtime_result=status;
  if (status!=ATLAS_HOST_OK) return 5;
  uint32_t load_clocks=after_load-load_before;
  uint32_t start_wait_clocks=atlas_host_direct_read32(0,ATLAS_CSR_BASE+ATLAS_CSR_CYCLES)-after_load;
#if M2_VERIFY || defined(M2_FAST)
#ifndef M2_FAST
  size_t end=launch->output_offset+launch->output_bytes;
#else
  size_t bad;
#endif
  bad=different(arena+launch->output_offset,launch->expected,launch->output_bytes,launch->output_offset);
#ifndef M2_FAST
  bad+=different(arena,launch->initial,launch->output_offset,0);
  bad+=different(arena+end,launch->initial+end,launch->arena_bytes-end,end);
#endif
  if (bad) { atlas_workload_mailbox.mismatches=bad; return 6; }
#endif
  /* Always consume the actual device bytes, after validation. */
  copy_bytes(output->data,(const uint8_t *)arena+launch->output_offset,launch->output_bytes);
  atlas_workload_mailbox.completed=index+1;
  uint64_t c1=cycle(),i1=instret();
  total_host_instret+=host_instret;
  total_launch_instret+=i1-i0;
  total_atlas_load_clocks+=load_clocks;
  total_atlas_start_wait_clocks+=start_wait_clocks;
  printf("RESNET50_LAUNCH index=%u status=0 host_cycles=%llu launch_cycles=%llu atlas_load_clocks=%u atlas_start_wait_clocks=%u host_instret=%llu launch_instret=%llu a_bytes=%lu out_bytes=%lu scale=%d\n",
    index,(unsigned long long)host_cycles,(unsigned long long)(c1-c0),load_clocks,start_wait_clocks,(unsigned long long)host_instret,
    (unsigned long long)(i1-i0),(unsigned long)launch->a_bytes,(unsigned long)launch->output_bytes,launch->input_scale);
  mark_cycle=cycle(); mark_instret=instret();
  return 0;
}
__attribute__((noipa)) static int maxpool(const m2_node *node) {
  const m2_tensor *input=&m2_tensors[node->input];
  m2_tensor *output=&m2_tensors[node->output];
  if (input->dtype!=M2_FP8_NHWC || output->dtype!=M2_FP8_NHWC || input->scale!=output->scale) return 1;
  memset(output->data,0,output->bytes);
  for (unsigned y=0;y<output->h;++y) for (unsigned x=0;x<output->w;++x)
    for (unsigned c=0;c<output->channels;++c) {
      uint8_t best=0;
      for (unsigned ky=0;ky<node->kernel;++ky) for (unsigned kx=0;kx<node->kernel;++kx) {
        int iy=(int)(y*node->stride+ky)-(int)node->padding;
        int ix=(int)(x*node->stride+kx)-(int)node->padding;
        if (iy>=0 && ix>=0 && iy<(int)input->h && ix<(int)input->w) {
          uint8_t value=input->data[((size_t)iy*input->w+(unsigned)ix)*input->padded_channels+c];
          if (value&128 || value==127) return 2;
          if (value>best) best=value;
        }
      }
      output->data[((size_t)y*output->w+x)*output->padded_channels+c]=best;
    }
  return 0;
}
__attribute__((noipa)) static int average(const m2_node *node) {
  const m2_tensor *input=&m2_tensors[node->input];
  m2_tensor *output=&m2_tensors[node->output];
  if (input->dtype!=M2_FP8_NHWC || output->dtype!=M2_F32_VECTOR) return 1;
  float *dest=(float *)output->data;
  size_t spatial=(size_t)input->h*input->w;
  for (unsigned c=0;c<input->channels;++c) {
    float sum=0;
    for (size_t row=0;row<spatial;++row) sum=sum+unpack_fp8(input->data[row*input->padded_channels+c],input->scale);
    dest[c]=sum/(float)spatial;
  }
  return 0;
}
__attribute__((noipa)) static int fc_bias(const m2_node *node) {
  const m2_tensor *input=&m2_tensors[node->input];
  if (input->dtype!=M2_BF16_TILES) return 1;
  for (unsigned c=0;c<M2_LOGITS;++c) {
    size_t offset=(c/32)*2048+((c%32)/16)*1024+(c%16)*2;
    uint16_t bits=(uint16_t)input->data[offset]|((uint16_t)input->data[offset+1]<<8);
    logits[c]=bf16_float(bits)+m2_fc_bias[c];
  }
  return 0;
}
__attribute__((noinline)) void atlas_block_trap_report(uint64_t cause,uint64_t pc,uint64_t value) {
  atlas_workload_mailbox.status=255;
  atlas_workload_mailbox.trap_cause=cause;
  atlas_workload_mailbox.trap_pc=pc;
  atlas_workload_mailbox.trap_value=value;
}
__attribute__((noipa,noreturn)) void smoke_done(void) {
  for (;;) __asm__ volatile("wfi");
}
#ifdef RESNET50_QEMU_HARNESS
extern volatile uint64_t tohost;
__attribute__((noinline,noreturn,used)) void finish(uint32_t result) {
  atlas_workload_mailbox.status=result;
  smoke_result.status=result ? 2 : 1;
  smoke_result.completed=atlas_workload_mailbox.completed;
  smoke_result.bad_launch=atlas_workload_mailbox.failing_index;
  smoke_result.mismatches=atlas_workload_mailbox.mismatches;
  if (result && !smoke_result.mismatches) smoke_result.mismatches=1;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  tohost=((uint64_t)result<<1)|1;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  smoke_done();
}
#endif
uintptr_t trap_handler(uintptr_t epc,uintptr_t cause,uintptr_t tval,uintptr_t regs[32]) {
  (void)regs;
  atlas_block_trap_report(cause,epc,tval);
#ifdef RESNET50_QEMU_HARNESS
  finish(255);
#else
  extern volatile uint64_t tohost;
  tohost=511;
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  for (;;) __asm__ volatile("wfi");
#endif
}
__attribute__((noinline)) uint32_t atlas_workload_main(void) {
  atlas_workload_mailbox.magic=0x41544c53;
  atlas_workload_mailbox.version=4;
  atlas_workload_mailbox.status=UINT32_MAX;
  atlas_workload_mailbox.bad_offset=UINT64_MAX;
  atlas_workload_mailbox.failing_index=UINT32_MAX;
  printf("ATLAS_KERNEL_BEGIN name=resnet50-v2 launches_expected=%u verify=%u\n",M2_LAUNCHES,M2_VERIFY);
  atlas_host_result status=atlas_host_init(&host,(atlas_host_map){ATLAS_IMEM_BASE,ATLAS_CSR_BASE,ATLAS_IMEM_BYTES},
    (atlas_host_mmio){atlas_host_direct_read32,atlas_host_direct_write32,0},(atlas_host_memory){1,0,0});
  if (status!=ATLAS_HOST_OK) return 254;
  mark_cycle=cycle(); mark_instret=instret();
  for (unsigned i=0;i<M2_NODES;++i) {
    const m2_node *node=&m2_nodes[i];
    int result=node->op==M2_LAUNCH ? run_launch(node->launch) : node->op==M2_MAXPOOL ? maxpool(node) :
      node->op==M2_AVGPOOL ? average(node) : fc_bias(node);
    if (result) {
      printf("RESNET50_GRAPH_FAIL node=%u op=%u result=%d launch=%u offset=%llu mismatches=%llu\n",i,node->op,result,
        atlas_workload_mailbox.failing_index,(unsigned long long)atlas_workload_mailbox.bad_offset,
        (unsigned long long)atlas_workload_mailbox.mismatches);
      return 1;
    }
  }
  total_host_instret+=instret()-mark_instret;
  unsigned matching=0,best=0,expected_best=0;
  for (unsigned i=0;i<M2_LOGITS;++i) {
    union {float f;uint32_t u;} x={.f=logits[i]};
    matching+=x.u==m2_expected_logits[i];
    if (logits[i]>logits[best]) best=i;
    union {uint32_t u;float f;} a={.u=m2_expected_logits[i]},b={.u=m2_expected_logits[expected_best]};
    if (a.f>b.f) expected_best=i;
  }
  printf("RESNET50_TOTAL launches=%u host_instret=%llu launch_instret=%llu atlas_load_clocks=%llu atlas_start_wait_clocks=%llu\n",
    atlas_workload_mailbox.completed,(unsigned long long)total_host_instret,(unsigned long long)total_launch_instret,
    (unsigned long long)total_atlas_load_clocks,(unsigned long long)total_atlas_start_wait_clocks);
  printf("RESNET50_RESULT %s logits_matching=%u/%u top1=%u expected_top1=%u\n",matching==M2_LOGITS?"PASS":"FAIL",
    matching,M2_LOGITS,best,expected_best);
  if (matching!=M2_LOGITS || atlas_workload_mailbox.completed!=M2_LAUNCHES) {
    atlas_workload_mailbox.mismatches=M2_LOGITS-matching;
    return 2;
  }
  atlas_workload_mailbox.failing_index=UINT32_MAX;
  printf("RESNET_V2 PASS\n");
  printf("ATLAS_KERNEL_PASS name=resnet50-v2 launches_completed=%u\n",M2_LAUNCHES);
  return 0;
}
int main(void) {
  uint32_t result=atlas_workload_main();
  atlas_workload_mailbox.status=result;
  printf(result?"*** FAILED ***\n":"*** PASSED ***\n");
  fflush(stdout);
#ifdef RESNET50_QEMU_HARNESS
  finish(result);
#endif
  return (int)result;
}
