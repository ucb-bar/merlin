/* Runner-owned rv64 carrier for one bounded Radiance GSIM numerical check.
 *
 * The Muon harness writes its 32 f32 result words at local address 0x10010000
 * and publishes READY at 0x10011000.  The SoC fuse maps the Muon address
 * space at +0x1_0000_0000, so Rocket observes those locations at the physical
 * addresses below.  Rocket publishes the comparison through the standard
 * one of two distinct, retained PC loops.  The GSIM harness already reports
 * Rocket's PC every 2,000 cycles, so a bounded run can distinguish PASS from
 * FAIL without relying on the broken UART or on dirty-cache DRAM inspection.
 * The Muon manager stays parked, preventing GPU-idle from racing the readback.
 */
#include <stdint.h>

#define RESULT ((volatile uint32_t *)0x110010000ULL)
#define STATUS ((volatile uint32_t *)0x110011000ULL)
#define READY 0x52503130u /* "RP10" */
#define ACK   0x41434b31u /* "ACK1" */

static const uint32_t lower[32] = {
  0xc005e780u, 0x3e979700u, 0xbd3582c0u, 0xbf5ca580u, 0x3fe4ed80u, 0xbfd57900u, 0xbf54e700u, 0xbf35cc80u,
  0x3e8a2d80u, 0xbf633f80u, 0xbec46680u, 0x3fc31700u, 0x40210980u, 0xbf642300u, 0x3ffcad00u, 0x3fbfc480u,
  0x3dd84d00u, 0x3f359d80u, 0x3f142580u, 0xc01bde00u, 0xc046c700u, 0xbf90ca80u, 0x3fa76780u, 0x3f1f5800u,
  0xbf08aac0u, 0x3ecc2180u, 0xbf297c00u, 0x3f4f9400u, 0x3eb87180u, 0xbef89400u, 0x3fb0e100u, 0xbfe28c80u,
};

static const uint32_t upper[32] = {
  0xbffbb100u, 0x3ebce900u, 0x3c984580u, 0xbf461a80u, 0x3ff45280u, 0xbfc70700u, 0xbf3e9900u, 0xbf207380u,
  0x3eaf1280u, 0xbf4c8080u, 0xbe9ed980u, 0x3fd16900u, 0x402a3680u, 0xbf4d5d00u, 0x40066980u, 0x3fcdfb80u,
  0x3e309980u, 0x3f4ba280u, 0x3f291a80u, 0xc0132200u, 0xc03cb900u, 0xbf847580u, 0x3fb4d880u, 0x3f34a800u,
  0xbee96a80u, 0x3ef31e80u, 0xbf148400u, 0x3f666c00u, 0x3edece80u, 0xbed16c00u, 0x3fbe9f00u, 0xbfd3b380u,
};

/* Monotonic key for finite IEEE-754 f32 values, implemented without FP so it
 * also runs on the scalar Rocket configuration in this RTL. */
static uint32_t ordered_f32(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__attribute__((noreturn, noinline, aligned(64))) static void pass_loop(void) {
  __asm__ volatile(".globl rp10_numeric_pass\n"
                   "rp10_numeric_pass:\n"
                   "wfi\n"
                   "j rp10_numeric_pass");
  __builtin_unreachable();
}

__attribute__((noreturn, noinline, aligned(64))) static void fail_loop(void) {
  __asm__ volatile(".globl rp10_numeric_fail\n"
                   "rp10_numeric_fail:\n"
                   "wfi\n"
                   "j rp10_numeric_fail");
  __builtin_unreachable();
}

int main(void) {
  while (STATUS[0] != READY) __asm__ volatile("fence r, r" ::: "memory");

  uint32_t bad = 0;
  uint32_t checksum = 2166136261u;
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t got = RESULT[i];
    checksum = (checksum ^ got) * 16777619u;
    uint32_t key = ordered_f32(got);
    bad += (key < ordered_f32(lower[i]) || key > ordered_f32(upper[i]));
  }
  STATUS[3] = checksum;
  STATUS[4] = bad;
  __asm__ volatile("fence rw, rw" ::: "memory");

  if (bad == 0) pass_loop();
  fail_loop();
}
