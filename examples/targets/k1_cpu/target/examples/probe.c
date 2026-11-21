#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

static inline uintptr_t read_vlenb(void) {
  uintptr_t value;
  __asm__ volatile("csrr %0, vlenb" : "=r"(value));
  return value;
}

static inline uintptr_t read_time(void) {
  uintptr_t value;
  __asm__ volatile("rdtime %0" : "=r"(value));
  return value;
}

int main(void) {
  printf("k1_cpu_probe_version=1\n");
  printf("online_harts=%ld\n", sysconf(_SC_NPROCESSORS_ONLN));
  printf("vlenb=%lu\n", (unsigned long)read_vlenb());
  printf("rdtime=%lu\n", (unsigned long)read_time());
  return 0;
}
