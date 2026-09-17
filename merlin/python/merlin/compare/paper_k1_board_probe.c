/* Trusted SpacemiT K1 board-state probe for paper benchmark receipts.
 *
 * The production path reads VLEN from the RISC-V vlenb CSR and reads frequency,
 * governor, identity, and thermal state from the running Linux system.  The
 * unit-test path is deliberately a different command and is rejected for K1
 * contracts by paper_ablation_generator.py.
 */
#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_text(const char *path, char *buffer, size_t capacity) {
  FILE *stream = fopen(path, "r");
  if (!stream) return -1;
  size_t count = fread(buffer, 1, capacity - 1, stream);
  int failed = ferror(stream);
  fclose(stream);
  if (failed || count == 0) return -1;
  buffer[count] = '\0';
  while (count && (buffer[count - 1] == '\n' || buffer[count - 1] == '\r' ||
                   buffer[count - 1] == '\0')) buffer[--count] = '\0';
  return count ? 0 : -1;
}

static int read_positive_long(const char *path, long *value) {
  char buffer[64];
  char *end = NULL;
  if (read_text(path, buffer, sizeof(buffer))) return -1;
  errno = 0;
  long parsed = strtol(buffer, &end, 10);
  if (errno || end == buffer || parsed <= 0) return -1;
  *value = parsed;
  return 0;
}

static long maximum_thermal_millic(void) {
  DIR *directory = opendir("/sys/class/thermal");
  if (!directory) return -1;
  long maximum = -1;
  struct dirent *entry;
  while ((entry = readdir(directory))) {
    if (strncmp(entry->d_name, "thermal_zone", 12)) continue;
    char path[512];
    long value;
    if (snprintf(path, sizeof(path), "/sys/class/thermal/%s/temp", entry->d_name) <= 0)
      continue;
    if (!read_positive_long(path, &value) && value > maximum) maximum = value;
  }
  closedir(directory);
  return maximum;
}

static void json_string(const char *text) {
  putchar('"');
  for (const unsigned char *cursor = (const unsigned char *)text; *cursor; ++cursor) {
    if (*cursor == '"' || *cursor == '\\') putchar('\\');
    if (*cursor >= 0x20) putchar(*cursor);
  }
  putchar('"');
}

static int production_probe(void) {
  unsigned long vlenb = 0;
#if defined(__riscv) && defined(__riscv_vector)
  __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
#else
  fputs("K1 probe requires a RISC-V vector build\n", stderr);
  return 2;
#endif
  char identity[256], governor[128];
  long current_khz, maximum_khz;
  if (read_text("/sys/firmware/devicetree/base/serial-number", identity, sizeof(identity)) &&
      read_text("/etc/machine-id", identity, sizeof(identity))) return 3;
  if (read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", governor,
                sizeof(governor))) return 4;
  if (read_positive_long("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                         &current_khz)) return 5;
  if (read_positive_long("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
                         &maximum_khz)) return 6;
  long thermal = maximum_thermal_millic();
  if (thermal <= 0 || vlenb == 0) return 7;
  fputs("{\"schema_version\":1,\"kind\":\"merlin_board_probe_v1\",\"identity\":", stdout);
  json_string(identity);
  printf(",\"vlen_bits\":%lu,\"vlen_source\":\"csr\",\"governor\":", vlenb * 8);
  json_string(governor);
  printf(",\"current_khz\":%ld,\"max_khz\":%ld,\"max_thermal_millic\":%ld}\n",
         current_khz, maximum_khz, thermal);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "--json")) return production_probe();
  if (argc == 2 && !strcmp(argv[1], "--unit-test-json")) {
    puts("{\"schema_version\":1,\"kind\":\"merlin_board_probe_v1\","
         "\"identity\":\"non-paper-unit-test\",\"vlen_bits\":256,"
         "\"vlen_source\":\"csr\",\"governor\":\"performance\","
         "\"current_khz\":1600000,\"max_khz\":1600000,"
         "\"max_thermal_millic\":42000}");
    return 0;
  }
  fputs("usage: paper_k1_board_probe {--json|--unit-test-json}\n", stderr);
  return 64;
}
