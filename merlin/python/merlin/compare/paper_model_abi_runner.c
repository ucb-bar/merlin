#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Implemented by the frozen backend package before private measurement inputs exist. */
extern int merlin_paper_step(const char *artifact_path, const unsigned char *input,
                             size_t input_size, unsigned char *output, size_t *output_size);

static int read_u64(FILE *stream, uint64_t *value) {
  unsigned char raw[8];
  if (fread(raw, 1, sizeof(raw), stream) != sizeof(raw)) return 0;
  *value = 0;
  for (int index = 7; index >= 0; --index) *value = (*value << 8) | raw[index];
  return 1;
}

static int write_u64(FILE *stream, uint64_t value) {
  unsigned char raw[8];
  for (int index = 0; index < 8; ++index) { raw[index] = value & 0xffu; value >>= 8; }
  return fwrite(raw, 1, sizeof(raw), stream) == sizeof(raw);
}

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  FILE *output = fopen(argv[2], "wb");
  if (!output || fwrite("MRLNFRM1", 1, 8, output) != 8) return 3;
  for (int argument = 3; argument < argc; ++argument) {
    FILE *input = fopen(argv[argument], "rb");
    char magic[8];
    if (!input || fread(magic, 1, sizeof(magic), input) != sizeof(magic)
        || memcmp(magic, "MRLNFRM1", sizeof(magic))) return 4;
    uint64_t length;
    while (read_u64(input, &length)) {
      if (!length || length > (1u << 26)) return 5;
      unsigned char *payload = malloc((size_t)length);
      size_t output_size = 1u << 26;
      unsigned char *result = malloc(output_size);
      if (!payload || !result || fread(payload, 1, (size_t)length, input) != length) return 6;
      if (merlin_paper_step(argv[1], payload, (size_t)length, result, &output_size)
          || !output_size || output_size > (1u << 26)
          || !write_u64(output, output_size)
          || fwrite(result, 1, output_size, output) != output_size) return 7;
      free(payload); free(result);
    }
    if (!feof(input)) return 8;
    fclose(input);
  }
  return fclose(output) ? 9 : 0;
}
