/* Hosted adapter for the generated owned-result driver and byte observer.
 * Caller-selected input bytes remain runtime data, outside source staging. */
#include "owned_result_readback.h"
#include <stdio.h>
#include <stdlib.h>

static int write_bytes(void *context, const void *data, size_t bytes) {
  (void)context;
  return fwrite(data, 1, bytes, stdout) != bytes;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  if (argc != 2) {
    fputs("usage: owned-model invocation.bin\n", stderr);
    return 1;
  }
  FILE *input_file = fopen(argv[1], "rb");
  if (!input_file)
    return 2;
  if (fseek(input_file, 0, SEEK_END)) {
    fclose(input_file);
    return 3;
  }
  long length = ftell(input_file);
  if (length < 0 || length > 64 * 1024 * 1024 || fseek(input_file, 0, SEEK_SET)) {
    fclose(input_file);
    return 4;
  }
  void *input = malloc(length ? (size_t)length : 1);
  if (!input) {
    fclose(input_file);
    return 5;
  }
  if (fread(input, 1, (size_t)length, input_file) != (size_t)length) {
    fclose(input_file);
    free(input);
    return 6;
  }
  fclose(input_file);
  MerlinOwnedReadback sink = {write_bytes, NULL};
  int status = merlin_owned_driver_run(input, (size_t)length, merlin_owned_readback_observe, &sink);
  free(input);
  if (status) {
    fputs(merlin_owned_driver_error(status), stderr);
    fputc('\n', stderr);
    return status;
  }
  if (fputs("DONE\n", stdout) == EOF || fflush(stdout))
    return 7;
  return 0;
}
