/* Host verification driver for the Merlin C runtime: load weights.bin, run forward(),
 * write the output buffer to a file. Compared against the torch golden in Python.
 * (Host-only; the spike build uses model_main.c with HTIF instead.) */
#include <stdio.h>
#include <stdlib.h>

#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

int main(int argc, char **argv) {
  const char *weights_path = argc > 1 ? argv[1] : "weights.bin";
  const char *out_path = argc > 2 ? argv[2] : "out.bin";

  FILE *f = fopen(weights_path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", weights_path); return 1; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *weights = malloc(n);
  if (fread(weights, 1, n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); return 1; }
  fclose(f);

  merlin_descriptor_t descs[MERLIN_N_ARGS];
  merlin_reset_session();
  merlin_prepare_step(0);
  merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, weights, MERLIN_INPUT_PTR,
                   MERLIN_OUTPUT_PTR, descs);
#if MERLIN_N_STATE_PAIRS > 0
  if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                          MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                          MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {
    fprintf(stderr, "state ABI mismatch\n"); return 1;
  }
#endif

  FILE *o = fopen(out_path, "wb");
  fwrite(MERLIN_OUTPUT_PTR[0], sizeof(float), MERLIN_OUT_ELEMS, o);
  fclose(o);
  printf("wrote %d output elems to %s\n", MERLIN_OUT_ELEMS, out_path);
  return 0;
}
