/* Serial returned-result adapter for the selected bare-metal console ABI.
 * The builder supplies the actual input-blob placement/extent from its layout. */
#include "htif.h"
#include "owned_result_readback.h"

#if !defined(MERLIN_OWNED_INPUT_BASE_ADDR) || !defined(MERLIN_OWNED_INPUT_BYTES)
#error "owned-result execution requires explicit input blob placement and size"
#endif

static int write_bytes(void *context, const void *data, size_t bytes) {
  (void)context;
  const unsigned char *source = data;
  htif_line_flush(0);
  for (size_t i = 0; i < bytes; ++i)
    htif_putc((char)source[i]);
  htif_line_flush(1);
  return 0;
}

_Noreturn void merlin_owned_results_failure(const char *reason) {
  htif_puts("OWNED_FAILURE ");
  htif_puts(reason);
  htif_putc('\n');
  htif_exit(2);
}

int main(int hart) {
  if (hart)
    for (;;) {}
  console_init();
  MerlinOwnedReadback sink = {write_bytes, NULL};
  int status = merlin_owned_driver_run((void *)(uintptr_t)MERLIN_OWNED_INPUT_BASE_ADDR,
      (size_t)MERLIN_OWNED_INPUT_BYTES, merlin_owned_readback_observe, &sink);
  if (status) {
    htif_puts("OWNED_DRIVER_FAILURE ");
    htif_puts(merlin_owned_driver_error(status));
    htif_putc('\n');
    htif_exit(status);
  }
  htif_puts("DONE\n");
  htif_exit(0);
}
