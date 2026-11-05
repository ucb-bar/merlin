#include <stdint.h>
#include <stdio.h>
#include "chip_config.h"

void atlas_workload_main(void);
extern volatile uint64_t smol_tohost;
#ifdef PENGUIN26_SMOLVLA_PHASE
extern volatile uint64_t smol_full_segment_request;
extern volatile uint64_t smol_full_segment_self_restore;
#endif

void app_init(void) {
#if defined(TERMINAL_DEVICE_UART0)
  UART_InitType config;
  config.baudrate = PENGUIN26_UART_BAUDRATE;
  config.mode = UART_MODE_TX_RX;
  config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &config);
#endif
}

int app_main(void) {
#ifdef PENGUIN26_SMOLVLA_PHASE
  smol_full_segment_request = PENGUIN26_SMOLVLA_PHASE;
  smol_full_segment_self_restore = 1;
#endif
  atlas_workload_main();
  uint64_t result = smol_tohost;
  if (result == 1)
    return 0;
  int status = (int)(result >> 1);
  return status ? status : 1;
}

int main(void) {
  app_init();
  int status = app_main();
  printf("*** %s *** code=%d\n", status ? "FAILED" : "PASSED", status);
  fflush(stdout);
  return status;
}
