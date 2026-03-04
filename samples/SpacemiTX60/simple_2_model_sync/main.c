#include <stdio.h>
#include <stdlib.h>

#include "runtime_scheduler.h"

static void print_usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <dronet.vmfb> <mlp.vmfb> [dronet_fn] [mlp_fn] [driver]\n\n"
          "Defaults:\n"
          "  dronet_fn = dronet.main_graph$async\n"
          "  mlp_fn    = mlp.main_graph$async\n"
          "  driver    = local-task\n",
          argv0);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const char* dronet_vmfb = argv[1];
  const char* mlp_vmfb = argv[2];
  const char* dronet_fn = (argc >= 4) ? argv[3] : "dronet.main_graph$async";
  const char* mlp_fn = (argc >= 5) ? argv[4] : "mlp.main_graph$async";
  const char* driver = (argc >= 6) ? argv[5] : "local-task";

  merlin_async_smoke_config_t config = {
      .dronet_vmfb_path = dronet_vmfb,
      .mlp_vmfb_path = mlp_vmfb,
      .dronet_function = dronet_fn,
      .mlp_function = mlp_fn,
      .driver_name = driver,
  };

  fprintf(stdout,
          "Async smoke config:\n"
          "  dronet_vmfb = %s\n"
          "  mlp_vmfb    = %s\n"
          "  dronet_fn   = %s\n"
          "  mlp_fn      = %s\n"
          "  driver      = %s\n",
          config.dronet_vmfb_path, config.mlp_vmfb_path, config.dronet_function,
          config.mlp_function, config.driver_name);
  fflush(stdout);

  return merlin_async_smoke_run(&config);
}