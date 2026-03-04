#include <stdio.h>
#include <string.h>

#include "runtime_scheduler.h"

static void usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <dronet.vmfb> <mlp.vmfb> [dronet_fn] [mlp_fn] [driver] [dump]\n\n"
          "Defaults:\n"
          "  dronet_fn = dronet.main_graph$async\n"
          "  mlp_fn    = mlp.main_graph$async\n"
          "  driver    = local-task\n"
          "  dump      = 0\n",
          argv0);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }

  const char* dronet_vmfb = argv[1];
  const char* mlp_vmfb = argv[2];
  const char* dronet_fn = (argc >= 4) ? argv[3] : "dronet.main_graph$async";
  const char* mlp_fn = (argc >= 5) ? argv[4] : "mlp.main_graph$async";
  const char* driver = (argc >= 6) ? argv[5] : "local-task";
  int dump = (argc >= 7) ? (strcmp(argv[6], "1") == 0) : 0;

  merlin_dual_model_async_gate_config_t config = {
      .dronet_vmfb_path = dronet_vmfb,
      .mlp_vmfb_path = mlp_vmfb,
      .dronet_function = dronet_fn,
      .mlp_function = mlp_fn,
      .driver_name = driver,
      .dump_output_bytes = dump,
  };

  fprintf(stdout,
          "Dual-model async gate:\n"
          "  dronet_vmfb = %s\n"
          "  mlp_vmfb    = %s\n"
          "  dronet_fn   = %s\n"
          "  mlp_fn      = %s\n"
          "  driver      = %s\n"
          "  dump        = %d\n",
          config.dronet_vmfb_path, config.mlp_vmfb_path, config.dronet_function,
          config.mlp_function, config.driver_name, config.dump_output_bytes);
  fflush(stdout);

  return merlin_dual_model_async_gate_run(&config);
}