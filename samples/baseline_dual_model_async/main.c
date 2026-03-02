#include <stdio.h>
#include <stdlib.h>

#include "runtime_scheduler.h"

static void print_usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <dronet.vmfb> <mlp.vmfb> [mlp_hz] [duration_s] "
          "[dronet_fn] [mlp_fn] [driver] [report_hz] "
          "[dronet_sensor_hz] [mlp_sensor_hz]\n\n"
          "Defaults:\n"
          "  mlp_hz     = 20.0\n"
          "  duration_s = 10.0 (<=0 means run forever)\n"
          "  dronet_fn  = dronet.main_graph\n"
          "  mlp_fn     = mlp.main_graph\n"
          "  driver     = local-task\n"
          "  report_hz  = 1.0\n"
          "  dronet_sensor_hz = 60.0\n"
          "  mlp_sensor_hz    = mlp_hz\n",
          argv0);
}

static double parse_double_or_default(const char* text, double default_value) {
  if (!text) return default_value;
  char* end = NULL;
  double value = strtod(text, &end);
  if (end == text) return default_value;
  return value;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  double mlp_hz = (argc >= 4) ? parse_double_or_default(argv[3], 20.0) : 20.0;
  double duration_s =
      (argc >= 5) ? parse_double_or_default(argv[4], 10.0) : 10.0;
  const char* dronet_fn = (argc >= 6) ? argv[5] : "dronet.main_graph";
  const char* mlp_fn = (argc >= 7) ? argv[6] : "mlp.main_graph";
  const char* driver_name = (argc >= 8) ? argv[7] : "local-task";
  double report_hz = (argc >= 9) ? parse_double_or_default(argv[8], 1.0) : 1.0;
  double dronet_sensor_hz =
      (argc >= 10) ? parse_double_or_default(argv[9], 60.0) : 60.0;
  double mlp_sensor_hz =
      (argc >= 11) ? parse_double_or_default(argv[10], mlp_hz) : mlp_hz;

  merlin_dual_model_runtime_config_t config = {
      .dronet_vmfb_path = argv[1],
      .mlp_vmfb_path = argv[2],
      .dronet_function = dronet_fn,
      .mlp_function = mlp_fn,
      .driver_name = driver_name,
      .mlp_frequency_hz = mlp_hz,
      .dronet_sensor_frequency_hz = dronet_sensor_hz,
      .mlp_sensor_frequency_hz = mlp_sensor_hz,
      .report_frequency_hz = report_hz,
      .run_duration_ms = (int64_t)(duration_s * 1000.0),
  };

  fprintf(stdout,
          "Dual-model baseline config:\n"
          "  dronet_vmfb = %s\n"
          "  mlp_vmfb    = %s\n"
          "  dronet_fn   = %s\n"
          "  mlp_fn      = %s\n"
          "  driver      = %s\n"
          "  mlp_hz      = %.3f\n"
          "  dronet_sensor_hz = %.3f\n"
          "  mlp_sensor_hz    = %.3f\n"
          "  report_hz   = %.3f\n"
          "  duration_ms = %lld\n",
          config.dronet_vmfb_path, config.mlp_vmfb_path, config.dronet_function,
          config.mlp_function, config.driver_name, config.mlp_frequency_hz,
          config.dronet_sensor_frequency_hz, config.mlp_sensor_frequency_hz,
          config.report_frequency_hz, (long long)config.run_duration_ms);
  fflush(stdout);

  return merlin_dual_model_runtime_run(&config);
}
