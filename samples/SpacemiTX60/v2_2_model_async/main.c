#include <stdio.h>
#include <stdlib.h>

#include "runtime_scheduler.h"

static double parse_double_or_default(const char* text, double dflt) {
  if (!text) return dflt;
  char* end = NULL;
  double v = strtod(text, &end);
  return (end == text) ? dflt : v;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
      "Usage:\n"
      "  %s <dronet.vmfb> <mlp.vmfb> [mlp_hz] [duration_s] "
      "[dronet_fn] [mlp_fn] [driver] [report_hz] "
      "[dronet_sensor_hz] [mlp_sensor_hz]\n\n"
      "Defaults:\n"
      "  mlp_hz           = 20.0\n"
      "  duration_s       = 10.0\n"
      "  dronet_fn        = dronet.main_graph$async\n"
      "  mlp_fn           = mlp.main_graph$async\n"
      "  driver           = local-task\n"
      "  report_hz        = 1.0\n"
      "  dronet_sensor_hz = 60.0\n"
      "  mlp_sensor_hz    = mlp_hz\n",
      argv[0]);
    return 1;
  }

  const char* dronet_vmfb = argv[1];
  const char* mlp_vmfb = argv[2];

  double mlp_hz = (argc >= 4) ? parse_double_or_default(argv[3], 20.0) : 20.0;
  double duration_s =
      (argc >= 5) ? parse_double_or_default(argv[4], 10.0) : 10.0;

  const char* dronet_fn =
      (argc >= 6) ? argv[5] : "dronet.main_graph$async";
  const char* mlp_fn =
      (argc >= 7) ? argv[6] : "mlp.main_graph$async";

  const char* driver =
      (argc >= 8) ? argv[7] : "local-task";

  double report_hz =
      (argc >= 9) ? parse_double_or_default(argv[8], 1.0) : 1.0;

  double dronet_sensor_hz =
      (argc >= 10) ? parse_double_or_default(argv[9], 60.0) : 60.0;
  double mlp_sensor_hz =
      (argc >= 11) ? parse_double_or_default(argv[10], mlp_hz) : mlp_hz;

  dual_model_async_config_t cfg = {
    .dronet_vmfb_path = dronet_vmfb,
    .mlp_vmfb_path = mlp_vmfb,
    .dronet_function = dronet_fn,
    .mlp_function = mlp_fn,
    .driver_name = driver,
    .mlp_frequency_hz = mlp_hz,
    .dronet_sensor_frequency_hz = dronet_sensor_hz,
    .mlp_sensor_frequency_hz = mlp_sensor_hz,
    .report_frequency_hz = report_hz,
    .run_duration_ms = (int64_t)(duration_s * 1000.0),
  };

  fprintf(stdout,
    "Dual-model async scheduler (input-driven):\n"
    "  dronet_vmfb=%s\n"
    "  mlp_vmfb   =%s\n"
    "  dronet_fn  =%s\n"
    "  mlp_fn     =%s\n"
    "  driver     =%s\n"
    "  mlp_hz     =%.3f\n"
    "  duration_ms=%lld\n"
    "  report_hz  =%.3f\n"
    "  dronet_sensor_hz=%.3f\n"
    "  mlp_sensor_hz   =%.3f\n",
    cfg.dronet_vmfb_path, cfg.mlp_vmfb_path,
    cfg.dronet_function, cfg.mlp_function,
    cfg.driver_name, cfg.mlp_frequency_hz,
    (long long)cfg.run_duration_ms, cfg.report_frequency_hz,
    cfg.dronet_sensor_frequency_hz, cfg.mlp_sensor_frequency_hz);
  fflush(stdout);

  return dual_model_async_scheduler_run(&cfg);
}