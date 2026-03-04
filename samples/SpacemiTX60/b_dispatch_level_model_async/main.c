#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_scheduler.h"

static void print_usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <dispatch_graph.json> [driver] [graph_iters] [dispatch_iters] [report_every] [--flags]\n"
          "\n"
          "Defaults:\n"
          "  driver         = local-task\n"
          "  graph_iters     = 1\n"
          "  dispatch_iters  = 1 (passed as i32 if entry takes i32)\n"
          "  report_every    = 0 (only final summary)\n"
          "\n"
          "Flags:\n"
          "  --vmfb_dir=<path>        Directory containing per-dispatch .vmfb files\n"
          "                           (recommended on-board)\n"
          "  --core_mask=0x3          Best-effort local-task pinning bitmask\n"
          "  --cores=1..64            Convenience: sets core_mask=(1<<cores)-1\n"
          "  --parallelism=1..128     Run ready nodes in parallel (default 1)\n"
          "  --out_json=<path>        Write summary JSON for plotting/reconstruction\n"
          "  --out_dot=<path>         Write DOT graph with latency labels\n"
          "  --trace_csv=<path>       Write per-dispatch timeline CSV (start/dur/thread)\n"
          "\n"
          "Examples:\n"
          "  %s dronet.q.int8_dispatch_graph.json local-task 100 1 10 \\\n"
          "     --vmfb_dir=/home/baseline/dispatches --cores=4 --parallelism=4 \\\n"
          "     --out_json=run.json --out_dot=run.dot --trace_csv=trace.csv\n",
          argv0, argv0);
}

static int parse_int_or_default(const char* text, int dflt) {
  if (!text) return dflt;
  char* end = NULL;
  long v = strtol(text, &end, 10);
  return (end == text) ? dflt : (int)v;
}

static uint64_t parse_u64_hex_or_default(const char* text, uint64_t dflt) {
  if (!text) return dflt;
  char* end = NULL;
  unsigned long long v = strtoull(text, &end, 0);
  return (end == text) ? dflt : (uint64_t)v;
}

// Minimal flag parser: --key=value
static const char* get_flag_value(const char* arg, const char* key) {
  size_t klen = strlen(key);
  if (strncmp(arg, key, klen) != 0) return NULL;
  if (arg[klen] != '=') return NULL;
  return arg + klen + 1;
}

static uint64_t core_mask_from_cores(int cores) {
  if (cores <= 0) return 0;
  if (cores >= 64) return UINT64_MAX;
  return (1ull << (uint64_t)cores) - 1ull;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const char* json_path = argv[1];
  const char* driver = (argc >= 3) ? argv[2] : "local-task";
  const int graph_iters = (argc >= 4) ? parse_int_or_default(argv[3], 1) : 1;
  const int dispatch_iters = (argc >= 5) ? parse_int_or_default(argv[4], 1) : 1;
  const int report_every = (argc >= 6) ? parse_int_or_default(argv[5], 0) : 0;

  uint64_t core_mask = 0;
  int parallelism = 1;
  const char* vmfb_dir = NULL;
  const char* out_json = NULL;
  const char* out_dot = NULL;
  const char* trace_csv = NULL;

  // Remaining args are flags.
  for (int i = 6; i < argc; ++i) {
    const char* v = NULL;
    if ((v = get_flag_value(argv[i], "--core_mask"))) {
      core_mask = parse_u64_hex_or_default(v, 0);
    } else if ((v = get_flag_value(argv[i], "--cores"))) {
      int cores = parse_int_or_default(v, 0);
      if (cores < 0) cores = 0;
      core_mask = core_mask_from_cores(cores);
    } else if ((v = get_flag_value(argv[i], "--parallelism"))) {
      parallelism = parse_int_or_default(v, 1);
      if (parallelism < 1) parallelism = 1;
      if (parallelism > 128) parallelism = 128;
    } else if ((v = get_flag_value(argv[i], "--vmfb_dir"))) {
      vmfb_dir = v;
    } else if ((v = get_flag_value(argv[i], "--out_json"))) {
      out_json = v;
    } else if ((v = get_flag_value(argv[i], "--out_dot"))) {
      out_dot = v;
    } else if ((v = get_flag_value(argv[i], "--trace_csv"))) {
      trace_csv = v;
    } else {
      fprintf(stderr, "Unknown arg: %s\n\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  dispatch_graph_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.graph_json_path = json_path;
  cfg.driver_name = driver;
  cfg.graph_iters = graph_iters;
  cfg.dispatch_iters = dispatch_iters;
  cfg.report_every = report_every;
  cfg.core_mask = core_mask;
  cfg.parallelism = parallelism;
  cfg.vmfb_root_dir = vmfb_dir;
  cfg.out_json_path = out_json;
  cfg.out_dot_path = out_dot;
  cfg.trace_csv_path = trace_csv;

  return dispatch_graph_run(&cfg);
}