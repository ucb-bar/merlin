#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_dispatch_graph.h"

static void print_usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <dispatch_graph.json> [driver] [graph_iters] [dispatch_iters] [report_every]\n"
          "\n"
          "Defaults:\n"
          "  driver         = local-task\n"
          "  graph_iters     = 1\n"
          "  dispatch_iters  = 1 (passed as i32 arg if entry takes i32)\n"
          "  report_every    = 0 (only final summary)\n"
          "\n"
          "Optional flags:\n"
          "  --core_mask=0x3        (best-effort local-task pinning)\n"
          "  --parallelism=4        (run ready nodes in parallel; default 1)\n"
          "\n"
          "Example:\n"
          "  %s graph.json local-task 100 10 10 --core_mask=0x3 --parallelism=4\n",
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

  // Remaining args can be flags.
  for (int i = 6; i < argc; ++i) {
    const char* v = NULL;
    if ((v = get_flag_value(argv[i], "--core_mask"))) {
      core_mask = parse_u64_hex_or_default(v, 0);
    } else if ((v = get_flag_value(argv[i], "--parallelism"))) {
      parallelism = parse_int_or_default(v, 1);
      if (parallelism < 1) parallelism = 1;
      if (parallelism > 128) parallelism = 128;
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

  return dispatch_graph_run(&cfg);
}