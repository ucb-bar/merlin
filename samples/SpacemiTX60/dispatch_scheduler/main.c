// samples/SpacemiTX60/dispatch_scheduler/main.c

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

#include "core/cli_utils.h"
#include "xpu-rt/scheduler_runner.h"

#define MAX_CLI_TARGETS 16

static void print_usage(const char *argv0) {
	fprintf(stderr,
		"Usage:\n"
		"  %s <dispatch_schedule.json> [driver] [graph_iters] [dispatch_iters] "
		"[report_every] [--flags]\n"
		"\n"
		"Defaults:\n"
		"  driver          = local-task\n"
		"  graph_iters     = 1\n"
		"  dispatch_iters  = 1\n"
		"  report_every    = 0 (final only)\n"
		"\n"
		"N-target flags (repeatable):\n"
		"  --target=NAME:CPU_IDS:VARIANT   Define a target (e.g. "
		"CPU_P:0,1,2,3:RVV)\n"
		"\n"
		"Legacy 2-target flags:\n"
		"  --cpu_p_cpu_ids=0,1,2,3     Logical CPUs for CPU_P\n"
		"  --cpu_e_cpu_ids=4,5         Logical CPUs for CPU_E\n"
		"  --visible_cores=8           Validate IDs are in [0, visible_cores)\n"
		"\n"
		"Common flags:\n"
		"  --vmfb_dir=<path>           Directory containing per-dispatch .vmfb "
		"files\n"
		"  --warmup_iters=N            Graph iterations to run before tracing "
		"(default 0)\n"
		"  --out_json=<path>           Write summary JSON\n"
		"  --out_dot=<path>            Write DOT graph\n"
		"  --trace_csv=<path>          Write trace CSV\n"
		"\n"
		"Notes:\n"
		"  - --target flags take precedence over "
		"--cpu_p_cpu_ids/--cpu_e_cpu_ids.\n"
		"  - If no --target flags are given, defaults to CPU_P:0,1,2,3:RVV + "
		"CPU_E:4,5:scalar.\n"
		"  - This runner expects async-external dispatch VMFB exports.\n"
		"\n"
		"Example (N-target):\n"
		"  %s schedule.json local-task 10 1 1 \\\n"
		"     --target=CPU_P:0,1,2,3:RVV --target=CPU_E:4,5:scalar \\\n"
		"     --target=CPU_AUX:6,7:scalar \\\n"
		"     --vmfb_dir=/home/baseline/dispatches \\\n"
		"     --visible_cores=8 --trace_csv=trace.csv\n"
		"\n"
		"Example (legacy 2-target):\n"
		"  %s schedule.json local-task 10 1 1 \\\n"
		"     --cpu_p_cpu_ids=0,1,2,3 --cpu_e_cpu_ids=4,5 --visible_cores=8\n",
		argv0, argv0, argv0);
}

// Parse --target=NAME:CPU_IDS:VARIANT into components.
// Returns 1 on success, 0 on parse failure.
static int parse_target_flag(const char *value, const char **out_name,
	const char **out_cpu_ids, const char **out_variant) {
	// We need to split on ':'. We duplicate the string so we can insert NULs.
	// The caller owns the original storage (argv), but we need mutable copies.
	static char target_bufs[MAX_CLI_TARGETS][256];
	static int target_buf_idx = 0;

	if (target_buf_idx >= MAX_CLI_TARGETS)
		return 0;

	char *buf = target_bufs[target_buf_idx++];
	size_t len = strlen(value);
	if (len >= 256)
		return 0;
	memcpy(buf, value, len + 1);

	// Split: NAME:CPU_IDS:VARIANT
	char *first_colon = strchr(buf, ':');
	if (!first_colon)
		return 0;
	*first_colon = '\0';
	*out_name = buf;

	char *rest = first_colon + 1;
	char *second_colon = strchr(rest, ':');
	if (!second_colon)
		return 0;
	*second_colon = '\0';
	*out_cpu_ids = rest;
	*out_variant = second_colon + 1;
	return 1;
}

int main(int argc, char **argv) {
	// Parse IREE global flags first (e.g. --task_topology_cpu_ids=0,1,2,3).
	// This removes consumed flags from argv so our arg parser sees only ours.
	iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);

	if (argc < 2) {
		print_usage(argv[0]);
		return 1;
	}

	const char *json_path = argv[1];
	const char *driver = (argc >= 3) ? argv[2] : "local-task";
	const int graph_iters = (argc >= 4) ? parse_int_or_default(argv[3], 1) : 1;
	const int dispatch_iters =
		(argc >= 5) ? parse_int_or_default(argv[4], 1) : 1;
	const int report_every = (argc >= 6) ? parse_int_or_default(argv[5], 0) : 0;

	const char *vmfb_dir = NULL;
	int visible_cores = 8;
	int warmup_iters = 0;
	const char *out_json = NULL;
	const char *out_dot = NULL;
	const char *trace_csv = NULL;

	// N-target arrays.
	int num_targets = 0;
	const char *target_names[MAX_CLI_TARGETS];
	const char *target_cpu_ids[MAX_CLI_TARGETS];
	const char *target_variant_dirs[MAX_CLI_TARGETS];

	// Legacy 2-target fields.
	const char *cpu_p_cpu_ids = NULL;
	const char *cpu_e_cpu_ids = NULL;

	for (int i = 6; i < argc; ++i) {
		const char *v = NULL;
		if ((v = get_flag_value(argv[i], "--vmfb_dir"))) {
			vmfb_dir = v;
		} else if ((v = get_flag_value(argv[i], "--target"))) {
			const char *name = NULL;
			const char *ids = NULL;
			const char *variant = NULL;
			if (!parse_target_flag(v, &name, &ids, &variant)) {
				fprintf(stderr,
					"Invalid --target format: '%s'\n"
					"Expected: --target=NAME:CPU_IDS:VARIANT\n",
					v);
				return 1;
			}
			if (num_targets >= MAX_CLI_TARGETS) {
				fprintf(stderr, "Too many --target flags (max %d)\n",
					MAX_CLI_TARGETS);
				return 1;
			}
			target_names[num_targets] = name;
			target_cpu_ids[num_targets] = ids;
			target_variant_dirs[num_targets] = variant;
			num_targets++;
		} else if ((v = get_flag_value(argv[i], "--cpu_p_cpu_ids"))) {
			cpu_p_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--cpu_e_cpu_ids"))) {
			cpu_e_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--visible_cores"))) {
			visible_cores = parse_int_or_default(v, 8);
		} else if ((v = get_flag_value(argv[i], "--warmup_iters"))) {
			warmup_iters = parse_int_or_default(v, 0);
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

	scheduler_runner_config_t cfg;
	memset(&cfg, 0, sizeof(cfg));
	cfg.graph_json_path = json_path;
	cfg.driver_name = driver;
	cfg.graph_iters = graph_iters;
	cfg.warmup_iters = warmup_iters;
	cfg.dispatch_iters = dispatch_iters;
	cfg.report_every = report_every;
	cfg.vmfb_root_dir = vmfb_dir;
	cfg.visible_cores = visible_cores;
	cfg.out_json_path = out_json;
	cfg.out_dot_path = out_dot;
	cfg.trace_csv_path = trace_csv;

	// SpacemiT X60-specific target configuration.
	cfg.target_platform = "spacemit_x60";
	cfg.elf_marker = "_embedded_elf_riscv_64";

	if (num_targets > 0) {
		// N-target mode from --target flags.
		cfg.num_targets = num_targets;
		cfg.target_names = target_names;
		cfg.target_cpu_ids = target_cpu_ids;
		cfg.target_variant_dirs = target_variant_dirs;
	} else if (cpu_p_cpu_ids) {
		// Legacy 2-target mode from --cpu_p_cpu_ids / --cpu_e_cpu_ids.
		cfg.cpu_p_cpu_ids = cpu_p_cpu_ids;
		cfg.cpu_e_cpu_ids = cpu_e_cpu_ids ? cpu_e_cpu_ids : "4,5";
		cfg.variant_p_dir = "RVV";
		cfg.variant_e_dir = "scalar";
	} else {
		// Default: SpacemiT X60 with 2 targets.
		cfg.cpu_p_cpu_ids = "0,1,2,3";
		cfg.cpu_e_cpu_ids = "4,5";
		cfg.variant_p_dir = "RVV";
		cfg.variant_e_dir = "scalar";
	}

	return scheduler_runner_run(&cfg);
}
