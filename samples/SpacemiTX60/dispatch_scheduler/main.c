// samples/SpacemiTX60/b_dispatch_level_model_async/main.c

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

#include "core/cli_utils.h"
#include "xpu-rt/scheduler_runner.h"

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
		"Flags:\n"
		"  --vmfb_dir=<path>           Directory containing per-dispatch .vmfb "
		"files\n"
		"  --cpu_p_cpu_ids=0,1,2,3     Exactly 4 logical CPUs for CPU_P\n"
		"  --cpu_e_cpu_ids=4,5         Exactly 2 logical CPUs for CPU_E\n"
		"  --visible_cores=8           Validate IDs are in [0, visible_cores)\n"
		"  --out_json=<path>           Write summary JSON\n"
		"  --out_dot=<path>            Write DOT graph\n"
		"  --trace_csv=<path>          Write trace CSV\n"
		"  --variant_p=RVV             ISA variant dir for CPU_P VMFBs\n"
		"  --variant_e=scalar          ISA variant dir for CPU_E VMFBs\n"
		"  --pin_per_core[=1]          Honour CPU_P#N placement: one pinned\n"
		"                              device per core instead of one per\n"
		"                              cluster. Required for a schedule built\n"
		"                              from single-core profiles.\n"
		"\n"
		"Notes:\n"
		"  - This runner is strict and expects async-external dispatch VMFB "
		"exports.\n"
		"  - It does not create application worker threads; IREE local-task "
		"does the work.\n"
		"\n"
		"Example:\n"
		"  %s real_multi_model_scheduling.json local-task 10 1 1 \\\n"
		"     --vmfb_dir=/home/baseline/dispatches \\\n"
		"     --cpu_p_cpu_ids=0,1,2,3 --cpu_e_cpu_ids=4,5 --visible_cores=8 "
		"\\\n"
		"     --out_json=run.json --out_dot=run.dot --trace_csv=trace.csv\n",
		argv0, argv0);
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
	const char *cpu_p_cpu_ids = "0,1,2,3";
	const char *cpu_e_cpu_ids = "4,5";
	int visible_cores = 8;
	const char *out_json = NULL;
	const char *out_dot = NULL;
	const char *trace_csv = NULL;
	/* ISA variant directories used to resolve per-dispatch VMFBs. These were
	 * hardcoded to RVV/scalar, which silently decided a scheduling experiment:
	 * a schedule built from RVV profiles on BOTH clusters would have loaded
	 * scalar binaries on CPU_E and been measured against the wrong timings.
	 * On the K1 both clusters implement RVV identically (measured: 0.996 median
	 * ratio on compute-bound dispatches), so RVV/RVV is a legitimate and in fact
	 * the expected configuration. Defaults preserve the old behaviour. */
	const char *variant_p = "RVV";
	const char *variant_e = "scalar";
	/* Off by default: honouring per-core placement changes what the run
	 * measures, so it should be asked for explicitly. */
	int pin_per_core = 0;

	for (int i = 6; i < argc; ++i) {
		const char *v = NULL;
		if ((v = get_flag_value(argv[i], "--vmfb_dir"))) {
			vmfb_dir = v;
		} else if ((v = get_flag_value(argv[i], "--cpu_p_cpu_ids"))) {
			cpu_p_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--cpu_e_cpu_ids"))) {
			cpu_e_cpu_ids = v;
		} else if ((v = get_flag_value(argv[i], "--visible_cores"))) {
			visible_cores = parse_int_or_default(v, 8);
		} else if ((v = get_flag_value(argv[i], "--out_json"))) {
			out_json = v;
		} else if ((v = get_flag_value(argv[i], "--out_dot"))) {
			out_dot = v;
		} else if ((v = get_flag_value(argv[i], "--trace_csv"))) {
			trace_csv = v;
		} else if ((v = get_flag_value(argv[i], "--variant_p"))) {
			variant_p = v;
		} else if ((v = get_flag_value(argv[i], "--variant_e"))) {
			variant_e = v;
		} else if ((v = get_flag_value(argv[i], "--pin_per_core"))) {
			pin_per_core = parse_int_or_default(v, 0);
		} else if (strcmp(argv[i], "--pin_per_core") == 0) {
			pin_per_core = 1;
		} else {
			fprintf(stderr, "Unknown arg: %s\n\n", argv[i]);
			print_usage(argv[0]);
			return 1;
		}
	}

	/* IME on cluster 1 traps; refuse rather than SIGILL on the first dispatch.
	 *
	 * Measured with a per-core SIGILL probe (see
	 * artifacts/k1_bringup/ * /ime_capability_probe.txt): smt.vmadot executes
	 * on cores 0-3 and traps on cores 4-7. xpu-rt/capabilities.py rejects the
	 * placement when a schedule is BUILT, but this is the only layer that sees
	 * the actual invocation -- a hand-typed --variant_e=IME bypasses the
	 * scheduler entirely, and until now it was accepted without comment.
	 *
	 * Checked by prefix so IME_ukernel and any future IME variant are covered. */
	if (strncmp(variant_e, "IME", 3) == 0) {
		fprintf(stderr,
			"--variant_e=%s: cluster 1 (cores %s) does not implement "
			"smt.vmadot -- an IME kernel there does not run slowly, it "
			"traps with SIGILL. IME is legal on cluster 0 only.\n",
			variant_e, cpu_e_cpu_ids);
		return 1;
	}

	scheduler_runner_config_t cfg;
	memset(&cfg, 0, sizeof(cfg));
	cfg.graph_json_path = json_path;
	cfg.driver_name = driver;
	cfg.graph_iters = graph_iters;
	cfg.dispatch_iters = dispatch_iters;
	cfg.report_every = report_every;
	cfg.vmfb_root_dir = vmfb_dir;
	cfg.cpu_p_cpu_ids = cpu_p_cpu_ids;
	cfg.cpu_e_cpu_ids = cpu_e_cpu_ids;
	cfg.visible_cores = visible_cores;
	cfg.out_json_path = out_json;
	cfg.out_dot_path = out_dot;
	cfg.trace_csv_path = trace_csv;

	// SpacemiT X60-specific target configuration.
	cfg.target_platform = "spacemit_x60";
	cfg.pin_per_core = pin_per_core;
	cfg.variant_p_dir = variant_p;
	cfg.variant_e_dir = variant_e;
	cfg.elf_marker = "_embedded_elf_riscv_64";

	return scheduler_runner_run(&cfg);
}
