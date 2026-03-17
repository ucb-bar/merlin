// dispatch_output.h — Trace CSV, summary JSON, and DOT graph output.

#ifndef DISPATCH_OUTPUT_H_
#define DISPATCH_OUTPUT_H_

#include <inttypes.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include "dispatch_types.h"
#include "runtime_scheduler.h"

//------------------------------------------------------------------------------
// Trace CSV writer
//------------------------------------------------------------------------------

struct TraceWriter {
	std::mutex mu;
	FILE *f = nullptr;
	bool wrote_header = false;

	bool Open(const char *path) {
		if (!path || !path[0])
			return false;
		f = fopen(path, "wb");
		if (!f)
			return false;
		setvbuf(f, nullptr, _IOFBF, 1 << 20);
		return true;
	}

	void Close() {
		if (f) {
			fflush(f);
			fclose(f);
		}
		f = nullptr;
		wrote_header = false;
	}

	void Header() {
		if (!f || wrote_header)
			return;
		fprintf(f,
			"graph_iter,dispatch_key,dispatch_id,ordinal,total,"
			"job_name,module_name,target,vmfb_path,"
			"planned_start_us,eligible_us,submit_us,complete_us,"
			"residency_us,dep_slip_us,planned_duration_us,"
			"ready_us,start_us,end_us,queue_delay_us,run_us,total_latency_"
			"us\n");
		wrote_header = true;
	}

	void WriteRow(int graph_iter, const DispatchNode &node,
		uint64_t planned_start_us, uint64_t ready_us, uint64_t start_us,
		uint64_t end_us) {
		if (!f)
			return;

		const uint64_t eligible_us = ready_us;
		const uint64_t submit_us = start_us;
		const uint64_t complete_us = end_us;
		const uint64_t residency_us =
			complete_us >= eligible_us ? (complete_us - eligible_us) : 0;
		const uint64_t dep_slip_us = eligible_us >= planned_start_us
			? (eligible_us - planned_start_us)
			: 0;
		const uint64_t queue_delay_us =
			start_us >= ready_us ? (start_us - ready_us) : 0;
		const uint64_t run_us = end_us >= start_us ? (end_us - start_us) : 0;
		const uint64_t total_latency_us =
			end_us >= planned_start_us ? (end_us - planned_start_us) : 0;

		std::lock_guard<std::mutex> lock(mu);
		Header();
		fprintf(f,
			"%d,%s,%d,%d,%d,"
			"%s,%s,%s,%s,"
			"%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
			"%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
			"%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
			"%" PRIu64 ",%" PRIu64 "\n",
			graph_iter, node.key.c_str(), node.id, node.ordinal, node.total,
			node.job_name.c_str(), node.module_name.c_str(),
			HardwareTargetName(node.hardware_target),
			node.vmfb_path_resolved.c_str(), planned_start_us, eligible_us,
			submit_us, complete_us, residency_us, dep_slip_us,
			MsToUs(node.planned_duration_ms), ready_us, start_us, end_us,
			queue_delay_us, run_us, total_latency_us);
	}
};

//------------------------------------------------------------------------------
// JSON / DOT output helpers
//------------------------------------------------------------------------------

inline void JsonWriteEscaped(FILE *f, const std::string &s) {
	fputc('"', f);
	for (char c : s) {
		switch (c) {
			case '\\':
				fputs("\\\\", f);
				break;
			case '"':
				fputs("\\\"", f);
				break;
			case '\n':
				fputs("\\n", f);
				break;
			case '\r':
				fputs("\\r", f);
				break;
			case '\t':
				fputs("\\t", f);
				break;
			default:
				fputc(c, f);
				break;
		}
	}
	fputc('"', f);
}

inline std::string DotEscapeLabel(const std::string &s) {
	std::string out;
	out.reserve(s.size() + 8);
	for (char c : s) {
		if (c == '"')
			out += "\\\"";
		else if (c == '\\')
			out += "\\\\";
		else if (c == '\n')
			out += "\\n";
		else
			out.push_back(c);
	}
	return out;
}

inline bool WriteSummaryJson(const char *path,
	const dispatch_graph_config_t *cfg, const GraphModel &model,
	const std::vector<int> &topo_order) {
	if (!path || !path[0])
		return true;
	FILE *f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "Failed to open out_json: %s\n", path);
		return false;
	}

	fprintf(f, "{\n");
	fprintf(f, "  \"config\": {\n");
	fprintf(f, "    \"graph_json_path\": ");
	JsonWriteEscaped(f, cfg->graph_json_path ? cfg->graph_json_path : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"driver\": ");
	JsonWriteEscaped(f, cfg->driver_name ? cfg->driver_name : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"graph_iters\": %d,\n", cfg->graph_iters);
	fprintf(f, "    \"dispatch_iters\": %d,\n", cfg->dispatch_iters);
	fprintf(f, "    \"report_every\": %d,\n", cfg->report_every);
	fprintf(f, "    \"vmfb_root_dir\": ");
	JsonWriteEscaped(f, cfg->vmfb_root_dir ? cfg->vmfb_root_dir : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"cpu_p_cpu_ids\": ");
	JsonWriteEscaped(f, cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"cpu_e_cpu_ids\": ");
	JsonWriteEscaped(f, cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"visible_cores\": %d,\n", cfg->visible_cores);
	fprintf(f, "    \"schedule_makespan_ms\": %.6f\n", model.makespan_ms);
	fprintf(f, "  },\n");

	fprintf(f, "  \"nodes\": [\n");
	for (size_t i = 0; i < model.nodes.size(); ++i) {
		const auto &n = model.nodes[i];
		fprintf(f, "    {\n");
		fprintf(f, "      \"key\": ");
		JsonWriteEscaped(f, n.key);
		fprintf(f, ",\n");
		fprintf(f, "      \"id\": %d,\n", n.id);
		fprintf(f, "      \"ordinal\": %d,\n", n.ordinal);
		fprintf(f, "      \"total\": %d,\n", n.total);
		fprintf(f, "      \"job_name\": ");
		JsonWriteEscaped(f, n.job_name);
		fprintf(f, ",\n");
		fprintf(f, "      \"target\": ");
		JsonWriteEscaped(f, HardwareTargetName(n.hardware_target));
		fprintf(f, ",\n");
		fprintf(f, "      \"start_time_ms\": %.6f,\n", n.start_time_ms);
		fprintf(
			f, "      \"planned_duration_ms\": %.6f,\n", n.planned_duration_ms);
		fprintf(f, "      \"module_name\": ");
		JsonWriteEscaped(f, n.module_name);
		fprintf(f, ",\n");
		fprintf(f, "      \"vmfb_path_resolved\": ");
		JsonWriteEscaped(f, n.vmfb_path_resolved);
		fprintf(f, ",\n");
		fprintf(f, "      \"time_dependency\": ");
		JsonWriteEscaped(f, n.time_dependency);
		fprintf(f, ",\n");

		fprintf(f, "      \"dependencies\": [");
		for (size_t d = 0; d < n.deps.size(); ++d) {
			if (d)
				fprintf(f, ", ");
			JsonWriteEscaped(f, n.deps[d]);
		}
		fprintf(f, "],\n");

		fprintf(f, "      \"all_predecessors\": [");
		for (size_t d = 0; d < n.all_predecessors.size(); ++d) {
			if (d)
				fprintf(f, ", ");
			JsonWriteEscaped(f, n.all_predecessors[d]);
		}
		fprintf(f, "],\n");

		fprintf(f, "      \"run_stats\": {\n");
		fprintf(f, "        \"count\": %" PRIu64 ",\n", n.run_stats.count);
		fprintf(f, "        \"avg_ms\": %.6f,\n", n.run_stats.AvgMs());
		fprintf(f, "        \"p50_ms\": %.6f,\n", n.run_stats.P50Ms());
		fprintf(f, "        \"p90_ms\": %.6f,\n", n.run_stats.P90Ms());
		fprintf(f, "        \"p99_ms\": %.6f,\n", n.run_stats.P99Ms());
		fprintf(f, "        \"min_ms\": %.6f,\n", n.run_stats.MinMs());
		fprintf(f, "        \"max_ms\": %.6f\n", n.run_stats.MaxMs());
		fprintf(f, "      }\n");

		fprintf(f, "    }%s\n", (i + 1 < model.nodes.size()) ? "," : "");
	}
	fprintf(f, "  ],\n");

	fprintf(f, "  \"topo_order\": [");
	for (size_t i = 0; i < topo_order.size(); ++i) {
		if (i)
			fprintf(f, ", ");
		JsonWriteEscaped(
			f, model.nodes[static_cast<size_t>(topo_order[i])].key);
	}
	fprintf(f, "]\n");
	fprintf(f, "}\n");

	fclose(f);
	return true;
}

inline bool WriteDotGraph(const char *path, const GraphModel &model) {
	if (!path || !path[0])
		return true;
	FILE *f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "Failed to open out_dot: %s\n", path);
		return false;
	}

	fprintf(f, "digraph dispatch_graph {\n");
	fprintf(f, "  rankdir=LR;\n");
	fprintf(f, "  node [shape=box];\n");

	for (const auto &n : model.nodes) {
		const std::string label = n.key + "\\njob=" + n.job_name +
			"\\ntarget=" + std::string(HardwareTargetName(n.hardware_target)) +
			"\\nstart=" + std::to_string(n.start_time_ms) + "ms" +
			"\\nplan=" + std::to_string(n.planned_duration_ms) + "ms" +
			"\\nrun_avg=" + std::to_string(n.run_stats.AvgMs()) + "ms";

		fprintf(f, "  \"%s\" [label=\"%s\"];\n", n.key.c_str(),
			DotEscapeLabel(label).c_str());
	}

	for (const auto &n : model.nodes) {
		for (const auto &dep : n.deps) {
			fprintf(f, "  \"%s\" -> \"%s\";\n", dep.c_str(), n.key.c_str());
		}
		if (!n.time_dependency.empty()) {
			bool already = false;
			for (const auto &dep : n.deps) {
				if (dep == n.time_dependency) {
					already = true;
					break;
				}
			}
			if (!already) {
				fprintf(f,
					"  \"%s\" -> \"%s\" [style=dashed,label=\"time\"];\n",
					n.time_dependency.c_str(), n.key.c_str());
			}
		}
	}

	fprintf(f, "}\n");
	fclose(f);
	return true;
}

#endif // DISPATCH_OUTPUT_H_
