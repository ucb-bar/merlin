/** @file dispatch_output.h
 *  @brief Trace CSV writer, DOT graph output, and JSON serialization helpers.
 *
 *  Generic output utilities for dispatch graph runners. These do not depend on
 *  any sample-specific config struct; sample-specific output (e.g., config
 *  sections in summary JSON) should be handled locally by each sample.
 */

#ifndef MERLIN_DISPATCH_OUTPUT_H_
#define MERLIN_DISPATCH_OUTPUT_H_

#include <inttypes.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include "dispatch/dispatch_types.h"

namespace merlin_bench {

//------------------------------------------------------------------------------
// Trace CSV writer
//------------------------------------------------------------------------------

/** @brief Thread-safe CSV writer for per-dispatch execution traces.
 *
 *  Writes one row per dispatch invocation with planned vs. actual timing
 *  columns. The header is written lazily on the first WriteRow() call.
 */
struct TraceWriter {
	std::mutex mu; /**< Guards all file I/O. */
	FILE *f = nullptr; /**< Output file handle. */
	bool wrote_header = false; /**< True after CSV header is written. */

	/** @brief Open the trace file for writing.
	 *  @param path Output file path (must be non-null and non-empty).
	 *  @return True if the file was opened successfully.
	 */
	bool Open(const char *path) {
		if (!path || !path[0])
			return false;
		f = fopen(path, "wb");
		if (!f)
			return false;
		setvbuf(f, nullptr, _IOFBF, 1 << 20);
		return true;
	}

	/** @brief Flush and close the trace file. */
	void Close() {
		if (f) {
			fflush(f);
			fclose(f);
		}
		f = nullptr;
		wrote_header = false;
	}

	/** @brief Write the CSV header row (idempotent). */
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

	/** @brief Write a single dispatch trace row.
	 *  @param graph_iter      Current graph iteration index.
	 *  @param node            The dispatch node that was executed.
	 *  @param planned_start_us Planned start time in microseconds.
	 *  @param ready_us        Time the node became ready (us).
	 *  @param start_us        Actual execution start time (us).
	 *  @param end_us          Actual execution end time (us).
	 *  @param target_name     Human-readable target name (overrides legacy
	 *                         lookup). Pass nullptr to use
	 * HardwareTargetName().
	 */
	void WriteRow(int graph_iter, const DispatchNode &node,
		uint64_t planned_start_us, uint64_t ready_us, uint64_t start_us,
		uint64_t end_us, const char *target_name = nullptr) {
		if (!f)
			return;

		const char *tname = target_name
			? target_name
			: HardwareTargetName(node.hardware_target);

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
			node.job_name.c_str(), node.module_name.c_str(), tname,
			node.vmfb_path_resolved.c_str(), planned_start_us, eligible_us,
			submit_us, complete_us, residency_us, dep_slip_us,
			MsToUs(node.planned_duration_ms), ready_us, start_us, end_us,
			queue_delay_us, run_us, total_latency_us);
	}
};

//------------------------------------------------------------------------------
// JSON / DOT output helpers
//------------------------------------------------------------------------------

/** @brief Write a JSON-escaped string (with surrounding quotes) to a file.
 *  @param f File handle.
 *  @param s String to escape and write.
 */
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

/** @brief Escape a string for use as a DOT graph label.
 *  @param s Raw label text.
 *  @return Escaped string safe for DOT label attributes.
 */
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

/** @brief Write the dispatch graph as a Graphviz DOT file.
 *  @param path  Output file path (no-op if null or empty).
 *  @param model Graph model containing nodes and edges.
 *  @return True on success or if path is null/empty (no-op).
 */
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

/** @brief Write dispatch node data as a JSON array fragment.
 *
 *  Writes the "nodes" key and its array value. Does not write surrounding
 *  object braces, so callers can compose sample-specific summary JSON.
 *
 *  @param f     Output file handle.
 *  @param nodes Dispatch nodes to serialize.
 */
inline void WriteNodesJson(FILE *f, const std::vector<DispatchNode> &nodes) {
	fprintf(f, "  \"nodes\": [\n");
	for (size_t i = 0; i < nodes.size(); ++i) {
		const auto &n = nodes[i];
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

		fprintf(f, "    }%s\n", (i + 1 < nodes.size()) ? "," : "");
	}
	fprintf(f, "  ]");
}

/** @brief Write topological order as a JSON array fragment.
 *  @param f          Output file handle.
 *  @param nodes      Dispatch nodes (used to map indices to keys).
 *  @param topo_order Topological ordering (indices into @p nodes).
 */
inline void WriteTopoOrderJson(FILE *f, const std::vector<DispatchNode> &nodes,
	const std::vector<int> &topo_order) {
	fprintf(f, "  \"topo_order\": [");
	for (size_t i = 0; i < topo_order.size(); ++i) {
		if (i)
			fprintf(f, ", ");
		JsonWriteEscaped(f, nodes[static_cast<size_t>(topo_order[i])].key);
	}
	fprintf(f, "]");
}

} // namespace merlin_bench

#endif // MERLIN_DISPATCH_OUTPUT_H_
