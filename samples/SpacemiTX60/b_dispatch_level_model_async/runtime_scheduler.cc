// samples/SpacemiTX60/b_dispatch_level_model_async/runtime_scheduler.cc

#include "runtime_scheduler.h"

#include <inttypes.h>
#include <math.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

namespace {

using Clock = std::chrono::steady_clock;

enum class HardwareTarget {
	kCpuP = 0,
	kCpuE = 1,
};

static const char *HardwareTargetName(HardwareTarget t) {
	switch (t) {
		case HardwareTarget::kCpuP:
			return "CPU_P";
		case HardwareTarget::kCpuE:
			return "CPU_E";
		default:
			return "UNKNOWN";
	}
}

static bool ParseHardwareTarget(const std::string &s, HardwareTarget *out) {
	if (s == "CPU_P") {
		*out = HardwareTarget::kCpuP;
		return true;
	}
	if (s == "CPU_E") {
		*out = HardwareTarget::kCpuE;
		return true;
	}
	return false;
}

static uint64_t MsToUs(double ms) {
	if (ms < 0.0)
		ms = 0.0;
	return (uint64_t)llround(ms * 1000.0);
}

// -----------------------------------------------------------------------------
// Stats
// -----------------------------------------------------------------------------

struct Log2Histogram {
	static constexpr int kBuckets = 64;
	uint64_t buckets[kBuckets];

	Log2Histogram() {
		Reset();
	}

	void Reset() {
		for (int i = 0; i < kBuckets; ++i)
			buckets[i] = 0;
	}

	static int BucketForUs(uint64_t us) {
		if (us == 0)
			return 0;
		int b = 0;
		while (us) {
			++b;
			us >>= 1;
			if (b >= kBuckets - 1)
				return kBuckets - 1;
		}
		return b;
	}

	void Add(uint64_t us) {
		buckets[BucketForUs(us)]++;
	}

	uint64_t Count() const {
		uint64_t c = 0;
		for (int i = 0; i < kBuckets; ++i)
			c += buckets[i];
		return c;
	}

	uint64_t ApproxPercentile(double pct) const {
		if (pct <= 0.0)
			return 0;
		if (pct >= 1.0)
			pct = 1.0;
		const uint64_t total = Count();
		if (total == 0)
			return 0;
		const uint64_t target = (uint64_t)((double)total * pct);
		uint64_t run = 0;
		for (int b = 0; b < kBuckets; ++b) {
			run += buckets[b];
			if (run >= target) {
				if (b == 0)
					return 0;
				if (b >= kBuckets - 1)
					return (uint64_t)-1;
				return (1ull << b);
			}
		}
		return (1ull << 63);
	}
};

struct RunningStats {
	uint64_t count = 0;
	uint64_t sum_us = 0;
	uint64_t min_us = UINT64_MAX;
	uint64_t max_us = 0;
	Log2Histogram hist;

	void Reset() {
		count = 0;
		sum_us = 0;
		min_us = UINT64_MAX;
		max_us = 0;
		hist.Reset();
	}

	void Add(uint64_t us) {
		++count;
		sum_us += us;
		if (us < min_us)
			min_us = us;
		if (us > max_us)
			max_us = us;
		hist.Add(us);
	}

	double AvgMs() const {
		if (count == 0)
			return 0.0;
		return ((double)sum_us / 1000.0) / (double)count;
	}

	double MinMs() const {
		return count == 0 ? 0.0 : ((double)min_us / 1000.0);
	}
	double MaxMs() const {
		return count == 0 ? 0.0 : ((double)max_us / 1000.0);
	}
	double P50Ms() const {
		return (double)hist.ApproxPercentile(0.50) / 1000.0;
	}
	double P90Ms() const {
		return (double)hist.ApproxPercentile(0.90) / 1000.0;
	}
	double P99Ms() const {
		return (double)hist.ApproxPercentile(0.99) / 1000.0;
	}
};

// -----------------------------------------------------------------------------
// Shared fatal state
// -----------------------------------------------------------------------------

struct SharedState {
	int fatal_code = IREE_STATUS_OK;
};

static bool HasFatal(const SharedState *s) {
	return s->fatal_code != IREE_STATUS_OK;
}

static void SetFatalOnce(SharedState *s, iree_status_t st, const char *tag) {
	if (iree_status_is_ok(st))
		return;
	if (s->fatal_code == IREE_STATUS_OK) {
		s->fatal_code = (int)iree_status_code(st);
		if (tag && tag[0])
			fprintf(stderr, "%s\n", tag);
		iree_status_fprint(stderr, st);
	}
	iree_status_ignore(st);
}

// -----------------------------------------------------------------------------
// Minimal JSON parser
// -----------------------------------------------------------------------------

struct JsonParser {
	const char *p = nullptr;
	const char *end = nullptr;

	void SkipWs() {
		while (p < end) {
			const char c = *p;
			if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
				++p;
				continue;
			}
			break;
		}
	}

	bool Consume(char expected) {
		SkipWs();
		if (p >= end || *p != expected)
			return false;
		++p;
		return true;
	}

	bool ParseString(std::string *out) {
		out->clear();
		SkipWs();
		if (p >= end || *p != '"')
			return false;
		++p;
		const char *start = p;
		bool has_escapes = false;
		while (p < end) {
			char c = *p++;
			if (c == '"')
				break;
			if (c == '\\') {
				has_escapes = true;
				if (p >= end)
					return false;
				++p;
			}
		}
		if (p > end)
			return false;
		const char *raw_end = p - 1;

		if (!has_escapes) {
			out->assign(start, raw_end - start);
			return true;
		}

		out->reserve((size_t)(raw_end - start));
		const char *r = start;
		while (r < raw_end) {
			char c = *r++;
			if (c != '\\') {
				out->push_back(c);
				continue;
			}
			if (r >= raw_end)
				return false;
			char e = *r++;
			switch (e) {
				case '"':
					out->push_back('"');
					break;
				case '\\':
					out->push_back('\\');
					break;
				case 'n':
					out->push_back('\n');
					break;
				case 'r':
					out->push_back('\r');
					break;
				case 't':
					out->push_back('\t');
					break;
				default:
					out->push_back(e);
					break;
			}
		}
		return true;
	}

	bool ParseInt(int *out) {
		*out = 0;
		SkipWs();
		if (p >= end)
			return false;
		char *endptr = nullptr;
		long v = strtol(p, &endptr, 10);
		if (endptr == p)
			return false;
		p = endptr;
		*out = (int)v;
		return true;
	}

	bool ParseDouble(double *out) {
		*out = 0.0;
		SkipWs();
		if (p >= end)
			return false;
		char *endptr = nullptr;
		double v = strtod(p, &endptr);
		if (endptr == p)
			return false;
		p = endptr;
		*out = v;
		return true;
	}

	bool SkipValue();
	bool SkipArray();
	bool SkipObject();
};

bool JsonParser::SkipArray() {
	if (!Consume('['))
		return false;
	SkipWs();
	if (Consume(']'))
		return true;
	while (true) {
		if (!SkipValue())
			return false;
		SkipWs();
		if (Consume(']'))
			return true;
		if (!Consume(','))
			return false;
	}
}

bool JsonParser::SkipObject() {
	if (!Consume('{'))
		return false;
	SkipWs();
	if (Consume('}'))
		return true;
	while (true) {
		std::string key;
		if (!ParseString(&key))
			return false;
		if (!Consume(':'))
			return false;
		if (!SkipValue())
			return false;
		SkipWs();
		if (Consume('}'))
			return true;
		if (!Consume(','))
			return false;
	}
}

bool JsonParser::SkipValue() {
	SkipWs();
	if (p >= end)
		return false;
	const char c = *p;
	if (c == '"') {
		std::string s;
		return ParseString(&s);
	} else if (c == '{') {
		return SkipObject();
	} else if (c == '[') {
		return SkipArray();
	} else if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
		double dummy = 0.0;
		return ParseDouble(&dummy);
	} else if (!strncmp(p, "true", 4)) {
		p += 4;
		return true;
	} else if (!strncmp(p, "false", 5)) {
		p += 5;
		return true;
	} else if (!strncmp(p, "null", 4)) {
		p += 4;
		return true;
	}
	return false;
}

// -----------------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------------

static std::string PathDirname(const std::string &path) {
	const size_t pos = path.find_last_of('/');
	if (pos == std::string::npos)
		return ".";
	if (pos == 0)
		return "/";
	return path.substr(0, pos);
}

static std::string PathJoin2(const std::string &a, const std::string &b) {
	if (a.empty())
		return b;
	if (b.empty())
		return a;
	if (b[0] == '/')
		return b;
	if (a.back() == '/')
		return a + b;
	return a + "/" + b;
}

static bool StartsWithDirPrefix(const std::string &s, const std::string &dir) {
	if (dir.empty())
		return false;
	if (s.size() < dir.size())
		return false;
	if (s.compare(0, dir.size(), dir) != 0)
		return false;
	if (s.size() == dir.size())
		return true;
	return s[dir.size()] == '/';
}

static std::string StripLeadingDir(
	const std::string &s, const std::string &dir) {
	if (!StartsWithDirPrefix(s, dir))
		return s;
	if (s.size() == dir.size())
		return std::string();
	return s.substr(dir.size() + 1);
}

static bool FileReadable(const std::string &path) {
	std::ifstream f(path, std::ios::binary);
	return (bool)f;
}

static bool EndsWith(const std::string &s, const std::string &suffix) {
	return s.size() >= suffix.size() &&
		s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// -----------------------------------------------------------------------------
// Schedule model
// -----------------------------------------------------------------------------

struct ModuleCallSignature {
	bool pass_dispatch_iters = false;
	std::string input_cc;
};

struct PerNodeModule {
	std::string vmfb_path;
	iree_runtime_session_t *session = nullptr;
	iree_vm_function_t entry_fn = {0};
	ModuleCallSignature sig;
};

struct DispatchNode {
	std::string key;
	int id = -1;
	int ordinal = 0;
	int total = 0;

	HardwareTarget hardware_target = HardwareTarget::kCpuP;
	double start_time_ms = 0.0;
	double planned_duration_ms = 0.0;

	std::string job_name;
	std::string module_name;
	std::string time_dependency;

	std::vector<std::string> deps;
	std::vector<std::string> all_predecessors;

	std::string vmfb_path_json;
	std::string vmfb_path_resolved;

	PerNodeModule module;
	RunningStats residency_stats;
};

struct GraphModel {
	std::vector<DispatchNode> nodes;
	std::string dispatch_vmfb_dir_from_json;
	double makespan_ms = 0.0;
};

static bool ParseDependenciesArray(
	JsonParser *jp, std::vector<std::string> *out) {
	out->clear();
	if (!jp->Consume('['))
		return false;
	jp->SkipWs();
	if (jp->Consume(']'))
		return true;
	while (true) {
		std::string dep;
		if (!jp->ParseString(&dep))
			return false;
		out->push_back(std::move(dep));
		jp->SkipWs();
		if (jp->Consume(']'))
			return true;
		if (!jp->Consume(','))
			return false;
	}
}

static bool ParseDispatchesObject(
	JsonParser *jp, std::vector<DispatchNode> *out_nodes) {
	out_nodes->clear();
	if (!jp->Consume('{'))
		return false;
	jp->SkipWs();
	if (jp->Consume('}'))
		return true;

	while (true) {
		std::string dispatch_key;
		if (!jp->ParseString(&dispatch_key))
			return false;
		if (!jp->Consume(':'))
			return false;
		if (!jp->Consume('{'))
			return false;

		DispatchNode node;
		node.key = std::move(dispatch_key);

		jp->SkipWs();
		if (!jp->Consume('}')) {
			while (true) {
				std::string field;
				if (!jp->ParseString(&field))
					return false;
				if (!jp->Consume(':'))
					return false;

				if (field == "id") {
					if (!jp->ParseInt(&node.id))
						return false;
				} else if (field == "ordinal") {
					if (!jp->ParseInt(&node.ordinal))
						return false;
				} else if (field == "total") {
					if (!jp->ParseInt(&node.total))
						return false;
				} else if (field == "hardware_target") {
					std::string s;
					if (!jp->ParseString(&s))
						return false;
					if (!ParseHardwareTarget(s, &node.hardware_target)) {
						fprintf(stderr, "Unknown hardware_target '%s' for %s\n",
							s.c_str(), node.key.c_str());
						return false;
					}
				} else if (field == "start_time") {
					if (!jp->ParseDouble(&node.start_time_ms))
						return false;
				} else if (field == "duration") {
					if (!jp->ParseDouble(&node.planned_duration_ms))
						return false;
				} else if (field == "job_name") {
					if (!jp->ParseString(&node.job_name))
						return false;
				} else if (field == "module_name") {
					if (!jp->ParseString(&node.module_name))
						return false;
				} else if (field == "time_dependency") {
					if (!jp->ParseString(&node.time_dependency))
						return false;
				} else if (field == "vmfb_path") {
					if (!jp->ParseString(&node.vmfb_path_json))
						return false;
				} else if (field == "dependencies") {
					if (!ParseDependenciesArray(jp, &node.deps))
						return false;
				} else {
					if (!jp->SkipValue())
						return false;
				}

				jp->SkipWs();
				if (jp->Consume('}'))
					break;
				if (!jp->Consume(','))
					return false;
			}
		}

		out_nodes->push_back(std::move(node));

		jp->SkipWs();
		if (jp->Consume('}'))
			return true;
		if (!jp->Consume(','))
			return false;
	}
}

static bool ParseMetadataObject(JsonParser *jp, double *out_makespan_ms) {
	*out_makespan_ms = 0.0;
	if (!jp->Consume('{'))
		return false;
	jp->SkipWs();
	if (jp->Consume('}'))
		return true;

	while (true) {
		std::string key;
		if (!jp->ParseString(&key))
			return false;
		if (!jp->Consume(':'))
			return false;
		if (key == "makespan") {
			if (!jp->ParseDouble(out_makespan_ms))
				return false;
		} else {
			if (!jp->SkipValue())
				return false;
		}
		jp->SkipWs();
		if (jp->Consume('}'))
			return true;
		if (!jp->Consume(','))
			return false;
	}
}

static bool ParseDispatchScheduleJson(
	const std::string &json_path, GraphModel *out_model) {
	out_model->nodes.clear();
	out_model->dispatch_vmfb_dir_from_json.clear();
	out_model->makespan_ms = 0.0;

	std::ifstream f(json_path, std::ios::binary);
	if (!f)
		return false;
	std::string buf(
		(std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (buf.empty())
		return false;

	JsonParser jp;
	jp.p = buf.data();
	jp.end = buf.data() + buf.size();

	if (!jp.Consume('{'))
		return false;

	bool saw_dispatches = false;
	while (true) {
		jp.SkipWs();
		if (jp.Consume('}'))
			break;

		std::string key;
		if (!jp.ParseString(&key))
			return false;
		if (!jp.Consume(':'))
			return false;

		if (key == "dispatches") {
			if (!ParseDispatchesObject(&jp, &out_model->nodes))
				return false;
			saw_dispatches = true;
		} else if (key == "dispatch_vmfb_dir") {
			if (!jp.ParseString(&out_model->dispatch_vmfb_dir_from_json))
				return false;
		} else if (key == "metadata") {
			if (!ParseMetadataObject(&jp, &out_model->makespan_ms))
				return false;
		} else {
			if (!jp.SkipValue())
				return false;
		}

		jp.SkipWs();
		if (jp.Consume('}'))
			break;
		if (!jp.Consume(','))
			return false;
	}

	return saw_dispatches && !out_model->nodes.empty();
}

static void ExpandAllPredecessors(std::vector<DispatchNode> *nodes) {
	for (auto &n : *nodes) {
		std::unordered_set<std::string> uniq;
		std::vector<std::string> all;
		all.reserve(n.deps.size() + (n.time_dependency.empty() ? 0 : 1));
		for (const auto &d : n.deps) {
			if (uniq.insert(d).second)
				all.push_back(d);
		}
		if (!n.time_dependency.empty() &&
			uniq.insert(n.time_dependency).second) {
			all.push_back(n.time_dependency);
		}
		n.all_predecessors = std::move(all);
	}
}

// -----------------------------------------------------------------------------
// Topological order
// -----------------------------------------------------------------------------

static bool TopoSort(const std::vector<DispatchNode> &nodes,
	std::vector<int> *out_order,
	std::vector<std::vector<int>> *out_dependents) {
	out_order->clear();
	out_dependents->assign(nodes.size(), {});

	std::unordered_map<std::string, int> index_of;
	index_of.reserve(nodes.size() * 2);
	for (int i = 0; i < (int)nodes.size(); ++i) {
		index_of[nodes[i].key] = i;
	}

	std::vector<int> indeg(nodes.size(), 0);
	for (int i = 0; i < (int)nodes.size(); ++i) {
		indeg[i] = (int)nodes[i].all_predecessors.size();
		for (const auto &pred : nodes[i].all_predecessors) {
			auto it = index_of.find(pred);
			if (it == index_of.end()) {
				fprintf(stderr, "Missing predecessor '%s' for node '%s'\n",
					pred.c_str(), nodes[i].key.c_str());
				return false;
			}
			(*out_dependents)[it->second].push_back(i);
		}
	}

	auto better = [&](int a, int b) {
		const auto &A = nodes[a];
		const auto &B = nodes[b];
		if (A.start_time_ms != B.start_time_ms)
			return A.start_time_ms < B.start_time_ms;
		if (A.id != B.id)
			return A.id < B.id;
		if (A.ordinal != B.ordinal)
			return A.ordinal < B.ordinal;
		return A.key < B.key;
	};

	std::vector<int> ready;
	ready.reserve(nodes.size());
	for (int i = 0; i < (int)nodes.size(); ++i) {
		if (indeg[i] == 0)
			ready.push_back(i);
	}

	while (!ready.empty()) {
		int best_i = 0;
		for (int i = 1; i < (int)ready.size(); ++i) {
			if (better(ready[i], ready[best_i]))
				best_i = i;
		}
		const int pick = ready[best_i];
		ready.erase(ready.begin() + best_i);
		out_order->push_back(pick);

		for (int child : (*out_dependents)[pick]) {
			indeg[child]--;
			if (indeg[child] == 0)
				ready.push_back(child);
		}
	}

	if (out_order->size() != nodes.size()) {
		fprintf(stderr, "Cycle detected in dispatch/time dependency graph\n");
		return false;
	}
	return true;
}

// -----------------------------------------------------------------------------
// CPU set parsing / validation
// -----------------------------------------------------------------------------

static bool SplitCpuIds(const char *text, std::vector<int> *out_ids) {
	out_ids->clear();
	if (!text || !text[0])
		return false;
	const char *cur = text;
	while (*cur) {
		char *endptr = nullptr;
		long v = strtol(cur, &endptr, 10);
		if (endptr == cur)
			return false;
		out_ids->push_back((int)v);
		if (*endptr == '\0')
			break;
		if (*endptr != ',')
			return false;
		cur = endptr + 1;
	}
	return !out_ids->empty();
}

static bool ValidateCorePartition(const dispatch_graph_config_t *cfg) {
	const int visible_cores = cfg->visible_cores > 0 ? cfg->visible_cores : 8;
	if (visible_cores != 8) {
		fprintf(stderr, "This strict runner expects visible_cores=8; got %d\n",
			visible_cores);
		return false;
	}

	std::vector<int> p_ids;
	std::vector<int> e_ids;
	if (!SplitCpuIds(
			cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "0,1,2,3", &p_ids)) {
		fprintf(stderr, "Invalid --cpu_p_cpu_ids\n");
		return false;
	}
	if (!SplitCpuIds(cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "4,5", &e_ids)) {
		fprintf(stderr, "Invalid --cpu_e_cpu_ids\n");
		return false;
	}
	if (p_ids.size() != 4) {
		fprintf(
			stderr, "CPU_P must have exactly 4 cores; got %zu\n", p_ids.size());
		return false;
	}
	if (e_ids.size() != 2) {
		fprintf(
			stderr, "CPU_E must have exactly 2 cores; got %zu\n", e_ids.size());
		return false;
	}

	std::unordered_set<int> seen;
	for (int v : p_ids) {
		if (v < 0 || v >= visible_cores) {
			fprintf(stderr, "CPU_P core %d out of range [0,%d)\n", v,
				visible_cores);
			return false;
		}
		if (!seen.insert(v).second) {
			fprintf(stderr, "Duplicate logical core %d in CPU_P set\n", v);
			return false;
		}
	}
	for (int v : e_ids) {
		if (v < 0 || v >= visible_cores) {
			fprintf(stderr, "CPU_E core %d out of range [0,%d)\n", v,
				visible_cores);
			return false;
		}
		if (!seen.insert(v).second) {
			fprintf(
				stderr, "CPU_E core %d overlaps CPU_P or is duplicated\n", v);
			return false;
		}
	}
	return true;
}

// -----------------------------------------------------------------------------
// VMFB path resolution
// -----------------------------------------------------------------------------

static std::string NormalizeModelFamilyFromKey(
	const std::string &dispatch_key) {
	const size_t pos = dispatch_key.find("_dispatch_");
	if (pos == std::string::npos)
		return dispatch_key;

	std::string prefix = dispatch_key.substr(0, pos);

	// mlp0, mlp1, ..., mlp16 -> mlp
	if (prefix.size() >= 3 && prefix.compare(0, 3, "mlp") == 0) {
		bool rest_is_digits = true;
		for (size_t i = 3; i < prefix.size(); ++i) {
			if (prefix[i] < '0' || prefix[i] > '9') {
				rest_is_digits = false;
				break;
			}
		}
		if (rest_is_digits)
			return "mlp";
	}

	return prefix;
}

static const char *VariantDirFromHardwareTarget(HardwareTarget target) {
	switch (target) {
		case HardwareTarget::kCpuP:
			return "RVV";
		case HardwareTarget::kCpuE:
			return "scalar";
		default:
			return "";
	}
}

static std::string BuildRepoVmfbPathFromDispatch(
	const dispatch_graph_config_t *cfg, const DispatchNode &node) {
	if (!cfg || !cfg->vmfb_root_dir || !cfg->vmfb_root_dir[0])
		return std::string();

	const std::string model_family = NormalizeModelFamilyFromKey(node.key);
	const std::string variant_dir =
		VariantDirFromHardwareTarget(node.hardware_target);

	char filename[512];
	snprintf(filename, sizeof(filename),
		"module_%s$async_dispatch_%d_embedded_elf_riscv_64_async.vmfb",
		model_family.c_str(), node.id);

	std::string path = cfg->vmfb_root_dir;
	path = PathJoin2(path, "gen");
	path = PathJoin2(path, "vmfb");
	path = PathJoin2(path, model_family);
	path = PathJoin2(path, "spacemit_x60");
	path = PathJoin2(path, variant_dir);
	path = PathJoin2(path, model_family + ".q.int8");
	path = PathJoin2(path, "async");
	path = PathJoin2(path, "vmfb");
	path = PathJoin2(path, filename);
	return path;
}

static std::string ResolveVmfbPath(const dispatch_graph_config_t *cfg,
	const std::string &json_dir, const std::string &json_dispatch_vmfb_dir,
	const DispatchNode &node) {
	if (!node.vmfb_path_json.empty()) {
		if (node.vmfb_path_json[0] == '/')
			return node.vmfb_path_json;
		if (cfg && cfg->vmfb_root_dir && cfg->vmfb_root_dir[0]) {
			std::string rel = node.vmfb_path_json;
			if (!json_dispatch_vmfb_dir.empty()) {
				rel = StripLeadingDir(rel, json_dispatch_vmfb_dir);
			} else {
				rel = StripLeadingDir(rel, "dispatches");
			}
			return PathJoin2(cfg->vmfb_root_dir, rel);
		}
		return PathJoin2(json_dir, node.vmfb_path_json);
	}

	std::string repo_path = BuildRepoVmfbPathFromDispatch(cfg, node);
	if (!repo_path.empty())
		return repo_path;

	if (!node.module_name.empty()) {
		std::string file = node.module_name;
		if (!EndsWith(file, ".vmfb"))
			file += ".vmfb";
		if (cfg->vmfb_root_dir && cfg->vmfb_root_dir[0]) {
			return PathJoin2(cfg->vmfb_root_dir, file);
		}
		return PathJoin2(json_dir, file);
	}

	return std::string();
}

// -----------------------------------------------------------------------------
// Strict async-external export handling
// Supported signatures:
//   rr   -> (wait_fence, signal_fence)
//   irr  -> (dispatch_iters: i32, wait_fence, signal_fence)
// -----------------------------------------------------------------------------

static std::string GetInputCallingConvention(const iree_vm_function_t &fn) {
	const iree_vm_function_signature_t fsig = iree_vm_function_signature(&fn);
	const iree_string_view_t cc = fsig.calling_convention;
	if (cc.size == 0 || cc.data[0] != '0')
		return std::string();
	const void *u = memchr(cc.data, '_', cc.size);
	const size_t upos =
		u ? (size_t)((const char *)u - cc.data) : (size_t)cc.size;
	if (upos < 1)
		return std::string();
	return std::string(cc.data + 1, cc.data + upos);
}

static bool ClassifyStrictAsyncSignature(
	const iree_vm_function_t &fn, ModuleCallSignature *out_sig) {
	out_sig->pass_dispatch_iters = false;
	out_sig->input_cc = GetInputCallingConvention(fn);
	if (out_sig->input_cc == "rr") {
		out_sig->pass_dispatch_iters = false;
		return true;
	}
	if (out_sig->input_cc == "irr") {
		out_sig->pass_dispatch_iters = true;
		return true;
	}
	return false;
}

static iree_status_t PickEntryFunctionStrict(iree_vm_module_t *module,
	iree_vm_function_t *out_fn, ModuleCallSignature *out_sig) {
	*out_fn = (iree_vm_function_t){0};
	*out_sig = ModuleCallSignature();

	const iree_vm_module_signature_t sig = iree_vm_module_signature(module);
	if (sig.export_function_count == 0) {
		return iree_make_status(IREE_STATUS_NOT_FOUND, "module has no exports");
	}

	for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
		iree_vm_function_t fn = {0};
		IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
			module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &fn));
		ModuleCallSignature msig;
		if (ClassifyStrictAsyncSignature(fn, &msig)) {
			*out_fn = fn;
			*out_sig = msig;
			return iree_ok_status();
		}
	}

	iree_vm_function_t first = {0};
	IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
		module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, &first));
	const std::string cc = GetInputCallingConvention(first);
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"strict scheduler requires async-external export signature rr or irr; "
		"first export input convention was '%s'",
		cc.c_str());
}

// -----------------------------------------------------------------------------
// Pinned local-task device creation
// -----------------------------------------------------------------------------

static iree_status_t CreateConfiguredLocalTaskDeviceFromCpuIds(
	iree_runtime_instance_t *instance, iree_allocator_t host_allocator,
	const char *cpu_ids_csv, iree_hal_device_t **out_device) {
	*out_device = nullptr;

	if (!instance) {
		return iree_make_status(
			IREE_STATUS_INVALID_ARGUMENT, "instance is null");
	}
	if (!cpu_ids_csv || !cpu_ids_csv[0]) {
		return iree_make_status(
			IREE_STATUS_INVALID_ARGUMENT, "cpu_ids_csv is empty");
	}

	iree_hal_driver_registry_t *registry =
		iree_runtime_instance_driver_registry(instance);
	if (!registry) {
		return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
			"runtime instance has no driver registry");
	}

	iree_hal_driver_t *driver = nullptr;
	IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(registry,
		iree_make_cstring_view("local-task"), host_allocator, &driver));

	iree_string_pair_t params[1];
	params[0].key = iree_make_cstring_view("task_topology_cpu_ids");
	params[0].value = iree_make_cstring_view(cpu_ids_csv);

	iree_status_t st = iree_hal_driver_create_device_by_path(driver,
		iree_make_cstring_view("local-task"), iree_string_view_empty(),
		IREE_ARRAYSIZE(params), params, host_allocator, out_device);

	iree_hal_driver_release(driver);
	return st;
}

// -----------------------------------------------------------------------------
// Per-node module loading
// One session per JSON node key.
// -----------------------------------------------------------------------------

static void ReleasePerNodeModule(PerNodeModule *m) {
	if (!m)
		return;
	if (m->session) {
		iree_runtime_session_release(m->session);
		m->session = nullptr;
	}
}

static iree_status_t LoadPerNodeModule(iree_runtime_instance_t *instance,
	iree_hal_device_t *device, const std::string &vmfb_path,
	PerNodeModule *out) {
	out->vmfb_path = vmfb_path;

	iree_runtime_session_options_t session_opts;
	iree_runtime_session_options_initialize(&session_opts);

	IREE_RETURN_IF_ERROR(
		iree_runtime_session_create_with_device(instance, &session_opts, device,
			iree_runtime_instance_host_allocator(instance), &out->session));

	IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
		out->session, vmfb_path.c_str()));

	iree_vm_context_t *ctx = iree_runtime_session_context(out->session);
	const iree_host_size_t module_count = iree_vm_context_module_count(ctx);
	if (module_count == 0) {
		return iree_make_status(
			IREE_STATUS_FAILED_PRECONDITION, "session context had 0 modules");
	}

	iree_vm_module_t *module = iree_vm_context_module_at(ctx, module_count - 1);
	IREE_RETURN_IF_ERROR(
		PickEntryFunctionStrict(module, &out->entry_fn, &out->sig));

	return iree_ok_status();
}

static iree_status_t CallPerNodeModuleAsync(PerNodeModule *m,
	int32_t dispatch_iters, iree_hal_fence_t *wait_fence,
	iree_hal_fence_t *signal_fence, iree_allocator_t host_alloc) {
	iree_vm_list_t *inputs = nullptr;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		/*initial_capacity=*/
		(iree_host_size_t)(m->sig.pass_dispatch_iters ? 3 : 2), host_alloc,
		&inputs));

	iree_status_t st = iree_ok_status();

	if (m->sig.pass_dispatch_iters) {
		iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
		st = iree_vm_list_push_value(inputs, &v);
		if (!iree_status_is_ok(st))
			goto cleanup;
	}

	// wait_fence may be null. In that case, pass a null ref for !hal.fence.
	{
		iree_vm_ref_t wait_ref = {};
		if (wait_fence) {
			wait_ref = iree_hal_fence_retain_ref(wait_fence);
		}
		st = iree_vm_list_push_ref_move(inputs, &wait_ref);
		if (!iree_status_is_ok(st))
			goto cleanup;
	}

	{
		if (!signal_fence) {
			st = iree_make_status(
				IREE_STATUS_INVALID_ARGUMENT, "signal_fence must not be null");
			goto cleanup;
		}
		iree_vm_ref_t signal_ref = iree_hal_fence_retain_ref(signal_fence);
		st = iree_vm_list_push_ref_move(inputs, &signal_ref);
		if (!iree_status_is_ok(st))
			goto cleanup;
	}

	st = iree_runtime_session_call(m->session, &m->entry_fn, inputs,
		/*output_list=*/nullptr);

cleanup:
	if (inputs)
		iree_vm_list_release(inputs);
	return st;
}

// -----------------------------------------------------------------------------
// Trace writer for Gantt reconstruction
// -----------------------------------------------------------------------------

struct TraceWriter {
	FILE *f = nullptr;
	bool wrote_header = false;

	bool Open(const char *path) {
		if (!path || !path[0])
			return false;
		f = fopen(path, "wb");
		return f != nullptr;
	}

	void Close() {
		if (f)
			fclose(f);
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
			"residency_us,dep_slip_us,planned_duration_us\n");
		wrote_header = true;
	}

	void WriteRow(int graph_iter, const DispatchNode &node,
		uint64_t planned_start_us, uint64_t eligible_us, uint64_t submit_us,
		uint64_t complete_us) {
		if (!f)
			return;
		Header();

		const uint64_t residency_us =
			complete_us >= eligible_us ? (complete_us - eligible_us) : 0;
		const uint64_t dep_slip_us = eligible_us >= planned_start_us
			? (eligible_us - planned_start_us)
			: 0;

		fprintf(f,
			"%d,%s,%d,%d,%d,"
			"%s,%s,%s,%s,"
			"%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
			"%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
			graph_iter, node.key.c_str(), node.id, node.ordinal, node.total,
			node.job_name.c_str(), node.module_name.c_str(),
			HardwareTargetName(node.hardware_target),
			node.vmfb_path_resolved.c_str(), planned_start_us, eligible_us,
			submit_us, complete_us, residency_us, dep_slip_us,
			MsToUs(node.planned_duration_ms));

		fflush(f);
	}
};

// -----------------------------------------------------------------------------
// JSON / DOT output helpers
// -----------------------------------------------------------------------------

static void JsonWriteEscaped(FILE *f, const std::string &s) {
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

static std::string DotEscapeLabel(const std::string &s) {
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

static bool WriteSummaryJson(const char *path,
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

		fprintf(f, "      \"residency_stats\": {\n");
		fprintf(
			f, "        \"count\": %" PRIu64 ",\n", n.residency_stats.count);
		fprintf(f, "        \"avg_ms\": %.6f,\n", n.residency_stats.AvgMs());
		fprintf(f, "        \"p50_ms\": %.6f,\n", n.residency_stats.P50Ms());
		fprintf(f, "        \"p90_ms\": %.6f,\n", n.residency_stats.P90Ms());
		fprintf(f, "        \"p99_ms\": %.6f,\n", n.residency_stats.P99Ms());
		fprintf(f, "        \"min_ms\": %.6f,\n", n.residency_stats.MinMs());
		fprintf(f, "        \"max_ms\": %.6f\n", n.residency_stats.MaxMs());
		fprintf(f, "      }\n");

		fprintf(f, "    }%s\n", (i + 1 < model.nodes.size()) ? "," : "");
	}
	fprintf(f, "  ],\n");

	fprintf(f, "  \"topo_order\": [");
	for (size_t i = 0; i < topo_order.size(); ++i) {
		if (i)
			fprintf(f, ", ");
		JsonWriteEscaped(f, model.nodes[(size_t)topo_order[i]].key);
	}
	fprintf(f, "]\n");
	fprintf(f, "}\n");

	fclose(f);
	return true;
}

static bool WriteDotGraph(const char *path, const GraphModel &model) {
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
			"\\nres_avg=" + std::to_string(n.residency_stats.AvgMs()) + "ms";

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

// -----------------------------------------------------------------------------
// Iteration runtime state
// -----------------------------------------------------------------------------

struct IterNodeState {
	iree_hal_semaphore_t *completion_semaphore = nullptr;
	iree_hal_fence_t *signal_fence = nullptr;

	uint64_t planned_start_us = 0;
	uint64_t planned_duration_us = 0;
	uint64_t submit_us = 0;
	uint64_t eligible_us = 0;
	uint64_t complete_us = 0;
	bool done = false;
	bool submitted = false;
};

static void ReleaseIterNodeState(IterNodeState *s) {
	if (!s)
		return;
	if (s->signal_fence)
		iree_hal_fence_release(s->signal_fence);
	if (s->completion_semaphore)
		iree_hal_semaphore_release(s->completion_semaphore);
	*s = IterNodeState();
}

// -----------------------------------------------------------------------------
// Polling helpers
// -----------------------------------------------------------------------------

static uint64_t UsSince(const Clock::time_point &base, Clock::time_point now) {
	return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
		now - base)
		.count();
}

static uint64_t ComputeEligibleUs(const DispatchNode &node,
	const std::unordered_map<std::string, int> &index_of,
	const std::vector<IterNodeState> &iter_states) {
	uint64_t eligible_us = MsToUs(node.start_time_ms);
	for (const auto &pred : node.all_predecessors) {
		auto it = index_of.find(pred);
		if (it == index_of.end())
			continue;
		const IterNodeState &ps = iter_states[(size_t)it->second];
		if (ps.complete_us > eligible_us)
			eligible_us = ps.complete_us;
	}
	return eligible_us;
}

static void PollCompletions(SharedState *shared,
	const std::vector<DispatchNode> &nodes,
	const std::unordered_map<std::string, int> &index_of,
	std::vector<IterNodeState> *iter_states, const Clock::time_point &iter_t0,
	int graph_iter, TraceWriter *trace) {
	const uint64_t now_us = UsSince(iter_t0, Clock::now());

	for (size_t i = 0; i < iter_states->size(); ++i) {
		IterNodeState &st = (*iter_states)[i];
		if (st.done || !st.submitted || !st.signal_fence)
			continue;

		iree_status_t q = iree_hal_fence_query(st.signal_fence);
		if (iree_status_is_ok(q)) {
			st.done = true;
			st.complete_us = now_us;
			st.eligible_us =
				ComputeEligibleUs(nodes[i], index_of, *iter_states);

			trace->WriteRow(graph_iter, nodes[i], st.planned_start_us,
				st.eligible_us, st.submit_us, st.complete_us);

			iree_status_ignore(q);
		} else if (iree_status_code(q) == IREE_STATUS_DEFERRED) {
			iree_status_ignore(q);
		} else {
			SetFatalOnce(shared, q, "[dispatch] fence query failed");
			return;
		}
	}
}

static void SleepWithPolling(SharedState *shared, uint64_t sleep_us,
	const std::vector<DispatchNode> &nodes,
	const std::unordered_map<std::string, int> &index_of,
	std::vector<IterNodeState> *iter_states, const Clock::time_point &iter_t0,
	int graph_iter, TraceWriter *trace) {
	if (sleep_us == 0) {
		PollCompletions(
			shared, nodes, index_of, iter_states, iter_t0, graph_iter, trace);
		return;
	}

	const useconds_t chunk = (useconds_t)(sleep_us > 250 ? 250 : sleep_us);
	usleep(chunk);
	PollCompletions(
		shared, nodes, index_of, iter_states, iter_t0, graph_iter, trace);
}

static bool AreAllPredecessorsDone(const DispatchNode &node,
	const std::unordered_map<std::string, int> &index_of,
	const std::vector<IterNodeState> &iter_states) {
	for (const auto &pred : node.all_predecessors) {
		auto it = index_of.find(pred);
		if (it == index_of.end())
			return false;
		if (!iter_states[(size_t)it->second].done)
			return false;
	}
	return true;
}

static bool WaitUntilNodeReady(SharedState *shared, const DispatchNode &node,
	const std::unordered_map<std::string, int> &index_of,
	std::vector<IterNodeState> *iter_states, const Clock::time_point &iter_t0,
	int graph_iter, TraceWriter *trace) {
	const uint64_t planned_start_us = MsToUs(node.start_time_ms);

	while (!HasFatal(shared)) {
		PollCompletions(shared, std::vector<DispatchNode>(), index_of,
			iter_states, iter_t0, graph_iter, trace);
		if (HasFatal(shared))
			return false;

		const uint64_t now_us = UsSince(iter_t0, Clock::now());
		if (now_us < planned_start_us) {
			const uint64_t remaining = planned_start_us - now_us;
			SleepWithPolling(shared, remaining, std::vector<DispatchNode>(),
				index_of, iter_states, iter_t0, graph_iter, trace);
			continue;
		}

		if (AreAllPredecessorsDone(node, index_of, *iter_states)) {
			return true;
		}

		SleepWithPolling(shared, 100, std::vector<DispatchNode>(), index_of,
			iter_states, iter_t0, graph_iter, trace);
	}

	return false;
}

// Overload helpers that keep the actual nodes reference available.
static void PollCompletionsWithNodes(SharedState *shared,
	const std::vector<DispatchNode> &nodes,
	const std::unordered_map<std::string, int> &index_of,
	std::vector<IterNodeState> *iter_states, const Clock::time_point &iter_t0,
	int graph_iter, TraceWriter *trace) {
	const uint64_t now_us = UsSince(iter_t0, Clock::now());

	for (size_t i = 0; i < iter_states->size(); ++i) {
		IterNodeState &st = (*iter_states)[i];
		if (st.done || !st.submitted || !st.signal_fence)
			continue;

		iree_status_t q = iree_hal_fence_query(st.signal_fence);
		if (iree_status_is_ok(q)) {
			st.done = true;
			st.complete_us = now_us;
			st.eligible_us =
				ComputeEligibleUs(nodes[i], index_of, *iter_states);

			trace->WriteRow(graph_iter, nodes[i], st.planned_start_us,
				st.eligible_us, st.submit_us, st.complete_us);

			iree_status_ignore(q);
		} else if (iree_status_code(q) == IREE_STATUS_DEFERRED) {
			iree_status_ignore(q);
		} else {
			SetFatalOnce(shared, q, "[dispatch] fence query failed");
			return;
		}
	}
}

static void SleepWithPollingAndNodes(SharedState *shared, uint64_t sleep_us,
	const std::vector<DispatchNode> &nodes,
	const std::unordered_map<std::string, int> &index_of,
	std::vector<IterNodeState> *iter_states, const Clock::time_point &iter_t0,
	int graph_iter, TraceWriter *trace) {
	if (sleep_us > 0) {
		const useconds_t chunk = (useconds_t)(sleep_us > 250 ? 250 : sleep_us);
		usleep(chunk);
	}
	PollCompletionsWithNodes(
		shared, nodes, index_of, iter_states, iter_t0, graph_iter, trace);
}

static bool WaitUntilNodeReadyWithNodes(SharedState *shared,
	const DispatchNode &node, const std::vector<DispatchNode> &nodes,
	const std::unordered_map<std::string, int> &index_of,
	std::vector<IterNodeState> *iter_states, const Clock::time_point &iter_t0,
	int graph_iter, TraceWriter *trace) {
	const uint64_t planned_start_us = MsToUs(node.start_time_ms);

	while (!HasFatal(shared)) {
		PollCompletionsWithNodes(
			shared, nodes, index_of, iter_states, iter_t0, graph_iter, trace);
		if (HasFatal(shared))
			return false;

		const uint64_t now_us = UsSince(iter_t0, Clock::now());
		if (now_us < planned_start_us) {
			const uint64_t remaining = planned_start_us - now_us;
			SleepWithPollingAndNodes(shared, remaining, nodes, index_of,
				iter_states, iter_t0, graph_iter, trace);
			continue;
		}

		if (AreAllPredecessorsDone(node, index_of, *iter_states)) {
			return true;
		}

		SleepWithPollingAndNodes(shared, 100, nodes, index_of, iter_states,
			iter_t0, graph_iter, trace);
	}

	return false;
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

} // namespace

extern "C" int dispatch_graph_run(const dispatch_graph_config_t *cfg) {
	if (!cfg || !cfg->graph_json_path || !cfg->graph_json_path[0]) {
		fprintf(stderr, "dispatch_graph_run: missing graph_json_path\n");
		return 1;
	}

	const char *driver = (cfg->driver_name && cfg->driver_name[0])
		? cfg->driver_name
		: "local-task";
	const int graph_iters = (cfg->graph_iters > 0) ? cfg->graph_iters : 1;
	const int dispatch_iters =
		(cfg->dispatch_iters > 0) ? cfg->dispatch_iters : 1;
	const int report_every = (cfg->report_every >= 0) ? cfg->report_every : 0;

	if (strcmp(driver, "local-task") != 0) {
		fprintf(stderr,
			"Strict split CPU_P/CPU_E scheduler requires driver=local-task; "
			"got '%s'\n",
			driver);
		return 1;
	}

	if (!ValidateCorePartition(cfg))
		return 1;

	fprintf(stdout,
		"Strict dispatch scheduler:\n"
		"  json          = %s\n"
		"  driver        = %s\n"
		"  graph_iters   = %d\n"
		"  dispatch_iters= %d\n"
		"  report_every  = %d\n"
		"  vmfb_root_dir = %s\n"
		"  CPU_P cores   = %s\n"
		"  CPU_E cores   = %s\n"
		"  visible_cores = %d\n"
		"  out_json      = %s\n"
		"  out_dot       = %s\n"
		"  trace_csv     = %s\n",
		cfg->graph_json_path, driver, graph_iters, dispatch_iters, report_every,
		cfg->vmfb_root_dir ? cfg->vmfb_root_dir : "",
		cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "",
		cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "", cfg->visible_cores,
		cfg->out_json_path ? cfg->out_json_path : "",
		cfg->out_dot_path ? cfg->out_dot_path : "",
		cfg->trace_csv_path ? cfg->trace_csv_path : "");
	fflush(stdout);

	GraphModel model;
	if (!ParseDispatchScheduleJson(cfg->graph_json_path, &model)) {
		fprintf(stderr, "Failed to parse schedule JSON: %s\n",
			cfg->graph_json_path);
		return 1;
	}
	ExpandAllPredecessors(&model.nodes);

	if (model.makespan_ms <= 0.0) {
		double max_end_ms = 0.0;
		for (const auto &n : model.nodes) {
			const double end_ms = n.start_time_ms + n.planned_duration_ms;
			if (end_ms > max_end_ms)
				max_end_ms = end_ms;
		}
		model.makespan_ms = max_end_ms;
	}

	const std::string json_dir = PathDirname(cfg->graph_json_path);
	for (auto &n : model.nodes) {
		n.vmfb_path_resolved = ResolveVmfbPath(
			cfg, json_dir, model.dispatch_vmfb_dir_from_json, n);
		if (n.vmfb_path_resolved.empty()) {
			fprintf(stderr, "Node %s had neither vmfb_path nor module_name\n",
				n.key.c_str());
			return 1;
		}
		if (!FileReadable(n.vmfb_path_resolved)) {
			fprintf(stderr,
				"VMFB not readable for node %s:\n"
				"  module_name        = %s\n"
				"  vmfb_path_json     = %s\n"
				"  vmfb_path_resolved = %s\n",
				n.key.c_str(), n.module_name.c_str(), n.vmfb_path_json.c_str(),
				n.vmfb_path_resolved.c_str());
			return 1;
		}
	}

	std::vector<int> topo_order;
	std::vector<std::vector<int>> dependents;
	if (!TopoSort(model.nodes, &topo_order, &dependents))
		return 1;

	fprintf(stdout, "Submit order (%zu nodes):\n", topo_order.size());
	for (size_t i = 0; i < topo_order.size(); ++i) {
		const auto &n = model.nodes[(size_t)topo_order[i]];
		fprintf(stdout, "  %zu) %s target=%s start=%.3fms dur=%.3fms\n", i + 1,
			n.key.c_str(), HardwareTargetName(n.hardware_target),
			n.start_time_ms, n.planned_duration_ms);
	}
	fflush(stdout);

	std::unordered_map<std::string, int> index_of;
	index_of.reserve(model.nodes.size() * 2);
	for (int i = 0; i < (int)model.nodes.size(); ++i) {
		index_of[model.nodes[i].key] = i;
	}

	SharedState shared;
	iree_allocator_t host_alloc = iree_allocator_system();
	iree_runtime_instance_t *instance = nullptr;
	iree_hal_device_t *device_p = nullptr;
	iree_hal_device_t *device_e = nullptr;

	TraceWriter trace;
	if (cfg->trace_csv_path && cfg->trace_csv_path[0]) {
		if (!trace.Open(cfg->trace_csv_path)) {
			fprintf(
				stderr, "Failed to open trace_csv: %s\n", cfg->trace_csv_path);
			return 1;
		}
	}

	{
		iree_runtime_instance_options_t opts;
		iree_runtime_instance_options_initialize(&opts);
		iree_runtime_instance_options_use_all_available_drivers(&opts);
		iree_status_t st =
			iree_runtime_instance_create(&opts, host_alloc, &instance);
		if (!iree_status_is_ok(st)) {
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			trace.Close();
			return 1;
		}
	}

	{
		iree_status_t st = CreateConfiguredLocalTaskDeviceFromCpuIds(
			instance, host_alloc, cfg->cpu_p_cpu_ids, &device_p);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating CPU_P device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	{
		iree_status_t st = CreateConfiguredLocalTaskDeviceFromCpuIds(
			instance, host_alloc, cfg->cpu_e_cpu_ids, &device_e);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating CPU_E device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			iree_hal_device_release(device_p);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	fprintf(stdout, "[dispatch] CPU_P pinned to {%s}, CPU_E pinned to {%s}\n",
		cfg->cpu_p_cpu_ids, cfg->cpu_e_cpu_ids);
	fflush(stdout);

	for (auto &n : model.nodes) {
		iree_hal_device_t *target_device =
			(n.hardware_target == HardwareTarget::kCpuP) ? device_p : device_e;

		iree_status_t st = LoadPerNodeModule(
			instance, target_device, n.vmfb_path_resolved, &n.module);
		if (!iree_status_is_ok(st)) {
			fprintf(
				stderr, "Failed loading module for node %s\n", n.key.c_str());
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);

			for (auto &node : model.nodes)
				ReleasePerNodeModule(&node.module);
			iree_hal_device_release(device_e);
			iree_hal_device_release(device_p);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	const auto run_t0 = Clock::now();

	for (int gi = 0; gi < graph_iters && !HasFatal(&shared); ++gi) {
		std::vector<IterNodeState> iter_states(model.nodes.size());
		const auto iter_t0 = Clock::now();

		// Pre-create one completion fence per node.
		for (size_t i = 0; i < model.nodes.size(); ++i) {
			const DispatchNode &node = model.nodes[i];
			IterNodeState &st = iter_states[i];

			st.planned_start_us = MsToUs(node.start_time_ms);
			st.planned_duration_us = MsToUs(node.planned_duration_ms);

			iree_hal_device_t *target_device =
				(node.hardware_target == HardwareTarget::kCpuP) ? device_p
																: device_e;

			iree_status_t x = iree_hal_semaphore_create(target_device,
				IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
				IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &st.completion_semaphore);
			if (!iree_status_is_ok(x)) {
				SetFatalOnce(&shared, x,
					"[dispatch] create completion semaphore failed");
				break;
			}

			x = iree_hal_fence_create_at(
				st.completion_semaphore, 1ull, host_alloc, &st.signal_fence);
			if (!iree_status_is_ok(x)) {
				SetFatalOnce(
					&shared, x, "[dispatch] create signal fence failed");
				break;
			}
		}

		// Host-enforced dependency scheduling.
		// We do NOT pass predecessor wait fences to the module.
		// Instead, we only submit a node after:
		//   1) its planned start time has arrived
		//   2) all predecessor nodes are complete
		for (int idx : topo_order) {
			if (HasFatal(&shared))
				break;

			DispatchNode &node = model.nodes[(size_t)idx];
			IterNodeState &st = iter_states[(size_t)idx];

			if (!WaitUntilNodeReadyWithNodes(&shared, node, model.nodes,
					index_of, &iter_states, iter_t0, gi, &trace)) {
				break;
			}
			if (HasFatal(&shared))
				break;

			st.eligible_us = ComputeEligibleUs(node, index_of, iter_states);
			st.submit_us = UsSince(iter_t0, Clock::now());

			iree_status_t x =
				CallPerNodeModuleAsync(&node.module, (int32_t)dispatch_iters,
					/*wait_fence=*/nullptr, st.signal_fence, host_alloc);
			if (!iree_status_is_ok(x)) {
				SetFatalOnce(&shared, x, "[dispatch] module submit failed");
				break;
			}

			st.submitted = true;

			PollCompletionsWithNodes(&shared, model.nodes, index_of,
				&iter_states, iter_t0, gi, &trace);
		}

		// Wait for remaining submitted nodes to finish.
		if (!HasFatal(&shared)) {
			std::vector<iree_hal_fence_t *> pending;
			pending.reserve(iter_states.size());
			for (const auto &s : iter_states) {
				if (s.submitted && !s.done && s.signal_fence)
					pending.push_back(s.signal_fence);
			}

			if (!pending.empty()) {
				iree_hal_fence_t *wait_all = nullptr;
				iree_status_t st = iree_hal_fence_join(
					pending.size(), pending.data(), host_alloc, &wait_all);
				if (iree_status_is_ok(st)) {
					st = iree_hal_fence_wait(wait_all, iree_infinite_timeout(),
						IREE_HAL_WAIT_FLAG_DEFAULT);
				}
				if (wait_all)
					iree_hal_fence_release(wait_all);
				if (!iree_status_is_ok(st)) {
					SetFatalOnce(
						&shared, st, "[dispatch] final iteration wait failed");
				}
			}
		}

		if (!HasFatal(&shared)) {
			PollCompletionsWithNodes(&shared, model.nodes, index_of,
				&iter_states, iter_t0, gi, &trace);

			const uint64_t now_us = UsSince(iter_t0, Clock::now());
			for (size_t i = 0; i < iter_states.size(); ++i) {
				IterNodeState &st = iter_states[i];
				if (!st.submitted)
					continue;

				if (!st.done) {
					st.done = true;
					st.complete_us = now_us;
					st.eligible_us = ComputeEligibleUs(
						model.nodes[i], index_of, iter_states);

					trace.WriteRow(gi, model.nodes[i], st.planned_start_us,
						st.eligible_us, st.submit_us, st.complete_us);
				}

				const uint64_t residency_us = st.complete_us >= st.eligible_us
					? (st.complete_us - st.eligible_us)
					: 0;
				model.nodes[i].residency_stats.Add(residency_us);
			}
		}

		for (auto &s : iter_states)
			ReleaseIterNodeState(&s);

		if (report_every > 0 && ((gi + 1) % report_every) == 0 &&
			!HasFatal(&shared)) {
			fprintf(stdout, "[graph_iter %d/%d]\n", gi + 1, graph_iters);
			for (int idx : topo_order) {
				const auto &n = model.nodes[(size_t)idx];
				fprintf(stdout,
					"  %s target=%s plan=%.3fms res_avg=%.3fms p90=%.3fms "
					"max=%.3fms\n",
					n.key.c_str(), HardwareTargetName(n.hardware_target),
					n.planned_duration_ms, n.residency_stats.AvgMs(),
					n.residency_stats.P90Ms(), n.residency_stats.MaxMs());
			}
			fflush(stdout);
		}
	}

	const auto run_t1 = Clock::now();
	const double total_s =
		std::chrono::duration_cast<std::chrono::duration<double>>(
			run_t1 - run_t0)
			.count();

	if (!HasFatal(&shared)) {
		fprintf(stdout,
			"Run complete:\n"
			"  total_wall_ms=%.3f\n"
			"  schedule_makespan_ms=%.3f\n"
			"  completed_graph_iters=%d\n",
			total_s * 1000.0, model.makespan_ms, graph_iters);

		for (int idx : topo_order) {
			const auto &n = model.nodes[(size_t)idx];
			fprintf(stdout,
				"  %s target=%s plan=%.3fms res_avg=%.3fms p50=%.3fms "
				"p90=%.3fms "
				"p99=%.3fms min=%.3fms max=%.3fms\n",
				n.key.c_str(), HardwareTargetName(n.hardware_target),
				n.planned_duration_ms, n.residency_stats.AvgMs(),
				n.residency_stats.P50Ms(), n.residency_stats.P90Ms(),
				n.residency_stats.P99Ms(), n.residency_stats.MinMs(),
				n.residency_stats.MaxMs());
		}
		fprintf(stdout, "Done.\n");
		fflush(stdout);
	}

	bool ok_write = true;
	ok_write = ok_write &&
		WriteSummaryJson(cfg->out_json_path, cfg, model, topo_order);
	ok_write = ok_write && WriteDotGraph(cfg->out_dot_path, model);
	if (!ok_write) {
		fprintf(stderr, "Warning: failed writing one or more outputs\n");
	}

	trace.Close();
	for (auto &n : model.nodes)
		ReleasePerNodeModule(&n.module);
	if (device_e)
		iree_hal_device_release(device_e);
	if (device_p)
		iree_hal_device_release(device_p);
	if (instance)
		iree_runtime_instance_release(instance);

	return HasFatal(&shared) ? 1 : 0;
}
