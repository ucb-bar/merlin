// dispatch_graph_parse.h — JSON schedule parsing + topological sort.

#ifndef DISPATCH_GRAPH_PARSE_H_
#define DISPATCH_GRAPH_PARSE_H_

#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree_bench/json_parser.h"

#include "dispatch_types.h"

inline bool ParseDispatchesObject(
	merlin_bench::JsonParser *jp, std::vector<DispatchNode> *out_nodes) {
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
					if (!merlin_bench::ParseDependenciesArray(jp, &node.deps))
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

inline bool ParseMetadataObject(
	merlin_bench::JsonParser *jp, double *out_makespan_ms) {
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

inline bool ParseDispatchScheduleJson(
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

	merlin_bench::JsonParser jp;
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

inline void ExpandAllPredecessors(std::vector<DispatchNode> *nodes) {
	for (auto &n : *nodes) {
		std::unordered_set<std::string> uniq;
		std::vector<std::string> all;
		all.reserve(n.deps.size() + (n.time_dependency.empty() ? 0 : 1));

		for (const auto &d : n.deps) {
			if (uniq.insert(d).second)
				all.push_back(d);
		}

		// For MLP, time_dependency is a soft scheduling hint, not a hard gate.
		// The hard ordering is only the explicit per-layer dependency chain.
		if (!IsMlpJob(n) && !n.time_dependency.empty() &&
			uniq.insert(n.time_dependency).second) {
			all.push_back(n.time_dependency);
		}

		n.all_predecessors = std::move(all);
	}
}

//------------------------------------------------------------------------------
// Topological order
//------------------------------------------------------------------------------

inline bool TopoSort(const std::vector<DispatchNode> &nodes,
	std::vector<int> *out_order,
	std::vector<std::vector<int>> *out_dependents) {
	out_order->clear();
	out_dependents->assign(nodes.size(), {});

	std::unordered_map<std::string, int> index_of;
	index_of.reserve(nodes.size() * 2);
	for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
		index_of[nodes[i].key] = i;
	}

	std::vector<int> indeg(nodes.size(), 0);
	for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
		indeg[i] = static_cast<int>(nodes[i].all_predecessors.size());
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
	for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
		if (indeg[i] == 0)
			ready.push_back(i);
	}

	while (!ready.empty()) {
		int best_i = 0;
		for (int i = 1; i < static_cast<int>(ready.size()); ++i) {
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

#endif // DISPATCH_GRAPH_PARSE_H_
