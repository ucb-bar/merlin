// buffer_registry: keyed cache of per-dispatch outputs for the data-flow
// scheduler. Each canonical dispatch name maps to a vector of buffer views
// (one per output binding). Predecessors' outputs become this dispatch's
// inputs. Cross-device propagation goes through host-mediated queue
// read/write — see DataFlowRunner::TransferAcrossDevices().
//
// Lifetime: the registry retains references on the buffer_view objects;
// they are released when the registry is destroyed or when the entry is
// overwritten. Single-threaded use (the data-flow runner is sequential).

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "iree/hal/api.h"

struct DispatchOutputs {
	std::vector<iree_hal_buffer_view_t *> views; // owned (retained)
	std::string device_uri; // where these views live
};

class BufferRegistry {
  public:
	BufferRegistry() = default;
	~BufferRegistry() {
		Clear();
	}

	BufferRegistry(const BufferRegistry &) = delete;
	BufferRegistry &operator=(const BufferRegistry &) = delete;

	// Set this dispatch's outputs. Releases any previously-stored views
	// for the same key, then retains and stores the new views.
	void Set(const std::string &dispatch_name,
		std::vector<iree_hal_buffer_view_t *> views, std::string device_uri) {
		auto it = entries_.find(dispatch_name);
		if (it != entries_.end()) {
			ReleaseAll(it->second.views);
		}
		for (auto *v : views) {
			if (v)
				iree_hal_buffer_view_retain(v);
		}
		entries_[dispatch_name] = {std::move(views), std::move(device_uri)};
	}

	const DispatchOutputs *Get(const std::string &dispatch_name) const {
		auto it = entries_.find(dispatch_name);
		return it == entries_.end() ? nullptr : &it->second;
	}

	bool Has(const std::string &dispatch_name) const {
		return entries_.find(dispatch_name) != entries_.end();
	}

	void Clear() {
		for (auto &kv : entries_)
			ReleaseAll(kv.second.views);
		entries_.clear();
	}

	size_t size() const {
		return entries_.size();
	}

  private:
	static void ReleaseAll(std::vector<iree_hal_buffer_view_t *> &views) {
		for (auto *v : views) {
			if (v)
				iree_hal_buffer_view_release(v);
		}
		views.clear();
	}
	std::unordered_map<std::string, DispatchOutputs> entries_;
};
