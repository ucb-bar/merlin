#ifndef MERLIN_TIMING_MEMORY_SOURCE_OWNERSHIP_HPP
#define MERLIN_TIMING_MEMORY_SOURCE_OWNERSHIP_HPP

#include "memory_milestones.hpp"
#include <deque>

namespace merlin_timing {

// Finite ordinary-request source ownership, separate from cache tags/fills and
// downstream return IDs. Callers supply actual accepted request edges and an
// explicitly closed reserved-source/set predicate. The tracker supplies no
// response latency, scheduler decision, retirement law or reserved-class model.
class MemorySourceOwnership {
public:
  struct Spec { std::size_t ordinary, total, secondary; };
  struct Request { std::uint64_t identity, set; Cycle accepted; };
  enum class Status { allocated, queued, blocked, unavailable };
  struct Admission { Status status; std::optional<MemorySourceLeaseV1> source; };
  explicit MemorySourceOwnership(Spec spec) : spec_(spec), slots_(spec.ordinary) {
    if (!spec.ordinary || spec.ordinary > spec.total || spec.total > 65536 ||
        !spec.secondary || spec.secondary > 65536)
      throw std::invalid_argument("invalid finite source ownership geometry");
  }
  Admission accept(const Request &r, bool reserved_context_closed, bool reserved_set_conflict) {
    if (!reserved_context_closed || !r.identity || r.identity <= last_request_ ||
        r.accepted < edge_) return {Status::unavailable, {}};
    if (reserved_set_conflict) return {Status::blocked, {}};
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      auto &s = slots_[i];
      if (s.active && s.active->set == r.set) {
        if (secondary_ == spec_.secondary) return {Status::blocked, {}};
        s.pending.push_back(r); ++secondary_; last_request_ = r.identity; edge_ = r.accepted;
        // The queued request has not yet become this slot's active owner.
        // Returning the prior request's lease would alias two generations.
        return {Status::queued, {}};
      }
    }
    // Source-defined lowest free ordinary slot. Reserved slots are NOT extra
    // ordinary capacity, even when idle. Never substitute a reader source ID.
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      auto &s = slots_[i];
      if (s.active) continue;
      if (s.generation == UINT64_MAX) return {Status::unavailable, {}};
      s.active = r; s.allocated = r.accepted;
      ++s.generation; last_request_ = r.identity; edge_ = r.accepted;
      return {Status::allocated, MemorySourceLeaseV1{i, s.generation, r.accepted, {}}};
    }
    return {Status::blocked, {}};
  }
  // The OWNER of the protocol supplies the actual no-wait + scheduler-fire
  // retirement/reload edge. A first beat, payload publication or reader credit
  // does not implicitly retire this source entry.
  std::optional<Request> retire(std::size_t source, std::uint64_t generation,
                                std::uint64_t request, Cycle at,
                                bool retirement_context_closed) {
    if (!retirement_context_closed || source >= slots_.size() || at < edge_)
      throw std::invalid_argument("unclosed or nonmonotonic source retirement");
    auto &s = slots_[source];
    if (!s.active || s.generation != generation || s.active->identity != request ||
        at < s.active->accepted) throw std::invalid_argument("source retirement does not own entry");
    edge_ = at;
    if (s.pending.empty()) { s.active.reset(); return {}; }
    if (s.generation == UINT64_MAX) throw std::overflow_error("source generation overflow");
    s.active = s.pending.front(); s.allocated = at;
    s.pending.pop_front(); --secondary_; ++s.generation;
    return s.active;
  }
  std::optional<MemorySourceLeaseV1> lease(std::size_t source) const {
    if (source >= slots_.size() || !slots_[source].active) return {};
    const auto &s = slots_[source];
    return MemorySourceLeaseV1{source, s.generation, s.allocated, {}};
  }
  std::size_t queued() const { return secondary_; }
private:
  struct Slot { std::optional<Request> active; std::uint64_t generation = 0; Cycle allocated = 0; std::deque<Request> pending; };
  Spec spec_;
  std::vector<Slot> slots_;
  std::size_t secondary_ = 0;
  std::uint64_t last_request_ = 0;
  Cycle edge_ = 0;
};

} // namespace merlin_timing
#endif
