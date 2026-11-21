#ifndef MERLIN_TIMING_READY_RESPONSE_ARBITER_HPP
#define MERLIN_TIMING_READY_RESPONSE_ARBITER_HPP

#include "memory_milestones.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

namespace merlin_timing {

// Finite, non-preemptive response-message arbitration. Source identities and
// every beat's readiness must come from the owning memory protocol. This class
// does not allocate MSHRs, turn cache slots into MSHRs, predict refill stages,
// or grant permission to replace another scheduler's arbitration.
//
// Scheduling advances only through an explicitly closed input frontier. A
// later request cannot retroactively win arbitration. Callers must supply ALL
// actions that change the shared round-robin pointer, including non-response
// actions; a D-only view is insufficient when the source scheduler shares it.
class ReadyResponseArbiter {
public:
  struct Spec { std::size_t sources, return_credits, pending; Cycle beat_interval; };
  struct Beat {
    std::uint64_t request, return_source, return_generation, source, source_generation;
    std::size_t index;
    Cycle offered, accepted;
    bool last;
  };
  using Accept = std::function<std::optional<Cycle>(const Beat &)>;
  explicit ReadyResponseArbiter(Spec spec) : spec_(spec), returns_(spec.return_credits), sources_(spec.sources) {
    if (!spec.sources || spec.sources > 65536 || !spec.return_credits ||
        spec.return_credits > 65536 || !spec.pending || spec.pending > 65536 ||
        !spec.beat_interval) throw std::invalid_argument("invalid bounded response geometry");
  }
  bool enqueue(const MemoryMilestonesV1 &m) {
    if (failed_ || !valid_memory_milestones(m, spec_.sources) ||
        m.coverage != MemoryMilestoneCoverage::source_closed ||
        m.request.accepted < closed_ || m.request.accepted < last_arrival_ ||
        m.request.return_source >= spec_.return_credits || pending_.size() >= spec_.pending ||
        m.request.request <= last_request_) return false;
    // A return source stays owned until its incoming LAST beat, irrespective
    // of destination publication. The cache/MSHR lease is a separate owner.
    for (const auto &p : pending_)
      if (p.request.return_source == m.request.return_source ||
          (p.source->source == m.source->source && p.source->generation != m.source->generation))
        return false;
    const auto &prior_return = returns_[m.request.return_source];
    const auto &prior_source = sources_[m.source->source];
    if (prior_return.generation && (m.request.return_generation <= prior_return.generation ||
        !prior_return.release || m.request.accepted < *prior_return.release)) return false;
    if (prior_source.generation && m.source->generation != prior_source.generation &&
        (m.source->generation < prior_source.generation || !prior_source.release ||
         m.source->allocated < *prior_source.release)) return false;
    last_arrival_ = m.request.accepted;
    last_request_ = m.request.request;
    returns_[m.request.return_source] = {m.request.return_generation, {}};
    sources_[m.source->source] = {m.source->generation, m.source->released};
    pending_.push_back(m);
    return true;
  }
  bool scheduler_action(Cycle at, std::size_t source) {
    if (failed_ || source >= spec_.sources || at == UINT64_MAX || at < closed_ || at < last_action_)
      return false;
    actions_.push_back({at, source}); last_action_ = at;
    return true;
  }
  // The caller certifies that all admissions and shared-scheduler actions
  // before `until` have been supplied. Until is exclusive.
  bool advance(Cycle until, bool closed_ready_set, const Accept &accept) {
    if (failed_ || !closed_ready_set || until < closed_) return false;
    while (true) {
      const auto next_action = actions_.empty() ? UINT64_MAX : actions_.front().at;
      Cycle next = UINT64_MAX;
      for (const auto &p : pending_)
        next = std::min(next, std::max({clock_, p.request.accepted, p.beat_available.front()}));
      // Shared non-response selections take effect at their actual edge.
      if (next_action <= next && next_action < until) {
        cursor_ = (actions_.front().source + 1) % spec_.sources;
        actions_.erase(actions_.begin()); clock_ = std::max(clock_, next_action + 1);
        continue;
      }
      if (next >= until || next == UINT64_MAX) break;
      std::optional<std::size_t> chosen;
      for (std::size_t distance = 0; distance < spec_.sources && !chosen; ++distance) {
        const auto source = (cursor_ + distance) % spec_.sources;
        for (std::size_t i = 0; i < pending_.size(); ++i)
          if (pending_[i].source->source == source &&
              pending_[i].request.accepted <= next && pending_[i].beat_available.front() <= next) {
            chosen = i; break;
          }
      }
      if (!chosen) { failed_ = true; return false; }
      const auto p = pending_[*chosen];
      Cycle beat_clock = next;
      // A selected message is indivisible on this port. Backpressure holds
      // its identity; another message cannot consume the same beat slot.
      for (std::size_t i = 0; i < p.beat_available.size(); ++i) {
        beat_clock = std::max(beat_clock, p.beat_available[i]);
        Beat b{p.request.request, p.request.return_source, p.request.return_generation,
               p.source->source, p.source->generation, i, beat_clock, 0,
               i + 1 == p.beat_available.size()};
        const auto accepted = accept(b);
        if (!accepted || *accepted < beat_clock || *accepted > UINT64_MAX - spec_.beat_interval) {
          failed_ = true; return false;
        }
        if (beats_.size() >= spec_.pending * 256) { failed_ = true; return false; }
        b.accepted = *accepted; beats_.push_back(b);
        beat_clock = *accepted + spec_.beat_interval;
      }
      // MSHR/source release is NOT inferred from this response; only reader
      // return-source ownership ends here. Line visibility remains unchanged.
      cursor_ = (p.source->source + 1) % spec_.sources;
      returns_[p.request.return_source].release = beat_clock;
      clock_ = beat_clock; pending_.erase(pending_.begin() + *chosen);
    }
    closed_ = until;
    return true;
  }
  const std::vector<Beat> &beats() const { return beats_; }
  std::vector<Beat> take_beats() { auto result = std::move(beats_); beats_.clear(); return result; }
  std::size_t pending() const { return pending_.size(); }
  bool failed() const { return failed_; }
private:
  struct Action { Cycle at; std::size_t source; };
  struct Lease { std::uint64_t generation = 0; std::optional<Cycle> release; };
  Spec spec_;
  Cycle clock_ = 0, closed_ = 0, last_arrival_ = 0, last_action_ = 0;
  std::size_t cursor_ = 0;
  std::uint64_t last_request_ = 0;
  bool failed_ = false;
  std::vector<MemoryMilestonesV1> pending_;
  std::vector<Action> actions_;
  std::vector<Lease> returns_, sources_;
  std::vector<Beat> beats_;
};
} // namespace merlin_timing
#endif
