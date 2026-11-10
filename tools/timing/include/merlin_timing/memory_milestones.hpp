#ifndef MERLIN_TIMING_MEMORY_MILESTONES_HPP
#define MERLIN_TIMING_MEMORY_MILESTONES_HPP

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <merlin_timing/engine.hpp>

namespace merlin_timing {

// Optional side channel, independent of a provider's existing service ABI.
// A request ID, a return-channel source ID, a cache slot and a fill/MSHR ID
// belong to DIFFERENT namespaces. None may be inferred from another.
struct MemoryRequestIdentityV1 {
  std::uint64_t request = 0, address = 0, bytes = 0;
  std::uint64_t return_source = 0, return_generation = 0;
  Cycle accepted = 0;
};
struct MemoryFillIdentityV1 {
  std::uint64_t cache = 0, line = 0, slot = 0, generation = 0;
  bool operator==(const MemoryFillIdentityV1 &b) const {
    return cache == b.cache && line == b.line && slot == b.slot && generation == b.generation;
  }
};
struct MemorySourceLeaseV1 {
  std::uint64_t source = 0, generation = 0;
  Cycle allocated = 0;
  std::optional<Cycle> released;
};
enum class MemoryMilestoneCoverage : std::uint32_t {
  unsupported = 0,
  // A complete-payload bound is not the first beat or a cache-line fill.
  payload_completion_bound = 1,
  source_closed = 2
};
struct MemoryLookupSnapshotV1 {
  std::uint64_t set = 0, line_bytes = 0;
  bool read_hit_without_probe_or_eviction = false;
  std::optional<Cycle> prior_fill_ready;
};
struct MemoryMilestonesV1 {
  std::uint32_t size = sizeof(MemoryMilestonesV1), version = 1;
  MemoryRequestIdentityV1 request;
  std::optional<MemoryFillIdentityV1> fill;
  std::optional<MemorySourceLeaseV1> source;
  MemoryMilestoneCoverage coverage = MemoryMilestoneCoverage::unsupported;
  std::optional<Cycle> lookup_complete, first_refill_beat, last_refill_beat;
  std::optional<Cycle> line_visible, payload_complete_bound;
  // Each entry describes availability of that actual output beat. A scalar
  // aggregate latency must never be divided to manufacture these milestones.
  std::uint32_t beat_bytes = 0;
  std::vector<Cycle> beat_available;
  bool line_pending_at_lookup = false;
  std::string unavailable;
};

inline bool valid_memory_milestones(const MemoryMilestonesV1 &m,
                                   std::size_t source_capacity,
                                   std::size_t max_beats = 256) {
  if (m.version != 1 || m.size != sizeof(MemoryMilestonesV1) ||
      !m.request.request || !m.request.bytes || !m.request.return_generation ||
      m.request.address > UINT64_MAX - (m.request.bytes - 1)) return false;
  const auto after_accept = [&](const std::optional<Cycle> &t) {
    return !t || *t >= m.request.accepted;
  };
  if (!after_accept(m.lookup_complete) || !after_accept(m.first_refill_beat) ||
      !after_accept(m.last_refill_beat) || !after_accept(m.line_visible) ||
      !after_accept(m.payload_complete_bound)) return false;
  if (m.first_refill_beat && (!m.last_refill_beat || *m.last_refill_beat < *m.first_refill_beat))
    return false;
  if (m.last_refill_beat && (!m.line_visible || *m.line_visible < *m.last_refill_beat)) return false;
  if (m.source && (m.source->source >= source_capacity || !m.source->generation ||
      (m.lookup_complete && m.source->allocated > *m.lookup_complete) ||
      (m.source->released && *m.source->released < m.source->allocated))) return false;
  if (m.fill && !m.fill->generation) return false;
  if (m.coverage == MemoryMilestoneCoverage::payload_completion_bound)
    return m.payload_complete_bound && m.beat_available.empty();
  if (m.coverage != MemoryMilestoneCoverage::source_closed)
    return m.coverage == MemoryMilestoneCoverage::unsupported && m.beat_available.empty();
  if (!m.source || !m.fill || !m.lookup_complete || !m.line_visible || !m.beat_bytes ||
      m.beat_available.empty() || m.beat_available.size() > max_beats ||
      m.request.bytes / m.beat_bytes + (m.request.bytes % m.beat_bytes != 0) != m.beat_available.size())
    return false;
  Cycle prior = std::max(m.request.accepted, *m.lookup_complete);
  for (const auto available : m.beat_available) {
    if (available < prior) return false;
    prior = available;
  }
  return true;
}

// A read-only last-observation query is useful for adapters whose ordinary
// memory callback already performed classification. It must not warm caches,
// run another transfer, allocate a fill, or supply timing not owned by that
// original observation. The request binds exact bytes and accepted frontier.
struct MemoryMilestoneQueryV1 {
  std::uint32_t size = sizeof(MemoryMilestoneQueryV1), version = 1;
  void *opaque = nullptr;
  bool (*last_observation)(void *, const MemoryRequestIdentityV1 *, MemoryMilestonesV1 *) = nullptr;
};
using InstallMemoryMilestoneQueryV1 = void (*)(void *, const MemoryMilestoneQueryV1 *);

// Added metadata uses a new optional symbol/result type. Never enlarge a
// previously issued V1 result behind an old caller's output pointer.
struct MemoryMilestonesV2 : MemoryMilestonesV1 {
  std::uint32_t detail_size = sizeof(MemoryMilestonesV2), detail_version = 2;
  std::optional<MemoryLookupSnapshotV1> lookup;
  std::uint64_t memory_observation = 0;
};
struct MemoryMilestoneQueryV2 {
  std::uint32_t size = sizeof(MemoryMilestoneQueryV2), version = 2;
  void *opaque = nullptr;
  bool (*last_observation)(void *, const MemoryRequestIdentityV1 *, MemoryMilestonesV2 *) = nullptr;
};
using InstallMemoryMilestoneQueryV2 = void (*)(void *, const MemoryMilestoneQueryV2 *);

// Separate optional epoch ABI. Admission records the original observation
// interval, not a new transfer. The owner must subsequently validate actual
// posted arrivals until the selected final edge; interference cannot silently
// retain a closed-context claim. It does not authorize any timing law itself.
struct MemoryResponseEpochV1 {
  std::uint32_t size = sizeof(MemoryResponseEpochV1), version = 1;
  std::uint64_t first_observation = 0, last_observation = 0, requests = 0;
  Cycle begin = 0, final_response = 0;
};
struct MemoryResponseEpochServiceV1 {
  std::uint32_t size = sizeof(MemoryResponseEpochServiceV1), version = 1;
  void *opaque = nullptr;
  bool (*admit)(void *, const MemoryResponseEpochV1 *) = nullptr;
};
using InstallMemoryResponseEpochServiceV1 = void (*)(void *, const MemoryResponseEpochServiceV1 *);

} // namespace merlin_timing
#endif
