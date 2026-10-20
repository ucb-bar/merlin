#ifndef MERLIN_TIMING_ENGINE_HPP
#define MERLIN_TIMING_ENGINE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace merlin_timing {

using Cycle = std::uint64_t;
using ResourceId = std::uint32_t;
using ValueId = std::uint64_t;

template <class T> struct View {
  const T *data = nullptr;
  std::size_t size = 0;
  const T &operator[](std::size_t index) const { return data[index]; }
};

struct ResourceSpec {
  ResourceId id;
  std::size_t capacity;
  std::size_t queue_capacity;
};
struct Demand {
  ResourceId resource;
  std::size_t units;
  std::optional<Cycle> occupancy;
};
struct Event {
  Cycle earliest = 0;
  View<ValueId> reads;
  View<ValueId> writes;
  View<Demand> demands;
  std::optional<Cycle> latency;
  bool hold_reads_until_ready = false;
  View<ValueId> retained_reads{};
};

enum class UnknownReason {
  none,
  missing_latency,
  missing_occupancy,
  dependency,
  resource,
  barrier,
  cycle_overflow
};
inline const char *name(UnknownReason value) {
  switch (value) {
  case UnknownReason::none: return "none";
  case UnknownReason::missing_latency: return "missing_latency";
  case UnknownReason::missing_occupancy: return "missing_occupancy";
  case UnknownReason::dependency: return "dependency";
  case UnknownReason::resource: return "resource";
  case UnknownReason::barrier: return "barrier";
  case UnknownReason::cycle_overflow: return "cycle_overflow";
  }
  return "invalid";
}
struct Result {
  std::optional<Cycle> issue;
  std::optional<Cycle> start;
  std::optional<Cycle> ready;
  UnknownReason unknown = UnknownReason::none;
};
struct ResourceReport {
  ResourceId id;
  std::uint64_t reservations;
  std::optional<Cycle> busy_unit_cycles;
  std::uint64_t unknown_events;
  std::size_t peak_queue_entries;
  std::size_t pending_queue_entries;
  bool timing_known;
};
struct Report {
  Cycle modeled_makespan;
  std::uint64_t events;
  std::uint64_t unknown_events;
  bool complete;
  bool accounting_overflow;
  std::size_t scoreboard_entries;
  std::size_t pending_queue_entries;
  std::vector<ResourceReport> resources;
};

// An online reservation model, not a cycle-accurate hardware simulator. Values
// name mutable storage: reads wait writers, writes wait writers and readers.
// Read operands are consumed at issue unless explicitly held until completion.
// All initial tokens are available at logical epoch zero. No price is inferred.
class Engine {
public:
  Engine(std::size_t value_count, const std::vector<ResourceSpec> &specs)
      : values_(value_count) {
    resources_.reserve(specs.size());
    for (const auto &spec : specs) {
      if (spec.capacity == 0)
        throw std::invalid_argument("resource capacity must be positive");
      for (const auto &resource : resources_)
        if (resource.spec.id == spec.id)
          throw std::invalid_argument("duplicate resource id");
      resources_.emplace_back(spec);
    }
    selected_.reserve(resources_.size());
  }

  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine(Engine &&) noexcept = default;
  Engine &operator=(Engine &&) noexcept = default;

  std::optional<Cycle> ready(ValueId value) const {
    if (value >= values_.size())
      throw std::out_of_range("value id exceeds scoreboard bound");
    return values_[static_cast<std::size_t>(value)].writer;
  }
  std::optional<Cycle> last_use(ValueId value) const {
    if (value >= values_.size())
      throw std::out_of_range("value id exceeds scoreboard bound");
    const auto &entry = values_[static_cast<std::size_t>(value)];
    if (!entry.writer || !entry.readers)
      return std::nullopt;
    return std::max(*entry.writer, *entry.readers);
  }
  std::optional<Cycle> issue_frontier() const { return issue_; }
  // Explicit asynchronous v2 scheduling. A controller event may start in the
  // future without moving CPU admission to that future dependency. Physical
  // reservation ownership must be managed separately by the selected adapter.
  // The original submit/barrier semantics and v1 callers are unchanged.
  Result submit_detached(const Event &event) {
    const auto saved = issue_;
    try {
      const auto result = submit(event);
      issue_ = saved;
      return result;
    } catch (...) {
      issue_ = saved;
      throw;
    }
  }
  void advance_issue(std::optional<Cycle> admission) {
    if (!admission || !issue_) { issue_.reset(); return; }
    issue_ = std::max(*issue_, *admission);
  }
  void reset(ValueId value) {
    const auto done = last_use(value);
    if (!issue_ || !done || *done > *issue_)
      throw std::logic_error("cannot recycle a live or unknown scoreboard value");
    values_[static_cast<std::size_t>(value)] = {*issue_, *issue_};
  }

  // Attach an already scheduled asynchronous result to a reusable value token.
  // Resource scheduling remains the caller's responsibility; publication must
  // not move the issue frontier to that result's future completion.
  void publish(ValueId value, std::optional<Cycle> completion) {
    const auto done = last_use(value);
    if (!issue_ || !done || *done > *issue_)
      throw std::logic_error("cannot publish over a live or unknown scoreboard value");
    if (completion && *completion < *issue_)
      throw std::invalid_argument("published result cannot precede the issue frontier");
    values_[static_cast<std::size_t>(value)] = {completion, Cycle{0}};
    if (completion)
      horizon_ = std::max(horizon_, *completion);
    else
      count(unknown_events_);
  }

  Result submit(const Event &event) {
    validate(event);
    count(events_);
    selected_.clear();
    for (std::size_t i = 0; i < event.demands.size; ++i)
      selected_.push_back(find(event.demands[i].resource));
    Cycle issue = event.earliest;
    if (!issue_)
      return unavailable(event, UnknownReason::barrier);
    issue = std::max(issue, *issue_);
    for (const auto reads : {event.reads, event.retained_reads})
      for (std::size_t i = 0; i < reads.size; ++i) {
        const auto value = ready(reads[i]);
        if (!value)
          return unavailable(event, UnknownReason::dependency);
        issue = std::max(issue, *value);
      }
    for (std::size_t i = 0; i < event.writes.size; ++i) {
      const auto done = last_use(event.writes[i]);
      if (!done)
        return unavailable(event, UnknownReason::dependency);
      issue = std::max(issue, *done);
    }
    for (std::size_t i = 0; i < selected_.size(); ++i) {
      if (!selected_[i]->known)
        return unavailable(event, UnknownReason::resource);
      if (!event.demands[i].occupancy)
        return unavailable(event, UnknownReason::missing_occupancy);
    }

    // A waiting queue contains reservations that have not started at issue.
    // Full queues stall admission. Compute and transfer prices remain separate.
    Cycle start;
    for (;;) {
      start = issue;
      for (std::size_t i = 0; i < selected_.size(); ++i)
        start = std::max(start, selected_[i]->lanes[event.demands[i].units - 1]);
      Cycle next_issue = issue;
      if (start > issue) {
        for (auto *resource : selected_) {
          resource->retire(issue);
          if (resource->spec.queue_capacity == 0)
            next_issue = std::max(next_issue, start);
          else if (resource->queue.size() >= resource->spec.queue_capacity)
            next_issue = std::max(next_issue, *std::min_element(resource->queue.begin(), resource->queue.end()));
        }
      }
      if (next_issue == issue)
        break;
      issue = next_issue;
    }
    // Preflight every addition before committing any reservation.
    for (std::size_t i = 0; i < event.demands.size; ++i)
      if (!addition(start, *event.demands[i].occupancy))
        return unavailable(event, UnknownReason::cycle_overflow);
    if (event.latency && !addition(start, *event.latency))
      return unavailable(event, UnknownReason::cycle_overflow);

    issue_ = issue;
    for (std::size_t i = 0; i < selected_.size(); ++i) {
      auto &resource = *selected_[i];
      const auto &demand = event.demands[i];
      resource.retire(issue);
      const Cycle finish = start + *demand.occupancy;
      for (std::size_t lane = 0; lane < demand.units; ++lane)
        resource.lanes[lane] = finish;
      std::sort(resource.lanes.begin(), resource.lanes.end());
      if (start > issue)
        resource.queue.push_back(start);
      resource.peak_queue = std::max(resource.peak_queue, resource.queue.size());
      count(resource.reservations);
      if (resource.busy) {
        if (demand.units > std::numeric_limits<Cycle>::max() ||
            (*demand.occupancy != 0 && demand.units > std::numeric_limits<Cycle>::max() / *demand.occupancy) ||
            !addition(*resource.busy, *demand.occupancy * static_cast<Cycle>(demand.units))) {
          resource.busy.reset();
          accounting_overflow_ = true;
        } else {
          *resource.busy += *demand.occupancy * static_cast<Cycle>(demand.units);
        }
      }
      horizon_ = std::max(horizon_, finish);
    }
    std::optional<Cycle> completion;
    if (event.latency) {
      completion = start + *event.latency;
      horizon_ = std::max(horizon_, *completion);
    } else {
      count(unknown_events_);
      for (auto *resource : selected_)
        count(resource->unknown_events);
    }
    const auto release = event.hold_reads_until_ready ? completion : std::optional<Cycle>{issue};
    for (std::size_t i = 0; i < event.reads.size; ++i) {
      auto &readers = values_[static_cast<std::size_t>(event.reads[i])].readers;
      readers = readers && release ? std::optional<Cycle>{std::max(*readers, *release)} : std::nullopt;
    }
    for (std::size_t i = 0; i < event.retained_reads.size; ++i) {
      auto &readers = values_[static_cast<std::size_t>(event.retained_reads[i])].readers;
      readers = readers && completion ? std::optional<Cycle>{std::max(*readers, *completion)} : std::nullopt;
    }
    for (std::size_t i = 0; i < event.writes.size; ++i)
      values_[static_cast<std::size_t>(event.writes[i])] = {completion, Cycle{0}};
    return {issue, start, completion, completion ? UnknownReason::none : UnknownReason::missing_latency};
  }

  // Drain all modeled work. This has no implicit fence-instruction price.
  // Once prior work has unknown timing, a global drain also has unknown timing.
  std::optional<Cycle> barrier() {
    if (unknown_events_ != 0 || accounting_overflow_) {
      issue_.reset();
      return std::nullopt;
    }
    issue_ = std::max(*issue_, horizon_);
    for (auto &resource : resources_)
      resource.retire(*issue_);
    return issue_;
  }

  Report report() const {
    Report result{horizon_, events_, unknown_events_, unknown_events_ == 0 && !accounting_overflow_,
                  accounting_overflow_, values_.size(), 0, {}};
    result.resources.reserve(resources_.size());
    for (const auto &resource : resources_) {
      std::size_t pending = 0;
      for (const auto start : resource.queue)
        if (!issue_ || start > *issue_)
          ++pending;
      result.pending_queue_entries += pending;
      result.resources.push_back({resource.spec.id, resource.reservations, resource.busy,
                                  resource.unknown_events, resource.peak_queue, pending, resource.known});
    }
    return result;
  }

private:
  struct Resource {
    ResourceSpec spec;
    std::vector<Cycle> lanes;
    std::vector<Cycle> queue;
    std::uint64_t reservations = 0;
    std::optional<Cycle> busy = Cycle{0};
    std::uint64_t unknown_events = 0;
    std::size_t peak_queue = 0;
    bool known = true;
    explicit Resource(ResourceSpec selection) : spec(selection), lanes(selection.capacity, 0) {
      queue.reserve(selection.queue_capacity);
    }
    void retire(Cycle now) {
      queue.erase(std::remove_if(queue.begin(), queue.end(), [now](Cycle start) { return start <= now; }), queue.end());
    }
  };
  struct Value {
    std::optional<Cycle> writer = Cycle{0};
    std::optional<Cycle> readers = Cycle{0};
  };
  std::vector<Value> values_;
  std::vector<Resource> resources_;
  std::vector<Resource *> selected_;
  std::optional<Cycle> issue_ = Cycle{0};
  Cycle horizon_ = 0;
  std::uint64_t events_ = 0;
  std::uint64_t unknown_events_ = 0;
  bool accounting_overflow_ = false;

  static bool addition(Cycle a, Cycle b) { return b <= std::numeric_limits<Cycle>::max() - a; }
  void count(std::uint64_t &value) {
    if (value == std::numeric_limits<std::uint64_t>::max())
      accounting_overflow_ = true;
    else
      ++value;
  }
  Resource *find(ResourceId id) {
    for (auto &resource : resources_)
      if (resource.spec.id == id)
        return &resource;
    throw std::invalid_argument("undeclared resource id");
  }
  template <class T> static void valid_view(View<T> view) {
    if (view.size != 0 && view.data == nullptr)
      throw std::invalid_argument("nonempty event view has no storage");
  }
  void validate(const Event &event) {
    valid_view(event.reads);
    valid_view(event.retained_reads);
    valid_view(event.writes);
    valid_view(event.demands);
    if (event.reads.size > values_.size() || event.retained_reads.size > values_.size() ||
        event.writes.size > values_.size() ||
        event.demands.size > resources_.size())
      throw std::invalid_argument("event exceeds declared state bounds");
    for (const auto view : {event.reads, event.retained_reads, event.writes})
      for (std::size_t i = 0; i < view.size; ++i)
        if (view[i] >= values_.size())
          throw std::out_of_range("value id exceeds scoreboard bound");
    for (std::size_t i = 0; i < event.demands.size; ++i) {
      const auto &demand = event.demands[i];
      auto *resource = find(demand.resource);
      if (demand.units == 0 || demand.units > resource->spec.capacity)
        throw std::invalid_argument("resource demand exceeds declared capacity");
      for (std::size_t j = 0; j < i; ++j)
        if (event.demands[j].resource == demand.resource)
          throw std::invalid_argument("event repeats a resource demand");
    }
  }
  Result unavailable(const Event &event, UnknownReason why) {
    count(unknown_events_);
    for (const auto reads : {event.reads, event.retained_reads})
      for (std::size_t i = 0; i < reads.size; ++i)
        values_[static_cast<std::size_t>(reads[i])].readers.reset();
    for (std::size_t i = 0; i < event.writes.size; ++i)
      values_[static_cast<std::size_t>(event.writes[i])] = {std::nullopt, Cycle{0}};
    for (auto *resource : selected_) {
      resource->known = false;
      count(resource->unknown_events);
    }
    return {std::nullopt, std::nullopt, std::nullopt, why};
  }
};

struct CacheSpec {
  std::size_t sets;
  std::size_t ways;
  std::uint64_t line_bytes;
};
struct CacheProbe {
  bool hit;
  bool dirty;
  std::optional<std::size_t> slot = std::nullopt;
};
struct CacheAccess {
  bool hit;
  std::optional<std::uint64_t> evicted_line;
  bool dirty_eviction;
  std::size_t slot = 0;
};
struct CacheReport {
  std::uint64_t accesses;
  std::uint64_t hits;
  std::uint64_t misses;
  std::size_t entries;
  std::size_t capacity;
  bool accounting_overflow;
};

// Classification only: no allocation, hit, miss, writeback or coherence price
// follows from a tag lookup. The caller supplies memory hierarchy policy.
class SetAssociativeCache {
public:
  explicit SetAssociativeCache(CacheSpec spec) : spec_(spec) {
    if (spec.sets == 0 || spec.ways == 0 || spec.line_bytes == 0 ||
        spec.sets > std::numeric_limits<std::size_t>::max() / spec.ways)
      throw std::invalid_argument("cache geometry must be positive and bounded");
    lines_.resize(spec.sets * spec.ways);
  }
  CacheProbe probe(std::uint64_t address) const {
    const auto line = address / spec_.line_bytes;
    const auto base = static_cast<std::size_t>(line % spec_.sets) * spec_.ways;
    for (std::size_t i = 0; i < spec_.ways; ++i) {
      const auto &slot = lines_[base + i];
      if (slot.valid && slot.line == line)
        return {true, slot.dirty, base + i};
    }
    return {false, false, std::nullopt};
  }
  CacheProbe invalidate(std::uint64_t address) {
    const auto line = address / spec_.line_bytes;
    const auto base = static_cast<std::size_t>(line % spec_.sets) * spec_.ways;
    for (std::size_t i = 0; i < spec_.ways; ++i) {
      auto &slot = lines_[base + i];
      if (slot.valid && slot.line == line) {
        const bool dirty = slot.dirty;
        for (std::size_t j = 0; j < spec_.ways; ++j)
          if (lines_[base + j].valid && lines_[base + j].rank > slot.rank)
            --lines_[base + j].rank;
        slot = {};
        --entries_;
        return {true, dirty, base + i};
      }
    }
    return {false, false, std::nullopt};
  }
  void clear() {
    std::fill(lines_.begin(), lines_.end(), Line{});
    entries_ = 0;
  }
  CacheAccess access(std::uint64_t address, bool write = false) {
    count(accesses_);
    const auto line = address / spec_.line_bytes;
    const auto base = static_cast<std::size_t>(line % spec_.sets) * spec_.ways;
    std::size_t victim = spec_.ways;
    for (std::size_t i = 0; i < spec_.ways; ++i) {
      auto &slot = lines_[base + i];
      if (slot.valid && slot.line == line) {
        count(hits_);
        touch(base, i);
        slot.dirty = slot.dirty || write;
        return {true, std::nullopt, false, base + i};
      }
      if (!slot.valid && victim == spec_.ways)
        victim = i;
    }
    count(misses_);
    if (victim == spec_.ways) {
      victim = 0;
      for (std::size_t i = 1; i < spec_.ways; ++i)
        if (lines_[base + i].rank > lines_[base + victim].rank)
          victim = i;
    }
    auto &slot = lines_[base + victim];
    const auto old = slot.valid ? std::optional<std::uint64_t>{slot.line} : std::nullopt;
    const bool dirty = slot.valid && slot.dirty;
    if (!slot.valid)
      ++entries_;
    touch(base, victim);
    slot = {true, write, line, 0};
    return {false, old, dirty, base + victim};
  }
  CacheReport report() const { return {accesses_, hits_, misses_, entries_, lines_.size(), overflow_}; }

private:
  struct Line {
    bool valid = false;
    bool dirty = false;
    std::uint64_t line = 0;
    std::size_t rank = 0;
  };
  CacheSpec spec_;
  std::vector<Line> lines_;
  std::size_t entries_ = 0;
  std::uint64_t accesses_ = 0, hits_ = 0, misses_ = 0;
  bool overflow_ = false;
  void count(std::uint64_t &value) {
    if (value == std::numeric_limits<std::uint64_t>::max())
      overflow_ = true;
    else
      ++value;
  }
  void touch(std::size_t base, std::size_t selected) {
    auto &slot = lines_[base + selected];
    const auto old_rank = slot.rank;
    for (std::size_t i = 0; i < spec_.ways; ++i)
      if (i != selected && lines_[base + i].valid && (!slot.valid || lines_[base + i].rank < old_rank))
        ++lines_[base + i].rank;
    slot.rank = 0;
  }
};

} // namespace merlin_timing
#endif
