#include <merlin_timing/engine.hpp>

#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace {
bool count_allocations = false;
std::uint64_t allocations = 0;
}
void *operator new(std::size_t bytes) {
  if (count_allocations)
    ++allocations;
  if (void *storage = std::malloc(bytes == 0 ? 1 : bytes))
    return storage;
  throw std::bad_alloc();
}
void operator delete(void *storage) noexcept { std::free(storage); }
void operator delete(void *storage, std::size_t) noexcept { std::free(storage); }
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void *storage) noexcept { ::operator delete(storage); }
void operator delete[](void *storage, std::size_t) noexcept { ::operator delete(storage); }

using namespace merlin_timing;
static_assert(!std::is_copy_constructible<Engine>::value, "stateful engines cannot alias resource scratch pointers");
static_assert(std::is_move_constructible<Engine>::value, "engines retain explicit ownership on move");

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class Error, class F> static void refuses(F action, const char *message) {
  try {
    action();
  } catch (const Error &) {
    return;
  }
  throw std::runtime_error(message);
}

static void dependencies_and_overlap() {
  const Demand a{1, 1, 1}, b{2, 1, 1};
  const ValueId x = 1, y = 2;
  Engine engine(4, {{1, 1, 2}, {2, 1, 2}});
  auto first = engine.submit({0, {}, {&x, 1}, {&a, 1}, 8});
  auto independent = engine.submit({0, {}, {&y, 1}, {&b, 1}, 2});
  require(first.ready == 8 && independent.start == 0 && independent.ready == 2,
          "independent resources must overlap without advancing issue to completion");
  auto consumer = engine.submit({0, {&x, 1}, {&y, 1}, {&b, 1}, 3});
  require(consumer.issue == 8 && consumer.ready == 11, "RAW must wait for the actual writer");
  require(engine.report().modeled_makespan == 11 && engine.report().complete,
          "the report must retain the complete modeled horizon");
}

static void mutable_hazards_and_recycling() {
  const ValueId x = 0;
  Engine writers(1, {});
  writers.submit({0, {}, {&x, 1}, {}, 9});
  require(writers.submit({0, {}, {&x, 1}, {}, 2}).ready == 11, "WAW must retain writer ordering");
  refuses<std::logic_error>([&] { writers.reset(x); }, "live writer recycling must refuse");
  require(writers.barrier() == 11, "drain must include all pending results");
  writers.reset(x);
  require(writers.ready(x) == 11 && writers.last_use(x) == 11, "completed tokens must recycle at the frontier");

  Engine held(1, {});
  held.submit({0, {&x, 1}, {}, {}, 9, true});
  refuses<std::logic_error>([&] { held.reset(x); }, "live reader recycling must refuse");
  require(held.submit({0, {}, {&x, 1}, {}, 2}).ready == 11, "WAR must retain asynchronous readers");
  Engine captured(1, {});
  captured.submit({0, {&x, 1}, {}, {}, 9, false});
  require(captured.submit({0, {}, {&x, 1}, {}, 2}).ready == 2,
          "operands captured at issue need not remain live through result latency");
}

static void asynchronous_publication() {
  const ValueId x = 0;
  Engine engine(1, {});
  engine.publish(x, 20);
  require(engine.issue_frontier() == 0 && engine.ready(x) == 20 && engine.report().modeled_makespan == 20,
          "publication must retain future readiness without stalling unrelated issue");
  require(engine.submit({0, {}, {}, {}, 1}).ready == 1, "independent CPU work must overlap asynchronous response");
  refuses<std::logic_error>([&] { engine.publish(x, 21); }, "publication cannot replace a pending writer");
  require(engine.submit({0, {&x, 1}, {}, {}, 1}).ready == 21, "consumers must wait the published response");
  refuses<std::invalid_argument>([&] { engine.publish(x, 19); }, "publication cannot backdate a result");
  require(engine.ready(x) == 20, "refused publication must preserve the prior value");
  engine.publish(x, 22);
  require(engine.ready(x) == 22 && engine.issue_frontier() == 20, "a completed token can publish its next result");

  Engine readers(1, {});
  readers.submit({0, {&x, 1}, {}, {}, 9, true});
  refuses<std::logic_error>([&] { readers.publish(x, 10); }, "publication cannot clobber pending readers");
  readers.barrier();
  readers.publish(x, std::nullopt);
  require(!readers.ready(x) && !readers.report().complete && readers.report().unknown_events == 1,
          "unknown response readiness must remain unknown and invalidate completeness");
  require(readers.submit({0, {&x, 1}, {}, {}, 1}).unknown == UnknownReason::dependency,
          "unknown publication must propagate to its consumers");
  refuses<std::logic_error>([&] { readers.publish(x, 20); }, "unknown response tokens cannot be silently repaired");
  refuses<std::out_of_range>([&] { readers.publish(1, 20); }, "unbounded response tokens must refuse");
}

static void separate_read_lifetimes() {
  const ValueId base = 0, memory = 1, result = 2;
  Engine engine(3, {});
  engine.submit({0, {&base, 1}, {&result, 1}, {}, 20, false, {&memory, 1}});
  require(engine.submit({0, {}, {&base, 1}, {}, 1}).ready == 1,
          "base operands captured at issue must not acquire memory-reader lifetimes");
  require(engine.submit({0, {}, {&memory, 1}, {}, 1}).ready == 21,
          "memory overwrite must retain the prior asynchronous reader until completion");

  Engine raw(3, {});
  raw.submit({0, {}, {&memory, 1}, {}, 5});
  require(raw.submit({0, {}, {&result, 1}, {}, 3, false, {&memory, 1}}).ready == 8,
          "retained operands must observe RAW dependencies before issue");

  Engine unknown(3, {});
  unknown.submit({0, {&base, 1}, {&result, 1}, {}, std::nullopt, false, {&memory, 1}});
  require(unknown.submit({0, {}, {&base, 1}, {}, 1}).ready == 1,
          "unknown result latency need not poison captured operand lifetimes");
  require(unknown.submit({0, {}, {&memory, 1}, {}, 1}).unknown == UnknownReason::dependency,
          "unknown retained-reader completion must block unsafe memory reuse");
  refuses<std::invalid_argument>([&] { engine.submit({0, {}, {}, {}, 1, false, {nullptr, 1}}); },
                                 "retained views require actual operand storage");
}

static void capacities_and_queues() {
  const Demand demand{1, 1, 10};
  Engine queued(0, {{1, 1, 1}});
  const auto first = queued.submit({0, {}, {}, {&demand, 1}, 10});
  const auto second = queued.submit({0, {}, {}, {&demand, 1}, 10});
  const auto third = queued.submit({0, {}, {}, {&demand, 1}, 10});
  require(first.start == 0 && second.issue == 0 && second.start == 10 && third.issue == 10 && third.start == 20,
          "a full waiting queue must backpressure issue");
  const auto summary = queued.report();
  require(summary.resources[0].reservations == 3 && summary.resources[0].busy_unit_cycles == 30 &&
              summary.resources[0].peak_queue_entries == 1,
          "reservation and busy-unit observations must reflect the original demands");
  require(queued.barrier() == 30 && queued.report().pending_queue_entries == 0,
          "draining must retire waiting reservations");

  Engine unbuffered(0, {{1, 1, 0}});
  unbuffered.submit({0, {}, {}, {&demand, 1}, 10});
  require(unbuffered.submit({0, {}, {}, {&demand, 1}, 10}).issue == 10,
          "zero queue capacity must wait at admission");

  Engine parallel(0, {{1, 2, 4}});
  const Demand short_job{1, 1, 4}, both{1, 2, 1};
  parallel.submit({0, {}, {}, {&demand, 1}, 10});
  require(parallel.submit({0, {}, {}, {&short_job, 1}, 4}).start == 0,
          "available lanes must overlap independent work");
  require(parallel.submit({0, {}, {}, {&both, 1}, 1}).start == 10,
          "multi-unit demand must wait for all required capacity");
}

static void unknowns_and_overflow() {
  const Demand known{1, 1, 5}, other{2, 1, 1}, missing{1, 1, std::nullopt};
  const ValueId x = 0;
  Engine incomplete(1, {{1, 1, 1}, {2, 1, 1}});
  const auto unknown = incomplete.submit({0, {}, {&x, 1}, {&known, 1}, std::nullopt});
  require(unknown.start == 0 && !unknown.ready && unknown.unknown == UnknownReason::missing_latency,
          "unknown result latency cannot become a zero-cycle result");
  require(incomplete.submit({0, {&x, 1}, {}, {&other, 1}, 1}).unknown == UnknownReason::dependency,
          "unknown result readiness must propagate");
  require(!incomplete.report().complete && incomplete.report().unknown_events == 2,
          "partial modeled work must be explicitly incomplete");
  require(!incomplete.barrier() && !incomplete.issue_frontier(), "unknown work cannot acquire a known drain time");
  require(incomplete.submit({0, {}, {}, {}, 1}).unknown == UnknownReason::barrier,
          "unknown fence timing must constrain later work");
  refuses<std::logic_error>([&] { incomplete.reset(x); }, "unknown token recycling must refuse");

  Engine occupancy(0, {{1, 1, 1}, {2, 1, 1}});
  require(occupancy.submit({0, {}, {}, {&missing, 1}, 1}).unknown == UnknownReason::missing_occupancy,
          "an unpriced resource cannot be treated as free");
  require(occupancy.submit({0, {}, {}, {&known, 1}, 1}).unknown == UnknownReason::resource,
          "unknown resource occupancy must constrain later reservations");
  require(occupancy.submit({0, {}, {}, {&other, 1}, 1}).ready == 1,
          "unknowns need not erase unrelated known observations");

  Engine overflow(0, {{1, 1, 1}});
  require(overflow.submit({std::numeric_limits<Cycle>::max() - 1, {}, {}, {&known, 1}, 2}).unknown ==
              UnknownReason::cycle_overflow,
          "cycle arithmetic cannot wrap into a short estimate");
  require(!overflow.report().complete, "overflow must invalidate completeness");
}

static void cache_lru() {
  SetAssociativeCache cache({2, 2, 4});
  require(!cache.access(0, true).hit && !cache.access(8).hit && cache.access(0).hit, "line hits must update recency");
  const auto eviction = cache.access(16);
  require(eviction.evicted_line == 2 && !eviction.dirty_eviction, "conflicts must evict the least recent line");
  const auto dirty = cache.access(24);
  require(dirty.evicted_line == 0 && dirty.dirty_eviction, "dirty state must remain explicit for caller pricing");
  require(!cache.access(4).hit && cache.access(5).hit, "set and line offsets must be distinguished");
  const auto summary = cache.report();
  require(summary.accesses == 7 && summary.hits == 2 && summary.misses == 5 &&
              summary.entries == 3 && summary.capacity == 4,
          "cache classification must preserve its complete bounded census");
}

static void cache_coherence_controls() {
  SetAssociativeCache cache({1, 3, 4});
  cache.access(0, true);
  cache.access(4);
  cache.access(8);
  require(cache.probe(1).hit && cache.probe(1).dirty && !cache.probe(12).hit,
          "probe must classify resident dirty lines without filling");
  require(cache.report().accesses == 3, "probes must not count as memory accesses");
  cache.access(12);
  require(!cache.probe(0).hit, "probe must not update replacement recency");
  const auto clean = cache.invalidate(4);
  require(clean.hit && !clean.dirty && !cache.invalidate(4).hit && cache.report().entries == 2,
          "invalidation must free the line once without inventing a writeback price");
  cache.access(16, true);
  const auto dirty = cache.invalidate(17);
  require(dirty.hit && dirty.dirty, "dirty invalidation must be reported to the policy owner");
  cache.clear();
  require(cache.report().entries == 0 && cache.report().accesses == 5 && !cache.probe(8).hit,
          "clear must remove all residency while preserving the observed access census");
  require(!cache.access(8).hit, "a cleared line must be cold");
}

static void cache_slot_identity() {
  SetAssociativeCache cache({2, 2, 4});
  const auto a = cache.access(0), b = cache.access(8), c = cache.access(4);
  require(a.slot == 0 && b.slot == 1 && c.slot == 2,
          "slot identity must be deterministic set times ways plus way");
  require(cache.access(1).slot == a.slot && cache.probe(0).slot == a.slot,
          "hits and probes must retain the same physical slot identity");
  const auto replacement = cache.access(16);
  require(replacement.slot == b.slot && replacement.evicted_line == 2 && !cache.probe(8).slot,
          "evictions must reuse the replaced slot and missing probes must have no slot");
  const auto invalid = cache.invalidate(0);
  require(invalid.slot == a.slot && cache.access(24).slot == a.slot,
          "invalidation must expose the freed slot and later fills must reuse it");
  cache.clear();
  require(!cache.probe(24).slot && cache.access(0).slot == 0,
          "clearing residency must preserve deterministic bounded slot numbering");
}

static void bounded_streaming() {
  Engine engine(1, {{1, 2, 3}});
  SetAssociativeCache cache({3, 2, 5});
  const Demand demand{1, 1, 7};
  const auto before = allocations;
  count_allocations = true;
  for (std::uint64_t i = 0; i < 10000; ++i) {
    engine.submit({0, {}, {}, {&demand, 1}, 9});
    cache.access(i * 11, (i % 2) != 0);
  }
  count_allocations = false;
  require(allocations == before, "streaming events and cache accesses must allocate no per-event state");
  const auto summary = engine.report();
  require(summary.events == 10000 && summary.scoreboard_entries == 1 &&
              summary.pending_queue_entries <= 3 && summary.resources[0].peak_queue_entries <= 3 &&
              cache.report().entries <= 6,
          "state must remain bounded independently of trace length");
}

static void invalid_inputs() {
  refuses<std::invalid_argument>([] { Engine engine(1, {{1, 0, 1}}); }, "zero resource capacity must refuse");
  refuses<std::invalid_argument>([] { Engine engine(1, {{1, 1, 1}, {1, 2, 1}}); }, "duplicate resources must refuse");
  refuses<std::invalid_argument>([] { SetAssociativeCache cache({0, 1, 1}); }, "zero cache geometry must refuse");
  Engine engine(1, {{1, 1, 1}, {2, 1, 1}});
  const ValueId bad = 1;
  refuses<std::out_of_range>([&] { engine.submit({0, {&bad, 1}, {}, {}, 1}); }, "unbounded value ids must refuse");
  const Demand oversized{1, 2, 1}, undeclared{3, 1, 1};
  const Demand duplicates[] = {{1, 1, 1}, {1, 1, 1}};
  refuses<std::invalid_argument>([&] { engine.submit({0, {}, {}, {&oversized, 1}, 1}); }, "over-capacity demand must refuse");
  refuses<std::invalid_argument>([&] { engine.submit({0, {}, {}, {&undeclared, 1}, 1}); }, "undeclared resources must refuse");
  refuses<std::invalid_argument>([&] { engine.submit({0, {}, {}, {duplicates, 2}, 1}); }, "duplicate demands must refuse");
  refuses<std::invalid_argument>([&] { engine.submit({0, {nullptr, 1}, {}, {}, 1}); }, "invalid views must refuse");
  require(engine.report().events == 0, "refused malformed inputs must not partially mutate event accounting");
}

int main() {
  try {
    dependencies_and_overlap();
    mutable_hazards_and_recycling();
    asynchronous_publication();
    separate_read_lifetimes();
    capacities_and_queues();
    unknowns_and_overflow();
    cache_lru();
    cache_coherence_controls();
    cache_slot_identity();
    bounded_streaming();
    invalid_inputs();
    std::cout << "11 timing core cases passed\n";
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
