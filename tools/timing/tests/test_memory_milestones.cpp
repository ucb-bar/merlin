#include "merlin_timing/ready_response_arbiter.hpp"
#include "merlin_timing/memory_source_ownership.hpp"
#include <cassert>

using namespace merlin_timing;
static MemoryMilestonesV1 request(std::uint64_t id, std::size_t reader,
                                  std::size_t source, Cycle accepted,
                                  std::vector<Cycle> available) {
  MemoryMilestonesV1 m;
  m.request = {id, 0x1000 + id * 64, available.size() * 16, reader, 1, accepted};
  m.fill = MemoryFillIdentityV1{8, m.request.address / 64, id + 32, 1};
  m.source = MemorySourceLeaseV1{source, 1, accepted, {}};
  m.coverage = MemoryMilestoneCoverage::source_closed;
  m.lookup_complete = accepted;
  m.line_visible = available.back() + 3;
  m.beat_bytes = 16; m.beat_available = std::move(available);
  return m;
}
int main() {
  // A later request can finish first. Return-source ID, cache-slot ID and
  // finite memory scheduler source ID intentionally differ.
  ReadyResponseArbiter a({12, 16, 16, 1});
  auto slow = request(1, 13, 3, 1, {20, 21});
  auto fast = request(2, 2, 7, 2, {6});
  assert(a.enqueue(slow)); assert(a.enqueue(fast));
  const auto accept = [](const auto &b) { return std::optional<Cycle>{b.offered}; };
  assert(!a.advance(30, false, accept)); // unclosed ready set cannot arbitrate
  assert(a.advance(30, true, accept)); assert(a.pending() == 0);
  assert(a.beats().size() == 3 && a.beats()[0].request == 2);
  assert(a.beats()[1].request == 1 && a.beats()[1].accepted == 20);
  assert(*slow.line_visible == 24); // consumer never rewrites line-fill time
  assert(!a.enqueue(request(3, 1, 0, 29, {31}))); // no retroactive admissions

  // Pointer advances after the winning SOURCE, not the returning reader ID.
  ReadyResponseArbiter rr({12, 16, 16, 1});
  assert(rr.scheduler_action(0, 4));
  assert(rr.enqueue(request(1, 0, 2, 1, {4})));
  assert(rr.enqueue(request(2, 1, 7, 1, {4})));
  assert(rr.advance(20, true, accept));
  assert(rr.beats()[0].source == 7 && rr.beats()[1].source == 2);

  // One response port; a stalled multibeat message keeps its identity.
  ReadyResponseArbiter bp({12, 16, 16, 1});
  assert(bp.enqueue(request(1, 2, 0, 1, {3, 4})));
  assert(bp.enqueue(request(2, 3, 1, 1, {4})));
  assert(!bp.enqueue(request(3, 2, 2, 1, {4}))); // reader credit is still owned
  assert(bp.advance(30, true, [](const auto &b) {
    return std::optional<Cycle>{b.request == 1 && b.index == 0 ? 10 : b.offered};
  }));
  assert(bp.beats()[0].accepted == 10 && bp.beats()[1].accepted == 11);
  assert(bp.beats()[2].accepted == 12 && bp.beats()[2].last);
  assert(bp.take_beats().size() == 3 && bp.beats().empty());
  auto reused = request(4, 2, 2, 31, {32});
  assert(!bp.enqueue(reused)); // generation cannot alias a retired response
  reused.request.return_generation = 2;
  assert(bp.enqueue(reused));

  // Crossing the observation frontier does not expose a future reader credit.
  ReadyResponseArbiter frontier({12, 16, 16, 1});
  assert(frontier.enqueue(request(1, 0, 0, 1, {3, 20})));
  assert(frontier.advance(4, true, accept));
  auto too_early = request(2, 0, 1, 5, {6});
  too_early.request.return_generation = 2;
  assert(!frontier.enqueue(too_early));

  // Aggregate completion is explicitly transportable, never promoted to
  // first beat, fill visibility, a synthetic MSHR, or permission to reorder.
  auto bound = request(1, 0, 0, 1, {9});
  bound.coverage = MemoryMilestoneCoverage::payload_completion_bound;
  bound.beat_available.clear(); bound.source.reset(); bound.fill.reset();
  bound.line_visible.reset(); bound.lookup_complete.reset(); bound.payload_complete_bound = 9;
  assert(valid_memory_milestones(bound, 12));
  ReadyResponseArbiter no_guess({12, 16, 16, 1});
  assert(!no_guess.enqueue(bound));
  auto bad = request(2, 0, 12, 1, {9});
  assert(!valid_memory_milestones(bad, 12));
  bad = request(2, 0, 0, 1, {9}); bad.line_visible = 10;
  bad.first_refill_beat = 8; bad.last_refill_beat = 11;
  assert(!valid_memory_milestones(bad, 12));
  bad = request(2, 0, 0, 1, {9}); bad.fill.reset();
  assert(!no_guess.enqueue(bad));
  assert(no_guess.enqueue(request(3, 0, 0, 1, {9})));
  assert(!no_guess.advance(20, true, [](const auto &) { return std::optional<Cycle>{}; }));
  assert(no_guess.failed()); // a missing handshake never becomes zero cost

  MemorySourceOwnership sources({2, 4, 1});
  assert(sources.accept({1, 7, 1}, false, false).status == MemorySourceOwnership::Status::unavailable);
  assert(sources.accept({1, 7, 1}, true, true).status == MemorySourceOwnership::Status::blocked);
  const auto s0=sources.accept({1, 7, 1}, true, false);
  const auto s1=sources.accept({2, 8, 2}, true, false);
  assert(s0.source && s0.source->source==0 && s1.source && s1.source->source==1);
  assert(sources.accept({3, 9, 3}, true, false).status == MemorySourceOwnership::Status::blocked);
  assert(sources.accept({3, 7, 3}, true, false).status == MemorySourceOwnership::Status::queued);
  assert(sources.accept({4, 7, 4}, true, false).status == MemorySourceOwnership::Status::blocked);
  const auto reload=sources.retire(0,1,1,5,true);
  assert(reload && reload->identity==3 && sources.queued()==0);
  assert(sources.lease(0)->generation==2 && sources.lease(0)->allocated==5);
  bool wrong_owner=false;
  try { sources.retire(0,1,1,6,true); } catch(const std::invalid_argument &) { wrong_owner=true; }
  assert(wrong_owner);
  assert(!sources.retire(0,2,3,6,true));
  const auto reused_source=sources.accept({4,9,7},true,false);
  assert(reused_source.source && reused_source.source->source==0 && reused_source.source->generation==3);
}
