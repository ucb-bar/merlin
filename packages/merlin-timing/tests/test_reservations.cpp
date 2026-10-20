#include <merlin_timing/reservations.hpp>
#include <cassert>

int main() {
  using namespace merlin_timing;
  ReservationWindow reservations(2);
  const auto load = reservations.allocate(2), preload = reservations.allocate(3);
  assert(load && preload && !reservations.allocate(4));
  // A lone excluded PRELOAD does not exclude unfinished DMA work.
  assert(!reservations.drained(4, preload));
  reservations.release(*load, 50);
  assert(reservations.drained(4, preload) == 50);
  assert(!reservations.drained(4));
  // Completion ownership, not start/service, frees the ordinary entry.
  auto next = reservations.allocate(5);
  assert(next && next->admitted == 50);
  reservations.release(*next, 51);
  bool refused = false;
  try { reservations.release(*load, 100); } catch (const std::logic_error &) { refused = true; }
  assert(refused);
  reservations.release(*preload, 80);
  assert(reservations.drained(6) == 80);

  Engine engine(4, {{0, 1, 0}});
  const ValueId token = 0;
  const Demand service{0, 1, 10};
  Event effect{20, {}, {&token, 1}, {&service, 1}, 30};
  const auto detached = engine.submit_detached(effect);
  assert(detached.issue == 20 && detached.ready == 50);
  assert(engine.issue_frontier() == 0 && engine.ready(token) == 50);
  engine.advance_issue(5);
  assert(engine.issue_frontier() == 5);
  assert(engine.barrier() == 50);
}
