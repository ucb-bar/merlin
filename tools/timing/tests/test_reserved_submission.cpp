#include <merlin_timing/reserved_submission.hpp>

#include <iostream>
#include <stdexcept>

using namespace merlin_timing;

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template <class Error, class F> static void refuses(F action) {
  try { action(); } catch (const Error &) { return; }
  throw std::runtime_error("expected refusal");
}

static void credits_and_service_are_distinct() {
  const Demand demand{0, 1, 2};
  const Event event{0, {}, {}, {&demand, 1}, 10};
  Engine one(0, {{0, 1, 0}}), two(0, {{0, 1, 0}}), start(0, {{0, 1, 0}});
  ReservationWindow one_credit(1), two_credits(2), start_credit(1);
  submit_reserved(one, one_credit, event, CreditReleaseStage::ready);
  const auto a = submit_reserved(one, one_credit, event, CreditReleaseStage::ready);
  submit_reserved(two, two_credits, event, CreditReleaseStage::ready);
  const auto b = submit_reserved(two, two_credits, event, CreditReleaseStage::ready);
  submit_reserved(start, start_credit, event, CreditReleaseStage::start);
  const auto c = submit_reserved(start, start_credit, event, CreditReleaseStage::start);
  require(a.admission == 10 && a.execution->start == 10 && a.release == 20,
          "ready-held credit must survive service end at2");
  require(b.admission == 0 && b.execution->issue == 2 && b.release == 12,
          "two credits admit before the single service lane is ready");
  require(c.admission == 0 && c.execution->start == 2 && c.release == 2,
          "explicit start release differs from completion ownership");
}

static void issue_start_and_waiting_queue() {
  const Demand demand{0, 1, 5};
  const Event event{0, {}, {}, {&demand, 1}, 20};
  Engine issue(0, {{0, 1, 1}}), start(0, {{0, 1, 1}});
  ReservationWindow issue_credit(1), start_credit(1);
  for (int i = 0; i < 2; ++i) {
    const auto a = submit_reserved(issue, issue_credit, event, CreditReleaseStage::issue);
    const auto b = submit_reserved(start, start_credit, event, CreditReleaseStage::start);
    require(a.admission == 0 && b.admission == 0, "first two admissions share epoch0");
    if (i == 1)
      require(a.execution->issue == 0 && a.execution->start == 5 && a.release == 0 && b.release == 5,
              "queued command releases issue credit before its service start");
  }
  const auto a = submit_reserved(issue, issue_credit, event, CreditReleaseStage::issue);
  const auto b = submit_reserved(start, start_credit, event, CreditReleaseStage::start);
  require(a.admission == 0 && b.admission == 5 && a.execution->issue == 5 && b.execution->issue == 5 &&
          a.execution->start == 10 && b.execution->start == 10,
          "queue-full backpressure and credit admission are separate stages");
  require(issue.report().resources[0].peak_queue_entries == 1, "start queue remains bounded");
}

static void explicit_structural_capacities() {
  // Synthetic inputs with the same sizes as a separately extracted census;
  // these numbers supply no price or target classification to the consumer.
  for (std::size_t capacity : {8, 16, 4}) {
    Engine engine(0, {{0, 1, 0}});
    ReservationWindow credits(capacity);
    const Demand demand{0, 1, 1};
    for (std::size_t i = 0; i < capacity; ++i) {
      const auto result = submit_reserved(engine, credits, {0, {}, {}, {&demand, 1}, 20}, CreditReleaseStage::ready);
      require(result.status == ReservedSubmissionStatus::complete && result.admission == (i == 0 ? 0 : i - 1),
              "service backpressure bounds the in-order frontier independently");
    }
    const auto result = submit_reserved(engine, credits, {0, {}, {}, {&demand, 1}, 20}, CreditReleaseStage::ready);
    require(result.admission == 20 && result.execution->start == 20 && result.release == 40,
            "the next credit waits for completion, not occupancy end");
    require(engine.report().resources[0].busy_unit_cycles == capacity + 1,
            "service accounting is not live-credit duration");
  }
}

static void unknowns_and_refusals() {
  Engine engine(0, {});
  ReservationWindow credits(1);
  const auto first = submit_reserved(engine, credits, {0, {}, {}, {}, {}}, CreditReleaseStage::ready);
  require(first.status == ReservedSubmissionStatus::execution_unknown && first.admission == 0 &&
          first.execution && !first.release && !credits.drained(0), "unknown readiness retains ownership");
  const auto next = submit_reserved(engine, credits, {0, {}, {}, {}, 1}, CreditReleaseStage::ready);
  require(next.status == ReservedSubmissionStatus::credit_unknown && !next.admission && !next.execution &&
          !next.release && !engine.issue_frontier(), "unresolved credit is not a zero-cycle event");
  ReservationWindow other(1);
  const auto blocked = submit_reserved(engine, other, {0, {}, {}, {}, 1}, CreditReleaseStage::issue);
  require(blocked.status == ReservedSubmissionStatus::frontier_unknown && !blocked.execution,
          "unknown in-order admission blocks independent later commands");

  Engine early(0, {});
  ReservationWindow early_credit(1);
  const auto unfinished = submit_reserved(early, early_credit, {0, {}, {}, {}, {}}, CreditReleaseStage::issue);
  require(unfinished.status == ReservedSubmissionStatus::execution_unknown && unfinished.release == 0 &&
          !unfinished.execution->ready && early_credit.drained(0) == 0,
          "explicit issue release supplies no missing completion price");

  Engine zero(0, {});
  ReservationWindow zero_credit(1);
  const auto immediate = submit_reserved(zero, zero_credit, {0, {}, {}, {}, 0}, CreditReleaseStage::ready);
  require(immediate.status == ReservedSubmissionStatus::complete && immediate.release == 0,
          "explicit zero latency remains distinct from unknown latency");
  refuses<std::invalid_argument>([] { ReservationWindow invalid(0); });
  refuses<std::invalid_argument>([] { ReservationWindow invalid(65537); });
  refuses<std::invalid_argument>([&] {
    submit_reserved(zero, zero_credit, {0, {}, {}, {}, 1}, static_cast<CreditReleaseStage>(99));
  });
  require(zero_credit.drained(0) == 0, "invalid stage must not allocate a credit");

  Engine invalid(0, {{0, 1, 0}});
  ReservationWindow held(1);
  const Demand excessive{0, 2, 1};
  refuses<std::invalid_argument>([&] {
    submit_reserved(invalid, held, {0, {}, {}, {&excessive, 1}, 1}, CreditReleaseStage::ready);
  });
  require(!held.drained(0) && !held.allocate(0), "failed submission must not free its allocated entry");

  Engine overflow(0, {});
  ReservationWindow overflow_credit(1);
  const auto result = submit_reserved(overflow, overflow_credit,
                                     {std::numeric_limits<Cycle>::max(), {}, {}, {}, 1}, CreditReleaseStage::ready);
  require(result.status == ReservedSubmissionStatus::execution_unknown && !result.release &&
          result.execution->unknown == UnknownReason::cycle_overflow && !overflow_credit.drained(0),
          "overflow cannot fabricate completion or release");

  Engine missing(0, {{0, 1, 0}});
  ReservationWindow missing_credit(1);
  const Demand unknown_service{0, 1, {}};
  const auto unknown = submit_reserved(missing, missing_credit, {0, {}, {}, {&unknown_service, 1}, 1},
                                       CreditReleaseStage::start);
  require(unknown.status == ReservedSubmissionStatus::execution_unknown && !unknown.release &&
          unknown.execution->unknown == UnknownReason::missing_occupancy && !missing_credit.drained(0),
          "a latency alone does not supply resource occupancy");
}

static void frontier_dependencies_and_parallel_windows() {
  const Demand a{0, 1, 1}, b{1, 1, 1};
  Engine engine(2, {{0, 1, 0}, {1, 1, 0}});
  ReservationWindow left(1), right(1);
  const ValueId value = 0;
  const auto x = submit_reserved(engine, left, {0, {}, {&value, 1}, {&a, 1}, 20}, CreditReleaseStage::ready);
  const auto y = submit_reserved(engine, right, {0, {}, {}, {&b, 1}, 7}, CreditReleaseStage::ready);
  require(x.execution->start == 0 && y.execution->start == 0 && y.release == 7,
          "separate resources and credit windows overlap");
  ReservationWindow reader(1);
  const auto dependent = submit_reserved(engine, reader, {0, {&value, 1}, {}, {}, 3}, CreditReleaseStage::ready);
  require(dependent.execution->issue == 20 && dependent.release == 23, "RAW waits at issue, not OOO dispatch");
  ReservationWindow after(1);
  const auto later = submit_reserved(engine, after, {1, {}, {}, {}, 1}, CreditReleaseStage::issue);
  require(later.admission == 20 && later.release == 20, "event earliest cannot backdate the current frontier");

  Engine memory(1, {});
  ReservationWindow readers(1), writers(1);
  submit_reserved(memory, readers, {0, {&value, 1}, {}, {}, 9, true}, CreditReleaseStage::ready);
  const auto overwrite = submit_reserved(memory, writers, {0, {}, {&value, 1}, {}, 2}, CreditReleaseStage::ready);
  require(overwrite.execution->issue == 9 && overwrite.release == 11, "retained reads enforce WAR");
  ReservationWindow next_writer(1);
  const auto waw = submit_reserved(memory, next_writer, {0, {}, {&value, 1}, {}, 2}, CreditReleaseStage::ready);
  require(waw.execution->issue == 11 && waw.release == 13, "WAW follows prior completion");
}

int main() {
  credits_and_service_are_distinct();
  issue_start_and_waiting_queue();
  explicit_structural_capacities();
  unknowns_and_refusals();
  frontier_dependencies_and_parallel_windows();
  std::cout << "reserved submission checks passed\n";
}
