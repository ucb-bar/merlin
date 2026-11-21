#ifndef MERLIN_TIMING_RESERVED_SUBMISSION_HPP
#define MERLIN_TIMING_RESERVED_SUBMISSION_HPP

#include "reservations.hpp"

namespace merlin_timing {

// An explicit original adapter chooses the stage. Its name supplies no hardware
// completion law, price, or permission to release an actual device entry.
enum class CreditReleaseStage { issue, start, ready };
enum class ReservedSubmissionStatus {
  complete,
  execution_unknown,
  credit_unknown,
  frontier_unknown
};

struct ReservedSubmission {
  ReservedSubmissionStatus status;
  std::optional<Cycle> admission;
  std::optional<Result> execution;
  std::optional<Cycle> release;
};

// Conservative, submitted-order advisory scheduling, NOT a ready-select ROB.
// A credit is owned independently of service occupancy and waiting-until-start
// queues. Only this call's locally allocated ticket can be released: callers
// cannot supply a copied or foreign ticket to this consumer.
//
// Unknown credit admission blocks the in-order frontier. Missing execution or
// release timing remains explicit; it is never a no-op or a zero-cycle price.
// An exception after allocation leaves that entry held, including when Engine
// rejects the event. Invalid release-stage values refuse before allocation.
// Aggregate completeness requires all submission statuses, Engine::report(),
// and the selected windows' drain stages; Engine does not own these windows.
inline ReservedSubmission submit_reserved(Engine &engine, ReservationWindow &window,
                                          const Event &event, CreditReleaseStage stage) {
  if (stage != CreditReleaseStage::issue && stage != CreditReleaseStage::start &&
      stage != CreditReleaseStage::ready)
    throw std::invalid_argument("unknown reservation release stage");
  const auto frontier = engine.issue_frontier();
  if (!frontier)
    return {ReservedSubmissionStatus::frontier_unknown, {}, {}, {}};
  const auto ticket = window.allocate(std::max(event.earliest, *frontier));
  if (!ticket) {
    engine.advance_issue(std::nullopt);
    return {ReservedSubmissionStatus::credit_unknown, {}, {}, {}};
  }
  engine.advance_issue(ticket->admitted);
  Event admitted = event;
  admitted.earliest = ticket->admitted;
  const auto result = engine.submit(admitted);
  const auto release = stage == CreditReleaseStage::issue ? result.issue
                       : stage == CreditReleaseStage::start ? result.start : result.ready;
  if (release && *release < ticket->admitted)
    throw std::logic_error("reservation release precedes admission");
  window.release(*ticket, release);
  const bool complete = result.unknown == UnknownReason::none && result.issue &&
                        result.start && result.ready && release;
  return {complete ? ReservedSubmissionStatus::complete : ReservedSubmissionStatus::execution_unknown,
          ticket->admitted, result, release};
}

} // namespace merlin_timing
#endif
