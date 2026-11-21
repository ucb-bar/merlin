#ifndef MERLIN_TIMING_RESERVATIONS_HPP
#define MERLIN_TIMING_RESERVATIONS_HPP

#include "engine.hpp"
#include <optional>
#include <vector>

namespace merlin_timing {

// A reservation entry stays owned through the explicitly selected release
// stage. This is distinct from Engine's waiting-until-start resource queues.
class ReservationWindow {
public:
  struct Ticket { std::size_t slot; std::uint64_t generation; Cycle admitted; };
  explicit ReservationWindow(std::size_t capacity) : entries_(capacity) {
    if (!capacity || capacity > 65536) throw std::invalid_argument("invalid reservation capacity");
  }
  std::optional<Ticket> allocate(Cycle earliest) {
    std::optional<std::size_t> selected;
    std::optional<Cycle> admitted;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const auto &entry = entries_[i];
      if (!entry.owned || entry.release) {
        const auto available = entry.owned ? std::max(earliest, *entry.release) : earliest;
        if (!admitted || available < *admitted) { selected = i; admitted = available; }
      }
    }
    if (!selected) return {};
    auto &entry = entries_[*selected];
    if (entry.generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("reservation generation overflow");
    entry.owned = true; entry.release.reset(); ++entry.generation;
    return Ticket{*selected, entry.generation, *admitted};
  }
  void release(const Ticket &ticket, std::optional<Cycle> when) {
    if (ticket.slot >= entries_.size() || !entries_[ticket.slot].owned ||
        entries_[ticket.slot].generation != ticket.generation ||
        (when && *when < ticket.admitted))
      throw std::logic_error("reservation release does not own its original entry");
    entries_[ticket.slot].release = when;
  }
  std::optional<Cycle> drained(Cycle earliest, std::optional<Ticket> except = {}) const {
    Cycle result = earliest;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const auto &entry = entries_[i];
      if (!entry.owned) continue;
      if (except && except->slot == i && except->generation == entry.generation) continue;
      if (!entry.release) return {};
      result = std::max(result, *entry.release);
    }
    return result;
  }
private:
  struct Entry { bool owned = false; std::optional<Cycle> release; std::uint64_t generation = 0; };
  std::vector<Entry> entries_;
};
} // namespace merlin_timing
#endif
