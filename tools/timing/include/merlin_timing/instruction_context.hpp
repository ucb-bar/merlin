#ifndef MERLIN_TIMING_INSTRUCTION_CONTEXT_HPP
#define MERLIN_TIMING_INSTRUCTION_CONTEXT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace merlin_timing {

// Optional side-channel ABI. It does not extend an existing provider vtable or
// service aggregate, grant a hardware law, or change any timing reservation.
enum class InstructionPhase : std::uint32_t { committed = 1, aborted = 2, before_report = 3 };
struct InstructionContextV1 {
  std::uint32_t size = sizeof(InstructionContextV1);
  std::uint32_t version = 1;
  InstructionPhase phase = InstructionPhase::committed;
  std::uint32_t length = 0;
  std::uint64_t sequence = 0, pc = 0, bits = 0, next_pc = 0;
  std::array<char, 65> primary_elf_sha256{};
};
using ObserveInstructionContextV1 = void (*)(void *, const InstructionContextV1 *);

inline bool valid_instruction_context(const InstructionContextV1 &value) {
  if (value.size != sizeof(InstructionContextV1) || value.version != 1 ||
      value.primary_elf_sha256[64] != '\0') return false;
  for (std::size_t i = 0; i < 64; ++i) {
    const char c = value.primary_elf_sha256[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  if (value.phase != InstructionPhase::committed)
    return value.phase == InstructionPhase::aborted || value.phase == InstructionPhase::before_report;
  return value.sequence && (value.length == 2 || value.length == 4);
}

enum class NeighborReason { adjacent, nonadjacent, different_image, dirty_epoch, interrupted, invalid };
struct InstructionNeighbor {
  InstructionContextV1 command, following;
  NeighborReason reason;
  bool adjacent() const { return reason == NeighborReason::adjacent; }
};

// Bounded *evidence* window, not command/timing dispatch. A target may inspect
// the following actual encoding using its own ISA and hardware law. No event
// is held, replayed, shortened or priced by this class.
class PostedInstructionEvidence {
public:
  bool arm(const InstructionContextV1 &command, bool resultless, bool clean_epoch) {
    if (pending_ || !resultless || command.phase != InstructionPhase::committed ||
        !valid_instruction_context(command)) return false;
    pending_ = Pending{command, clean_epoch};
    return true;
  }
  std::optional<InstructionNeighbor> observe(const InstructionContextV1 &following) {
    if (!pending_) return {};
    const auto prior = *pending_;
    pending_.reset();
    auto reason = NeighborReason::adjacent;
    if (!valid_instruction_context(following)) reason = NeighborReason::invalid;
    else if (following.phase != InstructionPhase::committed) reason = NeighborReason::interrupted;
    else if (prior.command.primary_elf_sha256 != following.primary_elf_sha256)
      reason = NeighborReason::different_image;
    else if (!prior.clean_epoch) reason = NeighborReason::dirty_epoch;
    else if (prior.command.sequence == UINT64_MAX ||
             following.sequence != prior.command.sequence + 1 ||
             prior.command.pc > UINT64_MAX - prior.command.length ||
             following.pc != prior.command.pc + prior.command.length ||
             prior.command.next_pc != following.pc)
      reason = NeighborReason::nonadjacent;
    return InstructionNeighbor{prior.command, following, reason};
  }
  bool pending() const { return bool(pending_); }
private:
  struct Pending { InstructionContextV1 command; bool clean_epoch; };
  std::optional<Pending> pending_;
};

} // namespace merlin_timing
#endif
