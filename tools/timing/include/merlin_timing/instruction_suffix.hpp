#ifndef MERLIN_TIMING_INSTRUCTION_SUFFIX_HPP
#define MERLIN_TIMING_INSTRUCTION_SUFFIX_HPP

#include "instruction_context.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace merlin_timing {

// Immutable image bytes are prospective evidence, never fabricated commits.
// The producer owns reopening the primary ELF and executable-segment identity.
struct ImmutableInstructionV1 {
  std::uint64_t pc = 0, bits = 0;
  std::uint32_t length = 0;
};
struct ImmutableInstructionSuffixV1 {
  std::uint32_t size = sizeof(ImmutableInstructionSuffixV1), version = 1;
  InstructionContextV1 current;
  std::uint32_t count = 0;
  std::array<ImmutableInstructionV1, 32> instructions{};
};
using ObserveImmutableInstructionSuffixV1 = void (*)(void *, const ImmutableInstructionSuffixV1 *);

enum class SuffixRole { permitted, closing_barrier, forbidden };
enum class SuffixProgress { none, pending, closed, invalid };

// A target-specific caller supplies instruction classification. The generic
// evidence tracker grants no target law, queue credit, latency or reservation.
// An armed source claim remains provisional until ALL predicted encodings and
// the actual closing barrier commit in order from the exact primary image.
class InstructionSuffixEvidence {
public:
  template<class Classify>
  bool arm(const ImmutableInstructionSuffixV1 &source, Classify classify) {
    if (pending_ || failed_ || source.size != sizeof(source) || source.version != 1 ||
        !valid_instruction_context(source.current) ||
        source.current.phase != InstructionPhase::committed || !source.count ||
        source.count > source.instructions.size() ||
        source.current.pc > UINT64_MAX - source.current.length ||
        source.current.next_pc != source.current.pc + source.current.length ||
        source.current.sequence > UINT64_MAX - source.count) return false;
    auto pc = source.current.next_pc;
    for (std::size_t i = 0; i < source.count; ++i) {
      const auto &instruction = source.instructions[i];
      if (instruction.pc != pc || (instruction.length != 2 && instruction.length != 4) ||
          (instruction.length == 2 && instruction.bits > UINT16_MAX) ||
          (instruction.length == 4 && instruction.bits > UINT32_MAX) ||
          pc > UINT64_MAX - instruction.length) return false;
      const auto role = classify(instruction);
      if (role == SuffixRole::forbidden ||
          (role == SuffixRole::closing_barrier) != (i + 1 == source.count)) return false;
      pc += instruction.length;
    }
    source_ = source; position_ = 0; pending_ = true;
    return true;
  }
  SuffixProgress observe(const InstructionContextV1 &actual) {
    if (!pending_) return failed_ ? SuffixProgress::invalid : SuffixProgress::none;
    const auto &expected = source_.instructions[position_];
    if (!valid_instruction_context(actual) || actual.phase != InstructionPhase::committed ||
        actual.primary_elf_sha256 != source_.current.primary_elf_sha256 ||
        actual.sequence != source_.current.sequence + position_ + 1 ||
        actual.pc != expected.pc || actual.bits != expected.bits || actual.length != expected.length ||
        actual.next_pc != expected.pc + expected.length) {
      pending_ = false; failed_ = true; return SuffixProgress::invalid;
    }
    if (++position_ == source_.count) { pending_ = false; return SuffixProgress::closed; }
    return SuffixProgress::pending;
  }
  bool pending() const { return pending_; }
  bool failed() const { return failed_; }
private:
  ImmutableInstructionSuffixV1 source_{};
  std::size_t position_ = 0;
  bool pending_ = false, failed_ = false;
};

} // namespace merlin_timing
#endif
