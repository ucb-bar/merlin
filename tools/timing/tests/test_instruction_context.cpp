#include "merlin_timing/instruction_context.hpp"
#include <cassert>
#include <algorithm>

using namespace merlin_timing;
InstructionContextV1 context(std::uint64_t sequence, std::uint64_t pc, std::uint64_t bits) {
  InstructionContextV1 value;
  value.sequence=sequence; value.pc=pc; value.bits=bits; value.length=4; value.next_pc=pc+4;
  std::fill_n(value.primary_elf_sha256.begin(),64,'a');
  return value;
}
// This fixture's ISA interpretation belongs to the consumer, not the helper.
bool full_fence(const InstructionNeighbor &n) {
  return n.adjacent() && n.following.length==4 && n.following.bits==0x0ff0000f;
}
int main() {
  PostedInstructionEvidence evidence;
  const auto command=context(10,0x100,0x0800307b);
  assert(evidence.arm(command,true,true));
  assert(!evidence.arm(command,true,true)); // exactly one evidence record
  auto fence=evidence.observe(context(11,0x104,0x0ff0000f));
  assert(fence && full_fence(*fence) && fence->command.pc==0x100 && !evidence.pending());
  assert(!evidence.observe(context(12,0x108,0x0ff0000f)));
  assert(evidence.arm(command,true,true));
  auto next_compute=evidence.observe(context(11,0x104,0x0800307b));
  assert(next_compute && next_compute->adjacent() && !full_fence(*next_compute));
  assert(evidence.arm(command,true,true));
  auto dependent_scalar=evidence.observe(context(11,0x104,0x00150513));
  assert(dependent_scalar && !full_fence(*dependent_scalar));
  assert(evidence.arm(command,true,false));
  auto dirty=evidence.observe(context(11,0x104,0x0ff0000f));
  assert(dirty && dirty->reason==NeighborReason::dirty_epoch && !full_fence(*dirty));
  assert(!evidence.arm(command,false,true)); // no result-bearing command
  for (int mutation=0;mutation<5;++mutation) {
    assert(evidence.arm(command,true,true));
    auto next=context(11,0x104,0x0ff0000f);
    if(mutation==0)next.sequence=12;
    if(mutation==1)next.pc=0x108;
    if(mutation==2)next.primary_elf_sha256[0]='b';
    if(mutation==3)next.phase=InstructionPhase::aborted;
    if(mutation==4)next.version=2;
    auto result=evidence.observe(next);
    assert(result && !full_fence(*result) && !evidence.pending());
  }
  assert(evidence.arm(command,true,true));
  auto done=context(11,0x104,0);done.phase=InstructionPhase::before_report;
  auto unfinished=evidence.observe(done);
  assert(unfinished && unfinished->reason==NeighborReason::interrupted);
}
