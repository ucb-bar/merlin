#include <merlin_timing/instruction_suffix.hpp>
#include <cassert>
#include <iostream>
using namespace merlin_timing;
InstructionContextV1 context(std::uint64_t seq,std::uint64_t pc,std::uint64_t bits,unsigned length=4){
 InstructionContextV1 c;c.sequence=seq;c.pc=pc;c.bits=bits;c.length=length;c.next_pc=pc+length;
 for(unsigned i=0;i<64;++i)c.primary_elf_sha256[i]='a';
 return c;
}
ImmutableInstructionSuffixV1 suffix(){
 ImmutableInstructionSuffixV1 s;s.current=context(1,100,11);s.count=3;
 s.instructions[0]={104,1,2};s.instructions[1]={106,2,4};s.instructions[2]={110,3,4};return s;
}
SuffixRole role(const ImmutableInstructionV1&i){return i.bits==3?SuffixRole::closing_barrier:i.bits==99?SuffixRole::forbidden:SuffixRole::permitted;}
int main(){
 auto s=suffix();InstructionSuffixEvidence e;assert(e.arm(s,role));assert(!e.arm(s,role));
 assert(e.observe(context(2,104,1,2))==SuffixProgress::pending);
 assert(e.observe(context(3,106,2))==SuffixProgress::pending);
 assert(e.observe(context(4,110,3))==SuffixProgress::closed);assert(!e.pending()&&!e.failed());
 for(unsigned mutation=0;mutation<7;++mutation){
  InstructionSuffixEvidence v;assert(v.arm(s,role));auto actual=context(2,104,1,2);
  if(mutation==0)actual.pc+=2;
  if(mutation==1)++actual.sequence;
  if(mutation==2)actual.bits=9;
  if(mutation==3)actual.primary_elf_sha256[1]='b';
  if(mutation==4)actual.next_pc+=2;
  if(mutation==5)actual.phase=InstructionPhase::aborted;
  if(mutation==6)actual.phase=InstructionPhase::before_report;
  assert(v.observe(actual)==SuffixProgress::invalid);assert(v.failed()&&!v.arm(s,role));
 }
 for(unsigned mutation=0;mutation<6;++mutation){
  auto bad=s;InstructionSuffixEvidence v;
  if(mutation==0)bad.instructions[0].bits=99;
  if(mutation==1)bad.instructions[0].bits=3;
  if(mutation==2)bad.instructions[2].bits=2;
  if(mutation==3)bad.instructions[1].pc+=2;
  if(mutation==4)bad.count=33;
  if(mutation==5)bad.current.next_pc+=2;
  assert(!v.arm(bad,role));
 }
 std::cout<<"immutable suffix is provisional until same-image exact commits and barrier; interrupted/mutated/unsupported paths denied\n";
}
