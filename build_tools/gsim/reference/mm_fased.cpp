#include "mm_fased.h"
#include "fased_native_core.h"
#include <deque>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
struct Read {
  uint64_t id;
  std::vector<uint64_t> words;
  size_t beat=0;
};
struct Write {
  uint64_t address, id, bytes, beats, beat=0;
};
}

struct mm_fased_t::Impl {
  merlin_fased::Core core;
  merlin_fased::Inputs in=merlin_fased::configuration();
  merlin_fased::Outputs out;
  std::map<uint64_t,std::deque<Read>> pending_reads;
  std::map<uint64_t,std::deque<uint64_t>> pending_writes;
  std::deque<Write> writes;
  std::deque<Read> read_egress;
  std::deque<uint64_t> write_egress;
  bool reset_previous=true;
  uint64_t response_word=0;

  void clear() {
    pending_reads.clear(); pending_writes.clear(); writes.clear();
    read_egress.clear(); write_egress.clear(); response_word=0;
  }
  void payload() {
    in.io_egressResp_rBits_data=read_egress.empty()?0:read_egress.front().words.at(read_egress.front().beat);
    in.io_egressResp_rBits_id=read_egress.empty()?0:read_egress.front().id;
    in.io_egressResp_rBits_last=!read_egress.empty() && read_egress.front().beat+1==read_egress.front().words.size();
    in.io_egressResp_bBits_id=write_egress.empty()?0:write_egress.front();
  }
};

mm_fased_t::mm_fased_t(size_t base,size_t size,size_t word,size_t line,backing_data_t& backing,
                       unsigned address_bits,unsigned id_bits):mm_t(base,size,word,line,backing),impl(new Impl) {
  require(word==8 && merlin_fased::data_bits==64 && id_bits>=1 && id_bits<=8 && base%8==0 && size>=8 && size%8==0,"FASED adapter requires an explicitly compiled 64-bit interface");
  require(base==merlin_fased::address_base && size==merlin_fased::address_size &&
          address_bits==merlin_fased::address_bits && id_bits==merlin_fased::id_bits,
          "SimDRAM geometry does not match pinned FASED native configuration");
}
mm_fased_t::~mm_fased_t()=default;
bool mm_fased_t::ar_ready(){return impl->out.io_tNasti_ar_ready;}
bool mm_fased_t::aw_ready(){return impl->out.io_tNasti_aw_ready;}
bool mm_fased_t::w_ready(){return impl->out.io_tNasti_w_ready;}
bool mm_fased_t::b_valid(){return impl->out.io_tNasti_b_valid;}
uint64_t mm_fased_t::b_resp(){return impl->out.io_tNasti_b_bits_resp;}
uint64_t mm_fased_t::b_id(){return impl->out.io_tNasti_b_bits_id;}
bool mm_fased_t::r_valid(){return impl->out.io_tNasti_r_valid;}
uint64_t mm_fased_t::r_resp(){return impl->out.io_tNasti_r_bits_resp;}
uint64_t mm_fased_t::r_id(){return impl->out.io_tNasti_r_bits_id;}
void* mm_fased_t::r_data(){return &impl->response_word;}
bool mm_fased_t::r_last(){return impl->out.io_tNasti_r_bits_last;}

void mm_fased_t::tick(bool reset,bool arv,uint64_t ara,uint64_t ari,uint64_t ars,uint64_t arl,
                      bool awv,uint64_t awa,uint64_t awi,uint64_t aws,uint64_t awl,
                      bool wv,uint64_t strb,void* raw,bool last,bool rr,bool br) {
  auto& s=*impl; auto& in=s.in;
  auto validate=[&](uint64_t address,uint64_t id,uint64_t size,uint64_t length,uint64_t maximum) {
    require(size<=3 && length<256 && id<(1ULL<<merlin_fased::id_bits),"unsupported AXI size, length or ID");
    uint64_t bytes=1ULL<<size, total=bytes*(length+1);
    require(total<=maximum && address%bytes==0 && address>=mem_base &&
            address-mem_base<=mem_size && total<=mem_size-(address-mem_base),"AXI request outside compiled memory/transfer contract");
    require((address&4095)+total<=4096,"AXI incrementing burst crosses 4KiB boundary");
  };
  if (!reset && arv) validate(ara,ari,ars,arl,merlin_fased::max_read_bytes);
  if (!reset && awv) validate(awa,awi,aws,awl,merlin_fased::max_write_bytes);
  require(strb<=255,"AXI write strobes exceed bus width");
  in.reset=reset;
  // SimDRAM advertises zero readiness during reset. The first release tick
  // cannot consume a request not previously acknowledged to the target.
  in.io_tNasti_ar_valid=!reset && !s.reset_previous && arv;
  in.io_tNasti_ar_bits_addr=ara;in.io_tNasti_ar_bits_id=ari;
  in.io_tNasti_ar_bits_size=ars;in.io_tNasti_ar_bits_len=arl;in.io_tNasti_ar_bits_burst=1;
  in.io_tNasti_aw_valid=!reset && !s.reset_previous && awv;
  in.io_tNasti_aw_bits_addr=awa;in.io_tNasti_aw_bits_id=awi;
  in.io_tNasti_aw_bits_size=aws;in.io_tNasti_aw_bits_len=awl;in.io_tNasti_aw_bits_burst=1;
  uint64_t word=0;std::memcpy(&word,raw,8);
  in.io_tNasti_w_valid=!reset && !s.reset_previous && wv;
  in.io_tNasti_w_bits_data=word;in.io_tNasti_w_bits_strb=strb;in.io_tNasti_w_bits_last=last;
  in.io_tNasti_r_ready=!reset && rr;in.io_tNasti_b_ready=!reset && br;
  s.payload();
  auto before=s.core.evaluate(in,false);
  if (!reset && !s.reset_previous) {
    require(before.io_tNasti_ar_ready==s.out.io_tNasti_ar_ready && before.io_tNasti_aw_ready==s.out.io_tNasti_aw_ready &&
            before.io_tNasti_w_ready==s.out.io_tNasti_w_ready,"FASED ready depends on new live input; registered SimDRAM boundary is incompatible");
    require(before.io_tNasti_r_valid==s.out.io_tNasti_r_valid && before.io_tNasti_b_valid==s.out.io_tNasti_b_valid,
            "FASED response phase differs from advertised SimDRAM response");
    if (before.io_tNasti_r_valid) require(before.io_tNasti_r_bits_id==s.out.io_tNasti_r_bits_id &&
        before.io_tNasti_r_bits_data==s.out.io_tNasti_r_bits_data && before.io_tNasti_r_bits_last==s.out.io_tNasti_r_bits_last &&
        before.io_tNasti_r_bits_resp==s.out.io_tNasti_r_bits_resp,"FASED read payload differs from advertised response");
    if (before.io_tNasti_b_valid) require(before.io_tNasti_b_bits_id==s.out.io_tNasti_b_bits_id &&
        before.io_tNasti_b_bits_resp==s.out.io_tNasti_b_bits_resp,"FASED write payload differs from advertised response");
  }
  if (reset) s.clear();
  else {
    if (before.io_tNasti_r_valid && rr) {
      require(!s.read_egress.empty(),"FASED consumed absent functional read response");
      auto& response=s.read_egress.front();
      if (++response.beat==response.words.size()) s.read_egress.pop_front();
    }
    if (before.io_tNasti_b_valid && br) {
      require(!s.write_egress.empty(),"FASED consumed absent functional write response");
      s.write_egress.pop_front();
    }
    if (before.io_tNasti_ar_ready && in.io_tNasti_ar_valid) {
      Read response{ari,{}};
      for (uint64_t beat=0;beat<=arl;++beat) {
        uint64_t addr=(ara+beat*(1ULL<<ars))&~7ULL;
        require(addr>=mem_base && addr-mem_base<=mem_size-8,"read lane outside backing mapping");
        uint64_t value;std::memcpy(&value,data+addr-mem_base,8);response.words.push_back(value);
      }
      s.pending_reads[ari].push_back(std::move(response));
    }
    if (before.io_tNasti_aw_ready && in.io_tNasti_aw_valid) s.writes.push_back({awa,awi,1ULL<<aws,awl+1});
    if (before.io_tNasti_w_ready && in.io_tNasti_w_valid) {
      require(!s.writes.empty(),"FASED accepted W without an accepted AW");
      auto& request=s.writes.front();uint64_t addr=request.address+request.beat*request.bytes;
      uint64_t allowed=((1ULL<<request.bytes)-1)<<(addr&7);
      require(!(strb&~allowed),"write strobes exceed addressed byte lanes");
      require(last==(request.beat+1==request.beats),"WLAST disagrees with accepted AWLEN");
      uint64_t aligned=addr&~7ULL;
      for (unsigned byte=0;byte<8;++byte) if (strb&(1ULL<<byte)) data[aligned-mem_base+byte]=(word>>(8*byte))&255;
      if (++request.beat==request.beats) {s.pending_writes[request.id].push_back(request.id);s.writes.pop_front();}
    }
    if (before.io_egressReq_r_valid) {
      auto found=s.pending_reads.find(before.io_egressReq_r_bits);
      require(found!=s.pending_reads.end()&&!found->second.empty(),"FASED selected unknown read ID");
      s.read_egress.push_back(std::move(found->second.front()));found->second.pop_front();
    }
    if (before.io_egressReq_b_valid) {
      auto found=s.pending_writes.find(before.io_egressReq_b_bits);
      require(found!=s.pending_writes.end()&&!found->second.empty(),"FASED selected write before data completion");
      s.write_egress.push_back(found->second.front());found->second.pop_front();
    }
  }
  s.core.evaluate(in,true);
  s.payload();s.out=s.core.evaluate(in,true);
  if (reset) s.out={};
  s.response_word=s.out.io_tNasti_r_bits_data;
  s.reset_previous=reset;
}
