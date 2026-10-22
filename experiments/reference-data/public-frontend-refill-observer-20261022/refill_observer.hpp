#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
extern bool gsim_done;
extern int gsim_exit_code;
// Read-only access to current model values; never reads or writes $NEXT.
class RefillRecorder {
  std::string path;
  std::vector<std::string> rows;
  uint64_t events=0, bytes=0, dropped=0, ps=0, period=0;
  bool previous_core_rising=false, previous_l2_rising=false;
  unsigned occurrence[2]={0,0};
  int current[2]={-1,-1};
  static constexpr uint64_t max_events=65536, max_bytes=32*1024*1024;
  static unsigned long long U(uint64_t x){return static_cast<unsigned long long>(x);}
  void record(const char* domain,const char* kind,const std::string& body){
    ++events;
    char prefix[320];
    int n=snprintf(prefix,sizeof(prefix),"{\"seq\":%llu,\"time_ps\":%llu,\"reference_cycle\":%llu,\"clock_domain\":\"%s\",\"phase\":\"post_step_rising_pre_next_commit\",\"kind\":\"%s\"",U(events),U(ps),U(ps/period),domain,kind);
    std::string row(prefix,n);row+=body;row+="}\n";
    if(rows.size()>=max_events || bytes+row.size()>max_bytes){++dropped;return;}
    bytes+=row.size();rows.push_back(std::move(row));
  }
public:
  RefillRecorder(int argc,char**argv){
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a.rfind("+refill-trace=",0)==0){if(!path.empty())throw std::runtime_error("duplicate refill trace path");path=a.substr(14);}}
    if(path!="stdout")throw std::runtime_error("diagnostic requires +refill-trace=stdout");
    rows.reserve(4096);
  }
  ~RefillRecorder(){
    std::ostream& out=std::cout;
    static constexpr const char* tag="REFILL_EVENT_JSON ";
    out<<tag<<"{\"schema\":\"operator.frontend_refill_events.v1\",\"kind\":\"header\",\"original_engine_sha256\":\"76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651\",\"firrtl_sha256\":\"6748b9756799c086a6fb06e6d80b4b052f2771f9eb1858c172e14dcf441ba149\",\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"line_bytes\":64,\"icache_beat_bytes\":16,\"icache_beat_count\":4,\"l2_sets\":1024,\"address_mapping\":\"line_address=(tag<<16)|(set<<6)\",\"phase\":\"post_step_rising_pre_next_commit\",\"max_events\":65536,\"max_record_bytes\":33554432,\"filter\":\"all events from reset; no pre-window discard\",\"scope_index_missing\":-1,\"scope_bindings\":[{\"begin_pc\":2147485504,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147485704,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":0,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":0},{\"begin_pc\":2147485504,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147485704,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":2,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":1},{\"begin_pc\":2147485504,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147485704,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":3,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":2},{\"begin_pc\":2147485504,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147485704,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":5,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":3},{\"begin_pc\":2147485504,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147485704,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":6,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":4},{\"begin_pc\":2147485504,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147485704,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":8,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":5},{\"begin_pc\":2147485868,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147486068,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":0,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":6},{\"begin_pc\":2147485868,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147486068,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":2,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":7},{\"begin_pc\":2147485868,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147486068,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":3,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":8},{\"begin_pc\":2147485868,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147486068,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":5,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":9},{\"begin_pc\":2147485868,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147486068,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":6,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":10},{\"begin_pc\":2147485868,\"elf_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"end_pc\":2147486068,\"evidence_sha256\":\"83560d424ded692f9117eb01d1b7f296e6ac4e6b5c035c0e4319bdc3b61dc046\",\"input_sha256\":\"cbb7cb1e1810180f1af592ed851a8fbb7a73aae8441e3d9ccfab2f886d9ba86e\",\"occurrence\":8,\"schema\":\"merlin.spike_scope_binding.v1\",\"scope_sha256\":\"82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525\",\"scope_index\":11}],\"reference_period_ps\":"<<period<<"}\n";
    for(const auto&r:rows)out<<tag<<r;
    out<<tag<<"{\"kind\":\"footer\",\"observed_events\":"<<events<<",\"written_events\":"<<rows.size()<<",\"dropped\":"<<dropped<<",\"truncated\":"<<(dropped?"true":"false")<<",\"simulator_done\":"<<(gsim_done?"true":"false")<<",\"exit_code\":"<<gsim_exit_code<<",\"record_bytes\":"<<bytes<<"}\n";
    out.flush();if(!out)fprintf(stderr,"[refill-trace] write failed\n");
  }
  void sample(const STestHarness& d,uint64_t time,uint64_t reference_period){
    ps=time;period=reference_period;char b[1024];
    if(d.__gsim_clock_304748$rising && !previous_core_rising){
      if(d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$core$wb_valid){
        uint64_t pc=d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc;
        const uint64_t begin[2]={2147485504ULL,2147485868ULL}, end[2]={2147485704ULL,2147486068ULL};
        const int selected[9]={0,-1,1,2,-1,3,4,-1,5};
        for(int layout=0;layout<2;++layout){
          if(pc==begin[layout]||pc==end[layout]){
            bool start=pc==begin[layout];
            if(start)current[layout]=int(occurrence[layout]++);
            int occ=current[layout];int scope=(occ>=0&&occ<9)?selected[occ]:-1;if(scope>=0)scope+=layout*6;
            snprintf(b,sizeof(b),",\"pc\":%llu,\"inst\":%llu,\"wb_valid\":true,\"marker\":\"%s\",\"layout\":%d,\"occurrence\":%d,\"scope_index\":%d",U(pc),U(d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_inst),start?"begin":"end",layout?44:0,occ,scope);
            record("core","scope_marker",b);
          }
        }
      }
      if(d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$frontend$icache$_refill_fire_T){
        uint64_t a=d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$frontend$icache$refill_paddr&~uint64_t(63);
        snprintf(b,sizeof(b),",\"address\":%llu,\"line_address\":%llu,\"source\":0,\"source_basis\":\"selected_local_source0\",\"size_log2\":6,\"beat_bytes\":16,\"beat_count\":4,\"handshake\":true",U(a),U(a));record("core","icache_get",b);
      }
      if(d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$frontend$icache$refill_one_beat){
        snprintf(b,sizeof(b),",\"refill_address\":%llu,\"source\":0,\"source_basis\":\"selected_local_source0\",\"beat_counter\":%llu,\"last\":%s,\"handshake\":true",U(d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$frontend$icache$refill_paddr),U(d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$frontend$icache$refill_cnt),d.chiptop0$system$tile_prci_domain$element_reset_domain$rockettile$frontend$icache$refill_done?"true":"false");record("core","icache_d",b);
      }
    }
    if(d.__gsim_clock_37682$rising && !previous_l2_rising){
      if(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$directory$ren1){
        snprintf(b,sizeof(b),",\"set\":%llu,\"tag\":%llu,\"hit\":%llu,\"state\":%llu,\"way\":%llu",U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$directory$set),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$directory$tag),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$directory$_io_result_bits_hit_T_3),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$directory$_view__T_139$$state),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$directory$_io_result_bits_way_T_9));record("l2","directory_result",b);
      }
      if(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s1_valid_r && d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$bankedStore$_sourceD_rreq_io_sourceD_radr_ready_T_1){
        snprintf(b,sizeof(b),",\"set\":%llu,\"way\":%llu,\"beat\":%llu,\"handshake\":true",U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s1_req$$set),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s1_req$$way),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s1_beat));record("l2","bank_read",b);
      }
      if(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_valid_d){
        snprintf(b,sizeof(b),",\"source\":%llu,\"sink\":%llu,\"set\":%llu,\"way\":%llu,\"request_opcode\":%llu,\"response_opcode\":%llu,\"size_log2\":%llu,\"beat\":%llu,\"ready\":%s,\"handshake\":%s",U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_req$$source),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_req$$sink),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_req$$set),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_req$$way),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_req$$opcode),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$_d_bits_opcode_T),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_req$$size),U(d.chiptop0$system$coh_wrapper$l2$inclusive_cache_bank_sched$sourceD$s3_beat),d.chiptop0$system$coh_wrapper$InclusiveCache_inner_TLBuffer$nodeIn_d_q$_empty_T?"true":"false",d.chiptop0$system$coh_wrapper$InclusiveCache_inner_TLBuffer$nodeIn_d_q$_empty_T?"true":"false");record("l2","l2_d",b);
      }
    }
    previous_core_rising=d.__gsim_clock_304748$rising; previous_l2_rising=d.__gsim_clock_37682$rising;
  }
};
