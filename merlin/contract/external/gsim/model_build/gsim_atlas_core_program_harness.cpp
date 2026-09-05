// GSIM program-driven oracle for the atlas NPU (top AtlasCore).
//
// Faithful port of the Verilator oracle sim_main_prog.cpp (vsim_oracle/) to the
// GSIM-generated C++ model, so the SAME spec JSON runs on both and the cycle
// counts are directly comparable.
//
// SEMANTIC NOTE (the one real difference): Verilator exposes settle() = eval()
// with no clock edge, and tick() = one clock edge. GSIM exposes ONLY step(),
// which evaluates combinational logic AND commits registers in one call. So a
// combinational output (e.g. TileLink a_ready) read after step() reflects the
// NEW register state, whereas Verilator's settle() reads it off the OLD state.
// Handshake-dependent phases can therefore differ by ~1 cycle per fire. The
// run-loop below keeps the same structure so the comparison is apples-to-apples
// up to that documented offset.
//
// Spec JSON (argv[1]) and stdout JSON are byte-identical in schema to the
// Verilator harness, plus optional per-cycle observability (argv[2] = trace csv).
#include "AtlasCore.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>

// atlas-specific constants -- copied verbatim from sim_main_prog.cpp so the two
// harnesses cannot drift.
static const uint32_t IMEM_BASE   = 0x20000;
static const uint32_t START_CSR   = 0x18;
static const uint32_t DRAM_WINDOW = 1u << 20;   // wrapping window (see report)
static const uint32_t BEAT_BYTES  = 32;
static const int      MAX_WAIT    = 5000;

static const int TL_PUT_FULL        = 0;
static const int TL_GET             = 4;
static const int TL_ACCESS_ACK      = 0;
static const int TL_ACCESS_ACK_DATA = 1;

typedef unsigned _BitInt(256) u256;

// --------------------------------------------------------------------------
struct Json {
    const char* p; const char* end;
    explicit Json(const std::string& s) : p(s.data()), end(s.data()+s.size()) {}
    void ws(){ while(p<end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r'||*p==','))++p; }
    bool eat(char c){ ws(); if(p<end && *p==c){++p;return true;} return false; }
    char peekc(){ ws(); return p<end?*p:'\0'; }
    uint64_t num(){ ws(); uint64_t v=0; bool neg=false;
        if(p<end&&*p=='-'){neg=true;++p;}
        while(p<end&&*p>='0'&&*p<='9'){v=v*10+(uint64_t)(*p-'0');++p;}
        return neg?(uint64_t)(-(int64_t)v):v; }
    std::string str(){ ws(); std::string o;
        if(p<end&&*p=='"'){++p; while(p<end&&*p!='"')o.push_back(*p++); if(p<end)++p;}
        return o; }
    std::string key(){ return str(); }
};
static std::vector<uint8_t> hex_to_bytes(const std::string& h){
    std::vector<uint8_t> o;
    auto nib=[](char c)->int{ if(c>='0'&&c<='9')return c-'0';
        if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
    for(size_t i=0;i+1<h.size();i+=2) o.push_back((uint8_t)((nib(h[i])<<4)|nib(h[i+1])));
    return o;
}
static std::string bytes_to_hex(const uint8_t* b,size_t n){
    static const char* d="0123456789abcdef"; std::string o; o.reserve(n*2);
    for(size_t i=0;i<n;++i){ o.push_back(d[b[i]>>4]); o.push_back(d[b[i]&0xF]); }
    return o;
}

// --------------------------------------------------------------------------
static SAtlasCore* dut = nullptr;
static inline void tick(){ dut->step(); }

static void reset_core(int cycles=12){
    dut->set_reset(1);
    for(int i=0;i<cycles;++i) tick();
    dut->set_reset(0);
}

static bool tl_put_imem(uint32_t address, uint32_t data, int size){
    dut->set_io$$imemTL$$a$$bits$$opcode(TL_PUT_FULL);
    dut->set_io$$imemTL$$a$$bits$$param(0);
    dut->set_io$$imemTL$$a$$bits$$size(size);
    dut->set_io$$imemTL$$a$$bits$$source(0);
    dut->set_io$$imemTL$$a$$bits$$address(address);
    dut->set_io$$imemTL$$a$$bits$$mask(0xF);
    dut->set_io$$imemTL$$a$$bits$$data(data);
    dut->set_io$$imemTL$$d$$ready(1);
    dut->set_io$$imemTL$$a$$valid(1);
    bool accepted=false;
    for(int i=0;i<MAX_WAIT;++i){
        tick();
        if(dut->get_io$$imemTL$$a$$ready()==1){ accepted=true; break; }
    }
    dut->set_io$$imemTL$$a$$valid(0);
    tick();
    return accepted;
}

static bool tl_put_csr(uint32_t address, uint32_t data, int size){
    dut->set_io$$csrTL$$a$$bits$$opcode(TL_PUT_FULL);
    dut->set_io$$csrTL$$a$$bits$$param(0);
    dut->set_io$$csrTL$$a$$bits$$size(size);
    dut->set_io$$csrTL$$a$$bits$$source(0);
    dut->set_io$$csrTL$$a$$bits$$address(address);
    dut->set_io$$csrTL$$a$$bits$$mask(0xF);
    dut->set_io$$csrTL$$a$$bits$$data(data);
    dut->set_io$$csrTL$$d$$ready(1);
    dut->set_io$$csrTL$$a$$valid(1);
    bool accepted=false;
    for(int i=0;i<MAX_WAIT;++i){
        tick();
        if(dut->get_io$$csrTL$$a$$ready()==1){ accepted=true; break; }
    }
    dut->set_io$$csrTL$$a$$valid(0);
    tick();
    return accepted;
}

// --------------------------------------------------------------------------
struct Response { uint8_t op; uint32_t size; uint32_t src; uint8_t data[BEAT_BYTES]; };

struct DramSlave {
    std::vector<uint8_t> mem;
    uint32_t mask;
    std::deque<Response> q;
    long reads=0, writes=0;
    long wrap_hits=0;              // instrumentation the Verilator harness lacks
    uint32_t raw_min=0xFFFFFFFFu, raw_max=0; long alias_collisions=0;
    std::vector<uint32_t> seen_wrapped; // (addr&mask) -> first raw seen, for true-aliasing detection
    std::vector<uint32_t> seen_raw;
    DramSlave() : mem(DRAM_WINDOW,0), mask(DRAM_WINDOW-1) {}

    void preload(uint32_t addr,const std::vector<uint8_t>& d){
        if(addr >= DRAM_WINDOW) wrap_hits++;
        uint32_t a=addr&mask;
        for(size_t i=0;i<d.size() && (a+i)<mem.size();++i) mem[a+i]=d[i];
    }
    std::vector<uint8_t> captured(uint32_t addr,size_t n){
        uint32_t a=addr&mask; std::vector<uint8_t> o(n,0);
        for(size_t i=0;i<n&&(a+i)<mem.size();++i) o[i]=mem[a+i];
        return o;
    }
    static void poke_d(const uint8_t* b){
        u256 v=0; for(int k=31;k>=0;--k){ v=(v<<8)|(u256)b[k]; }
        dut->set_io$$dmaTL$$d$$bits$$data(v);
    }
    static void peek_a(uint8_t* b){
        u256 v=dut->get_io$$dmaTL$$a$$bits$$data();
        for(int k=0;k<32;++k){ b[k]=(uint8_t)(uint64_t)(v & (u256)0xFF); v>>=8; }
    }
    // Drive slave outputs for this cycle. Must be called BEFORE tick().
    void drive(){
        if(!q.empty()){
            Response& f=q.front();
            dut->set_io$$dmaTL$$d$$bits$$opcode(f.op);
            dut->set_io$$dmaTL$$d$$bits$$size(f.size);
            dut->set_io$$dmaTL$$d$$bits$$source(f.src);
            poke_d(f.data);
            dut->set_io$$dmaTL$$d$$bits$$param(0);
            dut->set_io$$dmaTL$$d$$bits$$sink(0);
            dut->set_io$$dmaTL$$d$$bits$$denied(0);
            dut->set_io$$dmaTL$$d$$bits$$corrupt(0);
            dut->set_io$$dmaTL$$d$$valid(1);
        } else {
            dut->set_io$$dmaTL$$d$$valid(0);
        }
        dut->set_io$$dmaTL$$a$$ready(1);
    }
    // Account fires for this cycle. Must be called AFTER tick().
    void account(){
        bool have=!q.empty();
        bool d_fire = have && (dut->get_io$$dmaTL$$d$$ready()==1);
        bool a_fire = (dut->get_io$$dmaTL$$a$$valid()==1);
        if(d_fire) q.pop_front();
        if(a_fire){
            int op=dut->get_io$$dmaTL$$a$$bits$$opcode();
            uint32_t size=dut->get_io$$dmaTL$$a$$bits$$size();
            uint32_t src =dut->get_io$$dmaTL$$a$$bits$$source();
            uint32_t raw =dut->get_io$$dmaTL$$a$$bits$$address();
            if(raw < raw_min) raw_min=raw;
            if(raw > raw_max) raw_max=raw;
            if(raw >= DRAM_WINDOW) wrap_hits++;
            uint32_t addr=raw & mask;
            { bool found=false;
              for(size_t z=0;z<seen_wrapped.size();++z)
                if(seen_wrapped[z]==addr){ if(seen_raw[z]!=raw) alias_collisions++; found=true; break; }
              if(!found){ seen_wrapped.push_back(addr); seen_raw.push_back(raw); } }
            uint32_t nbytes=1u<<size;
            if(op==TL_GET){
                Response r; r.op=TL_ACCESS_ACK_DATA; r.size=size; r.src=src;
                for(uint32_t i=0;i<BEAT_BYTES;++i)
                    r.data[i]=(i<nbytes && (addr+i)<mem.size())?mem[addr+i]:0;
                q.push_back(r); reads++;
            } else {
                uint8_t db[BEAT_BYTES]; peek_a(db);
                uint32_t m=dut->get_io$$dmaTL$$a$$bits$$mask();
                for(uint32_t i=0;i<nbytes&&i<32;++i)
                    if(m&(1u<<i)){ if((addr+i)<mem.size()) mem[addr+i]=db[i]; }
                Response r; r.op=TL_ACCESS_ACK; r.size=size; r.src=src;
                std::memset(r.data,0,BEAT_BYTES);
                q.push_back(r); writes++;
            }
        }
    }
};

static void zero_inputs(){
    dut->set_io$$imemTL$$a$$valid(0); dut->set_io$$imemTL$$a$$bits$$opcode(0);
    dut->set_io$$imemTL$$a$$bits$$param(0); dut->set_io$$imemTL$$a$$bits$$size(0);
    dut->set_io$$imemTL$$a$$bits$$source(0); dut->set_io$$imemTL$$a$$bits$$mask(0);
    dut->set_io$$imemTL$$a$$bits$$corrupt(0); dut->set_io$$imemTL$$a$$bits$$address(0);
    dut->set_io$$imemTL$$a$$bits$$data(0); dut->set_io$$imemTL$$d$$ready(0);
    dut->set_io$$csrTL$$a$$valid(0); dut->set_io$$csrTL$$a$$bits$$opcode(0);
    dut->set_io$$csrTL$$a$$bits$$param(0); dut->set_io$$csrTL$$a$$bits$$size(0);
    dut->set_io$$csrTL$$a$$bits$$source(0); dut->set_io$$csrTL$$a$$bits$$mask(0);
    dut->set_io$$csrTL$$a$$bits$$corrupt(0); dut->set_io$$csrTL$$a$$bits$$address(0);
    dut->set_io$$csrTL$$a$$bits$$data(0); dut->set_io$$csrTL$$d$$ready(0);
    dut->set_io$$dmaTL$$a$$ready(0); dut->set_io$$dmaTL$$d$$valid(0);
    dut->set_io$$dmaTL$$d$$bits$$opcode(0); dut->set_io$$dmaTL$$d$$bits$$param(0);
    dut->set_io$$dmaTL$$d$$bits$$size(0); dut->set_io$$dmaTL$$d$$bits$$source(0);
    dut->set_io$$dmaTL$$d$$bits$$sink(0); dut->set_io$$dmaTL$$d$$bits$$denied(0);
    dut->set_io$$dmaTL$$d$$bits$$corrupt(0); dut->set_io$$dmaTL$$d$$bits$$data((u256)0);
    dut->set_io$$vmemTL$$a$$valid(0); dut->set_io$$vmemTL$$d$$ready(0);
    dut->set_reset(0);
}

int main(int argc,char** argv){
    if(argc<2){ fprintf(stderr,"usage: %s spec.json [trace.csv]\n",argv[0]); return 2; }
    std::ifstream f(argv[1]);
    if(!f){ fprintf(stderr,"cannot open spec %s\n",argv[1]); return 2; }
    std::stringstream ss; ss<<f.rdbuf(); std::string text=ss.str();

    std::vector<uint32_t> words;
    std::vector<std::pair<uint32_t,std::vector<uint8_t>>> preload;
    std::vector<std::pair<uint32_t,uint32_t>> reads;
    uint32_t max_cycles=20000;

    Json j(text); j.eat('{');
    while(j.peekc()!='}'&&j.peekc()!='\0'){
        std::string k=j.key(); j.eat(':');
        if(k=="words"){ j.eat('['); while(j.peekc()!=']'&&j.peekc()!='\0') words.push_back((uint32_t)j.num()); j.eat(']'); }
        else if(k=="preload"){ j.eat('[');
            while(j.peekc()=='['){ j.eat('['); uint32_t a=(uint32_t)j.num(); std::string h=j.str(); j.eat(']');
                preload.emplace_back(a,hex_to_bytes(h)); }
            j.eat(']'); }
        else if(k=="reads"){ j.eat('[');
            while(j.peekc()=='['){ j.eat('['); uint32_t a=(uint32_t)j.num(); uint32_t n=(uint32_t)j.num(); j.eat(']');
                reads.emplace_back(a,n); }
            j.eat(']'); }
        else if(k=="max_cycles"){ max_cycles=(uint32_t)j.num(); }
        else { if(j.peekc()=='"') j.str(); else j.num(); }
    }

    // optional per-cycle trace (bounded: streamed straight to disk, never buffered)
    FILE* tr = nullptr;
    if(argc>2){ tr=fopen(argv[2],"w");
        if(tr) fprintf(tr,"cycle,halted,pc,fetch_pc,dma_a_valid,mxu0Comp,mxu0Data,mxu1Comp,mxu1Data,lsuBusy,xluBusy,vpu_fsm_state,xlu_state,imem_state,vloadBusy,vstoreBusy\n"); }

    dut = new SAtlasCore();
    zero_inputs();
    reset_core(12);

    DramSlave slave;
    for(auto& pr:preload) slave.preload(pr.first,pr.second);

    uint32_t load_cycles_before = 0;
    for(size_t i=0;i<words.size();++i) tl_put_imem(IMEM_BASE+4u*(uint32_t)i, words[i], 2);
    tl_put_csr(START_CSR,1,2);

    bool started=false, halted=false;
    uint32_t cyc=0;
    for(cyc=0;cyc<max_cycles;++cyc){
        slave.drive();
        bool h=(dut->get_io$$halted()==1);
        if(!h) started=true;
        tick();
        slave.account();
        if(tr) fprintf(tr,"%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n", cyc,
            (unsigned)dut->get_io$$halted(),
            (unsigned)dut->scalar$pc_ctrl$pc_reg,
            (unsigned)dut->scalar$pc_ctrl$fetch_pc_reg,
            (unsigned)dut->get_io$$dmaTL$$a$$valid(),
            (unsigned)dut->io$$dbg$$mxu0CompBusy,
            (unsigned)dut->io$$dbg$$mxu0DataBusy,
            (unsigned)dut->io$$dbg$$mxu1CompBusy,
            (unsigned)dut->io$$dbg$$mxu1DataBusy,
            (unsigned)dut->io$$dbg$$lsuBusy,
            (unsigned)dut->io$$dbg$$xluBusy,
            (unsigned)dut->vpu$core$fsm$state,
            (unsigned)dut->xlu$state,
            (unsigned)dut->imem$state,
            (unsigned)dut->lsu$vloadBusy,
            (unsigned)dut->lsu$vstoreBusy);
        if(started&&h){ halted=true; break; }
    }
    if(tr) fclose(tr);

    int halt_reason=-1;
    printf("{\"halted\":%s,\"cycles\":%u,\"halt_reason\":%d,\"outputs\":[",
           halted?"true":"false", cyc, halt_reason);
    for(size_t i=0;i<reads.size();++i){
        std::vector<uint8_t> o=slave.captured(reads[i].first,reads[i].second);
        std::string hx=bytes_to_hex(o.data(),o.size());
        printf("%s\"%s\"",(i?",":""),hx.c_str());
    }
    printf("],\"reads\":%ld,\"writes\":%ld,\"wrap_hits\":%ld,\"raw_min\":%u,\"raw_max\":%u,\"raw_span\":%u,\"alias_collisions\":%ld,\"distinct_wrapped\":%zu}\n",
           slave.reads, slave.writes, slave.wrap_hits, slave.raw_min, slave.raw_max,
           (slave.raw_max>=slave.raw_min? slave.raw_max-slave.raw_min : 0u),
           slave.alias_collisions, slave.seen_wrapped.size());
    delete dut;
    return 0;
}
