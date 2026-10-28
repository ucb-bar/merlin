// Records the accelerator command stream of one functional-model run without a commit log.
//
// Loaded with --extlib and selected with --extension=command_recorder placed AFTER the accelerator's
// own --extension, so the accelerator's state is allocated exactly where an ordinary run allocates it.
// It rewires, in place, every instruction the named inner extension (MERLIN_COMMAND_RECORDER_INNER)
// registered on the hart, and adds the base FENCE encoding; each calls the original implementation
// unchanged and, after it retires, writes one record:
//
//   uint32 instruction bits, uint32 hart, uint64 x[rs1], uint64 x[rs2]   (values read before execution)
//
// to MERLIN_COMMAND_RECORDER_OUT, after an 8-byte header. At exit it appends a trailer record with
// instruction bits 0 (never a retired instruction): hart = flags, x[rs1] = minstret, x[rs2] = records.
// Flags: 1 a retirement-counter/control access (the retired count is then not the commit count),
// 2 a hart other than 0, 4 an inner instruction this recorder could not wrap (too many, or not yet
// registered on the hart). Any flag means the stream is not a complete equivalent of the commit log and
// the reader falls back to the commit log.
//
// Nothing here names an accelerator, opcode or command: the inner extension's own descriptors select
// what is wrapped. Build against the exact headers of the engine binary it is loaded into.

#include <riscv/csrs.h>
#include <riscv/decode.h>
#include <riscv/encoding.h>
#include <riscv/extension.h>
#include <riscv/processor.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#define RECORDER_BASE_INSN(name)                                  \
  extern reg_t fast_rv32i_##name(processor_t*, insn_t, reg_t);    \
  extern reg_t fast_rv64i_##name(processor_t*, insn_t, reg_t);    \
  extern reg_t fast_rv32e_##name(processor_t*, insn_t, reg_t);    \
  extern reg_t fast_rv64e_##name(processor_t*, insn_t, reg_t);    \
  extern reg_t logged_rv32i_##name(processor_t*, insn_t, reg_t);  \
  extern reg_t logged_rv64i_##name(processor_t*, insn_t, reg_t);  \
  extern reg_t logged_rv32e_##name(processor_t*, insn_t, reg_t);  \
  extern reg_t logged_rv64e_##name(processor_t*, insn_t, reg_t);
RECORDER_BASE_INSN(fence)
RECORDER_BASE_INSN(csrrw)
RECORDER_BASE_INSN(csrrs)
RECORDER_BASE_INSN(csrrc)
RECORDER_BASE_INSN(csrrwi)
RECORDER_BASE_INSN(csrrsi)
RECORDER_BASE_INSN(csrrci)

namespace {

// The engine's "re-execute this instruction serialized" return value (decode_macros.h, not installed);
// the commit log prints no line for it, so no record is written either.
constexpr reg_t kSerializeBefore = 3;
constexpr size_t kMaxInner = 16;
constexpr uint32_t kTouchedCounter = 1, kOtherHart = 2, kUnwrapped = 4, kUnsupported = 8, kLimit = 16;

// The hart's custom-instruction table is private; an explicit instantiation may name a private member,
// which is the one standard way to reach it without patching the engine.
struct CustomTable {
  using type = std::vector<insn_desc_t> processor_t::*;
  friend type table_of(CustomTable);
};
template <typename Tag, typename Tag::type Member>
struct Expose {
  friend typename Tag::type table_of(Tag) { return Member; }
};
template struct Expose<CustomTable, &processor_t::custom_instructions>;

struct Record {
  uint32_t bits;
  uint32_t hart;
  uint64_t rs1;
  uint64_t rs2;
};
static_assert(sizeof(Record) == 24, "record layout");

FILE* out = nullptr;
uint64_t records = 0;
uint32_t flags = 0;
uint64_t max_records = 0;
std::vector<insn_desc_t> inner;
std::shared_ptr<wide_counter_csr_t> minstret;

void write(const Record& r) {
  if (fwrite(&r, sizeof r, 1, out) != 1) {
    fprintf(stderr, "command recorder: write failed\n");
    abort();
  }
}

void record(processor_t* p, insn_t insn, reg_t rs1, reg_t rs2) {
  const uint64_t width = insn_length(insn.bits()) * 8;
  if (width != 32) { flags |= kUnsupported; return; }
  if (records >= max_records) { flags |= kLimit; return; }
  const uint64_t bits = width >= 64 ? insn.bits() : insn.bits() & ((uint64_t(1) << width) - 1);
  if (p->get_id() != 0) flags |= kOtherHart;
  write(Record{uint32_t(bits), uint32_t(p->get_id()), rs1, rs2});
  ++records;
}

insn_func_t variant(const insn_desc_t& d, int v) {
  const insn_func_t all[8] = {d.fast_rv32i,   d.fast_rv64i,   d.fast_rv32e,   d.fast_rv64e,
                              d.logged_rv32i, d.logged_rv64i, d.logged_rv32e, d.logged_rv64e};
  return all[v];
}

template <size_t D, int V>
reg_t inner_call(processor_t* p, insn_t insn, reg_t pc) {
  state_t* s = p->get_state();
  const reg_t rs1 = s->XPR[insn.rs1()], rs2 = s->XPR[insn.rs2()];
  const reg_t npc = variant(inner[D], V)(p, insn, pc);
  if (npc != kSerializeBefore) record(p, insn, rs1, rs2);
  return npc;
}

template <insn_func_t F>
reg_t fence_call(processor_t* p, insn_t insn, reg_t pc) {
  const reg_t npc = F(p, insn, pc);
  if (npc != kSerializeBefore) record(p, insn, 0, 0);
  return npc;
}

template <insn_func_t F>
reg_t counter_call(processor_t* p, insn_t insn, reg_t pc) {
  flags |= kTouchedCounter;
  return F(p, insn, pc);
}

bool same(const insn_desc_t& a, const insn_desc_t& b) {
  for (int v = 0; v < 8; v++)
    if (variant(a, v) != variant(b, v)) return false;
  return a.match == b.match && a.mask == b.mask;
}

template <size_t D>
void wrap_one(std::vector<insn_desc_t>& table) {
  if (D >= inner.size()) return;
  for (auto& entry : table) {
    if (same(entry, inner[D])) {
      entry = insn_desc_t{entry.match,       entry.mask,        inner_call<D, 0>, inner_call<D, 1>,
                          inner_call<D, 2>,  inner_call<D, 3>,  inner_call<D, 4>, inner_call<D, 5>,
                          inner_call<D, 6>,  inner_call<D, 7>};
      return;
    }
  }
  flags |= kUnwrapped;
}

template <size_t... D>
void wrap_inner(std::vector<insn_desc_t>& table, std::index_sequence<D...>) {
  (wrap_one<D>(table), ...);
}

#define RECORDER_DESC(match, mask, wrapper, name)                                                    \
  insn_desc_t {                                                                                      \
    insn_bits_t(match), insn_bits_t(mask), wrapper<fast_rv32i_##name>, wrapper<fast_rv64i_##name>,  \
        wrapper<fast_rv32e_##name>, wrapper<fast_rv64e_##name>, wrapper<logged_rv32i_##name>,        \
        wrapper<logged_rv64i_##name>, wrapper<logged_rv32e_##name>, wrapper<logged_rv64e_##name>     \
  }

void counter_descs(std::vector<insn_desc_t>& to, unsigned csr) {
  const insn_bits_t field = insn_bits_t(csr) << 20, select = insn_bits_t(0xfff) << 20;
  to.push_back(RECORDER_DESC(MATCH_CSRRW | field, MASK_CSRRW | select, counter_call, csrrw));
  to.push_back(RECORDER_DESC(MATCH_CSRRS | field, MASK_CSRRS | select, counter_call, csrrs));
  to.push_back(RECORDER_DESC(MATCH_CSRRC | field, MASK_CSRRC | select, counter_call, csrrc));
  to.push_back(RECORDER_DESC(MATCH_CSRRWI | field, MASK_CSRRWI | select, counter_call, csrrwi));
  to.push_back(RECORDER_DESC(MATCH_CSRRSI | field, MASK_CSRRSI | select, counter_call, csrrsi));
  to.push_back(RECORDER_DESC(MATCH_CSRRCI | field, MASK_CSRRCI | select, counter_call, csrrci));
}

struct Trailer {
  ~Trailer() {
    if (out == nullptr) return;
    write(Record{0, flags, minstret ? minstret->read() : ~uint64_t(0), records});
    fclose(out);
    out = nullptr;
  }
};
Trailer trailer;  // destroyed at exit, before `minstret` (constructed earlier in this unit)

class command_recorder_t : public extension_t {
 public:
  command_recorder_t() {
    if (out != nullptr) return;
    const char* limit = getenv("MERLIN_COMMAND_RECORDER_LIMIT");
    char* end = nullptr;
    max_records = limit == nullptr ? 0 : strtoull(limit, &end, 10);
    if (max_records == 0 || end == nullptr || *end != '\0') {
      fprintf(stderr, "command recorder: a finite command limit is required\n");
      abort();
    }
    const char* path = getenv("MERLIN_COMMAND_RECORDER_OUT");
    if (path == nullptr || (out = fopen(path, "wb")) == nullptr) {
      fprintf(stderr, "command recorder: MERLIN_COMMAND_RECORDER_OUT is not a writable path\n");
      abort();
    }
    static char buffer[1 << 20];
    setvbuf(out, buffer, _IOFBF, sizeof buffer);
    if (fwrite("MCREC001", 8, 1, out) != 1) abort();
  }

  std::vector<insn_desc_t> get_instructions(const processor_t& proc) override {
    if (inner.empty()) {
      const char* name = getenv("MERLIN_COMMAND_RECORDER_INNER");
      if (name == nullptr) {
        fprintf(stderr, "command recorder: MERLIN_COMMAND_RECORDER_INNER names no extension\n");
        abort();
      }
      // A separate instance only lends its descriptors to compare against the registered ones.
      inner = find_extension(name)()->get_instructions(proc);
    }
    if (inner.empty() || inner.size() > kMaxInner) flags |= kUnwrapped;
    // The inner extension registered first; its entries are rewired here, before the hart's decode
    // cache is rebuilt for this registration and before any instruction executes.
    processor_t& hart = const_cast<processor_t&>(proc);
    wrap_inner(hart.*table_of(CustomTable{}), std::make_index_sequence<kMaxInner>{});
    if (hart.get_id() != 0) flags |= kOtherHart;
    minstret = hart.get_state()->minstret;
    // Partial vector memory traps can be printed as commit lines without retiring. Keep the log
    // reader's exact bound for such ISAs by declining the recorder before accepting any result.
    if (hart.extension_enabled('V')) flags |= kUnsupported;
    std::vector<insn_desc_t> insns;
    insns.push_back(RECORDER_DESC(MATCH_FENCE, MASK_FENCE, fence_call, fence));
    counter_descs(insns, CSR_MINSTRET);
    counter_descs(insns, CSR_MINSTRETH);
    counter_descs(insns, CSR_MCOUNTINHIBIT);
    counter_descs(insns, CSR_SCOUNTINHIBIT);
    counter_descs(insns, CSR_MINSTRETCFG);
    counter_descs(insns, CSR_MINSTRETCFGH);
    return insns;
  }

  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override { return {}; }
  const char* name() const override { return "command_recorder"; }

  void reset(processor_t& p) override {
    if (p.get_id() != 0) flags |= kOtherHart;
    minstret = p.get_state()->minstret;
  }
};

}  // namespace

REGISTER_EXTENSION(command_recorder, []() { return new command_recorder_t; })
