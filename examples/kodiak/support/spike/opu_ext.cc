// Functional model of the Saturn outer-product unit (OPU) for spike, loaded with `--extlib`.
//
// WHAT IS MODELLED, AND FROM WHERE. Every instruction word comes from `opu_encodings.h`, which
// spike/build.py GENERATES from the unit's own Chisel (merlin.targetgen.rtl.opu_isa, cross-checked
// against the unit author's bme.h); nothing here spells an opcode, funct3 or funct6. The datapath widths,
// the matrix register count, the vector/datapath lengths and the LMUL bound are read the same way, from
// the elaborated config named in the provider's contract. What this file DOES state is the behaviour, and
// it is a transliteration of three RTL files, cited line by line below:
//
//   * backend/OuterProductSequencer.scala -- the iteration (row_idx/col_idx), the matrix-register index
//     (mrf_idx = Cat(tile, A, B)), the scalar row-index bit permutation, `rd(4)` as the column flag, and
//     the termination conditions (including the ones that never terminate);
//   * backend/Backend.scala               -- how a vector element group is spread over the array
//     (in_l(i)(j) := elems_l(i + j * yDim)) and where a readout lands (wvd_eg, full-width mask);
//   * exu/OuterProductUnit.scala          -- what a cell does with each control (macc / mvin / bcast /
//     bcast_col), and the 32-bit wrapping accumulate.
//
// The state is kept in the array's PHYSICAL coordinates -- (tile, A, B, cluster row, cell row, cluster
// column, cell column) -- and every instruction walks the same sequencer iterations the hardware does.
// That is deliberately not the "natural" C[i][j] += a[i] * b[j] description: the two agree for the forms
// the microkernel uses (and the corpus checks that), but the physical walk also reproduces what the RTL
// does with operands the description does not cover (a readout into v16..v31 swaps the subtile fields,
// because `rd(4)` is read as the column flag for every instruction), so a kernel relying on the
// description beyond its range fails here the way it would on the hardware.
//
// WHAT IS NOT MODELLED. Timing, and therefore the sequencer's read-intent hazard (see the hardware pin
// notes for saturn_opu_int8 and KernelSpec.row_vreg_alt in merlin): a functional model executes
// instructions one at a time, so a younger write cannot overtake an older accumulate's operand read.
// vl is honoured exactly as the RTL honours it: an instruction with vl == 0 is dropped at dispatch
// (frontend/Dispatch.scala), otherwise the sequencer never reads vl. The taped-out part's partial-N
// erratum is a property of a DIFFERENT revision, whose RTL this model was not built from.
//
// A configuration the RTL hangs on (an iteration count the sequencer's col_idx register cannot reach,
// e.g. a readout at e32/m1) raises illegal-instruction here and says why on stderr, rather than spinning.

#include "opu_encodings.h"

#include "extension.h"
#include "processor.h"
#include "trap.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- geometry: HasOPUParams (exu/OuterProductUnit.scala), from the derived parameters ----------------

constexpr unsigned ilog2(unsigned long v) { return v <= 1 ? 0 : 1 + ilog2(v >> 1); }
constexpr unsigned long pow2ceil(unsigned long v) { return v <= 1 ? 1 : 2ul << ilog2(v - 1); }
constexpr bool is_pow2(unsigned long v) { return v && !(v & (v - 1)); }

constexpr unsigned long VLEN = OPU_VLEN;
constexpr unsigned long DLEN = OPU_DLEN;
constexpr unsigned long SUB = VLEN / DLEN;                        // vLen/dLen: subtiles per dimension
constexpr unsigned long CLUSTER_X = OPU_C_BITS / OPU_B_BITS;      // clusterXdim = cWidth / bWidth
constexpr unsigned long CLUSTER_Y = OPU_C_BITS / OPU_A_BITS;      // clusterYdim = cWidth / aWidth
constexpr unsigned long Y_DIM = (DLEN / OPU_A_BITS) / CLUSTER_Y;  // yDim
constexpr unsigned long X_DIM = (DLEN / OPU_B_BITS) / CLUSTER_X;  // xDim
constexpr unsigned long MREGS = OPU_MATRIX_REGS;                  // nMrfRegs
constexpr unsigned long EGS_PER_VREG = VLEN / DLEN;
constexpr unsigned long EGS_TOTAL = 32 * EGS_PER_VREG;
constexpr unsigned long EDGE = VLEN / OPU_A_BITS;                 // the tile edge: VLMAX at e8/m1
// col_idx is UInt(log2Ceil(clusterXdim * maxLMUL * vLen/dLen).W) and next_col_idx = col_idx +& 1, so
// the largest count the sequencer can terminate at is exactly 2^width.
constexpr unsigned long COL_IDX_REACH = pow2ceil(CLUSTER_X * OPU_MAX_LMUL * SUB);

static_assert(OPU_A_BITS == 8 && OPU_B_BITS == 8, "this model reads int8 operands");
static_assert(OPU_C_BITS == 32, "this model accumulates in int32");
static_assert(is_pow2(SUB) && is_pow2(Y_DIM) && is_pow2(X_DIM) && is_pow2(CLUSTER_X) && is_pow2(CLUSTER_Y),
              "the sequencer's index fields are bit slices; non-power-of-two geometry is not modelled");
static_assert(is_pow2(MREGS), "mrf_idx truncation to the tile field assumes a power-of-two register count");
static_assert(Y_DIM * CLUSTER_Y * SUB == EDGE && X_DIM * CLUSTER_X * SUB == EDGE, "array does not tile the edge");

constexpr unsigned long CELLS_Y = Y_DIM * CLUSTER_Y;  // physical rows of cells  (= dLen / aWidth)
constexpr unsigned long CELLS_X = X_DIM * CLUSTER_X;  // physical cols of cells  (= dLen / bWidth)

enum class Op { macc, mvin, bcast, mvout };

class opu_t : public extension_t {
 public:
  const char* name() const override { return OPU_EXT_NAME_STR; }

  std::vector<insn_desc_t> get_instructions(const processor_t&) override;
  std::vector<disasm_insn_t*> get_disasms(const processor_t* = nullptr) override;
  void reset(processor_t&) override { std::fill(regs.begin(), regs.end(), 0u); }

  // regs[tile][A][B][cluster row][cell row][cluster col][cell col]: the cell registers of
  // OuterProductCell (Reg(Vec(regsPerCell, SInt(cWidth)))), regsPerCell = (vLen/dLen)^2 * nMrfRegs
  // indexed by mrf_idx = Cat(tile, A, B). Uninitialised in the RTL; zero here.
  std::vector<uint32_t> regs = std::vector<uint32_t>(MREGS * SUB * SUB * CELLS_Y * CELLS_X, 0u);

  uint32_t& cell(unsigned long tile, unsigned long a, unsigned long b, unsigned long c, unsigned long r,
                 unsigned long cj, unsigned long cc) {
    const unsigned long row = c * CLUSTER_Y + r, col = cj * CLUSTER_X + cc;
    return regs[(((tile * SUB + a) * SUB + b) * CELLS_Y + row) * CELLS_X + col];
  }
};

opu_t* unit_of(processor_t* p) {
  return static_cast<opu_t*>(p->get_extension(OPU_EXT_NAME_STR));
}

[[noreturn]] void illegal(insn_t insn) { throw trap_illegal_instruction(insn.bits()); }

// ---- vector register file access by ELEMENT GROUP, as RegisterAccess indexes it ------------------------
// An element group is dLen bits; eg = vreg * egsPerVReg + k, truncated to log2(egsTotal) bits.

uint8_t eg_byte(processor_t* p, unsigned long eg, unsigned long byte) {
  eg %= EGS_TOTAL;
  return p->VU.elt<uint8_t>(eg / EGS_PER_VREG, (eg % EGS_PER_VREG) * (DLEN / 8) + byte);
}

uint32_t eg_word(processor_t* p, unsigned long eg, unsigned long word) {
  eg %= EGS_TOTAL;
  return p->VU.elt<uint32_t>(eg / EGS_PER_VREG, (eg % EGS_PER_VREG) * (DLEN / 32) + word);
}

void eg_word_write(processor_t* p, unsigned long eg, unsigned long word, uint32_t v) {
  eg %= EGS_TOTAL;
  p->VU.elt<uint32_t>(eg / EGS_PER_VREG, (eg % EGS_PER_VREG) * (DLEN / 32) + word, true) = v;
}

// ---- the legality checks the hardware applies (frontend/EarlyDecode.scala, PipelinedFaultCheck.scala) --
// plus spike's own vector-state rule, which every surrounding vector instruction is already held to.
// Returns false when the instruction is dropped (vl <= vstart: Dispatch.scala never dispatches it).
bool admit(processor_t* p, insn_t insn) {
  state_t* s = p->get_state();
  if (!p->any_vector_extensions() || !s->sstatus->enabled(SSTATUS_VS)) illegal(insn);
  if (p->VU.vill) illegal(insn);                    // EarlyDecode: legal := !vtype.vill
  if (p->VU.vstart->read() != 0) illegal(insn);     // PipelinedFaultCheck: vstart =/= 0 && !vmu -> illegal
  if (p->VU.VLEN != VLEN) {
    std::fprintf(stderr,
                 "[%s] spike runs VLEN=%lu but this model was built for %s (vLen=%lu): rebuild the "
                 "extension for the simulated configuration or pass the matching --isa zvl\n",
                 OPU_EXT_NAME_STR, (unsigned long)p->VU.VLEN, OPU_CONFIG, VLEN);
    illegal(insn);
  }
  s->sstatus->dirty(SSTATUS_VS);
  return p->VU.vl->read() > p->VU.vstart->read();
}

// emul - eew as the sequencer computes it: two 2-bit registers subtracted, wrapping.
// emul = vlmul_sign ? 0 : vlmul_mag (PipelinedFaultCheck), eew = vtype.vsew.
unsigned long lmul_shift(processor_t* p) {
  const float flmul = p->VU.vflmul;
  const unsigned long emul = flmul >= 1.0f ? ilog2((unsigned long)flmul) : 0;
  const unsigned long eew = ilog2(p->VU.vsew / 8);
  return (emul - eew) & 3;
}

void hang(insn_t insn, const char* what, unsigned long count) {
  std::fprintf(stderr,
               "[%s] %s at this vtype needs %lu col_idx steps; the sequencer's col_idx can only reach "
               "%lu, so the RTL never retires it (it hangs). Raising illegal-instruction instead.\n",
               OPU_EXT_NAME_STR, what, count, COL_IDX_REACH);
  illegal(insn);
}

// The sequencer's scalar_row_idx: Cat(rs1[log2(yDim)-1:0], rs1[log2(yDim*clusterYdim*vLen/dLen)-1:log2(yDim)]).
unsigned long scalar_row_idx(reg_t rs1) {
  const unsigned long low = rs1 & (Y_DIM - 1);
  const unsigned long high = (rs1 >> ilog2(Y_DIM)) & (CLUSTER_Y * SUB - 1);
  return (low << ilog2(CLUSTER_Y * SUB)) | high;
}

reg_t execute(processor_t* p, insn_t insn, reg_t pc, Op op) {
  if (!admit(p, insn)) return pc + 4;
  opu_t* u = unit_of(p);
  const unsigned long rd = insn.rd(), rs1 = insn.rs1(), rs2 = insn.rs2();
  const bool col = (rd >> 4) & 1;  // mvin_col := dis_inst.rd(4) -- for EVERY instruction the unit runs
  const unsigned long shift = lmul_shift(p);

  // Iteration bounds (col_idx_tail / row_idx_tail).
  const unsigned long col_limit = op == Op::macc ? (SUB << shift) : ((CLUSTER_X * SUB) << shift);
  const unsigned long row_limit = op == Op::macc ? SUB : 1;
  if (col_limit > COL_IDX_REACH) hang(insn, op == Op::macc ? "the accumulate" : "this move", col_limit);

  // VX forms carry a scalar row index; the VV accumulate does not read one.
  const unsigned long R = op == Op::macc ? 0 : scalar_row_idx(p->get_state()->XPR[rs1]);
  const unsigned long cluster_row = (R >> ilog2(CLUSTER_Y * SUB)) & (Y_DIM - 1);  // scalar_cluster_row_idx
  const unsigned long r_sel = R & (CLUSTER_Y - 1);                               // row_idx into a cluster

  for (unsigned long row_idx = 0; row_idx < row_limit; ++row_idx) {
    for (unsigned long col_idx = 0; col_idx < col_limit; ++col_idx) {
      // mrf_eg / mrf_row_idx / mrf_col_idx, then mrf_idx = Cat(mrf_eg, X, Y) truncated to cellRegIdxBits.
      const unsigned long mrf_eg = (op == Op::mvout ? rs2 : rd) +
                                   (op == Op::macc ? (col_idx >> ilog2(SUB)) : (col_idx >> ilog2(CLUSTER_X * SUB)));
      const unsigned long mrf_row = (op == Op::macc ? row_idx : (R >> ilog2(CLUSTER_Y))) & (SUB - 1);
      const unsigned long mrf_col = (op == Op::macc ? col_idx : (col_idx >> ilog2(CLUSTER_X))) & (SUB - 1);
      const unsigned long tile = mrf_eg & (MREGS - 1);
      const unsigned long A = col ? mrf_col : mrf_row;
      const unsigned long B = col ? mrf_row : mrf_col;
      const unsigned long cc_sel = col_idx & (CLUSTER_X - 1);  // col_idx into a cluster

      switch (op) {
        case Op::macc: {
          // rvs1 eg = rs1 * egsPerVReg + row_idx; rvs2 eg = rs2 * egsPerVReg + col_idx (Backend.scala:
          // in_l(i)(j) := elems_l(i + j * yDim), in_t(i)(j) := elems_t(i + j * xDim)). Every cell whose
          // register matches mrf_idx accumulates in_l * in_t, signed 8x8, wrapping at 32 bits.
          const unsigned long eg1 = rs1 * EGS_PER_VREG + row_idx, eg2 = rs2 * EGS_PER_VREG + col_idx;
          int32_t in_l[Y_DIM][CLUSTER_Y], in_t[X_DIM][CLUSTER_X];
          for (unsigned long c = 0; c < Y_DIM; ++c)
            for (unsigned long r = 0; r < CLUSTER_Y; ++r) in_l[c][r] = (int8_t)eg_byte(p, eg1, c + r * Y_DIM);
          for (unsigned long cj = 0; cj < X_DIM; ++cj)
            for (unsigned long cc = 0; cc < CLUSTER_X; ++cc) in_t[cj][cc] = (int8_t)eg_byte(p, eg2, cj + cc * X_DIM);
          for (unsigned long c = 0; c < Y_DIM; ++c)
            for (unsigned long r = 0; r < CLUSTER_Y; ++r) {
              const int32_t a = in_l[c][r];
              uint32_t* row = &u->cell(tile, A, B, c, r, 0, 0);  // one physical row of cells, contiguous
              for (unsigned long cj = 0; cj < X_DIM; ++cj)
                for (unsigned long cc = 0; cc < CLUSTER_X; ++cc)
                  row[cj * CLUSTER_X + cc] += (uint32_t)(a * in_t[cj][cc]);  // 8x8 signed fits; += wraps at 32
            }
          break;
        }
        case Op::mvin: {
          const unsigned long eg = rs2 * EGS_PER_VREG + col_idx;
          if (!col) {
            // mvin(i) only for cluster row == scalar_cluster_row_idx; cell (row_idx, col_idx) of each
            // cluster takes in_t(cluster col): word `cluster col` of the element group.
            for (unsigned long cj = 0; cj < X_DIM; ++cj)
              u->cell(tile, A, B, cluster_row, r_sel, cj, cc_sel) = eg_word(p, eg, cj);
          } else {
            // mvin_col(j) only for cluster column == scalar_cluster_row_idx; cell (row = col_idx,
            // col = row_idx) of each cluster takes in_l(cluster row).
            for (unsigned long ci = 0; ci < Y_DIM; ++ci)
              u->cell(tile, A, B, ci, cc_sel, cluster_row, r_sel) = eg_word(p, eg, ci);
          }
          break;
        }
        case Op::bcast: {
          const unsigned long eg = rs2 * EGS_PER_VREG + col_idx;
          if (!col) {
            // bcast_match: every register of the tile whose LOW field equals B; cell column == col_idx,
            // every row, every cluster; data in_t(cluster col).
            for (unsigned long a = 0; a < SUB; ++a)
              for (unsigned long c = 0; c < Y_DIM; ++c)
                for (unsigned long r = 0; r < CLUSTER_Y; ++r)
                  for (unsigned long cj = 0; cj < X_DIM; ++cj)
                    u->cell(tile, a, B, c, r, cj, cc_sel) = eg_word(p, eg, cj);
          } else {
            // bcast_col_match: every register of the tile whose HIGH field equals A; cell row == col_idx,
            // every column, every cluster; data in_l(cluster row).
            for (unsigned long b = 0; b < SUB; ++b)
              for (unsigned long ci = 0; ci < Y_DIM; ++ci)
                for (unsigned long cj = 0; cj < X_DIM; ++cj)
                  for (unsigned long jj = 0; jj < CLUSTER_X; ++jj)
                    u->cell(tile, A, b, ci, cc_sel, cj, jj) = eg_word(p, eg, ci);
          }
          break;
        }
        case Op::mvout: {
          // Cluster row scalar_cluster_row_idx drives cell_outs(row_idx)(col_idx) down the pipe; output
          // word e is cluster column e. Written whole (mask = ~0) at wvd_eg = rd * egsPerVReg + col_idx.
          const unsigned long eg = rd * EGS_PER_VREG + col_idx;
          for (unsigned long e = 0; e < X_DIM; ++e)
            eg_word_write(p, eg, e, u->cell(tile, A, B, cluster_row, r_sel, e, cc_sel));
          break;
        }
      }
    }
  }
  return pc + 4;
}

reg_t do_macc(processor_t* p, insn_t insn, reg_t pc) { return execute(p, insn, pc, Op::macc); }
reg_t do_mvin(processor_t* p, insn_t insn, reg_t pc) { return execute(p, insn, pc, Op::mvin); }
reg_t do_bcast(processor_t* p, insn_t insn, reg_t pc) { return execute(p, insn, pc, Op::bcast); }
reg_t do_mvout(processor_t* p, insn_t insn, reg_t pc) { return execute(p, insn, pc, Op::mvout); }

insn_desc_t desc(insn_bits_t match, insn_bits_t mask, insn_func_t f) {
  return insn_desc_t{match, mask, f, f, f, f, f, f, f, f};
}

std::vector<insn_desc_t> opu_t::get_instructions(const processor_t&) {
  return {
      desc(OPU_ACCUMULATE_MATCH, OPU_ACCUMULATE_MASK, do_macc),
      desc(OPU_OPERAND_LOAD_MATCH, OPU_OPERAND_LOAD_MASK, do_mvin),
      desc(OPU_BROADCAST_MATCH, OPU_BROADCAST_MASK, do_bcast),
      desc(OPU_READOUT_MATCH, OPU_READOUT_MASK, do_mvout),
  };
}

// ---- disassembly: the matrix operand prints as m<n> (mc<n> when rd(4) selects the column form) --------

struct field_arg_t : public arg_t {
  enum Kind { mat_rd, mat_rs2, vec_rd, vec_rs1, vec_rs2, x_rs1 } kind;
  explicit field_arg_t(Kind k) : kind(k) {}
  std::string to_string(insn_t insn) const override {
    switch (kind) {
      case mat_rd:
        return ((insn.rd() >> 4) & 1 ? "mc" : "m") + std::to_string(insn.rd() & 15);
      case mat_rs2: return "m" + std::to_string(insn.rs2());
      case vec_rd: return "v" + std::to_string(insn.rd());
      case vec_rs1: return "v" + std::to_string(insn.rs1());
      case vec_rs2: return "v" + std::to_string(insn.rs2());
      case x_rs1: return "x" + std::to_string(insn.rs1());
    }
    return "?";
  }
};

std::vector<disasm_insn_t*> opu_t::get_disasms(const processor_t*) {
  static const field_arg_t md(field_arg_t::mat_rd), ms2(field_arg_t::mat_rs2), vd(field_arg_t::vec_rd),
      vs1(field_arg_t::vec_rs1), vs2(field_arg_t::vec_rs2), xs1(field_arg_t::x_rs1);
  return {
      new disasm_insn_t(OPU_ACCUMULATE_MNEMONIC, OPU_ACCUMULATE_MATCH, OPU_ACCUMULATE_MASK, {&md, &vs1, &vs2}),
      new disasm_insn_t(OPU_OPERAND_LOAD_MNEMONIC, OPU_OPERAND_LOAD_MATCH, OPU_OPERAND_LOAD_MASK, {&md, &xs1, &vs2}),
      new disasm_insn_t(OPU_BROADCAST_MNEMONIC, OPU_BROADCAST_MATCH, OPU_BROADCAST_MASK, {&md, &vs2}),
      new disasm_insn_t(OPU_READOUT_MNEMONIC, OPU_READOUT_MATCH, OPU_READOUT_MASK, {&vd, &xs1, &ms2}),
  };
}

}  // namespace

// REGISTER_EXTENSION stringises its first argument, so the generated name is expanded one level first.
#define OPU_REGISTER(name) REGISTER_EXTENSION(name, []() { return new opu_t; })
OPU_REGISTER(OPU_EXT_NAME)
