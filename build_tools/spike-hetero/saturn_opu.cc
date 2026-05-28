// Saturn Outer Product Unit — functional spike extension.
//
// Models the four OPU custom instructions defined in Saturn's
// generators/saturn/benchmarks/common/bme.h (branch origin/opu-fp8):
//
//   VOPACC md, vs2, vs1     md += vs2 ⊗ vs1     (i8 outer-product MAC)
//   OPMVINBCAST md, vs2     md[r, :] = vs2 for r in [0, dim)
//   VMV_VR vd, rs1, ms2     vd = ms2[rs1, :]
//   VMV_RV md, rs1, vs2     md[rs1, :] = vs2
//
// The OPU sits in encoding space that overlaps upstream RVV's Zvqdotq.
// All OPU asm-macro encodings set the V mask bit (vm=1); our MASK
// includes that bit, so Zvqdotq's vm=0 encodings still route through
// spike's base instruction table while OPU's vm=1 encodings hit our
// custom_instructions table first and win.
//
// Functional semantics only — no microarchitectural pipeline / VAT
// modeling. Each instruction executes atomically. That's sufficient
// for modelblaster-flow correctness verify since the asm sequence is
// deterministic at the architectural level.
//
// Matrix-register dim is derived from spike's runtime VLEN: each
// matrix has `dim = VLEN/8` rows × `dim` cols of i32, matching the
// HW's logical view (the per-PE accumulators distributed across the
// mesh + cell array collapse to a dim×dim i32 matrix in software).
// MRF storage is sized at the worst case (VLEN=512 -> 64×64) and
// indexed in-bounds via dim.
//
// Usage:
//   spike --extension=saturn_opu --isa=rv64gcv_zicntr <elf>

#include "decode_macros.h"
#include "extension.h"
#include "insn_macros.h"
#include "processor.h"
#include "trap.h"
#include "vector_unit.h"
#include <array>
#include <cstring>
#include <vector>

namespace {

// Encoding constants. opcode=0x57 (V), funct3+funct7 per bme.h. See
// modelblaster/notes/saturn_opu_spike_support.md for the derivation.
constexpr reg_t MATCH_VOPACC = 0xA2002057; // f3=0x2 f7=0x51
constexpr reg_t MATCH_OPMVINBCAST = 0xB2006057; // f3=0x6 f7=0x59
constexpr reg_t MATCH_VMV_VR = 0xBA006057; // f3=0x6 f7=0x5d
constexpr reg_t MATCH_VMV_RV = 0xAA006057; // f3=0x6 f7=0x55
constexpr reg_t MASK_OPU =
	0xFE00707F; // funct7 | funct3 | opcode (bit 25 included)

// Saturn opuParams (int8 OPU) fixes nMrfRegs at 4. The HW could carry
// more (or different cWidth) under a non-default config, but the asm
// macros only address m0..m3 so we hardcode here.
constexpr int N_MRF = 4;

// VLEN=512 is the largest Saturn target we ship; one OPU row is then
// 64 i32 cells. Static allocation avoids any per-instruction heap.
constexpr int MAX_DIM = 64;

// Decode helpers — equivalent to insn.rd()/.rs1()/.rs2() but keep
// the dependency surface here narrow (we don't want to drag in
// decode.h's full macro stack).
inline int rd_of(insn_t insn) {
	return (insn.bits() >> 7) & 0x1F;
}
inline int rs1_of(insn_t insn) {
	return (insn.bits() >> 15) & 0x1F;
}
inline int rs2_of(insn_t insn) {
	return (insn.bits() >> 20) & 0x1F;
}

} // anonymous namespace

class saturn_opu_t : public extension_t {
  public:
	const char *name() const override {
		return "saturn_opu";
	}

	saturn_opu_t() {
		reset_state();
	}

	void reset(processor_t &) override {
		reset_state();
	}

	std::vector<insn_desc_t> get_instructions(const processor_t &) override;
	std::vector<disasm_insn_t *> get_disasms(
		const processor_t * = nullptr) override;

	// Public state access — the static instruction dispatchers fetch
	// their state through processor_t::get_extension("saturn_opu") and
	// call into these.
	inline int32_t &mrf(int m, int r, int c) {
		return mrf_[m][r * MAX_DIM + c];
	}

  private:
	// [N_MRF][MAX_DIM*MAX_DIM] flat — easier to memset and bounds-check
	// than a nested vector. ~64 KiB total at MAX_DIM=64.
	std::array<std::array<int32_t, MAX_DIM * MAX_DIM>, N_MRF> mrf_;

	void reset_state() {
		for (auto &m : mrf_)
			m.fill(0);
	}
};

namespace {

// Look up the singleton extension state attached to `p`. spike's
// register_extension() puts our instance in p->custom_extensions
// keyed by name(). For the same reason gemmini's spike extension
// does this lookup once per insn, it's cheap — the map has 1 entry.
inline saturn_opu_t *state_for(processor_t *p) {
	return static_cast<saturn_opu_t *>(p->get_extension("saturn_opu"));
}

// dim = VLEN/8 — matches the HW's logical i32 matrix dimension and
// is what the asm macros use (`vsetvli zero, %0, e8, m1` → vlmax = VLEN/8).
inline int opu_dim(processor_t *p) {
	return (int)(p->VU.get_vlen() / 8);
}

// VOPACC md, vs2, vs1
//   md[r][c] += vs1[r] * vs2[c]  for r,c in [0, vl)
// Inputs are i8 (per opuParams aWidth=bWidth=8); accumulator is i32.
reg_t exec_vopacc(processor_t *p, insn_t insn, reg_t pc) {
	if (!p->any_vector_extensions())
		throw trap_illegal_instruction(0);
	auto *s = state_for(p);
	int md = rd_of(insn);
	int vs1 = rs1_of(insn);
	int vs2 = rs2_of(insn);
	if (md >= N_MRF)
		throw trap_illegal_instruction(0);
	const reg_t vl = p->VU.vl->read();
	const int dim = opu_dim(p);
	if ((int)vl > dim) {
		// vsetvli should never have produced vl > vlmax_e8m1; defensive.
		throw trap_illegal_instruction(0);
	}
	for (reg_t r = 0; r < vl; r++) {
		int32_t a = (int32_t)(int8_t)p->VU.elt<int8_t>(vs1, r);
		if (a == 0)
			continue; // micro-opt: zero rows contribute nothing
		int32_t *row = &s->mrf(md, (int)r, 0);
		for (reg_t c = 0; c < vl; c++) {
			int32_t b = (int32_t)(int8_t)p->VU.elt<int8_t>(vs2, c);
			row[c] += a * b;
		}
	}
	return pc + insn_length(insn.bits());
}

// OPMVINBCAST md, vs2
//   md[r][c] = vs2[c]  for r in [0, dim), c in [0, vl)
// In the upstream asm pattern this is preceded by `vsetvli ... e32, m4`
// + `vle32.v v0, (bias)` — vs2 contains i32 elements. Anything past
// the current vl is left at its prior value (tail-agnostic in the
// kernel, so we can also zero — but preserving is the safer model).
reg_t exec_opmvinbcast(processor_t *p, insn_t insn, reg_t pc) {
	if (!p->any_vector_extensions())
		throw trap_illegal_instruction(0);
	auto *s = state_for(p);
	int md = rd_of(insn);
	int vs2 = rs2_of(insn);
	if (md >= N_MRF)
		throw trap_illegal_instruction(0);
	const reg_t vl = p->VU.vl->read();
	const int dim = opu_dim(p);
	for (int r = 0; r < dim; r++) {
		int32_t *row = &s->mrf(md, r, 0);
		for (reg_t c = 0; c < vl; c++) {
			row[c] = (int32_t)p->VU.elt<int32_t>(vs2, c);
		}
	}
	return pc + insn_length(insn.bits());
}

// VMV_VR vd, rs1, ms2
//   vd[c] = ms2[rs1_val][c]  for c in [0, vl)
// rs1 is a scalar register holding the row index. Upstream usage
// always precedes this with `vsetvli ... e32, m4`.
reg_t exec_vmv_vr(processor_t *p, insn_t insn, reg_t pc) {
	if (!p->any_vector_extensions())
		throw trap_illegal_instruction(0);
	auto *s = state_for(p);
	int vd = rd_of(insn);
	int rs1 = rs1_of(insn);
	int ms2 = rs2_of(insn);
	if (ms2 >= N_MRF)
		throw trap_illegal_instruction(0);
	reg_t row_idx = p->get_state()->XPR[rs1];
	const reg_t vl = p->VU.vl->read();
	const int dim = opu_dim(p);
	if ((int)row_idx >= dim)
		throw trap_illegal_instruction(0);
	int32_t *row = &s->mrf(ms2, (int)row_idx, 0);
	for (reg_t c = 0; c < vl; c++) {
		p->VU.elt<int32_t>(vd, c, /*is_write=*/true) = row[c];
	}
	return pc + insn_length(insn.bits());
}

// VMV_RV md, rs1, vs2
//   md[rs1_val][c] = vs2[c]  for c in [0, vl)
reg_t exec_vmv_rv(processor_t *p, insn_t insn, reg_t pc) {
	if (!p->any_vector_extensions())
		throw trap_illegal_instruction(0);
	auto *s = state_for(p);
	int md = rd_of(insn);
	int rs1 = rs1_of(insn);
	int vs2 = rs2_of(insn);
	if (md >= N_MRF)
		throw trap_illegal_instruction(0);
	reg_t row_idx = p->get_state()->XPR[rs1];
	const reg_t vl = p->VU.vl->read();
	const int dim = opu_dim(p);
	if ((int)row_idx >= dim)
		throw trap_illegal_instruction(0);
	int32_t *row = &s->mrf(md, (int)row_idx, 0);
	for (reg_t c = 0; c < vl; c++) {
		row[c] = (int32_t)p->VU.elt<int32_t>(vs2, c);
	}
	return pc + insn_length(insn.bits());
}

} // anonymous namespace

std::vector<insn_desc_t> saturn_opu_t::get_instructions(const processor_t &) {
	// Same handler for all (xlen, rve, logged) variants — OPU semantics
	// are xlen-independent and the logged paths don't need extra work
	// beyond what processor_t already does when log_commits_enabled.
	auto desc = [](reg_t m, reg_t k, insn_func_t f) {
		return insn_desc_t{m, k, f, f, f, f, f, f, f, f};
	};
	return {
		desc(MATCH_VOPACC, MASK_OPU, exec_vopacc),
		desc(MATCH_OPMVINBCAST, MASK_OPU, exec_opmvinbcast),
		desc(MATCH_VMV_VR, MASK_OPU, exec_vmv_vr),
		desc(MATCH_VMV_RV, MASK_OPU, exec_vmv_rv),
	};
}

// Disassembly: prints VOPACC m1, v18, v16 etc. instead of raw .insn.
namespace {

// Argument formatters — adapted from cflush.cc's xrs1 idiom.
struct opu_md_arg : public arg_t {
	std::string to_string(insn_t insn) const override {
		return "m" + std::to_string(rd_of(insn));
	}
};
struct opu_ms2_arg : public arg_t {
	std::string to_string(insn_t insn) const override {
		return "m" + std::to_string(rs2_of(insn));
	}
};
struct opu_vs1_arg : public arg_t {
	std::string to_string(insn_t insn) const override {
		return "v" + std::to_string(rs1_of(insn));
	}
};
struct opu_vs2_arg : public arg_t {
	std::string to_string(insn_t insn) const override {
		return "v" + std::to_string(rs2_of(insn));
	}
};
struct opu_vd_arg : public arg_t {
	std::string to_string(insn_t insn) const override {
		return "v" + std::to_string(rd_of(insn));
	}
};
struct opu_xrs1_arg : public arg_t {
	std::string to_string(insn_t insn) const override {
		return xpr_name[rs1_of(insn)];
	}
};

static opu_md_arg k_md;
static opu_ms2_arg k_ms2;
static opu_vs1_arg k_vs1;
static opu_vs2_arg k_vs2;
static opu_vd_arg k_vd;
static opu_xrs1_arg k_xrs1;

} // anonymous namespace

std::vector<disasm_insn_t *> saturn_opu_t::get_disasms(const processor_t *) {
	return {
		// vopacc md, vs2, vs1
		new disasm_insn_t(
			"vopacc", MATCH_VOPACC, MASK_OPU, {&k_md, &k_vs2, &k_vs1}),
		// opmvinbcast md, vs2
		new disasm_insn_t(
			"opmvinbcast", MATCH_OPMVINBCAST, MASK_OPU, {&k_md, &k_vs2}),
		// vmv.vr vd, rs1, ms2
		new disasm_insn_t(
			"vmv.vr", MATCH_VMV_VR, MASK_OPU, {&k_vd, &k_xrs1, &k_ms2}),
		// vmv.rv md, rs1, vs2
		new disasm_insn_t(
			"vmv.rv", MATCH_VMV_RV, MASK_OPU, {&k_md, &k_xrs1, &k_vs2}),
	};
}

REGISTER_EXTENSION(saturn_opu, []() { return new saturn_opu_t; })
