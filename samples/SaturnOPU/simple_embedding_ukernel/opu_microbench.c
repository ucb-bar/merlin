// Saturn OPU instruction-cost microbenchmark.
//
// Two-point measurement: runs N1 and N2 VOPACCs back-to-back, measures
// cycles around each, subtracts to cancel constant overhead (rdcycle
// overhead, loop setup, printf, exit).  Marginal cost per op =
// (cyc(N2) - cyc(N1)) / (N2 - N1).
//
// The inner loop unrolls 64 VOPACCs with no branches between them, so
// the only per-iteration branch overhead is one bnez at the bottom —
// amortized 1/64 across VOPACCs, i.e. <0.05 cyc bias if VOPACC=1cyc.
//
// Uses alternating register pairs (v16/v18 vs v20/v22) to avoid
// register-read hazards on the outer-product inputs, matching the
// ukernel's own pair-alternation pattern in mmt4d_riscv_64_xopu.c.

#include <stdint.h>
#include <stdio.h>

static inline uint64_t rdcycle(void) {
	uint64_t c;
	asm volatile("rdcycle %0" : "=r"(c));
	return c;
}

// 64 VOPACCs per call, alternating v16/v18 and v20/v22 to avoid RAW.
static inline void vopacc_block64(void) {
	asm volatile(
#define VOPACC_AB                                                              \
	".insn r 87, 2, 81, x0, x16, x18\n\t"                                      \
	".insn r 87, 2, 81, x0, x20, x22\n\t"
#define VOPACC_AB4 VOPACC_AB VOPACC_AB VOPACC_AB VOPACC_AB
#define VOPACC_AB16 VOPACC_AB4 VOPACC_AB4 VOPACC_AB4 VOPACC_AB4
		VOPACC_AB16 VOPACC_AB16
		:
		:
		: "memory");
}

// 64 OPMVINBCASTs per call. Only targets rd=x0..x3 (the matrix
// registers m0..m3), no register-operand reuse hazards.
static inline void opmvinbcast_block64(void) {
	asm volatile(
#define BCAST_M0 ".insn r 87, 6, 89, zero, zero, zero\n\t"
#define BCAST_M0_4 BCAST_M0 BCAST_M0 BCAST_M0 BCAST_M0
#define BCAST_M0_16 BCAST_M0_4 BCAST_M0_4 BCAST_M0_4 BCAST_M0_4
		BCAST_M0_16 BCAST_M0_16 BCAST_M0_16 BCAST_M0_16
		:
		:
		: "memory");
}

// 64 VMV_VRs extracting row 0 of m0 into v0. No register hazards
// because there's no outer-product dependency — we just want the
// raw issue rate.
static inline void vmv_vr_block64(void) {
	asm volatile("li t0, 0\n\t"
#define VMV_VR_M0 ".insn r 87, 6, 93, x0, t0, x0\n\t"
#define VMV_VR_M0_4 VMV_VR_M0 VMV_VR_M0 VMV_VR_M0 VMV_VR_M0
#define VMV_VR_M0_16 VMV_VR_M0_4 VMV_VR_M0_4 VMV_VR_M0_4 VMV_VR_M0_4
				 VMV_VR_M0_16 VMV_VR_M0_16 VMV_VR_M0_16 VMV_VR_M0_16
				 :
				 :
				 : "t0", "memory");
}

static void setup_vregs(void) {
	// The host toolchain compiles without +v; enable the V extension
	// just for this block so the assembler accepts vsetvli / vmv.v.x.
	// VOPACC reads v16/v18 and v20/v22 — contents don't matter for
	// timing, but VL must be set so the instructions don't no-op.
	asm volatile(".option push\n\t"
				 ".option arch, +v\n\t"
				 "li t0, 16\n\t"
				 "vsetvli zero, t0, e8, m1, ta, ma\n\t"
				 "vmv.v.x v16, zero\n\t"
				 "vmv.v.x v18, zero\n\t"
				 "vmv.v.x v20, zero\n\t"
				 "vmv.v.x v22, zero\n\t"
				 ".option pop\n\t"
				 :
				 :
				 : "t0", "memory");
}

// Time N_OUTER × 64 VOPACCs. Returns cycles.
static uint64_t time_vopaccs(int n_outer) {
	uint64_t start = rdcycle();
	for (int i = 0; i < n_outer; ++i) {
		vopacc_block64();
	}
	uint64_t end = rdcycle();
	return end - start;
}

static uint64_t time_bcasts(int n_outer) {
	uint64_t start = rdcycle();
	for (int i = 0; i < n_outer; ++i) {
		opmvinbcast_block64();
	}
	uint64_t end = rdcycle();
	return end - start;
}

static uint64_t time_vmv_vrs(int n_outer) {
	uint64_t start = rdcycle();
	for (int i = 0; i < n_outer; ++i) {
		vmv_vr_block64();
	}
	uint64_t end = rdcycle();
	return end - start;
}

int main(void) {
	setup_vregs();

	// One round of each instruction, two measurement points.
	// N1 = 16 blocks × 64 = 1024 instructions
	// N2 = 256 blocks × 64 = 16384 instructions
	// Delta = 15360 instructions.
	const int N1_BLOCKS = 16;
	const int N2_BLOCKS = 256;
	const int N1_OPS = N1_BLOCKS * 64;
	const int N2_OPS = N2_BLOCKS * 64;
	const int DELTA_OPS = N2_OPS - N1_OPS;

	printf("OPU instruction-cost microbench\n");
	printf("  N1=%d ops  N2=%d ops  delta=%d ops\n", N1_OPS, N2_OPS, DELTA_OPS);
	printf("  (cyc/op = (cyc_N2 - cyc_N1) / delta)\n\n");

	// Warm rdcycle pipeline.
	(void)rdcycle();
	(void)rdcycle();

	// VOPACC
	uint64_t v_n1 = time_vopaccs(N1_BLOCKS);
	uint64_t v_n2 = time_vopaccs(N2_BLOCKS);
	// Integer arithmetic on cyc/op * 1000 for millicycle precision.
	uint64_t v_milli = ((v_n2 - v_n1) * 1000ULL) / (uint64_t)DELTA_OPS;
	printf("VOPACC        N1=%lu cyc  N2=%lu cyc  -> %lu.%03lu cyc/op\n",
		(unsigned long)v_n1, (unsigned long)v_n2,
		(unsigned long)(v_milli / 1000), (unsigned long)(v_milli % 1000));

	// OPMVINBCAST
	uint64_t b_n1 = time_bcasts(N1_BLOCKS);
	uint64_t b_n2 = time_bcasts(N2_BLOCKS);
	uint64_t b_milli = ((b_n2 - b_n1) * 1000ULL) / (uint64_t)DELTA_OPS;
	printf("OPMVINBCAST   N1=%lu cyc  N2=%lu cyc  -> %lu.%03lu cyc/op\n",
		(unsigned long)b_n1, (unsigned long)b_n2,
		(unsigned long)(b_milli / 1000), (unsigned long)(b_milli % 1000));

	// VMV_VR
	uint64_t m_n1 = time_vmv_vrs(N1_BLOCKS);
	uint64_t m_n2 = time_vmv_vrs(N2_BLOCKS);
	uint64_t m_milli = ((m_n2 - m_n1) * 1000ULL) / (uint64_t)DELTA_OPS;
	printf("VMV_VR        N1=%lu cyc  N2=%lu cyc  -> %lu.%03lu cyc/op\n",
		(unsigned long)m_n1, (unsigned long)m_n2,
		(unsigned long)(m_milli / 1000), (unsigned long)(m_milli % 1000));

	printf("\nCSV, opu_microbench, cyc_per_op, vopacc=%lu.%03lu, "
		   "bcast=%lu.%03lu, vmv_vr=%lu.%03lu\n",
		(unsigned long)(v_milli / 1000), (unsigned long)(v_milli % 1000),
		(unsigned long)(b_milli / 1000), (unsigned long)(b_milli % 1000),
		(unsigned long)(m_milli / 1000), (unsigned long)(m_milli % 1000));
	return 0;
}
