// Passthrough replacement for rocket-chip's EICG_wrapper clock-gate blackbox.
//
// Ports and module name must match the blackbox rocket-chip elaborates
// (freechips.rocketchip.util.EICG_wrapper), because it is selected by pointing
// ClockGateModelFile at this file rather than by changing any Scala class.
//
// WHY THIS EXISTS. The stock model is a latch-and-AND:
//
//     reg en_latched; always @(*) if (!in) en_latched = en || test_en;
//     assign out = en_latched && in;
//
// On an FPGA that is combinational logic in the clock path, so the tools must
// insert a *second* global clock buffer behind it to reach the gated domain's
// loads. That second global net cannot be delay-matched to its ungated parent,
// and every path crossing the gated/ungated boundary is still checked as one
// clock at zero skew. On FireSimOPUV256D128ShuttleConfig the gated domain (the
// OPU cluster array, 131,585 clock loads) ended up 3.79 ns later than the
// ungated control logic, giving 4.207 ns of clock skew, WHS -4.068 ns and
// 258,620 hold-failing endpoints -- unclosable, and unrelated to frequency,
// since hold and skew are both period-independent.
//
// Gating is a power optimisation, so on an FPGA it buys nothing and costs
// timing closure. Passing the clock through removes the second buffer.
//
// CORRECTNESS IS NOT ASSUMED HERE. Ungating is only sound if nothing in the
// gated domain relies on the clock stopping to retain state. In the OPU,
// OuterProductCell.regs is written under an explicit `when`, but
// OuterProductCluster.pipe is assigned unconditionally
// (`pipe := Mux(io.shift, io.in_pipe, cell_outs(...))`), so it does depend on
// the gate to hold. The sequencer drives the enable as
// `clock_enable := valid || mvout_valids =/= 0.U`, i.e. it is high for every
// cycle an instruction is in flight or a readout is pending, and `pipe` is
// re-primed from cell_outs at the start of each readout rather than
// accumulated across readouts -- so the clobbering that ungating introduces
// should only land on idle cycles whose pipe contents are never read. That is
// an argument, not a measurement: it is validated by re-running the certified
// OPU corpus against this model and requiring bit-identical digests.

/* verilator lint_off UNOPTFLAT */

module EICG_wrapper(
  output out,
  input en,
  input test_en,
  input in
);

  // Enable is intentionally ignored; `in` passes straight through so the gated
  // domain shares one clock tree with its parent.
  wire _unused_ok = &{1'b0, en, test_en, 1'b0};

  assign out = in;

endmodule
