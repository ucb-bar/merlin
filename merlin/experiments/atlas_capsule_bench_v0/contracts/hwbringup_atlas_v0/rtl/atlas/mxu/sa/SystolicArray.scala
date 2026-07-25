// ============================================================================
// SystolicArray.scala — Systolic-array MXU datapath.
//
// Computes the equivalent of PyTorch's F.linear (y = x·Aᵀ + b) in hardware.
//
// Uses the shared ComputeReq bundle: partial-sum data arrives through the
// `psum` field (pre-read by the sequencer), so the core never touches the
// accumulation buffer directly.
//
// Timing:
//   • Activations are skewed left-to-right across rows (delay = row index).
//   • Addends (partial sums) are skewed top-to-bottom across columns.
//   • Outputs are de-skewed bottom-to-top so they emerge as a single row.
//   • Total pipeline latency = MXU_ARRAY_ROWS + MXU_ARRAY_COLS − 2 cycles.
//     The bottom-row mac feeds the deskew chain combinationally; the accBuf
//     SyncReadMem write downstream provides the missing pipeline stage.
// ============================================================================

package atlas.sa

import chisel3._
import chisel3.util._
import atlas.common._
import atlas.mxu.ComputeReq

/** Systolic-array compute mesh.
  *
  * @param p  Systolic-array geometry and PE-architecture parameters.
  */
class SystolicArray(p: SystolicArrayParams) extends Module {

  val io = IO(new Bundle {
    /** Compute request: activations, partial sums, and control. */
    val computeReq = Flipped(Valid(new ComputeReq(p.mxu)))
    /** Weight tile from slot 0 — indexed as (row)(col). */
    val weights0   = Input(Vec(p.rows, Vec(p.cols, UInt(p.inT.ieeeWidth.W))))
    /** Weight tile from slot 1 — indexed as (row)(col). */
    val weights1   = Input(Vec(p.rows, Vec(p.cols, UInt(p.inT.ieeeWidth.W))))
    /** Result row (valid after pipeline flush). */
    val outputRow  = Valid(Vec(p.cols, UInt(p.outT.ieeeWidth.W)))
  })

  val req = io.computeReq.bits

  // ==========================================================================
  // PE mesh instantiation
  // ==========================================================================

  val peMesh = Module(new PEMesh(p.rows, p.cols, p.peArch, p.fmaStages))

  // Extra pipeline latency inside each PE's mac datapath (0 unless the FP8 FMA
  // is pipelined). The vertical (partial-sum) hop between rows is then C+1
  // registers (C inside the PE + 1 explicit RegNext in PEMesh), while the
  // horizontal (activation) hop stays 1 register. Keeping the wavefront aligned
  // therefore requires the per-row input skew to be (C+1) instead of 1; the
  // per-column addend skew and output de-skew stay at 1 (horizontal hop). This
  // is a bit-exact transform of the array — same result, +C·(rows-1) latency.
  val C = p.fmaStages

  // ==========================================================================
  // Left edge — skew and format activations + weight-slot select
  // ==========================================================================

  for (i <- 0 until p.rows) {
    val actSkewed       = ShiftRegister(req.act(i), i * (C + 1))
    val weightSelSkewed = ShiftRegister(req.weightBufSel, i * (C + 1))
    peMesh.io.actVec(i)           := p.peArch.formatMul(actSkewed)
    peMesh.io.weightReadSelVec(i) := weightSelSkewed
  }

  // ==========================================================================
  // Top edge — skew and format addends (partial sums)
  // ==========================================================================

  for (j <- 0 until p.cols) {
    val zero   = 0.U(p.outT.ieeeWidth.W)
    val addend = Mux(req.accumulate, req.psum(j), zero)
    peMesh.io.addendVec(j) := p.peArch.formatAdd(ShiftRegister(addend, j))
  }

  // ==========================================================================
  // Bottom edge — format and de-skew outputs
  // ==========================================================================

  for (j <- 0 until p.cols) {
    val formatted = p.peArch.formatOut(peMesh.io.outVec(j))
    io.outputRow.bits(j) := ShiftRegister(formatted, (p.cols - 1) - j)
  }

  // ==========================================================================
  // Z-axis — format and feed weights
  // ==========================================================================

  for (i <- 0 until p.rows; j <- 0 until p.cols) {
    peMesh.io.weights0(i)(j) := p.peArch.formatMul(io.weights0(i)(j))
    peMesh.io.weights1(i)(j) := p.peArch.formatMul(io.weights1(i)(j))
  }

  // ==========================================================================
  // Valid pipeline — total latency = rows + cols − 2
  // ==========================================================================

  io.outputRow.valid := ShiftRegister(
    io.computeReq.valid, (C + 1) * (p.rows - 1) + (p.cols - 1) + C, false.B, true.B)
}
