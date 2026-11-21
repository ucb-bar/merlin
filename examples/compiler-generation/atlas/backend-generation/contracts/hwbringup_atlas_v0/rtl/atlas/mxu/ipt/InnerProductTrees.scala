// ============================================================================
// InnerProductTrees.scala — Inner-product-tree MXU datapath.
//
// Uses the shared ComputeReq bundle (parameterized by MxuParams).
// Each lane is an independent AnchorAccumulationTree that computes one
// output column of the matrix product.
//
// Pipeline depth is controlled by `numPipeCuts`; the valid signal is
// shifted through a matching shift register so that `io.out.valid`
// aligns with the first result.
// ============================================================================

package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.InnerProductTreeParams
import atlas.mxu.ComputeReq

/** Inner-product-tree compute array.
  *
  * @param p  IPT geometry parameters (provides lane count, formats, etc.).
  */
class InnerProductTrees(
    p: InnerProductTreeParams = InnerProductTreeParams()
) extends Module {

  val io = IO(new Bundle {
    /** Compute request: activations, partial sums, and control. */
    val compute    = Flipped(Valid(new ComputeReq(p.mxu)))
    /** Weight tile from slot 0 — indexed as (lane)(vecLen). */
    val weightBuf0 = Input(Vec(p.numLanes, Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W))))
    /** Weight tile from slot 1 — indexed as (lane)(vecLen). */
    val weightBuf1 = Input(Vec(p.numLanes, Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W))))
    /** Result vector (valid after pipeline flush). */
    val out        = Valid(Vec(p.numLanes, UInt(p.outputFmt.ieeeWidth.W)))
  })

  // ==========================================================================
  // Valid pipeline — depth = numPipeCuts
  // ==========================================================================

  val outValid = RegInit(false.B)
  if (p.numPipeCuts == 0) {
    outValid := io.compute.valid
  } else {
    val validSr = RegInit(VecInit(Seq.fill(p.numPipeCuts)(false.B)))
    validSr(0) := io.compute.valid
    for (i <- 1 until p.numPipeCuts) validSr(i) := validSr(i - 1)
    outValid := validSr(p.numPipeCuts - 1)
  }
  io.out.valid := outValid

  // ==========================================================================
  // Compute lanes — one AnchorAccumulationTree per output column
  // ==========================================================================

  val req = io.compute.bits

  for (laneIdx <- 0 until p.numLanes) {
    val lane = Module(new AnchorAccumulationTree(p))
    lane.io.act        := req.act
    lane.io.weightBuf0 := io.weightBuf0(laneIdx)
    lane.io.weightBuf1 := io.weightBuf1(laneIdx)
    lane.io.psum       := req.psum(laneIdx)
    lane.io.bufReadSel := req.weightBufSel
    lane.io.accumulate := req.accumulate
    io.out.bits(laneIdx) := lane.io.out
  }
}
