// ============================================================================
// InnerProductTreesTop.scala — Top-level wrapper for the IPT MXU.
//
// Instantiates and wires:
//   • InnerProductTreesSequencer — 3-slot concurrent command FSMs,
//                                  FP8↔BF16 (de)quant.
//   • InnerProductTrees          — MXU datapath (reduction-tree lanes).
//   • WeightBuffers              — dual weight-tile storage (from atlas.mxu).
//   • AccumulationBuffers        — dual BF16 accum storage (from atlas.mxu).
// ============================================================================

package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.{InnerProductTreeParams, MregParams, MregReadReq, MregWriteReq}
import atlas.mxu.{MxuCmd, WeightBuffers, AccumulationBuffers}

/** Top-level wrapper for the inner-product-tree MXU.
  *
  * @param p      IPT geometry parameters.
  * @param mregP  Tensor register file parameters (spec: NUM_MREG, MREG_ROWS, …).
  */
class InnerProductTreesTop(
    p:     InnerProductTreeParams = InnerProductTreeParams(),
    mregP: MregParams             = MregParams()
) extends Module {

  require(p.vecLen * p.inputFmt.ieeeWidth == mregP.mregRowBits,
    s"vecLen(${p.vecLen}) × inputWidth(${p.inputFmt.ieeeWidth}) must equal mregRowBits(${mregP.mregRowBits})")
  require(p.tileRows <= mregP.mregRows,
    s"tileRows(${p.tileRows}) must not exceed mregRows(${mregP.mregRows})")
  require(p.numLanes <= mregP.mregRows,
    s"numLanes(${p.numLanes}) must not exceed mregRows(${mregP.mregRows}) for PushWeight")

  val io = IO(new Bundle {
    // ── Command input (fire-and-forget, no backpressure) ──
    val cmd = Flipped(Valid(new MxuCmd(mregP.mregIdBits)))

    // ── Tensor register file (mreg) ports ──
    val mregReadReq0  = Valid(new MregReadReq(mregP))
    val mregReadResp0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadReq1  = Valid(new MregReadReq(mregP))
    val mregReadResp1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq0 = Valid(new MregWriteReq(mregP))
    val mregWriteReq1 = Valid(new MregWriteReq(mregP))

    // ── Granular busy signals for scalar-core hazard logic ──
    val compBusy    = Output(Bool())
    val pushBusy    = Output(Bool())
    val popBusy     = Output(Bool())
    val dataBusy    = Output(Bool())
    val computeBusy = Output(Bool())

    // ── Active mreg bank reports for direction-aware TRF tracking ──
    val activeReads  = Output(Vec(2, Valid(UInt(mregP.mregIdBits.W))))
    val activeWrites = Output(Vec(2, Valid(UInt(mregP.mregIdBits.W))))
  })

  // ==========================================================================
  // Sub-modules
  // ==========================================================================

  val seq    = Module(new InnerProductTreesSequencer(p, mregP))
  val core   = Module(new InnerProductTrees(p))
  val wbuf   = Module(new WeightBuffers(p.mxu))
  val accBuf = Module(new AccumulationBuffers(p.mxu))

  // ==========================================================================
  // Command interface
  // ==========================================================================

  seq.io.cmd := io.cmd

  // ==========================================================================
  // Tensor register file ports (directly forwarded)
  // ==========================================================================

  io.mregReadReq0      <> seq.io.mregReadReq0
  seq.io.mregReadResp0 := io.mregReadResp0
  io.mregReadReq1      <> seq.io.mregReadReq1
  seq.io.mregReadResp1 := io.mregReadResp1
  io.mregWriteReq0     <> seq.io.mregWriteReq0
  io.mregWriteReq1     <> seq.io.mregWriteReq1

  // ==========================================================================
  // Busy signals
  // ==========================================================================

  io.compBusy    := seq.io.compBusy
  io.pushBusy    := seq.io.pushBusy
  io.popBusy     := seq.io.popBusy
  io.dataBusy    := seq.io.dataBusy
  io.computeBusy := seq.io.computeBusy

  // ==========================================================================
  // Active mreg bank reports (forwarded from sequencer)
  // ==========================================================================

  io.activeReads  := seq.io.activeReads
  io.activeWrites := seq.io.activeWrites

  // ==========================================================================
  // Sequencer → WeightBuffers
  // ==========================================================================

  wbuf.io.writeReq <> seq.io.weightWriteReq

  // ==========================================================================
  // WeightBuffers → Core (InnerProductTrees)
  // ==========================================================================

  core.io.weightBuf0 := wbuf.io.slot0
  core.io.weightBuf1 := wbuf.io.slot1

  // ==========================================================================
  // Sequencer ↔ Core
  // ==========================================================================

  core.io.compute := seq.io.compute
  seq.io.coreOut  := core.io.out

  // ==========================================================================
  // Sequencer ↔ AccumulationBuffers
  // ==========================================================================

  accBuf.io.computeWriteReq <> seq.io.accComputeWrite
  accBuf.io.computeReadAddr := seq.io.accComputeReadAddr
  accBuf.io.computeReadEn   := seq.io.accComputeReadEn
  seq.io.accComputeReadData := accBuf.io.computeReadData
  accBuf.io.loadReq         <> seq.io.accLoadReq
  accBuf.io.storeAddr       := seq.io.accStoreAddr
  accBuf.io.storeReadEn     := seq.io.accStoreReadEn
  seq.io.accStoreData       := accBuf.io.storeData
}
