// ============================================================================
// WeightBuffers.scala — Dual weight-tile register storage.
//
// Spec parameters:
//   WEIGHT_SLOTS_PER_MXU = 2      two weight slots per MXU
//   WEIGHT_SLOT_BYTES    = 1024   bytes per slot (32 × 32 × 1 FP8)
//   MXU_ARRAY_ROWS       = 32    rows  (= input-vector length)
//   MXU_ARRAY_COLS       = 32    cols  (= output lanes)
//
// Each weight slot holds one arrayRows × arrayCols tile of FP8 values,
// stored as registers so that the full tile is available combinationally
// to the compute datapath every cycle.
//
// Access model:
//   • Write port — the sequencer pushes one lane (column) per cycle during
//     a PushWeight command.  After arrayCols cycles the full tile is loaded.
//   • Read ports — both tiles are continuously exposed as combinational
//     outputs for the datapath to consume.
//
// Parameterized by MxuParams so both mxu0 (SA) and mxu1 (IPT) share the
// same module.
// ============================================================================

package atlas.mxu

import chisel3._
import chisel3.util._

/** Dual weight-tile storage for one MXU.
  *
  * @param p  MXU geometry parameters (provides array dimensions and FP format).
  */
class WeightBuffers(p: MxuParams) extends Module {

  // Shorthand for the tile type: arrayCols columns, each holding arrayRows FP8 values.
  private val elemWidth = p.inputFmt.ieeeWidth

  val io = IO(new Bundle {
    /** Write one column into the selected weight slot (one lane per cycle). */
    val writeReq = Flipped(Valid(new WeightWriteReq(p)))

    /** Combinational read-out of weight slot 0 (full tile). */
    val slot0 = Output(Vec(p.arrayCols, Vec(p.arrayRows, UInt(elemWidth.W))))
    /** Combinational read-out of weight slot 1 (full tile). */
    val slot1 = Output(Vec(p.arrayCols, Vec(p.arrayRows, UInt(elemWidth.W))))
  })

  // ==========================================================================
  // Storage — registered tiles, zero-initialized
  // ==========================================================================

  private val zeroLane = VecInit(Seq.fill(p.arrayRows)(0.U(elemWidth.W)))
  private val zeroTile = VecInit(Seq.fill(p.arrayCols)(zeroLane))

  val weightSlot0 = RegInit(zeroTile)
  val weightSlot1 = RegInit(zeroTile)

  // ==========================================================================
  // Write path — sequencer pushes one column per cycle
  // ==========================================================================

  when(io.writeReq.valid) {
    when(io.writeReq.bits.weightSlot) {
      weightSlot1(io.writeReq.bits.laneIdx) := io.writeReq.bits.data
    }.otherwise {
      weightSlot0(io.writeReq.bits.laneIdx) := io.writeReq.bits.data
    }
  }

  // ==========================================================================
  // Read ports — continuous combinational output
  // ==========================================================================

  io.slot0 := weightSlot0
  io.slot1 := weightSlot1
}
