/*
MxuParams.scala — Shared architectural parameters for both MXU datapaths.

Naming follows the frozen system-parameter spec:
  MXU_ARRAY_ROWS, MXU_ARRAY_COLS, WEIGHT_SLOTS_PER_MXU,
  ACCUM_BUFFER_ROWS, ACCUM_BUFFER_COLS_BF16, etc.

Both InnerProductTreeParams (mxu1) and SystolicArrayParams (mxu0) compose
this case class and add design-specific fields on top.
*/

package atlas.mxu

import chisel3._
import chisel3.util._
import sp26FPUnits.AtlasFPType

case class MxuParams(
  arrayRows:         Int = 32,   // MXU_ARRAY_ROWS   — dot-product length / weight-tile rows
  arrayCols:         Int = 32,   // MXU_ARRAY_COLS   — output lanes / weight-tile columns
  accumBufferRows:   Int = 32,   // ACCUM_BUFFER_ROWS
  weightSlotsPerMxu: Int = 2,    // WEIGHT_SLOTS_PER_MXU
) {
  require(arrayRows > 1,          s"arrayRows must be > 1, got $arrayRows")
  require(arrayCols > 1,          s"arrayCols must be > 1, got $arrayCols")
  require(accumBufferRows > 1,    s"accumBufferRows must be > 1, got $accumBufferRows")
  require(weightSlotsPerMxu >= 1, s"weightSlotsPerMxu must be >= 1, got $weightSlotsPerMxu")

  // ── Frozen formats ──
  val inputFmt:  AtlasFPType = AtlasFPType.E4M3    // activation & weight element format
  val accumFmt:  AtlasFPType = AtlasFPType.BF16    // accumulation buffer element format
  val outputFmt: AtlasFPType = AtlasFPType.BF16    // output element format

  // ── Spec-derived constants ──
  val accumBufferColsBF16: Int = arrayCols                                         // ACCUM_BUFFER_COLS_BF16
  val weightSlotBytes:     Int = arrayRows * arrayCols * (inputFmt.ieeeWidth / 8)  // WEIGHT_SLOT_BYTES
  val accumBufferBytes:    Int = accumBufferRows * accumBufferColsBF16 *           // ACCUM_BUFFER_BYTES
                                (accumFmt.ieeeWidth / 8)

  // ── Derived bit-widths (used by bundles and modules) ──
  val rowIdxBits: Int = log2Ceil(accumBufferRows)
  val colIdxBits: Int = log2Ceil(arrayCols)

  override def toString: String =
    s"MxuParams(${arrayRows}x${arrayCols}, accum=${accumBufferRows}x${accumBufferColsBF16}, " +
    s"wSlots=$weightSlotsPerMxu, in=$inputFmt, acc=$accumFmt)"
}
