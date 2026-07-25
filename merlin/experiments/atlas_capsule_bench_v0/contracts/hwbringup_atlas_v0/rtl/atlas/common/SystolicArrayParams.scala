package atlas.common

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import atlas.mxu.{MxuParams, FPUtils}

/** 
  * Defines the hardware architecture and arithmetic logic for a Processing Element (PE).
  *
  * This trait is used to configure the type of arithmetic units instantiated within
  * the systolic array grid. By using a sealed trait, the compiler ensures that 
  * all architecture types are exhaustively pattern-matched during hardware generation,
  * preventing unmapped configurations.
  *
  * ## Available Architectures
  * - **HardFloatFMA**: HardFloat BF16 Fused Multiply-Add. Medium area. Standard implementation.
  * - **FP32Addition**: HardFloat Multiply in BF16, HardFloat Add in FP32. High area.
  * - **CustomFMA**: Custom Fused Multiply-Add for (FP8*FP8)+BF16 operation. Low area.
  */
sealed trait PEArchitecture {
  def getMulWidth(): Int
  def getAddWidth(): Int
  def formatMul(raw: UInt): UInt
  def formatAdd(raw: UInt): UInt
  def formatOut(raw: UInt): UInt
}

/** Companion object containing the supported architectures for a Processing Element (PE). */
object PEArchitecture {

  /** HardFloat BF16 Fused Multiply-Add (FMA) module. */
  case object HardFloatFMA extends PEArchitecture {
    def getMulWidth(): Int = AtlasFPType.BF16.recodedWidth
    def getAddWidth(): Int = AtlasFPType.BF16.recodedWidth
    def formatMul(raw: UInt): UInt =
      FPUtils.convertHardFloat(FPUtils.recodeIEEEToHardFloat(raw, AtlasFPType.E4M3), AtlasFPType.E4M3, AtlasFPType.BF16)
    def formatAdd(raw: UInt): UInt =
      FPUtils.recodeIEEEToHardFloat(raw, AtlasFPType.BF16)
    def formatOut(raw: UInt): UInt =
      FPUtils.decodeHardFloatToIEEE(raw, AtlasFPType.BF16)
  }

  /** HardFloat Multiply in BF16, conversion to FP32, and addition in FP32. */
  case object FP32Addition extends PEArchitecture {
    def getMulWidth(): Int = AtlasFPType.BF16.recodedWidth
    def getAddWidth(): Int = AtlasFPType.FP32.recodedWidth
    def formatMul(raw: UInt): UInt =
      FPUtils.convertHardFloat(FPUtils.recodeIEEEToHardFloat(raw, AtlasFPType.E4M3), AtlasFPType.E4M3, AtlasFPType.BF16)
    def formatAdd(raw: UInt): UInt =
      FPUtils.convertHardFloat(FPUtils.recodeIEEEToHardFloat(raw, AtlasFPType.BF16), AtlasFPType.BF16, AtlasFPType.FP32)
    def formatOut(raw: UInt): UInt =
      FPUtils.decodeHardFloatToIEEE(FPUtils.convertHardFloat(raw, AtlasFPType.FP32, AtlasFPType.BF16), AtlasFPType.BF16)
  }

  /** Custom Fused Multiply-Add for (FP8*FP8)+BF16 operation. */
  case object CustomFMA extends PEArchitecture {
    def getMulWidth(): Int = AtlasFPType.E4M3.ieeeWidth
    def getAddWidth(): Int = AtlasFPType.BF16.ieeeWidth
    def formatMul(raw: UInt): UInt = raw
    def formatAdd(raw: UInt): UInt = raw
    def formatOut(raw: UInt): UInt = raw
  }
}

/**
  * Parameters for the systolic-array MXU (mxu0).
  *
  * Composes shared [[MxuParams]] and adds PE architecture selection.
  * Convenience aliases preserve the prior API surface.
  */
case class SystolicArrayParams(
  mxu:       MxuParams      = MxuParams(),
  peArch:    PEArchitecture = PEArchitecture.CustomFMA,
  fmaStages: Int            = 0,
) {
  require(mxu.arrayRows > 1, s"Systolic array must have >1 rows, got ${mxu.arrayRows}")
  require(mxu.arrayCols > 1, s"Systolic array must have >1 cols, got ${mxu.arrayCols}")
  require(fmaStages >= 0 && fmaStages <= 2, s"fmaStages must be 0..2, got $fmaStages")
  require(fmaStages == 0 || peArch == PEArchitecture.CustomFMA,
    s"fmaStages>0 is only supported with PEArchitecture.CustomFMA, got $peArch")

  // ── Convenience aliases ──
  val rows:    Int         = mxu.arrayRows
  val cols:    Int         = mxu.arrayCols
  val accSize: Int         = mxu.accumBufferRows
  val inT:     AtlasFPType = mxu.inputFmt
  val outT:    AtlasFPType = mxu.outputFmt

  override def toString: String =
    s"SystolicArrayParams(mxu=$mxu, peArch=$peArch)"
}
