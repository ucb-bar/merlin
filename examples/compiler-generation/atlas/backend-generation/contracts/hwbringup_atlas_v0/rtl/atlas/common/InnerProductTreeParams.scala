package atlas.common

import chisel3._
import chisel3.util._
import sp26FPUnits.AtlasFPType
import sp26FPUnits.E4M3ProdFmt
import atlas.mxu.MxuParams

case class InnerProductTreeParams(
  mxu:           MxuParams   = MxuParams(),
  accumIntWidth: Int         = 0,
  pipelineCuts:  Set[Int]    = Set(1), // default: 2-stage pipelined
) {
  require(pipelineCuts.forall(c => c >= 0 && c <= 3),
    s"pipelineCuts must be in {0,...,3}, got $pipelineCuts")
    
  require(mxu.arrayRows > 1, s"Inner product trees must have >1 rows, got ${mxu.arrayRows}")
  require(mxu.arrayCols > 1, s"Inner product trees must have >1 cols, got ${mxu.arrayCols}")

  // ── Convenience aliases (match prior API so downstream code compiles) ──
  val numLanes: Int = mxu.arrayCols
  val vecLen:   Int = mxu.arrayRows
  val tileRows: Int = mxu.accumBufferRows

  val inputFmt:  AtlasFPType = mxu.inputFmt
  val outputFmt: AtlasFPType = mxu.outputFmt
  val accumFmt:  AtlasFPType = mxu.accumFmt
  val mulFmt                 = E4M3ProdFmt

  // ── Anchor-accumulation derived widths ──
  val anchorHeadroom: Int = {
    val total = vecLen + 1
    BigInt(total).bitLength + 1
  }

  val intWidth:     Int = if (accumIntWidth > 0) accumIntWidth
                          else mulFmt.sigWidth + anchorHeadroom + 17
  val expWorkWidth: Int = Seq(inputFmt.expWidth, mulFmt.expWidth, outputFmt.expWidth).max + 4

  // ── Pipeline ──
  val numPipeCuts: Int = pipelineCuts.size
  val latency:     Int = numPipeCuts + 1
  val shiftBits:   Int = log2Ceil(intWidth + 1)
  val tileRowBits: Int = log2Ceil(tileRows)

  def isOriginalConfig: Boolean =
    numLanes == 32 && vecLen == 32 && tileRows == 32

  override def toString: String = {
    val cutsStr = if (pipelineCuts.isEmpty) "none"
                  else pipelineCuts.toSeq.sorted.mkString(",")
    s"InnerProductTreeParams(mxu=$mxu, intW=$intWidth, headroom=$anchorHeadroom, cuts={$cutsStr})"
  }
}

object InnerProductTreeParams {
  def withPipelineDepth(
    depth: Int,
    base:  InnerProductTreeParams = InnerProductTreeParams()
  ): InnerProductTreeParams = {
    require(depth >= 1 && depth <= 5)
    val cuts: Set[Int] = depth match {
      case 1 => Set.empty
      case 2 => Set(1)
      case 3 => Set(0, 2)
      case 4 => Set(0, 1, 2)
      case 5 => Set(0, 1, 2, 3)
    }
    base.copy(pipelineCuts = cuts)
  }
}
