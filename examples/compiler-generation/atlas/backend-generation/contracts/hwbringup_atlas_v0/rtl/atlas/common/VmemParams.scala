// ============================================================================
// VmemParams.scala — VMEM parameters and port bundles (architecture).
//
// Atlas uses a block-banked VMEM. Each bank owns one contiguous power-of-two
// `bankBytes` range, and the default Atlas memory map places this window at
// `AtlasMemMap.VMEM_BASE` with size `AtlasMemMap.VMEM_SIZE`.
//
// Banking on power-of-two bank windows of line address:
//   bank index        = lineAddr >> bankLineAddrBits
//   line within bank  = lineAddr(bankLineAddrBits-1, 0)
//
// This keeps each bank as one contiguous VMEM window while allowing a
// non-power-of-two number of banks, as long as the per-bank SRAM macro size
// remains a power of two.
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._
import atlas.scalar.AtlasMemMap

case class VmemParams(
  lineWidthBits: Int    = 256,
  capacityBytes: Int    = AtlasMemMap.VMEM_SIZE,
  base:          BigInt = BigInt(AtlasMemMap.VMEM_BASE),
  beatBytes:     Int    = 32,
  numBanks:      Int    = AtlasMemMap.VMEM_NUM_BANKS
) {
  private def isPowerOfTwo(x: Int): Boolean = x > 0 && (x & (x - 1)) == 0

  require(capacityBytes > 0)
  require(numBanks > 0)
  require(lineWidthBits % 8 == 0)

  val lineBytes: Int        = lineWidthBits / 8
  val bankBytes: Int        = capacityBytes / numBanks
  val linesPerBank: Int     = bankBytes / lineBytes
  val numLines: Int         = linesPerBank * numBanks
  val bankIdBits: Int       = log2Ceil(numBanks)
  val bankLineAddrBits: Int = log2Ceil(linesPerBank)
  val lineAddrBits: Int     = bankLineAddrBits + bankIdBits
  val wordWidth: Int        = 32
  val wordsPerLine: Int     = lineWidthBits / wordWidth
  val wordOffBits: Int      = log2Ceil(wordsPerLine)
  val numWords: Int         = numLines * wordsPerLine
  val wordAddrBits: Int     = log2Ceil(numWords)
  val byteAddrBits: Int     = log2Ceil(capacityBytes)

  // ── Banking ──
  val lineOffBits: Int      = log2Ceil(lineBytes)

  // ── Aliases ──
  def lineWidth: Int = lineWidthBits
  def sizeBytes: Int = capacityBytes

  require(bankBytes * numBanks == capacityBytes)
  require(bankBytes % lineBytes == 0)
  require(isPowerOfTwo(bankBytes))

  /** Extract bank index from a line address. */
  def getBankIdx(lineAddr: UInt): UInt =
    (lineAddr >> bankLineAddrBits)(bankIdBits - 1, 0)
  /** Extract bank-local line address from a line address. */
  def getBankAddr(lineAddr: UInt): UInt =
    lineAddr(bankLineAddrBits - 1, 0)
}

// ============================================================================
// Port bundles
// ============================================================================

/** Read request with pre-decomposed bank address. */
class VmemLineReadPort(p: VmemParams) extends Bundle {
  val bankIdx  = UInt(p.bankIdBits.W)
  val bankAddr = UInt(p.bankLineAddrBits.W)
}

/** Full-line write (no masking). Used by DMA and VSTORE. */
class VmemLineWritePort(p: VmemParams) extends Bundle {
  val bankIdx  = UInt(p.bankIdBits.W)
  val bankAddr = UInt(p.bankLineAddrBits.W)
  val data     = UInt(p.lineWidthBits.W)
}

/** Byte-masked write. Used by LSU scalar stores. */
class MaskedVmemLineWritePort(p: VmemParams) extends Bundle {
  val bankIdx  = UInt(p.bankIdBits.W)
  val bankAddr = UInt(p.bankLineAddrBits.W)
  val data     = UInt(p.lineWidthBits.W)
  val mask     = Vec(p.lineBytes, Bool())
}
