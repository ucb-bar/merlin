// ============================================================================
// MregParams.scala — Parameters and port bundles for the matrix register file.
//
// The spec defines matrix registers as "m registers" (mreg):
//   NUM_MREG       = 64   matrix registers
//   NUM_MREG_BANKS = 32   physical SRAM banks
//   MREG_ROWS      = 32   rows per register
//   MREG_ROW_BYTES = 32   bytes per row  (256 bits = 32 × fp8)
//   MREG_BYTES     = 1024 bytes per register
//
// Physical layout:
//   - Each SRAM bank is 32 B wide and 64 entries deep.
//   - Architectural m_i and m_{i+32} share bank i.
//   - Rows 0..31 hold the low architectural register; rows 32..63 hold
//     the high architectural register.
//
// Multiple functional units (MXU-0, MXU-1, VPU, XLU, LSU/DMA) access the
// register file through dedicated read and write ports.
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._

/** Design-time parameters for the matrix register file, named to match the
  * architectural spec.
  *
  * @param mregRowBytes  Bytes per row (spec: `MREG_ROW_BYTES`, default 32).
  * @param mregRows      Rows per matrix register (spec: `MREG_ROWS`, default 32).
  * @param numMreg       Number of matrix registers (spec: `NUM_MREG`, default 64).
  * @param numMregBanks  Number of physical SRAM banks.
  * @param numReadPorts  Number of independent read ports.
  * @param numWritePorts Number of independent write ports.
  */
case class MregParams(
    mregRowBytes: Int = 32, // MREG_ROW_BYTES — 32 bytes = 256 bits per row.
    mregRows:     Int = 32, // MREG_ROWS      — 32 rows per matrix register.
    numMreg:      Int = 64, // NUM_MREG       — 64 matrix registers.
    numMregBanks: Int = 32, // Physical SRAM banks; m_i aliases m_{i+32}.
    numReadPorts: Int = 8,
    numWritePorts: Int = 6
) {
  require(mregRows > 0 && (mregRows & (mregRows - 1)) == 0,
    s"mregRows($mregRows) must be a positive power of two")
  require(numMregBanks > 0 && (numMregBanks & (numMregBanks - 1)) == 0,
    s"numMregBanks($numMregBanks) must be a positive power of two")
  require(numMreg == 2 * numMregBanks,
    s"numMreg($numMreg) must be exactly twice numMregBanks($numMregBanks)")

  /** Bit-width of one SRAM row (MREG_ROW_BYTES × 8). */
  val mregRowBits: Int = mregRowBytes * 8

  /** Bits needed to address a row within a single register. */
  val mregRowAddrBits: Int = log2Ceil(mregRows)

  /** Bits needed to select a matrix register. */
  val mregIdBits: Int = log2Ceil(numMreg)

  /** Rows per physical SRAM bank: low half for m_i, high half for m_{i+32}. */
  val mregBankRows: Int = 2 * mregRows

  /** Bits needed to select a physical SRAM bank. */
  val mregBankIdBits: Int = log2Ceil(numMregBanks)

  /** Bits needed to address a row within one physical SRAM bank. */
  val mregBankRowAddrBits: Int = log2Ceil(mregBankRows)

  /** Total bytes per matrix register (MREG_BYTES). */
  val mregBytes: Int = mregRowBytes * mregRows

  /** Physical SRAM bank for an architectural mreg id. */
  def physicalBank(mregId: UInt): UInt =
    mregId(mregBankIdBits - 1, 0)

  /** Physical SRAM row for an architectural mreg id and logical row. */
  def physicalRow(mregId: UInt, row: UInt): UInt =
    Cat(mregId(mregIdBits - 1, mregBankIdBits), row)
}

// ============================================================================
// Port bundles
// ============================================================================

/** Read request: selects one row from one matrix register.
  *
  * Wrapped in `Valid(...)` — when valid is asserted the SRAM performs a
  * synchronous read; the result appears one cycle later.
  */
class MregReadReq(p: MregParams) extends Bundle {
  val mregId = UInt(p.mregIdBits.W)       // Which matrix register to read.
  val row    = UInt(p.mregRowAddrBits.W)  // Which row within that register.
}

/** Write request: addresses one row and carries the write payload. */
class MregWriteReq(p: MregParams) extends Bundle {
  val mregId = UInt(p.mregIdBits.W)       // Target matrix register.
  val row    = UInt(p.mregRowAddrBits.W)  // Target row within the register.
  val data   = UInt(p.mregRowBits.W)      // Data to write.
}
