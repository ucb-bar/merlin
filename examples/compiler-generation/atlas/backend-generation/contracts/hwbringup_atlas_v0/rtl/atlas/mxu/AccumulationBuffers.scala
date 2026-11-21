// ============================================================================
// AccumulationBuffers.scala — Dual BF16 accumulation-buffer storage.
//
// Spec parameters (per buffer):
//   ACCUM_BUFFER_ROWS      = 32   rows
//   ACCUM_BUFFER_COLS_BF16 = 32   BF16 columns
//   ACCUM_BUFFER_BYTES     = 2048 bytes  (32 × 32 × 2)
//
// Two identically-sized buffers are selected by the `accSel` bit in each
// bundle.  The module is parameterized by MxuParams so both mxu0 (systolic
// array) and mxu1 (inner-product tree) share the same implementation.
//
// Three independent access paths:
//
//   1. Compute write / read — the MXU datapath writes result rows and reads
//      partial sums during matrix-multiply operations.
//
//   2. Load (push path) — the sequencer writes BF16 data into a buffer
//      after optional FP8→BF16 dequantization (dequant lives in sequencer).
//
//   3. Store (pop path) — the sequencer reads raw BF16 rows out of a buffer
//      for optional BF16→FP8 quantization (quant lives in sequencer).
//
// All (de)quantization happens in the sequencer, not here.
//
// Port inference:
//   Each buffer is backed by a single SyncReadMem with exactly one read site
//   and one write site, so it maps to a 1R1W SRAM macro. Mutual exclusion
//   between the two write sources (compute / load) and the two read sources
//   (compute / store) is enforced by assertions below; the write- and
//   read-mux stages rely on those assertions to be safe.
// ============================================================================

package atlas.mxu

import chisel3._
import chisel3.util._

/** Dual accumulation-buffer storage for one MXU.
  *
  * @param p  MXU geometry parameters (provides buffer dimensions and FP format).
  */
class AccumulationBuffers(p: MxuParams) extends Module {

  // Shorthand for the row type used throughout this module.
  private val numCols  = p.accumBufferColsBF16
  private val colWidth = p.accumFmt.ieeeWidth
  private type RowT    = Vec[UInt]

  val io = IO(new Bundle {

    // ── Compute pipeline (MXU datapath ↔ accum buffer) ─────────────
    /** Write a BF16 result row from the compute pipeline. */
    val computeWriteReq = Flipped(Valid(new AccBufWriteReq(p)))
    /** Address for a compute-side partial-sum read. */
    val computeReadAddr = Input(new AccBufReadAddr(p))
    /** Pulse to request a compute-side partial-sum read. */
    val computeReadEn   = Input(Bool())
    /** Partial-sum row returned one cycle after `computeReadEn`. */
    val computeReadData = Output(Vec(numCols, UInt(colWidth.W)))

    // ── Load path (sequencer → accum buffer) ───────────────────────
    /** Write a BF16 row into the buffer (post-dequant push). */
    val loadReq = Flipped(Valid(new AccBufLoadReq(p)))

    // ── Store path (accum buffer → sequencer) ──────────────────────
    /** Address for a store/pop read-out. */
    val storeAddr   = Input(new AccBufStoreAddr(p))
    /** Pulse to request a store/pop read-out. */
    val storeReadEn = Input(Bool())
    /** Raw BF16 row returned one cycle after `storeReadEn`. */
    val storeData   = Output(Vec(numCols, UInt(colWidth.W)))
  })

  // ==========================================================================
  // Storage — one synchronous-read memory per buffer (1R1W)
  // ==========================================================================

  val buffer0 = SyncReadMem(p.accumBufferRows, Vec(numCols, UInt(colWidth.W)))
  val buffer1 = SyncReadMem(p.accumBufferRows, Vec(numCols, UInt(colWidth.W)))

  // ==========================================================================
  // Safety: compute-write and load must not target the same buffer
  // so each buffer only needs a single write port
  // ==========================================================================

  assert(!(io.computeWriteReq.valid && io.loadReq.valid &&
           io.computeWriteReq.bits.accSel === io.loadReq.bits.accSel),
    "AccumulationBuffers: compute write and load target the same buffer")

  // ==========================================================================
  // Safety: compute-read and store-read must not target the same buffer
  // so each buffer only needs a single read port
  // ==========================================================================

  assert(!(io.computeReadEn && io.storeReadEn &&
           io.computeReadAddr.accSel === io.storeAddr.accSel),
    "AccumulationBuffers: compute read and store read target the same buffer")

  // ==========================================================================
  // Unified write selection — one write request per buffer per cycle
  // Priority within a buffer: compute write over load
  // (same-buffer overlap is already forbidden by assertion)
  // ==========================================================================

  val write0IsCompute = io.computeWriteReq.valid && !io.computeWriteReq.bits.accSel
  val write0IsLoad    = io.loadReq.valid         && !io.loadReq.bits.accSel
  val write1IsCompute = io.computeWriteReq.valid &&  io.computeWriteReq.bits.accSel
  val write1IsLoad    = io.loadReq.valid         &&  io.loadReq.bits.accSel

  val write0En = write0IsCompute || write0IsLoad
  val write1En = write1IsCompute || write1IsLoad

  val write0Addr = Mux(write0IsCompute,
                       io.computeWriteReq.bits.rowIdx,
                       io.loadReq.bits.rowIdx)
  val write1Addr = Mux(write1IsCompute,
                       io.computeWriteReq.bits.rowIdx,
                       io.loadReq.bits.rowIdx)

  val write0Data = Mux(write0IsCompute,
                       io.computeWriteReq.bits.data,
                       io.loadReq.bits.data)
  val write1Data = Mux(write1IsCompute,
                       io.computeWriteReq.bits.data,
                       io.loadReq.bits.data)

  when(write0En) { buffer0.write(write0Addr, write0Data) }
  when(write1En) { buffer1.write(write1Addr, write1Data) }

  // ==========================================================================
  // Unified read selection — one read request per buffer per cycle
  // Priority within a buffer: compute read over store read
  // (same-buffer overlap is already forbidden by assertion)
  // ==========================================================================

  val read0En   = Wire(Bool())
  val read0Addr = Wire(UInt(log2Ceil(p.accumBufferRows).W))
  val read1En   = Wire(Bool())
  val read1Addr = Wire(UInt(log2Ceil(p.accumBufferRows).W))

  val read0IsCompute = Wire(Bool())
  val read0IsStore   = Wire(Bool())
  val read1IsCompute = Wire(Bool())
  val read1IsStore   = Wire(Bool())

  read0IsCompute := io.computeReadEn && !io.computeReadAddr.accSel
  read0IsStore   := io.storeReadEn   && !io.storeAddr.accSel
  read1IsCompute := io.computeReadEn &&  io.computeReadAddr.accSel
  read1IsStore   := io.storeReadEn   &&  io.storeAddr.accSel

  read0En := read0IsCompute || read0IsStore
  read1En := read1IsCompute || read1IsStore

  read0Addr := Mux(read0IsCompute, io.computeReadAddr.rowIdx, io.storeAddr.rowIdx)
  read1Addr := Mux(read1IsCompute, io.computeReadAddr.rowIdx, io.storeAddr.rowIdx)

  // Latch which kind of read each buffer served in the previous cycle.
  // Use RegNext so the tag clears back to false when that buffer is idle;
  // RegEnable would leave a stale "was store/compute" tag behind and could
  // mis-route a later response from the other buffer.
  val read0WasCompute = RegNext(read0IsCompute, false.B)
  val read0WasStore   = RegNext(read0IsStore,   false.B)
  val read1WasCompute = RegNext(read1IsCompute, false.B)
  val read1WasStore   = RegNext(read1IsStore,   false.B)

  val data0 = buffer0.read(read0Addr, read0En)
  val data1 = buffer1.read(read1Addr, read1En)

  // ==========================================================================
  // Read response routing
  // ==========================================================================

  val zeroRow = Wire(Vec(numCols, UInt(colWidth.W)))
  zeroRow := VecInit(Seq.fill(numCols)(0.U(colWidth.W)))

  io.computeReadData := zeroRow
  io.storeData       := zeroRow

  when(read0WasCompute) {
    io.computeReadData := data0
  }.elsewhen(read1WasCompute) {
    io.computeReadData := data1
  }

  when(read0WasStore) {
    io.storeData := data0
  }.elsewhen(read1WasStore) {
    io.storeData := data1
  }
}
