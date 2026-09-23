// ============================================================================
// DmaParams.scala — Configuration for the Atlas DMA engine.
//
// Naming follows the architectural spec:
//   DMA_CHANNELS = 8    → numChannels
//   DMA_ALIGN    = 32 B → beatBytes
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._

/** Parameters governing the DMA engine's capacity and data-path geometry.
  *
  * @param beatBytes        Bytes per TileLink beat; also the minimum transfer
  *                         granularity and alignment (spec: `DMA_ALIGN`, default 32).
  * @param tagBits          Bit-width of the TileLink source-ID field.
  *                         Must satisfy 2^tagBits ≥ maxInFlight.
  * @param numChannels      Number of independent DMA channels / command slots
  *                         (spec: `DMA_CHANNELS`, default 8).
  * @param channelIdBits    log2(numChannels); width of channel-index pointers.
  * @param maxInFlight      Maximum outstanding (un-acknowledged) TileLink beats.
  * @param maxTransferBytes Largest single DMA transfer in bytes (default 4 KiB).
  * @param name             Instance name used for TileLink node naming and debug output.
  */
case class DmaParams(
  beatBytes:        Int     = 32,    // DMA_ALIGN — 256-bit beats.
  numChannels:      Int     = 8,     // DMA_CHANNELS — 8 command slots.
  maxInFlight:      Int     = 64,    // Up to 64 beats on the wire at once.
  maxTransferBytes: Int     = 4096,  // 4 KiB maximum transfer.
  name:             String  = "dma"
) {
  val channelIdBits:    Int = log2Ceil(numChannels)
  val transferSizeBits: Int = log2Ceil(maxTransferBytes) + 1
  val tagBits:          Int = log2Ceil(maxInFlight)
  val queueSize:        Int = maxTransferBytes / beatBytes    // Queue size for pending stores.
}
