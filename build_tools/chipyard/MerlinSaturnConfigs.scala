package chipyard

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}

// ---------------------------------------------------------------------------
// Multicore Saturn-vectors SoCs.
//
// Every stock Saturn config in generators/saturn/chipyard/SaturnConfigs.scala is
// single-core (WithNShuttleCores(1) / WithNHugeCores(1)), and the only multi-tile
// SoC in HeteroConfigs.scala (GemminiAndOPUShuttleConfig) builds its vector unit on
// tile 1 only. Neither can host a multi-hart RVV workload.
//
// These configs give EVERY Shuttle tile its own Saturn vector unit:
// WithShuttleVectorUnit's `cores` parameter defaults to None, which its TilesLocated
// mapping reads as "build the vector unit on every ShuttleTileAttachParams", so the
// stock fragment already does what we need — only the core count changes.
//
// Parameters follow REFV256D128ShuttleConfig (vLen=256, dLen=128, refParams, 128-bit
// system bus, 16-byte tile beats), which matches the SpacemiT K1's VLEN=256 so a
// schedule tuned on the board transfers without a re-tune.
// ---------------------------------------------------------------------------

/** 2 Shuttle tiles, each with its own Saturn vector unit (vLen=256, dLen=128). */
class DualSaturnV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(2) ++
  new chipyard.config.AbstractConfig)

/** 4 Shuttle tiles, each with its own Saturn vector unit (vLen=256, dLen=128).
  * The default target for merlin's multicore-RVV Zephyr images. */
class MultiSaturnV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(4) ++
  new chipyard.config.AbstractConfig)
