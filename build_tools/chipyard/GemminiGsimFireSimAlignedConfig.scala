// Add this file, unchanged, at
// generators/firechip/chip/src/main/scala/GemminiGsimFireSimAlignedConfig.scala
// in a clean descendant of Chipyard 009e85b05ca817a15fcd6c3fe4eedd86cc2a3617.
// Keep the Gemmini gitlink at 63f0b68a68f185a016a2993f2fb32e1661ea9e6c.
package firechip.chip

import org.chipsalliance.cde.config.Config

// Use the FireSim chip settings, including 16 GiB target memory, SerialTL width,
// bus frequencies, peripherals, and IO cell models. The two leftmost binders
// replace the harness clock and serial-clock paths that gSIM cannot simulate as
// blackboxes. TestHarness needs a block-device model for the FireSim chip port.
class GemminiGsimFireSimAlignedConfig extends Config(
  new chipyard.WithGsimSerialTLClockPunchthrough ++
  new chipyard.WithGsimHarnessClockInstantiator ++
  new chipyard.harness.WithSimBlockDevice ++
  new WithDefaultFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.GemminiRocketConfig)
