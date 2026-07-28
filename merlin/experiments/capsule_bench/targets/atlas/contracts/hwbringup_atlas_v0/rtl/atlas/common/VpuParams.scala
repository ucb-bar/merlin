package atlas.common

import sp26FPUnits._

case class VpuParams(
  BF16: AtlasFPType = AtlasFPType.BF16,
  wordWidth: Int = AtlasFPType.BF16.wordWidth,
  expWidth: Int = AtlasFPType.BF16.expWidth,
  sigWidth: Int = AtlasFPType.BF16.sigWidth,
  numLanes: Int = 16,
  tagWidth: Int = 16,
  numOpMod: Int = 8
) {
  require((numLanes > 1) && ((numLanes & (numLanes - 1)) == 0), s"VPU must have >1 lanes and must be power of 2, got $numLanes")
}

