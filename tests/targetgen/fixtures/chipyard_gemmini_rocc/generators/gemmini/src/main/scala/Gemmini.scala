// Synthetic fixture.
package gemmini

import freechips.rocketchip.tile.LazyRoCC
import freechips.rocketchip.tile.OpcodeSet

class Gemmini extends LazyRoCC(OpcodeSet.custom3) {
  // synthetic body
}

class GemminiConfig extends Config((site, here, up) => {
  case Object => Some(new GemminiBlock())
})
