// Synthetic fixture.
package fftgen

import freechips.rocketchip.regmapper.TLRegisterRouter
import freechips.rocketchip.regmapper.RegField

class FFTGenerator extends LazyModule {
  val node = TLRegisterRouter(0x4000, "fft", Seq("ucb-bar,fft"), 0)
}
