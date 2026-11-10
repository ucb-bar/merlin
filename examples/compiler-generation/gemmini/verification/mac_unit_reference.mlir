// Authored verification reference for the source-selected MacUnit only.
// Treat each input as a signed 8-bit integer; multiply in 20 bits, add the
// low 20 bits of the accumulator, and return the 20-bit modular result.
hw.module @MacUnitReference(in %io_in_a: i8, in %io_in_b: i8,
                            in %io_in_c: i32, out io_out_d: i20) {
  %asign = comb.extract %io_in_a from 7 : (i8) -> i1
  %bsign = comb.extract %io_in_b from 7 : (i8) -> i1
  %aextend = comb.replicate %asign : (i1) -> i12
  %bextend = comb.replicate %bsign : (i1) -> i12
  %a20 = comb.concat %aextend, %io_in_a : i12, i8
  %b20 = comb.concat %bextend, %io_in_b : i12, i8
  %product = comb.mul %a20, %b20 : i20
  %acc = comb.extract %io_in_c from 0 : (i32) -> i20
  %result = comb.add %product, %acc : i20
  hw.output %result : i20
}
