FIRRTL version 3.3.0
circuit PrintfFilter :
  module PrintfFilter :
    input clock : Clock
    input reset : UInt<1>

    when not(reset) :
      printf(clock, UInt<1>(0h1), "diagnostic noise\n") : debug
