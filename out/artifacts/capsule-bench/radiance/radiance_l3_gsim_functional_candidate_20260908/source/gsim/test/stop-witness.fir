FIRRTL version 3.3.0
circuit StopWitness :
  module StopWitness :
    input clock : Clock
    input reset : UInt<1>

    when reset :
      stop(clock, UInt<1>(0h1), 1) : failure
    when not(reset) :
      stop(clock, UInt<1>(0h1), 0) : stop
