kernel hoist_fixture(input: i8[17] read, output: i8[23] write)
  attrs {fixture=phase1}
  unknown(slot=0, a=0, b=0)
  pair(slot=0, a=3, b=5)
  for i in 0..3:
    work(slot=0, a=0, b=0)
    work(slot=0, a=0, b=0)
  work(slot=0, a=0, b=0)
