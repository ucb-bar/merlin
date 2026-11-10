# ModelBlaster adapter

The installed entry point is `runner`. It may read host-local ModelBlaster and
FireSim configuration, but must not mutate the shared Chipyard checkout or
submit work in `preflight`. The queue is the default for shared FPGA execution.
