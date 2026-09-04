# K1 CPU-host evidence

The paper board reported eight online SpacemiT X60 harts on 2026-08-30.  `/proc/cpuinfo` reported
`rv64imafdcv` with `zfh`, `zvfh`, the Zve subsets, and the standard bit-manipulation extensions.  `lscpu`
reported 32 KiB L1 data and instruction caches per hart and two 512 KiB L2 instances.

The Linux kernel does not delegate `cycle` to userspace.  K1 performance claims therefore use monotonic
wall time as the silicon authority; `rdtime` is retained as a corroborating fixed-frequency counter, not
renamed to cycles.  Every scored run must record CPU affinity, requested and actual execution mode, core
count, VLEN and its source, memory policy, warmups, individual observation times, and peak RSS.

Run `examples/probe.c` immediately before a campaign.  A compiler may specialize for VLEN=256 only if the
probe reports `vlenb=32`; otherwise it must use a dynamic-VL loop or fail closed.
