"""Merlin runtime execution backends.

The Python simulator lives in :mod:`merlin.runtime.simulator`; this package holds the
backends that run the same command buffers on real ISAs/simulators:

- :mod:`spike` — bare-metal multicore RVV execution on the spike ISA simulator
  (chipyard toolchain), with the same reference-equality correctness gate.
- :mod:`rvv_codegen` — command buffer -> C driver around the hand-written RVV kernel.
- :mod:`vcs` — replay the same ELF on a pre-built Saturn VCS RTL simulator (gated).
"""
