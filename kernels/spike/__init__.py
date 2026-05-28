"""Spike simulator runner for RISC-V kernel objects.

`runner.py` compiles an MLIR fixture via `iree-compile` and executes the
resulting RISC-V ELF under `spike` with `pk`. Used by the `./merlin spike`
subcommand to validate Gemmini / Saturn-OPU kernels functionally.
"""
