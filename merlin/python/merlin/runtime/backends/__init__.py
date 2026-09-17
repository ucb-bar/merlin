"""Merlin runtime execution backends.

The Python simulator lives in :mod:`merlin.runtime.simulator`; this package holds the
backends that run the same command buffers on real ISAs/simulators. They are addressed by
**target class** (CPU / GPU / NPU), not by name — see the registry + shared console protocol in
:mod:`merlin.runtime.backends.base` (``list_backends`` / ``get_backend`` / ``class_of``). The
per-instance modules registered there share the ``Backend`` protocol (resolve toolchain →
``compile_command_buffer`` → ``run_elf`` → ``parse_output`` → ``run_command_buffer``); a new backend
is one registry entry. The in-tree reference instances are the CPU/RVV backends (``spike``,
``saturn_vec``, ``rvv_codegen``), the whole-model runners (``spike_model``, ``zephyr_model``), the
matmul-routing attribution backends (``xnnpack_board`` / ``openblas_board`` / ``ours_board`` /
``xnnpack_host``), and the NPU/GPU reference instances (``gemmini``, ``muon``).
"""
