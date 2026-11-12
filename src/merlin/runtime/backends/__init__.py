"""Merlin runtime execution backends.

The Python simulator lives in :mod:`merlin.runtime.simulator`; this package holds the
backends that run the same command buffers on real ISAs/simulators. They are addressed by
**target class** (CPU / GPU / NPU), not by name — see the registry + shared console protocol in
:mod:`merlin.runtime.backends.base` (``list_backends`` / ``get_backend`` / ``class_of``). The
per-instance modules registered there share the ``Backend`` protocol (resolve toolchain →
``compile_command_buffer`` → ``run_elf`` → ``parse_output`` → ``run_command_buffer``); a new backend
is one registry entry. The in-tree instances are the CPU/RVV backends (``spike``, ``rvv_codegen``),
the whole-model runners (``spike_model``, ``zephyr_model``) and the outlined int8 matmul route
(``outlined_int8_board``); accelerator (NPU/GPU)
instances register from the selected target's support provider.
"""
