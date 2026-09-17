"""Compile machinery behind ``merlin-compile``: bundles, the host lane, capacity, and mesh execution.

``merlin.compile_cli`` is the front door -- ``compile_rvv``, ``compile_model``, ``compile_oot`` and the CLI's
``main`` -- and re-exports every name defined here, so ``merlin.compile_cli.<name>`` keeps working for
callers. A re-export is a second binding, though: patching it does not reach the callers in the module
that DEFINES the name, which resolve it in their own namespace. Tests therefore patch a name where it is
defined; ``merlin/tests/infra/test_compile_cli_patch_targets.py`` fails on a patch aimed at a re-export.
"""
