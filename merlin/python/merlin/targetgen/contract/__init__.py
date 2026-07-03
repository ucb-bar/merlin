"""Experiment-ABI contract layer.

This package holds the Merlin-side machinery for the repo-independent *experiment ABI*:
the `merlin_iface` interface-grammar emitter/parser (:mod:`interface_emit`), the MLIR
toolchain resolver (:mod:`toolchain`), and (added by later phases) the generic
out-of-tree package runner. The contract bundle itself (the frozen, versioned spec a
package author reads) lives under ``merlin/contract/`` (repo-root-relative; no compat symlink).
"""
from __future__ import annotations
