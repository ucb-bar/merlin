"""Generate the merlin-target-<name>/ repository skeleton and its artifacts.

Each generator returns a list of :class:`merlin.common.artifacts.Artifact`. The pipeline
writes them under the output directory and then backfills AGENT.md coverage.
"""

from __future__ import annotations

# ``runtime_adapter`` deliberately is NOT imported eagerly.  It renders a callable route to Merlin's
# reference/simulator and is deny-masked in the assisted authoring sandbox.  Eagerly importing it made
# ``import merlin.targetgen.generate`` fail even though every allowed generator was present.  Host-side
# callers that explicitly request ``from merlin.targetgen.generate import runtime_adapter`` retain normal
# Python submodule-import behaviour; the safe package surface no longer loads the denied route as a side
# effect.
#
# ``oot_package`` IS imported eagerly and is not deny-masked. It renders the generated package's own
# self-registering backend, whose execution entries DECLINE -- it carries no import of
# ``merlin.runtime``'s reference or simulator, so it is not a callable route to the oracle. That
# absence is the load-bearing property, and a test asserts it rather than this comment.
from . import llvm_plan, mlir_scaffold, oot_package, target_repo, xdsl, zephyr_module

__all__ = [
    "target_repo",
    "xdsl",
    "mlir_scaffold",
    "zephyr_module",
    "llvm_plan",
    "oot_package",
]
