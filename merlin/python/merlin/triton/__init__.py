"""Triton as a target-independent KERNEL FRONTEND to Merlin (not a per-target backend).

A standard ``@triton.jit`` kernel enters Merlin ABOVE the target-selection boundary:

    @triton.jit  --(stock triton wheel, ASTSource.make_ir)-->  TTIR
    TTIR         --(bridge: structural, xDSL, fail-closed)-->  linalg-on-tensors
                 --> the SAME contract -> schedule -> interface -> target -> runtime
                     pipeline every other Merlin payload descends

The consequence we are after: once Merlin learns to compile generic computation to a new target,
that target becomes programmable from Triton with ZERO Triton-specific work — no
``Triton<Target>Backend``, no per-target Triton plan, no extra TargetGen prompt.

That property is only real if this package stays target-blind, so it is gate-enforced:
``merlin/tests/infra/test_triton_target_independence.py`` asserts no target-name literal, no
import of a target-specific module, and no ``import re`` (structural parsing only) anywhere under
this package. The target is a PARAMETER threaded in from the descriptor/contract; if supporting a
second target ever requires editing a file here, the abstraction failed and the fix belongs in
Merlin's shared lowering, not here.

Design + the invariants it must uphold: ``docs/design/triton_frontend.md``.
"""
from __future__ import annotations

__all__: list[str] = []
