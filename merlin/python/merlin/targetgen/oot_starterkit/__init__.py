"""OOT starter kit — hw-agnostic, answer-free framework plumbing for authoring an MLIR OOT backend.

The abc4 analysis showed agents rebuilt ~570 LOC of target-INDEPENDENT plumbing (input-grammar parser,
command-buffer serializer, dialect/entrypoint boilerplate) that the framework can amortize. This kit
provides exactly that plumbing + generic compiler transforms — and NOTHING target-specific:

  * parse_interface()  — wraps the existing contract.interface_emit parser (fixed merlin_iface grammar)
  * CommandBufferBuilder — emits SCHEMA-VALID command_buffer.json (fixed frozen ABI)
  * transforms.im2col / tile_to_dim — GENERIC, target-agnostic compiler transforms the agent CALLS
  * scaffold/ — a STRUCTURE-ONLY package skeleton (empty dialect + 4 entrypoints + verifier hooks; NO
    op lowering — the agent writes all lowering itself)

ANTI-CHEAT: contains only the contract-fixed input/output formats + textbook generic transforms. No target
funct table, no goldens, no target-specific op lowering. Identical for every arm that's allowed it. The
agent authors every target lowering (and, for the merlin arm, the ISA encoding).
"""
from .iface import parse_interface          # noqa: F401
from .cmdbuf import CommandBufferBuilder     # noqa: F401
from . import transforms                     # noqa: F401
from . import verify                         # noqa: F401  (C++-MLIR-verifier-equivalent for the Python path)
from .verify import validate, verify_module  # noqa: F401
