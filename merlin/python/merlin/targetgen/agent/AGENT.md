# AGENT.md — merlin/python/merlin/targetgen/agent

## Purpose

Agentic target-generation slots (Claude Code CLI dispatch + gated kernel synthesis).

## Modules

- `claude_cli.py` — Dispatch the Claude Code CLI as the agent for a generation slot.
- `gemmini_kernel_slot.py` — Kernel slot: agent synthesizes the Gemmini command-buffer -> C kernel, gated by the oracle.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
