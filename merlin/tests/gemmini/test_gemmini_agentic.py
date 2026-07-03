"""Agentic kernel generation (Claude Code CLI, Opus) — gate logic + held-out certification.

The live generation loop (which calls `claude -p`) is exercised by
`merlin.targetgen.agent.kernel_slot.generate_kernel` and is not run in CI (slow/networked).
What CI verifies deterministically:
  - the gate logic (cheat scan, prompt assembly) — no toolchain;
  - the *committed* agent-generated kernel certifies bit-exact on the HELD-OUT shapes it never
    saw (C4 multi-tile, C4e padded, C5 reuse) — spike-gated. That is the headline result,
    pinned as a regression.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import importlib.util
from pathlib import Path

import pytest

from merlin.runtime import outputs_match, reference_outputs, simulate
from merlin.runtime.backends import gemmini
from merlin.targetgen.eval.gemmini_conformance import build
from merlin.targetgen.agent import kernel_slot

REPO = repo_root()
AGENT_KERNEL = REPO / "merlin/experiments/gemmini_cert/agent_generated/kernel_codegen.py"

VISIBLE = ("C0", "C1")
HELDOUT = ("C4", "C4e", "C5")


def test_cheat_scan_blocks_peeking():
    assert kernel_slot._scan_cheat("def generate_driver(cb): return ''") == []
    assert "reference_outputs" in kernel_slot._scan_cheat("x = reference_outputs(cb)")
    assert "import numpy" in kernel_slot._scan_cheat("import numpy as np")


def test_prompt_excludes_golden():
    """The agent prompt carries example cb STRUCTURE but never reference outputs."""
    prompt = kernel_slot.build_prompt(VISIBLE, feedback=None)
    assert "generate_driver" in prompt and "RES_PACK" in prompt
    assert "reference_outputs" in prompt  # only as a forbidden-token rule
    # the prompt must instruct against peeking / hardcoding the golden
    low = prompt.lower()
    assert "hardware only" in low and "do not" in low


def _load_agent_generate_driver():
    spec = importlib.util.spec_from_file_location("agent_kernel_committed", AGENT_KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_driver


@pytest.mark.skipif(not AGENT_KERNEL.is_file(), reason="agent kernel artifact not present")
def test_agent_kernel_no_cheat():
    assert kernel_slot._scan_cheat(AGENT_KERNEL.read_text()) == []


@pytest.mark.skipif(not gemmini.available("spike"), reason="spike-gemmini unavailable")
@pytest.mark.skipif(not AGENT_KERNEL.is_file(), reason="agent kernel artifact not present")
@pytest.mark.parametrize("rung", VISIBLE + HELDOUT)
def test_agent_kernel_certifies(rung, tmp_path):
    """The committed agent-generated kernel is three-way bit-exact, including held-out shapes."""
    gen = _load_agent_generate_driver()
    cb = build(rung)
    src = gen(cb)
    res = gemmini.run_command_buffer(cb, workdir=tmp_path, simulator="spike",
                                     timeout=180, driver_src=src)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb)
    assert res["outputs"] == simulate(cb)["outputs"]
    assert outputs_match(res["outputs"], reference_outputs(cb))
