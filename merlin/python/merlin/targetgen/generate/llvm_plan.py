"""Generate the llvm/ scaffold from an llvm_extension_plan.

Out-of-tree first: emit the plan plus README placeholders for td/, patches/, tests/. No real
backend patches are produced -- a fork is only ever justified by the plan's fork_triggers.
"""
from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact, yaml_artifact


def generate(llvm_extension_plan: dict[str, Any]) -> list[Artifact]:
    """Return llvm/ artifacts for the given llvm_extension_plan."""
    target = llvm_extension_plan.get("target", "target")
    fork = llvm_extension_plan.get("requires_llvm_fork", False)
    strategy = llvm_extension_plan.get("initial_strategy", "runtime_calls_or_command_buffer")
    return [
        yaml_artifact("llvm/llvm_extension_plan.yaml", llvm_extension_plan,
                      header="Generated LLVM extension plan (llvm_extension_plan.schema.yaml)."),
        Artifact("llvm/td/README.md",
                 f"# LLVM TableGen fragments — {target}\n\n"
                 f"requires_llvm_fork: `{fork}` · initial_strategy: `{strategy}`\n\n"
                 "Out-of-tree `.td` fragments live here once a target needs real "
                 "instruction/register/codegen support. Empty during the MVP.\n"),
        Artifact("llvm/patches/README.md",
                 f"# LLVM patch series — {target}\n\n"
                 "Reviewable patch files against a pinned `third_party/llvm-project` go here, "
                 "with a `series` file. **None during the MVP** — do not fork LLVM before "
                 "target-dialect + simulator validation passes.\n"),
        Artifact("llvm/tests/README.md",
                 f"# LLVM lit/codegen/asm tests — {target}\n\n"
                 "Tests that expect a patched LLVM live here. Empty during the MVP.\n"),
    ]
