"""Load a declarative per-target experiment descriptor — the target-parameterized replacement for the
gemmini-hardcoded experiment setup.

Per the derive-first rule, the hardware FACTS (ISA/opcode set, memory map, mesh DIM, arc model) are
DERIVED from the RTL by mlc (``rtl_backend.target_profile`` / ``mlc_bridge``), never hand-written. What a
run genuinely cannot derive — which RTL repo, which hardware-spec files every arm gets, which capsule
corpus to grade on, how the simulator runs — is the irreducible SETUP, declared in a small YAML
descriptor (``target_experiment.yaml`` beside the experiment). A new accelerator drops its own descriptor
and registers its RTL with mlc; no per-target code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common.paths import repo_root


@dataclass(frozen=True)
class TargetExperiment:
    """The declarative SETUP for one target's experiment (derivable facts are NOT here)."""
    target: str
    isa_headers: tuple[Path, ...]      # the shared hardware-spec headers given to ALL arms
    hwbringup_set: Path                # the shared RTL/ISA/README/example set given to ALL arms
    capsule_corpus: Path               # the corpus the arms author against + are graded on
    sim_via: str                       # how the simulator runs (e.g. "chipyard")
    rtl_via: str                       # how RTL facts are obtained (e.g. "mlc" — DERIVED, not declared)
    path: Path                         # the descriptor file this came from


def load_target_experiment(descriptor: str | Path) -> TargetExperiment:
    """Load + validate a ``target_experiment.yaml`` descriptor, resolving its repo-root-relative paths."""
    p = Path(descriptor)
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict) or not doc.get("target"):
        raise ValueError(f"{p}: not a target-experiment descriptor (missing 'target')")
    root = repo_root()
    hw = doc.get("hardware_spec") or {}
    return TargetExperiment(
        target=str(doc["target"]),
        isa_headers=tuple(root / h for h in (hw.get("isa_headers") or [])),
        hwbringup_set=root / hw["hwbringup_set"] if hw.get("hwbringup_set") else None,
        capsule_corpus=root / doc["capsule_corpus"] if doc.get("capsule_corpus") else None,
        sim_via=str((doc.get("toolchain") or {}).get("sim_via", "")),
        rtl_via=str((doc.get("rtl") or {}).get("via", "mlc")),
        path=p,
    )
