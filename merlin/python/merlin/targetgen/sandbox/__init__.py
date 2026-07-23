"""Shared, descriptor+manifest-driven agentic bwrap sandbox.

Promotes the per-experiment gemmini sandbox to target-agnostic infra: a new target gets a correct,
continuously-guarded sandbox from its ``target_experiment.yaml`` (+ capability manifest) with ZERO copied
scripts. Routing is by compute-unit KIND / sim FAMILY (never a target name):

  * :mod:`.answer_surfaces` — the DERIVED answer-surface mask set (goldens/hidden/prior/oracle/grader/
    memory) + the single declared oracle/grader registry + the coverage guard's audit tokens.
  * :mod:`.toolchain`       — the legit tools bound back, universal + the descriptor's ``sim_via`` family,
    cross-checked by ``kind`` via :mod:`merlin.targetgen.families`.
  * :mod:`.bwrap`           — deny-by-default argv assembly + the hermetic mount-replay coverage proof.

:func:`build_sandbox` is the one entry point; the experiment ``scripts/*`` are thin delegators onto it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.targetgen.sandbox import bwrap, toolchain
from merlin.targetgen.sandbox.answer_surfaces import (AnswerSurface, answer_surfaces, audit_tokens,
                                                      golden_files)
from merlin.targetgen.sandbox.toolchain import ToolProbe, required_tool_probes
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment

__all__ = ["Sandbox", "build_sandbox", "resolve_kind", "AnswerSurface", "ToolProbe",
           "answer_surfaces", "audit_tokens", "golden_files", "required_tool_probes"]


def resolve_kind(te: TargetExperiment) -> str | None:
    """The target's primary compute-unit KIND (systolic|simt|vector|scalar), for family routing. Tries the
    registered capability manifest first; falls back to the ``target_contract`` the descriptor itself
    names (some targets — e.g. radiance — register their contract under a ``*_oot`` id the roster registry
    does not yet resolve). Returns None if no contract is reachable (the sandbox still builds — routing
    then rests on ``sim_via`` alone, and the isolation test records 'kind: unknown')."""
    from merlin.targetgen.families import family_profile   # noqa: F401 — ensures kind is a known family
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest
        return load_capability_manifest(te.target).kind
    except Exception:  # noqa: BLE001 — registry miss / no contract; fall through to the declared contract
        pass
    try:
        doc = yaml.safe_load(te.path.read_text()) or {}
        contract_rel = ((doc.get("hardware_spec") or {}).get("target_contract"))
        if not contract_rel:
            return None
        from merlin.common.paths import repo_root
        from merlin.targetgen import compute_units
        from merlin.targetgen.target_experiment import _primary_kind
        contract = yaml.safe_load((repo_root() / contract_rel).read_text()) or {}
        units = compute_units.compute_units(contract)
        return _primary_kind(units) if units else None
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class Sandbox:
    """A built, target-parameterized sandbox. All fields are DERIVED from the descriptor (+ its manifest);
    nothing is per-target code. ``argv``/``env``/``wrap`` produce the real bwrap command; ``answer_surfaces``
    / ``required_tools`` / ``coverage_gap`` drive the isolation test."""
    te: TargetExperiment
    ws: Path
    bundle: dict
    kind: str | None
    answer_surfaces: list[AnswerSurface]
    required_tools: list[ToolProbe]

    def argv(self) -> list[str]:
        return bwrap.full_argv(self.te, self.ws, self.bundle)

    def env(self) -> str:
        return toolchain.sandbox_env(self.te, self.ws)

    def wrap(self, inner: str) -> str:
        return bwrap.wrap(self.te, self.ws, inner, self.bundle)

    def coverage_gap(self) -> list[AnswerSurface]:
        """The answer surfaces still reachable under the built argv — MUST be empty (hermetic guard)."""
        return bwrap.coverage_gap(self.argv(), self.answer_surfaces)


def build_sandbox(descriptor: str | Path | TargetExperiment, ws: Path,
                  bundle: dict | None = None) -> Sandbox:
    """Build the sandbox for a target from its descriptor (path or loaded ``TargetExperiment``) + an
    optional arm bundle. This is the single seam the experiment scripts + the CI isolation test call."""
    te = descriptor if isinstance(descriptor, TargetExperiment) else load_target_experiment(descriptor)
    return Sandbox(te=te, ws=Path(ws), bundle=bundle or {}, kind=resolve_kind(te),
                   answer_surfaces=answer_surfaces(te), required_tools=required_tool_probes(te))
