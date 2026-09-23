"""Shared, descriptor+manifest-driven agentic bwrap sandbox.

Promotes the per-experiment gemmini sandbox to target-agnostic infra: a new target gets a correct,
continuously-guarded sandbox from its ``target_experiment.yaml`` (+ capability manifest) with ZERO copied
scripts. Routing is by compute-unit KIND / sim FAMILY (never a target name):

  * :mod:`.answer_surfaces` — the DERIVED answer-surface mask set (goldens/model weights/hidden/prior/
    oracle/grader/memory) + the single declared oracle/grader registry + the coverage guard's audit tokens.
  * :mod:`.toolchain`       — the legit tools bound back, universal + the descriptor's ``sim_via`` family,
    cross-checked by ``kind`` via :mod:`merlin.targetgen.families`.
  * :mod:`.bwrap`           — deny-by-default argv assembly + the hermetic mount-replay coverage proof.
  * :mod:`.preflight`       — whether the sandbox can actually be BUILT on this host, as a named condition.
  * :mod:`.cleanroom`       — prevention WITHOUT a kernel: a workspace in which the answer key is absent.
  * :mod:`.read_audit`      — detection: what a run's event log says the agent read, as CLEAN/CONTAMINATED/UNKNOWN.

WHEN THERE IS NO KERNEL TO FALL BACK ON. :func:`build_sandbox` is the strong mechanism and should be
used whenever :func:`~merlin.targetgen.sandbox.preflight.probe_sandbox` says it can be. Where it cannot
— a host whose policy forbids unprivileged user namespaces — the clean room and the read audit are what
remain, and they are defence in depth rather than a replacement: the first makes the answer key absent
from the workspace, the second reports afterwards what was actually read. Neither stops a process from
opening an absolute path elsewhere on the filesystem, and nothing here claims to.

:func:`build_sandbox` is the one entry point; the experiment ``scripts/*`` are thin delegators onto it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.targetgen.sandbox import bwrap, toolchain
from merlin.targetgen.sandbox.answer_surfaces import (
    AUDIT_ADVISORY_KINDS,
    AUDIT_VIOLATION_KINDS,
    AnswerSurface,
    DroppedDeclaration,
    DroppedDeclarations,
    answer_surfaces,
    audit_hit_is_violation,
    audit_tokens,
    declared_oracle_modules,
    dropped_declarations,
    golden_files,
    module_matches,
    module_name_for,
    prior_backend_exemptions,
    require_declarations_in_force,
    weight_files,
)
from merlin.targetgen.sandbox.bwrap import snapshot_input_paths
from merlin.targetgen.sandbox.cleanroom import (
    CleanRoom,
    CleanRoomRefused,
    CleanRoomVerdict,
    build_clean_room,
    verify_clean_room,
)
from merlin.targetgen.sandbox.preflight import (
    SANDBOX_ABSENT,
    SANDBOX_INOPERABLE,
    SANDBOX_OK,
    SANDBOX_UNKNOWN,
    SandboxProbe,
    SandboxUnavailable,
    probe_sandbox,
    require_working_sandbox,
)
from merlin.targetgen.sandbox.read_audit import (
    CLEAN,
    CONTAMINATED,
    UNKNOWN,
    AnswerKeyExposure,
    ReadAudit,
    audit_event_logs,
    audit_run,
)
from merlin.targetgen.sandbox.toolchain import ToolProbe, required_tool_probes
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment

__all__ = [
    "Sandbox",
    "build_sandbox",
    "snapshot_input_paths",
    "resolve_kind",
    "AnswerSurface",
    "ToolProbe",
    "answer_surfaces",
    "audit_tokens",
    # a declared deny rule that matched nothing is a silent no-op — report it
    "DroppedDeclaration",
    "DroppedDeclarations",
    "dropped_declarations",
    "prior_backend_exemptions",
    "require_declarations_in_force",
    "golden_files",
    "weight_files",
    "required_tool_probes",
    "AUDIT_ADVISORY_KINDS",
    "AUDIT_VIOLATION_KINDS",
    "audit_hit_is_violation",
    "declared_oracle_modules",
    "module_matches",
    "module_name_for",
    # containment without kernel isolation
    "CleanRoom",
    "CleanRoomRefused",
    "CleanRoomVerdict",
    "build_clean_room",
    "verify_clean_room",
    "ReadAudit",
    "AnswerKeyExposure",
    "audit_event_logs",
    "audit_run",
    "CLEAN",
    "CONTAMINATED",
    "UNKNOWN",
    # is the sandbox usable at all
    "SandboxProbe",
    "SandboxUnavailable",
    "probe_sandbox",
    "require_working_sandbox",
    "SANDBOX_OK",
    "SANDBOX_ABSENT",
    "SANDBOX_INOPERABLE",
    "SANDBOX_UNKNOWN",
]


def resolve_kind(te: TargetExperiment) -> str | None:
    """The target's primary compute-unit KIND (systolic|simt|vector|scalar), for family routing. Tries the
    registered capability manifest first; falls back to the ``target_contract`` the descriptor itself
    names (some targets — e.g. radiance — register their contract under a ``*_oot`` id the roster registry
    does not yet resolve). Returns None if no contract is reachable (the sandbox still builds — routing
    then rests on ``sim_via`` alone, and the isolation test records 'kind: unknown')."""
    from merlin.targetgen.families import family_profile  # noqa: F401 — ensures kind is a known family

    try:
        from merlin.targetgen.target_experiment import load_capability_manifest

        return load_capability_manifest(te.target).kind
    except Exception:  # noqa: BLE001 — registry miss / no contract; fall through to the declared contract
        pass
    try:
        doc = yaml.safe_load(te.path.read_text()) or {}
        contract_rel = (doc.get("hardware_spec") or {}).get("target_contract")
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
    _policy_test_live_inputs: bool = False

    def argv(self) -> list[str]:
        return bwrap.full_argv(self.te, self.ws, self.bundle, _policy_test_live_inputs=self._policy_test_live_inputs)

    def env(self) -> str:
        return toolchain.sandbox_env(self.te, self.ws)

    def wrap(self, inner: str) -> str:
        return bwrap.wrap(self.te, self.ws, inner, self.bundle, _policy_test_live_inputs=self._policy_test_live_inputs)

    def coverage_gap(self) -> list[AnswerSurface]:
        """The answer surfaces still reachable under the built argv — MUST be empty (hermetic guard)."""
        argv = self.argv()
        private = bwrap.host_input_surfaces(
            argv, self.ws, self.bundle, _policy_test_live_inputs=self._policy_test_live_inputs
        )
        return bwrap.coverage_gap(argv, [*self.answer_surfaces, *private])

    def dropped_declarations(self) -> list[DroppedDeclaration]:
        """Declared deny rules that contributed NO mask — MUST be empty, and is not covered by
        :meth:`coverage_gap`, which can only iterate the surfaces that survived the existence filter.
        An empty coverage gap over a set that quietly lost a rule proves nothing about that rule."""
        return dropped_declarations(self.te)

    def containment_record(self) -> dict[str, object]:
        """The block a run artifact embeds so "was this run actually contained?" is answerable after the
        fact — both halves together, because either one alone is satisfiable while the other is not."""
        return {
            "coverage_gap": [str(surface.path) for surface in self.coverage_gap()],
            "dropped_declarations": [d.as_record() for d in self.dropped_declarations()],
        }


def build_sandbox(
    descriptor: str | Path | TargetExperiment,
    ws: Path,
    bundle: dict | None = None,
    *,
    _policy_test_live_inputs: bool = False,
) -> Sandbox:
    """Build the sandbox for a target from its descriptor (path or loaded ``TargetExperiment``) + an
    optional arm bundle. This is the single seam the experiment scripts + the CI isolation test call."""
    te = descriptor if isinstance(descriptor, TargetExperiment) else load_target_experiment(descriptor)
    return Sandbox(
        te=te,
        ws=Path(ws),
        bundle=bundle or {},
        kind=resolve_kind(te),
        answer_surfaces=answer_surfaces(te),
        required_tools=required_tool_probes(te),
        _policy_test_live_inputs=_policy_test_live_inputs,
    )
