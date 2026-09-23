"""Explicit trusted Phase 2 workflow selection and shared action-contract construction.

Scientific authorities belong to the selected corpus or whole-model policy, not the
transport. Selection never comes from candidate metadata or a target-name default.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.target_experiment import TargetExperiment

from .broker import BrokerAction, placeholder_names
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json

CORPUS_FEEDBACK_V1 = "corpus-feedback-v1"
WHOLE_MODEL_V1 = "whole-model-v1"
DEVELOPMENT_FEEDBACK_ACTION = "tuning-gsim-feedback"
_HOST_ANALYSIS_SENTINEL = "__host_owned_command_buffer_analysis__"


class WholeModelAnalysis(Protocol):
    def __call__(
        self,
        baseline: Path,
        candidate: Path,
        sentinel: Any,
        *,
        timeout_s: int,
        peak_macs_per_cycle: int | None,
        achievable_macs_per_cycle: float | None,
        target: str,
    ) -> dict[str, Any]: ...


class CommandBufferAnalysis(Protocol):
    def __call__(
        self,
        baseline_json: Path,
        candidate_json: Path,
        *,
        candidate_root: Path,
        peak_macs_per_cycle: int | None,
        achievable_macs_per_cycle: float | None,
        target: str,
    ) -> dict[str, Any]: ...


class GlobalAnalysisView(Protocol):
    def __call__(
        self, record: Mapping[str, Any], *, complete_evidence: str, context_provider_installed: bool
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BrokerServices:
    """Only the native scientific projections which have not yet moved."""

    whole_model_analysis: WholeModelAnalysis | None = None
    command_buffer_analysis: CommandBufferAnalysis | None = None
    global_analysis_view: GlobalAnalysisView | None = None


_HOST_FEEDBACK_SENTINEL = "__host_owned_tuning_gsim_feedback__"
#: A FREE, ORDERING-ONLY analysis of two emitted command buffers. The measured feedback above costs
#: ~110 s a call and was the ONLY way to judge a candidate, so every dead end was paid for at full
#: price -- measured across three trials, two excursions of +5.9% and +11.1% cost ~220 s of oracle
#: time to discover. This prices the agent's own emitted artifacts instead: no oracle, no goldens,
#: no holdout, so it can be called as often as the agent likes and can leak nothing.
ANALYSIS_ACTION = "analyze-command-buffers"
INVENTORY_ACTION = "inspect-optimization-surfaces"
_HOST_INVENTORY_SENTINEL = "__host_owned_optimization_inventory__"
E2E_ANALYSIS_ACTION = "analyze-whole-model"
_HOST_E2E_ANALYSIS_SENTINEL = "__host_owned_whole_model_analysis__"
OCCUPANCY_PROFILE_ACTION = "profile-reduced-global-witness"
_HOST_OCCUPANCY_PROFILE_SENTINEL = "__host_owned_reduced_global_profile__"
CHANGED_REGION_ACTION = "qualify-changed-region"
_HOST_CHANGED_REGION_SENTINEL = "__host_owned_changed_region_qualification__"
SOURCE_CONVOLUTION_PREPARATION_ACTION = "prepare-source-convolution"
_HOST_SOURCE_CONVOLUTION_PREPARATION_SENTINEL = "__host_owned_source_convolution_preparation__"
SOURCE_CONTRACTION_PREPARATION_ACTION = "prepare-source-contraction"
_HOST_SOURCE_CONTRACTION_PREPARATION_SENTINEL = "__host_owned_source_contraction_preparation__"
SOURCE_CONTRACTION_QUALIFICATION_ACTION = "qualify-source-contraction"
_HOST_SOURCE_CONTRACTION_QUALIFICATION_SENTINEL = "__host_owned_source_contraction_qualification__"
CONTROLLED_CONTEXT_ACTION = "profile-controlled-context"
_HOST_CONTROLLED_CONTEXT_SENTINEL = "__host_owned_controlled_source_prefix__"
PAIRED_CONTEXT_ACTION = "compare-controlled-context"
_HOST_PAIRED_CONTEXT_SENTINEL = "__host_owned_paired_fixed_work_context__"
# Expensive evidence is a sparse promotion gate. One exploratory cohort query plus the mandatory
# final-byte query is enough to reject or promote a lever; the dedicated counter profile is a
# separate one-shot instrument for an occupancy/overlap unknown. Structural/analytical actions stay
# bounded only by the ordinary tool-call budget because they launch no simulator.
EXPENSIVE_ACTION_LIMITS = {
    DEVELOPMENT_FEEDBACK_ACTION: 2,
    OCCUPANCY_PROFILE_ACTION: 1,
    CONTROLLED_CONTEXT_ACTION: 1,
    PAIRED_CONTEXT_ACTION: 1,
}

#: WHICH ACTIONS NEED A HOST PROVIDER, and the refusal each one raises without it. ONE TABLE, read
#: by three consumers that used to decide independently: the handler that refuses the call, the
#: registry that advertises the action, and the stage context the agent reads.
#:
#: THE DEFECT THIS FIXES, measured on `phase2_arm4_frombest_20260909`. `profile-reduced-global-witness`
#: was registered, given a 1-per-round budget, described at length in the prompt as the only route to
#: a calibrated cost -- and refused at runtime with rc=125, because its provider is installed only
#: when the launcher is given a probe interface and this run was not. The same document reported
#: `probes_available: False` in a field nothing joined to the action list. So the loop advertised an
#: action it could not answer, the agent spent a call and part of its wall budget discovering that,
#: and no measured feedback reached it in any of the three rounds. An action that is registered,
#: budgeted and advertised and then refuses is worse than an absent one, because the loop reports it
#: as available. Each entry is (action, the broker attribute that must be installed, the refusal).
ACTION_PROVIDER_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    (OCCUPANCY_PROFILE_ACTION, "global_probe_provider", "host mechanism-equivalent probe extraction is unavailable"),
    (CHANGED_REGION_ACTION, "global_semantic_provider", "host changed-region semantic extraction is unavailable"),
    (CONTROLLED_CONTEXT_ACTION, "global_context_provider", "controlled source-prefix extraction is unavailable"),
    (PAIRED_CONTEXT_ACTION, "global_paired_context_provider", "paired fixed-work context extraction is unavailable"),
    (
        SOURCE_CONTRACTION_QUALIFICATION_ACTION,
        "global_source_pair_provider",
        "complete source-pair runtime provider is unavailable",
    ),
)


def unavailable_global_actions(providers: Mapping[str, Any]) -> dict[str, str]:
    """`{action: refusal}` for every global action whose host provider is not installed.

    ``providers`` is keyed by the attribute names in :data:`ACTION_PROVIDER_REQUIREMENTS`. A key the
    mapping does not carry is treated as NOT installed: an availability question that cannot be
    answered resolves to unavailable, because the other direction advertises an action nobody can
    answer -- which is the defect this exists for.
    """
    return {
        action: reason for action, attribute, reason in ACTION_PROVIDER_REQUIREMENTS if providers.get(attribute) is None
    }


#: never the exception's own text, so an evaluator message can never carry a hidden shape, a capsule
#: path or a golden value across the boundary even if it is worded to look procedural.
#:
#: THE DEFECT THIS FIXES, measured on the v15 run. The agent called `qualify-changed-region` three
#: times, spending 1,566 s of a 3,600 s authoring budget, and every call returned rc=125 with the
#: single word `(ValueError)` or `(TimeoutError)`. The host had recorded the real causes into a
#: host-private directory: once "candidate changed: recompile its full graph and global plan" and
#: twice a wall-budget timeout. The first is purely PROCEDURAL -- it says the agent must re-run the
#: analysis action before qualifying, which it would have done had it been told. Instead it retried
#: the identical call twice more. `changed_region_semantic_qualification` is one of the two standing
#: promotion blockers, and this is why it never cleared.
#:
#: Each entry is (substring to look for in the host's own message, what the agent is told).
AGENT_VISIBLE_REFUSAL_REASONS: tuple[tuple[str, str], ...] = (
    (
        "candidate changed: recompile its full graph and global plan",
        "the candidate's bytes changed since the last analysis; re-run the whole-model analysis "
        "action before qualifying a changed region",
    ),
    (
        "exceeded its wall budget",
        "this action exceeded its wall budget; it did not fail, it ran out of time, so retrying it "
        "unchanged will spend the budget again",
    ),
    (
        "is unavailable",
        "the host did not install a provider for this action in this run; no retry will make it available",
    ),
    (
        "compiler edit authority or input integrity refused candidate execution",
        "the candidate falls outside the frozen compiler edit authority; revert the out-of-scope edit "
        "before invoking a host action",
    ),
    (
        "a corpus microbenchmark sweep is not a global iteration",
        "this action is not available in global mode; use the whole-model analysis action instead",
    ),
    (
        "a calibration probe must reduce the full-model repetition count",
        "the probe must repeat its mechanism fewer times than the full model does",
    ),
    (
        "probe mechanism differs in representation, resource, capacity, or edge domain",
        "the probe does not emit the same mechanism signature as the full model, so it cannot calibrate it",
    ),
    (
        "stale graph, global plan, compiler, or target evidence",
        "the probe was extracted against evidence that has since changed; re-analyze, then re-extract",
    ),
)


def agent_visible_refusal(lead: str, exc: BaseException) -> str:
    """The refusal text an action returns to the agent: a DECLARED remediation, or just the type.

    See :data:`AGENT_VISIBLE_REFUSAL_REASONS` for why this exists and what may cross the boundary.
    When nothing matches, the result is byte-identical to what these actions returned before, so the
    default is unchanged and only declared cases gain a reason.
    """
    detail = str(exc)
    for needle, remediation in AGENT_VISIBLE_REFUSAL_REASONS:
        if needle in detail:
            return f"{lead} ({type(exc).__name__}): {remediation}"
    return f"{lead} ({type(exc).__name__})"


def _record_host_refusal(stage: Any, exc: BaseException, *, round_index: Any, call_index: Any) -> None:
    """Write the full reason for a host-side refusal where the HOST can read it.

    Deliberately not on the agent's path and deliberately best-effort: a failure to record a failure
    must not replace it with a different one. It lands beside the evaluator's own work root, which is
    host-private, so nothing here widens what the agent can see.
    """
    try:
        import traceback  # noqa: PLC0415

        evaluator = getattr(stage, "feedback_evaluator", None)
        macro = getattr(stage, "global_experiment", None)
        private_root = getattr(macro, "output", None) or getattr(evaluator, "work_root", None)
        if private_root is None:
            return  # Never fall back to a cwd which could be part of an agent-visible workspace.
        base = Path(private_root)
        # A RELATIVE root resolves against the process cwd, which is the repo checkout for a
        # launcher run from the tree -- measured: two refusal files landed at the repository root.
        # Recording is best-effort, so an unanchored root is dropped, never re-anchored.
        if not base.is_absolute():
            return
        base = base.resolve()
        checkout = Path(repo_root()).resolve()
        if base == checkout or checkout.is_relative_to(base):
            return  # The checkout (or one of its parents) is never a host-private work root.
        root = base / "host_refusals"
        root.mkdir(parents=True, exist_ok=True)
        # NEVER OVERWRITE AN EARLIER REFUSAL. The round index is None on the macro path -- every
        # round's broker gets a fresh call counter and no round number -- so the name was
        # `round_None_call_4.txt` in all three rounds of `phase2_arm4_frombest_20260909` and the
        # later round silently replaced the earlier one. The only host-side record of WHY the
        # measured-witness probe refused in round 0 was overwritten by an unrelated edit-scope
        # refusal in round 2, and the reason had to be recovered from a stderr hash. A refusal
        # record that a later refusal can destroy is a record of the last event, not of the run.
        stem = f"round_{round_index}_call_{call_index}"
        path = root / f"{stem}.txt"
        suffix = 0
        while path.exists() or path.is_symlink():
            suffix += 1
            path = root / f"{stem}.{suffix}.txt"
        path.write_text(
            f"{type(exc).__name__}: {exc}\n\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - recording is never allowed to mask the thing being recorded
        pass


@dataclass(frozen=True)
class ActionAdmission:
    """Selected workflow's budget rule, enforced atomically by the broker."""

    limit: int | None
    reserved: bool
    limit_refusal: str
    reserve_refusal: str


class WorkflowPolicy:
    """One invocation's scientific authority; constructed only by trusted host selection.

    The broker allocates indices and bounds time before dispatch. This policy owns
    candidate checks, scientific invocation order, redaction and workflow evidence.
    """

    workflow_id: str

    def __init__(
        self,
        *,
        candidate: Path,
        target_experiment: TargetExperiment,
        receipt_path: Path,
        services: BrokerServices = BrokerServices(),
        feedback_evaluator: Any | None = None,
        feedback_round: int | None = None,
        functional_base: Path | None = None,
        e2e_sentinel: Any | None = None,
        global_experiment: Any | None = None,
        global_probe_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_semantic_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_context_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_paired_context_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_source_pair_provider: Callable[..., Mapping[str, Any]] | None = None,
    ):
        self.candidate = candidate
        self.target_experiment = target_experiment
        self.receipt_path = receipt_path
        self.services = services
        self.feedback_evaluator = feedback_evaluator
        self.feedback_round = feedback_round
        self.functional_base = functional_base
        self.e2e_sentinel = e2e_sentinel
        self.global_experiment = global_experiment
        self.global_probe_provider = global_probe_provider
        self.global_semantic_provider = global_semantic_provider
        self.global_context_provider = global_context_provider
        self.global_paired_context_provider = global_paired_context_provider
        self.global_source_pair_provider = global_source_pair_provider
        self.stop_verdict = None

    def admission(self, name: str) -> ActionAdmission:
        """Invocation cap and eligibility for the reserved analysis deadline."""
        limit = EXPENSIVE_ACTION_LIMITS.get(name)
        return ActionAdmission(
            limit,
            name == E2E_ANALYSIS_ACTION,
            f"expensive action {name!r} is limited to {limit} invocation(s) "
            "per round; iterate with structural/analytical actions and reserve the final "
            "tuning query for the exact bytes you seal",
            "performance stage wall-clock budget is reserved for mandatory whole-model analysis",
        )

    def validate_actions(self, actions) -> None:
        unavailable = self.unavailable
        disagrees = sorted(
            name
            for name, _attribute, _reason in ACTION_PROVIDER_REQUIREMENTS
            if name in actions and actions[name].available is not (name not in unavailable)
        )
        if disagrees:
            raise StageGateError(f"broker action availability disagrees with its installed providers: {disagrees}")

    def _analyze(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            document = self._analyze_document(request, rendered, call_index, timeout_s)
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - inability to inspect the objective is a refusal
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": (f"whole-model analysis refused ({type(exc).__name__}: {str(exc)[:200]})"),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _inventory(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            document = self._inventory_document(request, rendered, call_index, timeout_s)
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - malformed declarations are an actionable refusal
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": (f"optimization-surface inventory refused ({type(exc).__name__}: {str(exc)[:200]})"),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _profile(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            document = self._profile_document(request, rendered, call_index, timeout_s)
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - partial counter evidence is never invented
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("reduced global profile refused by the host-owned evaluator", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _command_buffers(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            evaluator = self.feedback_evaluator
            if self.services.command_buffer_analysis is None:
                raise StageGateError("command-buffer analysis service is unavailable")
            document = self.services.command_buffer_analysis(
                Path(rendered["baseline_json"]),
                Path(rendered["candidate_json"]),
                candidate_root=Path(self.candidate),
                peak_macs_per_cycle=getattr(evaluator, "peak_macs_per_cycle", None),
                achievable_macs_per_cycle=getattr(evaluator, "achievable_macs_per_cycle", None),
                target=str(getattr(getattr(evaluator, "target_experiment", None), "target", "") or ""),
            )
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - an unreadable buffer is a refusal, not a crash
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": f"command-buffer analysis refused ({type(exc).__name__}: {str(exc)[:200]})",
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def build_registry(self):
        raise NotImplementedError

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        raise NotImplementedError


def select_workflow(workflow_id: str, **inputs: Any) -> WorkflowPolicy:
    """Resolve one explicit, versioned host workflow; never guess or fall back."""
    from .corpus_feedback import CorpusFeedbackPolicy
    from .whole_model import WholeModelPolicy

    registry = {CORPUS_FEEDBACK_V1: CorpusFeedbackPolicy, WHOLE_MODEL_V1: WholeModelPolicy}
    try:
        owner = registry[workflow_id]
    except (KeyError, TypeError) as exc:
        raise StageGateError(f"unknown broker workflow policy: {workflow_id!r}") from exc
    if (inputs.get("global_experiment") is not None) != (workflow_id == WHOLE_MODEL_V1):
        raise StageGateError("broker workflow policy disagrees with its scientific experiment")
    return owner(**inputs)


def _probe_slug(label: str) -> str:
    """Historical ASCII probe-name runs, without a regex-dependent parser."""
    result = []
    invalid = False
    for character in label:
        allowed = character.isascii() and (character.isalnum() or character in "._-")
        if allowed:
            result.append(character)
        elif not invalid:
            result.append("-")
        invalid = not allowed
    return "".join(result).strip("-").lower()


def action_registry(
    workflow_id: str,
    candidate: Path,
    target_experiment: TargetExperiment,
    *,
    unavailable: Mapping[str, str] | None = None,
) -> tuple[BrokerAction, ...]:
    """Construct the selected host contract before the round's evaluator exists."""
    if workflow_id not in (CORPUS_FEEDBACK_V1, WHOLE_MODEL_V1):
        raise StageGateError(f"unknown broker workflow policy: {workflow_id!r}")
    return _build_action_registry(
        candidate, target_experiment, global_optimization=workflow_id == WHOLE_MODEL_V1, unavailable=unavailable
    )


def _build_action_registry(
    candidate: Path,
    target_experiment: TargetExperiment,
    *,
    global_optimization: bool = False,
    unavailable: Mapping[str, str] | None = None,
) -> tuple[BrokerAction, ...]:
    """Create named candidate-manifest actions; no caller-selected executable is accepted.

    ``unavailable`` comes from :func:`unavailable_global_actions` over the SAME providers the broker
    was constructed with. An action named there is still registered -- removing it would make a
    legitimate call look like an unknown action rather than an uninstalled provider -- but it is
    advertised carrying the refusal it would return, so nothing downstream can read it as available.
    """
    manifest_path = candidate / "manifest.yaml"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StageGateError("functional candidate has no real manifest.yaml")
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries, commands = document.get("entrypoints"), document.get("commands")
    if not isinstance(entries, Mapping) or not isinstance(commands, Mapping) or not commands:
        raise StageGateError("functional candidate manifest has no brokerable commands")
    tool = entries.get("tool")
    if not isinstance(tool, str) or not tool:
        raise StageGateError("functional candidate manifest has no tool entrypoint")
    tool_path = candidate / tool
    if tool_path.is_symlink() or not tool_path.is_file():
        raise StageGateError("functional candidate tool entrypoint is absent or linked")
    actions: list[BrokerAction] = []
    for command_name in sorted(commands):
        row = commands[command_name]
        raw = row.get("argv") if isinstance(row, Mapping) else None
        if not isinstance(raw, list) or not raw or any(not isinstance(v, str) for v in raw):
            raise StageGateError(f"candidate manifest command {command_name!r} has malformed argv")
        argv = tuple(str(tool_path) if value == "{tool}" else value for value in raw)
        placeholders = tuple(sorted({match for value in argv for match in placeholder_names(value)}))
        if "tool" in placeholders:
            raise StageGateError(f"candidate manifest command {command_name!r} has embedded tool token")
        actions.append(
            BrokerAction(
                f"candidate-{command_name.replace('_', '-')}",
                argv,
                placeholders,
                f"candidate manifest command {command_name}",
                not global_optimization,
            )
        )
    for probe in TC.required_tool_probes(target_experiment):
        slug = _probe_slug(probe.label)
        if not slug:
            raise StageGateError("required target tool probe has no safe action name")
        actions.append(
            BrokerAction(
                f"probe-{slug}", ("bash", "-c", probe.cmd), (), f"descriptor-derived probe for {probe.label}", False
            )
        )
    actions.append(
        BrokerAction(
            DEVELOPMENT_FEEDBACK_ACTION,
            (_HOST_FEEDBACK_SENTINEL,),
            (),
            "sparse host-owned frozen-tuning correctness and certified GSIM cycle deltas; at most one "
            "exploratory call, then reserve the second and final call for the exact bytes being sealed",
            not global_optimization,
        )
    )
    actions.append(
        BrokerAction(
            E2E_ANALYSIS_ACTION,
            (_HOST_E2E_ANALYSIS_SENTINEL,),
            (),
            (
                "optional in-round host-owned baseline/candidate emission and structural analysis of the "
                "fixed declared complete-model objective; the macro controller separately validates the "
                "submitted bytes after authoring; no simulator and no timing claim"
            )
            if global_optimization
            else (
                "required host-owned baseline/candidate emission and structural analysis of the fixed "
                "declared complete-model objective; no simulator and no timing claim"
            ),
            not global_optimization,
        )
    )
    actions.append(
        BrokerAction(
            ANALYSIS_ACTION,
            (_HOST_ANALYSIS_SENTINEL, "{baseline_json}", "{candidate_json}"),
            ("baseline_json", "candidate_json"),
            "host-owned comparison of two emitted command buffers from the buffers alone: declared work "
            "volume per arm, the derived ceilings, the change in completion points (barriers), and a "
            "cycle LOWER BOUND per arm, and structural inefficiencies tagged by the optimisation level "
            "they live at. Costs no oracle time -- use it to investigate structural changes before "
            "spending a measurement. It cannot certify a speedup or regression: nothing here predicts which of two "
            "orderings is faster, and the block it returns says so",
            False,
        )
    )
    actions.append(
        BrokerAction(
            INVENTORY_ACTION,
            (_HOST_INVENTORY_SENTINEL,),
            (),
            "host-owned live AST inventory of compiler command ownership and verified manifest "
            "optimization surfaces; rerun after changing source or manifest declarations",
            False,
        )
    )
    actions.append(
        BrokerAction(
            OCCUPANCY_PROFILE_ACTION,
            (_HOST_OCCUPANCY_PROFILE_SENTINEL,),
            (),
            "host-owned warm profile of one preselected frozen reduced witness; returns total compute "
            "cycles and only proved occupancy, movement, and overlap evidence; absent executed-encoding "
            "evidence remains explicitly UNKNOWN. The witness calibrates the complete-model plan and "
            "is never a whole-model result",
            False,
        )
    )
    if global_optimization:
        actions.append(
            BrokerAction(
                SOURCE_CONTRACTION_PREPARATION_ACTION,
                (
                    _HOST_SOURCE_CONTRACTION_PREPARATION_SENTINEL,
                    "{comparison_arm}",
                    "{source_op_index}",
                    "{max_m}",
                    "{max_n}",
                    "{max_k}",
                ),
                ("comparison_arm", "source_op_index", "max_m", "max_n", "max_k"),
                "prepare one explicitly selected current-source contraction through both exact normal "
                "compiler arms under 60 seconds; explicit baseline/previous and integer source/bounds only. "
                "Returns preparation SHA and independent typed oracle; no runtime or numerical pass",
                False,
            )
        )
        actions.append(
            BrokerAction(
                SOURCE_CONTRACTION_QUALIFICATION_ACTION,
                (_HOST_SOURCE_CONTRACTION_QUALIFICATION_SENTINEL, "{preparation_sha256}"),
                ("preparation_sha256",),
                "execute a current host-prepared reduced source pair through the optional host runtime "
                "provider under 60 seconds, warm1/measured1 per arm; only a preparation SHA, never a file "
                "path. Full-model route relevance, numerics and timing remain separate obligations",
                False,
            )
        )
        actions.append(
            BrokerAction(
                SOURCE_CONVOLUTION_PREPARATION_ACTION,
                (_HOST_SOURCE_CONVOLUTION_PREPARATION_SENTINEL, "{comparison_arm}"),
                ("comparison_arm",),
                "prepare actual changed host-to-convolution source using cached full-model proofs and both "
                "normal compiler entrypoints, within 60 seconds; comparison_arm must explicitly be "
                "optimization_baseline or previous. Returns source opportunities and allowed edit surfaces; "
                "no simulator, runtime admission or numerical pass",
                False,
            )
        )
        actions.append(
            BrokerAction(
                CHANGED_REGION_ACTION,
                (_HOST_CHANGED_REGION_SENTINEL,),
                (),
                "host-selected reduced semantic witness for the actual changed full-model region; "
                "same candidate compiler under answer-masked policy plus independent reference. "
                "Reports only the qualified mechanism/domain, never full-model numerics or cycles",
                False,
            )
        )
        actions.append(
            BrokerAction(
                CONTROLLED_CONTEXT_ACTION,
                (_HOST_CONTROLLED_CONTEXT_SENTINEL,),
                (),
                "host-extracted bounded prefix of the current emitted model, including queued loads; "
                "one warm and one measured prefix under a total 60-second budget. Reports controlled "
                "occupancy only, not full task/model equivalence or global cost calibration",
                False,
            )
        )
        actions.append(
            BrokerAction(
                PAIRED_CONTEXT_ACTION,
                (_HOST_PAIRED_CONTEXT_SENTINEL,),
                (),
                "host-projected identical bounded work in previous/current emitted schedules; "
                "warm1/measured1 per arm under one total 60-second budget. Reports controlled "
                "fixed-work cycle differences, never full-model cycles or statistical confirmation",
                False,
            )
        )
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise StageGateError("broker action registry contains duplicate names")
    refused = dict(unavailable or {})
    unknown = sorted(set(refused) - set(names))
    if unknown:
        # An availability verdict about an action nobody registered is a wiring error, not a
        # refusal: it would silently mark nothing while reading as though it had.
        raise StageGateError(f"availability was declared for unregistered actions: {unknown}")
    return tuple(
        action if action.name not in refused else replace(action, unavailable_reason=refused[action.name])
        for action in actions
    )
