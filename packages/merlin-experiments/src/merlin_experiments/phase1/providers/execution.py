"""One agent-turn execution boundary: provider selection, billing and candidate transport.

Inputs are explicit; no native launcher is imported and no target is selected at
module import. Sandbox composition preserves the existing Phase 1 policy, which
is intentionally not yet consolidated with the differing shared full_argv path.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from merlin.common import arrival_stamp as AS
from merlin.common.cpu_pool import AUTHOR, with_priority
from merlin.targetgen import experiment_tokens as ET

from .. import author_subagents
from ..context import InvocationContext
from ..feedback import lifecycle as FL
from ..telemetry import startup as _START
from . import codex_context

_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"
_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"
_MODEL_HOST_SNAPSHOT_RECORD_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"


@dataclass(frozen=True)
class ProviderConfig:
    driver: str = "auto"
    provider: str = "subscription"
    subagent_model: str = ""
    background_model: str = ""
    codex_runtime: object | None = None
    codex_max_subagents: int | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    context: InvocationContext
    provider: ProviderConfig
    resolved_tools: Callable[[], tuple[str, ...]]
    timing_file: Path
    sim_max_jobs: int = 0
    selected_rtl_facts: Path | None = None
    sim_per_capsule_timeout: int = 0
    model_gate_interval: int = 0


def automatic_model_gate_interval(context, options) -> int:
    """Schedule optional visible-model feedback only for a declared full-model scope.

    Capsule authoring retains the model tool without starting model workers.
    The descriptor supplies the capability; target names grant nothing.
    """
    if getattr(options, "joint_campaign", ""):
        return options.grade_interval
    if options.experiment != "full" or getattr(options, "validation_scope", "full") != "full":
        return 0
    from ..feedback.private_full_models import program_requirements_for

    return options.grade_interval if program_requirements_for(context.descriptor) else 0


_DRIVER_MODULES = {
    "codex": "merlin_experiments.phase1.providers.codex_agent",
    "opencode": "merlin_experiments.phase1.providers.opencode_agent",
    "converse": "merlin_experiments.phase1.providers.bedrock_agent",
}  # 'claudecode' drives the claude CLI directly


def resolve_driver(model: str, *, config: ProviderConfig) -> str:
    """Resolve which agent driver handles ``model``. ``--driver`` (``config.driver``) is authoritative
    when set to a concrete driver; ``auto`` (the default, behavior-preserving) routes by model id — the
    Bedrock Converse loop for a non-Anthropic id, else the ``claude`` CLI."""
    from merlin_experiments.phase1.providers import bedrock_agent as _BA

    if config.driver and config.driver != "auto":
        return config.driver
    return "converse" if _BA.is_converse_model(model) else "claudecode"


def trust_cli_cost(model: str, *, config: ProviderConfig) -> bool:
    """False when the round reached its model through the bridge.

    A CLI prices what it believes it ran. Pointed at the proxy it bills a foreign model at its own
    catalogue's rates, which is both a wrong artifact and an active hazard: the inflated figure counts
    against --max-spend-usd and kills a run that has spent almost nothing.
    """
    try:
        from merlin_experiments.phase1.providers import agent_bridge as _BR

        return not _BR.bridged_name(model, resolve_driver(model, config=config))
    except Exception:
        return True


def billing_mode(model: str, *, config: ProviderConfig) -> str:
    """How the run that ``model`` produces is BILLED — asked of the driver, never inferred from the
    model id. A driver module declares ``BILLING_MODE``; anything that does not is metered (an API
    key charged per token). This is what keeps a subscription-seat run from reporting a dollar spend:
    a Codex round once landed ``estimated_cost_usd: 17.2103`` in the ledger, priced at opus rates for
    a model no price table knows, on an account that is not billed per token at all."""
    # A driver's declared BILLING_MODE describes ITS OWN account. A bridged round does not use that
    # account: it is our Bedrock key, charged per token, whichever CLI happens to be driving it. The
    # codex driver declares subscription_notional because it normally runs on a ChatGPT seat, so a
    # bridged codex round was booking real Bedrock spend as notional -- money that is never counted
    # against --max-spend-usd or the campaign budget. This is the mirror of trusting a CLI's own
    # total_cost_usd for a model it does not bill, and it under-reports instead of over-reporting.
    drv = resolve_driver(model, config=config)
    try:
        from merlin_experiments.phase1.providers import agent_bridge as _BR

        if _BR.bridged_name(model, drv):
            return ET.METERED
    except Exception:
        pass
    mod_name = _DRIVER_MODULES.get(drv)
    if not mod_name:
        # NO DRIVER MODULE means the `claude` CLI driven directly, and for that one only the PROVIDER
        # knows whether the tokens are billed: `--provider subscription` runs on the machine's own
        # ~/.claude SEAT, which is not charged per token at all, while `--provider bedrock` runs on our
        # AWS key and is. Returning METERED for both made the two indistinguishable in the ledger and
        # priced a seat run's tokens at Anthropic list rates -- exactly the defect
        # `subscription_notional` was introduced to fix for the codex seat, one driver over. A seat run
        # keeps `estimated_cost_usd: None` and reports the projection as `subscription_notional_usd`,
        # so a notional figure can never be spent against a real budget ceiling.
        return ET.SUBSCRIPTION_NOTIONAL if config.provider == "subscription" else ET.METERED
    try:
        import importlib

        return getattr(importlib.import_module(mod_name), "BILLING_MODE", ET.METERED)
    except ImportError:
        return ET.METERED


_ASSEMBLY_VERIFIED: dict[str, bool] = {}


def _assemble(stage, function, /, *args, **kwargs):
    """One sandbox assembly step, computed in a grader worker process once proven identical there.

    Assembly is Python in the runner, which also hosts the background grader's threads: measured, the
    same steps took 10-32 s in the launch preflight and 59-174 s for the first turn while a grade ran,
    all of it waiting for the interpreter lock. The first call of each step in this process (the
    preflight) computes the argv both in process and in a worker and keeps the in-process one; later
    calls use the worker only if the two were identical, and a worker failure falls back in process
    (:func:`merlin.targetgen.grade_offload.call`). A mismatch keeps the step in process for the run."""
    from merlin.targetgen import grade_offload

    grade_offload.register(function)
    verified = _ASSEMBLY_VERIFIED.get(stage)
    if verified is None:
        inline = function(*args, **kwargs)
        try:
            _ASSEMBLY_VERIFIED[stage] = grade_offload.call(function, *args, **kwargs) == inline
        except Exception:  # noqa: BLE001 -- an unverifiable step simply stays in process
            _ASSEMBLY_VERIFIED[stage] = False
        return inline
    if verified:
        return grade_offload.call(function, *args, **kwargs)
    return function(*args, **kwargs)


# Sandbox verification is on the author launch/resume critical path. Change only
# admission priority: do not reserve a core or hold a lease over the paid turn.
@with_priority(AUTHOR)
def sandbox_plan(
    inner: str,
    ws: Path,
    bundle: dict,
    extra_binds: list[str] | None = None,
    *,
    context: InvocationContext,
    private_run_dir: Path | None = None,
    codex_mode: bool = False,
) -> str:
    """bwrap argv (deny-by-default) + provider runtime binds + TOOLCHAIN binds (the legit build+sim tools,
    bound back over the /scratch* masks) + the DERIVED answer-mask pass + toolchain env. The mask set now
    comes from the shared descriptor-driven answer surface (goldens/hidden/prior/oracle/grader/memory) and
    is coverage-proven by test_sandbox_isolation; masking is bundle-independent (a bind that re-exposes an
    answer surface is re-masked here, not left to the bundle's denied list).

    No mode mounts the operator's Claude state (``~/.claude``, ``~/.claude.json``). An ``inner`` that runs
    the ``claude`` CLI gets a home staged fresh for that launch (``sandbox.claude_home``) instead."""
    from merlin.targetgen.sandbox import bwrap as _BW
    from merlin.targetgen.sandbox import claude_home as _CH
    from merlin.targetgen.target_experiment import load_target_experiment

    base_options = {"repo": context.repo, "include_claude_home": False}
    if codex_mode:
        base_options["inherit_environment"] = False
    parts = _START.call("sandbox_base", _assemble, "sandbox_base", _BW.base_argv, ws, bundle, **base_options)
    home = None
    if not codex_mode:
        parts += _CH.without_operator_state(_BW.claude_runtime_binds())
        if _CH.launches_claude(inner):
            home = _CH.stage(private_run_dir.name if private_run_dir is not None else "agent", workspace=ws)
            parts += home.binds()
    target = load_target_experiment(context.descriptor)
    from merlin.targetgen.sandbox import toolchain as TC

    parts += TC.toolchain_binds(target)
    # These point at operator-only snapshot storage for the out-of-sandbox grader. The agent receives
    # the declared bytes through read-only mounts and never needs (or gets) their host storage path.
    parts += ["--unsetenv", _MODEL_HOST_SNAPSHOT_ROOT_ENV, "--unsetenv", _MODEL_HOST_SNAPSHOT_REQUIRED_ENV]
    parts += ["--unsetenv", _MODEL_HOST_SNAPSHOT_RECORD_ENV]
    parts += ["--unsetenv", "MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"]
    from .. import joint_control
    if joint_control.selected() is not None:
        parts += ["--unsetenv", "MERLIN_JOINT_CAMPAIGN_STATE"]
        parts += ["--unsetenv", "MERLIN_JOINT_PERFORMANCE_SELECTION"]
        parts += ["--unsetenv", "MERLIN_JOINT_PERFORMANCE_REQUEST"]
        parts += ["--unsetenv", "MERLIN_NATIVE_CALIBRATION_INITIAL_REVIEW"]
    # Per-driver runtime binds (e.g. the Codex CLI's package dir + an isolated
    # CODEX_HOME) go in BEFORE the mask pass, so a bind can never re-expose an
    # answer surface: masking is applied last and therefore wins.
    parts += list(extra_binds or [])
    # A trusted toolchain bind may overlap a declared arm grant (LLVM is both).
    # Reassert the frozen per-run bundle after every such bind so no live
    # worktree/tool input can override the snapshot, then apply answer masks.
    parts = _START.call("snapshot_overlay", _assemble, "snapshot_overlay", _BW.reapply_bundle_snapshot,
                        parts, ws, bundle, repo=context.repo)
    from ..compiler_library import sandbox_overlay

    parts, library_environment = _START.call("library_overlay", sandbox_overlay,
        parts, workspace=ws, bundle=bundle, context=context, private_run_dir=private_run_dir
    )
    parts = _START.call("final_answer_masks", _assemble, "final_answer_masks", _BW.apply_final_answer_masks,
        parts, load_target_experiment(context.descriptor), ws, bundle, repo=context.repo
    )
    # Sibling homes hold other launches' sessions; only this launch's home, at the CLI's paths, is visible.
    if home is not None and _BW.is_exposed(parts, home.root.parent):
        raise RuntimeError("staged Claude homes are visible at their host path in the agent sandbox")
    if private_run_dir is not None:
        from merlin_experiments.phase1.semantic_diagnostics import assert_private_mounts

        assert_private_mounts(parts, private_run_dir)
    from .. import joint_control
    if joint_control.selected() is not None:
        parts += ["--setenv", "MERLIN_JOINT_DEVELOPMENT", "1"]
    tool_environment = TC.sandbox_env(load_target_experiment(context.descriptor), ws, bundle=bundle)
    return SandboxPlan(tuple(parts), tool_environment + " " + library_environment, ws)


def sandbox_script(rounds: Path, rnd: int, turn: int, command: str) -> list[str]:
    """Run a bwrap command from a FILE rather than as a ``bash -c`` argument.

    execve refuses any single argument over MAX_ARG_STRLEN (128 KiB) with E2BIG, and the error names
    only the interpreter -- `Argument list too long: 'bash'` -- so a size refusal is indistinguishable
    from a missing shell. Measured 2026-09-06 on the perf campaign: the composed sandbox command was
    178,303 bytes and every trial died before its first round. The bind list has a size escape already
    (``bwrap --args`` from a file descriptor), but the shell string that carries it does not, and the
    mask set grows with the corpus, so the ceiling is reachable by adding capsules.

    A file has no such limit. It lives beside the round's other artifacts and NOT in the workspace:
    the agent must not be able to read or edit the command that sandboxes it.
    """
    import os
    import tempfile

    script = rounds / f"round_{rnd:02d}.sandbox.turn{turn:02d}.sh"
    # A completed launch seals this file 0500.  A checkpoint resume may retry the same round/turn and
    # therefore the same path; opening the sealed inode for writing fails before Codex starts.  Replace
    # it atomically with a newly sealed inode instead of temporarily making the old launch record
    # writable.  The behaviour is identical for every driver invocation and independent of target.
    rounds.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=rounds, prefix=f".{script.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(command)
            staged = Path(handle.name)
        staged.chmod(0o500)
        os.replace(staged, script)
        staged = None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return ["bash", str(script)]


@dataclass(frozen=True)
class SandboxPlan:
    """One reviewed mount policy; composing a command never adds a grant."""
    parts: tuple[str, ...]
    environment: str
    workspace: Path

    def command(self, inner: str) -> str:
        from merlin.targetgen.sandbox import bwrap
        payload = self.environment + inner
        return bwrap.compose_command(list(self.parts), " bash -c " + shlex.quote(payload), self.workspace)


def sandbox_command(inner, ws, bundle, extra_binds=None, **options):
    return sandbox_plan(inner, ws, bundle, extra_binds, **options).command(inner)


@dataclass(frozen=True)
class CodexSandboxCommand:
    """Launch-local plan for the permission probe and initial paid command.

    Continuations rebuild their policy. Both adjacent initial commands use the
    exact same runtime grants and final masks; no payload can add a grant.
    """
    context: InvocationContext
    private_run_dir: Path
    runtime: object | None = None

    @with_priority(AUTHOR)
    def prepare(self, ws, bundle, extra_binds=None):
        if self.runtime is not None:
            _verify_codex_runtime(self.runtime, self.context, ws, bundle)
        return sandbox_plan("", ws, bundle, extra_binds, context=self.context,
                            private_run_dir=self.private_run_dir, codex_mode=True)

    def __call__(self, inner, ws, bundle, extra_binds=None):
        return self.prepare(ws, bundle, extra_binds).command(inner)


def _verify_codex_runtime(runtime, context, ws, bundle):
    from merlin.targetgen.sandbox import toolchain as TC
    from merlin.targetgen.target_experiment import load_target_experiment

    _START.call("runtime_verify", runtime.verify)
    target = load_target_experiment(context.descriptor)
    if (tuple(TC.toolchain_binds(target)) != runtime.toolchain_mounts
            or TC.sandbox_env(target, ws, bundle=bundle) != runtime.tool_environment):
        raise ValueError("explicit client toolchain mount selection changed")


@with_priority(AUTHOR)
def codex_sandbox_command(inner, ws, bundle, extra_binds=None, *, context, private_run_dir, runtime):
    return CodexSandboxCommand(context, private_run_dir, runtime)(inner, ws, bundle, extra_binds)


def codex_call_kwargs(runtime, *, context, run_dir, model) -> dict:
    if runtime is None:
        return {"sandbox_command": CodexSandboxCommand(context, run_dir)}
    return {
        **runtime.round_kwargs(),
        "sandbox_command": CodexSandboxCommand(context, run_dir, runtime),
    }


def launch(
    ws: Path,
    run_dir: Path,
    model: str,
    effort: str,
    sandbox: str,
    bundle: dict,
    rnd: int,
    timeout: int,
    arm: str = "raw_baseline",
    continuous: bool = False,
    *,
    config: ExecutionConfig,
    capsules_root: Path | None = None,
    policy_root: Path | None = None,
    contract: Path | None = None,
) -> tuple[int, Path]:
    from merlin.targetgen.target_experiment import load_target_experiment

    # TASK.md must live INSIDE the bound workspace: run_dir is under runs/ which bwrap tmpfs-masks,
    # so a stdin redirect from run_dir/TASK.md is invisible inside the sandbox (empty stdin).
    ws_task = ws / "TASK.md"
    # Setup stages and seals the task BEFORE writing environment.yaml.  Rebuilding here used to leave
    # environment provenance unable to say what prompt was served, and a resumed round zero could silently
    # pick up changed bundle prose.  Missing after setup is corruption, not an invitation to regenerate.
    if not ws_task.is_file():
        raise RuntimeError(f"sealed task is missing before agent launch: {ws_task}")
    # Under bwrap the oracle is masked, so the agent's self-check (agent_selfcheck.py) can't grade in-box.
    # Start the driver-side BROKER (oracle available, outside the sandbox) + stage the in-box shim, so the
    # agent gets a REDACTED on-demand self-check (numeric diff, no goldens) without the oracle ever entering
    # its sandbox. Started BEFORE the backend split so BOTH the claude CLI and the Bedrock Converse agent
    # get the identical mid-round feedback loop.
    broker = (
        FL.start_brokers(
            ws,
            FL.BrokerConfig(
                config.context,
                tuple(config.resolved_tools()),
                config.timing_file,
                config.sim_max_jobs,
                capsules_root,
                policy_root,
                contract,
                config.selected_rtl_facts,
                run_dir / "input_bundle_manifest.yaml" if bundle.get("compiler_tool_profile") is not None else None,
                sim_per_capsule_timeout=config.sim_per_capsule_timeout,
                raw_evidence_root=(run_dir / "_qa_work" / "selfchecks").resolve(),
            ),
        )
        if sandbox == "bwrap"
        else None
    )
    model_gate = None
    if config.model_gate_interval > 0 and sandbox == "bwrap":
        from merlin_experiments.phase1.feedback import visible_model_gate

        model_gate = visible_model_gate.start(
            ws, run_dir, interval_s=config.model_gate_interval, context=config.context,
            compiler_bundle_manifest=(run_dir / "input_bundle_manifest.yaml"
                                      if bundle.get("compiler_tool_profile") is not None else None),
        )
    primary_error = None
    calibration_watcher = None
    try:
        if sandbox == "bwrap":
            from .. import joint_calibration_lifecycle
            calibration_watcher = joint_calibration_lifecycle.start_for_author(
                run_dir=run_dir, context=config.context, round_index=rnd)
        # Route to the selected driver (explicit --driver, or auto-by-model-id). A non-Anthropic model can't
        # drive the claude CLI (Anthropic API only) → the Bedrock Converse backend runs the same masked-
        # sandbox agentic loop (incl. the self_check tool wired to the shim above) + a compatible transcript;
        # 'opencode' drives the provider-agnostic OpenCode CLI; 'claudecode'/Anthropic uses the claude CLI.
        drv = resolve_driver(model, config=config.provider)
        from ..conformance_sources import execution_context

        conformance_context = execution_context(
            ws,
            config.context.target,
            config.selected_rtl_facts,
            [
                ws / "rtl_facts/facts.json",
                *([bundle["selected_rtl_facts_file"]] if bundle.get("selected_rtl_facts_file") else []),
            ],
        )
        if config.provider.codex_runtime is not None and drv != "codex":
            raise ValueError("explicit client runtime requires the Codex provider")
        if config.provider.codex_max_subagents is not None and (drv != "codex" or sandbox != "bwrap"):
            raise ValueError("Codex subagents require the Codex provider and bwrap sandbox")
        # A (model, harness) pairing that needs the bridge needs the proxy running. Started here rather
        # than by the launcher so EVERY entry point (launch_ab_batch, chia_ab_batch, watchdog resume, a
        # bare run) gets it, and idempotently so concurrent arms of one campaign share one instance.
        from merlin_experiments.phase1.providers import agent_bridge as _BR

        if _BR.bridged_name(model, drv):
            _pi = _BR.start_proxy(run_dir / "logs" / "litellm_proxy.log")
            (run_dir / "bridge.json").write_text(
                json.dumps({**_BR.record(model, harness=drv), "proxy_started": _pi}, indent=1)
            )
        if drv == "converse":
            from merlin_experiments.phase1.providers import bedrock_agent as _BA

            return _BA.run_round(
                ws,
                run_dir,
                model,
                bundle,
                load_target_experiment(config.context.descriptor),
                sandbox,
                rnd,
                timeout,
                subagent_model=config.provider.subagent_model,
                background_model=config.provider.background_model,
            )
        if drv == "opencode":
            try:
                from merlin_experiments.phase1.providers import opencode_agent as _OA
            except ImportError as e:  # Phase 3 not landed yet
                raise SystemExit(f"--driver opencode is not available yet: {e}")
            # effort threads here for the same reason it does for codex below — see that comment.
            return _OA.run_round(
                ws,
                run_dir,
                model,
                bundle,
                load_target_experiment(config.context.descriptor),
                sandbox,
                rnd,
                timeout,
                subagent_model=config.provider.subagent_model,
                background_model=config.provider.background_model,
                effort=effort,
                sandbox_command=partial(sandbox_command, context=config.context, private_run_dir=run_dir),
                conformance_context=conformance_context,
            )
        if drv == "codex":
            from .. import joint_control
            try:
                from merlin_experiments.phase1.providers import codex_agent as _CA
            except ImportError as e:
                raise SystemExit(f"--driver codex is not available: {e}")
            # effort is threaded through: codex takes it as a config override, and an
            # arm that silently ran at a different reasoning effort is a different arm.
            # `codex exec` is one TURN, not one session -- it returns when the model stops. Under
            # --schedule continuous the harness launches ONE session and expects it to spend the wall
            # budget, so without this a run ends after a single turn with most of the budget unspent
            # (measured four times; the last left 14772s of 23890s). Keep the SAME thread alive.
            return _CA.run_round(
                ws,
                run_dir,
                model,
                bundle,
                load_target_experiment(config.context.descriptor),
                sandbox,
                rnd,
                timeout,
                subagent_model=config.provider.subagent_model,
                background_model=config.provider.background_model,
                effort=effort,
                continue_session=continuous,
                **({"continuation_stop": joint_control.stop_reason} if joint_control.selected() else {}),
                conformance_context=conformance_context,
                **(
                    {"max_concurrent_subagents": config.provider.codex_max_subagents}
                    if config.provider.codex_max_subagents is not None
                    else {}
                ),
                # The helper-thread ceiling this run RECORDED (environment.yaml), so the frozen home
                # carries exactly the audited value, including across a resume.
                **author_subagents.launch_kwargs(run_dir, sandbox),
                # The recorded context ceiling and task-skill selection (codex_context), likewise.
                **codex_context.launch_kwargs(run_dir, sandbox),
                **codex_call_kwargs(
                    config.provider.codex_runtime, context=config.context, run_dir=run_dir, model=model
                ),
            )
        # claudecode. The claude CLI speaks the Anthropic Messages API, so a NON-Anthropic model reaches
        # it only through the LiteLLM bridge (ANTHROPIC_BASE_URL -> our proxy -> Bedrock). This is what
        # makes the harness the experimental variable instead of a fixed property of the model: nemotron
        # and glm5 can now be driven by the same harness that drives opus. An Anthropic model returns an
        # empty env here and takes its existing native/Bedrock path unchanged.
        from merlin_experiments.phase1.providers import agent_bridge as _BR

        _bridge_env = _BR.claude_env(model)
        _cli_model = _BR.claude_model_name(model, provider=config.provider.provider)
        inner = (
            f"claude --print --model {_cli_model} --effort {effort} "
            f"--permission-mode bypassPermissions --add-dir {ws} "
            f"--output-format stream-json --verbose < {ws_task}"
        )
        if _bridge_env:
            # Exported INSIDE the sandbox: bwrap keeps the network namespace (loopback reaches the
            # proxy) but not the environment.
            _exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in _bridge_env.items())
            inner = f"env {_exports} {inner}"
        cmd = (
            sandbox_command(inner, ws, bundle, context=config.context, private_run_dir=run_dir)
            if sandbox == "bwrap"
            else inner
        )
        tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        epath = run_dir / "rounds" / f"round_{rnd:02d}.stderr.log"
        # Observe each event as it arrives, retaining the untouched stream beside it. The
        # shared launcher also kills the whole bash -> sandbox -> agent group on timeout.
        rc = AS.stream_stamped(
            ["bash", "-c", cmd],
            cwd=ws,
            transcript=tpath,
            stderr_path=epath,
            timeout=timeout,
            raw_path=tpath.with_name(f"round_{rnd:02d}.stream.raw.jsonl"),
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if model_gate is not None:
            visible_model_gate.stop_gate(model_gate)
        if calibration_watcher is not None:
            # Keep brokers and leases alive until the current bounded native
            # calibration operation drains. Never kill its simulator children.
            calibration_watcher.stop()
        FL.stop_brokers(ws, broker, primary_error=primary_error)
    return rc, tpath
