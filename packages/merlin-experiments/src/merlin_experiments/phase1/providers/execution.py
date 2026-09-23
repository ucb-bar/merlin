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
from merlin.targetgen import experiment_tokens as ET

from ..context import InvocationContext
from ..feedback import lifecycle as FL

_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"
_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"
_MODEL_HOST_SNAPSHOT_RECORD_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"


@dataclass(frozen=True)
class ProviderConfig:
    driver: str = "auto"
    provider: str = "subscription"
    subagent_model: str = ""
    background_model: str = ""


@dataclass(frozen=True)
class ExecutionConfig:
    context: InvocationContext
    provider: ProviderConfig
    resolved_tools: Callable[[], tuple[str, ...]]
    timing_file: Path
    sim_max_jobs: int = 0


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


def sandbox_command(
    inner: str, ws: Path, bundle: dict, extra_binds: list[str] | None = None, *, context: InvocationContext
) -> str:
    """bwrap argv (deny-by-default) + claude runtime binds + TOOLCHAIN binds (the legit build+sim tools,
    bound back over the /scratch* masks) + the DERIVED answer-mask pass + toolchain env. The mask set now
    comes from the shared descriptor-driven answer surface (goldens/hidden/prior/oracle/grader/memory) and
    is coverage-proven by test_sandbox_isolation; masking is bundle-independent (a bind that re-exposes an
    answer surface is re-masked here, not left to the bundle's denied list)."""
    from merlin.targetgen.sandbox import bwrap as _BW
    from merlin.targetgen.target_experiment import load_target_experiment

    parts = _BW.base_argv(ws, bundle, repo=context.repo) + _BW.claude_runtime_binds()
    target = load_target_experiment(context.descriptor)
    from merlin.targetgen.sandbox import toolchain as TC

    parts += TC.toolchain_binds(target)
    # These point at operator-only snapshot storage for the out-of-sandbox grader. The agent receives
    # the declared bytes through read-only mounts and never needs (or gets) their host storage path.
    parts += ["--unsetenv", _MODEL_HOST_SNAPSHOT_ROOT_ENV, "--unsetenv", _MODEL_HOST_SNAPSHOT_REQUIRED_ENV]
    parts += ["--unsetenv", _MODEL_HOST_SNAPSHOT_RECORD_ENV]
    # Per-driver runtime binds (e.g. the Codex CLI's package dir + an isolated
    # CODEX_HOME) go in BEFORE the mask pass, so a bind can never re-expose an
    # answer surface: masking is applied last and therefore wins.
    parts += list(extra_binds or [])
    # A trusted toolchain bind may overlap a declared arm grant (LLVM is both).
    # Reassert the frozen per-run bundle after every such bind so no live
    # worktree/tool input can override the snapshot, then apply answer masks.
    parts = _BW.reapply_bundle_snapshot(parts, ws, bundle, repo=context.repo)
    parts = _BW.apply_final_answer_masks(
        parts, load_target_experiment(context.descriptor), ws, bundle, repo=context.repo
    )
    payload = f"{TC.sandbox_env(load_target_experiment(context.descriptor), ws)} {inner}"
    # Single-quote the whole payload for the OUTER `bash -c`, escaping any embedded single quotes (the
    # POSIX '\'' idiom). ``inner`` may itself be shlex-quoted by the caller (the opencode driver quotes
    # its prompt arg, e.g. "…(if present)…"), so a naive f"…'{inner}'" would let those quotes close the
    # wrapper early and expose a `(` to the outer shell (opencode arm died rc=2 on exactly this). The
    # INNER bash still re-parses the payload as a script, so $(…)/\( \) in mask_selftest keep working.
    # Through the SHARED composer, which moves the bind list into a file descriptor once the string
    # would exceed the execve per-argument limit. This whole string is handed to `bash -c` as ONE
    # argument, and a single execve argument may not exceed MAX_ARG_STRLEN (128 KiB). At the 1,172
    # answer surfaces this checkout carries, the /dev/null and tmpfs masks alone are ~167 KB, so the
    # inline form dies with E2BIG naming `bash` before the first round -- for every arm.
    #
    # This line was already fixed once (34e0296f, "one composer for the argv size rule") and the fix
    # was lost again in a later merge that took the pre-fix side of this file. Restored here, and now
    # pinned by merlin/tests/infra/test_bwrap_cmd_argv_size.py so a merge cannot silently drop it a
    # third time.
    return _BW.compose_command(parts, " bash -c '" + payload.replace("'", "'\\''") + "'", ws)


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
            ),
        )
        if sandbox == "bwrap"
        else None
    )
    primary_error = None
    try:
        # Route to the selected driver (explicit --driver, or auto-by-model-id). A non-Anthropic model can't
        # drive the claude CLI (Anthropic API only) → the Bedrock Converse backend runs the same masked-
        # sandbox agentic loop (incl. the self_check tool wired to the shim above) + a compatible transcript;
        # 'opencode' drives the provider-agnostic OpenCode CLI; 'claudecode'/Anthropic uses the claude CLI.
        drv = resolve_driver(model, config=config.provider)
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
                sandbox_command=partial(sandbox_command, context=config.context),
            )
        if drv == "codex":
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
                sandbox_command=partial(sandbox_command, context=config.context),
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
        cmd = sandbox_command(inner, ws, bundle, context=config.context) if sandbox == "bwrap" else inner
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
        FL.stop_brokers(ws, broker, primary_error=primary_error)
    return rc, tpath
