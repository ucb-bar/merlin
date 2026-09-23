"""Invocation-scoped host setup, independent of native launcher location.

Preparation does not mutate process state. Application is for serialized controller
invocations only: existing graders inherit os.environ, so parallel experiments must
use separate processes. This is not an integrity-policy or credential sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sysconfig
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .options import RunOptions

RETIRED_READONLY_ENV = "MERLIN_PINNED_SUBMISSION_READ_ONLY"
RETIRED_READONLY_MESSAGE = "pinned-submission read-only mode is unsupported; no certified-resume policy is active"


@dataclass(frozen=True)
class PreparedEnvironment:
    """Host-only environment and deliberately small, receipt-safe account metadata.

    Never serialize this object: environment values can contain credentials. Only
    account is suitable for the existing environment receipt.
    """

    environment: Mapping[str, str] = field(repr=False)
    account: Mapping[str, object]
    refusal: int | None = None
    message: str = ""


def framework_import_roots() -> tuple[str, ...]:
    """Active installation roots; no compatibility promise for another Python ABI."""
    from merlin.common.paths import python_import_roots

    paths = sysconfig.get_paths()
    roots = (*python_import_roots(), paths.get("purelib"), paths.get("platlib"))
    return tuple(dict.fromkeys(str(Path(root).resolve()) for root in roots if root))


def prepare_runtime_environment(
    options: RunOptions,
    base_env: Mapping[str, str],
    *,
    machine_defaults: Mapping[str, str],
    library_paths: tuple[Path, ...] = (),
) -> PreparedEnvironment:
    """Prepare the historical host setup with explicit machine configuration.

    The native edge resolves .env through its existing reader and supplies defaults;
    explicit base_env values win. Account probing is the only subprocess, runs only
    when requested, and receives the prepared environment explicitly. Errors never
    echo provider credentials or subprocess output.
    """
    environment = dict(base_env)
    account: dict[str, object] = {"config_dir": None, "email": None, "orgId": None}

    def set_value(key, value):
        environment[key] = value

    def default(key, value):
        if key not in environment:
            set_value(key, value)

    def remove(key):
        environment.pop(key, None)

    def result(refusal=None, message=""):
        return PreparedEnvironment(
            MappingProxyType(dict(environment)), MappingProxyType(dict(account)), refusal, message
        )

    # Refuse the historical request before authentication or provider setup. A
    # mount alone would not restore its removed hash-bound resume lifecycle.
    if environment.get(RETIRED_READONLY_ENV, "").strip() == "1":
        return result(7, RETIRED_READONLY_MESSAGE)

    budget = options.qa_timeout if options.model_budget_s is None else options.model_budget_s
    if budget:
        set_value("MERLIN_MODEL_BUDGET_S", str(budget))
    else:
        remove("MERLIN_MODEL_BUDGET_S")
    if library_paths:
        set_value(
            "LD_LIBRARY_PATH",
            os.pathsep.join(map(str, library_paths)) + os.pathsep + environment.get("LD_LIBRARY_PATH", ""),
        )
    if options.provider == "bedrock":
        from .providers.model_tiers import resolve

        set_value("CLAUDE_CODE_USE_BEDROCK", "1")
        set_value("AWS_REGION", options.aws_region)
        default("AWS_DEFAULT_REGION", options.aws_region)
        if options.aws_profile:
            set_value("AWS_PROFILE", options.aws_profile)
        if not environment.get("AWS_BEARER_TOKEN_BEDROCK"):
            bearer = machine_defaults.get("AWS_BEARER_TOKEN_BEDROCK")
            if bearer:
                set_value("AWS_BEARER_TOKEN_BEDROCK", bearer)
        if options.subagent_model:
            set_value("CLAUDE_CODE_SUBAGENT_MODEL", resolve(options.subagent_model))
        if options.background_model:
            background = resolve(options.background_model)
            set_value("ANTHROPIC_SMALL_FAST_MODEL", background)
            set_value("ANTHROPIC_DEFAULT_HAIKU_MODEL", background)
        for key, value in {
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "us.anthropic.claude-opus-4-6-v1",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "us.anthropic.claude-sonnet-4-6",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "ANTHROPIC_SMALL_FAST_MODEL": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "CLAUDE_CODE_SUBAGENT_MODEL": "us.anthropic.claude-sonnet-4-6",
        }.items():
            default(key, value)
    else:
        remove("CLAUDE_CODE_USE_BEDROCK")

    if options.experiment == "realistic" and options.arm == "merlin_assisted":
        roots = framework_import_roots()
        previous = environment.get("PYTHONPATH")
        set_value("PYTHONPATH", os.pathsep.join((*roots, *((previous,) if previous else ()))))
        if importlib.util.find_spec("xdsl") is None:
            return result(7, "xdsl is unavailable in the active controller interpreter")

    if options.account_config_dir:
        selected = options.account_config_dir
        if selected == "~" or selected.startswith("~/"):
            # Match expanduser's HOME-then-user-database precedence, but against
            # the prepared baseline rather than another invocation's ambient HOME.
            import pwd

            home = environment.get("HOME")
            if home is None:
                home = pwd.getpwuid(os.getuid()).pw_dir
            selected = home + selected[1:]
        else:
            # Explicit ~user expansion uses the user database, not HOME.
            selected = str(Path(selected).expanduser())
        config = str(Path(selected).absolute())
        set_value("CLAUDE_CONFIG_DIR", config)
        account["config_dir"] = config
        try:
            response = subprocess.run(
                ["claude", "auth", "status"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            status = json.loads(response.stdout)
            account.update({key: status.get(key) for key in ("email", "orgId", "loggedIn")})
            if not status.get("loggedIn"):
                return result(6, "selected account configuration is not logged in")
        except Exception:  # account refusal must not disclose command output or credentials
            return result(6, "could not read selected account authentication status")
    return result()


@contextmanager
def applied_environment(prepared: PreparedEnvironment):
    """Apply the complete prepared baseline; restore the exact pre-context environment.

    A supplied base may differ from ambient state. Auth and authoring must see
    identical environments, including absent keys. Continuation-created or changed
    keys are also restored. Concurrent use is unsupported.
    """
    before = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(prepared.environment)
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)
