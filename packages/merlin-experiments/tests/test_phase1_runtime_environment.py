"""No network/auth/provider launches: explicit host preparation and restoration."""

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import runtime_environment as R
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.options import parse_options


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected auth subprocess")

    monkeypatch.setattr(R.subprocess, "run", forbidden)
    context = InvocationContext(
        tmp_path,
        tmp_path / "descriptor.yaml",
        tmp_path,
        "fixture",
        tmp_path / "runs",
        tmp_path / "reports",
        tmp_path / "bundles",
        (),
    )
    return context, parse_options(["--run-id", "fixture"])


def prepare(inputs, env=None, **options):
    context, arguments = inputs
    return R.prepare_runtime_environment(
        replace(arguments, **options),
        env or {},
        machine_defaults={"MERLIN_EXT_CHIPYARD": "/machine/chipyard", "AWS_BEARER_TOKEN_BEDROCK": "secret-default"},
    )


def test_preparation_is_nonmutating_and_base_wins(inputs):
    before = dict(os.environ)
    base = {
        "MERLIN_EXT_CHIPYARD": "/explicit",
        "LD_LIBRARY_PATH": "original",
        "AWS_DEFAULT_REGION": "kept",
        "AWS_BEARER_TOKEN_BEDROCK": "secret-explicit",
        "CLAUDE_CODE_SUBAGENT_MODEL": "kept-model",
    }
    got = prepare(inputs, base, provider="bedrock", aws_region="selected", aws_profile="profile")
    assert dict(os.environ) == before
    assert got.environment["LD_LIBRARY_PATH"] == "original"
    assert got.environment["AWS_DEFAULT_REGION"] == "kept"
    assert got.environment["AWS_REGION"] == "selected"
    assert got.environment["AWS_PROFILE"] == "profile"
    assert got.environment["AWS_BEARER_TOKEN_BEDROCK"] == "secret-explicit"
    assert got.environment["CLAUDE_CODE_SUBAGENT_MODEL"] == "kept-model"
    assert "secret" not in repr(got)
    assert "secret" not in json.dumps(dict(got.account))
    with pytest.raises(TypeError):
        got.environment["X"] = "change"


@pytest.mark.parametrize("driver", ["auto", "converse", "claudecode", "opencode", "codex"])
@pytest.mark.parametrize("provider", ["subscription", "bedrock"])
@pytest.mark.parametrize("sandbox", ["bwrap", "none"])
def test_retired_readonly_refuses_before_auth_or_provider_setup(inputs, driver, provider, sandbox):
    base = {"MERLIN_PINNED_SUBMISSION_READ_ONLY": " 1 "}
    before = dict(os.environ)
    got = prepare(inputs, base, driver=driver, provider=provider, sandbox=sandbox, account_config_dir="/unused")
    assert got.refusal == 7
    assert "unsupported" in got.message
    assert dict(got.environment) == base
    assert dict(os.environ) == before
    assert got.account == {"config_dir": None, "email": None, "orgId": None}


@pytest.mark.parametrize("value", [None, "", "0"])
def test_retired_readonly_uses_explicit_baseline(inputs, monkeypatch, value):
    monkeypatch.setenv("MERLIN_PINNED_SUBMISSION_READ_ONLY", "1")
    base = {} if value is None else {"MERLIN_PINNED_SUBMISSION_READ_ONLY": value}
    assert prepare(inputs, base).refusal is None


@pytest.mark.parametrize("budget,expected", [(None, "900"), (0, None), (17, "17")])
def test_budget_and_subscription_cleanup(inputs, budget, expected):
    got = prepare(
        inputs, {"CLAUDE_CODE_USE_BEDROCK": "1", "MERLIN_MODEL_BUDGET_S": "old"}, qa_timeout=900, model_budget_s=budget
    )
    assert got.environment.get("MERLIN_MODEL_BUDGET_S") == expected
    assert "CLAUDE_CODE_USE_BEDROCK" not in got.environment


def test_provider_defaults_and_explicit_model_overrides(inputs):
    from merlin_experiments.phase1.providers.model_tiers import resolve

    got = prepare(inputs, provider="bedrock", subagent_model="sonnet", background_model="haiku")
    assert got.environment["AWS_BEARER_TOKEN_BEDROCK"] == "secret-default"
    assert got.environment["CLAUDE_CODE_SUBAGENT_MODEL"] == resolve("sonnet")
    assert got.environment["ANTHROPIC_SMALL_FAST_MODEL"] == resolve("haiku")
    assert got.environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == resolve("haiku")


@pytest.mark.parametrize("available,refusal", [(True, None), (False, 7)])
def test_framework_uses_current_installation_without_integrity_mutation(inputs, monkeypatch, available, refusal):
    from merlin.targetgen import package_runtime

    original = package_runtime._FORBIDDEN
    monkeypatch.setattr(R, "framework_import_roots", lambda: ("/installed/core", "/venv/lib/python3.14/site-packages"))
    monkeypatch.setattr(R.importlib.util, "find_spec", lambda name: object() if available else None)
    got = prepare(inputs, {"PYTHONPATH": "prior"}, experiment="realistic", arm="merlin_assisted")
    assert got.refusal == refusal
    assert got.environment["PYTHONPATH"] == "/installed/core:/venv/lib/python3.14/site-packages:prior"
    assert package_runtime._FORBIDDEN == original


@pytest.mark.parametrize(
    "payload,refusal",
    [('{"loggedIn":true,"email":"e","orgId":"o"}', None), ('{"loggedIn":false}', 6), ("not json secret", 6)],
)
def test_account_probe_explicit_env_and_safe_metadata(inputs, monkeypatch, tmp_path, payload, refusal):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=payload, returncode=0)

    monkeypatch.setattr(R.subprocess, "run", run)
    got = prepare(inputs, provider="bedrock", account_config_dir=str(tmp_path / "account"))
    assert got.refusal == refusal
    argv, kwargs = calls[0]
    assert argv == ["claude", "auth", "status"]
    assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "account")
    assert kwargs["env"]["AWS_BEARER_TOKEN_BEDROCK"] == "secret-default"
    assert kwargs["timeout"] == 60
    assert "secret" not in got.message
    assert set(got.account) <= {"config_dir", "email", "orgId", "loggedIn"}


def test_account_exception_is_sanitized(inputs, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("secret from provider")

    monkeypatch.setattr(R.subprocess, "run", fail)
    got = prepare(inputs, account_config_dir="account")
    assert got.refusal == 6
    assert "secret" not in got.message


def test_application_restores_touched_keys_on_exception(inputs, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "original")
    monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    got = prepare(inputs, dict(os.environ), model_budget_s=19)
    before = dict(os.environ)
    with pytest.raises(RuntimeError), R.applied_environment(got):
        assert "CLAUDE_CODE_USE_BEDROCK" not in os.environ
        assert os.environ["MERLIN_MODEL_BUDGET_S"] == "19"
        raise RuntimeError("stop")
    assert dict(os.environ) == before


def test_import_roots_deduplicate_active_sysconfig(inputs, monkeypatch, tmp_path):
    from merlin.common import paths

    monkeypatch.setattr(paths, "python_import_roots", lambda: (tmp_path / "core",))
    monkeypatch.setattr(
        R.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path / "site"), "platlib": str(tmp_path / "site")}
    )
    assert R.framework_import_roots() == (str(tmp_path / "core"), str(tmp_path / "site"))


@pytest.mark.parametrize("prior", [None, "previous"])
def test_explicit_library_paths_only(inputs, tmp_path, prior):
    context, options = inputs
    base = {} if prior is None else {"LD_LIBRARY_PATH": prior}
    plain = R.prepare_runtime_environment(options, base, machine_defaults={})
    assert plain.environment.get("LD_LIBRARY_PATH") == prior
    selected = R.prepare_runtime_environment(
        options, base, machine_defaults={}, library_paths=(tmp_path / "lib", tmp_path / "vendor")
    )
    assert selected.environment["LD_LIBRARY_PATH"] == f"{tmp_path}/lib:{tmp_path}/vendor:{prior or ''}"


def test_auth_and_applied_environment_share_explicit_baseline(inputs, monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient-token")
    monkeypatch.setenv("PATH", "/ambient/bin")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "ambient-model")
    monkeypatch.setenv("ONLY_AMBIENT", "remove-during-invocation")
    before = dict(os.environ)
    observed = []

    def run(argv, **kwargs):
        observed.append(dict(kwargs["env"]))
        return SimpleNamespace(stdout='{"loggedIn": true}')

    monkeypatch.setattr(R.subprocess, "run", run)
    base = {
        "AWS_BEARER_TOKEN_BEDROCK": "explicit-token",
        "PATH": "/selected/bin",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "selected-model",
    }
    prepared = prepare(inputs, base, provider="bedrock", account_config_dir=str(tmp_path))
    with R.applied_environment(prepared):
        assert dict(os.environ) == observed[0] == dict(prepared.environment)
        assert "ONLY_AMBIENT" not in os.environ
    assert dict(os.environ) == before


@pytest.mark.parametrize("raises", [False, True])
def test_continuation_environment_changes_never_escape(inputs, monkeypatch, raises):
    monkeypatch.setenv("UNCHANGED_AT_ENTRY", "original")
    monkeypatch.delenv("CONTINUATION_NEW_KEY", raising=False)
    before = dict(os.environ)
    prepared = prepare(inputs, before)

    def invoke():
        with R.applied_environment(prepared):
            os.environ["UNCHANGED_AT_ENTRY"] = "changed-by-continuation"
            os.environ["CONTINUATION_NEW_KEY"] = "new"
            os.environ.pop("PATH", None)
            if raises:
                raise RuntimeError("continuation failed")

    if raises:
        with pytest.raises(RuntimeError, match="continuation failed"):
            invoke()
    else:
        invoke()
    assert dict(os.environ) == before


def test_account_tilde_uses_explicit_home(inputs, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "ambient"))
    captured = []

    def run(argv, **kwargs):
        captured.append(kwargs["env"]["CLAUDE_CONFIG_DIR"])
        return SimpleNamespace(stdout='{"loggedIn": true}')

    monkeypatch.setattr(R.subprocess, "run", run)
    result = prepare(inputs, {"HOME": str(tmp_path / "selected")}, account_config_dir="~/account")
    assert result.refusal is None
    assert captured == [str(tmp_path / "selected/account")]
    assert result.account["config_dir"] == captured[0]
