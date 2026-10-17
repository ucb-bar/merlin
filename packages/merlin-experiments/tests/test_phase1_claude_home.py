"""The Claude author's home: no operator Claude state is mounted, only a home staged for the launch.

Builds the actual Phase 1 sandbox argv for a ``claude`` launch against a synthetic operator home
(every name a real ``~/.claude`` holds) and replays its mount table; one guarded test runs it
under bwrap. Controls show the same check flags a bound ``~/.claude``. A mount-table replay is
policy regression, not operating-system isolation qualification.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase1.providers import execution as E

from merlin.targetgen import target_experiment
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import claude_home as CH
from merlin.targetgen.sandbox import toolchain
from merlin.targetgen.sandbox.preflight import probe_sandbox

MARKER = "synthetic operator material"
CREDENTIAL = "synthetic credential"
#: Operator state a real ~/.claude holds, by name: prompt history, edit backups, plans, todos,
#: session transcripts and memory, shell snapshots, settings and hooks, instructions, skills, caches.
OPERATOR_FILES = (
    "history.jsonl",
    "file-history/session/backup@v1",
    "plans/reset-charter.md",
    "todos/agent-todos.json",
    "projects/-work-repo/session.jsonl",
    "projects/-work-repo/memory/MEMORY.md",
    "shell-snapshots/snapshot-bash.sh",
    "sessions/session.json",
    "session-env/session/env",
    "settings.json",
    "settings.json.before-task",
    "settings.local.json",
    "CLAUDE.md",
    "skills/operator-skill/SKILL.md",
    "plugins/installed.json",
    "backups/.claude.json.backup",
    "paste-cache/paste.txt",
    "ide/lock",
    "statsig/cache",
    "stats-cache.json",
    "cache/changelog.md",
    "downloads/download.bin",
)


@pytest.fixture
def operator(tmp_path, monkeypatch):
    home = tmp_path / "operator-home"
    directory = home / ".claude"
    for name in OPERATOR_FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(MARKER)
    (directory / ".credentials.json").write_text(CREDENTIAL)
    (home / ".claude.json").write_text(json.dumps({"projects": {"/work/repo": {"history": [MARKER]}}}))
    launcher = home / ".local/bin"
    launcher.mkdir(parents=True)
    (home / ".local/share/claude/versions").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return SimpleNamespace(
        home=home,
        directory=directory,
        config=home / ".claude.json",
        credential=directory / ".credentials.json",
        homes=tmp_path / "out/artifacts/cache/claude_home",
    )


@pytest.fixture
def composer(tmp_path, monkeypatch, operator):
    """The real provider composition; only target loading, toolchain and the shell string are stubbed."""
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = tmp_path / "run/workspace"
    workspace.mkdir(parents=True)
    run_dir = tmp_path / "host-run"
    run_dir.mkdir()
    BW.materialize_bundle_inputs(workspace, {}, repo=repo)
    context = InvocationContext(repo, repo / "descriptor", repo, "fixture", repo, repo, repo, ())
    target = SimpleNamespace(target="fixture", capsule_corpus=None, corpus_siblings=lambda: [])
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: target)
    monkeypatch.setattr(BW, "repo_root", lambda: repo)
    monkeypatch.setattr(toolchain, "toolchain_binds", lambda te: [])
    monkeypatch.setattr(toolchain, "sandbox_env", lambda te, ws: "true;")
    monkeypatch.setattr(BW, "answer_surfaces", lambda te: [])
    surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(surfaces, "answer_surfaces", lambda te: [])
    monkeypatch.setattr(BW, "compose_command", lambda argv, payload, ws: argv)

    def command(inner, extra=None, **options):
        return E.sandbox_command(inner, workspace, {}, extra, context=context, private_run_dir=run_dir, **options)

    yield SimpleNamespace(command=command, ws=workspace, repo=repo, run_dir=run_dir, target=target, context=context)
    BW.remove_bundle_snapshot(workspace)


def _operator_paths(operator) -> list[Path]:
    """Everything of the operator's Claude state except the credential file the CLI must reach."""
    return [path for path in operator.directory.rglob("*") if path != operator.credential] + [operator.config]


def _served_source(argv: list[str], path: Path) -> Path | None:
    """The host bytes the ordered mount table serves at ``path`` (longest destination, then latest)."""
    best = None
    for index, (state, source, destination) in enumerate(BW._mounts(argv)):
        if path.is_relative_to(destination) and (best is None or (len(destination), index) >= best[0]):
            best = ((len(destination), index), state, source, Path(destination))
    if best is None or best[1] == "hide":
        return None
    return Path(best[2]) / path.relative_to(best[3])


def _operator_leaks(argv: list[str], operator) -> list[Path]:
    """Operator Claude paths a mount sources (directly or through a parent), or served at their own path."""
    private = _operator_paths(operator)
    leaks = set()
    for path in private:
        served = _served_source(argv, path)
        if served is not None and served in private and BW.is_exposed(argv, path):
            leaks.add(path)
    for state, source, _destination in BW._mounts(argv):
        if state == "expose":
            leaks.update(path for path in private if path.is_relative_to(Path(source)))
    return sorted(leaks)


def _operator_destinations(argv: list[str], operator) -> list[tuple[str, str]]:
    return [
        (source, destination)
        for state, source, destination in BW._mounts(argv)
        if state == "expose" and CH.is_operator_state(destination, operator.home)
    ]


def _homes(operator) -> list[Path]:
    return sorted(operator.homes.iterdir()) if operator.homes.is_dir() else []


def test_claude_launch_mounts_a_fresh_home_and_no_operator_state(composer, operator):
    argv = composer.command("claude --print --model fixture --output-format stream-json < TASK.md")

    assert _operator_leaks(argv, operator) == []
    [home] = _homes(operator)
    assert home.name.startswith(f"{composer.run_dir.name}_")
    # Mounted where the CLI looks; the credential is the only operator file any mount names.
    assert _operator_destinations(argv, operator) == [
        (str(home / ".claude"), str(operator.directory)),
        (str(home / ".claude.json"), str(operator.config)),
        (str(operator.credential), str(operator.credential)),
    ]
    sources = [Path(source) for state, source, _ in BW._mounts(argv) if state == "expose"]
    assert [source for source in sources if CH.is_operator_state(source, operator.home)] == [operator.credential]
    assert BW.is_exposed(argv, operator.credential)
    assert argv[argv.index("CLAUDE_CONFIG_DIR") - 1] == "--unsetenv"
    # Nothing of the operator's was copied: an empty directory, a credential mount point, a minimal config.
    assert sorted(path.relative_to(home) for path in home.rglob("*")) == [
        Path(".claude"),
        Path(".claude/.credentials.json"),
        Path(".claude.json"),
    ]
    assert (home / ".claude/.credentials.json").read_text() == ""
    assert json.loads((home / ".claude.json").read_text()) == {"hasCompletedOnboarding": True}

    composer.command("env A=b claude --print < TASK.md")
    assert len(_homes(operator)) == 2, "a launch never reuses another launch's home"


def test_leak_check_flags_a_bound_operator_claude_home(composer, operator):
    """Control: the composition before this fix, and a bind slipped into the real composer, are caught."""
    legacy = BW.base_argv(composer.ws, {}, repo=composer.repo) + BW.claude_runtime_binds()
    leaks = set(_operator_leaks(legacy, operator))
    for name in ("history.jsonl", "file-history/session/backup@v1", "plans/reset-charter.md", "todos/agent-todos.json"):
        assert operator.directory / name in leaks
        assert BW.is_exposed(legacy, operator.directory / name)
    for name in ("shell-snapshots/snapshot-bash.sh", "sessions/session.json", "settings.json", "CLAUDE.md"):
        assert operator.directory / name in leaks
    assert operator.config in leaks

    injected = composer.command("claude --print", ["--ro-bind", str(operator.directory), str(operator.directory)])
    assert operator.directory / "history.jsonl" in _operator_leaks(injected, operator)
    assert operator.directory / "plans/reset-charter.md" in _operator_leaks(injected, operator)
    config_bind = composer.command("claude --print", ["--ro-bind", str(operator.config), "/agent/config.json"])
    assert operator.config in _operator_leaks(config_bind, operator)


@pytest.mark.parametrize("inner", ["opencode run --model fixture 'Read TASK.md'", "test -s probe && printf DONE"])
def test_commands_that_do_not_run_claude_get_no_claude_state(composer, operator, inner):
    argv = composer.command(inner)
    assert _operator_leaks(argv, operator) == []
    assert _operator_destinations(argv, operator) == []
    assert not BW.is_exposed(argv, operator.credential)
    assert _homes(operator) == []


def test_converse_policy_mounts_no_operator_claude_state(composer, operator):
    argv = BW.full_argv(composer.target, composer.ws, {})
    assert _operator_leaks(argv, operator) == []
    assert _operator_destinations(argv, operator) == []
    assert not BW.is_exposed(argv, operator.credential)


def test_codex_composition_is_unchanged(composer, operator, tmp_path, monkeypatch):
    real_codex = operator.home / ".codex"
    real_codex.mkdir()
    (real_codex / "auth.json").write_text(CREDENTIAL)
    monkeypatch.setenv("CODEX_HOME", str(real_codex))
    codex_home = tmp_path / "isolated-codex-home"
    codex_home.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("the Codex provider must not request Claude runtime mounts or a Claude home")

    monkeypatch.setattr(BW, "claude_runtime_binds", forbidden)
    monkeypatch.setattr(CH, "stage", forbidden)
    extra = CA.codex_runtime_binds(codex_home)
    for inner in ("codex exec --json -", "claude --print"):
        argv = composer.command(inner, extra, codex_mode=True)
        # The historical Codex composition, mount for mount.
        expected = BW.base_argv(
            composer.ws, {}, repo=composer.repo, include_claude_home=False, inherit_environment=False
        )
        expected += ["--unsetenv", "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"]
        expected += ["--unsetenv", "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"]
        expected += ["--unsetenv", "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"]
        expected += ["--unsetenv", "MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION", *extra]
        expected = BW.reapply_bundle_snapshot(expected, composer.ws, {}, repo=composer.repo)
        expected = BW.apply_final_answer_masks(expected, composer.target, composer.ws, {}, repo=composer.repo)
        assert argv == expected
        assert _operator_leaks(argv, operator) == []
        assert _operator_destinations(argv, operator) == []
    assert _homes(operator) == []


def test_bedrock_launch_gets_no_oauth_credential(composer, operator, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    argv = composer.command("claude --print")
    [home] = _homes(operator)
    assert _operator_leaks(argv, operator) == []
    assert _operator_destinations(argv, operator) == [
        (str(home / ".claude"), str(operator.directory)),
        (str(home / ".claude.json"), str(operator.config)),
    ]
    assert not BW.is_exposed(argv, operator.credential)
    assert sorted(path.relative_to(home) for path in home.rglob("*")) == [Path(".claude"), Path(".claude.json")]


def test_staged_homes_visible_at_their_host_path_are_refused(composer, operator):
    out = operator.homes.parent.parent.parent
    with pytest.raises(RuntimeError, match="staged Claude homes are visible"):
        composer.command("claude --print", ["--ro-bind", str(out), str(out)])


def test_staging_refuses_overlap_and_unsafe_names(operator, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for root in (workspace / "homes", tmp_path, operator.home, operator.directory / "homes"):
        with pytest.raises(ValueError, match="overlap"):
            CH.stage("run", workspace=workspace, root=root)
    for name in ("", ".", "..", "a/b", "../escape"):
        with pytest.raises(ValueError, match="path component"):
            CH.stage(name, workspace=workspace, root=tmp_path / "homes")
    first = CH.stage("run", workspace=workspace, root=tmp_path / "homes")
    second = CH.stage("run", workspace=workspace, root=tmp_path / "homes")
    assert first.root != second.root and first.root.parent == second.root.parent == tmp_path / "homes"
    assert first.root.stat().st_mode & 0o777 == 0o700
    assert (first.root / ".claude.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("claude --print --model m < TASK.md", True),
        ("env ANTHROPIC_BASE_URL='http://127.0.0.1:4000' claude -p", True),
        ("timeout 60 /opt/bin/claude --print", True),
        ("opencode run --model m 'Read TASK.md'", False),
        ("codex exec --json -", False),
        ("echo claudette", False),
        ("printf 'unbalanced", False),
    ],
)
def test_claude_launch_detection(command, expected):
    assert CH.launches_claude(command) is expected


def test_without_operator_state_keeps_every_other_mount(operator):
    keep = ["--ro-bind", "/usr", "/usr", "--setenv", "A", "b", "--tmpfs", "/tmp"]
    drop = ["--bind", str(operator.config), str(operator.config), "--tmpfs", str(operator.directory / "projects")]
    drop += ["--ro-bind", "/elsewhere", str(operator.directory / "plans")]
    assert CH.without_operator_state([*keep, *drop]) == keep


_LIVE_PROBE = r"""
import json, pathlib, sys
home = pathlib.Path(sys.argv[1])
directory = home / ".claude"
leaked = []
for name in json.loads(sys.argv[2]):
    try:
        if (directory / name).read_text() == "synthetic operator material":
            leaked.append(name)
    except OSError:
        pass
try:
    if "synthetic operator material" in (home / ".claude.json").read_text():
        leaked.append(".claude.json")
except OSError:
    pass
assert not leaked, leaked
assert (directory / ".credentials.json").read_text() == "synthetic credential"
(directory / ".credentials.json").write_text("rotated credential")
(directory / "history.jsonl").write_text("author history")
print(sorted(path.name for path in directory.iterdir()))
"""


def test_live_claude_home_holds_no_operator_state(composer, operator):
    probe = probe_sandbox()
    if not probe.usable:
        pytest.skip(probe.describe())
    names = json.dumps(list(OPERATOR_FILES))
    argv = composer.command("claude --print")
    result = subprocess.run(
        [*argv, "/usr/bin/python3", "-c", _LIVE_PROBE, str(operator.home), names],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['.credentials.json', 'history.jsonl']"
    [home] = _homes(operator)
    # The author's writes land in its own home; a token rotation reaches the operator's credential.
    assert (home / ".claude/history.jsonl").read_text() == "author history"
    assert (operator.directory / "history.jsonl").read_text() == MARKER
    assert operator.credential.read_text() == "rotated credential"
    assert (home / ".claude/.credentials.json").read_text() == ""

    # Live control: the composition before this fix lets the same probe read operator state.
    operator.credential.write_text(CREDENTIAL)
    legacy = BW.base_argv(composer.ws, {}, repo=composer.repo) + BW.claude_runtime_binds()
    control = subprocess.run(
        [*legacy, "/usr/bin/python3", "-c", _LIVE_PROBE, str(operator.home), names],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert control.returncode != 0 and "history.jsonl" in control.stderr, control.stderr
