"""Input identity and actual child delivery; no compiler/runtime qualification."""

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_SPEC = importlib.util.spec_from_file_location(
    "native_input_qualification", repo_root() / "build_tools/scripts/qualify_installed.py"
)
Q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(Q)
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
SUITE = "original-pointwise-host"


def _source(tmp_path):
    root = tmp_path / "frontend"
    (root / "m2m").mkdir(parents=True)
    (root / "m2m/__init__.py").write_text("value = 1\n")
    (root / "m2m/other.py").write_text("value = 2\n")
    (root / "pyproject.toml").write_text('[project]\nname="owned-source-control"\nversion="0"\n')
    for argv in (
        ["git", "init", "-q", root],
        ["git", "-C", root, "add", "."],
        [
            "git",
            "-C",
            root,
            "-c",
            "user.name=Control",
            "-c",
            "user.email=control@example.invalid",
            "commit",
            "-qm",
            "source",
        ],
    ):
        subprocess.run(argv, env=ENV, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], env=ENV).decode().strip()
    return root, commit


def _tools(tmp_path):
    prefix = tmp_path / "selected-python"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin/python"
    python.symlink_to(Path(sys.executable).resolve())
    (prefix / "pyvenv.cfg").write_text("owned identity control, not a runnable venv\n")
    return Q.capture_native_tools(
        SUITE,
        ("compiler-python=" + str(python), "mlir-translate=" + sys.executable, "llvm-llc=" + sys.executable),
    )


def test_closed_inputs_reach_actual_clean_child_with_selected_interpreter_prefix(tmp_path, monkeypatch):
    root, commit = _source(tmp_path)
    tools = _tools(tmp_path)
    sources = Q.capture_native_sources(SUITE, ("m2m=" + str(root) + "@" + commit,))
    report = {"commands": [], "native_tools": tools, "native_sources": sources}
    monkeypatch.setenv("MERLIN_UNADMITTED_CONTROL", "excluded")
    monkeypatch.setenv("MERLIN_M2M_DIR", "/unselected/source")
    recorder = Q.Recorder(tmp_path, report, 10)
    output = tmp_path / "observed.json"
    recorder.run(
        "actual-clean-input-delivery",
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import json,os; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith('MERLIN')}))",
        ],
        tmp_path,
        stdout=output,
    )
    assert json.loads(output.read_text()) == Q.native_environment(report)
    assert recorder.environment["MERLIN_COMPILER_PYTHON"] == tools["compiler-python"]["selected_path"]
    assert sources["m2m"]["identity"]["source_files"].keys() == {"m2m/__init__.py", "m2m/other.py", "pyproject.toml"}


@pytest.mark.parametrize("defect", ("link", "prefix", "missing", "source", "untracked", "bytecode", "symlink", "head"))
@pytest.mark.parametrize("during_child", (False, True))
def test_input_drift_refuses_before_or_after_actual_child(tmp_path, defect, during_child):
    root, commit = _source(tmp_path)
    tools = _tools(tmp_path)
    sources = Q.capture_native_sources(SUITE, ("m2m=" + str(root) + "@" + commit,))
    report = {"commands": [], "native_tools": tools, "native_sources": sources}
    recorder = Q.Recorder(tmp_path, report, 10)
    python = Path(tools["compiler-python"]["selected_path"])
    changes = {
        "link": f"p=Path({str(python)!r}); p.unlink(); p.symlink_to('/usr/bin/false')",
        "prefix": f"Path({str(python.parent.parent / 'pyvenv.cfg')!r}).write_text('changed')",
        "missing": f"Path({str(root / 'm2m/other.py')!r}).unlink()",
        "source": f"Path({str(root / 'm2m/other.py')!r}).write_text('value=3')",
        "untracked": f"Path({str(root / 'm2m/new.py')!r}).write_text('value=3')",
        "bytecode": f"Path({str(root / 'm2m/untracked.pyc')!r}).write_bytes(b'owned invalid bytecode')",
        "symlink": f"Path({str(root / 'm2m/new.py')!r}).symlink_to({str(root / 'm2m/other.py')!r})",
        "head": f"Path({str(root / '.git/HEAD')!r}).write_text('0'*40+'\\n')",
    }
    code = "from pathlib import Path; " + changes[defect]
    if not during_child:
        subprocess.run([sys.executable, "-I", "-B", "-c", code], env=ENV, check=True)
    with pytest.raises(Q.QualificationFailed):
        recorder.run(
            "actual-drift-control", [sys.executable, "-I", "-B", "-c", code if during_child else "pass"], tmp_path
        )
    if during_child:
        assert report["commands"][0]["returncode"] == 0
        assert report["commands"][0]["status"] == "inputs_changed"
    else:
        assert report["commands"] == []


@pytest.mark.parametrize(
    "defect",
    ("unknown", "duplicate", "relative", "short-commit", "wrong-commit", "dirty", "staged-deletion", "wrong-suite"),
)
def test_source_capture_rejects_invalid_or_incomplete_original_bytes(tmp_path, defect):
    root, commit = _source(tmp_path)
    selection, suite = "m2m=" + str(root) + "@" + commit, SUITE
    if defect == "unknown":
        selection = "provider=" + str(root) + "@" + commit
    elif defect == "relative":
        selection = "m2m=relative@" + commit
    elif defect == "short-commit":
        selection = "m2m=" + str(root) + "@" + commit[:12]
    elif defect == "wrong-commit":
        selection = "m2m=" + str(root) + "@" + "0" * 40
    elif defect == "dirty":
        (root / "m2m/other.py").write_text("changed")
    elif defect == "staged-deletion":
        subprocess.run(["git", "-C", root, "rm", "-q", "m2m/other.py"], env=ENV, check=True)
    elif defect == "wrong-suite":
        suite = "phase1"
    with pytest.raises(Q.QualificationFailed):
        Q.capture_native_sources(suite, (selection, selection) if defect == "duplicate" else (selection,))


def test_normal_cli_forwards_only_explicit_native_inputs(tmp_path, monkeypatch):
    from merlin.common import paths

    captured = {}
    monkeypatch.setattr(paths, "build_dir", lambda: tmp_path)
    monkeypatch.setattr(Q, "resolve_ref", lambda *_: "a" * 40)
    monkeypatch.setattr(Q, "qualify", lambda *args, **kwargs: captured.update(kwargs) or False)
    assert (
        Q.main(
            [
                "--ref",
                "HEAD",
                "--suite",
                SUITE,
                "--label",
                "owned-cli-control",
                "--native-tool",
                "compiler-python=/selected/python",
                "--native-source",
                "m2m=/selected/source@" + "b" * 40,
            ]
        )
        == 1
    )
    assert captured["native_tools"] == ["compiler-python=/selected/python"]
    assert captured["native_sources"] == ["m2m=/selected/source@" + "b" * 40]


def test_actual_older_source_archive_does_not_require_new_qualification_tooling(tmp_path, monkeypatch):
    root = repo_root()
    helper = "build_tools/scripts/installed_native_inputs.py"
    addition = (
        subprocess.check_output(
            ["git", "-C", root, "log", "--diff-filter=A", "--format=%H", "-1", "HEAD", "--", helper], env=ENV
        )
        .decode()
        .strip()
    )
    old_ref = (
        subprocess.check_output(["git", "-C", root, "rev-parse", addition + "^" if addition else "HEAD"], env=ENV)
        .decode()
        .strip()
    )
    assert subprocess.check_output(["git", "-C", root, "ls-tree", "--name-only", old_ref, "--", helper], env=ENV) == b""
    original_run = Q.Recorder.run

    def bounded_run(self, label, argv, cwd, **kwargs):
        if label not in ("resource-manifest", "source-archive"):
            # Stop before dependency/build work; this is an archive control only.
            raise Q.QualificationFailed("owned archive control stops before wheel construction")
        return original_run(self, label, argv, cwd, **kwargs)

    monkeypatch.setattr(Q.Recorder, "run", bounded_run)
    assert not Q.qualify(root, tmp_path, old_ref, "phase1", 15)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed" and "owned archive control" in report["error"]
    assert report["tool_sources"][helper] == Q.digest(root / helper)
    assert [(row["step"], row["returncode"]) for row in report["commands"]] == [
        ("resource-manifest", 0),
        ("source-archive", 0),
    ]
    with tarfile.open(tmp_path / "source.tar") as archive:
        assert "pyproject.toml" in archive.getnames() and helper not in archive.getnames()
