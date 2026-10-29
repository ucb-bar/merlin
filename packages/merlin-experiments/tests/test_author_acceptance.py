"""The pre-launch author acceptance suite stays green on the committed tree.

``build_tools/scripts/preflight_author_workflows.py`` is run by an operator on a frozen export before a
launch. This runs its fast checks on a ``git archive`` of HEAD -- the same shape as an export -- so a
change that breaks an author workflow (a withheld import in the public tree, an empty extent reader, a
skill script that no longer starts) fails here rather than mid-run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest
from merlin_experiments.phase1 import author_acceptance as AA

from merlin.common.paths import repo_root

#: The parts of a framework export the fast checks read: the harness, the compiler sources, the test
#: fixtures and the schema data.
_EXPORTED = ("src", "packages", "merlin")


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    root = tmp_path_factory.mktemp("export")
    archive = subprocess.run(["git", "-C", str(repo_root()), "archive", "HEAD", *_EXPORTED],
                             capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive.stdout, check=True)
    return root


def test_fast_author_workflows_pass_on_an_archive_of_head(export, tmp_path):
    report = AA.run(export, None, work_root=tmp_path / "reports", only=set(AA.FAST))
    rows = {row["name"]: row for row in report["checks"]}
    assert set(AA.FAST) | {"staging"} == set(rows)
    failed = {name: row["cause"] for name, row in rows.items() if row["status"] == AA.FAIL}
    assert not failed, failed
    assert report["status"] == AA.PASS
    assert json.loads(open(report["report_path"]).read())["status"] == AA.PASS
    # The author path is the staged tree and the workspace, in that order, and nothing else.
    assert report["author_sys_path"][0].endswith("submission/compiler_library")
    assert report["author_sys_path"][-1] == report["staged"]["workspace"]


def test_the_script_prints_a_table_and_fails_on_any_failure(export, tmp_path):
    script = repo_root() / "build_tools" / "scripts" / "preflight_author_workflows.py"
    proc = subprocess.run([sys.executable, "-B", str(script), "--framework", str(export), "--work-root",
                           str(tmp_path), "--only", "skills,tuning-knobs"], capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "skills" in proc.stdout and "tuning-knobs" in proc.stdout and "report:" in proc.stdout
    broken = tmp_path / "broken"
    broken.mkdir()
    proc = subprocess.run([sys.executable, "-B", str(script), "--framework", str(broken), "--work-root",
                           str(tmp_path), "--only", "skills"], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1 and "staging" in proc.stdout and "FAIL" in proc.stdout


def test_a_recorded_sandbox_argument_file_is_audited_for_mounted_withheld_modules(tmp_path):
    args = tmp_path / "bwrap.args"
    args.write_bytes(b"\0".join([b"--ro-bind", b"/src/sdk/merlin/kernels/cca.py", b"/sdk/merlin/kernels/cca.py",
                                 b"--ro-bind", b"/src/sdk/merlin/common/x.py", b"/sdk/merlin/common/x.py",
                                 b"--ro-bind", b"/dev/null", b"/sdk/merlin/mining/y.py"]) + b"\0")
    check = AA.audit_sandbox_args(args, ("merlin.kernels", "merlin.mining"))
    assert check.status == AA.FAIL and check.detail["withheld_mounted"] == [
        "merlin.kernels.cca at /sdk/merlin/kernels/cca.py"]
    assert AA.audit_sandbox_args(args, ("merlin.mining",)).status == AA.PASS
