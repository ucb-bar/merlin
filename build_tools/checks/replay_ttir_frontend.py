"""Replay the frozen TTIR integration tests against the current source layout.

The permanent suite remains compact. This opt-in replay extracts exact reviewed
Git blobs and applies a hash-checked layout adapter under ignored out/build.
Native compiler, cross compiler and simulator selections come from the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    tools = repo / "build_tools/checks"
    selection = json.loads((tools / "ttir_layout_replay.json").read_text())
    patch = tools / "ttir_layout_replay.patch"
    if digest(patch.read_bytes()) != selection["patch_sha256"]:
        raise ValueError("TTIR replay adapter digest differs from its reviewed selection")
    parent = repo / "out/build/ttir-replay"
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="layout-", dir=parent))
    for member in selection["files"]:
        relative = Path(member["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "tests":
            raise ValueError("replay members must remain within tests/")
        data = subprocess.check_output(
            ["git", "show", selection["source_commit"] + ":" + member["source"]], cwd=repo
        )
        if digest(data) != member["sha256"]:
            raise ValueError("frozen replay source digest differs: " + member["source"])
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=work, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=work, check=True)
    evidence = {
        "schema": "merlin.ttir-layout-replay.v1",
        "source_commit": selection["source_commit"],
        "patch_sha256": selection["patch_sha256"],
        "reviewed_profile_updates": selection.get("reviewed_profile_updates", []),
        "consumer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "consumer_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)),
        "consumer_source_sha256": {
            str(p.relative_to(repo)): digest(p.read_bytes())
            for p in sorted((repo / "src/merlin").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "replay_source_sha256": {
            str(p.relative_to(work)): digest(p.read_bytes())
            for p in sorted((work / "tests").rglob("*")) if p.is_file()
        },
        "python": str(args.python.absolute()),
        "status": "prepared",
    }
    report = work / "replay.json"
    report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(work, flush=True)
    if args.prepare_only:
        return 0
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repo / "src"), str(work / "tests/fixtures"))
    )
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [
        str(args.python), "-m", "pytest", str(work / "tests"),
        "-c", str(repo / "pyproject.toml"), "-q", "--tb=short",
        "--basetemp=" + str(work / "pytest"),
    ]
    with (work / "pytest.log").open("w") as log:
        done = subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
    evidence.update(status="passed" if done.returncode == 0 else "failed", returncode=done.returncode)
    report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print((work / "pytest.log").read_text()[-12000:])
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
