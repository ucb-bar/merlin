"""Freeze one declared-call engineering qualification and one fresh replay.

Failures are retained and sealed as unqualified. This command does not approve
contracts, repair earlier contexts, run numerics, or publish source changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify import lean_graph_regions as base
from merlin.verify.lean_graph_regions_declared_check import save
from merlin.verify.lean_graph_regions_declared_verify import fresh_identity, replay, validate


def sources():
    root = repo_root()
    return (
        sorted((root / "src/merlin/verify/lean/Merlin").glob("GraphRegionsDeclared*.lean"))
        + sorted((root / "src/merlin/verify").glob("lean_graph_regions_declared*.py"))
        + [root / "merlin/tests/infra/test_lean_graph_regions_declared.py"]
    )


def command(arguments, destination, environment=None, timeout=600):
    with destination.with_suffix(".stdout").open("w") as stdout, destination.with_suffix(".stderr").open("w") as stderr:
        result = subprocess.run(
            arguments, cwd=repo_root(), env=environment, stdout=stdout, stderr=stderr, timeout=timeout
        )
    save(destination.with_suffix(".json"), dict(command=arguments, returncode=result.returncode))
    return result.returncode


def source_patch(files, destination, head):
    """Exact owned diff from current HEAD; never stage, commit or apply it."""
    root = repo_root()
    parts = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        tracked = subprocess.run(["git", "cat-file", "-e", head + ":" + relative], cwd=root, capture_output=True)
        args = (
            ["git", "diff", "--binary", head, "--", relative]
            if tracked.returncode == 0
            else ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative]
        )
        result = subprocess.run(args, cwd=root, capture_output=True)
        if result.returncode not in (0, 1):
            raise ValueError("owned source-only diff failed")
        parts.append(result.stdout)
    destination.write_bytes(b"".join(parts))


def seal(destination, record):
    refs = {
        str(path.relative_to(destination)): base.file_ref(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "SEAL.json"
    }
    save(destination / "SEAL.json", dict(status=record, files=refs, retention_policy="all"))


def run(name):
    root = repo_root()
    artifact_root = artifacts_dir() / "verification/graph-regions-declared"
    destination = artifact_root / name
    destination.mkdir(parents=True, exist_ok=False)
    files = sources()
    roster = {}
    for path in files:
        relative = path.relative_to(root)
        target = destination / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        roster[str(relative)] = dict(original=base.file_ref(path), snapshot=base.file_ref(target))
    profile = dict(
        schema="merlin.graph-regions.declared-qualification.v1",
        source_roster=roster,
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        native_execution_run=False,
        numerical_execution_run=False,
        independently_reviewed=False,
        source_operational_bridge="UNQUALIFIED; not imported",
        retry_policy="no retry-to-promote",
    )
    save(destination / "PROFILE.json", profile)
    environment = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "MERLIN_GRAPH_DECLARED_RUN": name + "-originals"}
    pytest = [sys.executable, "-B", "-m", "pytest", "-q", "merlin/tests/infra/test_lean_graph_regions_declared.py"]
    status = dict(
        engineering_qualified=False,
        phase="collection",
        full_model_verified=False,
        primitive_fidelity_proved=False,
        independently_reviewed=False,
    )
    try:
        if command(pytest + ["--collect-only"], destination / "collection", environment) != 0:
            raise ValueError("fixed roster collection failed")
        nodes = [
            line
            for line in (destination / "collection.stdout").read_text().splitlines()
            if line.startswith("merlin/tests/infra/test_lean_graph_regions_declared.py::")
        ]
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("missing or duplicate fixed test roster")
        save(destination / "ROSTER.json", nodes)
        for row in roster.values():
            base.fresh_ref(row["original"])
        status["phase"] = "originals"
        if command(pytest + ["--tb=short"], destination / "originals", environment) != 0:
            raise ValueError("original qualification failed; preserve first failure")
        summary = (destination / "originals.stdout").read_text().splitlines()[-1].split()
        if summary[:2] != [str(len(nodes)), "passed"]:
            raise ValueError("not every fixed case passed; skips/unavailable are not passes")
        original_root = artifact_root / (name + "-originals")
        receipts = [original_root / ("verified-" + label) / "EVIDENCE.json" for label in ("i8", "i16shared")]
        for path in receipts:
            validate(json.loads(path.read_text()))
        status["phase"] = "one fresh replay batch"
        replay_root = destination / "fresh-replay"
        worker = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--replay-batch",
            str(replay_root),
            "--receipts",
            *map(str, receipts),
        ]
        if command(worker, destination / "fresh-process", environment) != 0:
            raise ValueError("fresh replay failed; preserve original and fresh evidence")
        for row in roster.values():
            if base.fresh_ref(row["original"]).read_bytes() != base.fresh_ref(row["snapshot"]).read_bytes():
                raise ValueError("source roster drifted after qualification")
        for path in receipts:
            receipt = json.loads(path.read_text())
            validate(receipt)
            fresh_identity(json.loads(base.fresh_ref(receipt["identity"]).read_text()))
        original_receipt = json.loads(receipts[0].read_text())
        build_root = Path(original_receipt["identity"]["path"]).parent
        external = {}
        for label, directory in (("originals", original_root), ("build", build_root)):
            external[label] = {
                str(path.relative_to(directory)): base.file_ref(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
        save(destination / "EXTERNAL-RETENTION.json", external)
        identity = json.loads(Path(original_receipt["identity"]["path"]).read_text())
        project_sources = {}
        for row in identity["actual_import_closure"].values():
            if row["project"]:
                path = base.fresh_ref(row["source"])
                relative = path.relative_to(root)
                target = destination / "project-source-closure" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                project_sources[str(relative)] = dict(original=row["source"], snapshot=base.file_ref(target))
        save(destination / "PROJECT-SOURCE-CLOSURE.json", project_sources)
        source_patch(files, destination / "owned-source.patch", profile["head"])
        status.update(
            engineering_qualified=True,
            phase="complete",
            fixed_cases=len(nodes),
            original_receipts=[base.file_ref(path) for path in receipts],
            fresh_process_batches=1,
            graph_contract_review="pending",
            ordinary_compiler_gating=False,
        )
    except Exception as error:
        status.update(failure=repr(error), phase_failed=status["phase"])
        source_patch(files, destination / "owned-source-partial.patch", profile["head"])
    save(destination / "STATUS.json", status)
    seal(destination, status)
    print(json.dumps(dict(destination=str(destination), status=status), indent=2))
    return 0 if status["engineering_qualified"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--replay-batch", type=Path)
    parser.add_argument("--receipts", nargs="*", type=Path)
    args = parser.parse_args()
    if args.replay_batch is not None:
        args.replay_batch.mkdir(parents=True, exist_ok=False)
        for index, path in enumerate(args.receipts or []):
            replay(json.loads(path.read_text()), args.replay_batch / str(index))
        if not args.receipts:
            raise ValueError("empty replay batch")
        return 0
    if args.name is None:
        raise ValueError("explicit fresh qualification name required")
    return run(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
