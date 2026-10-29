"""Freeze the additive ABI/full-after-effect cut over immutable qualification v1."""

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
from merlin.verify import lean_graph_regions_declared_bound as bound
from merlin.verify.lean_graph_regions_declared_check import save
from merlin.verify.lean_graph_regions_declared_qualify import command, seal, source_patch


def sources():
    root = repo_root()
    return [
        root / path
        for path in (
            "src/merlin/verify/lean/Merlin/GraphRegionsDeclaredBindings.lean",
            "src/merlin/verify/lean/Merlin/GraphRegionsDeclaredEffects.lean",
            "src/merlin/verify/lean_graph_regions_declared_bound.py",
            "src/merlin/verify/lean_graph_regions_declared_bound_qualify.py",
            "merlin/tests/infra/test_lean_graph_regions_declared_bound.py",
        )
    ]


def run(name):
    root = repo_root()
    parent = artifacts_dir() / "verification/graph-regions-declared"
    destination = parent / name
    destination.mkdir(parents=True, exist_ok=False)
    fixed = bound.fixed_v1()
    roster = {}
    for path in sources():
        relative = path.relative_to(root)
        target = destination / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        roster[str(relative)] = dict(original=base.file_ref(path), snapshot=base.file_ref(target))
    profile = dict(
        fixed_v1=fixed,
        sources=roster,
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        relation="common-total-primitive, per-instruction contracts and complete observer preservation",
        native_fidelity_proved=False,
        independent_review="pending",
        ordinary_compiler_gating=False,
    )
    save(destination / "PROFILE.json", profile)
    status = dict(
        engineering_qualified=False,
        phase="collection",
        independently_reviewed=False,
        native_execution_run=False,
        numerical_execution_run=False,
        full_model_verified=False,
    )
    environment = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "MERLIN_GRAPH_DECLARED_BOUND_RUN": name + "-originals"}
    test = "merlin/tests/infra/test_lean_graph_regions_declared_bound.py"
    pytest = [sys.executable, "-B", "-m", "pytest", "-q", test]
    try:
        if command(pytest + ["--collect-only"], destination / "collection", environment) != 0:
            raise ValueError("new fixed roster unavailable")
        nodes = [
            line
            for line in (destination / "collection.stdout").read_text().splitlines()
            if line.startswith(test + "::")
        ]
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("empty or duplicate new roster")
        save(destination / "ROSTER.json", nodes)
        for row in roster.values():
            base.fresh_ref(row["original"])
        status["phase"] = "originals"
        if command(pytest + ["--tb=short"], destination / "originals", environment) != 0:
            raise ValueError("new original qualification failed; first failure retained")
        if (destination / "originals.stdout").read_text().splitlines()[-1].split()[:2] != [str(len(nodes)), "passed"]:
            raise ValueError("not every fixed case passed; no skips/unavailable promotion")
        originals = parent / (name + "-originals")
        receipts = [originals / ("verified-" + label) / "EVIDENCE.json" for label in ("i8", "i16shared")]
        for path in receipts:
            bound.validate_bound(json.loads(path.read_text()))
        status["phase"] = "one fresh replay batch"
        args = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--replay",
            str(destination / "fresh-replay"),
            "--receipts",
            *map(str, receipts),
        ]
        if command(args, destination / "fresh-process", environment) != 0:
            raise ValueError("fresh replay failed; preserve original and fresh evidence")
        for row in roster.values():
            if base.fresh_ref(row["original"]).read_bytes() != base.fresh_ref(row["snapshot"]).read_bytes():
                raise ValueError("source changed after frozen run")
        if bound.fixed_v1() != fixed:
            raise ValueError("fixed v1 dependency changed")
        first = json.loads(receipts[0].read_text())
        base_receipt = json.loads(base.fresh_ref(first["base"]).read_text())
        build = Path(base_receipt["identity"]["path"]).parent
        identity = json.loads(base.fresh_ref(base_receipt["identity"]).read_text())
        external = {}
        for label, directory in (("originals", originals), ("build", build)):
            external[label] = {
                str(path.relative_to(directory)): base.file_ref(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
        save(destination / "EXTERNAL-RETENTION.json", external)
        closure = {}
        for row in identity["actual_import_closure"].values():
            if row["project"]:
                path = base.fresh_ref(row["source"])
                relative = path.relative_to(root)
                target = destination / "project-source-closure" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                closure[str(relative)] = dict(original=row["source"], snapshot=base.file_ref(target))
        save(destination / "PROJECT-SOURCE-CLOSURE.json", closure)
        source_patch(sources(), destination / "owned-additive-source.patch", profile["head"])
        status.update(
            engineering_qualified=True,
            phase="complete",
            fixed_cases=len(nodes),
            fresh_process_batches=1,
            receipts=[base.file_ref(path) for path in receipts],
            primitive_fidelity_proved=False,
        )
    except Exception as error:
        status.update(failure=repr(error), phase_failed=status["phase"])
        source_patch(sources(), destination / "owned-additive-partial.patch", profile["head"])
    save(destination / "STATUS.json", status)
    seal(destination, status)
    print(json.dumps(dict(destination=str(destination), status=status), indent=2))
    return 0 if status["engineering_qualified"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--receipts", nargs="*", type=Path)
    args = parser.parse_args()
    if args.replay is not None:
        args.replay.mkdir(parents=True, exist_ok=False)
        if not args.receipts:
            raise ValueError("empty replay batch")
        for index, receipt in enumerate(args.receipts):
            bound.replay_bound(json.loads(receipt.read_text()), args.replay / str(index))
        return 0
    if args.name is None:
        raise ValueError("explicit fresh run name required")
    return run(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
