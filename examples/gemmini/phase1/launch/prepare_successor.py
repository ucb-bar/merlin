"""Bind already prepared inputs to a fresh launch; never derive, certify or start it."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml
from merlin_experiments.phase1 import launch_config as L

from merlin.common.digest import sha256_file


def bind(
    template: Path, *, task_root: Path, prep: Path, run_id: str, output: Path, overrides: dict[str, str] | None = None
) -> dict:
    """Pin selected bytes, preserving the native session's independent SDK admission.

    This is operator configuration, not a source, corpus, runtime or compiler
    qualification. A new corpus must already have its ordinary reviewed seal.
    """
    selected = L.load(template)
    launch = L.build(selected, task_root=task_root, prep=prep, run_id=run_id, overrides=overrides)
    document = copy.deepcopy(selected.document)
    if (
        document.get("validation_scope") != "capsules"
        or document["origin"]["resume"]
        or document["origin"]["seed_submission"] is not None
    ):
        raise ValueError("successor preparation requires fresh unseeded capsule scope")
    total = document["budgets"]["supervisor_backstop_s"]
    author = document["budgets"]["authoring_s"]
    reserve = int(launch.variables["SUPERVISOR_FINAL_RESERVE_S"])
    if author <= 0 or reserve <= 0 or author + reserve != total:
        raise ValueError("author and final reserve must partition the capsule clock")
    repo = launch.path(document["inputs"]["repo"]).resolve()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True)
    if dirty.strip():
        raise ValueError("commit the selected producer sources before binding a launch")
    output = output.absolute()
    protected = [
        repo,
        template.resolve(),
        *(
            launch.path(launch.variables[name]).resolve()
            for name in ("RELEASE", "NUMERICS_BUNDLE", "CHIPYARD", "SPIKE_RUNTIME")
            if name in launch.variables
        ),
    ]
    if (
        output.exists()
        or output.is_symlink()
        or any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in protected)
    ):
        raise ValueError("choose a fresh configuration output outside frozen inputs")
    # Preserve every resolved default/override: the output config lives elsewhere.
    document["variables"].update(
        {name: value for name, value in launch.variables.items() if name not in (*L.BUILTINS, "HOME")}
    )
    document["variables"]["SOURCE_REVISION"] = source
    for section in ("runtime_pins", "target_selection"):
        for row in document.get(section, {}).values():
            row["path"] = str(launch.path(row["path"]))
            row["sha256"] = sha256_file(Path(row["path"]))
    for name in ("oracle_timing",):
        row = document["inputs"][name]
        row["path"] = str(launch.path(row["path"]))
        row["sha256"] = sha256_file(Path(row["path"]))
    for name in ("descriptor", "bundle_manifest", "corpus_seal"):
        path = launch.path(document["inputs"][name])
        document["runtime_pins"]["selected_" + name] = {"path": str(path), "sha256": sha256_file(path)}
    selection = document["instruction_selection"]
    for row in (selection, selection["approval"]):
        row["path"] = str(launch.path(row["path"]))
        row["sha256"] = sha256_file(Path(row["path"]))
    errata = document["errata"]
    errata_path = selected.path.parent / errata["source"]
    errata["source"] = str(errata_path.resolve())
    errata["sha256"] = sha256_file(errata_path)
    release = copy.deepcopy(selected.release)
    if release is not None:
        for name in release["variables"]:
            release["variables"][name] = launch.variables[name]
        for row in release["prepare"].values():
            if isinstance(row, dict) and set(row) == {"path", "sha256"}:
                row["path"] = str(launch.path(row["path"]))
                row["sha256"] = sha256_file(Path(row["path"]))
        observed_seal = json.loads(launch.path(document["inputs"]["corpus_seal"]).read_bytes())
        review = observed_seal.get("review") or {}
        if not all(isinstance(review.get(key), str) and review[key].strip() for key in ("reviewed_by", "note")):
            raise ValueError("selected successor release has no retained operator review")
        release["seal"].update(reviewed_by=review["reviewed_by"], review_note=review["note"])
        document["release_config"] = str(output / "release.yaml")
    # No request to inspect private tensors, execute a compiler or grade a seed.
    manifest = launch.path(document["inputs"]["bundle_manifest"])
    bundle = yaml.safe_load(manifest.read_text())
    sdk = bundle.get("compiler_tool_profile") if isinstance(bundle, dict) else None
    if not isinstance(sdk, dict) or sdk.get("common_substrate") != "merlin.common_compiler_substrate.v6":
        raise ValueError("selected successor corpus requires the explicit V6 compiler tool profile")
    candidate = launch.path(launch.variables["RUN"])
    if candidate.exists():
        raise ValueError("successor run ID already exists")
    output.mkdir(parents=True)
    if release is not None:
        (output / "release.yaml").write_text(yaml.safe_dump(release, sort_keys=False))
    config = output / "config.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False))
    bound = L.build(L.load(config), task_root=task_root, prep=prep, run_id=run_id)
    L.render(bound, output / "rendered")
    record = {
        "schema": "merlin.phase1.successor_preparation.v1",
        "status": "configuration_bound",
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "source_revision": source,
        "compiler_tool_profile": sdk,
        "sdk_admission": "pending_native_session",
        "capsule_budget_s": total,
        "author_budget_s": author,
        "final_reserve_s": reserve,
        "candidate_required_for_preparation": False,
        "candidate_qualified": False,
        "controller_started": False,
        "phase1_complete": False,
        "argv": list(bound.argv),
        "manager_cpus": list(bound.manager_cpus),
    }
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("el4-successor.yaml"))
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--prep", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--var", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = bind(
            args.template,
            task_root=args.task_root,
            prep=args.prep,
            run_id=args.run_id,
            output=args.output,
            overrides=L._overrides(args.var),
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"successor preparation refused: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
