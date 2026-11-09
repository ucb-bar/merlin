"""A CELL run end to end: prepared from a measured loop run, launched detached, read back.

    merlin-experiment cell prepare <loop_run> --cell ID [--cells FILE] [--groups 1,2] [--form-of N ...]
                                   [--seed PKG] [--baseline-package PKG] [--held-out CAPSULE ...]
                                   [--collateral-share RESULT] [--screen-capsules NAMES] --why TEXT
    merlin-experiment cell launch <cell_run> --profile P
    merlin-experiment cell status [<cell_run> | --target T]
    merlin-experiment cell board --package-job JOB --reference-job JOB --groups 1,70 [--prepare-only]

``prepare`` composes the cell's objective config from the loop's own (:func:`.cell_prep.prepare`: the
held-out groups by form, the reference arm, the baseline and collateral measured on a VERIFIED package)
into a new phase-2 run of method ``<loop method>_cell_<id>`` -- so the loop's no-FSM naming and its
prohibited roles carry over -- and records where everything came from in ``cell.json``.  A cell is
named by its groups, its anchors' FORMS (``--form-of``), or an entry of a cells file: per-model data
in the target's ``examples/<target>/phase2/`` directory, which states the model it was read from and is
refused against any other.

THE SEED DEFAULTS TO THE CURRENT CHAMPION RECORD: the loop run's confirmed ``best`` (its OOT ``best``
tag, attributable in the store), else the target's one exported champion; its exact bytes are exported
from the OOT history and checked against the digest they were measured as, and that measurement rides
along as the seed's import evidence.  A champion is a verified package, so it is also the default
baseline.  With neither, or with several champions to choose from, ``--seed`` is required: a cell is
never seeded from whatever a workspace holds now.

``launch`` always DETACHES (:mod:`.launch`): its own session, output appended to ``launch.log`` in the
run directory, never a pipe.

``board`` confirms a cell result on the BOARD: both arms' one-group programs and the reference control
in one batch (:func:`.group_capsules_board.measure_on_board`), each count admitted only on its own
evidence.  The arms are the two whole-model jobs' own build options, and the board, functional model and
control are the package job's own machine -- nothing restated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import cells as CELLS
from .identity import package_digest, read_json, write_json_atomic

CELL_RECORD = "cell.json"
CELL_SCHEMA = "merlin.phase2.whole_model_measured.cell_run.v1"
CELLS_SCHEMA = "merlin.phase2.whole_model_measured.cells.v1"
CELL_PREP_DIR = "cell"


class CellRunError(RuntimeError):
    """The cell run cannot be prepared, launched or read as asked; the reason is stated."""


# ------------------------------------------------------------------------------------- definitions
def load_cells(path: Path) -> dict[str, Any]:
    """A cells file: ``{schema, model, cells: {id: {groups?, form_of?, screen_capsules?, why?}}}``."""
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if document.get("schema") != CELLS_SCHEMA:
        raise CellRunError(f"{path} is not a cells file ({CELLS_SCHEMA})")
    if not document.get("model") or not isinstance(document.get("cells"), Mapping):
        raise CellRunError(f"{path} names no model or no cells")
    for cell_id, body in document["cells"].items():
        if not isinstance(body, Mapping) or not (body.get("groups") or body.get("form_of")):
            raise CellRunError(f"cell {cell_id!r} in {path} names neither groups nor form_of anchors")
    return document


def _component(cell_id: str) -> str:
    if not cell_id or "/" in cell_id or cell_id in (".", "..") or not cell_id.replace("_", "").isalnum():
        raise CellRunError(f"a cell id is one identifier, not {cell_id!r}")
    return cell_id


def cell_groups(
    loop_config: Mapping[str, Any],
    *,
    target: str,
    groups: Sequence[int],
    form_of: Sequence[int],
) -> list[int]:
    """The cell's groups: the named ones, plus one per distinct shape of each anchor's form."""
    from . import cell_prep as CP
    from . import forms as FORMS

    found = {int(g) for g in groups}
    if form_of:
        capsule = loop_config["certifier"]["build_options"]["model_capsule"]
        found |= set(CP.form_cell_groups(FORMS.statement_forms(capsule, target=target), list(form_of)))
    if not found:
        raise CellRunError("name the cell's groups, its form anchors, or an entry of a cells file")
    return sorted(found)


# ------------------------------------------------------------------------------------- the seed
def _attributable(store_root: Path, digest: str) -> bool:
    from . import jobs as J

    return J.attributable(read_json(Path(store_root) / digest / J.ATTRIBUTION_FILE))


def _export(repo: Path, commit: str, destination: Path, digest: str) -> Path:
    from merlin.common import oot_repo

    oot_repo.export(repo, commit, destination)
    if package_digest(destination) != digest:
        raise CellRunError(f"{repo}@{commit[:12]} exports bytes other than the {digest[:12]} it was measured as")
    return destination


def champion_seed(loop_run: Path, *, target: str, destination: Path) -> dict[str, Any] | None:
    """The current champion record's exact bytes, exported to ``destination``; None when there is none.

    The loop's own confirmed ``best`` first (attributable in its store), else the target's single
    exported champion.  Several champions and no loop best is ambiguous and refused."""
    from merlin.common import oot_repo

    loop_run = Path(loop_run)
    repo = loop_run / "oot"
    stores = (read_json(loop_run / "resumed_seed.json") or {}).get("store_roots") or {}
    if repo.is_dir() and oot_repo.BEST_TAG in oot_repo.tags(repo):
        commit = oot_repo.resolve(repo, oot_repo.BEST_TAG)
        digest = oot_repo.tree_digest(repo, commit)
        screen = Path(str(stores.get("screen") or ""))
        if stores.get("screen") and not _attributable(screen, digest):
            raise CellRunError(f"the loop's best {digest[:12]} is not attributable to an authored round")
        evidence = screen / digest / "result.json" if stores.get("screen") else None
        return {
            "kind": "loop_best",
            "package": str(_export(repo, commit, destination, digest)),
            "package_sha256": digest,
            "commit": commit,
            "evidence": str(evidence) if evidence is not None and evidence.is_file() else None,
        }
    from merlin.common.paths import out_dir
    from merlin.targetgen.champions import champions_root, read_champion

    root = champions_root(target)
    found = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.is_dir() else []
    if not found:
        return None
    if len(found) > 1:
        raise CellRunError(f"{len(found)} exported champions of {target!r} ({[p.name for p in found]}); name --seed")
    provenance = read_champion(found[0])["provenance"]
    phase2 = provenance.get("phase2") or {}
    run = Path(str(phase2.get("run") or ""))
    run = run if run.is_absolute() else out_dir() / run
    digest, commit = str(provenance.get("package_digest") or ""), str(phase2.get("best_commit") or "")
    if not (run / "oot").is_dir() or not digest or not commit:
        raise CellRunError(f"champion {found[0].name} names no phase-2 run history to export its bytes from")
    return {
        "kind": "champion",
        "champion": str(found[0]),
        "package": str(_export(run / "oot", commit, destination, digest)),
        "package_sha256": digest,
        "commit": commit,
        "evidence": None,
    }


# ------------------------------------------------------------------------------------- prepare
def _phase_run(*, target: str, method: str) -> Path:
    from merlin.common.artifacts import start_phase_run

    return Path(start_phase_run(target=target, phase=2, method=method).run_dir)


def _compose(
    loop_run: Path,
    loop: Mapping[str, Any],
    run_dir: Path,
    *,
    target: str,
    method: str,
    cell_id: str,
    groups: Sequence[int],
    why: str,
    seed: Path | None,
    baseline_package: Path | None,
    held_out: Sequence[str],
    collateral_share: Path | None,
    collateral_tolerance: float,
    screen: str | None,
    max_cycles: int | None,
    functional_model: str | None,
    roles: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """The seed, the cell's composed config and the prepared run, in ``run_dir``."""
    from merlin.common import oot_repo

    from . import cell_prep as CP
    from . import cli as CLI
    from . import forms as FORMS
    from . import runs as RUNS

    prep = run_dir / CELL_PREP_DIR
    if seed is not None:
        source = {"kind": "explicit", "package": str(Path(seed).resolve()), "package_sha256": package_digest(seed)}
    else:
        source = champion_seed(loop_run, target=target, destination=prep / "seed")
        if source is None:
            raise CellRunError(f"neither the loop nor {target!r} has a champion record to seed from; name --seed")
    if baseline_package is None and source["kind"] != "explicit":
        baseline_package = Path(source["package"])  # a champion is a verified package
    prepared = CP.prepare(
        loop,
        target=target,
        cell_id=cell_id,
        groups=list(groups),
        held_out_capsules=list(held_out),
        collateral_share=FORMS.load_counts(collateral_share) if collateral_share else None,
        baseline_package=baseline_package,
        collateral_tolerance=collateral_tolerance,
        out=prep,
        max_cycles=max_cycles,
        functional_model=CLI._functional_model(loop, functional_model),
        screen_capsules=screen,
    )
    run = RUNS.prepare(
        target=target,
        method=method,
        objective_config=prepared["config"],
        seed=Path(source["package"]),
        prohibited_roles=list(roles),
        why=why,
        run_factory=lambda **_: run_dir,
        oot=oot_repo,
        import_evidence=Path(source["evidence"]) if source.get("evidence") else None,
    )
    return source, {**prepared, "baseline_package": baseline_package}, run


def prepare_cell_run(
    loop_run: Path,
    *,
    cell_id: str,
    why: str,
    groups: Sequence[int] = (),
    form_of: Sequence[int] = (),
    cells_file: Path | None = None,
    seed: Path | None = None,
    baseline_package: Path | None = None,
    held_out: Sequence[str] = (),
    collateral_share: Path | None = None,
    collateral_tolerance: float = 0.01,
    screen_capsules: str | None = None,
    max_cycles: int | None = None,
    functional_model: str | None = None,
    run_factory: Any = None,
) -> dict[str, Any]:
    """Prepare one cell run from the measured loop run ``loop_run`` (see the module docstring)."""
    from . import cli as CLI
    from . import runs as RUNS

    if not str(why or "").strip():
        raise CellRunError("a cell run states why it exists")
    cell_id = _component(cell_id)
    loop_run = CLI.resolve_run(loop_run)
    record = read_json(loop_run / "run.json") or {}
    loop = json.loads((loop_run / RUNS.CONFIG_NAME).read_text(encoding="utf-8"))
    target = str(record["target"])
    definition: dict[str, Any] = {}
    if cells_file is not None:
        cells = load_cells(cells_file)
        capsule = Path(str(loop["certifier"]["build_options"]["model_capsule"]))
        if capsule.name != str(cells["model"]):
            raise CellRunError(f"{cells_file} is for {cells['model']!r}; this loop measures {capsule.name!r}")
        if cell_id not in cells["cells"]:
            raise CellRunError(f"{cells_file} declares no cell {cell_id!r} ({sorted(cells['cells'])})")
        definition = dict(cells["cells"][cell_id])
    chosen = cell_groups(
        loop,
        target=target,
        groups=[*groups, *(definition.get("groups") or ())],
        form_of=[*form_of, *(definition.get("form_of") or ())],
    )
    screen = screen_capsules or definition.get("screen_capsules")
    if isinstance(screen, list):
        screen = ",".join(str(name) for name in screen)
    method = f"{record['method']}_cell_{cell_id}"
    run_dir = Path((run_factory or _phase_run)(target=target, method=method))
    try:
        source, prepared, run = _compose(
            loop_run,
            loop,
            run_dir,
            target=target,
            method=method,
            cell_id=cell_id,
            groups=chosen,
            why=why,
            seed=seed,
            baseline_package=baseline_package,
            held_out=held_out,
            collateral_share=collateral_share,
            collateral_tolerance=collateral_tolerance,
            screen=screen,
            max_cycles=max_cycles,
            functional_model=functional_model,
            roles=list(record.get("prohibited_instruction_roles") or ()),
        )
    except (CellRunError, CELLS.CellError, RUNS.RunError) as exc:
        # Never deleted (another session may already read it): named, so the operator can set it aside.
        raise CellRunError(f"{exc} (the unfinished cell run is left at {run_dir})") from exc
    baseline_package = prepared["baseline_package"]
    document = {
        "schema": CELL_SCHEMA,
        "cell_id": cell_id,
        "groups": chosen,
        "loop_run": str(loop_run),
        "cells_file": str(Path(cells_file).resolve()) if cells_file is not None else None,
        "definition": definition or None,
        "seed": source,
        "baseline_package": str(baseline_package) if baseline_package is not None else None,
        "held_out": prepared.get("held_out"),
        "screen_capsules": screen,
        "config_sha256": run.config_sha256,
        "why": str(why).strip(),
    }
    write_json_atomic(run_dir / CELL_RECORD, document)
    return {"run_dir": str(run_dir), **document}


# ------------------------------------------------------------------------------------- status
def cell_runs(target: str) -> list[Path]:
    from merlin.common.paths import phase_runs_root

    return sorted(p.parent for p in phase_runs_root(target, 2).glob(f"*/{CELL_RECORD}"))


def cell_status(run_dir: Path, *, objective: Any = None) -> dict[str, Any]:
    """The run's status (:func:`.progress.run_status`) with its cell: what it covers, its bar and its
    baseline, read from the run's own records."""
    from . import progress as PROGRESS

    run_dir = Path(run_dir)
    record = read_json(run_dir / CELL_RECORD)
    if record is None:
        raise CellRunError(f"{run_dir} is not a cell run (no {CELL_RECORD})")
    document = PROGRESS.run_status(run_dir, objective=objective)
    config = read_json(run_dir / "whole_model_objective_config.json") or {}
    cell = ((config.get("screen") or {}).get("machine") or {}).get("cell") or {}
    reference = read_json(Path(str((config.get("screen") or {}).get("reference") or ""))) or {}
    document["cell"] = {
        "id": record.get("cell_id"),
        "groups": record.get("groups"),
        "loop_run": record.get("loop_run"),
        "seed": {k: (record.get("seed") or {}).get(k) for k in ("kind", "package_sha256", "champion")},
        "reference_cycles": reference.get("objective_cycles"),
        "reference_status": reference.get("timing_status"),
        "baseline_groups": len((cell.get(CELLS.BASELINE) or {}).get("groups") or ()),
        "collateral_groups": len((cell.get(CELLS.COLLATERAL) or {}).get("groups") or ()),
        "held_out": [row.get("label") for row in cell.get("held_out") or ()],
    }
    return document


def format_cell(document: Mapping[str, Any]) -> str:
    from . import progress as PROGRESS

    cell = document["cell"]
    head = (
        f"cell {cell['id']} groups {cell['groups']}: reference {cell['reference_cycles']} "
        f"({cell['reference_status']}); seed {cell['seed']['kind']} {str(cell['seed']['package_sha256'])[:12]}; "
        f"{cell['collateral_groups']} collateral, {cell['baseline_groups']} baseline group(s)"
    )
    return head + "\n" + PROGRESS.format_status(document)


# ------------------------------------------------------------------------------------- the board
def board_validate(
    package_job: Path,
    reference_job: Path,
    groups: Sequence[int],
    *,
    package_dir: Path | None = None,
    out: Path | None = None,
    submit: bool = True,
    timeout_s: float = 1800,
) -> dict[str, Any]:
    """Both arms of ``groups`` on the package job's board in one batch with its reference control."""
    from merlin.common.artifacts import new_product

    from . import group_capsules as GC
    from . import group_capsules_board as GCB

    job = read_json(Path(package_job))
    if not job:
        raise CellRunError(f"{package_job} is not a job record")
    machine = dict(job.get("machine") or {})
    if not machine.get("timing") or not machine.get("local"):
        raise CellRunError(f"{package_job}'s machine has no board half and functional-model half to run both on")
    target = str(job["target"])
    arms = GC.arms_from_jobs(package_job, reference_job)
    package_dir = Path(package_dir) if package_dir is not None else Path(package_job).parent / "package"
    if out is None:
        product = new_product(
            "perf-group-board", version=1, target=target, sources=[str(package_job), str(reference_job)]
        )
        out = product.path
        product.add_artifact("board_rows.json")
        product.write_manifest()
    return GCB.measure_on_board(
        arms,
        [int(g) for g in groups],
        package_dir=package_dir,
        model_capsule=(job.get("build_options") or {})["model_capsule"],
        target=target,
        out=Path(out),
        machine=GCB.board_machine_spec(machine["timing"]),
        local=machine["local"],
        control=machine.get("control"),
        timeout_s=timeout_s,
        submit=submit,
    )


# ------------------------------------------------------------------------------------- command line
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merlin experiment cell", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("prepare", help="prepare a cell run from a measured loop run")
    child.add_argument("loop_run", type=Path, help="the measured loop run the cell is composed from")
    child.add_argument("--cell", required=True, dest="cell_id", help="the cell's id (one identifier)")
    child.add_argument("--why", required=True)
    child.add_argument("--cells", type=Path, help="a cells file naming the cell's groups and screen")
    child.add_argument("--groups", default="", help="the cell's groups, comma-separated")
    child.add_argument("--form-of", type=int, action="append", default=[], help="a group whose FORM the cell is")
    child.add_argument("--seed", type=Path, help="seed package (default: the current champion record)")
    child.add_argument("--baseline-package", type=Path, help="the VERIFIED package baselines are measured on")
    child.add_argument("--held-out", action="append", default=[], help="a held-out model capsule directory")
    child.add_argument("--collateral-share", type=Path, help="a result picking each other form's representative")
    child.add_argument("--collateral-tolerance", type=float, default=0.01)
    child.add_argument("--screen-capsules", help="restrict the loop's capsule screen to these (comma-separated)")
    child.add_argument("--max-cycles", type=int)
    child.add_argument("--functional-model", help="a registry machine the efficiency census runs on")
    child = sub.add_parser("launch", help="start a prepared cell run detached")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--profile", required=True)
    child.add_argument("--round-driver")
    child.add_argument("--price-table", type=Path)
    child = sub.add_parser("board", help="time both arms' one-group programs on the board in one batch")
    child.add_argument("--package-job", type=Path, required=True, help="the package arm's whole-model job.json")
    child.add_argument("--reference-job", type=Path, required=True, help="the reference arm's job.json")
    child.add_argument("--groups", required=True, help="the groups, comma-separated")
    child.add_argument("--package", type=Path, help="the package (default: the package job's own snapshot)")
    child.add_argument("--out", type=Path, help="default: a perf-group-board product under out/artifacts")
    child.add_argument("--prepare-only", action="store_true", help="build, check and grade locally; submit nothing")
    child.add_argument("--timeout", type=float, default=1800)
    child = sub.add_parser("status", help="a cell run's status, or every cell run of a target")
    child.add_argument("run_dir", type=Path, nargs="?")
    child.add_argument("--target")
    child.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from . import cli as CLI
    from . import launch as LAUNCH
    from . import progress as PROGRESS

    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "prepare":
            document = prepare_cell_run(
                args.loop_run,
                cell_id=args.cell_id,
                why=args.why,
                groups=[int(g) for g in args.groups.split(",") if g.strip()],
                form_of=args.form_of,
                cells_file=args.cells,
                seed=args.seed,
                baseline_package=args.baseline_package,
                held_out=args.held_out,
                collateral_share=args.collateral_share,
                collateral_tolerance=args.collateral_tolerance,
                screen_capsules=args.screen_capsules,
                max_cycles=args.max_cycles,
                functional_model=args.functional_model,
            )
            print(json.dumps(document, indent=1, default=str))
            return 0
        if args.command == "board":
            document = board_validate(
                args.package_job,
                args.reference_job,
                [int(g) for g in args.groups.split(",") if g.strip()],
                package_dir=args.package,
                out=args.out,
                submit=not args.prepare_only,
                timeout_s=args.timeout,
            )
            print(json.dumps({k: document.get(k) for k in ("rows", "control", "run")}, indent=1, default=str))
            return 0
        if args.command == "launch":
            run_dir = CLI.resolve_run(args.run_dir)
            if not (run_dir / CELL_RECORD).is_file():
                raise CellRunError(f"{run_dir} is not a cell run (no {CELL_RECORD})")
            document = LAUNCH.launch(
                run_dir,
                profile=args.profile,
                round_driver=args.round_driver or CLI.DEFAULT_ROUND_DRIVER,
                price_table=args.price_table,
            )
            print(json.dumps(document, default=str))
            return 0
        if args.run_dir is None:
            if not args.target:
                raise CellRunError("name a cell run, or --target to list every cell run of a target")
            rows = []
            for run_dir in cell_runs(args.target):
                record = read_json(run_dir / CELL_RECORD) or {}
                status = PROGRESS.run_status(run_dir)
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "cell": record.get("cell_id"),
                        "groups": record.get("groups"),
                        "launcher_alive": status["launcher"]["alive"],
                        "stopped": (status.get("stopped") or {}).get("kind"),
                        "jobs": (status["stores"].get("screen") or {}).get("jobs"),
                    }
                )
            print(json.dumps(rows, indent=1) if args.json else "\n".join(json.dumps(r) for r in rows))
            return 0
        run_dir = CLI.resolve_run(args.run_dir)
        objective, error = CLI.objective_or_error(run_dir)
        document = cell_status(run_dir, objective=objective)
        if error:
            document["objective_error"] = error
        print(json.dumps(document, indent=1, default=str) if args.json else format_cell(document))
        return 0
    except (CellRunError, CELLS.CellError, ValueError) as exc:
        raise SystemExit(f"cell {args.command}: {exc}") from exc


__all__ = [
    "CELL_RECORD",
    "CellRunError",
    "board_validate",
    "cell_groups",
    "cell_runs",
    "cell_status",
    "champion_seed",
    "load_cells",
    "main",
    "prepare_cell_run",
]
