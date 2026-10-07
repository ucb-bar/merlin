"""Command line of the ``whole_model_measured`` mode.

prepare ...               prepare a run (seed, frozen inputs, policy-stamped config, oot/) and print it
start <run_dir> ...       run the authoring sessions of a prepared run until evidence or budget stops them
run ...                   prepare, then start (``--resume`` continues the latest run of the same method)
resume <run_dir> ...      prepare the next run of THIS run (method, roles, store kept), optionally
                          from another seed (``--seed``) or a revised config, and ``--launch`` it
status <run_dir>          the run's status from its own records (``--poll`` advances its objective)
follow <run_dir>          print each change (jobs, rounds, best, launcher, stop, holds) until it ends
audit-round <run_dir> N   replay round N's transcript audit, receipts and edits; compare the status
roofline ...              each group's derived roofline against measured results, per form
launch <run_dir> ...       start a prepared run DETACHED (own session, output to <run_dir>/launch.log)
stop <run_dir> --why ...  ask a run to stop at its next session boundary (``stop_requested.json``)
watch <run_dir> <pid> ... relaunch a run whose launcher exits, keeping its method and roles
export-champion <run_dir> export the run's confirmed best (its ``best`` tag) as a retention-pinned champion
work <job_dir>            run one job to its result (the detached worker)
batch <store_root>        run one board batch of a batched store (the detached batch runner)
fast <tier_dir>           run one fast tier described by its spec (the detached tier runner)
cell-prepare <run_dir>    compose a cell run's objective config from a prepared run's own config
admin <store_root> ...    operator surgery on a store (see store_admin)

The authoring round is driven by a round driver (``--round-driver module:callable``, called as
``driver(profile=, run_dir=, objective=) -> run_round``); the default is :func:`.rounds.round_driver`,
the profile's agent in the shared phase-2 sandbox, broker, transcript audit and edit authority.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import MODE

DEFAULT_ROUND_DRIVER = "merlin_experiments.phase2.whole_model_measured.rounds:round_driver"


def _price_table(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    from merlin.common.paths import _dotenv

    raw = os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or ""
    if not raw:
        raise SystemExit("no price table: pass --price-table or set AET_PRICE_TABLE")
    return Path(raw)


def _inputs(pairs: list[str]) -> dict[str, Path]:
    found = {}
    for pair in pairs or ():
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--input takes name=path, not {pair!r}")
        found[name] = Path(path)
    return found


def _latest_run(target: str, method: str) -> Path:
    from merlin.common.paths import phase_runs_root

    runs = sorted(
        p.parent
        for p in phase_runs_root(target, 2).glob("*/run.json")
        if (json.loads(p.read_text()) or {}).get("method") == method
    )
    if not runs:
        raise SystemExit(f"no prepared {MODE} run of method {method!r} to resume")
    return runs[-1]


def _oot():
    from merlin.common import oot_repo

    return oot_repo


def machine_warnings(run_dir: Path) -> list[str]:
    """The capability warnings the run's launch recorded (``machine_capabilities.json``)."""
    from . import capabilities as CAP
    from .identity import read_json

    return list((read_json(Path(run_dir) / CAP.RECORD) or {}).get("warnings") or ())


def print_machine_warnings(warnings: list[str] | tuple[str, ...]) -> None:
    """Every capability warning as its own line, beside the command's JSON (which carries the same list
    under ``machine_warnings`` on stdout): a launch never removes a lever silently.  The lines go to
    stderr so stdout stays one JSON document; a detached launch writes both into ``launch.log``."""
    for warning in warnings:
        print(f"WARNING (machine capabilities): {warning}", file=sys.stderr, flush=True)


def _prepare(args: argparse.Namespace):
    from . import runs as RUNS

    manifest = Path(args.phase0_manifest) if getattr(args, "phase0_manifest", None) else None
    if args.resume:
        if args.no_fused_regions:
            raise SystemExit("--no-fused-regions changes the experiment; resume a named run with a revised config")
        return RUNS.resume(
            _latest_run(args.target, args.method),
            why=args.why,
            oot=_oot(),
            method=args.method,
            phase0_manifest=manifest,
        )
    return RUNS.prepare(
        target=args.target,
        method=args.method,
        objective_config=_regions_opt_out(
            json.loads(Path(args.objective_config).read_text(encoding="utf-8")), args.no_fused_regions
        ),
        seed=Path(args.seed),
        prohibited_roles=list(args.prohibited_instruction_role or ()),
        inputs=_inputs(args.input),
        phase1_oot=Path(args.phase1_oot) if args.phase1_oot else None,
        why=args.why,
        oot=_oot(),
        import_evidence=Path(args.import_evidence) if args.import_evidence else None,
        phase0_manifest=manifest,
    )


def _regions_opt_out(config: dict[str, Any], opted_out: bool) -> dict[str, Any]:
    """``config`` with fused regions turned off when the operator opted out (``--no-fused-regions``)."""
    from . import config as CFG

    return {**config, CFG.FUSED_REGIONS: False} if opted_out else config


def start(run_dir: Path, *, profile_name: str, round_driver: str, price_table: Path | None) -> dict[str, Any]:
    """Run the authoring sessions of a prepared run, after checking its launch profile."""
    from . import profiles as P
    from . import rounds as RND
    from . import sessions as SES
    from .identity import load_builder

    profile = P.load(profile_name)
    checked = P.check(profile, price_table=_price_table(price_table))
    print_machine_warnings(machine_warnings(Path(run_dir)))
    _record, objective = _objective_of(Path(run_dir))
    # THIS PROCESS IS THE LAUNCHER: it writes the run's heartbeat (pid, last activity, last measurement).
    from . import liveness as LIVE

    objective.heartbeat = LIVE.Heartbeat(
        Path(run_dir),
        stores=lambda: [objective.screen.root, *([objective.certifier.root] if objective.certifier else [])],
    )
    objective.heartbeat.tick("launched", force=True)
    # THE SEED IS MEASURED FIRST: the bar, the coverage floor and the noise margin are all the seed's,
    # and a candidate requested before any of them exists carries no coverage gate.  Idempotent: a
    # relaunch finds the seed's job in the store and requests nothing new.
    seed = Path(run_dir) / "seed" / "submission"
    if seed.is_dir():
        objective.measure(seed, label="seed", seed=True)
    stage = Path(run_dir) / "stage"
    _recover_killed_rounds(Path(run_dir), objective)
    run_round = load_builder(round_driver)(profile=profile, run_dir=Path(run_dir), objective=objective)
    return SES.run_sessions(
        objective,
        run_round=run_round,
        stage_root=stage,
        run=Path(run_dir).name,
        max_sessions=int(profile["max_sessions"]),
        total_seconds=float(profile["total_authoring_seconds"]),
        driver=str(profile["driver"]),
        model=str(checked["resolved_model"]),
        stop_request=Path(run_dir) / SES.OPERATOR_STOP_FILE,
        first_session=RND.next_session(stage),
    )


def _recover_killed_rounds(run_dir: Path, objective) -> list[dict[str, Any]]:
    """Close the rounds a killed driver left open: this run's own (a relaunched ``start``) and the run it
    was resumed from (a relaunch by the watchdog), whose store this run keeps."""
    from . import rounds as RND
    from .identity import read_json

    recovered = RND.recover_killed_rounds(run_dir / "stage", attribute=objective.attribute, run_name=run_dir.name)
    previous = (read_json(run_dir / "resumed_seed.json") or {}).get("resumed_from_run")
    if previous and Path(previous).is_dir():
        # The previous run's own history is its ledger's; this store's attribution is the shared fact.
        recovered += RND.recover_killed_rounds(
            Path(previous) / "stage", attribute=objective.screen.attribute, run_name=Path(previous).name
        )
    return recovered


def _objective_of(run_dir: Path):
    from . import config as CFG
    from . import runs as RUNS
    from .identity import read_json
    from .ledger import ITERATIONS, OotLedger

    record = read_json(Path(run_dir) / "run.json") or {}
    if record.get("schema") != RUNS.RUN_SCHEMA:
        raise SystemExit(f"{run_dir} is not a prepared {MODE} run")
    config = json.loads((Path(run_dir) / RUNS.CONFIG_NAME).read_text(encoding="utf-8"))
    objective = CFG.from_config(config, target=str(record["target"]))
    # THE STORE IS THE ONE THE RUN WAS PREPARED ON.  Its key includes the builder's source, so a code
    # change between `prepare` and `start` (or a relaunched `start`) silently opened a new, empty store:
    # the seed was requested again (hours on the screen) and the history the run was prepared on was
    # out of view.  `resume` refuses that move without a reason; so does everything that opens a run.
    prepared = ((read_json(Path(run_dir) / "resumed_seed.json") or {}).get("store_roots") or {}).get("screen")
    if prepared and Path(objective.screen.root) != Path(prepared):
        raise SystemExit(
            f"{run_dir} was prepared on the store {prepared}, but this code keys it as "
            f"{objective.screen.root} (the builder's source changed); run the code it was prepared with, "
            "or resume it with allow_new_store and a reason"
        )
    if (Path(run_dir) / "oot").is_dir():
        objective.ledger = OotLedger(
            Path(run_dir) / "oot",
            run_id=Path(run_dir).name,
            records=Path(run_dir) / ITERATIONS,
            sandbox_roots=(Path(run_dir) / "workspace", Path(run_dir) / "stage" / "agent_workspaces"),
        )
    return record, objective


def objective_or_error(run_dir: Path) -> tuple[Any, str | None]:
    """The run's objective, or None and why it could not be opened (a status never fails for it)."""
    try:
        return _objective_of(Path(run_dir))[1], None
    except SystemExit as exc:
        return None, str(exc.code)
    except Exception as exc:  # noqa: BLE001 -- reported beside the records that could be read
        return None, f"{type(exc).__name__}: {exc}"


def resume_run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare the next run of ``args.run_dir`` and, with ``--launch``, start it detached."""
    from . import launch as LAUNCH
    from . import runs as RUNS

    if args.launch and not args.profile:
        raise SystemExit("--launch needs --profile")  # refused before a run is prepared, not after
    previous = resolve_run(args.run_dir)
    config = json.loads(Path(args.objective_config).read_text(encoding="utf-8")) if args.objective_config else None
    try:
        prepared = RUNS.resume(
            previous,
            why=args.why,
            workspace=Path(args.seed) if args.seed else None,
            objective_config=config,
            allow_new_store=args.allow_new_store,
            oot=_oot(),
        )
    except RUNS.RunError as exc:
        raise SystemExit(str(exc)) from exc
    document: dict[str, Any] = {
        "run_dir": str(prepared.run_dir),
        "resumed_from": str(previous),
        "method": prepared.method,
        "config_sha256": prepared.config_sha256,
        "seed_package_sha256": prepared.seed_package_sha256,
        "store_roots": dict(prepared.store_roots),
        "machine_warnings": list(prepared.machine_warnings),
    }
    print_machine_warnings(prepared.machine_warnings)
    if args.launch:
        document["launch"] = LAUNCH.launch(
            prepared.run_dir, profile=args.profile, round_driver=args.round_driver, price_table=args.price_table
        )
    return document


def roofline_report(args: argparse.Namespace) -> Path:
    """Write the roofline report as a versioned product and return its path."""
    from merlin.common.artifacts import new_product

    from . import roofline as ROOF
    from . import runs as RUNS
    from .identity import read_json

    target, capsule = args.target, args.model_capsule
    if args.run is not None:
        run_dir = resolve_run(args.run)
        record = read_json(run_dir / "run.json") or {}
        config = read_json(run_dir / RUNS.CONFIG_NAME) or {}
        section = config.get("certifier") or config.get("screen") or {}
        target = target or record.get("target")
        capsule = capsule or (section.get("build_options") or {}).get("model_capsule")
    if not target or not capsule:
        raise SystemExit("name --target and --model-capsule, or a --run whose config names them")
    results = _inputs(args.result)
    if not results:
        raise SystemExit("name at least one --result label=result.json to confront the rooflines with")
    document = ROOF.report(str(target), str(capsule), results, emulator=args.emulator)
    product = new_product("perf-roofline", version=1, target=str(target), sources=[str(p) for p in results.values()])
    path = product.add_artifact("roofline.json")
    path.write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    product.write_manifest()
    return path


def export_champion(args: argparse.Namespace) -> Path:
    from .ledger import export_best

    record, objective = _objective_of(args.run_dir)
    if objective.ledger is None:
        raise SystemExit(f"{args.run_dir} has no oot/ repository to export from")
    objective.poll()
    return export_best(
        objective,
        objective.ledger,
        target=str(record["target"]),
        package_id=args.package_id,
        roles=list(record.get("prohibited_instruction_roles") or ()),
        phase1_run=args.phase1_run,
        frozen_commit=args.frozen_commit,
        corpus_seal_digest=args.corpus_seal_digest,
        phase0_evidence_digest=args.phase0_evidence_digest,
    )


def _spawn_start(prepared, *, profile: str, round_driver: str, price_table: Path | None) -> int:
    from . import launch as LAUNCH

    return int(
        LAUNCH.launch(prepared.run_dir, profile=profile, round_driver=round_driver, price_table=price_table)["pid"]
    )


def resolve_run(path: Path) -> Path:
    """The measured run ``path`` names: a prepared run directory itself, or a phase orchestration
    directory whose phase-2 record points at one (``phase2/phase_run.json``)."""
    from . import runs as RUNS
    from .identity import read_json

    path = Path(path).expanduser().resolve()
    if (read_json(path / "run.json") or {}).get("schema") == RUNS.RUN_SCHEMA:
        return path
    pointer = (read_json(path / "phase2" / "phase_run.json") or {}).get("run_dir")
    if pointer and (read_json(Path(pointer) / "run.json") or {}).get("schema") == RUNS.RUN_SCHEMA:
        return Path(pointer).resolve()
    raise SystemExit(f"{path} is neither a prepared {MODE} run nor an orchestration run that points at one")


def successors(run_dir: Path) -> list[str]:
    """The runs prepared by resuming ``run_dir`` (a watchdog relaunch continues in a NEW directory)."""
    from merlin.common.paths import phase_runs_root

    from .identity import read_json

    record = read_json(Path(run_dir) / "run.json") or {}
    if not record.get("target"):
        return []
    found = []
    for path in sorted(phase_runs_root(str(record["target"]), 2).glob("*/resumed_seed.json")):
        previous = (read_json(path) or {}).get("resumed_from_run")
        if previous and Path(previous).resolve() == Path(run_dir).resolve():
            found.append(str(path.parent))
    return found


def stop(run_dir: Path, *, why: str) -> dict[str, Any]:
    """Ask the run to stop at its next session boundary.  The request names the run it was written to;
    when that run was already resumed into a newer one, the newer run is named so the operator can stop
    the run that is actually going."""
    from . import launch as LAUNCH
    from . import sessions as SES

    run_dir = resolve_run(run_dir)
    try:
        request = SES.request_stop(run_dir, why=why)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return {
        "run_dir": str(run_dir),
        "request": request,
        "request_path": str(run_dir / SES.OPERATOR_STOP_FILE),
        "launcher_alive": LAUNCH.launcher_alive(run_dir),
        "resumed_into": successors(run_dir),
        "note": "the run finishes its current session first; nothing is signalled",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=MODE, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        child = sub.add_parser(name)
        child.add_argument("--target", required=True)
        child.add_argument("--method", required=True)
        child.add_argument("--why", required=True)
        child.add_argument("--objective-config", type=Path)
        child.add_argument("--seed", type=Path)
        child.add_argument("--input", action="append", default=[], help="name=path: a large input frozen by content")
        child.add_argument("--prohibited-instruction-role", action="append", default=[])
        child.add_argument(
            "--phase0-manifest",
            type=Path,
            help="the sealed Phase 0 corpus MANIFEST.yaml whose instruction_policy the roles are held to "
            "(default: the config's phase0_manifest, else the selected descriptor's corpus manifest)",
        )
        child.add_argument("--phase1-oot", type=Path)
        child.add_argument(
            "--import-evidence",
            type=Path,
            help="the result.json measuring an IMPORTED seed (no Phase 1 freeze); recorded as imported, not frozen",
        )
        child.add_argument("--resume", action="store_true", help="continue the latest run of this method")
        child.add_argument(
            "--no-fused-regions",
            action="store_true",
            help="opt out of fused regions (on by default for a closed model: a package may claim adjacent groups)",
        )
        if name == "run":
            child.add_argument("--profile", required=True)
            child.add_argument("--round-driver", default=DEFAULT_ROUND_DRIVER)
            child.add_argument("--price-table", type=Path)
            child.add_argument("--record-dir", type=Path, help="where to write a pointer to the phase run")
    child = sub.add_parser("start")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--profile", required=True)
    child.add_argument("--round-driver", default=DEFAULT_ROUND_DRIVER)
    child.add_argument("--price-table", type=Path)
    child = sub.add_parser("resume", help="prepare the next run of this run, keeping its method, roles and store")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--why", required=True)
    child.add_argument("--seed", type=Path, help="seed the next run from this package, not the run's workspace")
    child.add_argument("--objective-config", type=Path, help="a revised objective config for the next run")
    child.add_argument("--allow-new-store", help="why the next run may open a new measurement store")
    child.add_argument("--launch", action="store_true", help="start the prepared run detached")
    child.add_argument("--profile")
    child.add_argument("--round-driver", default=DEFAULT_ROUND_DRIVER)
    child.add_argument("--price-table", type=Path)
    child = sub.add_parser("status", help="the run's status, read from its own records")
    child.add_argument("run_dir", type=Path)
    child.add_argument(
        "--stall-hours", type=float, default=None, help="hours without a measured candidate that mean STALLED"
    )
    child.add_argument("--poll", action="store_true", help="advance the objective first (dispatches pending jobs)")
    child.add_argument("--json", action="store_true")
    child.add_argument("--history", type=int, default=10, help="how many recent measurements to show")
    child = sub.add_parser("follow", help="print each change to the run until it is over")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--interval", type=float, default=60.0)
    child.add_argument("--max-seconds", type=float)
    child = sub.add_parser("audit-round", help="replay one recorded round's audit and compare its status")
    child.add_argument("run_dir", type=Path)
    child.add_argument("round", type=int, help="the round's index, as its files are named (round_NN)")
    child.add_argument("--line", type=int, action="append", default=[], help="also print this transcript line")
    child.add_argument("--json", action="store_true")
    child = sub.add_parser("roofline", help="every group's derived roofline against measured results, per form")
    child.add_argument("--run", type=Path, help="take the target and model capsule from this measured run")
    child.add_argument("--target")
    child.add_argument("--model-capsule", type=Path)
    child.add_argument("--result", action="append", default=[], help="label=result.json: one measured arm")
    child.add_argument("--emulator", type=Path, help="the dump-capable emulator whose elaboration is read")
    child = sub.add_parser("launch", help="start a prepared run detached, its output appended to launch.log")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--profile", required=True)
    child.add_argument("--round-driver", default=DEFAULT_ROUND_DRIVER)
    child.add_argument("--price-table", type=Path)
    child = sub.add_parser("stop", help="ask a run to stop at its next session boundary")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--why", required=True)
    child = sub.add_parser("watch")
    child.add_argument("run_dir", type=Path)
    child.add_argument("pid", type=int)
    child.add_argument("--profile", required=True)
    child.add_argument("--round-driver", default=DEFAULT_ROUND_DRIVER)
    child.add_argument("--price-table", type=Path)
    child.add_argument("--why", required=True)
    child.add_argument("--max-relaunches", type=int, default=6)
    child.add_argument("--stall-hours", type=float, default=None, help="hours without a measurement that mean STALLED")
    child.add_argument(
        "--notify-command",
        help="a command run once per stall (split like a shell would, never run by one; {run_dir} and "
        "{reason} are substituted)",
    )
    child = sub.add_parser("export-champion")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--package-id", required=True)
    child.add_argument("--phase1-run", required=True)
    child.add_argument("--frozen-commit", required=True)
    child.add_argument("--corpus-seal-digest", required=True)
    child.add_argument("--phase0-evidence-digest", required=True)
    sub.add_parser("work").add_argument("job_dir", type=Path)
    sub.add_parser("batch").add_argument("root", type=Path)
    sub.add_parser("fast").add_argument("root", type=Path)
    child = sub.add_parser("cell-prepare")
    child.add_argument("run_dir", type=Path, help="a prepared run whose config the cell's is composed from")
    child.add_argument("--cell-id", required=True)
    child.add_argument("--groups", default="", help="the cell's groups (comma-separated)")
    child.add_argument("--form-of", type=int, action="append", default=[], help="a group whose whole FORM is the cell")
    child.add_argument("--held-out", action="append", default=[], help="a held-out model capsule directory")
    child.add_argument(
        "--collateral-share", type=Path, help="a result or rows document picking each other form's representative"
    )
    child.add_argument(
        "--baseline-package", type=Path, help="the VERIFIED package the collateral baseline is measured on"
    )
    child.add_argument("--collateral-tolerance", type=float, default=0.01)
    child.add_argument("--max-cycles", type=int)
    child.add_argument(
        "--functional-model",
        help="a machine of the loop's own registry a candidate's efficiency census runs on (a paired "
        "machine's local half is taken); without it the cell's efficiency rows say none was given",
    )
    child.add_argument("--out", required=True, type=Path)
    return parser


def _functional_model(loop: Mapping[str, Any], name: str | None) -> dict[str, Any] | None:
    """Registry machine ``name`` of the loop's own screen registry, as a functional-model spec: a paired
    or batched machine's local half, a single machine as itself. None when no name is given."""
    if not name:
        return None
    from . import registry as REG

    registry = ((loop.get("screen") or {}).get("machine") or {}).get("registry")
    if not registry:
        raise SystemExit("--functional-model names a registry machine, and the loop's screen names no registry")
    spec = REG.resolve(registry, name)
    return dict(spec["local"]) if spec.get("kind") in ("paired", "batched") else spec


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["admin"]:
        from .store_admin import main as admin

        return admin(raw[1:])
    args = _parser().parse_args(raw)
    if args.command in ("prepare", "run"):
        if not args.resume and (args.objective_config is None or args.seed is None):
            raise SystemExit("a new run needs --objective-config and --seed (or --resume)")
        prepared = _prepare(args)
        print_machine_warnings(prepared.machine_warnings)
        print(
            json.dumps(
                {
                    "run_dir": str(prepared.run_dir),
                    "config_sha256": prepared.config_sha256,
                    "method": prepared.method,
                    "machine_warnings": list(prepared.machine_warnings),
                }
            )
        )
        if args.command == "prepare":
            return 0
        if args.record_dir is not None:
            args.record_dir.mkdir(parents=True, exist_ok=True)
            (args.record_dir / "phase_run.json").write_text(json.dumps({"run_dir": str(prepared.run_dir)}) + "\n")
        document = start(
            prepared.run_dir, profile_name=args.profile, round_driver=args.round_driver, price_table=args.price_table
        )
        print(json.dumps(document.get("stopped"), default=str))
        return 0
    if args.command == "start":
        document = start(
            args.run_dir, profile_name=args.profile, round_driver=args.round_driver, price_table=args.price_table
        )
        print(json.dumps(document.get("stopped"), default=str))
        return 0
    if args.command == "resume":
        print(json.dumps(resume_run(args), default=str))
        return 0
    if args.command == "status":
        from . import progress as PROGRESS

        run_dir = resolve_run(args.run_dir)
        objective, error = objective_or_error(run_dir)
        document = PROGRESS.run_status(
            run_dir, objective=objective, poll=args.poll, history=args.history, stall_hours=args.stall_hours
        )
        if error:
            document["objective_error"] = error
        print(json.dumps(document, indent=1, default=str) if args.json else PROGRESS.format_status(document))
        return 0
    if args.command == "follow":
        from . import progress as PROGRESS

        run_dir = resolve_run(args.run_dir)
        objective, error = objective_or_error(run_dir)
        if error:
            print(f"(the objective could not be opened, so no best is followed: {error})", flush=True)
        ended = PROGRESS.follow(
            run_dir,
            objective=objective,
            interval=args.interval,
            max_seconds=args.max_seconds,
            out=lambda line: print(line, flush=True),
        )
        print(json.dumps(ended))
        return 0
    if args.command == "audit-round":
        from . import round_audit as AUDIT

        run_dir = resolve_run(args.run_dir)
        try:
            document = AUDIT.audit_round(run_dir, args.round)
        except AUDIT.RoundAuditError as exc:
            raise SystemExit(str(exc)) from exc
        if args.line:
            document["transcript_lines"] = AUDIT.transcript_lines(run_dir, args.round, args.line)
        print(json.dumps(document, indent=1, default=str) if args.json else AUDIT.format_audit(document))
        for number, text in (document.get("transcript_lines") or {}).items() if not args.json else ():
            print(f"---- line {number}\n{text}")
        return 0
    if args.command == "roofline":
        print(roofline_report(args))
        return 0
    if args.command == "launch":
        from . import launch as LAUNCH

        try:
            document = LAUNCH.launch(
                resolve_run(args.run_dir),
                profile=args.profile,
                round_driver=args.round_driver,
                price_table=args.price_table,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        warnings = machine_warnings(resolve_run(args.run_dir))
        print_machine_warnings(warnings)
        print(json.dumps({**document, "machine_warnings": warnings}, default=str))
        return 0
    if args.command == "stop":
        print(json.dumps(stop(args.run_dir, why=args.why), default=str))
        return 0
    if args.command == "watch":
        from . import runs as RUNS
        from . import watchdog as WD

        document = WD.watch(
            args.run_dir,
            args.pid,
            launch=lambda prepared: _spawn_start(
                prepared, profile=args.profile, round_driver=args.round_driver, price_table=args.price_table
            ),
            why=args.why,
            policy=WD.WatchPolicy(
                max_relaunches=args.max_relaunches,
                **({"stall_hours": args.stall_hours} if args.stall_hours is not None else {}),
                notify_command=shlex.split(args.notify_command) if args.notify_command else None,
            ),
            resume=lambda run_dir, *, why: RUNS.resume(run_dir, why=why, oot=_oot()),
        )
        print(json.dumps(document, default=str))
        return 0
    if args.command == "export-champion":
        print(export_champion(args))
        return 0
    if args.command == "cell-prepare":
        from . import cell_prep as CP
        from . import forms as FORMS
        from . import runs as RUNS

        record = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
        loop = json.loads((args.run_dir / RUNS.CONFIG_NAME).read_text(encoding="utf-8"))
        target = str(record["target"])
        groups = [int(g) for g in args.groups.split(",") if g.strip()]
        if args.form_of:
            capsule = loop["certifier"]["build_options"]["model_capsule"]
            groups = sorted(
                set(groups) | set(CP.form_cell_groups(FORMS.statement_forms(capsule, target=target), args.form_of))
            )
        if not groups:
            raise SystemExit("name the cell's --groups or a --form-of anchor")
        prepared = CP.prepare(
            loop,
            target=target,
            cell_id=args.cell_id,
            groups=groups,
            held_out_capsules=args.held_out,
            collateral_share=FORMS.load_counts(args.collateral_share) if args.collateral_share else None,
            baseline_package=args.baseline_package,
            collateral_tolerance=args.collateral_tolerance,
            out=args.out,
            max_cycles=args.max_cycles,
            functional_model=_functional_model(loop, args.functional_model),
        )
        print(
            json.dumps(
                {
                    "config": str(args.out / "cell_objective_config.json"),
                    "groups": groups,
                    "held_out": prepared["held_out"],
                    # None means the candidate's programs run under the flat bound: say so, never imply.
                    "baseline": prepared["baseline"],
                }
            )
        )
        return 0
    if args.command == "fast":
        from . import fast as FAST

        FAST.run(args.root)
        return 0
    if args.command == "work":
        from .worker import worker_main

        return worker_main(args.job_dir.resolve())
    if args.command == "batch":
        from .batch import batch_main

        return batch_main(args.root.resolve())
    return 2


__all__ = ["main", "objective_or_error", "resolve_run", "resume_run", "start", "stop", "successors"]
