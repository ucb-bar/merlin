"""The fast tiers of a measured run: what an edit did, in minutes, before any board time.

    structure     Tier A: the board's own program, run on the functional model of the screen machine
                  (its ``local`` half) and screened -- every group's local correctness, its simulator
                  cycles, and, refit from the store's own (simulator, board) pairs, which kinds that
                  simulator is blind to.
    group-timing  Tier B: only the groups the candidate CHANGED against a baseline (``best``, ``seed``
                  or a digest), each built as a one-group program for the certifier's machine and
                  timed on the elaborated-RTL emulator, candidate and baseline side by side
                  (:mod:`merlin.perf.whole_model_group_timing`).
    group-check   One group's program on the functional model, on the inputs the model hands it,
                  saying per output axis where it is wrong.

None of them is the objective, and each says so in what it returns.  Their cycles are their own
device's: the functional model's are never the board's, the emulator's are compared only with the
same emulator and say which DIRECTION an edit moved a group.  A tier runs detached, one directory per
(kind, bytes[, baseline | group]) under the screen store's ``fast/``; asking again returns its state,
or its result once it has one.  A finished tier keeps its result and drops its build tree.

Nothing here names a target: the builder, the machines and the header all come from the run's own
launch config, as the screen and the certifier build them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import service as SERVICE
from .identity import load_builder, package_digest, read_json, write_json_atomic
from .jobs import ROLE_CANDIDATE

FAST_KINDS = ("structure", "group-timing", "group-check")
#: How many runs of one kind may be in progress at once (each holds simulator or emulator processes).
FAST_CONCURRENCY = 2
SPEC_SCHEMA = "merlin.phase2.whole_model_measured.fast_tier.v1"
NOT_THE_OBJECTIVE = "a fast tier: it orients an edit, it is never the measured objective"


# --------------------------------------------------------------------------------------- the tiers


def _options(spec: Mapping[str, Any], section: str) -> dict[str, Any]:
    options = dict(spec.get(f"{section}_options") or {})
    if not options.get("model_capsule") or not options.get("machine") or not options.get("header"):
        raise ValueError(f"the run's {section} build options name no model capsule, machine or header")
    return options


def _board_build(package: str | Path, *, spec: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """The board's program for ``package``, built by the run's own builder exactly as the screen builds it."""
    builder = load_builder(str(spec["builder"]), expected_sha256=spec.get("builder_sha256"))
    record = dict(builder(Path(package), target=str(spec["target"]), out_dir=out, **_options(spec, "screen")))
    full = json.loads(Path(record["notes"]["build_record"]).read_text(encoding="utf-8"))
    return {"service_record": record, "build_record": full}


def _prune(build_dir: Path) -> None:
    """Drop what a whole-model build keeps only to link; keep its records and its ELF."""
    from merlin.perf import whole_model_builder as B

    for name in ("lower", "objects", "harness"):
        shutil.rmtree(build_dir / name, ignore_errors=True)
    for pattern in B.PRUNABLE:
        for path in build_dir.glob(pattern):
            if path.is_file() and path.suffix != ".elf":
                path.unlink()


def structure(package: str | Path, *, spec: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    """Tier A.  The calibration is refit at call time from every (simulator, board) pair under the
    store base; the screen is filed there by the program's digest, so the day this ELF is measured on
    the board it becomes a calibration pair too."""
    from merlin.perf import whole_model_screen as S

    out = Path(out)
    built = _board_build(package, spec=spec, out=out / "build")
    record, full = built["service_record"], built["build_record"]
    store = Path(str(spec.get("store_base") or ""))
    calibration = None
    if store.is_dir():
        pairs = S.collect_pairs([store])
        calibration = S.fit_calibration(pairs, margin=screen_margin(store, pairs))
    screen = S.structure_screen(
        record["elf"],
        groups={g: e["compare"] for g, e in record["expectations"]["groups"].items()},
        target=str(spec["target"]),
        out=out / "screen",
        elf_sha256=record["elf_sha256"],
        templates=record.get("protocol"),
        calibration=calibration,
        routes=S.routes_of(full),
    )
    if store.is_dir() and screen.get("status") == "screened":
        filed = S.screen_dir(store, record["elf_sha256"])
        filed.mkdir(parents=True, exist_ok=True)
        for name in ("console.txt", S.SCREEN_FILE):
            shutil.copyfile(out / "screen" / name, filed / name)
    screen["text"] = render_structure(screen)
    _prune(out / "build")
    return screen


def screen_margin(store_base: Path, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The noise margin of the board the calibration pairs were measured on (:func:`.noise.margin`), from
    the solo readings of every store under ``store_base`` -- the error bound its validation holds the
    refit to. None when the pairs name no one board, which leaves the screen's ranking unvalidated."""
    from . import noise as NOISE
    from .objective import NOISE_FLOOR

    devices = {str((pair.get("domain") or {}).get("binary_sha256")) for pair in pairs if pair.get("domain")}
    if len(devices) != 1:
        return None
    roots = [p for p in sorted(Path(store_base).iterdir()) if p.is_dir() and not p.name.startswith("_")]
    machine = NOISE.machine_noise(
        NOISE.solo_readings(roots), device=next(iter(devices)), controls=NOISE.control_readings(roots)
    )
    return {**NOISE.margin(machine, floor=NOISE_FLOOR), "machine": machine}


def group_timing(
    candidate: str | Path, baseline: str | Path, *, spec: Mapping[str, Any], out: str | Path, max_parallel: int = 8
) -> dict[str, Any]:
    """Tier B.  The groups are diffed on the two packages' board builds; their one-group programs are
    built for the certifier's machine and timed on its emulator."""
    from merlin.perf import whole_model_group_timing as T

    out = Path(out)
    records = {}
    for side, package in (("candidate", candidate), ("baseline", baseline)):
        records[side] = _board_build(package, spec=spec, out=out / f"{side}_build")["build_record"]
        _prune(out / f"{side}_build")
    options = _options(spec, "certifier")
    document = T.changed_group_timing(
        {
            "package": str(candidate),
            "record": records["candidate"],
            "decline": records["candidate"].get("declined_by_caller"),
        },
        {
            "package": str(baseline),
            "record": records["baseline"],
            "decline": records["baseline"].get("declined_by_caller"),
        },
        target=str(spec["target"]),
        model_capsule=options["model_capsule"],
        machine=options["machine"],
        header=options["header"],
        header_sha256=options.get("header_sha256"),
        out=out / "timing",
        max_parallel=max_parallel,
        phase0_recipe=options.get("phase0_recipe"),
        descriptor=options.get("descriptor"),
        harness_overrides=list(options.get("harness_overrides") or ()),
    )
    document["text"] = render_group_timing(document)
    return document


def group_check(package: str | Path, group: int, *, spec: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    """ONE group's program on the screen machine's functional model, built under the run's own build
    options and verified ``local_map``: the core recomputes the group from the exact inputs it read
    and says, per output axis, where its output differs.  No board, no emulator, never an objective."""
    from merlin.perf import whole_model_group_timing as T

    from .machines import machine_from_spec

    out = Path(out)
    options = _options(spec, "screen")
    local = dict(spec.get("local_machine") or {})
    if not local:
        raise ValueError("the run's screen machine declares no functional model (its local half)")
    programs = T.build_group_programs(
        Path(package),
        [int(group)],
        model_capsule=options["model_capsule"],
        target=str(spec["target"]),
        machine=options["machine"],
        header=options["header"],
        header_sha256=options.get("header_sha256"),
        out=out / "build",
        verify="local_map",
        prohibited_roles=list(options.get("prohibited_roles") or ()),
        phase0_recipe=options.get("phase0_recipe"),
        descriptor=options.get("descriptor"),
        harness_overrides=list(options.get("harness_overrides") or ()),
    )
    program = programs[int(group)]
    document: dict[str, Any] = {
        "schema": "whole_model_group_check_v1",
        "group": int(group),
        "on": program.get("on"),
        "cause": program.get("cause"),
        "elf_sha256": program.get("elf_sha256"),
        "note": "one group's program on the functional model, on the model's own inputs to it; says nothing "
        "about timing and nothing about a hardware ordering race",
    }
    if program.get("refusal"):
        document.update(status="refused", refusal=program["refusal"])
    else:
        run = machine_from_spec({**local, "target": spec["target"]}).run(
            Path(program["elf"]), out / "run", timeout_s=1800
        )
        if not run.get("completed"):
            document.update(status="refused", refusal=f"the run did not complete: {run.get('incomplete_reason')}")
        else:
            text = Path(run["uart_log"]).read_text(encoding="utf-8", errors="replace")
            document.update(parse_group_check(text, int(group), target=str(spec["target"])))
    _prune(out / "build" / f"g{int(group)}")
    document["text"] = render_group_check(document)
    return document


def parse_group_check(text: str, group: int, *, target: str) -> dict[str, Any]:
    """The group's local verdict and mismatch map from its console, read by the target driver's own
    line templates."""
    from merlin.perf.whole_model_verdict import match_line
    from merlin.runtime.backends import base as backends

    uart = backends.whole_model_driver(target).program.UART
    found: dict[str, Any] = {"maps": {}, "samples": [], "bounded": None, "local": None, "sign": None}
    for line in text.splitlines():
        line = line.strip()
        fields = match_line(uart["local"], line)
        if fields and int(fields["group"]) == group:
            found["local"] = {k: int(fields[k]) for k in ("mismatches", "elements", "first")}
            continue
        fields = match_line(uart["local_map"], line)
        if fields and int(fields["group"]) == group:
            wrong = {}
            if fields["wrong"] != "none":
                for pair in fields["wrong"].split(","):
                    index, _, count = pair.partition(":")
                    wrong[int(index)] = int(count)
            found["maps"][fields["axis"]] = {"extent": int(fields["extent"]), "wrong": wrong}
            continue
        fields = match_line(uart["local_sample"], line)
        if fields and int(fields["group"]) == group:
            found["samples"].append({k: int(v) for k, v in fields.items() if k != "group"})
            continue
        fields = match_line(uart["local_sign"], line)
        if fields and int(fields["group"]) == group:
            found["sign"] = {k: int(v) for k, v in fields.items() if k != "group"}
            continue
        fields = match_line(uart["bounded"], line)
        if fields and int(fields["group"]) == group:
            found["bounded"] = dict(fields)
    if found["local"] is not None:
        found["status"] = "correct" if found["local"]["mismatches"] == 0 else "incorrect"
    elif found["bounded"] is not None:
        found["status"] = "correct" if str(found["bounded"].get("over")) == "0" else "incorrect"
    else:
        found["status"] = "refused"
        found["refusal"] = "the console carries no local verdict for this group"
    return found


# ----------------------------------------------------------------------------------- what is read


def render_structure(screen: Mapping[str, Any]) -> str:
    lines = [f"# {screen.get('label')}", ""]
    if screen.get("status") != "screened":
        return "\n".join([*lines, f"REFUSED: {screen.get('refusal')}", ""])
    calibration = screen.get("calibration") or {}
    lines += [
        f"- ELF `{str(screen.get('elf_sha256'))[:16]}` on the {screen.get('simulator')} functional simulator "
        f"({screen.get('wall_s')} s); THESE CYCLES ARE NOT THE BOARD'S and never an objective",
        f"- every group locally correct: **{screen.get('all_groups_correct')}** "
        f"(not correct: {screen.get('groups_not_correct')})",
        f"- calibration refit from {calibration.get('pairs')} board pairs; a group marked blind is one whose "
        f"board cost this simulator does not see (board/simulator ratio over {calibration.get('blind_ratio')})",
        _ranking_line(screen.get("ranking")),
        "",
        "| g | kind | route | local | sim cycles | blind | board/sim ratio of its class (median [p10, p90]) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in screen.get("groups") or ():
        local = row["local"] + (f" {json.dumps(row['failure'])}" if row.get("failure") else "")
        lines.append(
            f"| {row['group']} | {row['kind']} | {row.get('on') or ''} | {local} | {row['spike_cycles']:,} | "
            f"{row.get('spike_blind')} | {_ratio(row.get('class_ratio'))} |"
        )
    return "\n".join(lines) + "\n"


def _ranking_line(ranking: Mapping[str, Any] | None) -> str:
    """Whether the class ratios below may be read as a board ordering: only a validated refit says so."""
    ranking = ranking or {}
    if ranking.get("status") == "validated":
        return "- screen ranking: VALIDATED against held-out board readings of this store"
    why = "; ".join(str(r) for r in ranking.get("reasons") or ()) or "no validation was recorded"
    return f"- screen ranking: UNVALIDATED -- the ratios below are not a board ordering ({why})"


def _ratio(ratio: Mapping[str, Any] | None) -> str:
    if not ratio:
        return ""
    return f"{ratio['median_ratio']} [{ratio['p10']}, {ratio['p90']}] n={ratio['n']}"


def render_group_timing(document: Mapping[str, Any]) -> str:
    diff = document.get("diff") or {}
    lines = [
        "# Changed groups, timed on the RTL emulator",
        "",
        f"- device: {document.get('device')}",
        f"- changed groups: {diff.get('changed')} (unchanged: {len(diff.get('unchanged') or [])}); build "
        f"{document.get('build_wall_s')} s, total {document.get('total_wall_s')} s",
        "",
        "| g | kind | candidate cycles | baseline cycles | ratio | direction | candidate correct | baseline correct |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in document.get("groups") or ():
        note = row.get("candidate_refusal") or row.get("baseline_refusal") or ""
        lines.append(
            f"| {row['group']} | {row.get('kind')} | {row.get('candidate_cycles')} | {row.get('baseline_cycles')} | "
            f"{row.get('ratio', '')} | {row.get('direction', note[:80])} | {row.get('candidate_correct')} | "
            f"{row.get('baseline_correct')} |"
        )
    return "\n".join(lines) + "\n"


def _ranges(indices: list[int]) -> str:
    """``0-3,7,9-10``: a sorted index set as runs."""
    runs, start, prev = [], None, None
    for i in sorted(indices):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = i
    if start is not None:
        runs.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(runs)


def render_group_check(document: Mapping[str, Any]) -> str:
    group = document.get("group")
    lines = [f"# g{group} on the functional model, alone, on the model's inputs to it", ""]
    if document.get("status") == "refused":
        return "\n".join([*lines, f"REFUSED: {document.get('refusal')}", ""])
    local = document.get("local") or {}
    if local:
        lines.append(
            f"- {str(document.get('status')).upper()}: {local.get('mismatches')} of {local.get('elements')} "
            f"elements differ from the local reference (first flat index {local.get('first')})"
        )
    elif document.get("bounded"):
        lines.append(f"- {str(document.get('status')).upper()}: bounded check {document['bounded']}")
    sign = document.get("sign") or {}
    if sign and local.get("mismatches"):
        lines.append(
            f"- above the reference: {sign.get('over')}, below: {sign.get('under')}, largest |difference|: "
            f"{sign.get('max_abs')}"
        )
    for axis, row in (document.get("maps") or {}).items():
        wrong = row.get("wrong") or {}
        extent = row.get("extent")
        if not wrong:
            lines.append(f"- {axis} (extent {extent}): none wrong")
            continue
        full = [i for i, c in wrong.items() if extent and c]
        lines.append(
            f"- {axis} (extent {extent}): {len(full)} of {extent} have a wrong element -- {_ranges(full)[:300]}; "
            f"most wrong: {', '.join(f'{i}:{c}' for i, c in sorted(wrong.items(), key=lambda kv: -kv[1])[:6])}"
        )
    for sample in (document.get("samples") or [])[:8]:
        lines.append(
            f"  - index {sample['index']} (row {sample['row']}, col {sample['col']}, channel {sample['channel']}): "
            f"got {sample['got']}, want {sample['want']}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- detached, per bytes


def tier_spec(objective: Any) -> dict[str, Any]:
    """What a detached tier needs, from the run's own launch config and its resolved screen machine."""
    config = dict(getattr(objective, "config", None) or {})
    screen = objective.screen
    machine = dict(getattr(screen, "machine", None) or {})
    return {
        "schema": SPEC_SCHEMA,
        "target": str(screen.target),
        "builder": str((screen.builder or {}).get("spec")),
        "builder_sha256": (screen.builder or {}).get("sha256"),
        "screen_options": dict(getattr(screen, "build_options", None) or {}),
        "certifier_options": dict(getattr(objective.certifier, "build_options", None) or {})
        if getattr(objective, "certifier", None) is not None
        else None,
        "local_machine": machine.get("local") if isinstance(machine.get("local"), Mapping) else None,
        "store_base": str(Path(screen.root).parent),
        "max_parallel": int(config.get("fast_max_parallel") or 8),
    }


def _baseline(objective: Any, name: str) -> tuple[str, Path]:
    """``best`` (the run's ATTRIBUTABLE best), ``seed`` (the store's first candidate) or a digest."""
    name = str(name or "best").strip()
    if name == "best":
        best = objective._screen_best_unretracted()
        if best is None:
            raise ValueError("there is no correct measured best yet to time against; name baseline=seed or a digest")
        name = best["package_sha256"]
    elif name == "seed":
        jobs = sorted(
            (
                j
                for j in objective.screen.jobs()
                if j.get("role", ROLE_CANDIDATE) == ROLE_CANDIDATE and not j.get("replicate")
            ),
            key=lambda j: float(j.get("requested_epoch") or 0),
        )
        if not jobs:
            raise ValueError("this store has no seed yet")
        name = jobs[0]["package_sha256"]
    package = Path(objective.screen.root) / name / "package"
    if not package.is_dir():
        raise ValueError(f"no measured package {name!r} in this run's store")
    return name, package


def _view(kind: str, digest: str, baseline: str | None, status: Mapping[str, Any], result: Mapping | None) -> dict:
    view: dict[str, Any] = {
        "tier": kind,
        "package_sha256": digest,
        "baseline_sha256": baseline,
        "state": "done" if result is not None else "running",
        "started_at": status.get("started_at"),
        "not_the_objective": NOT_THE_OBJECTIVE,
    }
    if result is not None:
        view["text"] = result.get("text") or ""
        view["status"] = result.get("status")
    else:
        view["text"] = f"{kind} is running on these bytes; ask again (or wait-for-result) to read it"
    return view


def _finalize(root: Path) -> None:
    """A FINISHED tier keeps its spec, status and result; its build tree and package copy go (they are
    regenerable from the store's own snapshot of the same bytes)."""
    shutil.rmtree(root / "package", ignore_errors=True)
    shutil.rmtree(root / "baseline", ignore_errors=True)
    for child in (root / "out").iterdir() if (root / "out").is_dir() else ():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def request(
    objective: Any, kind: str, candidate: Path, *, baseline: str = "best", group: int | None = None
) -> dict[str, Any]:
    """Start (or read) one fast tier on the exact bytes of ``candidate``; never blocks on it."""
    if kind not in FAST_KINDS:
        raise ValueError(f"unknown fast tier {kind!r}; one of {FAST_KINDS}")
    digest = package_digest(Path(candidate))
    base_digest, base_package = None, None
    if kind == "group-timing":
        base_digest, base_package = _baseline(objective, baseline)
    if kind == "group-check":
        if group is None:
            raise ValueError("a group check names its group")
        key = f"{digest}__g{int(group)}"
    else:
        key = digest if base_digest is None else f"{digest}__vs_{base_digest[:16]}"
    root = Path(objective.screen.root) / "fast" / kind / key
    status = read_json(root / "status.json") or {}
    result = read_json(root / "out" / "result.json")
    if result is not None:
        _finalize(root)
        return _view(kind, digest, base_digest, status, result)
    owner = Path(objective.screen.root)
    if status.get("pid") and SERVICE.alive(status.get("pid"), owner):
        return _view(kind, digest, base_digest, status, None)
    running = [
        p
        for p in (owner / "fast" / kind).glob("*/status.json")
        if SERVICE.alive((read_json(p) or {}).get("pid"), owner)
    ]
    if len(running) >= FAST_CONCURRENCY:
        return {
            "tier": kind,
            "package_sha256": digest,
            "state": "busy",
            "not_the_objective": NOT_THE_OBJECTIVE,
            "text": f"{len(running)} {kind} run(s) already in progress; ask again when one finishes",
        }
    if root.exists():
        shutil.rmtree(root)  # a dead, resultless attempt: its log is not a result
    root.mkdir(parents=True)
    shutil.copytree(Path(candidate), root / "package", symlinks=True)
    if package_digest(root / "package") != digest:
        shutil.rmtree(root)
        raise ValueError("the package changed while it was being copied; ask again")
    spec = {
        **tier_spec(objective),
        "kind": kind,
        "package": str(root / "package"),
        "baseline": str(base_package) if base_package is not None else None,
        "group": int(group) if group is not None else None,
    }
    write_json_atomic(root / "spec.json", spec)
    with (root / "run.log").open("ab") as log:
        process = SERVICE.spawn(
            [sys.executable, "-m", SERVICE.WORKER_MODULE, "fast", str(root)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, **dict(getattr(objective.screen, "environment", None) or {})},
            cwd=str(root),
        )
    status = {
        "tier": kind,
        "pid": process.pid,
        "started_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "package_sha256": digest,
        "baseline_sha256": base_digest,
    }
    write_json_atomic(root / "status.json", status)
    return _view(kind, digest, base_digest, status, None)


def run(root: Path) -> dict[str, Any]:
    """The detached half: run the tier ``root/spec.json`` describes and write ``root/out/result.json``.
    A tier that raises writes a REFUSED result naming why, never nothing."""
    root = Path(root)
    spec = read_json(root / "spec.json") or {}
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    kind = spec.get("kind")
    try:
        if kind == "structure":
            result = structure(spec["package"], spec=spec, out=out)
        elif kind == "group-timing":
            result = group_timing(
                spec["package"], spec["baseline"], spec=spec, out=out, max_parallel=int(spec.get("max_parallel") or 8)
            )
        elif kind == "group-check":
            result = group_check(spec["package"], int(spec["group"]), spec=spec, out=out)
        else:
            raise ValueError(f"unknown fast tier {kind!r}")
    except Exception as exc:  # noqa: BLE001 -- a tier that cannot answer says why
        result = {"status": "refused", "refusal": f"{type(exc).__name__}: {exc}"[:2000]}
        result["text"] = f"# {kind}\n\nREFUSED: {result['refusal']}\n"
    result["not_the_objective"] = NOT_THE_OBJECTIVE
    write_json_atomic(out / "result.json", result)
    return result


__all__ = [
    "FAST_CONCURRENCY",
    "FAST_KINDS",
    "group_check",
    "group_timing",
    "parse_group_check",
    "render_group_check",
    "render_group_timing",
    "render_structure",
    "request",
    "run",
    "structure",
    "tier_spec",
]
