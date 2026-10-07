"""``merlin experiment inspect <candidate|package> --group gN [--stage S] [--trace]``: one group, rebuilt.

Debugging one group of a whole-model candidate used to mean a whole-model build and a hand search
through its work tree. This rebuilds THE GROUP ALONE through machinery that already exists, in a fresh
directory under the purgeable cache, and prints where everything is:

1. **The candidate.** A measured-mode job directory (``job.json`` names the target and the build
   options; ``package/`` is the snapshot that was measured), or a package directory with the same
   build options given explicitly (``--build-options FILE`` or ``--capsule``/``--machine``/...).
2. **The group's program.** :func:`merlin.perf.whole_model_group_timing.build_group_programs` with
   ``ask_only``: the package is asked for that group alone, every other group is stated by the
   reference, and the group becomes a one-step program of the target's own whole-model driver -- the
   interface the package was given, its command buffer and target artifact, the object, the program
   source and its ELF.
3. **Its IR at every stage.** The rebuild runs inside a compile trace that dumps every stage
   (:mod:`merlin.common.compile_trace`); a package whose compiler lowers through Merlin reports its
   stages from its own process. ``--stage S`` prints the IR of one stage, or one product.
4. **An instruction trace** (``--trace``): the group's program on the candidate's own functional
   model -- the ``spike`` machine its job declares -- with the simulator's execution log, stopped
   after ``--run-to`` instructions.
5. **Source-line attribution** (``--trace``): the program is linked a second time from the same
   inputs with debug information added, and that companion is admitted only if its allocated bytes and
   relocations are the program's (:func:`merlin.perf.debug_companion.verify_debug_companion`). The
   functional model's PC histogram of the program is then symbolized against the companion by the LLVM
   symbolizer of the target's own toolchain (beside the compiler its build recipe names) and attributed
   per function and source line (:func:`merlin.perf.debug_companion.attribute_symbolized_pcs`). Any step
   that cannot be taken leaves the attribution ``UNKNOWN``, with the reason.

Each step uses a hook the TARGET provides (its whole-model driver, a functional-model machine). A
target or candidate without one is told "not available for this target" and why; nothing is guessed,
and no target is named here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Instructions the execution log covers when ``--run-to`` is not given: enough to see a group's
#: kernel start, small enough that the log stays tens of megabytes.
DEFAULT_RUN_TO = 100_000
#: Lines of a stage's IR printed inline (the whole file is at the printed path).
DEFAULT_LINES = 60
#: The build options a group program needs, as a measured job's ``build_options`` names them.
_REQUIRED = ("model_capsule", "machine", "header")
_SPIKE = "spike"


class InspectError(RuntimeError):
    """The candidate cannot be inspected; the message says which input or hook is missing."""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """The group-inspection options, on the ``inspect`` verb (they apply only with ``--group``)."""
    group = parser.add_argument_group("one group of a candidate (with --group)")
    group.add_argument("--group", help="rebuild this group alone (gN) and show its IR, program and trace")
    group.add_argument("--stage", help="print the IR of this stage (a stage the trace reached, or a product name)")
    group.add_argument(
        "--trace", action="store_true", help="run the group's program on the functional model with an instruction log"
    )
    group.add_argument(
        "--run-to", type=int, default=DEFAULT_RUN_TO, help="instructions the log covers (default %(default)s)"
    )
    group.add_argument("--lines", type=int, default=DEFAULT_LINES, help="lines of the stage's IR to print")
    group.add_argument("--out", type=Path, help="work directory (default: a fresh one under the cache)")
    group.add_argument("--json", action="store_true", help="print the inspection as JSON")
    group.add_argument("--target", help="package mode: the target")
    group.add_argument(
        "--build-options", type=Path, help="package mode: JSON/YAML build options (model_capsule, machine, header, ...)"
    )
    group.add_argument("--capsule", help="package mode: the model capsule directory")
    group.add_argument("--machine", help="package mode: the hardware-registry machine the program is built for")
    group.add_argument("--header", help="package mode: that machine's parameter header")
    group.add_argument("--phase0-recipe", help="package mode: the Phase 0 recipe naming the corpus binding")
    group.add_argument("--descriptor", help="package mode: the target descriptor")
    group.add_argument("--machine-spec", type=Path, help="package mode: a JSON functional-model machine for --trace")


# ------------------------------------------------------------------------------------- candidate


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise InspectError(f"{path} holds no mapping")
    return dict(value)


def functional_machine(machine: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The ``spike``-kind machine a job's machine spec runs its local functional grade on, if any: the
    spec itself, or the ``local`` half of a composite one."""
    if not isinstance(machine, Mapping):
        return None
    if machine.get("kind") == _SPIKE:
        return dict(machine)
    local = machine.get("local")
    return dict(local) if isinstance(local, Mapping) and local.get("kind") == _SPIKE else None


def resolve_candidate(source: str | Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    """``{package, target, options, machine, source}`` for a measured job directory or a package directory."""
    source = Path(source).expanduser()
    job_file = source / "job.json"
    if job_file.is_file():
        job = json.loads(job_file.read_text(encoding="utf-8"))
        package = source / "package"
        if not package.is_dir():
            raise InspectError(f"{source} is a job directory without its package/ snapshot")
        return {
            "source": str(source),
            "kind": "measured job",
            "package": package,
            "target": str(job["target"]),
            "options": dict(job.get("build_options") or {}),
            "machine": functional_machine(job.get("machine")),
        }
    if not source.is_dir():
        raise InspectError(f"{source} is neither a measured job directory nor a package directory")
    args = args or argparse.Namespace()
    options = _load_mapping(args.build_options) if getattr(args, "build_options", None) else {}
    for key, flag in (
        ("model_capsule", "capsule"),
        ("machine", "machine"),
        ("header", "header"),
        ("phase0_recipe", "phase0_recipe"),
        ("descriptor", "descriptor"),
    ):
        if getattr(args, flag, None):
            options[key] = getattr(args, flag)
    target = getattr(args, "target", None) or options.pop("target", None)
    if not target:
        raise InspectError("a package directory needs --target (a measured job directory names its own)")
    spec = _load_mapping(args.machine_spec) if getattr(args, "machine_spec", None) else None
    return {
        "source": str(source),
        "kind": "package",
        "package": source,
        "target": str(target),
        "options": options,
        "machine": functional_machine(spec),
    }


# --------------------------------------------------------------------------------------- rebuild


def _group(text: str) -> int:
    from merlin.perf.whole_model_partial import parse_groups

    groups = parse_groups([text])
    if len(groups) != 1:
        raise InspectError(f"--group names one group (gN), not {text!r}")
    return groups[0]


def _owned(path: Path, root: Path, group: int) -> bool:
    """Whether a file the rebuild wrote belongs to ``group``: a path component that IS the group's tag
    (``g12/``) or starts with it (``g12.iface.mlir``, ``g12.generated/``)."""
    tag = f"g{group}"
    return any(part == tag or part.startswith(tag + ".") for part in path.relative_to(root).parts)


def rebuild(
    candidate: Mapping[str, Any], group: int, work: Path, *, timeout: int = 900, debug_companion: bool = False
) -> dict[str, Any]:
    """Build ``group`` of ``candidate`` alone under ``work``, inside a compile trace that dumps every stage.
    ``debug_companion`` also links the program's debug-information companion (for source attribution)."""
    from merlin.common import compile_trace as T
    from merlin.compile import debug
    from merlin.perf import whole_model_group_timing as GT
    from merlin.runtime.backends import base as backends

    options = candidate["options"]
    missing = [key for key in _REQUIRED if not options.get(key)]
    if missing:
        raise InspectError(f"the candidate's build options name no {missing}")
    target = candidate["target"]
    try:
        backends.whole_model_driver(target)
    except (NotImplementedError, LookupError, ImportError) as exc:
        raise InspectError(f"not available for this target: {exc}") from None
    build_dir = work / "build"
    request = T.Request(directory=str(work / "trace"), dump_after=(T.ALL,))
    with debug.opened(request, ["merlin experiment inspect", str(candidate["source"]), f"--group g{group}"]):
        before = T.snapshot(work)
        records = GT.build_group_programs(
            candidate["package"],
            [group],
            model_capsule=options["model_capsule"],
            target=target,
            machine=str(options["machine"]),
            header=str(options["header"]),
            header_sha256=options.get("header_sha256"),
            out=build_dir,
            prohibited_roles=list(options.get("prohibited_roles") or ()),
            harness_overrides=list(options.get("harness_overrides") or ()),
            ask_only=True,
            phase0_recipe=options.get("phase0_recipe"),
            descriptor=options.get("descriptor"),
            timeout=timeout,
            jobs=1,
            keep_statement=True,
            debug_companion=debug_companion,
        )
        written = [p for p in T.written_since(build_dir, before) if _owned(p, build_dir, group)]
    record = dict(records.get(group) or {})
    return {
        "group": f"g{group}",
        "record": record,
        "work": str(work),
        "trace": str(Path(request.directory).absolute() / T.INDEX),
        "products": {str(p.relative_to(build_dir)): str(p) for p in written},
    }


def stage_files(result: Mapping[str, Any], stage: str) -> list[str]:
    """The files of ``stage``: a stage the trace reached (``mlir:cse#2`` names an occurrence), else a
    product whose path names it (``command_buffer``, ``artifact``, ``iface``, ``.o``)."""
    from merlin.common import compile_trace as T

    index_path = Path(result["trace"])
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"stages": []}
    name, ordinal = T.parse_selector(stage)
    seen = 0
    for event in index.get("stages") or ():
        if event.get("stage") != name or event.get("when") == "before":
            continue
        seen += 1
        if ordinal not in (None, seen):
            continue
        files = event.get("files") or ([event["file"]] if event.get("file") else [])
        if files:
            return [str(index_path.parent / f) for f in files]
    return [path for rel, path in sorted(result["products"].items()) if stage in rel]


# ------------------------------------------------------------------------------ instruction trace


def instruction_trace(candidate: Mapping[str, Any], elf: str | None, work: Path, *, run_to: int) -> dict[str, Any]:
    """The group's program on the candidate's functional model, with the simulator's execution log.

    The machine is the one the candidate's own job runs its local grade on; only a ``spike``-kind
    machine has an execution log (``-l --log=FILE --instructions=N`` are that simulator's own options)."""
    machine = candidate.get("machine")
    if machine is None:
        return {
            "available": False,
            "why": "not available for this target: the candidate declares no functional-model (spike) machine",
        }
    if not elf or not Path(elf).is_file():
        return {"available": False, "why": "the group's program did not build, so there is nothing to run"}
    command = [str(c) for c in machine.get("command") or ()]
    if not command:
        return {"available": False, "why": "the functional-model machine names no command"}
    log = work / "instruction_trace.log"
    argv = [command[0], "-l", f"--log={log}", f"--instructions={int(run_to)}", *command[1:], str(elf)]
    env = {**os.environ, **{str(k): str(v) for k, v in (machine.get("environment") or {}).items()}}
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "argv": argv, "error": f"{type(exc).__name__}: {exc}", "log": str(log)}
    lines = sum(1 for _ in log.open(encoding="utf-8", errors="replace")) if log.is_file() else 0
    return {
        "available": True,
        "argv": argv,
        "returncode": done.returncode,
        "log": str(log) if log.is_file() else None,
        "log_lines": lines,
        "stdout_tail": (done.stdout or "")[-1500:],
    }


# ---------------------------------------------------------------------------- source attribution

#: The symbolizer whose JSON records :func:`merlin.perf.debug_companion.attribute_symbolized_pcs` reads.
SYMBOLIZER = "llvm-symbolizer"
UNKNOWN = "UNKNOWN"


def symbolizer_beside(compiler: str | Path | None) -> Path | None:
    """The LLVM symbolizer of the toolchain that built the program: the one in the directory of the
    compiler its build recipe names. None when that toolchain has none -- never one from elsewhere,
    which would read another toolchain's view of the program's debug information."""
    if not compiler:
        return None
    candidate = Path(str(compiler)).with_name(SYMBOLIZER)
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _unknown(why: str, **known: Any) -> dict[str, Any]:
    return {"status": UNKNOWN, "why": why, **known}


def symbolize(symbolizer: Path, elf: str | Path, addresses: Sequence[int]) -> list[dict[str, Any]]:
    """The symbolizer's JSON record for every address, one per address, against ``elf``."""
    done = subprocess.run(
        [str(symbolizer), f"--obj={elf}", "--output-style=JSON", "--inlining"],
        input="".join(f"{hex(address)}\n" for address in addresses),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if done.returncode != 0:
        raise ValueError(f"{symbolizer} exited {done.returncode}: {(done.stderr or '').strip()[-400:]}")
    return [json.loads(line) for line in done.stdout.splitlines() if line.strip()]


def source_attribution(candidate: Mapping[str, Any], record: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """The group's program, its PC histogram on the functional model, attributed to functions and source
    lines through its verified debug companion -- or ``UNKNOWN`` and why.

    The counts are the functional model's instruction executions of the whole one-group program (its
    setup and checks included), never cycles and never a measured region."""
    from merlin.perf import debug_companion as DC
    from merlin.perf import group_efficiency as E

    machine = candidate.get("machine")
    if machine is None:
        return _unknown("the candidate declares no functional-model (spike) machine, so there is no PC census")
    elf, program_object = record.get("elf"), record.get("program_object")
    if not elf or not Path(elf).is_file():
        return _unknown("the group's program did not build, so there is nothing to attribute")
    companion = record.get("debug_companion")
    if not isinstance(companion, Mapping):
        return _unknown("no debug companion was built for the program")
    if companion.get("refusal"):
        return _unknown(f"the debug companion did not build: {companion['refusal']}")
    debug_elf, debug_object = companion.get("elf"), companion.get("program_object")
    if not all(p and Path(p).is_file() for p in (debug_elf, program_object, debug_object)):
        return _unknown("the program or its debug companion names no object and image to compare")
    try:
        image = DC.verify_debug_companion(Path(elf).read_bytes(), Path(debug_elf).read_bytes())
        objects = DC.verify_debug_companion(
            Path(program_object).read_bytes(), Path(debug_object).read_bytes(), relocatable=True
        )
    except (OSError, ValueError) as exc:
        return _unknown(f"the debug companion is not the program: {exc}")
    admitted = {"image": image, "object": objects, "option": companion.get("option")}
    symbolizer = symbolizer_beside(companion.get("compiler"))
    if symbolizer is None:
        return _unknown(
            f"the target's toolchain has no {SYMBOLIZER} beside its compiler {companion.get('compiler')}",
            companion=admitted,
        )
    run = E.run_functional_model(machine, Path(elf), work / "pc_census")
    if run["returncode"] != 0:
        return _unknown(f"the functional model exited {run['returncode']}", companion=admitted)
    histogram = E.pc_histogram(Path(run["histogram"]).read_text(encoding="utf-8", errors="replace"))
    if not histogram:
        return _unknown("the functional model wrote no PC histogram", companion=admitted)
    try:
        attribution = DC.attribute_symbolized_pcs(histogram, symbolize(symbolizer, debug_elf, sorted(histogram)))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return _unknown(f"the PC histogram could not be attributed: {exc}", companion=admitted)
    lines = [
        {"file": file, "line": line, "executions": count}
        for (file, line), count in sorted(attribution["lines"].items(), key=lambda item: (-item[1], item[0]))
    ]
    document = {
        "status": "attributed",
        "symbolizer": str(symbolizer),
        "histogram": str(run["histogram"]),
        "companion": admitted,
        "total": attribution["total"],
        "functions": dict(sorted(attribution["functions"].items(), key=lambda item: (-item[1], item[0]))),
        "lines": lines,
        "scope": "functional-model instruction executions of the whole one-group program; not cycles",
    }
    (work / "source_attribution.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    document["file"] = str(work / "source_attribution.json")
    return document


# ------------------------------------------------------------------------------------------- cli


def inspect_group(source: str | Path, group: str, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve, rebuild, and (optionally) trace one group; the JSON the command prints."""
    from merlin.common.artifacts import cache_dir

    candidate = resolve_candidate(source, args)
    index = _group(group)
    if args.out is not None:
        work = Path(args.out)
        if work.exists() and any(work.iterdir()):
            raise InspectError(f"{work} is not empty; name a new directory")
        work.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        work = Path(tempfile.mkdtemp(prefix=f"{candidate['target']}-g{index}-", dir=cache_dir("group-inspect")))
    result = rebuild(candidate, index, work, debug_companion=bool(args.trace))
    record = result["record"]
    out: dict[str, Any] = {
        "candidate": candidate["source"],
        "candidate_kind": candidate["kind"],
        "target": candidate["target"],
        **result,
        "answered_by": record.get("on"),
        "cause": record.get("cause"),
        "why": record.get("why"),
        "refusal": record.get("refusal"),
        "program": {
            key: record.get(key) for key in ("interface", "command_buffer", "program_source", "elf", "memory_map")
        },
    }
    if args.stage:
        files = stage_files(result, args.stage)
        out["stage"] = {"name": args.stage, "files": files}
        if not files:
            out["stage"]["why"] = "no stage or product of this rebuild has that name; see trace and products"
    if args.trace:
        out["instruction_trace"] = instruction_trace(candidate, record.get("elf"), work, run_to=args.run_to)
        out["source_attribution"] = source_attribution(candidate, record, work)
    return out


def _print(result: Mapping[str, Any], *, lines: int) -> None:
    print(f"group {result['group']} of {result['candidate']} ({result['candidate_kind']}, target {result['target']})")
    print(
        f"  answered by: {result.get('answered_by')}"
        + (f" [{result.get('cause')}]: {result.get('why')}" if result.get("cause") else "")
    )
    if result.get("refusal"):
        print(f"  program not built: {result['refusal']}")
    print(f"  work:  {result['work']}")
    print(f"  trace: {result['trace']}")
    for key, path in (result.get("program") or {}).items():
        if path:
            print(f"  {key}: {path}")
    print("  products:")
    for rel in sorted(result.get("products") or {}):
        print(f"    {rel}")
    stage = result.get("stage")
    if stage:
        if not stage["files"]:
            print(f"  stage {stage['name']}: {stage['why']}")
        for path in stage["files"]:
            print(f"  ---- {stage['name']}: {path} (first {lines} lines)")
            with open(path, encoding="utf-8", errors="replace") as fh:
                for _, line in zip(range(lines), fh):
                    print(f"  {line.rstrip()}")
    trace = result.get("instruction_trace")
    if trace:
        if not trace.get("available"):
            print(f"  instruction trace: {trace['why']}")
        else:
            print(
                f"  instruction trace: {trace.get('log')} ({trace.get('log_lines')} lines; exit {trace.get('returncode')})"
            )
            if trace.get("error"):
                print(f"    {trace['error']}")
    attribution = result.get("source_attribution")
    if attribution:
        if attribution.get("status") != "attributed":
            print(f"  source attribution: {attribution['status']}: {attribution['why']}")
        else:
            total = f"{attribution['total']:,} executions"
            print(f"  source attribution: {attribution['file']} ({total}; {attribution['scope']})")
            for row in attribution["lines"][:lines]:
                print(f"    {row['executions']:>12,}  {row['file'] or '?'}:{row['line']}")


def run_from_args(args: argparse.Namespace) -> int:
    try:
        result = inspect_group(args.spec, args.group, args)
    except InspectError as exc:
        print(f"merlin experiment inspect: {exc}")
        return 2
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print(result, lines=args.lines)
    return 0 if not result.get("refusal") else 1


__all__: Sequence[str] = (
    "InspectError",
    "configure_parser",
    "functional_machine",
    "inspect_group",
    "instruction_trace",
    "rebuild",
    "resolve_candidate",
    "run_from_args",
    "source_attribution",
    "stage_files",
    "symbolize",
    "symbolizer_beside",
)
