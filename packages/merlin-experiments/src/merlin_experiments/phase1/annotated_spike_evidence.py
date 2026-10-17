"""Evidence-only annotated-Spike event counts for a Phase 1 run's gSIM-labelled programs.

For every training-eligible public capsule execution of a finished run, this host-side pass runs the
exact program gSIM measured on the observer-enabled Spike with an explicitly selected timing provider and
a count-only model (every price unknown), scoped to the caller's own kernel-call counter window. The
report's per-class event counts (commands, bytes, MACs, primitive classes, CPU instruction classes) are
the features a tuned Spike model is fitted on. Each program is also run on the run's own functional Spike
configuration, and the two consoles must be byte-identical, so observation never changes a functional
result. Where the run kept the functional tier's own program, the rerun must also reproduce its retained
console.

This is AUXILIARY evidence. It never grades, never runs during authoring, leases shared compute at the
lowest priority when a pool is selected, and writes only under its destination. The counter window is
derived from the linked ELF by disassembly with an explicitly selected tool, matched against the
caller's reviewed counter instruction text and the target contract's entry/completion symbols; nothing
here knows an opcode, a register field or a target name.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path, PurePosixPath

CONFIG_SCHEMA = "merlin.annotated_spike_evidence_config.v1"
REVIEW_SCHEMA = "merlin.counter_window_review.v1"
EVIDENCE_SCHEMA = "merlin.annotated_spike_evidence.v1"
MODEL_SCHEMA = "merlin.annotated_spike_model.v1"
EVIDENCE_DIR = "annotated_spike"
#: The program the RTL label measured, and the functional tier's own program when the run kept it.
PROGRAM_ROLE = "gsim_program"
FUNCTIONAL_PROGRAM_ROLE = "spike_program"
_TIMEOUT_S = 600


class EvidenceError(ValueError):
    """The selected inputs cannot establish an exact, scoped observation."""


def canonical_digest(value) -> str:
    """The timing contracts' canonical digest: SHA256 of sorted-key, compact, finite JSON."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"selected evidence input is not an ordinary file: {path}")
    return {"path": str(path), "sha256": _file_sha256(path)}


# ---------------------------------------------------------------------------------------------------
# Counter window: which two executed counter reads bracket the kernel call, read from the linked ELF
# ---------------------------------------------------------------------------------------------------


def _instruction_form(text: str) -> tuple[str, tuple[str, ...]]:
    """``mnemonic`` and comma-separated operands, whitespace-normalized (a disassembler's or a template's)."""
    mnemonic, _, operands = text.strip().partition(" ")
    if "\t" in mnemonic:
        mnemonic, _, rest = mnemonic.partition("\t")
        operands = rest + " " + operands
    fields = tuple(field.strip() for field in operands.replace("\t", " ").split(",") if field.strip())
    return mnemonic.strip(), fields


def _symbols(comment: str) -> set[str]:
    """Exact ``<symbol>`` references; an offset reference ``<symbol+0x4>`` is not a call to ``symbol``."""
    found = set()
    for piece in comment.split("<")[1:]:
        name, closed, _ = piece.partition(">")
        if closed and name and "+" not in name and "-" not in name:
            found.add(name)
    return found


def parse_disassembly(text: str) -> list[dict]:
    """Instructions of a ``-d --no-show-raw-insn`` listing, each with its address and enclosing function."""
    rows, function = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, _, rest = stripped.partition(" ")
        if rest.startswith("<") and rest.endswith(">:"):
            try:
                int(head, 16)
            except ValueError:
                continue
            function = rest[1:-2]
            continue
        address, colon, body = stripped.partition(":")
        if not colon or not body.strip():
            continue
        try:
            pc = int(address, 16)
        except ValueError:
            continue
        instruction, _, comment = body.partition("#")
        mnemonic, operands = _instruction_form(instruction)
        if mnemonic:
            rows.append(
                {
                    "pc": pc,
                    "function": function,
                    "mnemonic": mnemonic,
                    "operands": operands,
                    "symbols": sorted(_symbols(comment) | _symbols(instruction)),
                }
            )
    return rows


def counter_window(rows: list[dict], *, counter_asm: str, entry_symbol: str, completion_symbol: str) -> dict:
    """The unique counter-read pair bracketing the caller's entry and completion calls, or refuse.

    ``counter_asm`` is the reviewed instruction template with its output operand spelled ``%0``; an
    instruction matches when its mnemonic and every other operand equal the template's. The caller's
    function must call ``entry_symbol`` and then ``completion_symbol``, read the counter exactly twice,
    and place one read before the entry call and the other after the completion call."""
    pattern = _instruction_form(counter_asm)
    if not pattern[0] or pattern[1].count("%0") != 1:
        raise EvidenceError("counter instruction template must name one %0 output operand")

    def counter(row):
        return (
            row["mnemonic"] == pattern[0]
            and len(row["operands"]) == len(pattern[1])
            and all(want == "%0" or have == want for have, want in zip(row["operands"], pattern[1], strict=True))
        )

    callers = sorted(
        {row["function"] for row in rows if entry_symbol in row["symbols"]}
        & {row["function"] for row in rows if completion_symbol in row["symbols"]}
    )
    if len(callers) != 1 or callers[0] is None:
        raise EvidenceError(f"expected one function calling both {entry_symbol} and {completion_symbol}: {callers}")
    body = [row for row in rows if row["function"] == callers[0]]
    reads = [row for row in body if counter(row)]
    entry = min(row["pc"] for row in body if entry_symbol in row["symbols"])
    completion = max(row["pc"] for row in body if completion_symbol in row["symbols"])
    if len(reads) != 2 or not reads[0]["pc"] < entry < completion < reads[1]["pc"]:
        raise EvidenceError(
            f"{callers[0]} does not read the counter exactly once before {entry_symbol} and once after "
            f"{completion_symbol}: {[hex(row['pc']) for row in reads]}"
        )
    return {
        "caller": callers[0],
        "begin_pc": reads[0]["pc"],
        "end_pc": reads[1]["pc"],
        # The caller runs once per program, so its window is the first (zero-based) observed.
        "occurrence": 0,
        "entry_call_pc": entry,
        "completion_call_pc": completion,
        "counter_reads": [
            {"pc": row["pc"], "mnemonic": row["mnemonic"], "operands": list(row["operands"])} for row in reads
        ],
        "program_counter_reads": sum(counter(row) for row in rows),
    }


def review_elf(elf: Path, *, objdump: Path, counter_asm: str, entry_symbol: str, completion_symbol: str) -> dict:
    """Disassemble ``elf`` with the selected tool and record the reviewed window with every identity."""
    listing = subprocess.run(
        [str(objdump), "-d", "--no-show-raw-insn", str(elf)],
        capture_output=True,
        text=True,
        check=True,
        timeout=_TIMEOUT_S,
    ).stdout
    window = counter_window(
        parse_disassembly(listing),
        counter_asm=counter_asm,
        entry_symbol=entry_symbol,
        completion_symbol=completion_symbol,
    )
    return {
        "schema": REVIEW_SCHEMA,
        "elf_sha256": _file_sha256(elf),
        "disassembler": _pin(objdump),
        "disassembly_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "counter_asm": counter_asm,
        "entry_symbol": entry_symbol,
        "completion_symbol": completion_symbol,
        **window,
        "scope": "static disassembly of the linked program; the observer delimits windows on executed reads",
    }


# ---------------------------------------------------------------------------------------------------
# Selected configuration: the run's own functional Spike invocation plus the observer selections
# ---------------------------------------------------------------------------------------------------


def functional_invocation(run_dir: Path, records: list[dict]) -> dict:
    """The functional tier's exact argv (without its program), from the run's own process records.

    The functional engine is the binary the L2 tier records name; its retained invocation shows every
    flag it was launched with. All of a run's L2 launches must agree, or this refuses."""
    engines = {((row.get("executions") or {}).get("L2") or {}).get("engine_sha256") for row in records} - {None}
    if len(engines) != 1:
        raise EvidenceError(f"run has no single functional engine identity: {sorted(engines)}")
    engine = next(iter(engines))
    templates = {}
    for row in records:
        generated = (run_dir / row["result"]).parent / "generated"
        for path in sorted((generated / "invocations").glob("*/invocation.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            argv = record.get("argv") or []
            if (record.get("executable") or {}).get("sha256") != engine or len(argv) < 2:
                continue
            # The program argument is this capsule's own build output (matched on its trailing
            # components, so a moved run directory still identifies its launches).
            here = (generated.parent.name, generated.name)
            program = [i for i, value in enumerate(argv) if PurePosixPath(value).parts[-3:-1] == here]
            if program != [len(argv) - 1]:
                continue
            templates[tuple(argv[:-1])] = record
        if len(templates) > 1:
            raise EvidenceError("functional launches in this run disagree on their arguments")
    if not templates:
        raise EvidenceError("no retained functional launch identifies the functional configuration")
    ((argv, record),) = templates.items()
    return {"argv": list(argv), "executable_sha256": engine, "cwd": record.get("cwd")}


def _extlibs(argv: list[str]) -> list[str]:
    return [value.partition("=")[2] for value in argv if value.startswith("--extlib=")]


def count_only_model(
    *,
    device_profile: dict,
    device_sha256: str,
    feature_engine_sha256: str,
    feature_schema_sha256: str,
    label_engine_sha256: str,
    scope_sha256: str,
    libraries: list[str],
    isa: str,
) -> dict:
    """An observer model with every price unknown: it counts events and estimates no cycles.

    ``device_profile`` is the simulator repository's exported profile, optionally with RTL structural
    facts already placed by role (``spike-timing apply-capacities``). Its hardware facts configure the
    provider, its ``structure`` and ``constraints`` ride along for providers and fitting, and every
    resource whose lanes and start queue are both known becomes an engine resource. A resource with
    either unknown stays unbound and is reported unresolved -- never given an invented default. The
    generic engine needs at least one resource, so a profile with none known gets a named placeholder
    that nothing binds (the CPU issue and accelerator resources are then reported unspecified)."""
    known = [
        {"name": name, "capacity": slot["capacity"], "queue_capacity": slot["queue_capacity"]}
        for name, slot in sorted((device_profile.get("resources") or {}).items())
        if isinstance(slot, dict)
        and type(slot.get("capacity")) is int
        and slot["capacity"] > 0
        and type(slot.get("queue_capacity")) is int
        and slot["queue_capacity"] >= 0
    ]
    resources = [{"id": index, **row} for index, row in enumerate(known)] or [
        {"id": 0, "name": "unbound.placeholder", "capacity": 1, "queue_capacity": 0}
    ]
    target = device_profile["target"]
    return {
        "schema": MODEL_SCHEMA,
        "profile": "phase1-public",
        "domain": {
            "device_sha256": device_sha256,
            "feature_engine_sha256": feature_engine_sha256,
            "feature_schema_sha256": feature_schema_sha256,
            "label_engine": "gsim",
            "label_engine_sha256": label_engine_sha256,
            "memory_regime_sha256": canonical_digest({"cache_warmth": "unknown", "protocol": None}),
            "scope_sha256": scope_sha256,
        },
        "scope": {"kind": "counter_pc_pair"},
        "libraries": libraries,
        "cpu": {"isa": isa},
        "resources": resources,
        "parameters": {},
        "target": {
            "hardware": dict(target["hardware"]),
            "prices": {},
            **{key: target[key] for key in ("structure", "constraints") if isinstance(target.get(key), dict)},
        },
        **({"capacity_facts": device_profile["capacity_facts"]} if "capacity_facts" in device_profile else {}),
    }


def kernel_call_scope(selection: dict) -> str:
    """The timing contracts' kernel-call scope digest for one reviewed counter selection."""
    return canonical_digest(
        {
            "schema": "merlin.kernel_call_scope.v1",
            "cycle_scope": "entry-through-completion",
            "timing_selection_sha256": selection["selection_sha256"],
            "selection_record_sha256": canonical_digest(selection),
        }
    )


def embedded_input(harness_sha256: str) -> str:
    return canonical_digest({"schema": "merlin.embedded_input_context.v1", "harness_sha256": harness_sha256})


def _run(argv: list[str], *, cwd: str | None) -> dict:
    """One Spike process at auxiliary priority on a leased shared CPU when a pool is selected."""
    from merlin.common.cpu_pool import AUXILIARY, cpu_priority, shared_compute_pool

    with cpu_priority(AUXILIARY), shared_compute_pool():
        started = time.monotonic()
        result = subprocess.run(argv, capture_output=True, timeout=_TIMEOUT_S, cwd=cwd)
        seconds = time.monotonic() - started
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "seconds": seconds}


def observe(
    elf: Path,
    *,
    harness_sha256: str,
    selection: dict,
    config: dict,
    functional: dict,
    model_path: Path,
    model_scope: str,
    out: Path,
    retained_console: Path | None = None,
) -> dict:
    """Observe one program: count-only annotated run vs the run's functional configuration."""
    out.mkdir(parents=True, exist_ok=True)
    review = review_elf(
        elf,
        objdump=Path(config["objdump"]),
        counter_asm=selection["counter_asm"],
        entry_symbol=config["entry_symbol"],
        completion_symbol=config["completion_symbol"],
    )
    if review["elf_sha256"] != _file_sha256(elf):
        raise EvidenceError("program changed during its counter review")
    input_sha = embedded_input(harness_sha256)
    binding = {
        "schema": "merlin.spike_scope_binding.v1",
        "elf_sha256": review["elf_sha256"],
        "input_sha256": input_sha,
        "scope_sha256": model_scope,
        "begin_pc": review["begin_pc"],
        "end_pc": review["end_pc"],
        "occurrence": review["occurrence"],
        "evidence_sha256": canonical_digest(review),
    }
    (out / "counter_window_review.json").write_text(json.dumps(review, sort_keys=True, indent=1) + "\n")
    (out / "scope_binding.json").write_text(json.dumps(binding, sort_keys=True) + "\n")
    report = out / "report.json"
    report.unlink(missing_ok=True)
    Path(str(report) + ".partial").unlink(missing_ok=True)  # an interrupted earlier report never blocks this one
    plain = _run([*functional["argv"], str(elf)], cwd=functional.get("cwd"))
    annotated_argv = [config["annotated_spike"], *functional["argv"][1:]]
    last_extlib = max(i for i, value in enumerate(annotated_argv) if value.startswith("--extlib="))
    annotated_argv.insert(last_extlib + 1, "--extlib=" + config["timing_provider"])
    annotated_argv += [
        "--timing-model=" + str(model_path),
        "--timing-report=" + str(report),
        "--timing-scope-binding=" + str(out / "scope_binding.json"),
        "--timing-input-sha256=" + input_sha,
        str(elf),
    ]
    annotated = _run(annotated_argv, cwd=functional.get("cwd"))
    (out / "functional_console.log").write_bytes(plain["stdout"])
    (out / "annotated_console.log").write_bytes(annotated["stdout"])
    retained = retained_console.read_bytes() if retained_console is not None else None
    document = json.loads(report.read_text()) if report.is_file() else None
    windows = (document or {}).get("counter_windows") or []
    return {
        "elf_sha256": review["elf_sha256"],
        "input_sha256": input_sha,
        "begin_pc": review["begin_pc"],
        "end_pc": review["end_pc"],
        "functional_identical": plain["stdout"] == annotated["stdout"]
        and plain["returncode"] == annotated["returncode"],
        "functional_returncode": plain["returncode"],
        "annotated_returncode": annotated["returncode"],
        "retained_console_reproduced": None if retained is None else retained == plain["stdout"] == annotated["stdout"],
        "seconds": {"functional": round(plain["seconds"], 3), "annotated": round(annotated["seconds"], 3)},
        "report": {"path": str(report), "sha256": _file_sha256(report)} if document is not None else None,
        "report_error": None if document is not None else annotated["stderr"][-2000:].decode(errors="replace"),
        "window_instructions": windows[review["occurrence"]]["instructions"]
        if len(windows) > review["occurrence"]
        else None,
        "feature_counts": (document or {}).get("feature_counts"),
        "unresolved": (document or {}).get("unresolved"),
        "timing_complete": (document or {}).get("timing_complete"),
        "functional_status": (document or {}).get("functional_status"),
    }


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema",
        "annotated_spike",
        "annotated_source_manifest",
        "timing_provider",
        "provider_bundle_manifest",
        "device_profile",
        "objdump",
        "contract",
    }
    if not isinstance(config, dict) or set(config) != required or config["schema"] != CONFIG_SCHEMA:
        raise EvidenceError(f"evidence configuration requires exactly {sorted(required)}")
    import yaml

    contract = yaml.safe_load(Path(config["contract"]).read_text(encoding="utf-8")) or {}
    abi = contract.get("harness_abi") or {}
    if not isinstance(abi.get("entry_symbol"), str) or not isinstance(abi.get("fence_symbol"), str):
        raise EvidenceError("target contract declares no harness entry and completion symbols")
    counter = (contract.get("logical_harness") or {}).get("cycle_counter_asm")
    return {
        **config,
        "entry_symbol": abi["entry_symbol"],
        "completion_symbol": abi["fence_symbol"],
        "contract_counter_asm": counter,
    }


def collect(run_dir: Path, config_path: Path, destination: Path, *, jobs: int = 1, capsules=None) -> dict:
    """Observe every training-eligible public gSIM program of ``run_dir`` (and retained Spike programs)."""
    from . import capsule_timing_archive as archive

    run_dir = Path(run_dir).resolve()
    config = load_config(config_path)
    index = archive.audit(run_dir)
    rows = [row for row in index["records"] if row["visibility"] == "public"]
    functional = functional_invocation(run_dir, rows)
    if capsules:
        rows = [row for row in rows if row["capsule"] in set(capsules)]
    source = json.loads(Path(config["annotated_source_manifest"]).read_text(encoding="utf-8"))
    device = json.loads(Path(config["device_profile"]).read_text(encoding="utf-8"))
    libraries = [*_extlibs(functional["argv"]), config["timing_provider"]]
    selections = {
        canonical_digest(row["timing_selection"]): row["timing_selection"] for row in rows if row["training_eligible"]
    }
    if len(selections) != 1:
        raise EvidenceError(f"run's eligible records carry {len(selections)} distinct counter selections")
    (selection,) = selections.values()
    if config["contract_counter_asm"] not in (None, selection.get("counter_asm")):
        raise EvidenceError("caller counter instruction differs from the target contract's")
    engines = {row["sim_provenance"]["sha256"] for row in rows if row["training_eligible"]}
    if len(engines) != 1:
        raise EvidenceError("eligible records name more than one gSIM binary")
    scope = kernel_call_scope(selection)
    isa = next((value.partition("=")[2] for value in functional["argv"] if value.startswith("--isa=")), "")
    # Spike runs in the functional tier's own working directory, so every path it is given is absolute.
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model = count_only_model(
        device_profile=device,
        device_sha256=_file_sha256(Path(config["device_profile"])),
        feature_engine_sha256=source["sha256"],
        feature_schema_sha256=_file_sha256(Path(config["provider_bundle_manifest"])),
        label_engine_sha256=next(iter(engines)),
        scope_sha256=scope,
        libraries=[_file_sha256(Path(library)) for library in libraries],
        isa=isa,
    )
    model_path = destination / "count_only_model.json"
    model_path.write_text(json.dumps(model, sort_keys=True, indent=1) + "\n")
    tasks = []
    for row in rows:
        capsule = (run_dir / row["result"]).parent
        target = destination / capsule.relative_to(run_dir) / EVIDENCE_DIR
        spike = (row["executions"] or {}).get("L2") or {}
        if row["training_eligible"]:
            tasks.append((PROGRAM_ROLE, row, run_dir / row["elf"], row["harness_sha256"], target, None))
        retained = row["artifacts"].get("L2_executed_elf")
        if retained:
            caller = row["artifacts"].get("L2_executed_harness") or {}
            tasks.append(
                (
                    FUNCTIONAL_PROGRAM_ROLE,
                    row,
                    run_dir / retained["path"],
                    caller.get("sha256"),
                    target,
                    row["artifacts"].get("functional_console"),
                )
            )
        elif spike.get("elf_sha256") == row["elf_sha256"] and row["artifacts"].get("functional_console"):
            # The run left the functional tier's own program in place: rerun it against its own console.
            tasks.append(
                (
                    FUNCTIONAL_PROGRAM_ROLE,
                    row,
                    run_dir / row["elf"],
                    row["harness_sha256"],
                    target,
                    row["artifacts"]["functional_console"],
                )
            )

    def work(task):
        role, row, elf, harness_sha, target, console = task
        try:
            observed = observe(
                elf,
                harness_sha256=harness_sha,
                selection=row["timing_selection"] or selection,
                config=config,
                functional=functional,
                model_path=model_path,
                model_scope=scope,
                out=target / role,
                retained_console=run_dir / console["path"] if console else None,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            observed = {"error": f"{type(exc).__name__}: {exc}"}
        return {"role": role, "result": row["result"], "capsule": row["capsule"], **observed}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        observations = list(pool.map(work, tasks))
    summary = {
        "schema": EVIDENCE_SCHEMA,
        "run_dir": str(run_dir),
        "scope": "evidence only: event counts under a count-only model; no grade, price or cycle estimate",
        "functional_invocation": functional,
        "annotated_spike": _pin(Path(config["annotated_spike"])),
        "timing_provider": _pin(Path(config["timing_provider"])),
        "feature_engine_sha256": source["sha256"],
        "model": {"path": str(model_path), "sha256": _file_sha256(model_path)},
        "kernel_call_scope_sha256": scope,
        "observations": observations,
        "counts": {
            "programs": len(observations),
            "errors": sum("error" in row for row in observations),
            "functional_identical": sum(row.get("functional_identical") is True for row in observations),
            "retained_consoles_reproduced": sum(row.get("retained_console_reproduced") is True for row in observations),
            "retained_consoles_checked": sum(
                row.get("retained_console_reproduced") is not None for row in observations
            ),
            "reports": sum(bool(row.get("report")) for row in observations),
        },
    }
    (destination / "annotated_spike_evidence.json").write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True, help=f"{CONFIG_SCHEMA} selections")
    parser.add_argument("--out", type=Path, help="destination (default: the run directory's capsules)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--capsule", action="append", help="observe only these capsules (repeatable)")
    args = parser.parse_args(argv)
    try:
        summary = collect(args.run_dir, args.config, args.out or args.run_dir, jobs=args.jobs, capsules=args.capsule)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"annotated spike evidence: {exc}\n")
    print(json.dumps(summary["counts"], sort_keys=True))
    counts = summary["counts"]
    return 0 if counts["programs"] and counts["functional_identical"] == counts["programs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
