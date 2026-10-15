"""Run replay images on the RTL-equivalent native simulator (gSIM) of the selected FIRRTL, beside Spike.

gSIM models one board: the board catalog's entry for the RTL configuration the selected RTL facts name.
Its DRAM span is far smaller than the span Spike is told to model, so a gSIM replay image is a *board
image*: code, ``.bss`` and stack at ``dram_base`` within the board's code reserve, then every operand,
then one poisoned result slot per call, all inside ``[dram_base, dram_base + dram_bytes)``. The image
states that span in its symbol table, so the SAME ELF runs on Spike (with exactly that memory) and on
gSIM, and the two engines' digests are compared call by call (``engine_agreement``).

The emulator is the one the backend binds: the RTL facts' FIRRTL must be the FIRRTL its build receipt
pins, and the citation and ELF are rechecked around every run. It is started with the backend's
command template plus the same-ELF DRAM preload (``+loadmem``) and a ``+max-cycles`` hang bound; the
cycles of interest are each call's own ``mcycle`` delta, which the preload does not enter. A run stopped
by its cycle or wall bound still reports every call it finished (the image flushes per call).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.model_check import replay as RP

#: Simulated cycles allowed for boot, console and exit around the calls of one image.
BOOT_CYCLES = 2_000_000
#: Simulated cycles allowed per result byte for the image's digest pass (measured well below this).
DIGEST_CYCLES_PER_BYTE = 4
_ALIGN = 1 << 20  # a result/operand region base needs no finer alignment
_PT_LOAD = 1  # ELF program header type of a loadable segment (ELF specification)
#: What Merlin's gSIM harness prints when the harness itself fails (chipyard_harness/main.cpp).
_HARNESS_ERROR = "GSIM harness error"
#: Run statuses that say nothing about the program: a bound stopped it or the engine failed.
UNFINISHED = frozenset({"timeout", "cycle_cap", "error"})


class EngineError(RP.ReplayError):
    """An engine or board binding that cannot be honored (an infrastructure failure, not a verdict)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------ board image


@dataclass(frozen=True)
class BoardImage:
    """Where a replay image lives on a board: code at ``dram_base``, operands then results above it."""

    name: str
    catalog: str
    catalog_sha256: str
    dram_base: int
    dram_bytes: int
    code_reserve: int

    @property
    def weights_base(self) -> int:
        return (self.dram_base + self.code_reserve + _ALIGN - 1) // _ALIGN * _ALIGN

    @property
    def capacity(self) -> int:
        """Bytes available to one image's operands and result slots."""
        return self.dram_base + self.dram_bytes - self.weights_base

    def defsyms(self) -> dict[str, int]:
        from merlin.runtime.backends import spike_model

        return {
            "MERLIN_WEIGHTS_BASE": self.weights_base,
            spike_model.DRAM_BASE_SYMBOL: self.dram_base,
            spike_model.DRAM_SPAN_SYMBOL: self.dram_bytes,
        }

    def record(self) -> dict[str, Any]:
        return {
            "board": self.name,
            "catalog": self.catalog,
            "catalog_sha256": self.catalog_sha256,
            "dram_base": hex(self.dram_base),
            "dram_bytes": self.dram_bytes,
            "code_reserve": self.code_reserve,
            "weights_base": hex(self.weights_base),
        }


def board_catalog_path(target: str, catalog: str | Path | None = None) -> Path:
    """``catalog``, else ``$MERLIN_BOARD_CATALOG``, else the target's ``examples/<target>/platform`` one."""
    from merlin.common.paths import repo_root
    from merlin.runtime import boards

    selected = catalog or os.environ.get(boards.BOARD_CATALOG_ENV)
    path = Path(selected) if selected else repo_root() / "examples" / target / "platform" / "board-catalog.yaml"
    if not path.is_file():
        raise EngineError(f"no board catalog at {path}")
    return path


def board_for(target: str, config: str, *, catalog: str | Path | None = None) -> BoardImage:
    """The one bare-metal HTIF board of ``target`` whose RTL simulation config is ``config``."""
    from merlin.runtime import boards

    path = board_catalog_path(target, catalog)
    found = [b for b in boards.load_boards(path).values() if b.target == target and b.rtl_sim_config == config]
    if len(found) != 1:
        raise EngineError(f"{path}: {len(found)} boards of {target} declare rtl_sim_config {config!r}; need one")
    board = found[0]
    if board.flow != boards.FLOW_BAREMETAL or board.console != boards.CONSOLE_HTIF or not board.code_reserve:
        raise EngineError(f"board {board.name} is not a bare-metal HTIF board with a code reserve")
    return BoardImage(
        name=board.name,
        catalog=str(path.resolve()),
        catalog_sha256=_sha256(path),
        dram_base=int(board.dram_base),
        dram_bytes=int(board.dram_bytes),
        code_reserve=int(board.code_reserve),
    )


def load_spans(elf: Path) -> list[tuple[int, int]]:
    """``[(start, end)]`` physical spans of the ELF64's loadable segments (memory size, not file size)."""
    data = Path(elf).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise EngineError(f"{elf}: not an ELF64 image")
    order = "<" if data[5] == 1 else ">"
    phoff = struct.unpack_from(order + "Q", data, 0x20)[0]
    phentsize, phnum = struct.unpack_from(order + "HH", data, 0x36)
    spans = []
    for index in range(phnum):
        p_type, _flags, _offset, _vaddr, paddr, _filesz, memsz = struct.unpack_from(
            order + "IIQQQQQ", data, phoff + index * phentsize
        )
        if p_type == _PT_LOAD and memsz:
            spans.append((paddr, paddr + memsz))
    return spans


def check_fits(elf: Path, board: BoardImage) -> dict[str, int]:
    """Fail closed unless the low segments end below the operand base and all of them are in DRAM."""
    spans = load_spans(elf)
    low = [end for start, end in spans if start < board.weights_base]
    high = [end for start, end in spans if start >= board.weights_base]
    top = board.dram_base + board.dram_bytes
    if not spans or min(start for start, _ in spans) < board.dram_base:
        raise EngineError(f"{elf.name}: a segment starts below the board's DRAM at {hex(board.dram_base)}")
    if low and max(low) > board.weights_base:
        raise EngineError(f"{elf.name}: code/data/stack end {hex(max(low))} overruns the code reserve")
    if max(end for _, end in spans) > top:
        raise EngineError(f"{elf.name}: image ends at {hex(max(end for _, end in spans))}, beyond DRAM {hex(top)}")
    return {"code_end": max(low) if low else board.dram_base, "image_end": max(high) if high else board.weights_base}


def case_bytes(case: RP.ReplayCase) -> tuple[int, int]:
    """``(operand bytes, result-slot bytes)`` a case adds to a board image (an upper bound)."""
    operands = sum(int(a.nbytes) for a in case.operands)
    results = sum((int(a.nbytes) + 63) // 64 * 64 for a in case.expected)
    return operands, results


def board_chunks(cases: Sequence[RP.ReplayCase], board: BoardImage) -> list[list[RP.ReplayCase]]:
    """One image per symbol (split further only when its calls overrun the board's operand region)."""
    by_symbol: dict[str, list[RP.ReplayCase]] = {}
    for case in cases:
        by_symbol.setdefault(case.symbol, []).append(case)
    chunks: list[list[RP.ReplayCase]] = []
    for group in by_symbol.values():
        current: list[RP.ReplayCase] = []
        size = 0
        for case in group:
            need = sum(case_bytes(case)) + 64 * (len(case.operands) + 1)
            if need > board.capacity:
                raise EngineError(
                    f"{case.label}: {need} operand/result bytes exceed the board's {board.capacity}-byte region"
                )
            if current and size + need > board.capacity:
                chunks.append(current)
                current, size = [], 0
            current.append(case)
            size += need
        if current:
            chunks.append(current)
    return chunks


def image_max_cycles(cases: Sequence[RP.ReplayCase], *, call_max_cycles: int) -> int:
    """The image's hang bound from a per-call bound: boot, then each call and its digest pass."""
    return BOOT_CYCLES + sum(call_max_cycles + DIGEST_CYCLES_PER_BYTE * case_bytes(c)[1] for c in cases)


# ------------------------------------------------------------------------------------------ gSIM engine


def _unlimited_stack() -> None:  # pragma: no cover - runs in the child process
    """A small stack makes the generated model warn onto the console; lift it for the child."""
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass


@dataclass
class EngineRun:
    engine: str
    #: completed | fault (the program trapped, exited non-zero or made the design assert: a verdict) |
    #: timeout | cycle_cap | error (the engine itself failed): the last three are :data:`UNFINISHED`.
    status: str
    console: str
    wall_s: float
    returncode: int | None
    cycles: int | None  # the engine's own count for the whole run (gSIM harness), when it prints one
    detail: str = ""
    argv: tuple[str, ...] = ()
    cpu_s: float | None = None  # the engine process's own CPU time, when it was reaped

    def record(self) -> dict[str, Any]:
        row = {"engine": self.engine, "status": self.status, "seconds": round(self.wall_s, 1)}
        if self.returncode is not None:
            row["returncode"] = self.returncode
        if self.cycles is not None:
            row["cycles"] = self.cycles
        if self.cpu_s is not None:
            row["cpu_seconds"] = round(self.cpu_s, 1)
            if self.cycles:
                row["cycles_per_cpu_second"] = round(self.cycles / max(self.cpu_s, 1e-3))
        if self.detail:
            row["detail"] = self.detail[-1500:]
        return row


class GsimEngine:
    """The receipt-bound gSIM emulator for the selected RTL facts, and the board it models."""

    name = "gsim"

    def __init__(self, target: str, rtl_facts: str | Path, *, board_catalog: str | Path | None = None) -> None:
        from merlin.compile import model_execution_inputs as MI

        facts_path = Path(rtl_facts)
        document = json.loads(facts_path.read_text())
        config = ((document.get("facts") or {}).get("source") or {}).get("config")
        if not isinstance(config, str) or not config:
            raise EngineError(f"{facts_path}: the RTL facts name no source config")
        self.target = target
        self.facts = MI.selected_firrtl(facts_path, target=target, config=config)
        self.board = board_for(target, config, catalog=board_catalog)
        backend, citation, revalidate, _prepare = MI.native_engine(target, "gsim", self.facts)
        self.backend = backend
        self.citation = citation
        self._revalidate = revalidate
        block = backend._engine("gsim")  # noqa: SLF001 -- the bound runner's own selected engine block
        binary = Path(block["binary"]["path"])
        if binary.resolve() != Path(citation["path"]).resolve() or block["binary"]["sha256"] != citation.get(
            "binary_sha256"
        ):
            raise EngineError("the backend's gSIM block names different bytes than the bound citation")
        self.binary = binary
        self.environment = dict(block.get("environment") or {})
        self.failure_markers = tuple(block.get("failure_markers") or ())
        template = list(backend.execution_argv_template("gsim"))
        caps = [a for a in template if a.startswith("+max-cycles=")]
        if "{elf}" not in template or len(caps) != 1 or not caps[0].removeprefix("+max-cycles=").isdigit():
            raise EngineError(f"the selected gSIM command template has no ELF and unique cycle bound: {template}")
        self.default_max_cycles = int(caps[0].removeprefix("+max-cycles="))
        if not any(a.startswith("+loadmem=") for a in template):
            template.append("+loadmem={elf}")  # the backend's own same-ELF DRAM preload transform
        self.template = tuple(template)

    def identity(self) -> dict[str, Any]:
        receipt = self.citation.get("receipt") or {}
        return {
            "engine": self.name,
            "binary": str(self.binary),
            "binary_sha256": self.citation.get("binary_sha256"),
            "receipt": receipt.get("receipt_path"),
            "receipt_sha256": receipt.get("receipt_sha256"),
            "firrtl_sha256": self.facts["firrtl_sha256"],
            "rtl_facts_sha256": self.facts["sha256"],
            "config": self.facts["config"],
            "command_template": list(self.template),
            "load": "same-ELF DRAM preload (+loadmem); call cycles are in-image mcycle deltas",
            "board": self.board.record(),
        }

    def argv(self, elf: Path, *, max_cycles: int) -> list[str]:
        if type(max_cycles) is not int or not 100 < max_cycles < 1 << 62:
            raise EngineError(f"gSIM max cycles must be a positive bounded integer, got {max_cycles!r}")
        out = []
        for arg in self.template:
            if arg.startswith("+max-cycles="):
                arg = f"+max-cycles={max_cycles}"
            out.append(arg.replace("{elf}", str(elf)))
        return [str(self.binary), *out]

    def run(self, elf: Path, *, max_cycles: int, timeout_s: float, workdir: Path) -> EngineRun:
        """Run ``elf`` to its exit, its cycle bound or ``timeout_s``; the console survives either bound."""
        from merlin.runtime.backends import chipyard_rocc
        from merlin.targetgen.rtl_engine_policy import gsim_runtime_slot

        elf = Path(elf).resolve()
        elf_sha = _sha256(elf)
        argv = self.argv(elf, max_cycles=max_cycles)
        self._revalidate()
        stdout_path, stderr_path = workdir / "gsim.stdout", workdir / "gsim.stderr"
        timed_out = False
        with gsim_runtime_slot(wait_timeout_s=timeout_s), stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            started = time.monotonic()
            proc = subprocess.Popen(  # noqa: S603 -- argv is the bound engine and our own ELF
                argv,
                cwd=workdir,
                env=self.environment,
                stdout=out,
                stderr=err,
                start_new_session=True,
                preexec_fn=_unlimited_stack,  # noqa: PLW1509 -- no threads touch the child before exec
            )
            cpu_s = _reap(proc, deadline=started + timeout_s)
            if cpu_s is None:
                timed_out = True
                _kill_group(proc)
            wall = time.monotonic() - started
        stdout = stdout_path.read_bytes().decode("latin-1")
        stderr = stderr_path.read_bytes().decode("latin-1")
        self._revalidate()
        if _sha256(elf) != elf_sha:
            raise EngineError(f"{elf}: the image changed while gSIM ran it")
        cycles = None
        for line in stderr.splitlines():
            if line.startswith("Cycles:") and line.split()[-1].isdigit():
                cycles = int(line.split()[-1])
        code = proc.returncode
        status, detail = "completed", ""
        asserted = chipyard_rocc._RTL_ASSERTION_MARKER  # noqa: SLF001 -- the design's own assertion text
        if timed_out:
            status, detail = "timeout", f"wall bound {timeout_s:.0f} s reached"
        elif code == chipyard_rocc._GSIM_CYCLE_CAP_EXIT:  # noqa: SLF001 -- the harness's cycle-cap status
            status, detail, cycles = "cycle_cap", f"+max-cycles={max_cycles} reached", max_cycles
        elif asserted in stdout or asserted in stderr:  # the program did something the design refuses
            status, detail = "fault", "the design asserted: " + stderr[-1200:]
        elif any(marker in stdout + stderr for marker in self.failure_markers):
            status, detail = "fault", "a declared failure marker was printed: " + stdout[-1200:]
        elif code is None or code < 0 or (code != 0 and _HARNESS_ERROR in stderr):
            status, detail = "error", f"the emulator failed (exit {code}): {stderr[-1200:]}"
        elif code != 0:
            status, detail = "fault", f"the program exited {code}: {(stdout + stderr)[-1200:]}"
        elif "MCDONE" not in stdout:
            status, detail = "error", f"the run ended without the image's completion line: {stdout[-1200:]}"
        return EngineRun("gsim", status, stdout, wall, code, cycles, detail, tuple(argv), cpu_s)


def _reap(proc: subprocess.Popen, *, deadline: float) -> float | None:
    """Wait for ``proc`` until ``deadline``; its CPU seconds when it exited, ``None`` at the deadline.

    The CPU time is the load-independent cost of the run: on a shared host the wall time is not.
    """
    while True:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            proc.returncode = os.waitstatus_to_exitcode(status)
            return usage.ru_utime + usage.ru_stime
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_spike(target: str, elf: Path, board: BoardImage, *, timeout_s: float) -> EngineRun:
    """The same board image on the functional model, given exactly the board's memory span."""
    started = time.monotonic()
    try:
        console, wall = RP.run_image(
            target, elf, dram_base=board.dram_base, dram_bytes=board.dram_bytes, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return EngineRun("spike", "timeout", "", time.monotonic() - started, None, None, f"wall bound {timeout_s} s")
    except RP.ReplayError as error:
        return EngineRun("spike", "error", "", time.monotonic() - started, None, None, str(error))
    return EngineRun("spike", "completed", console, wall, 0, None)


# ------------------------------------------------------------------------------------------ replay


@dataclass
class ImageJob:
    index: int
    directory: Path
    cases: list[RP.ReplayCase]
    plan: list[dict[str, Any]]
    elf: Path
    max_cycles: int
    layout: dict[str, int]


def build_images(
    cases: Sequence[RP.ReplayCase],
    kernels: Mapping[str, str],
    interfaces: Mapping[str, str],
    objects: Sequence[Path],
    toolchain: RP.Toolchain,
    board: BoardImage,
    work: Path,
    *,
    max_cycles: int | None,
    call_max_cycles: int | None,
    default_max_cycles: int,
) -> list[ImageJob]:
    """Board images for ``cases`` (one per symbol unless a symbol's calls overrun the board)."""
    jobs = []
    for index, chunk in enumerate(board_chunks(cases, board)):
        directory = work / f"board_image_{index}"
        directory.mkdir(parents=True, exist_ok=True)
        main_c, data_s, plan = RP._write_image_sources(  # noqa: SLF001
            directory, chunk, kernels, interfaces, preload_outputs=True
        )
        elf = RP.link_image(directory, main_c, data_s, objects, toolchain, defsyms=board.defsyms())
        layout = check_fits(elf, board)
        if max_cycles is not None:
            bound = max_cycles
        elif call_max_cycles is not None:
            bound = image_max_cycles(chunk, call_max_cycles=call_max_cycles)
        else:
            bound = default_max_cycles
        jobs.append(ImageJob(index, directory, list(chunk), plan, elf, bound, layout))
    return jobs


def np_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(d) for d in getattr(value, "shape", ()))


def _case_rows(job: ImageJob, gsim: EngineRun, spike: EngineRun, call_max_cycles: int | None) -> list[dict]:
    """Per call: the gSIM verdict (the judged one), the Spike verdict and whether the digests agree."""
    verdicts = RP.judge_console(gsim.console, job.plan)
    spike_verdicts = RP.judge_console(spike.console, job.plan)
    gsim_seen, _ = RP.console_records(gsim.console)
    spike_seen, _ = RP.console_records(spike.console)
    rows = []
    for case, entry, verdict, functional in zip(job.cases, job.plan, verdicts, spike_verdicts, strict=True):
        keys = [(entry["case"], oi) for oi in range(len(entry["outputs"]))]
        reported = all(k in gsim_seen for k in keys)
        if reported:
            status = "completed"
        elif gsim.status == "completed":
            status = "unreported"
        else:
            status = gsim.status
        row = {
            **verdict,
            "engine": "gsim",
            "image": job.index,
            "shapes": {role: list(np_shape(value)) for role, value in zip(case.roles, case.operands, strict=True)},
            "status": status,
            "spike": {"exact": functional["exact"], "instret_cycles": functional.get("cycles")},
            "engines_agree": all(k in gsim_seen and k in spike_seen and gsim_seen[k] == spike_seen[k] for k in keys)
            if reported
            else None,
        }
        if not reported and gsim.detail:
            row["stopped"] = gsim.detail[:300]
        if call_max_cycles is not None and verdict.get("cycles") is not None:
            row["over_call_budget"] = verdict["cycles"] > call_max_cycles
        rows.append(row)
    return rows


def replay_on_board(
    target: str,
    engine: GsimEngine,
    cases: Sequence[RP.ReplayCase],
    kernels: Mapping[str, str],
    interfaces: Mapping[str, str],
    objects: Sequence[Path],
    toolchain: RP.Toolchain,
    work: Path,
    *,
    timeout_s: float,
    jobs: int = 1,
    max_cycles: int | None = None,
    call_max_cycles: int | None = None,
    deadline: float | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build board images for ``cases``, run each on Spike then gSIM; ``(case rows, image records)``.

    ``jobs`` images run at once (each holds one machine-wide gSIM slot while its emulator runs). Every
    run is bounded by ``timeout_s`` and by ``deadline`` (``time.monotonic()``), whichever comes first; a
    run the bound stops, or one never started because the budget was spent, reports why.
    """
    images = build_images(
        cases,
        kernels,
        interfaces,
        objects,
        toolchain,
        engine.board,
        work,
        max_cycles=max_cycles,
        call_max_cycles=call_max_cycles,
        default_max_cycles=engine.default_max_cycles,
    )
    lock = threading.Lock()

    def remaining() -> float:
        return timeout_s if deadline is None else min(timeout_s, deadline - time.monotonic())

    def spent(name: str) -> EngineRun:
        return EngineRun(name, "timeout", "", 0.0, None, None, "the --timeout-s budget was spent before this run")

    def run(job: ImageJob) -> tuple[list[dict], dict]:
        budget = remaining()
        spike = run_spike(target, job.elf, engine.board, timeout_s=budget) if budget > 1 else spent("spike")
        budget = remaining()
        gsim = (
            engine.run(job.elf, max_cycles=job.max_cycles, timeout_s=budget, workdir=job.directory)
            if budget > 1
            else spent("gsim")
        )
        (job.directory / "spike.console").write_text(spike.console)
        rows = _case_rows(job, gsim, spike, call_max_cycles)
        record = {
            "image": job.index,
            "cases": len(job.cases),
            "symbols": sorted({c.symbol for c in job.cases}),
            "engine": "gsim",
            "elf": str(job.elf),
            "elf_sha256": _sha256(job.elf),
            "elf_bytes": job.elf.stat().st_size,
            "max_cycles": job.max_cycles,
            "layout": {k: hex(v) for k, v in job.layout.items()},
            "seconds": round(gsim.wall_s, 1),
            "gsim": gsim.record(),
            "spike": spike.record(),
        }
        if log is not None:
            with lock:
                log(
                    f"[gsim] image {job.index} {record['symbols']}: {gsim.status} in {gsim.wall_s:.0f} s, "
                    f"{gsim.cycles} cycles; calls "
                    + ", ".join(f"{r['label']}={r['status']}/{r['exact']}/{r.get('cycles')}" for r in rows)
                )
        return rows, record

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        for job_rows, record in pool.map(run, images):
            rows += job_rows
            records.append(record)
    return rows, records


def annotate_kernels(kernels: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]], *, cycles_kind: str) -> None:
    """Add each kernel's call cycles (``functional``: Spike's count; ``rtl``: gSIM's) and, for board
    replays, engine agreement and unfinished calls to its report row (in place)."""
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], []).append(row)
    for kernel in kernels:
        mine = by_symbol.get(kernel["symbol"])
        if not mine:
            continue
        cycles = [r["cycles"] for r in mine if r.get("cycles") is not None]
        if cycles:
            kernel["cycles"] = {"kind": cycles_kind, "min": min(cycles), "max": max(cycles), "calls": len(cycles)}
        if any("engines_agree" in r for r in mine):
            kernel["engines_agree"] = sum(bool(r.get("engines_agree")) for r in mine)
            kernel["spike_exact"] = sum(bool((r.get("spike") or {}).get("exact")) for r in mine)
        if mine[0].get("shapes"):
            kernel["shapes"] = mine[0]["shapes"]
        unfinished = [r for r in mine if r.get("status") in UNFINISHED]
        if unfinished:
            kernel["unfinished"] = [r["label"] for r in unfinished][:5]
            kernel["stopped"] = unfinished[0].get("stopped", unfinished[0]["status"])
            if kernel.get("status") == "fail":
                kernel["status"] = "unfinished"
        faults = [r for r in mine if r.get("status") == "fault"]
        if faults:
            kernel["fault"] = faults[0].get("stopped", "fault")
        if any(r.get("over_call_budget") for r in mine):
            kernel["over_call_budget"] = sum(bool(r.get("over_call_budget")) for r in mine)
