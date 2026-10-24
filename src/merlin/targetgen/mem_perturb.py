"""A memory-order-perturbing variant of an elaborated-RTL simulator, and a seed sweep over it.

WHY THIS EXISTS. Every cheap engine this repo certifies on answers memory in ISSUE order. Spike has no
memory system; the chipyard RTL simulators (Verilator and GSIM alike) bind the SoC's AXI port to
testchipip's ``mm_magic_t``, which queues each read and returns it the next cycle, in the order it was
accepted, forever. A program that is correct only while two independent requests complete in the order
they were issued therefore passes every tier, and fails on a board whose memory controller does not keep
that order. Measured: a residual add whose overwriting accumulator load and accumulating load are not
ordered by completion passed L2 and L3 and then failed on FireSim with exactly the first operand's value
in the failing elements, differently from run to run.

WHAT IT CHANGES. Exactly one thing, and only when asked: ``mm_perturb_t``
(``merlin/contract/external/sim_memory/``) lets read and write responses with DIFFERENT AXI ids complete
out of order, each after a seeded random latency. Everything AXI4 guarantees is kept (same-id order,
contiguous bursts); read data is sampled at acceptance and writes land at their W beat, as in the stock
model. The model is chosen at run time by a plusarg -- without ``+mem_perturb_seed`` the variant builds
the stock ``mm_magic_t`` -- so the same binary gives the in-order answer and N reordered ones.

HOW IT IS BUILT, target-agnostically. Nothing here names a target, a design or a harness file. A
chipyard Verilator build leaves its objects and its makefile behind; the makefile's own dry run
(``make -n``) states the exact compile line of every harness source and the exact link line. The harness
source that constructs the memory model is FOUND by content (the one user source that builds an
``mm_magic_t``), patched structurally to build through ``mm_perturb_make`` instead, compiled with its own
recorded flags, and linked in place of the original object -- into a NEW directory. The source build is
never written to. A control relink of the UNMODIFIED objects is compared byte-for-byte to the declared
simulator, so the variant's receipt can say whether its only difference from the certified engine is the
memory model.

WHAT A VERDICT MEANS. ``sweep`` runs one ELF under the in-order model and under N seeds and reports
``order_sensitive`` when any seed's judged outcome differs from the in-order one's. That is a statement
about the PROGRAM under the ordering freedom the bus protocol grants; it is not a statement that any
particular chip will reorder those two requests. A clean sweep is evidence, not proof: N seeds sample
the schedule space, they do not cover it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: The stock memory model's constructor, as it appears in a harness source. The source that contains it
#: is the one this module patches; zero or several matches is a refusal, not a guess.
MAGIC_CTOR = "new mm_magic_t("
#: The factory the patched source calls instead (declared in mm_perturb.h).
FACTORY = "mm_perturb_make"
MODEL_HEADER = "mm_perturb.h"
MODEL_SOURCE = "mm_perturb.cc"
RECEIPT_NAME = "build_receipt.json"
BINARY_NAME = "emulator"
SCHEMA = "merlin.mem-perturb-engine.v1"
#: The environment spelling of the knobs (the model reads MERLIN_MEM_PERTURB_<KNOB> when no plusarg names it).
ENV_PREFIX = "MERLIN_MEM_PERTURB_"


class PerturbBuildError(RuntimeError):
    """The variant cannot be built as specified. Raised rather than approximated: a variant that
    silently kept the stock memory model would certify nothing while reporting that it had."""


def model_sources_dir() -> Path:
    """The model's C++ sources, resolved through the contract directory like every contract reader."""
    from merlin.common.paths import contract_dir

    return contract_dir() / "external" / "sim_memory"


def _model_sources() -> Path:
    """The model sources, or a refusal naming the one that is absent (an installed wheel does not ship
    them; building a variant needs the source checkout and the simulator's own build tree anyway)."""
    sources = model_sources_dir()
    for name in (MODEL_HEADER, MODEL_SOURCE):
        if not (sources / name).is_file():
            raise PerturbBuildError(f"the memory model source {sources / name} is absent")
    return sources


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise PerturbBuildError("unbalanced parentheses after the memory-model constructor")


def patch_harness_source(text: str, argc_expr: str, argv_expr: str) -> str:
    """Rewrite the ONE ``new mm_magic_t(<args>)`` in ``text`` into ``mm_perturb_make(<cfg>, <args>)``.

    The constructor's own argument list is carried over verbatim, so whatever base/size/word/line the
    harness computes is what the perturbing model gets. ``argc_expr``/``argv_expr`` name the plusargs as
    the harness sees them. Raises unless there is exactly one constructor to rewrite.
    """
    count = text.count(MAGIC_CTOR)
    if count != 1:
        raise PerturbBuildError(f"expected exactly one {MAGIC_CTOR!r} in the harness source, found {count}")
    start = text.index(MAGIC_CTOR)
    open_idx = start + len(MAGIC_CTOR) - 1
    close_idx = _matching_paren(text, open_idx)
    args = text[open_idx + 1 : close_idx]
    cfg = f"mm_perturb_parse({argc_expr}, (const char *const *)({argv_expr}))"
    patched = text[:start] + f"{FACTORY}({cfg}, {args})" + text[close_idx + 1 :]
    include = f'#include "{MODEL_HEADER}"\n'
    return include + patched


# ---------------------------------------------------------------------------------------------------
# the Verilator build plan, read from the build's own makefile
# ---------------------------------------------------------------------------------------------------


@dataclass
class LinkPlan:
    obj_dir: Path
    makefile: Path
    compile_lines: dict[str, list[str]]  # object name -> argv that builds it
    link_line: list[str]
    output: Path  # the simulator the makefile builds
    user_sources: dict[str, Path] = field(default_factory=dict)  # object name -> source path


def _dry_run(obj_dir: Path, makefile: str, touched: Path) -> list[list[str]]:
    proc = subprocess.run(
        ["make", "-n", "-C", str(obj_dir), "-f", makefile, "-W", str(touched)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise PerturbBuildError(f"make -n failed ({proc.returncode}): {proc.stderr.strip()[:400]}")
    lines = []
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("make:") or ln.startswith("make["):
            continue
        lines.append(shlex.split(ln))
    return lines


def _out_of(argv: Sequence[str]) -> str | None:
    for i, tok in enumerate(argv[:-1]):
        if tok == "-o":
            return argv[i + 1]
    return None


def link_plan(obj_dir: str | Path, makefile: str) -> LinkPlan:
    """What the build's makefile would run to rebuild every harness object and relink the simulator.

    Obtained from a DRY RUN with every harness source marked new, so nothing in ``obj_dir`` is written.
    """
    obj_dir = Path(obj_dir)
    mk = obj_dir / makefile
    if not mk.is_file():
        raise PerturbBuildError(f"no makefile at {mk}")
    # Mark every user source new by naming one; make then lists the compile of each object that
    # depends on it plus the link. One dry run per source keeps the attribution exact.
    sources = _user_sources(mk)
    compile_lines: dict[str, list[str]] = {}
    user_sources: dict[str, Path] = {}
    link_line: list[str] | None = None
    for src in sources:
        for argv in _dry_run(obj_dir, makefile, src):
            out = _out_of(argv)
            if out is None:
                continue
            if "-c" in argv:
                compile_lines[out] = argv
                if str(src) in argv:
                    user_sources[out] = src
            else:
                link_line = argv
    if link_line is None:
        raise PerturbBuildError(f"{mk}: the dry run names no link step")
    out = _out_of(link_line)
    return LinkPlan(obj_dir, mk, compile_lines, link_line, Path(out), user_sources)


def _mk_var(text: str, name: str) -> list[str]:
    """The whitespace-separated values of a ``NAME = \\`` continued makefile variable."""
    vals: list[str] = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        head, sep, rest = ln.partition("=")
        if sep and head.strip() == name:
            chunk = [rest]
            j = i
            while chunk[-1].rstrip().endswith("\\") and j + 1 < len(lines):
                j += 1
                chunk.append(lines[j])
            for c in chunk:
                vals.extend(t for t in c.replace("\\", " ").split() if t)
            return vals
    return vals


def _user_sources(mk: Path) -> list[Path]:
    text = mk.read_text(encoding="utf-8", errors="replace")
    classes = _mk_var(text, "VM_USER_CLASSES")
    dirs = [Path(d) for d in _mk_var(text, "VM_USER_DIR")]
    out: list[Path] = []
    for cls in classes:
        for d in dirs:
            for ext in (".cc", ".cpp", ".c"):
                p = d / f"{cls}{ext}"
                if p.is_file():
                    out.append(p)
                    break
    if not out:
        raise PerturbBuildError(f"{mk}: no user harness sources found (VM_USER_CLASSES/VM_USER_DIR)")
    return out


def memory_harness(plan: LinkPlan) -> tuple[str, Path]:
    """(object name, source) of the ONE harness source that constructs the stock memory model."""
    hits = [(obj, src) for obj, src in plan.user_sources.items() if MAGIC_CTOR in src.read_text(errors="replace")]
    if len(hits) != 1:
        raise PerturbBuildError(
            f"expected one harness source constructing {MAGIC_CTOR!r}, found {[str(s) for _, s in hits]}"
        )
    return hits[0]


def _relocate(argv: Sequence[str], obj_dir: Path, out_dir: Path, replace: dict[str, Path]) -> list[str]:
    """Rewrite a makefile command so every relative object resolves in ``obj_dir`` (read-only) and every
    output lands in ``out_dir``. ``replace`` substitutes whole tokens (an object we rebuilt)."""
    res: list[str] = []
    for i, tok in enumerate(argv):
        prev = argv[i - 1] if i else ""
        if tok in replace:
            res.append(str(replace[tok]))
        elif prev == "-o":
            res.append(str(out_dir / Path(tok).name))
        elif not tok.startswith("-") and not Path(tok).is_absolute() and (obj_dir / tok).exists():
            res.append(str(obj_dir / tok))
        elif tok.startswith("-I") and not Path(tok[2:]).is_absolute():
            res.append("-I" + str(obj_dir / tok[2:]))
        else:
            res.append(tok)
    return res


def _run(argv: list[str], cwd: Path, log: list[dict]) -> None:
    t0 = time.time()
    proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, check=False)
    log.append({"argv": argv, "cwd": str(cwd), "rc": proc.returncode, "wall_s": round(time.time() - t0, 2)})
    if proc.returncode != 0:
        raise PerturbBuildError(f"{argv[0]} failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")


#: The spellings looked for BY CONTENT in the build's own sources and libraries (see derive_invocation).
PRELOAD_PLUSARG = "+loadmem="
WRAP_PLUSARGS = ("+permissive", "+permissive-off")


def _link_libraries(link_line: Sequence[str]) -> list[Path]:
    """The library files a link line names (-L dirs x -l names), those that exist."""
    dirs = [Path(t[2:]) for t in link_line if t.startswith("-L") and len(t) > 2]
    names = [t[2:] for t in link_line if t.startswith("-l") and len(t) > 2]
    found = []
    for n in names:
        for d in dirs:
            for suffix in (".a", ".so"):
                f = d / f"lib{n}{suffix}"
                if f.is_file():
                    found.append(f)
    return found


def derive_invocation(plan: LinkPlan, harness_src: Path) -> dict:
    """How to run the variant so its memory model actually sees the program's data.

    A simulator front end that loads the program through the SoC (a serial host interface writing
    through the coherent bus) leaves every operand resident in the last-level cache, and then no request
    reaches the memory model at all: measured, a residual add that fails on most seeds when its image is
    preloaded into the backing store passed every seed when the host interface loaded it. So the program
    must be PRELOADED into the backing store, when the harness offers that.

    Both facts are found by CONTENT, never assumed: the preload plusarg is used only if the memory
    harness source parses it, and the wrap that lets the front end accept it only if a library the link
    line names carries both spellings. Returns ``{"args": [...], "evidence": {...}}``, where ``{elf}`` in
    an arg stands for the executable; empty args when neither is found (the check then runs on whatever
    the front end does, and says so).
    """
    evidence: dict = {"preload": None, "wrap": None}
    args: list[str] = []
    if PRELOAD_PLUSARG in harness_src.read_text(errors="replace"):
        evidence["preload"] = str(harness_src)
        args = [PRELOAD_PLUSARG + "{elf}"]
        for lib in _link_libraries(plan.link_line):
            blob = lib.read_bytes()
            if all(w.encode() in blob for w in WRAP_PLUSARGS):
                evidence["wrap"] = str(lib)
                args = [WRAP_PLUSARGS[0], *args, WRAP_PLUSARGS[1]]
                break
    return {"args": args, "evidence": evidence}


def invocation_for(emulator: str | Path) -> list[str]:
    """The invocation args recorded in the build receipt beside ``emulator``; [] when there is none.

    A receipt that exists and cannot be read raises: running the engine without the preload it records
    would leave the operands cached and perturb nothing, which reads exactly like a clean sweep."""
    rec = Path(emulator).parent / RECEIPT_NAME
    if not rec.is_file():
        return []
    receipt = json.loads(rec.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise PerturbBuildError(f"{rec} is not a build receipt")
    return [str(arg) for arg in ((receipt.get("invocation") or {}).get("args") or [])]


def build_verilator_variant(
    obj_dir: str | Path,
    out_dir: str | Path,
    *,
    makefile: str,
    argc_expr: str = "info.argc",
    argv_expr: str = "info.argv",
    control: bool = True,
    base_digest: str = "",
) -> dict:
    """Build ``<out_dir>/emulator``: the simulator ``obj_dir``'s makefile builds, relinked with its
    memory harness rebuilt through the perturbing model. Returns (and writes) the build receipt.

    ``base_digest`` is the declared digest of the stock simulator (a hardware-pins artifact). With
    ``control`` the UNMODIFIED objects are relinked too and compared to it, which is what lets the
    receipt say the variant differs from the certified engine in its memory model and nothing else.
    """
    # Absolute, because every command runs with ``out_dir`` as its working directory: a relative one
    # would name the patched source relative to itself and the compile would find nothing.
    obj_dir, out_dir = Path(obj_dir), Path(out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    msrc = _model_sources()
    plan = link_plan(obj_dir, makefile)
    obj, src = memory_harness(plan)
    log: list[dict] = []

    patched = out_dir / src.name
    patched.write_text(patch_harness_source(src.read_text(), argc_expr, argv_expr))
    base_compile = plan.compile_lines[obj]
    harness_argv = [t if t != str(src) else str(patched) for t in base_compile]
    harness_argv = _relocate(harness_argv, obj_dir, out_dir, {}) + [f"-I{msrc}"]
    _run(harness_argv, out_dir, log)
    model_argv = [
        t if t != str(src) else str(msrc / MODEL_SOURCE) for t in _relocate(base_compile, obj_dir, out_dir, {})
    ]
    model_obj = out_dir / "mm_perturb.o"
    model_argv[model_argv.index(str(out_dir / Path(obj).name))] = str(model_obj)
    model_argv.append(f"-I{msrc}")
    _run(model_argv, out_dir, log)

    link = _relocate(plan.link_line, obj_dir, out_dir, {obj: out_dir / Path(obj).name})
    link[link.index(str(out_dir / plan.output.name))] = str(out_dir / BINARY_NAME)
    link.insert(link.index(str(out_dir / Path(obj).name)) + 1, str(model_obj))
    _run(link, out_dir, log)

    control_rec: dict = {"performed": False}
    if control:
        ctl = _relocate(plan.link_line, obj_dir, out_dir, {})
        ctl[ctl.index(str(out_dir / plan.output.name))] = str(out_dir / "control_relink")
        _run(ctl, out_dir, log)
        got = _sha(out_dir / "control_relink")
        control_rec = {
            "performed": True,
            "digest": got,
            "declared_base_digest": base_digest or None,
            "base_on_disk_digest": _sha(plan.output) if plan.output.is_file() else None,
            "matches_declared_base": (got == base_digest) if base_digest else None,
        }
        (out_dir / "control_relink").unlink()

    receipt = {
        "schema_version": SCHEMA,
        "binary": {"path": str(out_dir / BINARY_NAME), "sha256": _sha(out_dir / BINARY_NAME)},
        "base_simulator": {"path": str(plan.output), "makefile": str(plan.makefile)},
        "memory_harness": {"object": obj, "source": str(src), "source_sha256": _sha(src)},
        "patched_harness_sha256": _sha(patched),
        "model_sources": {n: _sha(msrc / n) for n in (MODEL_HEADER, MODEL_SOURCE)},
        "control_relink": control_rec,
        "commands": log,
        "default_behaviour": "stock mm_magic_t unless +mem_perturb_seed is given",
        "invocation": derive_invocation(plan, src),
    }
    (out_dir / RECEIPT_NAME).write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


# ---------------------------------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------------------------------


@dataclass
class SeedRun:
    seed: int | None  # None = the in-order model
    rc: int
    wall_s: float
    outcome: str  # what the caller's judge said: e.g. "pass" / "fail:<detail>"
    console_sha256: str


def run_once(
    emulator: str | Path,
    elf: str | Path,
    *,
    seed: int | None,
    max_latency: int = 64,
    tail_permille: int = 0,
    region_log2: int = 0,
    slow_permille: int = 500,
    via: str = "plusarg",
    extra_args: Sequence[str] = (),
    plusarg_wrap: tuple[str, str] | None = None,
    timeout: float = 3600,
    console: str | Path | None = None,
) -> tuple[int, str, float]:
    """One run. Returns (rc, console text, wall seconds).

    The knobs reach the model as plusargs (``via="plusarg"``) or as environment variables
    (``via="env"``). The environment form leaves the simulator's command line exactly what its own
    backend would run, which is what a grade wants. ``plusarg_wrap`` brackets plusargs for a front end
    that rejects ones it does not know; the caller names the spelling, since it belongs to the simulator.
    """
    if via not in ("plusarg", "env"):
        raise ValueError(f"via must be 'plusarg' or 'env', got {via!r}")
    knobs: dict[str, int] = {}
    if seed is not None:
        knobs = {"seed": seed, "max_latency": max_latency, "tail_permille": tail_permille}
        if region_log2:
            knobs.update({"region_log2": region_log2, "slow_permille": slow_permille})
    plus = list(extra_args)
    env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIX)}
    if via == "plusarg":
        plus += [f"+mem_perturb_{k}={v}" for k, v in knobs.items()]
    else:
        env.update({f"{ENV_PREFIX}{k.upper()}": str(v) for k, v in knobs.items()})
    if plusarg_wrap and plus:
        plus = [plusarg_wrap[0], *plus, plusarg_wrap[1]]
    argv = [str(emulator), *plus, str(elf)]
    t0 = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        rc, text = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = -9
        text = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        text += "\n[mem-perturb] TIMEOUT\n"
    wall = time.time() - t0
    if console is not None:
        Path(console).write_text(text)
    return rc, text, wall


def sweep(
    emulator: str | Path,
    elf: str | Path,
    seeds: Sequence[int],
    judge: Callable[[int, str], str],
    **kw,
) -> dict:
    """Run ``elf`` in order and under each seed; ``judge(rc, console) -> outcome string``.

    The verdict compares OUTCOMES, never raw consoles (a reordered run legitimately takes a different
    number of cycles, so its console differs even when its answer does not):

      ``order_sensitive``  some seed's outcome differs from the in-order outcome
      ``order_stable``     every seed agrees with the in-order outcome
      ``unjudged``         the in-order run itself did not produce a judgeable outcome
    """
    console_dir = kw.pop("console_dir", None)
    workers = max(1, int(kw.pop("workers", 1)))
    # A caller sweeping several knob profiles over one executable needs the in-order run only once;
    # without it the seeds are compared against "pass", the only outcome a certified program has.
    include_in_order = bool(kw.pop("include_in_order", True))

    def one(seed: int | None) -> SeedRun:
        con = None
        if console_dir is not None:
            con = Path(console_dir) / f"console_{'inorder' if seed is None else f'seed{seed}'}.log"
        rc, text, wall = run_once(emulator, elf, seed=seed, console=con, **kw)
        return SeedRun(seed, rc, round(wall, 2), judge(rc, text), hashlib.sha256(text.encode()).hexdigest())

    t0 = time.time()
    order = [None, *seeds] if include_in_order else list(seeds)
    if workers == 1:
        runs = [one(s) for s in order]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(one, order))  # map keeps the in-order run first
    elapsed = time.time() - t0
    base = runs[0].outcome if include_in_order else "pass"
    perturbed = runs[1:] if include_in_order else runs
    if not base or base.startswith("unjudged"):
        verdict = "unjudged"
    else:
        verdict = "order_sensitive" if any(r.outcome != base for r in perturbed) else "order_stable"
    return {
        "verdict": verdict,
        "in_order_outcome": base,
        "diverging_seeds": [r.seed for r in perturbed if r.outcome != base],
        "runs": [r.__dict__ for r in runs],
        "emulator_sha256": _sha(emulator),
        "elf_sha256": _sha(elf),
        "cpu_wall_s_total": round(sum(r.wall_s for r in runs), 2),
        "elapsed_s": round(elapsed, 2),
        "workers": workers,
    }
