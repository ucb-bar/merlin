"""Kernel-performance CEILING harness (S4.2) — compile a CURATED expert kernel
*standalone* and MEASURE its cycle count on **spike**, to establish the bar our
compiler's RVV codegen is judged against.

    attainment = ceiling_cycles / compiler_cycles

(1.0 == we matched the expert; < 1.0 == the expert kernel is faster than what we
emit; > 1.0 == we beat the expert). The ceiling is *measured*, never modeled.

Which corpus, and how it builds
-------------------------------
The ceiling is measured on a *standalone-benchmark* corpus (registry layout
``standalone_benchmarks`` in ``src/merlin/resources/contracts/corpora.yaml``): each benchmark ships its own
``main`` + golden data + an htif printf path, builds to a single ``.riscv`` ELF, and runs
under ``spike`` printing a cycle count per kernel invocation. No bare-metal main is
generated; the bench already is one. ``VOPACC`` benches are excluded per the mining
contract.

Everything specific to that corpus is DATA owned by the target the corpus registry names as
its owner: ``kernel_ceiling.yaml`` in the target's package (:class:`BenchConfig`), found
through the target registry. It carries the toolchain flags, the spike ISA/harts/memory map,
the bench tree layout, and which bench measures which op through which console reader. This
module holds only the generic build/run/read machinery, so a second corpus is a new config
file, not an edit here.

Honesty contract
----------------
:func:`run_kernel_ceiling` returns ``None`` (never a fabricated number) when the
toolchain/spike is unavailable, the bench won't build, the run fails, or the requested
``(M,N,K)`` is not among the sizes the bench actually executes. Each ceiling row records
exactly which bench/kernel produced it and how the cycle count was parsed.

Reading the console is deliberately three-valued (:class:`CycleReading`): PARSED, ABSENT (this
bench does not sweep that size), and UNPARSEABLE (it DID measure the size but the console did not
say so in a shape we recognize). The last is a tooling defect and is reported, never quietly
downgraded to "not measured" and never turned into a cycle count of 0.

The ``fingerprint_key`` ``(op, dtype, shape_regime)`` matches
:class:`merlin.kernels.compare.RvvFingerprint`'s key, so a ceiling row joins 1:1 to a
compiler-measured cycle count for the same op-shape.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..common.paths import repo_root
from . import build_asm

DEFAULT_CEILING_PATH = "out/artifacts/ceiling/ceiling.jsonl"

#: File name of a target's kernel-ceiling bench config, at the root of its package.
CONFIG_NAME = "kernel_ceiling.yaml"
CONFIG_SCHEMA = "merlin.kernel_ceiling.v1"


# --------------------------------------------------------------------------- shape_regime
def shape_regime(op: str, M: int, N: int, K: int) -> str:
    """A compact, deterministic regime STRING for the fingerprint key.

    compare.py uses a single string (e.g. "square_small"), not the feature-extractor's
    regime list, so we produce one here that is stable for a given (op, M, N, K). The
    vocabulary mirrors features/shape_regime.py's intent (skinny / square / large) at a
    granularity that joins cleanly to a compiler-measured cycle count.
    """
    mn = min(M, N)
    if mn <= 1:  # dot / elementwise: N=K=1, M=length
        return "vector"
    if M == N == K:
        if M >= 256:
            return "square_large"
        if M >= 32:
            return "square_medium"
        return "square_small"
    if mn <= 16:
        return "skinny"
    return "rectangular"


# ------------------------------------------------------- console reading (three states, no regex)
#
# These benches have no machine-readable output mode: their printf'd console text IS the contract.
# It is fixed by the corpus's own C sources, so we anchor on those literals and then VALIDATE the
# token we pulled out, refusing anything that is not a plain non-negative decimal.
#
# Three outcomes, never collapsed:
#   ABSENT       the bench does not measure this size (its sweep never printed that header)
#   UNPARSEABLE  the size WAS measured but the cycle line could not be read -> the console format
#                drifted; we must say so
#   PARSED       a real number
# UNPARSEABLE never becomes ABSENT and neither ever becomes 0: an unmeasurable ceiling is UNKNOWN.

_PARSED = "parsed"
_ABSENT = "absent"
_UNPARSEABLE = "unparseable"

_DIGITS = frozenset("0123456789")


@dataclass(frozen=True)
class CycleReading:
    """What one bench console said about the requested size."""

    state: str  # _PARSED / _ABSENT / _UNPARSEABLE
    cycles: int | None = None
    instructions: int | None = None
    note: str = ""  # how it was read (goes on the ceiling row)
    detail: str = ""  # what went wrong, when state is _UNPARSEABLE

    @classmethod
    def absent(cls) -> "CycleReading":
        return cls(_ABSENT)

    @classmethod
    def unparseable(cls, detail: str) -> "CycleReading":
        return cls(_UNPARSEABLE, detail=detail)


def _leading_count(text: str) -> tuple[int | None, int]:
    r"""Read ``\s*(\d+)`` off the FRONT of ``text`` -> ``(value, index just past the digits)``.

    Only ASCII digits: the spike console is ASCII, and refusing an exotic digit is the safe side of
    "refuse rather than coerce". A sign, a dot, or a letter stops the run, so ``-5`` and ``1.2`` are
    refused, exactly as the old ``(\d+)`` (which could not match a leading '-') did.
    """
    i = 0
    while i < len(text) and text[i].isspace():
        i += 1
    j = i
    while j < len(text) and text[j] in _DIGITS:
        j += 1
    if j == i:
        return None, 0
    return int(text[i:j]), j


def _square_shape_header(line: str, size: int) -> bool:
    r"""True for a square-sweep bench's header at ``size``.

    Live console line (spike, a square-sweep integer GEMM bench)::

        Calculating a (32 x 32) x (32 x 32) matrix multiplication...

    printf source: ``"Calculating a (%d x %d) x (%d x %d) matrix multiplication...\n"`` with the
    same ``s`` in all four slots. The old pattern was
    ``\(\s*S\s*x\s*S\s*\)\s*x\s*\(\s*S\s*x\s*S\s*\)`` — the four-operand shape with arbitrary
    whitespace ANYWHERE inside it, including none. Deleting all whitespace from the line and
    looking for the canonical spelling accepts exactly that set. The token is specific enough to
    anchor on (it names the size four times), so it cannot pick a number out of another line: for
    S=4 it does not match ``(64 x 64) x (64 x 64)`` or ``(14 x 4) x (4 x 4)``.
    """
    return f"({size}x{size})x({size}x{size})" in "".join(line.split())


def _took_cycles(line: str) -> int | None:
    r"""``The execution took 26055 cycles.`` -> 26055; ``None`` when the line is not that.

    printf source: ``"The execution took %d cycles.\n"``. The old pattern was ``took\s+(\d+)\s+cycles``:
    whitespace-delimited, so the count is always a whole token. Anchoring on the exact token
    ``took`` is marginally stricter than the old substring match (which would also have fired on
    ``mistook``), which no spelling this bench emits can hit. A ``%d`` that printed negative has no
    all-digit token and is refused, as the old pattern also refused it.
    """
    parts = line.split()
    for i, token in enumerate(parts):
        if token != "took" or i + 2 >= len(parts):
            continue
        if any(c not in _DIGITS for c in parts[i + 1]) or not parts[i + 1]:
            continue
        if parts[i + 2].startswith("cycles"):
            return int(parts[i + 1])
    return None


def _length_header(line: str, avl: int) -> bool:
    r"""True for a length-sweep bench's per-length header at ``avl``.

    Live console line::

        Calulating 64b dotp with vectors with length = 512

    printf source: ``"Calulating <w>b dotp with vectors with length = %lu\n"`` (the bench's own
    typo). The old pattern was ``length\s*=\s*AVL\b``; this walk is that literally — find
    ``length``, skip whitespace, require '=', skip whitespace, then the digits must be exactly
    ``avl`` and must not run on into another word character (so ``length = 512`` does not answer a
    request for 51).
    """
    want = str(avl)
    at = line.find("length")
    while at >= 0:
        i = at + len("length")
        while i < len(line) and line[i].isspace():
            i += 1
        if i < len(line) and line[i] == "=":
            i += 1
            while i < len(line) and line[i].isspace():
                i += 1
            if line.startswith(want, i):
                end = i + len(want)
                if end == len(line) or not (line[end].isalnum() or line[end] == "_"):
                    return True
        at = line.find("length", at + 1)
    return False


def _vector_cycles(line: str) -> tuple[int, int] | None:
    r"""``Vector cycles: 401 instructions: 401`` -> ``(401, 401)``; ``None`` otherwise.

    printf source: ``"Vector cycles: %ld instructions: %ld\n"``. The old pattern was
    ``Vector cycles:\s*(\d+)\s+instructions:\s*(\d+)`` — both counts anchored directly to their own
    named literal, which is what makes this safe to read positionally.
    """
    marker, gap = "Vector cycles:", "instructions:"
    at = line.find(marker)
    while at >= 0:
        rest = line[at + len(marker) :]
        cycles, end = _leading_count(rest)
        if cycles is not None:
            after = rest[end:]
            if after[:1].isspace() and after.lstrip().startswith(gap):  # the old `\s+instructions:`
                instructions, _ = _leading_count(after.lstrip()[len(gap) :])
                if instructions is not None:
                    return cycles, instructions
        at = line.find(marker, at + 1)
    return None


def _named_counter(line: str, name: str) -> int | None:
    r"""``mcycle = 137509`` -> 137509, for a ``setStats`` counter dump.

    printf source: the corpus's ``benchmarks/common/syscalls.c`` prints ``"%s = %d\n"`` for each enabled
    counter, and ``setStats(0)`` stores the DELTA over the measured region (``csr -= counters[i]``),
    so ``mcycle`` here is the region's cycle count, not an absolute CSR read. Anchored on the
    counter name as the whole left-hand side, and the right-hand side must be a bare count.
    """
    lhs, sep, rhs = line.partition("=")
    if not sep or lhs.strip() != name:
        return None
    value, end = _leading_count(rhs)
    if value is None or rhs[end:].strip():  # nothing but the number may follow
        return None
    return value


def _colon_cycles(line: str) -> int | None:
    r"""``core   0: 137509 cycles, ... CPI`` -> 137509 (the riscv-tests / pk ``setStats`` spelling).

    This is what the retired ``:\s*(\d+)\s+cycles`` pattern read. The registered counter-dump corpus
    does NOT print it (see :func:`_read_counter_dump`), but a pk-hosted console would, so we keep
    accepting it.
    """
    at = line.find(":")
    while at >= 0:
        value, end = _leading_count(line[at + 1 :])
        if value is not None:
            after = line[at + 1 + end :]
            if after[:1].isspace() and after.lstrip().startswith("cycles"):
                return value
        at = line.find(":", at + 1)
    return None


# ------------------------------------------------------------------------- console readers
# Each reader maps a requested ``(M, N, K)`` to a :class:`CycleReading` for one console FORMAT. A bench
# config names its bench's reader by key (:data:`CONSOLE_READERS`), so the formats are generic code and
# the choice of which bench prints which is data.


def _read_square_sweep(console: str, M: int, N: int, K: int) -> CycleReading:
    """A square sweep (e.g. s=4,8,16,32,64) printing a shape header then
    ``The execution took N cycles.`` a couple of lines later."""
    if not (M == N == K):
        return CycleReading.absent()
    lines = console.splitlines()
    first_window = ""
    for i, line in enumerate(lines):
        if not _square_shape_header(line, M):
            continue
        window = lines[i : i + 4]
        for nxt in window:
            cycles = _took_cycles(nxt)
            if cycles is not None:
                return CycleReading(
                    _PARSED, cycles, note=f"'The execution took N cycles.' under the ({M} x {M}) header"
                )
        # Keep scanning: a later header for the same size may carry the line (the old pattern
        # searched on too). Only once every one of them is exhausted is this UNPARSEABLE.
        first_window = first_window or " | ".join(x.strip() for x in window)
    if first_window:
        return CycleReading.unparseable(
            f"the bench printed the ({M} x {M}) header but no 'took N cycles' line followed it: " + first_window
        )
    return CycleReading.absent()  # this size is simply not in the sweep


def _read_length_sweep(console: str, M: int, N: int, K: int) -> CycleReading:
    """A length sweep (e.g. avl=8,64,512 for each element width), printing
    ``Calulating <w>b dotp with vectors with length = <avl>`` then
    ``Vector cycles: C instructions: I``.

    The requested length is M (N==K==1). Width is implied by dtype but the 64b block is measured
    first; we match on the length header regardless of width and take the FIRST (64-bit)
    occurrence so the key is deterministic."""
    if N != 1 or K != 1:
        return CycleReading.absent()
    lines = console.splitlines()
    first_window = ""
    for i, line in enumerate(lines):
        if not _length_header(line, M):
            continue
        window = lines[i : i + 3]
        for nxt in window:
            got = _vector_cycles(nxt)
            if got is not None:
                return CycleReading(
                    _PARSED, got[0], got[1], note=f"'Vector cycles: C instructions: I' under length = {M}"
                )
        # This header repeats once per element width, so keep scanning the later ones (as the old
        # pattern's search did) before calling the console unreadable.
        first_window = first_window or " | ".join(x.strip() for x in window)
    if first_window:
        return CycleReading.unparseable(
            f"the bench printed the length = {M} header but no 'Vector cycles:' line followed it: " + first_window
        )
    return CycleReading.absent()


def _read_counter_dump(console: str, M: int, N: int, K: int) -> CycleReading:
    r"""A bench that runs ONE region between ``setStats(1)``/``setStats(0)``, which dumps the enabled
    hardware counters. Such a bench measures one fixed shape, declared as ``shape`` in its config and
    enforced by :meth:`Bench.read` before this reader runs.

    The retired pattern here was ``:\s*(\d+)\s+cycles`` — the riscv-tests / pk spelling
    ``<code>: C cycles, ... CPI``. **A counter-dump console does not print that.** Verified by building
    and running such a bench on spike from this tree; the whole console is::

        sgemm M,N,K = 71,71,71
        mcycle = 137509
        minstret = 137514

    (the corpus's ``benchmarks/common/syscalls.c`` prints ``"%s = %d\n"`` per counter, and
    ``setStats(0)`` has already turned each into a delta over the measured region.) So the old pattern
    matched nothing and this bench's ceiling was silently unmeasurable — the exact failure the
    "parse structurally" rule exists to stop. We now read the counter dump the bench actually emits,
    and still accept the pk spelling for a pk-hosted console.
    """
    cycles = instructions = None
    for line in console.splitlines():
        if cycles is None:
            cycles = _named_counter(line, "mcycle")
        if instructions is None:
            instructions = _named_counter(line, "minstret")
    if cycles is not None:
        return CycleReading(_PARSED, cycles, instructions, note="setStats counter dump ('mcycle = N' / 'minstret = N')")
    for line in console.splitlines():
        legacy = _colon_cycles(line)
        if legacy is not None:
            return CycleReading(_PARSED, legacy, note="pk setStats line ('<code>: N cycles')")
    return CycleReading.unparseable(
        f"the bench ran ({M},{N},{K}) but neither a 'mcycle = N' counter dump nor a "
        "'<code>: N cycles' stats line was found in its console: " + console.strip()[:300]
    )


#: Console formats a bench config may name (``benches.<name>.console``).
CONSOLE_READERS: dict[str, Callable[[str, int, int, int], CycleReading]] = {
    "square_sweep": _read_square_sweep,
    "length_sweep": _read_length_sweep,
    "counter_dump": _read_counter_dump,
}


# ------------------------------------------------------------------------- bench config (data)
class BenchConfigError(ValueError):
    """A kernel-ceiling bench config is malformed, or disagrees with what selected it."""


class BenchConfigMissing(LookupError):
    """The selected target's package ships no kernel-ceiling bench config."""


@dataclass(frozen=True)
class Bench:
    """How to read a cycle number out of one standalone benchmark.

    ``op``/``dtype`` are the curated kernel's canonical op + element type. ``console`` names the
    :data:`CONSOLE_READERS` entry for the bench's printf format; ``shape``, when set, is the one
    ``(M, N, K)`` the bench measures. :meth:`read` maps a requested ``(M, N, K)`` to a
    :class:`CycleReading` — a measured count, an honest ABSENT when that size is not among the ones
    the bench executes, or UNPARSEABLE when the size WAS measured but the console did not say what we
    know how to read.
    """

    bench: str
    op: str
    dtype: str
    kernel_ref: str
    console: str
    shape: tuple[int, int, int] | None = None

    def read(self, console: str, M: int, N: int, K: int) -> CycleReading:
        if self.shape is not None and (M, N, K) != self.shape:
            return CycleReading.absent()
        return CONSOLE_READERS[self.console](console, M, N, K)


@dataclass(frozen=True)
class BenchConfig:
    """One target's kernel-ceiling configuration, read from its ``kernel_ceiling.yaml``."""

    target: str
    corpus: str
    path: Path
    isa: str
    harts: int
    spike_memory: str
    cflags: tuple[str, ...]
    link_opts: tuple[str, ...]
    sources: tuple[str, ...]
    includes: tuple[str, ...]
    linker_script: str
    toolchain_headers: tuple[str, ...]
    benches: dict[str, Bench]

    def benchmarks_dir(self) -> Path | None:
        """The benchmark tree of this config's corpus, or None when the corpus is not registered."""
        return build_asm.benchmarks_dir(self.corpus)

    def encoding_include_dir(self) -> Path | None:
        """The dir holding the first declared toolchain header present under the spike toolchain's
        chipyard root, or None."""
        from ..runtime.backends import spike

        chip = spike.chipyard_root()
        for rel in self.toolchain_headers:
            p = chip / rel
            if p.is_file():
                return p.parent
        return None

    def bench_for(self, kernel_ref: str | None, op: str | None, dtype: str | None) -> Bench | None:
        """The bench ``kernel_ref`` names, else the best (op, dtype) match, else None."""
        if kernel_ref in self.benches:
            return self.benches[kernel_ref]
        op = (op or "").lower()
        dtype = (dtype or "").lower()
        for spec in self.benches.values():
            if spec.op == op and (not dtype or spec.dtype == dtype):
                return spec
        for spec in self.benches.values():
            if spec.op == op:
                return spec
        return None


def _strings(doc: dict, key: str, path: Path) -> tuple[str, ...]:
    value = doc.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise BenchConfigError(f"{path}: `{key}` must be a list of non-empty strings")
    return tuple(value)


def read_bench_config(path: str | Path) -> BenchConfig:
    """Parse and validate one ``kernel_ceiling.yaml``. Raises :class:`BenchConfigError` on any defect:
    a config that half-loads would build with the wrong flags and report a plausible wrong ceiling."""
    from ..common.yaml import load_yaml  # deferred: importing this module must stay dependency-free

    path = Path(path)
    doc = load_yaml(path)
    if not isinstance(doc, dict) or doc.get("schema") != CONFIG_SCHEMA:
        raise BenchConfigError(f"{path}: expected a mapping with schema {CONFIG_SCHEMA!r}")
    spike_doc, build, benches = doc.get("spike"), doc.get("build"), doc.get("benches")
    if not isinstance(spike_doc, dict) or not isinstance(build, dict) or not isinstance(benches, dict):
        raise BenchConfigError(f"{path}: `spike`, `build` and `benches` must be mappings")
    for key in ("target", "corpus"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise BenchConfigError(f"{path}: `{key}` must name the owning {key}")
    harts = spike_doc.get("harts")
    if not isinstance(harts, int) or isinstance(harts, bool) or harts < 1:
        raise BenchConfigError(f"{path}: `spike.harts` must be a positive integer")
    for key in ("isa", "memory"):
        if not isinstance(spike_doc.get(key), str) or not spike_doc[key]:
            raise BenchConfigError(f"{path}: `spike.{key}` must be a non-empty string")
    if not isinstance(build.get("linker_script"), str) or not build["linker_script"]:
        raise BenchConfigError(f"{path}: `build.linker_script` must be a non-empty string")
    parsed: dict[str, Bench] = {}
    for name, spec in benches.items():
        if not isinstance(spec, dict):
            raise BenchConfigError(f"{path}: bench {name!r} must be a mapping")
        fields = {key: spec.get(key) for key in ("op", "dtype", "kernel_ref", "console")}
        if not all(isinstance(v, str) and v for v in fields.values()):
            raise BenchConfigError(f"{path}: bench {name!r} needs op, dtype, kernel_ref and console")
        if fields["console"] not in CONSOLE_READERS:
            raise BenchConfigError(
                f"{path}: bench {name!r} names unknown console reader "
                f"{fields['console']!r} (known: {', '.join(sorted(CONSOLE_READERS))})"
            )
        shape = spec.get("shape")
        if shape is not None:
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape)
            ):
                raise BenchConfigError(f"{path}: bench {name!r} `shape` must be three positive integers")
            shape = (shape[0], shape[1], shape[2])
        parsed[str(name)] = Bench(str(name), shape=shape, **fields)
    if not parsed:
        raise BenchConfigError(f"{path}: declares no benches")
    return BenchConfig(
        target=doc["target"],
        corpus=doc["corpus"],
        path=path,
        isa=spike_doc["isa"],
        harts=harts,
        spike_memory=spike_doc["memory"],
        cflags=_strings(build, "cflags", path),
        link_opts=_strings(build, "link_opts", path),
        sources=_strings(build, "sources", path),
        includes=_strings(build, "includes", path),
        linker_script=build["linker_script"],
        toolchain_headers=_strings(build, "toolchain_headers", path),
        benches=parsed,
    )


def bench_config_path(target: str) -> Path | None:
    """Where ``target``'s kernel-ceiling bench config lives, or None when its package ships none.

    Looked up through the target registry: the package SELECTED for ``target`` first (an explicit
    ``MERLIN_TARGET_PATH`` provider that ships its own config wins), then the target's authored
    reference metadata. Reading it inspects an authored input; no provider code is imported.
    """
    from ..targetgen import target_registry

    bases = [target_registry.resolve(target).base]
    reference = target_registry.reference_targets().get(target)
    if reference is not None:
        bases.append(reference)
    for base in bases:
        path = Path(base) / CONFIG_NAME
        if path.is_file():
            return path
    return None


def load_bench_config(target: str) -> BenchConfig:
    """``target``'s bench config. Raises :class:`BenchConfigMissing` when its package ships none, and
    :class:`BenchConfigError` when the file is malformed or declares a different target."""
    path = bench_config_path(target)
    if path is None:
        raise BenchConfigMissing(
            f"target {target!r} ships no {CONFIG_NAME} (looked in its selected package and its "
            f"reference metadata); a kernel ceiling cannot be measured without one"
        )
    config = read_bench_config(path)
    if config.target != target:
        raise BenchConfigError(f"{path}: declares target {config.target!r}, selected as {target!r}")
    return config


def bench_config_for_source(source: str | None = None) -> BenchConfig | None:
    """The bench config of the standalone-benchmark corpus ``source`` names (default: the corpus
    registry's first), loaded from the target the corpus registry names as its owner.

    None when ``source`` names no standalone-benchmark corpus or the corpus declares no owner.
    Raises :class:`BenchConfigMissing` when the owner ships no config, and :class:`BenchConfigError`
    when the owner's config is about a different corpus.
    """
    from ..targetgen.corpora import kernel_corpus_target

    corpus = build_asm.benchmark_source(source)
    owner = kernel_corpus_target(corpus) if corpus else None
    if owner is None:
        return None
    config = load_bench_config(owner)
    if config.corpus != corpus:
        raise BenchConfigError(
            f"{config.path}: configures corpus {config.corpus!r}, but the corpus "
            f"registry names {owner!r} as the owner of {corpus!r}"
        )
    return config


# ------------------------------------------------------------------------- build + run
def corpus_available(config: BenchConfig | None) -> bool:
    """True when the riscv gcc + spike + ``config``'s corpus + its toolchain header are all present."""
    from ..runtime.backends import spike

    if config is None:
        return False
    root = config.benchmarks_dir()
    return spike.available() and root is not None and root.is_dir() and config.encoding_include_dir() is not None


def _layout_path(benchmarks: Path, rel: str, bench: str) -> Path:
    """A config layout entry under the benchmark tree; ``{bench}`` is the bench's own directory."""
    return benchmarks / rel.replace("{bench}", bench)


def _build_bench_elf(bench: str, workdir: Path, config: BenchConfig, *, timeout: int = 300) -> Path | None:
    """Build a benchmark of ``config``'s corpus to a spike ELF, mirroring the bench Makefile recipe
    the config transcribes.

    Returns the ELF path, or ``None`` on any failure (missing toolchain, bench, or a
    compile/link error) — never raises for an ordinary build failure.
    """
    from ..runtime.backends import spike

    gcc = spike.gcc_path()
    if not gcc.is_file():
        return None
    benchmarks = config.benchmarks_dir()
    if benchmarks is None:
        return None
    bench_dir = benchmarks / bench
    linker_script = _layout_path(benchmarks, config.linker_script, bench)
    if not bench_dir.is_dir() or not linker_script.parent.is_dir():
        return None
    enc = config.encoding_include_dir()
    if enc is None:
        return None

    srcs: list[str] = []
    for rel in config.sources:
        d = _layout_path(benchmarks, rel, bench)
        if d.is_dir():
            srcs += [str(p) for p in sorted(d.glob("*.c"))]
            srcs += [str(p) for p in sorted(d.glob("*.S"))]
    incs: list[str] = []
    for rel in config.includes:
        incs += ["-I", str(_layout_path(benchmarks, rel, bench))]
    incs += ["-I", str(enc)]
    elf = workdir / f"{bench}.riscv"
    cmd = [str(gcc), *incs, *config.cflags, "-o", str(elf), *srcs, *config.link_opts, "-T", str(linker_script)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    # ld warns about RWX LOAD segments (harmless); only a nonzero exit / missing ELF fails.
    if proc.returncode != 0 or not elf.is_file():
        return None
    return elf


def _run_bench_elf(elf: Path, config: BenchConfig, *, isa: str | None = None, timeout: int = 300) -> str | None:
    """Run a benchmark ELF on spike with ``config``'s harts and memory map, and return console text, or
    None on failure."""
    from ..runtime.backends import spike

    cmd = [str(spike.spike_path()), f"--isa={isa or config.isa}", f"-p{config.harts}", config.spike_memory, str(elf)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


# ------------------------------------------------------------------------- public API
def run_kernel_ceiling(
    source: str,
    kernel_ref: str,
    op: str,
    dtype: str,
    MNK: tuple[int, int, int],
    *,
    target: str = "spike",
    isa: str | None = None,
    timeout: int = 300,
    diagnostics: list[str] | None = None,
    config: BenchConfig | None = None,
) -> dict | None:
    """Build a curated kernel standalone, run it on ``target`` (spike), and return one
    ceiling row, or ``None`` if it cannot build/run or the requested size is not measured.

    ``config`` is the bench config to measure with; by default the one the corpus ``source`` names
    (:func:`bench_config_for_source`). ``isa`` defaults to the config's spike ISA.

    ``kernel_ref`` doubles as a bench name of the config (e.g. ``vec-igemm``); when it is a
    configured bench it also fixes op/dtype/reader, so callers may pass the bench name as
    ``kernel_ref`` and let the config supply op/dtype.

    Returns a dict with: op, dtype, M, N, K, shape_regime, source, target, cycles,
    fingerprint_key (+ bench, kernel_ref, instructions, isa, note).

    ``None`` covers two different things and they must not be confused by the caller: the size was
    never measured (ordinary, silent), or the bench DID measure it but its console could not be
    read. The second is a tooling defect, so it is written to stderr and appended to
    ``diagnostics`` if a list is passed. Neither ever becomes a cycle count of 0.
    """
    if target != "spike":
        return None
    corpus = build_asm.benchmark_source(source or "")
    if corpus is None:  # not a registered standalone-benchmark corpus
        return None
    if "vopacc" in (kernel_ref or "").lower():  # mining-contract exclusion
        return None
    if config is None:
        try:
            config = bench_config_for_source(corpus)
        except BenchConfigMissing as exc:  # surfaced, never read as "not measured"
            if diagnostics is not None:
                diagnostics.append(str(exc))
            print(f"kernel-bench: {exc}", file=sys.stderr)
            return None
    if config is None or config.corpus != corpus:
        return None
    isa = isa or config.isa

    spec = config.bench_for(kernel_ref, op, dtype)
    if spec is None:
        return None
    # Config op/dtype win when the caller passed the bench name; else honor caller's.
    op = spec.op if kernel_ref in config.benches else (op or spec.op)
    dtype = spec.dtype if kernel_ref in config.benches else (dtype or spec.dtype)

    if not corpus_available(config):
        return None

    M, N, K = (int(x) for x in MNK)
    with tempfile.TemporaryDirectory(prefix="merlin_ceiling_") as tmp:
        elf = _build_bench_elf(spec.bench, Path(tmp), config, timeout=timeout)
        if elf is None:
            return None
        console = _run_bench_elf(elf, config, isa=isa, timeout=timeout)
    if console is None:
        return None
    reading = spec.read(console, M, N, K)
    if reading.state == _UNPARSEABLE:
        message = (
            f"{spec.bench}: measured the requested size but its console could not be read "
            f"-> ceiling UNKNOWN (not zero): {reading.detail}"
        )
        if diagnostics is not None:
            diagnostics.append(message)
        print(f"kernel-bench: {message}", file=sys.stderr)
        return None
    if reading.state != _PARSED or reading.cycles is None:
        return None
    cycles, instr = reading.cycles, reading.instructions

    regime = shape_regime(op, M, N, K)
    row = {
        "op": op,
        "dtype": dtype,
        "M": M,
        "N": N,
        "K": K,
        "shape_regime": regime,
        "source": corpus,
        "target": target,
        "bench": spec.bench,
        "kernel_ref": spec.kernel_ref,
        "cycles": int(cycles),
        "isa": isa,
        "fingerprint_key": fingerprint_key(op, dtype, regime),
        "note": f"{corpus} {spec.bench} on spike; read {reading.note}",
    }
    if instr is not None:
        row["instructions"] = int(instr)
    return row


def fingerprint_key(op: str, dtype: str, regime: str) -> dict[str, str]:
    """The 1:1 join key to merlin.kernels.compare.RvvFingerprint (``{op,dtype,shape_regime}``)."""
    return {"op": op, "dtype": dtype, "shape_regime": regime}


# ------------------------------------------------------------------------- jsonl store
def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def append_ceiling(row: dict, path: str | Path = DEFAULT_CEILING_PATH) -> Path:
    """Append one ceiling row as a JSON line, creating the file/dir as needed."""
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def load_ceiling(path: str | Path = DEFAULT_CEILING_PATH) -> list[dict]:
    """Load all ceiling rows (skipping blank lines); empty list when the file is absent."""
    out = _resolve(path)
    if not out.is_file():
        return []
    rows: list[dict] = []
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def attainment(ceiling_row: dict, compiler_cycles: float) -> float | None:
    """attainment = ceiling_cycles / compiler_cycles.

    1.0 == our compiler matched the expert ceiling; < 1.0 == the expert is faster (we are
    leaving performance on the table); > 1.0 == we beat the expert. Returns ``None`` when
    inputs are missing or ``compiler_cycles`` is non-positive.
    """
    ceil = ceiling_row.get("cycles") if isinstance(ceiling_row, dict) else None
    if ceil is None or compiler_cycles is None or compiler_cycles <= 0:
        return None
    return float(ceil) / float(compiler_cycles)


def find_ceiling(rows: Iterable[dict], op: str, dtype: str, regime: str) -> dict | None:
    """First ceiling row whose fingerprint key matches (op, dtype, shape_regime)."""
    want = fingerprint_key(op, dtype, regime)
    for r in rows:
        if r.get("fingerprint_key") == want:
            return r
    return None


# ------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kernel-bench", description="Measure a curated kernel's spike cycle ceiling (S4.2)."
    )
    default_corpus = build_asm.benchmark_source()
    ap.add_argument(
        "--source", default=default_corpus, help=f"curated standalone-benchmark corpus (default: {default_corpus})"
    )
    ap.add_argument(
        "--target",
        default=None,
        help="target whose kernel_ceiling.yaml configures the benches (default: the owner "
        "the corpus registry declares for --source)",
    )
    ap.add_argument(
        "--bench",
        required=True,
        help="benchmark name: a `benches` key of the selected config (else matched by --op/--dtype)",
    )
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    ap.add_argument("--op", default="", help="override op (else from the bench config)")
    ap.add_argument("--dtype", default="", help="override dtype (else from the bench config)")
    ap.add_argument("--isa", default=None, help="spike ISA (default: the selected config's)")
    ap.add_argument(
        "--out", default=DEFAULT_CEILING_PATH, help="ceiling jsonl path (relative paths resolve under repo root)"
    )
    ap.add_argument(
        "--compiler-cycles", type=float, default=None, help="if given, also print attainment = ceiling/compiler cycles"
    )
    ap.add_argument("--no-append", action="store_true", help="measure + print only; do not write the jsonl")
    args = ap.parse_args(argv)

    try:
        config = load_bench_config(args.target) if args.target else bench_config_for_source(args.source)
    except (BenchConfigMissing, BenchConfigError) as exc:
        print(f"kernel-bench: {exc}")
        return 2
    if config is None:
        print(
            f"kernel-bench: corpus {args.source!r} is not a standalone-benchmark corpus with an "
            f"owning target (src/merlin/resources/contracts/corpora.yaml); no bench config to measure with."
        )
        return 2
    if not corpus_available(config):
        from ..targetgen.corpora import kernel_corpus_env

        print(
            f"kernel-bench: spike/riscv-gcc/{config.corpus}-corpus/encoding.h unavailable; "
            f"cannot measure a ceiling (set MERLIN_CHIPYARD / {kernel_corpus_env(config.corpus)})."
        )
        return 2

    diagnostics: list[str] = []
    row = run_kernel_ceiling(
        config.corpus,
        args.bench,
        args.op,
        args.dtype,
        (args.M, args.N, args.K),
        isa=args.isa,
        diagnostics=diagnostics,
        config=config,
    )
    if row is None:
        if diagnostics:
            # The bench ran and measured this size; only reading it back failed. Say which,
            # rather than blaming the build or the size.
            print(
                f"kernel-bench: UNKNOWN ceiling for bench={args.bench} "
                f"(M,N,K)=({args.M},{args.N},{args.K}): " + "; ".join(diagnostics)
            )
        else:
            print(
                f"kernel-bench: no ceiling for bench={args.bench} "
                f"(M,N,K)=({args.M},{args.N},{args.K}) — build/run failed or size not measured."
            )
        return 1

    print(json.dumps(row, sort_keys=True))
    if not args.no_append:
        out = append_ceiling(row, args.out)
        print(f"appended -> {out}")
    if args.compiler_cycles is not None:
        att = attainment(row, args.compiler_cycles)
        print(
            f"attainment (ceiling/compiler) = {att:.4f}  "
            f"(ceiling={row['cycles']} cycles, compiler={args.compiler_cycles:g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
