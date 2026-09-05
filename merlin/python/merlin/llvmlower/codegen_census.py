"""Post-codegen census: did the model that went IN to the backend come OUT of it?

A build that emits an EMPTY entry point still links, still produces a binary, and still reports
success — and on a board it then reports a spectacular speedup for computing nothing. That is not
hypothetical. ``smolvla_int8_consistent`` linked a 512 MB ELF whose ``forward`` was 3,654 bytes with
a complete external call set of ``malloc``/``memset``/``roundevenf``: a ~500M-parameter
vision-language-action model, forty times SMALLER than a 9 MB LSTM/ViT bundle built minutes earlier
from the same feature set. The cause was upstream (a float->i1 ``arith.fptosi`` is poison in LLVM,
and a branch on that poison let ``simplifycfg`` delete every block after it), but nothing between
the lowering and the timing run could SEE that the model was gone.

The check here is the missing sight. It is a *census*, not a threshold:

* the obligation count comes from the PREPARED IR — the module lowering actually receives — as the
  number of ``linalg`` structured ops whose result transitively reaches a ``func.return``. Ops
  whose results feed nothing are excluded, so a legitimately dead op is never charged for;
* the delivery count comes from the OBJECT — instructions actually emitted for the entry symbol,
  disassembled by the LLVM objdump from the same install that compiled it;
* the rule needs no constant: **a computation that reaches the output cannot be performed in zero
  instructions**, so ``emitted_instructions >= live_structured_ops``. Nothing is hardcoded, nothing
  is per-target, and the bound is loose by orders of magnitude — a real build clears it by ~20x
  (lstmnetvit: 39,875 instructions for 2,000-odd live ops), while an erased one misses it by 3x.

FAIL CLOSED. A missing objdump, an absent symbol, an object that disassembles to nothing, and an
unparseable prepared module are all REFUSALS, never skips: this file exists because a check that
could not fail reported success. "Unparseable" means unparseable in BOTH MLIR forms -- see
:func:`live_structured_ops_in_file`; a module the MLIR printer left in custom form is normalized,
not refused, because refusing it would make this census a build-breaker for every model containing
an op xDSL has no custom-format parser for.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class CodegenCensusError(RuntimeError):
    """The emitted code is not commensurate with the model that was compiled."""


@dataclass(frozen=True)
class Census:
    live_structured_ops: int
    total_structured_ops: int
    emitted_instructions: int       # over the whole object (outlined helpers included)
    entry_instructions: int         # the entry symbol alone
    symbol: str

    @property
    def ratio(self) -> float:
        return self.emitted_instructions / self.live_structured_ops if self.live_structured_ops else 0.0

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "live_structured_ops": self.live_structured_ops,
                "total_structured_ops": self.total_structured_ops,
                "emitted_instructions": self.emitted_instructions,
                "entry_instructions": self.entry_instructions,
                "instructions_per_live_op": round(self.ratio, 3)}


def _defining_op(value):
    """The op that defines ``value`` (None for a block argument)."""
    owner = getattr(value, "owner", None)
    return owner if owner is not None and hasattr(owner, "operands") else None


def live_structured_ops(module) -> tuple[int, int]:
    """``(live, total)`` counts of ``linalg`` structured ops in ``module``.

    ``live`` = reachable backwards from a ``func.return`` operand through SSA def-use. This is the
    honest obligation count: an op whose result feeds nothing MAY legitimately be deleted by the
    backend, so charging codegen for it would make the census refuse correct builds.
    """
    total = 0
    for op in module.walk():
        if op.name.startswith("linalg.") and op.results:
            total += 1

    seen: set[int] = set()
    live_ops: set[int] = set()
    stack = []
    for op in module.walk():
        if op.name == "func.return":
            stack.extend(op.operands)
    while stack:
        val = stack.pop()
        if id(val) in seen:
            continue
        seen.add(id(val))
        src = _defining_op(val)
        if src is None:
            continue
        if src.name.startswith("linalg.") and src.results:
            live_ops.add(id(src))
        # `src.walk()`, not `src.operands`: an op's REGIONS may capture outer SSA values that never
        # appear in its operand list -- a `tensor.extract` inside a linalg.generic body is how a
        # gather reads its source table. Walking only the operand list severs the chain there and
        # under-counts the live set by an order of magnitude (measured on a conv autoencoder:
        # 20 live ops instead of 354, which would have left the census blind on that model).
        for inner in src.walk():
            stack.extend(inner.operands)
    return len(live_ops), total


def live_structured_ops_in_file(prepared_mlir: str | Path) -> tuple[int, int]:
    """``live_structured_ops`` of the module on disk, in whichever form MLIR printed it.

    The module lowering receives is NOT always the generic-form text a capture ships. Two prepared-
    module rewrites (`perop_blocks.tag_prepared_mlir`, `prov_cse.rewrite_prepared_file`) round-trip
    it through the MLIR printer, which prints CUSTOM form -- and xDSL, whose tensor dialect has no
    custom-format parser for `tensor.extract_slice` (nor for the next op nobody has hit yet),
    refuses the whole module. MEASURED: that refusal blocked every tiny_llama int8 build carrying
    `perop_register_block` / `cse_through_provenance` the day this census started reading the
    prepared module. `parse_mlir_file_any_form` re-prints through the same interpreter that did the
    printing and parses that; it is not a tolerance for unparseable input -- an un-normalizable
    module still raises, because a census that quietly counted zero live ops would pass every
    erased model, which is the exact failure this file exists to catch.
    """
    from .generic_form import parse_mlir_file_any_form
    return live_structured_ops(parse_mlir_file_any_form(Path(prepared_mlir)))


def disassembly_census(obj: str | Path, objdump: str | Path | None = None) -> dict[str, int]:
    """``{symbol: instruction count}`` for every function llvm-objdump decodes in ``obj``.

    Parsed structurally, not by pattern. llvm-objdump prints a function banner as
    ``<hex address> <name>:`` and one instruction per line as
    ``<hex address>:\t<encoding>\t<mnemonic> ...``, so a banner is a line whose last character is
    ``:`` with a ``<name>`` between angle brackets, and an instruction is a line whose text before
    the first ``:`` parses as hexadecimal and whose remainder contains a tab. Neither test assumes
    a mnemonic spelling, an encoding width, or an instruction set.
    """
    from . import toolchain

    tool = Path(objdump) if objdump is not None else toolchain.objdump()
    if not Path(tool).is_file():
        raise CodegenCensusError(
            f"post-codegen census cannot run: no llvm-objdump at {tool} (set MERLIN_OBJDUMP). "
            "Refusing rather than passing a build nothing inspected.")
    proc = subprocess.run([str(tool), "-d", str(obj)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise CodegenCensusError(f"llvm-objdump failed on {obj}: {proc.stderr[-2000:]}")
    counts: dict[str, int] = {}
    current = ""
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and "<" in stripped and ">" in stripped:
            inner = stripped[stripped.index("<") + 1:stripped.rindex(">")]
            if inner:
                current = inner
                counts.setdefault(current, 0)
                continue
        head, sep, rest = line.partition(":")
        if not sep or "\t" not in rest:
            continue
        head = head.strip()
        if not head:
            continue
        try:
            int(head, 16)
        except ValueError:
            continue
        counts[current] = counts.get(current, 0) + 1
    return counts


def census(prepared_mlir: str | Path, obj: str | Path, symbol: str,
           objdump: str | Path | None = None) -> Census:
    """Census of ``obj`` against the prepared module, keyed to entry point ``symbol``.

    The delivery count is the instruction total over the WHOLE object, not over ``symbol`` alone:
    outlining is legitimate and routine — an OpenMP-parallel lowering leaves ``forward`` a 31-
    instruction shell that calls ``forward..omp_par``, so charging only the entry symbol would
    refuse every correct multicore build. ``symbol`` must still be DEFINED in the object; an entry
    point that vanished is the failure, not a technicality.
    """
    live, total = live_structured_ops_in_file(prepared_mlir)
    per_symbol = disassembly_census(obj, objdump)
    if symbol not in per_symbol:
        raise CodegenCensusError(
            f"post-codegen census: entry point {symbol!r} is not defined in {obj} "
            f"(symbols: {sorted(per_symbol)[:12]}). Refusing the build.")
    return Census(live_structured_ops=live, total_structured_ops=total,
                  emitted_instructions=sum(per_symbol.values()),
                  entry_instructions=per_symbol[symbol], symbol=str(symbol))


def require_commensurate(prepared_mlir: str | Path, obj: str | Path, symbol: str,
                         objdump: str | Path | None = None) -> Census:
    """Raise :class:`CodegenCensusError` unless the emitted entry point can plausibly BE the model.

    Returns the :class:`Census` so a caller can record it on the build artifact.
    """
    c = census(prepared_mlir, obj, symbol, objdump)
    if c.emitted_instructions == 0:
        raise CodegenCensusError(
            f"post-codegen census: {symbol} in {obj} disassembles to NO instructions "
            f"(prepared IR has {c.live_structured_ops} live structured ops). The object cannot be "
            "the model; refusing the build.")
    if c.live_structured_ops and c.emitted_instructions < c.live_structured_ops:
        raise CodegenCensusError(
            f"post-codegen census: {symbol} emitted {c.emitted_instructions} instructions for "
            f"{c.live_structured_ops} live structured ops in the prepared IR "
            f"({c.total_structured_ops} total) — fewer than one instruction per op that reaches an "
            "output, so compute the model requires is not in the object. The usual cause is poison "
            "or undefined behaviour in the lowered IR letting the backend delete reachable code "
            "(see fix_bool_fptosi). Refusing the build rather than timing an empty kernel.")
    return c
