"""Derive a target SDK's console/clock facts from that SDK's *own* headers.

Why this module exists, concretely: our bare-metal images printed over HTIF, a **host-assisted**
channel — the program writes ``tohost`` and spins until a host debugger (fesvr, inside spike /
FireSim / uart_tsi) clears it. Every substrate we could test on has such a host, so HTIF passed
everywhere and was wrong in exactly one place: **bare silicon**, where nothing clears ``tohost`` and
the second character spins forever inside the first print, before any model work. From the outside
that is indistinguishable from a core that never booted, which is what it looked like to the people
running our binaries.

Talking to the real console instead needs a fistful of chip facts — the UART base, its register
offsets, which bit says "TX FIFO full", the clock the baud divisor is computed against, and (because
raising the PLL changes that clock) the PLL and clock-selector registers. Those are precisely the
kind of facts this repo forbids hardcoding: a literal ``0x10020000`` in library code would be a fact
about one tapeout baked into a shared path, and it would be silently wrong for the next chip.

So they are **extracted from the SDK's own headers at build time** and injected into the harness as
macros. The parsing is structural (a tokenizer + recursive-descent evaluator over ``#define`` bodies,
field order for register offsets) rather than line-matching, because a too-narrow pattern drops
valid-but-differently-spelled input silently — the failure mode the no-regex rule exists to prevent.
Two independent checks guard the result: struct offsets are cross-checked against the trailing
``// 0x..`` comments the vendor headers carry, and every fact is mandatory — a value that cannot be
derived raises rather than falling back to a default, because a wrong console address is a silent
hang and a defaulted one is a wrong answer nobody can see.

The derivation was validated against a **working vendor binary** (a whisper ELF built by their own
flow): the values this module extracts — UART base ``0x10020000``, clock selector ``0x130000``, PLL
``0x140000``, baud ``115200``, ``DIV`` at byte offset 24, system clock 50 MHz — are the same ones
that appear in that binary's disassembly of ``init_test``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Byte sizes for the scalar types these register-map structs are built from. A field whose type is
#: not here is an error rather than an assumed width -- guessing a size shifts every offset after it.
TYPE_SIZES = {
    "int8_t": 1, "uint8_t": 1, "char": 1,
    "int16_t": 2, "uint16_t": 2, "short": 2,
    "int32_t": 4, "uint32_t": 4, "int": 4, "unsigned": 4,
    "int64_t": 8, "uint64_t": 8, "long": 8,
}

#: Console baud. One constant shared by the bare-metal harness and the RTOS config so the two cannot
#: disagree -- and 115200 specifically because it is what the target SDK's own bring-up uses, i.e.
#: what a terminal already pointed at that board is set to.
DEFAULT_BAUD = 115200

#: Qualifiers to skip when reading a field declaration. ``__IO``/``__I``/``__O`` are the CMSIS-style
#: volatile aliases these SDKs use.
QUALIFIERS = frozenset({"__IO", "__I", "__O", "__IM", "__OM", "__IOM", "volatile", "const"})

_PUNCT = ("<<", ">>", "&&", "||", "==", "!=", "<=", ">=")
_SINGLE = set("()<>|&+-*/~^%!")


class SdkFactError(RuntimeError):
    """A fact could not be derived from the SDK sources. Never downgraded to a default."""


# --------------------------------------------------------------------------------- lexing / eval --
def strip_comments(text: str) -> str:
    """Remove C comments, preserving newlines so line structure survives.

    Done by scanning states rather than matching patterns: a ``//`` inside a string literal is not a
    comment, and getting that wrong silently deletes code.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and nxt == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
        elif c in "\"'":
            quote = c
            out.append(c)
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i])
                    i += 1
                out.append(text[i])
                i += 1
            if i < n:
                out.append(text[i])
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(expr: str) -> list[str]:
    """Split a C constant expression into identifiers, integer literals and operators."""
    toks: list[str] = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            if c == "0" and i + 1 < n and expr[i + 1] in "xX":
                j = i + 2
                while j < n and (expr[j].isdigit() or expr[j].lower() in "abcdef"):
                    j += 1
            else:
                while j < n and expr[j].isdigit():
                    j += 1
            lit = expr[i:j]
            while j < n and expr[j] in "uUlL":       # integer suffixes: 0x10020000U, 500000000ULL
                j += 1
            toks.append(lit)
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            toks.append(expr[i:j])
            i = j
            continue
        if expr[i:i + 2] in _PUNCT:
            toks.append(expr[i:i + 2])
            i += 2
            continue
        if c in _SINGLE:
            toks.append(c)
            i += 1
            continue
        raise SdkFactError(f"cannot tokenize {c!r} in {expr!r}")
    return toks


class _Eval:
    """Recursive-descent evaluator for the integer subset of C constant expressions.

    Handles what vendor register headers actually contain: hex/decimal literals with U/L suffixes,
    shifts and bitwise ops, parentheses, references to other ``#define``s, and casts -- including
    pointer casts like ``((ClockSel_Type*)(RCC_BASE + 0x30000))``, which is how the clock-selector
    base is spelled on one of these chips.

    A parenthesised identifier is a **cast** only when it is not itself a known define: ``(UART_BASE)``
    is a value, ``(uint32_t)`` is a cast. That single rule keeps both spellings working without a
    hardcoded list of type names.
    """

    def __init__(self, defines: dict[str, str], seen: frozenset[str] = frozenset()):
        self.defines, self.seen = defines, seen
        self.toks: list[str] = []
        self.pos = 0

    def run(self, expr: str) -> int:
        self.toks, self.pos = tokenize(expr), 0
        val = self._or()
        if self.pos != len(self.toks):
            raise SdkFactError(f"trailing tokens in {expr!r} at {self.toks[self.pos:]}")
        return val

    def _peek(self, k: int = 0) -> str | None:
        j = self.pos + k
        return self.toks[j] if j < len(self.toks) else None

    def _take(self, tok: str) -> None:
        if self._peek() != tok:
            raise SdkFactError(f"expected {tok!r}, got {self._peek()!r}")
        self.pos += 1

    def _binary(self, ops: tuple[str, ...], sub) -> int:
        val = sub()
        while self._peek() in ops:
            op = self.toks[self.pos]
            self.pos += 1
            rhs = sub()
            if op == "|":
                val |= rhs
            elif op == "&":
                val &= rhs
            elif op == "^":
                val ^= rhs
            elif op == "<<":
                val <<= rhs
            elif op == ">>":
                val >>= rhs
            elif op == "+":
                val += rhs
            elif op == "-":
                val -= rhs
            elif op == "*":
                val *= rhs
            elif op == "/":
                val //= rhs
            elif op == "%":
                val %= rhs
        return val

    def _or(self) -> int:
        return self._binary(("|",), self._xor)

    def _xor(self) -> int:
        return self._binary(("^",), self._and)

    def _and(self) -> int:
        return self._binary(("&",), self._shift)

    def _shift(self) -> int:
        return self._binary(("<<", ">>"), self._add)

    def _add(self) -> int:
        return self._binary(("+", "-"), self._mul)

    def _mul(self) -> int:
        return self._binary(("*", "/", "%"), self._unary)

    def _unary(self) -> int:
        tok = self._peek()
        if tok == "-":
            self.pos += 1
            return -self._unary()
        if tok == "+":
            self.pos += 1
            return self._unary()
        if tok == "~":
            self.pos += 1
            return ~self._unary()
        return self._primary()

    def _is_cast(self) -> bool:
        """``( IDENT [*...] )`` where IDENT is not a known define -> a type cast, not a value."""
        if self._peek() != "(":
            return False
        ident = self._peek(1)
        if ident is None or not (ident[0].isalpha() or ident[0] == "_"):
            return False
        if ident in self.defines:
            return False
        k = 2
        while self._peek(k) == "*":
            k += 1
        return self._peek(k) == ")"

    def _primary(self) -> int:
        tok = self._peek()
        if tok is None:
            raise SdkFactError("unexpected end of expression")
        if self._is_cast():
            while self._peek() != ")":
                self.pos += 1
            self._take(")")
            return self._unary()
        if tok == "(":
            self.pos += 1
            val = self._or()
            self._take(")")
            return val
        self.pos += 1
        if tok[0].isdigit():
            return int(tok, 16) if tok[:2].lower() == "0x" else int(tok, 10)
        if tok in self.defines:
            if tok in self.seen:
                raise SdkFactError(f"cyclic #define reference through {tok}")
            return _Eval(self.defines, self.seen | {tok}).run(self.defines[tok])
        raise SdkFactError(f"unknown identifier {tok!r} in a constant expression")


def parse_defines(text: str) -> dict[str, str]:
    """``{name: body}`` for every object-like ``#define`` in ``text``.

    Function-like macros (a ``(`` touching the name) are skipped: they are not constants, and a
    parameterised body would evaluate to nonsense.
    """
    src = strip_comments(text)
    # Join backslash continuations so a multi-line define stays one define.
    src = src.replace("\\\n", " ")
    out: dict[str, str] = {}
    for line in src.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        body = line[1:].lstrip()
        if not body.startswith("define"):
            continue
        rest = body[len("define"):]
        if not rest[:1].isspace():
            continue
        rest = rest.strip()
        if not rest:
            continue
        head, _, value = rest.partition(" ")
        if "(" in head:                      # function-like macro
            continue
        name = head.strip()
        value = value.strip()
        if name and value:
            out[name] = value
    return out


def eval_define(name: str, defines: dict[str, str]) -> int:
    """Integer value of ``#define name``. Raises if absent or not an integer constant."""
    if name not in defines:
        raise SdkFactError(f"#define {name} not found")
    try:
        return _Eval(defines).run(defines[name])
    except SdkFactError as exc:
        raise SdkFactError(f"cannot evaluate #define {name} = {defines[name]!r}: {exc}") from exc


# ---------------------------------------------------------------------------- struct / enum maps --
def _find_typedef_body(text: str, kind: str, name: str) -> str:
    """Body of ``typedef <kind> { ... } <name>;``, located by brace matching."""
    needle = f"typedef {kind}"
    at = 0
    while True:
        at = text.find(needle, at)
        if at < 0:
            raise SdkFactError(f"typedef {kind} ... {name} not found")
        open_brace = text.find("{", at)
        if open_brace < 0:
            raise SdkFactError(f"malformed typedef {kind} near {name}")
        depth, i = 1, open_brace + 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        tail = text[i:text.find(";", i)] if ";" in text[i:] else ""
        if tail.strip().rstrip("*").strip() == name:
            return text[open_brace + 1:i - 1]
        at = i


def _field_name(decl: str) -> str | None:
    """Field name of a single declaration, or None if it is not one."""
    toks = [t for t in decl.replace("*", " ").split() if t not in QUALIFIERS]
    return toks[-1] if len(toks) >= 2 and toks[-2] in TYPE_SIZES else None


def _stated_offsets(raw: str) -> dict[str, int]:
    """``{field: offset}`` for fields whose line carries an explicit ``// 0x..`` offset comment.

    Kept separate from the offset computation on purpose: this is the *independent* number, and
    comparing two derivations is what makes a misread register map loud instead of silent.
    """
    out: dict[str, int] = {}
    for line in raw.splitlines():
        decl, comment = line, ""
        if "//" in line:
            decl, _, comment = line.partition("//")
        elif "/*" in line:
            decl, _, rest = line.partition("/*")
            comment = rest.partition("*/")[0]
        if ";" not in decl:
            continue
        head = comment.strip().split()[0] if comment.strip() else ""
        if head[:2].lower() != "0x":
            continue
        name = _field_name(decl.partition(";")[0])
        if name is None:
            continue
        try:
            out[name] = int(head, 16)
        except ValueError:
            continue
    return out


def parse_struct_offsets(text: str, struct: str) -> dict[str, int]:
    """``{field: byte_offset}`` for a register-map struct, from declaration order.

    Offsets come from field widths with natural alignment, and are **cross-checked against the
    header's own ``// 0x..`` offset comments** wherever it carries them (these vendor headers do for
    the PLL and clock-selector maps). A disagreement raises: it means the layout was misread, and a
    misread register map writes to the wrong address -- on a console, a silent hang.

    Comments are stripped before the fields are split, because a ``/** ... */`` description sitting
    after a ``;`` belongs to the line before it and would otherwise be read as part of the *next*
    declaration -- which silently dropped every second register the first time around.
    """
    raw = _find_typedef_body(text, "struct", struct)
    stated = _stated_offsets(raw)
    offsets: dict[str, int] = {}
    cursor = 0
    for stmt in strip_comments(raw).split(";"):
        decl = stmt.strip()
        if not decl:
            continue
        toks = [t for t in decl.replace("*", " ").split() if t not in QUALIFIERS]
        if len(toks) < 2:
            raise SdkFactError(f"cannot read field declaration {decl!r} of {struct}")
        fname = toks[-1]
        if "[" in fname:                       # arrays would need an extent; none in these maps
            raise SdkFactError(f"array field {fname!r} in {struct} is not supported")
        ctype = toks[-2]
        if ctype not in TYPE_SIZES:
            raise SdkFactError(f"unknown field type {ctype!r} for {struct}.{fname}")
        size = TYPE_SIZES[ctype]
        if cursor % size:
            cursor += size - (cursor % size)
        if fname in stated and stated[fname] != cursor:
            raise SdkFactError(
                f"{struct}.{fname}: computed offset {hex(cursor)} disagrees with the header's own "
                f"{hex(stated[fname])} -- the register map was misread")
        offsets[fname] = cursor
        cursor += size
    if not offsets:
        raise SdkFactError(f"no fields parsed for {struct}")
    return offsets


def parse_enum(text: str, enum: str) -> dict[str, int]:
    """``{name: value}`` for a C enum, honouring explicit values and implicit increments."""
    raw = strip_comments(_find_typedef_body(text, "enum", enum))
    out: dict[str, int] = {}
    nxt = 0
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, eq, value = item.partition("=")
        name = name.strip()
        if not name:
            continue
        if eq:
            out[name] = _Eval({}).run(value.strip())
        else:
            out[name] = nxt
        nxt = out[name] + 1
    if not out:
        raise SdkFactError(f"no enumerators parsed for {enum}")
    return out


# ------------------------------------------------------------------------------------- the facts --
@dataclass(frozen=True)
class UartConsoleFacts:
    """Everything the harness needs to speak a chip's own console, all of it derived."""

    uart_base: int
    reg: dict[str, int]                 # TXDATA / RXDATA / TXCTRL / RXCTRL / DIV byte offsets
    tx_full_bit: int
    txen_bit: int
    rxen_bit: int
    nstop_bit: int
    sys_clk_hz: int
    #: CLINT/mtime timebase. Not a detail: an RTOS derives its peripheral clock from the mtime rate
    #: times a divider, so a wrong value here sets a wrong baud divisor and the console emits garbage
    #: rather than nothing -- which reads as a corrupt program instead of a misconfigured UART.
    mtime_hz: int
    pll_base: int
    pll: dict[str, int]                 # PLL register byte offsets
    clksel_base: int
    clksel: dict[str, int]              # clock-selector register byte offsets
    clksel_slow: int
    clksel_pll: int
    provenance: dict[str, str] = field(default_factory=dict)

    def macros(self, *, baud: int = DEFAULT_BAUD, stopbits: int = 2,
               chip_freq_hz: int | None = None) -> list[str]:
        """``-D`` flags injecting these facts into ``console_uart.c``.

        ``chip_freq_hz`` None means "do not touch the PLL": the image runs at the chip's reset clock
        and the baud divisor is computed against that. Set it to raise the PLL first, mirroring the
        SDK's own ``init_test`` -- which also rewrites the divisor, because the divisor is relative
        to a clock the PLL just changed.
        """
        need_reg = ("TXDATA", "TXCTRL", "RXCTRL", "DIV")
        missing = [r for r in need_reg if r not in self.reg]
        if missing:
            raise SdkFactError(f"UART register map lacks {missing}")
        need_pll = ("PLLEN", "MDIV_RATIO", "RATIO", "FRACTION", "ZDIV0_RATIO", "ZDIV1_RATIO",
                    "LDO_ENABLE", "POWERGOOD_VNN", "PLLFWEN_B")
        defs = [
            f"-DMERLIN_UART_BASE={hex(self.uart_base)}ULL",
            f"-DMERLIN_UART_TXDATA_OFF={self.reg['TXDATA']}",
            f"-DMERLIN_UART_TXCTRL_OFF={self.reg['TXCTRL']}",
            f"-DMERLIN_UART_RXCTRL_OFF={self.reg['RXCTRL']}",
            f"-DMERLIN_UART_DIV_OFF={self.reg['DIV']}",
            f"-DMERLIN_UART_TX_FULL_BIT={self.tx_full_bit}",
            f"-DMERLIN_UART_TXEN_BIT={self.txen_bit}",
            f"-DMERLIN_UART_RXEN_BIT={self.rxen_bit}",
            f"-DMERLIN_UART_NSTOP_BIT={self.nstop_bit}",
            f"-DMERLIN_UART_STOPBITS={stopbits}",
            f"-DMERLIN_UART_BAUD={baud}",
            f"-DMERLIN_SYS_CLK_HZ={self.sys_clk_hz}ULL",
            # No shell is involved (these go to subprocess as a list), so the quotes must NOT be
            # backslash-escaped -- gcc would see the backslashes as part of the string literal.
            '-DMERLIN_CONSOLE_NAME="uart"',
        ]
        if chip_freq_hz:
            missing_pll = [r for r in need_pll if r not in self.pll]
            if missing_pll:
                raise SdkFactError(f"PLL register map lacks {missing_pll}")
            # The harness walks the clock-selector domains as one contiguous array of 32-bit
            # registers (that is what the SDK's set_all_clocks does). Verify the derived map really
            # is contiguous rather than assuming it: a gap would send a write to a neighbour.
            want = list(range(0, 4 * len(self.clksel), 4))
            if sorted(self.clksel.values()) != want:
                raise SdkFactError(
                    f"clock-selector registers are not a contiguous 32-bit array: {self.clksel}")
            defs += [f"-DMERLIN_CHIP_FREQ_HZ={chip_freq_hz}ULL",
                     f"-DMERLIN_PLL_BASE={hex(self.pll_base)}ULL",
                     f"-DMERLIN_CLKSEL_BASE={hex(self.clksel_base)}ULL",
                     f"-DMERLIN_CLKSEL_SLOW={self.clksel_slow}",
                     f"-DMERLIN_CLKSEL_PLL={self.clksel_pll}",
                     f"-DMERLIN_CLKSEL_N={len(self.clksel)}"]
            defs += [f"-DMERLIN_PLL_{r}_OFF={self.pll[r]}" for r in need_pll]
        return defs


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        raise SdkFactError(f"cannot read {path}: {exc}") from exc


def _find_header(sdk: Path, filename: str, *, must_contain: tuple[str, ...],
                 prefer_under: str | None = None) -> Path:
    """Locate a header by name, disambiguated by content rather than by an assumed path.

    These SDKs ship several files called ``uart.h`` (one per vendor driver); picking by path would
    hardcode one SDK's layout, so the caller says what the right one must *contain*.
    """
    hits = [p for p in sorted(sdk.rglob(filename)) if ".git" not in p.parts]
    good = [p for p in hits if all(tok in _read(p) for tok in must_contain)]
    if not good:
        raise SdkFactError(
            f"no {filename} under {sdk} contains all of {must_contain} "
            f"({len(hits)} candidate(s) by name)")
    if prefer_under:
        scoped = [p for p in good if prefer_under in p.parts]
        if scoped:
            return scoped[0]
    return good[0]


def derive_uart_console(sdk_dir: str | Path, chip: str) -> UartConsoleFacts:
    """Derive ``chip``'s console facts from the SDK checkout at ``sdk_dir``.

    ``chip`` selects the platform directory (an SDK covering several tapeouts has one per chip); it
    is a *lookup key into the target's own tree*, not a fact about the chip.
    """
    sdk = Path(sdk_dir).resolve()
    if not sdk.is_dir():
        raise SdkFactError(f"SDK directory not found: {sdk}")

    cfg_path = _find_header(sdk, "chip_config.h", must_contain=("UART",), prefer_under=chip)
    if chip not in cfg_path.parts:
        raise SdkFactError(f"no chip_config.h under a '{chip}' directory in {sdk} (found {cfg_path})")
    cfg = parse_defines(_read(cfg_path))

    uart_path = _find_header(sdk, "uart.h",
                             must_contain=("UART_Type", "UART_TXDATA_FULL_POS"))
    uart_text = _read(uart_path)
    uart_defs = parse_defines(uart_text)

    pll_path = _find_header(sdk, "pll.h", must_contain=("PLL_Type",))
    rcc_path = _find_header(sdk, "hal_rcc.h", must_contain=("ClockSel_Type", "ClockSel_Opts"),
                            prefer_under=chip)
    rcc_text = _read(rcc_path)

    # UART instance base. `UART0_BASE` is the console instance on these chips; it may be spelled as a
    # reference to a generic `UART_BASE`, which the evaluator resolves.
    uart_base = eval_define("UART0_BASE", cfg)
    # The clock selector is derived from RCC_CLOCK_SELECTOR rather than RCC_BASE: on one of these
    # chips the selector sits at an offset inside the RCC block, and the cast-bearing define is the
    # only place that offset is written down.
    clksel_base = eval_define("RCC_CLOCK_SELECTOR", cfg)
    pll_base = eval_define("PLL_BASE", cfg)
    sys_clk = eval_define("SYS_CLK_FREQ", cfg)

    clkopts = parse_enum(rcc_text, "ClockSel_Opts")
    for want in ("CLKSEL_SLOW", "CLKSEL_PLL0"):
        if want not in clkopts:
            raise SdkFactError(f"{rcc_path.name} lacks enumerator {want}")

    return UartConsoleFacts(
        uart_base=uart_base,
        reg=parse_struct_offsets(uart_text, "UART_Type"),
        tx_full_bit=eval_define("UART_TXDATA_FULL_POS", uart_defs),
        txen_bit=eval_define("UART_TXCTRL_TXEN_POS", uart_defs),
        rxen_bit=eval_define("UART_RXCTRL_RXEN_POS", uart_defs),
        nstop_bit=eval_define("UART_TXCTRL_NSTOP_POS", uart_defs),
        sys_clk_hz=sys_clk,
        mtime_hz=eval_define("MTIME_FREQ", cfg),
        pll_base=pll_base,
        pll=parse_struct_offsets(_read(pll_path), "PLL_Type"),
        clksel_base=clksel_base,
        clksel=parse_struct_offsets(rcc_text, "ClockSel_Type"),
        clksel_slow=clkopts["CLKSEL_SLOW"],
        clksel_pll=clkopts["CLKSEL_PLL0"],
        provenance={"chip_config.h": str(cfg_path.relative_to(sdk)),
                    "uart.h": str(uart_path.relative_to(sdk)),
                    "pll.h": str(pll_path.relative_to(sdk)),
                    "hal_rcc.h": str(rcc_path.relative_to(sdk))},
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m merlin.runtime.sdk_facts <sdk-dir> <chip>`` -- print the derived facts."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Derive console/clock facts from an SDK's headers")
    ap.add_argument("sdk_dir")
    ap.add_argument("chip")
    ap.add_argument("--chip-freq", type=int, default=None)
    a = ap.parse_args(argv)

    facts = derive_uart_console(a.sdk_dir, a.chip)
    print(json.dumps({
        "uart_base": hex(facts.uart_base), "reg": facts.reg,
        "tx_full_bit": facts.tx_full_bit, "txen_bit": facts.txen_bit,
        "rxen_bit": facts.rxen_bit, "nstop_bit": facts.nstop_bit,
        "sys_clk_hz": facts.sys_clk_hz, "pll_base": hex(facts.pll_base),
        "clksel_base": hex(facts.clksel_base), "clksel": facts.clksel,
        "clksel_slow": facts.clksel_slow, "clksel_pll": facts.clksel_pll,
        "provenance": facts.provenance,
        "macros": facts.macros(chip_freq_hz=a.chip_freq),
    }, indent=2))
    return 0


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
