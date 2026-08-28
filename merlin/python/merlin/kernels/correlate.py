"""Join a model region to the instructions it became — and refuse to fake it when we cannot.

THE CHAIN, AND WHERE IT BREAKS. ``prov.region_id`` / ``prov.fqn`` are stamped at capture and survive
a long way: model op -> dispatch (``outline._prov``) -> kernel symbol, which encodes the region id in
its name (``forward$kernel_3__rmatmul_7``) -> the runtime program's ``KernelEntry`` -> measured ticks
joined by ``fqn``. It survives the int8 and matrix-unit rewrites, which split one captured op into a
contraction and a requant epilogue and keep both attributable by adding ``prov.role``.

It breaks at ONE link: symbol -> instruction range. There is no MLIR ``Location`` threading and no
debug info, so once the emitted image is monolithic the only compute symbol is ``forward`` and every
instruction in the model belongs to it.

This module closes that link where the symbols exist -- a per-kernel build does carry them -- and
REPORTS the monolithic case instead of returning a single all-encompassing mapping, which would be
technically true and read as success. `attributable` is the field that distinguishes the two.

⚠️ This claims a REGION-granular attribution, not a per-op or per-pass one. Knowing which
instructions came from ``matmul_7`` is not knowing which pass chose their tile size; see
``kernels.asm_provenance``, which is explicit that it attributes a decision to a SEAM and cannot
localize a codegen bug.
"""
from __future__ import annotations

from dataclasses import dataclass


def symbol_of(section: str) -> str | None:
    """The symbol named in an objdump section header, e.g. ``10002000 <main>`` -> ``main``.

    Structural, not a pattern: split on the delimiters objdump actually emits. A header with no
    angle-bracketed name (``.text``) is a section, not a symbol, and yields None.
    """
    _, sep, rest = (section or "").partition("<")
    if not sep:
        return None
    name, close, _ = rest.partition(">")
    return name if close and name else None


@dataclass(frozen=True)
class SymbolSpan:
    """The contiguous instruction span attributed to one emitted symbol."""

    symbol: str
    #: index range into the decoded stream, end-exclusive.
    first_index: int
    last_index: int
    #: byte addresses, end-INCLUSIVE of the last instruction's address.
    low_addr: int
    high_addr: int
    n_insns: int
    #: the model region this symbol was tagged with, when the outliner encoded one.
    region_id: str | None = None


@dataclass
class Attribution:
    """Which instructions belong to which region, and whether that question was answerable."""

    spans: tuple[SymbolSpan, ...] = ()
    #: False when the image cannot support per-region attribution (see ``reason``).
    attributable: bool = False
    reason: str = ""

    def by_region(self) -> dict[str, SymbolSpan]:
        """region_id -> span, for the spans that carry one. Empty when not attributable."""
        if not self.attributable:
            return {}
        return {s.region_id: s for s in self.spans if s.region_id}

    def region_of_index(self, index: int) -> str | None:
        """Which region the instruction at stream position ``index`` came from, if any."""
        if not self.attributable:
            return None
        for s in self.spans:
            if s.first_index <= index < s.last_index:
                return s.region_id
        return None


def attribute(raws) -> Attribution:
    """Group a decoded stream into per-symbol spans and recover each symbol's region id.

    ``raws`` are :class:`kernels.decode.objdump.RawInsn`, which carry the enclosing section header.
    """
    from merlin.xdsl_dialects.lowering.outline import region_id_of_symbol

    spans: list[SymbolSpan] = []
    cur_sym: str | None = None
    start = 0
    for i, r in enumerate(list(raws) + [None]):
        sym = symbol_of(getattr(r, "section", "")) if r is not None else None
        if sym != cur_sym:
            if cur_sym is not None and i > start:
                window = list(raws)[start:i]
                addrs = [int(getattr(w, "addr", 0)) for w in window]
                spans.append(SymbolSpan(
                    symbol=cur_sym, first_index=start, last_index=i,
                    low_addr=min(addrs) if addrs else 0,
                    high_addr=max(addrs) if addrs else 0,
                    n_insns=len(window),
                    region_id=region_id_of_symbol(cur_sym)))
            cur_sym, start = sym, i

    if not spans:
        return Attribution(reason="no symbol headers in the disassembly: nothing to attribute to")

    tagged = [s for s in spans if s.region_id]
    if not tagged:
        # The honest monolithic case. Returning the single span with attributable=True would be
        # literally correct and would read as "we can attribute instructions to regions", which is
        # the claim this refuses to make.
        return Attribution(
            spans=tuple(spans), attributable=False,
            reason=(f"{len(spans)} symbol(s), none carrying a region id — a monolithic image "
                    f"attributes every instruction to one symbol, which is not an attribution. "
                    f"Build per-kernel symbols, or slice the model by region and rebuild."))
    return Attribution(spans=tuple(spans), attributable=True,
                       reason=f"{len(tagged)}/{len(spans)} symbol(s) carry a region id")
