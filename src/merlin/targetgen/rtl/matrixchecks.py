"""Cross-check derived matrix encodings against independent headers and opcode tables.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .matrixencoding import MaskedEncoding, MatrixIsa, _field_mask, _put


@dataclass(frozen=True)
class HeaderSite:
    """One ``.insn r`` in a C header, with the bits it spells as literals."""

    site: str  # enclosing macro or function name
    match: int
    mask: int
    template: str


def _asm_template(text: str, start: int) -> str:
    """The asm template beginning at the string literal that contains ``start``.

    Adjacent literals concatenate; a macro call between them (``" VME_V(vd) "``) is an operand spelled at
    preprocessing time and becomes ``@``. Stops at the operand/clobber section or the closing paren.
    """
    q = text.rfind('"', 0, start)
    out: list[str] = []
    i = q + 1
    n = len(text)
    while i < n:
        end = text.find('"', i)
        if end < 0:
            break
        out.append(text[i:end])
        j = end + 1
        while j < n and text[j] in " \t\\\n":
            j += 1
        if j >= n or text[j] in ":);":
            break
        if text[j] == '"':
            i = j + 1
            continue
        # A preprocessor-spelled operand between two literals.
        depth, k = 0, j
        while k < n:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                if depth == 0:
                    break
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            elif text[k] == '"' and depth == 0:
                break
            k += 1
        out.append("@")
        while k < n and text[k] in " \t\\\n":
            k += 1
        if k < n and text[k] == '"':
            i = k + 1
            continue
        break
    return "".join(out)


def _blank_c_comments(text: str) -> str:
    """``text`` with C comments replaced by spaces (newlines kept), so positions stay valid and a
    ``.insn r`` mentioned in prose is not read as an instruction."""
    out = list(text)
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


_C_CONTROL = frozenset({"if", "for", "while", "switch", "do", "else"})


def _enclosing_name(text: str, pos: int) -> str | None:
    """The macro whose definition line, or the function whose body, contains ``pos``."""
    line_start = text.rfind("\n", 0, pos) + 1
    # A macro body may continue across backslash-newlines; walk back to its #define.
    k = line_start
    while True:
        line = text[k : text.find("\n", k) if text.find("\n", k) >= 0 else len(text)].lstrip()
        if line.startswith("#define "):
            head = line[len("#define ") :]
            return head.split("(", 1)[0].split()[0]
        prev_end = k - 1
        if prev_end <= 0 or not text[:prev_end].rstrip(" \t").endswith("\\"):
            break
        k = text.rfind("\n", 0, prev_end) + 1
    # Otherwise the innermost function definition whose brace-balanced body contains pos.
    depth = 0
    i = pos
    while i > 0:
        i -= 1
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                head = text[text.rfind("\n", 0, text.rfind(")", 0, i)) + 1 : i]
                paren = head.find("(")
                if paren > 0 and head[:paren].split():
                    name = head[:paren].split()[-1].lstrip("*")
                    if name.isidentifier() and name not in _C_CONTROL:
                        return name
                    if name in _C_CONTROL:
                        continue  # a nested block: keep walking out to the function
                return None
            depth -= 1
    return None


def header_insn_sites(text: str) -> dict[str, list[HeaderSite]]:
    """``{site: [HeaderSite]}`` for every ``.insn r`` in a C header, keyed by enclosing macro/function.

    The literal fields (opcode, funct3, funct7 and any register written as ``x<n>``) form the site's
    ``(match, mask)``; operands spelled through ``%n`` or a macro stay unmasked.
    """
    out: dict[str, list[HeaderSite]] = {}
    marker = ".insn r"
    text = _blank_c_comments(text)
    pos = text.find(marker)
    while pos >= 0:
        template = _asm_template(text, pos)
        name = _enclosing_name(text, pos)
        body = template[template.find(marker) + len(marker) :] if marker in template else ""
        fields = [f.strip() for f in body.split(",")]
        if name and len(fields) == 6:
            ints = []
            for f in fields[:3]:
                try:
                    ints.append(int(f, 0))
                except ValueError:
                    ints = []
                    break
            if len(ints) == 3:
                match = _put(ints[0], "opcode") | _put(ints[1], "funct3") | _put(ints[2], "funct7")
                mask = _field_mask("opcode") | _field_mask("funct3") | _field_mask("funct7")
                for fname, tok in zip(("rd", "rs1", "rs2"), fields[3:]):
                    if tok.startswith("x") and tok[1:].isdigit():
                        match |= _put(int(tok[1:]), fname)
                        mask |= _field_mask(fname)
                out.setdefault(name, []).append(HeaderSite(name, match, mask, template.strip()))
        pos = text.find(marker, pos + len(marker))
    return out


_EMIT_FIELDS = ("opcode", "funct3", "funct7")


def crosscheck_header(isa: MatrixIsa, header: str | Path, *, pairs: Mapping[str, str]) -> MatrixIsa:
    """Check every derived encoding against the header site ``pairs`` names, and fill emission bits.

    Agreement means: every bit BOTH sources state has the same value, and the header states every bit of
    the derived opcode/funct3/funct6. Bits the header spells that the RTL decoder ignores, within the
    fields an ``.insn r`` must carry, become the encoding's ``fill`` -- recorded as coming from the header.
    A site the header does not have, or a name with no declared pair, is a failed check, never a pass.
    """
    label = str(header)
    try:
        sites = header_insn_sites(Path(header).read_text(encoding="utf-8"))
    except OSError as exc:
        return MatrixIsa(
            encodings=isa.encodings,
            sources={**isa.sources, "header": label},
            gaps=isa.gaps,
            crosschecks=isa.crosschecks + ({"agrees": False, "source": label, "reason": f"unreadable: {exc}"},),
        )
    emit_mask = 0
    for f in _EMIT_FIELDS:
        emit_mask |= _field_mask(f)
    new: dict[str, MaskedEncoding] = {}
    records: list[dict[str, Any]] = []
    for name, enc in sorted(isa.encodings.items()):
        site_name = pairs.get(name)
        found = sites.get(site_name or "")
        if not site_name or not found:
            records.append(
                {
                    "instruction": name,
                    "source": label,
                    "site": site_name,
                    "agrees": False,
                    "reason": "no cross-check site declared" if not site_name else f"{site_name!r} not in header",
                }
            )
            new[name] = enc
            continue
        site_records = []
        for s in found:
            both = enc.mask & s.mask
            clash = (enc.match ^ s.match) & both
            identity = _field_mask("opcode") | _field_mask("funct3") | _field_mask("funct6")
            unstated = enc.mask & identity & ~s.mask
            site_records.append((s, clash, unstated))
        bad = [(s, c, u) for s, c, u in site_records if c or u]
        site = found[0]
        if bad:
            s, clash, unstated = bad[0]
            records.append(
                {
                    "instruction": name,
                    "source": label,
                    "site": site_name,
                    "agrees": False,
                    "reason": (
                        f"bits {clash:#010x} differ (rtl {enc.match & clash:#010x}, header {s.match & clash:#010x})"
                        if clash
                        else f"header does not state identity bits {unstated:#010x}"
                    ),
                    "template": s.template,
                }
            )
            new[name] = enc
            continue
        # Emission fill: bits of the always-spelled fields (and of any register the header fixes) that the
        # RTL does not decode. Every site must agree on them, or the header itself is ambiguous.
        fill_mask = (emit_mask | site.mask) & site.mask & ~enc.mask
        fills = {s.match & fill_mask for s in found}
        if len(fills) > 1:
            records.append(
                {
                    "instruction": name,
                    "source": label,
                    "site": site_name,
                    "agrees": False,
                    "reason": "the header's sites disagree on bits the RTL does not decode",
                }
            )
            new[name] = enc
            continue
        new[name] = MaskedEncoding(
            mnemonic=enc.mnemonic,
            match=enc.match,
            mask=enc.mask,
            origin=enc.origin,
            flags=enc.flags,
            vtype=enc.vtype,
            fill=site.match & fill_mask,
            fill_mask=fill_mask,
            fill_source=f"{label}:{site_name}" if fill_mask else "",
        )
        records.append(
            {
                "instruction": name,
                "source": label,
                "site": site_name,
                "agrees": True,
                "checked_bits": f"{enc.mask & site.mask:#010x}",
                "filled_bits": f"{fill_mask:#010x}",
                "sites": len(found),
            }
        )
    return MatrixIsa(
        encodings=new,
        sources={**isa.sources, "header": label},
        gaps=isa.gaps,
        crosschecks=isa.crosschecks + tuple(records),
    )


def opcode_table_defs(text: str) -> dict[str, tuple[int, int]]:
    """``{NAME: (match, mask)}`` from a riscv-opcodes style header's ``#define MATCH_NAME`` / ``MASK_NAME``."""
    matches: dict[str, int] = {}
    masks: dict[str, int] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) != 3 or parts[0] != "#define":
            continue
        name, value = parts[1], parts[2]
        try:
            v = int(value, 0)
        except ValueError:
            continue
        if name.startswith("MATCH_"):
            matches[name[len("MATCH_") :]] = v
        elif name.startswith("MASK_"):
            masks[name[len("MASK_") :]] = v
    return {k: (matches[k], masks[k]) for k in matches if k in masks}


def crosscheck_opcode_table(isa: MatrixIsa, table: str | Path, *, pairs: Mapping[str, str]) -> MatrixIsa:
    """Check the derivation against a simulator's MATCH/MASK table (e.g. a riscv-opcodes ``encoding.h``).

    Agreement: on every bit both decoders look at, the values are equal, and on every bit the simulator
    decodes that the derivation FILLED from its header, the fill equals the simulator's value -- so an
    emitted word is one this simulator decodes as the paired instruction. Bits only the simulator decodes
    (e.g. a register-field bit the kernel supplies as an operand) are reported, not judged.
    """
    label = str(table)
    try:
        defs = opcode_table_defs(Path(table).read_text(encoding="utf-8"))
    except OSError as exc:
        return MatrixIsa(
            encodings=isa.encodings,
            sources={**isa.sources, "opcode_table": label},
            gaps=isa.gaps,
            crosschecks=isa.crosschecks + ({"agrees": False, "source": label, "reason": f"unreadable: {exc}"},),
        )
    records: list[dict[str, Any]] = []
    for name, enc in sorted(isa.encodings.items()):
        other = pairs.get(name)
        if other is None or other not in defs:
            records.append(
                {"instruction": name, "source": label, "entry": other, "agrees": False, "reason": "no table entry"}
            )
            continue
        m, k = defs[other]
        clash = (enc.match ^ m) & (enc.mask & k)
        fill_clash = (enc.fill ^ m) & (enc.fill_mask & k)
        only_sim = k & ~(enc.mask | enc.fill_mask)
        agrees = not clash and not fill_clash
        records.append(
            {
                "instruction": name,
                "source": label,
                "entry": other,
                "agrees": agrees,
                "checked_bits": f"{enc.mask & k:#010x}",
                "operand_bits_only_simulator_decodes": f"{only_sim:#010x}",
                **(
                    {}
                    if agrees
                    else {"reason": f"decoded bits differ {clash:#010x}, filled bits differ {fill_clash:#010x}"}
                ),
            }
        )
    return MatrixIsa(
        encodings=isa.encodings,
        sources={**isa.sources, "opcode_table": label},
        gaps=isa.gaps,
        crosschecks=isa.crosschecks + tuple(records),
    )
