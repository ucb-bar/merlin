"""Derive a vector-unit matrix-extension's instruction encodings from the RTL that implements them.

The extension this first serves — Saturn's outer-product unit — has no assembler support anywhere: every
real binary for it is a raw ``.insn r`` with the fields written as literals. Something has to decide what
those literals are, and the repo's cardinal rule says it must not be a table typed into Python. So this
module reads them out of the hardware's own Chisel sources and then checks its answer against a second,
independent source (the expert C header's ``.insn`` literals). **Agreement is the evidence; disagreement
fails closed** and refuses to emit an encoding, because a wrong opcode does not fail loudly — it decodes
as a different instruction and silently computes something else.

The derivation is four links, none of which needs a regex (each is a tokenizer over Chisel declarations):

1. **opcode** and **funct3** — ``def opcVector = "b1010111".U`` and ``def OPMVV = "b010".U(3.W)`` inside a
   consts trait. Bit-literal defs, parsed by splitting on ``=`` and reading the quoted literal.
2. **funct6** — a ``ChiselEnum``'s value *is its ordinal*, so the funct6 of a name is the number of
   ``Value`` slots declared before it, counting ``val a, b = Value`` as two and a placeholder
   ``val _ = Value`` as one. This is the fragile link and the reason the whole derivation must be re-run
   per RTL revision: an upstream edit that inserts one ``Value`` shifts every later opcode silently.
3. **which instructions the unit has, and each one's operand form** — the parameter object's instruction
   sequence (``Seq(OPMACC.VV, OPMVIN.VX, ...)``), which pairs a mnemonic with the funct3 class.
4. **operand roles** — the instruction objects' ``props`` list (``ReadsVS1.Y``, ``WritesVD.N``), which is
   what says which field carries which operand and which instruction is the only readout.

Nothing here names a target, a tree, or a mnemonic: the caller supplies the source paths and the symbol
names to look for, so a different vector unit with the same Chisel idioms is a parameter change. Each
derived field carries ``{value, derived, source, evidence}`` like the other fact extractors in this
package, and a field that cannot be grounded is reported ``derived=False`` with the reason rather than
guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Encoding", "IsaDerivation", "bit_literal_defs", "chisel_enum_ordinals", "crosscheck",
           "derive", "insn_r_macros", "instruction_props", "scala_call_args", "scala_param_names",
           "unit_instruction_forms", "vector_unit_params"]

#: A ``ChiselEnum`` placeholder — a reserved slot that consumes an ordinal without naming it.
_PLACEHOLDER = "_"

#: How a vector-unit mixin is recognised in a config body. Chipyard's naming convention
#: (``WithShuttleVectorUnit`` / ``WithRocketVectorUnit``) is the only thing shared between the host-core
#: flavours, so matching the suffix reads both without naming either.
_VECTOR_UNIT_MIXIN_SUFFIX = "VectorUnit"


# ---------------------------------------------------------------------------------------------
# Chisel readers
# ---------------------------------------------------------------------------------------------


def _block_after(text: str, header: str) -> str | None:
    """The brace-balanced block following ``header``, or None when the header is absent.

    Balanced-brace scanning rather than a line pattern, so a nested block inside the body cannot end the
    read early and a body that opens on a later line is still found.
    """
    i = text.find(header)
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _balanced_after(text: str, header: str, opener: str = "{", closer: str = "}") -> str | None:
    """The delimiter-balanced span following ``header``. ``_block_after`` is the brace case of this.

    Scala configs are *parenthesised* (``extends Config( … )``) while its enums and traits are braced, so
    both delimiters are needed to read the same file.
    """
    i = text.find(header)
    if i < 0:
        return None
    j = text.find(opener, i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == opener:
            depth += 1
        elif text[k] == closer:
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at nesting depth zero, so ``VectorParams(a, b)`` stays one argument."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def scala_call_args(text: str, callee: str) -> list[str] | None:
    """The positional argument tokens of the first call to ``callee``, or None when it is absent.

    ``callee`` matches on the SIMPLE name, so a fully-qualified call site
    (``new saturn.shuttle.WithShuttleVectorUnit(…)``) is found by the class name alone — the package
    prefix is a spelling choice at the call site, not part of the fact being read.
    """
    body = _balanced_after(_strip_comments(text), callee, "(", ")")
    if body is None:
        return None
    return _split_top_level(body)


def scala_param_names(text: str, decl: str) -> list[str]:
    """The declared parameter names of ``decl`` (e.g. ``class WithShuttleVectorUnit``), in order.

    This is what makes a positional call site readable without assuming an argument order. Guessing that
    order is a real hazard here: ``(vLen, dLen)`` and ``(dLen, vLen)`` are both plausible, both parse, and
    swapping them yields a tile edge that is wrong by exactly a factor of two — which would silently
    certify a corpus against the wrong geometry rather than fail.
    """
    body = _balanced_after(_strip_comments(text), decl, "(", ")")
    if body is None:
        return []
    names = []
    for arg in _split_top_level(body):
        name, sep, _ = arg.partition(":")
        name = name.strip()
        if sep and name.isidentifier():
            names.append(name)
    return names


def vector_unit_params(config_text: str, config_class: str, *,
                       mixin_text: str | None = None) -> dict[str, int]:
    """``{param_name: value}`` for the vector-unit mixin a config instantiates.

    The mixin is DISCOVERED rather than named: whichever call in the config's body has a simple name
    ending in ``VectorUnit`` is the one read, so a shuttle config and a rocket config are handled by the
    same code. Its positional integer arguments are then bound to the names in the mixin's own
    declaration (``mixin_text``), which is why ``vLen`` comes back under that name instead of as
    "argument 0".

    NAMED arguments are bound by their own name, and they do not consume a positional slot. Scala allows
    both forms and mixes them freely, and real configs use both: the extension's own generator writes
    ``WithShuttleVectorUnit(256, 128, params)`` while an integrating SoC writes
    ``WithShuttleVectorUnit(vLen = 128, dLen = 64, params = ..., cores = Some(Seq(1)))``. Reading only the
    positional form returned nothing at all for the second, which is how a config that a real bitstream was
    built from looked ungroundable.

    Returns only what it could ground. An empty result means the caller must record UNKNOWN and stop --
    every consumer of this needs a number that is *right*, and a defaulted vector length produces a
    plausible, wrong tile edge.
    """
    body = _balanced_after(_strip_comments(config_text), f"class {config_class} extends Config(",
                           "(", ")")
    if body is None:
        return {}
    callee = None
    for mixin in _split_top_level(body, "+"):
        head = mixin.strip().removeprefix("new ").strip().split("(", 1)[0]
        simple = head.strip().rsplit(".", 1)[-1]      # drop any package qualification
        if simple.endswith(_VECTOR_UNIT_MIXIN_SUFFIX):
            callee = simple
            break
    if callee is None:
        return {}
    args = scala_call_args(body, callee)
    if not args:
        return {}
    names = scala_param_names(mixin_text, f"class {callee}") if mixin_text else []
    out: dict[str, int] = {}
    position = 0                          # only POSITIONAL arguments advance this
    for arg in args:
        name, sep, rhs = arg.partition("=")
        if sep and name.strip().isidentifier():
            # A named argument. `Some(Seq(1))` and the like fail the int parse below and are skipped, as
            # they are for positional args -- what matters is that a named one never claims a slot.
            key, token = name.strip(), rhs.strip()
        else:
            key = names[position] if position < len(names) else None
            token, position = arg.strip(), position + 1
        try:
            value = int(token.removesuffix(".U").strip(), 0)
        except ValueError:
            continue                      # a params object or an Option, not a scalar we can bind
        if key:
            out[key] = value
    return out


def _strip_comments(text: str) -> str:
    """Drop ``//`` line comments. A commented-out ``val x = Value`` must NOT consume an ordinal, and
    the OPU's own sources carry exactly that kind of annotation next to the ops of interest."""
    out = []
    for line in text.splitlines():
        i = line.find("//")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def chisel_enum_ordinals(text: str, enum_name: str) -> dict[str, int]:
    """``{member: ordinal}`` for a ``ChiselEnum``, which is what its funct6 encoding is.

    Handles the three declaration shapes these enums use: ``val a, b, c = Value`` (consecutive
    ordinals), ``val _ = Value`` (a reserved slot that advances the counter without naming anything), and
    ``val x = Value(0x40.U)`` — an EXPLICIT ordinal, which both names that value and resets where
    counting continues from, so treating it as just another slot would corrupt everything after it.
    ``def alias = other`` lines are aliases, not slots, and are skipped.
    """
    body = _block_after(text, f"object {enum_name}")
    if body is None:
        return {}
    ordinals: dict[str, int] = {}
    nxt = 0
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line.startswith("val ") or "=" not in line:
            continue
        names_part, value_part = line[len("val "):].split("=", 1)
        value_part = value_part.strip()
        if not value_part.startswith("Value"):
            continue                      # `val x = something_else` is not an enum slot
        names = [n.strip() for n in names_part.split(",") if n.strip()]
        if not names:
            continue
        explicit = _explicit_ordinal(value_part)
        if explicit is not None:
            # `Value(0x40.U)` pins this member and continues counting from there.
            if len(names) == 1 and names[0] != _PLACEHOLDER:
                ordinals[names[0]] = explicit
            nxt = explicit + 1
            continue
        for name in names:
            if name != _PLACEHOLDER:
                ordinals[name] = nxt
            nxt += 1
    return ordinals


def _explicit_ordinal(value_part: str) -> int | None:
    """The integer in ``Value(0x40.U)``, or None for a bare ``Value``."""
    if not value_part.startswith("Value("):
        return None
    inner = value_part[len("Value("):]
    close = inner.find(")")
    if close < 0:
        return None
    token = inner[:close].strip().removesuffix(".U").strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def bit_literal_defs(text: str, container: str) -> dict[str, int]:
    """``{name: value}`` for ``def NAME = "b1010111".U`` style declarations inside a trait/object.

    Accepts an optional width suffix (``.U(3.W)``) and both binary and hex literals, because the same
    trait spells the opcode in binary and other constants in hex, and a reader that only understood one
    would drop the other silently.
    """
    body = _block_after(text, f"trait {container}")
    if body is None:
        body = _block_after(text, f"object {container}")
    if body is None:
        return {}
    out: dict[str, int] = {}
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line.startswith("def ") or "=" not in line:
            continue
        name, value = line[len("def "):].split("=", 1)
        name = name.strip()
        if not name or not name.isidentifier():
            continue
        got = _bit_literal(value.strip())
        if got is not None:
            out[name] = got
    return out


def _bit_literal(value: str) -> int | None:
    """The integer in ``"b1010111".U`` / ``"h2a".U`` / ``0x40.U``, or None when it is not a literal."""
    if value.startswith('"'):
        end = value.find('"', 1)
        if end < 0:
            return None
        token = value[1:end]
        base = {"b": 2, "h": 16, "o": 8}.get(token[:1].lower())
        if base is None:
            return None
        digits = token[1:].replace("_", "")
        try:
            return int(digits, base)
        except ValueError:
            return None
    token = value.split()[0].removesuffix(".U").removesuffix(".W").strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def instruction_props(text: str) -> dict[str, dict[str, Any]]:
    """``{OBJECT: {"funct6_member": ..., "flags": {...}}}`` from ``object X extends ... { val props = ... }``.

    ``props`` is a ``Seq`` of markers: ``F6(SomeEnum.member)`` names the funct6 member, and
    ``ReadsVS1.Y`` / ``WritesVD.N`` style entries are boolean flags. Both are read positionally from the
    token, never by matching a fixed vocabulary, so a marker this module has not seen before is carried
    through as a flag instead of being dropped.
    """
    out: dict[str, dict[str, Any]] = {}
    for raw in _strip_comments(text).splitlines():
        line = raw.strip()
        if not line.startswith("object ") or "val props" not in line:
            continue
        name = line[len("object "):].split(None, 1)[0].strip()
        seq = line[line.find("Seq(") + 4:] if "Seq(" in line else ""
        rec: dict[str, Any] = {"funct6_member": None, "flags": {}}
        for tok in seq.split(","):
            # The final token carries the closing `) }` of both the Seq and the object body, possibly
            # with whitespace between them; strip braces and spaces together so the LAST marker in the
            # list is not silently discarded. That marker is `WritesVD`, which is what identifies the
            # single readout instruction -- exactly the field whose loss would be least visible.
            tok = tok.strip(" \t)}")
            if not tok:
                continue
            if tok.startswith("F6("):
                member = tok[len("F6("):].rstrip(")").strip()
                rec["funct6_member"] = member.rsplit(".", 1)[-1] if "." in member else member
            elif "." in tok:
                key, _, val = tok.partition(".")
                if val in ("Y", "N"):
                    rec["flags"][key.strip()] = (val == "Y")
        if rec["funct6_member"] is not None:
            out[name] = rec
    return out


def unit_instruction_forms(text: str, seq_name: str) -> list[tuple[str, str]]:
    """``[(OBJECT, form)]`` from ``def <seq_name> = Seq(pkg.OPMACC.VV, pkg.OPMVIN.VX, ...)``.

    The form (``VV`` / ``VX`` / ...) is what selects the funct3 class, so this is the link that says an
    accumulate is OPMVV and a row move is OPMVX. Only this sequence's members count as the unit's
    instructions: the instruction file defines the whole vector ISA, and treating all of it as the unit's
    would claim capability the hardware does not have.
    """
    i = text.find(f"def {seq_name}")
    if i < 0:
        return []
    j = text.find("Seq(", i)
    if j < 0:
        return []
    depth, end = 0, len(text)
    for k in range(j + 3, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                end = k
                break
    out: list[tuple[str, str]] = []
    for tok in _strip_comments(text[j + 4:end]).split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(".")
        if len(parts) < 2:
            continue
        out.append((parts[-2], parts[-1]))
    return out


def insn_r_macros(text: str) -> dict[str, dict[str, Any]]:
    """``{MACRO: {opcode, funct3, funct7, funct6, args}}`` from ``.insn r`` literals in a C header.

    This is the INDEPENDENT source the Chisel derivation is checked against. ``.insn r`` takes
    ``opcode, funct3, funct7, rd, rs1, rs2``, and for a vector-format instruction ``funct7 = funct6 << 1
    | vm``, so the funct6 recovered here has to equal the ChiselEnum ordinal. ``args`` is the macro's own
    parameter list, which is what reveals that a macro's argument order can differ from the field order
    it expands to.
    """
    out: dict[str, dict[str, Any]] = {}
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("#define ") or "(" not in line:
            continue
        head = line[len("#define "):]
        name = head[:head.find("(")].strip()
        args = [a.strip() for a in head[head.find("(") + 1:head.find(")")].split(",") if a.strip()]
        # The body may continue on following lines via a trailing backslash.
        body, k = head[head.find(")") + 1:], idx
        while body.rstrip().endswith("\\") and k + 1 < len(lines):
            k += 1
            body = body.rstrip().rstrip("\\") + lines[k]
        marker = ".insn r"
        i = body.find(marker)
        if i < 0:
            continue
        fields: list[int] = []
        for tok in body[i + len(marker):].split(","):
            got = _c_int(tok)
            if got is None:
                break                      # the register operands are strings, not literals
            fields.append(got)
        if len(fields) < 3:
            continue
        opcode, funct3, funct7 = fields[0], fields[1], fields[2]
        out[name] = {"opcode": opcode, "funct3": funct3, "funct7": funct7,
                     "funct6": funct7 >> 1, "vm": funct7 & 1, "args": args}
    return out


def _c_int(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    try:
        return int(token, 0)
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------
# the derived table
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoding:
    """One instruction's fields, with where each came from."""

    mnemonic: str
    form: str                       # the funct3 class this instance was instantiated as (VV / VX / ...)
    opcode: int
    funct3: int
    funct6: int
    funct6_member: str
    flags: dict[str, bool] = field(default_factory=dict)

    @property
    def funct7(self) -> int:
        """The ``.insn r`` funct7 field for the unmasked (``vm = 1``) form."""
        return (self.funct6 << 1) | 1

    def insn_r(self, rd: str, rs1: str, rs2: str) -> str:
        """The ``.insn r`` directive that encodes this instruction — the only way to emit it, since no
        assembler knows the mnemonic."""
        return (f".insn r {self.opcode:#x}, {self.funct3:#x}, {self.funct7:#x}, "
                f"{rd}, {rs1}, {rs2}")


@dataclass(frozen=True)
class IsaDerivation:
    """The derived encodings plus everything that could not be grounded.

    ``ok`` is True only when every requested instruction was fully derived AND every cross-check that
    ran agreed. A caller must not emit code from a derivation that is not ``ok``.
    """

    encodings: dict[str, Encoding] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()
    crosschecks: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return (not self.gaps and bool(self.encodings)
                and all(c.get("agrees") for c in self.crosschecks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sources": dict(self.sources),
            "gaps": list(self.gaps),
            "crosschecks": [dict(c) for c in self.crosschecks],
            "encodings": {
                name: {"mnemonic": e.mnemonic, "form": e.form, "opcode": e.opcode,
                       "funct3": e.funct3, "funct6": e.funct6, "funct7": e.funct7,
                       "funct6_member": e.funct6_member, "flags": dict(e.flags)}
                for name, e in sorted(self.encodings.items())},
        }


def derive(*, consts: "str | Path", instructions: "str | Path", params: "str | Path",
           funct6_enum: str, consts_container: str, insn_seq: str,
           opcode_name: str, form_funct3: dict[str, str]) -> IsaDerivation:
    """Derive every instruction in ``insn_seq`` from the three Chisel sources.

    ``form_funct3`` maps an instantiation form to the consts name holding its funct3 (e.g.
    ``{"VV": "OPMVV", "VX": "OPMVX"}``) — the one piece of Chisel convention that is not itself written
    down in the sources, since the mapping lives in a class definition rather than in a table. Every
    other field is read. Missing pieces become ``gaps``; nothing is defaulted.
    """
    paths = {"consts": Path(consts), "instructions": Path(instructions), "params": Path(params)}
    texts: dict[str, str] = {}
    gaps: list[str] = []
    for label, path in paths.items():
        try:
            texts[label] = path.read_text(encoding="utf-8")
        except OSError as exc:
            gaps.append(f"{label}: unreadable ({exc})")
    if gaps:
        return IsaDerivation(sources={k: str(v) for k, v in paths.items()}, gaps=tuple(gaps))

    consts_defs = bit_literal_defs(texts["consts"], consts_container)
    ordinals = chisel_enum_ordinals(texts["consts"], funct6_enum)
    props = instruction_props(texts["instructions"])
    forms = unit_instruction_forms(texts["params"], insn_seq)

    if opcode_name not in consts_defs:
        gaps.append(f"opcode {opcode_name!r} not found in {consts_container}")
    if not ordinals:
        gaps.append(f"funct6 enum {funct6_enum!r} yielded no members")
    if not forms:
        gaps.append(f"instruction sequence {insn_seq!r} is empty or absent")

    encodings: dict[str, Encoding] = {}
    opcode = consts_defs.get(opcode_name)
    for obj, form in forms:
        prop = props.get(obj)
        if prop is None:
            gaps.append(f"{obj}: no props declaration found")
            continue
        member = prop["funct6_member"]
        if member not in ordinals:
            gaps.append(f"{obj}: funct6 member {member!r} not in {funct6_enum}")
            continue
        f3_name = form_funct3.get(form)
        if f3_name is None or f3_name not in consts_defs:
            gaps.append(f"{obj}: no funct3 for form {form!r} (looked for {f3_name!r})")
            continue
        if opcode is None:
            continue
        encodings[obj] = Encoding(mnemonic=obj, form=form, opcode=opcode,
                                  funct3=consts_defs[f3_name], funct6=ordinals[member],
                                  funct6_member=member, flags=dict(prop["flags"]))
    return IsaDerivation(encodings=encodings, sources={k: str(v) for k, v in paths.items()},
                         gaps=tuple(gaps))


def crosscheck(derivation: IsaDerivation, header: "str | Path", *,
               pairs: dict[str, str]) -> IsaDerivation:
    """Check the derivation against a C header's ``.insn r`` literals and record the result.

    ``pairs`` maps a derived instruction name to the header's macro name for it (the two vocabularies
    differ: the RTL calls it ``OPMACC``, the expert header calls it ``VOPACC``). Each pair yields one
    record; a disagreement on any field makes the returned derivation NOT ``ok``, so the caller cannot
    emit from it. A macro with no counterpart is recorded as unchecked rather than as agreeing.
    """
    try:
        macros = insn_r_macros(Path(header).read_text(encoding="utf-8"))
    except OSError as exc:
        return IsaDerivation(
            encodings=derivation.encodings, sources={**derivation.sources, "header": str(header)},
            gaps=derivation.gaps,
            crosschecks=derivation.crosschecks + ({"agrees": False, "reason": f"unreadable: {exc}"},))

    records: list[dict[str, Any]] = []
    for name, enc in sorted(derivation.encodings.items()):
        macro_name = pairs.get(name)
        if macro_name is None:
            records.append({"instruction": name, "agrees": False,
                            "reason": "no cross-check macro declared for this instruction"})
            continue
        macro = macros.get(macro_name)
        if macro is None:
            records.append({"instruction": name, "macro": macro_name, "agrees": False,
                            "reason": f"macro {macro_name!r} not found in the header"})
            continue
        disagreements = [f"{f}: rtl={got} header={macro[f]}"
                         for f, got in (("opcode", enc.opcode), ("funct3", enc.funct3),
                                        ("funct6", enc.funct6))
                         if macro[f] != got]
        records.append({"instruction": name, "macro": macro_name,
                        "agrees": not disagreements,
                        "fields": {"opcode": enc.opcode, "funct3": enc.funct3,
                                   "funct6": enc.funct6, "funct7": enc.funct7},
                        "macro_args": macro["args"],
                        "reason": "; ".join(disagreements)})
    return IsaDerivation(encodings=derivation.encodings,
                         sources={**derivation.sources, "header": str(header)},
                         gaps=derivation.gaps,
                         crosschecks=derivation.crosschecks + tuple(records))
