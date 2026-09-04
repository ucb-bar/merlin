"""Per-module PORT facts, read structurally out of a target's own elaborated FIRRTL.

The fact bundle records interfaces by NAME plus an evidence string and carries no port list at all.
Two measurements that the performance layer needs are consequently unobtainable even on a target whose
RTL plainly provides them:

``explicit_completion``
    whether a unit SIGNALS that it finished. The trait reported "none of the declared interfaces
    reports per-engine completion", which reads as a fact about the hardware. On the interlocked target
    here it is a fact about our extractor: each of the three command controllers carries
    ``completed : { flip ready, valid, bits : UInt<6> }`` — a decoupled channel tagged with the
    reservation-station id, which is exactly the per-engine completion a scheduling family needs to
    price an issue against a wait.

``a top-level busy port per engine``
    the same three modules each carry ``busy : UInt<1>``. This matters because an unmeasured unit that
    reads as *idle* has already produced a wrong headline number in this repo: a vector unit with no
    exposed port counted zero everywhere, and correcting it moved a corpus idle figure from 76.7% to
    46.2%. A busy port that exists and is not extracted is the same failure waiting to happen.

So this module reads the ports. Structurally, and with no assumption about which names a target uses:
the caller asks for a field name, and every module that exposes it at the top level of a port bundle is
returned with the bundle's own leaf fields. Nothing here knows what a controller is called, and a
target whose engines signal completion differently is served by asking for a different field.

**Why not the Scala.** The Scala source says what the generator CAN emit; the elaborated FIRRTL says
what this configuration DID emit. Those differ — the same generator elaborates an fp6 and an int8
build from one source tree — and a capability read off the source rather than the elaboration is the
over-declaration this repo has already shipped once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: FIRRTL indents a module's declaration by two spaces and its body by four. Depth is taken from the
#: text rather than assumed, so a differently-formatted emitter is reported unreadable instead of
#: mis-parsed.
_MODULE_KEYWORDS = ("module", "extmodule", "intmodule")
_PORT_DIRECTIONS = ("input", "output")


@dataclass(frozen=True)
class PortField:
    """One field at the top level of a module port's bundle, with the leaves it carries."""

    name: str
    leaves: tuple[str, ...] = ()
    direction: str = ""
    port: str = ""
    #: leaf name -> declared TYPE TEXT. The names alone answer "does this unit signal completion";
    #: a memory's geometry is entirely in the types, so both halves are kept.
    types: dict = field(default_factory=dict)
    #: This field's OWN declared type text (``UInt<8>``, ``{ ready, valid }``, ``SInt<20>[4]``). ``types``
    #: keeps the types of a field's LEAVES and drops the field's own, which is exactly the half a scalar
    #: field has: a compute cell declares its operand and accumulation widths as bare scalars at the top
    #: level of one bundle, and without this they were read as fields with no type at all.
    type_text: str = ""
    #: Whether the declaration carries FIRRTL's ``flip`` orientation marker. Orientation is not a
    #: decoration: inside an ``output`` bundle a flipped field flows INTO the module, so a reader that
    #: drops it cannot tell an accumulator's input from its result and would read a cell's operands and
    #: its outputs as one undifferentiated set.
    flipped: bool = False

    def is_input(self) -> bool:
        """Whether data flows INTO the declaring module through this field.

        The port's own direction XOR the field's flip: an ``input`` port's plain field and an ``output``
        port's flipped field both carry data inward. A field on a port whose direction was not recorded
        is reported as not-inward, which is why callers requiring the distinction check both halves.
        """
        return (self.direction == "input") != bool(self.flipped)

    def is_decoupled(self) -> bool:
        """A ready/valid handshake. ``bits`` is optional — a bare ready/valid pair is still decoupled."""
        return "ready" in self.leaves and "valid" in self.leaves

    def to_dict(self) -> dict:
        return {"name": self.name, "leaves": list(self.leaves), "direction": self.direction,
                "port": self.port, "decoupled": self.is_decoupled()}


@dataclass
class ModulePorts:
    """The port fields one module exposes."""

    module: str
    fields: list = field(default_factory=list)          # [PortField]

    def field_named(self, name: str):
        return next((f for f in self.fields if f.name == name), None)

    def to_dict(self) -> dict:
        return {"module": self.module, "fields": [f.to_dict() for f in self.fields]}


def _split_top_level(body: str) -> list[str]:
    """Split a FIRRTL bundle body on commas that sit at brace depth 0.

    A bundle field's type may itself be a bundle or a vector, so splitting on every comma shreds the
    nesting and yields field names that do not exist. Depth tracking is the whole point.
    """
    out, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(body[start:i])
            start = i + 1
    tail = body[start:]
    if tail.strip():
        out.append(tail)
    return out


def _field_name(chunk: str) -> str:
    """The declared name of one bundle field, stripping FIRRTL's ``flip`` orientation marker."""
    head = chunk.split(":", 1)[0].strip()
    parts = head.split()
    if parts and parts[0] == "flip":
        parts = parts[1:]
    return parts[0] if parts else ""


def _is_flipped(chunk: str) -> bool:
    """Whether one bundle-field declaration carries FIRRTL's ``flip`` orientation marker."""
    head = chunk.split(":", 1)[0].strip().split()
    return bool(head) and head[0] == "flip"


#: FIRRTL's source-locator token. It trails a top-level declaration (``busy : UInt<1> @[Foo.scala 3:5]``)
#: and is not part of the type, so a reader that keeps it reads the type as unparseable.
_INFO_MARKER = "@["


def _own_type(chunk: str) -> str:
    """The declared type TEXT of one field/port declaration, without FIRRTL's trailing source locator."""
    text = chunk.partition(":")[2]
    cut = text.find(_INFO_MARKER)
    return (text[:cut] if cut != -1 else text).strip()


def _leaf_names(chunk: str) -> tuple[str, ...]:
    """The immediate sub-field names of one bundle field, or ``()`` when its type is not a bundle."""
    _, _, rest = chunk.partition(":")
    rest = rest.strip()
    open_i = rest.find("{")
    if open_i == -1:
        return ()
    depth, close_i = 0, -1
    for i in range(open_i, len(rest)):
        if rest[i] == "{":
            depth += 1
        elif rest[i] == "}":
            depth -= 1
            if depth == 0:
                close_i = i
                break
    if close_i == -1:
        return ()
    inner = rest[open_i + 1:close_i]
    return tuple(n for n in (_field_name(c) for c in _split_top_level(inner)) if n)


def _is_identifier(token: str) -> bool:
    """Whether ``token`` could be a FIRRTL declaration name (so a statement is not read as a module)."""
    return bool(token) and not token[0].isdigit() and all(c == "_" or c.isalnum() for c in token)


def _module_name(line: str) -> str:
    """``module Foo :`` -> ``Foo``, for any of FIRRTL's module keywords; ``""`` when not a module line."""
    parts = line.strip().split()
    # FIRRTL 6.0 marks a circuit's entry point `public module Foo :`. Matching only on the FIRST token
    # made every such module invisible: `parse_ports` returned an EMPTY dict for the whole design, so
    # the completion/busy probes reported "this target exposes no such port" for a design that exposes
    # it plainly. Skip a leading qualifier rather than enumerating the ones seen so far -- and then
    # require the NAME to look like a declaration name, so a statement that merely happens to carry the
    # word in its second position is not read as a module header.
    for i, token in enumerate(parts[:2]):
        if token in _MODULE_KEYWORDS:
            parts = parts[i:]
            break
    else:
        return ""
    if len(parts) < 2:
        return ""
    name = parts[1].rstrip(":")
    return name if _is_identifier(name) else ""


#: Public alias. Every reader of a FIRRTL circuit has to agree on what a module header IS -- the
#: qualifier-tolerant reading above -- or one of them silently sees a circuit with no modules in it.
module_name = _module_name


def parse_ports(fir_text: str) -> dict:
    """``module name -> ModulePorts`` for every module whose ports declare a bundle.

    Reads the port DECLARATION lines only. A FIRRTL circuit for a whole SoC is tens of megabytes and
    almost all of it is statements, so this walks lines and keeps only the ones that declare a port.
    """
    out: dict = {}
    current = ""
    for raw in fir_text.splitlines():
        name = _module_name(raw)
        if name:
            current = name
            continue
        if not current:
            continue
        stripped = raw.strip()
        parts = stripped.split(None, 1)
        if len(parts) < 2 or parts[0] not in _PORT_DIRECTIONS:
            continue
        direction = parts[0]
        port_name = _field_name(parts[1])
        leaves = _leaf_names(parts[1])
        if not leaves:
            # A scalar port. Recorded as a field of its own so a bare `busy : UInt<1>` at the top level
            # of a module (rather than inside an `io` bundle) is still found.
            out.setdefault(current, ModulePorts(current)).fields.append(
                PortField(name=port_name, leaves=(), direction=direction, port=port_name,
                          type_text=_own_type(parts[1]), flipped=_is_flipped(parts[1])))
            continue
        mp = out.setdefault(current, ModulePorts(current))
        _, _, rest = parts[1].partition(":")
        open_i = rest.find("{")
        depth, close_i = 0, -1
        for i in range(open_i, len(rest)):
            if rest[i] == "{":
                depth += 1
            elif rest[i] == "}":
                depth -= 1
                if depth == 0:
                    close_i = i
                    break
        inner = rest[open_i + 1:close_i] if close_i != -1 else ""
        for chunk in _split_top_level(inner):
            fname = _field_name(chunk)
            if fname:
                mp.fields.append(PortField(name=fname, leaves=_leaf_names(chunk),
                                           direction=direction, port=port_name,
                                           types=_leaf_types(chunk), type_text=_own_type(chunk),
                                           flipped=_is_flipped(chunk)))
    return out


_HW_DECLS = ("hw.module", "hw.module.extern", "hw.module.generated", "hw.module private")
#: Port direction keywords the hw dialect spells before a port name.
_HW_DIRECTIONS = ("in", "out", "inout")


def _hw_port_entries(sig: str) -> list[str] | None:
    """Split a `hw.module` signature's port list on top-level commas, or ``None`` if unreadable.

    Depth is tracked over ``<>``, ``()``, ``[]`` and ``{}`` together, because an hw port type is
    routinely nested (``!hw.array<4xi8>``, ``!hw.struct<a: i1, b: i2>``) and splitting on every comma
    shreds it into names that do not exist. Returning None rather than a partial list is the point: a
    signature this cannot read must be recorded as UNREADABLE, never as a module with no ports -- that
    is the silent skip this file's own history warns about.
    """
    open_i = sig.find("(")
    if open_i == -1:
        return None
    depth, close_i = 0, -1
    pairs = {"(": ")", "<": ">", "[": "]", "{": "}"}
    closers = set(pairs.values())
    for i in range(open_i, len(sig)):
        ch = sig[i]
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth -= 1
            if depth == 0:
                close_i = i
                break
    if close_i == -1:
        return None
    body, out, cur, depth = sig[open_i + 1:close_i], [], [], 0
    for ch in body:
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [e for e in (x.strip() for x in out) if e]


def _hw_port_name(entry: str) -> tuple[str, str] | None:
    """``(direction, port name)`` for one signature entry, or ``None`` when it cannot be read.

    An entry is ``in %clock : i1`` or ``out done : i1``: a direction keyword, an optionally
    ``%``-prefixed name, then the type after a colon. Read by splitting, not by pattern -- and an entry
    whose direction keyword is not one of the three is refused rather than guessed at.
    """
    head, sep, _type = entry.partition(":")
    if not sep:
        return None
    words = head.split()
    if len(words) < 2 or words[0] not in _HW_DIRECTIONS:
        return None
    return words[0], words[1].lstrip("%")


def hw_module_ports(text: str) -> tuple[dict, dict[str, str]]:
    """``({module: ModulePorts}, {module: why})`` from a CIRCT **hw dialect** elaboration.

    ⚠️ hw-DIALECT BUNDLES ARE FLATTENED. Where FIRRTL writes ``completed : { flip ready, valid, bits :
    UInt<6> }``, the hw dialect writes three separate ports -- ``completed_ready``, ``completed_valid``,
    ``completed_bits`` -- so the bundle has to be REGROUPED to answer the same question. A field is a
    name segment, and its leaves are the segments that follow it, which is why a decoupled channel is
    still recognisable as one.

    The second return value is the modules whose signature could NOT be read, with the reason. A caller
    that ignores it is back to treating an unreadable module as a module with no ports.
    """
    ports: dict = {}
    unreadable: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not any(line.startswith(d) for d in _HW_DECLS):
            continue
        at = line.find("@")
        if at == -1:
            unreadable[line[:60]] = "the declaration names no symbol"
            continue
        rest = line[at + 1:]
        stop = min((i for i in (rest.find("("), rest.find(" ")) if i != -1), default=-1)
        module = rest[:stop] if stop != -1 else rest
        module = module.strip().strip('"')
        if not module:
            unreadable[line[:60]] = "the symbol after @ is empty"
            continue
        entries = _hw_port_entries(rest)
        if entries is None:
            unreadable[module] = "the port list is not delimited by a matching paren pair"
            continue
        # field -> {leaf}, plus the direction seen for it. A port with no underscore is a scalar field
        # of its own, matching the .fir reader's treatment of a bare top-level port.
        fields: dict[str, set] = {}
        directions: dict[str, str] = {}
        refused = 0
        for entry in entries:
            got = _hw_port_name(entry)
            if got is None:
                refused += 1
                continue
            direction, name = got
            segs = [s for s in name.split("_") if s]
            for i, seg in enumerate(segs):
                leaf = segs[i + 1] if i + 1 < len(segs) else None
                fields.setdefault(seg, set())
                directions.setdefault(seg, direction)
                if leaf is not None:
                    fields[seg].add(leaf)
        if refused:
            unreadable[module] = (f"{refused} of {len(entries)} port entries could not be read; the "
                                  f"module's field list is therefore incomplete")
            continue
        mp = ModulePorts(module)
        for name, leaves in sorted(fields.items()):
            mp.fields.append(PortField(name=name, leaves=tuple(sorted(leaves)),
                                       direction=directions.get(name, ""), port=name))
        ports[module] = mp
    return ports, unreadable


def modules_exposing(ports: dict, field_name: str) -> dict:
    """``module -> PortField`` for every module exposing ``field_name`` at a port's top level."""
    found = {}
    for mod, mp in sorted(ports.items()):
        f = mp.field_named(field_name)
        if f is not None:
            found[mod] = f
    return found


def elaboration_kind(target: str) -> tuple[str, str]:
    """``(kind, detail)`` for the elaboration this target's facts were extracted from.

    ⚠️ "THE NAMED FILE IS MISSING" AND "THE FACTS NAME NO .fir AT ALL" ARE DIFFERENT REPAIRS, and
    collapsing them sent a reader looking for a file that was never named. Measured: atlas's facts name
    ``inputs.hw_mlir``/``core_hw_mlir`` and a ``facts.interfaces.hw_source`` pointing at a CIRCT
    ``hw.mlir`` that EXISTS (4.3 MB on this host) -- there is no ``source.fir`` to locate. gemmini's
    facts do name one, which is why its port facts derive and atlas's do not.

    ``kind`` is ``"fir"`` (a .fir is named), ``"hw_mlir"`` (the elaboration is recorded as CIRCT hw
    dialect, which this reader does not parse yet), or ``"none"``.
    """
    try:
        from merlin.targetgen.rtl.facts import load_facts
        doc = load_facts(target) or {}
    except Exception as exc:                                   # noqa: BLE001
        return "none", f"facts unreadable: {type(exc).__name__}: {exc}"
    body = doc.get("facts") or {}
    # `facts.source` is a MAPPING carrying a `fir` key on one target and a bare STRING on another.
    # Both are read; assuming either shape turns a target with a recorded elaboration into one that
    # reports none, which is the direction that hides a fixable gap.
    source = body.get("source")
    if isinstance(source, dict):
        named = str(source.get("fir") or "").strip()
    elif isinstance(source, str):
        named = source.strip() if source.strip().endswith(".fir") else ""
    else:
        named = ""
    if named:
        return "fir", named
    inputs = doc.get("inputs") or {}
    hw = [str(v) for k, v in inputs.items() if isinstance(v, str) and str(v).endswith(".mlir")]
    # `facts.interfaces` is a MAPPING on one target and a LIST of interface records on another, so
    # both shapes are walked rather than one assumed -- an AttributeError here would report a target
    # with a perfectly good elaboration as having none.
    ifaces = body.get("interfaces")
    records: list = []
    if isinstance(ifaces, dict):
        records = [ifaces] + [v for v in ifaces.values() if isinstance(v, dict)]
    elif isinstance(ifaces, list):
        records = [v for v in ifaces if isinstance(v, dict)]
    for rec in records:
        src = rec.get("hw_source") or ""
        if isinstance(src, str) and src.endswith(".mlir"):
            hw.append(src)
    if hw:
        return "hw_mlir", ", ".join(sorted(set(hw))[:2])
    return "none", "the facts name no elaboration artifact of either kind"


def hw_path_for(target: str) -> Path | None:
    """The CIRCT hw-dialect elaboration this target's facts name, when it is on this host.

    An absolute path the facts carry is used as-is; a bare filename is looked for beside the other
    named artifact, since the extractor writes them together.
    """
    kind, detail = elaboration_kind(target)
    if kind != "hw_mlir":
        return None
    names = [n.strip() for n in detail.split(",") if n.strip()]
    absolute = [Path(n) for n in names if n.startswith("/")]
    for cand in absolute:
        if cand.is_file():
            return cand
    for base in absolute:                          # a sibling of a named absolute path
        for n in names:
            if n.startswith("/"):
                continue
            cand = base.parent / n
            if cand.is_file():
                return cand
    return None


def _why_no_fir(target: str) -> str:
    """Why no FIRRTL resolved, distinguishing a missing file from a different artifact kind."""
    kind, detail = elaboration_kind(target)
    if kind == "fir":
        return f"the elaborated FIRRTL this target's facts name ({detail}) could not be located"
    if kind == "hw_mlir":
        if hw_path_for(target) is None:
            return (f"this target's elaboration is recorded as CIRCT hw dialect ({detail}) and that "
                    f"file is not on this host")
        return (f"this target's elaboration is CIRCT hw dialect ({detail}); it was read")
    return f"no elaboration artifact is named by this target's facts ({detail})"


def fir_path_for(target: str) -> Path | None:
    """The elaborated FIRRTL this target's facts were extracted from, or ``None``.

    Resolved from the fact bundle's own ``source.fir`` — the file the facts NAME — so this cannot drift
    from what everything else derived, and no path is typed here.
    """
    try:
        from merlin.targetgen.rtl.facts import load_facts
        body = (load_facts(target) or {}).get("facts") or {}
        name = ((body.get("source") or {}).get("fir") or "").strip()
    except Exception:                                          # noqa: BLE001
        return None
    if not name:
        return None
    try:
        from merlin.targetgen.rtl.introspect import find_artifacts
        found = find_artifacts()
        for cand in (found or {}).values() if isinstance(found, dict) else ():
            p = Path(str(cand))
            if p.name == name and p.is_file():
                return p
    except Exception:                                          # noqa: BLE001
        pass
    try:
        from merlin.common.paths import _dotenv
        root = (_dotenv() or {}).get("MERLIN_CHIPYARD")
    except Exception:                                          # noqa: BLE001
        root = None
    if not root:
        return None
    for p in Path(root).rglob(name):
        if p.is_file():
            return p
    return None


def port_facts(target: str, *, fields=("completed", "busy"), fir=None) -> dict:
    """Which modules of ``target``'s elaboration expose each of ``fields``.

    Returns a three-state record. ``status`` is ``derived`` when the FIRRTL was read, ``unavailable``
    when it could not be found or read — never an empty answer that reads like "the RTL has none".
    """
    # TWO ELABORATION DIALECTS, one question. A target's facts name either a FIRRTL `.fir` or a CIRCT
    # `hw.mlir`, and the port question is the same in both -- so both are read rather than one dialect
    # deciding whether the trait is knowable. Measured: atlas's facts name only an `hw.mlir`, and while
    # this read `.fir` alone its ports were reported UNKNOWN against a 4.5 MB file sitting on disk.
    #
    # `dialect` travels in the record because the two are not interchangeable evidence: hw-dialect
    # bundles are FLATTENED, so a field's leaves are reconstructed from name segments rather than read
    # from a bundle, and a reader should be able to see which reading produced its answer.
    path = Path(fir) if fir else fir_path_for(target)
    dialect = "fir"
    if path is None or not path.is_file():
        hw = hw_path_for(target)
        if hw is not None:
            path, dialect = hw, "hw"
    if path is None or not path.is_file():
        return {"status": "unavailable", "fir": str(path) if path else None,
                "why": (f"{_why_no_fir(target)}; port facts are UNKNOWN, which is not the same as "
                        f"the RTL exposing no such port"),
                "fields": {}}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"status": "unavailable", "fir": str(path),
                "why": f"unreadable: {type(e).__name__}: {e}", "fields": {}}
    if dialect == "hw":
        ports, unreadable = hw_module_ports(text)
        if unreadable:
            # A signature this could not read leaves the field list INCOMPLETE, and an incomplete list
            # answered as though complete is how "no completion port" gets concluded from a parse gap.
            return {"status": "unavailable", "fir": str(path), "dialect": dialect,
                    "why": (f"{len(unreadable)} module signature(s) could not be read "
                            f"({sorted(unreadable)[:3]}), so the port list is incomplete and port "
                            f"facts are UNKNOWN"),
                    "fields": {}}
    else:
        ports = parse_ports(text)
    out: dict = {"status": "derived", "fir": str(path), "dialect": dialect,
                 "n_modules": len(ports), "fields": {}}
    for name in fields:
        found = modules_exposing(ports, name)
        out["fields"][name] = {
            "modules": sorted(found),
            "decoupled": sorted(m for m, f in found.items() if f.is_decoupled()),
            "leaves": {m: list(f.leaves) for m, f in sorted(found.items())},
        }
    return out


# ---------------------------------------------------------------- banked-store geometry (from ports)
def _leaf_types(chunk: str) -> dict[str, str]:
    """Immediate sub-field name -> its declared TYPE TEXT, for one bundle field.

    :func:`_leaf_names` keeps the names and drops the types, which is all the completion/busy probes
    need. A memory's GEOMETRY is entirely in the types, so this keeps the other half.
    """
    _, _, rest = chunk.partition(":")
    rest = rest.strip()
    open_i = rest.find("{")
    if open_i == -1:
        return {}
    depth, close_i = 0, -1
    for i in range(open_i, len(rest)):
        if rest[i] == "{":
            depth += 1
        elif rest[i] == "}":
            depth -= 1
            if depth == 0:
                close_i = i
                break
    if close_i == -1:
        return {}
    out: dict[str, str] = {}
    for c in _split_top_level(rest[open_i + 1:close_i]):
        name = _field_name(c)
        if name:
            out[name] = c.partition(":")[2].strip()
    return out


def _uint_width(type_text: str) -> int | None:
    """``UInt<13>`` -> 13. ``None`` for any other type, including a vector or a nested bundle."""
    t = type_text.strip()
    if not t.startswith("UInt<") or not t.endswith(">"):
        return None
    inner = t[len("UInt<"):-1]
    return int(inner) if inner.isdigit() else None


def _uint_vec(type_text: str) -> tuple[int, int] | None:
    """``UInt<1>[32]`` -> ``(1, 32)`` (element bits, length). ``None`` when not a UInt vector."""
    t = type_text.strip()
    if not t.endswith("]") or "[" not in t:
        return None
    head, _, tail = t.rpartition("[")
    length = tail[:-1]
    bits = _uint_width(head)
    if bits is None or not length.isdigit():
        return None
    return bits, int(length)


#: FIRRTL's ORDERING comparison primops. A bank count is pinned by a RANGE check ("is this line address
#: inside the store?"), which is what these four spell. ``eq``/``neq`` are deliberately excluded: they
#: test identity against one value, which is not a bound on a capacity.
_ORDER_COMPARISONS = ("lt", "leq", "gt", "geq")
#: Radix prefixes FIRRTL writes literals in. ``UInt<16>(0hc000)`` is the common one; the others are read
#: rather than assumed absent, because a literal this reader cannot parse becomes a bound it cannot pin
#: and would be reported as "the design compares against nothing", which is a claim about the RTL.
_RADIX = {"h": 16, "o": 8, "b": 2}
_IDENT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _radix_int(token: str) -> int | None:
    """``0hc000`` -> 49152, ``0o17`` -> 15, ``0b101`` -> 5, ``42`` -> 42; ``None`` when unreadable."""
    t = token.strip()
    negative = t.startswith("-")
    if negative:
        t = t[1:]
    if len(t) > 2 and t[0] == "0" and t[1] in _RADIX:
        try:
            value = int(t[2:], _RADIX[t[1]])
        except ValueError:
            return None
    elif t.isdigit():
        value = int(t)
    else:
        return None
    return -value if negative else value


def _uint_literal(text: str) -> tuple[int, int | None] | None:
    """``UInt<16>(0hc000)`` -> ``(49152, 16)``; ``UInt(0h5)`` -> ``(5, None)``; else ``None``.

    The DECLARED width travels with the value because it is half the admissibility test: a literal wider
    than the address it would bound cannot be that address's limit, whatever its value happens to be.
    """
    t = text.strip()
    if not t.startswith("UInt"):
        return None
    rest = t[len("UInt"):]
    width: int | None = None
    if rest.startswith("<"):
        close = rest.find(">")
        if close == -1:
            return None
        inner = rest[1:close]
        if not inner.isdigit():
            return None
        width = int(inner)
        rest = rest[close + 1:]
    if not (rest.startswith("(") and rest.endswith(")")):
        return None
    value = _radix_int(rest[1:-1])
    return None if value is None else (value, width)


def _split_call_args(body: str) -> list[str]:
    """Split a primop's argument list on commas that sit at bracket depth 0."""
    out, depth, cur = [], 0, []
    for ch in body:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [a for a in (x.strip() for x in out) if a]


def _primop_calls(line: str, names: tuple[str, ...]) -> list[list[str]]:
    """Argument lists of every call to one of ``names`` on ``line``.

    The callee is the WHOLE identifier before the paren, so a node called ``_my_lt`` is not read as an
    ``lt``. Depth is tracked so a nested call's arguments stay inside their own list.
    """
    out: list[list[str]] = []
    for i, ch in enumerate(line):
        if ch != "(":
            continue
        start = i
        while start > 0 and line[start - 1] in _IDENT_CHARS:
            start -= 1
        if line[start:i] not in names:
            continue
        depth, close = 0, -1
        for k in range(i, len(line)):
            if line[k] == "(":
                depth += 1
            elif line[k] == ")":
                depth -= 1
                if depth == 0:
                    close = k
                    break
        if close == -1:
            continue
        out.append(_split_call_args(line[i + 1:close]))
    return out


def module_bodies(fir_text: str, modules) -> dict[str, list[str]]:
    """``module -> its statement lines``, for the named modules only.

    Restricted to the modules asked for because a whole-SoC elaboration is tens of megabytes and the
    only statements that can pin one port's geometry are the ones in the module that declares it.
    """
    wanted = set(modules)
    out: dict[str, list[str]] = {m: [] for m in wanted}
    current = ""
    for raw in fir_text.splitlines():
        name = _module_name(raw)
        if name:
            current = name
            continue
        if current in wanted:
            out[current].append(raw.strip())
    return out


def pin_bank_count(lines, *, rows_per_bank: int, addr_bits: int,
                   max_banks: int) -> tuple[int | None, str]:
    """``(banks, evidence)`` when the module's own statements pin the bank count, else ``(None, why)``.

    A banked store's port geometry BOUNDS the bank count (the index is ``n`` bits, so at most ``2**n``
    banks) but does not pin it: a design is free to elaborate fewer banks than the index can address.
    The count is pinned where the design RANGE-CHECKS a line address against its own elaborated limit --
    ``lt(<line address>, UInt<16>(0hc000))`` -- because that limit is the TOTAL row count, and total
    rows over rows-per-bank is the bank count.

    A literal is admissible only when it is a whole multiple of ``rows_per_bank``, when the multiple is
    within the bound the index allows, and when it fits the line address it would bound (both its value
    and its DECLARED width). Those three filters are what separate the line limit from every other
    constant in a module -- measured on a real LSU, the sibling byte-capacity literal is a whole
    multiple of the per-bank rows too, and is rejected on the other two.

    Ambiguity FAILS CLOSED. Two admissible literals that disagree mean the module states two different
    limits and this reader cannot tell which one is the store's; answering with either would be a guess
    wearing a derivation's clothes, so the bound is reported instead.
    """
    if rows_per_bank <= 0 or max_banks <= 0:
        return None, "the port geometry yields no per-bank row count, so no literal can be tested against it"
    candidates: dict[int, str] = {}
    for line in lines:
        for args in _primop_calls(line, _ORDER_COMPARISONS):
            for arg in args:
                lit = _uint_literal(arg)
                if lit is None:
                    continue
                value, width = lit
                if value <= 0 or value % rows_per_bank:
                    continue
                banks = value // rows_per_bank
                if banks < 1 or banks > max_banks:
                    continue
                if width is not None and width > addr_bits:
                    continue
                if value > (1 << addr_bits):
                    continue
                candidates.setdefault(banks, line)
    if not candidates:
        return None, ("no ordering comparison in this module tests an address against a whole multiple "
                      f"of the {rows_per_bank} rows a bank holds that also fits the {addr_bits}-bit line "
                      "address, so the bank count is BOUNDED by the index width and not pinned")
    if len(candidates) > 1:
        return None, (f"{len(candidates)} admissible line-address limits disagree on the bank count "
                      f"({sorted(candidates)}), so which one bounds this store is UNKNOWN; the index "
                      "width bound is reported instead of a guess")
    banks, line = next(iter(candidates.items()))
    return banks, line


def banked_store_ports(fir_text: str) -> list[dict]:
    """Every BANKED MEMORY WRITE PORT an elaborated design exposes, with the geometry it implies.

    Identified with NO name matching, because a port name is a target's vocabulary and this must work
    on any of them. The signature is a byte enable: a bundle carrying a ``UInt<W>`` data field AND a
    ``UInt<1>[N]`` vector with ``N == W/8`` is a masked write to a memory whose row is W bits wide --
    nothing else in a design has a Bool vector whose length is exactly the byte count of a sibling
    field. The remaining plain ``UInt`` fields of that bundle are its address: the WIDEST is the
    intra-bank row address (its width is ``log2(rows per bank)``) and a narrower one is the bank index.

    This exists because a memory reached through PORTS is invisible to a ``cmem``/``smem`` census. The
    atlas VMEM is instantiated at the tile level that the standalone elaboration cannot build, so the
    census returned no memories at all and every memory-regime question about that target was
    unanswerable -- while its LSU plainly declares the geometry in the elaboration we CAN build.
    """
    out: list[dict] = []
    ports = parse_ports(fir_text)
    for module, mp in ports.items():
        for f in mp.fields:
            if not f.leaves:
                continue
            top = dict(getattr(f, "types", None) or {})
            if not top:
                continue
            # Descend through a handshake wrapper. A memory port is almost always carried inside a
            # `Valid`/`Decoupled` bundle, so the data and its byte enable sit under `bits` rather than
            # at this level; descending one level finds them without knowing the wrapper's name.
            #
            # Each level is its OWN namespace and they are never merged. Merging them lets a `data` in
            # one sub-bundle pair with a `mask` in a SIBLING sub-bundle -- two fields that are not on
            # the same port at all -- and invent a memory; it also drags the wrapper's own `valid` into
            # the address candidates, where a wide one would be read as the row address.
            levels = [top] + [sub for t in top.values() if (sub := _leaf_types(f":{t}"))]
            row_bits = row_addr_bits = None
            bank_id_bits = 0
            for types in levels:
                data = [(n, w) for n, t in types.items() if (w := _uint_width(t)) is not None]
                masks = [(n, ev) for n, t in types.items() if (ev := _uint_vec(t)) is not None]
                found = next((w for _, w in data
                              for _, (mb, ml) in masks
                              if mb == 1 and w == ml * 8), None)
                if found is None:
                    continue
                addrs = sorted((w for n, w in data if w != found), reverse=True)
                if not addrs:
                    continue
                row_bits, row_addr_bits = found, addrs[0]
                bank_id_bits = addrs[1] if len(addrs) > 1 else 0
                break
            if row_bits is None or row_addr_bits is None:
                continue
            rows_per_bank = 1 << row_addr_bits
            max_banks = (1 << bank_id_bits) if bank_id_bits else 1
            rec = {
                "module": module, "port": f.port, "field": f.name,
                "row_bits": row_bits, "row_bytes": row_bits // 8,
                "row_addr_bits": row_addr_bits, "rows_per_bank": rows_per_bank,
                "bank_id_bits": bank_id_bits,
                "line_addr_bits": row_addr_bits + bank_id_bits,
                "max_banks": max_banks,
                # The index width is a BOUND, and it travels whether or not the count gets pinned, so a
                # reader of a pinned record can still see what the geometry alone allowed.
                "banks_min": 1, "banks_max": max_banks,
                "banks": None, "banks_exact": False, "total_rows": None, "bytes": None,
                "evidence": f"{module}.{f.port}.{f.name}: {row_bits}-bit row with a {row_bits // 8}-lane "
                            f"byte enable, {row_addr_bits}-bit row address"
                            + (f", {bank_id_bits}-bit bank index" if bank_id_bits else ""),
            }
            out.append(rec)
    if not out:
        return out
    # THE INDEX WIDTH BOUNDS THE BANK COUNT; the design's own range check PINS it. Statements are read
    # only for the modules that declared a banked port, because a whole-SoC elaboration is tens of
    # megabytes of statements and none of the rest can bound this store.
    bodies = module_bodies(fir_text, {r["module"] for r in out})
    for rec in out:
        banks, why = pin_bank_count(bodies.get(rec["module"], ()),
                                    rows_per_bank=rec["rows_per_bank"],
                                    addr_bits=rec["line_addr_bits"],
                                    max_banks=rec["max_banks"])
        if banks is None:
            # NOT a guess and NOT a silent drop: the record keeps the bound it did establish and says
            # in its own words why the exact count is UNKNOWN.
            rec["banks_unknown"] = why
            continue
        rec.update({
            "banks": banks, "banks_exact": True,
            "total_rows": banks * rec["rows_per_bank"],
            "bytes": banks * rec["rows_per_bank"] * rec["row_bytes"],
            "banks_evidence": why,
        })
        rec["evidence"] += (f"; the module range-checks a line address against {banks * rec['rows_per_bank']} "
                            f"lines ({why}), which is {banks} banks of {rec['rows_per_bank']} rows")
    return out
