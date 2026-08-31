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


def _module_name(line: str) -> str:
    """``module Foo :`` -> ``Foo``, for any of FIRRTL's module keywords; ``""`` when not a module line."""
    parts = line.strip().split()
    if len(parts) >= 2 and parts[0] in _MODULE_KEYWORDS:
        return parts[1].rstrip(":")
    return ""


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
                PortField(name=port_name, leaves=(), direction=direction, port=port_name))
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
                                           direction=direction, port=port_name))
    return out


def modules_exposing(ports: dict, field_name: str) -> dict:
    """``module -> PortField`` for every module exposing ``field_name`` at a port's top level."""
    found = {}
    for mod, mp in sorted(ports.items()):
        f = mp.field_named(field_name)
        if f is not None:
            found[mod] = f
    return found


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
    path = Path(fir) if fir else fir_path_for(target)
    if path is None or not path.is_file():
        return {"status": "unavailable", "fir": str(path) if path else None,
                "why": "the elaborated FIRRTL this target's facts name could not be located; port "
                       "facts are UNKNOWN, which is not the same as the RTL exposing no such port",
                "fields": {}}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"status": "unavailable", "fir": str(path),
                "why": f"unreadable: {type(e).__name__}: {e}", "fields": {}}
    ports = parse_ports(text)
    out: dict = {"status": "derived", "fir": str(path), "n_modules": len(ports), "fields": {}}
    for name in fields:
        found = modules_exposing(ports, name)
        out["fields"][name] = {
            "modules": sorted(found),
            "decoupled": sorted(m for m, f in found.items() if f.is_decoupled()),
            "leaves": {m: list(f.leaves) for m, f in sorted(found.items())},
        }
    return out
