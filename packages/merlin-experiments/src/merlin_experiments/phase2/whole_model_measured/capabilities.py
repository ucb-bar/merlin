"""What a measurement machine CAN do, from its own facts -- and what the chosen one lacks that another can.

Board choice silently removed optimizations once: a lean board without a full-width accumulator
readout made a classifier on the accelerator impossible, and a whole campaign optimized for a weaker
machine than the comparison used, with nothing at launch saying so.  So every launch states each
machine's capabilities, every result carries them, and the launch WARNS -- in the record and on
stdout -- when the chosen machine lacks a capability another machine registered for the same target has.

WHERE THE FACTS COME FROM.  A machine's software-visible capabilities are its own ABI header's: every
object-like macro the header states (a flag -- defined with no value -- or an integer value such as a
buffer's rows or a transfer's bytes), read structurally (:func:`merlin.targetgen.capability_discovery.
parse_c_header`), never by name.  The header is found BY CONTENT: the registry entry's
``program_header_sha256`` names it, and it is looked for among the section's own ``header`` and the
registry's ``capability_headers`` (environment references).  A machine whose header cannot be found is
``UNKNOWN`` and the report says so -- never assumed equal to another.  The registry's own ``cannot_express``
limits are capabilities too (a limit one machine declares and another does not is a lack).  The target's
RTL facts are one elaboration for every machine of the target, so they are cited (artifact and digest),
not compared.

HOW MACHINES ARE COMPARED.  Generically, as capability sets: a flag the chosen machine's header lacks
and another's states is a LACK; a declared ``cannot_express`` limit the chosen machine has and another
does not is a LACK; an integer value that differs is a DIFFERENCE (stated with both values -- larger is
not always better).  Only machines of the same kind as the chosen machine's timing half are compared.
Nothing here names a target, a macro or a machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import registry as R

SCHEMA = "merlin.phase2.whole_model_measured.machine_capabilities.v1"
RECORD = "machine_capabilities.json"
UNKNOWN = "UNKNOWN"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header_capabilities(path: str | Path) -> dict[str, Any]:
    """The capabilities a C header states: flags (object-like macros with no value), integer values, and
    the names of macros whose value is an expression (stated, not evaluated)."""
    from merlin.targetgen.capability_discovery import parse_c_header

    model = parse_c_header(Path(path))
    flags, values, expressions = set(), {}, set()
    for macro in model.macros:
        if macro.is_function:
            continue
        if not macro.body.strip():
            flags.add(macro.name)
        elif macro.int_value is not None:
            values[macro.name] = int(macro.int_value)
        else:
            expressions.add(macro.name)
    return {
        "status": "derived",
        "path": str(path),
        "sha256": _sha256(Path(path)),
        "flags": sorted(flags),
        "values": dict(sorted(values.items())),
        "expressions": sorted(expressions),
    }


def _resolved_paths(values: Any, *, environment: Mapping[str, str] | None, base: Path) -> tuple[list[Path], list[str]]:
    """Each declared header location: an environment reference, or a path relative to the registry file."""
    paths, notes = [], []
    for index, value in enumerate(values or ()):
        try:
            resolved = R._resolve(value, name=f"capability_headers[{index}]", environment=environment)
        except R.RegistryError as exc:
            notes.append(str(exc))
            continue
        path = Path(str(resolved))
        path = path if path.is_absolute() else base / path
        (paths if path.is_file() else notes).append(path if path.is_file() else f"{path} is not a file")
    return paths, notes


def _header_for(digest: str | None, explicit: Sequence[Path], candidates: Sequence[Path]) -> dict[str, Any]:
    """The machine's header by content: an explicit header must BE the declared digest; else the first
    candidate that hashes to it.  ``UNKNOWN`` with why when none does."""
    for path in explicit:
        if not Path(path).is_file():
            continue
        found = _sha256(Path(path))
        if digest and found != digest:
            return {
                "status": "conflict",
                "why": f"the section's header {path} is {found[:12]}, the machine declares {str(digest)[:12]}",
            }
        return header_capabilities(path)
    if not digest:
        return {"status": UNKNOWN, "why": "the machine declares no program_header_sha256 to find its header by"}
    for path in candidates:
        if _sha256(Path(path)) == digest:
            return header_capabilities(path)
    return {"status": UNKNOWN, "why": f"no declared header file hashes to {digest[:12]}"}


def _timing_entry(machines: Mapping[str, Any], name: str) -> tuple[str, Mapping[str, Any]]:
    entry = machines.get(name) or {}
    if entry.get("kind") in ("paired", "batched") and isinstance(entry.get("timing"), str):
        return _timing_entry(machines, str(entry["timing"]))
    return name, entry


def machine_report(
    name: str, entry: Mapping[str, Any], *, explicit: Sequence[Path] = (), candidates: Sequence[Path] = ()
) -> dict[str, Any]:
    """One machine's capability report (see the module doc)."""
    timing = entry.get("timing") if isinstance(entry.get("timing"), Mapping) else entry
    digest = timing.get("program_header_sha256")
    return {
        "machine": name,
        "kind": timing.get("kind"),
        "hw_config": timing.get("hw_config"),
        "program_header_sha256": digest,
        "header": _header_for(digest, explicit, candidates),
        "cannot_express": [dict(limit) for limit in timing.get("cannot_express") or ()],
        "adjudicates": list(timing.get("adjudicates") or ()),
    }


def _limit_key(limit: Mapping[str, Any]) -> str:
    return json.dumps({k: v for k, v in limit.items() if k != "reason"}, sort_keys=True)


def compare(chosen: Mapping[str, Any], peers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The chosen machine's lacks and differences against each peer, as capability sets."""
    lacks: dict[str, dict[str, Any]] = {}
    differs: dict[str, dict[str, Any]] = {}
    unknown = []
    mine = chosen.get("header") or {}
    my_limits = {_limit_key(limit): limit for limit in chosen.get("cannot_express") or ()}
    for name, peer in sorted(peers.items()):
        theirs = peer.get("header") or {}
        their_limits = {_limit_key(limit) for limit in peer.get("cannot_express") or ()}
        for key, limit in my_limits.items():
            if key not in their_limits:
                row = lacks.setdefault(
                    f"can express {key}",
                    {
                        "capability": f"can express {key}",
                        "basis": "cannot_express",
                        "why": limit.get("reason"),
                        "has": [],
                    },
                )
                row["has"].append(name)
        if theirs.get("status") != "derived":
            unknown.append({"machine": name, "why": theirs.get("why")})
            continue
        if mine.get("status") != "derived":
            continue
        for flag in sorted(set(theirs.get("flags") or ()) - set(mine.get("flags") or ())):
            row = lacks.setdefault(flag, {"capability": flag, "basis": "header flag", "has": []})
            row["has"].append(name)
        for key, value in (theirs.get("values") or {}).items():
            own = (mine.get("values") or {}).get(key)
            if own is not None and own != value:
                differs.setdefault(key, {"value": key, "mine": own, "others": {}})["others"][name] = value
    warnings = [
        f"{chosen.get('machine')} lacks {row['capability']}, which {', '.join(row['has'])} "
        f"{'has' if len(row['has']) == 1 else 'have'}" + (f" ({row['why']})" if row.get("why") else "")
        for row in lacks.values()
    ]
    warnings += [
        f"{chosen.get('machine')} has {row['value']}={row['mine']} where "
        + ", ".join(f"{peer} has {value}" for peer, value in row["others"].items())
        for row in differs.values()
    ]
    if mine.get("status") != "derived":
        warnings.insert(0, f"{chosen.get('machine')}'s capabilities are {mine.get('status')}: {mine.get('why')}")
    return {
        "lacks": list(lacks.values()),
        "differs": list(differs.values()),
        "unknown_peers": unknown,
        "warnings": warnings,
    }


def report(
    registry: str | Path,
    name: str,
    *,
    header: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    target_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The capability report of registry machine ``name`` (its timing half for a composite machine),
    compared with every other machine of the same kind the registry declares."""
    document = R.load(registry)
    machines = document.get("machines") or {}
    timing_name, timing = _timing_entry(machines, name)
    candidates, notes = _resolved_paths(
        document.get("capability_headers"), environment=environment, base=Path(registry).parent
    )
    explicit = [Path(str(header))] if header else []
    chosen = machine_report(timing_name, timing, explicit=explicit, candidates=candidates)
    peers = {
        other: machine_report(other, entry, candidates=candidates)
        for other, entry in machines.items()
        if other != timing_name and isinstance(entry, Mapping) and entry.get("kind") == timing.get("kind")
    }
    out = {
        "schema": SCHEMA,
        "target": document.get("target"),
        "registry": str(registry),
        "selected": name,
        "chosen": chosen,
        "peers": peers,
        **compare(chosen, peers),
        "header_sources": {"found": [str(p) for p in candidates], "unavailable": notes},
        "rtl_facts": dict(target_facts) if target_facts else rtl_facts_citation(str(document.get("target") or "")),
        "basis": "each machine's own ABI header (found by content) and its declared cannot_express limits; "
        "the target's RTL facts are one elaboration for every machine and are cited, not compared",
    }
    return out


def inline_report(machine: Mapping[str, Any], *, header: str | Path | None = None) -> dict[str, Any]:
    """The report of a machine given as a full spec (no registry to compare against): its own facts only."""
    name = str(machine.get("registry_name") or machine.get("kind") or "machine")
    chosen = machine_report(name, machine, explicit=[Path(str(header))] if header else [])
    return {
        "schema": SCHEMA,
        "target": machine.get("target"),
        "registry": None,
        "selected": name,
        "chosen": chosen,
        "peers": {},
        **compare(chosen, {}),
        "basis": "the machine's own ABI header; no registry declares other machines to compare with",
    }


def rtl_facts_citation(target: str) -> dict[str, Any]:
    """Which RTL facts artifact the target has (path and digest), without extracting anything."""
    if not target:
        return {"status": UNKNOWN, "why": "no target"}
    try:
        from merlin.targetgen.rtl.facts import find_facts

        path = find_facts(target)
    except Exception as exc:  # noqa: BLE001 -- cited as unknown, never extracted at launch
        return {"status": UNKNOWN, "why": f"{type(exc).__name__}: {exc}"}
    if path is None or not Path(path).is_file():
        return {"status": UNKNOWN, "why": "no derived RTL facts artifact for this target"}
    return {"status": "cited", "path": str(path), "sha256": _sha256(Path(path))}


def section_report(section: Mapping[str, Any], *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The report for one objective-config section's machine (registry reference or full spec); a
    failure to read is a report that says so -- a launch is never refused for its capability report."""
    machine = dict(section.get("machine") or {})
    header = (section.get("build_options") or {}).get("header")
    try:
        if "registry" in machine:
            return report(machine["registry"], str(machine.get("name") or ""), header=header, environment=environment)
        return inline_report(machine, header=header)
    except Exception as exc:  # noqa: BLE001 -- stated as unknown, loudly
        return {
            "schema": SCHEMA,
            "selected": machine.get("name") or machine.get("kind"),
            "chosen": {"header": {"status": UNKNOWN, "why": f"{type(exc).__name__}: {exc}"}},
            "warnings": [f"the machine's capabilities could not be read: {type(exc).__name__}: {exc}"],
        }


def compact(document: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What every result carries of its machine's report: the machine, its header's capabilities, its
    limits, and the launch's warnings -- with the full report's digest."""
    if not document:
        return None
    chosen = document.get("chosen") or {}
    header = chosen.get("header") or {}
    return {
        "schema": SCHEMA,
        "machine": chosen.get("machine"),
        "header_sha256": header.get("sha256"),
        "header_status": header.get("status"),
        "flags": header.get("flags"),
        "values": header.get("values"),
        "cannot_express": chosen.get("cannot_express"),
        "lacks": [row.get("capability") for row in document.get("lacks") or ()],
        "warnings": list(document.get("warnings") or ()),
        "report_sha256": hashlib.sha256(json.dumps(dict(document), sort_keys=True, default=str).encode()).hexdigest(),
    }


def lines(document: Mapping[str, Any], *, section: str = "") -> list[str]:
    """The report as a few lines for a launch's stdout: the machine, its header's flags, and each warning."""
    chosen = document.get("chosen") or {}
    header = chosen.get("header") or {}
    where = f"{section}: " if section else ""
    head = f"machine capabilities -- {where}{chosen.get('machine') or document.get('selected')}"
    if header.get("status") == "derived":
        head += f" (header {str(header.get('sha256'))[:12]}: {len(header.get('flags') or ())} flag(s), "
        head += f"{len(header.get('values') or {})} value(s))"
    else:
        head += f" (header {header.get('status')}: {header.get('why')})"
    return [head, *(f"  WARNING: {warning}" for warning in document.get("warnings") or ())]


__all__ = [
    "RECORD",
    "SCHEMA",
    "compact",
    "compare",
    "header_capabilities",
    "inline_report",
    "lines",
    "machine_report",
    "report",
    "rtl_facts_citation",
    "section_report",
]
