"""Behavioural hardware facts, derived from a target's own sources: what a program needs to KNOW to fix
what a failure signature SHOWS.

WHY THIS EXISTS. The structural facts (``rtl.facts``) say what a device HAS -- a mesh of this size, a
store of that capacity, these instruction selectors. A failing program usually needs something else:
what the device DOES with a legal command stream. Measured on the first phase-2 whole-model run: the
optimizing agent was told, for all 16 residual adds, that "the worst element equals the lhs operand's
contribution ALONE", and never fixed one, because nothing it could read said that two loads into the
same accumulator rows are ordered by ISSUE and not by COMPLETION, or that a scaled load saturates
before it accumulates. Both facts are written in the design. This module reads them out.

THREE GENERIC PROPERTIES, any accelerator can have them:

* ``load_completion_ordering`` -- for each pair of command classes, what releases a later command's
  dependency on an earlier one (the earlier one's ISSUE or its COMPLETION, and whether only on an
  address overlap); and which instruction establishes completion order when the tracker does not.
* ``load_scale_saturation`` -- whether a scaled load saturates, to which width, and whether that happens
  before the value reaches the accumulator.
* ``accumulator_readout_width`` -- per MACHINE, whether the accumulator can be read out at its full
  width, from the header that machine's programs are compiled against, checked against an elaboration.

HOW. A target declares WHERE its sources state these facts in its selected support provider's
``contracts/semantic_probes.yaml`` (module, signal and macro names -- data, never code). The
derivation reads the elaborated FIRRTL with :mod:`.firrtl_struct` (a parser, not a line matcher) and
the generated header with the structural C reader, and compares what it finds as DATA. Every fact
carries its provenance: the kind of reading
(``firrtl_structural`` / ``generated_header`` are DERIVED; ``source_reading`` is a citation into Scala
and is flagged ``needs_review``), the content digest of every byte read, the registry entry it was
checked against, and the measured validation the target recorded for it. A fact that cannot be read is
``UNKNOWN`` with the reason -- never a default, never omitted.

DERIVED, NEVER SHIPPED.  The facts are derived where they are used, from the sources the probes name
(:func:`derive`); no provider ships a copy, so no copy can go stale against its hardware.  A machine's
header can be named BY ITS REGISTRY DIGEST (``{abi_header_of: <machine>}``): the derivation then reads
whichever caller-supplied or declared header has the ABI header digest the hardware registry declares
for that machine, so no host path is written into the probes.  :func:`compact` and :func:`brief` render
a derived document for an agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from . import firrtl_struct as F

SCHEMA = "semantic_facts_v1"
PROBES_SCHEMA = "semantic_probes_v1"
#: The producer named by its import path, so a run as `python -m` does not record `__main__`.
GENERATOR = "merlin.targetgen.rtl.semantic_facts"
PROBES_FILE = "semantic_probes.yaml"
FACT_NAMES = ("load_completion_ordering", "load_scale_saturation", "accumulator_readout_width", "accumulate_on_load")

DERIVED = "derived"
SOURCE_READING = "source_reading"
UNKNOWN = "UNKNOWN"
CONFLICT = "CONFLICT"

#: Release events a dependency can wait for.
RELEASED_AT_ISSUE = "issue"
RELEASED_AT_COMPLETION = "completion"
ISSUE_ORDERED_ONLY = "issue_ordered_only"
COMPLETION_ORDERED = "completion_ordered"


class Unknown(Exception):
    """A fact that could not be read, with the reason. Caught per fact, never per document."""


# ------------------------------------------------------------------------------------------ locating


def contracts_dir(target: str) -> Path:
    """The contracts directory of the provider ``target`` resolves to (the selected support provider)."""
    from merlin.targetgen.target_registry import resolve

    return Path(resolve(target).contract_path).parent


def probes_path(target: str) -> Path:
    return contracts_dir(target) / PROBES_FILE


def load_probes(target: str, path: str | Path | None = None) -> dict[str, Any] | None:
    import yaml

    p = Path(path) if path else probes_path(target)
    if not p.is_file():
        return None
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if doc.get("schema") != PROBES_SCHEMA:
        raise ValueError(f"{p}: schema {doc.get('schema')!r} is not {PROBES_SCHEMA!r}")
    if doc.get("target") not in (None, target):
        raise ValueError(f"{p} declares target {doc.get('target')!r}, not {target!r}")
    return doc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Resolved:
    """One source a probe names: where it is, what it is by content, what the registry says of it."""

    def __init__(self, locator: Mapping[str, Any], path: Path | None, registry: dict[str, Any], why: str = ""):
        self.locator = dict(locator)
        self.path = path
        self.registry = registry
        self.why = why

    @property
    def ok(self) -> bool:
        return self.path is not None and self.path.is_file() and not self.why

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"locator": {k: v for k, v in self.locator.items() if k != "sha256"}}
        if self.path is not None and self.path.is_file():
            out["sha256"] = _sha256(self.path)
        out["registry"] = self.registry
        if self.why:
            out["unavailable"] = self.why
        return out


def registry_abi_header(machine: str) -> tuple[str | None, str | None]:
    """``(declaring entry, ABI header sha256)`` the hardware registry declares for ``machine``, walking
    its ``built_from`` chain; ``(None, None)`` when nothing declares one."""
    import yaml

    from merlin.common import provenance as P

    doc = yaml.safe_load(P.pins_path().read_text(encoding="utf-8")) or {}
    entries = {**(doc.get("pins") or {}), **(doc.get("artifacts") or {})}
    frontier, seen = [machine], set()
    while frontier:
        name = frontier.pop(0)
        if name in seen or name not in entries:
            continue
        seen.add(name)
        entry = entries[name] or {}
        if entry.get("abi_header_sha256"):
            return name, str(entry["abi_header_sha256"])
        frontier.extend(str(n) for n in entry.get("built_from") or ())
    return None, None


def resolve_source(locator: Mapping[str, Any], *, headers: Iterable[str | Path] = ()) -> Resolved:
    """Resolve one declared source. Never raises; an unresolvable source carries its reason.

    ``headers`` are the candidate header files an ``abi_header_of`` locator is matched against BY
    CONTENT (the caller's declared headers and the probes' own header sources)."""
    from merlin.common import provenance as P
    from merlin.common.paths import ext_path, out_dir

    loc = dict(locator)
    registry: dict[str, Any] = {}
    path: Path | None = None
    try:
        if "abi_header_of" in loc:
            declared_by, digest = registry_abi_header(str(loc["abi_header_of"]))
            registry = {"abi_header_of": loc["abi_header_of"], "declared_by": declared_by, "sha256": digest}
            if digest is None:
                return Resolved(loc, None, registry, f"the registry declares no ABI header for {loc['abi_header_of']}")
            path = next(
                (Path(h) for h in headers if Path(h).is_file() and _sha256(Path(h)) == digest),
                None,
            )
            if path is None:
                return Resolved(loc, None, registry, f"no supplied header has the declared ABI digest {digest[:12]}")
        elif "artifact" in loc:
            check = P.verify_artifact(str(loc["artifact"]))
            registry = {"artifact": loc["artifact"], "digest_matches": bool(check.ok), "gaps": list(check.gaps)}
            art = P.load_artifacts().get(str(loc["artifact"]))
            path = art.resolve() if art is not None else None
        elif "pin" in loc:
            pin = P.pin(str(loc["pin"]))
            root = pin.checkout()
            path = (Path(root) / str(loc["path"])) if root is not None else None
            try:
                registry = {"pin": loc["pin"], "citation": P.citation(str(loc["pin"]), reads=[str(loc["path"])])}
            except Exception as exc:  # noqa: BLE001 - a citation failure is recorded, not fatal
                registry = {"pin": loc["pin"], "citation": f"{UNKNOWN}: {exc}"}
        elif "ext_root" in loc:
            path = Path(ext_path(str(loc["ext_root"]))) / str(loc["path"])
            registry = {"registry": "undeclared", "correspondence": loc.get("correspondence")}
        elif "out" in loc:
            path = Path(out_dir()) / str(loc["out"])
            registry = {"registry": "undeclared"}
        elif "file" in loc:  # an explicit path: tests and one-off re-derivations
            path = Path(str(loc["file"]))
            registry = {"registry": "explicit_path"}
        else:
            return Resolved(loc, None, {}, f"locator {loc} names no known source kind")
    except Exception as exc:  # noqa: BLE001 - an unresolvable source is a reason, not a crash
        return Resolved(loc, None, registry, f"{type(exc).__name__}: {exc}")
    if path is None or not Path(path).is_file():
        return Resolved(loc, None, registry, f"not present on this host ({path})")
    want = loc.get("sha256")
    if want and _sha256(Path(path)) != want:
        return Resolved(
            loc, Path(path), registry, f"content {_sha256(Path(path))[:12]} is not the declared {str(want)[:12]}"
        )
    return Resolved(loc, Path(path), registry)


def _source(probes: Mapping[str, Any], ref: Any, cache: dict[str, Resolved]) -> Resolved:
    """A probe's source reference: a NAME in ``sources`` or an inline locator.  An ``abi_header_of``
    locator is matched against ``probes["_headers"]`` (see :func:`derive`)."""
    if isinstance(ref, str):
        key = ref
        loc = (probes.get("sources") or {}).get(ref)
        if loc is None:
            return Resolved({"name": ref}, None, {}, f"source {ref!r} is not declared under `sources`")
    else:
        loc = dict(ref or {})
        key = json.dumps(loc, sort_keys=True)
    if key not in cache:
        cache[key] = resolve_source(loc, headers=probes.get("_headers") or ())
    return cache[key]


def _provenance(kind: str, sources: Iterable[tuple[str, Resolved]], **extra: Any) -> dict[str, Any]:
    from merlin.common import provenance as P

    srcs = list(sources)
    paths = [r.path for _role, r in srcs if r.path is not None]
    return {
        "kind": kind,
        "derived": kind != SOURCE_READING,
        "sources": [{"role": role, **r.record()} for role, r in srcs],
        # provenance.source_digest hashes the PATH strings with the bytes, so it is host-specific; the
        # per-source sha256 above is the host-independent identity.
        "source_digest": P.source_digest(paths) if paths else None,
        **extra,
    }


# ---------------------------------------------------------------------------------- module loading


class _Modules:
    """Parsed FIRRTL modules per elaboration, loaded once per derivation."""

    def __init__(self):
        self._cache: dict[tuple[str, str], F.ModuleBody | None] = {}
        self._wanted: dict[str, set[str]] = {}

    def want(self, fir: Path, names: Iterable[str]) -> None:
        self._wanted.setdefault(str(fir), set()).update(names)

    def get(self, fir: Path, name: str) -> F.ModuleBody:
        key = (str(fir), name)
        if key not in self._cache:
            names = self._wanted.get(str(fir), set()) | {name}
            found = F.load_modules(fir, names)
            for n in names:
                self._cache[(str(fir), n)] = found.get(n)
        mod = self._cache[key]
        if mod is None:
            raise Unknown(f"the elaboration {fir.name} defines no module {name!r}")
        return mod


# ------------------------------------------------------------------------------- instruction names


def _instruction_names(target: str) -> dict[int, str]:
    """``{selector: name}`` from the target's derived decode table, or ``{}`` when it is not derivable."""
    # Imported outside the guard: a missing decoder is a broken install, not an absent fact.
    from merlin.compile.analysis.rocc_commands import funct_table_for

    try:
        names = funct_table_for(target).get("names") or {}
    except Exception:  # noqa: BLE001 - names are a rendering aid; selectors stand without them
        return {}
    out: dict[int, str] = {}
    for sel, name in names.items():
        try:
            out[int(str(sel), 0)] = str(name)
        except ValueError:
            continue
    return out


def _roles_by_name(target: str) -> dict[str, list[str]]:
    from merlin.compile.analysis.compute_endpoints import endpoints_for

    try:
        endpoints = endpoints_for(target)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, list[str]] = {}
    for name in _instruction_names(target).values():
        roles = sorted({r for e in endpoints for r in e.roles_of(name)})
        if roles:
            out[name] = roles
    return out


# ------------------------------------------------------------------------ load_completion_ordering


def _entry_fields(leaves: Iterable[str], entries: str) -> set[str]:
    """The field paths of ``entries[i]`` a dependency expression reads (``valid``, ``bits.issued``, ...)."""
    out: set[str] = set()
    for leaf in leaves:
        if not F.ref_matches(leaf, entries):
            continue
        rest = leaf[len(entries) :]
        if rest.startswith("["):
            rest = rest[rest.index("]") + 1 :]
        out.add(rest.lstrip("."))
    return out


def _field_hit(fields: Iterable[str], name: str) -> bool:
    return any(f == name or f.startswith(name + ".") for f in fields)


def _cite(module: F.ModuleBody, line: int, loc: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"module": module.name, "firrtl_line": line}
    c = F.locator_citation(loc)
    if c:
        out.update({"scala_file": c["file"], "scala_line": c["line"]})
    return out


def _membership(mod: F.ModuleBody, node: str, field_pattern: str) -> list[int]:
    if node not in mod.nodes and node not in mod.decls:
        raise Unknown(f"{mod.name} defines no signal {node!r}")
    _leaves, prims, _lines = F.Cone(mod).walk(F.parse_expr(node))
    return sorted(F.eq_constants(prims, field_pattern))


def _ordering(target: str, probes: Mapping[str, Any], mods: _Modules, cache: dict) -> dict[str, Any]:
    spec = probes.get("load_completion_ordering")
    if not isinstance(spec, Mapping):
        raise Unknown("the target declares no `load_completion_ordering` probe")
    src = _source(probes, spec.get("source"), cache)
    if not src.ok:
        raise Unknown(f"elaboration unavailable: {src.why}")
    fir = src.path
    trk = spec["tracker"]
    bar = spec.get("barrier") or {}
    mods.want(
        fir,
        [trk["module"], bar.get("host_module", ""), bar.get("accelerator_module", "")]
        + [(spec.get("sync_decode") or {}).get("module", "")],
    )
    mod = mods.get(fir, trk["module"])
    names = _instruction_names(target)
    queues: Mapping[str, Mapping[str, Any]] = trk["queues"]
    live, issued = trk["entry_fields"]["live"], trk["entry_fields"]["issued"]
    addr_fields = list(trk.get("address_fields") or ())

    classes: dict[str, Any] = {}
    selector_classes: dict[int, list[str]] = {}
    for cls, q in queues.items():
        functs = _membership(mod, q["membership"], trk["command_field"])
        classes[cls] = {"selectors": functs, "names": [names.get(s, str(s)) for s in functs]}
        for s in functs:
            selector_classes.setdefault(s, []).append(cls)
    for cls in classes.values():
        cls["sub_typed_selectors"] = [s for s in cls["selectors"] if len(selector_classes.get(s, [])) > 1]

    pairs: list[dict[str, Any]] = []
    cone = F.Cone(mod)
    for later, lq in queues.items():
        for earlier, eq in queues.items():
            conns = [
                c
                for c in mod.connects
                if F.ref_matches(c.lhs.text(), eq["dependency"]) and F.guards_match(c.guards, lq["guard"])
            ]
            row: dict[str, Any] = {"later": later, "earlier": earlier}
            if not conns:
                row.update({"released_at": UNKNOWN, "why": "no dependency assignment under this class's guard"})
                pairs.append(row)
                continue
            leaves: set[str] = set()
            for c in conns:
                leaves |= cone.leaves(c.rhs)
            fields = _entry_fields(leaves, eq["entries"])
            if _field_hit(fields, issued):
                released = RELEASED_AT_ISSUE
            elif _field_hit(fields, live):
                released = RELEASED_AT_COMPLETION
            else:
                released = UNKNOWN
            row.update(
                {
                    "released_at": released,
                    "address_conditional": any(_field_hit(fields, a) for a in addr_fields),
                    "reads": sorted(f for f in fields if not any(_field_hit([f], a) for a in addr_fields)),
                    "cite": _cite(mod, conns[0].line, conns[0].locator),
                }
            )
            pairs.append(row)

    # The two field meanings the verdicts rest on, checked rather than assumed: `live` is cleared under
    # the retire port, `issued` is set under the issue port.
    def _set_under(field: str, value: int, port: str) -> bool | None:
        hits = []
        for q in queues.values():
            for c in mod.connects:
                t = c.lhs.text()
                if not F.ref_matches(t, q["entries"]):
                    continue
                rest = _entry_fields([t], q["entries"])
                if field not in rest or not isinstance(c.rhs, F.Lit) or c.rhs.value != value:
                    continue
                guard_leaves: set[str] = set()
                gcone = F.Cone(mod, include_guards=True)
                for g in c.guards:
                    guard_leaves |= gcone.leaves(g.cond)
                hits.append(any(F.ref_matches(x, port) for x in guard_leaves))
        return None if not hits else any(hits)

    field_checks = {
        "live_cleared_under_retire_port": _set_under(live, 0, trk["retire_port"]),
        "issued_set_under_issue_port": _set_under(issued, 1, trk["issue_port"]),
    }

    same_class: dict[str, str] = {}
    for cls in queues:
        row = next(p for p in pairs if p["later"] == cls and p["earlier"] == cls)
        if row["released_at"] == RELEASED_AT_ISSUE:
            same_class[cls] = ISSUE_ORDERED_ONLY
        elif row["released_at"] == RELEASED_AT_COMPLETION:
            same_class[cls] = COMPLETION_ORDERED
        else:
            same_class[cls] = UNKNOWN

    barrier = _barrier(bar, trk, queues, mods, fir) if bar else {"status": UNKNOWN, "why": "no barrier probe declared"}
    sync = _sync_decode(spec.get("sync_decode"), mods, fir, names, bar.get("tracker_instance"), target)

    subject = trk.get("subject")
    subject_order = same_class.get(subject, UNKNOWN) if subject else UNKNOWN
    status = DERIVED if subject_order != UNKNOWN and all(v is not False for v in field_checks.values()) else UNKNOWN
    value: dict[str, Any] = {
        "subject_class": subject,
        "same_class": same_class,
        "classes": classes,
        "pairs": pairs,
        "field_checks": field_checks,
        "completion_barrier": barrier,
        "sync_role_instructions": sync,
    }
    value["summary"] = _ordering_summary(subject, same_class, barrier)
    return {
        "status": status,
        "value": value,
        "provenance": _provenance(
            "firrtl_structural",
            [("elaboration", src)],
            reading=(
                "for each pair of command classes, the expression the dependency tracker assigns to a NEW "
                "command's dependency on an OLDER entry, walked to the older entry's fields: reading its "
                "issue marker releases the dependency at issue; reading only its live bit holds it until "
                "the entry retires"
            ),
        ),
        "needs_review": False,
    }


def _barrier(bar, trk, queues, mods: _Modules, fir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"instruction": bar.get("instruction"), "executes_on": "host"}
    try:
        host = mods.get(fir, bar["host_module"])
        stall_leaves = _visits(host, bar["stall"], [bar["accelerator_busy_input"], bar["decode_field"]])
        out["stall_waits_on_accelerator_busy"] = bar["accelerator_busy_input"] in stall_leaves
        out["stall_triggered_by_instruction_decode"] = bar["decode_field"] in stall_leaves
        acc = mods.get(fir, bar["accelerator_module"])
        acc_leaves = F.Cone(acc).leaves(F.parse_expr(bar["accelerator_busy_output"]))
        inst = bar["tracker_instance"]
        out["accelerator_busy_includes_tracker"] = any(
            F.ref_matches(x, f"{inst}.{trk['busy_output']}") for x in acc_leaves
        )
        tracker = mods.get(fir, trk["module"])
        busy_leaves = F.Cone(tracker).leaves(F.parse_expr(trk["busy_output"]))
        live = trk["entry_fields"]["live"]
        out["tracker_busy_covers_every_queue_until_retired"] = all(
            live in _entry_fields(busy_leaves, q["entries"]) for q in queues.values()
        )
        checks = [
            out["stall_waits_on_accelerator_busy"],
            out["stall_triggered_by_instruction_decode"],
            out["accelerator_busy_includes_tracker"],
            out["tracker_busy_covers_every_queue_until_retired"],
        ]
        out["status"] = DERIVED if all(checks) else UNKNOWN
        if not all(checks):
            out["why"] = "one link of stall -> accelerator busy -> tracker entries was not found"
        out["cite"] = {"host_module": host.name, "accelerator_module": acc.name}
    except (Unknown, KeyError, F.FirrtlParseError) as exc:
        out.update({"status": UNKNOWN, "why": str(exc)})
    return out


def _visits(mod: F.ModuleBody, signal: str, targets: list[str]) -> set[str]:
    """Which of ``targets`` the value of ``signal`` reads, stopping at each target (so a wire target is
    recorded as reached, not expanded into its own fan-in)."""
    reached: set[str] = set()
    seen: set[tuple] = set()
    work: list[Any] = [F.parse_expr(signal)]
    while work:
        e = work.pop()
        for r in F._refs_in(e):
            key = (r.root, r.segments)
            if key in seen:
                continue
            seen.add(key)
            text = r.text()
            hit = [t for t in targets if F.ref_matches(text, t)]
            if hit:
                reached.update(hit)
                continue
            if r.root in mod.nodes:
                work.append(mod.nodes[r.root][0])
            elif (mod.decls.get(r.root) or ("",))[0] in F._DRIVEN_DECLS:
                work.extend(c.rhs for c in mod.connects_to(r.root) if F._compatible(c.lhs, r))
    return reached


def _sync_decode(spec, mods: _Modules, fir: Path, names: dict[int, str], tracker_inst, target) -> list[dict[str, Any]]:
    if not isinstance(spec, Mapping):
        return []
    out: list[dict[str, Any]] = []
    try:
        mod = mods.get(fir, spec["module"])
    except Unknown as exc:
        return [{"status": UNKNOWN, "why": str(exc)}]
    roles = _roles_by_name(target)
    for sel in spec.get("selectors") or ():
        try:
            functs = _membership(mod, sel["membership"], spec["command_field"])
        except Unknown as exc:
            out.append({"status": UNKNOWN, "why": str(exc)})
            continue
        guarded = [c for c in mod.connects if F.guards_match(c.guards, sel["guard"])]
        effects = sorted({f"{c.lhs.text()} = {c.rhs.value if isinstance(c.rhs, F.Lit) else 'expr'}" for c in guarded})
        read = set()
        for c in guarded:
            read |= F.Cone(mod).leaves(c.rhs)
        touches = bool(tracker_inst) and any(
            x.startswith(f"{tracker_inst}.") for x in read | {c.lhs.text() for c in guarded}
        )
        for f in functs:
            name = names.get(f, str(f))
            out.append(
                {
                    "selector": f,
                    "name": name,
                    "declared_roles": roles.get(name),
                    "decoded_effects": effects,
                    "touches_dependency_tracker": touches,
                    "is_completion_barrier": False if not touches else UNKNOWN,
                    "status": DERIVED,
                }
            )
    return out


def _ordering_summary(subject, same_class, barrier) -> str:
    order = same_class.get(subject, UNKNOWN)
    if order == ISSUE_ORDERED_ONLY:
        text = (
            f"{subject} commands are ISSUE-ORDERED ONLY: a later {subject} waits for an earlier one to issue, "
            f"not to complete, so two {subject}s into the same local rows (an overwriting one and an "
            f"accumulating one) can complete in either order"
        )
    elif order == COMPLETION_ORDERED:
        text = f"{subject} commands are COMPLETION-ORDERED: a later {subject} waits until an earlier one retires"
    else:
        return f"{subject} command ordering is UNKNOWN"
    if barrier.get("status") == DERIVED:
        text += (
            f"; `{barrier['instruction']}` (host) stalls until the accelerator reports idle, which it does only "
            f"once every tracked command has retired -- it is the completion barrier"
        )
    return text


# --------------------------------------------------------------------------- load_scale_saturation


def _c_tokens(body: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            i += 1
        elif c.isalnum() or c == "_" or c == ".":
            j = i
            while j < n and (body[j].isalnum() or body[j] in "_."):
                j += 1
            out.append(body[i:j])
            i = j
        else:
            two = body[i : i + 2]
            if two in (">=", "<=", "==", "!=", "&&", "||", "<<", ">>"):
                out.append(two)
                i += 2
            else:
                out.append(c)
                i += 1
    return out


def _stdint_bits(name: str) -> tuple[int, bool] | None:
    """``(bits, signed)`` of a C ``<stdint.h>`` type or limit name (``int8_t``, ``INT8_MAX``, ...)."""
    t = name.strip()
    signed = True
    low = t.lower()
    if low.startswith("uint"):
        signed, digits = False, low[4:]
    elif low.startswith("int"):
        digits = low[3:]
    else:
        return None
    num = ""
    for ch in digits:
        if ch.isdigit():
            num += ch
        else:
            break
    rest = digits[len(num) :]
    if not num or rest not in ("_t", "_max", "_min"):
        return None
    return int(num), signed


def _type_bits(model, alias: str, depth: int = 0) -> tuple[int, bool] | None:
    got = _stdint_bits(alias)
    if got is not None or depth > 4:
        return got
    for name, underlying, _line in model.typedefs:
        if name == alias:
            return _type_bits(model, underlying.split()[-1], depth + 1)
    return None


def _bound_value(model, tok: str) -> int | None:
    std = _stdint_bits(tok)
    if std is not None and tok.upper() == tok:  # a limit macro (INT8_MAX), not a type
        bits, signed = std
        if tok.upper().endswith("_MAX"):
            return (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
        return -(1 << (bits - 1)) if signed else 0
    try:
        return int(tok, 0)
    except ValueError:
        pass
    m = model.macro(tok)
    return m.int_value if m is not None else None


def analyze_scale_macro(model, name: str) -> dict[str, Any]:
    """What a generated load-scale macro does to its argument, read from its body's tokens."""
    macro = model.macro(name)
    if macro is None:
        raise Unknown(f"the header defines no macro {name!r}")
    toks = _c_tokens(macro.body)
    params = list(macro.params)
    stripped = [t for t in toks if t not in ("(", ")")]
    if params and stripped == [params[0]]:
        return {"macro": name, "body": macro.body.strip(), "line": macro.line, "scaled": False, "saturates": False}
    upper: list[int] = []
    lower: list[int] = []
    for i, t in enumerate(toks[:-1]):
        if t in (">", ">=", "<", "<="):
            v = _bound_value(model, toks[i + 1])
            if v is not None:
                (upper if t in (">", ">=") else lower).append(v)
    casts = [
        toks[i + 1]
        for i in range(len(toks) - 2)
        if toks[i] == "(" and toks[i + 2] == ")" and _type_bits(model, toks[i + 1]) is not None
    ]
    calls = sorted({toks[i] for i in range(len(toks) - 1) if toks[i + 1] == "(" and toks[i][:1].isalpha()})
    out: dict[str, Any] = {
        "macro": name,
        "body": macro.body.strip(),
        "line": macro.line,
        "scaled": True,
        "saturates": bool(upper and lower),
        "applies": calls,
        "result_types": sorted(set(casts)),
    }
    if upper and lower:
        hi, lo = min(upper), max(lower)
        bits = max(hi.bit_length() + 1, (-lo - 1).bit_length() + 1) if lo < 0 else hi.bit_length()
        out.update({"range": [lo, hi], "saturates_to_bits": bits, "signed": lo < 0})
    return out


def _saturation(target: str, probes: Mapping[str, Any], mods: _Modules, cache: dict) -> dict[str, Any]:
    from merlin.targetgen.capability_discovery import parse_c_header

    spec = probes.get("load_scale_saturation")
    if not isinstance(spec, Mapping):
        raise Unknown("the target declares no `load_scale_saturation` probe")
    hdr = _source(probes, spec.get("header"), cache)
    if not hdr.ok:
        raise Unknown(f"header unavailable: {hdr.why}")
    model = parse_c_header(hdr.path)
    operand = analyze_scale_macro(model, spec["macros"]["operand"])
    accumulator = (
        analyze_scale_macro(model, spec["macros"]["accumulator"]) if spec["macros"].get("accumulator") else None
    )
    op_bits = _type_bits(model, spec["types"]["operand"])
    acc_bits = _type_bits(model, spec["types"]["accumulator"])
    sources: list[tuple[str, Resolved]] = [("header", hdr)]

    entry: dict[str, Any] = {"status": UNKNOWN}
    ae = spec.get("accumulator_entry")
    if isinstance(ae, Mapping):
        el = _source(probes, ae.get("source"), cache)
        if el.ok:
            sources.append(("elaboration", el))
            try:
                mods.want(el.path, [ae["module"]])
                mod = mods.get(el.path, ae["module"])
                entry = _accumulator_entry(mod, ae["accumulator_write"], ae["scaled_operand"])
            except (Unknown, F.FirrtlParseError) as exc:
                entry = {"status": UNKNOWN, "why": str(exc)}
        else:
            entry = {"status": UNKNOWN, "why": f"elaboration unavailable: {el.why}"}

    citation = _scala_citation(spec.get("source_citation"), cache)
    corroboration = _corroborate(probes, spec.get("corroboration"), cache)

    stage = UNKNOWN
    stage_kind = None
    if entry.get("status") == DERIVED and op_bits and acc_bits and entry.get("enters_at_bits") == [op_bits[0]]:
        if op_bits[0] < acc_bits[0]:
            stage, stage_kind = "before_accumulate", DERIVED
    elif citation.get("found"):
        stage, stage_kind = "before_accumulate", SOURCE_READING
    operand.update({"stage": stage, "stage_provenance": stage_kind})
    if not operand.get("scaled"):
        status = DERIVED
    elif operand.get("saturates") and stage_kind == DERIVED:
        status = DERIVED
    elif operand.get("saturates") and stage_kind == SOURCE_READING:
        status = SOURCE_READING
    else:
        status = UNKNOWN
    value = {
        "operand_load": operand,
        "accumulator_load": accumulator,
        "operand_bits": op_bits[0] if op_bits else None,
        "accumulator_bits": acc_bits[0] if acc_bits else None,
        "accumulator_entry": entry,
        "source_citation": citation,
        "corroboration": corroboration,
    }
    value["summary"] = _saturation_summary(operand, value)
    kind = (
        "generated_header+firrtl_structural"
        if stage_kind == DERIVED
        else (SOURCE_READING if stage_kind == SOURCE_READING else "generated_header")
    )
    return {
        "status": status,
        "value": value,
        "provenance": _provenance(kind, sources, pin_citation=citation.get("citation")),
        "needs_review": status == SOURCE_READING,
    }


def _accumulator_entry(mod: F.ModuleBody, write_pattern: str, scaled: str) -> dict[str, Any]:
    conns = [c for c in mod.connects if F.ref_matches(c.lhs.text(), write_pattern)]
    if not conns:
        raise Unknown(f"{mod.name} assigns nothing matching {write_pattern!r}")
    widths: set[int] = set()
    cite = None
    cone = F.Cone(mod)
    for c in conns:
        leaves, prims, _lines = cone.walk(c.rhs)
        if not any(F.ref_matches(x, scaled) for x in leaves):
            continue
        cite = cite or _cite(mod, c.line, c.locator)
        for p in prims:
            # a sign bit taken out of the scaled operand: `bits(x, k, k)` means x is k+1 bits wide
            if p.op == "bits" and len(p.args) == 3 and isinstance(p.args[0], F.Ref):
                hi, lo = p.args[1], p.args[2]
                if isinstance(hi, F.IntArg) and isinstance(lo, F.IntArg) and hi.value == lo.value:
                    if F.ref_matches(p.args[0].text(), scaled):
                        widths.add(hi.value + 1)
    if not widths:
        raise Unknown(f"no assignment to {write_pattern!r} sign-extends a value read from {scaled!r}")
    return {
        "status": DERIVED,
        "enters_at_bits": sorted(widths),
        "reading": "the accumulator write port's data is the scaled operand sign-extended from its own width",
        "cite": cite,
    }


def _scala_citation(spec, cache) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        return {"found": False, "why": "no source citation declared"}
    r = _source({}, {k: spec[k] for k in ("pin", "path") if k in spec}, cache)
    out: dict[str, Any] = {"pin": spec.get("pin"), "path": spec.get("path"), "line": spec.get("line")}
    if not r.ok:
        return {**out, "found": False, "why": r.why}
    lines = r.path.read_text(encoding="utf-8", errors="replace").splitlines()
    ln = int(spec.get("line") or 0)
    text = lines[ln - 1] if 0 < ln <= len(lines) else ""
    out.update(
        {
            "found": str(spec.get("contains") or "") in text,
            "text": text.strip(),
            "sha256": r.record().get("sha256"),
            "citation": r.registry.get("citation"),
            "provenance": SOURCE_READING,
            "needs_review": True,
        }
    )
    return out


def _corroborate(probes, spec, cache) -> dict[str, Any] | None:
    if not isinstance(spec, Mapping):
        return None
    r = _source(probes, spec.get("source"), cache)
    if not r.ok:
        return {"status": UNKNOWN, "why": r.why}
    text = r.path.read_text(encoding="utf-8", errors="replace")
    want = list(spec.get("contains_all") or ())
    return {
        "status": "present" if all(w in text for w in want) else "absent",
        "contains_all": want,
        "sha256": r.record().get("sha256"),
    }


def _saturation_summary(operand: Mapping[str, Any], value: Mapping[str, Any]) -> str:
    if not operand.get("scaled"):
        return "an operand-width load is not scaled, so it does not saturate"
    if not operand.get("saturates"):
        return "a scaled operand-width load is scaled without saturation (it wraps or is unbounded)"
    lo, hi = operand["range"]
    stage = operand.get("stage")
    where = "BEFORE it reaches the accumulator" if stage == "before_accumulate" else "at an UNKNOWN stage"
    acc = value.get("accumulator_load") or {}
    tail = "" if acc.get("scaled", True) else "; an accumulator-width load is not scaled at all"
    return (
        f"a scaled operand-width load saturates to [{lo}, {hi}] ({operand['saturates_to_bits']}-bit) {where}: "
        f"a load multiplier > 1 clips every element whose scaled magnitude exceeds that range, and the sum "
        f"of two such loads is a sum of clipped values -- load at a multiplier <= 1 and carry the gain on "
        f"the readout scale{tail}"
    )


# ------------------------------------------------------------------------ accumulator_readout_width


def _readout(target: str, probes: Mapping[str, Any], mods: _Modules, cache: dict) -> dict[str, Any]:
    from merlin.targetgen.capability_discovery import parse_c_header

    spec = probes.get("accumulator_readout_width")
    if not isinstance(spec, Mapping):
        raise Unknown("the target declares no `accumulator_readout_width` probe")
    machines: dict[str, Any] = {}
    sources: list[tuple[str, Resolved]] = []
    for machine, mspec in (spec.get("machines") or {}).items():
        row: dict[str, Any] = {}
        hdr = _source(probes, mspec.get("header"), cache)
        if not hdr.ok:
            machines[machine] = {"status": UNKNOWN, "why": f"no header: {hdr.why}"}
            continue
        sources.append((f"{machine}.header", hdr))
        model = parse_c_header(hdr.path)
        full = model.macro(spec["macros"]["full"]) is not None
        narrow = model.macro(spec["macros"]["narrow"]) is not None
        fbits = _type_bits(model, spec["types"]["full"])
        nbits = _type_bits(model, spec["types"]["narrow"])
        dim_m = model.macro(spec["row_elements_macro"])
        dim = dim_m.int_value if dim_m is not None else None
        readable = sorted(b for b, on in ((fbits and fbits[0], full), (nbits and nbits[0], narrow)) if on and b)
        row["header"] = {
            "full_width_macro": full,
            "narrow_width_macro": narrow,
            "readout_bits": readable,
            "row_elements": dim,
            "sha256": hdr.record().get("sha256"),
            "registry_abi_header": _registry_abi(machine, hdr),
        }
        row["full_width_readout"] = full
        row["status"] = DERIVED
        row["provenance"] = "generated_header"
        fir = _source(probes, mspec.get("firrtl"), cache) if mspec.get("firrtl") else None
        if fir is not None and fir.ok and dim:
            sources.append((f"{machine}.firrtl", fir))
            try:
                w = spec["writer"]
                mods.want(fir.path, [w["module"]])
                mod = mods.get(fir.path, w["module"])
                width = F.type_width(F.field_type(mod.port_type(w["port"]), w["path"]))
                if width is None:
                    raise Unknown(f"{w['module']}.{w['port']} has no sized field {w['path']}")
                per = width // dim
                row["firrtl"] = {
                    "writer_data_bits": width,
                    "bits_per_element": per,
                    "full_width_datapath": bool(fbits and per >= fbits[0]),
                    "correspondence": fir.locator.get("correspondence") or "declared",
                }
                agrees = row["firrtl"]["full_width_datapath"] == full
                row["firrtl"]["agrees_with_header"] = agrees
                row["provenance"] = "generated_header+firrtl_structural"
                if not agrees:
                    row["status"] = CONFLICT
                    row["why"] = "the header and the elaboration disagree on the readout width; neither is believed"
            except (Unknown, F.FirrtlParseError, KeyError) as exc:
                row["firrtl"] = {"status": UNKNOWN, "why": str(exc)}
        elif fir is not None:
            row["firrtl"] = {"status": UNKNOWN, "why": fir.why or "no row-element count to divide by"}
        machines[machine] = row
    if not machines:
        raise Unknown("no machine is declared")
    statuses = {m["status"] for m in machines.values()}
    status = (
        DERIVED
        if statuses == {DERIVED}
        else (CONFLICT if CONFLICT in statuses else (DERIVED if DERIVED in statuses else UNKNOWN))
    )
    narrow_only = sorted(m for m, r in machines.items() if r.get("status") == DERIVED and not r["full_width_readout"])
    summary = (
        "full-width accumulator readout on: "
        + ", ".join(sorted(m for m, r in machines.items() if r.get("full_width_readout")) or ["none"])
        + "; NARROW readout only (a full-width mvout of the accumulator cannot return its int rows) on: "
        + ", ".join(narrow_only or ["none"])
    )
    return {
        "status": status,
        "value": {"machines": machines, "narrow_only_machines": narrow_only, "summary": summary},
        "provenance": _provenance("generated_header+firrtl_structural", sources),
        "needs_review": False,
    }


def _accumulate_on_load(target: str, probes: Mapping[str, Any], mods: _Modules, cache: dict) -> dict[str, Any]:
    """See :mod:`.accumulate_fact`: a load that adds into the accumulator, selected by its address."""
    from . import accumulate_fact as AF

    spec = probes.get(AF.FACT)
    if not isinstance(spec, Mapping):
        raise Unknown(f"the target declares no `{AF.FACT}` probe")
    el = _source(probes, spec.get("source"), cache)
    if not el.ok:
        raise Unknown(f"elaboration unavailable: {el.why}")
    names = [str((spec.get(part) or {}).get("module")) for part in ("store", "load")]
    mods.want(el.path, names)
    try:
        value = AF.derive(spec, {n: mods.get(el.path, n) for n in names}, _roles_by_name(target))
    except (AF.Unreadable, F.FirrtlParseError) as exc:
        raise Unknown(str(exc)) from exc
    value["summary"] = (
        f"a load ({', '.join(value['roles'])}) can add into the accumulator: the store writes the adder's sum "
        "when the write's accumulate bit is set, and that bit is the load address's own accumulate field"
    )
    return {
        "status": DERIVED,
        "value": value,
        "provenance": _provenance("firrtl_structural", [("elaboration", el)]),
        "needs_review": False,
    }


def _registry_abi(machine: str, hdr: Resolved) -> dict[str, Any]:
    """Whether the hardware registry declares an ABI header digest for ``machine`` and whether it agrees."""
    try:
        declared_by, digest = registry_abi_header(machine)
    except Exception as exc:  # noqa: BLE001
        return {"status": UNKNOWN, "why": str(exc)}
    if digest is None:
        return {"status": "undeclared"}
    return {"declared_by": declared_by, "agrees": hdr.record().get("sha256") == digest}


# ---------------------------------------------------------------------------------------- derive


def derive(
    target: str,
    *,
    probes_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    headers: Iterable[str | Path] = (),
    only: Iterable[str] | None = None,
) -> dict:
    """Derive every semantic fact ``target`` declares probes for (``only`` narrows it to the named
    facts). Never raises for a missing source: each fact that cannot be read is recorded UNKNOWN with its
    reason.

    ``overrides`` replaces named ``sources`` locators (``{"elaboration": {"file": "..."}}``): the hook a
    test uses to point the same probes at a mutated source.  ``headers`` are candidate header files an
    ``abi_header_of`` locator may resolve to, by content."""
    try:
        probes = load_probes(target, probes_file)
    except Exception as exc:  # noqa: BLE001 -- an unreadable probes file is UNKNOWN, never a crash
        return {
            "schema": SCHEMA,
            "target": target,
            "status": UNKNOWN,
            "why": f"{type(exc).__name__}: {exc}",
            "facts": {},
        }
    doc: dict[str, Any] = {"schema": SCHEMA, "target": target, "generated_by": GENERATOR}
    if probes is None:
        doc.update({"status": UNKNOWN, "why": f"{target} declares no {PROBES_FILE}", "facts": {}})
        return doc
    if overrides:
        probes = dict(probes)
        probes["sources"] = {**(probes.get("sources") or {}), **overrides}
    probes = dict(probes)
    probes["_headers"] = [str(h) for h in headers]
    p = Path(probes_file) if probes_file else probes_path(target)
    doc["probes"] = {"file": _repo_rel(p), "sha256": _sha256(p)}
    mods = _Modules()
    cache: dict[str, Resolved] = {}
    facts: dict[str, Any] = {}
    wanted = set(only) if only is not None else set(FACT_NAMES)
    for name, fn in (
        ("load_completion_ordering", _ordering),
        ("load_scale_saturation", _saturation),
        ("accumulator_readout_width", _readout),
        ("accumulate_on_load", _accumulate_on_load),
    ):
        if name not in wanted:
            continue
        try:
            facts[name] = fn(target, probes, mods, cache)
        except Unknown as exc:
            facts[name] = {"status": UNKNOWN, "unknown_reason": str(exc), "needs_review": False}
        except Exception as exc:  # noqa: BLE001 - any failure to read a fact is that fact UNKNOWN, with why
            facts[name] = {"status": UNKNOWN, "unknown_reason": f"{type(exc).__name__}: {exc}", "needs_review": False}
        facts[name]["validation"] = _validation((probes.get("validation") or {}).get(name))
    doc["facts"] = facts
    doc["contract_conflicts"] = _contract_conflicts(target, facts)
    doc["status"] = DERIVED if all(f["status"] in (DERIVED, SOURCE_READING) for f in facts.values()) else "partial"
    return doc


def readout_machines(target: str, *, headers: Iterable[str | Path] = ()) -> dict[str, Any]:
    """The DERIVED ``accumulator_readout_width`` fact for ``target`` (``{}`` when not derivable)."""
    doc = derive(target, headers=headers, only=("accumulator_readout_width",))
    return dict((doc.get("facts") or {}).get("accumulator_readout_width") or {})


def _validation(rows) -> list[dict[str, Any]]:
    out = []
    for row in rows or ():
        rec = dict(row)
        ev = row.get("evidence")
        if isinstance(ev, Mapping):
            r = resolve_source(ev)
            present = r.path is not None and (r.path.exists())
            if r.path is None and "out" in ev:
                from merlin.common.paths import out_dir

                present = (Path(out_dir()) / str(ev["out"])).exists()
            rec["evidence_present_at_derivation"] = bool(present)
            if r.ok:
                rec["evidence_sha256"] = r.record().get("sha256")
        out.append(rec)
    return out


def _contract_conflicts(target: str, facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A declared contract statement a derived fact contradicts. Reported, never silently reconciled."""
    try:
        import yaml

        from .facts import target_contract_path

        contract = yaml.safe_load(Path(target_contract_path(target)).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - no contract is no conflict to report
        return []
    out = []
    declared = ((contract.get("hardware") or {}).get("memory_model") or contract.get("memory_model") or {}).get(
        "hazard_resolution"
    )
    ordering = facts.get("load_completion_ordering") or {}
    same = ((ordering.get("value") or {}).get("same_class")) or {}
    if declared == "interlocked" and ISSUE_ORDERED_ONLY in same.values():
        out.append(
            {
                "contract_key": "memory_model.hazard_resolution",
                "declared": declared,
                "derived": {k: v for k, v in same.items() if v == ISSUE_ORDERED_ONLY},
                "why": (
                    "the contract says the accelerator resolves every hazard itself, but a same-class "
                    "dependency is released at ISSUE: a write-after-write between two such commands into "
                    "the same rows is NOT interlocked, so reordering a legal stream CAN change the answer"
                ),
            }
        )
    return out


def _repo_rel(p: Path) -> str:
    try:
        from merlin.common.paths import repo_root

        return str(Path(p).resolve().relative_to(Path(repo_root()).resolve()))
    except (ValueError, OSError):
        return str(p)


# ------------------------------------------------------------------------------------------ reading


def class_members(ordering_value: Mapping[str, Any], cls: str | None) -> list[str]:
    """The instructions that ARE commands of ``cls``: its selectors minus the sub-typed ones (a selector
    that also appears under another class is a multi-purpose command whose sub-type picks the class)."""
    row = (ordering_value.get("classes") or {}).get(cls) or {}
    if not row.get("selectors"):
        return list(row.get("names") or ())
    shared = set(row.get("sub_typed_selectors") or ())
    return [n for s, n in zip(row.get("selectors") or (), row.get("names") or (), strict=False) if s not in shared]


def compact(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The agent-facing form: per fact its status, provenance kind, review flag and one-line summary,
    plus the values a program decision needs. Small enough to embed in a stage context."""
    out: dict[str, Any] = {
        "schema": "semantic_facts_brief_v1",
        "target": doc.get("target"),
        "status": doc.get("status"),
    }
    if doc.get("why"):
        out["why"] = doc["why"]
    facts: dict[str, Any] = {}
    for name in FACT_NAMES:
        f = (doc.get("facts") or {}).get(name)
        if not isinstance(f, Mapping):
            facts[name] = {"status": UNKNOWN, "why": "not derived"}
            continue
        v = f.get("value") or {}
        row: dict[str, Any] = {
            "status": f.get("status"),
            "provenance": (f.get("provenance") or {}).get("kind"),
            "needs_review": bool(f.get("needs_review")),
        }
        if f.get("status") == UNKNOWN:
            row["why"] = f.get("unknown_reason")
        if v.get("summary"):
            row["summary"] = v["summary"]
        if name == "load_completion_ordering" and v:
            row["same_class"] = v.get("same_class")
            row["subject_class"] = v.get("subject_class")
            row["subject_members"] = class_members(v, v.get("subject_class"))
            b = v.get("completion_barrier") or {}
            row["completion_barrier"] = {
                "instruction": b.get("instruction"),
                "executes_on": b.get("executes_on"),
                "status": b.get("status"),
            }
            row["not_barriers"] = [
                s["name"] for s in v.get("sync_role_instructions") or () if s.get("is_completion_barrier") is False
            ]
        elif name == "load_scale_saturation" and v:
            op = v.get("operand_load") or {}
            row["operand_load"] = {k: op.get(k) for k in ("scaled", "saturates", "range", "saturates_to_bits", "stage")}
        elif name == "accumulator_readout_width" and v:
            row["full_width_readout"] = {
                m: r.get("full_width_readout") if r.get("status") == DERIVED else r.get("status")
                for m, r in (v.get("machines") or {}).items()
            }
        row["validated"] = [x.get("engine") for x in f.get("validation") or () if x.get("agrees")]
        facts[name] = row
    out["facts"] = facts
    if doc.get("contract_conflicts"):
        out["contract_conflicts"] = [c.get("why") for c in doc["contract_conflicts"]]
    return out


def brief(doc: Mapping[str, Any]) -> str:
    """The Markdown section appended to a target's ISA brief. Empty when the target has no document."""
    if not doc or doc.get("status") == UNKNOWN and not doc.get("facts"):
        return ""
    c = compact(doc)
    lines = [
        "",
        f"## Semantic facts: {c.get('target')}",
        (
            "_What the hardware DOES with a legal command stream, derived from its own elaboration and generated "
            "header. `derived` = read structurally; `source_reading` = cited from source and needs review; "
            "UNKNOWN = could not be read (never a default)._"
        ),
        "",
    ]
    for name in FACT_NAMES:
        row = c["facts"][name]
        head = f"- **{name}** [{row.get('status')}; {row.get('provenance') or 'n/a'}"
        head += "; needs review]" if row.get("needs_review") else "]"
        lines.append(f"{head}: {row.get('summary') or row.get('why') or 'no summary'}")
        if name == "load_completion_ordering" and row.get("same_class"):
            lines.append(f"  - per class, a later command vs an earlier one of the SAME class: `{row['same_class']}`")
            if row.get("subject_members"):
                lines.append(f"  - {row.get('subject_class')} class instructions: {', '.join(row['subject_members'])}")
            if row.get("not_barriers"):
                lines.append(
                    f"  - NOT a completion barrier (its decode never touches the dependency tracker): "
                    f"{', '.join(row['not_barriers'])}"
                )
        if row.get("validated"):
            lines.append(f"  - validated against measured behaviour on: {'; '.join(row['validated'])}")
    for why in c.get("contract_conflicts") or ():
        lines.append(f"- **contract conflict**: {why}")
    return "\n".join(lines) + "\n"


def task_paragraph(compact_doc: Mapping[str, Any] | None) -> str:
    """One plain-text paragraph of :func:`compact`'s facts, for a task statement. ``""`` when there are none.

    Takes the COMPACT form because that is what a stage context already carries
    (``declared_instruction_set.semantic_facts``), so a task renderer needs nothing else."""
    if not isinstance(compact_doc, Mapping) or not compact_doc.get("facts"):
        return ""
    parts = []
    for name in FACT_NAMES:
        row = (compact_doc.get("facts") or {}).get(name) or {}
        tag = f"{row.get('status')}, {row.get('provenance') or 'n/a'}"
        if row.get("needs_review"):
            tag += ", needs review"
        parts.append(f"{name} [{tag}]: {row.get('summary') or row.get('why') or UNKNOWN}.")
    return (
        "The target's semantic facts (what its hardware DOES with a legal command stream, derived from its "
        "own elaboration and header): " + " ".join(parts)
    )


def brief_for(target: str, *, headers: Iterable[str | Path] = ()) -> str:
    """:func:`brief` of the derived document, or ``""`` for a target that declares no probes."""
    doc = derive(target, headers=headers)
    return brief(doc) if doc.get("facts") else ""


# --------------------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-semantic-facts", description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True)
    ap.add_argument("--header", action="append", default=[], help="a candidate header file (by content)")
    ap.add_argument("--brief", action="store_true", help="print the agent-facing brief of the derived document")
    args = ap.parse_args(argv)
    if args.brief:
        sys.stdout.write(brief_for(args.target, headers=args.header))
        return 0
    print(json.dumps(derive(args.target, headers=args.header), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
