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

from merlin.target.rtl.firrtl import structure as F

# ---- semantic_facts ----------------------------------------------------------------------------

SCHEMA = "semantic_facts_v1"


PROBES_SCHEMA = "semantic_probes_v1"


GENERATOR = "merlin.target.rtl.firrtl.semantic_sources"


PROBES_FILE = "semantic_probes.yaml"


FACT_NAMES = ("load_completion_ordering", "load_scale_saturation", "accumulator_readout_width", "accumulate_on_load")


DERIVED = "derived"


SOURCE_READING = "source_reading"


UNKNOWN = "UNKNOWN"


CONFLICT = "CONFLICT"


RELEASED_AT_ISSUE = "issue"


RELEASED_AT_COMPLETION = "completion"


ISSUE_ORDERED_ONLY = "issue_ordered_only"


COMPLETION_ORDERED = "completion_ordered"


class Unknown(Exception):
    """A fact that could not be read, with the reason. Caught per fact, never per document."""


def contracts_dir(target: str) -> Path:
    """The contracts directory of the provider ``target`` resolves to (the selected support provider)."""
    from merlin.target.registry.identity import resolve

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

    from merlin.common.provenance_claims import pins_path

    doc = yaml.safe_load(pins_path().read_text(encoding="utf-8")) or {}
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
    from merlin.common.provenance_claims import load_artifacts, verify_artifact

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
            check = verify_artifact(str(loc["artifact"]))
            registry = {"artifact": loc["artifact"], "digest_matches": bool(check.ok), "gaps": list(check.gaps)}
            art = load_artifacts().get(str(loc["artifact"]))
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


def _instruction_names(target: str) -> dict[int, str]:
    """``{selector: name}`` from the target's derived decode table, or ``{}`` when it is not derivable."""
    # Imported outside the guard: a missing decoder is a broken install, not an absent fact.
    from merlin.analysis.observation.rocc_commands import funct_table_for

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
    from merlin.analysis.observation.compute_endpoints import endpoints_for

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
