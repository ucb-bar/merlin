"""``accumulate_on_load``: can a plain LOAD add into the accumulator, chosen per command by its address?

A two-operand sum needs no contraction if the accumulator can add an arriving row to what it holds.
Whether it can, and what decides it, is written in the elaborated design, and this reads it there --
never from a loop sequencer. Two structural readings, both of the target's own FIRRTL, plus the
instruction roles the target declares for the commands that reach them:

* the STORE: the accumulator memory's write data is the adder's sum when the write's accumulate bit
  is set, and the arriving data otherwise (``mux(<accumulate bit>, <adder sum>, <data>)``);
* the LOAD PATH: the accumulator bank's write accumulate bit is driven from the local address a load
  command carries (the address metadata's accumulate field, set by the program), and the module's
  load request delivers that field into the path.

The names (modules, signals, the select and sum fields) are the target's, declared in its
``semantic_probes.yaml`` under ``accumulate_on_load``; the library compares what it finds as data.
The roles are declared there too and checked against the closed role vocabulary and the target's
own instruction roles, so a policy that prohibits a role (a hardware-loop descriptor, say) can tell
whether this capability depends on one. A reading that fails is UNKNOWN with the reason.

The queues between the load request and the bank write (a reader, a scale unit, a pixel repeater)
are crossed by their declared field names, not traced through each submodule: the reading proves
the request field enters the path and the path's output drives the bank, and says so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from . import firrtl_struct as F

FACT = "accumulate_on_load"


class Unreadable(Exception):
    """The fact cannot be read from these sources; the message is the reason."""


def _cite(module: F.ModuleBody, line: int, loc: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"module": module.name, "firrtl_line": line}
    c = F.locator_citation(loc)
    if c:
        out.update({"scala_file": c["file"], "scala_line": c["line"]})
    return out


def accumulating_store(mod: F.ModuleBody, write_data: str, select: str, summed: str) -> dict[str, Any]:
    """The memory's write data is ``mux(select, summed, data)``: it adds when the write says so."""
    conns = [c for c in mod.connects if F.ref_matches(c.lhs.text(), write_data)]
    if not conns:
        raise Unreadable(f"{mod.name} assigns nothing matching {write_data!r}")
    cone = F.Cone(mod)
    for c in conns:
        _leaves, prims, _lines = cone.walk(c.rhs)
        for p in prims:
            if p.op != "mux" or len(p.args) != 3 or not isinstance(p.args[0], F.Ref):
                continue
            arms = [a.text() for a in p.args[1:] if isinstance(a, F.Ref)]
            if F.ref_matches(p.args[0].text(), select) and any(F.ref_matches(a, summed) for a in arms):
                return {
                    "write_data": write_data,
                    "select": select,
                    "sum": summed,
                    "reading": "the store writes the adder's sum when the write's accumulate bit is set",
                    "cite": _cite(mod, c.line, c.locator),
                }
    raise Unreadable(f"no write to {write_data!r} in {mod.name} selects {summed!r} by {select!r}")


def load_selected_accumulate(
    mod: F.ModuleBody, write_select: str, carried_by: Iterable[str], request: str
) -> dict[str, Any]:
    """The bank's accumulate bit comes from the load path, and the load request feeds that path."""
    carried = list(carried_by)
    conns = [c for c in mod.connects if F.ref_matches(c.lhs.text(), write_select)]
    if not conns:
        raise Unreadable(f"{mod.name} assigns nothing matching {write_select!r}")
    cone = F.Cone(mod)
    reached: set[str] = set()
    cite = None
    for c in conns:
        leaves, _prims, _lines = cone.walk(c.rhs)
        hit = {pattern for pattern in carried for leaf in leaves if F.ref_matches(leaf, pattern)}
        if hit:
            reached |= hit
            cite = cite or _cite(mod, c.line, c.locator)
    if not reached:
        raise Unreadable(f"no assignment to {write_select!r} in {mod.name} reads any of {carried}")
    entry = None
    for c in mod.connects:  # one hop: a connect that reads the request field directly
        if any(F.ref_matches(ref.text(), request) for ref in F._refs_in(c.rhs)):  # noqa: SLF001
            entry = _cite(mod, c.line, c.locator)
            break
    if entry is None:
        raise Unreadable(f"{mod.name} never reads the load request's {request!r}")
    return {
        "write_select": write_select,
        "carried_by": sorted(reached),
        "request": request,
        "reading": "the bank's accumulate bit is the load's address-metadata accumulate field",
        "cite": cite,
        "request_cite": entry,
        "between": "the request field enters the load path and the path's output drives the bank; the "
        "queues in between are crossed by their declared field names, not traced per submodule",
    }


def validated_roles(declared: Any, roles_by_name: Mapping[str, list[str]]) -> dict[str, Any]:
    """The declared roles, each in the closed vocabulary, with the instructions that carry it."""
    from merlin.compile.analysis.instruction_roles import ROLES

    if not isinstance(declared, list) or not declared or not all(isinstance(r, str) for r in declared):
        raise Unreadable("the probe declares no `roles` list for the commands that use this capability")
    unknown = sorted(set(declared) - set(ROLES))
    if unknown:
        raise Unreadable(f"roles {unknown} are not in the closed role vocabulary")
    carriers = {role: sorted(n for n, rs in roles_by_name.items() if role in rs) for role in declared}
    missing = sorted(role for role, names in carriers.items() if roles_by_name and not names)
    if missing:
        raise Unreadable(f"no instruction of this target carries role(s) {missing}")
    return {"roles": list(declared), "instructions": carriers if roles_by_name else "UNKNOWN"}


def derive(spec: Mapping[str, Any], modules: Mapping[str, F.ModuleBody], roles_by_name) -> dict[str, Any]:
    """The fact's value from already-loaded modules; raises :class:`Unreadable` with the reason."""
    store, load = spec.get("store"), spec.get("load")
    if not isinstance(store, Mapping) or not isinstance(load, Mapping):
        raise Unreadable("the probe needs a `store` and a `load` block")
    smod, lmod = modules.get(store.get("module")), modules.get(load.get("module"))
    if smod is None or lmod is None:
        raise Unreadable(f"the elaboration lacks module {store.get('module')!r} or {load.get('module')!r}")
    return {
        "accumulates_on_load": True,
        "selected_per_command": True,
        "store": accumulating_store(smod, store["write_data"], store["select"], store["sum"]),
        "load_path": load_selected_accumulate(lmod, load["write_select"], load["carried_by"], load["request"]),
        **validated_roles(spec.get("roles"), roles_by_name),
    }
