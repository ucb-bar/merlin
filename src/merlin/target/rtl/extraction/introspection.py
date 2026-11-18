"""merlin-rtl-introspect — structure-only fact extraction from an elaborated SoC's FIRRTL.

Reads the CIRCT/firtool-produced artifacts (the SoC ``.fir`` + ``top_module_hierarchy.json``)
and extracts STRUCTURE ONLY: memories (with sizes), the compute array, datapath widths,
and the command interface. No semantic hypotheses (no role/kind guesses) — those are the
agent's classify/spec slots. Each fact carries its source evidence.

Two extraction paths live here, and the difference between them is the whole point:

* the role-anchored probes. Their module/source markers and published role names come entirely
  from the selected target contract. The accumulator probe reads its data-port path from that
  contract and its element type from the exact FIRRTL field;
* the SCOPED STRUCTURAL CENSUS (:func:`census_facts`). Nothing here matches a construct by name.
  A target's modules are the ones the elaboration's own ``@[generators/<gen>/...]`` provenance
  annotation says were defined in that target's generator tree; its accelerator unit is the
  outermost such module in the instance hierarchy; its SRAMs are every ``smem``/``cmem`` declared
  inside that unit, grouped by the Scala source SITE that declared them, counted PER UNIT (a dual-
  core SoC instantiates the unit twice, and a compiler schedules into one of them); and its array is
  the widest same-module sibling group under that unit. Every number is read off the design.

The census RUNS ONLY WHERE THE ROLE-ANCHORED PROBES FOUND NOTHING of that kind. That precedence is
deliberate and not a convenience: the role names are what ``address_space``/``memory_regime``/
``perf.profile`` link stores to datapaths by, and what mlc's discovery de-duplicates against, so a
design those probes DO understand must keep them. A design they do not understand — every target
that is not the one they were written for — used to get an empty bundle, and now gets the census.

`validate_against_contract` checks the contract's declared `compute_units` cover the discovered
datapath dtypes (the mesh/scratchpad/dtype capacities are no longer hand-declared — they ARE these
facts, read directly by consumers, so there is nothing left to cross-check for them).

WHERE the elaboration lives is never a table in this module: it is read from the TARGET'S OWN
declaration (:func:`declared_rtl_source`) — an ``rtl_source:`` block in its contract, or an
``rtl.elaboration:`` block in its experiment descriptor. The role-anchored reader of a chipyard
simulator's own elaboration (:func:`sim_elaboration_facts`) takes its config from the harness config the
target's contract declares for its simulators (``runtime.rtl_sim_config``). A target that declares none is
reported as undeclared; nothing here guesses a config name.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.paths import ext_path
from merlin.target.rtl.extraction.declared_readers import accumulator_layout, firrtl_role_probe, observed_memory_lines

# ---- introspect --------------------------------------------------------------------------------


def _channel_payload(typ: str) -> dict[str, str] | None:
    """The ``bits`` payload fields of a decoupled (ready/valid) channel type, else None."""
    from merlin.target.rtl.extraction.census import _directed_fields, _inner_bundle

    leaves = {name: t for name, _flip, t in _directed_fields(_inner_bundle(typ))}
    if "ready" not in leaves or "valid" not in leaves or "bits" not in leaves:
        return None
    return {name: t for name, _flip, t in _directed_fields(_inner_bundle(leaves["bits"]))}


def bus_master_edges(port_text: str) -> list[dict[str, Any]]:
    """Every outward MEMORY-MASTER edge a module's ports declare, with its data width each way.

    Structural: a port field whose bundle holds a request channel ``a`` pointing OUT of the module whose
    payload carries an ``address`` and a ``data`` field, and a response channel ``d`` pointing IN whose
    payload carries ``data``. That pair is a master edge onto a memory bus; a manager edge has the same
    channels the other way round and is not counted. Write data leaves on ``a``; read data returns on
    ``d``. Each channel is a ready/valid handshake, so it retires at most one payload per cycle and its
    ``data`` width is that direction's per-cycle ceiling on this edge.
    """
    from merlin.target.rtl.extraction.census import _directed_fields, _inner_bundle, _uint_width

    edges: list[dict[str, Any]] = []
    for line in port_text.splitlines():
        _, _, rest = line.partition(":")
        for node, node_flipped, typ in _directed_fields(_inner_bundle(rest)):
            channels = {name: (flip != node_flipped, t) for name, flip, t in _directed_fields(_inner_bundle(typ))}
            request, response = channels.get("a"), channels.get("d")
            if request is None or response is None or request[0] or not response[0]:
                continue
            a_bits, d_bits = _channel_payload(request[1]), _channel_payload(response[1])
            if a_bits is None or d_bits is None or "address" not in a_bits or "data" not in a_bits:
                continue
            write, read = _uint_width(a_bits["data"]), _uint_width(d_bits.get("data", ""))
            if write is None or read is None:
                continue
            edges.append({"node": node, "write_data_bits": write, "read_data_bits": read})
    return edges


def _receives_host_command(port_text: str) -> str | None:
    """The name of a decoupled port field pointing INTO the module whose payload carries a RISC-V
    instruction bundle (the co-processor command handoff, received), else None."""
    from merlin.target.rtl.extraction.census import _INSTRUCTION_FIELDS, _bundle_fields, _directed_fields, _inner_bundle

    for line in port_text.splitlines():
        _, _, rest = line.partition(":")
        for name, flipped, typ in _directed_fields(_inner_bundle(rest)):
            if not flipped:
                continue
            payload = _channel_payload(typ)
            if payload is None:
                continue
            for sub in payload.values():
                if _INSTRUCTION_FIELDS <= {f for f, _ in _bundle_fields(_inner_bundle(sub))}:
                    return name
    return None


def memory_path(fir: str | Path) -> dict[str, Any]:
    """The accelerator's memory path from an elaborated FIRRTL: the per-cycle byte ceiling each way.

    Generator-free and name-free. The accelerator is a module that RECEIVES a host co-processor command
    (:func:`_receives_host_command`) AND masters at least one memory edge (:func:`bus_master_edges`);
    a command queue receives the command but masters no edge, and the host core masters edges but sends
    the command, so neither qualifies. Its edges are summed per direction (every master edge it has can
    move a payload in the same cycle). When several modules qualify -- a wrapper re-exposing the same
    ports -- the WIDEST is taken, so the result never understates what one accelerator can move: a cycle
    floor built from it stays a floor.

    Streamed one module at a time: the elaborations are 80-130 MB on a host other sessions share.
    Returns ``{"status": "derived", "read_bytes_per_cycle", "write_bytes_per_cycle", "modules"}`` or
    ``{"status": "unknown", "reason"}`` -- never a default width.
    """
    from merlin.target.rtl.extraction.census import _module_head

    found: list[dict[str, Any]] = []

    def close(module: str, ports: list[str]) -> None:
        if not module or not ports:
            return
        text = "".join(ports)
        command = _receives_host_command(text)
        if command is None:
            return
        edges = bus_master_edges(text)
        if not edges:
            return
        found.append(
            {
                "module": module,
                "command_port": command,
                "edges": edges,
                "read_data_bits": sum(e["read_data_bits"] for e in edges),
                "write_data_bits": sum(e["write_data_bits"] for e in edges),
            }
        )

    current, ports, in_ports = "", [], False
    with Path(fir).open(encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            stripped = raw.strip()
            head = _module_head(stripped)
            if head is not None:
                close(current, ports)
                current, ports, in_ports = head[0], [], True
                continue
            if not in_ports:
                continue
            if stripped.startswith(("input ", "output ")):
                ports.append(stripped + "\n")
            elif stripped:
                in_ports = False  # ports are declared first; the body begins
    close(current, ports)
    if not found:
        return {
            "status": "unknown",
            "reason": "no module both receives a host co-processor command and masters a memory edge, so "
            "the accelerator's memory path is not derivable from this elaboration",
        }
    best = max(found, key=lambda m: (m["read_data_bits"] + m["write_data_bits"], m["module"]))
    if best["read_data_bits"] % 8 or best["write_data_bits"] % 8:
        return {"status": "unknown", "reason": f"module {best['module']}'s data widths are not whole bytes"}
    return {
        "status": "derived",
        "read_bytes_per_cycle": best["read_data_bits"] // 8,
        "write_bytes_per_cycle": best["write_data_bits"] // 8,
        "module": best["module"],
        "modules": found,
        "evidence": f"module {best['module']} receives the host command on `{best['command_port']}` and masters "
        + ", ".join(
            f"`{e['node']}` (a.data {e['write_data_bits']}b out, d.data {e['read_data_bits']}b in)"
            for e in best["edges"]
        ),
    }


def _module_port_field_type(fir: Path, module: str, dotted_field: str) -> str | None:
    """Resolve one declared dotted FIRRTL port field inside an exact module, without guessing width."""
    from merlin.target.rtl.extraction.census import _bundle_fields, _inner_bundle

    parts = dotted_field.split(".")
    in_module = False
    with fir.open(encoding="utf-8", errors="ignore") as source:
        for raw in source:
            code = raw.split("@[", 1)[0].strip()
            head = code.partition(" :")[0].split()
            if len(head) >= 2 and head[-2] == "module":
                in_module = head[-1] == module
                continue
            if not in_module:
                continue
            for direction in ("input", "output"):
                prefix = f"{direction} {parts[0]} : "
                if not code.startswith(prefix):
                    continue
                typ = code[len(prefix) :]
                for field in parts[1:]:
                    typ = dict(_bundle_fields(_inner_bundle(typ))).get(field, "")
                    if not typ:
                        return None
                return typ
    return None


def _firrtl_integer_type(typ: str | None) -> tuple[int, str, str] | None:
    """Element bit width, dtype and source token from a FIRRTL integer port type."""
    if typ is None:
        return None
    for marker, dtype_prefix in (("SInt<", "i"), ("UInt<", "u")):
        if typ.startswith(marker):
            width, sep, _ = typ[len(marker) :].partition(">")
            if sep and width.isdigit() and int(width) > 0:
                return int(width), f"{dtype_prefix}{width}", f"{marker}{width}>"
    return None


def _count_children_under_parent(hierarchy: Path, *, parent_contains: str, child_prefix: str) -> int:
    tree = json.loads(hierarchy.read_text())
    n = 0

    def walk(node: dict, path: str):
        nonlocal n
        mod = node.get("module_name", "")
        if mod.startswith(child_prefix) and parent_contains in path:
            n += 1
        for child in node.get("instances", []):
            walk(child, path + "/" + mod)

    walk(tree, "")
    return n


def extract_facts(
    fir: str | Path,
    hierarchy: str | Path,
    *,
    target: str | None = None,
    role_target: str | None = None,
    generator: str | None = None,
    config: str | None = None,
) -> dict[str, Any]:
    """Return a structure-only facts dict extracted from the elaborated FIRRTL.

    Two readers, in a fixed precedence (see the module docstring). The ROLE-ANCHORED probes below
    match target-declared constructs and, where they fire, produce target-declared role names
    downstream consumers link stores to datapaths by. ``generator`` (or ``target``,
    from which it defaults) additionally enables the SCOPED STRUCTURAL CENSUS, which fills each fact
    KIND the probes left empty — so a design those probes do not recognize stops yielding an empty
    bundle, and a design they do recognize is untouched.

    ``role_target`` selects the target-owned role-probe layout without enabling the structural census.
    ``config``/``target`` are recorded as given so ports/provenance readers can find the exact
    elaboration these facts came from. A caller that supplies neither records neither: no config or
    target is assumed here.
    """
    from merlin.target.rtl.extraction.census import _uint_dims, census_facts

    fir, hierarchy = Path(fir), Path(hierarchy)
    generator = generator or target
    facts: dict[str, Any] = {
        "target": target,
        "source": {
            "kind": "firrtl",
            "config": config,
            "fir": fir.name,
            "hierarchy": hierarchy.name,
            "fir_path": str(fir),
            "fir_sha256": _sha256(fir),
            "hierarchy_path": str(hierarchy),
            "hierarchy_sha256": _sha256(hierarchy),
        },
        "arrays": [],
        "memories": [],
        "datapaths": [],
        "interfaces": [],
    }
    if generator:
        facts["source"]["generator"] = generator

    layout_target = role_target or target
    roles = firrtl_role_probe(layout_target) if layout_target else None
    accumulator = accumulator_layout(layout_target) if layout_target else None
    accumulator_module = accumulator["module"] if accumulator and accumulator.get("firrtl_data_field") else None

    # ---------------------------------------------------------------- role-anchored probes
    array = roles.get("array") if roles else None
    if array:
        tiles = _count_children_under_parent(
            hierarchy,
            parent_contains=array["parent_module_contains"],
            child_prefix=array["child_module_prefix"],
        )
        if tiles:
            side = int(math.isqrt(tiles))
            facts["arrays"].append(
                {
                    "name": array["name"],
                    "tiles": tiles,
                    "rows": side,
                    "cols": side,
                    "square": side * side == tiles,
                    "evidence": (
                        f"{hierarchy.name}: {array['child_module_prefix']} instances "
                        f"under {array['parent_module_contains']}"
                    ),
                }
            )

    memory = roles.get("memory") if roles else None
    interfaces = roles.get("interfaces", []) if roles else []
    needles = (
        (memory["declaration_contains"], memory["source_contains"]) if memory else (),
        (f"module {accumulator_module} :",) if accumulator_module else (),
        *((f"module {item['module']}",) for item in interfaces),
    )
    probe = _probe_lines(fir, *needles)
    sp = [ln for ln in probe[0] if _uint_dims(ln) is not None] if memory else []
    if sp:
        ebits, row_elems, depth = _uint_dims(sp[0])
        banks = len(sp)
        facts["memories"].append(
            {
                "name": memory["name"],
                "banks": banks,
                "row_elems": row_elems,
                "depth": depth,
                "elem_bits": ebits,
                "bytes": banks * row_elems * (ebits // 8) * depth,
                "evidence": (
                    f"{banks}x `{memory['declaration_contains']}{ebits}>"
                    f"[{row_elems}] [{depth}]` @ {memory['source_contains']}"
                ),
            }
        )

    # Accumulator memory: its role/module/field come from the target, its element type from FIRRTL.
    acc_type = (
        _firrtl_integer_type(_module_port_field_type(fir, accumulator_module, accumulator["firrtl_data_field"]))
        if probe[1] and accumulator is not None and accumulator_module is not None
        else None
    )
    if probe[1]:
        evidence = (
            f"module {accumulator_module}; acc datapath {acc_type[2]}"
            if acc_type is not None
            else f"module {accumulator_module}; declared field {accumulator['firrtl_data_field']} type unresolved"
        )
        facts["memories"].append(
            {
                "name": accumulator["memory_name"],
                "elem_bits": acc_type[0] if acc_type is not None else None,
                "bytes": None,
                "evidence": evidence,
                "note": "depth not extracted from this artifact (v1)",
            }
        )

    # Datapath element widths, for the stores the probes above actually found. Emitted ONLY when
    # those stores exist: the pair used to be appended unconditionally, so a design with neither
    # memory still had an 8-bit input and a 32-bit accumulator asserted about it, sourced from
    # nothing. A dtype nobody measured is worse than a dtype nobody has.
    if sp:
        facts["datapaths"].append(
            {
                "name": memory["datapath_name"],
                "dtype": f"i{ebits}",
                "evidence": f"{memory['name']} smem UInt<{ebits}>",
            }
        )
    if acc_type is not None:
        facts["datapaths"].append(
            {
                "name": accumulator["memory_name"],
                "dtype": acc_type[1],
                "evidence": f"{accumulator_module} {acc_type[2]}",
            }
        )

    # Interface role is target-declared; module presence remains an observed FIRRTL fact.
    for index, item in enumerate(interfaces, start=2):
        if probe[index]:
            facts["interfaces"].append({"name": item["name"], "evidence": item["evidence"]})

    # ---------------------------------------------------------------- scoped structural census
    if not generator:
        facts["census"] = {
            "status": "not_run",
            "why": "no generator scope was given, so the elaboration's own "
            "provenance annotations cannot be used to tell this target's "
            "modules from the rest of the SoC",
        }
        return facts
    census = census_facts(fir, hierarchy, generator=generator)
    facts["census"] = census.pop("census", {})
    for kind in ("arrays", "memories", "datapaths", "interfaces"):
        found = census.pop(kind, [])
        if found and not facts[kind]:
            facts[kind] = found
        elif found:
            facts["census"].setdefault("superseded", []).append(
                f"{kind}: the role-anchored probes produced "
                f"{[f.get('name') for f in facts[kind]]}, so the census's "
                f"{[f.get('name') for f in found]} was not used (the probe names are the ones "
                f"downstream links stores to datapaths by)"
            )
    facts.update(census)  # replication_groups / *_undeterminable / notes — evidence, unconsumed
    return facts


def _probe_lines(fir: Path, *needle_sets: tuple[str, ...]) -> list[list[str]]:
    """One streaming pass answering several substring probes at once.

    The per-probe helper reads the whole file each time it is asked; four probes over a 130 MB
    elaboration is half a gigabyte of redundant I/O on a shared host, and the answers are all
    available from a single walk.
    """
    out: list[list[str]] = [[] for _ in needle_sets]
    try:
        handle = Path(fir).open(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    with handle as fh:
        for line in fh:
            for i, needles in enumerate(needle_sets):
                if needles and all(nd in line for nd in needles):
                    out[i].append(line.rstrip("\n"))
    return out


def validate_against_contract(facts: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Return problems where the extracted RTL datapaths are not covered by the contract's declared
    compute_units. Mesh geometry, scratchpad capacity and datapath dtypes are no longer hand-declared
    in the contract — they ARE these facts — so there is nothing to cross-check for them; the only
    surviving check is that the contract's compute_units cover the discovered datapath dtype(s)."""
    records = facts.get("datapaths") or []
    if not records:
        return ["UNKNOWN: no RTL datapath evidence is available"]
    if not contract.get("compute_units"):
        return ["UNKNOWN: contract declares no compute_units to cross-check"]
    dt = {d["name"]: d.get("dtype") for d in records if isinstance(d, dict) and d.get("name")}
    unknown = [f"UNKNOWN: datapath {name} has no resolved dtype" for name, dtype in dt.items() if not dtype]
    if not dt:
        unknown.append("UNKNOWN: no named RTL datapath evidence is available")
    # If the contract declares compute_units, the RTL datapaths must be covered by them: the input
    # datapath dtype by some unit's declared formats, and the accumulator by some accumulate rule
    # (when a unit still declares one).
    return unknown + _check_compute_units(dt, contract)


def _check_compute_units(datapaths: dict[str, str], contract: dict[str, Any]) -> list[str]:
    from merlin.target.capabilities import compute_units as _cu

    units = _cu.compute_units(contract)
    if not units:
        return []
    eff = [_cu.effective(u, units) for u in units]
    tokens: set[str] = set()
    accs: set[str] = set()
    for u in eff:
        tokens |= _cu.datatype_tokens(u)
        accs |= {a.acc for a in u.accumulate}
    out: list[str] = []
    inp = datapaths.get("input")
    if inp and inp not in tokens:
        out.append(f"datapath input {inp} not covered by any compute_unit dtypes {sorted(tokens)}")
    acc = datapaths.get("accumulator")
    if acc and accs and acc not in accs:
        out.append(f"datapath accumulator {acc} not in any compute_unit accumulate acc {sorted(accs)}")
    return out


def emit_facts_yaml(facts: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(facts, sort_keys=False)


GENERATOR_VERSION = "rtl-introspect-v1-declared-accumulator"  # CIRCT pass remains v2


def _src_sha(path: str) -> str:
    proc = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


def dump_facts(
    out_path: str | Path,
    *,
    target: str | None = None,
    chipyard_root: str | Path | None = None,
    config: str | None = None,
) -> dict[str, Any]:
    """Extract facts and write a REPRODUCIBLE rtl_facts.yaml (facts + generator version + source
    SHAs + extraction method). This makes RTL-fact extraction a recorded, attributable input —
    the thing an agent_spec target-generation experiment consumes.

    Addressed by ``target`` (its simulator elaboration, :func:`sim_elaboration_facts`) or by an explicit
    ``config``. One of the two is required; neither is assumed."""
    import yaml

    from merlin.target.rtl.extraction.census import find_artifacts, sim_elaboration_facts

    if target is not None and config is None:
        arts, facts = sim_elaboration_facts(target, chipyard_root)
    else:
        arts = find_artifacts(chipyard_root, config)
        facts = extract_facts(arts["fir"], arts["hierarchy"], role_target=target, config=config)
        facts["target"] = target
    cy = str(chipyard_root)
    shas = {"chipyard": _src_sha(cy)}
    if target is not None:
        # the target's generator tree, by the chipyard `generators/<t>` convention isa_scala_path uses
        shas[target] = _src_sha(f"{cy}/generators/{target}")
    record = {
        "schema_version": "1.0",
        "generator": {
            "name": "merlin.target.rtl.extraction.census",
            "version": GENERATOR_VERSION,
            "method": "grep/regex over firtool-produced FIRRTL + hierarchy JSON (NOT yet a CIRCT hw/seq MLIR pass)",
        },
        "source_shas": shas,
        "facts": facts,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return record


CENSUS_VERSION = "rtl-introspect-firrtl-census-v1"


def _sha256_16(path: Path) -> str:
    """A digest of the BYTES actually read, or ``missing`` — never a silently absent provenance field."""
    digest = _sha256(path)
    return digest if digest == "missing" else digest[:16]


def _sha256(path: Path) -> str:
    """Full identity of a consumed source, not a human-facing digest prefix."""
    import hashlib

    if not Path(path).is_file():
        return "missing"
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkout_sha(path: str | Path) -> str:
    """The revision of an external checkout, or WHY there is none.

    ``unknown`` used to cover two different situations and the difference matters: a directory that is
    not under version control at all can never yield a revision (recording it as unknown invites
    someone to go looking for the git failure), whereas a checkout whose ``rev-parse`` failed is a
    fixable environment problem. Kept apart, the way every other three-state field here is.
    """
    p = Path(path)
    if not p.is_dir():
        return "absent"
    if not (p / ".git").exists():
        return "not_a_git_checkout"
    return _src_sha(str(p))


def build_facts_record(target: str) -> dict[str, Any]:
    """The schema-2.0 facts bundle for ``target``, extracted from the elaboration it declares.

    Same shape ``merlin.target.rtl.facts.load_facts`` reads: ``{schema_version, generator, inputs,
    facts}``. ``inputs`` names both artifacts AND their digests, so a bundle can never claim an
    elaboration it did not read.
    """
    from merlin.target.rtl.extraction.census import declared_rtl_source

    src = declared_rtl_source(target)
    arts = src.artifacts()
    fir, hier = Path(arts["fir"]), Path(arts["hierarchy"])
    for path, what in ((fir, "elaborated FIRRTL"), (hier, "module hierarchy")):
        if not path.is_file():
            raise FileNotFoundError(
                f"{target}: the {what} its own declaration ({src.origin}) points at is not on disk: "
                f"{path}. The DECLARATION is fine; the elaboration has not been built (or lives "
                f"elsewhere) — that is a build to run, not a fact to guess."
            )
    facts = extract_facts(fir, hier, target=target, generator=src.generator, config=src.config)
    return {
        "schema_version": "2.0",
        "generator": {
            "name": "merlin.target.rtl.extraction.census",
            "version": CENSUS_VERSION,
            "method": "structural census over the firtool-produced FIRRTL + module hierarchy "
            "(modules scoped by their own @[generators/<gen>/...] provenance; SRAMs grouped "
            "by Chisel declaration site and counted per accelerator unit; array = the widest "
            "same-module sibling group)",
        },
        "inputs": {
            "target": target,
            "config": src.config,
            "generator": src.generator,
            "declared_by": str(src.origin),
            "rtl_root": str(src.root),
            "fir": fir.name,
            "fir_path": str(fir.resolve()),
            "fir_sha256": _sha256(fir),
            "fir_sha": _sha256_16(fir),
            "hierarchy": hier.name,
            "hierarchy_path": str(hier.resolve()),
            "hierarchy_sha256": _sha256(hier),
            "hierarchy_sha": _sha256_16(hier),
            "rtl_sha": _checkout_sha(src.root),
            "generator_sha": _checkout_sha(Path(src.root) / "generators" / src.generator),
            "extractor_sha": _sha256_16(Path(__file__)),
            "extractor_sha256": _sha256(Path(__file__)),
        },
        "source_consistency": {
            "status": "unverified",
            "reason": "selected FIRRTL and hierarchy have no shared production receipt or reviewed equivalence",
            "config": src.config,
        },
        "facts": facts,
    }


CENSUS_KINDS = ("arrays", "memories", "datapaths", "interfaces")


def augment_facts_file(path: str | Path, target: str) -> dict[str, Any] | None:
    """Fill fact kinds an existing bundle at ``path`` left EMPTY, from the target's own elaboration.

    Additive only, and only into a hole: a kind the primary extractor populated is never touched, so
    this can never quietly replace a richer derivation with a poorer one. Returns the merged record,
    or ``None`` when the target declares no elaboration / the elaboration is not on disk (both of
    which are reported by the caller, never swallowed into a silently unchanged file).
    """
    from merlin.target.rtl.extraction.census import RtlSourceUndeclared
    from merlin.target.rtl.facts import write_facts_guarded

    p = Path(path)
    try:
        census = build_facts_record(target)
    except (RtlSourceUndeclared, FileNotFoundError):
        return None
    if not p.is_file():
        write_facts_guarded(p, census)
        return census
    try:
        existing = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        write_facts_guarded(p, census)
        return census
    body = existing.get("facts")
    if not isinstance(body, dict):
        body = {}
    filled: list[str] = []
    for kind in CENSUS_KINDS:
        if body.get(kind):
            continue
        supplied = census["facts"].get(kind)
        if supplied:
            body[kind] = supplied
            filled.append(kind)
    for key in ("source", "census", "replication_groups", "memories_undeterminable"):
        if key in census["facts"] and key not in body:
            body[key] = census["facts"][key]
    if not filled and existing.get("facts"):
        return existing
    body.setdefault("target", target)
    existing["facts"] = body
    existing.setdefault("inputs", {}).update(
        {
            f"firrtl_{k}": v
            for k, v in census["inputs"].items()
            if k
            in (
                "config",
                "generator",
                "fir",
                "fir_sha",
                "hierarchy",
                "hierarchy_sha",
                "declared_by",
                "rtl_sha",
                "generator_sha",
            )
        }
    )
    existing["generator"] = {
        **(existing.get("generator") or {}),
        "firrtl_census": f"{CENSUS_VERSION} filled {filled or 'nothing'}",
    }
    write_facts_guarded(p, existing)
    return existing


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract structure-only RTL facts.")
    ap.add_argument("--out", default="rtl_facts.yaml")
    ap.add_argument("--chipyard", default=None, help="external sim checkout (default: resolved from .env when needed)")
    ap.add_argument(
        "--config",
        default=None,
        help="without --target: the elaborated chipyard config to read (required; no config is assumed)",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="extract from the elaboration THIS TARGET declares (rtl_source / "
        "rtl.elaboration) and write the schema-2.0 JSON bundle",
    )
    ap.add_argument(
        "--json-out", default=None, help="with --target: where to write the bundle (default: the target's rtl cache)"
    )
    args = ap.parse_args()
    if args.target:
        from merlin.target.rtl.facts import rtl_facts_path

        rec = build_facts_record(args.target)
        out = Path(args.json_out) if args.json_out else rtl_facts_path(args.target)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        body = rec["facts"]
        print(
            f"wrote {out}: {len(body.get('arrays', []))} arrays, "
            f"{len(body.get('memories', []))} memories, "
            f"{len(body.get('datapaths', []))} datapaths, "
            f"{len(body.get('interfaces', []))} interfaces "
            f"[{rec['generator']['version']}]"
        )
        return 0
    if not args.config:
        ap.error("pass --target, or --config naming the elaborated chipyard config to read")
    rec = dump_facts(args.out, chipyard_root=args.chipyard, config=args.config)
    print(
        f"wrote {args.out}: {len(rec['facts'].get('arrays', []))} arrays, "
        f"{len(rec['facts'].get('memories', []))} memories, "
        f"{len(rec['facts'].get('interfaces', []))} interfaces "
        f"[generator {rec['generator']['version']}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
