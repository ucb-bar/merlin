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

_MODULE_KEYWORDS = ("module", "public module", "extmodule", "intmodule")


_DEFINED_MODULE_KEYWORDS = ("module", "public module")  # keywords that carry a body we can read


_MEM_KEYWORDS = ("smem", "cmem", "mem")  # source and explicit lowered memory declarations


_UINT_MARKERS = ("UInt<", "SInt<")


_INSTRUCTION_FIELDS = frozenset({"funct", "opcode"})


def default_chipyard() -> str:
    """The external sim checkout, resolved WHEN NEEDED rather than at import.

    Resolving it at module scope made importing this module require the external checkout: inside the
    agent sandbox `.env` is not reachable, so `ext_path` raised KeyError during import and took down
    every consumer with it -- including `gen_isa_module`, which an arm-4 agent is granted and which only
    needs the COMMITTED rtl_facts for most work. A missing external path should fail the call that
    actually needs the checkout, not the import of anything that mentions it."""
    return f"{ext_path('chipyard')}"


def find_artifacts(chipyard_root: str | Path | None, config: str) -> dict[str, Path]:
    """``{fir, hierarchy}`` of the chipyard TestHarness elaboration of ``config`` under ``chipyard_root``
    (the external checkout when None). ``config`` is required: which elaboration is meant is a fact about
    the target (:func:`sim_config` / :func:`declared_rtl_source`), never a default here."""
    if not config:
        raise RtlSourceUndeclared(
            "find_artifacts needs the elaborated config; none was given, and "
            "picking one would be a guess about the hardware"
        )
    chipyard_root = default_chipyard() if chipyard_root is None else chipyard_root
    base = Path(chipyard_root) / "sims/verilator/generated-src" / f"chipyard.harness.TestHarness.{config}"
    return {"fir": base / f"chipyard.harness.TestHarness.{config}.fir", "hierarchy": base / "top_module_hierarchy.json"}


@dataclass(frozen=True)
class RtlSource:
    """WHERE one target's elaborated RTL is, as that target declares it.

    Every field comes from a file the TARGET owns. Nothing in this module knows which config any
    target elaborates, which external checkout holds it, or which generator directory defines its
    modules: a second target is onboarded by adding this block to its own contract, never by editing
    a table here.
    """

    target: str
    root: Path  # the external checkout root (from `.env` MERLIN_EXT_<NAME>)
    config: str  # the elaborated config, e.g. the chipyard TestHarness config name
    generator: str  # the generator source dir whose modules ARE this target's
    origin: Path  # the declaration file this came from — provenance, so a wrong pin is traceable
    source_revision: str | None = None
    gitlinks: tuple[tuple[str, str], ...] = ()
    artifact_sha256: tuple[tuple[str, str], ...] = ()

    def artifacts(self) -> dict[str, Path]:
        return find_artifacts(self.root, self.config)


class RtlSourceUndeclared(RuntimeError):
    """This target declares no elaborated-RTL source, so nothing can be extracted from one.

    Distinct from "the declared elaboration is not on disk": one is a gap in the target's own
    definition that a human closes by declaring it, the other is a build that has not been run. They
    demand different actions, so they never share a message.
    """


class RtlSourceInvalid(RtlSourceUndeclared):
    """A selected declaration exists but is unusable; an ambient cache cannot replace it."""


def _yaml_doc(path: Path) -> dict[str, Any]:
    """A yaml mapping at ``path``, or ``{}``. Never raises — an unreadable declaration is "none"."""
    try:
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent/broken declaration is simply no declaration
        return {}
    return doc if isinstance(doc, dict) else {}


def _declaration_files(target: str) -> list[Path]:
    """Every file in which ``target`` may declare its own RTL source, most authoritative first.

    All three are the target's OWN files, resolved from the target name by the repo's existing
    conventions — the contract path resolver, the target registry, and the experiment-descriptor
    layout. No path here names a target.
    """
    from merlin.target.rtl.facts import target_contract_path

    out: list[Path] = []

    def add(p: Path | None) -> None:
        if p is not None and Path(p).is_file() and Path(p) not in out:
            out.append(Path(p))

    add(target_contract_path(target))
    try:  # the registry resolves OOT packages the path helper does not
        from merlin.target.registry.identity import resolve

        add(resolve(target).contract_path)
    except Exception:  # noqa: BLE001 — an unresolvable target has no registry contract
        pass
    try:  # the ONE module allowed to know the experiments/ layout
        from merlin.target.registry.descriptor_loading import descriptor_path

        add(descriptor_path(target))
    except Exception:  # noqa: BLE001 — no descriptor is simply no declaration there
        pass
    return out


def _rtl_source_block(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The declaration block out of one of the target's own files, whichever spelling it uses.

    A capability contract declares ``rtl_source:`` at the top level; an experiment descriptor already
    owns an ``rtl:`` block (``via``, ``operand_store``) and declares the elaboration inside it as
    ``rtl.elaboration:``. Both are read so a target may declare it wherever it already keeps its RTL
    facts, rather than the reader dictating a new file.
    """
    for block in (doc.get("rtl_source"), (doc.get("rtl") or {}).get("elaboration")):
        if isinstance(block, dict) and block:
            return block
    return None


def declared_rtl_source(target: str) -> RtlSource:
    """``target``'s elaborated-RTL source, read from the target's own declaration.

    Raises :class:`RtlSourceUndeclared` when no file of the target's declares one — a GAP to close in
    that target's contract, reported as such rather than papered over with the first config this repo
    happens to ship.
    """
    seen: list[str] = []
    for path in _declaration_files(target):
        seen.append(str(path))
        block = _rtl_source_block(_yaml_doc(path))
        if not block:
            continue
        ext = block.get("ext_root")
        root = block.get("root")
        if ext:
            try:
                root = ext_path(str(ext))
            except KeyError as exc:
                raise RtlSourceInvalid(
                    f"{target}: selected RTL declaration {path} cannot resolve {ext!r}: {exc}"
                ) from exc
        if not root:
            raise RtlSourceInvalid(
                f"{target}: {path} declares an RTL source with neither `ext_root` (a .env "
                f"MERLIN_EXT_<NAME> key) nor `root`, so the checkout it means is not resolvable"
            )
        config = block.get("config")
        if not config:
            raise RtlSourceInvalid(
                f"{target}: {path} declares an RTL source with no `config`, so WHICH elaboration it "
                f"means is not resolvable — an SoC checkout holds many, and picking one would be a "
                f"guess about the hardware"
            )
        from merlin.target.rtl.extraction.source_custody import (
            declared_artifact_pins,
            declared_revision,
            verify_artifact_digests,
        )

        revision, gitlinks = declared_revision(block, Path(str(root)), where=f"{target}: {path}")
        source = RtlSource(
            target=target,
            root=Path(str(root)),
            config=str(config),
            generator=str(block.get("generator") or target),
            origin=path,
            source_revision=revision,
            gitlinks=gitlinks,
            artifact_sha256=declared_artifact_pins(block, where=f"{target}: {path}"),
        )
        verify_artifact_digests(source)
        return source
    raise RtlSourceUndeclared(
        f"{target}: no file of this target's declares an elaborated-RTL source. Add an `rtl_source:` "
        f"block (ext_root / config / generator) to its contract, or an `rtl.elaboration:` block to its "
        f"experiment descriptor. Files consulted: {seen or 'none found'}"
    )


def sim_config(target: str) -> str:
    """The chipyard harness config ``target``'s simulators are built from, from the target's own contract.

    Read through :func:`merlin.target.simulators.spike_runtime.rtl_sim_config` (capability manifest
    ``runtime.rtl_sim_config``), the one reader of that key: it names the elaborated design every chipyard
    simulator of the target is built from, so it is the elaboration those simulators' ``generated-src``
    holds. Raises :class:`RtlSourceUndeclared` when the target declares none -- a gap in its contract,
    never a config this module picks."""
    from merlin.target.simulators.spike_runtime import rtl_sim_config

    cfg = rtl_sim_config(target)
    if not cfg:
        raise RtlSourceUndeclared(
            f"{target}: its contract declares no `runtime.rtl_sim_config`, so which chipyard elaboration "
            f"its simulators are built from is not resolvable. Declare it in the target's contract."
        )
    return str(cfg)


def sim_elaboration_facts(
    target: str, chipyard_root: str | Path | None = None
) -> tuple[dict[str, Path], dict[str, Any]]:
    """``(artifacts, facts)`` from the role-anchored probes over ``target``'s simulator elaboration.

    The config is :func:`sim_config` (the target's own declaration); the checkout is ``chipyard_root``,
    else the external one. The census is NOT scoped to the target here -- this is the reader that predates
    it, and its callers publish exactly what the probes found -- so ``target`` is recorded, not used as a
    generator scope."""
    from merlin.target.rtl.extraction.introspection import extract_facts

    cfg = sim_config(target)
    arts = find_artifacts(chipyard_root, cfg)
    facts = extract_facts(arts["fir"], arts["hierarchy"], role_target=target, config=cfg)
    facts["target"] = target
    return arts, facts


def artifacts_for(target: str) -> dict[str, Any]:
    """``{fir, hierarchy, source}`` for ``target``, from its own declaration. Never guesses a config."""
    src = declared_rtl_source(target)
    arts = src.artifacts()
    arts["source"] = src
    return arts


def _fir_lines(fir: Path, *needles: str) -> list[str]:
    """FIRRTL lines containing ALL of ``needles`` (regex-free substring scan).

    Now a one-probe call into :func:`_probe_lines`, which streams. The previous body read the whole
    elaboration into a string AND built a list of every line in it; at 130 MB, per probe, on a shared
    host, that was the single most expensive thing this module did.
    """
    from merlin.target.rtl.extraction.introspection import _probe_lines

    return _probe_lines(Path(fir), needles)[0]


def _uint_dims(line: str) -> tuple[int, int, int] | None:
    """Parse ``UInt<W>[E] [D]`` widths out of a FIRRTL smem line structurally (marker + delimiters),
    returning ``(W, E, D)`` or None."""
    idx = line.find("UInt<")
    if idx == -1:
        return None
    width, sep, after = line[idx + len("UInt<") :].partition(">")
    if not sep or not width.strip().isdigit():
        return None
    dims: list[str] = []
    pos = 0
    while len(dims) < 2:
        lb = after.find("[", pos)
        rb = after.find("]", lb) if lb != -1 else -1
        if lb == -1 or rb == -1:
            break
        dims.append(after[lb + 1 : rb].strip())
        pos = rb + 1
    if len(dims) < 2 or not (dims[0].isdigit() and dims[1].isdigit()):
        return None
    return int(width.strip()), int(dims[0]), int(dims[1])


def _module_head(stripped: str) -> tuple[str, str] | None:
    """``("Foo", "@[gen/... 1:2]")`` for a FIRRTL module declaration line, else None.

    FIRRTL writes ``  module Foo : @[<source site>]`` — the site being firtool's record of the Chisel
    that emitted the module. Split structurally on the declaration colon; the tail is the provenance.
    """
    for kw in _MODULE_KEYWORDS:
        if stripped.startswith(kw + " "):
            head, sep, ann = stripped.partition(":")
            if not sep:
                return None
            name = head.split()[-1]
            return (name, ann.strip()) if kw in _DEFINED_MODULE_KEYWORDS else ("", ann.strip())
    return None


def _mem_decl(stripped: str) -> tuple[str, str, str, str] | None:
    """``(keyword, name, type text, source site)`` for a FIRRTL memory declaration, else None.

    Line form: ``smem banks_0 : UInt<8>[32] [8192] @[<file> <line>:<col>]``. Parsed by the language's
    own delimiters, never by a pattern over the identifier — a design that names its SRAM anything at
    all is read the same way.
    """
    for kw in _MEM_KEYWORDS:
        if not stripped.startswith(kw + " "):
            continue
        _, rest = stripped.split(None, 1)
        name, sep, typ = rest.partition(":")
        if not sep:
            return None
        typ = typ.strip()
        site = ""
        marker = typ.find("@[")
        if marker != -1:
            site = typ[marker + 2 :].rstrip("]").strip()
            typ = typ[:marker].strip()
        return kw, name.strip(), typ, site
    return None


def _mem_shape(typ: str) -> tuple[int, int, int] | None:
    """``(element_bits, row_elements, depth)`` for a FIRRTL memory type, else None.

    ``UInt<8>[32] [8192]`` is a depth-8192 memory whose row is 32 elements of 8 bits; ``UInt<256> [64]``
    is a depth-64 memory whose row is one flat 256-bit word (row_elements 1 — the RTL declares no
    element decomposition, and inventing one is how a store's bank count comes out wrong). A bundle
    type (a queue's payload) returns None: it has no element width to report.
    """
    for marker in _UINT_MARKERS:
        if not typ.startswith(marker):
            continue
        width, sep, after = typ[len(marker) :].partition(">")
        if not sep or not width.strip().isdigit():
            return None
        dims: list[int] = []
        pos = 0
        while True:
            lb = after.find("[", pos)
            rb = after.find("]", lb) if lb != -1 else -1
            if lb == -1 or rb == -1:
                break
            tok = after[lb + 1 : rb].strip()
            if not tok.isdigit():
                break
            dims.append(int(tok))
            pos = rb + 1
        if len(dims) == 1:
            return int(width), 1, dims[0]
        if len(dims) == 2:
            return int(width), dims[0], dims[1]
        return None
    return None


def scan_fir(fir: str | Path, generator: str) -> tuple[set[str], dict[str, list[str]], dict[str, str]]:
    """One streaming pass over the elaborated FIRRTL.

    Returns ``(modules defined in <generator>'s source tree, {module: [mem decl lines]},
    {module: port declaration lines})`` — the last only for the modules in scope.

    Streamed a line at a time on purpose: these elaborations are 130 MB, and the alternative
    (``read_text().splitlines()``) holds the file plus a list of every line at once on a machine other
    sessions share.
    """
    needle = f"generators/{generator}/"
    scoped: set[str] = set()
    mems: dict[str, list[str]] = {}
    ports: dict[str, str] = {}
    current, in_scope = "", False
    with Path(fir).open(encoding="utf-8", errors="ignore") as fh:
        for raw in observed_memory_lines(fh):
            stripped = raw.strip()
            head = _module_head(stripped)
            if head is not None:
                current, ann = head
                in_scope = bool(current) and needle in ann
                if in_scope:
                    scoped.add(current)
                continue
            if not in_scope:
                continue
            if _mem_decl(stripped) is not None:
                mems.setdefault(current, []).append(stripped)
            elif stripped.startswith(("input ", "output ")):
                ports[current] = ports.get(current, "") + stripped + "\n"
    return scoped, mems, ports


def _hierarchy_units(tree: dict, scoped: set[str]) -> list[tuple[str, dict[str, int], dict[tuple[str, str], int]]]:
    """Per ACCELERATOR UNIT: ``(root module, {module: instances}, {(parent, child): widest group})``.

    A unit is an instance of a scoped module none of whose ancestors is scoped — the outermost thing
    the target's own generator emitted. Counting per unit rather than per design is what makes a
    dual-core SoC report the capacity one core actually has instead of twice it.
    """
    units: list[tuple[str, dict[str, int], dict[tuple[str, str], int]]] = []

    def descend(node: dict, counts: dict[str, int], widest: dict[tuple[str, str], int]) -> None:
        kids = node.get("instances") or []
        seen: dict[str, int] = {}
        for kid in kids:
            seen[kid["module_name"]] = seen.get(kid["module_name"], 0) + 1
        for child, n in seen.items():
            if n > 1:
                key = (node["module_name"], child)
                widest[key] = max(widest.get(key, 0), n)
        for kid in kids:
            counts[kid["module_name"]] = counts.get(kid["module_name"], 0) + 1
            descend(kid, counts, widest)

    def walk(node: dict, inside: bool) -> None:
        mod = node["module_name"]
        here = mod in scoped
        if here and not inside:
            counts: dict[str, int] = {mod: 1}
            widest: dict[tuple[str, str], int] = {}
            descend(node, counts, widest)
            units.append((mod, counts, widest))
        for kid in node.get("instances") or []:
            walk(kid, inside or here)

    walk(tree, False)
    return units


def _dominant_unit(units: list) -> tuple[str, int, dict[str, int], dict[tuple[str, str], int]] | None:
    """``(root module, instance count, counts, widest)`` for the unit whose subtree is largest.

    "Largest subtree" and not "first found": a target's generator also emits small leaf modules that a
    HOST instantiates directly (a decoder inside the scalar core, a rounding cell inside the FPU), and
    those are scoped roots too. They are real, and they are not the accelerator; the unit that carries
    the design is the one that contains the most of it. When several instances of the SAME root exist
    (a dual-core SoC), one instance's counts are used and the instance count is reported beside them.
    """
    if not units:
        return None
    by_root: dict[str, list] = {}
    for root, counts, widest in units:
        by_root.setdefault(root, []).append((counts, widest))
    root = max(by_root, key=lambda r: sum(by_root[r][0][0].values()))
    counts, widest = by_root[root][0]
    return root, len(by_root[root]), counts, widest


def _store_name(site: str, decl: str) -> str:
    """A store's name, derived from the Chisel source file that declared it plus the declaration's own
    base identifier: ``.../VMEM.scala 62:16`` + ``banks_3`` -> ``vmem.banks``.

    Derived and not chosen, so two targets that both declare a register file get names that differ
    exactly where their RTL differs. The ``_N`` suffix a bank sibling carries is dropped so the group
    is named once; a declaration whose own name ends in a number keeps it only if the base would
    otherwise be empty.
    """
    stem = Path(site.split()[0]).stem.casefold() if site else "unknown"
    parts = decl.rsplit("_", 1)
    base = parts[0] if len(parts) == 2 and parts[1].isdigit() and parts[0] else decl
    return f"{stem}.{base}"


def _element_token(bits: int) -> str:
    """The machine spelling of an unsigned RTL storage element of ``bits`` width.

    ``UInt`` is what the FIRRTL says and ``u<N>`` is what it means: an N-bit unsigned storage word.
    Signedness of the VALUE the datapath puts there is NOT declared by an ``smem`` (a signed operand
    is stored in a UInt cell), so it is not claimed here — the width is the fact.
    """
    return f"u{bits}"


def census_facts(fir: str | Path, hierarchy: str | Path, *, generator: str) -> dict[str, Any]:
    """The scoped structural census: arrays, memories, datapaths, interfaces — all read off the design.

    ``generator`` names the source tree whose modules are this target's, which is what the elaboration's
    own provenance annotations record. Everything else follows from the design: the unit is the
    outermost scoped module, the capacities are its SRAMs, the array is its widest sibling group.
    """
    scoped, mems, ports = scan_fir(fir, generator)
    tree = json.loads(Path(hierarchy).read_text(encoding="utf-8"))
    dominant = _dominant_unit(_hierarchy_units(tree, scoped))
    out: dict[str, Any] = {"arrays": [], "memories": [], "datapaths": [], "interfaces": [], "replication_groups": []}
    if dominant is None:
        out["census"] = {
            "generator": generator,
            "scoped_modules": len(scoped),
            "unit_root": None,
            "units": 0,
            "note": f"no module in the instance hierarchy is defined under "
            f"generators/{generator}/, so this elaboration contains no unit of "
            f"this target: nothing is derivable from it (UNKNOWN, not absent)",
        }
        return out
    root, n_units, counts, widest = dominant
    out["census"] = {
        "generator": generator,
        "scoped_modules": len(scoped),
        "unit_root": root,
        "units": n_units,
        "unit_module_instances": sum(counts.values()),
    }

    # ---- memories: every SRAM inside ONE unit, grouped by the Chisel site that declared it.
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for module, lines in sorted(mems.items()):
        instances = counts.get(module, 0)
        if not instances:
            continue  # declared in the target's tree but not inside this unit
        for line in lines:
            parsed = _mem_decl(line)
            if parsed is None:
                continue
            kw, decl, typ, site = parsed
            key = (site, _store_name(site, decl), typ)
            rec = groups.setdefault(
                key, {"banks": 0, "copies": 1, "modules": set(), "kind": kw, "line": line, "module_declarations": {}}
            )
            # BANKS ARE SIBLINGS, COPIES ARE NOT. Instances of the declaring module that sit side by side
            # under one parent are the banks of one store (a register file built from N bank modules);
            # instances under DIFFERENT parents are separate stores of separate devices (the same buffer
            # module in each of two compute units). Summing every instance as a bank reported one
            # accumulator twice the size of either, which a schedule would address past the end of.
            group = max([n for (_parent, child), n in widest.items() if child == module] or [1])
            if instances % group:
                group = 1  # siblings do not tile the instances evenly: count none as banks
            rec["banks"] += group
            rec["copies"] = max(rec["copies"], instances // group)
            rec["modules"].add(module)
            rec["module_declarations"][module] = rec["module_declarations"].get(module, 0) + 1
    undeterminable: list[dict[str, Any]] = []
    for (site, name, typ), rec in sorted(groups.items(), key=lambda kv: -kv[1]["banks"]):
        shape = _mem_shape(typ)
        mods = sorted(rec["modules"])
        # Elaboration may specialize one source memory into several module
        # variants. Group their instances by actual parent before deciding banks:
        # variants under different compute units are copies, not a larger store.
        declarations = rec["module_declarations"]
        unit_stack = [tree]
        while unit_stack and unit_stack[-1]["module_name"] != root:
            node = unit_stack.pop()
            unit_stack.extend(reversed(node.get("instances") or []))
        bank_groups = list(declarations.values())
        unit_stack = unit_stack[-1:]  # Exactly one selected accelerator unit.
        while unit_stack:
            node = unit_stack.pop()
            children = node.get("instances") or []
            bank_groups.append(sum(declarations.get(child["module_name"], 0) for child in children))
            unit_stack.extend(children)
        banks = max(bank_groups, default=1)
        total_instances = sum(counts.get(module, 0) * declarations[module] for module in mods)
        if total_instances % banks:
            banks = 1
        rec["copies"] = max(total_instances // banks, 1)
        if shape is None:
            undeterminable.append(
                {
                    "name": name,
                    "banks": banks,
                    "type": typ,
                    "modules": mods,
                    "site": site,
                    "reason": "the declared type is not a UInt/SInt memory, so it carries no element "
                    "width or depth this reader can state",
                }
            )
            continue
        elem_bits, row_elems, depth = shape
        row_bits = row_elems * elem_bits
        mem: dict[str, Any] = {
            "name": name,
            "banks": banks,
            "depth": depth,
            "row_elems": row_elems,
            "elem_bits": elem_bits,
            "row_bits_rtl": row_bits,
            "bytes": (banks * row_bits * depth) // 8 if row_bits * depth % 8 == 0 else None,
            "source": "firrtl_census",
            "modules": mods,
            "evidence": f"{banks}x `{rec['kind']} {name.split('.')[-1]} : {typ}` in {mods} "
            f"@ {site} (per {root} unit; {n_units} unit(s) in this elaboration)",
        }
        if rec["copies"] > 1:
            mem["copies"] = rec["copies"]
            mem["copies_note"] = (
                f"{rec['copies']} instances of {mods} under different parents: separate "
                "stores of separate devices, each with the banks/depth/bytes above -- "
                "not banks of one address space"
            )
        if mem["bytes"] is None:
            mem["bytes_unknown"] = (
                f"one bank holds {row_bits} bits x {depth} rows, which is not a "
                f"whole number of bytes; a byte capacity would be a rounding, not "
                f"a measurement"
            )
        out["memories"].append(mem)
        # A datapath is emitted ONLY when the RTL declares the element decomposition (a vector row).
        # A flat `UInt<256>` row states a word width and nothing about elements; deriving one would
        # hand every consumer a bank count that is wrong by the packing factor.
        if row_elems > 1:
            out["datapaths"].append(
                {
                    "name": name,
                    "dtype": _element_token(elem_bits),
                    "elem_bits": elem_bits,
                    "evidence": f"{rec['kind']} row `{typ}` @ {site}: an {elem_bits}-bit unsigned storage "
                    f"element ({row_elems} per row). The FIRRTL declares the WIDTH; the "
                    f"signedness/format of the value stored is not declared by a memory.",
                }
            )
    if undeterminable:
        out["memories_undeterminable"] = undeterminable

    # ---- arrays: the widest same-module sibling group inside the unit.
    ranked = sorted(widest.items(), key=lambda kv: (-kv[1], kv[0]))
    for (parent, child), n in ranked:
        out["replication_groups"].append({"container": parent, "element": child, "instances": n})
    if ranked:
        top = ranked[0][1]
        tied = [(p, c) for (p, c), n in ranked if n == top]
        side = math.isqrt(top)
        array: dict[str, Any] = {
            "instances": top,
            "container": ranked[0][0][0],
            "element": ranked[0][0][1],
            "source": "firrtl_census",
            "evidence": f"{top} instances of {ranked[0][0][1]} under one {ranked[0][0][0]} — the widest "
            f"same-module sibling group inside the {root} unit",
        }
        if len(tied) > 1:
            array["ambiguous_with"] = [{"container": p, "element": c} for p, c in tied[1:]]
            array["geometry_unknown"] = (
                f"{len(tied)} sibling groups tie at {top} instances ({tied}); which one is the compute "
                f"array — and therefore what a store's row spans — is not decidable from replication "
                f"alone, so no row/column extent is claimed"
            )
        elif side * side == top:
            array.update({"rows": side, "cols": side, "square": True})
        else:
            array["geometry_unknown"] = (
                f"{top} is not a perfect square, so a row/column extent is not derivable from the "
                f"instance count; the count itself is the fact"
            )
        # NAMED BY WHAT WAS ESTABLISHED, not by what was hoped for. A group that resolves a
        # row/column extent is the compute array and takes the role name every consumer (and mlc's
        # own discovery, which de-duplicates on it) uses. A group that does not is exactly one fact —
        # "these N siblings are the widest replication in the unit" — and calling that a mesh would
        # assert a geometry the very next field says is unknown. On a vector unit the widest group is
        # a compare pipe, and it is not a mesh at all.
        array["name"] = "mesh" if "geometry_unknown" not in array else "widest_replication"
        out["arrays"].append(array)

    # ---- interfaces: a host co-processor command port on the unit boundary, read from its ports.
    cmd = _host_command_port(ports.get(root, ""))
    if cmd:
        out["interfaces"].append(
            {
                "name": "rocc_cmd",
                "source": "firrtl_census",
                "evidence": f"module {root} exposes a decoupled `{cmd}` port whose payload carries a "
                f"RISC-V instruction bundle ({sorted(_INSTRUCTION_FIELDS)}): a host-decoded "
                f"co-processor command handoff",
            }
        )
    else:
        out["interfaces_note"] = (
            f"module {root} exposes no decoupled port carrying a RISC-V instruction bundle, so this "
            f"unit is not reached by a host co-processor command queue. Data-movement and completion "
            f"interfaces are NOT derived by this census — their absence here is unexamined, not absent."
        )
    return out


def _bundle_fields(text: str) -> list[tuple[str, str]]:
    """``[(field name, field type text)]`` for a FIRRTL bundle body, split at brace depth 0."""
    out: list[tuple[str, str]] = []
    depth, start = 0, 0
    chunks: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            chunks.append(text[start:i])
            start = i + 1
    if text[start:].strip():
        chunks.append(text[start:])
    for chunk in chunks:
        head, sep, typ = chunk.partition(":")
        if not sep:
            continue
        parts = head.split()
        if parts and parts[0] == "flip":
            parts = parts[1:]
        if parts:
            out.append((parts[0], typ.strip()))
    return out


def _inner_bundle(typ: str) -> str:
    """The body of the outermost ``{...}`` in ``typ``, or ``""`` when the type is not a bundle."""
    open_i = typ.find("{")
    if open_i == -1:
        return ""
    depth = 0
    for i in range(open_i, len(typ)):
        if typ[i] == "{":
            depth += 1
        elif typ[i] == "}":
            depth -= 1
            if depth == 0:
                return typ[open_i + 1 : i]
    return ""


def _host_command_port(port_text: str) -> str | None:
    """The name of a decoupled port field carrying a RISC-V instruction bundle, or None.

    Structural, three levels deep and no deeper: a port field that is a ready/valid handshake, whose
    ``bits`` payload contains a sub-bundle declaring the instruction-format fields. That shape IS the
    co-processor handoff; a design that does not have it is not reached that way.
    """
    for line in port_text.splitlines():
        _, _, rest = line.partition(":")
        for name, typ in _bundle_fields(_inner_bundle(rest)):
            leaves = dict(_bundle_fields(_inner_bundle(typ)))
            if "ready" not in leaves or "valid" not in leaves:
                continue
            payload = leaves.get("bits", "")
            for _sub, subtyp in _bundle_fields(_inner_bundle(payload)):
                fields = {f for f, _ in _bundle_fields(_inner_bundle(subtyp))}
                if _INSTRUCTION_FIELDS <= fields:
                    return name
    return None


def _directed_fields(text: str) -> list[tuple[str, bool, str]]:
    """``[(field name, flipped, field type text)]`` for a FIRRTL bundle body; like :func:`_bundle_fields`
    but keeping the ``flip`` that says the field points INTO the module."""
    out: list[tuple[str, bool, str]] = []
    depth, start = 0, 0
    chunks: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            chunks.append(text[start:i])
            start = i + 1
    if text[start:].strip():
        chunks.append(text[start:])
    for chunk in chunks:
        head, sep, typ = chunk.partition(":")
        if not sep:
            continue
        parts = head.split()
        flipped = bool(parts) and parts[0] == "flip"
        if flipped:
            parts = parts[1:]
        if parts:
            out.append((parts[0], flipped, typ.strip()))
    return out


def _uint_width(typ: str) -> int | None:
    """The width of a ``UInt<W>`` type text, else None."""
    head, sep, rest = typ.strip().partition("<")
    if head.strip() != "UInt" or not sep:
        return None
    width, close, _ = rest.partition(">")
    return int(width) if close and width.strip().isdigit() else None
