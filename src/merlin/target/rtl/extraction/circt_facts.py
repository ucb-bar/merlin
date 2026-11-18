"""Deterministic, target-agnostic CIRCT/FIRRTL structural fact extraction.

Prefer the version-matched ModelIR HW discovery for decoded opcodes, geometry,
memory and timing. Use declared FIRRTL for census and arithmetic-cell evidence;
legacy HW-port/ISA parsing is available only when its source is selected.
Instruction codes come from RTL. Their names are declared ISA vocabulary, not
additional hardware observations.

Record exact source-byte commitments and report missing evidence as unknown.
Source consistency requires a separate production/equivalence receipt; matching
configuration names alone cannot establish it. ``validate`` reports contradictions
against software declarations instead of reconciling them silently.

CLI: ``python -m merlin.target.rtl.extraction.circt_decode --target <t> --out facts.json``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.common.target_data import interface as _facts_interface
from merlin.target.rtl.extraction import census as V1
from merlin.target.rtl.extraction import declared_readers as extraction_contract
from merlin.target.rtl.extraction import introspection as _merlin_introspection_3
from merlin.target.rtl.extraction.declared_readers import accumulator_layout as _accumulator_layout
from merlin.target.rtl.extraction.declared_readers import boolean_feature_gate as _boolean_feature_gate
from merlin.target.rtl.extraction.declared_readers import extract_boolean_feature_gate
from merlin.target.rtl.extraction.declared_readers import scala_funct_block as _scala_funct_block
from merlin.target.rtl.facts import rtl_cache_dir
from merlin.target.rtl.firrtl.ports import _module_port_sig, extract_accumulator, memories_from_port_geometry

# ---- circt_introspect --------------------------------------------------------------------------


def elaborated_firrtl(target: str) -> list[Path]:
    """Every elaborated ``.fir`` in ``target``'s purgeable RTL cache, sorted.

    A design whose memory is instantiated ABOVE the module a standalone elaboration can build is
    invisible to a cmem/smem census of that elaboration -- but the module below it still declares the
    store's geometry on its PORTS. These are the elaborations that carry those ports.
    """
    from merlin.target.rtl.extraction.circt_decode import _FIRRTL_SUBDIR

    d = rtl_cache_dir(target) / _FIRRTL_SUBDIR
    return sorted(p for p in d.glob("*.fir")) if d.is_dir() else []


def _memories_from_declared_elaboration(target: str, facts: dict) -> list[str]:
    """Fill ``facts['memories']`` from the SRAM declarations in the elaboration the target DECLARES.

    Where the chipyard census above did not run (no ``<T>ISA.scala`` by the generator convention) but the
    target's descriptor names its elaboration (``rtl.elaboration``: checkout, config, generator), that
    elaboration's FIRRTL declares every SRAM the target's own generator emitted -- scoped by the
    ``@[generators/<gen>/...]`` annotation, so a host core's caches are not counted. This reads them
    before the port-geometry filler, which exists for the case where no declaration is visible and which
    sees only stores with banked write ports (a compute unit's local accumulation buffers have none).

    A NO-OP when memories are already known, when nothing is declared, or when the declared elaboration
    is not on disk. Mutates ``facts``; returns the provenance names sourced.
    """
    from merlin.target.rtl.extraction.circt_decode import _sha256

    if facts.get("memories"):
        return []
    try:
        source = V1.declared_rtl_source(target)
        arts = source.artifacts()
    except Exception:  # noqa: BLE001 -- undeclared or unresolvable: the next filler decides
        return []
    fir, hierarchy = Path(arts["fir"]), Path(arts["hierarchy"])
    if not fir.is_file() or not hierarchy.is_file():
        return []
    census = V1.census_facts(fir, hierarchy, generator=source.generator)
    facts.setdefault("census", census.get("census"))
    facts.setdefault(
        "source",
        {
            "kind": "firrtl_census",
            "config": source.config,
            "generator": source.generator,
            "fir": fir.name,
            "hierarchy": hierarchy.name,
            "fir_path": str(fir.resolve()),
            "fir_sha256": _sha256(fir),
            "hierarchy_path": str(hierarchy.resolve()),
            "hierarchy_sha256": _sha256(hierarchy),
            "declared_by": str(source.origin),
            "note": "memories from the SRAM declarations of the elaboration this target's descriptor names",
        },
    )
    if not census.get("memories"):
        return []
    facts["memories"] = census["memories"]
    return [f"memories({len(census['memories'])} from the declared elaboration's census)"]


def _selected_firrtl(target: str, facts: dict) -> list[Path]:
    """Use the selected census elaboration, not unrelated ambient cache files.

    An explicit declaration remains authoritative even when missing. Only a
    target with no resolvable declaration may use legacy cached elaborations.
    """
    source = facts.get("source") or {}
    if source.get("fir_path"):
        return [Path(source["fir_path"])]
    try:
        return [Path(V1.declared_rtl_source(target).artifacts()["fir"])]
    except V1.RtlSourceInvalid as exc:
        facts.setdefault("datapaths_undeterminable", []).append(str(exc))
        return []
    except (V1.RtlSourceUndeclared, KeyError, FileNotFoundError):
        return elaborated_firrtl(target)


def _record_firrtl_reads(facts: dict, paths: Iterable[Path]) -> None:
    """Record the exact bytes consumed by the port and compute-cell readers."""
    from merlin.target.rtl.extraction.circt_decode import _sha256

    reads = facts.setdefault("source", {"kind": "firrtl_reader_inputs"}).setdefault("firrtl_inputs", [])
    for path in paths:
        receipt = {"path": str(path.resolve()), "sha256": _sha256(path)}
        if receipt not in reads:
            reads.append(receipt)


def _memories_from_ports(target: str, facts: dict) -> list[str]:
    """Fill ``facts['memories']`` from banked write ports when the census produced none. Mutates
    ``facts``; returns the provenance names sourced.

    A NO-OP for any target whose census (or mlc discovery) already produced memories: this is the gap
    filler for the elaboration a census cannot see into, never a competing reading of one it can.
    """
    if facts.get("memories"):
        return []
    paths = _selected_firrtl(target, facts)
    _record_firrtl_reads(facts, paths)
    mems = memories_from_port_geometry(paths)
    if not mems:
        return []
    facts["memories"] = mems
    return [f"memories({len(mems)} from port geometry)"]


def _datapaths_from_cells(target: str, facts: dict) -> list[str]:
    """Fill ``facts['datapaths']`` from the COMPUTE ELEMENT's own port geometry when the census
    produced none. Mutates ``facts``; returns the provenance names sourced.

    A NO-OP for any target whose role-anchored probes or census already produced datapaths, and the
    reason is sharper than precedence: THE TWO READINGS ARE NOT THE SAME QUANTITY. The existing one is
    the accumulator SRAM's declared element; this one is the compute cell's partial-sum chain, and on a
    real systolic design those differ (measured: a 20-bit chain feeding a 32-bit accumulator store).
    Both are true, they answer different questions, and the SRAM reading is the one every consumer has
    been calibrated against — so it is never displaced by this one, which fills the gap where there is
    no reading at all.

    Where the census produced NOTHING, the alternative to this is not silence -- it is a target contract
    declaring an operand and accumulate format from a generator's parameter class and admitting it is
    not RTL-grounded. A width read off the cell's ports, with a format the cell's own arithmetic
    children NAME, is the fact that declaration was standing in for.

    Refusals are RECORDED, never swallowed: a kind whose compute element is not locatable, an
    elaboration that is absent, a width no registered format is named for -- each lands in
    ``facts['datapaths_undeterminable']`` so the gap is visible instead of reading as "this design has
    no datapath".
    """
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge
    from merlin.target.rtl.firrtl.datapaths import datapaths_from_compute_cells

    fir = _selected_firrtl(target, facts)
    if not fir:
        facts.setdefault("datapaths_undeterminable", []).append(
            "no elaborated FIRRTL is selected for this target, so its compute element's port geometry "
            "could not be read (UNKNOWN, not absent)"
        )
        return []
    _record_firrtl_reads(facts, fir)
    dps, notes = datapaths_from_compute_cells(facts, fir, _merlin_mlc_discovery_1.compute_unit_kinds(target))
    if notes:
        facts.setdefault("datapaths_undeterminable", []).extend(notes)
    if not dps:
        return []
    facts["compute_datapaths"] = dps
    if facts.get("datapaths"):
        facts["storage_datapaths"] = facts["datapaths"]
        # Arithmetic format evidence is distinct from storage width. Preserve
        # existing named compatibility roles and add only missing input/accumulator.
        names = {row.get("name") for row in facts["datapaths"]}
        facts["datapaths"] = [*facts["datapaths"], *(row for row in dps if row.get("name") not in names)]
    else:
        facts["datapaths"] = dps
    return [f"datapaths({len(dps)} from cell geometry)"]


def _lane_datapaths_from_units(target: str, facts: dict) -> list[str]:
    """Fill ``facts['lane_datapaths']`` with the element format of each LANE engine beside the compute
    array (a vector unit): the units that replicate their arithmetic once per lane, with the format that
    per-lane arithmetic names. Mutates ``facts``; returns the provenance names sourced.

    The array cell reading above says what the ARRAY consumes and accumulates in. A target can carry a
    second engine with its own element format -- a bf16 vector unit beside an fp8 array -- and without
    this reading the capability deriver could only guess that engine's formats from the array's, which
    is how an accumulator format came to be claimed as an operand format. Refusals are recorded in
    ``datapaths_undeterminable`` like the cell reader's.
    """
    from merlin.target.rtl.firrtl.datapaths import lane_datapaths

    fir = _selected_firrtl(target, facts)
    if not fir:
        return []
    _record_firrtl_reads(facts, fir)
    lanes, notes = lane_datapaths(facts, fir)
    if notes:
        facts.setdefault("datapaths_undeterminable", []).extend(notes)
    if not lanes:
        return []
    facts["lane_datapaths"] = lanes
    return [f"lane_datapaths({sum(1 for r in lanes if r.get('dtype'))} named from lane replication)"]


def _timing_from_discovery(target: str, facts: dict) -> list[str]:
    """Add RTL-DERIVED per-module pipeline depth (:mod:`.timing`). Mutates ``facts``; returns the
    provenance names sourced.

    Timing is the fact class merlin never had, and it is the one a statically-scheduled target cannot
    be compiled correctly without: where the ISA has no interlocks, the delays live in the PROGRAM, so
    a wrong latency is a wrong ANSWER rather than a slow one. It is derived by counting register
    stages in the target's own RTL -- never read from a vendor performance model, which is a
    hand-written artifact that may disagree with the RTL it claims to describe and will not exist for
    the next target.

    Absent (not empty) when the RTL is unreachable: an empty list would say "this design has no
    timing", which is a claim about hardware, where the truth is that nobody could look.
    """
    from merlin.target.rtl.hw import timing as _timing

    try:
        recs = _timing.discovered_timing(target)
    except Exception:  # noqa: BLE001 -- unreachable/unparseable RTL is UNKNOWN, never a fabricated depth
        return []
    if not recs:
        return []
    facts["timing"] = recs
    resolved = sum(1 for r in recs if r.get("pipeline_depth") is not None)
    return [f"timing({resolved}/{len(recs)} modules)"]


def extract_elaborated_rtl_features(target: str, facts: dict, fir_path: Path | str | None = None) -> dict[str, Any]:
    """Normalize feature gates from exact elaborated RTL, with source config as corroboration only."""
    from merlin.target.rtl.extraction.circt_decode import _sha256

    gate = _boolean_feature_gate(target)
    source = facts.get("source") or {}
    path = Path(fir_path or source.get("fir_path")) if (fir_path or source.get("fir_path")) else None
    observed = None
    if gate is not None and path is not None and path.is_file():
        try:
            observed = extract_boolean_feature_gate(path.read_text(encoding="utf-8", errors="replace"), gate=gate)
        except OSError:
            observed = None

    corroboration: dict[str, Any] = {"status": "unavailable"}
    try:
        from merlin.target.capabilities.discovery import configuration as _merlin_configuration_2
        from merlin.target.capabilities.discovery import sources as discovery

        config = _merlin_configuration_2.elaborated_config(target, facts) if gate is not None else None
        if config is not None and gate is not None:
            field_name = gate["config_field"]
            field = config.fields.get(field_name)
            config_value = config.boolean(field_name) if not config.ambiguities else None
            corroboration = {
                "status": (
                    "agree"
                    if observed is not None and config_value is observed["value"]
                    else "diverge"
                    if observed is not None and isinstance(config_value, bool)
                    else "unknown"
                ),
                "config": config.name,
                "instantiated": config.instantiated or None,
                field_name: config_value,
                "source": field.locator if field is not None else None,
                "line": field.line if field is not None else None,
            }
    except Exception:  # noqa: BLE001 — corroboration never decides the hardware fact
        pass

    value = observed["value"] if observed is not None else None
    status = "derived" if isinstance(value, bool) else "unknown"
    if gate is None:
        evidence = "target declares no FIRRTL boolean feature gate"
    elif path is not None and observed is not None:
        evidence = f"{path}:{observed['line']}: {observed['expression']}"
    else:
        evidence = "no unique declared structural FIRRTL boolean build gate was derivable"
    return {
        "name": "elaborated_rtl_features",
        "features": {gate["feature"]: value} if gate is not None else {},
        "status": status,
        "source": str(path) if gate is not None and path is not None else None,
        "source_sha256": _sha256(path) if gate is not None and path is not None else "unresolved",
        "method": f"structural FIRRTL {gate['node']} boolean build gate" if gate is not None else "undeclared",
        "evidence": evidence,
        "corroboration": corroboration,
    }


def build_facts(
    hw_path: Path | str | None = None,
    isa_path: Path | str | None = None,
    chipyard_root: str | Path | None = None,
    target: str | None = None,
    source_bundle: Path | str | None = None,
    generic_output: Path | str | None = None,
) -> dict[str, Any]:
    """Extract selected sources; an explicit source bundle is authoritative."""
    if source_bundle is None:
        return _build_facts(hw_path, isa_path, chipyard_root, target)
    if hw_path is not None:
        raise ValueError("source_bundle and hw_path are mutually exclusive")
    from merlin.common.external_tools import discovery_imports
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge
    from merlin.target.rtl.extraction.source_selection import (
        load_selection,
        production_consistency,
        selected_sources,
    )

    selected = load_selection(source_bundle, target=target)
    consistency = production_consistency(selected)
    selected["_generic_hw_output"] = str(
        Path(generic_output)
        if generic_output is not None
        else rtl_cache_dir(selected["target"]) / "genericized" / f"{selected['sources']['core_hw']['sha256']}.mlir"
    )
    with selected_sources(selected), discovery_imports(mlc_bridge.mlc_dir()):
        record = _build_facts(Path(selected["sources"]["soc_hw"]["path"]), isa_path, chipyard_root, target)
    record["inputs"]["source_bundle_path"] = selected["selection_path"]
    record["inputs"]["source_bundle_sha256"] = selected["selection_sha256"]
    genericization = selected.get("_genericization")
    if genericization is not None:
        record["inputs"]["generic_hw_path"] = genericization["output"]["path"]
        record["inputs"]["generic_hw_sha256"] = genericization["output"]["sha256"]
        consistency["genericization"] = genericization
        consistency["sources"].append({"role": "core_hw_generic", **genericization["output"]})
    record["source_consistency"] = consistency
    return record


def _build_facts(
    hw_path: Path | str | None = None,
    isa_path: Path | str | None = None,
    chipyard_root: str | Path | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """Assemble the RTL facts for ``target``. PREFERS mlc RTL discovery (target-agnostic: mesh DIM +
    memory capacities + the decoder-derived ISA); the chipyard FIRRTL grep + HW-port parse is the
    legacy FALLBACK, run only for a target that ships a Chisel ISA source and skipped
    entirely otherwise (a non-chipyard target relies wholly on discovery — no chipyard path is touched).

    Every input path is resolved FROM ``target`` when not explicitly overridden: the SoC HW cache
    (``<t>_soc.hw.mlir``), the Chisel ISA source (``<T>ISA.scala`` by generator convention), the
    declared ISA headers. Provenance-stamped; does not write (see :func:`dump_facts`).

    ``target`` is required (last so the path overrides can stay positional); a missing target is a
    loud error, never a silent default-target fallback."""
    from merlin.target.rtl.extraction.circt_decode import (
        _ROCC_FUNCT3_DEFAULT,
        GENERATOR_VERSION,
        _bundle_layouts,
        _core_hw_input,
        _facts_from_discovery,
        _funct_name_table,
        _reconcile_funct,
        _rocc_custom_opcode,
        _sha,
        _sha256,
        _soc_hw_path,
        extract_funct_table_via_decoder,
        isa_scala_path,
    )

    if target is None:
        raise ValueError("build_facts requires an explicit target (no default is assumed)")
    from merlin.target.rtl.extraction import declared_readers as feature_reader
    from merlin.target.rtl.extraction import (
        declared_readers as memory_roles,
    )
    from merlin.target.rtl.extraction import (
        mlc_toolchain as mlc_bridge,
    )
    from merlin.target.rtl.extraction import (
        source_selection,
    )
    from merlin.target.rtl.facts import target_contract_path
    from merlin.target.rtl.firrtl import datapaths as datapath_reader
    from merlin.target.rtl.hw import command_ordering, timing
    from merlin.target.rtl.hw import graph as hw_graph
    from merlin.target.rtl.hw import ordering_search as _merlin_ordering_search_4
    from merlin.target.rtl.hw import timing as timing_structure

    readers = [
        {"path": str(Path(module.__file__).resolve()), "sha256": _sha256(Path(module.__file__))}
        for module in (
            V1,
            extraction_contract,
            datapath_reader,
            feature_reader,
            source_selection,
            hw_graph,
            mlc_bridge,
            timing,
            memory_roles,
            timing_structure,
        )
    ]
    extractor_digest = _sha256(Path(__file__))

    try:
        extraction_contract_path = target_contract_path(target)
    except (KeyError, FileNotFoundError):
        extraction_contract_path = None
    if chipyard_root is None:
        try:
            chipyard_root = V1.declared_rtl_source(target).root
        except V1.RtlSourceInvalid:
            raise
        except (V1.RtlSourceUndeclared, KeyError, FileNotFoundError):
            chipyard_root = V1.default_chipyard()
    hw_path = _soc_hw_path(target) if hw_path is None else Path(hw_path)
    isa_path = isa_scala_path(target, chipyard_root) if isa_path is None else Path(isa_path)

    from merlin.target.rtl.extraction.source_selection import active_selection

    selected = active_selection(target)
    fir_sha = fir_sha256 = isa_sha = "n/a"
    elaborated_fir: Path | None = None
    v1: dict[str, Any] = {}
    if selected is not None:
        elaborated_fir = Path(selected["sources"]["firrtl"]["path"])
        selected_hierarchy = Path(selected["sources"]["hierarchy"]["path"])
        source_selection.require_generator_scope(selected)  # a wrong scope would census nothing
        v1 = V1.census_facts(elaborated_fir, selected_hierarchy, generator=selected["generator"])
        # Legacy declared role probes are supplemental readers of the same bytes,
        # never an alternate selected elaboration.
        if extraction_contract.firrtl_role_probe(target) is not None:
            probes = _merlin_introspection_3.extract_facts(
                elaborated_fir, selected_hierarchy, role_target=target, config=selected["config"]
            )
            for field in ("arrays", "datapaths", "interfaces"):
                if probes.get(field):
                    v1[field] = probes[field]
            # Declared roles name census stores; with no census the role probes are the stores.
            v1["memories"] = memory_roles.label_roles(v1.get("memories") or [], probes.get("memories") or [])
        v1["source"] = {
            "kind": "explicit_source_bundle",
            "config": selected["config"],
            "generator": selected["generator"],
            "fir_path": str(elaborated_fir),
            "fir_sha256": _sha256(elaborated_fir),
            "hierarchy_path": str(selected_hierarchy),
            "hierarchy_sha256": _sha256(selected_hierarchy),
            "declared_by": selected["selection_path"],
        }
        layout = _accumulator_layout(target)
        if layout is not None:
            address_space = extract_accumulator(hw_path.read_text(), layout=layout)
            if address_space is not None:
                address_space.update(
                    {
                        "source": "selected_hw_port_geometry",
                        "view_kind": "address_space",
                        "physical_aliases_status": "unknown",
                        "accounting_note": "logical address-space view; do not sum with physical census stores",
                    }
                )
                # Preserve the named compatibility view consumed by profiles; census rows that
                # physically back it move to census.physical_memories (never counted twice).
                v1.setdefault("memories", []).append(address_space)
                v1["memories"] = memory_roles.separate_backings(v1["memories"], v1["memories"], v1)
                v1["memory_address_spaces"] = [row for row in v1["memories"] if row.get("view_kind") == "address_space"]
    elif isa_path.is_file():  # a chipyard target with a Chisel ISA source -> legacy FIRRTL grep + HW-port
        accumulator_layout = _accumulator_layout(target)
        try:
            # the elaboration the target's own simulators are built from (runtime.rtl_sim_config)
            arts, v1 = V1.sim_elaboration_facts(target, chipyard_root)
            elaborated_fir = Path(arts["fir"])
            hw_text = hw_path.read_text(errors="replace") if hw_path.is_file() else ""
            acc = extract_accumulator(hw_text, layout=accumulator_layout) if hw_text and accumulator_layout else None
            if acc:
                v1["memories"] = [m for m in v1.get("memories", []) if m.get("name") != acc["name"]] + [acc]
            fir_sha, fir_sha256, isa_sha = (_sha(elaborated_fir), _sha256(elaborated_fir), _sha(isa_path))
        except Exception:  # noqa: BLE001 — chipyard artifacts absent/broken: rely on mlc discovery
            v1 = {}

    # PREFER mlc discovery (target-agnostic) for mesh + memory capacities.
    sourced = _facts_from_discovery(target, v1)
    # Only where nothing above produced a memory list: a store the census cannot see (instantiated
    # above the module a standalone elaboration builds) still declares its geometry on the ports below
    # it, and without this the target's whole memory-mapping axis is unanswerable.
    sourced += _memories_from_declared_elaboration(target, v1)
    sourced += _memories_from_ports(target, v1)
    # Only where nothing above produced a datapath: the compute element declares what it consumes and
    # what it accumulates in on its own ports, and without this a target with no census datapath had its
    # operand/accumulate formats DECLARED by a contract rather than measured.
    sourced += _datapaths_from_cells(target, v1)
    sourced += _lane_datapaths_from_units(target, v1)
    sourced += _timing_from_discovery(target, v1)
    sourced += timing_structure.add_to_facts(target, v1)  # capacities, in-flight limits, fixed delays

    # Decoder comparison field: preserve RTL-observed values over the weaker
    # header parse, without treating either as an exhaustive legal set. NAMES
    # are sourced target-agnostically (Chisel ISA source > declared ISA headers > generic funct_<code>).
    header_funct = _funct_name_table(target, isa_path)
    decoder_funct = extract_funct_table_via_decoder(target)
    funct = _reconcile_funct(decoder_funct, header_funct)
    if funct:
        # The RoCC major opcode is a SoC-config fact the accelerator decoder does not carry: resolve it
        # from the target's reviewed custom-slot (RISC-V standard slot->opcode), UNKNOWN if undeclared.
        funct["custom_opcode"] = _rocc_custom_opcode(target)
        funct.setdefault("funct3", _ROCC_FUNCT3_DEFAULT)
        v1.setdefault("interfaces", []).append(funct)
        sourced.append("funct" if funct.get("method", "").startswith("decoder") else "funct(header)")

    # Packed command-register layouts come from the target's own Chisel ISA Bundle declarations. They
    # are separate from the funct decode table: the decoder says WHICH instruction this is, while this
    # record says how that instruction's rs payload is partitioned. An unavailable ISA source leaves the
    # record absent, and a code generator needing one must fail closed rather than copy bit positions.
    if isa_path.is_file():
        layouts, unresolved_bundles = _bundle_layouts(isa_path.read_text(errors="replace"))
        if layouts or unresolved_bundles:
            v1.setdefault("interfaces", []).append(
                {
                    "name": "register_bundle_layouts",
                    "bundles": layouts,
                    "unresolved": unresolved_bundles,
                    "source": str(isa_path),
                    "method": "chisel_bundle_reverse_declaration_order",
                    "evidence": (
                        f"{isa_path.name}: {len(layouts)} fully-derived packed Bundle layout(s); "
                        "field widths from the file's val declarations"
                    ),
                }
            )
            sourced.append(f"register_layouts({len(layouts)})")

    # ISA fields prove that pooling is encodable, not that the elaborated StoreController contains the
    # datapath. Resolve its literal build gate from the exact elaborated FIRRTL and publish a three-state
    # value beside the other interfaces. The source configuration is corroboration only. Consumers
    # require literal True; false/unknown can never manufacture pooled coverage.
    build_features = extract_elaborated_rtl_features(target, v1, elaborated_fir)
    v1.setdefault("interfaces", []).append(build_features)
    sourced.append(f"elaborated_features({build_features['status']})")
    # COMMAND ORDERING (queues, release points, spans): probes the dispatch module with the layouts above
    sourced += _merlin_ordering_search_4.add_to_facts(target, v1)

    # WHO THESE FACTS ARE ABOUT, inside the body -- not only in ``inputs``. A pin is matched to a target
    # by what its body DECLARES (``facts._facts_declares`` reads ``facts.target``), never by the
    # directory it sits in, so a body without this key can be matched to no target at all. The
    # FIRRTL-census path already stamps it (``introspect.py``: ``body.setdefault("target", target)``);
    # this path did not.
    v1.setdefault("target", target)
    # WHICH ELABORATION they describe. The census path records this as ``source`` (the config it read,
    # e.g. a Chipyard ``*Config``, with the .fir it came from). Where the census did not run, these facts
    # were extracted from a cached HW dialect and NOTHING said which configuration that dialect is of --
    # so the artifact could not be attributed to a device, which is the repo's hardware-provenance rule
    # exactly inverted. The target's own descriptor names it (``rtl.elaboration``), so record that,
    # under its own ``kind`` so a DECLARED elaboration is never read as a census-DERIVED one. Undeclared
    # stays absent: an unattributable artifact says so rather than naming a config nobody verified.
    if not v1.get("source"):
        try:
            declared = V1.declared_rtl_source(target)
            v1["source"] = {
                "kind": "declared_elaboration",
                "config": declared.config,
                "generator": declared.generator,
                "root": str(declared.root),
                "declared_by": str(declared.origin),
                "note": (
                    "the elaboration this target's own descriptor names; the facts here were "
                    "extracted from the HW dialect cached for it, and this records WHICH "
                    "configuration that dialect is of -- it is not a second extraction of it"
                ),
            }
        except Exception:  # noqa: BLE001 — undeclared/unresolvable: leave `source` absent, never guess
            pass

    # Byte identity is not production identity. The independently selected HW
    # dialect and census elaboration have no shared build receipt today. Record
    # that gap explicitly; matching names, configurations or timestamps cannot
    # qualify the combined evidence as one hardware revision.
    source = v1.get("source") or {}
    core_input = _core_hw_input(target)
    if elaborated_fir is None and source.get("fir_path"):
        elaborated_fir = Path(source["fir_path"])
    hierarchy = Path(source["hierarchy_path"]) if source.get("hierarchy_path") else None
    fir_sha = _sha(elaborated_fir) if elaborated_fir is not None else fir_sha
    fir_sha256 = _sha256(elaborated_fir) if elaborated_fir is not None else fir_sha256
    inputs = {
        "hw_path": str(hw_path.resolve()),
        **core_input,
        "fir_path": str(elaborated_fir.resolve()) if elaborated_fir is not None else None,
        "hierarchy_path": str(hierarchy.resolve()) if hierarchy is not None else None,
        "hierarchy_sha256": _sha256(hierarchy) if hierarchy is not None else "unresolved",
        "firrtl_inputs": source.get("firrtl_inputs", []),
        "isa_path": str(isa_path.resolve()),
        "isa_sha256": _sha256(isa_path),
        "reader_sources": readers,
    }
    if _sha256(Path(__file__)) != extractor_digest or any(
        _sha256(Path(reader["path"])) != reader["sha256"] for reader in readers
    ):
        raise RuntimeError("RTL extraction source changed during the observation; retry against stable source bytes")

    return {
        "schema_version": "2.0",
        "generator": {
            "name": "merlin.target.rtl.extraction.circt_decode",
            "version": GENERATOR_VERSION,
            "method": f"mlc RTL discovery (target-agnostic: {', '.join(sourced) or 'none'}) + chipyard "
            "FIRRTL grep/HW-port fallback",
        },
        "inputs": {
            "target": target,
            "hw_mlir": hw_path.name,
            "hw_sha": _sha(hw_path),
            "hw_sha256": _sha256(hw_path),
            # The SoC dialect above is only one of the two HW inputs, and for most targets it is the
            # one that is ABSENT: it feeds the legacy accumulator port-parse. What mlc discovery and
            # the pipeline-depth walk actually read is the CORE dialect, resolved separately by
            # ``mlc_bridge.core_hw_mlir``. Recording only the SoC path made these facts name an input
            # they had not read (``hw_sha: "missing"``) while omitting the one they had, which reads as
            # provenance rather than as the gap it is. Both are recorded now, and the three states are
            # kept distinct: a digest, ``missing`` for a resolved-but-absent file, and ``unresolved``
            # when mlc cannot resolve one at all. A term's validity domain cannot name its elaboration
            # unless this field does.
            **inputs,
            "fir_sha": fir_sha,
            "fir_sha256": fir_sha256,
            "isa_sha": isa_sha,
            "extractor_sha": extractor_digest[:16],
            "extractor_sha256": extractor_digest,
            "extraction_contract_sha256": (
                _sha256(extraction_contract_path) if extraction_contract_path is not None else "unresolved"
            ),
            "extraction_reader_sha256": _sha256(Path(extraction_contract.__file__)),
        },
        "source_consistency": {
            "status": "unverified",
            "reason": "no shared production receipt or reviewed equivalence binds the selected HW/FIRRTL/hierarchy",
            "config": source.get("config"),
            "sources": [
                {"role": role, "path": inputs[path_key], "sha256": digest}
                for role, path_key, digest in (
                    ("core_hw", "core_hw_path", core_input.get("core_hw_sha256")),
                    ("soc_hw", "hw_path", _sha256(hw_path)),
                    ("firrtl", "fir_path", fir_sha256),
                    ("hierarchy", "hierarchy_path", inputs["hierarchy_sha256"]),
                )
                if inputs.get(path_key) and digest not in ("missing", "unresolved", "n/a", None)
            ]
            + [
                {"role": "firrtl_reader", **receipt}
                for receipt in inputs["firrtl_inputs"]
                if receipt["path"] != inputs["fir_path"] and receipt["sha256"] != "missing"
            ],
        },
        "facts": v1,
    }


def dump_facts(out_path: Path | str | None = None, **kw) -> dict[str, Any]:
    """Build facts and write the GENERATED artifact to ``out_path`` (default: the purgeable cache dir,
    NOT merlin/); cache-hit (no rebuild) when input SHAs are unchanged. This is the writer
    :func:`merlin.target.rtl.facts.ensure_facts` calls to fill a cold cache."""
    out = Path(out_path) if out_path is not None else rtl_cache_dir(kw["target"]) / "facts.json"
    if kw.get("source_bundle") is not None:
        kw.setdefault("generic_output", out.parent / "core.hw.generic.mlir")
    rec = build_facts(**kw)
    if out.is_file():
        try:
            old = json.loads(out.read_text())
            if old.get("inputs") == rec["inputs"] and old.get("generator") == rec["generator"]:
                return old  # deterministic cache hit
        except Exception:
            pass
    # Guarded: an extraction that lost the decode table looks identical to a good one on the way out.
    # Same ratchet every fact-extraction family is held to (see facts.write_facts_guarded).
    from merlin.target.rtl.facts import write_facts_guarded

    write_facts_guarded(out, rec)
    return rec


def validate(
    facts_rec: dict, contract: dict | None = None, rocc_funct_class: dict | None = None
) -> dict[str, list[str]]:
    """Cross-check selected evidence, distinguishing agreement, divergence and unknown checks.

    Divergence is information (e.g. an unconfirmed contract capacity), not a hard error — the whole
    point of RTL extraction is to *correct* curated guesses."""
    facts = facts_rec["facts"]
    agree, diverge, unknown = [], [], []

    if contract:
        # reuse v1's compute_units-coverage check (mesh/scratchpad/dtype/accumulator capacities are
        # no longer hand-declared in the contract — they ARE these facts, so nothing to cross-check).
        problems = _merlin_introspection_3.validate_against_contract(facts, contract)
        for p in problems:
            (unknown if p.startswith("UNKNOWN:") else diverge).append(f"contract: {p}")
        if not problems:
            agree.append("contract: RTL datapaths covered by declared compute_units")

    if rocc_funct_class is not None:
        funct = _facts_interface(facts, "funct_decode_table")
        if funct and funct.get("scope") == "complete_rocc_funct7" and funct.get("complete_isa") is True:
            legal = set(funct["legal_funct"])
            classifier = set(int(k) for k in rocc_funct_class)
            extra = classifier - legal
            if extra:
                diverge.append(f"rocc_decode classifies funct(s) {sorted(extra)} not in RTL legal set")
            else:
                agree.append(
                    f"rocc_decode funct classifier {sorted(classifier)} ⊆ RTL legal "
                    f"{funct['legal_funct'][0]}..{funct['legal_funct'][-1]}"
                )
        else:
            unknown.append("rocc_decode: complete RTL RoCC funct legality unavailable; observed field is diagnostic")
    consistency = facts_rec.get("source_consistency") or {}
    if consistency.get("status") != "verified":
        unknown.append(f"source consistency: {consistency.get('reason', 'legacy evidence has no production receipt')}")
    return {"agree": agree, "diverge": diverge, "unknown": unknown}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Deterministic, target-agnostic RTL facts via CIRCT HW (v2).")
    ap.add_argument("--target", required=True, help="the accelerator target whose RTL facts to extract")
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: purgeable artifacts/cache/rtl_introspect/<target>/facts.json)",
    )
    ap.add_argument("--hw", default=None, help="override the SoC HW-dialect input (default: per-target cache)")
    ap.add_argument("--source-bundle", default=None, help="authoritative byte-bound FIRRTL/HW/hierarchy selection")
    ap.add_argument("--validate", action="store_true", help="cross-check vs contract + rocc_decode")
    a = ap.parse_args(argv)
    out = a.out or str(rtl_cache_dir(a.target) / "facts.json")
    rec = dump_facts(out, target=a.target, hw_path=(Path(a.hw) if a.hw else None), source_bundle=a.source_bundle)
    facts = rec["facts"]
    acc = next((m for m in facts.get("memories", []) if m.get("name") == "accumulator"), {})
    funct = _facts_interface(facts, "funct_decode_table") or {}
    print(f"wrote {out}")
    print(f"  accumulator: depth={acc.get('depth')} bytes={acc.get('bytes')} addr_width={acc.get('addr_width')}")
    print(f"  funct legal: {funct.get('legal_funct')}")
    if a.validate:
        import yaml

        from merlin.target.rtl.facts import target_contract_path

        contract_unknown = None
        try:
            contract = yaml.safe_load(target_contract_path(a.target).read_text())
        except (FileNotFoundError, KeyError) as exc:
            contract = None
            contract_unknown = f"contract: selected contract is unavailable ({exc})"
        rocc_class = None
        try:
            from merlin.target.isa.rocc import decode as rocc_decode

            rocc_class = rocc_decode.funct_class_for(a.target)
        except Exception:  # noqa: BLE001 — no classifier for this target: skip that cross-check
            pass
        res = validate(rec, contract, rocc_class)
        if contract_unknown is not None:
            res["unknown"].append(contract_unknown)
        print("  AGREE:")
        [print(f"    + {x}") for x in res["agree"]]
        print("  DIVERGE:")
        [print(f"    ! {x}") for x in res["diverge"]] or print("    (none)")
        print("  UNKNOWN:")
        [print(f"    ? {x}") for x in res["unknown"]] or print("    (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
