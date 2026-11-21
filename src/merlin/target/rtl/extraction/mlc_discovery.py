"""Bridge to the sibling ``mlc`` model-ladder compiler — the CIRCT+xDSL RTL frontend we reuse for
RTL *needle* extraction (finding specific ISA facts in the HW-dialect op graph).

``mlc`` is an EXTERNAL dependency resolved via ``.env MERLIN_MLC_DIR`` (it will be upstreamed to its own
open-source repo). Rather than re-implement the op-graph parser + decoder analysis, we reuse mlc's
``discover.irgraph`` (``circt-opt --mlir-print-op-generic`` -> xDSL ``HwGraph``) and ``discover.decode``
(observed field values from the decoder's ``comb.icmp eq`` fan-out), plus mlc's own prebuilt CIRCT
binaries. ``mlc`` is pip-installed editable into merlin's venv from ``MERLIN_MLC_DIR`` (see the dev
setup docs), so its Python is a normal import. Imports are still function-local behind an availability
guard (``chia_bridge`` style) so importing this module never hard-requires mlc, and a machine without
the mlc package installed degrades honestly rather than crashing. Its NON-python assets (the prebuilt
``circt-opt`` binary, the cached ``runs/circt-arc/<target>`` arc outputs, schemas) still resolve by
directory path via ``MERLIN_MLC_DIR``.

Why this matters: a header parse and an RTL comparison set can disagree. Recording
the selected decoder field exposes that discrepancy; neither alone proves an
exhaustive executable ISA or authorizes a code-generation endpoint.

TARGET-AGNOSTIC: every entry point here takes a ``target`` argument and holds no target name. mlc is
target-parameterized (``artifact_paths(target)`` / ``discovered_memory_map(target)`` /
``discover_opcode_set(graph)`` / per-target ``runs/circt-arc/<target>``), so the same code plugs any HW
RTL repo mlc knows (systolic arrays, NPUs, SIMT cores, CPUs, ...).
"""

from __future__ import annotations

import copy
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common import external_tools as _modelir

# ---- mlc_bridge --------------------------------------------------------------------------------


def discovered_memories(target: str) -> list[dict] | None:
    """The target's SRAM banks (name/depth/row_bytes), DISCOVERED from the RTL by mlc. Target-agnostic."""
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir
    from merlin.target.rtl.extraction.source_selection import active_selection

    if active_selection(target) is not None:
        return None
    if mlc_dir() is None:
        return None
    try:
        with _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import load_interface

            return list(load_interface(target).get("memories", []))
    except Exception:  # noqa: BLE001
        return None


def discovered_capacities(target: str) -> dict | None:
    """Total operand-scratchpad + accumulator capacity (bytes) DERIVED from mlc discovery — sum the
    sibling banks of the representative operand/accumulator memory named in the discovered ``memory_map``
    (banks share the name with the bank index stripped). Target-agnostic: no hardcoded bank names.
    Returns ``{operand_bytes, accumulator_bytes, operand_depth, accumulator_depth}`` (values None when
    not derivable)."""
    from merlin.target.rtl.extraction.mlc_roles import discovered_memory_map

    mm = discovered_memory_map(target)
    mems = discovered_memories(target)
    if not mm or not mems:
        return None

    def _bank_prefix(rep_name: str | None) -> str | None:
        # sibling banks share the name with the bank-index segment ('<base>_<int>') truncated to
        # '<base>_' — structural split, no regex (e.g. 'spad/spad_mems_0/mem_ext' -> 'spad/spad_mems_').
        if not rep_name:
            return None
        segs = rep_name.split("/")
        for i, s in enumerate(segs):
            base, sep, num = s.rpartition("_")
            if sep and num.isdigit():
                return "/".join(segs[:i] + [base + "_"])
        return None

    def _group(rep_name: str | None):
        prefix = _bank_prefix(rep_name)
        if not prefix:
            return None, None
        banks = [x for x in mems if x.get("name", "").startswith(prefix)]
        if not banks:
            return None, None
        total = sum(b["depth"] * b["row_bytes"] for b in banks)
        depth = banks[0]["depth"] if len({b["depth"] for b in banks}) == 1 else None
        return total, depth

    op_bytes, op_depth = _group(mm.get("operand_mem"))
    acc_bytes, acc_depth = _group(mm.get("accum_mem"))
    return {
        "operand_bytes": op_bytes,
        "accumulator_bytes": acc_bytes,
        "operand_depth": op_depth,
        "accumulator_depth": acc_depth,
    }


def target_fact_bundle(target: str) -> dict:
    """The STATIC, provenance-tagged fact bundle DERIVED from a target's CIRCT HW dialect — no model run.

    This is the groundable, non-labeling half of extraction, and it serves BOTH consumers: it is handed
    to the agent as information (the legal ISA it must target, the mesh geometry, the on-chip capacities
    it must respect) AND it is the source of the cheap FileCheck literals. Each field carries provenance
    {value, source, derived}; a field mlc cannot ground is reported derived=False (value None), never
    guessed — so a target with no HW dialect (SIMT/prototype) yields an honest all-unavailable bundle
    rather than a fabricated one. The opcode->role LABELING is deliberately NOT here (see
    semantic_roles / the cross-module analysis it needs); this bundle is exactly what is derivable today.
    """

    from merlin.target.rtl.extraction.mlc_roles import discovered_dim, discovered_memory_map
    from merlin.target.rtl.extraction.mlc_toolchain import discover_legal_opcodes

    def _try(fn, *a):
        try:
            return fn(*a)
        except Exception:  # noqa: BLE001 — a field that can't be grounded is honest-unavailable
            return None

    legal = _try(discover_legal_opcodes, target) or {}
    dim = _try(discovered_dim, target)
    mm = _try(discovered_memory_map, target)
    caps = _try(discovered_capacities, target)
    opcodes = legal.get("legal_opcodes")
    fields = {
        "legal_opcodes": {
            "value": opcodes,
            "derived": opcodes is not None,
            "source": legal.get("method"),
            "evidence": legal.get("evidence"),
        },
        "mesh_dim": {"value": dim, "derived": dim is not None, "source": "mlc.discover.geometry.discover_mesh_dim"},
        "memory_map": {"value": mm, "derived": mm is not None, "source": "mlc discovered memory_map"},
        "capacities": {
            "value": caps,
            "derived": bool(caps and caps.get("operand_bytes")),
            "source": "sibling-bank sum over discovered memory_map",
        },
    }
    return {
        "target": target,
        "method": "static CIRCT HW-dialect discovery (no model run)",
        "fields": fields,
        "n_derived": sum(1 for f in fields.values() if f["derived"]),
    }


def render_fact_bundle(target: str, bundle: dict | None = None) -> str:
    """Render :func:`target_fact_bundle` as an agent-facing ISA brief (Markdown). This is the information
    handed to the agent so it can target the real ISA/geometry/capacities — every line is DERIVED and
    stamped with its provenance, and an ungrounded field is stated as unavailable, never invented."""
    b = bundle or target_fact_bundle(target)
    f = b["fields"]
    # Robust to non-systolic bundle shapes: a simt bundle (mlc_bridge._simt_fact_bundle) has no
    # legal_opcodes/mesh_dim/capacities keys, so a missing OR underived field renders as "unavailable"
    # rather than raising. A systolic target has all three present -> byte-identical brief.
    # TODO(simt-render): a dedicated SIMT/vector renderer so a simt target's brief surfaces its
    # registers/shared-memory/fp-datapath facts instead of this honest-degraded all-"unavailable" line.
    lo_f = f.get("legal_opcodes") or {}
    md_f = f.get("mesh_dim") or {}
    cap_f = f.get("capacities") or {}
    lo = lo_f.get("value")
    lines = [
        f"# Target ISA facts: {b['target']}",
        f"_Derived by {b['method']}. {b['n_derived']}/4 fields grounded; ungrounded = unavailable, not guessed._",
        "",
    ]
    if lo_f.get("derived") and lo is not None:
        lines += [f"- **Legal opcodes** ({len(lo)}): `{lo}`", f"  - source: {lo_f.get('source')}"]
    else:
        lines.append("- **Legal opcodes**: unavailable (no decoder signal / no HW dialect for this target)")
    lines.append(f"- **Mesh DIM**: {md_f['value'] if md_f.get('derived') else 'unavailable'}")
    caps = cap_f.get("value") if cap_f.get("derived") else None
    if caps:
        lines.append(
            f"- **On-chip capacity (bytes)**: operand={caps.get('operand_bytes')}, "
            f"accumulator={caps.get('accumulator_bytes')}"
        )
    else:
        lines.append("- **On-chip capacity**: unavailable")
    return "\n".join(lines) + "\n"


_SPATIAL_INTROSPECTS: dict[str, object] = {}


def register_spatial_introspect(module) -> None:
    """Register a spatial tensor-tile reader for every arc model it names in ``ARC_TARGETS``. Idempotent."""
    for arc in getattr(module, "ARC_TARGETS", ()):
        _SPATIAL_INTROSPECTS[arc] = module


def _register_declared_spatial_introspects() -> None:
    """Register every spatial reader a target contract declares (``plugin.spatial_introspect``).

    The reader is the target's data, loaded from its own support provider, so core names no unit. A
    declaration that does not load registers nothing and the fact bundle says so.
    """
    try:
        from merlin.target.registry import identity as _tr
        from merlin.target.registry.plugins import load_declared

        names = _tr.all_targets()
    except Exception:  # noqa: BLE001 — no registry, no declared reader; the bundle reports it
        return
    for name in names:
        try:
            if not _tr.resolve(name).plugin().get("spatial_introspect"):
                continue
            register_spatial_introspect(load_declared(name, "spatial_introspect"))
        except Exception:  # noqa: BLE001 — unresolvable declaration: register nothing, report honestly
            continue


def resolve_spatial_introspect(target: str):
    """The declared spatial reader serving ``target``'s arc model, or None."""
    from merlin.target.rtl.extraction.mlc_toolchain import _arc_target

    if not _SPATIAL_INTROSPECTS:
        _register_declared_spatial_introspects()
    return _SPATIAL_INTROSPECTS.get(_arc_target(target))


_SPATIAL_FIELDS = ("tile_dim", "mrf_depth", "element_widths", "dtypes", "fma_latency", "op_categories", "accum_kind")


def _spatial_unavailable(target: str, reason: str) -> dict:
    """An honest all-unavailable spatial bundle — never a fabricated tile."""
    return {
        "target": target,
        "method": "declared spatial state-manifest reader (no model run)",
        "kind": "spatial",
        "fields": {k: {"value": None, "derived": False, "source": None, "evidence": reason} for k in _SPATIAL_FIELDS},
        "n_derived": 0,
        "reason": reason,
    }


def spatial_fact_bundle(target: str) -> dict:
    """The STATIC, provenance-tagged SPATIAL tensor-tile fact bundle for ``target`` from its declared
    reader, or an all-unavailable bundle naming why. Same ``{target, method, fields}`` shape as the
    systolic :func:`target_fact_bundle`."""
    reader = resolve_spatial_introspect(target)
    if reader is None:
        return _spatial_unavailable(target, f"no declared spatial reader serves {target!r}")
    return reader.build_fact_bundle(target)


def render_spatial_fact_bundle(target: str, bundle: dict | None = None) -> str:
    """Render :func:`spatial_fact_bundle` as an agent-facing spatial-tile brief (Markdown)."""
    reader = resolve_spatial_introspect(target)
    if reader is None:
        b = bundle or spatial_fact_bundle(target)
        return f"## Spatial tile facts: {target}\n\nUnavailable: {b.get('reason')}\n"
    return reader.render_fact_bundle(target, bundle)


def spatial_facts(target: str) -> dict:
    """The spatial facts artifact for ``target`` from its declared reader; fails closed without one."""
    reader = resolve_spatial_introspect(target)
    if reader is None:
        raise RuntimeError(f"no declared spatial reader serves {target!r}; select its support provider")
    return reader.spatial_facts(target)


def _arc_model_kind(target: str) -> str | None:
    """The kind the arc-model registry declares for a model merlin reaches under ``target``'s own name.

    Consulted only for a target with no capability manifest (an arc-only target, whose facts come from
    the arc artifacts rather than an in-tree contract), so fact_bundle_for can route it by KIND without a
    per-target branch. None when the registry lists no kind for it."""
    from merlin.target.rtl.extraction.mlc_toolchain import arc_model

    return arc_model(target).get("kind")


class KindUnresolved(RuntimeError):
    """``target``'s compute-unit kind could not be resolved because its MANIFEST FAILED TO LOAD.

    Not the same state as "this target has no manifest", and the difference is the whole point. A
    target with no contract is a fresh or arc-only target: nothing has declared its class yet, the arc
    registry is asked, and a caller that still gets nothing may fall back to the generic static reader.
    A target whose contract EXISTS and does not load is a target whose declaration is broken -- and
    until this existed both landed in the same bare ``except Exception`` and came back as ``kind=None``,
    which every caller read as "no declaration" and routed to the systolic extractor.

    Measured, live, in this tree: ``k1_cpu``'s contract declares ``int16``/``int32`` on its compute
    units, the quant-format registry knew neither, ``compute_units._unit`` raised, the raise was
    swallowed, and a general-purpose vector CPU with no RoCC mesh anywhere in it was extracted by the
    systolic reader -- silently, for every consumer, with no record that a declaration had failed. A
    class routed by a swallowed exception is not routed; it is defaulted.
    """


def _kinds_from_manifest(target: str) -> tuple[str, ...] | None:
    """``target``'s declared compute-unit kinds (primary first), or ``None`` when it has NO manifest.

    Raises :class:`KindUnresolved` when a manifest exists and could not be turned into a kind set: the
    contract is there, it says something, and what it says could not be read. Refusing is the only
    honest answer -- the alternative on that path is to pick a class for silicon nobody has described.

    ``FileNotFoundError`` (which :class:`~merlin.target.registry.identity.TargetContractMissing`
    subclasses) is the structural test for "no contract", so absence is distinguished from breakage by
    the registry's own answer rather than by inspecting a message.
    """
    from merlin.target.capabilities import compute_units as _cu
    from merlin.target.registry.experiment_descriptor import _primary_kind, load_capability_manifest

    try:
        contract = load_capability_manifest(target).contract
    except FileNotFoundError:
        return None  # no contract at all: a fresh / arc-only target, not a broken declaration
    except Exception as exc:  # noqa: BLE001 — a contract that exists and does not load is a REFUSAL
        raise KindUnresolved(
            f"{target!r}: a capability contract exists but its compute-unit kind could not be "
            f"resolved ({type(exc).__name__}: {exc}). REFUSING rather than defaulting: routing an "
            f"unresolved kind to a fact extractor picks a datapath class for silicon whose own "
            f"declaration did not load, and the resulting artifact is indistinguishable from one "
            f"that was extracted correctly."
        ) from exc
    try:
        units = _cu.compute_units(contract)
    except Exception as exc:  # noqa: BLE001 — same refusal, for a contract whose units do not parse
        raise KindUnresolved(
            f"{target!r}: its capability contract declares compute_units that do not parse "
            f"({type(exc).__name__}: {exc}). REFUSING rather than defaulting to a class."
        ) from exc
    if not units:
        # A DECLARED absence, not a failure: the neutral reference / skeleton contracts genuinely
        # declare no compute unit, and `families.contract_endpoint_kind` already reads that as None.
        return ()
    return tuple(dict.fromkeys([_primary_kind(units)] + [u.kind for u in units]))


def _resolve_kind(target: str) -> str | None:
    """The PRIMARY compute-unit ``kind`` for ``target``: its capability manifest's kind when one is
    committed/regenerated, else the kind the arc-model registry declares for it, else None (no
    declaration anywhere). Never fabricates a kind, and never converts a FAILED resolution into
    ``None`` — that raises :class:`KindUnresolved`; see :func:`_kinds_from_manifest`."""
    kinds = _resolve_kinds(target)
    return kinds[0] if kinds else None


def fact_bundle_for(target: str) -> dict:
    """KIND-routed RTL fact bundle dispatch — the seam that keeps fact extraction target-name-free.

    Resolves ``target``'s compute-unit kind and routes to the fact-extraction family named by that kind's
    :class:`~merlin.target.capabilities.families.FamilyProfile.fact_extractor`:

      * ``circt_static`` (systolic/vector/scalar, and the default when no kind resolves) -> the existing
        CIRCT HW-dialect static bundle :func:`target_fact_bundle` (BYTE-IDENTICAL to the pre-dispatch
        path for every current caller);
      * ``simt_config`` (simt) -> the SIMT config+FIRRTL introspect, adapted to the bundle shape;
      * ``spatial_manifest`` -> the target's declared spatial state-manifest reader
        :func:`spatial_fact_bundle`.

    All branches return the uniform ``{target, method, fields, n_derived}`` shape, so consumers
    (onboarding, prompt rendering) read ``n_derived`` uniformly. A kind nothing DECLARES degrades to the
    systolic static path — exactly the pre-existing behavior. A kind that fails to RESOLVE does not:
    that raises :class:`KindUnresolved` rather than picking a datapath class for silicon whose own
    declaration did not load.

    MEMOIZED in-memory per target: the extraction (circt-opt on the target's ``hw.mlir`` / mlc discovery)
    is deterministic for a given target within a process but costs ~seconds, so it ran redundantly on
    every call (dominating test time — re-discovery per test). It now runs ONCE per target and later
    calls return a deep COPY (mutation-safe: no caller can poison the cache). The cache lives only in
    process memory — nothing is written to disk, so nothing is committed. Call
    :func:`clear_fact_bundle_cache` if a process regenerates a target's RTL facts and must re-read."""
    if target not in _FACT_BUNDLE_CACHE:
        _FACT_BUNDLE_CACHE[target] = _fact_bundle_uncached(target)
    return copy.deepcopy(_FACT_BUNDLE_CACHE[target])


_FACT_BUNDLE_CACHE: dict[str, dict] = {}


def clear_fact_bundle_cache() -> None:
    """Drop the in-memory fact-bundle memo (use after regenerating a target's RTL facts in-process)."""
    _FACT_BUNDLE_CACHE.clear()


def _resolve_kinds(target: str) -> tuple[str, ...]:
    """EVERY compute-unit kind the target declares, primary first, deduped.

    The set, not a primary. A hybrid runs one extractor per datapath family, and selecting a single
    kind extracted facts for half its silicon: a SIMT cluster embedding a systolic mesh resolved to
    ``simt``, ran the config introspect, and never looked at the mesh -- so the mesh geometry, the
    on-chip capacities and the legal opcodes were all absent for hardware that has them.

    Primary first so single-unit targets are unchanged and a hybrid's headline kind is still its
    outermost unit.

    Raises :class:`KindUnresolved` when a manifest EXISTS and fails to resolve. ``()`` keeps its old
    meaning -- nothing declares a kind for this target -- and is now reachable only from a declared
    absence, never from a swallowed exception."""
    kinds = _kinds_from_manifest(target)
    if not kinds:
        # No contract at all, or a contract that declares no compute unit. Either way nothing in-tree
        # has stated this target's class, so the arc-model registry is the next declaration to ask --
        # the same order as before, with the swallowed-exception path removed from it.
        k = _arc_model_kind(target)
        return (k,) if k else ()
    return kinds


def compute_unit_kinds(target: str) -> tuple[str, ...]:
    """EVERY compute-unit kind ``target`` declares, primary first — the public reading of the kind set.

    The routing axis for anything that must serve a hybrid honestly (fact extraction, the compute-element
    read): a caller given only the PRIMARY kind extracts facts for half the silicon. Empty when no kind
    resolves, which is a caller's cue to say so rather than to assume one."""
    return _resolve_kinds(target)


def _extractors_for(target: str) -> tuple[str, ...]:
    """The fact extractors this target needs, in kind order, deduped.

    ``("circt_static",)`` when nothing DECLARES a kind — the documented default for a target nobody has
    classified. It propagates :class:`KindUnresolved` rather than returning that default when a
    declaration exists and failed to load: the two states demand opposite responses, and collapsing
    them is what routed a vector CPU through the systolic reader."""
    from merlin.target.capabilities.families import family_profile, known_kinds

    kinds = _resolve_kinds(target)
    if not kinds:
        return ("circt_static",)
    out = [family_profile(k).fact_extractor if k in known_kinds() else "circt_static" for k in kinds]
    return tuple(dict.fromkeys(out))


def _run_extractor(name: str, target: str) -> dict:
    if name == "spatial_manifest":
        return spatial_fact_bundle(target)
    if name == "simt_config":
        return _simt_fact_bundle(target)
    return target_fact_bundle(target)


def _merge_fact_bundles(bundles: list[dict], extractors: tuple[str, ...] = ()) -> dict:
    """Union several extractors' bundles into one ``{target, method, fields, n_derived}``.

    Field collisions are resolved by EVIDENCE, never by extractor precedence: a derived value beats an
    undrived one, and two extractors that both derive a field and DISAGREE keep both readings under
    ``conflicts`` rather than letting the first-listed datapath silently win. A hybrid whose two halves
    report different mesh geometries has a real problem, and a merge that hides it converts that problem
    into a wrong number nobody can trace."""
    head = bundles[0]
    fields: dict = {}
    conflicts: list[dict] = []
    for b in bundles:
        for name, rec in (b.get("fields") or {}).items():
            prev = fields.get(name)
            if prev is None or (not prev.get("derived") and rec.get("derived")):
                fields[name] = rec
                continue
            if prev.get("derived") and rec.get("derived") and prev.get("value") != rec.get("value"):
                conflicts.append({"field": name, "kept": prev, "also": rec})
    out = {
        "target": head.get("target"),
        "method": " + ".join(dict.fromkeys(str(b.get("method")) for b in bundles if b.get("method"))),
        "kind": head.get("kind"),
        "fields": fields,
        "n_derived": sum(1 for r in fields.values() if r.get("derived")),
    }
    if len(bundles) > 1:
        # Only stamped on a genuine hybrid, so a single-datapath bundle stays byte-identical. Taken
        # from the extractors that actually ran, not from each bundle's `kind` -- the static CIRCT
        # bundle does not carry one, so reading it back yields None for the half that has no label.
        out["extractors"] = list(extractors) or [str(b.get("kind")) for b in bundles]
    if conflicts:
        out["conflicts"] = conflicts
    return out


def _fact_bundle_uncached(target: str) -> dict:
    """The un-memoized extraction (see :func:`fact_bundle_for`)."""
    extractors = _extractors_for(target)
    if len(extractors) == 1:
        return _run_extractor(extractors[0], target)
    return _merge_fact_bundles([_run_extractor(e, target) for e in extractors], extractors)


def render_simt_fact_bundle(target: str, bundle: dict | None = None) -> str:
    """Render a SIMT fact bundle as an agent-facing ISA brief (Markdown).

    The systolic renderer asks three systolic-shaped questions — legal opcodes (a RoCC funct decode), mesh
    DIM, operand/accumulator capacity — none of which a SIMT core has. Routing a SIMT target through it
    printed "unavailable" three times and a nonsensical "5/4 fields grounded" header while the bundle
    held every fact the agent needs (execution geometry, register budget, shared-memory capacity, FP
    datapath, and the core's own instruction encoding + classes). An arm told its ISA facts are
    unavailable cannot derive its lowering from them, which is the whole point of the RTL-checks arm.

    Every line is DERIVED from the bundle and carries its provenance; an underived field is stated as
    unavailable, never invented. Field-driven, so a SIMT introspect that grows a fact surfaces it here
    without editing this function.
    """
    b = bundle or _simt_fact_bundle(target)
    f = b.get("fields") or {}
    n_total = len(f) or 1
    lines = [
        f"# Target ISA facts: {b.get('target', target)}",
        f"_Derived by {b.get('method')}. {b.get('n_derived', 0)}/{n_total} fields grounded; "
        f"ungrounded = unavailable, not guessed._",
        "",
    ]
    if b.get("rtl_present") is False:
        lines += [
            "> RTL module hierarchy was not present for this extraction: geometry comes from the "
            "target's own config + ISA docs. Treat the encoding/classes as authoritative and the "
            "geometry as declared-not-elaborated.",
            "",
        ]

    #: (bundle field, heading, the sub-keys worth surfacing) — the SIMT analog of the systolic triple.
    _SHAPE = (
        ("simt", "Execution geometry", ("lanes_per_warp", "warps_per_core", "cores", "threads_per_core")),
        ("registers", "Register budget", ("arch_max", "compiler_limit", "config")),
        ("shared_memory", "Shared memory", ("bytes_per_cluster",)),
        ("fp_datapath", "FP datapath", ("dtype", "flop_per_fma", "peak_flops_per_cycle", "clock_hz", "peak_gflops")),
        (
            "isa",
            "Instruction encoding",
            ("encoding_bits", "max_src_operands", "max_dst_operands", "predicated_execution", "address_spaces"),
        ),
    )
    for key, heading, subkeys in _SHAPE:
        spec = f.get(key) or {}
        val = spec.get("value")
        if not spec.get("derived") or not isinstance(val, dict):
            lines.append(f"- **{heading}**: unavailable")
            continue
        shown = ", ".join(f"{k}={val[k]}" for k in subkeys if val.get(k) is not None)
        lines.append(f"- **{heading}**: {shown or 'derived (no scalar fields)'}")
        if key == "isa":
            classes = list(val.get("instruction_classes") or [])
            if classes:
                lines.append(f"  - **instruction classes** ({len(classes)}): `{classes}`")
        ev = val.get("evidence") or spec.get("evidence")
        if ev:
            lines.append(f"  - source: {spec.get('source')} — {ev}")
    return "\n".join(lines) + "\n"


def render_fact_bundle_for(target: str, bundle: dict | None = None) -> str:
    """KIND-routed sibling of :func:`fact_bundle_for` for RENDERING — the render half of the same seam.

    Routes ``target``'s kind to the matching bundle renderer: ``opu`` (spatial) ->
    :func:`render_spatial_fact_bundle`; ``simt`` -> :func:`render_simt_fact_bundle`; everything else
    (systolic/vector/scalar ``circt_static``, and the default when no kind resolves) ->
    :func:`render_fact_bundle`. BYTE-IDENTICAL to :func:`render_fact_bundle` for every current
    ``circt_static`` caller — the same reasoning as ``fact_bundle_for``: a systolic target resolves
    ``kind='systolic'`` -> ``fact_extractor='circt_static'`` -> the ``return render_fact_bundle(...)``
    fall-through, on the same ``bundle`` object."""
    # Routes on the extractor SET, so a hybrid that has a spatial tile still gets the tile brief; `kind`
    # is resolved separately for the SIMT branch below, which is a kind and not an extractor.
    kind = _resolve_kind(target)
    if "opu" in _extractors_for(target):
        return render_spatial_fact_bundle(target, bundle)
    if kind == "simt":
        return render_simt_fact_bundle(target, bundle)
    return render_fact_bundle(target, bundle)


_SIMT_INTROSPECTS: dict = {}


def register_simt_introspect(module) -> None:
    """Register a SIMT RTL introspect (must expose ``TARGET`` + ``build_facts()``). Idempotent."""
    _SIMT_INTROSPECTS[module.TARGET] = module


def _register_declared_simt_introspects() -> None:
    """Register every SIMT introspect a target contract DECLARES, without naming any target here.

    A target contributes its introspect as data: ``plugin.simt_introspect: <plugin.backend>:<attribute>``
    names an attribute of its own registered backend package, resolved through the backend registry, so
    nothing is imported a second time under another name. This keeps the on-demand guarantee the old
    hardcoded fallback gave (it fires whenever the registry is empty, independent of discovery order, so
    the SIMT introspection a composite target relies on cannot silently go missing) while core carries no
    SIMT target's name. A declaration that does not resolve registers nothing and the fact bundle says so.
    """
    try:
        from merlin.runtime.backends import registry as _bk
        from merlin.target.registry import identity as _tr
        from merlin.target.registry.plugins import ATTR_SEP

        names = _tr.all_targets()
    except Exception:  # noqa: BLE001 — no registry, no declared introspect; _simt_fact_bundle reports it
        return
    for name in names:
        try:
            plugin = _tr.resolve(name).plugin()
        except Exception:  # noqa: BLE001 — a contract that will not parse declares nothing usable
            continue
        ref = plugin.get("simt_introspect")
        if not isinstance(ref, str) or ATTR_SEP not in ref:
            continue
        module_ref, _, attr = ref.partition(ATTR_SEP)
        if module_ref != plugin.get("backend"):
            continue  # only the target's own registered backend package is consulted
        try:
            register_simt_introspect(getattr(_bk.get_backend(name), attr))
        except Exception:  # noqa: BLE001 — unresolvable declaration: register nothing, report honestly
            continue


def _resolve_simt_introspect(target: str):
    """The registered SIMT introspect whose declared identity matches ``target``'s arc alias, or None.
    Serves by the ARC-target alias (not the merlin name): a composite SIMT target whose
    RTL IS the introspect's config (its cluster) resolves to that introspect via ``_arc_target``."""
    from merlin.target.rtl.extraction.mlc_toolchain import _arc_target

    if not _SIMT_INTROSPECTS:
        _register_declared_simt_introspects()
    return _SIMT_INTROSPECTS.get(_arc_target(target))


_DERIVED_STATE = "derived"


def _simt_field(name: str, block: dict | None, source: str | None) -> dict:
    """Adapt ONE fact block from a SIMT introspect to the bundle field shape.

    ``derived`` is read from the block's own DECLARED ``state``, never from the block merely existing.
    That distinction is the bug this replaced: the old expression was ``"derived": name in f``, so a
    block of baked default values — the Muon introspect published ``lanes_per_warp=16,
    warps_per_core=8, cores=2, threads_per_core=128`` from ``cfg.get("num_lanes", 16)`` defaults after
    its config path resolved to a placeholder that exists on no machine — arrived here stamped
    ``derived: True`` and was rendered to the agent as a grounded RTL fact. Key presence proves the
    extractor RAN; it proves nothing about whether it READ anything.

    An introspect that declares no ``state`` is treated as NOT derived and says so, rather than being
    granted the benefit of the doubt: fail closed, per the repo's derive-or-report rule.
    """
    if not isinstance(block, dict):
        return {
            "value": None,
            "derived": False,
            "source": source,
            "evidence": f"introspect emitted no {name!r} fact block",
        }
    state = block.get("state")
    if state is None:
        return {
            "value": block,
            "derived": False,
            "source": source,
            "evidence": (
                f"{name!r} block declares no `state` — cannot be counted as derived "
                f"(introspect contract: state in derived/absent/undeterminable). "
                f"{block.get('evidence') or ''}"
            ).strip(),
        }
    return {
        "value": block,
        "derived": state == _DERIVED_STATE,
        "state": state,
        "source": source,
        "evidence": block.get("evidence"),
    }


def _simt_fact_bundle(target: str) -> dict:
    """Adapt the SIMT RTL introspect to the uniform fact-bundle shape. The introspect is resolved from
    the registry by the target's declared identity, so a SIMT target no registered introspect serves is
    reported honestly rather than mis-attributed to another core's geometry. Keys on that declared
    identity, never a literal target name."""
    intro = _resolve_simt_introspect(target)
    if intro is None:
        served = sorted(_SIMT_INTROSPECTS)
        fields = {
            "simt": {
                "value": None,
                "derived": False,
                "source": None,
                "evidence": f"no SIMT RTL introspect serves {target!r} (registered: {served})",
            }
        }
        return {
            "target": target,
            "method": f"SIMT RTL introspect (registered: {served})",
            "kind": "simt",
            "fields": fields,
            "n_derived": 0,
        }
    facts = intro.build_facts()
    f = facts.get("facts", {})
    present = facts.get("inputs", {}).get("rtl_present", False)
    fields = {
        name: _simt_field(name, f.get(name), facts.get("generator", {}).get("name"))
        for name in ("simt", "registers", "shared_memory", "fp_datapath", "isa")
    }
    return {
        "target": target,
        "method": facts.get("generator", {}).get("method", f"{getattr(intro, 'TARGET', 'SIMT')} RTL introspect"),
        "kind": "simt",
        "rtl_present": present,
        "fields": fields,
        "n_derived": sum(1 for v in fields.values() if v["derived"]),
    }


def simt_facts(target: str) -> dict:
    """SIMT self-hosted-ISA facts adapted to the ``facts.json`` body shape, so the generic manifest
    deriver (:func:`merlin.target.capabilities.manifest.derive_manifest`) grounds ``endpoint_kind``
    from them like any other RTL facts. Delegates to the registered SIMT RTL introspect (via
    the arc-target alias); the core's OWN instruction encoding is surfaced as a ``self_hosted_isa``
    interface (its ``encoding_bits`` + ``instruction_classes``) — the SIMT analog of a decode table.
    Returns ``{}`` when no SIMT introspect serves the target (honest — the deriver then falls back to the
    family default, never a fabricated endpoint)."""
    bundle = _simt_fact_bundle(target)
    fields = bundle.get("fields") or {}

    def _derived(name: str) -> dict:
        """The field's value ONLY when the introspect declared it derived — otherwise ``{}``.

        Guards the manifest deriver against the failure this function was itself party to: it used to
        read every field's value unconditionally, so a bundle whose fields were baked defaults (see
        :func:`_simt_field`) grounded ``capabilities.simt`` and the SMEM capacity in the manifest,
        which is how a placeholder config path ended up as a published hardware capability."""
        rec = fields.get(name) or {}
        return (rec.get("value") or {}) if rec.get("derived") else {}

    isa = _derived("isa")
    if not isa.get("encoding_bits"):
        return {}
    itf = {
        "name": "self_hosted_isa",
        "encoding_bits": isa.get("encoding_bits"),
        "instruction_classes": list(isa.get("instruction_classes") or []),
        "source": bundle.get("method"),
    }
    body: dict = {"interfaces": [itf], "source": bundle.get("method"), "target": target}
    # SIMT execution geometry (lanes/warp, warps, cores) + shared-memory CAPACITY, read out of the
    # ELABORATION by the introspect — surfaced so the manifest deriver grounds capabilities.simt / the
    # SMEM capacity instead of a residual literal. No memory-map BASE here (not recoverable from the op
    # graph — stays residue). Taken through _derived(), so a block the introspect could not read is
    # simply absent from the manifest rather than grounding a capability on a number nobody measured.
    simt = _derived("simt")
    if isinstance(simt.get("lanes_per_warp"), int):
        body["simt"] = {
            k: simt[k] for k in ("lanes_per_warp", "warps_per_core", "cores") if isinstance(simt.get(k), int)
        }
    smem = _derived("shared_memory")
    if isinstance(smem.get("bytes_per_cluster"), int):
        body["memories"] = [{"name": "shared_memory", "bytes": smem["bytes_per_cluster"]}]
    return {"schema_version": "simt-facts/v0", "facts": body}


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        description="Emit a target's DERIVED ISA fact bundle (agent info + "
        "FileCheck source). Prints Markdown; --json prints the raw bundle."
    )
    ap.add_argument("target")
    ap.add_argument("--json", action="store_true", help="print the raw provenance-tagged bundle as JSON")
    a = ap.parse_args(argv)
    b = target_fact_bundle(a.target)
    print(_json.dumps(b, indent=2) if a.json else render_fact_bundle(a.target, b), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


@contextmanager
def _mlc_cwd():
    """Keep configured-root lookup inside the shared artifact lock."""
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir

    with _modelir.artifact_context(resolve_root=mlc_dir):
        yield


def arc_available(target: str) -> bool:
    """True iff mlc has a prebuilt arc model (.so/.o) + state manifest for ``target`` (any target)."""
    from merlin.target.rtl.extraction.mlc_toolchain import _arc_target, mlc_dir

    if mlc_dir() is None:
        return False
    try:
        with _mlc_cwd():
            from mlc.runtime.backend import available

            return bool(available(_arc_target(target)))
    except Exception:  # noqa: BLE001
        return False


def _arc_model_present(target: str) -> bool:
    """Un-aliased structural check: does mlc register an arc model under THIS EXACT target name? Structural
    consumers (the RoCC role / mesh probes) gate on this, NOT the oracle-aliased :func:`arc_available` — a
    composite SIMT target must not borrow the embedding cluster's arc for structural role
    derivation (that would fabricate systolic roles it does not have); its oracle still reaches the SIMT core's model
    via the alias, but its structural profile stays honestly empty."""
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir

    if mlc_dir() is None:
        return False
    try:
        with _mlc_cwd():
            from mlc.runtime.backend import available

            return bool(available(target))
    except Exception:  # noqa: BLE001
        return False


def _ensure_interface_cache(target: str) -> None:
    from mlc.discover.cache import discovered_memory_map, dump_cache

    try:
        discovered_memory_map(target)
    except Exception:  # noqa: BLE001 — build the fingerprint-gated discovery cache on demand
        dump_cache(target)


def arc_core(target: str):
    """The target-AGNOSTIC ctypes arc model (mlc ``CosimCore``) for ``target``: poke/peek/step any signal
    by NAME. Works for every target mlc compiled from RTL — the compile-from-RTL oracle primitive. The
    high-level per-op driver (matmul-WS, SIMT, ...) is target-specific and provided by mlc; use
    :func:`arc_run_command_buffer` for the agnostic high-level path."""
    from merlin.target.rtl.extraction.mlc_toolchain import _arc_target, mlc_dir, require_mlc

    require_mlc()
    if not arc_available(target):
        raise RuntimeError(f"mlc arc model absent for target {target!r} (runs/circt-arc/{_arc_target(target)})")
    at = _arc_target(target)
    with _mlc_cwd():
        _ensure_interface_cache(at)
        from mlc.backends.cosim_core import CosimCore
        from mlc.discover.fingerprint import artifact_paths

        p = artifact_paths(at)
        d = mlc_dir()
        return CosimCore(str((d / p["so"]).resolve()), str((d / p["man"]).resolve()))


def arc_run_command_buffer(cb: dict, target: str | None = None) -> dict:
    """Answer a merlin command buffer on the RTL-derived arc model for ``target``. Returns mlc's backend
    contract ``{outputs, metrics, correct, oracle, ...}`` where ``metrics`` carries the RTL's internal
    counts (cycles, bytes_moved, resident_hits, accumulator_commits, ...) — verilator-level state without
    verilator, for ANY target.

    The target is PASSED, not inferred. mlc's ``run_command_buffer`` routes on its ``target`` argument and
    defaults it to one specific accelerator, so omitting it silently answered every other target's buffer
    on the wrong model: a saturn_opu buffer reached the systolic cosim and died on its
    dims-divisible-by-DIM assertion. Falls back to the buffer's own ``target`` field, and raises rather
    than letting a default stand in.
    """
    from merlin.target.rtl.extraction.mlc_toolchain import _arc_target, require_mlc

    require_mlc()
    t = target or cb.get("target")
    if not t:
        raise ValueError(
            "arc_run_command_buffer needs a target: pass one, or declare it in the "
            "command buffer — the oracle must never fall back to a default model"
        )
    with _mlc_cwd():
        from mlc.runtime.backend import run_command_buffer

        return run_command_buffer(cb, target=_arc_target(str(t)))
