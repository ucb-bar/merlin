"""Target-AGNOSTIC capability-manifest deriver + per-target residual discovery.

A capability manifest is a target **definition** (a ``target_contract.yaml`` with ``compute_units``)
generated as an ``out/`` artifact and plugged into the routing tooling — the same shape a real
out-of-tree target repo ships. There are NO per-target manifest dicts baked into this core module:
every manifest is reconstructed by :func:`derive_manifest` from three sources —

  * **CIRCT FACTS** (``facts.json``): mesh / memory capacities / datapath dtypes / observed decode fields,
  * **FAMILY defaults** (:func:`merlin.targetgen.families.family_profile`, keyed by compute-unit kind):
    non-authorizing runner hints when no executable endpoint is established,
  * a small **RESIDUAL** side-input (intent + prose RTL cannot ground) that lives WITH the target
    package at ``<target_base>/contracts/residual.yaml`` — the discovered target dir, never a Python
    literal here.

Discovery (:func:`discovered_targets`) scans ``artifacts/targets/*/contracts/residual.yaml``, and
``manifest_for(name)`` runs the single agnostic derive path over each. A new accelerator brings itself
up by dropping a descriptor + a residual and letting mlc extract facts — with zero edits to core.

The shipped residuals include ``rvv``, ``mx_gemmini``, ``radiance`` and ``atlas``.
An observed Atlas decoder field does not by itself establish its encoding or endpoint.
All are provenance-tagged prototypes flagged
``requires_human_review`` — NOT RTL-certified.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from merlin.common import quant_formats as _qf
from merlin.common import schemas as _schemas
from merlin.common.paths import artifacts_dir as _artifacts_dir
from merlin.common.yaml import write_yaml
from merlin.targetgen import compute_units as _cu
from merlin.targetgen import families as _families
from merlin.targetgen.rtl.facts import target_base as _target_base

_COMMON: dict[str, Any] = {
    "version": "0.1",
    "status": "prototype",
    "requires_human_review": True,
}


# --------------------------------------------------------------------------- residual discovery
#
# The residual is the ONLY per-target input, and it is NOT a Python literal in core: it lives with the
# target package at ``<target_base>/contracts/residual.yaml``. This module discovers those residuals
# and runs the target-agnostic ``derive_manifest`` over each — so onboarding a target is "drop a
# descriptor + a residual, let mlc extract facts", with no code change here.


def _residual_path(name: str) -> Path:
    """The residual side-input path for a target: ``<target_base>/contracts/residual.yaml``."""
    return _target_base(name) / "contracts" / "residual.yaml"


def _load_residual(name: str) -> dict[str, Any]:
    p = _residual_path(name)
    if not p.is_file():
        raise KeyError(
            f"no capability residual for {name!r} at {p} — drop a contracts/residual.yaml "
            "in the target package (merlin.targetgen.capability_manifests)."
        )
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: residual is not a mapping")
    return doc


def discovered_targets() -> list[str]:
    """Every target that ships a capability residual (``<base>/*/contracts/residual.yaml``) — the
    DISCOVERED manifest set that replaces the retired hardcoded name list. A new target appears the
    moment it drops a residual; core needs no edit.

    Legacy generated residuals and the registry's reference metadata are included.
    References include the legacy shelf and physical-checkout authored examples;
    the registry owns precedence and explicit reference-root overrides. No example
    paths are inferred in an installed distribution.

    Genuinely OUT-OF-TREE packages are included too, via ``target_registry.external_targets()`` (which
    honours ``MERLIN_TARGET_PATH`` and the freshly-generated home). Scanning only the two in-tree roots
    made eviction self-defeating: a target moved out of the tree kept its contract and its backend but
    silently lost its manifest, and with it ``kind``, ``sim_via`` and ``endpoint_kind`` — so the very
    step that removes a target from core would quietly break oracle selection for it."""
    names: dict[str, None] = {}
    root = _artifacts_dir() / "targets"
    if root.is_dir():
        for p in root.glob("*/contracts/residual.yaml"):
            names[p.parent.parent.name] = None
    try:
        from .target_registry import external_targets, reference_targets, resolve
    except Exception:  # noqa: BLE001 — registry unavailable: report the in-tree set, never fail discovery
        return sorted(names)
    names.update(dict.fromkeys(reference_targets()))
    names.update(dict.fromkeys(external_targets() or {}))
    return sorted(name for name in names if (resolve(name).base / "contracts" / "residual.yaml").is_file())


def manifest_for(name: str) -> dict[str, Any]:
    """Build a target's capability manifest the AGNOSTIC way: load its residual side-input, load RTL
    facts when the residual marks ``facts_source: rtl`` (else none — an all-residual prototype), and run
    :func:`derive_manifest`. This is the single path for EVERY target (atlas proved it); there is no
    per-target builder or literal manifest dict in core."""
    residual = _load_residual(name)
    facts_source = residual.pop("facts_source", "none")
    # arc_target / facts_target are mlc-key side-inputs (consumed by mlc_bridge._arc_target and the facts
    # loader), NOT manifest body — pop them so they never leak a foreign target name into the derived
    # manifest. facts_target: which target's RTL facts ground the structural body (a config variant that
    # shares another target's decoder/mesh reuses those facts; datapath dtypes still come from THIS
    # target's residual/mlc).
    residual.pop("arc_target", None)
    facts_target = residual.pop("facts_target", None) or name
    facts: dict[str, Any] = {}
    if facts_source == "rtl":
        from .rtl import facts as _facts  # lazy: pulls circt_introspect only when RTL facts are needed

        facts = _facts.load_facts(facts_target)  # regenerates from the RTL if the cache is cold (mlc)
    elif facts_source == "simt":
        # A SIMT self-hosted core: its facts come from the SIMT RTL introspect (a standalone instruction
        # encoding, not a host RoCC decode table), adapted to the facts body shape so the SAME deriver
        # grounds endpoint_kind from them. Empty {} means the required interface is unresolved.
        from .rtl import mlc_bridge as _mb

        facts = _mb.simt_facts(facts_target)
    elif facts_source == "spatial":
        # A spatial tensor tile (a cluster x cell accumulator grid driven by a command buffer, with no
        # opcode decode at all). Its facts come from the OuterProductUnit state manifest + hw.mlir rather
        # than a RoCC decode table, and they carry a DIFFERENT shape than facts.json -- which
        # ``derive_manifest`` already recognises by its ``tile_dim`` field. The only thing missing was a
        # way to LOAD them here, and without it such a target's contract could be derived once by hand and
        # never regenerated: deriving it from the residual alone silently drops the tile geometry and
        # collapses the named fp8 datapaths to an unnamed float width, which is a quietly weaker contract
        # rather than a failure. Empty {} when mlc or the OPU artifacts are unavailable, so the deriver
        # falls back to the family default instead of a fabricated tile.
        from .rtl import spatial_introspect as _si

        facts = _si.build_fact_bundle(facts_target)
    return derive_manifest({"target": name, "facts_source": facts_source}, facts, residual=residual)


def __getattr__(attr: str):
    """``MANIFESTS`` is DISCOVERED, not a literal: ``{name -> zero-arg builder}`` for every target that
    ships a residual. Exposed as a module attribute (PEP 562) for the existing ``for name in
    cm.MANIFESTS`` / ``cm.MANIFESTS[name]()`` / ``name in cm.MANIFESTS`` call sites."""
    if attr == "MANIFESTS":
        return {name: (lambda n=name: manifest_for(n)) for name in discovered_targets()}
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def validate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Schema-validate the contract and parse its compute_units (raises on any problem)."""
    _schemas.validate_or_raise(manifest, "target_contract")
    _cu.compute_units(manifest)  # validates kinds/dtypes/scaling
    return manifest


def write(name: str, base: Path | None = None) -> Path:
    """Write a target's derived manifest to ``<base or target_base(name)>/contracts/target_contract.yaml``."""
    manifest = manifest_for(name)  # derive_manifest already schema-validated it
    root = base if base is not None else _target_base(name)
    path = root / "contracts" / "target_contract.yaml"
    write_yaml(
        path,
        manifest,
        header=f"GENERATED capability manifest for {name} "
        "(merlin.targetgen.capability_manifests). Provenance-tagged; "
        "requires_human_review. Regenerable from contracts/residual.yaml.",
    )
    return path


def write_all(base_root: Path | None = None) -> list[Path]:
    """Write every DISCOVERED target's manifest (:func:`discovered_targets`). ``base_root`` overrides the
    per-target base dir (each target lands under ``base_root/<name>/``)."""
    return [write(n, base=(base_root / n) if base_root is not None else None) for n in discovered_targets()]


def dialect_plan_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive a schema-valid dialect_plan from a manifest's compute_units (ops + a type per unit).

    The generated dialect exposes one ``!<target>.<unit>_tensor`` type per compute unit and the union
    of the units' ops — the machine-readable dialect spec the out-of-tree repo formalizes into real
    xDSL/MLIR. Types/ops are DERIVED from the capability model, not invented.
    """
    name = manifest["name"]
    units = _cu.compute_units(manifest)
    ops = sorted({op for u in units for op in u.ops})
    types = [{"name": f"{u.name}_tensor"} for u in units]
    return {
        "target": name,
        "dialect_name": name.replace("_", ""),
        "ops": [{"name": op} for op in ops],
        "types": types,
        "lowering": [{"op": op, "to": f"{name.replace('_', '')}.{op}"} for op in ops],
        "tests": [],
    }


# --------------------------------------------------------------------------- generic deriver
#
# ``derive_manifest`` is the target-AGNOSTIC path the per-target ``*_manifest()`` builders above will
# be retired into: a capability manifest = CIRCT facts (mesh / capacities / datapath dtypes / legal-funct
# codes — already in ``facts.json``) + FAMILY defaults (runner/endpoint from the compute-unit kind) +
# a small RESIDUAL side-input (the ABI ``encoding`` sub-block, ``requant.ref`` and human prose that RTL
# cannot ground). Onboarding a target becomes: drop a descriptor, let mlc extract facts, hand it the
# shrinking residual. The residual is exactly the shape of gemmini's stripped ``target_contract.yaml``.


def _descriptor_get(descriptor: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a descriptor that may be a bare target-name string, a mapping, or an object
    (e.g. :class:`merlin.targetgen.target_experiment.TargetExperiment`)."""
    if descriptor is None:
        return default
    if isinstance(descriptor, str):
        return descriptor if key == "target" else default
    if isinstance(descriptor, dict):
        return descriptor.get(key, default)
    return getattr(descriptor, key, default)


def _facts_body(facts: dict[str, Any]) -> dict[str, Any]:
    """The facts payload. Accepts a full ``facts.json`` (``{schema_version, inputs, facts: {...}}``,
    as :func:`merlin.targetgen.rtl.facts.load_facts` returns) OR a bare hand stub already at the
    ``{arrays, memories, datapaths, interfaces}`` level (non-arc targets have no arc — their facts are
    hand literals today)."""
    if isinstance(facts, dict) and isinstance(facts.get("facts"), dict):
        return facts["facts"]
    return facts or {}


def _fmt_name(token: str) -> str:
    """Canonical quant-format name for a datapath element token (``i8`` -> ``int8`` via the registry
    aliases); unknown tokens pass through unchanged (a raw accumulator token stays raw)."""
    try:
        return _qf.get(token).name
    except Exception:  # noqa: BLE001 — not a storage format (e.g. an i32 accumulator token)
        return token


def _mesh_from_facts(body: dict[str, Any]) -> dict[str, int] | None:
    """``capabilities.mesh`` {rows, cols} from the CIRCT-discovered ``mesh`` array, if present."""
    for arr in body.get("arrays") or []:
        if arr.get("name") == "mesh" and arr.get("rows") is not None and arr.get("cols") is not None:
            return {"rows": arr["rows"], "cols": arr["cols"]}
    return None


def _capacities_from_facts(body: dict[str, Any]) -> dict[str, int]:
    """``<memory>_bytes`` capacity facts (scratchpad/accumulator/...) from the CIRCT memory list."""
    out: dict[str, int] = {}
    for mem in body.get("memories") or []:
        name, nbytes = mem.get("name"), mem.get("bytes")
        if name and nbytes is not None:
            out[f"{name}_bytes"] = nbytes
    return out


def _datapaths_from_facts(body: dict[str, Any]) -> tuple[str | None, list[dict[str, str]]]:
    """The primary input storage dtype (quant-format name) + the ``(in, weight) -> acc`` accumulate
    matrix, both grounded in the CIRCT datapath facts (input element dtype x accumulator dtype)."""
    dps = body.get("datapaths") or []
    inp = next((d for d in dps if d.get("name") == "input"), None)
    acc = next((d for d in dps if d.get("name") == "accumulator"), None)
    in_dtype = _fmt_name(inp["dtype"]) if inp and inp.get("dtype") else None
    acc_tok = acc.get("dtype") if acc else None
    accumulate = [{"in": in_dtype, "weight": in_dtype, "acc": acc_tok}] if in_dtype and acc_tok else []
    return in_dtype, accumulate


def _accumulator_only_formats(body: dict[str, Any]) -> dict[str, str]:
    """``format -> evidence`` for formats the datapath facts ground ONLY as the array's accumulator.

    The compute cell's facts name what it consumes (``input``) and what it accumulates in
    (``accumulator``). A format named only in the second role is not an operand format of that unit, and
    a declaration that lists it among the unit's dtypes would admit operands of that format to the
    unit's contraction -- the accumulator type counted as an operand type.
    """
    dps = [d for d in (body.get("datapaths") or []) if isinstance(d, dict) and d.get("dtype")]
    inputs = {_fmt_name(str(d["dtype"])) for d in dps if d.get("name") == "input"}
    out: dict[str, str] = {}
    for d in dps:
        if d.get("name") == "accumulator":
            fmt = _fmt_name(str(d["dtype"]))
            if inputs and fmt not in inputs:
                out[fmt] = str(d.get("evidence") or d.get("source") or "datapath facts: accumulator")
    return out


def _drop_accumulator_only(unit: dict, declared: list[str], body: dict[str, Any]) -> list[str]:
    """``declared`` without the formats the facts ground only as this unit's accumulator.

    Recorded on the unit (``dtype_corrections``) and applied to the unit's declared semantic
    capabilities too, so the correction is visible rather than a silent narrowing.
    """
    acc_only = _accumulator_only_formats(body)
    dropped = [d for d in declared if _fmt_name(d) in acc_only]
    if not dropped:
        return declared
    unit["dtype_corrections"] = [
        {
            "dtype": d,
            "action": "removed_from_operand_dtypes",
            "reason": "the RTL datapath facts name this format only as the unit's accumulator, not as an "
            "input it consumes",
            "evidence": acc_only[_fmt_name(d)],
        }
        for d in dropped
    ]
    for cap in unit.get("semantic_capabilities") or []:
        if isinstance(cap, dict) and cap.get("dtypes"):
            cap["dtypes"] = [d for d in cap["dtypes"] if d not in dropped]
    return [d for d in declared if d not in dropped]


def _encoding_codes_from_facts(body: dict[str, Any]) -> dict[str, Any]:
    """The observed RoCC funct field, never a standalone instruction encoding.

    A generic equality fan-out can also find one field of a self-hosted ISA.
    Only a selected RoCC custom slot supplies the transport context in which
    these values are meaningful as funct codes. The observation is not an
    exhaustive executable-ISA legality claim.
    """
    for itf in body.get("interfaces") or []:
        if itf.get("name") == "funct_decode_table" and itf.get("custom_opcode") is not None:
            fields = ("custom_opcode", "funct3")
            if itf.get("scope") == "complete_rocc_funct7" and itf.get("complete_isa") is True:
                fields += ("legal_funct",)
            return {k: itf[k] for k in fields if k in itf}
    return {}


# RoCC's funct field is architecturally 7 bits. This bound validates an
# independently scoped RoCC interface; it does not classify an arbitrary field.
_ROCC_FUNCT7_MAX = 0x7f  # derived-ok: standard RoCC ABI — funct7 is a 7-bit field, max 2^7-1 (not target-specific)  # fmt: skip
_RISCV_CUSTOM_MAJOR_OPCODES = frozenset((0x0B, 0x2B, 0x5B, 0x7B))


def _endpoint_from_facts(body: dict[str, Any]) -> str | None:
    """Select an endpoint only from an explicit executable-interface fact.

    Equality comparisons over a field are not an ISA classification: a 5-bit
    sub-op may belong to a 32-bit self-hosted instruction, and a 14-bit slice
    does not prove the rest of that instruction's legality. Legacy tables with
    no scope are equally non-authorizing. An observed RoCC command transport
    with a selected custom major opcode, a scoped complete RoCC interface, or
    an independently established self-hosted instruction interface can decide
    the endpoint; decoder values alone return unknown.
    """
    for itf in body.get("interfaces") or []:
        if itf.get("name") == "self_hosted_isa" and itf.get("encoding_bits"):
            return "external_backend"
    # The host command transport plus its selected custom major opcode proves
    # how instructions reach a RoCC device. It says nothing about which funct
    # values are exhaustive or legal; those field observations remain scoped.
    interfaces = [itf for itf in (body.get("interfaces") or []) if isinstance(itf, dict)]
    has_rocc_command = any(itf.get("name") == "rocc_cmd" for itf in interfaces)
    has_custom_slot = any(
        itf.get("name") == "funct_decode_table"
        and isinstance(itf.get("custom_opcode"), int)
        and itf["custom_opcode"] in _RISCV_CUSTOM_MAJOR_OPCODES
        for itf in interfaces
    )
    if has_rocc_command and has_custom_slot:
        return "inline_asm_insn"
    for itf in body.get("interfaces") or []:
        if (
            itf.get("name") == "funct_decode_table"
            and itf.get("scope") == "complete_rocc_funct7"
            and itf.get("complete_isa") is True
            and itf.get("custom_opcode") is not None
        ):
            legal = itf.get("legal_funct") or []
            if not legal:
                return None
            if all(isinstance(v, int) and 0 <= v <= _ROCC_FUNCT7_MAX for v in legal):
                return "inline_asm_insn"
    return None


def _spatial_fields(body: dict[str, Any]) -> dict[str, Any] | None:
    """The SPATIAL (OuterProductUnit) fact fields ``{name: {value, derived, ...}}`` when ``body`` is a
    spatial fact bundle (:func:`merlin.targetgen.rtl.spatial_introspect.build_fact_bundle`) — detected by
    its ``tile_dim`` field. A spatial tile's facts carry a DIFFERENT shape than the systolic ``facts.json``
    (no ``arrays``/``memories``/``datapaths``/``interfaces``), so they get their own readers below."""
    fields = body.get("fields")
    if isinstance(fields, dict) and "tile_dim" in fields:
        return fields
    return None


def _spatial_datapaths_from_fields(fields: dict[str, Any]) -> tuple[str | None, list[str], list[dict[str, str]]]:
    """(primary input dtype, ALL storage dtypes, ``(in,weight)->acc`` matrix) from the OPU ``dtypes``
    datapath fact — the spatial analog of :func:`_datapaths_from_facts`. The OPU is MULTI-format (an int8
    MAC datapath + fp8 e4m3/e5m2 FMA datapaths), so it grounds a full dtype list, not a single dtype."""
    dts = ((fields.get("dtypes") or {}).get("value")) or []
    storage: list[str] = []
    accumulate: list[dict[str, str]] = []
    primary: str | None = None
    for d in dts:
        nm = d.get("name")
        if not nm:
            continue
        nm = _fmt_name(nm)
        if nm not in storage:
            storage.append(nm)
        acc = d.get("accumulator")
        if acc:
            accumulate.append({"in": nm, "weight": nm, "acc": acc})
        if primary is None:
            primary = nm
    return primary, storage, accumulate


def _spatial_capabilities_from_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """``capabilities`` geometry from the OPU tile facts: the ``tile`` {rows, cols} accumulator grid +
    ``mrf_depth`` (register-file bank count) — the spatial analog of :func:`_mesh_from_facts` (an OPU has
    a cluster x cell tile, NOT a systolic mesh)."""
    out: dict[str, Any] = {}
    tv = (fields.get("tile_dim") or {}).get("value") or {}
    if tv.get("rows") and tv.get("cols"):
        out["tile"] = {"rows": tv["rows"], "cols": tv["cols"]}
    mrf = (fields.get("mrf_depth") or {}).get("value")
    if mrf is not None:
        out["mrf_depth"] = mrf
    return out


def _kind_for_facet(facet: str) -> str | None:
    """The compute-unit KIND a facet implies, or None when the facet does not determine one.

    ``spatial`` maps to two kinds (``systolic`` and ``spatial``) and a role census cannot tell a
    stationary-weight wavefront from a rank-1 outer-product tile -- both push weights and drain
    accumulators. Synthesizing either would assert a datapath nobody observed, so the honest answer is
    None and the caller declines to synthesize."""
    from merlin.kernels import engines as _eng

    kinds = [k for k, f in _eng.ENGINE_FACET.items() if f == facet]
    return kinds[0] if len(kinds) == 1 else None


def _derived_units_for_undeclared_engines(name: str, manifest: dict, facts: dict) -> list[dict]:
    """Compute units for engines the target's OWN evidence reaches but its residual never declared.

    This is the acting half of the engine audit. The audit reports that atlas's ISA census evidences a
    vector engine while its contract declares one systolic ``mxu``; without this, that finding stays a
    line in a report and the eligibility oracle keeps attributing atlas's elementwise work to the
    matrix array -- the recorded `atlas-0of11-is-agent-not-tooling` shape, where the agent was never
    told which engine owned which family.

    Three constraints make this safe to run inside a generator:

    * it only ADDS units, never edits or removes a declared one, so authored intent always wins;
    * it declines any facet that does not determine a single kind (see :func:`_kind_for_facet`) rather
      than guessing a datapath;
    * every synthesized unit carries ``derived_from`` naming the rung and the literal observation, so a
      derived unit is never mistaken for a reviewed one.

    Ops and semantic families come from the ROLES that evidenced the engine, not from a default: a unit
    invented with a plausible-looking capability list would be a fabricated hardware claim wearing
    derivation's clothes.
    """
    from . import capability_derive as _cd
    from . import semantic_families as _sf

    declared = {u.get("kind") for u in (manifest.get("compute_units") or []) if isinstance(u, dict)}
    try:
        derived = _cd.derive_engines(name, manifest, facts)
    except Exception:  # noqa: BLE001 — no evidence is not a reason to fail generation
        return []

    # Which roles evidenced each facet, so the synthesized unit's capability is read off the same
    # census rather than defaulted.
    try:
        from . import isa_taxonomy as _it

        by_role = _it._classes_by_role(_it.taxonomy_for_target(name) or {})
    except Exception:  # noqa: BLE001
        by_role = {}

    out: list[dict] = []
    for facet, ev in sorted(derived.evidenced.items()):
        kind = _kind_for_facet(facet)
        if kind is None or kind in declared:
            continue
        roles = [r for r in sorted(by_role) if _cd._ROLE_ENGINE.get(r) == facet and by_role.get(r)]
        fams, ops = [], []
        for role in roles:
            mapped = _sf.ISA_ROLE_FAMILY.get(role)
            fam = mapped[0] if isinstance(mapped, tuple) else mapped
            if fam and fam not in fams:
                fams.append(fam)
        if not fams:
            continue  # evidenced an engine but nothing says what it computes
        lane = _lane_formats_for_kind(kind, facts)
        dtypes = lane[0] if lane else _unit_dtypes_for_synthesis(manifest)
        ops = sorted({_SYNTH_OP_FOR[f] for f in fams if f in _SYNTH_OP_FOR})
        if not ops:
            continue  # a family with no op token binds nothing; claim nothing
        unit = {
            "name": f"{kind}_unit",
            "kind": kind,
            "dtypes": list(dtypes),
            "ops": ops,
            "scaling": "none",
            "requant": {"ref": "none"},
            "semantic_capabilities": [{"family": f, "dtypes": list(dtypes)} for f in fams],
            "derived_from": f"{ev.source}: {ev.evidence}",
        }
        if lane:
            unit["dtypes_derived_from"] = lane[1]
        out.append(unit)
    return out


#: The op token a synthesized unit may claim for a family it was evidenced to compute.
#:
#: Deliberately small and explicit. An op token is a claim about what the compiler's ROUTING may bind
#: to this unit, so a generator widening that surface on its own would move the ARR numerator without
#: evidence. A family absent here synthesizes no unit rather than defaulting to a plausible op --
#: ``movement`` in particular evidences no engine and licenses no binding.
_SYNTH_OP_FOR = {"elementwise_map": "elementwise", "contraction": "matmul"}


def _lane_formats_for_kind(kind: str, facts: dict) -> tuple[tuple[str, ...], str] | None:
    """``(formats, evidence)`` the RTL facts name for a synthesized unit of ``kind``, or ``None``.

    A kind whose compute element is a LANE replication (:mod:`merlin.targetgen.families`) is the engine
    the facts' ``lane_datapaths`` describe: a unit beside the array whose arithmetic is replicated once
    per lane, with the element format that arithmetic names. Taken only when exactly one such unit
    resolved a format -- two lane engines naming different formats leave this engine's format open,
    and the caller keeps its stated fallback rather than picking one.
    """
    try:
        if _families.family_profile(kind).compute_element != "lane_replication":
            return None
    except Exception:  # noqa: BLE001 - an unknown kind has no compute-element rule
        return None
    lanes = [r for r in (_facts_body(facts).get("lane_datapaths") or []) if isinstance(r, dict) and r.get("dtype")]
    formats = tuple(dict.fromkeys(_fmt_name(str(r["dtype"])) for r in lanes))
    if len(formats) != 1:
        return None
    rec = lanes[0]
    return formats, f"rtl_facts.lane_datapaths[{rec.get('unit_module')}] ({rec.get('source')}): {rec.get('evidence')}"


def _unit_dtypes_for_synthesis(manifest: dict) -> tuple[str, ...]:
    """Formats a synthesized unit may claim: the union already declared by the target's own units.

    A lane engine that shares a register file with the array runs the same formats; inventing a wider
    set would be a fabricated capability, and inventing a narrower one would shrink the ARR denominator.
    """
    seen: list[str] = []
    for u in manifest.get("compute_units") or []:
        for d in (u.get("dtypes") or []) if isinstance(u, dict) else []:
            if d not in seen:
                seen.append(d)
    return tuple(seen)


def derive_manifest(
    descriptor: Any, facts: dict[str, Any], *, residual: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Derive a schema-valid capability manifest from a descriptor + CIRCT facts + a small residual.

    Field provenance (the three-way split this proves):

    - **CIRCT FACTS** (``facts``): ``capabilities.mesh`` (arrays), ``memory_model.<mem>_bytes``
      (memories), each compute unit's ``dtypes`` + ``accumulate`` matrix (datapaths), and the encoding
      CODES ``custom_opcode``/``funct3``/``legal_funct`` (funct_decode_table interface).
    - **FAMILY defaults** (:func:`merlin.targetgen.families.family_profile`, keyed by the primary
      compute-unit ``kind``): ``runner.suite`` and, only when no executable
      facts source was requested, a provisional codegen ``endpoint_kind`` fallback. The
      remaining generation defaults (rtl_tiers/perf_fields/trace_gate) are filled at read time by
      :func:`merlin.targetgen.target_experiment.load_capability_manifest`, so they are not duplicated
      into the emitted contract.
    - **RESIDUAL** (``residual``, the intent+prose side-input, exactly the shape of gemmini's stripped
      ``target_contract.yaml``): the compute-unit intent (name/kind/ops/scaling) + ``requant.ref``, the
      encoding ABI sub-block (addr_len/readout_bits/semantic_class/config_subtype), and the human prose
      (family/features/obligations/promises/oracle_ladder/provenance/notes/runner/runtime/...). Facts
      OVERRIDE the residual where they overlap (grounded dtypes/codes win over declared ones).

    ``facts`` accepts a full ``facts.json`` (arc/mlc-derived) or a bare hand stub (non-arc targets).
    Returns a schema-valid manifest (raises via :func:`validate` otherwise)."""
    from .target_experiment import _primary_kind  # lazy: avoid any import-order surprise

    manifest: dict[str, Any] = copy.deepcopy(dict(residual or {}))
    body = _facts_body(facts)

    name = _descriptor_get(descriptor, "target") or manifest.get("name")
    if not name:
        raise ValueError("derive_manifest: no target name (descriptor.target / residual.name)")
    manifest["name"] = name
    manifest.setdefault("version", _COMMON["version"])

    family = _descriptor_get(descriptor, "family") or manifest.get("family")
    if family:
        manifest["family"] = family

    # --- compute units: residual INTENT (name/kind/ops/scaling/requant) + FACTS (dtypes/accumulate) ---
    spatial = _spatial_fields(body)  # OuterProductUnit fact bundle vs the systolic facts.json shape
    if spatial is not None:
        _in_dtype, _storage, accumulate = _spatial_datapaths_from_fields(spatial)
    else:
        _in_dtype, accumulate = _datapaths_from_facts(body)
        _storage = [_in_dtype] if _in_dtype else []
    units = manifest.get("compute_units")
    if not units:
        kind_hint = _descriptor_get(descriptor, "kind") or manifest.get("kind")
        if not kind_hint:
            raise ValueError(f"{name}: no compute_units in residual and no descriptor/residual kind to synthesize one")
        units = [{"name": f"{kind_hint}_unit", "kind": kind_hint, "ops": ["matmul"]}]
        manifest["compute_units"] = units
    primary = units[0]
    # Ground each compute unit's INPUT dtypes from mlc's GENERAL datapath-dtype extractor — target-agnostic
    # for ANY target (typed MAC-mesh/FPU + spatial OPU), it reproduces the OPU-specific facts value and also
    # grounds systolic/FPU cores. It is PREFERRED over the OPU-only ``_spatial_datapaths_from_fields`` /
    # systolic ``_datapaths_from_facts`` storage, which stays the FALLBACK for the primary unit when mlc is
    # unavailable or the target is unsupported (``compute_unit_dtypes`` returns None) — so nothing regresses.
    # The extractor is keyed by unit; its per-unit lists map positionally onto the residual's compute_units
    # (a structural correspondence, not a literal name table — the primary unit takes the primary datapath).
    from .rtl import mlc_bridge as _mlc_bridge  # lazy: mlc access is guarded/context-managed inside

    _ext_lists = list((_mlc_bridge.compute_unit_dtypes(name) or {}).values())
    for _i, _unit in enumerate(units):
        # extractor dtypes (positional) win; the fact-bundle storage is the primary unit's fallback.
        _src = _ext_lists[_i] if _i < len(_ext_lists) else (_storage if _i == 0 else [])
        # The extractor may report a width-only ``float<N>`` for a float datapath whose sub-format identity
        # the RTL does not NAME (fail-closed — never a fabricated fp8_e4m3/fp8_e5m2 guessed from width +
        # port presence). That marker is honest but is NOT an actionable quant format, so keep the manifest's
        # dtype list to registry-known formats and SURFACE the dropped markers (never silently) under
        # ``unnamed_float_datapaths`` so the identity gap is visible instead of hidden.
        _src_known = [d for d in _src if d and _qf.has(d)]
        _src_unnamed = [d for d in _src if d and not _qf.has(d)]
        # AUGMENT (never drop) any human-reviewed formats the residual declared — a multi-format unit's
        # reviewed matrix can be richer than a single grounded datapath.
        _declared = list(dict.fromkeys([*(_unit.get("dtypes") or []), *_src_known]))
        if _i == 0:
            _declared = _drop_accumulator_only(_unit, _declared, body)
        if _declared:
            _unit["dtypes"] = _declared
        if _src_unnamed:
            _unit["unnamed_float_datapaths"] = list(dict.fromkeys(_src_unnamed))
    if accumulate and not primary.get("accumulate"):
        primary["accumulate"] = accumulate  # the (in,weight)->acc matrix is a datapath fact

    # primary compute-unit kind -> family generation defaults (reuse the shared registry + resolver)
    kind = _primary_kind(_cu.compute_units(manifest))
    profile = _families.family_profile(kind)
    # A field-local decode observation cannot choose an executable endpoint.
    # Keep an explicitly authored endpoint, or a complete independent interface
    # fact; otherwise mark a target with instruction observations unresolved.
    endpoint = _endpoint_from_facts(body)
    if endpoint:
        manifest["endpoint_kind"] = endpoint
    elif manifest.get("endpoint_kind"):
        pass  # reviewed target-owned interface declaration, not inferred from width
    elif _descriptor_get(descriptor, "facts_source") in {"rtl", "simt"} or any(
        itf.get("name") in {"funct_decode_table", "self_hosted_isa"}
        for itf in (body.get("interfaces") or [])
        if isinstance(itf, dict)
    ):
        manifest["endpoint_kind"] = "unresolved"
        manifest["endpoint_resolution"] = {
            "status": "unverified",
            "reason": "selected facts do not establish an executable endpoint",
        }
    else:
        manifest.setdefault("endpoint_kind", profile.endpoint_kind_default)

    runner = dict(manifest.get("runner") or {})
    runner.setdefault("suite", f"{name}-capsule-bench")
    manifest["runner"] = runner
    runtime = dict(manifest.get("runtime") or {})
    runtime.setdefault("backends", ["simulator"])
    manifest["runtime"] = runtime

    # --- capabilities.mesh + memory capacities: pure CIRCT facts layered onto the residual ---
    caps = dict(manifest.get("capabilities") or {})
    mesh = _mesh_from_facts(body)
    if mesh:
        caps["mesh"] = mesh
    # SIMT execution geometry (lanes/warp, warps, cores) DERIVED from the SIMT facts overrides the residual
    # literal — the same facts-win-over-residual rule as mesh (a SIMT core's lane count is grounded by the
    # introspect, not hand-declared).
    simt_geo = body.get("simt") or {}
    if isinstance(simt_geo.get("lanes_per_warp"), int):
        caps["simt"] = {
            **(caps.get("simt") or {}),
            **{
                k: simt_geo[k]
                for k in ("lanes_per_warp", "warps_per_core", "cores")
                if isinstance(simt_geo.get(k), int)
            },
        }
    if spatial is not None:  # OPU tile geometry (cluster x cell) + MRF bank depth
        caps.update(_spatial_capabilities_from_fields(spatial))
    manifest["capabilities"] = caps

    memory_model = dict(manifest.get("memory_model") or {})
    memory_model.update(_capacities_from_facts(body))
    manifest["memory_model"] = memory_model

    # --- encoding: residual ABI sub-block + facts CODES (codes win on overlap) ---
    codes = _encoding_codes_from_facts(body)
    residual_encoding = manifest.get("encoding")
    if codes or residual_encoding:
        manifest["encoding"] = {**dict(residual_encoding or {}), **codes}

    # --- schema-required top-level fields the stripped residual may omit ---
    for req in ("compiler_obligations", "hardware_promises", "runtime_promises", "legality"):
        manifest.setdefault(req, [])

    # --- compute units the evidence reaches but the residual never declared ---
    # Additive and evidence-bearing (see _derived_units_for_undeclared_engines): the audit below would
    # otherwise report the same undeclared engine on every run forever while the eligibility oracle kept
    # attributing that engine's work to the wrong datapath. Runs BEFORE the audit so the audit scores
    # the manifest that will actually ship.
    try:
        _synth = _derived_units_for_undeclared_engines(name, manifest, facts)
    except Exception:  # noqa: BLE001 — synthesis is an improvement, never a precondition
        _synth = []
    if _synth:
        manifest["compute_units"] = list(manifest.get("compute_units") or []) + _synth
        manifest["derived_compute_units"] = [u["name"] for u in _synth]

    # --- operation contract: one identity envelope across dialect IR and machine instructions ---
    # ``dialect_plan_from_manifest`` is already the canonical compute-unit -> target-dialect
    # projection.  The ISA taxonomy is independently discovered from the target's own instruction
    # definition.  Joining both here gives prompt/preflight consumers one domain/dialect/operation
    # registry instead of parallel scalar-, vector-, or target-specific capability tables.
    #
    # A declaration starts UNKNOWN: decoding an instruction is not proof that its architectural effect
    # works.  Optional residual observations are merged last and therefore refine the declaration to
    # SUPPORTED/UNSUPPORTED only when their exact operation identity exists.  A stale or misspelled
    # observation fails closed rather than manufacturing a capability.
    from .operation_capabilities import operation_contract_for_target

    manifest["operation_capabilities"] = operation_contract_for_target(name, manifest)

    # --- semantic capability: derived from this target's OWN evidence, every run ---
    # The ARR denominator is a claim about hardware, and a claim nothing checks is an assertion. So the
    # facts are produced HERE, while the contract is being derived, and recorded beside the declaration
    # rather than replacing it: a derivation bug must never be able to silently move the denominator,
    # and a target whose evidence is thin (empty RTL facts) must not have its reviewed families deleted.
    # `capability_evidence` is what the gate reads; `semantic_capabilities_unknown` is what keeps an
    # undecidable family out of BOTH sides of the ratio instead of flattering one of them.
    try:
        from . import capability_derive as _cd
        from . import eligibility as _el

        derived = _cd.derive(name, manifest, facts)
        manifest.update(derived.to_dict())
        evidence = {
            "drift": _cd.reconcile(_el.capability_map_from_contract(manifest), derived),
            "derived_families": derived.families(),
            "undetermined_families": sorted(derived.unknown),
        }

        # ENGINES, on the same terms: which compute datapaths the evidence reaches, recorded beside the
        # declared `compute_units` and never replacing them. A target described only by its declaration
        # can be most of a machine short -- atlas declares one systolic unit with ops ('matmul',) while
        # 50 of its 137 expert kernels drive a vector engine exclusively -- and that gap is invisible
        # until something compares the two.
        #
        # `unchecked_engine` findings are the load-bearing part. No rung can observe a SIMT engine
        # today, so a SIMT declaration is UNCHECKED rather than unsupported, and the report says which
        # it is. Collapsing those two would action a gap in our instruments as a fact about silicon.
        from merlin.kernels import engines as _eng

        engines = _cd.derive_engines(name, manifest, facts)
        evidence.update(engines.to_dict())
        declared_engines = _eng.facet_families_of_units(_cu.compute_units(manifest))
        evidence["engine_drift"] = _cd.reconcile_engines(declared_engines, engines)
        evidence["engines_declared"] = sorted(declared_engines)
        manifest["capability_evidence"] = evidence
    except Exception as exc:  # noqa: BLE001 — never block contract derivation on the audit
        manifest["capability_evidence"] = {"error": f"{type(exc).__name__}: {exc}"}

    return validate(manifest)


def write_oot_target(name: str, root: Path) -> Path:
    """Materialize a complete out-of-tree target package at ``root`` (discoverable via MERLIN_TARGET_PATH).

    Writes ``contracts/target_contract.yaml`` (capability manifest + plugin block),
    ``contracts/dialect_plan.yaml`` (derived from the compute units), the self-registering
    ``backend/`` the plugin block names, the experiment-ABI ``manifest.yaml`` WORK ORDER plus the
    declining ``tools/compile.py`` it points at, and an ``AGENT.md`` describing the compute units /
    datatypes.

    THE PLUGIN BLOCK IS THE POINT. This docstring claimed to write one for a long time and nothing did,
    so every package this function has ever produced resolved through ``target_registry`` and then
    raised ``KeyError`` from ``get_backend``. The package shape now comes from
    :mod:`merlin.targetgen.generate.oot_package`, shared with ``pipeline.build`` — the other generator —
    so the two cannot emit different things and leave the convergence with a second entrance.
    """
    from .generate import oot_package as _pkg

    manifest = manifest_for(name)  # derive_manifest already schema-validated it
    manifest = {**manifest, "plugin": {**dict(manifest.get("plugin") or {}), **_pkg.plugin_block()}}
    root = Path(root)
    write_yaml(
        root / "contracts" / "target_contract.yaml",
        manifest,
        header=f"GENERATED out-of-tree target manifest for {name} — plug in via MERLIN_TARGET_PATH.",
    )
    write_oot_runtime(name, root, manifest)
    write_yaml(
        root / "contracts" / "dialect_plan.yaml",
        dialect_plan_from_manifest(manifest),
        header=f"GENERATED dialect plan for {name} (derived from compute_units).",
    )
    units = _cu.compute_units(manifest)
    lines = [
        f"# {name} — out-of-tree target package",
        "",
        f"Generated by `merlin.targetgen.capability_manifests` (provenance: {manifest.get('provenance', 'n/a')}).",
        "",
        "Plug in: `MERLIN_TARGET_PATH=<this dir>`; the dialect + lowering the `plugin` block names "  # target-ok: example target named in generated README prose  # fmt: skip
        "live in the out-of-tree repo (e.g. radiance-mlir).",
        "",
        "## Compute units (datatype -> unit -> op)",
    ]
    for u in units:
        eff = _cu.effective(u, units)
        lines.append(
            f"- **{u.name}** ({u.kind}): dtypes {sorted(eff.dtypes)}; ops {sorted(eff.ops)}; "
            f"scaling {u.scaling}; requant {u.requant}" + (f"; contains {list(u.contains)}" if u.contains else "")
        )
    (root / "AGENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def write_oot_runtime(name: str, root: Path, manifest: dict[str, Any]) -> list[Path]:
    """Write the EXECUTABLE half of a generated package: the backend, its derived encoding, the work order.

    Split out of :func:`write_oot_target` so the other generator (``pipeline.build``, which assembles
    its repo out of :class:`~merlin.common.artifacts.Artifact` objects rather than writing YAML
    directly) emits byte-identical files through the same code rather than a parallel copy.

    Returns the paths written, so a caller can record what it produced.
    """
    from .generate import oot_package as _pkg

    root = Path(root)
    backend_dir = root / _pkg.BACKEND_DIR
    backend_dir.mkdir(parents=True, exist_ok=True)
    init = backend_dir / "__init__.py"
    init.write_text(_pkg.backend_module_source(name), encoding="utf-8")
    encoding = backend_dir / _pkg.ENCODING_FILE
    write_yaml(
        encoding,
        _pkg.command_encoding(name, manifest),
        header=f"GENERATED opcode map for {name} — DERIVED from this target's own manifest encoding.",
    )
    tool = root / _pkg.COMPILER_TOOL
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(_pkg.compiler_tool_source(name), encoding="utf-8")
    tool.chmod(0o755)
    work_order = root / _pkg.COMPILER_MANIFEST
    write_yaml(
        work_order,
        _pkg.compiler_manifest(name),
        header=(
            f"GENERATED experiment-ABI WORK ORDER for {name} — NOT a compiler. Phase 1 fills in the "
            f"four commands and flips package_capabilities.compiler.provided."
        ),
    )
    return [init, encoding, tool, work_order]


def materialize_generated_target(name: str, dest: Path | None = None) -> Path:
    """Materialize a target's OOT package into the ZERO-ENV generated home and return its root.

    The generated home is ``out/build/generated/<name>/`` (:func:`merlin.targetgen.target_registry.
    generated_target_home`), the location :func:`target_registry.resolve` auto-discovers WITHOUT any
    ``MERLIN_TARGET_PATH`` — the seamless default for a freshly generated target. ``dest`` overrides the
    destination directory (e.g. a pinned/versioned copy).

    This is the shared target-independent materialization entry point. Example-owned setup tools
    and other explicit callers use the same mechanism; target identity is always supplied by the caller.
    """
    from .target_registry import generated_target_home  # lazy: avoid an import cycle at module load

    root = Path(dest) if dest is not None else generated_target_home() / name
    return write_oot_target(name, root)
