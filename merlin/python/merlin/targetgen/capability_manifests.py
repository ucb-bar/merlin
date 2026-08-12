"""Target-AGNOSTIC capability-manifest deriver + per-target residual discovery.

A capability manifest is a target **definition** (a ``target_contract.yaml`` with ``compute_units``)
generated as an ``out/`` artifact and plugged into the routing tooling — the same shape a real
out-of-tree target repo ships. There are NO per-target manifest dicts baked into this core module:
every manifest is reconstructed by :func:`derive_manifest` from three sources —

  * **CIRCT FACTS** (``facts.json``): mesh / memory capacities / datapath dtypes / legal-funct codes,
  * **FAMILY defaults** (:func:`merlin.targetgen.families.family_profile`, keyed by compute-unit kind):
    the codegen endpoint + runner fallback,
  * a small **RESIDUAL** side-input (intent + prose RTL cannot ground) that lives WITH the target
    package at ``<target_base>/contracts/residual.yaml`` — the discovered target dir, never a Python
    literal here.

Discovery (:func:`discovered_targets`) scans ``artifacts/targets/*/contracts/residual.yaml``, and
``manifest_for(name)`` runs the single agnostic derive path over each. A new accelerator brings itself
up by dropping a descriptor + a residual and letting mlc extract facts — with zero edits to core.

The shipped residuals: ``rvv`` (K1 vector unit — regular floats + int8, no low-bit datapath),
``mx_gemmini`` (microscaling systolic PE — mxfp4/6/8 + int8/bf16), ``radiance`` (SIMT tensor core that
composes the gemmini-mx PE), and ``atlas`` (self-hosted-ISA NPU MXU whose mesh/encoding/endpoint are
DERIVED from RTL facts — ``facts_source: rtl``). All are provenance-tagged prototypes flagged
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
from merlin.common.paths import targets_dir as _targets_dir
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
        raise KeyError(f"no capability residual for {name!r} at {p} — drop a contracts/residual.yaml "
                       "in the target package (merlin.targetgen.capability_manifests).")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: residual is not a mapping")
    return doc


def discovered_targets() -> list[str]:
    """Every target that ships a capability residual (``<base>/*/contracts/residual.yaml``) — the
    DISCOVERED manifest set that replaces the retired hardcoded name list. A new target appears the
    moment it drops a residual; core needs no edit.

    Both target-home roots are scanned, matching where :func:`_residual_path` /
    :func:`merlin.targetgen.rtl.facts.target_base` look: the generated home
    ``artifacts/targets/<t>/`` (the four all-residual/rtl prototypes live here) AND the curated
    reference home ``merlin/targets/<t>/`` (a reference target such as gemmini ships its residual
    beside its committed ``target_contract.yaml``). A name is reported once; the reference base — the
    one ``target_base`` resolves to when it exists — wins on any overlap, so ``manifest_for`` loads the
    same file this set advertises."""
    names: dict[str, None] = {}
    for root in (_artifacts_dir() / "targets", _targets_dir()):
        if root.is_dir():
            for p in root.glob("*/contracts/residual.yaml"):
                names[p.parent.parent.name] = None
    return sorted(names)


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
        from .rtl import facts as _facts   # lazy: pulls circt_introspect only when RTL facts are needed
        facts = _facts.load_facts(facts_target)  # regenerates from the RTL if the cache is cold (mlc)
    elif facts_source == "simt":
        # A SIMT self-hosted core: its facts come from the SIMT RTL introspect (a standalone instruction
        # encoding, not a host RoCC decode table), adapted to the facts body shape so the SAME deriver
        # grounds endpoint_kind from them. Empty {} when no introspect serves the target (family default).
        from .rtl import mlc_bridge as _mb
        facts = _mb.simt_facts(facts_target)
    return derive_manifest({"target": name}, facts, residual=residual)


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
    _cu.compute_units(manifest)   # validates kinds/dtypes/scaling
    return manifest


def write(name: str, base: Path | None = None) -> Path:
    """Write a target's derived manifest to ``<base or target_base(name)>/contracts/target_contract.yaml``."""
    manifest = manifest_for(name)   # derive_manifest already schema-validated it
    root = base if base is not None else _target_base(name)
    path = root / "contracts" / "target_contract.yaml"
    write_yaml(path, manifest, header=f"GENERATED capability manifest for {name} "
                                       "(merlin.targetgen.capability_manifests). Provenance-tagged; "
                                       "requires_human_review. Regenerable from contracts/residual.yaml.")
    return path


def write_all(base_root: Path | None = None) -> list[Path]:
    """Write every DISCOVERED target's manifest (:func:`discovered_targets`). ``base_root`` overrides the
    per-target base dir (each target lands under ``base_root/<name>/``)."""
    return [write(n, base=(base_root / n) if base_root is not None else None)
            for n in discovered_targets()]


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
    accumulate = ([{"in": in_dtype, "weight": in_dtype, "acc": acc_tok}]
                  if in_dtype and acc_tok else [])
    return in_dtype, accumulate


def _encoding_codes_from_facts(body: dict[str, Any]) -> dict[str, Any]:
    """The RTL-grounded encoding CODES (``custom_opcode`` / ``funct3`` / ``legal_funct``) from the
    ``funct_decode_table`` interface — the only encoding fields facts can ground (the ABI sub-block
    lives in the residual)."""
    for itf in body.get("interfaces") or []:
        if itf.get("name") == "funct_decode_table":
            return {k: itf[k] for k in ("custom_opcode", "funct3", "legal_funct") if k in itf}
    return {}


# RoCC's funct field is architecturally 7 bits — a legal opcode above 0x7f cannot be a RoCC funct7, so a
# decode table with wider opcodes is a standalone instruction decode (a self-hosted ISA core with its own
# opcodes/PC/IMEM), not a RoCC co-processor. This is the load-bearing, RTL-grounded distinction between a
# host-driven ``.insn`` endpoint and a device-kernel (``external_backend``) endpoint — never hand-set.
_ROCC_FUNCT7_MAX = 0x7f  # derived-ok: standard RoCC ABI — funct7 is a 7-bit field, max 2^7-1 (not target-specific)


def _endpoint_from_facts(body: dict[str, Any]) -> str | None:
    """Codegen ``endpoint_kind`` DERIVED from the CIRCT decode facts, never hand-set per target.

    A ``funct_decode_table`` whose legal opcodes ALL fit RoCC's 7-bit funct field (``<= 0x7f``) is a RoCC
    co-processor decoded from the host pipeline -> ``inline_asm_insn`` (emit host ``.insn``). One with any
    wider opcode (a standalone instruction decode — e.g. atlas's 14-bit ``ScalarDecoder``, values to
    ``0x26d7``) is a self-hosted ISA core -> ``external_backend`` (emit a device ``kernel.S`` the target's
    own assembler builds). No decode table -> ``None`` (caller falls back to the family default, e.g. a
    spatial command-buffer target has one-hot op ports, not an opcode decode)."""
    for itf in body.get("interfaces") or []:
        if itf.get("name") == "funct_decode_table":
            legal = itf.get("legal_funct") or []
            if not legal:
                return None
            return "inline_asm_insn" if max(legal) <= _ROCC_FUNCT7_MAX else "external_backend"
    # A self-hosted-ISA core carries its OWN instruction encoding (``encoding_bits``) + instruction
    # classes, NOT a host-decoded RoCC funct — so it emits a device kernel the target's own toolchain
    # builds -> ``external_backend``. This is the SIMT analog of "opcodes too wide for RoCC funct7": the
    # signal is a standalone instruction encoding, surfaced as a ``self_hosted_isa`` interface (e.g. a
    # 64-bit Muon/Vortex SIMT encoding), never a target-name test.
    for itf in body.get("interfaces") or []:
        if itf.get("name") == "self_hosted_isa" and itf.get("encoding_bits"):
            return "external_backend"
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


def _spatial_datapaths_from_fields(
        fields: dict[str, Any]) -> tuple[str | None, list[str], list[dict[str, str]]]:
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


def derive_manifest(descriptor: Any, facts: dict[str, Any], *,
                    residual: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive a schema-valid capability manifest from a descriptor + CIRCT facts + a small residual.

    Field provenance (the three-way split this proves):

    - **CIRCT FACTS** (``facts``): ``capabilities.mesh`` (arrays), ``memory_model.<mem>_bytes``
      (memories), each compute unit's ``dtypes`` + ``accumulate`` matrix (datapaths), and the encoding
      CODES ``custom_opcode``/``funct3``/``legal_funct`` (funct_decode_table interface).
    - **FAMILY defaults** (:func:`merlin.targetgen.families.family_profile`, keyed by the primary
      compute-unit ``kind``): the codegen ``endpoint_kind`` and the ``runner.suite`` fallback. The
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
    spatial = _spatial_fields(body)   # OuterProductUnit fact bundle vs the systolic facts.json shape
    if spatial is not None:
        _in_dtype, _storage, accumulate = _spatial_datapaths_from_fields(spatial)
    else:
        _in_dtype, accumulate = _datapaths_from_facts(body)
        _storage = [_in_dtype] if _in_dtype else []
    units = manifest.get("compute_units")
    if not units:
        kind_hint = _descriptor_get(descriptor, "kind") or manifest.get("kind")
        if not kind_hint:
            raise ValueError(f"{name}: no compute_units in residual and no descriptor/residual kind "
                             "to synthesize one")
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
    from .rtl import mlc_bridge as _mlc_bridge   # lazy: mlc access is guarded/context-managed inside
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
        if _declared:
            _unit["dtypes"] = _declared
        if _src_unnamed:
            _unit["unnamed_float_datapaths"] = list(dict.fromkeys(_src_unnamed))
    if accumulate and not primary.get("accumulate"):
        primary["accumulate"] = accumulate      # the (in,weight)->acc matrix is a datapath fact

    # primary compute-unit kind -> family generation defaults (reuse the shared registry + resolver)
    kind = _primary_kind(_cu.compute_units(manifest))
    profile = _families.family_profile(kind)
    # endpoint_kind: FACTS win (the decode-width signal) over the residual over the family default. A
    # self-hosted-ISA systolic core (atlas: 14-bit ScalarDecoder) derives external_backend; a RoCC
    # systolic co-processor (gemmini: 7-bit funct7) derives inline_asm_insn — neither hand-set.
    endpoint = _endpoint_from_facts(body)
    if endpoint:
        manifest["endpoint_kind"] = endpoint
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
        caps["simt"] = {**(caps.get("simt") or {}),
                        **{k: simt_geo[k] for k in ("lanes_per_warp", "warps_per_core", "cores")
                           if isinstance(simt_geo.get(k), int)}}
    if spatial is not None:                       # OPU tile geometry (cluster x cell) + MRF bank depth
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

    return validate(manifest)


def write_oot_target(name: str, root: Path) -> Path:
    """Materialize a complete out-of-tree target package at ``root`` (discoverable via MERLIN_TARGET_PATH).

    Writes ``contracts/target_contract.yaml`` (capability manifest + plugin block),
    ``contracts/dialect_plan.yaml`` (derived from the compute units), and an ``AGENT.md`` describing the
    compute units / datatypes — the first materialization of the out-of-tree target repo (e.g. the
    radiance-mlir repo will host the real dialect + lowering the plugin block references).
    """
    manifest = manifest_for(name)   # derive_manifest already schema-validated it
    root = Path(root)
    write_yaml(root / "contracts" / "target_contract.yaml", manifest,
               header=f"GENERATED out-of-tree target manifest for {name} — plug in via MERLIN_TARGET_PATH.")
    write_yaml(root / "contracts" / "dialect_plan.yaml", dialect_plan_from_manifest(manifest),
               header=f"GENERATED dialect plan for {name} (derived from compute_units).")
    units = _cu.compute_units(manifest)
    lines = [f"# {name} — out-of-tree target package", "",
             f"Generated by `merlin.targetgen.capability_manifests` (provenance: "
             f"{manifest.get('provenance', 'n/a')}).", "",
             "Plug in: `MERLIN_TARGET_PATH=<this dir>`; the dialect + lowering the `plugin` block names "
             "live in the out-of-tree repo (e.g. radiance-mlir).", "",
             "## Compute units (datatype -> unit -> op)"]
    for u in units:
        eff = _cu.effective(u, units)
        lines.append(f"- **{u.name}** ({u.kind}): dtypes {sorted(eff.dtypes)}; ops {sorted(eff.ops)}; "
                     f"scaling {u.scaling}; requant {u.requant}"
                     + (f"; contains {list(u.contains)}" if u.contains else ""))
    (root / "AGENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
