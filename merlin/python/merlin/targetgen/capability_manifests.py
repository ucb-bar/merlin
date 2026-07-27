"""Capability manifests for the two vertical-slice targets: ``rvv`` and ``mx_gemmini``.

These are target **definitions** (a `target_contract.yaml` with `compute_units`) generated as `out/`
artifacts and plugged into the routing tooling — the same shape a real out-of-tree target
(radiance-mlir) will ship. They exist to prove datatype -> compute-unit routing across two very
different targets:

- ``rvv`` (SpacemiT K1 vector unit): the regular formats fp32/fp16/bf16 + int8 — no fp4/fp6/native-fp8
  datapath, so those route to an honest gap.
- ``mx_gemmini`` (microscaling systolic PE): mxfp4/mxfp6/mxfp8 + int8/bf16 with an E8M0 block-scale
  requant — accepts the low-bit formats RVV rejects.

Both are provenance-tagged prototypes flagged ``requires_human_review`` — the numbers/modes trace to a
source but are NOT RTL-certified. The requant reference is opaque (target-specific lowering lives
elsewhere: RVV's software requant in Merlin; gemmini-mx's MxRequantizer out-of-tree).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from merlin.common import quant_formats as _qf
from merlin.common import schemas as _schemas
from merlin.common.yaml import write_yaml
from merlin.targetgen import compute_units as _cu
from merlin.targetgen import families as _families
from merlin.targetgen import target_registry as _tr

_COMMON: dict[str, Any] = {
    "version": "0.1",
    "status": "prototype",
    "requires_human_review": True,
}


def rvv_manifest() -> dict[str, Any]:
    """K1/RVV vector unit — regular floats + int8 (fp8 is decode-to-f32, not a native format)."""
    return {
        **_COMMON,
        "name": "rvv",
        "family": "cpu_vector",
        "provenance": "SpacemiT K1 X60 rv64gcv (VLEN=256); RVV datapaths per merlin.rvvgen.k1 + "
                      "llvmlower.passes_quant_int (int8 W8A8 vwmacc) + passes_xdsl.lower_bf16_matmul_f32acc.",
        "features": ["vector_unit"],
        "capabilities": {"ops": ["matmul", "elementwise"]},
        "memory_model": {"resident": False},
        "compiler_obligations": [],
        "hardware_promises": ["ieee_float", "widening_integer_accumulate"],
        "runtime_promises": ["metrics"],
        "legality": [],
        "runtime": {"default_backend": "baremetal"},
        "compute_units": [
            {
                "name": "vector",
                "kind": "vector",
                "dtypes": ["fp32", "fp16", "bf16", "int8"],
                "ops": ["matmul", "elementwise"],
                "accumulate": [
                    {"in": "fp32", "weight": "fp32", "acc": "f32"},
                    {"in": "fp16", "weight": "fp16", "acc": "f32"},
                    {"in": "bf16", "weight": "bf16", "acc": "f32"},
                    {"in": "int8", "weight": "int8", "acc": "i32"},
                ],
                "scaling": "per_channel",
                # RVV requant is OUR software lowering (scale-multiply), not a hardware unit.
                "requant": {"ref": "merlin.llvmlower.passes_quant_int"},
            }
        ],
    }


def mx_gemmini_manifest() -> dict[str, Any]:
    """Gemmini-MX microscaling systolic PE — fp4/fp6/fp8 (MX, E8M0 block scale) + int8/bf16."""
    return {
        **_COMMON,
        "name": "mx_gemmini",
        "family": "tensor_resident",
        "provenance": "chipyard generators/gemmini (branch gemmini-mx) MxParameters.scala: "
                      "mxGemminiConfig=[mode0(fp4/fp4), mode4(fp6/fp6), mode8(fp8/fp8)]; block scale via "
                      "ScaleFactorMem + MxExp (E8M0) + QuantLut. TypeSupport permits independent act/wei "
                      "modes (not enumerated here — shipped config is symmetric).",
        "features": ["systolic_array", "microscaling", "resident_packed_tensor", "accumulator_commit"],
        "capabilities": {"ops": ["matmul"], "mesh": {"rows": 16, "cols": 16}},
        "memory_model": {"resident": True, "accumulators": True, "block_scale_memory": True},
        "compiler_obligations": ["must_tile_to_mesh_shape", "must_supply_e8m0_block_scales"],
        "hardware_promises": ["microscaling_block_exponent", "widening_accumulate"],
        "runtime_promises": ["command_buffer", "metrics"],
        "legality": ["mx block-scale granularity == 32"],
        "runtime": {"default_backend": "firesim"},
        "compute_units": [
            {
                "name": "mx_pe",
                "kind": "systolic",
                "dtypes": ["mxfp4", "mxfp6", "mxfp8", "int8", "bf16"],
                "ops": ["matmul"],
                "accumulate": [
                    {"in": "mxfp4", "weight": "mxfp4", "acc": "f32"},   # PE_MxMode mode0
                    {"in": "mxfp6", "weight": "mxfp6", "acc": "f32"},   # mode4
                    {"in": "mxfp8", "weight": "mxfp8", "acc": "f32"},   # mode8
                    {"in": "int8", "weight": "int8", "acc": "i32"},
                ],
                "scaling": "block_e8m0",
                # gemmini-mx requant (MxRequantizer + E8M0 exponent add + QuantLut) is target-specific,
                # out-of-tree — Merlin only references it.
                "requant": {"ref": "mx_gemmini.mx_requantizer"},
            }
        ],
    }


def radiance_manifest() -> dict[str, Any]:
    """Radiance (Muon SIMT tensor core) — regular fp32/fp16/bf16, COMPOSING the gemmini-mx MX PE.

    Demonstrates the composition the hardware allows: gemmini-mx is a compute unit that can stand
    alone (see :func:`mx_gemmini_manifest`) OR be embedded inside a radiance cluster. Here the SIMT
    cluster ``contains`` the ``mx_pe`` unit, so the target's effective capability is the union: regular
    half/single floats on the SIMT lanes + the low-bit MX formats on the embedded PE. Carries a
    ``plugin`` block for the out-of-tree discovery seam (the real dialect + lowering live in a
    radiance-mlir repo; this artifact is the first materialization under out/artifacts/targets).
    """
    mx_pe = mx_gemmini_manifest()["compute_units"][0]   # reuse the exact gemmini-mx PE (composition)
    return {
        **_COMMON,
        "name": "radiance",
        "family": "simt_tensor",
        "provenance": "chipyard generators/radiance (Muon SIMT): CVFPU FP32/FP16/BF16 + INT32 lanes "
                      "(muon backend/fp/FPU.scala; muon_introspect fp_datapath); tensor-core matmul via "
                      "nu.invoke. Embeds the gemmini-mx MX PE as a contained low-bit unit. Not RTL-certified.",
        "features": ["simt", "cvfpu", "tensor_core", "microscaling"],
        "capabilities": {"ops": ["matmul", "elementwise"], "simt": {"lanes_per_warp": 16}},
        "memory_model": {"resident": True, "shared_memory": True},
        "compiler_obligations": ["must_map_to_warps"],
        "hardware_promises": ["ieee_float", "fp32_accumulate"],
        "runtime_promises": ["metrics"],
        "legality": [],
        "runtime": {"default_backend": "simulator"},
        "plugin": {
            "dialect_module": "radiance_mlir.dialect",
            "lowering_entrypoint": "radiance_mlir.lowering:lower",
        },
        "compute_units": [
            {
                "name": "simt_cluster",
                "kind": "simt",
                "dtypes": ["fp32", "fp16", "bf16"],
                "ops": ["matmul", "elementwise"],
                "accumulate": [
                    {"in": "fp32", "weight": "fp32", "acc": "f32"},
                    {"in": "fp16", "weight": "fp16", "acc": "f32"},
                    {"in": "bf16", "weight": "bf16", "acc": "f32"},
                ],
                "scaling": "none",
                "requant": {"ref": "none"},
                "contains": ["mx_pe"],   # gemmini-mx embedded as a sub-unit
            },
            mx_pe,
        ],
    }


def _atlas_residual() -> dict[str, Any]:
    """The IRREDUCIBLE declared side-input for atlas — only what the CIRCT facts cannot (yet) ground.

    Grounded FROM FACTS by :func:`derive_manifest`, so DELIBERATELY ABSENT here: ``endpoint_kind`` (the
    14-bit ScalarDecoder opcode set is too wide for RoCC funct7 -> ``external_backend``), ``capabilities.mesh``
    (the 32x32 array), and the ``encoding`` codes (the 42-entry legal opcode set). What remains is the
    compute-unit datapath INTENT (fp8/bf16 dtypes + accumulate matrix + E8M0 scaling + requant ref) — NOT
    yet RTL-grounded, pending mlc datapath discovery (the storage/accumulator dtypes are absent from this
    ``facts.json`` schema) — plus the human prose (family/features/obligations/promises/runner/runtime)."""
    return {
        **_COMMON,
        "name": "atlas",
        "family": "tensor_resident",
        "provenance": "endpoint_kind + capabilities.mesh + encoding codes are DERIVED from the pinned mlc "
                      "CIRCT facts (a 42-entry ScalarDecoder decode of 14-bit opcodes — too wide for RoCC "
                      "funct7 -> self-hosted ISA -> external_backend; a 32x32 discovered array -> mesh). The "
                      "compute-unit datapath (inputFmt=E4M3 fp8 activation+weight; accumFmt=outputFmt=BF16, "
                      "with an FP32Addition PE variant; E8M0 block scale via ScalingFactorRegFile) is DECLARED "
                      "intent from the atlas-npu chipyard generator (MxuParams.scala) — NOT yet RTL-grounded "
                      "(dtypes are absent from this facts.json schema; pending mlc datapath discovery). "
                      "NOT RTL-certified.",
        "features": ["systolic_array", "fp8_mxu", "microscaling", "bf16_accumulate"],
        # The program oracle (targetgen.program_oracle) assembles the emitted kernel.S via npu_model's OWN
        # assembler (``model_ext``) and runs the target's mlc arc cosim — declared intent, not an RTL fact.
        "runner": {"model_ext": "npu_model", "fourth_output_name": "kernel.S"},
        "capabilities": {"ops": ["matmul"]},          # mesh comes from facts; ops is compute-unit intent
        "memory_model": {"resident": True, "accumulators": True, "block_scale_memory": True},
        "compiler_obligations": ["must_tile_to_mesh_shape", "must_supply_e8m0_block_scales"],
        "hardware_promises": ["fp8_multiply", "bf16_accumulate"],
        "runtime_promises": ["metrics"],
        "legality": ["matmul tiles to the 32x32 mesh"],
        "runtime": {"default_backend": "simulator"},
        "compute_units": [
            {
                "name": "mxu",
                "kind": "systolic",
                "dtypes": ["fp8_e4m3", "bf16"],
                "ops": ["matmul"],
                "accumulate": [
                    {"in": "fp8_e4m3", "weight": "fp8_e4m3", "acc": "bf16"},  # CustomFMA(default)/HardFloatFMA
                    {"in": "fp8_e4m3", "weight": "fp8_e4m3", "acc": "f32"},   # FP32Addition PE variant
                ],
                "scaling": "block_e8m0",
                "requant": {"ref": "atlas.e4m3_block_requant"},
            }
        ],
    }


def _load_atlas_facts() -> dict[str, Any]:
    """The atlas CIRCT facts (mlc discovery) the mesh/endpoint/encoding are derived from — read via the
    standard resolver (the purgeable ``rtl_introspect`` cache; regenerated by mlc discovery when absent).
    atlas is a generated/out-of-tree target, so there is no in-tree pin."""
    from .rtl import facts as _facts
    return _facts.load_facts("atlas")


def atlas_manifest() -> dict[str, Any]:
    """ATLAS NPU systolic MXU manifest — DERIVED via :func:`derive_manifest` from the mlc CIRCT facts +
    the irreducible residual. ``endpoint_kind`` (``external_backend``), the 32x32 mesh, and the encoding
    codes fall out of the facts (never hand-set per target); the residual carries only the datapath intent
    + prose the facts cannot yet ground. This is the same generic path any target uses; the result is
    materialized as an out-of-tree package (``write_oot_target``), discovered via ``MERLIN_TARGET_PATH`` —
    no in-tree ``merlin/targets/atlas`` directory."""
    descriptor = {"target": "atlas", "kind": "systolic", "family": "tensor_resident"}
    return derive_manifest(descriptor, _load_atlas_facts(), residual=_atlas_residual())


MANIFESTS = {"rvv": rvv_manifest, "mx_gemmini": mx_gemmini_manifest,
             "radiance": radiance_manifest, "atlas": atlas_manifest}


def validate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Schema-validate the contract and parse its compute_units (raises on any problem)."""
    _schemas.validate_or_raise(manifest, "target_contract")
    _cu.compute_units(manifest)   # validates kinds/dtypes/scaling
    return manifest


def write(name: str, base: Path | None = None) -> Path:
    """Write a manifest to ``<base or resolve(name).base>/contracts/target_contract.yaml``."""
    manifest = validate(MANIFESTS[name]())
    root = base if base is not None else _tr.resolve(name).base
    path = root / "contracts" / "target_contract.yaml"
    write_yaml(path, manifest, header=f"GENERATED capability manifest for {name} "
                                       "(merlin.targetgen.capability_manifests). Provenance-tagged; "
                                       "requires_human_review. Regenerable.")
    return path


def write_all(base_root: Path | None = None) -> list[Path]:
    """Write all manifests (rvv, mx_gemmini, radiance). ``base_root`` overrides the per-target base dir."""
    return [write(n, base=(base_root / n) if base_root is not None else None) for n in MANIFESTS]


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
_ROCC_FUNCT7_MAX = 0x7f


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
    # Ground the primary unit's storage dtype from the datapath fact, AUGMENTING (never dropping) any
    # human-reviewed formats the residual declared — a multi-format unit's reviewed matrix is richer
    # than the single primary datapath facts can ground.
    declared = list(dict.fromkeys([*(primary.get("dtypes") or []), *(d for d in _storage if d)]))
    if declared:
        primary["dtypes"] = declared
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
    manifest = validate(MANIFESTS[name]())
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
