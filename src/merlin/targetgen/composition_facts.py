"""The hardware facts a declared composition chain is generated and judged against, derived.

Phase 0 composition generation (:mod:`merlin.targetgen.composition_corpus`) and its truth gate
(:mod:`merlin.targetgen.functional_truth_compositions`) read a small set of target facts: the
operand and mesh widths, the dataflows, the K-extension bound, the shared-input residency, the
weight-panel width, the store-path pooling bounds and where each producer may run. Each is read
here from the selected RTL facts, the selected contract and the selected software spec. Nothing
names a target: field and memory names come from the contract's own declarations.

A fact that no source establishes is left out and its reason recorded under ``unknown``, so the
consumer refuses whatever needs it. A value read from an authored declaration rather than the
RTL carries that declaration's review status under ``evidence``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Composition dataflow names for the structural modes :mod:`.rtl.timing_structure` classifies.
DATAFLOW_NAMES = {"weight_stationary": "WS", "output_stationary": "OS"}


def _body(facts: Mapping[str, Any] | None) -> Mapping[str, Any]:
    inner = (facts or {}).get("facts")
    return inner if isinstance(inner, Mapping) else (facts or {})


def _datapath(body: Mapping[str, Any], name: str) -> str | None:
    for key in ("storage_datapaths", "datapaths"):
        for row in body.get(key) or ():
            if isinstance(row, Mapping) and row.get("name") == name and isinstance(row.get("dtype"), str):
                return row["dtype"]
    return None


def _signed_bits(dtype: str | None) -> int | None:
    from merlin.common import quant_formats as qf

    if not dtype or not qf.has(dtype):
        return None
    fmt = qf.get(dtype)
    if fmt.exp_bits is not None or not fmt.signed:
        return None
    return int(qf.machine_bits(dtype))


def _register_field_width(body: Mapping[str, Any], bundle: str, field: str) -> int | None:
    for item in body.get("interfaces") or ():
        if isinstance(item, Mapping) and item.get("name") == "register_bundle_layouts":
            placed = (((item.get("bundles") or {}).get(bundle) or {}).get("fields") or {}).get(field) or {}
            width = placed.get("width") or placed.get("slot_width")
            return width if type(width) is int and width > 0 else None
    return None


def _role_field(contract: Mapping[str, Any], name: str) -> tuple[str, str] | None:
    """The (bundle, field) the contract's operand roles name ``name`` by, when exactly one does."""
    found = {
        (row["bundle"], row["field"])
        for klass in ((contract.get("rocc_operand_roles") or {}).get("classes") or {}).values()
        for row in (klass or {}).get("fields") or ()
        if isinstance(row, Mapping) and row.get("name") == name and row.get("bundle") and row.get("field")
    }
    return found.pop() if len(found) == 1 else None


class _Facts:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.evidence: dict[str, Any] = {}
        self.unknown: dict[str, str] = {}

    def set(self, key: str, value: Any, evidence: Any) -> None:
        self.values[key] = value
        self.evidence[key] = evidence

    def why(self, key: str, reason: str) -> None:
        self.unknown[key] = reason


def _widths(out: _Facts, body: Mapping[str, Any]) -> tuple[int | None, int | None]:
    operand, accumulator = _datapath(body, "input"), _datapath(body, "accumulator")
    operand_bits, accumulator_bits = _signed_bits(operand), _signed_bits(accumulator)
    if operand_bits:
        out.set("operand_signed_bits", operand_bits, f"rtl facts input datapath {operand}")
    else:
        out.why("operand_signed_bits", f"the input datapath {operand!r} is not a signed integer format")
    if operand_bits and accumulator_bits:
        # Every extended-K prefix of signed operand products stays an exact accumulator value.
        max_k = ((1 << (accumulator_bits - 1)) - 1) // (1 << (operand_bits - 1)) ** 2
        out.set(
            "k_extension_within",
            {"max_k": max_k, "partial_sum_bits": accumulator_bits},
            f"rtl facts datapaths input {operand} and accumulator {accumulator}",
        )
    else:
        out.why("k_extension_within", f"the accumulator datapath {accumulator!r} is not a signed integer format")
    return operand_bits, accumulator_bits


def _mesh(out: _Facts, body: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    semantics = spec.get("numerical_semantics") or {}
    mesh = (semantics.get("native_unit_contract") or {}).get("mesh_mac") or {}
    internal = semantics.get("internal_arithmetic") or {}
    bits = mesh.get("signed_bits")
    if type(bits) is int and bits > 1 and internal.get("mac_result_bits") in (None, bits):
        out.set(
            "mesh_signed_bits",
            bits,
            {"declaration": "numerical_semantics.native_unit_contract.mesh_mac", "status": mesh.get("evidence")},
        )
    else:
        out.why("mesh_signed_bits", "the software spec declares no mesh MAC width agreeing with its MAC result width")
    structure = (body.get("timing_structure") or {}).get("dataflow") or {}
    if structure.get("status") == "derived" and structure.get("support") == "both":
        names = sorted({DATAFLOW_NAMES[m] for m in (structure.get("modes") or {}).values() if m in DATAFLOW_NAMES})
        out.set("dataflows", names, {"rtl": "timing_structure.dataflow", "method": structure.get("method")})
        return
    declared = mesh.get("dataflows")
    if isinstance(declared, list) and declared and all(isinstance(d, str) and d for d in declared):
        out.set(
            "dataflows",
            list(declared),
            {
                "declaration": "numerical_semantics.native_unit_contract.mesh_mac.dataflows",
                "status": mesh.get("evidence"),
                "rtl": structure.get("reason") or "timing_structure.dataflow is not in the selected facts",
            },
        )
    else:
        out.why("dataflows", "neither the RTL timing structure nor the software spec establishes the dataflows")


def _storage(out: _Facts, body: Mapping[str, Any], contract: Mapping[str, Any], operand_bits: int | None) -> None:
    probe = ((contract.get("rtl_extraction") or {}).get("firrtl_role_probe") or {}).get("memory") or {}
    memory = next(
        (m for m in body.get("memories") or () if isinstance(m, Mapping) and m.get("name") == probe.get("name")), None
    )
    capacity = (memory or {}).get("bytes")
    if type(capacity) is int and capacity > 0:
        out.set("shared_input_resident", {"capacity_bytes": capacity}, f"rtl facts memory {memory.get('evidence')}")
    else:
        out.why("shared_input_resident", f"the operand memory {probe.get('name')!r} has no derived capacity")
    payload = ((contract.get("memory_model") or {}).get("dma") or {}).get("max_transfer_bytes")
    arrays = [a for a in body.get("arrays") or () if isinstance(a, Mapping)]
    edge = arrays[0].get("cols") if len(arrays) == 1 else None
    if type(payload) is int and payload > 0 and type(edge) is int and edge > 0 and operand_bits:
        tiles = payload // (edge * ((operand_bits + 7) // 8))
        if tiles >= 1:
            out.set(
                "weight_panel_width",
                tiles * edge,
                f"contract memory_model.dma.max_transfer_bytes {payload}; rtl array edge {edge}",
            )
            return
    out.why("weight_panel_width", "transfer payload, one array edge or operand width is not established")


def _pooling(out: _Facts, body: Mapping[str, Any], contract: Mapping[str, Any], path: Mapping[str, Any]) -> None:
    pool = next((s for s in (path or {}).get("readout") or () if s.get("kind") == "pool"), None)
    if pool is not None and "max" in (pool.get("reductions") or ()):
        out.set("max_requires_store_path_pooling", True, "derived output path carries a store-path max pool")
    else:
        out.why("max_requires_store_path_pooling", "the derived output path has no store-path max pool")
    widths = {}
    for key, role in (("size_bits", "pool_size"), ("stride_bits", "pool_stride")):
        located = _role_field(contract, role)
        width = _register_field_width(body, *located) if located else None
        if width is None:
            out.why("pool_window_within_register_fields", f"no derived register field carries {role}")
            return
        widths[key] = width
    out.set("pool_window_within_register_fields", widths, "rtl register layouts of the contract's pool fields")


def _placement(out: _Facts, spec: Mapping[str, Any]) -> None:
    placements: dict[str, list[str]] = {}
    for chain in (spec.get("composition_resolution") or {}).get("chains") or ():
        for producer, placed in (chain.get("producer_placement") or {}).items():
            if isinstance(placed, list) and placed:
                placements[producer] = sorted(set(placements.get(producer, ())) | set(placed))
    if placements:
        out.set("producer_placement", placements, "resolved software spec composition_resolution")
    else:
        out.why("producer_placement", "the resolved software spec derives no producer placement")


def _bypass_realization(kind, capability, contract, width, field):
    if kind == "activation":
        identity = capability.get("identity") or {}
        code = identity.get("code")
        if type(code) is int and 0 <= code < 1 << width and identity.get("cite"):
            return {"identity": identity, "contract": "readout_semantics/epilogue_capability"}
    if kind == "pool":
        disable = capability.get("disable") or {}
        code = disable.get("code")
        configured = (disable.get("probe") or {}).get("configured_by", "").rsplit(".", 1)[-1]
        if type(code) is int and 0 <= code < 1 << width and disable.get("source") and configured == field:
            return {"disable": disable, "contract": "readout_semantics/epilogue_capability"}
    if kind == "scale":
        return _scale_identity(capability, contract, width)
    return None


def _scale_identity(capability, contract, width):
    from merlin.common import quant_formats as qf

    abi = (contract.get("readout_semantics") or {}).get("scalar_abi") or {}
    if capability.get("dtype") != abi.get("scale_dtype") or capability.get("dtype") != "f32":
        return None
    if width != qf.machine_bits(capability["dtype"]):
        return None
    identities = {
        name: value
        for name, value in ((contract.get("rocc_operand_roles") or {}).get("isa_constants") or {}).items()
        if type(value.get("f32_bits")) in (int, float) and value["f32_bits"] == 1.0
    }
    if identities:
        return {"identity": identities, "contract": "readout_semantics/scalar_abi", "operation": "multiply by one"}
    return None


def _bypasses(out: _Facts, body, contract, path) -> None:
    epilogue = contract.get("output_epilogue") or {}
    semantics = contract.get("readout_semantics") or {}
    output_dtype = next((s.get("to") for s in path.get("readout", []) if s["kind"] == "requant"), None)
    applies = {
        stage
        for row in semantics.get("epilogue_capability", [])
        if row.get("selector") == output_dtype and row.get("evidence")
        for stage in row.get("applies", [])
    }
    for index, stage in enumerate(path.get("readout") or ()):
        key, kind = f"bypass.{index}", stage["kind"]
        if kind in ("convert", "requant"):
            out.why(
                key,
                f"{kind} has no identity on the narrow path; the full-width accumulator exit "
                "bypasses the entire readout, not an individual narrow-path stage",
            )
            continue
        capability = epilogue.get(kind) or {}
        role = {"activation": "acc_act", "scale": "acc_scale", "pool": "pool_stride"}.get(kind)
        located = _role_field(contract, role) if role else None
        width = _register_field_width(body, *located) if located else None
        evidence = {"rtl": f"register_bundle_layouts/{located[0]}/{located[1]}"} if width else {}
        operation = {"activation": "relu", "scale": "acc_scale", "pool": "maxpool"}.get(kind)
        if capability.get("status") != "derived" or not width or operation not in applies:
            out.why(key, f"{kind} has no derived control field and readout semantics")
            continue
        realization = _bypass_realization(kind, capability, contract, width, located[1])
        if realization:
            out.set(key, True, {**evidence, **realization})
        else:
            out.why(key, f"{kind} has no established identity or disable realization")


def _mean(out: _Facts, body, spec, path) -> None:
    from merlin.compile.composition_fusion import mean_realization

    schemes = ((spec.get("quantization") or {}).get("schemes") or {}).get("schemes") or []
    if not any(s.get("readout_strategy") == "f32_product_rne" for s in schemes):
        out.why("mean_realization", "no declared scheme supplies the ones-matmul single f32 boundary")
        return
    try:
        value = mean_realization(
            path, operand_dtype=_datapath(body, "input"), accumulator_bits=_signed_bits(_datapath(body, "accumulator"))
        )
    except ValueError as exc:
        out.why("mean_realization", str(exc))
        return
    out.set(
        "mean_realization",
        value,
        {"plan": "composition_fusion._mean_plan", "rtl": "datapaths", "contract": "output_epilogue/output_path"},
    )


def load_state_mapping(*, facts, contract, isa):
    """Derive the audit's load-state facts, refusing incomplete source selections."""
    try:
        return _load_state_mapping(facts=facts, contract=contract, isa=isa)
    except (KeyError, TypeError, StopIteration) as error:
        raise ValueError("UNKNOWN: incomplete load-state source selection") from error


def _load_state_mapping(*, facts, contract, isa):
    """Bind load classes to the configured stride and scale banks in RTL.

    The target selects source wires; the existing structural reader proves the
    state selector and bank writes. Scale must use that very same bank index.
    """
    from dataclasses import replace

    from merlin.common.quant_formats import storage_bits
    from merlin.targetgen.rtl import firrtl_struct as F
    from merlin.targetgen.rtl.firrtl_load_states import derive_load_state_mapping
    from merlin.targetgen.rtl.firrtl_operand_slices import _source, derive_operand_layouts

    body = _body(facts)
    probe = (contract.get("rtl_extraction") or {}).get("load_states")
    if not probe:
        raise ValueError("UNKNOWN: load-class state selection is underived")
    source = body.get("source") or {}
    layouts = derive_operand_layouts(
        source["fir_path"],
        expected_sha256=source["fir_sha256"],
        selections=probe["fields"],
        word_bits=next(
            row["bundles"][probe["selection"]["config_field"]["bundle"]]["width"]
            for row in body["interfaces"]
            if row.get("name") == "register_bundle_layouts"
        ),
    )
    mapping = derive_load_state_mapping(layouts, probe["selection"], isa)
    selection, scale = probe["selection"], probe["scale"]
    module = F.parse_module(
        selection["module"], F.read_module_bodies(mapping.source, [selection["module"]])[selection["module"]]
    )
    bank = F.Ref(scale["register"], (None,), (F.parse_expr(selection["state"]),))
    stride = F.Ref(selection["stride_register"], (None,), (F.parse_expr(selection["state"]),))
    writers = [c for c in module.connects if c.lhs.root == scale["register"]]
    stride_writers = [c for c in module.connects if c.lhs == stride]
    sinks = [c for c in module.connects if c.lhs == F.parse_expr(scale["sink"])]
    declaration = module.decls.get(scale["register"])
    typ = F.parse_type(declaration[1]) if declaration and declaration[0] == "reg" else None
    field = next((r for r in layouts["fields"] if r["field"] == scale["input"]), {})
    decoded = [r for r in isa["OPERAND_FIELDS"][selection["config_class"]]["fields"] if r["name"] == "scale_bits"]
    if (
        typ != ("vec", ("uint", field.get("width")), len(mapping.states))
        or field.get("status") != "derived"
        or len(decoded) != 1
        or decoded[0].get("operand") != "rs1"
        or any(decoded[0].get(k) != field.get(k) for k in ("offset", "width"))
        or len(writers) != 1
        or writers[0].lhs != bank
        or writers[0].rhs != F.parse_expr(scale["input"])
        or len(stride_writers) != 1
        or writers[0].guards != stride_writers[0].guards
        or len(sinks) != 1
        or sinks[0].rhs != bank
        or sinks[0].guards
    ):
        raise ValueError("UNKNOWN: load scale does not use the configured load-state bank")
    bits = storage_bits(_datapath(body, "accumulator"))
    if bits <= 0 or bits % 8:
        raise ValueError("UNKNOWN: accumulator load storage width is unresolved")
    _source(mapping.source, mapping.source_sha256)
    return replace(mapping, max_element_bytes=bits // 8)


def derive_runtime_readout_scale(*, facts, contract, output_path) -> dict:
    """Locate the selected runtime scale slot using roles, RTL layout and scalar ABI.

    Parametric layouts use slot_width only after the selected f32 ABI fixes the
    effective storage width. Authored capability assertions alone are insufficient.
    """
    from merlin.common.quant_formats import storage_bits
    from merlin.verify.lean_contract import document_digest

    body, contract, path = _body(facts), contract or {}, output_path or {}
    evidence = []

    def unknown(reason):
        return {"status": "unknown", "reason": reason, "evidence": evidence}

    scales = [s for s in path.get("readout", ()) if s.get("kind") == "scale"]
    if (
        len(scales) != 1
        or scales[0].get("dtype") != "f32"
        or scales[0].get("granularity") not in ("per_command", "tensor")
    ):
        return unknown("selected output path has no tensor-wide f32 scale")
    evidence.append({"output_path_sha256": document_digest(path)})
    roles = ((contract.get("rocc_operand_roles") or {}).get("classes") or {}).get("CONFIG_ST") or {}
    selected = [r for r in roles.get("fields", ()) if r.get("name") in ("acc_scale", "acc_scale_bits")]
    locations = {(r.get("operand"), r.get("bundle"), r.get("field")) for r in selected}
    if len(locations) != 1 or not selected:
        return unknown("CONFIG_ST scale roles are absent or conflict")
    operand, bundle, field = locations.pop()
    if operand != "rs2" or not isinstance(bundle, str) or not isinstance(field, str) or not bundle or not field:
        return unknown("CONFIG_ST scale role does not select an rs2 register field")
    if any(r.get("as", "f32") != "f32" for r in selected):
        return unknown("CONFIG_ST scale role conflicts with f32 storage")
    evidence.append({"contract": "rocc_operand_roles.classes.CONFIG_ST.fields", "roles": selected})
    layouts = [i for i in body.get("interfaces", ()) if i.get("name") == "register_bundle_layouts"]
    if len(layouts) != 1 or not layouts[0].get("evidence") or not layouts[0].get("source"):
        return unknown("selected register layout has no unique source evidence")
    layout = layouts[0]
    placed = ((layout.get("bundles") or {}).get(bundle) or {}).get("fields", {}).get(field)
    if not isinstance(placed, Mapping):
        return unknown("selected scale register field is missing")
    abi = (contract.get("readout_semantics") or {}).get("scalar_abi") or {}
    if (
        abi.get("schema") != "scalar_narrow_readout_contract_v1"
        or abi.get("scale_dtype") != "f32"
        or abi.get("accumulator_dtype") != "i32"
        or abi.get("output_dtype") != "i8"
        or abi.get("clamp_min") != -128
        or abi.get("clamp_max") != 127
    ):
        return unknown("selected scalar ABI does not establish signed i32/f32/i8 readout")
    bits = storage_bits(abi["scale_dtype"])

    def extent(row):
        size = row.get("width")
        if size is None:
            size = row.get("slot_width")
        return row.get("offset"), size

    offset, width = extent(placed)
    register = ((layout.get("bundles") or {}).get(bundle) or {}).get("width")
    others = [
        extent(row)
        for name, row in ((layout.get("bundles") or {}).get(bundle) or {}).get("fields", {}).items()
        if name != field and isinstance(row, Mapping)
    ]
    if (
        type(width) is not int
        or type(offset) is not int
        or type(register) is not int
        or width != bits
        or bits != 32
        or not 0 <= offset <= register - width
        or ("slot_width" in placed and (type(placed["slot_width"]) is not int or placed["slot_width"] != bits))
    ):
        return unknown("selected f32 scale slot is not one whole binary32 field inside its register bundle")
    if any(
        type(start) is not int or type(size) is not int or start < offset + width and offset < start + size
        for start, size in others
    ):
        return unknown("selected f32 scale slot overlaps another field of its register bundle")
    evidence.extend(
        [
            {
                "rtl": f"interfaces.register_bundle_layouts.bundles.{bundle}.fields.{field}",
                "layout": dict(placed),
                "source": layout["source"],
                "evidence": layout["evidence"],
            },
            {"contract": "readout_semantics.scalar_abi", "value": dict(abi), "storage_bits": bits},
        ]
    )
    return {
        "status": "derived",
        "value": {
            "schema": "merlin.runtime_readout_scale.v1",
            "dtype": "f32",
            "granularity": "per_command",
            "operand": operand,
            "bundle": bundle,
            "field": field,
            "offset": offset,
            "width": width,
            "encoding": "ieee754_binary32_bits",
        },
        "evidence": evidence,
    }


def derive(
    *,
    facts: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    software_spec: Mapping[str, Any] | None,
    output_path: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``{"facts": {...}, "evidence": {...}, "unknown": {...}}`` for composition generation."""
    body, contract, spec = _body(facts), contract or {}, software_spec or {}
    out = _Facts()
    operand_bits, _accumulator = _widths(out, body)
    _mesh(out, body, spec)
    _storage(out, body, contract, operand_bits)
    _pooling(out, body, contract, output_path or {})
    _placement(out, spec)
    _bypasses(out, body, contract, output_path or {})
    _mean(out, body, spec, output_path or {})
    if any(stage.get("kind") == "scale" for stage in (output_path or {}).get("readout", ())):
        runtime = derive_runtime_readout_scale(facts=facts, contract=contract, output_path=output_path)
        if runtime["status"] == "derived":
            out.set("runtime_readout_scale", runtime["value"], runtime["evidence"])
        else:
            out.why("runtime_readout_scale", runtime["reason"])
    return {"facts": out.values, "evidence": out.evidence, "unknown": dict(sorted(out.unknown.items()))}
