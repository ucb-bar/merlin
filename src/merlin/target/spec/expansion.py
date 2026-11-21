"""Software-spec selection, expansion and normalization."""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from merlin.target.spec.semantic_sources import (
    Composition,
    Source,
    SpecError,
    canonical,
    digest,
    fields,
    parse,
    read_source,
    sha256,
)

# ---- load --------------------------------------------------------------------------------------
# Explicit compact-spec selection, source-bound expansion and inspection.
#
# This milestone composes existing v1 records, not arbitrary numerical programs.
# Typed scalar computations use the existing instruction-semantics representation;
# their v1 executable projection remains explicitly unsupported. Source review and
# structural projection confer no hardware qualification or equivalence proof.


def _registry(expected: object = None, *, legacy: bool = False) -> tuple[Source, dict, tuple[Source, ...]]:
    from merlin.common import target_data as quant_formats
    from merlin.common.paths import schemas_dir

    overlay = os.environ.get("MERLIN_QUANT_FORMATS")
    if overlay and not legacy:
        raise SpecError(
            "compact specs do not yet admit MERLIN_QUANT_FORMATS overlays; "
            "select legacy v1 or remove the ambient overlay"
        )
    path = schemas_dir() / "quant_formats.registry.yaml"
    snapshot = Source(path, read_source(path), "format_registry")
    if expected is not None and snapshot.sha256 != sha256(expected, "format_registry_sha256"):
        raise SpecError("selected format registry differs from the authored SHA-256 pin")
    document = parse(snapshot.raw, str(path))
    raw_formats = dict(document["formats"])
    overlay_sources = ()
    if overlay:
        overlay_path = Path(overlay).expanduser().resolve()
        overlay_source = Source(overlay_path, read_source(overlay_path), "format_overlay")
        raw_formats.update(parse(overlay_source.raw, str(overlay_path))["formats"])
        overlay_sources = (overlay_source,)
    frozen = {name: quant_formats._build(name, row) for name, row in raw_formats.items()}
    if quant_formats.registry() != frozen:
        raise SpecError("ambient/cached quant-format registry differs from selected source bytes")
    aliases = {}
    for name, fmt in frozen.items():
        for alias in (name, *fmt.aliases):
            if alias in aliases and aliases[alias] != name and not legacy:
                raise SpecError(f"ambiguous selected quant-format alias {alias!r}")
            aliases[alias] = name
    if any(quant_formats.get(alias) != frozen[name] for alias, name in aliases.items()):
        raise SpecError("cached quant-format alias resolution differs from selected source bytes")
    return (
        snapshot,
        {
            "path": str(path),
            "sha256": snapshot.sha256,
            "formats": {name: asdict(fmt) for name, fmt in frozen.items()},
            "aliases": aliases,
            "overlays": [{"path": str(source.path), "sha256": source.sha256} for source in overlay_sources],
        },
        overlay_sources,
    )


def _tools() -> tuple[Source, ...]:
    from merlin.common import target_data as quant_formats
    from merlin.target.isa import spec as isa_spec
    from merlin.target.spec import instructions, quantization, semantic_sources
    from merlin.target.spec import software as software_spec
    from merlin.target.spec import software as transfer_contracts

    modules = (
        semantic_sources,
        software_spec,
        instructions,
        isa_spec,
        quantization,
        quant_formats,
        transfer_contracts,
    )
    sources = [Source(Path(__file__).resolve(), Path(__file__).read_bytes(), __name__)]
    sources.extend(
        Source(Path(module.__file__).resolve(), Path(module.__file__).read_bytes(), module.__name__)
        for module in modules
    )
    return tuple(sources)


@dataclass(frozen=True)
class ExpandedSpec:
    """An immutable captured expansion with separate raw and canonical identities."""

    _record_json: bytes
    _sources: tuple[Source, ...]
    _registry: Source
    _tools: tuple[Source, ...]

    def verify_sources(self) -> None:
        for source in (*self._sources, self._registry, *self._tools):
            source.verify()
        record = self.record(expanded=True)
        _, current_registry, _ = _registry(
            self._registry.sha256, legacy=record["source_schema"] == "merlin.software_spec.v1"
        )
        if canonical(current_registry) != canonical(record["format_registry"]):
            raise SpecError("selected format registry/overlay binding changed after expansion")

    def record(self, *, expanded: bool = False) -> dict:
        """Return the captured snapshot; use verify_sources for fresh custody checks."""
        import json

        record = json.loads(self._record_json)
        if not expanded:
            record.pop("software_spec")
            record.pop("computations")
            record.pop("authored_text")
            record["format_registry"].pop("formats")
            record["format_registry"].pop("aliases")
            for source in record["sources"]:
                source.pop("authored_text")
            for definition in record["definitions"]:
                definition.pop("value")
        return record

    def source_references(self) -> dict[str, Path]:
        """Exact imported/data/tool files needed to reproduce this projection."""
        self.verify_sources()
        references = {f"semantic_import_{index}": source.path for index, source in enumerate(self._sources[1:])}
        references["semantic_format_registry"] = self._registry.path
        references.update({f"semantic_tool_{index}": source.path for index, source in enumerate(self._tools)})
        return references

    def software_spec(self) -> dict:
        """Return an existing v1 consumer view only when every selected item projects."""
        self.verify_sources()
        record = self.record(expanded=True)
        if record["projection"]["status"] != "supported":
            raise SpecError("unsupported compact-spec projection: " + "; ".join(record["projection"]["obligations"]))
        result = record["software_spec"]
        self.verify_sources()
        return result


def load_spec(path: str | Path, target: str | None = None) -> ExpandedSpec:
    """Load exactly one explicit v2 authority and its hash-pinned definition imports.

    Application preparation profiles are selected by the recipe caller. They are
    not discovered here or merged into the target's numerical semantics.
    """

    source_path = Path(path).expanduser().resolve()
    raw = read_source(source_path)
    document = parse(raw, str(source_path), legacy=True)
    legacy = document.get("schema") == "merlin.software_spec.v1"
    if not legacy:
        document = parse(raw, str(source_path))
    registry_source, registry_record, overlays = _registry(
        document.get("format_registry_sha256") if not legacy else None, legacy=legacy
    )
    if legacy:
        from merlin.target.spec.software import validate_software_spec

        candidate = validate_software_spec(document, target=target, source=source_path)
        computations, obligations, definitions = {}, [], []
        sources = [Source(source_path, raw, "")]
    else:
        composition = Composition(source_path)
        if composition.sources[0].raw != raw:
            raise SpecError("selected root source changed while loading")
        candidate, computations, obligations, definitions = normalize(composition)
        sources = composition.sources
    if target is not None and candidate["target"] != target:
        raise SpecError("compact spec target differs from selected target")
    tools = _tools()
    root = sources[0]
    record = {
        "schema": "merlin.software_spec_expansion.v1",
        "source_schema": document["schema"],
        "target": candidate["target"],
        "status": candidate["status"],
        "projection": {"status": "unsupported" if obligations else "supported", "obligations": obligations},
        "qualification": "not_established",
        "scope": "structural expansion and existing-consumer projection",
        "tool_versions": {"python": platform.python_version(), "pyyaml": yaml.__version__},
        "source_sha256": root.sha256,
        "canonical_sha256": digest(candidate),
        "sources": [source.record() for source in sources + list(overlays)],
        "tools": [{"module": source.namespace, "sha256": source.sha256} for source in tools],
        "format_registry": registry_record,
        "definitions": definitions,
        "authored_text": root.raw.decode("utf-8"),
        "software_spec": candidate,
        "computations": computations,
    }
    from merlin.target.spec.quantization import semantic_extensions

    extensions = semantic_extensions(candidate)
    if extensions:
        record["semantic_extensions"] = extensions
    record["expansion_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
    result = ExpandedSpec(canonical(record), tuple(sources) + overlays, registry_source, tools)
    result.verify_sources()
    return result


# ---- normalize ---------------------------------------------------------------------------------
# Lossless composition into the established numerical and ISA record formats.

_NUMERICAL = {
    "model",
    "operand_dtype",
    "accumulator_dtype",
    "readout_dtype",
    "subnormal_operand_flush",
    "input_domain",
    "output_domain",
    "rounding",
    "reduction_order",
    "reduction_cadence",
    "product_rounding",
    "overflow",
    "internal_arithmetic",
    "readout",
    "block_size",
    "scale_encoding",
    "operand_rounding",
    "scale_rule",
    "bf16_source_subnormal_flush",
    "zero_block_scale_e8m0",
}
_OPERATION = {
    "id",
    "ops",
    "families",
    "family",
    "placement",
    "numerical_contract",
    "evidence",
    "status",
    "description",
    "hardware",
    "signature",
}
_FORMAT = {
    "id",
    "operand_dtype",
    "accumulator_dtype",
    "eligible_operations",
    "unit",
    "numerical_semantics",
    "framework",
    "evidence",
    "description",
    "activation_zero_point",
    "weight_zero_point",
    "block_size",
    "subnormal_operand_flush",
    "activation_mode",
    "weight_mode",
    "site_modes",
    "scale_encoding",
    "activation_granularity",
    "weight_granularity",
}


def check_record(value: dict, kind: str) -> None:
    """Close lossy v1 boundaries before invoking their existing normalizers."""
    from merlin.target.spec.software import _SIGNATURE_FIELDS, validate_numerical_semantics

    if kind == "numerical_semantics":
        fields(value, _NUMERICAL, kind)
        fields(value.get("model"), {"engine", "source_root_env", "source_root_path"}, "numerical model", {"engine"})
        if "internal_arithmetic" in value:
            internal = fields(
                value["internal_arithmetic"],
                {
                    "signed_operand_bits",
                    "mac_result_bits",
                    "mac_result_overflow",
                    "full_operation_overflow_policy",
                    "operand_domain",
                    "addend_domain",
                    "output_overflow",
                    "product",
                    "accumulator",
                    "cross_tile",
                },
                "internal arithmetic",
            )
            if "operand_domain" in internal:
                fields(internal["operand_domain"], {"exponent_range", "signed_zero"}, "operand domain")
        if "readout" in value:
            fields(
                value["readout"],
                {"acc_scale_rounding", "narrowing", "requant_rounding", "hardware_integer_shift_rounding", "shift"},
                "numerical readout",
            )
        validate_numerical_semantics(value)
    elif kind == "signature":
        fields(value, _SIGNATURE_FIELDS, kind)
    elif kind == "operation":
        fields(value, _OPERATION | set(_SIGNATURE_FIELDS), kind)
        if "signature" in value:
            check_record(value["signature"], "signature")
    elif kind == "quantization_format":
        fields(value, _FORMAT, kind)
        if "numerical_semantics" in value:
            check_record(value["numerical_semantics"], "numerical_semantics")
    elif kind == "quantization_element_formats":
        from merlin.target.spec.quantization import validate_element_formats

        validate_element_formats(value)
    elif kind == "quantization_output_requantization":
        from merlin.target.spec.quantization import validate_output_requantization

        validate_output_requantization(value)
    elif kind == "isa":
        from merlin.target.isa.spec import validate_isa_spec

        validate_isa_spec(value)
    elif kind == "transfer_contracts":
        from merlin.target.spec.software import validate_transfer_contracts

        fields(value, {"status", "declarations"}, kind, {"declarations"})
        declarations = value["declarations"]
        rows = (
            declarations.values()
            if isinstance(declarations, dict)
            else declarations
            if isinstance(declarations, list)
            else []
        )
        for row in rows:
            fields(row, {"id", "from", "to", "copy", "valid_window", "evidence", "signature", "semantics"}, "transfer")
        validate_transfer_contracts(value)
    elif kind == "computation":
        if value.get("status") == "UNKNOWN":
            fields(value, {"status", "reason"}, kind, {"status", "reason"})
        else:
            fields(value, {"kind", "indexing_maps", "indexing_maps_explicit", "iterator_types", "scalar_body"}, kind)
            check_record(value.get("scalar_body"), "scalar_body")
        from merlin.target.spec.semantic_sources import _computation

        _computation(value, label="computation", unknowns=[])
    elif kind == "scalar_body":
        fields(value, {"arguments", "captures", "operations", "yields", "effects"}, kind)
        from merlin.target.spec.semantic_sources import _scalar_body

        _scalar_body(value, label="scalar body", unknowns=[])
        for row in value.get("operations", []):
            fields(row, {"op", "operands", "results", "attributes", "effects", "regions"}, "scalar operation")
        for row in value.get("captures", []):
            fields(row, {"source", "shape", "dtype", "result_index", "const_value"}, "scalar capture")


def normalize(composition: Composition) -> tuple[dict, dict, list[str], list[dict]]:
    from merlin.target.spec.software import SCHEMA as V1_SCHEMA
    from merlin.target.spec.software import validate_software_spec

    source = fields(
        composition.document,
        {
            "schema",
            "target",
            "status",
            "imports",
            "definitions",
            "format_registry_sha256",
            "numerical_semantics",
            "operations",
            "quantization",
            "restrictions",
            "transfer_contracts",
            "isa",
            "capability_contract",
            "evidence",
            "computations",
        },
        "software spec v2",
        {"schema", "target", "numerical_semantics", "operations"},
    )
    candidate = {"schema": V1_SCHEMA, "target": source["target"], "status": source.get("status", "unreviewed")}
    for field in ("evidence", "restrictions", "capability_contract"):
        if field in source:
            candidate[field] = source[field]
    for field in ("numerical_semantics", "isa", "transfer_contracts"):
        if field in source:
            candidate[field] = composition.expand(source[field], field)
            check_record(candidate[field], field)
    operations = source["operations"]
    if not isinstance(operations, dict) or not operations:
        raise SpecError("software spec v2 operations must be a nonempty mapping")
    candidate["operations"] = {}
    for name, row in operations.items():
        expanded = composition.expand(row, "operation")
        check_record(expanded, "operation")
        candidate["operations"][name] = expanded
    if "quantization" in source:
        quantization = fields(
            source["quantization"],
            {"formats", "element_formats", "output_requantization"},
            "v2 quantization (application methodology must be explicitly selected separately)",
            {"formats"},
        )
        if not isinstance(quantization["formats"], list):
            raise SpecError("quantization.formats must be a list")
        formats = [composition.expand(row, "quantization_format") for row in quantization["formats"]]
        for row in formats:
            check_record(row, "quantization_format")
        candidate["quantization"] = {"formats": formats}
        for field in ("element_formats", "output_requantization"):
            if field in quantization:
                kind = f"quantization_{field}"
                expanded = composition.expand(quantization[field], kind)
                check_record(expanded, kind)
                candidate["quantization"][field] = expanded
    candidate = validate_software_spec(candidate, source="compact expansion")
    selected = source.get("computations", {})
    if not isinstance(selected, dict):
        raise SpecError("computations must be a mapping of typed computation records")
    computations, obligations = {}, []
    for name, row in selected.items():
        from merlin.target.spec.semantic_sources import _computation

        expanded = composition.expand(row, "computation")
        check_record(expanded, "computation")
        unknowns = []
        computations[name] = {
            "computation": _computation(expanded, label=str(name), unknowns=unknowns),
            "unknowns": unknowns,
        }
        obligations.append(
            f"computations.{name}: no admitted v1 execution/proof projection for an authored typed computation"
        )
    # Validate every imported definition as part of the visible review closure.
    # Merely importing a definition does not select it as an operation or scheme.
    selected_names = set(composition.used)
    definitions = []
    for name, (kind, value, scope) in sorted(composition.definitions.items()):
        expanded = composition.expand(value, kind, scope=scope, stack=(name,))
        check_record(expanded, kind)
        definitions.append({"name": name, "kind": kind, "used": name in selected_names, "value": expanded})
    return candidate, computations, obligations, definitions
