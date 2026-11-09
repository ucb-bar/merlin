"""Explicit compact-spec selection, source-bound expansion and inspection.

This milestone composes existing v1 records, not arbitrary numerical programs.
Typed scalar computations use the existing instruction-semantics representation;
their v1 executable projection remains explicitly unsupported. Source review and
structural projection confer no hardware qualification or equivalence proof.
"""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from merlin.target.spec.source import Composition, Source, SpecError, canonical, digest, parse, read_source, sha256


def _registry(expected: object = None, *, legacy: bool = False) -> tuple[Source, dict, tuple[Source, ...]]:
    from merlin.common import quant_formats
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
    from merlin.common import quant_formats
    from merlin.target.spec import computation, instructions, normalize, quantization, signature, source
    from merlin.targetgen import isa_spec, software_spec, transfer_contracts

    modules = (
        source,
        normalize,
        software_spec,
        computation,
        instructions,
        signature,
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
    from merlin.target.spec.normalize import normalize

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
        from merlin.targetgen.software_spec import validate_software_spec

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
