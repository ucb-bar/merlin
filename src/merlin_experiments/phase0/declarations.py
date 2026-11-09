"""Explicit derivation inputs declared by experiments, without reading private sidecars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from merlin_experiments.spec import ExperimentSpec, SpecError, catalog, load_spec


@dataclass(frozen=True)
class DerivationDeclaration:
    definition: Path
    id: str
    target: str
    profile: str
    descriptor: Path
    recipe: Path
    performance_template: Path
    conformance_spec: Path | None
    synth_profile: Path | None
    smt_profile: Path | None
    hidden_profile: Path | None

    def profile_inputs(self) -> dict[str, Path | None]:
        return {
            name: getattr(self, name)
            for name in (
                "recipe",
                "performance_template",
                "conformance_spec",
                "synth_profile",
                "smt_profile",
                "hidden_profile",
            )
        }


def _declaration(spec: ExperimentSpec) -> DerivationDeclaration:
    phase = spec.document["phases"].get("0")
    if spec.document.get("kind") == "template" or not phase or phase["adapter"] != "capsule_derivation":
        raise SpecError(f"{spec.path}: not a concrete capsule derivation definition")
    config = phase["config"]
    for name in ("descriptor", "recipe", "performance_template"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise SpecError(f"{spec.path}: explicit phase-0 {name} is required")
    # Declarations are lexical inputs, not an admission check. Path.resolve() probes
    # the filesystem (including private holdouts) and erases dangling aliases.
    # The consuming loader/verifier checks the inputs it is authorized to open.
    paths = {
        name: Path(os.path.abspath(spec.path.parent / Path(config[name]).expanduser())) if config.get(name) else None
        for name in (
            "descriptor",
            "recipe",
            "performance_template",
            "conformance_spec",
            "synth_profile",
            "smt_profile",
            "hidden_profile",
        )
    }
    return DerivationDeclaration(spec.path, spec.id, spec.target, config.get("profile", spec.target), **paths)


def from_definition(path: str | Path) -> DerivationDeclaration:
    return _declaration(load_spec(path))


def all_declarations(*, catalog_path: Path | None = None) -> tuple[DerivationDeclaration, ...]:
    declarations = []
    for name, path in catalog(catalog_path).items():
        spec = load_spec(path)
        if spec.id != name:
            raise SpecError(f"catalog id {name!r} does not match definition id {spec.id!r}")
        phase = spec.document["phases"].get("0")
        if spec.document.get("kind") == "template" or not phase or phase["adapter"] != "capsule_derivation":
            continue
        declarations.append(_declaration(spec))
    return tuple(declarations)


def for_target(selector: str, *, catalog_path: Path | None = None) -> DerivationDeclaration:
    matches = [item for item in all_declarations(catalog_path=catalog_path) if selector in (item.target, item.profile)]
    if len(matches) != 1:
        raise SpecError(f"phase-0 selector {selector!r} requires one declaration; found {len(matches)}")
    return matches[0]
