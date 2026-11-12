"""Per-framework contract descriptors — the caller-side assumptions (prepack/transpose/layout/
accumulator/dtype) that are NOT in a kernel's body or assembly, so they can't be mined from code
alone. Hand-authored once per framework (~the XNNPACK-transpose knowledge), agent-refined, and
loaded by the dossier so the agent reads the contract alongside the code facts.

The ``feature_extraction/`` subdir holds built-in, target-neutral ISA-class contracts. A
target-specific contract is instead selected explicitly from its target owner with
``use_feature_contract``. No checkout or target-name lookup supplies one implicitly.

Which ``kernel.source`` spellings name which framework is not held here: it belongs to the corpus
registry (``src/merlin/resources/contracts/corpora.yaml``, read through :mod:`merlin.development.targetgen.corpora`), so a source
alias and the corpus location it selects cannot drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common.yaml import load_yaml

_DIR = Path(__file__).resolve().parent
_FEATURE_DIR = _DIR / "feature_extraction"
_SELECTED_FEATURE: ContextVar[tuple[dict[str, Any], str, Path] | None] = ContextVar(
    "selected_feature_contract", default=None
)
_SELECTED_FRAMEWORK: ContextVar[tuple[dict[str, Any], str, Path] | None] = ContextVar(
    "selected_framework_contract", default=None
)


def _framework_stem(framework: str | None) -> str:
    """Contract file stem for a source/framework spelling: the framework a registered corpus alias names,
    else the spelling itself (lower-cased)."""
    from merlin.development.targetgen.corpora import kernel_corpus_for_source

    name = (framework or "").lower()
    return kernel_corpus_for_source(name) or name


@cache
def _load_builtin_contract(framework: str) -> dict[str, Any]:
    """Load a framework contract by source/framework name. Returns {} if none exists (the kernel
    simply has no recorded caller contract — e.g. an unmapped source)."""
    path = _DIR / f"{_framework_stem(framework)}.yaml"
    if not path.is_file():
        return {}
    return load_contract_file(path)


def load_contract(framework: str) -> dict[str, Any]:
    """Read a built-in source-class contract or an explicit target-owned caller contract."""
    selected = _SELECTED_FRAMEWORK.get()
    if selected is not None and _framework_stem(framework) == selected[0]["framework"]:
        return selected[0]
    return _load_builtin_contract(framework)


def load_contract_file(path: Path) -> dict[str, Any]:
    return load_yaml(path) or {}


def available_frameworks() -> list[str]:
    names = {p.stem for p in _DIR.glob("*.yaml")}
    selected = _SELECTED_FRAMEWORK.get()
    if selected is not None:
        names.add(selected[0]["framework"])
    return sorted(names)


def selected_framework_contract() -> dict[str, str] | None:
    selected = _SELECTED_FRAMEWORK.get()
    if selected is None:
        return None
    doc, digest, path = selected
    return {"framework": doc["framework"], "sha256": digest, "path": str(path)}


@contextmanager
def use_framework_contract(path: Path | str) -> Iterator[dict[str, str]]:
    """Select an exact caller-side framework declaration without ambient discovery."""
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    doc = load_contract_file(source)
    if not isinstance(doc, dict) or not isinstance(doc.get("framework"), str) or not doc["framework"]:
        raise ValueError(f"framework contract {source} needs a framework name")
    framework = doc["framework"].lower()
    if (_DIR / f"{framework}.yaml").is_file():
        raise ValueError(f"selected framework contract may not replace built-in source class {framework!r}")
    doc["framework"] = framework
    token = _SELECTED_FRAMEWORK.set((doc, sha256(raw).hexdigest(), source))
    try:
        yield selected_framework_contract() or {}
    finally:
        _SELECTED_FRAMEWORK.reset(token)


def feature_families() -> list[str]:
    """Built-in ISA classes plus the explicitly selected target family, if any."""
    families = {p.stem for p in _FEATURE_DIR.glob("*.yaml")}
    selected = _SELECTED_FEATURE.get()
    if selected is not None:
        families.add(selected[0]["family"])
    return sorted(families)


@cache
def _load_builtin_feature_contract(family: str) -> dict[str, Any]:
    path = _FEATURE_DIR / f"{(family or '').lower()}.yaml"
    if not path.is_file():
        return {}
    return load_contract_file(path)


def load_feature_contract(family: str) -> dict[str, Any]:
    """Read a built-in class or an explicitly selected target-owned feature contract."""
    selected = _SELECTED_FEATURE.get()
    if selected is not None:
        if family.lower() == selected[0]["family"]:
            return selected[0]
        extension = (selected[0].get("feature_extensions") or {}).get(family.lower())
        if extension:
            base = _load_builtin_feature_contract(family)
            merged = dict(base)
            for key in ("schedule_directives",):
                if key in extension:
                    merged[key] = list(dict.fromkeys([*(base.get(key) or []), *extension[key]]))
            if "markers" in extension:
                markers = {motif: list(patterns) for motif, patterns in (base.get("markers") or {}).items()}
                for motif, patterns in extension["markers"].items():
                    markers[motif] = list(dict.fromkeys([*markers.get(motif, []), *patterns]))
                merged["markers"] = markers
            return merged
    return _load_builtin_feature_contract(family)


def selected_feature_contract() -> dict[str, Any] | None:
    """Return exact selected-file identity for a mining receipt; never search for a target."""
    selected = _SELECTED_FEATURE.get()
    if selected is None:
        return None
    doc, digest, path = selected
    return {"family": doc["family"], "targets": doc["targets"], "sha256": digest, "path": str(path)}


@contextmanager
def use_feature_contract(path: Path | str) -> Iterator[dict[str, Any]]:
    """Select one target-owned feature contract for this extraction scope only.

    The file is mandatory, parsed once, and cannot override a built-in class or claim one
    of its target spellings. Consumers must pass the path explicitly; no ambient lookup.
    """
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    doc = load_contract_file(source)
    if not isinstance(doc, dict):
        raise ValueError(f"feature contract {source} must be a mapping")
    family = str(doc.get("family", "")).lower()
    if not family or not isinstance(doc.get("targets"), list) or not doc["targets"]:
        raise ValueError(f"feature contract {source} needs family and non-empty targets")
    if family in {p.stem for p in _FEATURE_DIR.glob("*.yaml")}:
        raise ValueError(f"selected feature contract may not replace built-in ISA class {family!r}")
    from merlin.analysis.kernels.markers import MOTIFS, _target_families

    overlap = set(map(str.lower, map(str, doc["targets"]))) & set(_target_families())
    if overlap:
        raise ValueError(f"selected feature contract claims built-in target spelling(s): {sorted(overlap)}")
    unknown = set(doc.get("markers") or {}) - set(MOTIFS)
    if unknown:
        raise ValueError(f"selected feature contract has unknown motif(s): {sorted(unknown)}")
    extensions = doc.get("feature_extensions") or {}
    if not isinstance(extensions, dict):
        raise ValueError("feature_extensions must be a mapping of built-in ISA classes")
    for base_family, extension in extensions.items():
        if not _load_builtin_feature_contract(str(base_family)) or not isinstance(extension, dict):
            raise ValueError(f"feature extension {base_family!r} needs a built-in ISA class")
        extra = set(extension) - {"markers", "schedule_directives"}
        if extra:
            raise ValueError(f"feature extension {base_family!r} has non-additive fields: {sorted(extra)}")
        markers = extension.get("markers") or {}
        if not isinstance(markers, dict) or set(markers) - set(MOTIFS):
            raise ValueError(f"feature extension {base_family!r} has invalid marker vocabulary")
        for patterns in markers.values():
            if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
                raise ValueError(f"feature extension {base_family!r} marker patterns must be strings")
            for pattern in patterns:
                re.compile(pattern)  # regex-ok: validate explicitly authored marker data
        directives = extension.get("schedule_directives") or []
        if not isinstance(directives, list) or not all(isinstance(d, str) and d for d in directives):
            raise ValueError(f"feature extension {base_family!r} directives must be non-empty strings")
    doc["family"] = family
    token = _SELECTED_FEATURE.set((doc, sha256(raw).hexdigest(), source))
    try:
        yield selected_feature_contract() or {}
    finally:
        _SELECTED_FEATURE.reset(token)


def verify_index_feature_contract(index: dict[str, Any], selected: dict[str, Any] | None) -> None:
    """Fail closed when reprocessing a target index with missing/different feature bytes."""
    recorded = index.get("feature_contract")
    if recorded is not None:
        if selected is None or any(recorded.get(k) != selected.get(k) for k in ("family", "sha256")):
            raise ValueError("index feature contract requires the same explicitly selected YAML bytes")
    elif selected is not None and str(index.get("target", "")).lower() in {
        str(target).lower() for target in selected["targets"]
    }:
        raise ValueError("target index has no feature-contract identity; re-index with explicit selection")
