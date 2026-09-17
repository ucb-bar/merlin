"""Per-framework contract descriptors — the caller-side assumptions (prepack/transpose/layout/
accumulator/dtype) that are NOT in a kernel's body or assembly, so they can't be mined from code
alone. Hand-authored once per framework (~the XNNPACK-transpose knowledge), agent-refined, and
loaded by the dossier so the agent reads the contract alongside the code facts.

The ``feature_extraction/`` subdir holds a second, distinct kind of contract keyed by ISA *family*
(rvv / gemmini / exo_schedule / triton / ...): the per-family data that used to be an
``if fam == "gemmini"`` branch inside ``features/{roles,dispatch,loops}.py`` — how to measure RHS
reuse, which accelerator dispatch opcodes to count, whether the tiling marker is an explicit
directive — together with the family's motif-marker patterns and the ``kernel.target`` spellings that
select it (read by ``markers.py``). Loading it as data keeps those extractors framework-agnostic (see
``load_feature_contract``); a new ISA family is a new file there, not a code edit.

Which ``kernel.source`` spellings name which framework is not held here: it belongs to the corpus
registry (``merlin/contract/corpora.yaml``, read through :mod:`merlin.targetgen.corpora`), so a source
alias and the corpus location it selects cannot drift apart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ...common.yaml import load_yaml

_DIR = Path(__file__).resolve().parent
_FEATURE_DIR = _DIR / "feature_extraction"


def _framework_stem(framework: str | None) -> str:
    """Contract file stem for a source/framework spelling: the framework a registered corpus alias names,
    else the spelling itself (lower-cased)."""
    from ...targetgen.corpora import kernel_corpus_for_source

    name = (framework or "").lower()
    return kernel_corpus_for_source(name) or name


@lru_cache(maxsize=None)
def load_contract(framework: str) -> dict[str, Any]:
    """Load a framework contract by source/framework name. Returns {} if none exists (the kernel
    simply has no recorded caller contract — e.g. an unmapped source)."""
    path = _DIR / f"{_framework_stem(framework)}.yaml"
    if not path.is_file():
        return {}
    return load_contract_file(path)


def load_contract_file(path: Path) -> dict[str, Any]:
    return load_yaml(path) or {}


def available_frameworks() -> list[str]:
    return sorted(p.stem for p in _DIR.glob("*.yaml"))


def feature_families() -> list[str]:
    """Every ISA family that ships a feature-extraction contract (the ``feature_extraction/*.yaml``
    stems), sorted."""
    return sorted(p.stem for p in _FEATURE_DIR.glob("*.yaml"))


@lru_cache(maxsize=None)
def load_feature_contract(family: str) -> dict[str, Any]:
    """Load the per-ISA-family FEATURE-EXTRACTION contract (``feature_extraction/<family>.yaml``).

    This is the data that used to be an ``if fam == ...`` branch in ``features/{roles,dispatch,
    loops}.py``: the RHS-reuse measurement method, the accelerator dispatch opcode set, and whether
    the tiling marker is an explicit directive. Returns {} for a family with no file — the extractors
    then apply their target-agnostic defaults."""
    path = _FEATURE_DIR / f"{(family or '').lower()}.yaml"
    if not path.is_file():
        return {}
    return load_contract_file(path)
