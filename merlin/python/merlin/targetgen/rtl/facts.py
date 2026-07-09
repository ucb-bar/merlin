"""Target-agnostic resolution of RTL-derived facts and the introspect run/cache location.

The committed ``facts.json`` under a target is a **pinned cache** of an introspect run — the run
(``circt_introspect``) is the source of truth, regenerated from the RTL whenever the CIRCT/firtool
toolchain is present. The pin lets the RTL-derived checks run when that toolchain is absent (CI,
fresh checkouts). This module is the single place that maps a target name -> its facts pin and its
purgeable scratch dir, so no consumer hardcodes the gemmini path (they used to, with three
different ``parents[]`` depths). Mirrors :func:`merlin.targetgen.contract.schemas.contract_dir`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root, targets_dir


def target_base(target: str = "gemmini") -> Path:
    """Per-target home: the curated ``merlin/targets/<t>`` if it exists, else the generated
    ``artifacts/targets/<t>`` (covers targets like muon that have no hand-curated reference dir)."""
    ref = targets_dir() / target
    if ref.is_dir():
        return ref
    return repo_root() / "artifacts" / "targets" / target


def rtl_facts_path(target: str = "gemmini", *, explicit: str | Path | None = None) -> Path:
    """Resolve the RTL facts pin: explicit > ``$MERLIN_RTL_FACTS`` > ``<target base>/contracts/rtl_facts/facts.json``."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MERLIN_RTL_FACTS")
    if env:
        return Path(env)
    return target_base(target) / "contracts" / "rtl_facts" / "facts.json"


def target_contract_path(target: str = "gemmini", *, explicit: str | Path | None = None) -> Path:
    """Resolve the target contract yaml: explicit > ``$MERLIN_TARGET_CONTRACT`` > ``<base>/contracts/target_contract.yaml``."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MERLIN_TARGET_CONTRACT")
    if env:
        return Path(env)
    return target_base(target) / "contracts" / "target_contract.yaml"


def load_facts(target: str = "gemmini", *, explicit: str | Path | None = None) -> dict[str, Any]:
    """Load and parse the resolved facts pin (raises if absent — fail-closed)."""
    return json.loads(rtl_facts_path(target, explicit=explicit).read_text(encoding="utf-8"))


def rtl_cache_dir(target: str = "gemmini", *, ensure: bool = False) -> Path:
    """Purgeable introspect scratch (hw.mlir input, ``*.ll``/``*.o``, arcilator bins, per-run
    facts.json) under ``artifacts/cache/rtl_introspect/<target>/`` — never inside ``merlin/``.

    Mirrors :func:`merlin.common.artifacts.cache_dir` (``artifacts/cache/<ns>/``, PURGEABLE) without
    forcing directory creation at import time; pass ``ensure=True`` when about to write."""
    d = repo_root() / "artifacts" / "cache" / "rtl_introspect" / target
    if ensure:
        d.mkdir(parents=True, exist_ok=True)
    return d
