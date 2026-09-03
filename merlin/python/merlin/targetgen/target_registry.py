"""The single source of target identity: name -> (paths, backend, kind).

Target identity used to be smeared across three hardcoded maps (`pipeline.DEFAULT_BACKEND`,
`target_lowering._specs()`/`LOWERING_TABLES`, `synthesize.dialect_plan.CURATED_TARGETS`) plus ~6
ad-hoc `parents[N]/"merlin/targets/..."` path readers. This module resolves everything a target needs
from one place, reusing the path resolvers in `merlin.targetgen.rtl.facts`.

Two kinds of target:
- ``reference`` — a curated definition under ``merlin/targets/<name>/`` (toy_npu, saturn, gemmini).
- ``generated`` — an isolated package under ``artifacts/targets/<name>/<run_id>/``, loaded by
  :func:`merlin.targetgen.registry.load_target`. This module resolves the reference kind and the
  base paths; the parametric dialect (from the plan) is built by
  ``merlin.xdsl_dialects.targets.factory``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import build_dir, targets_dir
from .rtl.facts import dialect_plan_path, rtl_facts_path, target_base, target_contract_path

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Target-package RESOLUTION — how Merlin picks WHICH definition package to use for a target name.
#
# A target definition is a self-contained OUT-OF-TREE PACKAGE — a directory with
# ``contracts/target_contract.yaml`` (the capability manifest + optional plugin block) + a
# ``contracts/dialect_plan.yaml`` — exactly the layout ``capability_manifests.write_oot_target`` emits
# and the published ``<target>-mlir`` repos ship. The SAME format is the interchange format: anyone can
# clone such a repo anywhere and plug it in. Nothing target-specific is committed into merlin for this.
#
# ``resolve(name)`` walks an ORDERED search path and takes the FIRST package whose contract ``name``
# matches. Precedence (highest first) — see docs/guides/target_resolution.md:
#   1. ``MERLIN_TARGET_PATH`` entries  — EXPLICIT selection: a specific versioned/named package, or a
#      user's separately-cloned ``<target>-mlir`` repo. ``os.pathsep``-separated, left-to-right; each
#      entry is either a package root (has ``contracts/target_contract.yaml``) or a dir OF such roots.
#   2. in-tree ``merlin/targets/<name>/``  — the curated REFERENCE package shipped in merlin (gemmini).
#   3. ``out/build/generated/<name>/``  — the FRESHLY-GENERATED OOT home (``write_oot_target`` /
#      onboarding drop packages here), so a just-generated target resolves with ZERO env.
#   4. ``out/artifacts/targets/<name>/``  — legacy generated location (fallback).
# To pin a specific version/location, put it first on ``MERLIN_TARGET_PATH``; it wins over every default.
_ENV_TARGET_PATH = "MERLIN_TARGET_PATH"


def generated_target_home() -> Path:
    """Where freshly generated OOT target packages are dropped (``out/build/generated/``) and
    auto-discovered — the zero-env default for a just-generated target."""
    return build_dir() / "generated"

# Generic runtime backend for a target whose contract declares no default (no name -> backend map).
_GENERIC_BACKEND = "simulator"


@dataclass(frozen=True)
class TargetInfo:
    """Resolved identity + locations for one target."""

    name: str
    kind: str                 # "reference" | "generated" | "external"
    base: Path
    contract_path: Path
    dialect_plan_path: Path
    facts_path: Path          # rtl facts pin (may not exist for non-RTL targets)
    backend: str
    external_root: Path | None = None   # OOT package root, when kind == "external"

    def load_contract(self) -> dict[str, Any]:
        if not self.contract_path.is_file():
            # The fallback branch in `resolve` promises this surfaces the absence honestly; a bare
            # FileNotFoundError from deep inside a caller's stack is not that. Say which target, which
            # path, and that the package may simply not be generated.
            raise TargetContractMissing(
                f"{self.name!r}: no capability contract at {self.contract_path}. Either the target's "
                f"package has not been generated, or the name asked for is a DIRECTORY name whose "
                f"descriptor declares a different `target:` (see `declared_target_for`)")
        return yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))

    def load_dialect_plan(self) -> dict[str, Any]:
        return yaml.safe_load(self.dialect_plan_path.read_text(encoding="utf-8"))

    def plugin(self) -> dict[str, Any]:
        """The out-of-tree ``plugin`` block from the contract (dialect + lowering entry-points).

        Merlin reads (never executes) these references; importing the dialect / calling the lowering
        is the caller's job, guarded — so nothing target-specific runs at resolution time. The OOT
        package root is injected as ``path`` so a caller can put it on ``sys.path``.
        """
        block = dict(self.load_contract().get("plugin", {}))
        if self.external_root is not None:
            block.setdefault("path", str(self.external_root))
        return block


def _backend_from_contract(contract_path: Path) -> str:
    """A target's DECLARED default runtime backend, read from its contract's ``runtime.default_backend``
    (a declared target fact, not a name -> backend map). Generic ``simulator`` when the file or the field
    is absent — an unknown target degrades honestly rather than inheriting another target's backend."""
    try:
        doc = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return _GENERIC_BACKEND
    val = (doc.get("runtime") or {}).get("default_backend")
    return str(val) if val else _GENERIC_BACKEND


def backend_for(name: str) -> str:
    """Default runtime backend for a target, DERIVED from its contract (never a hardcoded name map)."""
    return _backend_from_contract(target_contract_path(name))


def _is_target_root(p: Path) -> bool:
    return (p / "contracts" / "target_contract.yaml").is_file()


def _target_name(root: Path) -> str:
    """The target's declared name (contract ``name``), falling back to the directory name."""
    try:
        doc = yaml.safe_load((root / "contracts" / "target_contract.yaml").read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("name"):
            return str(doc["name"])
    except (OSError, yaml.YAMLError):
        pass
    return root.name


def _roots_under(entry: Path) -> list[Path]:
    """A search-path entry expands to the package root itself (if it has a contract) or, if it is a
    directory OF packages, each immediate child that is a package root."""
    if _is_target_root(entry):
        return [entry]
    if entry.is_dir():
        return [c for c in sorted(entry.iterdir()) if _is_target_root(c)]
    return []


def _env_target_roots() -> list[Path]:
    """The ``MERLIN_TARGET_PATH`` search entries, in declared (left-to-right) order."""
    raw = os.environ.get(_ENV_TARGET_PATH, "")
    return [Path(e) for e in raw.split(os.pathsep) if e]


def _discover(entries: list[Path]) -> dict[str, Path]:
    """``{contract-name: package_root}`` for every package reachable from ``entries`` (later entries win
    on a name clash — callers order entries so the desired precedence is achieved)."""
    found: dict[str, Path] = {}
    for entry in entries:
        for root in _roots_under(entry):
            found[_target_name(root)] = root
    return found


def external_targets() -> dict[str, Path]:
    """Discover out-of-tree target packages -> ``{name: package_root}``, across the ``MERLIN_TARGET_PATH``
    entries AND the freshly-generated home (``out/build/generated/``). Env entries take precedence over the
    generated home (they are applied last, overwriting). In-tree reference targets are resolved separately
    (see :func:`resolve`); this returns only OOT packages."""
    # generated-home first (lower precedence), then env (higher) — later writes win in `_discover`.
    return _discover([generated_target_home(), *_env_target_roots()])


def _resolve_external(name: str, root: Path) -> TargetInfo:
    contracts = root / "contracts"
    return TargetInfo(
        name=name, kind="external", base=root,
        contract_path=contracts / "target_contract.yaml",
        dialect_plan_path=contracts / "dialect_plan.yaml",
        facts_path=contracts / "rtl_facts" / "facts.json",
        backend=_backend_from_contract(contracts / "target_contract.yaml"),
        external_root=root)


class TargetContractMissing(FileNotFoundError):
    """A target resolved, but the capability contract it points at does not exist."""


def declared_target_for(directory_name: str) -> str | None:
    """The name a capsule-bench descriptor DECLARES, when it differs from its directory name.

    ⚠️ A DIRECTORY NAME IS NOT ALWAYS THE TARGET NAME, and this repo has now paid for that four
    separate times: the conformance-coverage gate exited 0 for two targets it could not resolve, the
    conformance specs were audited under the wrong key, `generate_corpus --target <declared>` dies on
    a missing descriptor, and the shipped-capsule boundary gate raised FileNotFoundError on a contract
    that exists under the declared name. A descriptor sits in a short directory and declares a
    configuration-qualified name, which is the key every artifact path uses.

    Returns None when the directory has no descriptor or the two names agree, so a caller can treat
    "no hop available" and "hop to X" distinctly.
    """
    from merlin.common.paths import repo_root

    desc = (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"
            / str(directory_name) / "target_experiment.yaml")
    if not desc.is_file():
        return None
    try:
        doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    declared = str(doc.get("target") or "")
    return declared or None if declared and declared != str(directory_name) else None


def resolve(name: str) -> TargetInfo:
    """Resolve a target's identity + paths by walking the ordered search path (module docstring),
    first-match-wins:

    1. ``MERLIN_TARGET_PATH`` (explicit selection: a specific versioned/named package, or a user's
       separately-cloned ``<target>-mlir`` repo) -> ``kind='external'``.
    2. curated in-tree ``merlin/targets/<name>/`` -> ``kind='reference'``.
    3. freshly-generated ``out/build/generated/<name>/`` -> ``kind='external'`` (zero-env default for a
       just-generated target).
    4. legacy ``out/artifacts/targets/<name>/`` -> ``kind='generated'`` (fallback).

    So an explicit env pointer always wins; otherwise a curated reference beats an incidental generated
    package; otherwise a freshly generated OOT package is picked up automatically."""
    # 1. explicit env selection — highest precedence
    env = _discover(_env_target_roots())
    if name in env:
        return _resolve_external(name, env[name])
    # 2. curated in-tree reference
    if (targets_dir() / name).is_dir():
        return TargetInfo(
            name=name, kind="reference", base=target_base(name),
            contract_path=target_contract_path(name), dialect_plan_path=dialect_plan_path(name),
            facts_path=rtl_facts_path(name), backend=backend_for(name))
    # 3. freshly-generated OOT home
    gen = _discover([generated_target_home()])
    if name in gen:
        return _resolve_external(name, gen[name])
    # 3b. opt-in native fetch of the published <target>-mlir repo into the generated home, then
    # re-discover. Off by default (no surprise network calls); set MERLIN_TARGET_AUTOFETCH=1 to enable.
    if os.environ.get("MERLIN_TARGET_AUTOFETCH", "").strip() not in ("", "0", "false", "False"):
        from .oot_fetch import fetch, FetchError  # lazy: oot_fetch imports from this module
        try:
            fetch(name, champion=os.environ.get("MERLIN_TARGET_CHAMPION") or None)
        except FetchError:
            pass
        else:
            gen = _discover([generated_target_home()])
            if name in gen:
                return _resolve_external(name, gen[name])
    # 4. legacy generated location. A DISCOVERED residual-target (ships a contracts/residual.yaml but no
    # committed contract) is materialized on demand here — the "drop a descriptor + residual, let mlc
    # derive" path: its structural facts come from RTL via mlc and its datapath from the residual. This
    # only fires when nothing is committed (reference targets are caught at step 2), so it changes no
    # committed target; it fails closed (writes nothing) when mlc can't derive, and load_contract then
    # surfaces the missing contract honestly rather than fabricating one.
    if not target_contract_path(name).is_file():
        _materialize_discovered(name)
    # 5. ONE HOP TO THE DECLARED NAME. Nothing resolved for the name as given, and a descriptor in a
    # directory of that name may declare the configuration-qualified name every artifact path uses.
    # Tried last so it can never shadow a target that resolves on its own, and exactly once so a
    # descriptor pointing at itself cannot loop.
    if not target_contract_path(name).is_file():
        declared = declared_target_for(name)
        if declared and target_contract_path(declared).is_file():
            return resolve(declared)
    return TargetInfo(
        name=name, kind="generated", base=target_base(name),
        contract_path=target_contract_path(name), dialect_plan_path=dialect_plan_path(name),
        facts_path=rtl_facts_path(name), backend=backend_for(name))


def _materialize_discovered(name: str) -> None:
    """Materialize a discovered residual-target's ``target_contract.yaml`` + ``dialect_plan.yaml`` from its
    residual + mlc-derived RTL facts, into its legacy generated home. Best-effort and ATOMIC: the manifest
    is fully derived (and schema-validated) before anything is written, so a failure (mlc absent, schema
    mismatch, not a discovered generator) writes nothing — never a partial or fabricated contract."""
    try:
        from . import capability_manifests as cm
        if name not in cm.discovered_targets():
            return
        cm.write_oot_target(name, target_base(name))
    except Exception:  # noqa: BLE001 — no derivation possible in this env: leave it missing, fail closed
        return


def list_targets() -> list[str]:
    """Curated reference targets (dirs under merlin/targets/ with a target_contract.yaml)."""
    root = targets_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if (p / "contracts" / "target_contract.yaml").is_file())


def all_targets() -> list[str]:
    """Curated reference targets plus any discovered out-of-tree (MERLIN_TARGET_PATH) targets."""
    return sorted(set(list_targets()) | set(external_targets()))


def load_contract(name: str) -> dict[str, Any]:
    return resolve(name).load_contract()


def load_dialect_plan(name: str) -> dict[str, Any]:
    return resolve(name).load_dialect_plan()
