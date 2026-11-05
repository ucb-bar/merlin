"""The single source of target identity: name -> (paths, backend, kind).

Target identity used to be smeared across three hardcoded maps (`pipeline.DEFAULT_BACKEND`,
`target_lowering._specs()`/`LOWERING_TABLES`, `synthesize.dialect_plan.CURATED_TARGETS`) plus ~6
ad-hoc `parents[N]/"merlin/targets/..."` path readers. This module resolves everything a target needs
from one place, reusing the path resolvers in `merlin.targetgen.rtl.facts`.

Two kinds of target:
- ``reference`` — curated metadata on the legacy reference shelf or in physical-checkout examples.
- ``generated`` — an isolated package under ``artifacts/targets/<name>/<run_id>/``, loaded by
  :func:`merlin.targetgen.registry.load_target`. This module resolves the reference kind and the
  base paths; the parametric dialect (from the plan) is built by
  ``merlin.xdsl_dialects.targets.factory``.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import artifacts_dir, build_dir, checkout_root, targets_dir
from merlin.common.yaml import safe_load_text

from .providers import Provider, ProviderRole, read_provider
from .rtl.facts import rtl_facts_path

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Target-package RESOLUTION — how Merlin picks WHICH definition package to use for a target name.
#
# A target definition is a self-contained OUT-OF-TREE PACKAGE — a directory with
# ``contracts/target_contract.yaml`` (the capability manifest + optional plugin block) + a
# ``contracts/dialect_plan.yaml`` — exactly the layout ``capability_manifests.write_oot_target`` emits
# emits. Published compiler candidates and host schedules are DIFFERENT roles and do not necessarily
# ship support contracts. Only support providers participate in target-definition discovery.
#
# ``resolve(name)`` walks an ORDERED search path and takes the FIRST package whose contract ``name``
# matches. Precedence (highest first) — see docs/guides/target_resolution.md:
#   1. ``MERLIN_TARGET_PATH`` entries  — EXPLICIT selection: a specific versioned/named package, or a
#      user's separately-cloned ``<target>-mlir`` repo. ``os.pathsep``-separated, left-to-right; each
#      entry is either a package root (has ``contracts/target_contract.yaml``) or a dir OF such roots.
#      UNSET, it defaults to the checkout's vendored support providers (``examples/*/support``, keyed
#      by each provider's declared target; see :func:`in_repo_support`). Set to the EMPTY string, it
#      selects nothing, which is how a caller asks for no executable support at all.
#   2. physical-checkout ``examples/<name>/target/`` first, then legacy
#      ``merlin/targets/<name>/`` compatibility links. MERLIN_TARGETS_DIR replaces this search.
#   3. ``out/build/generated/<name>/``  — the FRESHLY-GENERATED OOT home (``write_oot_target`` /
#      onboarding drop packages here), so a just-generated target resolves with ZERO env.
#   4. ``out/artifacts/targets/<name>/``  — legacy generated location (fallback).
# To pin a specific version/location, put it first on ``MERLIN_TARGET_PATH``; it wins over every default.
_ENV_TARGET_PATH = "MERLIN_TARGET_PATH"
_OBSERVED_CONTRACTS = contextvars.ContextVar("merlin_observed_contracts", default={})
_OBSERVED_CONTRACT_PATHS = contextvars.ContextVar("merlin_observed_contract_paths", default={})


@contextlib.contextmanager
def observed_contract(name: str, contract: dict[str, Any], *, source_path: Path | None = None):
    """Bind deterministic derivation to selected bytes without changing providers.

    This scoped bridge is for existing name-based readers. It does not grant
    support code, select a compiler, or certify an authored capability claim.
    Readers receive detached copies so a consumer cannot mutate the selection.
    """
    token = _OBSERVED_CONTRACTS.set({**_OBSERVED_CONTRACTS.get(), name: copy.deepcopy(contract)})
    paths_token = _OBSERVED_CONTRACT_PATHS.set({**_OBSERVED_CONTRACT_PATHS.get(), name: source_path})
    try:
        yield
    finally:
        _OBSERVED_CONTRACT_PATHS.reset(paths_token)
        _OBSERVED_CONTRACTS.reset(token)


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
    kind: str  # "reference" | "generated" | "external"
    base: Path
    contract_path: Path
    dialect_plan_path: Path
    facts_path: Path  # rtl facts pin (may not exist for non-RTL targets)
    backend: str
    external_root: Path | None = None  # OOT package root, when kind == "external"

    @property
    def provider(self) -> Provider | None:
        """Package role and origin, not a trust decision or compiler qualification."""
        return read_provider(self.base)

    @property
    def capability_contract_path(self) -> Path:
        """Selected capability bytes; ``contract_path`` still owns executable support."""
        observed = _OBSERVED_CONTRACT_PATHS.get().get(self.name)
        if observed is not None:
            return Path(observed)
        override = os.environ.get("MERLIN_TARGET_CONTRACT")
        return Path(override).expanduser() if override else self.contract_path

    def load_contract(self) -> dict[str, Any]:
        if self.name in _OBSERVED_CONTRACTS.get():
            return copy.deepcopy(_OBSERVED_CONTRACTS.get()[self.name])
        # An experiment may select a capability contract alongside an OOT support provider.
        # Keep executable plugin ownership at the provider's own contract (below), but make all
        # capability consumers read the same explicitly selected view as RTL extraction.
        path = self.capability_contract_path
        if path != self.contract_path:
            if not path.is_file():
                raise TargetContractMissing(f"{self.name!r}: selected capability contract does not exist: {path}")
            selected = safe_load_text(path.read_text(encoding="utf-8"))
            if not isinstance(selected, dict):
                raise ValueError(f"{path}: selected capability contract must be a mapping")
            if selected.get("name") != self.name:
                raise ValueError(f"{path}: selected capability contract names a different target")
            return selected
        return self._load_provider_contract()

    def _load_provider_contract(self) -> dict[str, Any]:
        """Read support-owned metadata, independently of an observed capability view."""
        if not self.contract_path.is_file():
            # The fallback branch in `resolve` promises this surfaces the absence honestly; a bare
            # FileNotFoundError from deep inside a caller's stack is not that. Say which target, which
            # path, and that the package may simply not be generated.
            raise TargetContractMissing(
                f"{self.name!r}: no capability contract at {self.contract_path}. Either the target's "
                f"package has not been generated, or the name asked for is a DIRECTORY name whose "
                f"descriptor declares a different `target:` (see `declared_target_for`)"
            )
        return safe_load_text(self.contract_path.read_text(encoding="utf-8"))

    def load_dialect_plan(self) -> dict[str, Any]:
        return yaml.safe_load(self.dialect_plan_path.read_text(encoding="utf-8"))

    def plugin(self) -> dict[str, Any]:
        """The out-of-tree ``plugin`` block from the contract (dialect + lowering entry-points).

        Merlin reads (never executes) these references; importing the dialect / calling the lowering
        is the caller's job, guarded — so nothing target-specific runs at resolution time. The OOT
        package root is injected as ``path`` so a caller can put it on ``sys.path``.
        """
        selected = self.load_contract()
        runner = selected.get("runner", {})
        if not isinstance(runner, dict):
            raise ValueError("selected runtime declaration must be a mapping")
        if "backend" in runner:
            if runner["backend"] != "chipyard_rocc" or selected.get("plugin"):
                raise ValueError("data-bound runtime selection cannot grant executable provider hooks")
            return {}
        # Legacy observations cannot inject executable plugin references.
        block = dict(self._load_provider_contract().get("plugin", {}))
        if self.external_root is not None:
            block.setdefault("path", str(self.external_root))
        return block


def _backend_from_contract(contract_path: Path) -> str:
    """A target's DECLARED default runtime backend, read from its contract's ``runtime.default_backend``
    (a declared target fact, not a name -> backend map). Generic ``simulator`` when the file or the field
    is absent — an unknown target degrades honestly rather than inheriting another target's backend."""
    try:
        doc = safe_load_text(contract_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return _GENERIC_BACKEND
    val = (doc.get("runtime") or {}).get("default_backend")
    return str(val) if val else _GENERIC_BACKEND


def backend_for(name: str) -> str:
    """Default runtime backend for a target, DERIVED from its contract (never a hardcoded name map)."""
    return resolve(name).backend


def _is_target_root(p: Path) -> bool:
    # Candidate manifests are not support definitions. Ignore unrelated manifests on a shelf;
    # an explicit provider declaration, however, must never be silently skipped when invalid.
    if not (p / "provider.yaml").is_file() and not (p / "contracts" / "target_contract.yaml").is_file():
        return False
    provider = read_provider(p)
    return provider is not None and provider.role == ProviderRole.SUPPORT


def _target_name(root: Path) -> str:
    """The support provider's declared target name."""
    provider = read_provider(root)
    if provider is None or provider.role != ProviderRole.SUPPORT:
        raise ValueError(f"{root}: not a target support provider")
    return provider.target


def _roots_under(entry: Path) -> list[Path]:
    """A search-path entry expands to the package root itself (if it has a contract) or, if it is a
    directory OF packages, each immediate child that is a package root."""
    if _is_target_root(entry):
        return [entry]
    if entry.is_dir():
        return [c for c in sorted(entry.iterdir()) if _is_target_root(c)]
    return []


#: Where a checkout keeps a target's vendored support provider: ``examples/<example>/<this>``. The
#: example directory is not the target's identity; the provider's own declaration is.
IN_REPO_SUPPORT_DIR = "support"


class TargetCollisionError(ValueError):
    """One unordered shelf contains different providers for the same target."""


def in_repo_support() -> dict[str, Path]:
    """The checkout's vendored support providers -> ``{declared target: provider root}``.

    Target support is target-specific code, so it lives beside the target's example, at
    ``examples/<example>/support``. Each root is keyed by the target its ``provider.yaml`` declares,
    never by its directory name, so a lookup takes the target as a parameter and this module names
    none. Only support providers count; an invalid declaration raises rather than vanishing from
    the selection, and two examples declaring one target are refused instead of picked by sort order.

    An installed distribution has no checkout and therefore no default: it selects support only
    through ``MERLIN_TARGET_PATH``.
    """
    checkout = checkout_root()
    if checkout is None:
        return {}
    found: dict[str, Path] = {}
    for candidate in sorted((checkout / "examples").glob(f"*/{IN_REPO_SUPPORT_DIR}")):
        if not candidate.is_dir() or not _is_target_root(candidate):
            continue
        root = candidate.resolve()
        name = _target_name(root)
        if name in found and found[name] != root:
            raise TargetCollisionError(f"two in-repo support providers declare {name!r}: {found[name]} and {root}")
        found[name] = root
    return found


def vendored_support_dirs() -> tuple[Path, ...]:
    """Every ``examples/<example>/support`` directory in the checkout, whether or not it declares a
    provider.

    :func:`in_repo_support` SELECTS, so it counts only valid declarations. What must be WITHHELD is
    wider: a support tree without a ``provider.yaml`` (or before its declaration is written) is still
    target support bytes, so a mask keyed on the declaration alone would leave it readable. Empty for an
    installed distribution, which has no checkout.
    """
    checkout = checkout_root()
    if checkout is None:
        return ()
    found = (checkout / "examples").glob(f"*/{IN_REPO_SUPPORT_DIR}")
    return tuple(sorted(path.resolve() for path in found if path.is_dir()))


def default_support_root(target: str) -> Path | None:
    """The in-repo support provider ``MERLIN_TARGET_PATH`` defaults to for ``target``, if one exists."""
    return in_repo_support().get(target)


def _env_target_roots() -> list[Path]:
    """The ``MERLIN_TARGET_PATH`` search entries, in declared (left-to-right) order.

    Unset, the entries are the in-repo support providers (one per declared target). An explicit
    value, including the empty string, replaces that default entirely.
    """
    raw = os.environ.get(_ENV_TARGET_PATH)
    if raw is None:
        return list(in_repo_support().values())
    return [Path(e) for e in raw.split(os.pathsep) if e]


def effective_target_path() -> str:
    """The selection in ``MERLIN_TARGET_PATH`` syntax, with the in-repo default spelled out.

    For a caller that hands the selection to a child process or prepends an entry to it: an unset
    variable must not become an explicit selection that silently drops the default.
    """
    raw = os.environ.get(_ENV_TARGET_PATH)
    return raw if raw is not None else os.pathsep.join(str(root) for root in _env_target_roots())


@dataclass(frozen=True)
class TargetShadow:
    name: str
    selected: Path
    shadowed: Path


@dataclass(frozen=True)
class Discovery:
    targets: dict[str, Path]
    shadows: tuple[TargetShadow, ...]


def discover(entries: list[Path]) -> Discovery:
    """Read-only, first-entry-wins discovery, with explicit shadows and ambiguous-shelf errors.

    Ordering separate search entries is an intentional selection. Alphabetical child directory
    order within one shelf is not: two different packages declaring one target there are an error.
    """
    found: dict[str, Path] = {}
    shadows: list[TargetShadow] = []
    for entry in entries:
        shelf: dict[str, Path] = {}
        for root in _roots_under(entry):
            root = root.resolve()
            name = _target_name(root)
            if name in shelf and shelf[name] != root:
                raise TargetCollisionError(f"{entry}: ambiguous target {name!r}: {shelf[name]} and {root}")
            shelf[name] = root
        for name, root in shelf.items():
            if name not in found:
                found[name] = root
            elif found[name] != root:
                shadows.append(TargetShadow(name, found[name], root))
    return Discovery(found, tuple(shadows))


def _discover(entries: list[Path]) -> dict[str, Path]:
    return discover(entries).targets


def explicit_targets() -> dict[str, Path]:
    """Support providers selected on MERLIN_TARGET_PATH, without implicit artifact discovery.

    Use this inventory to authorize executable plugins. Reference definitions and freshly
    generated directories remain inspectable metadata, not permission to execute code. With the
    variable unset, the selection is the checkout's reviewed in-repo support (:func:`in_repo_support`),
    which is tracked source, not a discovered artifact.
    """
    return _discover(_env_target_roots())


def external_targets() -> dict[str, Path]:
    """Discover out-of-tree target packages -> ``{name: package_root}``, across the ``MERLIN_TARGET_PATH``
    entries AND the freshly-generated home (``out/build/generated/``). Env entries take precedence over the
    generated home (first configured entry wins). In-tree reference targets are resolved separately
    (see :func:`resolve`); this returns only OOT packages."""
    return _discover([*_env_target_roots(), generated_target_home()])


def _resolve_external(name: str, root: Path) -> TargetInfo:
    provider = read_provider(root)
    if provider is None or provider.role != ProviderRole.SUPPORT or provider.contract_path is None:
        raise ValueError(f"{root}: not a target support provider")
    contracts = root / "contracts"
    return TargetInfo(
        name=name,
        kind="external",
        base=root,
        contract_path=provider.contract_path,
        dialect_plan_path=contracts / "dialect_plan.yaml",
        facts_path=contracts / "rtl_facts" / "facts.json",
        backend=_backend_from_contract(provider.contract_path),
        external_root=root,
    )


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
    from merlin.targetgen import corpora

    # The convention path, not the MERLIN_TARGET_EXPERIMENT override: this asks what ONE directory
    # declares, and an override names a single descriptor that is very likely another directory's.
    desc = corpora.standard_descriptor_path(str(directory_name))
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
    2. legacy reference shelf, then physical-checkout examples -> ``kind='reference'``.
       An explicit ``MERLIN_TARGETS_DIR`` replaces both reference locations.
    3. freshly-generated ``out/build/generated/<name>/`` -> ``kind='external'`` (zero-env default for a
       just-generated target).
    4. legacy ``out/artifacts/targets/<name>/`` -> ``kind='generated'`` (fallback).

    So an explicit env pointer always wins; otherwise a curated reference beats an incidental generated
    package; otherwise a freshly generated OOT package is picked up automatically.

    This operation NEVER derives resources, imports provider code, or fetches repositories, even
    when the legacy MERLIN_TARGET_AUTOFETCH environment variable is set. Call :func:`materialize`
    or :func:`merlin.targetgen.oot_fetch.fetch` explicitly before resolving an absent package.
    """
    return _resolve(name, allow_alias=True)


def _physical_reference_alias(base: Path, name: str) -> Path:
    """Resolve a checkout compatibility link only against an explicitly selected repo.

    Installed distributions do not discover arbitrary ``examples/`` under cwd.
    An explicitly selected repository may, however, contain old ``merlin/targets``
    links to its own physical example packages.  Use the self-contained package
    for provider ownership; never accept a link to another root or target.
    """
    selected = os.environ.get("MERLIN_REPO_ROOT")
    link = base / "contracts/target_contract.yaml"
    if not selected or not link.is_symlink():
        return base
    repo = Path(selected).resolve()
    examples = repo / "examples"
    try:
        contract = link.resolve(strict=True)
    except (OSError, RuntimeError):
        return base
    candidate = contract.parent.parent
    if (
        contract.name == "target_contract.yaml"
        and contract.parent.name == "contracts"
        and candidate.name == "target"
        and candidate.parent.parent == examples
        and _target_name(candidate) == name
    ):
        return candidate
    return base


def _resolve(name: str, *, allow_alias: bool) -> TargetInfo:
    # 1. explicit env selection — highest precedence
    env = _discover(_env_target_roots())
    if name in env:
        return _resolve_external(name, env[name])
    # 2. curated in-tree reference
    base = reference_targets().get(name)
    if base is not None:
        base = _physical_reference_alias(base, name)
        contract = Path(os.environ.get("MERLIN_TARGET_CONTRACT") or base / "contracts/target_contract.yaml")
        return TargetInfo(
            name=name,
            kind="reference",
            base=base,
            contract_path=contract,
            dialect_plan_path=base / "contracts/dialect_plan.yaml",
            facts_path=rtl_facts_path(name),
            backend=_backend_from_contract(contract),
        )
    # 3. freshly-generated OOT home
    gen = _discover([generated_target_home()])
    if name in gen:
        return _resolve_external(name, gen[name])
    # 4. Legacy generated location is read as-is. Generation is an explicit operation.
    # 5. ONE HOP TO THE DECLARED NAME. Nothing resolved for the name as given, and a descriptor in a
    # directory of that name may declare the configuration-qualified name every artifact path uses.
    # Tried last so it can never shadow a target that resolves on its own, and exactly once so a
    # descriptor pointing at itself cannot loop.
    base = artifacts_dir() / "targets" / name
    contract = Path(os.environ.get("MERLIN_TARGET_CONTRACT") or base / "contracts/target_contract.yaml")
    if allow_alias and not contract.is_file():
        declared = declared_target_for(name)
        if declared:
            alternate = _resolve(declared, allow_alias=False)
            if alternate.contract_path.is_file():
                return alternate
    return TargetInfo(
        name=name,
        kind="generated",
        base=base,
        contract_path=contract,
        dialect_plan_path=base / "contracts/dialect_plan.yaml",
        facts_path=rtl_facts_path(name),
        backend=_backend_from_contract(contract),
    )


def materialize(name: str, *, destination: Path | None = None) -> TargetInfo:
    """Explicitly derive a support package, propagating derivation failures to the caller.

    Existing destinations are not overwritten: regeneration needs its own explicitly managed output.
    The generator owns schema validation and derivation; this function adds no fallback facts.
    """
    from . import capability_manifests as cm

    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid target name {name!r}")
    destination = Path(destination) if destination is not None else generated_target_home() / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{destination}: materialization requires a new destination")
    cm.write_oot_target(name, destination)
    return _resolve_external(name, destination.resolve())


def reference_targets() -> dict[str, Path]:
    """Read-only reference metadata; discovery never authorizes provider execution.

    Authored checkout examples take precedence over legacy reference roots. An
    explicit reference-root override disables checkout example discovery entirely.
    Installed distributions never infer an examples directory from a working dir.
    """
    root = targets_dir()
    references = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if (path / "contracts" / "target_contract.yaml").is_file():
                references[path.name] = path
    checkout = checkout_root() if "MERLIN_TARGETS_DIR" not in os.environ else None
    if checkout is not None:
        examples: dict[str, Path] = {}
        for contract in sorted((checkout / "examples").glob("*/target/contracts/target_contract.yaml")):
            if contract.is_file():
                base = contract.parent.parent
                name = _target_name(base)
                if name in examples:
                    raise TargetCollisionError(
                        f"duplicate reference examples for {name!r}: {examples[name]} and {base}"
                    )
                examples[name] = base
        references.update(examples)
    return references


def list_targets() -> list[str]:
    """Names of available reference metadata, including physical-checkout examples."""
    return sorted(reference_targets())


def all_targets() -> list[str]:
    """Curated reference targets plus any discovered out-of-tree (MERLIN_TARGET_PATH) targets."""
    return sorted(set(list_targets()) | set(external_targets()))


def load_contract(name: str) -> dict[str, Any]:
    if name in _OBSERVED_CONTRACTS.get():
        return copy.deepcopy(_OBSERVED_CONTRACTS.get()[name])
    return resolve(name).load_contract()


def load_matrix_contract(name: str) -> dict[str, Any]:
    """Read the selected support provider's declared matrix metadata, without importing code.

    Reference metadata alone is not a provider selection. Missing declarations or
    resources never fall back to a checkout-global matrix contract.
    """
    from .plugins import resolve_support
    from .providers import ProviderError, contained_resource

    info = resolve_support(name)
    reference = info.load_contract().get("matrix_contract")
    if not isinstance(reference, str) or not reference:
        raise ProviderError(f"{name!r}: selected support declares no matrix_contract")
    path = contained_resource(info.base, reference)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("units"), dict):
        raise ProviderError(f"{path}: matrix contract requires a units mapping")
    return payload


def load_dialect_plan(name: str) -> dict[str, Any]:
    return resolve(name).load_dialect_plan()
