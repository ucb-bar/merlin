"""Declared simulator capabilities and engine selection, independent of evaluation.

This is the canonical registry shared with the optional capsule evaluator. Registering a
plugin or choosing an engine does not import AET or execute a capsule. Concrete grading
adapters are loaded only when requested. Legacy capsule_runner exports the SAME objects.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any


class OracleMetadataUnavailable(RuntimeError):
    """A declared oracle cannot describe its tiers without executing an evaluator."""


@dataclass(frozen=True)
class OracleTierPlan:
    """Host-owned tier metadata, never an execution result or a certification.

    ``tiers`` is the selected/advertised inventory, including advisory tiers.
    Fallible adapter constructors may omit selected tiers, so this inventory
    cannot define capsule requirements unless ``requirements_inference_safe``
    explicitly guarantees construction does not silently change its tier keys.
    Neither that guarantee nor selection proves successful program execution.
    """

    tiers: tuple[str, ...]
    route: str = "plugin"
    sim_via: str = ""
    model_ext: str | None = None
    selection: dict[str, Any] | None = None
    unavailable_reason: str | None = None
    requirements_inference_safe: bool = False

    def __post_init__(self):
        if not isinstance(self.tiers, tuple) or any(not isinstance(tier, str) or not tier for tier in self.tiers):
            raise ValueError("oracle tier metadata must be a tuple of nonempty tier names")
        if len(set(self.tiers)) != len(self.tiers):
            raise ValueError("oracle tier metadata must not contain duplicate tiers")
        if not isinstance(self.requirements_inference_safe, bool):
            raise ValueError("requirements inference safety must be an explicit boolean")
        # Own the complete nested evidence, not just the outer dataclass. Keep
        # the public dataclass field for replace()/asdict() compatibility.
        object.__setattr__(self, "selection", deepcopy(object.__getattribute__(self, "selection")))

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        # This one field has copy-on-read semantics: evaluators still receive
        # ordinary dicts/lists and may serialize or annotate their own copy.
        # Returning the stored dict would make a frozen plan externally mutable.
        return deepcopy(value) if name == "selection" else value


_SIM_ADAPTER_FACTORIES: dict[str, Callable[[str], dict]] = {}


def bind_sim_oracle_adapters(sim_via: str, factory: Callable[[str], dict]) -> None:
    """Bind trusted evaluator factories without making core import their owner.

    This is dependency injection for built-in policies, not candidate plugin
    discovery. Out-of-tree support plugins retain ``register_sim_oracle``.
    """
    _SIM_ADAPTER_FACTORIES[sim_via] = factory


def _chipyard_adapters(target: str) -> dict:
    factory = _SIM_ADAPTER_FACTORIES.get("chipyard")
    if factory is None:
        raise OracleMetadataUnavailable(
            "concrete simulator adapters require merlin-experiments; use oracle_tier_plan for core-only metadata"
        )
    return factory(target)


def chipyard_tier_plan(target: str) -> OracleTierPlan:
    """The concrete simulator contribution; ARC fallback is merged by routing."""
    try:
        selection = select_chipyard_engine(target)
    except Exception as exc:  # matches the evaluator's existing absent-RTL policy
        return OracleTierPlan(("L2",), unavailable_reason=str(exc), requirements_inference_safe=True)
    # These constructors wrap execution in closures; they do not omit keys.
    return OracleTierPlan(("L2", "L3"), selection=selection, requirements_inference_safe=True)


def _endpoint_of(target: str) -> tuple[str | None, str | None]:
    """(endpoint_kind, model_ext) from the target's contract, best-effort (None,None if no contract).
    Lets ``oracle_adapters`` self-route without threading manifest fields through every caller."""
    try:
        from .target_experiment import load_capability_manifest

        m = load_capability_manifest(target)
        model_ext = (m.contract.get("runner") or {}).get("model_ext") or (m.contract.get("toolchain") or {}).get(
            "model"
        )
        return m.endpoint_kind, model_ext
    except Exception:  # noqa: BLE001 — no contract / not resolvable -> fall back to the arc default
        return None, None


def _bespoke_sim_via(target: str) -> str:
    """Recover a target's DECLARED bespoke-sim engine from its contract's ``runner.sim_via`` block — the
    sim engine is a declared target fact, NOT inferred from a hardcoded ``{spike,verilator} -> chipyard``
    map. A target that ships the chipyard spike/verilator tiers declares ``sim_via: chipyard``; an arc-only
    target declares none and reads back as ``""``. Lets :func:`oracle_adapters` self-route when a caller
    omits ``sim_via``. The harness passes the descriptor's authoritative ``toolchain.sim_via`` explicitly;
    when it doesn't (a standalone preflight/validator), we fall back to the contract's ``runner.sim_via``
    and then to the descriptor's ``toolchain.sim_via`` — so a SIMT target's ``cyclotron`` engine resolves
    even without a contract, instead of mis-defaulting to the arc path (which would false-green the
    preflight)."""
    try:
        from .target_experiment import load_capability_manifest

        via = str((load_capability_manifest(target).contract.get("runner") or {}).get("sim_via") or "")
        if via:
            return via
    except Exception:  # noqa: BLE001 — no contract; fall through to the descriptor
        pass
    try:
        from .corpora import descriptor_path
        from .target_experiment import load_target_experiment

        # Through `corpora`: it owns the experiments/ layout and honours MERLIN_TARGET_EXPERIMENT,
        # which the convention path built here silently ignored.
        p = descriptor_path(target)
        if p.is_file():
            return str(getattr(load_target_experiment(p), "sim_via", "") or "")
    except Exception:  # noqa: BLE001 — no descriptor -> no bespoke sim (arc-default)
        pass
    return ""


def chipyard_l3_selection(target: str) -> dict:
    """Which elaborated-RTL engine certifies ``target`` on the chipyard sim, and what it was chosen over.

    Routed through :mod:`rtl_engine_policy` rather than naming a binary, for the reason that module was
    written: a tier index is a FIDELITY, not a simulator, and binding ``L3 = verilator`` made the choice
    invisible and unchangeable. The program-oracle path already selects this way; the chipyard path did
    not, so on a chipyard target the policy never ran at all and a faster engine could not be adopted
    without editing this function.

    The probe is the backend's own availability check. A backend that does not know an engine RAISES
    (measured: ``available('gsim')`` -> ``GemminiError: unknown simulator 'gsim'``), which the policy
    records as a reason rather than treating as a crash -- so an engine this target cannot run is
    reported as passed-over WITH why, and the day it can run it is selected with no change here.
    """
    from ..runtime.backends import base as _backends
    from . import rtl_engine_policy as _pol

    backend = _backends.get_backend(target)

    def _probe(engine: str):
        def run() -> tuple[bool, str]:
            # A backend MAY answer with its own sentence via ``<engine>_status() -> (ok, reason)``. That
            # is the whole difference between a readable selection record and an unreadable one: "gsim
            # reports unavailable" does not distinguish a binary nobody has built yet from one that was
            # REFUSED because its build receipt describes different bytes, and only the backend knows
            # which. The attribute name is DERIVED from the engine, so a backend opts in by defining it
            # and no shared code learns an engine or a target name.
            detail = getattr(backend, f"{engine}_status", None)
            if callable(detail):
                ok, why = detail()
                return bool(ok), str(why or "")
            ok = bool(backend.available(engine))  # may raise; the policy records that as a reason
            return (
                ok,
                f"{engine} reports available for {target!r}" if ok else f"{engine} reports unavailable for {target!r}",
            )

        return run

    probes = {engine: _probe(engine) for engine in _pol.ENGINE_PRIORITY}
    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    if required:
        if required not in probes:
            raise RuntimeError(f"required RTL engine {required!r} is not registered for chipyard selection")
        # A campaign pin constrains the available choices; cost ordering cannot
        # substitute another engine when the declared one is unavailable.
        selected = _pol.select(target, {required: probes[required]})
        selected["required_engine"] = required
        selected["selection_constraint"] = "MERLIN_REQUIRED_RTL_ENGINE"
        return selected
    return _pol.select(target, probes)


@dataclass(frozen=True)
class _SimOracle:
    """A DECLARED bespoke simulator's oracle contribution, keyed by the sim-ENGINE name a target's
    contract/descriptor declares (``sim_via``) — the same additive-registry pattern as
    :func:`_sim_engine_adapters` and ``sandbox.toolchain.SIM_TOOLCHAINS``, so the shared dispatch below
    never branches on a literal engine name.

    ``exclusive`` engines grade the emitted kernel ELF DIRECTLY and REPLACE the arc/program-oracle default
    entirely — a self-hosted SIMT core (cyclotron) must not be graded by the arc command-buffer path,
    which would grade the wrong artifact. Non-exclusive engines are ADDITIVE: their higher-fidelity tiers
    layer on top of the arc default (chipyard: spike L2 / verilator L3 over the arc L3). A new bespoke sim
    registers ONE entry here + ships its adapter module; the dispatch is unchanged."""

    adapters: Callable[[str], dict]  # target -> {tier: adapter}
    available: Callable[[str], tuple[bool, str]]  # target -> (ok, reason) pre-spend probe
    exclusive: bool  # replaces (True) vs augments (False) the arc default
    has_memmap: bool = False  # exposes an SoC memory map (DRAM base derivable from the build)
    is_compile_based: bool = False  # lowers the kernel via an oracle-side compile toolchain (smoke-testable)
    l3_selection: Callable[[str], dict] | None = None
    tier_plan: Callable[[str], OracleTierPlan] | None = None
    #: target -> the rtl_engine_policy selection record for this sim's cert tier, when it has a CHOICE
    #: of elaborated-RTL engine to make. Optional: a sim with exactly one engine has nothing to report.
    #: It exists so :func:`describe_l3_engine` can ask the PLUGIN which engine it picked instead of
    #: reaching into a named backend — the same eviction the adapters and the availability probe already
    #: went through, for the same reason: a shared code path that knows one sim's backend by name is a
    #: shared code path a second sim cannot be added to without editing it.


def _chipyard_available(target: str) -> tuple[bool, str]:
    """chipyard (gemmini/mx-gemmini): the loop-tier spike binary carries GO; the mlc arc model is the
    fallback gold tier when spike is absent. Preserves the prior gemmini availability semantics exactly."""
    from .rtl import mlc_bridge

    arc_ok = mlc_bridge.arc_available(target)
    try:
        from ..runtime.backends import base as _bk

        _gem = _bk.get_backend(target)  # resolve THIS target's backend (chipyard spike availability)
        spike_ok = bool(_gem.available("spike"))
    except Exception:  # noqa: BLE001 — an unimportable backend is honestly unavailable
        spike_ok = False
    if spike_ok:
        return True, f"{target!r}: chipyard spike oracle available (loop tier)"
    if arc_ok:
        return True, f"{target!r}: chipyard sim absent but mlc arc oracle available (fallback)"
    return False, f"{target!r}: neither the chipyard spike sim nor the mlc arc oracle is available"


#: DECLARED bespoke-sim oracle registry, keyed by sim ENGINE (``sim_via``) — the seam that keeps oracle
#: routing target-name-free (a new sim engine registers here; the dispatch below is untouched). The
#: self-hosted SIMT oracle is discovered from an explicitly selected OOT support
#: package's ``plugin.sim_oracle``, which calls :func:`register_sim_oracle` at import
#: via :func:`_ensure_sim_oracles_discovered`. Reference metadata is not execution authority.
_SIM_ORACLES: dict[str, _SimOracle] = {
    "chipyard": _SimOracle(
        lambda t: _chipyard_adapters(t),
        _chipyard_available,
        exclusive=False,
        has_memmap=True,
        is_compile_based=True,
        tier_plan=chipyard_tier_plan,
    ),
}


def sim_oracle_caps(sim_via: str | None):
    """The registered :class:`_SimOracle` for a sim engine (its capability flags), or None. The
    contract-routed way for other layers (e.g. runtime_build) to ask 'does this sim expose a memory
    map / a compile toolchain?' without branching on the engine NAME. Runs plugin discovery first so a
    target-contributed engine is visible."""
    _ensure_sim_oracles_discovered()
    return _SIM_ORACLES.get(sim_via or "")


def register_sim_oracle(
    sim_via: str,
    *,
    adapters: Callable[[str], dict],
    available: Callable[[str], tuple[bool, str]],
    exclusive: bool,
    has_memmap: bool = False,
    is_compile_based: bool = False,
    l3_selection: Callable[[str], dict] | None = None,
    tier_plan: Callable[[str], OracleTierPlan] | None = None,
) -> None:
    """Register a bespoke-sim oracle under its ``sim_via`` engine name (idempotent) — the public seam a
    NEW simulator uses to plug into oracle routing without editing :func:`oracle_adapters` /
    :func:`oracle_available`. ``exclusive=True`` replaces the arc/program default (a self-hosted SIMT
    core graded on its own kernel ELF); ``exclusive=False`` layers additive tiers on top of the arc
    default (a chipyard-style sim). See :class:`_SimOracle`."""
    _SIM_ORACLES[sim_via] = _SimOracle(
        adapters=adapters,
        available=available,
        exclusive=exclusive,
        has_memmap=has_memmap,
        is_compile_based=is_compile_based,
        l3_selection=l3_selection,
        tier_plan=tier_plan,
    )


_sim_oracle_env_seen: str | None = None
_sim_oracle_lock = threading.RLock()
_sim_metadata_env_seen: str | None = None
_sim_metadata_lock = threading.RLock()


def _ensure_sim_metadata_discovered() -> None:
    """Discover only support modules that explicitly opt into core-only metadata.

    Legacy ``sim_oracle`` modules can import the optional evaluator just to
    register. Do not execute those modules to answer a metadata-only question.
    The shared synthetic namespace keeps opt-in modules idempotent across both
    discovery paths, with one registry and one plugin implementation.
    """
    global _sim_metadata_env_seen
    import os

    from ..runtime.backends import base as backends

    backends._assert_oot_plugin_ownership()
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _sim_metadata_env_seen:
        return
    # Always acquire the shared loader lock first: backend imports may discover
    # oracle metadata, and oracle imports may query backends in the same thread.
    with backends._oot_lock, _sim_metadata_lock:
        if key == _sim_metadata_env_seen:
            return
        for name, path in backends._oot_plugin_modules("sim_oracle_metadata"):
            backends._load_oot_backend(name, path, ns="merlin._oot_sim_oracles")
        _sim_metadata_env_seen = key


def _ensure_sim_oracles_discovered() -> None:
    """Load any bespoke-sim oracle a target contributes via its contract ``plugin.sim_oracle`` — a module
    that calls :func:`register_sim_oracle` at import — through the SAME OOT/reference plugin discovery the
    runtime backends use. This is what WIRES the registry: a new target adds its oracle as DATA (a plugin
    path in its contract), never a core edit to the ``_SIM_ORACLES`` literal. Re-scans only when
    ``MERLIN_TARGET_PATH`` changes; registration is idempotent, so repeated scans are harmless.

    THREAD-SAFE (double-checked locking): the ``_sim_oracle_env_seen`` marker is published only AFTER the
    discovery loop has registered every plugin oracle, under a lock. A grade fans capsules across worker
    threads that each call this; the old code set the marker BEFORE discovering, so a second thread could
    observe "already scanned" and race past with e.g. cyclotron not yet registered — collapsing an
    exclusive-sim target to the external_backend program-oracle path (the spurious 'no runner.model_ext'
    crash). Under the lock the first caller finishes registering before any other proceeds."""
    global _sim_oracle_env_seen
    import os

    from ..runtime.backends import base as _bk

    _bk._assert_oot_plugin_ownership()
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _sim_oracle_env_seen:  # fast path: already discovered for this env
        return
    with _bk._oot_lock, _sim_oracle_lock:
        if key == _sim_oracle_env_seen:  # re-check under lock (another thread may have finished)
            return
        try:
            for name, path in _bk._oot_plugin_modules("sim_oracle"):
                _bk._load_oot_backend(name, path, ns="merlin._oot_sim_oracles")
        except _bk.PluginOwnershipError:
            raise
        except Exception:  # noqa: BLE001 — discovery is best-effort; a broken plugin must not break routing
            pass
        _sim_oracle_env_seen = key  # publish ONLY after every plugin oracle is registered


def _screen_tiers_of(target: str | None) -> tuple[tuple[str, str], ...]:
    """The (tier, sim) pairs BELOW this target's RTL tiers -- the cheap functional screens, in order.

    Derived from the target's own ``tier_sim`` map minus its declared ``rtl_tiers``, so it is the
    contract that names the screen, never a "spike" literal here. A target that declares no cheap tier
    (an adapter-supplied ladder with an empty ``tier_sim``) yields ``()`` and the caller must then say
    the screen is UNAVAILABLE rather than invent one.
    """
    if not target:
        return ()
    try:
        from .runner_config import runner_config_from_manifest
        from .target_experiment import load_capability_manifest

        cfg = runner_config_from_manifest(load_capability_manifest(target))
    except Exception:  # noqa: BLE001 — unresolvable manifest
        return ()
    return tuple((t, cfg.tier_sim[t]) for t in cfg.oracle_tiers if t not in cfg.rtl_tiers and t in cfg.tier_sim)


# Existing callers patch the evaluator's selectors. Honor those overrides when that
# module is already loaded, without making compiler-side selection load evaluation.
_DEFAULT_POLICIES = {
    "_bespoke_sim_via": _bespoke_sim_via,
    "_endpoint_of": _endpoint_of,
    "chipyard_l3_selection": chipyard_l3_selection,
    "_screen_tiers_of": _screen_tiers_of,
}


def _selected_policy(name: str, target):
    legacy = sys.modules.get("merlin.targetgen.capsule_runner")
    override = getattr(legacy, name, None) if legacy is not None else None
    if override is not None and override is not _DEFAULT_POLICIES[name]:
        return override(target)
    return globals()[name](target)


def selected_sim_via(target: str) -> str:
    """Declared simulator, retaining already-loaded legacy evaluator overrides."""
    return _selected_policy("_bespoke_sim_via", target)


def selected_endpoint(target: str) -> tuple[str | None, str | None]:
    """Declared endpoint, retaining already-loaded legacy evaluator overrides."""
    return _selected_policy("_endpoint_of", target)


def select_chipyard_engine(target: str) -> dict:
    """Engine policy without importing the optional evaluator."""
    return _selected_policy("chipyard_l3_selection", target)


def selected_screen_tiers(target: str | None) -> tuple[tuple[str, str], ...]:
    """Declared screen ladder without importing the optional evaluator."""
    return _selected_policy("_screen_tiers_of", target)


def describe_l3_engine(target: str, sim_via: str | None = None) -> dict:
    """Describe selected elaborated-RTL engine evidence, never construct adapters.

    Core-only callers discover only metadata-safe support plugins. An already
    loaded evaluator retains its legacy discovery, overrides and reporting
    fallback; this function never imports that optional owner to obtain them.
    Engine selection is not proof that an adapter or submitted program can run.
    """
    legacy = sys.modules.get("merlin.targetgen.capsule_runner")
    if legacy is None:
        _ensure_sim_metadata_discovered()
    else:
        legacy._ensure_sim_oracles_discovered()
    if sim_via is None:
        sim_via = selected_sim_via(target)
    registry = getattr(legacy, "_SIM_ORACLES", _SIM_ORACLES)
    try:
        if sim_via == "chipyard":
            selection = select_chipyard_engine(target)
        else:
            oracle = registry.get(sim_via or "")
            if oracle is not None and oracle.l3_selection is not None:
                selection = oracle.l3_selection(target)
            elif oracle is not None and oracle.exclusive:
                return {
                    "available": False,
                    "target": target,
                    "sim_via": sim_via or "",
                    "reason": (
                        f"the {sim_via!r} sim oracle owns this target's cert tier and reports no engine selection"
                    ),
                }
            else:
                if sim_via and oracle is None and legacy is None:
                    raise OracleMetadataUnavailable(
                        f"declared sim {sim_via!r} has no metadata registration; add plugin.sim_oracle_metadata "
                        "with an l3_selection callback"
                    )
                from . import program_engine_policy

                selection = program_engine_policy._context().select_rtl_engine(target)
    except Exception as exc:  # unavailable engine is a report, not a certification or a substituted tier
        return {
            "available": False,
            "target": target,
            "sim_via": sim_via or "",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    from . import rtl_engine_policy

    out = dict(selection)
    out.update(
        {
            "available": True,
            "target": target,
            "sim_via": sim_via or "",
            "summary": rtl_engine_policy.describe(selection),
        }
    )
    return out


_DEFAULT_L3_REPORTER = describe_l3_engine


def selected_l3_engine_report(target: str, sim_via: str | None = None) -> dict:
    """Shared report, honoring an already-loaded legacy reporter override."""
    legacy = sys.modules.get("merlin.targetgen.capsule_runner")
    override = getattr(legacy, "describe_l3_engine", None)
    if override is not None and override is not _DEFAULT_L3_REPORTER:
        return override(target, sim_via)
    return describe_l3_engine(target, sim_via)


def sim_tier_plan(sim_via: str, target: str) -> OracleTierPlan:
    """Ask a trusted support plugin for metadata, never instantiate its adapters.

    Adapter-only plugins remain valid for evaluation. They must add a metadata
    callback before core-only corpus derivation can infer their tier inventory.
    """
    _ensure_sim_metadata_discovered()
    oracle = _SIM_ORACLES.get(sim_via)
    if oracle is None:
        raise OracleMetadataUnavailable(
            f"{target!r}: declared sim engine {sim_via!r} has no registered metadata; install a support plugin "
            "declaring plugin.sim_oracle_metadata "
            "or provide explicit required_oracle_tiers in the corpus profile (no ARC fallback inferred)"
        )
    if oracle.tier_plan is None:
        raise OracleMetadataUnavailable(
            f"{target!r}: sim engine {sim_via!r} has no tier metadata callback; add tier_plan= to "
            "register_sim_oracle or provide explicit required_oracle_tiers in the corpus profile"
        )
    plan = oracle.tier_plan(target)
    if not isinstance(plan, OracleTierPlan):
        raise OracleMetadataUnavailable(f"{target!r}: {sim_via!r} tier_plan must return OracleTierPlan")
    return replace(plan, sim_via=sim_via)


def program_tier_plan(target: str, model_ext: str | None) -> OracleTierPlan:
    """Program-ABI tier inventory from the same read-only engine probes as evaluation."""
    if not model_ext:
        raise ValueError(
            f"external_backend target {target!r}: contract declares no runner.model_ext — the program "
            f"oracle needs the model project to lay out operands; set runner.model_ext in the contract"
        )
    from . import program_engine_policy as policy

    try:
        selection = policy._context().select_rtl_engine(target)
    except policy.OracleUnavailable as exc:
        return OracleTierPlan(
            ("L2",), route="program", model_ext=model_ext, unavailable_reason=str(exc), requirements_inference_safe=True
        )
    # The selected engine is passed into the closure constructor, not selected
    # again; construction has no catch-and-drop tier policy.
    return OracleTierPlan(
        ("L2", "L3"), route="program", model_ext=model_ext, selection=selection, requirements_inference_safe=True
    )


def oracle_tier_plan(target: str, sim_via: str | None = None) -> OracleTierPlan:
    """Resolve advertised tier inventory without loading optional evaluation.

    Mirrors exclusive-sim > program endpoint > ARC/additive dispatch. Unknown
    declared engines and legacy plugins without metadata are explicit refusals;
    metadata discovery never falls through to another artifact's oracle.
    """
    _ensure_sim_metadata_discovered()
    via = selected_sim_via(target) if sim_via is None else sim_via
    oracle = _SIM_ORACLES.get(via)
    if via and oracle is None:
        return sim_tier_plan(via, target)  # actionable unavailable, never guessed ARC
    if oracle is not None and oracle.exclusive:
        return replace(sim_tier_plan(via, target), route="exclusive")
    endpoint_kind, model_ext = selected_endpoint(target)
    if endpoint_kind == "external_backend":
        return replace(program_tier_plan(target, model_ext), sim_via=via)
    if oracle is not None:
        plan = sim_tier_plan(via, target)
        # Preserve the existing ARC default even when the additive RTL tier is
        # unavailable. This is inventory parity, not a new fidelity assertion.
        return replace(plan, tiers=tuple(sorted({"L3", *plan.tiers})), route="arc")
    return OracleTierPlan(("L3",), route="arc", sim_via=via, requirements_inference_safe=True)


def inferred_oracle_tiers(target: str, sim_via: str | None = None) -> tuple[str, ...]:
    """Infer requirements only from an explicitly construction-stable inventory."""
    plan = oracle_tier_plan(target, sim_via)
    if not plan.requirements_inference_safe:
        raise OracleMetadataUnavailable(
            f"{target!r}: {plan.sim_via or plan.route!r} oracle metadata advertises tiers but does not guarantee "
            "adapter construction preserves them; declare explicit datapath.required_oracle_tiers in the "
            "corpus profile, or provide a tier_plan with a proven requirements_inference_safe=True contract"
        )
    return plan.tiers
