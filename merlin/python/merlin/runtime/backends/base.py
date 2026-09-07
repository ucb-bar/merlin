"""Backend taxonomy + registry — address runtime backends by target CLASS, not instance.

The runtime backends are per-instance modules (``spike``, ``saturn_vec``, ``gemmini``, ``muon``,
``spike_model``, ``zephyr_model``) that share one shape — a ``Backend``: resolve toolchain →
``compile_command_buffer`` (→ ELF) → ``run_elf`` → ``parse_output`` (→ outputs+metrics) →
``run_command_buffer`` (compile+run+parse, gated on the reference oracle). Historically each was
imported by name; this module classifies them by **target class** (CPU / GPU / NPU) so callers and
tooling can reason about "the CPU/RVV backend" or "the NPU/systolic backend" rather than a specific
silicon instance — the same instance→class generalization the dialect layer got via
``xdsl_dialects.targets.factory``.

Scope (step 1): the taxonomy + a registry (name → module + class) + the shared ``Backend`` Protocol.
The per-instance modules keep their current behavior; collapsing their copy-pasted plumbing
(toolchain resolve / ``OUT/METRIC/DONE`` parse / reference gate) into a shared base is the follow-up
(and must re-certify the frozen gemmini path byte-for-byte).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class TargetClass(str, Enum):
    """The coarse hardware class a backend targets (not the specific silicon instance).

    The members are DERIVED, in the sense that matters: which class a target is follows from the
    compute ENGINES it declares, and that relationship lives in one place --
    :data:`merlin.kernels.engines.TARGET_CLASS_OF_ENGINE` -- rather than being restated here in a
    comment. ``targetgen.families`` used to assert in prose that the compute-unit kinds were "aligned
    with" this enum; five kinds cannot align with three tokens by inspection, so it is now a check
    (:func:`merlin.kernels.engines.check_class_map_is_total`).

    The enum itself stays, because a backend genuinely declares its class when it registers and that
    is a fact about the backend. What is derived is the correspondence, so the two cannot drift.

    Deliberately NOT annotated with which silicon is which: naming targets here is how a coarse class
    turns into a lookup table for specific hardware. Use :func:`target_class_for` to ask.
    """

    CPU = "cpu"     # a scalar and/or lane (vector) engine — a CPU, with or without SIMD
    GPU = "gpu"     # a threads-of-control (SIMT) engine, with or without a tensor unit inside it
    NPU = "npu"     # an array engine (systolic wavefront or outer-product tile) as the outermost unit


def target_class_for(target: str) -> "TargetClass | None":
    """The class a target's DECLARED engines imply, or None when it declares none.

    None is a real answer -- "nobody has said what silicon this is" -- and must not be defaulted to
    CPU: guessing the class of an undeclared accelerator is how a result gets attributed to the wrong
    kind of device. Imported lazily so the runtime does not depend on the compiler packages at import
    time; returns None if they are unavailable rather than failing a backend lookup.
    """
    try:
        from merlin.kernels import engines as _engines
    except Exception:                        # noqa: BLE001 — kernels unavailable in this sandbox
        return None
    got = _engines.target_class_for(_engines.engines_for(target))
    return TargetClass(got) if got else None


class BackendKind(str, Enum):
    KERNEL = "kernel"          # compiles+runs one command buffer (spike, gemmini, muon, saturn_vec)
    WHOLE_MODEL = "whole_model"  # runs a whole captured model (spike_model, zephyr_model)
    MATMUL_ROUTE = "matmul_route"  # routes matmuls to an external/hand GEMM for attribution
                                   # (xnnpack/openblas/ours on a board; xnnpack on the host)


#: Closed vocabulary for SOFTWARE execution capabilities a backend may declare.  These are kept out
#: of ``merlin.perf.profile.TRAITS`` deliberately: whether the runner can build a whole-program ABI or
#: place a warm and measured invocation under one counter is a property of the backend/harness, not a
#: discovered property of the accelerator RTL.  A shared performance family may require both kinds of
#: evidence, but it must not turn missing tooling into a claim about the hardware.
EXECUTION_CAPABILITIES: tuple[str, ...] = (
    "whole_program_kernel_abi",
    "warm_single_counter_region_cycles",
)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    target_class: TargetClass
    kind: BackendKind
    module: str                  # dotted import path, loaded lazily via get_backend()


# The registry is populated two ways so the CORE never hardcodes a shipped-accelerator name:
#  * the generic ISA-class instances (RVV/host CPU kernels, whole-model runners, matmul-route
#    attribution) are seeded here directly — their identifiers are ISA-class/tool names, not silicon
#    instances, so naming them keeps the core target-agnostic;
#  * the silicon-specific reference instances (the NPU/GPU accelerator backends) SELF-REGISTER from
#    their own module via ``register(...)`` and are discovered lazily by ``_ensure_discovered()`` — so
#    a new accelerator backend is one ``register`` line in ITS OWN module (or an out-of-tree package
#    reached via ``MERLIN_TARGET_PATH``), and the core carries no name -> module map for it.
_REGISTRY: dict[str, BackendInfo] = {
    "spike":        BackendInfo("spike", TargetClass.CPU, BackendKind.KERNEL,
                                "merlin.runtime.backends.spike"),
    # NB: "saturn_vec" is NOT seeded here — it was evicted to its own reference package
    # (merlin/targets/saturn/backend/) and self-registers via plugin discovery (see _ensure_oot_discovered).
    "spike_model":  BackendInfo("spike_model", TargetClass.CPU, BackendKind.WHOLE_MODEL,
                                "merlin.runtime.backends.spike_model"),
    "zephyr_model": BackendInfo("zephyr_model", TargetClass.CPU, BackendKind.WHOLE_MODEL,
                                "merlin.runtime.backends.zephyr_model"),
    # matmul-routing attribution backends (CPU-class): route routable matmuls to an external/hand
    # GEMM — the RVV board variants (xnnpack/openblas/ours) + the x86 host xnnpack reference.
    "xnnpack_board":  BackendInfo("xnnpack_board", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.xnnpack_board"),
    "openblas_board": BackendInfo("openblas_board", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.openblas_board"),
    "ours_board":     BackendInfo("ours_board", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.ours_board"),
    "xnnpack_host":   BackendInfo("xnnpack_host", TargetClass.CPU, BackendKind.MATMUL_ROUTE,
                                  "merlin.runtime.backends.xnnpack_host"),
}

# Names seeded above (the generic ISA-class instances). Discovery imports every OTHER submodule of this
# package so any accelerator backend that self-registers is picked up — without the core naming it.
_SEEDED: frozenset[str] = frozenset(_REGISTRY)
_discovered = False

#: Backends whose module RAISED while loading, ``name -> "ExcType: message"``. A load failure used to
#: be swallowed whole, which made a BROKEN backend indistinguishable from one that was never declared.
#: That is not hypothetical: one reference package's backend imported a core module that had since been
#: evicted out of the core tree, discovery ate the ImportError, the backend never registered, and its
#: whole suite SKIPPED -- 17 skipped, 0 run -- for as long as that lasted. Green, and measuring nothing.
#: Recording the reason keeps the repo's fail-closed rule: an unloadable backend is UNKNOWN and says
#: why, never silently absent. No target is named here; the core holds no name -> module map.
_LOAD_FAILURES: dict[str, str] = {}


def load_failures() -> dict[str, str]:
    """``name -> reason`` for every backend module that raised while loading (empty when all loaded)."""
    return dict(_LOAD_FAILURES)


def register(info: BackendInfo) -> None:
    """Register a backend (idempotent per name). A reference accelerator backend calls this from its
    OWN module at import time, so the core never carries a name -> module map for a specific silicon."""
    _REGISTRY[info.name] = info


def _ensure_discovered() -> None:
    """Import the not-yet-seeded submodules of this package so their ``register(...)`` calls run, then
    load any backend modules declared by out-of-tree target packages on ``MERLIN_TARGET_PATH`` (so an
    evicted accelerator backend self-registers from ITS OWN package, not from core). Guarded per module:
    a backend whose optional deps are missing simply fails to register instead of breaking the registry.

    In-tree discovery runs at most once; OOT discovery re-checks whenever ``MERLIN_TARGET_PATH`` changes
    (so a caller that sets the env then queries the registry sees the package's backend). Neither runs at
    ``import base`` — only on the first registry query — so importing the registry does NOT eagerly pull
    in the accelerator backends."""
    _ensure_oot_discovered()  # per-env (cheap when MERLIN_TARGET_PATH is unchanged); before the one-shot
    global _discovered
    if _discovered:
        return
    _discovered = True  # set first: a backend importing base during its own load must not re-enter
    pkg = importlib.import_module(__name__.rsplit(".", 1)[0])
    for mod in pkgutil.iter_modules(pkg.__path__):
        leaf = mod.name
        if leaf.startswith("_") or leaf == __name__.rsplit(".", 1)[1] or leaf in _SEEDED:
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{leaf}")
        except Exception as exc:  # noqa: BLE001 — missing optional deps: do not register, but SAY SO
            _LOAD_FAILURES[leaf] = f"{type(exc).__name__}: {exc}"


# --- out-of-tree backend discovery (MERLIN_TARGET_PATH) ---------------------------------------------
# A target definition is a self-contained OUT-OF-TREE PACKAGE (``merlin.targetgen.target_registry``);
# its contract's ``plugin`` block may name a runtime backend module (``plugin.backend``, a path relative
# to the package root). We import that module BY FILE PATH so its ``register(...)`` runs — so a reference
# accelerator backend evicted to a published ``<target>-mlir`` package plugs back in with ZERO core
# changes and the core carries no name -> module map for it. Target-agnostic: the PACKAGE names its own
# backend file; nothing here is keyed on a specific target.
#
# ``plugin.backend`` may point at EITHER a single ``.py`` file OR a directory (a package with an
# ``__init__.py``). A directory is loaded as a package with its own ``__path__``, so a backend whose
# implementation spans several relative-import-coupled modules (``from .codegen import ...``) works
# unchanged out-of-tree — the module that would otherwise force a single-file backend to inline all its
# helpers. Only the parent-relative imports of an evicted backend need rewriting to absolute
# (``from ..metrics`` -> ``from merlin.runtime.metrics``); sibling imports inside the package stay relative.
_oot_env_seen: str | None = None


def _oot_plugin_modules(key: str = "backend") -> list[tuple[str, Path]]:
    """``(target-name, module-file/dir)`` for every target whose contract's ``plugin`` block declares a
    module under ``key`` (``backend`` for a runtime backend, ``sim_oracle`` for a bespoke-sim oracle) —
    OOT packages reachable via ``MERLIN_TARGET_PATH`` / the freshly-generated home, AND curated in-tree
    reference targets (so a reference module evicted into its own package dir is auto-loaded too). Empty
    when targetgen is unavailable or nothing declares one — the core degrades honestly. This is the ONE
    seam that lets a new target contribute a backend OR an oracle as data (a plugin path), never a core
    edit to a registry literal."""
    try:
        from ...targetgen import target_registry
    except Exception:  # noqa: BLE001 — targetgen optional; no OOT discovery without it
        return []
    try:
        names = list(target_registry.external_targets())
    except Exception:  # noqa: BLE001 — a malformed search path must not break the registry
        return []
    # curated in-tree REFERENCE targets may ALSO declare a plugin.backend — the physical eviction of a
    # reference backend into its own package dir under merlin/targets/<name>/backend/. They are NOT in
    # external_targets() (which is env + generated only), so add them here; ``plugin()`` omits ``path`` for a
    # reference (external_root is None), so inject the reference package root. External wins a name clash.
    try:
        ref_names = [n for n in target_registry.list_targets() if n not in names]
    except Exception:  # noqa: BLE001 — a missing reference tree must not break discovery
        ref_names = []
    out: list[tuple[str, Path]] = []
    for name in [*names, *ref_names]:
        try:
            info = target_registry.resolve(name)
            plugin = info.plugin()
        except Exception:  # noqa: BLE001 — skip a package whose contract will not parse
            continue
        rel = plugin.get(key)
        root = plugin.get("path") or str(info.base)   # reference: base is the package root (no injected path)
        if not rel or not root:
            continue
        path = Path(root) / rel
        # A plugin module is a single .py file OR a package directory (dir with __init__.py).
        if path.is_file() or (path.is_dir() and (path / "__init__.py").is_file()):
            out.append((name, path))
    return out


def _oot_backend_modules() -> list[tuple[str, Path]]:
    """Back-compat shim: the ``plugin.backend`` modules (see :func:`_oot_plugin_modules`)."""
    return _oot_plugin_modules("backend")


def _load_oot_backend(name: str, path: Path, *, ns: str = "merlin._oot_backends") -> None:
    """Import an OOT plugin module by file path (under a stable synthetic module name) so its module-level
    self-registration (``register(...)`` for a backend, ``register_sim_oracle(...)`` for an oracle) runs.
    Idempotent: a module already loaded is left as-is. The backend's ``BackendInfo.module`` is its own
    ``__name__`` (this synthetic name), so ``get_backend`` re-resolves it from ``sys.modules`` without the
    core knowing the package layout.

    ``path`` is either a single ``.py`` file (loaded as a leaf module) or a package DIRECTORY (loaded
    as ``merlin._oot_backends.<name>`` with ``submodule_search_locations`` set to the dir, so the
    backend's own ``from .sibling import ...`` resolve). Either way the top-level name is the same, so
    ``BackendInfo(module=__name__)`` in the file or the package ``__init__`` lands identically.

    ``ns`` is the synthetic top-level namespace, so a target's ``plugin.backend`` and its
    ``plugin.sim_oracle`` (loaded via the same mechanism) get distinct module names and never collide."""
    modname = f"{ns}.{name}"
    if modname in sys.modules:
        return
    if path.is_dir():
        init = path / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            modname, init, submodule_search_locations=[str(path)]
        )
    else:
        spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)  # runs the backend's register(...) self-registration
    except Exception as exc:  # noqa: BLE001 — missing deps: do not register, but RECORD the reason
        sys.modules.pop(modname, None)
        _LOAD_FAILURES[name] = f"{type(exc).__name__}: {exc}"


_oot_lock = threading.RLock()   # guards OOT discovery: the sentinel is published only after it completes


def _ensure_oot_discovered() -> None:
    """Load OOT backend modules for the current ``MERLIN_TARGET_PATH``. Re-scans only when the env value
    changes (cheap no-op otherwise); registration is idempotent, so repeated scans are harmless. Note:
    a backend that has already self-registered stays registered even if the env is later cleared (Python
    cannot un-import it) — a fresh process is the clean way to observe the env-unset state."""
    global _oot_env_seen
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _oot_env_seen:
        return
    # Publish the sentinel only AFTER every module has registered, under a lock.
    #
    # Setting it first is a check-then-set race, and grading fans capsules out across threads: worker A
    # passes the check and starts loading (imports + RTL facts, not fast), worker B sees the sentinel
    # already set, returns immediately, and then reads an EMPTY registry -> `KeyError: 'gemmini'`, which
    # the runner reports as `spike invocation failed: 'gemmini'` / tool_crash. Measured on the first
    # gemmini arm-4 run: 15 of 20 capsules crashed that way and 5 passed, purely on thread timing -- an
    # agent that had actually solved all 20 graded 5.
    with _oot_lock:
        if key == _oot_env_seen:          # another thread completed discovery while we waited
            return
        for name, path in _oot_backend_modules():
            _load_oot_backend(name, path)
        _oot_env_seen = key


@runtime_checkable
class Backend(Protocol):
    """The shape every kernel backend module exposes (module-level functions)."""

    def available(self) -> bool: ...
    def compile_command_buffer(self, cb: dict[str, Any], workdir: Any, **kw: Any) -> Any: ...
    def run_elf(self, elf: Any, **kw: Any) -> str: ...
    def parse_output(self, text: str) -> tuple[dict, dict]: ...
    def run_command_buffer(self, cb: dict[str, Any], **kw: Any) -> dict: ...


def list_backends() -> list[str]:
    _ensure_discovered()
    return sorted(_REGISTRY)


def info(name: str) -> BackendInfo:
    _ensure_discovered()
    return _REGISTRY[name]


def class_of(name: str) -> TargetClass:
    _ensure_discovered()
    return _REGISTRY[name].target_class


def backends_of_class(target_class: TargetClass) -> list[str]:
    _ensure_discovered()
    return sorted(n for n, b in _REGISTRY.items() if b.target_class == target_class)


from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe


def harness_build_recipe(target: str) -> HarnessBuildRecipe:
    """The build recipe ``target``'s backend declares, or a clear refusal.

    Optional capability: a backend that never builds a bare-metal harness (a host or simulator-only
    backend) simply does not define it, and callers that need one get told which target lacks it
    rather than an AttributeError from somewhere deeper.
    """
    backend = get_backend(target)
    factory = getattr(backend, "harness_build_recipe", None)
    if factory is None:
        raise NotImplementedError(
            f"backend for target {target!r} declares no harness_build_recipe; it cannot build a "
            f"runner-owned bare-metal harness. Add one to the backend module if it should.")
    return factory()


def harness_renderer(target: str):
    """``target``'s runner-owned harness renderer — ``render_harness(cb, *, target) -> str``.

    Optional, like :func:`harness_build_recipe`, and separate from it on purpose: the BUILD is a
    toolchain description a contract could plausibly carry, whereas the harness body pads to a
    target's tile edge and lays out its accumulator readout. That is codegen, and putting it behind a
    contract key would define a key no second target could implement.
    """
    backend = get_backend(target)
    render = getattr(backend, "render_harness", None)
    if render is None:
        raise NotImplementedError(
            f"backend for target {target!r} declares no render_harness; the runner cannot write a "
            f"harness for it. Add one to the backend module if it should.")
    return render


def name_of_module(module_name: str) -> str:
    """The registered backend name for an already-imported backend module.

    The reverse of :func:`get_backend`, and it exists for one specific situation: a caller that is
    still welded to a particular backend (it imports the module directly) and needs that backend's
    TARGET NAME to look something up — a contract, a manifest. Reading the identity of the module it
    already holds is strictly better than writing the name again as a literal, because the literal
    would be a second, independent place to update and the gates cannot tell it apart from a real
    hardcoded target. When the weld is removed the call goes with it.
    """
    _ensure_discovered()
    for info in _REGISTRY.values():
        if info.module == module_name:
            return info.name
    raise KeyError(f"no registered backend for module {module_name!r}")


def get_backend(name: str):
    """Lazily import + return the backend module for ``name``.

    Raises ``KeyError`` if unregistered -- and when the name is missing because its module RAISED
    while loading, the recorded reason is in the message. Without that, a backend broken by a
    refactor elsewhere in the tree presents exactly like a backend that was never declared, and the
    suites that depend on it skip green instead of failing.
    """
    _ensure_discovered()
    if name not in _REGISTRY:
        why = _LOAD_FAILURES.get(name)
        raise KeyError(f"{name}: backend module failed to load -- {why}" if why else name)
    return importlib.import_module(_REGISTRY[name].module)


def execution_capability_facts(target: str) -> dict[str, dict[str, Any]]:
    """Tri-state, evidence-bearing software capabilities for ``target``'s registered backend.

    A loaded backend settles every name in :data:`EXECUTION_CAPABILITIES`: a declared name is true and
    an omitted one is false.  If the backend itself cannot be resolved, support is UNKNOWN rather than
    false because no implementation was available to inspect.  Nothing is inferred from a target name,
    target class, opcode, or hardware trait; adding a new target means declaring capabilities in that
    target's own backend module.
    """
    try:
        backend = get_backend(target)
    except Exception as exc:  # noqa: BLE001 -- an unavailable plugin is UNKNOWN and carries why
        why = f"{type(exc).__name__}: {exc}"
        return {
            name: {
                "satisfied": None,
                "tier": "not_established",
                "evidence": f"backend {target!r} could not be resolved: {why}",
                "missing": [f"a loadable backend declaring execution capability {name!r}"],
            }
            for name in EXECUTION_CAPABILITIES
        }

    raw = getattr(backend, "EXECUTION_CAPABILITIES", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(
            f"backend {target!r} EXECUTION_CAPABILITIES must be a name -> evidence mapping")
    unknown = sorted(set(raw) - set(EXECUTION_CAPABILITIES))
    if unknown:
        raise ValueError(
            f"backend {target!r} declares unknown execution capabilities {unknown}; canonical names "
            f"are {list(EXECUTION_CAPABILITIES)}")
    facts: dict[str, dict[str, Any]] = {}
    for name in EXECUTION_CAPABILITIES:
        evidence = raw.get(name)
        if evidence is not None and (not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError(
                f"backend {target!r} execution capability {name!r} needs non-empty evidence text")
        supported = evidence is not None
        facts[name] = {
            "satisfied": supported,
            "tier": "backend_declared",
            "evidence": (f"backend {target!r} ({backend.__name__}) declares: {evidence}"
                         if supported else
                         f"backend {target!r} ({backend.__name__}) does not declare this capability"),
            "missing": ([] if supported else
                        [f"backend declaration and implementation of {name!r}"]),
        }
    return facts


# --- shared backend plumbing (the copy-pasted console protocol, collapsed) --------------------------
def _strip_warning_fragments(text: str) -> str:
    """Drop each line's ``%Warning:``/``Warning:`` fragment onward (stray Verilator noise), keeping
    any text before it — the structural equivalent of the old ``%?Warning:[^\\n]*`` substitution."""
    out = []
    for line in text.split("\n"):
        idx = line.find("Warning:")
        if idx != -1:
            if idx > 0 and line[idx - 1] == "%":
                idx -= 1
            line = line[:idx]
        out.append(line)
    return "\n".join(out)


def parse_console(text: str, *, error_cls: type[Exception] = RuntimeError,
                  strip_warnings: bool = False, tolerant_metric: bool = False,
                  value_parser=int) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the shared ``OUT``/``METRIC``/``DONE`` backend console protocol into (outputs, raw_metrics).

    Every backend prints results the same way — ``OUT <name> <rows> <cols> v...`` /
    ``METRIC <name> <int>`` / ``DONE`` — so the parser is shared; the small per-backend variations are
    flags: ``strip_warnings`` drops Verilator ``%Warning:`` fragments (gemmini/verilator),
    ``tolerant_metric`` skips malformed METRIC lines instead of raising (gemmini), ``value_parser`` is
    ``int`` for int8/systolic/CPU targets and ``float`` for fp SIMT targets. ``error_cls`` is the
    backend's own exception type (so messages/raises are unchanged from the hand-written versions)."""
    if strip_warnings:
        text = _strip_warning_fragments(text)
    outputs: dict[str, list] = {}
    raw: dict[str, int] = {}
    done = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "OUT":
            name, rows, cols = parts[1], int(parts[2]), int(parts[3])
            vals = [value_parser(v) for v in parts[4:]]
            if len(vals) != rows * cols:
                raise error_cls(f"OUT {name}: expected {rows * cols} values, got {len(vals)}")
            outputs[name] = [vals[r * cols:(r + 1) * cols] for r in range(rows)]
        elif parts[0] == "METRIC":
            if tolerant_metric:
                try:
                    raw[parts[1]] = int(parts[2])
                except (IndexError, ValueError):
                    pass
            else:
                raw[parts[1]] = int(parts[2])
        elif parts[0] == "DONE":
            done = True
    if not done:
        raise error_cls(f"run did not reach DONE; output was:\n{text[:2000]}")
    return outputs, raw


from merlin.runtime.fp8_formats import float_format_of


def decode_float_readback(outputs: dict[str, list], dtypes: dict[str, str]) -> dict[str, list]:
    """Decode a FLOAT-declared output that arrived over the console as its stored CONTAINER word.

    The shared ``OUT <name> <rows> <cols> <v...>`` protocol carries whatever the device harness
    printed. A harness whose ``printf`` is integer-only (the systolic/CPU targets: no float formatting
    in a bare-metal libc, and none wanted — a decimal round-trip would be the lossy step) prints the
    destination buffer's raw word, so an ``f32`` result comes back as the 32-bit pattern that was
    stored and a ``bf16`` result as its 16-bit pattern. The VALUE is recovered here, from the dtype
    the buffer was DECLARED in — the same fact the harness sized that buffer from.

    Keyed on the declared dtype, never on a target: a backend prints floats or it prints containers,
    and which one it did is visible in the values themselves. An output is decoded only when its
    declared dtype is a float format AND every value arrived as an integer; a backend that already
    formats decimals (the fp SIMT consoles parse with ``value_parser=float``) hands back floats and is
    left untouched. Integer-declared outputs are returned unchanged, so the integer path is
    byte-identical.

    Returns a new mapping; the input is not mutated.
    """
    from merlin.runtime import fp8_formats as _ff
    decoded: dict[str, list] = {}
    for name, rows in outputs.items():
        fmt = float_format_of(dtypes.get(name, ""))
        flat = [v for row in rows for v in row] if rows and isinstance(rows[0], list) else list(rows)
        if fmt is None or not flat or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in flat):
            decoded[name] = rows
            continue
        values = [float(v) for v in _ff.codes_to_f32(flat, fmt)]
        if rows and isinstance(rows[0], list):
            out, at = [], 0
            for row in rows:
                out.append(values[at:at + len(row)])
                at += len(row)
            decoded[name] = out
        else:
            decoded[name] = values
    return decoded
