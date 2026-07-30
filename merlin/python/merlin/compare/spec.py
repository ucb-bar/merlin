"""Target-agnostic comparison SPEC — the single source of truth a ``merlin-compare`` run is driven by.

A spec names the comparison POINTS (``configs``), the ``workloads`` (whole-model names and/or
isolated GEMM shapes), the ``target`` (``k1`` implemented; ``spike``/``gemmini``/``npu`` are seams),
and the measurement ``metric``/``reps``. Nothing here is RVV-specific; the RVV/K1 mapping lives in
the ingest layer. Parsing is total and validating so a malformed spec fails loud, not silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# The comparison-point kinds. A config is EITHER an "ours" feature-set, OR a kernel backend
# (xnnpack/openblas), OR an EXTERNAL baseline framework (its own end-to-end compiler/runtime), OR
# the frozen baseline.
_OURS_PREFIX = "ours"
_KERNEL_BACKENDS = ("xnnpack", "openblas")
# Independent external compilers/runtimes (see merlin.baselines.contract.FRAMEWORKS and
# third_party/baselines/). Unlike kernel backends, these run the whole model on their OWN stack;
# their measurements are produced by the merlin.baselines harness and ingested from its result JSON.
_EXTERNAL_FRAMEWORKS = ("tvm", "executorch", "buddy", "exo", "ggml")
_BASELINE = "baseline"

# Platform execution SUBSTRATES that are not dialect targets under merlin/targets/ (a board or a
# simulator seam, not a registered dialect) — kept first-class so existing callers keep working. The
# dialect targets are DISCOVERED from the target registry and unioned in at parse time, so a newly
# registered dialect (e.g. gemmini) becomes comparable with no edit here. Only k1 is implemented in
# v1; every other known target is a declared seam.
_PLATFORM_TARGETS = ("k1", "spike", "npu")
_IMPLEMENTED_TARGETS = ("k1",)


def _known_targets() -> tuple[str, ...]:
    """The comparable target set: the platform substrates above unioned with the dialect targets the
    registry discovers (curated in-tree targets + any ``MERLIN_TARGET_PATH`` packages). A registry
    import/discovery failure degrades to the platform substrates alone (never fatal)."""
    discovered: tuple[str, ...] = ()
    try:
        from merlin.targetgen.target_registry import all_targets
        discovered = tuple(all_targets())
    except Exception:
        discovered = ()
    return tuple(sorted(set(_PLATFORM_TARGETS) | set(discovered)))

_METRICS = ("wall", "instret")


@dataclass(frozen=True)
class Config:
    """One comparison point.

    kind: ``baseline`` | ``ours`` | ``kernel_backend`` | ``external``.
    name: the spec-facing name (e.g. ``ours_wholemodel_vf``, ``xnnpack``, ``baseline``).
    compiler_features: for ``ours`` configs, the feature-set (informational here; the ingest layer
        maps the spec name to the cached JSON key). For non-ours, empty.
    """
    name: str
    kind: str
    compiler_features: tuple[str, ...] = ()

    @staticmethod
    def parse(raw: Any) -> "Config":
        if isinstance(raw, str):
            name, feats = raw, ()
        elif isinstance(raw, dict):
            name = raw.get("name")
            feats = tuple(raw.get("compiler_features", ()) or ())
            if not name:
                raise ValueError(f"config dict missing 'name': {raw!r}")
        else:
            raise ValueError(f"config must be str or dict, got {type(raw).__name__}: {raw!r}")
        if name == _BASELINE:
            kind = "baseline"
        elif name in _KERNEL_BACKENDS:
            kind = "kernel_backend"
        elif name in _EXTERNAL_FRAMEWORKS:
            kind = "external"
        elif name.startswith(_OURS_PREFIX):
            kind = "ours"
        else:
            raise ValueError(
                f"unknown config '{name}': must be 'baseline', one of {_KERNEL_BACKENDS}, "
                f"one of {_EXTERNAL_FRAMEWORKS}, or start with '{_OURS_PREFIX}'")
        return Config(name=name, kind=kind, compiler_features=feats)


@dataclass(frozen=True)
class Workload:
    """A workload to compare over: a whole-model (``kind='model'``, e.g. ``openvla``) or an isolated
    GEMM shape (``kind='gemm'``, ``gemm:64`` -> M=N=K=64, or ``gemm:17x192x576``)."""
    name: str          # the spec token, e.g. "openvla" or "gemm:64"
    kind: str          # "model" | "gemm"
    mnk: tuple[int, int, int] | None = None   # for gemm

    @staticmethod
    def parse(raw: str) -> "Workload":
        if not isinstance(raw, str):
            raise ValueError(f"workload must be str, got {type(raw).__name__}: {raw!r}")
        if raw.startswith("gemm:"):
            body = raw.split(":", 1)[1]
            if "x" in body:
                parts = body.split("x")
                if len(parts) != 3:
                    raise ValueError(f"gemm shape '{raw}' must be 'gemm:N' or 'gemm:MxNxK'")
                m, n, k = (int(p) for p in parts)
            else:
                m = n = k = int(body)
            return Workload(name=raw, kind="gemm", mnk=(m, n, k))
        return Workload(name=raw, kind="model")


@dataclass(frozen=True)
class Spec:
    configs: tuple[Config, ...]
    workloads: tuple[Workload, ...]
    target: str = "k1"
    metric: str = "wall"
    reps: int = 5
    label: str = "compare"
    # extra spec-level provenance (free-form, recorded into the manifest verbatim).
    notes: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(raw: dict) -> "Spec":
        if not isinstance(raw, dict):
            raise ValueError("spec must be a mapping")
        cfgs = raw.get("configs")
        wls = raw.get("workloads")
        if not cfgs:
            raise ValueError("spec must list at least one 'configs' entry")
        if not wls:
            raise ValueError("spec must list at least one 'workloads' entry")
        target = raw.get("target", "k1")
        known = _known_targets()
        if target not in known:
            raise ValueError(f"unknown target '{target}'; known: {known}")
        if target not in _IMPLEMENTED_TARGETS:
            raise ValueError(
                f"target '{target}' is a declared seam but not implemented in v1 "
                f"(implemented: {_IMPLEMENTED_TARGETS})")
        metric = raw.get("metric", "wall")
        if metric not in _METRICS:
            raise ValueError(f"unknown metric '{metric}'; known: {_METRICS}")
        return Spec(
            configs=tuple(Config.parse(c) for c in cfgs),
            workloads=tuple(Workload.parse(w) for w in wls),
            target=target,
            metric=metric,
            reps=int(raw.get("reps", 5)),
            label=str(raw.get("label", "compare")),
            notes=dict(raw.get("notes", {}) or {}),
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "Spec":
        import yaml
        return Spec.parse(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "target": self.target,
            "metric": self.metric,
            "reps": self.reps,
            "configs": [
                {"name": c.name, "kind": c.kind,
                 "compiler_features": list(c.compiler_features)}
                for c in self.configs
            ],
            "workloads": [
                {"name": w.name, "kind": w.kind, "mnk": list(w.mnk) if w.mnk else None}
                for w in self.workloads
            ],
            "notes": dict(self.notes),
        }
