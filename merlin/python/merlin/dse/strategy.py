"""Compilation strategy: a first-class, hashable compilation approach.

Generalizes the ``baseline / software_visible / hardware_managed / oracle`` variant enum: a
strategy is the ``lowering_pipeline`` (a named-pass spec) plus the contract assumptions,
schedule policies, interface features, target, and cost-model overrides it implies (schema:
``compilation_strategy.schema.yaml``). Running a strategy means applying its pipeline to a
region's baseline plan and costing the result — so two strategies that differ only in pipeline
or exposed features are two directly comparable approaches.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import schemas
from merlin.common.yaml import load_yaml
from merlin.dse.cost_model import evaluate_cost
from merlin.dse.hardware_space import default_cost_model
from merlin.dse.variants import contract_plans
from merlin.dse.pipelines.builder import build_pipeline


@dataclass(frozen=True)
class Strategy:
    """One way of compiling a workload (a hashable, comparable approach)."""

    id: str
    variant_class: str
    target: str
    lowering_pipeline: str
    contract_assumptions: tuple[str, ...] = ()
    schedule_policies: tuple[str, ...] = ()
    interface_features: tuple[str, ...] = ()
    cost_model_overrides: tuple[tuple[str, float], ...] = ()
    description: str = ""

    def overrides(self) -> dict:
        return {k: v for k, v in self.cost_model_overrides}

    def pipeline(self):
        return build_pipeline(self.lowering_pipeline)

    def hash(self) -> str:
        return strategy_id(self)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "variant_class": self.variant_class,
            "target": self.target,
            "lowering_pipeline": self.lowering_pipeline,
            "contract_assumptions": list(self.contract_assumptions),
            "schedule_policies": list(self.schedule_policies),
            "interface_features": list(self.interface_features),
            "cost_model_overrides": self.overrides(),
        }
        if self.description:
            d["description"] = self.description
        return d


def strategy_id(strategy: Strategy) -> str:
    """Stable short hash keyed on the fields that change the compiled result."""
    key = "|".join([
        strategy.id, strategy.variant_class, strategy.target, strategy.lowering_pipeline,
        ",".join(sorted(strategy.interface_features)),
        ",".join(f"{k}={v}" for k, v in sorted(strategy.overrides().items())),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _baseline_plan(rpv: dict) -> dict:
    """The pre-lowering plan a strategy's pipeline transforms (the opaque I0 plan)."""
    return dict(contract_plans(rpv)["I0"])


def evaluate_strategy(strategy: Strategy, rpv: dict, cost_model: dict | None = None) -> dict:
    """Run ``strategy``'s pipeline on ``rpv`` and return its cost + provenance."""
    cm = dict(cost_model or default_cost_model())
    cm.update(strategy.overrides())
    plan = strategy.pipeline().run(_baseline_plan(rpv))
    cost = evaluate_cost(rpv, plan, cm)
    return {
        "strategy_id": strategy.hash(),
        "strategy": strategy.id,
        "variant_class": strategy.variant_class,
        "interface_features": list(strategy.interface_features),
        "plan": plan,
        "cycles": cost["cycles"],
        "energy": cost["energy"],
        "breakdown": cost["breakdown"],
    }


def default_strategies(target: str = "toy_npu") -> list[Strategy]:
    """The four canonical strategies, expressed as pass pipelines (not an enum)."""
    pre, post = "merlin-contract,merlin-schedule", "interface-lower,toynpu-lower"
    return [
        Strategy(
            id="opaque_baseline", variant_class="baseline", target=target,
            lowering_pipeline=f"{pre},{post}",
            description="opaque call; pack/load weight every step"),
        Strategy(
            id="hardware_managed_reuse", variant_class="hardware_managed", target=target,
            lowering_pipeline=f"{pre},hw-cache,{post}",
            contract_assumptions=("rhs_immutable_across_region",),
            description="hardware caches the loaded weight; no exposed residency"),
        Strategy(
            id="resident_sw_visible", variant_class="software_visible", target=target,
            lowering_pipeline=f"{pre},hoist-pack,make-resident,{post}",
            contract_assumptions=("rhs_immutable_across_region",),
            schedule_policies=("packed_rhs_policy",),
            interface_features=("resident_packed_tensor",),
            description="expose resident packed RHS to software"),
        Strategy(
            id="resident_commit_sw_visible", variant_class="software_visible", target=target,
            lowering_pipeline=f"{pre},hoist-pack,make-resident,defer-commit,{post}",
            contract_assumptions=("rhs_immutable_across_region",),
            schedule_policies=("packed_rhs_policy", "accumulator_commit_policy"),
            interface_features=("resident_packed_tensor", "accumulator_commit"),
            description="resident RHS + accumulator commit"),
        Strategy(
            id="oracle", variant_class="oracle", target=target,
            lowering_pipeline=f"{pre},hoist-pack,make-resident,defer-commit,batch-dispatch,{post}",
            interface_features=("resident_packed_tensor", "accumulator_commit"),
            description="perfect residency, commit, and dispatch batching"),
    ]


# Which pass exposes which interface feature, and which variant class it implies.
FEATURE_PASSES = {
    "make-resident": "resident_packed_tensor",
    "defer-commit": "accumulator_commit",
}
_PRE = ["merlin-contract", "merlin-schedule"]
_POST = ["interface-lower", "toynpu-lower"]
_EFFECT_ORDER = ["hw-cache", "hoist-pack", "make-resident", "defer-commit", "batch-dispatch"]


def strategy_from_passes(effect_passes, target: str = "toy_npu",
                         id: str | None = None) -> Strategy:
    """Assemble a Strategy from a set of effect passes (features/variant_class derived)."""
    chosen = [p for p in _EFFECT_ORDER if p in set(effect_passes)]
    features = tuple(FEATURE_PASSES[p] for p in chosen if p in FEATURE_PASSES)
    if "batch-dispatch" in chosen and {"make-resident", "defer-commit"} <= set(chosen):
        variant = "oracle"
    elif features:
        variant = "software_visible"
    elif "hw-cache" in chosen:
        variant = "hardware_managed"
    else:
        variant = "baseline"
    pipeline = ",".join(_PRE + chosen + _POST)
    sid = id or ("strat_" + "_".join(chosen) if chosen else "opaque_baseline")
    return Strategy(id=sid, variant_class=variant, target=target,
                    lowering_pipeline=pipeline, interface_features=features)


def effect_passes(strategy: Strategy) -> list[str]:
    """The effect passes in a strategy's pipeline (drops the structural lowering passes)."""
    from merlin.dse.pipelines.builder import parse_spec
    return [p for p in parse_spec(strategy.lowering_pipeline) if p in _EFFECT_ORDER]


def behavior_descriptors(strategy: Strategy) -> dict:
    """Map a strategy to MAP-Elites behavior dimensions (see search_space.schema.yaml)."""
    eff = set(effect_passes(strategy))
    feats = set(strategy.interface_features)
    if "resident_packed_tensor" in feats:
        memory = "resident_object"
    elif "hw-cache" in eff:
        memory = "hardware_cache"
    else:
        memory = "scratchpad"
    control = "persistent_command_buffer" if "batch-dispatch" in eff else "blocking"
    granularity = "fused_region" if "accumulator_commit" in feats else "tile_op"
    return {"memory_abstraction": memory, "control_abstraction": control,
            "granularity": granularity}


def from_dict(d: dict) -> Strategy:
    """Build a Strategy from a ``compilation_strategy``-schema mapping."""
    schemas.validate_or_raise(d, "compilation_strategy")
    ov = d.get("cost_model_overrides", {}) or {}
    return Strategy(
        id=d["id"],
        variant_class=d["variant_class"],
        target=d["target"],
        lowering_pipeline=d["lowering_pipeline"],
        contract_assumptions=tuple(d.get("contract_assumptions", []) or []),
        schedule_policies=tuple(d.get("schedule_policies", []) or []),
        interface_features=tuple(d.get("interface_features", []) or []),
        cost_model_overrides=tuple(sorted((k, v) for k, v in ov.items())),
        description=d.get("description", ""),
    )


def load_strategies(source=None) -> list[Strategy]:
    """Load strategies. ``source`` may be a directory of YAML files, a list of paths, a list of
    dicts, or None (the built-in :func:`default_strategies`)."""
    if source is None:
        return default_strategies()
    if isinstance(source, (str, Path)):
        p = Path(source)
        files = sorted(p.glob("*.yaml")) if p.is_dir() else [p]
        return [from_dict(load_yaml(f)) for f in files]
    out: list[Strategy] = []
    for item in source:
        out.append(from_dict(item) if isinstance(item, dict) else from_dict(load_yaml(item)))
    return out
