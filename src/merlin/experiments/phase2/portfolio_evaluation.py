"""Stateful, bounded analytical portfolio evaluation and frozen provider admission."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import broker_evidence as BE
from . import contracts as P2_CONTRACTS
from . import stage_inputs as INPUTS


class FastPortfolioEvaluation:
    """Own one optional evaluator, its declared quality policy and serialized calls."""

    def __init__(
        self,
        portfolio_sentinels: Sequence[INPUTS.StageE2ESentinel],
        *,
        provider: Callable[..., Mapping[str, Any]] | None = None,
        policy: Any | None = None,
        quality_budgets: Mapping[str, Any] | None = None,
        provider_binding: Mapping[str, Any] | None = None,
    ) -> None:
        self.portfolio_sentinels = tuple(portfolio_sentinels)
        member_hashes = [member.capsule_sha256 for member in self.portfolio_sentinels]
        self._lock = threading.Lock()
        from merlin.perf.phase2_portfolio import (
            PORTFOLIO_MEMBER_COUNT,
            FastEvaluationPolicy,
            QualityBudget,
        )

        configured_fast_parts = (
            provider,
            policy,
            quality_budgets,
            provider_binding,
        )
        if any(part is not None for part in configured_fast_parts) and not all(
            part is not None for part in configured_fast_parts
        ):
            raise ValueError(
                "fast evaluation requires provider, policy, per-model quality budgets, "
                "and an immutable provider binding"
            )
        self.provider = provider
        self._provider_identity = provider
        self.policy = policy
        self.quality_budgets = dict(quality_budgets or {})
        self.provider_binding = copy.deepcopy(provider_binding)
        if provider is not None:
            if len(member_hashes) != PORTFOLIO_MEMBER_COUNT:
                raise ValueError("fast evaluation requires exactly four content-addressed portfolio members")
            if not isinstance(policy, FastEvaluationPolicy):
                raise TypeError("fast_evaluation_policy must be FastEvaluationPolicy")
            if set(self.quality_budgets) != set(member_hashes) or any(
                not isinstance(value, QualityBudget) for value in self.quality_budgets.values()
            ):
                raise ValueError("quality budgets must exactly cover the content-addressed portfolio")
            quality_profiles = [budget.profile for budget in self.quality_budgets.values()]
            if (
                quality_profiles.count("classification_top1") != 1
                or quality_profiles.count("numerical_similarity") != PORTFOLIO_MEMBER_COUNT - 1
            ):
                raise ValueError("four-model quality policy requires one top-1 and three numerical budgets")
            provider_binding = self.provider_binding
            if (
                not isinstance(provider_binding, Mapping)
                or provider_binding.get("schema") != "host_fast_analytical_evaluator_binding_v1"
                or not BE._is_sha256(provider_binding.get("implementation_sha256"))
                or provider_binding.get("execution") != "host_analytical_only"
                or provider_binding.get("full_model_simulation_allowed") is not False
                or provider_binding.get("resource_admission") != "serialized_one_model_at_a_time"
                or isinstance(provider_binding.get("maximum_model_seconds"), bool)
                or not isinstance(provider_binding.get("maximum_model_seconds"), (int, float))
                or not 0 < provider_binding["maximum_model_seconds"] <= 60
                or not isinstance(provider_binding.get("calibration_sha256s"), Sequence)
                or isinstance(provider_binding.get("calibration_sha256s"), (str, bytes))
                or not provider_binding["calibration_sha256s"]
                or any(not BE._is_sha256(value) for value in provider_binding["calibration_sha256s"])
            ):
                raise ValueError("fast analytical provider requires an exact implementation binding")
            self.binding = {
                "schema": "phase2_fast_evaluation_binding_v1",
                "provider": provider_binding,
                "policy": policy.to_dict(),
                "quality_budgets": {model: self.quality_budgets[model].to_dict() for model in member_hashes},
                "execution": "host_analytical_only_no_complete_model_or_layer_simulation",
            }
        else:
            self.binding = None
        self.binding_sha256 = P2_CONTRACTS.document_sha256(self.binding)

    def check_integrity(self) -> None:
        if P2_CONTRACTS.document_sha256(self.binding) != self.binding_sha256:
            raise ValueError("host fast-evaluation binding changed")
        if self.provider is not self._provider_identity:
            raise ValueError("host fast-evaluation provider changed")
        if self.provider is not None:
            current_fast_binding = {
                "schema": "phase2_fast_evaluation_binding_v1",
                "provider": copy.deepcopy(self.provider_binding),
                "policy": self.policy.to_dict(),
                "quality_budgets": {
                    member.capsule_sha256: self.quality_budgets[member.capsule_sha256].to_dict()
                    for member in self.portfolio_sentinels
                },
                "execution": "host_analytical_only_no_complete_model_or_layer_simulation",
            }
            if current_fast_binding != self.binding:
                raise ValueError("host fast-evaluation policy or quality budget changed")

    def authorized_surfaces(
        self, analysis: Mapping[str, Any], *, edit_contract: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], ...]:
        """Return only analysis surfaces already present in host-frozen edit authority."""
        if edit_contract is None:
            return ()
        authorized = {
            (row.get("surface_id"), row.get("path"), row.get("symbol"))
            for row in edit_contract.get("existing_symbols") or ()
            if isinstance(row, Mapping)
        }
        result: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any]] = set()
        brief = analysis.get("optimization_brief") or {}
        for action in brief.get("ranked_actions") or ():
            for surface in action.get("edit_surfaces") or ():
                key = (surface.get("id"), surface.get("path"), surface.get("symbol"))
                if key not in authorized or key in seen:
                    continue
                seen.add(key)
                result.append(
                    {field: copy.deepcopy(surface.get(field)) for field in ("id", "path", "symbol", "scope", "effects")}
                )
        return tuple(result)

    def evaluate(
        self,
        members: Sequence[tuple[INPUTS.StageE2ESentinel, Mapping[str, Any], Mapping[str, Any]]],
        *,
        edit_contract: Mapping[str, Any] | None,
        target_descriptor: Path | None,
        target_sha256: str,
        portfolio_sha256: str,
    ) -> dict[str, Any]:
        """Run the optional bounded host analytical adapter and apply the shared gate."""
        from merlin.perf.phase2_portfolio import (
            evaluate_fast_portfolio,
            unavailable_fast_evaluation,
        )

        self.check_integrity()
        if self.provider is None:
            return unavailable_fast_evaluation(
                reason=(
                    "held-out quality corpus and host analytical evaluator are not both bound; "
                    "approximate transformations remain disabled"
                )
            )
        rows = []
        surfaces: dict[str, Sequence[Mapping[str, Any]]] = {}
        provider_wall_seconds: dict[str, float] = {}
        for sentinel, analysis, artifacts in members:
            model_id = sentinel.capsule_sha256
            surfaces[model_id] = self.authorized_surfaces(analysis, edit_contract=edit_contract)
            try:
                provider_started = time.monotonic()
                with self._lock:
                    raw = self.provider(
                        analysis=analysis,
                        artifacts=artifacts,
                        sentinel=sentinel,
                        target_descriptor=target_descriptor,
                        target_sha256=target_sha256,
                        portfolio_sha256=portfolio_sha256,
                        provider_binding=copy.deepcopy(self.provider_binding),
                    )
                provider_elapsed = time.monotonic() - provider_started
                provider_wall_seconds[model_id] = provider_elapsed
                if provider_elapsed > self.provider_binding["maximum_model_seconds"]:
                    raise TimeoutError("host analytical provider exceeded its per-model wall budget")
                if not isinstance(raw, Mapping):
                    raise TypeError("host analytical provider must return a mapping")
                rows.append(
                    {**copy.deepcopy(dict(raw)), "model_id": model_id, "provider_elapsed_seconds": provider_elapsed}
                )
            except Exception as exc:  # fail closed without discarding valid compiler analysis
                rows.append(
                    {
                        "model_id": model_id,
                        "baseline": {},
                        "candidate": {},
                        "provider_failure": {"type": type(exc).__name__, "reason": str(exc)[:20000]},
                    }
                )
        report = evaluate_fast_portfolio(
            rows,
            quality_budgets=self.quality_budgets,
            policy=self.policy,
            authorized_surfaces=surfaces,
            expected_models=[member.capsule_sha256 for member in self.portfolio_sentinels],
        )
        return {
            **report,
            "binding": copy.deepcopy(self.binding),
            "binding_sha256": self.binding_sha256,
            "provider_wall_seconds_by_model": provider_wall_seconds,
            "provider_evaluation_order": [member.capsule_sha256 for member in self.portfolio_sentinels],
            "resource_admission": "serialized_one_model_at_a_time",
        }
