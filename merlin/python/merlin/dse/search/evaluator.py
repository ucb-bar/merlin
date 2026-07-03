"""Shared scoring for all three search methods.

Scoring rubric (early-work priority order: correctness first, speed last):

    score =  correctness + compile_success + verifier_success + workload_coverage
           + compiler_exploitability + speedup_or_cost_improvement - complexity_penalty
    priority: correctness > compile_success > coverage > exploitability > speedup

Prioritising correctness/compile/coverage stops search from finding fast-but-invalid junk: a
strategy that exposes a contract the region does not legally support scores correctness 0. The
legality verifiers are the mined policies (via ``design_pressure.synthesize``); cost/speedup
come from ``dse.strategy.evaluate_strategy`` over the supplied workload regions.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.design_pressure import synthesize as S
from merlin.dse.exploitability import exploitability as _exploitability
from merlin.dse.hardware_space import default_cost_model
from merlin.dse.strategy import default_strategies, effect_passes, evaluate_strategy


@dataclass(frozen=True)
class Score:
    correctness: float
    compile_success: float
    verifier_success: float
    coverage: float
    exploitability: float
    speedup: float
    complexity_penalty: float

    @property
    def total(self) -> float:
        return (self.correctness + self.compile_success + self.verifier_success
                + self.coverage + self.exploitability + self.speedup
                - self.complexity_penalty)

    def priority_key(self) -> tuple:
        """Lexicographic key (higher is better): correctness first, speedup last."""
        return (self.correctness, self.compile_success, self.verifier_success,
                self.coverage, self.exploitability, self.speedup, -self.complexity_penalty)


class Evaluator:
    """Scores candidates over a fixed set of workload regions and a cost model."""

    def __init__(self, regions: list[tuple[str, dict]], cost_model: dict | None = None,
                 policies=None):
        if not regions:
            raise ValueError("Evaluator needs at least one (name, rpv) region")
        self.regions = regions
        self.cost_model = cost_model or default_cost_model()
        self.policies = policies if policies is not None else S.load_policies()
        defaults = {s.variant_class: s for s in default_strategies()}
        self._baseline = defaults["baseline"]
        self._oracle = defaults["oracle"]
        # Per-region baseline / oracle cycles.
        self._base = {n: evaluate_strategy(self._baseline, r, self.cost_model)["cycles"]
                      for n, r in regions}
        self._orc = {n: evaluate_strategy(self._oracle, r, self.cost_model)["cycles"]
                     for n, r in regions}

    def _legal_features(self, rpv: dict) -> set[str]:
        return set(S.recommended_features(
            rpv, self.policies,
            resident_store_bytes=self.cost_model.get("resident_store_bytes")))

    def evaluate(self, candidate) -> Score:
        strategy = candidate.strategy() if hasattr(candidate, "strategy") else candidate
        feats = set(strategy.interface_features)

        legal_count = 0
        sup, expl = [], []
        for name, rpv in self.regions:
            legal_feats = self._legal_features(rpv)
            region_ok = feats <= legal_feats  # every claimed feature must be legal here
            # Only apply (and score) the strategy where it is legal — never apply it illegally.
            if region_ok:
                legal_count += 1
                e = evaluate_strategy(strategy, rpv, self.cost_model)
                base, orc = self._base[name], self._orc[name]
                speedup = base / e["cycles"] if e["cycles"] else 1.0
                oracle_speedup = base / orc if orc else 1.0
                sup.append(speedup)
                expl.append(_exploitability(speedup, oracle_speedup))

        n = len(self.regions)
        coverage = legal_count / n
        # compile_success: the pipeline resolves with no unknown passes.
        compile_ok = 1.0 if not strategy.pipeline().unknown else 0.0
        # Correct unless the strategy claims features that are legal in NO region (pure junk):
        # applying it only where legal is correct by construction; coverage captures specialisation.
        is_junk = bool(feats) and legal_count == 0
        correctness = 0.0 if is_junk else 1.0
        verifier = correctness
        mean_speedup = sum(sup) / len(sup) if sup else 0.0
        mean_expl = sum(expl) / len(expl) if expl else 0.0
        # Normalise speedup into [0,1)-ish so it stays below the integer-weighted gates.
        speedup_term = 1.0 - 1.0 / mean_speedup if mean_speedup > 0 else 0.0
        complexity = 0.1 * len(feats) + 0.02 * len(effect_passes(strategy))

        return Score(
            correctness=correctness, compile_success=compile_ok, verifier_success=verifier,
            coverage=round(coverage, 4), exploitability=round(mean_expl, 4),
            speedup=round(speedup_term, 4), complexity_penalty=round(complexity, 4))


def make_evaluator(regions, cost_model=None, policies=None) -> Evaluator:
    return Evaluator(regions, cost_model=cost_model, policies=policies)
