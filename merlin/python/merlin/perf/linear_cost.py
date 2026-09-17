"""A per-command LINEAR cycle model, calibrated per target: ``cycles ~= const + sum_e coeff[e] * n_e``.

The regressor (fit, load/save, predict, band) is target-agnostic. Everything about one machine -- the
command kinds it prices (``events``), the fitted coefficients, the calibration error band and any
derived-count folds -- is DATA, read from the target's own reference directory through
:func:`cost_model_artifact`. Adding a target is a calibration artifact, not a code edit: nothing here
names a target, and no caller has to import a per-target class to reach its model.

WHAT IT MAY SAY. Every artifact this reads declares its own fidelity as serial with no overlap. It
prices a HISTOGRAM of command kinds, so it is blind to command order and to how much work one command
carries (the measured refutation of using its sum as a ceiling is recorded in
:mod:`merlin.perf.compose_estimate`). It may screen a candidate; it never certifies a cycle count, and
:meth:`LinearCostModel.predict_with_band` carries the calibration error so a caller cannot quote the
central value without it.

FOLDS. A backend can issue a command kind that is the same transfer as a priced one but moves a
different number of bytes per element -- an accumulator-width load next to an input-width load. Such a
kind is declared in the artifact as a fold onto the priced event, with a scale that is the RATIO OF TWO
DATAPATH ELEMENT WIDTHS read from the target's RTL facts. The ratio is never written down here; when it
cannot be derived, a prediction that needs it raises :class:`CostModelUnavailable` naming why.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CostModelUnavailable", "LinearCostModel", "cost_model_artifact", "datapath_bits", "fit_linear"]

#: A target's cost model lives in ``<targets>/<target>/cost_model/``: the fitted ``coefficients.json``
#: (const, coeff, error, meta -- what calibration writes) and the declared ``vocabulary.json`` (the
#: priced events in summation order, and folds -- what the target's author writes).
MODEL_DIR = "cost_model"
COEFFICIENTS = "coefficients.json"
VOCABULARY = "vocabulary.json"


class CostModelUnavailable(RuntimeError):
    """A cost the model cannot state honestly. The message names what is missing."""


def cost_model_artifact(target: str) -> Path | None:
    """The target's calibrated coefficients, resolved from the target NAME, or None when it has none."""
    from merlin.common.paths import targets_dir  # noqa: PLC0415

    if not target:
        return None
    candidate = targets_dir() / target / MODEL_DIR / COEFFICIENTS
    return candidate if candidate.is_file() else None


def datapath_bits(target: str, datapath: str) -> int:
    """Element width in bits of one named datapath, from the target's RTL facts.

    Raises :class:`CostModelUnavailable` when the facts do not name the datapath or its dtype carries no
    parseable width -- a fold scaled by a guessed width would misprice every folded command.
    """
    from merlin.perf.lane_cost import dtype_bits  # noqa: PLC0415
    from merlin.targetgen.rtl import facts as rtl_facts  # noqa: PLC0415

    try:
        body = rtl_facts.load_facts(target)
    except Exception as exc:  # noqa: BLE001 - an unreadable fact bundle is a refusal, and says so
        raise CostModelUnavailable(f"RTL facts for {target!r} did not load ({type(exc).__name__}: {exc})") from exc
    facts = body.get("facts") or body
    for row in facts.get("datapaths") or []:
        if isinstance(row, Mapping) and row.get("name") == datapath:
            bits = dtype_bits(row.get("dtype"))
            if not bits:
                raise CostModelUnavailable(
                    f"datapath {datapath!r} of {target!r} has dtype {row.get('dtype')!r}, whose width is not derivable"
                )
            return bits
    raise CostModelUnavailable(f"RTL facts for {target!r} declare no {datapath!r} datapath")


@dataclass
class LinearCostModel:
    const: float = 0.0
    coeff: dict[str, float] = field(default_factory=dict)
    error: dict[str, float] = field(default_factory=dict)  # mape, max_abs_pct, n_points, ...
    meta: dict = field(default_factory=dict)
    #: The priced command vocabulary, in summation order. Empty means "the coefficient keys, sorted".
    events: tuple[str, ...] = ()
    #: ``{event: {"into": priced_event, "scale": {"datapath_bits_ratio": [num, den]}}}``.
    folds: dict[str, dict] = field(default_factory=dict)
    #: The target whose RTL facts resolve fold scales. Set by :meth:`for_target`.
    target: str = ""
    _scales: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "LinearCostModel":
        """A bare coefficient file: its keys are the vocabulary (sorted) and it declares no folds."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(const=d["const"], coeff=d["coeff"], error=d.get("error", {}), meta=d.get("meta", {}))

    @classmethod
    def for_target(cls, target: str) -> "LinearCostModel":
        """``target``'s calibrated coefficients plus its declared vocabulary and folds.

        Raises :class:`CostModelUnavailable` naming what is missing. A target that ships coefficients
        without a vocabulary is priced over its coefficient keys, exactly as :meth:`load` would.
        """
        artifact = cost_model_artifact(target)
        if artifact is None:
            raise CostModelUnavailable(
                f"no calibrated cost model for target {target!r} "
                f"(expected <targets>/{target}/{MODEL_DIR}/{COEFFICIENTS})"
            )
        model = cls.load(artifact)
        model.target = target
        vocab_path = artifact.with_name(VOCABULARY)
        if vocab_path.is_file():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            model.events = tuple(vocab.get("events") or ())
            model.folds = dict(vocab.get("folds") or {})
            unfitted = [e for e in model.events if e not in model.coeff]
            if unfitted:
                raise CostModelUnavailable(
                    f"{target!r} declares priced event(s) {unfitted} that its calibration never fitted"
                )
        return model

    def save(self, path: str | Path) -> None:
        """Write the fitted part only; the vocabulary is declared by the target, not by a fit."""
        doc: dict[str, Any] = {"const": self.const, "coeff": self.coeff, "error": self.error, "meta": self.meta}
        Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")

    def priced_events(self) -> tuple[str, ...]:
        """The command vocabulary this model prices.

        The artifact declares it. When it does not, the coefficient keys ARE the vocabulary -- without
        this the model would price nothing and return the bare intercept, which reads as a real (and
        badly wrong) cycle count rather than as a refusal.
        """
        return self.events or tuple(sorted(self.coeff))

    def fold_scale(self, event: str) -> float:
        """How many priced-event units one ``event`` command is worth, derived from RTL facts."""
        if event in self._scales:
            return self._scales[event]
        spec = (self.folds.get(event) or {}).get("scale") or {}
        ratio = spec.get("datapath_bits_ratio") if isinstance(spec, Mapping) else None
        if not isinstance(ratio, Sequence) or isinstance(ratio, str) or len(ratio) != 2:
            raise CostModelUnavailable(
                f"fold {event!r} declares no derivable scale (expected datapath_bits_ratio: [num, den])"
            )
        if not self.target:
            raise CostModelUnavailable(
                f"fold {event!r} is scaled by RTL datapath widths, but this model was loaded without a "
                "target to read them from (use LinearCostModel.for_target)"
            )
        num, den = (str(part) for part in ratio)
        scale = datapath_bits(self.target, num) / datapath_bits(self.target, den)
        self._scales[event] = scale
        return scale

    def predict(self, events: Mapping[str, float]) -> float:
        """Region cycles from a per-command count dict (over the priced events, plus declared folds)."""
        cyc = self.const
        for e in self.priced_events():
            cyc += self.coeff.get(e, 0.0) * events.get(e, 0.0)
        for event, fold in self.folds.items():
            if event in events:
                cyc += self.coeff.get(fold["into"], 0.0) * self.fold_scale(event) * events[event]
        return cyc

    def predict_with_band(self, events: Mapping[str, float]) -> tuple[float, float]:
        """``(cycles, +/- band)`` with the calibration MAPE as the relative band."""
        c = self.predict(events)
        return c, c * self.error.get("mape", 0.0)


def fit_linear(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[str],
    *,
    meta: Mapping[str, Any] | None = None,
    folds: Mapping[str, dict] | None = None,
    target: str = "",
) -> LinearCostModel:
    """Relative-error-weighted least squares over calibration rows ``{"events": {...}, "cycles": n}``.

    Weighting by 1/cycles minimizes MAPE (what callers act on), so tiny runs are not swamped by large
    ones -- unweighted absolute-residual lstsq drives a near-free command's coefficient to ~0 and
    inflates the intercept.
    """
    import numpy as np  # noqa: PLC0415

    A = np.array([[1.0] + [r["events"][e] for e in events] for r in rows])
    b = np.array([r["cycles"] for r in rows], dtype=float)
    w = 1.0 / np.maximum(b, 1.0)
    coef, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    ape = np.abs(A @ coef - b) / np.maximum(b, 1)
    return LinearCostModel(
        const=float(coef[0]),
        coeff={e: float(coef[i + 1]) for i, e in enumerate(events)},
        error={"mape": float(ape.mean()), "max_abs_pct": float(ape.max()), "n_points": len(b)},
        meta=dict(meta or {}),
        events=tuple(events),
        folds=dict(folds or {}),
        target=target,
    )
