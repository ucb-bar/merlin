"""How far the Spike timing model is from measured gSIM cycles, per family, from the run's own pairs.

The author compares variants by the Spike timing model's ``estimated_whole_soc_cycles``. Measured on
one attempt, the model read about 2x the gSIM cycles of the same executable at the median, with an
interquartile range of measured/modeled 0.33-0.82, and a variant that wins on the model by less than
that spread may be no faster on the hardware. This module pairs the two only where they describe the
SAME program: the same test case and the same ``execution_digest`` (the identity promotion already uses
to bind a certificate to an executable), from the agent's own self-check and gSIM responses. It fits a
robust ratio (median and quartiles of measured / modeled) at three levels -- the semantic family of the
test case's operation when its public declaration is in the workspace, the roofline limiter class
(compute or memory requests), and all pairs -- and uses the most specific level with enough pairs.

A calibrated value is an estimate with an interval, labelled as such; it is never a measurement, it
carries no authority, and a family with too few pairs says so instead of guessing.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from merlin.common import stat_memo

from .performance_headroom import _bound

SCHEMA = "merlin.timing_model_calibration.v1"
#: Fewest same-executable pairs a level needs before its ratio is offered.
MIN_PAIRS = 5
#: Newest response files read per channel; older pairs describe compilers long gone.
MAX_FILES = 600
_MAX_FILE_BYTES = 16 << 20
USE = ("Legacy measured/model ratios are descriptive program-pair diagnostics, not a tuned timing "
       "runtime or a qualification claim. Rank variants using complete same-provider and same-ROI "
       "cycles. Request small/medium gSIM witnesses only through the existing physical policy "
       "for an unresolved mechanism; large performance workloads remain Spike-only in development.")


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _rows(path: Path) -> list[tuple]:
    """``(name, digest, modeled, measured, limiter)`` rows of one response file (cached by stat)."""
    def parse():
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
                return []
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        gsim = isinstance(document, dict) and document.get("sim") == "gsim"
        found = []
        for row in (document.get("per_capsule") if isinstance(document, dict) else None) or ():
            if not isinstance(row, dict):
                continue
            name, digest = row.get("capsule"), row.get("execution_digest")
            if not isinstance(name, str) or not isinstance(digest, str) or not digest:
                continue
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            modeled = performance.get("estimated_whole_soc_cycles")
            if not (_positive(modeled) and performance.get("complete_event_accounting") is True):
                modeled = None
            measured = row.get("barrier_cycles")
            if not (gsim and row.get("barrier_tier") == "L3" and row.get("pass") is not False and _positive(measured)):
                measured = None
            if modeled is None and measured is None:
                continue
            _, limiter = _bound(performance.get("structural_roofline") or {})
            found.append((name, digest, modeled, measured, limiter))
        return found

    return stat_memo.cached("model-calibration-rows", str(path), lambda: stat_memo.file_signature(path), parse) or []


def semantic_family(workspace: Path, name: str) -> str | None:
    """The semantic family of a test case's declared operation, from its public declaration if present."""
    declaration = workspace / "isa" / name / "capsule.yaml"
    if not declaration.is_file():
        return None
    try:
        import yaml

        from merlin.targetgen import semantic_families

        document = yaml.safe_load(declaration.read_text()) or {}
        operation = document.get("operation") if isinstance(document.get("operation"), dict) else {}
        return semantic_families.from_op(operation.get("op"))
    except (OSError, ValueError, ImportError, AttributeError):
        return None


def pairs(workspace) -> list[dict]:
    """Same-executable (modeled, measured) pairs from the workspace's self-check and gSIM responses."""
    workspace = Path(workspace)
    channel = workspace / ".qa_channel"
    if not channel.is_dir():
        return []
    files = [p for p in channel.iterdir() if p.suffix == ".json" and p.name.startswith(("resp_", "simresp_"))]
    try:
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    modeled, measured, limiters = {}, {}, {}
    for path in files[:MAX_FILES]:
        for name, digest, model, measure, limiter in _rows(path):
            key = (name, digest)
            if model is not None:
                modeled.setdefault(key, model)  # newest first: keep the newest estimate
            if measure is not None:
                measured.setdefault(key, measure)
            if limiter:
                limiters.setdefault(key, limiter)
    families: dict[str, str | None] = {}
    out = []
    for key in sorted(set(modeled) & set(measured)):
        name = key[0]
        if name not in families:
            families[name] = semantic_family(workspace, name)
        limiter = limiters.get(key)
        out.append({"test_case": name, "execution_digest": key[1], "modeled": modeled[key], "measured": measured[key],
                    "ratio": measured[key] / modeled[key], "family": families[name],
                    "bound_class": None if limiter is None else ("compute" if limiter == "compute" else "memory requests")})
    return out


def _fit(ratios: list[float]) -> dict:
    ratios = sorted(ratios)
    if len(ratios) >= 4:
        q1, median, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
    else:
        q1, median, q3 = ratios[0], statistics.median(ratios), ratios[-1]
    return {"n": len(ratios), "median_measured_over_modeled": round(median, 3),
            "interquartile": [round(q1, 3), round(q3, 3)], "spread": round(q3 - q1, 3)}


def fit(found: list[dict], *, min_pairs: int = MIN_PAIRS) -> dict:
    """Robust ratios per semantic family, per limiter class and overall; small levels are uncalibrated."""
    levels: dict[str, dict] = {"family": {}, "bound_class": {}, "all": {}}
    groups: dict[tuple, list[float]] = {}
    for pair in found:
        groups.setdefault(("all", "all"), []).append(pair["ratio"])
        if pair.get("family"):
            groups.setdefault(("family", pair["family"]), []).append(pair["ratio"])
        if pair.get("bound_class"):
            groups.setdefault(("bound_class", pair["bound_class"]), []).append(pair["ratio"])
    for (level, name), ratios in groups.items():
        levels[level][name] = _fit(ratios) if len(ratios) >= min_pairs else {"n": len(ratios), "status": "uncalibrated"}
    from .model_accuracy import legacy_card
    return {"schema": SCHEMA, "pairs": len(found), "min_pairs": min_pairs, "levels": levels,
            "accuracy_card": legacy_card(found), "qualification_issued": False,
            "basis": "same test case and same execution_digest; measured = gSIM L3 barrier cycles",
            "use": USE}


def calibration(workspace) -> dict | None:
    """The workspace's calibration, or None when it has no same-executable pairs at all."""
    found = pairs(workspace)
    return fit(found) if found else None


def apply(modeled_cycles, calibrated: dict | None, *, family: str | None = None,
          bound_class: str | None = None) -> dict | None:
    """A labelled calibrated estimate for one modeled value, from the most specific calibrated level."""
    if calibrated is None or not _positive(modeled_cycles):
        return None
    levels = calibrated.get("levels") or {}
    for level, name in (("family", family), ("bound_class", bound_class), ("all", "all")):
        fitted = (levels.get(level) or {}).get(name) if name else None
        if fitted and "median_measured_over_modeled" in fitted:
            low, high = fitted["interquartile"]
            return {
                "calibrated_estimate": round(modeled_cycles * fitted["median_measured_over_modeled"]),
                "calibrated_interval": [round(modeled_cycles * low), round(modeled_cycles * high)],
                "model_confidence": {"level": level, "group": name, "pairs": fitted["n"],
                                     "median_measured_over_modeled": fitted["median_measured_over_modeled"],
                                     "spread": fitted["spread"]},
                "calibration_label": "estimate: Spike model scaled by this run's measured/modeled pairs; not a measurement",
            }
    return {"model_confidence": {"status": "uncalibrated", "pairs": calibrated.get("pairs", 0)},
            "calibration_label": "no calibrated level has enough same-executable gSIM pairs; confirm on gSIM"}
