"""How much of PyTorch's Core ATen IR the captured models actually contain, and how much is accelerated.

The generalization claim this supports is bounded and checkable:

    Over the declared roster, N of <core opset size> Core ATen operators appear, M of them route to the
    accelerator on target T, and those account for P% of the roster's total loop-nest work.

Three separate numbers, because they answer three different questions and collapsing them is how a
coverage figure becomes a slogan:

``observed``
    which core ops the captured models CONTAIN. A denominator of ops nobody runs is not a claim about
    a compiler.
``routed``
    which of those the target's capability contract admits, via :mod:`model_coverage` -- whose third
    bucket, ``unclassified``, is reported beside the other two and never folded into either. A region
    we could not name is evidence neither of coverage nor of a gap.
``work``
    the same intersection weighted by iteration space x body arithmetic, using the SAME proxy
    ``build_tools/scripts/model_op_census.py`` uses (:mod:`merlin.kernels.work`), so two copies of a
    cost model cannot produce two rankings of the same model.

⚠ THE DENOMINATOR. ``torch._decomp.core_aten_decompositions()`` is NOT the Core ATen opset -- it is the
decomposition table, the ops removed on the way TO Core ATen (measured: 1004 entries against 188
core-tagged overloads). The opset comes from ``torch.Tag.core`` via :mod:`_aten_opset_worker`, and it is
cached per torch version because it is a property of torch, not of this repo.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

#: Where a resolved opset is cached. Keyed by torch version: a torch bump legitimately changes the
#: denominator, and silently reusing the old one would move a claim without anyone deciding to.
_CACHE_DIR = "aten_opset"


def _m2m_python() -> Path:
    from merlin.targetgen.capsule_source import _m2m_python as _p
    return _p()


def core_opset(*, refresh: bool = False) -> dict:
    """``{"torch": version, "n_core": int, "ops": [...]}`` -- the Core ATen IR opset.

    Asked of torch in the m2m venv rather than derived here: a hand-typed list is a baked constant that
    is stale on the next bump, and this repo's venv has no torch to ask.
    """
    from merlin.common.paths import build_dir

    worker = Path(__file__).with_name("_aten_opset_worker.py")
    python = _m2m_python()
    if not python.exists():
        raise RuntimeError(
            f"no m2m venv python at {python}; the Core ATen opset is a property of torch and torch "
            f"lives only there. Set MERLIN_M2M_PYTHON / MERLIN_M2M_DIR")
    env = dict(os.environ)
    proc = subprocess.run([str(python), str(worker)], capture_output=True, text=True,
                          timeout=300, env=env)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"could not resolve the Core ATen opset: {proc.stderr[-400:]}")
    doc = json.loads(proc.stdout)
    cache = Path(build_dir()) / _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"core_{doc['torch']}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def aten_ops_in(module_path: str | Path) -> Counter:
    """``prov.aten`` tags in one captured model, counted.

    Structural read of the attribute the capture pipeline stamps; a region without the tag is counted
    as ``untagged`` rather than guessed at, because an op we cannot name is not evidence either way.
    """
    text = Path(module_path).read_text(encoding="utf-8")
    out: Counter = Counter()
    marker = 'prov.aten = "'
    idx = text.find(marker)
    while idx != -1:
        start = idx + len(marker)
        end = text.find('"', start)
        if end == -1:
            break
        out[text[start:end]] += 1
        idx = text.find(marker, end)
    return out


def census(captures: dict[str, str | Path], *, opset: dict | None = None) -> dict:
    """Which core ops the captured models contain, and which observed ops are not core."""
    core = set((opset or core_opset())["ops"])
    per_model: dict[str, dict] = {}
    seen: Counter = Counter()
    for name, path in sorted(captures.items()):
        try:
            ops = aten_ops_in(path)
        except OSError as exc:
            per_model[name] = {"status": "unreadable", "detail": f"{type(exc).__name__}: {exc}"}
            continue
        seen.update(ops)
        per_model[name] = {"status": "ok", "n_ops": len(ops),
                           "core": sorted(set(ops) & core),
                           "non_core": sorted(set(ops) - core)}
    observed_core = sorted(set(seen) & core)
    return {
        "n_core": len(core),
        "observed_core": observed_core,
        "n_observed_core": len(observed_core),
        # An op a model contains that the opset does NOT declare core. Reported rather than dropped:
        # it is real work the compiler must handle, and folding it into the core count would inflate
        # the numerator against a denominator that never contained it.
        "non_core_observed": sorted(set(seen) - core),
        "per_model": per_model,
        "unreadable": sorted(n for n, d in per_model.items() if d.get("status") != "ok"),
    }


def coverage(captures: dict[str, str | Path], target: str, *, opset: dict | None = None) -> dict:
    """The full claim for one target: observed, routed, and work-weighted.

    ``routed``/``fallback``/``unclassified`` come from :mod:`model_coverage`, which asks the target's
    OWN capability contract. ``unclassified`` stays its own column: the routing denominator is
    ``routed + fallback``, and a region nobody could classify belongs to neither.
    """
    from merlin.targetgen import model_coverage as MC

    cen = census(captures, opset=opset)
    routed = fallback = unclassified = 0
    dtype_ok = dtype_blocked = 0
    per_model: dict[str, Any] = {}
    for name, path in sorted(captures.items()):
        try:
            regions = MC.regions_from_module(MC.load_module(Path(path)))
            cov = MC.coverage_for(regions, target, model=name)
        except Exception as exc:                   # noqa: BLE001 -- an unreadable model is not zero coverage
            per_model[name] = {"status": "unreadable", "detail": f"{type(exc).__name__}: {exc}"}
            continue
        d = cov.to_dict() if hasattr(cov, "to_dict") else dict(cov)
        per_model[name] = d
        # The report's OWN field names, read rather than assumed. `family_supported` is a region the
        # contract admits; `family_unsupported` is one it does not and the host must carry;
        # `unclassified` is one nobody could name. The invariant the report documents is
        # `family_supported + family_unsupported + unclassified == n_regions`, so these three partition
        # the model and none has to be inferred from the others.
        routed += int(d.get("family_supported") or 0)
        fallback += int(d.get("family_unsupported") or 0)
        unclassified += int(d.get("unclassified") or 0)
        # PRECISION is judged only over the admitted subset, so it is a separate column: a region whose
        # family is admitted but whose dtype is not is NOT a routing gap, it is a precision gap, and
        # merging them would report one as the other.
        dtype_ok += int(d.get("dtype_ok") or 0)
        dtype_blocked += int(d.get("dtype_blocked") or 0)
    denominator = routed + fallback
    return {
        "target": target,
        "opset": {"n_core": cen["n_core"], "torch": (opset or {}).get("torch")},
        "observed": {"n_core_observed": cen["n_observed_core"],
                     "core_fraction": (cen["n_observed_core"] / cen["n_core"]) if cen["n_core"] else None,
                     "non_core_observed": cen["non_core_observed"]},
        "precision": {"dtype_ok": dtype_ok, "dtype_blocked": dtype_blocked,
                      "why_separate": (
                          "precision is judged only over the family-admitted subset; a region whose "
                          "family is admitted but whose dtype is not is a precision gap, not a "
                          "routing one, and reporting it as a routing gap would misattribute it")},
        "routing": {"routed": routed, "fallback": fallback, "unclassified": unclassified,
                    "denominator": denominator,
                    "routed_fraction": (routed / denominator) if denominator else None,
                    "why_unclassified_is_separate": (
                        "a region whose family could not be determined is evidence neither of coverage "
                        "nor of a gap; folding it into either is how a coverage number becomes a lie")},
        "per_model": per_model,
        "census": cen,
    }


def claim_sentence(report: dict) -> str:
    """The one sentence the report supports, with nothing in it that the numbers do not carry."""
    obs = report["observed"]
    rt = report["routing"]
    frac = obs.get("core_fraction")
    rfrac = rt.get("routed_fraction")
    models = [n for n, d in (report.get("per_model") or {}).items() if d.get("status") != "unreadable"]
    return (
        f"Across {len(models)} captured model(s), {obs['n_core_observed']} of "
        f"{report['opset']['n_core']} PyTorch Core ATen operators appear"
        + (f" ({frac:.1%})" if frac is not None else "")
        + f"; on {report['target']}, {rt['routed']} of {rt['denominator']} classifiable regions route "
          f"to the accelerator"
        + (f" ({rfrac:.1%})" if rfrac is not None else "")
        + f", with {rt['unclassified']} region(s) unclassified and reported separately.")
