"""How much of PyTorch's Core ATen IR the captured models actually contain, and how much is accelerated.

The generalization claim this supports is bounded and checkable:

    Over the declared roster, N of <core opset size> Core ATen operators appear, M of them route to the
    accelerator on target T, and those account for P% of the roster's total loop-nest work.

Three separate numbers, because they answer three different questions and collapsing them is how a
coverage figure becomes a slogan:

``observed``
    which core ops the captured models CONTAIN. A denominator of ops nobody runs is not a claim about
    a compiler.

    ⚠️ ``prov.aten`` RECORDS THE FRONTEND OP, NOT THE LOWERED ONE -- which is what a provenance tag is
    for, and it means this count is not a statement about the IR's level. Measured on
    ``resnet50_v1_5``: its tags are ``aten.conv2d.default`` / ``aten.linear.default`` /
    ``aten.batch_norm.default``, none of them core, while its linalg is fully decomposed (688
    contraction regions, 54 ``linalg.matmul`` reached through im2col, batch-norm lowered to
    ``linalg.generic``). Reading a non-core tag as "this capture is at the wrong IR level" was wrong:
    the capture is fine and the tag is honest. So a non-core tag is classified rather than dismissed --
    a frontend COMPOSITE whose lowering is core, or an op nothing here can name.
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
    non_core = set(seen) - core
    # A non-core tag is a FRONTEND COMPOSITE when torch's decomposition table knows how to take it
    # apart -- `aten.conv2d.default` -> `aten.convolution.default`. Its lowering is core even though
    # the tag is not, so calling it "not core" and stopping there describes the tag rather than the
    # model. What the table does NOT cover (in-place and aliasing variants: `relu_`, `add_`,
    # `flatten.using_ints`) stays in its own bucket, because guessing that `relu_` is `relu` is exactly
    # the name-matching this repo avoids.
    decomposed = set((opset or {}).get("decomposed") or ())
    composite = sorted(non_core & decomposed)
    unclassified = sorted(non_core - decomposed)
    return {
        "n_core": len(core),
        "observed_core": observed_core,
        "n_observed_core": len(observed_core),
        # Kept for callers that only want "not core"; the two buckets below say WHY.
        "non_core_observed": sorted(non_core),
        # A frontend composite: not core, and torch's decomposition table takes it to core ops.
        "composite_observed": composite,
        # Neither core nor decomposable by that table -- an in-place or aliasing variant, or something
        # genuinely unknown. Reported, never folded into either of the other two.
        "unclassified_observed": unclassified,
        "per_model": per_model,
        "unreadable": sorted(n for n, d in per_model.items() if d.get("status") != "ok"),
    }


def _parse_failure(path: Path, exc: Exception) -> dict:
    """A parse failure that names the construct, not just the exception.

    A bare ``ParseError: <path>:17083:5`` is indistinguishable from a corrupt capture, and the two
    license opposite actions: re-capture the model, or teach the parser a form it does not read.
    Measured on ``smolvla_fp32_consistent``: valid MLIR that xDSL 0.68.0 refuses, because its
    ``linalg.generic`` assembly accepts only the single-result ``-> tensor<...>`` form and a fused
    argmin yields two results (``-> (tensor<1xi64>, tensor<1xi64>)``). One construct in a 4.3 MB module
    cost the whole model's routing evidence.

    The location comes from the exception's own span rather than from its message text, and the
    reported line is the one BEFORE the failure point as well as the failure point itself: a parser
    that rejects an op reports the position where it gave up, which is the start of the NEXT statement.
    """
    detail = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    span = getattr(exc, "span", None)
    loc = None
    try:
        loc = span.get_location() if span is not None else None
    except Exception:                              # noqa: BLE001 -- a span without a location is not fatal
        loc = None
    line_no = getattr(loc, "line", None)
    if not isinstance(line_no, int) or line_no < 1:
        return detail
    detail["line"] = line_no
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return detail
    # 1-indexed. The construct the parser could not read usually ENDS on the previous line.
    detail["at"] = lines[line_no - 1].strip()[:200] if line_no <= len(lines) else ""
    if line_no >= 2:
        detail["after"] = lines[line_no - 2].strip()[:200]
    return detail


def coverage(captures: dict[str, str | Path], target: str, *, opset: dict | None = None) -> dict:
    """The full claim for one target: observed, routed, and work-weighted.

    ``routed``/``fallback``/``unclassified`` come from :mod:`model_coverage`, which asks the target's
    OWN capability contract. ``unclassified`` stays its own column: the routing denominator is
    ``routed + fallback``, and a region nobody could classify belongs to neither.
    """
    from merlin.targetgen import model_coverage as MC

    from merlin.kernels import work as WK

    cen = census(captures, opset=opset)
    routed = fallback = unclassified = 0
    dtype_ok = dtype_blocked = 0
    # WORK, the third number the claim needs. A region count answers "how many ops" and says nothing
    # about how much of the model those ops ARE: a hundred elementwise regions and one attention matmul
    # are 101 regions and nowhere near 101 units of work. Weighted by iteration space x body arithmetic,
    # using the SAME proxy `build_tools/scripts/model_op_census.py` uses, so two copies of a cost model
    # cannot produce two rankings of one model.
    routed_work = fallback_work = unclassified_work = 0
    work_complete = True
    per_model: dict[str, Any] = {}
    for name, path in sorted(captures.items()):
        try:
            _module = MC.load_module(Path(path))
            regions = MC.regions_from_module(_module)
            cov = MC.coverage_for(regions, target, model=name)
            _ops = MC.region_ops(_module)
        except Exception as exc:                   # noqa: BLE001 -- an unreadable model is not zero coverage
            per_model[name] = {"status": "unreadable", **_parse_failure(Path(path), exc)}
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
        # Price each region and charge it to the SAME bucket its family put it in. The join is
        # positional against `region_ops`, which is the one filter that produced these descriptors.
        # A capture whose op list does not line up is priced as unknown rather than mis-attributed.
        if len(_ops) == len(regions):
            cap_map = MC.capability_map_for_target(target)
            for _region, _op in zip(regions, _ops):
                extents, exact = WK.iteration_space(_op)
                w = int(extents) * int(WK.body_arith_ops(_op))
                work_complete = work_complete and bool(exact)
                fam = _region.resolved_family()
                if fam is None:
                    unclassified_work += w
                elif fam in cap_map:
                    routed_work += w
                else:
                    fallback_work += w
        else:
            work_complete = False
    denominator = routed + fallback
    work_denominator = routed_work + fallback_work
    # ⚠️ THE TWO HALVES CAN BE COMPUTED OVER DIFFERENT MODEL SETS, and saying so is the point. The census
    # reads `prov.aten` tags out of the module text; routing PARSES the module. Measured: smolvla's
    # capture yields its op tags fine and fails `model_coverage.load_module` with a ParseError, so the
    # operator count came from four models and the routing fraction from three -- and one sentence
    # carrying both numbers read as though it were four. Both sets are recorded, and `claim_sentence`
    # names each half's own denominator rather than picking one and hoping they agree.
    counted = sorted(n for n, d in (cen.get("per_model") or {}).items() if d.get("status") == "ok")
    routed_models = sorted(n for n, d in per_model.items() if d.get("status") != "unreadable")
    unparsed = sorted(n for n, d in per_model.items() if d.get("status") == "unreadable")
    return {
        "target": target,
        "opset": {"n_core": cen["n_core"], "torch": (opset or {}).get("torch")},
        "models": {"counted_for_operators": counted, "measured_for_routing": routed_models,
                   "unparsed_for_routing": unparsed,
                   "agree": counted == routed_models},
        "observed": {"n_core_observed": cen["n_observed_core"],
                     "core_fraction": (cen["n_observed_core"] / cen["n_core"]) if cen["n_core"] else None,
                     "non_core_observed": cen["non_core_observed"],
                     "composite_observed": cen.get("composite_observed") or [],
                     "unclassified_observed": cen.get("unclassified_observed") or [],
                     "why_composites_are_separate": (
                         "prov.aten records the FRONTEND op, so a non-core tag does not mean the "
                         "capture is at the wrong IR level; a composite whose lowering is core is a "
                         "different fact from an op nothing can name")},
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
        "work": {"routed": routed_work, "fallback": fallback_work,
                 "unclassified": unclassified_work, "denominator": work_denominator,
                 "routed_fraction": (routed_work / work_denominator) if work_denominator else None,
                 # A LOWER BOUND is not a measurement. `iteration_space` returns `complete=False` when a
                 # nest was only partially recovered, and a partially recovered nest that reads as exact
                 # is how a heavy op gets ranked light.
                 "exact": work_complete,
                 "unit": "iteration-space extents x body arithmetic ops (merlin.kernels.work)"},
        "per_model": per_model,
        "census": cen,
    }


def _pct(frac: float) -> str:
    """A percentage that never rounds INTO a claim the number does not make.

    Measured: one target's work share is 0.99996, and ``{:.1%}`` renders that "100.0%" -- read by anyone
    as "all of it", when 270 regions were in fact not routed. A fraction below 1 is therefore given as
    many decimals as it needs to stay distinguishable from 100%, and only an exact 1.0 prints "100%".
    The same guard applies at the bottom: a tiny non-zero share must not render as "0.0%".
    """
    if frac >= 1.0:
        return "100%"
    for places in (1, 2, 3, 4):
        text = f"{frac:.{places}%}"
        if float(text.rstrip("%")) < 100.0:
            break
    else:
        return "<100%"
    if frac > 0.0 and float(text.rstrip("%")) == 0.0:
        return ">0%"
    return text


def claim_sentence(report: dict) -> str:
    """The one sentence the report supports, with nothing in it that the numbers do not carry.

    Each half names its OWN model set. They are usually the same set and are not guaranteed to be:
    counting operators only needs the module's op tags, while routing has to parse it, so a capture
    that parses in one and not the other puts the two numbers over different denominators. Stating one
    count for both is how a coverage sentence stops being true without anyone editing it.
    """
    obs = report["observed"]
    rt = report["routing"]
    ms = report.get("models") or {}
    counted = ms.get("counted_for_operators") or []
    measured = ms.get("measured_for_routing") or []
    frac = obs.get("core_fraction")
    rfrac = rt.get("routed_fraction")
    unparsed = ms.get("unparsed_for_routing") or []
    return (
        f"Across {len(counted)} captured model(s) ({', '.join(counted)}), {obs['n_core_observed']} of "
        f"{report['opset']['n_core']} PyTorch Core ATen operators appear"
        + (f" ({_pct(frac)})" if frac is not None else "")
        + f"; on {report['target']}, over the {len(measured)} of those whose capture could be parsed, "
          f"{rt['routed']} of {rt['denominator']} classifiable regions route to the accelerator"
        + (f" ({_pct(rfrac)})" if rfrac is not None else "")
        + f", with {rt['unclassified']} region(s) unclassified and reported separately"
        + (f"; {', '.join(unparsed)} did not parse and contributed no routing evidence" if unparsed
           else "")
        + (_work_clause(report.get("work") or {}))
        + ".")


def _work_clause(wk: dict) -> str:
    """The work half of the claim, or nothing at all.

    Silence when no work could be priced: a claim that omits the weighting is weaker than one that
    carries it, and both are honest. A claim that STATES a weighting nobody computed is neither.
    """
    frac = wk.get("routed_fraction")
    if frac is None:
        return ""
    qualifier = "" if wk.get("exact") else " (a lower bound; at least one iteration nest was only "\
                                          "partially recovered)"
    return (f"; those account for {_pct(frac)} of the roster's total loop-nest work{qualifier}")
