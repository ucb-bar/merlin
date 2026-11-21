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
    ``the earlier model-op census`` uses (:mod:`merlin.analysis.programs.work`), so two copies of a
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---- aten_coverage -----------------------------------------------------------------------------

#: Where a resolved opset is cached. Keyed by torch version: a torch bump legitimately changes the
#: denominator, and silently reusing the old one would move a claim without anyone deciding to.
_CACHE_DIR = "aten_opset"
_OPSET_SCHEMA = "merlin.pytorch_opset.v1"


def _m2m_python() -> Path:
    from merlin.capture.pytorch import _m2m_python as _p

    return _p()


def observe_opset(*, python: str | Path | None = None, timeout: float = 30) -> dict:
    """Read the selected capture environment's catalog without writing a cache.

    This is an explicit live observation, not a frozen-replay dependency. Missing
    frameworks/interpreters return ``not_available`` with unknown denominators;
    they never fall back to the compiler interpreter or pretend an empty registry.
    """
    unavailable = {
        "schema": _OPSET_SCHEMA,
        "status": "not_available",
        "torch": None,
        "n_all_aten": None,
        "all_ops": [],
        "n_core": None,
        "ops": [],
        "n_decomposed": None,
        "decomposed": [],
        "components": {
            key: {"status": "not_available", "count": None} for key in ("registered_aten", "core", "decompositions")
        },
        "support_proven": False,
    }
    try:
        executable = Path(python) if python is not None else _m2m_python()
        worker = Path(__file__).with_name("_aten_opset_worker.py")
        unavailable["observation"] = {"python": str(executable), "worker": str(worker)}
        if not executable.is_file() or not os.access(executable, os.X_OK):
            unavailable["diagnostics"] = [f"capture Python is missing or not executable: {executable}"]
            return unavailable
        if not worker.is_file():
            unavailable["diagnostics"] = [f"ATen observation worker is missing: {worker}"]
            return unavailable
        proc = subprocess.run(
            [str(executable), str(worker)], capture_output=True, text=True, timeout=timeout, env=dict(os.environ)
        )
        if proc.returncode != 0:
            unavailable["diagnostics"] = [f"ATen observer exited {proc.returncode}: {proc.stderr[-1000:]}"]
            return unavailable
        doc = json.loads(proc.stdout)
        if not isinstance(doc, dict) or doc.get("schema") != _OPSET_SCHEMA:
            raise ValueError("ATen observer returned an unrecognized catalog schema")
        doc["observation"] = unavailable["observation"]
        return doc
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        unavailable["diagnostics"] = [f"{type(exc).__name__}: {str(exc)[:1000]}"]
        return unavailable


def core_opset(*, refresh: bool = False) -> dict:
    """``{"torch": version, "n_core": int, "ops": [...]}`` -- the Core ATen IR opset.

    Asked of torch in the m2m venv rather than derived here: a hand-typed list is a baked constant that
    is stale on the next bump, and this repo's venv has no torch to ask.
    """
    from merlin.common.paths import build_dir

    doc = observe_opset(timeout=300)
    if doc.get("status") != "available" or doc.get("components", {}).get("core", {}).get("status") != "available":
        raise RuntimeError(f"could not resolve the Core ATen opset: {doc.get('diagnostics') or doc.get('components')}")
    cache = Path(build_dir()) / _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"core_{doc['torch']}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def aten_ops_in(module_path: str | Path) -> Counter:
    """``prov.aten`` tags in one captured model, counted.

    Read the attribute the capture pipeline stamps. A region without the tag is not counted or
    guessed at, so this is a lower bound on observed frontend calls, never a complete model inventory.
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
    resolved_opset = opset if opset is not None else core_opset()
    core = set(resolved_opset["ops"])
    decomposed = set(resolved_opset.get("decomposed") or ())
    per_model: dict[str, dict] = {}
    seen: Counter = Counter()
    for name, path in sorted(captures.items()):
        try:
            ops = aten_ops_in(path)
        except OSError as exc:
            per_model[name] = {"status": "unreadable", "detail": f"{type(exc).__name__}: {exc}"}
            continue
        seen.update(ops)
        non_core = set(ops) - core
        per_model[name] = {
            "status": "ok",
            # These are provenance-tag observations, not a complete account of all IR operations:
            # a capture may contain untagged regions. Counts are needed to distinguish one rare
            # operator from one repeated thousands of times, while a zero-tag capture must not
            # read as a model requiring no operators.
            "n_ops": len(ops),
            "n_tagged_calls": sum(ops.values()),
            "op_counts": dict(sorted(ops.items())),
            "core": sorted(set(ops) & core),
            "non_core": sorted(non_core),
            "composite": sorted(non_core & decomposed),
            "unclassified": sorted(non_core - decomposed),
            "inventory_complete": False,
            "inventory_scope": "prov.aten-tagged calls only; untagged IR and lowering support are not proven",
        }
    observed_core = sorted(set(seen) & core)
    non_core = set(seen) - core
    # A non-core tag is a FRONTEND COMPOSITE when torch's decomposition table knows how to take it
    # apart -- `aten.conv2d.default` -> `aten.convolution.default`. Its lowering is core even though
    # the tag is not, so calling it "not core" and stopping there describes the tag rather than the
    # model. What the table does NOT cover (in-place and aliasing variants: `relu_`, `add_`,
    # `flatten.using_ints`) stays in its own bucket, because guessing that `relu_` is `relu` is exactly
    # the name-matching this repo avoids.
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
        "untagged_or_empty": sorted(n for n, d in per_model.items() if d.get("status") == "ok" and not d["op_counts"]),
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
    except Exception:  # noqa: BLE001 -- a span without a location is not fatal
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
    from merlin.analysis.programs import work as WK
    from merlin.development.targetgen.conformance import model_coverage as MC

    cen = census(captures, opset=opset)
    routed = fallback = unclassified = 0
    dtype_ok = dtype_blocked = 0
    # WORK, the third number the claim needs. A region count answers "how many ops" and says nothing
    # about how much of the model those ops ARE: a hundred elementwise regions and one attention matmul
    # are 101 regions and nowhere near 101 units of work. Weighted by iteration space x body arithmetic,
    # using the SAME proxy `the earlier model-op census` uses, so two copies of a cost model
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
        except Exception as exc:  # noqa: BLE001 -- an unreadable model is not zero coverage
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
        "models": {
            "counted_for_operators": counted,
            "measured_for_routing": routed_models,
            "unparsed_for_routing": unparsed,
            "agree": counted == routed_models,
        },
        "observed": {
            "n_core_observed": cen["n_observed_core"],
            "core_fraction": (cen["n_observed_core"] / cen["n_core"]) if cen["n_core"] else None,
            "non_core_observed": cen["non_core_observed"],
            "composite_observed": cen.get("composite_observed") or [],
            "unclassified_observed": cen.get("unclassified_observed") or [],
            "why_composites_are_separate": (
                "prov.aten records the FRONTEND op, so a non-core tag does not mean the "
                "capture is at the wrong IR level; a composite whose lowering is core is a "
                "different fact from an op nothing can name"
            ),
        },
        "precision": {
            "dtype_ok": dtype_ok,
            "dtype_blocked": dtype_blocked,
            "why_separate": (
                "precision is judged only over the family-admitted subset; a region whose "
                "family is admitted but whose dtype is not is a precision gap, not a "
                "routing one, and reporting it as a routing gap would misattribute it"
            ),
        },
        "routing": {
            "routed": routed,
            "fallback": fallback,
            "unclassified": unclassified,
            "denominator": denominator,
            "routed_fraction": (routed / denominator) if denominator else None,
            "why_unclassified_is_separate": (
                "a region whose family could not be determined is evidence neither of coverage "
                "nor of a gap; folding it into either is how a coverage number becomes a lie"
            ),
        },
        "work": {
            "routed": routed_work,
            "fallback": fallback_work,
            "unclassified": unclassified_work,
            "denominator": work_denominator,
            "routed_fraction": (routed_work / work_denominator) if work_denominator else None,
            # A LOWER BOUND is not a measurement. `iteration_space` returns `complete=False` when a
            # nest was only partially recovered, and a partially recovered nest that reads as exact
            # is how a heavy op gets ranked light.
            "exact": work_complete,
            "unit": "iteration-space extents x body arithmetic ops (merlin.analysis.programs.work)",
        },
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
        + (f"; {', '.join(unparsed)} did not parse and contributed no routing evidence" if unparsed else "")
        + (_work_clause(report.get("work") or {}))
        + "."
    )


def _work_clause(wk: dict) -> str:
    """The work half of the claim, or nothing at all.

    Silence when no work could be priced: a claim that omits the weighting is weaker than one that
    carries it, and both are honest. A claim that STATES a weighting nobody computed is neither.
    """
    frac = wk.get("routed_fraction")
    if frac is None:
        return ""
    qualifier = "" if wk.get("exact") else " (a lower bound; at least one iteration nest was only partially recovered)"
    return f"; those account for {_pct(frac)} of the roster's total loop-nest work{qualifier}"


# ---- conv_geometry -----------------------------------------------------------------------------
# The convolution geometry a captured model contains, recovered from bound evidence.
#
# Integerization can replace a convolution and its im2col gather with an integer
# matmul. For those captures, the receipt-bound original frontend graph supplies
# the source window; concrete operand/result shapes must verify its parameters.
# Captures without a source convolution trace use structural MLIR recovery below.
#
# A padded convolution is a defect class this corpus could not observe. Measured by
# ``check_defect_reach``: no capsule on any target declares a non-zero padding, so a lowering that
# loses the padding identity is wrong only in border rows nothing here computes -- and that is exactly
# how a fused convolution/max-pool shipped with a ``-128`` padding identity dropped and 119 wrong
# outputs.
#
# Reading ``padding``/``stride``/``dilation`` off lowered MLIR does not work:
# torch-mlir emits convolutions as **im2col**, so the lowered program contains a
# gather and a matmul and no convolution op at all.
#
# WHAT IS STILL THERE IS THE GEOMETRY ITSELF, in two structures:
#
# ``the gather's affine map``   ``(d3, d0, d4 * S + d1 * D, d5 * S + d2 * D)`` indexes the padded input.
#                               The coefficient on the OUTPUT dim is the stride; the coefficient on the
#                               KERNEL dim is the dilation; the kernel extent is that dim's extent in
#                               the iteration space. A ``k7x7 s1`` and a ``k3x3 s2`` are different maps,
#                               and they are different here even though the op names are identical.
# ``the producer insert_slice`` A padded convolution writes the real input into the interior of a larger
#                               zero tensor. ``static_offsets`` is the padding BEFORE, the destination
#                               extent minus offset minus size is the padding AFTER (which is how an
#                               ASYMMETRIC pad survives), and ``static_strides`` above 1 is an input
#                               dilation -- a transposed convolution, which this capture contains and
#                               which no hand-written corpus entry anticipated.
#
# NOTHING HERE IS GUESSED, AND THE ONE AMBIGUITY IS CHECKED. Within ``d_a * c_a + d_b * c_b`` the map
# alone does not say which dim is the kernel and which is the output: the expression is symmetric under
# swapping the pair. The extents break it -- a kernel is smaller than its output -- but an extent
# comparison is a heuristic, so it is only a PROPOSAL here. It is then verified against the geometry
# identity
#
#     padded_extent == (out - 1) * stride + (kernel - 1) * dilation + 1
#
# which the swapped assignment does not satisfy except in degenerate cases. A gather failing that
# identity yields ``None``: an unverified geometry is not recorded, because a wrong stride would demand
# capsules for a convolution the model does not contain.


@dataclass(frozen=True)
class ConvGeometry:
    """One convolution's window, recovered from the program rather than from an attribute."""

    kernel: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    pad_before: tuple[int, ...]
    pad_after: tuple[int, ...]
    input_dilation: tuple[int, ...]  # >1 means a TRANSPOSED convolution
    pad_known: bool  # False when no readable padding producer was found
    in_spatial: tuple[int, ...]  # the real input, before padding
    out_spatial: tuple[int, ...]
    channels_in: int
    dtype: str

    @property
    def padded(self) -> "bool | None":
        """True, False, or None when no padding producer was readable. Never False for unknown."""
        if not self.pad_known:
            return None
        return any(self.pad_before) or any(self.pad_after)

    @property
    def symmetric_pad(self) -> bool:
        return self.pad_before == self.pad_after

    def signature(self) -> str:
        """What the compiler must DO with this window — the grouping key.

        Extents are deliberately absent: two 3x3 stride-1 pad-1 convolutions exercise one lowering
        whatever their channel counts, and a corpus with a member per extent would grow without
        covering anything new. What IS in the key is every axis that changes the code: the window, how
        it steps, how it is spaced, whether the border is padded, whether that padding is symmetric,
        and whether the input is dilated.
        """
        k = "x".join(str(v) for v in self.kernel)
        s = "x".join(str(v) for v in self.stride)
        d = "x".join(str(v) for v in self.dilation)
        pb = "x".join(str(v) for v in self.pad_before)
        pa = "x".join(str(v) for v in self.pad_after)
        if not self.pad_known:
            pad = "UNKNOWN"
        else:
            pad = pb if self.symmetric_pad else f"{pb}_{pa}"
        base = f"k{k}/s{s}/d{d}/pad{pad}"
        return base + (
            "/indilated" + "x".join(str(v) for v in self.input_dilation)
            if any(v != 1 for v in self.input_dilation)
            else ""
        )

    def to_dict(self) -> dict:
        return {
            "signature": self.signature(),
            "kernel": list(self.kernel),
            "stride": list(self.stride),
            "dilation": list(self.dilation),
            "pad_before": list(self.pad_before),
            "pad_after": list(self.pad_after),
            "input_dilation": list(self.input_dilation),
            "pad_known": self.pad_known,
            "in_spatial": list(self.in_spatial),
            "out_spatial": list(self.out_spatial),
            "channels_in": self.channels_in,
            "dtype": self.dtype,
            "padded": self.padded,
            "symmetric_pad": self.symmetric_pad,
        }


def _terms(expr) -> "dict[int, int] | None":
    """``{iteration dim: coefficient}`` for an affine result, or None when it is not affine-linear.

    Fails closed on anything it does not model -- a ``mod``, a ``floordiv``, a symbol. Those appear in
    tiled or packed forms whose geometry is not read the same way, and treating one as linear would
    return a stride that is not the program's.
    """
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

    if isinstance(expr, AffineDimExpr):
        return {int(expr.position): 1}
    if not isinstance(expr, AffineBinaryOpExpr):
        return None
    lhs, rhs = _terms(expr.lhs), _terms(expr.rhs)
    if expr.kind is AffineBinaryOpKind.Add:
        if lhs is None or rhs is None:
            return None
        out = dict(lhs)
        for d, c in rhs.items():
            out[d] = out.get(d, 0) + c
        return out
    if expr.kind is AffineBinaryOpKind.Mul:
        # Exactly one side must be a constant; a dim-by-dim product is not affine.
        for a, b in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            if isinstance(a, AffineConstantExpr):
                inner = _terms(b)
                if inner is None:
                    return None
                return {d: c * int(a.value) for d, c in inner.items()}
    return None


def _pad_chain(value) -> "tuple[dict, bool]":
    """Walk back through ``tensor.insert_slice`` producers, accumulating padding and input dilation.

    A transposed convolution pads in TWO steps -- first a strided insert that spaces the input out,
    then an offset insert that adds the border -- so a reader that stops at the first producer sees
    the spacing and misses the border, or the reverse. Both are collected, and the walk is bounded so
    a malformed chain cannot spin.

    The second return value is whether a producer chain was found at all. ``False`` means the gather
    reads its input directly, which is a genuine zero padding, NOT an unknown one.
    """
    from merlin.compiler.ir import mlir_query as mq

    pad_before: dict[int, int] = {}
    pad_after: dict[int, int] = {}
    in_dil: dict[int, int] = {}
    found = False
    for _ in range(8):
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "name"):
            break
        if mq.op_name(owner) != "tensor.insert_slice":
            break
        tables = list(mq._attr_tables(owner))  # noqa: PLC2701 -- the same reader indexing_maps uses
        prop = {}
        for t in tables:
            for k in ("static_offsets", "static_sizes", "static_strides"):
                if k in t and k not in prop:
                    prop[k] = [int(v) for v in getattr(t[k], "get_values", lambda: t[k].data)()]
        if len(prop) != 3:
            break
        dest = None
        for res in getattr(owner, "results", ()):
            shp = getattr(getattr(res, "type", None), "get_shape", None)
            if shp is not None:
                dest = [int(v) for v in shp()]
        if dest is None or len(dest) != len(prop["static_offsets"]):
            break
        found = True
        for i, (off, size, strd) in enumerate(
            zip(prop["static_offsets"], prop["static_sizes"], prop["static_strides"])
        ):
            # A strided insert SPACES the source out: it occupies (size-1)*stride + 1 of the
            # destination. Anything past that on either side is border padding.
            span = (size - 1) * strd + 1
            pad_before[i] = pad_before.get(i, 0) + off
            pad_after[i] = pad_after.get(i, 0) + (dest[i] - off - span)
            if strd != 1:
                in_dil[i] = strd
        value = owner.operands[0] if getattr(owner, "operands", None) else None
        if value is None:
            break
    return {"before": pad_before, "after": pad_after, "input_dilation": in_dil}, found


def _im2col_km(op) -> "tuple[int, int] | None":
    """``(K, M)`` of the matrix the im2col gather feeds, or None.

    THIS IS THE DISCRIMINATOR, and nothing weaker works. Inside ``d_a * c_a + d_b * c_b`` the affine
    map is symmetric under swapping the pair: one dim is the kernel and one is the output, and the map
    alone cannot say which. The obvious tie-break -- "a kernel is smaller than its output" -- is FALSE,
    and it was measured false on a real capture in this store: a 4x4 kernel producing a 2x3 output read
    as a 2x3 kernel with dilation 4, which is a different convolution from the one the model contains.
    The convolution output identity does not separate them either; both assignments satisfy it there.

    What does separate them is the shape the gather is reshaped INTO. im2col produces
    ``[Cin * prod(kernel), N * prod(output)]``, so ``K`` names the kernel side unambiguously. It is
    read by following the gather's result forward through the reshapes to the first rank-2 value --
    structurally, not by assuming a fixed op sequence -- and a gather whose chain does not reach one
    yields None, so the geometry is dropped rather than guessed.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.ir.contraction_shapes import _shaped  # noqa: PLC2701

    frontier = [r for r in getattr(op, "results", ())]
    for _ in range(6):
        nxt = []
        for value in frontier:
            for use in getattr(value, "uses", ()):
                user = getattr(use, "operation", None)
                if user is None or mq.op_name(user) not in ("tensor.collapse_shape", "tensor.expand_shape"):
                    continue
                for res in getattr(user, "results", ()):
                    shaped = _shaped(res)
                    if shaped is not None and len(shaped[0]) == 2:
                        return (int(shaped[0][0]), int(shaped[0][1]))
                    nxt.append(res)
        if not nxt:
            return None
        frontier = nxt
    return None


def _resolve_axes(axes, *, bare, padded, k_total: int, m_total: int):
    """Decide kernel-vs-output for every window axis at once, or return None.

    Every axis is resolved TOGETHER rather than one at a time, because the evidence is a single
    product: ``K == channel * prod(kernel extents)``. Both orderings of the two bare dims (batch and
    channel) are tried, since which is which is a layout convention this reader declines to assume.

    Returns ``(axes, channel_extent)``, or None when zero assignments fit or when more than one does.
    More than one is the important case: it means the evidence does not determine the geometry, and
    emitting either would put a convolution in the requirement that the model may not contain.

    The channel extent falls out of the same arithmetic (``K // prod(kernel)``) rather than being
    scanned for separately, so it cannot disagree with the assignment it came from.
    """
    from itertools import product

    fits = []
    for choice in product((0, 1), repeat=len(axes)):
        kernels, strides, dils, poss, outs = [], [], [], [], []
        for (pos, da, ca, db, cb, ea, eb), pick in zip(axes, choice):
            # pick 0: `da` is the kernel dim; pick 1: `db` is.
            kdim, odim = (da, db) if pick == 0 else (db, da)
            kext, oext = (ea, eb) if pick == 0 else (eb, ea)
            dil = ca if kdim == da else cb
            strd = cb if kdim == da else ca
            kernels.append(kext)
            strides.append(strd)
            dils.append(dil)
            poss.append(pos)
            outs.append(oext)
        kprod = 1
        for v in kernels:
            kprod *= v
        oprod = 1
        for v in outs:
            oprod *= v
        if kprod <= 0 or k_total % kprod:
            continue
        channel = k_total // kprod
        # The two bare dims are batch and channel in some order; the K side takes one and the M side
        # the other, and the M side must then account for m_total exactly.
        rest = list(bare)
        if channel not in rest:
            continue
        rest.remove(channel)
        if oprod * rest[0] != m_total:
            continue
        # The convolution output identity, applied last as a consistency check rather than as the
        # discriminator it cannot be.
        if any(
            o != (padded[pos] - ((k - 1) * d + 1)) // s + 1
            for k, s, d, pos, o in zip(kernels, strides, dils, poss, outs)
        ):
            continue
        fits.append((list(zip(kernels, strides, dils, poss, outs)), channel))
    if len(fits) != 1:
        return None  # undetermined, or contradictory -- never guessed
    return fits[0]


def geometries(src) -> list[ConvGeometry]:
    """Every im2col convolution in ``src``, as verified geometries.

    Returns an EMPTY list rather than raising on an unreadable module, matching
    ``kernels.shapes.observe_contractions``: a geometry observer that fails must degrade to "I saw
    nothing" so the caller falls back rather than failing a build over one capture.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.ir.contraction_shapes import (
        _shaped,
        indexing_maps,  # noqa: PLC2701
    )

    try:
        module = mq.parse(src)
    except Exception:  # noqa: BLE001
        return []

    out: list[ConvGeometry] = []
    for op in mq.walk(module, "linalg.generic"):
        try:
            maps = indexing_maps(op)
            if not maps or len(maps) < 2:
                continue
            res = _shaped(getattr(op, "results", (None,))[0])
            src_shape = _shaped(op.operands[0]) if getattr(op, "operands", None) else None
            if res is None or src_shape is None:
                continue
            extents, _ = res
            padded, dtype = src_shape

            # The output map must be the identity for the result shape to BE the iteration space.
            # Anything else and the extents below would be indexed wrongly, so it is required rather
            # than assumed.
            out_terms = [_terms(r) for r in maps[-1]]
            if len(out_terms) != len(extents) or any(t is None or t != {i: 1} for i, t in enumerate(out_terms)):
                continue

            # The K side of the im2col, read from the reshape the gather feeds. This is what DECIDES
            # which dim of each `d_a * c_a + d_b * c_b` is the kernel and which is the output; see
            # `_im2col_km`.
            km = _im2col_km(op)
            if km is None:
                continue
            k_total, m_total = km

            axes: list[tuple[int, int, int, int, int, int, int]] = []  # pos + both candidates
            bare: list[int] = []
            ok = True
            for pos, r in enumerate(maps[0]):
                t = _terms(r)
                if t is None:
                    ok = False
                    break
                if len(t) == 1:
                    bare.append(extents[next(iter(t))])
                    continue  # batch or channel, not a window axis
                if len(t) != 2:
                    ok = False
                    break
                (da, ca), (db, cb) = sorted(t.items())
                axes.append((pos, da, ca, db, cb, extents[da], extents[db]))
            if not ok or not axes or len(bare) != 2:
                continue

            resolved = _resolve_axes(axes, bare=bare, padded=padded, k_total=k_total, m_total=m_total)
            if resolved is None:
                continue
            spatial, cin = resolved

            # ⚠️ NO PADDING PRODUCER MEANS UNKNOWN, NOT ZERO. Measured: this capture's first
            # convolution is padded by an `aten.index.Tensor` gather -- a REFLECTION pad, whose
            # identity is not zero and whose offsets are nowhere to read. Reporting it as `pad0` would
            # have been wrong twice: it claims no padding where there is padding, and it claims the
            # zero identity for one that reflects. An unknown padding is recorded as unknown and
            # becomes its own obligation class, which is a finding about the corpus rather than a
            # silent default.
            pads, pad_known = _pad_chain(op.operands[0])
            pb = tuple(pads["before"].get(pos, 0) for (_, _, _, pos, _) in spatial)
            pa = tuple(pads["after"].get(pos, 0) for (_, _, _, pos, _) in spatial)
            idil = tuple(pads["input_dilation"].get(pos, 1) for (_, _, _, pos, _) in spatial)
            # The real input, before the padding was added and before any input dilation spaced it.
            ins = tuple(
                ((padded[pos] - pb[i] - pa[i]) - 1) // idil[i] + 1 for i, (_, _, _, pos, _) in enumerate(spatial)
            )
            out.append(
                ConvGeometry(
                    kernel=tuple(k for k, _, _, _, _ in spatial),
                    stride=tuple(s for _, s, _, _, _ in spatial),
                    dilation=tuple(d for _, _, d, _, _ in spatial),
                    pad_before=pb,
                    pad_after=pa,
                    input_dilation=idil,
                    pad_known=pad_known,
                    in_spatial=ins,
                    out_spatial=tuple(o for *_, o in spatial),
                    channels_in=cin,
                    dtype=dtype,
                )
            )
        except Exception:  # noqa: BLE001 -- one bad op never kills the walk
            continue
    return out


#: Parsed-geometry cache, keyed by the capture path. A capture's convolution windows are a property of
#: the CAPTURE and not of any target, but the requirement is derived once per target -- so without this
#: the same 29 model.mlir files are re-parsed for every target, which is most of what a spec
#: regeneration spends its time on. Keyed by (path, mtime, size) so an edited capture re-parses.
_GEOMETRY_CACHE: dict = {}


def _cached_geometries(path) -> list[ConvGeometry]:
    from pathlib import Path as _P

    try:
        st = _P(path).stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return geometries(path)  # not a file we can stamp; parse it and do not cache
    hit = _GEOMETRY_CACHE.get(key)
    if hit is None:
        hit = geometries(path)
        _GEOMETRY_CACHE[key] = hit
    return hit


def _trace_geometries(path: Path) -> list[ConvGeometry] | None:
    """Read source convolution windows when integerization erased the im2col gather.

    The original frontend graph is part of the capture receipt, not an independently
    supplied model. Reject malformed geometry instead of inferring it from an int_mm.
    ``None`` means there is no source convolution trace, so MLIR recovery applies.
    """
    trace_path = path.parent / "frontend-trace.json"
    if not trace_path.is_file():
        return None
    from merlin.development.targetgen.applications.inventory import verify_capture_receipt

    receipt = verify_capture_receipt(path)
    if receipt["status"] != "verified_materialized":
        raise ValueError(f"capture receipt does not bind the frontend trace: {receipt['errors']}")
    receipt_doc = json.loads((path.parent / "capture_receipt.json").read_bytes())
    if "frontend-trace.json" not in (receipt_doc.get("artifacts") or {}):
        raise ValueError("frontend trace is not a capture receipt artifact")
    doc = json.loads(trace_path.read_bytes())
    if doc.get("schema") != "m2m.frontend_trace.v1" or doc.get("blockers"):
        raise ValueError("frontend trace is unsupported or blocked")
    graph = (doc.get("graphs") or {}).get("original")
    if not isinstance(graph, dict) or graph.get("status") != "complete" or not isinstance(graph.get("nodes"), list):
        raise ValueError("original frontend graph is incomplete or has no node list")
    nodes = graph["nodes"]
    convolutions = [node for node in nodes if node.get("target") == "aten.conv2d.default"]
    declared_count = (graph.get("by_target") or {}).get("aten.conv2d.default")
    if declared_count is not None and declared_count != len(convolutions):
        raise ValueError("original frontend convolution count differs from the node list")
    if not convolutions:
        return None
    values = {
        value.get("id"): value for node in nodes for value in (node.get("results") or []) if isinstance(value, dict)
    }

    def shape(ref: object) -> tuple[int, ...]:
        if not isinstance(ref, dict) or not isinstance(ref.get("value_id"), str):
            raise ValueError("convolution tensor operand has no source value identity")
        dims = (values.get(ref["value_id"]) or {}).get("shape")
        if not isinstance(dims, list) or not dims or any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError("convolution tensor operand has no concrete positive shape")
        return tuple(dims)

    def pair(value: object, name: str) -> tuple[int, int]:
        if type(value) is int:
            value = [value, value]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(type(v) is not int or v < (0 if name == "padding" else 1) for v in value)
        ):
            raise ValueError(f"convolution {name} is not a concrete two-dimensional extent")
        return tuple(value)

    found = []
    for node in convolutions:
        args = node.get("args") or []
        # A bias-free convolution exports with just (input, weight): two positional arguments.
        if not isinstance(args, list) or not 2 <= len(args) <= 7 or node.get("kwargs"):
            raise ValueError("convolution source arguments are not in the supported positional form")
        input_shape, weight_shape = shape(args[0]), shape(args[1])
        results = node.get("results") or []
        if len(results) != 1:
            raise ValueError("convolution source has no unique result")
        output_shape = shape({"value_id": results[0].get("id")})
        if any(len(s) != 4 for s in (input_shape, weight_shape, output_shape)):
            raise ValueError("convolution source is not NCHW/OIHW rank four")
        stride = pair(args[3] if len(args) > 3 else [1, 1], "stride")
        padding = pair(args[4] if len(args) > 4 else [0, 0], "padding")
        dilation = pair(args[5] if len(args) > 5 else [1, 1], "dilation")
        groups = args[6] if len(args) > 6 else 1
        if type(groups) is not int or groups <= 0 or input_shape[1] != weight_shape[1] * groups:
            raise ValueError("convolution source channel/group relation is invalid")
        if input_shape[0] != output_shape[0] or weight_shape[0] != output_shape[1]:
            raise ValueError("convolution source batch/output-channel relation is invalid")
        expected = tuple(
            (input_shape[2 + axis] + 2 * padding[axis] - dilation[axis] * (weight_shape[2 + axis] - 1) - 1)
            // stride[axis]
            + 1
            for axis in range(2)
        )
        if expected != output_shape[2:]:
            raise ValueError("convolution source window does not reproduce the captured output shape")
        found.append(
            ConvGeometry(
                kernel=weight_shape[2:],
                stride=stride,
                dilation=dilation,
                pad_before=padding,
                pad_after=padding,
                input_dilation=(1, 1),
                pad_known=True,
                in_spatial=input_shape[2:],
                out_spatial=output_shape[2:],
                channels_in=input_shape[1],
                dtype=str(results[0].get("dtype") or "unknown"),
            )
        )
    return found


def geometry_classes(captures: dict) -> dict:
    """The distinct convolution geometries a set of captures contains, with their evidence.

    ``captures`` is ``{label: path to model.mlir}``, the shape every other derivation reader takes.
    """
    by_sig: dict[str, dict] = {}
    unreadable: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            traced = _trace_geometries(Path(path)) if isinstance(path, (str, Path)) else None
            found = traced if traced is not None else _cached_geometries(path)
        except Exception as e:  # noqa: BLE001 -- reported, never skipped silently
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"
            continue
        for g in found:
            row = by_sig.setdefault(g.signature(), {**g.to_dict(), "n_regions": 0, "sources": []})
            row["n_regions"] += 1
            if label not in row["sources"]:
                row["sources"].append(label)
    return {
        "required": [by_sig[s] for s in sorted(by_sig)],
        "n_classes": len(by_sig),
        "captures_unreadable": unreadable,
        "axis_basis": (
            "convolution windows in receipt-bound original frontend graphs, with input/weight/output "
            "shapes checked against stride, padding and dilation; when a capture has no source "
            "convolution trace, recover from the im2col gather and its padding producer. Unreadable "
            "evidence is reported, never converted to an inferred window"
        ),
    }


__all__ = [
    "ConvGeometry",
    "geometries",
    "geometry_classes",
]
