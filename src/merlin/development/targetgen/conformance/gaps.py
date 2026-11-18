"""Measure declared source-chain coverage independently of runtime placement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from merlin.development.targetgen.applications.demand_inventory import application_demand_inventory
from merlin.development.targetgen.conformance.requirements import (
    DECLARED,
    OBSERVED,
    OBSERVED_VIA_PRIMITIVES,
    Boundaries,
    Cell,
    CellOrigin,
)

# ---- conformance_scope -------------------------------------------------------------------------


def _scope_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required CHAIN signatures no capsule in the corpus presents.

    A capsule witnesses a chain when its own program contains that run of adjacent regions. Almost no
    capsule does: the corpus is built one region at a time, which is exactly the finding. Reporting it as
    a measured gap rather than leaving the rung empty is the point -- an unpopulated rung reads as "not
    covered yet", while a measured zero against a derived requirement says the corpus cannot observe the
    lever at all.

    A capsule's chain is read from its own linalg when it ships one. A capsule with no linalg module
    contributes NOTHING rather than counting as a chain of one: absence of a program is not evidence of
    a single-region program, and treating it as one would manufacture coverage of the shortest chains.
    """

    import yaml

    from merlin.compiler.ir import mlir_query as mq
    from merlin.development.targetgen.applications import operation_plan as SC

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    have: dict[str, list[str]] = {}
    unreadable = 0
    no_program = 0
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if name in exclude:
                continue
            selected = cap.get("linalg_mlir", "capsule.linalg.mlir")
            if not isinstance(selected, str) or not selected:
                unreadable += 1
                continue
            relative = Path(selected)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                unreadable += 1
                continue
            linalg = cy.parent
            for part in relative.parts:
                linalg = linalg / part
                if linalg.is_symlink():
                    unreadable += 1
                    break
            else:
                if not linalg.is_file():
                    no_program += 1
                    continue
                try:
                    for chain in SC.chains(mq.parse(linalg), max_length=64):
                        have.setdefault(chain.signature, []).append(name)
                except Exception:  # noqa: BLE001
                    unreadable += 1
                continue
    want = [r["signature"] for r in (required or [])]
    missing = [sig for sig in want if sig not in have]
    # ONE VOCABULARY ACROSS EVERY AXIS. This function used to answer in its own words -- `required`,
    # `covered`, `missing`, and no `status` at all -- while every sibling gap function returns
    # `status`/`n_required`/`n_covered`/`uncovered`. Nothing outside `uncovered()` read it, so the
    # difference cost nothing until a caller tried to treat the axes uniformly: the gate's reporter
    # printed `scope: missing` (its default for a dict with no `status`) and its debt accumulator looked
    # for an `uncovered` key that was never there, so an adjacency gap could not reach the verdict
    # however the accumulator was written. An axis that spells its answer differently from its siblings
    # is an axis that will be skipped by anything written to handle them together.
    return {
        "status": "ok",
        "n_required": len(want),
        "n_covered": len(want) - len(missing),
        "uncovered": missing,
        "witnessed_by": {sig: sorted(set(v)) for sig, v in have.items() if sig in set(want)},
        # Reported rather than folded into `uncovered`: a capsule with no linalg module cannot be asked
        # what chain it contains, and counting it as covering nothing is a different claim from counting
        # it as containing nothing.
        "capsules_without_a_program": no_program,
        "capsules_unreadable": unreadable,
        "qualification": "declared graph-input adjacency only; target compilation and execution are not observed",
        "note": (
            "an uncovered signature is an adjacency chain real captures present that no capsule "
            "contains: the corpus is built one region at a time, so a compiler that mishandles a "
            "boundary between two regions is tested by nothing here"
        ),
    }


# ---- conformance -------------------------------------------------------------------------------
# What a target's whole-model corpus MUST cover, derived — never typed.
#
# A corpus of hand-picked whole-model capsules describes whichever architectures somebody happened to
# author. Radiance's did exactly that: the only model capsule ever graded is a LLaMA decoder, so its
# captured linalg carries no ``tanh``, no ``erf`` and no convolution, and nothing in the scorecard said the
# evidence was one architecture wide. Widening it by hand-writing "the corpus shall contain these eleven
# op families" trades one overfit for another: the list is stale the moment a target or a target-model is
# added, and it cannot report what it is missing.
#
# So the REQUIREMENT is computed from three independent sources, and the corpus is measured against it:
#
# ``admitted``
#     what the silicon declares it can compute — the target's capability manifest
#     (:func:`merlin.target.capabilities.eligibility.capability_map_for_target`), family x dtype x rank. A target
#     with no fp8 datapath does not get fp8 cells.
#
# ``observed``
#     what the real target-models actually contain — the family census of each captured model
#     (:func:`merlin.development.targetgen.conformance.model_coverage.regions_from_module`). A family no real model uses is not
#     required of the corpus, however capable the hardware is.
#
# ``boundaries``
#     where the hardware's own edges are — the mesh/array edge and the MX block-scale group, read from the
#     target's facts. Edge cases are generated at those boundaries rather than imagined.
#
# The required set is ``admitted INTERSECT observed``, expressed in the SAME cell vocabulary
# :func:`merlin.development.targetgen.corpus.materialization.cert_capsule_cover` already uses — ``(semantic_family,
# dtype, tile_alignment)`` — so the existing cover machinery measures conformance for free and the two can
# never disagree about what a cell is.
#
# Every cell carries its provenance: which capture observed the family, which compute unit admitted the
# dtype, or ``DECLARED`` plus a citation where a target-model could not be captured at all. A declared cell
# is never silently indistinguishable from an observed one — that distinction is the whole point of
# deriving, and DeepSeek-R1-Distill-Qwen has no capture, so its column can only be declared until a
# ``model2MLIR`` loader for it exists.


def _conv_geometry_axis(captures: dict) -> dict:
    """The convolution windows this target's captures contain, as an obligation set.

    Kept out of the cell cross-product deliberately, for the same reason the geometry and composition
    axes are: crossing windows with dtypes would demand a 7x7 reflection-padded convolution at every
    format the target admits, and no model contains that.
    """
    from merlin.development.targetgen.applications import capture_census as CG

    return CG.geometry_classes(captures)


def _conv_geometry_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required convolution windows no capsule in the corpus presents.

    A capsule declares its window as attributes -- there is no im2col to read on this side, because a
    capsule is authored as a convolution rather than captured as one. The two are brought into the same
    vocabulary by building a :class:`conv_geometry.ConvGeometry` from the declaration and taking its
    signature, so "the corpus covers this window" means the same thing on both sides of the comparison.

    A capsule declaring a window but no padding is ``pad_known=True`` with zero padding: unlike the
    capture side, where an unreadable producer means the padding could not be seen, here the absence of
    a padding attribute is the capsule AUTHOR saying there is none.
    """
    import yaml

    from merlin.development.targetgen.applications.capture_census import ConvGeometry
    from merlin.development.targetgen.conformance.requirements import _capsule_paths

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)

    def _pair(v, default):
        if v is None:
            return (default, default)
        if isinstance(v, (list, tuple)):
            vals = [int(x) for x in v]
            return (vals[0], vals[-1]) if vals else (default, default)
        return (int(v), int(v))

    have: dict[str, list[str]] = {}
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if name in exclude:
                continue
            a = ((cap.get("operation") or {}).get("attributes")) or {}
            if not isinstance(a, dict):
                continue
            kh, kw = a.get("kh"), a.get("kw")
            if kh is None and kw is None:
                pool = a.get("pool_size")
                if pool is None:
                    continue
                kh, kw = _pair(pool, 1)
            kernel = (int(kh if kh is not None else kw), int(kw if kw is not None else kh))
            stride = _pair(a.get("stride") if a.get("stride") is not None else a.get("pool_stride"), 1)
            dilation = _pair(a.get("dilation"), 1)
            pad = _pair(a.get("padding") if a.get("padding") is not None else a.get("pool_padding"), 0)
            g = ConvGeometry(
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                pad_before=pad,
                pad_after=pad,
                input_dilation=(1, 1),
                pad_known=True,
                in_spatial=(),
                out_spatial=(),
                channels_in=0,
                dtype=str(a.get("dtype") or ""),
            )
            have.setdefault(g.signature(), []).append(name)

            # A DECLARED obligation is honoured only where the derived one CANNOT express it, and the
            # window is still verified. A capsule built for a `padUNKNOWN` class declares zero padding
            # -- the only padding that can be spelled -- so re-deriving its signature yields `pad0x0`
            # and the obligation it was built for could never be satisfied by it. The declaration
            # closes that gap without becoming a way to claim anything: the kernel, stride and
            # dilation of the declared signature must match what the capsule actually carries, so only
            # the unspellable padding component is taken on trust.
            declared = str(((cap.get("semantic") or {}).get("conv_window")) or "")
            if declared and declared != g.signature():
                derived_window = g.signature().rsplit("/", 1)[0]
                if declared.startswith(derived_window + "/"):
                    have.setdefault(declared, []).append(name)

    want = [str(r["signature"]) for r in (required or [])]
    missing = [sig for sig in want if sig not in have]
    return {
        "status": "ok",
        "n_required": len(want),
        "n_covered": len(set(want) & set(have)),
        "uncovered": missing,
        "covered_by": {k: sorted(v) for k, v in sorted(have.items()) if k in set(want)},
        "corpus_windows": sorted(have),
        "note": (
            "a required window no capsule presents means a lowering that loses the padding "
            "identity, mis-steps the stride, or mis-spaces the dilation is wrong only in rows "
            "this corpus never computes. A `padUNKNOWN` obligation is a window whose padding the "
            "capture applies by a route this reader cannot see -- a reflection pad, for one -- "
            "and it is a real obligation: its identity is NOT zero"
        ),
    }


_GROUP_OP_CLASS = {"matmul": "matmul", "linear": "matmul", "fused_matmul_bias": "matmul", "conv2d": "conv2d"}


def _group_signature(op: str, epilogue, dtype: str | None) -> str:
    from merlin.development.targetgen.conformance.requirements import capsule_dtype
    from merlin.runtime.engine.command_buffer import EPILOGUE_STAGES

    # In the ABI's declared stage order, so two listings of one combination are one signature:
    # what is demanded is that the stages fuse together, which the listing order does not change.
    rank = {stage: index for index, stage in enumerate(EPILOGUE_STAGES)}
    stages = sorted((str(s) for s in epilogue), key=lambda s: (rank.get(s, len(rank)), s))
    return (
        f"{_GROUP_OP_CLASS.get(str(op), str(op))}|{'+'.join(stages) or '-'}|"
        f"{capsule_dtype(str(dtype)) if dtype else '?'}"
    )


def _derived_epilogue_groups(target: str, oracle) -> list[dict]:
    """Fused-epilogue group demands from the target's DERIVED output path, not from a capture.

    A capture forms a group only where its numbers let the readout hold them: a model quantized per
    output channel on a unit holding one scale per command forms none, and a corpus derived from such
    captures alone would never ask a backend for the epilogue the hardware performs on every
    contraction. So each contraction format the target declares gets the requantizing chain its
    output path is DERIVED to apply on move-out -- a contraction-scoped bias route, then the scale,
    then the rectifier -- and the same chain closed by the derived pool. Only stages the epilogue axis
    requires are chained, so every demand is one a capsule builder writes and a readout declaration
    does not contradict. Extents and values are the capsule builder's own: nothing here is a
    workload's shape.
    """
    from merlin.development.targetgen.conformance.axes import _epilogue_axis

    readout = getattr(oracle, "readout", None)
    from merlin.runtime.engine.command_buffer import BIAS_STAGES

    path = _declared_output_path(target)
    preloads = False
    if path is not None:
        # The ordered output path: each declared stage kind in the command-buffer stage vocabulary.
        derived = set(_path_stages(path))
        preloads = any(stage.get("kind") == "preload" for stage in path.get("accumulator_stages") or ())
    else:
        derived = {stage for facet in getattr(readout, "facets", ()) or () for stage in facet.derived_stages}
    if not derived:
        return []
    writable = {str(row.get("stage")) for row in (_epilogue_axis(target) or {}).get("required") or ()}
    routes = getattr(oracle, "routes", ()) or ()
    bias = [
        stage
        for stage in BIAS_STAGES
        if stage in writable
        and (preloads or any(r.stage == stage and r.composed_with == "contraction" for r in routes))
    ][:1]
    core = [*bias, *(stage for stage in ("acc_scale", "relu") if stage in derived and stage in writable)]
    chains = [core] + ([[*core, "maxpool"]] if "maxpool" in derived and "maxpool" in writable else [])
    contraction = (getattr(oracle, "cap_map", {}) or {}).get("contraction")
    granularities = [g for g in getattr(readout, "scale_granularities", ()) if g == "tensor"]
    rows = []
    for chain in chains:
        if len(chain) < 2 or "acc_scale" not in chain:
            continue  # a single stage is the epilogue axis's member; an unscaled chain is not a readout
        for dtype in getattr(contraction, "dtypes", ()) or ():
            rows.append(
                {
                    "signature": _group_signature("matmul", chain, dtype),
                    "op": "matmul",
                    "epilogue": list(chain),
                    "dtype": dtype,
                    "scale_granularities": granularities,
                    "groups": 0,
                    "observed_in": [],
                    "evidenced_by": ["derived_output_epilogue"],
                }
            )
    return rows


def _declared_output_path(target: str):
    try:
        from merlin.compiler.partition.groups import output_fusion as OF

        return OF.declared_path(target)
    except (KeyError, ValueError):  # an unreadable declaration evidences nothing
        return None


def _path_stages(path) -> list[str]:
    """The command-buffer stages an output path's readout performs, in its order."""
    out = []
    for stage in path.get("readout") or ():
        kind = stage.get("kind")
        if kind == "scale":
            out.append("acc_scale")
        elif kind == "activation" and stage.get("function") == "relu":
            out.append("relu")
        elif kind == "pool" and "max" in (stage.get("reductions") or ()):
            out.append("maxpool")
    return out


def _unstatable_path_stages(path) -> dict[str, str]:
    """Declared output-path capabilities no capsule stage can demand yet, each with why."""
    from merlin.development.targetgen.corpus.derived_models import unstated

    return unstated(path)


def _group_axis(captures: dict[str, str | Path], target: str) -> dict:
    """The compute groups the target's captured models form, as capsule demands.

    Derived by the pass that decides placement (:mod:`compute_groups`), so a demand here is a
    group the compiler will actually try to put on a unit, with the stages it will try to fuse.
    A group whose stages the capsule vocabulary cannot state is counted under ``unstated`` with
    the reason: it is a group no capsule can demand yet, which is a finding about the vocabulary.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition import memory_plan as stream_plan
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )
    from merlin.compiler.partition.groups import plan_compute_groups as _merlin_plan_compute_groups_1

    required: dict[str, dict] = {}
    unstated: Counter = Counter()
    unreadable: dict[str, str] = {}
    try:
        oracle = CG.TargetOracle(target)
    except Exception as exc:  # noqa: BLE001 -- no contract, no groups to derive
        return {
            "required": [],
            "unstated": {},
            "captures_unreadable": {},
            "status": "not_derivable",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    for model, capture in sorted(captures.items()):
        try:
            # Through the file door: it re-prints a module MLIR wrote in a custom form this parser
            # does not read, so a capture is unreadable here only when it is unreadable.
            demand = _merlin_plan_compute_groups_1.demand(
                _merlin_formation_compute_groups_1.form_groups(mq.parse(str(capture)), target, oracle=oracle),
                weight_args=stream_plan.weight_args_beside(capture),
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never silently dropped
            unreadable[str(model)] = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        for reason, count in demand["unstated"].items():
            unstated[reason] += count
        for entry in demand["entries"]:
            signature = _group_signature(entry["op"], entry["epilogue"], entry.get("operand_dtype"))
            row = required.setdefault(
                signature,
                {
                    "signature": signature,
                    "op": _GROUP_OP_CLASS.get(entry["op"], entry["op"]),
                    "epilogue": list(entry["epilogue"]),
                    "dtype": entry.get("operand_dtype"),
                    "scale_granularities": [],
                    "groups": 0,
                    "observed_in": [],
                },
            )
            row["groups"] += int(entry["count"])
            if str(model) not in row["observed_in"]:
                row["observed_in"].append(str(model))
            granularity = entry.get("scale_granularity")
            if granularity and granularity not in row["scale_granularities"]:
                row["scale_granularities"].append(granularity)
    for row in _derived_epilogue_groups(target, oracle):
        required.setdefault(row["signature"], row)
    from merlin.development.targetgen.corpus.derived_models import requirements_for_oracle

    for row in requirements_for_oracle(_declared_output_path(target), oracle, required.values()):
        required.setdefault(row["signature"], row)
    for name, why in _unstatable_path_stages(_declared_output_path(target)).items():
        unstated[f"derived output path {name}: {why}"] += 1
    return {
        "status": "ok",
        "required": sorted(required.values(), key=lambda r: (-r["groups"], r["signature"])),
        "unstated": dict(unstated),
        "captures_unreadable": unreadable,
        "axis_basis": (
            "the compute groups this target's captured models form, and the fused epilogue its "
            "DERIVED output path performs on every contraction family it declares (whether or not a "
            "capture happens to form it): a contraction with the stages "
            "its unit absorbs, by op, epilogue COMBINATION and operand format. A corpus that "
            "demands each stage once demands no combination, so a backend that fuses any one stage "
            "and no two passes it; and the model route is only as good as the patterns some capsule "
            "has certified"
        ),
    }


def _carried_state_axis(target: str) -> dict:
    """One requirement per configuration-only stage this target can fuse onto a contraction."""
    from merlin.development.targetgen.conformance.axes import _epilogue_axis
    from merlin.runtime.engine.command_buffer import CONFIGURATION_ONLY_STAGES

    fused = (_epilogue_axis(target) or {}).get("required") or ()
    required = [
        {"stage": str(row.get("stage")), "evidenced_by": row.get("evidenced_by")}
        for row in fused
        if str(row.get("stage")) in CONFIGURATION_ONLY_STAGES
    ]
    return {
        "required": required,
        "axis_basis": (
            "a readout stage with no operand and no parameter is configuration a unit stays in; a "
            "program is two commands over one resident weight, the stage on the FIRST and none on the "
            "second, on a stimulus that goes negative, so a backend that does not restore its "
            "configuration returns the second result with the first one's stage applied"
        ),
    }


def _carried_state_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which configuration-only stages no capsule follows with a command that lacks them."""
    import yaml

    from merlin.development.targetgen.conformance.requirements import _capsule_paths
    from merlin.runtime.engine.command_buffer import STIMULUS_RANGE_KEY

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    want = [str(r.get("stage")) for r in (required or []) if r.get("stage")]
    shown: dict[str, list[str]] = {}
    wrong_order: dict[str, list[str]] = {}
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if cap.get("label") not in labels or name in exclude:
                continue
            attrs = (cap.get("operation") or {}).get("attributes")
            commands = attrs.get("matmuls") if isinstance(attrs, dict) else None
            if not isinstance(commands, list) or len(commands) < 2:
                continue
            stages = [[str(x) for x in (c.get("epilogue") or ())] for c in commands if isinstance(c, dict)]
            signed = (cap.get(STIMULUS_RANGE_KEY) or [0])[0] < 0
            for stage in want:
                leaks_visible = any(
                    stage in stages[i] and any(stage not in later for later in stages[i + 1 :])
                    for i in range(len(stages))
                )
                if leaks_visible and signed:
                    shown.setdefault(stage, []).append(name)
                elif any(stage in seq for seq in stages):
                    wrong_order.setdefault(stage, []).append(name)
    missing = sorted(set(want) - set(shown))
    return {
        "status": "ok",
        "n_required": len(want),
        "n_covered": len(want) - len(missing),
        "uncovered": missing,
        "demanded_by": {k: sorted(v) for k, v in sorted(shown.items())},
        "cannot_fail": {k: sorted(v) for k, v in sorted(wrong_order.items())},
        "note": (
            "`cannot_fail` lists capsules that carry the stage in a multi-command program without "
            "being able to show it leaking: the stage comes last, or the stimulus cannot go negative"
        ),
    }


def _group_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required group signatures no capsule in the corpus demands."""
    import yaml

    from merlin.development.targetgen.conformance.requirements import _capsule_paths
    from merlin.development.targetgen.corpus.derived_models import matches

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    want = {str(r["signature"]): r for r in (required or []) if r.get("signature")}
    demanded: dict[str, list[str]] = {}
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if cap.get("label") not in labels or name in exclude:
                continue
            op = cap.get("operation") or {}
            attrs = op.get("attributes") if isinstance(op.get("attributes"), dict) else {}
            inputs = cap.get("inputs") if isinstance(cap.get("inputs"), list) else []
            dtype = next((str(i.get("dtype")) for i in inputs if isinstance(i, dict) and i.get("dtype")), None)
            signature = _group_signature(str(op.get("op") or ""), attrs.get("epilogue") or (), dtype)
            if signature in want:
                demanded.setdefault(signature, []).append(name)
            for key, requirement in want.items():
                if requirement.get("column_readout") and matches(cap, requirement):
                    demanded.setdefault(key, []).append(name)
    missing = sorted(set(want) - set(demanded))
    return {
        "status": "ok",
        "n_required": len(want),
        "n_covered": len(want) - len(missing),
        "uncovered": missing,
        "uncovered_groups": sum(int(want[s].get("groups") or 0) for s in missing),
        "required_groups": sum(int(r.get("groups") or 0) for r in want.values()),
        "demanded_by": {k: sorted(v) for k, v in sorted(demanded.items())},
        "note": (
            "an uncovered signature is a group real models form on this target that no "
            "capsule certifies: the model route lowers it with nothing having tested it, or "
            "falls back without anything having required it"
        ),
    }


def _epilogue_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required epilogue STAGES no capsule in the corpus demands.

    A cell says which arithmetic the datapath runs; it cannot say which stage rides the contraction. So
    a corpus derived from cells alone can fuse one stage, pass, and report the whole fusion capability
    covered -- which is how a missing per-column bias and a missing standalone requantization both
    reached a bundle as omissions rather than as failures.

    TWO WAYS A CAPSULE DEMANDS A STAGE, and both count, because the defect presented in each shape. The
    ordinary one is a contraction naming the stage in ``operation.attributes.epilogue``. The other is a
    capsule that IS the stage standing alone -- a bare requant or bias member -- which is the form that
    catches a backend with no lowering for it at all, as opposed to one that cannot fuse it.

    ``fused_by``/``standalone_by`` are kept apart rather than summed. A stage evidenced ONLY standalone
    says the lowering exists and the fusion is untested, which is a different remedy from a stage
    evidenced only fused, and one number cannot carry both.
    """
    import yaml

    from merlin.development.targetgen.conformance.requirements import _capsule_paths

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    want = {str(r["stage"]) for r in (required or []) if r.get("stage")}

    fused: dict[str, list[str]] = {}
    standalone: dict[str, list[str]] = {}
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if name in exclude:
                continue
            op = cap.get("operation") or {}
            attrs = op.get("attributes") if isinstance(op.get("attributes"), dict) else {}
            for stage in (attrs or {}).get("epilogue") or ():
                if str(stage) in want:
                    fused.setdefault(str(stage), []).append(name)
            # The stage as the whole operation. Compared against the DECLARED stage names rather than
            # against a spelling table, so a target that names its stages differently is measured by its
            # own vocabulary instead of by ours.
            opname = str(op.get("op") or "")
            for stage in want:
                if opname == stage or opname.endswith(f"_{stage}") or opname == f"{stage}_add":
                    standalone.setdefault(stage, []).append(name)

    proven = set(fused) | set(standalone)
    return {
        "status": "ok",
        "n_required": len(want),
        "n_covered": len(want & proven),
        "uncovered": sorted(want - proven),
        "fused_by": {k: sorted(v) for k, v in sorted(fused.items())},
        "standalone_by": {k: sorted(v) for k, v in sorted(standalone.items())},
        "fused_only": sorted(set(fused) - set(standalone)),
        "standalone_only": sorted(set(standalone) - set(fused)),
        "note": (
            "a required epilogue stage no capsule demands means a backend that cannot emit it "
            "fails nothing here; a stage evidenced only standalone means its lowering is tested "
            "and its FUSION is not"
        ),
    }


def _accumulator_output_gap(bound: dict, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Measure the output boundary from actual capsule operand shapes."""
    import yaml

    from merlin.development.targetgen.conformance.requirements import _capsule_paths

    status = bound.get("status")
    if status != "resolved":
        return {
            "status": status or "not_measured",
            "detail": bound.get("reason") or "regenerate the conformance requirement",
            "n_required": 0,
            "n_covered": 0,
            "uncovered": [],
        }
    tile = int(bound["tile_edge"])
    capacity = int(bound["capacity_rows"])
    selected = set(labels or {"public"})
    omitted = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    covered_by = []
    for root in roots:
        for path in _capsule_paths(root):
            try:
                cap = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(cap.get("name") or path.parent.name)
            if name in omitted or cap.get("label") not in selected:
                continue
            op = cap.get("operation") or {}
            if op.get("op") not in {"matmul", "linear", "fused_matmul_bias"}:
                continue
            inputs = {str(t.get("name")): t for t in (cap.get("inputs") or ()) if isinstance(t, dict)}
            attrs = op.get("attributes") or {}
            lhs = inputs.get(str(attrs.get("lhs"))) or {}
            weight = inputs.get(str(attrs.get("weight"))) or {}
            lhs_shape, weight_shape = lhs.get("shape") or (), weight.get("shape") or ()
            if len(lhs_shape) != 2 or len(weight_shape) != 2:
                continue
            try:
                m, k = map(int, lhs_shape)
                wk, n = map(int, weight_shape)
            except (TypeError, ValueError):
                continue
            if k != wk or min(m, k, n) <= 0:
                continue
            resident_rows = ((m + tile - 1) // tile) * ((n + tile - 1) // tile) * tile
            if resident_rows > capacity:
                covered_by.append(name)
    return {
        "status": "ok",
        "n_required": 1,
        "n_covered": int(bool(covered_by)),
        "uncovered": [] if covered_by else ["accumulator_output_capacity"],
        "capacity_rows": capacity,
        "covered_by": sorted(set(covered_by)),
    }


def uncovered(spec_doc: dict, corpus_roots, *, labels=None, tile_dim: int | None = None, exclude=None) -> dict:
    """Which required cells the corpus does NOT cover — the gate's question.

    Coverage is measured with :func:`contract.materialize.cert_capsule_cover`, the same function the cert
    tier uses to pick a representative subset, so "covered" means exactly what it means everywhere else.
    """
    from merlin.development.targetgen.conformance.axes import _geometry_gap
    from merlin.development.targetgen.corpus.materialization import cert_capsule_cover

    got = cert_capsule_cover(corpus_roots, labels=labels, tile_dim=tile_dim, exclude=exclude)
    have = set(got.get("cells") or ())
    want = [c["cell"] for c in (spec_doc.get("cells") or ())]
    missing = sorted(set(want) - have)
    out = {
        "n_required": len(want),
        "n_covered": len(set(want) & have),
        "uncovered": missing,
        "corpus_cells": sorted(have),
        "extra_cells": sorted(have - set(want)),
        "note": (
            "a required cell with no capsule means the corpus cannot evidence a family/dtype/"
            "alignment the hardware admits and a real target-model uses"
        ),
    }
    app_demands = spec_doc.get("application_demands")
    out["application_demands"] = (
        {
            "status": app_demands.get("status"),
            "coverage_status": app_demands.get("coverage_status", "unverified"),
            "n_operations": app_demands.get("n_operations", 0),
            "n_signatures": app_demands.get("n_signatures", 0),
            "n_operation_groups": len(app_demands.get("operation_groups") or ()),
            "full_inventory_sha256": app_demands.get("full_inventory_sha256"),
            "sidecar": app_demands.get("sidecar"),
            "applications": {
                label: {
                    "n_operations": row.get("n_operations"),
                    "counts": row.get("counts"),
                    "status": row.get("status"),
                }
                for label, row in sorted((app_demands.get("applications") or {}).items())
            },
            "note": "cell coverage does not establish per-operation application or whole-model coverage",
        }
        if isinstance(app_demands, dict)
        else {
            "status": "not_measured",
            "coverage_status": "unverified",
            "note": "this spec predates application operation inventory; cell coverage cannot stand in for it",
        }
    )
    # THE COMPOSITION AXIS, measured on the same corpus and reported beside the cells rather than folded
    # into them. A spec written before this axis existed carries no `composition` block; that is reported
    # as "not measured", never as "nothing required" -- an axis a stale spec cannot express must not read
    # as an axis with no gaps.
    comp_req = (spec_doc.get("composition") or {}).get("required")
    if comp_req is None:
        out["composition"] = {
            "status": "not_measured",
            "detail": "this spec predates the composition axis; regenerate it with --write to derive the requirement",
        }
    else:
        from merlin.development.targetgen.grading import boundary as BD

        corpus = BD.corpus_boundaries(corpus_roots, str(spec_doc.get("target") or ""), labels=labels, exclude=exclude)
        gap = BD.uncovered_boundaries({"by_kind": comp_req}, corpus)
        gap["status"] = "ok"
        gap["covered_by"] = corpus["by_kind"]
        out["composition"] = gap

    # THE GEOMETRY AXIS, measured by classifying each capsule's own contraction extents with the SAME
    # taxonomy the requirement is written in. A stale spec carrying no block reads "not measured",
    # never "no geometry is required".
    geom_req = (spec_doc.get("shape_geometry") or {}).get("required")
    if geom_req is None:
        out["shape_geometry"] = {
            "status": "not_measured",
            "detail": "this spec predates the geometry axis; regenerate it with --write to derive the requirement",
        }
    else:
        out["shape_geometry"] = _geometry_gap(geom_req, corpus_roots, labels=labels, exclude=exclude)

    out["accumulator_output_boundary"] = _accumulator_output_gap(
        spec_doc.get("accumulator_output_boundary") or {}, corpus_roots, labels=labels, exclude=exclude
    )

    scope_req = (spec_doc.get("scope") or {}).get("required")
    if scope_req is None:
        out["scope"] = {
            "status": "not_measured",
            "reason": "the spec carries no scope axis, so adjacency coverage was not asked",
        }
    else:
        out["scope"] = _scope_gap(scope_req, corpus_roots, labels=labels, exclude=exclude)

    # THE NEGATIVE LANE, measured the same way and reported beside the others. A family the hardware
    # does not admit must be shown landing on the HOST -- and shown by a capsule built to prove it, not
    # by one that merely happens to contain a host stretch. `corpus_boundaries` credits `H` to any
    # capsule containing one, so without the family clause below `H` is trivially covered by every
    # routing-shaped capsule and means nothing as a requirement.
    from merlin.development.targetgen.grading import boundary as BD

    out["host_only"] = BD.host_only_coverage(spec_doc, corpus_roots, labels=labels, exclude=exclude)

    # THE HOST-LANE AXIS. The cell vocabulary is `admitted INTERSECT observed`, so by construction it
    # can never contain a pair the hardware does not admit -- and the work in such a pair is not
    # out-of-scope, it is work the compiler must place on the HOST. The spec derives that lane pair by
    # pair and, until now, nothing read it back: 12 obligations on one target, written into every spec
    # and checked by nobody. A requirement that is derived, recorded and never measured is
    # indistinguishable from one that is met.
    from merlin.development.targetgen.grading import boundary as BD

    out["host_lane"] = BD.host_lane_coverage(spec_doc, corpus_roots, labels=labels, exclude=exclude)

    # THE EPILOGUE AXIS. A cell says WHICH arithmetic rides the datapath and cannot say which stage
    # rides the contraction, so a corpus derived from cells alone fuses one stage and reports the whole
    # fusion capability covered. Measured the way the corpus declares it -- a capsule either names the
    # stage in `operation.attributes.epilogue` or IS that stage standing alone -- because a standalone
    # requant member is how the missing-lowering defect actually presented.
    epi_req = (spec_doc.get("epilogue") or {}).get("required")
    if epi_req is None:
        out["epilogue"] = {
            "status": "not_measured",
            "detail": "this spec predates the epilogue axis; regenerate it with --write to derive the requirement",
        }
    else:
        out["epilogue"] = _epilogue_gap(epi_req, corpus_roots, labels=labels, exclude=exclude)

    # THE GROUP AXIS: the stage COMBINATIONS real models form, which the per-stage axis cannot ask for.
    group_req = (spec_doc.get("groups") or {}).get("required")
    if group_req is None:
        out["groups"] = {
            "status": "not_measured",
            "detail": "this spec predates the group axis; regenerate it with --write to derive the requirement",
        }
    else:
        out["groups"] = _group_gap(group_req, corpus_roots, labels=labels, exclude=exclude)

    carried_req = (spec_doc.get("carried_state") or {}).get("required")
    if carried_req is None:
        out["carried_state"] = {
            "status": "not_measured",
            "detail": "this spec predates the carried-state axis; regenerate it with --write to derive the requirement",
        }
    else:
        out["carried_state"] = _carried_state_gap(carried_req, corpus_roots, labels=labels, exclude=exclude)

    # THE CONVOLUTION-WINDOW AXIS, measured against what the corpus's capsules DECLARE. A capsule
    # states its window in `operation.attributes` (kh/kw, stride, padding, dilation); it is compared to
    # the required signature through the same `ConvGeometry` used to derive it, so a capsule and an
    # obligation cannot be written in two different vocabularies and read as agreeing.
    conv_req = (spec_doc.get("conv_geometry") or {}).get("required")
    if conv_req is None:
        out["conv_geometry"] = {
            "status": "not_measured",
            "detail": "this spec predates the convolution-window axis; regenerate "
            "it with --write to derive the requirement",
        }
    else:
        out["conv_geometry"] = _conv_geometry_gap(conv_req, corpus_roots, labels=labels, exclude=exclude)

    mem_req = (spec_doc.get("memory_mapping") or {}).get("required")
    if mem_req is None:
        out["memory_mapping"] = {
            "status": "not_measured",
            "detail": "this spec predates the memory-mapping axis; regenerate it "
            "with --write to derive the requirement",
        }
    else:
        from merlin.development.targetgen.policies import memory_regime as MR

        mem_corpus = MR.corpus_regimes(corpus_roots, str(spec_doc.get("target") or ""), labels=labels, exclude=exclude)
        if mem_req and mem_corpus["capacity_rows"] is None:
            # No capsule could be placed in ANY regime because the operand store itself did not resolve
            # (no RTL facts for the target on this host). That measures nothing; reporting every required
            # regime as uncovered would blame the corpus for the host. Same shape as the phase-0 reader.
            out["memory_mapping"] = {
                "status": "not_measured",
                "detail": "the target's operand-store capacity did not resolve here (no RTL facts), so no "
                "capsule's regime could be measured",
                "required": mem_req,
                "n_required": len([r for r in mem_req if r != MR.UNKNOWN]),
                "n_covered": 0,
                "uncovered": [],
            }
        else:
            mgap = MR.uncovered_regimes({"by_regime": mem_req}, mem_corpus)
            mgap["status"] = "ok"
            mgap["covered_by"] = mem_corpus["by_regime"]
            mgap["region_counts"] = (spec_doc.get("memory_mapping") or {}).get("region_counts") or {}
            out["memory_mapping"] = mgap
    return out
