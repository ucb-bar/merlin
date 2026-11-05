"""Let the shape panel choose between registered library pipelines instead of hand-written if cutoffs.

    python library_pipeline_policy.py [--facts rtl_facts/facts.json] [--library DIR]
        [--panel VARIANT=FILE ...] [--pins pins.json] [--holdout CLASS,...] [--execute M,K,N ...]
        [--adapter tools/policy_owner.py]

One dense family, three registered PipelineVariants built from library passes on one shared upstream,
each realized through the checked entry ``dense_ready_readout.schedule_dense_pipeline``:

* ``compact``   = dense_capacity_view -> compact_dense                          (fallback, unconditional)
* ``retained``  = dense_capacity_view -> retained_panel_pipeline -> dense_array_retention
* ``segmented`` = dense_capacity_view -> segmented_block -> segmented_readout_overlap

Each non-fallback variant carries its LEGALITY predicate (a capacity-legal block exists for its premise,
as a ``ResourcePredicate`` over facts derived from the RTL capacities). The PERFORMANCE choice is
``merlin.transforms.shape_policy``: two ``ShapeScheduleClass`` guards over accumulator fit
(``accumulator_output_copies >= 1`` / ``accumulator_output_stores_needed >= 2``), each with the incumbent
``compact`` and a frozen ``ValidationPartition`` of the panel geometries (``--holdout`` classes held out).

Evidence: one ``agent_selfcheck.py --panel dense`` response per variant (built with that variant active),
passed as ``--panel VARIANT=FILE``. Each panel class carries a ``shape_policy`` record (geometry +
observation); the pins it leaves to you (``author_supplies``) come from ``--pins`` (JSON with abi, inputs,
reference, library, provider, roi, artifact_format, implementation and, if the panel lacks them, facts,
scope and basis); this script derives ``source_sha256`` and ``context_sha256`` from its own session.
Without ``--panel`` it runs a SYNTHETIC demo panel (labelled so; never evidence) that shows both outcomes:
one class switches, the other keeps its incumbent because one alternative lacks complete timing.

``assess_shape_policy`` then reports per class: selected vs incumbent and, per alternative, what is
missing (geometries, boundary cases, complete pairs, a single performance domain, or a regression).
Finally every panel shape plus ``--execute`` shapes run through ``AssessedShapePolicy.execute`` on a
fresh session: the selected path's passes really run, and the emitted kernel is executed on host
integers against the source product.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

FAMILY = "dense"
LEVEL = "dense_capacity_source"
AUTHOR_PINS = (
    "abi_sha256",
    "inputs_sha256",
    "reference_sha256",
    "library_sha256",
    "provider_sha256",
    "roi_sha256",
    "artifact_format",
    "implementation_sha256",
)


def _support():
    if "recipe_support" not in sys.modules:
        spec = importlib.util.spec_from_file_location("recipe_support", Path(__file__).with_name("recipe_support.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["recipe_support"] = module
        spec.loader.exec_module(module)
    return sys.modules["recipe_support"]


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class DenseSource:
    """The pipeline IR of this example: one GEMM's extents and declared operand shapes (immutable)."""

    m: int
    k: int
    n: int
    operand_shapes: tuple[tuple[int, ...], ...]


def session(source, caps):
    from merlin.transforms.pipeline_state import PipelineContext, PipelineSession

    def verify(module):
        if type(module) is not DenseSource:
            raise ValueError("dense source changed type")

    context = PipelineContext(
        hardware=_sha(caps["array"]),
        resources=_sha([caps["operand"], caps["accumulator"]]),
        library="selected-compiler-library",
    )
    return PipelineSession(
        source, context=context, ir_level=LEVEL, fingerprint=lambda m: _sha(asdict(m)), verify_ir=verify
    )


# ---- legality facts and the three pipelines ---------------------------------------------------------


def _segmented_blocks(p, s, r):
    from dataclasses import replace

    if p.tiles[1] < 2:
        return []
    window = s.accumulator.capacity // 2
    return [
        b
        for b in _support().blocks(p, s, replace(r, max_k_blocks=p.tiles[1] - 1))
        if b.k < p.tiles[1] and b.m * b.n * p.tile_m <= window and p.tiles[2] // b.n > 1
    ]


def _full_k_blocks(p, s, r):
    return [b for b in _support().blocks(p, s, r) if b.k == p.tiles[1]]


def facts_for(sess, caps):
    """PolicyFacts of the current source: shapes as declared, resources derived from the capacities."""
    from merlin.transforms.pipeline_policy import PolicyFacts

    S = _support()
    src = sess.module
    p, s, r = S.dense_view((src.m, src.k, src.n), caps)
    rows, store = S.output_rows(p), caps["accumulator"]["rows"]
    resources = (
        ("accumulator_output_copies", store // rows),
        ("accumulator_output_stores_needed", -(-rows // store)),
        ("full_k_panel_blocks", len(_full_k_blocks(p, s, r))),
        ("segmented_blocks", len(_segmented_blocks(p, s, r))),
    )
    return PolicyFacts(
        sess.stamp(), shapes=src.operand_shapes, operations=frozenset({"dense"}), resources=tuple(sorted(resources))
    )


def registry(caps):
    """Library passes as RegisteredPass entries over the DenseSource IR (analysis-only: IR never mutates).

    Every family goes through the one checked entry ``dense_ready_readout.schedule_dense_pipeline``;
    the passes before it choose the block and state the typed plan, the passes after it transform the
    checked result (array retention)."""
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.dense_array_retention import (
        DenseArrayRetentionAdmission,
        propose_dense_array_retention,
        verify_dense_array_retention,
    )
    from merlin.compile.scheduling.dense_panel_pipeline import (
        dense_retained_compute_order,
        dense_retained_operand_hoists,
    )
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec

    S = _support()
    ex = S.example()

    def entry(name, stage, requires, produces, action):
        return name, RegisteredPass(
            PassSpec(name, stage, requires=requires, produces=produces, mutates_ir=False, ir_level=LEVEL), action
        )

    def view(sess):
        src = sess.module
        p, s, r = S.dense_view((src.m, src.k, src.n), caps)
        return {"dense_view": (p, s, r, S.adapter())}

    def compact(sess):  # no plan: the ordinary compact planner
        p, s, r, a = sess.analysis("dense_view")
        result = schedule_dense_pipeline(
            p,
            r,
            s,
            a,
            name="compact",
            dtypes=S.dtypes(),
            static_budget=128,
            external_buffers_disjoint=True,
            timeout_ms=20_000,
        )
        return {"dense_program": (result.kernel, a, p)}

    def retained(sess):
        p, s, r, a = sess.analysis("dense_view")
        b = S.pick(_full_k_blocks(p, s, r), lambda b: True)
        law = ex._laws(a, b, segmented=False)
        raw = emit_compact_dense(p, b, s, a, name="retained", dtypes=S.dtypes(), outer_order=("m", "n"))
        transforms = dense_retained_operand_hoists(raw, a, role="lhs")
        transforms += dense_retained_compute_order(propose_loop_transforms(raw, transforms), a, order=("k", "n", "m"))
        plan = make_dense_schedule_plan(
            p,
            r,
            s,
            a,
            source_identity=S.SOURCE,
            dtypes=S.dtypes(),
            static_budget=128,
            block_tiles=(b.m, b.k, b.n),
            outer_order=("m", "n"),
            loop_transforms=transforms,
            pipeline_kind="retained_full_k",
            pipeline_prefetch=False,
        )
        result = schedule_dense_pipeline(
            p,
            r,
            s,
            a,
            name="retained",
            dtypes=S.dtypes(),
            pipeline_admission=law,
            schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),
            source_identity=S.SOURCE,
            external_buffers_disjoint=True,
            static_budget=128,
        )
        return {"dense_retained": (result, b, law)}

    def array_retention(sess):
        p, s, r, a = sess.analysis("dense_view")
        result, b, law = sess.analysis("dense_retained")
        array_law = DenseArrayRetentionAdmission(
            a.configuration_identity, a.numerical_policy_identity, "test-only-array-capture-stay-law"
        )
        proposal = propose_dense_array_retention(result.kernel, a, array_law)
        verify_dense_array_retention(proposal, p, b, r, s, a, law, static_budget=256)
        return {"dense_program": (proposal.kernel, a, p)}

    def segmented_block(sess):
        p, s, r, _ = sess.analysis("dense_view")
        return {"dense_segment_block": S.pick(_segmented_blocks(p, s, r), lambda b: True)}

    def readout_overlap(sess):
        p, s, r, a = sess.analysis("dense_view")
        block = sess.analysis("dense_segment_block")
        result, _ = S.segmented_pipeline(p, s, r, a, block, name="segmented", batch=True, readout_mode="disjoint")
        return {"dense_program": (result.kernel, a, p)}

    return dict(
        (
            entry("dense_capacity_view", 0, (), ("dense_view",), view),
            entry("compact_dense", 1, ("dense_view",), ("dense_program",), compact),
            entry("retained_panel_pipeline", 1, ("dense_view",), ("dense_retained",), retained),
            entry("dense_array_retention", 2, ("dense_view", "dense_retained"), ("dense_program",), array_retention),
            entry("segmented_block", 1, ("dense_view",), ("dense_segment_block",), segmented_block),
            entry(
                "segmented_readout_overlap",
                2,
                ("dense_view", "dense_segment_block"),
                ("dense_program",),
                readout_overlap,
            ),
        )
    )


def variants():
    from merlin.transforms.pipeline_policy import PipelineVariant, ResourcePredicate

    return (
        PipelineVariant("compact", ("dense_capacity_view", "compact_dense")),
        PipelineVariant(
            "retained",
            ("dense_capacity_view", "retained_panel_pipeline", "dense_array_retention"),
            (ResourcePredicate("full_k_panel_blocks", 1),),
        ),
        PipelineVariant(
            "segmented",
            ("dense_capacity_view", "segmented_block", "segmented_readout_overlap"),
            (ResourcePredicate("segmented_blocks", 1),),
        ),
    )


GUARDS = {
    "output_fits_accumulator": ("accumulator_output_copies", 1),
    "output_overflows_accumulator": ("accumulator_output_stores_needed", 2),
}


# ---- panel evidence -----------------------------------------------------------------------------------


def load_panel(path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    panel = doc.get("panel", doc) if isinstance(doc, dict) else None
    if not isinstance(panel, dict) or not isinstance(panel.get("classes"), dict):
        raise SystemExit(f"{path}: no shape panel (a self-check response with 'panel', or the panel block)")
    return panel


def demo_panels(caps) -> tuple[dict, dict]:
    """A synthetic panel per variant in the self-check record form, with both guard boundaries covered.
    Cycles are a toy table (NOT evidence): retained helps while the output fits the accumulator,
    segmented once it overflows; the thin case has no complete timing for ``retained``."""
    edge, rows = caps["array"][1], caps["accumulator"]["rows"]
    half = max(1, rows // (2 * edge))
    cases = {
        "square": (2 * edge, 2 * edge, 2 * edge),
        "wide": (2 * edge, 3 * edge, half * edge),
        "tall": ((half + 1) * edge, 3 * edge, 2 * edge),
        "thin": (1, 3 * edge, 2 * edge),
        "ragged": (2 * edge - 1, 3 * edge, (half + 1) * edge - 1),
        "model_layer": (2 * edge, 4 * edge, 2 * half * edge),
    }
    factor = {"compact": (1.0, 1.0), "retained": (0.85, 1.15), "segmented": (1.1, 0.8)}
    pins = {name: hashlib.sha256(f"demo:{name}".encode()).hexdigest() for name in AUTHOR_PINS}
    pins["artifact_format"] = "demo-program"
    panels = {}
    for variant, (fits, overflows) in factor.items():
        classes = {}
        for klass, (m, k, n) in cases.items():
            fit = m * n <= rows * edge  # rough label only; the policy reads the derived resources
            cycles = (
                None
                if (klass, variant) == ("thin", "retained")
                else round((m * k * n / edge + 400) * (fits if fit else overflows))
            )
            observation = {
                "program_sha256": _sha([variant, m, k, n]),
                "feedback_sha256": _sha([variant, klass]),
                "numerical_status": "pass",
                "timing_status": "complete" if cycles else "unknown",
                "complete_roi": bool(cycles),
                "cycles": cycles,
            }
            classes[klass] = {
                "case": f"demo-{klass}",
                "extents": {"m": m, "k": k, "n": n},
                "shape_policy": {
                    "geometry": {"family": FAMILY, "operand_shapes": [[m, k], [k, n]]},
                    "identity": {
                        "basis": "modeled_roi",
                        "facts_sha256": _sha(caps),
                        "scope_sha256": _sha("demo-scope"),
                    },
                    "author_supplies": list(AUTHOR_PINS),
                    "observation": observation,
                },
            }
        panels[variant] = {"family": FAMILY, "classes": classes, "evidence": "SYNTHETIC DEMO, not timing"}
    return panels, pins


def feedback(panels, pins, caps):
    """ShapeCaseFeedback per panel geometry, pairing the variants' observations of that geometry."""
    from merlin.transforms.evaluated_exploration import (
        CompletedObservation,
        ExplorationIdentity,
        exploration_context_identity,
    )
    from merlin.transforms.pass_exploration import ExplorationGeometry
    from merlin.transforms.shape_policy import ShapeCaseFeedback

    cases, skipped = {}, []
    for variant, panel in panels.items():
        for klass, entry in panel["classes"].items():
            record = entry.get("shape_policy")
            if not record:
                skipped.append((variant, klass, entry.get("verdict") or "no shape_policy record"))
                continue
            shapes = tuple(tuple(s) for s in record["geometry"]["operand_shapes"])
            extents = entry["extents"]
            source = DenseSource(extents["m"], extents["k"], extents["n"], shapes)
            sess = session(source, caps)
            identity = {
                **{k: v for k, v in record["identity"].items() if v is not None},
                **{k: v for k, v in pins.items() if k != "implementation_sha256"},
                "source_sha256": sess.stamp().source_sha256,
                "context_sha256": exploration_context_identity(sess.context),
            }
            observation = record["observation"]
            if observation.get("program_sha256") is None:
                skipped.append((variant, klass, "no program digest: not evidence"))
                continue
            row = cases.setdefault(shapes, {"class": klass, "source": source, "observations": []})
            row["observations"].append((variant, CompletedObservation(ExplorationIdentity(**identity), **observation)))
    out = []
    for shapes, row in cases.items():
        sess = session(row["source"], caps)
        geometry = ExplorationGeometry(FAMILY, shapes)
        out.append(
            (
                row["class"],
                ShapeCaseFeedback(
                    geometry, facts_for(sess, caps), tuple(row["observations"]), pins["implementation_sha256"]
                ),
            )
        )
    return out, skipped


def assess(cases, holdout):
    from merlin.transforms.pass_exploration import ValidationPartition
    from merlin.transforms.pipeline_policy import ResourcePredicate
    from merlin.transforms.shape_policy import ShapeScheduleClass, assess_shape_policy

    training = tuple(c.geometry for klass, c in cases if klass not in holdout)
    held = tuple(c.geometry for klass, c in cases if klass in holdout)
    groups = tuple(
        ShapeScheduleClass(
            name,
            (ResourcePredicate(resource, edge),),
            "compact",
            ValidationPartition(training, held) if training and held else None,
        )
        for name, (resource, edge) in GUARDS.items()
    )
    return assess_shape_policy(groups, variants(), tuple(c for _, c in cases), fallback="compact")


def reading(alt) -> str:
    """One line per alternative: why a class switched to it, or why it kept its incumbent."""
    if alt["supported"]:
        return "supported on the declared panel"
    if alt["missing_geometries"]:
        return f"missing panel geometries {len(alt['missing_geometries'])}: incumbent kept"
    if alt["missing_boundaries"]:
        return (
            "missing boundary cases (value, outside value): "
            + ", ".join(f"{edge}/{outside}" for _, _, edge, outside in alt["missing_boundaries"])
            + ": incumbent kept"
        )
    if alt["reasons"]:
        return "; ".join(sorted({reason for _, reason in alt["reasons"]})) + ": incumbent kept"
    if alt["mixed_performance_domains"]:
        return "observations from more than one performance domain: incumbent kept"
    ratios = alt["matched_cycle_ratios"]
    return f"regresses on a matched case (max ratio {max(ratios):.3f})" if ratios else "no matched case"


def summary(policy) -> list[dict]:
    rows = []
    for group in policy.report["classes"]:
        rows.append(
            {
                "class": group["shape_class"],
                "status": group["status"],
                "incumbent": group["incumbent"],
                "selected": group["selected"],
                "alternatives": [
                    {
                        "alternative": a["alternative"],
                        "supported": a["supported"],
                        "training_ratio": None
                        if a["training_geometric_mean_ratio"] is None
                        else round(a["training_geometric_mean_ratio"], 3),
                        "reading": reading(a),
                    }
                    for a in group["alternatives"]
                ],
            }
        )
    return rows


def execute(policy, caps, shape, domain) -> dict:
    """Run the selected path's real passes for one shape and check the emitted kernel on host integers."""
    S = _support()
    m, k, n = shape
    sess = session(DenseSource(m, k, n, ((m, k), (k, n))), caps)
    facts = facts_for(sess, caps)
    result = policy.execute(sess, registry(caps), facts, output="dense_program", domain=domain)
    kernel, a, p = result.artifact
    return {
        "shape": shape,
        "resources": dict(facts.resources),
        "variant": result.receipt["selected_variant"],
        "decisions": [(d["variant"], d["status"]) for d in result.receipt["decisions"]],
        "calls": S.execute_dense(kernel, a, p),
    }


def unseen_shapes(caps) -> list[tuple[int, int, int]]:
    """Shapes no panel measured: one overflowing the accumulator (the class guard decides), and one that
    overflows with a single K tile, where the selected path's LEGALITY predicate fails and the
    incumbent compiles instead."""
    edge, rows = caps["array"][1], caps["accumulator"]["rows"]
    half = max(1, rows // (2 * edge))
    return [(3 * edge - 1, 5 * edge, (half + 2) * edge + 1), (2 * edge, edge, (half + 1) * edge)]


def target_example(owner):
    """Use an authored target's original sessions, emitter and cycle owner.

    ``prepare`` returns training, alternatives, design, incumbent and optional
    axes/features. Only training observations reach proposal selection. Optional
    ``boundaries(requests)`` returns source PolicyFacts before ``evaluate`` sees
    the frozen requests. ``evaluate`` returns genuine ShapeCaseFeedback for the
    requested heldout/boundary sources, using the ordinary numerical/timing owner.
    ``binding(geometry)`` returns a fresh session, registry, facts and output name.
    Its terminal registered pass must return EmittedProgram at the exact boundary
    named by the observation. No neutral registry or global identity pins apply.
    """
    from merlin.transforms.pass_exploration import EmittedProgram
    from merlin.transforms.pipeline_policy import PipelinePolicy, execute_policy
    from merlin.transforms.shape_policy import PerformanceDomain, propose_shape_policies

    if any(not callable(getattr(owner, name, None))
           for name in ("prepare", "evaluate", "binding")):
        raise ValueError("target adapter requires prepare, evaluate and binding hooks")
    setup = owner.prepare()
    training = setup["training"]
    paths = setup["alternatives"]
    incumbent = setup["incumbent"]
    proposals = propose_shape_policies(training, paths, design=setup["design"], incumbent=incumbent,
        axes=setup.get("axes", ()), features=setup.get("features", ()))
    if not proposals:
        return dict(evidence="target owner feedback", proposals=[], executions=[],
                    status="no_training_supported_proposal", performance_qualification=False)
    # The first TRAINING-ranked proposal is frozen before any heldout callback.
    proposal = proposals[0]
    requests = proposal.requests(training)
    if requests["missing_boundary_sources"] and callable(getattr(owner, "boundaries", None)):
        facts = tuple(owner.boundaries(requests))
        from merlin.transforms.pass_exploration import ExplorationGeometry
        proposal = proposal.freeze_boundaries(tuple(ExplorationGeometry(
            training[0].geometry.family, f.shapes) for f in facts), facts=facts)
        requests = proposal.requests(training)
    extra = tuple(owner.evaluate(requests))
    cases = (*training, *extra)
    policy = proposal.assess(cases)
    fallback = next(p for p in paths if p.name == incumbent)
    executions = []
    for case in cases:
        case.__post_init__()
        observed = dict(case.observations)
        domain = PerformanceDomain.from_observation(next(iter(observed.values())).identity,
            implementation_sha256=case.implementation_sha256)
        row = dict(geometry_sha256=case.geometry.identity)
        artifacts = []
        for label in ("incumbent", "selected"):
            bound = owner.binding(case.geometry)
            if bound["facts"] != case.facts:
                raise ValueError("target binding differs from original observed source facts")
            args = (bound["session"], bound["registry"], bound["facts"])
            result = (execute_policy(args[0], PipelinePolicy((fallback,), incumbent),
                      args[1], args[2], output=bound["output"]) if label == "incumbent" else
                      policy.execute(*args, output=bound["output"], domain=domain))
            program = result.artifact
            if type(program) is not EmittedProgram:
                raise ValueError("target terminal pass must return its actual EmittedProgram")
            program.__post_init__()
            digest = hashlib.sha256(program.data).hexdigest()
            variant = result.receipt["selected_variant"]
            original = observed.get(variant)
            if (original is None or original.program_sha256 != digest
                    or original.identity.artifact_format != program.format):
                raise ValueError("current target emission differs from the original observed program")
            row[label] = dict(variant=variant, program_sha256=digest,
                              feedback_sha256=original.feedback_sha256,
                              cycles=original.cycles, eligible=original.eligible,
                              passes=[p["name"] for p in result.receipt["passes"]])
            artifacts.append(program.data)
        row["changed_emission"] = artifacts[0] != artifacts[1]
        executions.append(row)
    return dict(evidence="target owner feedback; original emitted bytes rechecked",
                proposal=proposal.report, requests=proposal.requests(cases), policy=policy.report,
                executions=executions, performance_qualification=False,
                qualification="declared panel only; target owner retains numerical/timing authority")


def example(facts=None, panel_files=None, pins_file=None, holdout=("model_layer", "thin"), extra=()) -> dict:
    from merlin.transforms.shape_policy import PerformanceDomain

    S = _support()
    caps = S.capacities(facts)
    if panel_files:
        panels = {name: load_panel(path) for name, path in panel_files.items()}
        pins = json.loads(Path(pins_file).read_text()) if pins_file else {}
        evidence = "panel files"
    else:
        panels, pins = demo_panels(caps)
        evidence = "SYNTHETIC DEMO panel (not timing evidence); pass --panel VARIANT=FILE"
    missing = [p for p in AUTHOR_PINS if p not in pins]
    if missing:
        raise SystemExit(f"--pins lacks {missing}: the panel leaves these to your own pipeline session")
    cases, skipped = feedback(panels, pins, caps)
    policy = assess(cases, set(holdout))
    first = next(o for _, c in cases for _, o in c.observations).identity
    domain = PerformanceDomain.from_observation(first, implementation_sha256=pins["implementation_sha256"])
    shapes = [tuple(c.facts.shapes[0]) + (c.facts.shapes[1][1],) for _, c in cases]
    shapes += [tuple(s) for s in extra] or unseen_shapes(caps)
    runs = [execute(policy, caps, shape, domain) for shape in shapes]
    return {
        "facts": caps["source"],
        "evidence": evidence,
        "variants": [v.name for v in variants()],
        "holdout": sorted(holdout),
        "skipped_records": skipped,
        "policy": summary(policy),
        "executions": runs,
        "policy_sha256": policy.report["policy_sha256"],
        "verified": "every executed path reproduced the source product from its emitted kernel",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="See the heavy-pass-recipes skill, reference/pipeline-policy.md."
    )
    ap.add_argument("--facts", default=None, help="RTL facts JSON (default: rtl_facts/facts.json if present)")
    ap.add_argument(
        "--panel",
        action="append",
        default=[],
        metavar="VARIANT=FILE",
        help="a --panel dense response built with VARIANT active (repeat per variant)",
    )
    ap.add_argument("--pins", default=None, help="JSON with the identity pins the panel leaves to you")
    ap.add_argument("--holdout", default="model_layer,thin", help="panel classes held out of training")
    ap.add_argument(
        "--execute",
        action="append",
        default=[],
        metavar="M,K,N",
        help="extra shapes to compile (default: two unseen shapes derived from the capacities)",
    )
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    ap.add_argument("--adapter", help="authored target owner with prepare/evaluate/binding hooks")
    a = ap.parse_args(argv)
    _support().use_library(a.library)
    if a.adapter:
        if a.panel or a.pins or a.execute:
            ap.error("--adapter uses its original owner evidence, not neutral demo panels/pins")
        spec = importlib.util.spec_from_file_location("authored_policy_owner", Path(a.adapter))
        owner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = owner
        spec.loader.exec_module(owner)
        print(json.dumps(target_example(owner), indent=1, default=str))
        return 0
    files = {}
    for item in a.panel:
        name, sep, path = item.partition("=")
        if not sep or name not in ("compact", "retained", "segmented"):
            ap.error("--panel takes VARIANT=FILE with VARIANT one of compact, retained, segmented")
        files[name] = path
    extra = [tuple(int(x) for x in s.split(",")) for s in a.execute]
    if any(len(s) != 3 for s in extra):
        ap.error("--execute takes M,K,N")
    out = example(a.facts, files, a.pins, tuple(x for x in a.holdout.split(",") if x), extra)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
