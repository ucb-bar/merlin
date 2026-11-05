"""Executable, library-first reconstruction of reference scheduling mechanisms.

Backward requirements describe a search hypothesis, never proof that a pass is
missing. Forward paths use the ordinary pass manager and exact emitted artifacts.
Adapters keep target selection, source-region equivalence and execution ownership
explicit. No reference identity is a production applicability predicate.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType

from merlin.transforms.pass_exploration import (
    EmittedProgram,
    ExplorationRefused,
    ExploredCandidate,
    PassFamily,
    explore_candidates,
    generate_candidates,
)
from merlin.transforms.pipeline_policy import RegisteredPass
from merlin.transforms.pipeline_state import PassSpec


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _names(values, label):
    if (type(values) is not tuple or len(values) > 128
            or any(type(v) is not str or not v or len(v) > 256 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(label + " requires bounded unique names")


@dataclass(frozen=True)
class TransformationBinding:
    """Executable library family plus conservative backward mechanism contract.

    ``requires`` and ``establishes`` are discovery hints. Only the real PassSpec,
    original checkers and post-emission observation can admit a realization.
    Public compiler primitives remain independent of this reference workflow.
    """
    family: PassFamily
    requires: tuple[str, ...] = ()
    establishes: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()
    emission_consumers: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.family) is not PassFamily:
            raise ValueError("binding must wrap a callable existing PassFamily")
        for name in ("requires", "establishes", "verification", "emission_consumers"):
            _names(getattr(self, name), name)

    def summary(self):
        try:
            implementation = hashlib.sha256(inspect.getsource(self.family.bind).encode()).hexdigest()
        except (OSError, TypeError):
            implementation = None
        return dict(factory_source_sha256=implementation, family=self.family.name, knobs=self.family.knobs,
                    optional=self.family.optional, requires=self.requires,
                    establishes=self.establishes, verification=self.verification,
                    emission_consumers=self.emission_consumers)


@dataclass(frozen=True)
class MechanismObservation:
    """Mechanisms independently observed in the actual emitted candidate.

    Witnesses must identify the actual representation/check and its result.
    A declaration that a binding can establish a property is not an observation.
    Qualification for numerical execution or timing belongs to the evaluator.
    """
    established: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    witnesses: Mapping = field(default_factory=dict)
    cca: object = None

    def __post_init__(self):
        _names(self.established, "observed mechanisms")
        _names(self.unknown, "unknown mechanisms")
        if (set(self.established) & set(self.unknown) or not isinstance(self.witnesses, Mapping)
                or set(self.established) - self.witnesses.keys()):
            raise ValueError("observed mechanisms require disjoint unknowns and actual witnesses")
        # Freeze a JSON copy so later callback mutations cannot rewrite evidence.
        copied = json.loads(json.dumps(dict(self.witnesses), sort_keys=True))
        if any(copied[name] in (None, {}, [], "") for name in self.established):
            raise ValueError("an established mechanism requires a nonempty actual witness")
        object.__setattr__(self, "witnesses", MappingProxyType(copied))
        if self.cca is not None:
            from .cca import CCA
            if type(self.cca) is not CCA or self.cca.scope_problems():
                raise ValueError("actual CCA observation must use the existing valid facet vocabulary")
            object.__setattr__(self, "cca", deepcopy(self.cca))

    def summary(self):
        return dict(established=self.established, unknown=self.unknown, witnesses=dict(self.witnesses),
                    cca=None if self.cca is None else self.cca.to_dict())


@dataclass(frozen=True)
class ReconstructionAdapter:
    """Caller-selected frontend/library/target boundary for automated exploration.

    Source-level fusion and layout passes can precede schedule construction in
    ``bindings``. Each pass remains at its declared IR level; native lowerings
    return verified products instead of silently changing a session's IR level.
    ``emission`` serializes the actual final product. ``observe`` must read that
    product, never infer success from requested decisions or reference names.
    """
    source: object
    bindings: tuple[TransformationBinding, ...]
    facts: Callable
    emission: Callable
    observe: Callable
    output: str = "program"
    before: tuple[tuple[str, str], ...] = ()
    session_options: Mapping = field(default_factory=dict)
    feedback: Callable | None = None
    snapshot: Callable | None = None

    def __post_init__(self):
        if (type(self.bindings) is not tuple or not 1 <= len(self.bindings) <= 16
                or any(type(b) is not TransformationBinding for b in self.bindings)
                or len({b.family.name for b in self.bindings}) != len(self.bindings)
                or any(not callable(f) for f in (self.facts, self.emission, self.observe))
                or self.feedback is not None and not callable(self.feedback)
                or self.snapshot is not None and not callable(self.snapshot)
                or not isinstance(self.session_options, Mapping)
                or type(self.output) is not str or not self.output):
            raise ValueError("reconstruction requires executable bindings and explicit observation adapters")
        object.__setattr__(self, "session_options", MappingProxyType(dict(self.session_options)))


@dataclass(frozen=True)
class ReconstructionGap:
    kind: str
    goals: tuple[str, ...]
    reason: str
    next_action: str
    witnesses: tuple[dict, ...] = ()

    def summary(self):
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionPath:
    candidate: ExploredCandidate
    intermediates: tuple[dict, ...]
    observation: MechanismObservation
    goals: tuple[str, ...]
    library_sha256: str
    reference_cca: object = None

    @property
    def satisfied(self):
        return (self.candidate.emission is not None
                and self.candidate.evaluation.status == "compiled"
                and not set(self.goals) - set(self.observation.established))

    def summary(self):
        candidate = self.candidate
        result = candidate.evaluation.result
        receipt = result.receipt if result is not None else None
        summary = candidate.summary()
        comparison = None
        if self.reference_cca is not None and self.observation.cca is not None:
            from .cca_compare import compare, uncomparable_axes
            divergences = compare(self.reference_cca, self.observation.cca)
            comparison = dict(divergences=tuple(asdict(row) for row in divergences),
                              unknown_axes=uncomparable_axes(self.reference_cca, self.observation.cca),
                              purpose="explain structural differences; not a performance objective")
        summary.update(satisfied=self.satisfied, observation=self.observation.summary(), cca_comparison=comparison,
                       intermediates=self.intermediates, receipt=receipt,
                       replay=dict(bindings=candidate.bindings,
                                   source_sha256=receipt["source"]["source_sha256"] if receipt else None,
                                   context=receipt["source"]["context"] if receipt else None,
                                   library_sha256=self.library_sha256,
                                   program_sha256=summary["program_sha256"],
                                   intermediate_sha256=tuple(row["sha256"] for row in self.intermediates)))
        return summary


@dataclass(frozen=True)
class ReconstructionResult:
    goals: tuple[str, ...]
    backward_requirements: tuple[dict, ...]
    paths: tuple[ReconstructionPath, ...]
    gaps: tuple[ReconstructionGap, ...]
    search: Mapping
    library: tuple[dict, ...]
    reference_cca: object = None
    goal_relations: tuple[dict, ...] = ()

    def summary(self):
        return dict(schema_version="merlin.kernel_reconstruction.v1", goals=self.goals,
                    backward_requirements=self.backward_requirements,
                    candidates=tuple(p.summary() for p in self.paths),
                    gaps=tuple(g.summary() for g in self.gaps), search=dict(self.search),
                    library=self.library, goal_relations=self.goal_relations,
                    reference_cca=None if self.reference_cca is None else self.reference_cca.to_dict(),
                    qualification=("reconstruction only; numerical execution and complete-region cycles "
                                   "require evaluation"))


def backward_requirements(bindings, goals):
    """Find possible providers and recursively expose their mechanism inputs.

    Alternative providers remain alternatives; requirements are not conflated
    into a single proof obligation. Cycles terminate with explicit graph edges.
    """
    _names(goals, "reconstruction goals")
    pending, seen, rows = list(goals), set(), []
    while pending:
        goal = pending.pop(0)
        if goal in seen:
            continue
        seen.add(goal)
        providers = tuple(b for b in bindings if goal in b.establishes)
        rows.append(dict(property=goal, providers=tuple(dict(family=b.family.name,
                    requires=b.requires, verification=b.verification,
                    emission_consumers=b.emission_consumers) for b in providers),
                    status="declared_provider" if providers else "unbound_requirement",
                    meaning="discovery hypothesis; actual forward check required"))
        pending.extend(r for b in providers for r in b.requires if r not in seen)
    return tuple(rows)


def _snapshot(name, value):
    """Retain actual portable IR where supported, otherwise no invented hash."""
    from xdsl.ir import Operation
    if isinstance(value, Operation):
        from merlin.transforms.certificates import ir_text
        return "xdsl", ir_text(value)
    if hasattr(value, "kernel") and callable(getattr(value.kernel, "text", None)):
        return "merlin.counted_commands.v1", value.kernel.text()
    if callable(getattr(value, "text", None)):
        return "text", value.text()
    try:
        return "json", json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _instrument(adapter):
    # Hold sessions until all variants finish: object IDs cannot be recycled.
    sessions, by_module = {}, {}
    families = []
    for binding in adapter.bindings:
        family = binding.family
        def bind(knobs, family=family):
            implementation = family.bind(knobs)
            if type(implementation) is not RegisteredPass:
                raise ValueError("library binding must return its real RegisteredPass")
            def run(session):
                records = sessions.setdefault(session, [])
                by_module[id(session.module)] = records
                products = implementation.action(session)
                by_module[id(session.module)] = records
                for name, value in products.items():
                    captured = (adapter.snapshot or _snapshot)(name, value)
                    if captured is None:
                        continue
                    format_name, content = captured
                    if type(format_name) is not str or type(content) is not str:
                        raise ValueError("intermediate adapter must return a format and actual text")
                    records.append(dict(pass_name=implementation.spec.name, product=name,
                                        format=format_name, text=content,
                                        sha256=hashlib.sha256(content.encode()).hexdigest()))
                return products
            return RegisteredPass(implementation.spec, run)
        families.append(replace(family, bind=bind))
    return tuple(families), sessions, by_module


def reconstruct(adapter, goals=(), *, reference_cca=None, max_candidates=8, max_search_steps=4096):
    """Run bounded library composition, emitting replayable paths and gaps.

    Missing provider declarations and finite exhaustion are actionable discovery
    tasks. Neither establishes that a transformation or dialect is missing.
    Infrastructure/observer errors propagate instead of becoming learning gaps.
    """
    if type(adapter) is not ReconstructionAdapter:
        raise ValueError("select an explicit executable reconstruction adapter")
    _names(goals, "reconstruction goals")
    reference_cca = deepcopy(reference_cca)
    goal_plan = goals_from_cca(reference_cca) if reference_cca is not None else {"goals": (), "relations": ()}
    goals = tuple(dict.fromkeys((*goals, *goal_plan["goals"])))
    _names(goals, "combined reconstruction goals")
    library = tuple(b.summary() for b in adapter.bindings)
    library_sha256 = _digest(library)
    backward = backward_requirements(adapter.bindings, goals)
    families, retained_sessions, intermediates = _instrument(adapter)
    try:
        space = generate_candidates(families, output=adapter.output, before=adapter.before,
                                    max_candidates=max_candidates, max_search_steps=max_search_steps)
    except ExplorationRefused as exc:
        gap = ReconstructionGap("bounded_search_exhaustion", goals, str(exc),
                                "repair_library_order_or_prerequisites_or_expand_bounded_search",
                                (exc.summary(),))
        return ReconstructionResult(goals, backward, (), (gap,),
            dict(steps=None, truncated=None, max_candidates=max_candidates,
                 max_search_steps=max_search_steps), library, reference_cca, goal_plan["relations"])
    paths, gaps = [], []
    for candidate in explore_candidates(adapter.source, space, adapter.facts, output=adapter.output,
            emission=adapter.emission, feedback=adapter.feedback, **adapter.session_options):
        result = candidate.evaluation.result
        observation = (adapter.observe(candidate) if candidate.emission is not None
                       else MechanismObservation(unknown=goals))
        if type(observation) is not MechanismObservation:
            raise ValueError("post-emission observer must return actual MechanismObservation")
        snapshots = tuple(intermediates.get(id(result.module), ())) if result is not None else ()
        path = ReconstructionPath(candidate, snapshots, observation, goals, library_sha256, reference_cca)
        paths.append(path)
        outcome = candidate.outcome()
        if outcome["status"] in ("legality_refused", "legality_unknown"):
            gaps.append(ReconstructionGap("legality_or_capability_restriction", goals,
                outcome["status"], outcome["next_action"], (outcome,)))
        elif candidate.emission is None:
            kind = "adapter_gap" if outcome["layer"] in ("adapter", "emission") else "forward_execution_failure"
            gaps.append(ReconstructionGap(kind, goals, outcome["reason"] or outcome["status"],
                outcome["next_action"], (outcome,)))
    if not any(path.satisfied for path in paths):
        unmet = tuple(g for g in goals if not any(g in p.observation.established for p in paths))
        gaps.append(ReconstructionGap("bounded_search_exhaustion", unmet or goals,
            "no emitted candidate established all requested mechanisms within this library/search space",
            "inspect_actual_differences_then_try_parameters_order_adapters_or_new_library_bindings",
            tuple(dict(variant=p.candidate.evaluation.variant,
                       missing=tuple(g for g in goals if g not in p.observation.established)) for p in paths)))
    # Keep explicit unbound hypotheses even when another mechanism already succeeds.
    for row in backward:
        if row["status"] == "unbound_requirement":
            gaps.append(ReconstructionGap("unbound_library_requirement", (row["property"],),
                "no selected binding declares a provider; absence from the library is not proved",
                "discover_or_bind_existing_transformation_before_proposing_new_code", (row,)))
    return ReconstructionResult(goals, backward, tuple(paths), tuple(gaps),
        dict(steps=space.search_steps, truncated=space.truncated, max_candidates=max_candidates,
             max_search_steps=max_search_steps, refusals=space.refusals), library,
             reference_cca, goal_plan["relations"])



def goals_from_cca(cca):
    """Bridge existing CCA levers to library mechanism search goals.

    Axis values and their evidence remain attached to every relation. Unknown
    or contradicted axes create no goal. Other populated LEVER axes become
    ordinary unbound library requirements, keeping automatic gap discovery open
    to generated target dialects. Identity fields and metrics do not select an
    optimization. Exact differences use the existing CCA comparator.
    """
    from dataclasses import is_dataclass

    from .cca import CCA
    from .cca_contract import FIELD_REGISTRY, LEVER
    if type(cca) is not CCA or cca.scope_problems():
        raise ValueError("reference goals require an existing valid CCA")
    goals, relations, unknown = [], [], []
    mappings = {
        "memory.onchip_resident": {"a": ("reuse.lhs",), "b": ("reuse.rhs",),
                                    "both": ("reuse.lhs", "reuse.rhs"), "none": ()},
        "memory.panel_reuse": {True: ("memory.panel_reuse",), False: ()},
        "compute.accumulator_resident": {True: ("compute.accumulator_resident",), False: ()},
    }
    for facet_name, facet in vars(cca).items():
        if not is_dataclass(facet):
            continue
        for name, value in vars(facet).items():
            axis = facet_name + "." + name
            spec = FIELD_REGISTRY.get(axis)
            if spec is None or (spec.classification != LEVER and axis not in mappings):
                continue
            axis_evidence = cca.provenance.get("axis_evidence", {}).get(axis, {})
            status = axis_evidence.get("status", "unspecified")
            if value is None or status in ("unknown", "contradicted"):
                unknown.append(axis)
                continue
            supported = mappings[axis].get(value, ()) if axis in mappings else (axis,)
            goals.extend(supported)
            relations.append(dict(axis=axis, value=value, goals=supported, status=status,
                                  evidence=deepcopy(axis_evidence or cca.provenance),
                                  meaning="reference hypothesis; actual consumer verification required"))
    return dict(goals=tuple(dict.fromkeys(goals)), relations=tuple(relations), unknown=tuple(unknown))


def replay(adapter, record, *, goals=()):
    """Rebind exact recorded calls and reject source, context or output drift."""
    library = tuple(b.summary() for b in adapter.bindings)
    if not isinstance(record, Mapping) or record.get("library_sha256") != _digest(library):
        raise ValueError("replay library binding identities differ")
    selected = []
    by_name = {b.family.name: b for b in adapter.bindings}
    for row in record["bindings"]:
        binding = by_name[row["family"]]
        options = row["knobs"]
        domains = dict(binding.family.knobs)
        if set(options) != set(domains) or any(value not in domains[key] for key, value in options.items()):
            raise ValueError("replay options escape selected library domains")
        family = replace(binding.family, knobs=tuple((key, (value,)) for key, value in options.items()), optional=False)
        selected.append(replace(binding, family=family))
    names = tuple(b.family.name for b in selected)
    result = reconstruct(replace(adapter, bindings=tuple(selected), before=tuple(zip(names, names[1:]))),
                         goals, max_candidates=1)
    if len(result.paths) != 1:
        raise ValueError("recorded path did not replay")
    current = result.paths[0].summary()["replay"]
    for key in ("source_sha256", "context", "program_sha256", "intermediate_sha256"):
        if json.dumps(current[key], sort_keys=True) != json.dumps(record[key], sort_keys=True):
            raise ValueError("replay differs at " + key)
    # Narrowed knob domains are a replay implementation detail. Keep the
    # selected original library identity so the returned path is replayable too.
    return replace(result, library=library,
                   paths=tuple(replace(path, library_sha256=record["library_sha256"]) for path in result.paths))



def dense_operand_retention(problem, decision, report):
    """Distinguish once-only traffic from whole-operand private residency.

    Call after the ordinary source/lifetime verifier on the current actual
    command Kernel and the consumed schedule decision. A full K panel alone
    says nothing about all M rows or N columns being simultaneously assigned.
    Whole-operand credit additionally requires the entire source panel in the
    original private row mapping; streamed successive panels cannot borrow it.
    """
    mt, kt, nt = problem.tiles
    full_k = decision.block.k == kt
    rows = []
    for role, requested, unique, complete_panel, view, required_rows in (
        ("lhs", report["requested_lhs_bytes"], problem.m * problem.k * problem.lhs_bytes,
         decision.block.m == mt and full_k, decision.storage.lhs, mt * kt * problem.tile_m),
        ("rhs", report["requested_rhs_bytes"], problem.k * problem.n * problem.rhs_bytes,
         decision.block.n == nt and full_k, decision.storage.rhs, kt * nt * problem.tile_k)):
        if type(requested) is not int or requested < 0:
            raise ValueError("operand retention requires actual nonnegative requested byte counts")
        fits = view is not None and required_rows <= view.capacity
        read_once = requested == unique
        rows.append(dict(role=role, read_once=read_once,
            retained=bool(read_once and complete_panel and fits),
            scope="complete_invocation_full_reduction", actual_requested_bytes=requested,
            unique_source_bytes=unique, full_k_panel=full_k, complete_source_panel=complete_panel,
            required_private_rows=required_rows, assigned_private_rows=0 if view is None else view.capacity,
            required_private_bytes=0 if view is None else required_rows * view.unit_bytes,
            assigned_private_bytes=0 if view is None else view.capacity * view.unit_bytes,
            kernel_sha256=report["kernel_sha256"],
            check="actual source cover/lifetimes, consumed complete operand panel and original private geometry"))
    return tuple(rows)

def make_dense_reconstruction_adapter(
    source, *, problem, resources, storage, target_adapter,
    block_tiles=((1, 1, 1),), outer_orders=(("m", "n"), ("n", "m")),
    loop_transform_sets=((),), retained_operands=("none",), dtypes=("i8", "i8", "i32"), static_budget=256,
    external_buffers_disjoint, verification_timeout_ms=2000, emission=None, feedback=None,
):
    """Bind existing dense library passes through executable ``schedule.plan``.

    This initial adapter accepts a standalone zero-seeded signed integer dense
    source function. Grouped/fused regions use their own checked source binding
    in ReconstructionAdapter, before lower-level scheduling. Tile/resource and
    physical effect facts are supplied by the selected target adapter, never
    inferred from the reference. No runtime or instruction qualification follows.

    Without a machine emission callback the artifact is actual counted-command
    IR, explicitly distinct from machine code. Callbacks receive the ordinary
    PolicyResult containing the checked CompactDenseResult as its artifact.
    """
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.linalg.ops import FillOp, GenericOp

    from merlin.compile.scheduling.compact_dense import emit_compact_dense, schedule_compact_dense
    from merlin.compile.scheduling.compact_verification import verify_compact_dense
    from merlin.compile.scheduling.library_sources import shared_library_sources
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
        read_dense_schedule_plan,
    )
    from merlin.transforms.certificates import ir_digest
    from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry
    from merlin.transforms.pipeline_policy import PolicyFacts, ShapePredicate
    from merlin.transforms.pipeline_state import PipelineContext

    from .cca import CCA, ComputeFacet, DispatchFacet, MemoryFacet

    if (type(block_tiles) is not tuple or not 1 <= len(block_tiles) <= 16
            or any(type(v) is not tuple or len(v) != 3 or any(type(i) is not int or i <= 0 for i in v)
                   for v in block_tiles)
            or type(outer_orders) is not tuple or not 1 <= len(outer_orders) <= 2
            or any(v not in (("m", "n"), ("n", "m")) for v in outer_orders)
            or type(loop_transform_sets) is not tuple or not 1 <= len(loop_transform_sets) <= 16
            or any(type(v) is not tuple for v in loop_transform_sets)
            or type(retained_operands) is not tuple or not retained_operands
            or len(set(retained_operands)) != len(retained_operands)
            or any(role not in ("none", "lhs", "weight") for role in retained_operands)
            or type(external_buffers_disjoint) is not bool or not external_buffers_disjoint):
        raise ValueError("dense reconstruction needs bounded explicit schedule domains and disjoint source buffers")

    def read_source(module):
        operations = tuple(module.body.block.ops)
        if len(operations) != 1 or type(operations[0]) is not func.FuncOp:
            raise ValueError("dense reconstruction requires one standalone source function")
        function = operations[0]
        if len(function.body.blocks) != 1:
            raise ValueError("dense reconstruction source needs one straight-line block")
        body = tuple(function.body.block.ops)
        contractions = tuple(op for op in body if type(op) is GenericOp)
        if len(contractions) != 1:
            raise ValueError("dense reconstruction needs exactly one matched contraction region")
        contraction = contractions[0]
        geometry = static_zero_integer_matmul_geometry(contraction, dtypes)
        if geometry != (problem.m, problem.k, problem.n) or problem.bias:
            raise ValueError("matched source geometry or initializer differs from the dense problem")
        initializer = contraction.outputs[0].owner
        if type(initializer) is FillOp:
            allowed = {initializer, initializer.inputs[0].owner, initializer.outputs[0].owner, contraction, body[-1]}
            if type(initializer.inputs[0].owner) is not arith.ConstantOp:
                raise ValueError("dense region needs its original exact zero initializer")
        elif type(initializer) is tensor.SplatOp:
            allowed = {initializer, initializer.input.owner, contraction, body[-1]}
        else:
            raise ValueError("dense region has unsupported initialization")
        if (set(body) != allowed or type(body[-1]) is not func.ReturnOp
                or tuple(body[-1].operands) != tuple(contraction.results)
                or tuple(contraction.inputs) != tuple(function.body.block.args)):
            raise ValueError("dense region has other live source operations, operands or outputs")
        return dict(geometry=geometry, source_sha256=ir_digest(module),
                    semantics="signed modular ordered reduction with exact zero initialization")

    def original(_):
        return RegisteredPass(PassSpec("dense-source", 0, produces=("source_region",), mutates_ir=False),
                              lambda session: {"source_region": read_source(session.module)})

    def plan(knobs):
        def action(session):
            admitted = session.analysis("source_region")
            tiles = block_tiles[knobs["block"]]
            order = outer_orders[knobs["order"]]
            requests = loop_transform_sets[knobs["transforms"]]
            if knobs["retention"] != "none":
                from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_operand_hoists
                from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
                # Read a checked geometric proposal, then inspect the actual
                # compact command nest. Motion requests remain ordinary schedule
                # dialect operations, with the final independent checker intact.
                geometric = make_dense_schedule_plan(problem, resources, storage, target_adapter,
                    source_identity=admitted["source_sha256"], dtypes=dtypes, static_budget=static_budget,
                    block_tiles=tiles, outer_order=order)
                decision = read_dense_schedule_plan(geometric, problem, resources, storage, target_adapter,
                    source_identity=admitted["source_sha256"], dtypes=dtypes, static_budget=static_budget)
                if decision.block.k != problem.tiles[1]:
                    raise ValueError("operand retention requires a complete original K panel")
                raw = emit_compact_dense(problem, decision.block, decision.storage, target_adapter,
                    name="reconstructed", dtypes=dtypes, outer_order=order)
                hoists = dense_retained_operand_hoists(raw, target_adapter, role=knobs["retention"])
                moved = propose_loop_transforms(raw, hoists, index_bits=target_adapter.index_bits)
                if requests:
                    # Additional requests refer to the current post-motion IR.
                    propose_loop_transforms(moved, requests, index_bits=target_adapter.index_bits)
                requests = (*hoists, *requests)
            proposal = make_dense_schedule_plan(problem, resources, storage, target_adapter,
                source_identity=admitted["source_sha256"], dtypes=dtypes, static_budget=static_budget,
                block_tiles=tiles, outer_order=order, loop_transforms=requests)
            # Roundtrip through the same editable dialect used by generated compilers.
            return {"schedule_plan": parse_dense_schedule_plan(dense_schedule_text(proposal))}
        return RegisteredPass(PassSpec("dense-decisions", 1, requires=("source_region",),
                              produces=("schedule_plan",), mutates_ir=False), action)

    def consume(_):
        def action(session):
            admitted = session.analysis("source_region")
            plan = session.analysis("schedule_plan")
            decision = read_dense_schedule_plan(plan, problem, resources, storage, target_adapter,
                source_identity=admitted["source_sha256"], dtypes=dtypes, static_budget=static_budget)
            result = schedule_compact_dense(problem, resources, storage, target_adapter,
                name="reconstructed", dtypes=dtypes, static_budget=static_budget,
                external_buffers_disjoint=external_buffers_disjoint, schedule_plan=plan,
                source_identity=admitted["source_sha256"],
                verification_timeout_ms=verification_timeout_ms, max_repairs=0)
            if result is None:
                raise ExplorationRefused("selected schedule has no supported checked realization", layer="adapter")
            result.revalidate()
            if result.realization["schedule_ir"]["sha256"] != decision.plan_sha256:
                raise ValueError("actual consumer did not retain the selected scheduling dialect")
            return {"program": result}
        return RegisteredPass(PassSpec("dense-consumer", 2,
            requires=("source_region", "schedule_plan"), produces=("program",), mutates_ir=False), action)

    def emit(result):
        def current():
            result.artifact.revalidate()
            if read_source(result.module)["source_sha256"] != result.receipt["after"]["source_sha256"]:
                raise ValueError("emission changed its original source region")
        current()
        emitted = (emission(result) if emission is not None else
                   EmittedProgram(result.artifact.kernel.text().encode(),
                                  "merlin.counted_commands.v1", result.artifact.kernel.digest()))
        current()
        return emitted

    def observe(candidate):
        result = candidate.evaluation.result.artifact
        result.revalidate()
        original = candidate.evaluation.result.receipt["source"]["source_sha256"]
        schedule = result.realization["schedule_ir"]
        decision = read_dense_schedule_plan(parse_dense_schedule_plan(schedule["text"]),
            problem, resources, storage, target_adapter, source_identity=original,
            dtypes=dtypes, static_budget=static_budget)
        if f":schedule={decision.plan_sha256}:" not in result.checked_emission.context.execution:
            raise ValueError("reported plan differs from the checked executable schedule binding")
        report = verify_compact_dense(result.kernel, problem, resources, decision.storage, target_adapter,
                                      static_budget=static_budget, timeout_ms=verification_timeout_ms)
        actual = result.kernel.digest()
        if candidate.emission.schedule_sha256 != actual:
            raise ValueError("emission does not bind the actual checked schedule")
        # These counts are independently reconstructed by verify_compact_dense.
        baseline = problem.tiles[0] * problem.tiles[1] * problem.tiles[2]
        witnesses = {
            "source.integer_dense": {"source_sha256": candidate.evaluation.result.receipt["source"]["source_sha256"]},
            "source.reduction_order": {"kernel_sha256": actual, "check": "verify_compact_dense"},
            "schedule.consumed": {"kernel_sha256": actual,
                                  "plan_sha256": decision.plan_sha256,
                                  "consumer": "schedule_compact_dense"},
        }
        if decision.block.m > 1 or decision.block.n > 1 or decision.block.k > 1:
            witnesses["schedule.blocked"] = {"kernel_sha256": actual, "block": asdict(decision.block)}
        if report["lhs_loads"] < baseline:
            witnesses["reuse.lhs"] = {"kernel_sha256": actual, "actual_loads": report["lhs_loads"],
                                     "streamed_loads": baseline, "check": "verify_compact_dense"}
        if report["rhs_loads"] < baseline:
            witnesses["reuse.rhs"] = {"kernel_sha256": actual, "actual_loads": report["rhs_loads"],
                                     "streamed_loads": baseline, "check": "verify_compact_dense"}
        if decision.loop_transforms:
            untransformed = emit_compact_dense(problem, decision.block, decision.storage, target_adapter,
                name=result.kernel.name, dtypes=dtypes, outer_order=decision.outer_order)
            if untransformed.digest() != actual:
                witnesses["schedule.loop_transformed"] = {"kernel_sha256": actual,
                    "executed_requests": tuple(asdict(request) for request in decision.loop_transforms),
                    "check": "verify_compact_dense", "untransformed_sha256": untransformed.digest()}
        retention = dense_operand_retention(problem, decision, report)
        for row in retention:
            role = row["role"]
            if row["read_once"]:
                witnesses["memory." + role + "_read_once"] = row
            if row["retained"]:
                witnesses["memory." + role + "_retained_full_reduction"] = row
        left, right = report["lhs_loads"] < baseline, report["rhs_loads"] < baseline
        resident = "both" if left and right else "a" if left else "b" if right else "none"
        cca = CCA(op="matmul", backend=[],
            compute=ComputeFacet(op="matmul", accumulator_dtype=dtypes[2],
                register_block=(min(problem.m, decision.block.m * problem.tile_m),
                                min(problem.n, decision.block.n * problem.tile_n)), accumulator_resident=True),
            memory=MemoryFacet(panel_reuse=left or right, onchip_resident=resident, capacity_fit=True),
            dispatch=DispatchFacet(n_dispatches=report["executed_instruction_calls"]),
            provenance=dict(source=actual, level="checked_counted_commands",
                check="verify_compact_dense", machine_emission="not_established",
                timing="unknown", scope="complete standalone dense source region"))
        cca.provenance["axis_evidence"] = {
            axis: dict(status="observed", evidence=(actual,),
                       reason="independently checked actual counted-command operands and lifetimes")
            for axis in ("compute.register_block", "compute.accumulator_resident", "memory.panel_reuse",
                         "memory.onchip_resident", "memory.capacity_fit", "dispatch.n_dispatches")}
        cca.provenance["residency_scope"] = "reuse within checked macro-block/panel lifetimes"

        for axis, value in (("compute.register_block", cca.compute.register_block),
                            ("compute.accumulator_resident", True),
                            ("memory.panel_reuse", left or right)):
            if value:
                witnesses[axis] = {"axis": axis, "value": value, "kernel_sha256": actual,
                                   "check": "verify_compact_dense"}
        return MechanismObservation(tuple(witnesses), witnesses=witnesses, cca=cca)

    def retention_predicates(knobs):
        return (() if knobs["retention"] == "none" else
                (ShapePredicate(0, 1, at_most=block_tiles[knobs["block"]][1] * problem.tile_k),))

    bindings = (
        TransformationBinding(PassFamily("source", original), establishes=("source.integer_dense",),
            verification=("static_zero_integer_matmul_geometry", "complete_standalone_source_region")),
        TransformationBinding(PassFamily("decisions", plan, knobs=(
            ("block", tuple(range(len(block_tiles)))), ("order", tuple(range(len(outer_orders)))),
            ("transforms", tuple(range(len(loop_transform_sets)))),
            ("retention", retained_operands)), predicates=retention_predicates),
            requires=("source.integer_dense",), establishes=("schedule.explicit_plan",),
            verification=("read_dense_schedule_plan",), emission_consumers=("schedule_compact_dense",)),
        TransformationBinding(PassFamily("consume", consume), requires=("schedule.explicit_plan",),
            establishes=("schedule.consumed", "schedule.blocked", "reuse.lhs", "reuse.rhs",
                         "schedule.loop_transformed", "source.reduction_order",
                         "compute.register_block", "compute.accumulator_resident", "memory.panel_reuse",
                         "memory.lhs_retained_full_reduction", "memory.rhs_retained_full_reduction",
                         "memory.lhs_read_once", "memory.rhs_read_once"),
            verification=("verify_compact_dense", "CompactDenseResult.revalidate"),
            emission_consumers=("CompactDenseResult.emit_c", "CompactDenseResult.emit_llvm")),
    )
    context = PipelineContext(hardware=target_adapter.configuration_identity,
        numerical_policy=target_adapter.numerical_policy_identity,
        resources=_digest((asdict(resources), asdict(storage))),
        execution=target_adapter.qualification_identity or "unqualified_counted_command_ir",
        library=_digest(shared_library_sources(
            version=14 if any(role != "none" for role in retained_operands) else 13)))
    return ReconstructionAdapter(source, bindings,
        lambda session: PolicyFacts(session.stamp(),
            shapes=((problem.m, problem.k), (problem.k, problem.n))),
        emit, observe, session_options={"context": context}, feedback=feedback)
