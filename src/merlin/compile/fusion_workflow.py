"""Source-driven fusion exploration around existing compiler APIs, not a pass runner.

Discovery grants no legality or speedup. Realization returns the actual native
product and original alternative for the caller's ordinary emission pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from merlin.common.digest import is_sha256
from merlin.transforms.certificates import ir_digest


ROUTES = {
    "multioutput": "merlin.compile.scheduling.multioutput_fusion.compile_source_multioutput",
    "adjacent_elementwise": "merlin.compile.scheduling.fusion.plan_fusion",
    "epilogue": "merlin.xdsl_dialects.lowering.output_fusion.decisions",
    "shared_input": "merlin.compile.scheduling.shared_input_schedule.schedule_shared_input",
    "connected_residency": "merlin.compile.connected_regions.compile_connected_pointwise",
    "preparation_reuse": "merlin.compile.compiler_observations.install_prepared_observed",
}


# Stable public surface: discovery remains separate from physical realization.
from .fusion_discovery import discover, plan_materialization_frontier, _structural_family  # noqa: F401


@dataclass(frozen=True)
class Realization:
    artifact: object
    unfused: object
    receipt: dict
    objects: tuple = ()


def realize_adjacent(kernel, objects, banks, adapter, capabilities, **limits):
    """Return actual checked target calls, retaining the original unfused Kernel.

The existing finite physical-fusion verifier remains the owner. Counted-loop
regions must use their compact route; this helper never unrolls them.
"""
    from merlin.compile.scheduling.fusion import plan_fusion
    from merlin.sched.ir import Call

    before = kernel.digest()
    if any(type(node) is not Call for node in kernel.body):
        return Realization(kernel, kernel, {
            "route": "adjacent_elementwise", "status": "unsupported",
            "reason": "finite synchronous call route only; retain compact loops unchanged",
            "source_schedule_sha256": before, "schedule_sha256": before,
            "hardware_unsupported": False, "whole_region_cost": None,
            "planned": {}, "emitted": {},
        }, tuple(objects))
    candidate = plan_fusion(kernel, objects, banks, adapter, capabilities, **limits)
    if kernel.digest() != before:
        raise ValueError("native fusion mutated the retained unfused schedule")
    artifact = kernel if candidate is None else candidate.kernel
    return Realization(artifact, kernel, {
        "route": "adjacent_elementwise", "status": "no-match" if candidate is None else "applied",
        "reason": "native finite composition checks passed" if candidate is not None else
                  "no native composition satisfies the selected semantics, effects and resources",
        "planned": {} if candidate is None else {"source_regions": 1}, "emitted": {},
        "maturity": "checked_schedule_not_instruction_image",
        "source_schedule_sha256": before, "schedule_sha256": artifact.digest(),
        "checks": [] if candidate is None else list(candidate.checks),
        "objects": [asdict(obj) for obj in (objects if candidate is None else candidate.objects)],
        "emission": "actual returned Kernel must enter ordinary allocation/emission",
        "execution_qualification": "not_established", "whole_region_cost": None,
    }, tuple(objects if candidate is None else candidate.objects))


def realize_shared_input(module, group, **selection):
    """Return the native checked logical schedule; target encoding is still required.

    The original module is retained unchanged as the ordinary unfused alternative.
    Each output and later epilogue stays independent. Never call logical effects an
    emitted target image or claim that planned byte reduction is measured traffic.
    """
    from merlin.compile.scheduling.shared_input_schedule import schedule_shared_input
    from merlin.verify.lean_contract import document_digest

    before = ir_digest(module)
    candidate = schedule_shared_input(module, group, **selection)
    if ir_digest(module) != before:
        raise ValueError("shared-input scheduling mutated the retained source")
    record = None if candidate is None else asdict(candidate)
    return Realization(candidate if candidate is not None else module, module, {
        "route": "shared_input", "status": "no-match" if candidate is None else "applied",
        "reason": "native source/storage checks passed" if candidate is not None else
                  "selected tails, output blocking or storage capacities admit no realization",
        "planned": {} if candidate is None else {"source_regions": 1}, "emitted": {},
        "source_ir_sha256": before,
        "maturity": "logical_schedule_not_target_instruction_image",
        "schedule_sha256": None if record is None else document_digest(record),
        "schedule": record,
        "boundary": group.record(),
        "storage": {key: asdict(selection[key]) for key in ("resources", "storage")},
        "emission": "consume returned operations through target adapter and independent actual-effect checker",
        "execution_qualification": "not_established", "whole_region_cost": None,
    })


def realize_multioutput(module, operations, **selection):
    """Consume the existing closed fan-out compiler and its guarded actual images.

    Invalid effects/source are refused by the native compiler. This is a narrow
    pointwise route, not general contraction or whole-graph fusion.
    """
    from merlin.compile.scheduling.multioutput_fusion import compile_source_multioutput
    from merlin.compile.compiler_observations import emitted_multioutput

    before = ir_digest(module)
    product = compile_source_multioutput(module, operations, **selection)
    if ir_digest(module) != before:
        raise ValueError("multioutput compiler mutated the retained source")
    observed = emitted_multioutput(product)
    return Realization(product, module, {
        **observed, "route": "adjacent_elementwise", "form": "two_output_pointwise_fanout",
        "status": "applied", "source_ir_sha256": before,
        "emission": "consume the returned guarded compilation images in ordinary target emission",
        "whole_region_cost": None, "execution_qualification": "not_established",
    })


def compare_region_costs(baseline, candidate, *, allow_provisional=False):
    """Compare explicit complete-region observations; issue no placement authority.

    Observation trust remains with the caller's timing/correctness owner. Missing
    measurements cannot turn planned transformations into a gain. Explicit
    allow_provisional admits complete, same-model exploratory comparisons without
    promoting them to qualified or measured speedups. Physical observations need
    no fictitious cycle-model identity.
    """
    keys = ("source_ir_sha256", "numeric_policy_sha256", "hardware_sha256", "scope_sha256")
    if type(allow_provisional) is not bool:
        raise ValueError("provisional comparison selection must be an explicit Boolean")
    unknown = {"status": "unknown", "saved_cycles": None, "placement_authority": False}
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return {**unknown, "reason": "missing whole-region observations"}
    if any(baseline.get(k) != candidate.get(k) or not is_sha256(baseline.get(k)) for k in keys):
        return {**unknown, "reason": "source/numeric/hardware/scope mismatch"}
    method = baseline.get("method")
    if method not in {"physical", "model"} or candidate.get("method") != method:
        return {**unknown, "reason": "unknown or mismatched timing method"}
    qualifications = {"qualified", "provisional"} if allow_provisional else {"qualified"}
    if method == "model" and (
            not is_sha256(baseline.get("model_sha256"))
            or baseline.get("model_sha256") != candidate.get("model_sha256")
            or baseline.get("model_qualification") != candidate.get("model_qualification")
            or baseline.get("model_qualification") not in qualifications):
        return {**unknown, "reason": "unbound, mismatched or unqualified cycle model"}
    if any(row.get("scope") != "complete_region" or row.get("complete") is not True
           or row.get("correctness") != "pass"
           or type(row.get("cycles")) not in (int, float) or not math.isfinite(row["cycles"])
           or row["cycles"] < 0 or not is_sha256(row.get("artifact_sha256"))
           for row in (baseline, candidate)):
        return {**unknown, "reason": "partial, incorrect, unbound or invalid complete-region observations"}
    return {"status": "reported_comparison", "saved_cycles": baseline["cycles"] - candidate["cycles"],
            "baseline": baseline, "candidate": candidate, "placement_authority": False,
            "basis": "modeled_roi" if method == "model" else "measured_roi",
            "provisional": method == "model" and baseline.get("model_qualification") == "provisional",
            "qualification": "supplied observations; native timing and correctness owners remain authoritative"}


def realize_candidate(module, candidate, *, capability_contract, numerical_policy_sha256,
                      preparation_requests=(), max_region_operations=8, connected=None, **selection):
    """Revalidate one paged opportunity, then use its existing closed realizer.

    ``connected=...`` selects the source-owned sequential composition bridge
    with explicit target stage options, storage and runtime continuation. No
    backend or runtime contract is inferred. Unsupported graph routes return
    the unchanged source with a concrete reason.
    Resource/search-budget refusals are local to this alternative. Invalid source
    or contract bindings refuse before any native action. No arbitrary graph
    rewrite is inferred from the advisory capability declarations.
    """
    from merlin.compile.scheduling.shared_input_groups import SharedInputGroup
    from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise

    index = candidate.get('candidate_index')
    binding = candidate.get('binding_sha256')
    page = discover(module, capability_contract=capability_contract,
        numerical_policy_sha256=numerical_policy_sha256, preparation_requests=preparation_requests,
        max_region_operations=max_region_operations, max_candidates=1,
        cursor={'schema': 'merlin.fusion_cursor.v1', 'binding_sha256': binding, 'offset': index})
    from merlin.verify.lean_contract import document_digest
    if document_digest(page['candidates']) != document_digest([candidate]):
        raise ValueError('fusion candidate differs from the current bound discovery row')
    before = ir_digest(module)
    route = candidate['route']
    try:
        legality = candidate.get('source_legality')
        if legality is not None and legality.get('status') == 'declined':
            raise ValueError(legality['reason'])
        if connected is not None and route in {'adjacent_elementwise', 'connected_residency', 'epilogue'}:
            from merlin.compile.connected_regions import plan_connected_region, compile_connected_pointwise, compile_connected_region
            members = tuple(module.walk())
            region = plan_connected_region(module, tuple(members[i] for i in candidate['operation_ordinals']))
            compiler = compile_connected_region if 'dense_options' in connected else compile_connected_pointwise
            product = compiler(region, **connected)
            if ir_digest(module) != before:
                raise ValueError('connected realization mutated the retained source')
            return Realization(product, module, {'route': 'connected_mixed' if 'dense_options' in connected else 'connected_pointwise', 'status': 'applied',
                'source_ir_sha256': before, 'connected': product.report,
                'maturity': 'executable_unfused_composition_not_cross_stage_fusion',
                'whole_region_cost': None, 'physical_savings_claimed': False,
                'execution_qualification': 'not_established'})
        if route == 'shared_input' and candidate['operation_ordinals']:
            fields = candidate['detail']
            group = SharedInputGroup(fields['source_ir_sha256'], tuple(fields['source_op_indices']),
                tuple(fields['contraction_ordinals']), fields['m'], fields['k'], tuple(fields['widths']))
            return realize_shared_input(module, group, **selection)
        if route == 'multioutput':
            operations = tuple(module.walk())
            return realize_multioutput(module, tuple(operations[i] for i in candidate['operation_ordinals']), **selection)
        status = 'unsupported'
        reason = ('use realize_adjacent after ordinary finite schedule lowering' if route == 'adjacent_elementwise'
                  else 'this discovery route has no automatic whole-region realization; retain source')
    except (ValueError, UnsupportedElementwise) as error:
        if ir_digest(module) != before:
            raise ValueError('failed fusion realization mutated the retained source') from error
        reason = str(error)
        status = 'budget_exhausted' if 'budget' in reason else 'declined'
    return Realization(module, module, {'route': route, 'status': status, 'reason': reason,
        'source_ir_sha256': before, 'whole_region_cost': None, 'planned': {}, 'emitted': {},
        'hardware_unsupported': False, 'execution_qualification': 'not_established'})


@dataclass(frozen=True)
class FusionAlternative:
    """One compiler-selected native realization on its private source clone."""
    name: str
    realize: object
    predicates: tuple = ()


@dataclass(frozen=True)
class FusionEvaluation:
    name: str
    status: str
    evaluation: object
    realization: Realization | None
    reason: str | None = None


def evaluate_fusions(source, alternatives, emit, *, context=None, facts=None,
                     clone=None, fingerprint=None, verify_ir=None, ir_level='structured', unfused_objects=()):
    """Emit unfused plus at most fifteen realized alternatives using native search.

    ``emit(realization, session)`` receives the actual returned artifact/objects,
    never only a receipt. It must use the caller's normal allocation/target
    emission path. Independent correctness and performance grading stay external.
    Source IR/schedules and contract identities remain immutable across clones.
    Discovery pages are resumable; parameterizations for one region are bounded
    to sixteen paths including the unconditional unfused fallback per round.
    """
    from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass, evaluate_variants
    from merlin.transforms.pipeline_state import PassSpec

    alternatives = tuple(alternatives)
    if (len(alternatives) > 15 or any(type(a) is not FusionAlternative or not callable(a.realize)
            or not isinstance(a.name, str) or not a.name or a.name == 'unfused' for a in alternatives)
            or len({a.name for a in alternatives}) != len(alternatives) or not callable(emit)):
        raise ValueError('fusion round requires an emitter and at most fifteen unique alternatives plus unfused')
    outcomes = {}
    def prepare(session, alternative=None):
        before = session.stamp()
        if alternative is None:
            product = Realization(session.module, session.module, {'status': 'unfused',
                'source_ir_sha256': before.source_sha256, 'execution_qualification': 'not_established'},
                tuple(unfused_objects))
        else:
            product = alternative.realize(session.module)
        name = 'unfused' if alternative is None else alternative.name
        if type(product) is not Realization or product.unfused is not session.module or session.stamp() != before:
            raise ValueError('fusion realization lost its unchanged private unfused source')
        outcomes[name] = product
        if product.receipt.get('status') not in {'applied', 'unfused'}:
            raise ValueError(product.receipt.get('reason', 'native realization unavailable'))
        return {'fusion': product}
    registry = {'emit': RegisteredPass(PassSpec('emit', 1, requires=('fusion',), produces=('artifact',),
                        mutates_ir=False, ir_level=ir_level),
                        lambda s: {'artifact': emit(s.analysis('fusion'), s)})}
    variants = []
    for alternative in (None, *alternatives):
        name = 'unfused' if alternative is None else alternative.name
        prepare_name = 'realize:' + name
        registry[prepare_name] = RegisteredPass(PassSpec(prepare_name, 0, produces=('fusion',),
            mutates_ir=False, ir_level=ir_level), lambda s, a=alternative: prepare(s, a))
        variants.append(PipelineVariant(name, (prepare_name, 'emit'), () if alternative is None else alternative.predicates))
    for result in evaluate_variants(source, tuple(variants), registry,
            facts or (lambda s: PolicyFacts(s.stamp())), output='artifact', context=context,
            ir_level=ir_level, clone=clone, fingerprint=fingerprint, verify_ir=verify_ir):
        product = outcomes.get(result.variant)
        refused = product is not None and product.receipt.get('status') not in {'applied', 'unfused'}
        yield FusionEvaluation(result.variant, product.receipt['status'] if refused else result.status,
                               result, product, result.error)
