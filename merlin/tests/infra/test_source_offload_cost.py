"""Synthetic complete-cost ownership through ordinary source preparation.

These cases exercise routing and custody only. They measure no timer and qualify
neither an accelerator nor the supplied synthetic cost observations.
"""

import json
import runpy
from dataclasses import replace

import pytest
from test_offload_plan import choose, example
from test_source_dense_lowering import run, selection
from xdsl.dialects import builtin, func

from merlin.common import mlir_query as mq
from merlin.common.jsonio import canonical_sha256
from merlin.common.paths import merlin_dir
from merlin.llvmlower.offload_plan import bind_source, realize_offload_plan
from merlin.llvmlower.source_dense import offload_context, ordinary_dense_route_identity
from merlin.llvmlower.source_model import source_provider_builder
from merlin.perf.offload_profitability import compare_complete_offload_cost
from merlin.perf.term import PerformanceTerm


def cost_case(tmp_path, *, bounds=(70, 80), ownership=True, prepare=None, unknown=False, **cost_changes):
    """Use the existing synthetic original owner before it freezes its query."""
    config, problem, seen = selection(tmp_path, prepare=prepare, private_ownership=ownership)
    module = example(shape=(problem.m, problem.k, problem.n), widths=config.selected_contracts.dense.dtypes)[0].module
    context = offload_context(config)
    bound = bind_source(module, entry="forward", context=context)
    device = tuple(
        replace(
            alternative,
            id=f"device_{index}",
            implementation=config.adapter.qualification_identity,
            placement=config.selected_contracts.target,
        )
        for index, (operation, alternative) in enumerate(zip(bound.operations, bound.cpu_alternatives, strict=True))
        if operation.name == "linalg.generic"
    )
    plan = realize_offload_plan(bound, choose(bound, device), device_alternatives=device, context=context)

    def original_inputs(query):
        query.update(
            stamp=plan.stamp,
            original_selection_sha256=ordinary_dense_route_identity(plan.stamp),
            candidate_selection_sha256=plan.plan.digest,
        )
        query.update(cost_changes)
        if unknown:
            term = query["candidate_costs"]["warm"]
            query["candidate_costs"]["warm"] = PerformanceTerm.unknown(
                term.name, term.unit, term.provenance, term.validity, "no qualified complete timing"
            )

    # Cross-bucket fixture access stays explicit and checkout-relative. Its
    # original verifier fingerprints these fields after original_inputs runs.
    request = runpy.run_path(str(merlin_dir() / "tests/dse/test_offload_profitability.py"))["request"]
    query, evidence, calls = request(bounds=bounds, before=original_inputs)
    configured = replace(config, offload_plan=plan)
    if cost_changes:
        # Deliberately mismatched externally constructed route IDs must still
        # refuse at the existing explicit decision boundary.
        decision = compare_complete_offload_cost(**query)
        configured = replace(configured, offload_comparison=decision)
    else:
        configured = configured.with_complete_offload_cost(
            **{
                key: value
                for key, value in query.items()
                if key not in ("stamp", "original_selection_sha256", "candidate_selection_sha256")
            }
        )
    return configured, module, seen, configured.offload_comparison, evidence, calls


def test_complete_cost_factory_requires_a_current_original_plan(tmp_path):
    config, _, _ = selection(tmp_path)
    with pytest.raises(ValueError, match="current source plan"):
        config.with_complete_offload_cost(
            original=None,
            candidate=None,
            original_costs=None,
            candidate_costs=None,
            scope=None,
            regime="warm",
            verify_original=lambda **kwargs: pytest.fail("no unbound verifier call"),
        )


@pytest.mark.parametrize(
    "bounds,unknown,chosen",
    [((70, 80), False, True), ((90, 110), False, False), ((130, 150), False, False), ((70, 80), True, False)],
)
def test_complete_cost_choice_changes_actual_prepared_membership_only_for_strict_opportunity(
    tmp_path, bounds, unknown, chosen
):
    config, module, seen, decision, _, _ = cost_case(tmp_path, bounds=bounds, unknown=unknown)
    output, builder = run(tmp_path, config, module)
    body = next(op for op in mq.parse(output).body.block.ops if op.sym_name.data == "forward")
    assert len(seen) == (1 if chosen else 2)
    assert sum(type(op) is func.CallOp for op in body.walk()) == len(seen)
    assert sum(op.name == "linalg.generic" for op in body.walk()) == (1 if chosen else 0)
    assert source_provider_builder(builder) is builder
    record = json.loads((output.parent / "source_dense.json").read_text())["offload_placement"][0]
    assert record["actual_route"] == ("candidate" if chosen else "original")
    assert record["comparison"]["candidate_selection_sha256"] == config.offload_plan.plan.digest
    assert record["comparison"]["original_selection_sha256"] == ordinary_dense_route_identity(decision.stamp)
    assert len(record["selected_dense_ordinals"]) == len(seen)
    assert len(record["retained_cpu_dense_ordinals"]) == (1 if chosen else 0)
    if chosen:
        assert seen[0].compilation.operation_ordinal in config.offload_plan.selected_ordinals


@pytest.mark.parametrize("unknown", [False, True])
def test_rejected_cost_preserves_ordinary_source_when_candidate_ownership_is_unavailable(tmp_path, unknown):
    config, module, seen, _, _, _ = cost_case(tmp_path, bounds=(90, 110), ownership=False, unknown=unknown)
    output, builder = run(tmp_path, config, module)
    assert not seen and source_provider_builder(builder) is None
    assert output.read_bytes() == (tmp_path / "model.mlir").read_bytes()


def test_cost_opportunity_never_grants_missing_runtime_ownership(tmp_path):
    config, module, seen, _, _, _ = cost_case(tmp_path, ownership=False)
    with pytest.raises(ValueError, match="legal current provider"):
        run(tmp_path, config, module)
    assert not seen


@pytest.mark.parametrize("key", ["original_selection_sha256", "candidate_selection_sha256"])
def test_same_source_cost_observations_cannot_select_another_route(tmp_path, key):
    with pytest.raises(ValueError, match="exact route selections"):
        cost_case(tmp_path, **{key: canonical_sha256("different plan, identical source")})


@pytest.mark.parametrize("change", ["source", "hardware", "packing", "storage", "evidence"])
def test_current_source_full_dense_context_and_live_evidence_are_reopened(tmp_path, change):
    config, module, seen, _, evidence, _ = cost_case(tmp_path)
    if change == "source":
        module.attributes["prov.changed"] = builtin.UnitAttr()
    elif change == "hardware":
        config.selected_contracts.facts_path.write_bytes(config.selected_contracts.facts_bytes + b"\n")
    elif change == "packing":
        object.__setattr__(config, "borrow_contiguous", not config.borrow_contiguous)
    elif change == "storage":
        object.__setattr__(config.storage.lhs, "capacity", config.storage.lhs.capacity + 8)
    else:
        evidence["current"] = False
    with pytest.raises(ValueError, match="changed|stale|differs|measurement owner"):
        run(tmp_path, config, module)
    assert not seen


@pytest.mark.parametrize("mutation", ["replace_verifier", "equal_verifier", "expire_evidence"])
def test_implementation_callback_cannot_replace_or_expire_original_cost_ownership(tmp_path, mutation):
    original = {}

    def callback(bindings, directory):
        assert len(bindings) == 1
        if mutation in ("replace_verifier", "equal_verifier"):
            decision = original["decision"]
            retained_report = decision._record

            class EqualVerifier:
                def __eq__(self, other):
                    return True

                def __call__(self):
                    return retained_report

            replacement = EqualVerifier() if mutation == "equal_verifier" else lambda: retained_report
            object.__setattr__(decision, "_check", replacement)
        else:
            original["evidence"]["current"] = False

    config, module, _, decision, evidence, _ = cost_case(tmp_path, prepare=callback)
    original.update(decision=decision, evidence=evidence)
    with pytest.raises(ValueError, match="changed|measurement owner"):
        run(tmp_path, config, module)
    assert not (tmp_path / "preparation/prepared_model_transform.json").exists()
