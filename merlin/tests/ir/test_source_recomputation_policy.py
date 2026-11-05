"""Pure original policy transport; synthetic cost terms qualify no timer."""

import gc
import importlib.util
import weakref
from dataclasses import fields

import pytest
from test_recomputation import budgets, permissions, source
from test_source_recomputation_selection import original_owner, settings, synthetic_comparison
from xdsl.dialects.builtin import StringAttr

from merlin.llvmlower import source_recomputation_policy as P
from merlin.llvmlower.source_recomputation import SourceRecomputationPreparation
from merlin.transforms.certificates import ir_digest, ir_text
from merlin.transforms.pipeline_state import PipelineContext


def policy(**changes):
    return P.SourceRecomputationPolicy(
        changes.pop("context", PipelineContext("hardware", "numeric", "resources", "runtime", "library")),
        changes.pop("permissions", permissions()),
        changes.pop("budgets", budgets()),
        verify_current=changes.pop("verify_current", original_owner),
        **changes,
    )


def prepared():
    return ir_text(source())


def test_missing_cost_is_exact_original_and_binds_actual_prepared_stamp():
    selected = policy()
    text, actual = prepared(), settings()
    bound = selected.bind(text, settings=actual)
    selected.revalidate()
    assert bound.plan.stamp.source_sha256 == ir_digest(text)
    assert bound.plan.stamp.context == selected.context
    consumed = SourceRecomputationPreparation(text, bound, actual)
    assert consumed.selected_source == text
    assert consumed.report["status"] == "MISSING_COST" and consumed.report["selected"] is False
    with pytest.raises(ValueError, match="one binding"):
        selected.bind(text, settings=actual)


@pytest.mark.parametrize("bounds,chosen", [((70, 80), True), ((110, 140), False), ((130, 140), False)])
def test_only_existing_matching_complete_cost_controls_selection(bounds, chosen):
    calls = []

    def comparison_for(*, plan, settings):
        calls.append(plan)
        assert settings["index_bits"] == 64
        return synthetic_comparison(plan, bounds=bounds)

    selected = policy(comparison_for=comparison_for)
    text, actual = prepared(), settings()
    bound = selected.bind(text, settings=actual)
    assert calls == [bound.plan]
    consumed = SourceRecomputationPreparation(text, bound, actual)
    assert consumed.report["selected"] is chosen
    assert consumed.selected_source == (ir_text(bound.plan.candidate) if chosen else text)
    selected.revalidate()
    assert calls == [bound.plan]  # Revalidate never resolves or plans again.


@pytest.mark.parametrize("permission", [entry.name for entry in fields(P.R.RecomputationPermissions)][1:])
def test_each_missing_original_permission_keeps_source_without_cost_or_owner_callbacks(permission):
    def forbidden(**kwargs):
        pytest.fail("missing permission must not request costs or qualification")

    selected = policy(
        permissions=permissions(**{permission: False}), comparison_for=forbidden, verify_current=forbidden
    )
    assert selected.bind(prepared(), settings=settings()) is None
    selected.revalidate()
    with pytest.raises(ValueError, match="one binding"):
        selected.bind(prepared(), settings=settings())


@pytest.mark.parametrize("text,limits", [("module {}", None), (None, budgets(max_source_elements=1))])
def test_unsupported_or_exhausted_source_returns_none_without_resolver(text, limits):
    def forbidden(**kwargs):
        pytest.fail("unsupported original source cannot resolve costs")

    selected = policy(budgets=limits or budgets(), comparison_for=forbidden, verify_current=forbidden)
    assert selected.bind(text or prepared(), settings=settings()) is None


@pytest.mark.parametrize("field", ["context", "permissions", "budgets"])
def test_original_policy_input_identity_cannot_be_substituted_even_equal(field):
    selected = policy()
    original = getattr(selected, field)
    object.__setattr__(selected, field, type(original)(**vars(original)))
    with pytest.raises(ValueError, match="selection changed"):
        selected.bind(prepared(), settings=settings())


@pytest.mark.parametrize(
    "field,name,value",
    [
        ("context", "execution", "foreign"),
        ("permissions", "inputs_stable_during_region", False),
        ("budgets", "max_candidates", 7),
    ],
)
def test_retained_context_permissions_and_budget_mutations_refuse_before_resolver(field, name, value):
    calls = []

    def comparison_for(**kwargs):
        calls.append(kwargs)

    selected = policy(comparison_for=comparison_for)
    object.__setattr__(getattr(selected, field), name, value)
    with pytest.raises(ValueError, match="selection changed"):
        selected.bind(prepared(), settings=settings())
    assert not calls


@pytest.mark.parametrize(
    "mutation",
    ["source", "candidate", "source_clone", "stamp", "baseline_id", "candidate_id", "settings", "permission", "budget"],
)
def test_resolver_cannot_change_issued_ssa_context_alternatives_or_original_inputs(mutation):
    holder = {}

    def comparison_for(*, plan, settings):
        holder["plan"] = plan
        if mutation in ("source", "candidate"):
            getattr(plan, mutation).attributes["late"] = StringAttr("changed")
        elif mutation == "source_clone":
            object.__setattr__(plan, "source", plan.source.clone())
        elif mutation == "stamp":
            object.__setattr__(plan.stamp.context, "execution", "foreign-runtime")
        elif mutation.endswith("_id"):
            field = "baseline_selection_sha256" if mutation == "baseline_id" else "candidate_selection_sha256"
            object.__setattr__(plan, field, "0" * 64)
        elif mutation == "settings":
            settings["index_bits"] = 32
        elif mutation == "permission":
            object.__setattr__(holder["policy"].permissions, "inputs_stable_during_region", False)
        else:
            object.__setattr__(holder["policy"].budgets, "max_candidates", 1)
        return None

    holder["policy"] = selected = policy(comparison_for=comparison_for)
    with pytest.raises(ValueError, match="changed|original|issued|replan"):
        selected.bind(prepared(), settings=settings())
    assert "plan" in holder
    with pytest.raises(ValueError):
        selected.bind(prepared(), settings=settings())  # Never repair/replan.


@pytest.mark.parametrize("mutation", ["foreign_stamp", "wrong_type"])
def test_resolver_cannot_return_foreign_cost_or_untyped_record(mutation):
    def comparison_for(*, plan, settings):
        if mutation == "wrong_type":
            return {"choose_candidate": True}
        foreign = P.R.plan_source_recomputation(source(), PipelineContext("other"), permissions(), budgets())
        return synthetic_comparison(foreign)

    with pytest.raises(ValueError, match="current source|original|decision|selection|recomputation"):
        policy(comparison_for=comparison_for).bind(prepared(), settings=settings())


@pytest.mark.parametrize("method", ["bind", "revalidate"])
def test_original_resolver_reentrancy_fails_and_consumes_attempt(method):
    holder = {}

    def comparison_for(**kwargs):
        if method == "bind":
            holder["policy"].bind(prepared(), settings=settings())
        else:
            holder["policy"].revalidate()

    holder["policy"] = selected = policy(comparison_for=comparison_for)
    with pytest.raises(ValueError, match="not reentrant|binding failed"):
        selected.bind(prepared(), settings=settings())
    with pytest.raises(ValueError, match="failed"):
        selected.revalidate()


@pytest.mark.parametrize(
    "mutation", ["resolver_code", "verifier_identity", "reader_identity", "checker_code", "binder_code", "core_helper"]
)
def test_original_callback_cannot_replace_bound_code_or_direct_helpers(monkeypatch, mutation):
    holder, restore = {}, []

    def comparison_for(**kwargs):
        selected = holder["policy"]
        if mutation == "resolver_code":
            restore.append((comparison_for, comparison_for.__code__))
            comparison_for.__code__ = comparison_for.__code__.replace(co_name="changed")
        elif mutation == "verifier_identity":
            object.__setattr__(selected, "verify_current", comparison_for)
        elif mutation == "reader_identity":
            original = P._functions()
            monkeypatch.setattr(P, "_functions", lambda: original)
        elif mutation in ("checker_code", "binder_code"):
            function = selected._check if mutation == "checker_code" else selected._bind
            restore.append((function, function.__code__))
            function.__code__ = function.__code__.replace(co_name="changed")
        else:
            monkeypatch.setattr(P.R, "_state", lambda module: ())

    holder["policy"] = selected = policy(comparison_for=comparison_for)
    try:
        with pytest.raises(ValueError, match="changed"):
            selected.bind(prepared(), settings=settings())
    finally:
        for function, code in restore:
            function.__code__ = code


def test_original_callback_file_is_pinned_before_resolver(tmp_path):
    path = tmp_path / "original_owner.py"
    path.write_text(
        "from pathlib import Path\n"
        "def verify_current(**kwargs): pass\n"
        "def comparison_for(**kwargs):\n"
        "    p=Path(__file__)\n"
        "    p.write_text(p.read_text()+'# changed\\n')\n"
    )
    spec = importlib.util.spec_from_file_location("original_recomputation_policy_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = policy(verify_current=module.verify_current, comparison_for=module.comparison_for)
    with pytest.raises(ValueError, match="changed"):
        selected.bind(prepared(), settings=settings())


@pytest.mark.parametrize("mutation", ["plan", "comparison", "verifier", "settings", "source"])
def test_bound_policy_revalidation_keeps_original_selection_and_never_rebinds(mutation):
    selected, actual = policy(), settings()
    bound = selected.bind(prepared(), settings=actual)
    if mutation == "plan":
        object.__setattr__(
            bound, "plan", P.R.plan_source_recomputation(source(), selected.context, permissions(), budgets())
        )
    elif mutation == "comparison":
        object.__setattr__(bound, "comparison", synthetic_comparison(bound.plan))
    elif mutation == "verifier":
        object.__setattr__(bound, "verify_current", original_owner)
    elif mutation == "settings":
        actual["runtime"] = "foreign-runtime"
    else:
        bound.plan.source.attributes["late"] = StringAttr("changed")
    with pytest.raises(ValueError, match="changed|original|replan"):
        selected.revalidate()


def test_new_lowering_cannot_change_original_prepared_settings():
    selected = policy()
    text, actual = prepared(), settings()
    bound = selected.bind(text, settings=actual)
    with pytest.raises(ValueError, match="binding differs"):
        SourceRecomputationPreparation(text, bound, {**actual, "index_bits": 32})


@pytest.mark.parametrize("bad_settings", [{"runtime": float("nan")}, {"runtime": object()}, {"huge": "x" * (65 << 10)}])
def test_bad_settings_fail_before_original_callback_and_consume_attempt(bad_settings):
    selected = policy()
    with pytest.raises(ValueError):
        selected.bind(prepared(), settings=bad_settings)
    with pytest.raises(ValueError, match="failed"):
        selected.revalidate()


def test_unissued_or_replaced_guard_cannot_bind():
    selected = policy()
    object.__setattr__(selected, "_check", lambda: None)
    with pytest.raises(ValueError, match="guard changed"):
        selected.bind(prepared(), settings=settings())


def test_policy_and_bound_source_are_collectable_without_a_strong_registry_value():
    selected = policy()
    bound = selected.bind(prepared(), settings=settings())
    owner, module = weakref.ref(selected), weakref.ref(bound.plan.source)
    del selected, bound
    gc.collect()
    assert owner() is None and module() is None


def test_source_byte_ceiling_refuses_before_parser_or_original_callbacks():
    selected = policy()
    with pytest.raises(ValueError, match="bounded actual prepared"):
        selected.bind("x" * (P._MAX_SOURCE_BYTES + 1), settings=settings())
    with pytest.raises(ValueError, match="failed"):
        selected.revalidate()


@pytest.mark.parametrize("mutation", ["context", "settings", "plan", "limit", "reentrant"])
def test_original_verifier_is_guarded_on_its_first_entry(mutation, monkeypatch):
    holder = {}

    def verifier(**kwargs):
        if mutation == "context":
            object.__setattr__(holder["policy"].context, "execution", "foreign-runtime")
        elif mutation == "settings":
            kwargs["settings"]["index_bits"] = 32
        elif mutation == "plan":
            kwargs["plan"]["candidate_sha256"] = "0" * 64
        elif mutation == "limit":
            monkeypatch.setattr(P, "_MAX_SOURCE_BYTES", 1 << 30)
        else:
            holder["policy"].revalidate()

    holder["policy"] = selected = policy(verify_current=verifier)
    with pytest.raises(ValueError, match="changed|reentrant|replan|binding failed"):
        selected.bind(prepared(), settings=settings())


def test_all_unknown_context_values_remain_unknown_and_grant_nothing():
    def verifier(**kwargs):
        assert all(value is None for value in kwargs["plan"]["stamp"]["context"].values())
        raise ValueError("original owner has no current runtime qualification")

    selected = policy(context=PipelineContext(), verify_current=verifier)
    with pytest.raises(ValueError, match="no current runtime qualification"):
        selected.bind(prepared(), settings=settings())


@pytest.mark.parametrize("field", ["context", "permissions", "budgets"])
def test_untyped_original_inputs_refuse_at_construction(field):
    with pytest.raises(ValueError, match="exact original"):
        policy(**{field: None})


def test_catching_nested_validation_refusal_cannot_resurrect_outer_binding():
    holder = {}

    def comparison_for(**kwargs):
        with pytest.raises(ValueError, match="not reentrant"):
            holder["policy"].revalidate()
        return None

    holder["policy"] = selected = policy(comparison_for=comparison_for)
    with pytest.raises(ValueError, match="binding failed"):
        selected.bind(prepared(), settings=settings())
    with pytest.raises(ValueError, match="binding failed"):
        selected.revalidate()


@pytest.mark.parametrize("alias", ["_callback", "_check_callback"])
def test_actual_policy_callback_alias_cannot_hide_a_replaced_original_verifier(monkeypatch, alias):
    calls = []

    def verifier(**kwargs):
        calls.append("original")

    def replacement(**kwargs):
        calls.append("replacement")

    def comparison_for(**kwargs):
        monkeypatch.setattr(P, alias, lambda *args: None)
        verifier.__code__ = replacement.__code__

    selected = policy(verify_current=verifier, comparison_for=comparison_for)
    original_code = verifier.__code__
    try:
        with pytest.raises(ValueError, match="implementation changed"):
            selected.bind(prepared(), settings=settings())
        assert not calls  # Refuse before invoking the replacement owner.
    finally:
        verifier.__code__ = original_code
