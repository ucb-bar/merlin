"""Original bounded control custody before ordinary source/provider lowering."""

import json
import runpy
from dataclasses import replace
from threading import Event, Thread

import pytest
from test_dense_source import source
from test_source_dense_lowering import selection
from test_source_offload_cost import cost_case

from merlin.common import mlir_query
from merlin.common.digest import sha256_file
from merlin.common.paths import merlin_dir
from merlin.llvmlower import lower
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.source_choices import REQUEST_SCHEMA, SourceChoiceResult, SourceDenseChoice, SourceDenseChoices
from merlin.llvmlower.source_dense import ordinary_dense_route_identity
from merlin.llvmlower.source_model import bind_source_lowerings
from merlin.transforms.certificates import ir_text


def owner(tmp_path, *, choices=None, config=None, module=None, **limits):
    if config is None:
        config, _, _ = selection(tmp_path)
    if module is None:
        module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    path = tmp_path / "fixed.mlir"
    path.write_text(ir_text(module))
    choices = choices or (
        SourceDenseChoice("original"),
        SourceDenseChoice("reverse", (("outer_order", ("n", "m")),)),
    )
    return SourceDenseChoices(
        path, entry="forward", baseline=config, choices=choices, output_root=tmp_path / "private", **limits
    )


def request(original, choice="original"):
    return {"schema": REQUEST_SCHEMA, "source_sha256": original.source_sha256, "choice_id": choice}


class PreparedOnly(RuntimeError):
    pass


def stop_after_checked_preparation(monkeypatch, seen):
    def prepare(path, directory, *, shared_source_lowerings):
        (config,) = shared_source_lowerings
        transform, _ = bind_source_lowerings((config,), None, None, None)
        prepared = apply_prepared_model_transform(path, directory / "checked", transform)
        mlir_query.parse(prepared).verify()
        seen.append((config, prepared))
        raise PreparedOnly("unit boundary stops after actual checked preparation, before native compilation")

    monkeypatch.setattr(lower, "lower_model_file", prepare)


def test_original_source_and_control_roster_are_finite_and_path_free(tmp_path):
    original = owner(tmp_path)
    advertised = original.advertisement()
    assert advertised["source_sha256"] == sha256_file(original.source)
    assert [row["choice_id"] for row in advertised["choices"]] == ["original", "reverse"]
    assert str(tmp_path) not in json.dumps(advertised)
    assert "qualification" not in advertised["choices"][0]
    assert original.results == ()
    assert original.source.read_text() == ir_text(mlir_query.parse(original.source))


@pytest.mark.parametrize("choice,order", [("original", ("m", "n")), ("reverse", ("n", "m"))])
def test_chosen_controls_reach_actual_existing_checked_source_lifecycle(tmp_path, monkeypatch, choice, order):
    original = owner(tmp_path)
    seen = []
    stop_after_checked_preparation(monkeypatch, seen)
    with pytest.raises(PreparedOnly):
        original.compile(request(original, choice))
    assert len(seen) == 1
    config, prepared = seen[0]
    assert config.outer_order == order
    assert config.prepare is original.baseline.prepare and config.build is original.baseline.build
    assert config.adapter is original.baseline.adapter and config.storage is original.baseline.storage
    assert config.selected_contracts is original.baseline.selected_contracts
    assert (prepared.parent / "source_dense.json").is_file()
    assert original.results == () and not list(original.output_root.rglob("source_choice.json"))


@pytest.mark.parametrize(
    "field", ["source", "output_dir", "callbacks", "semantics", "effects", "qualification", "costs"]
)
def test_request_cannot_select_paths_implementation_or_authority(tmp_path, field):
    original = owner(tmp_path)
    payload = {**request(original), field: "candidate-selected"}
    with pytest.raises(ValueError, match="unknown or missing"):
        original.compile(payload)
    assert not list(original.output_root.iterdir())


@pytest.mark.parametrize("changed", ["schema", "source_sha256", "choice_id", "typed", "bytes"])
def test_bad_request_identity_type_and_budget_refuse_before_lowering(tmp_path, changed):
    original = owner(tmp_path, **({"max_request_bytes": 32} if changed == "bytes" else {}))
    payload = request(original)
    if changed == "typed":
        payload["choice_id"] = ["original"]
    elif changed != "bytes":
        payload[changed] = "unknown"
    with pytest.raises(ValueError, match="schema|identity|unavailable|strings|byte budget"):
        original.compile(payload)
    assert original.results == () and not list(original.output_root.iterdir())


@pytest.mark.parametrize("field", ["resources", "adapter", "prepare", "build", "private_ownership", "numerical_policy"])
def test_choices_cannot_override_capabilities_or_semantics(field):
    with pytest.raises(ValueError, match="non-tuning"):
        SourceDenseChoice("bad", ((field, True),))


@pytest.mark.parametrize(
    "controls",
    [
        (("outer_order", ("k", "m")),),
        (("borrow_contiguous", 1),),
        (("max_convolution_packing_bytes", -1),),
        (("outer_order", ("m", "n")), ("outer_order", ("n", "m"))),
    ],
)
def test_control_types_and_duplicate_names_are_closed(controls):
    with pytest.raises(ValueError, match="typed bounded|repeated"):
        SourceDenseChoice("bad", controls)


@pytest.mark.parametrize("mutation", ["source", "resource", "baseline", "later_choice", "facts", "callback_code"])
def test_original_roster_source_resources_and_callback_freshness(tmp_path, mutation):
    original = owner(tmp_path)
    function, old_code = original.baseline.prepare, original.baseline.prepare.__code__
    try:
        if mutation == "source":
            original.source.write_bytes(original.source.read_bytes() + b"\n")
        elif mutation == "resource":
            view = original.baseline.storage.lhs
            object.__setattr__(view, "capacity", view.capacity + 1)
        elif mutation == "baseline":
            object.__setattr__(original.baseline, "private_ownership", False)
        elif mutation == "later_choice":
            object.__setattr__(original.choices[1], "controls", (("borrow_contiguous", False),))
        elif mutation == "facts":
            path = original.baseline.selected_contracts.facts_path
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            function.__code__ = old_code.replace(co_name="changed_in_memory")
        with pytest.raises(ValueError, match="changed|stale|differs"):
            original.compile(request(original))
    finally:
        function.__code__ = old_code
    assert original.results == ()


@pytest.mark.parametrize("mutation", ["later_choice", "request", "recursive"])
def test_actual_prepare_callback_cannot_change_frozen_choice_or_request(tmp_path, monkeypatch, mutation):
    shared = {}

    def mutate(bindings, directory):
        if mutation == "later_choice":
            object.__setattr__(shared["owner"].choices[1], "controls", (("borrow_contiguous", False),))
        elif mutation == "request":
            shared["request"]["choice_id"] = "reverse"
        else:
            with pytest.raises(ValueError, match="concurrently or recursively"):
                shared["owner"].compile(shared["request"])

    config, _, _ = selection(tmp_path, prepare=mutate)
    original = owner(tmp_path, config=config)
    shared.update(owner=original, request=request(original))
    stop_after_checked_preparation(monkeypatch, [])
    error = PreparedOnly if mutation == "recursive" else ValueError
    with pytest.raises(error):
        original.compile(shared["request"])
    assert original.results == () and not list(original.output_root.rglob("source_choice.json"))


@pytest.mark.parametrize("kind", ["outside", "root", "existing", "symlink"])
def test_original_output_location_is_fresh_private_descendant(tmp_path, kind):
    original = owner(tmp_path)
    output = original.output_root / "next"
    if kind == "outside":
        output = tmp_path / "outside"
    elif kind == "root":
        output = original.output_root
    elif kind == "existing":
        output.mkdir()
    else:
        output.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="fresh private"):
        original.compile(request(original), output)
    assert original.results == ()


def test_device_selection_keeps_existing_ordinary_unsupported_diagnostic(tmp_path):
    config, _, seen = selection(tmp_path)
    original = owner(tmp_path, config=replace(config, execution_domain="device"))
    with pytest.raises(ValueError, match="native shared source lowering supports only host-domain"):
        original.compile(request(original))
    assert not seen and original.results == ()


def test_current_plan_never_rebinds_to_changed_controls(tmp_path):
    config, module, _, _, _, _ = cost_case(tmp_path)
    baseline = replace(config, offload_plan=None, offload_comparison=None)
    choice = SourceDenseChoice("stale", (("borrow_contiguous", False),), config.offload_plan)
    with pytest.raises(ValueError, match="context|stale|differs"):
        owner(tmp_path, config=baseline, module=module, choices=(choice,))


def test_unknown_complete_cost_keeps_existing_original_route(tmp_path, monkeypatch):
    config, module, seen, decision, _, _ = cost_case(tmp_path, unknown=True)
    baseline = replace(config, offload_plan=None, offload_comparison=None)
    original = owner(
        tmp_path,
        config=baseline,
        module=module,
        choices=(SourceDenseChoice("unknown", offload_plan=config.offload_plan, offload_comparison=decision),),
    )
    products = []
    stop_after_checked_preparation(monkeypatch, products)
    with pytest.raises(PreparedOnly):
        original.compile(request(original, "unknown"))
    assert len(seen) == 2
    record = json.loads((products[0][1].parent / "source_dense.json").read_text())
    assert record["offload_placement"][0]["actual_route"] == "original"


def test_constructor_freezes_later_choice_before_any_original_cost_callback(tmp_path):
    config, module, _, _, _, _ = cost_case(tmp_path)
    baseline = replace(config, offload_plan=None, offload_comparison=None)
    plan = config.offload_plan
    fixture = runpy.run_path(str(merlin_dir() / "tests/dse/test_offload_profitability.py"))

    def bind(query):
        query.update(
            stamp=plan.stamp,
            original_selection_sha256=ordinary_dense_route_identity(plan.stamp),
            candidate_selection_sha256=plan.plan.digest,
        )

    query, _, _ = fixture["request"](before=bind)
    verify, shared = query["verify_original"], {"armed": False}

    def verify_and_mutate(**arguments):
        verify(**arguments)
        if shared["armed"]:
            object.__setattr__(shared["later"], "controls", (("outer_order", ("n", "m")),))

    query["verify_original"] = verify_and_mutate
    priced = replace(baseline, offload_plan=plan).with_complete_offload_cost(
        **{
            key: value
            for key, value in query.items()
            if key not in ("stamp", "original_selection_sha256", "candidate_selection_sha256")
        }
    )
    later = SourceDenseChoice("later")
    shared.update(armed=True, later=later)
    with pytest.raises(ValueError, match="roster changed during construction"):
        owner(
            tmp_path,
            config=baseline,
            module=module,
            choices=(
                SourceDenseChoice("priced", offload_plan=plan, offload_comparison=priced.offload_comparison),
                later,
            ),
        )


def test_original_entry_must_exist_and_match_preissued_plan(tmp_path):
    config, module, _, _, _, _ = cost_case(tmp_path)
    path = tmp_path / "source.mlir"
    path.write_text(ir_text(module))
    with pytest.raises(ValueError, match="entry is absent"):
        SourceDenseChoices(
            path,
            entry="absent",
            baseline=config,
            choices=(SourceDenseChoice("only"),),
            output_root=tmp_path / "private",
        )


def test_constructor_binds_parsed_raw_bytes_even_if_original_file_changes(tmp_path, monkeypatch):
    config, _, _ = selection(tmp_path)
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    path = tmp_path / "fixed.mlir"
    path.write_text(ir_text(module))
    parse = mlir_query.parse

    def parse_and_change(value):
        result = parse(value)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(mlir_query, "parse", parse_and_change)
    with pytest.raises(ValueError, match="original source changed during construction"):
        SourceDenseChoices(
            path,
            entry="forward",
            baseline=config,
            choices=(SourceDenseChoice("only"),),
            output_root=tmp_path / "private",
        )


def test_concurrent_calls_cannot_share_original_provider_state(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def wait(bindings, directory):
        entered.set()
        assert release.wait(10)

    config, _, _ = selection(tmp_path, prepare=wait)
    original = owner(tmp_path, config=config)
    stop_after_checked_preparation(monkeypatch, [])
    outcomes = []

    def first_call():
        try:
            original.compile(request(original))
        except BaseException as exc:
            outcomes.append(exc)

    thread = Thread(target=first_call)
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(ValueError, match="concurrently or recursively"):
            original.compile(request(original, "reverse"))
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and len(outcomes) == 1 and isinstance(outcomes[0], PreparedOnly)
    assert original.results == ()


@pytest.mark.parametrize("kind", ["roster", "source", "symlink", "budget", "duplicate"])
def test_original_constructor_boundaries_are_checked(tmp_path, kind):
    config, _, _ = selection(tmp_path)
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    path = tmp_path / "fixed.mlir"
    path.write_text(ir_text(module))
    choices = (SourceDenseChoice("only"),)
    limits = {}
    if kind == "roster":
        choices = ()
    elif kind == "source":
        limits["max_source_bytes"] = 1
    elif kind == "symlink":
        alias = tmp_path / "linked.mlir"
        alias.symlink_to(path)
        path = alias
    elif kind == "budget":
        limits["max_choices"] = True
    else:
        choices = (SourceDenseChoice("same"), SourceDenseChoice("same"))
    with pytest.raises(ValueError, match="bounded nonempty|byte budget|unlinked|positive|unique"):
        SourceDenseChoices(
            path, entry="forward", baseline=config, choices=choices, output_root=tmp_path / "private", **limits
        )


def test_unissued_or_serialized_result_cannot_mint_compilation_custody(tmp_path):
    original = owner(tmp_path)
    forged = SourceChoiceResult(
        lower.LowerResult(tmp_path, tmp_path / "none.ll"),
        tmp_path / "none.json",
        original,
        "{}",
        (),
        "{}",
        (),
    )
    with pytest.raises(ValueError, match="original compilation binding"):
        forged.record()
    assert original.results == ()


@pytest.mark.parametrize("field", ["max_convolution_candidates", "max_convolution_added_operations"])
def test_choices_cannot_raise_original_search_or_growth_limits(tmp_path, field):
    config, _, _ = selection(tmp_path, **{field: 2})
    with pytest.raises(ValueError, match="original search or growth budget"):
        owner(tmp_path, config=config, choices=(SourceDenseChoice("raise", ((field, 3),)),))
    original = owner(tmp_path, config=config, choices=(SourceDenseChoice("restrict", ((field, 1),)),))
    assert original.advertisement()["choices"][0]["controls"][field] == 1


@pytest.mark.parametrize("cap", [None, 65])
def test_choices_cannot_raise_or_remove_original_packing_cap(tmp_path, cap):
    config, _, _ = selection(tmp_path, max_convolution_packing_bytes=64)
    with pytest.raises(ValueError, match="raise or remove its original packing cap"):
        owner(tmp_path, config=config, choices=(SourceDenseChoice("raise", (("max_convolution_packing_bytes", cap),)),))
