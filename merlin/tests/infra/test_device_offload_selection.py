"""Package-route identity and conditional narrowing; no execution/timing qualification."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
from test_offload_plan import choose, example
from xdsl.dialects import builtin

from merlin.common.digest import sha256_file
from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.device_ir_scalability import DeviceIRBudget
from merlin.llvmlower.device_offload_selection import bind_device_offload_source
from merlin.llvmlower.device_shim import KernelAbi
from merlin.llvmlower.offload_plan import bind_source, realize_offload_plan
from merlin.perf.term import PerformanceTerm
from merlin.system.model import Link
from merlin.targetgen.capability_manifest import CapabilityManifest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.transforms.certificates import ir_text


def no_evaluation(**kwargs):
    pytest.fail("this structural test never executes an artifact or admission policy")


class Selector:
    def __init__(self):
        self.accept = True
        self.hook = None
        self.calls = 0

    def select(self, shape):
        self.calls += 1
        if self.hook is not None:
            self.hook()
        return self.accept


def fixture(tmp_path, monkeypatch, *, selector=None, numeric_policy=None, **options):
    from merlin.llvmlower import device_shim
    from merlin.system import derive
    from merlin.targetgen import target_experiment
    from merlin.targetgen.rtl import facts

    package = tmp_path / "package"
    package.mkdir()
    (package / "driver.py").write_text("# original diagnostic package tool, never executed\n")
    (package / "manifest.yaml").write_text(
        json.dumps(
            dict(
                artifact_type="mlir_oot_target_backend",
                target="neutral",
                language="python",
                authoring=dict(mode="hand_curated"),
                integrity_exempt=False,
                entrypoints=dict(tool="driver.py"),
                commands={
                    name: dict(argv=["python", "driver.py", "{input_mlir}"])
                    for name in (
                        "parse",
                        "lower_interface_to_target",
                        "emit_command_buffer",
                        "emit_target_artifact",
                        "lower_target_to_llvm",
                    )
                },
            )
        )
    )
    contract = dict(
        compute_units=[
            dict(
                accumulate=[dict(**{"in": "i8"}, weight="i8", acc="i32")],
                semantic_capabilities=[dict(family="contraction", ranks=[2])],
            )
        ]
    )
    manifest = CapabilityManifest(
        "neutral",
        "scalar",
        "command_buffer",
        "neutral-contract",
        "i8xi8_i32",
        None,
        {},
        (),
        (),
        None,
        None,
        False,
        {},
        contract,
    )
    hardware = {"interfaces": [dict(name="dma_tlb")], "generation": 1}
    abi = KernelAbi("owned_kernel", "lhs ++ rhs ++ out", "row_major", 2)
    link = Link(command_transport="command_buffer", operand_placement="pointer_args", address_translation="identity")
    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda target: manifest)
    monkeypatch.setattr(facts, "body_if_present", lambda target: hardware)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda target: abi)
    monkeypatch.setattr(derive, "link_for", lambda target, endpoint: link)
    selector = Selector() if selector is None else selector
    service = LinkedElfAdmissionService(
        "neutral", no_evaluation, ((str(Path(__file__).resolve()), sha256_file(Path(__file__))),)
    )
    routing = DeviceRouting(
        "neutral",
        package,
        "i8",
        "i32",
        select=selector.select,
        numeric_policy={"comparison": "exact"} if numeric_policy is None else numeric_policy,
        linked_elf_admission=service,
        device_ir_budget=DeviceIRBudget(),
    )
    module = example()[0].module
    tool = tmp_path / "original-runtime"
    tool.write_text("original runtime declaration, not a runtime qualification")
    bound = bind_device_offload_source(
        module,
        routing=routing,
        source_pins=((str(tool), sha256_file(tool)),),
        configuration=(("index_bits", 64),),
        **options,
    )
    alternatives = tuple(
        replace(alt, id=f"device_{index}", placement=routing.device, implementation=bound.implementation_identity)
        for index, (op, alt) in enumerate(zip(bound.source.operations, bound.source.cpu_alternatives, strict=True))
        if op.name == "linalg.generic"
    )
    plan = realize_offload_plan(
        bound.source, choose(bound.source, alternatives), device_alternatives=alternatives, context=bound.context
    )
    return bound, bound.select(plan), selector, hardware, abi, link, tool


def priced(selection, *, bounds=(70, 80), unknown=False, before=None):
    request = runpy.run_path(str(merlin_dir() / "tests/dse/test_offload_profitability.py"))["request"]

    def bind(query):
        query.update(
            stamp=selection.owner.stamp,
            original_selection_sha256=selection.owner.original_selection_sha256,
            candidate_selection_sha256=selection.plan.plan.digest,
        )
        if unknown:
            term = query["candidate_costs"]["warm"]
            query["candidate_costs"]["warm"] = PerformanceTerm.unknown(
                term.name, term.unit, term.provenance, term.validity, "no original complete cost"
            )
        if before is not None:
            before(query)

    query, evidence, _ = request(bounds=bounds, before=bind)
    cost = selection.with_complete_cost(
        **{
            key: value
            for key, value in query.items()
            if key not in ("stamp", "original_selection_sha256", "candidate_selection_sha256")
        }
    )
    return cost, evidence


@pytest.mark.parametrize(
    "mode,expected",
    [("unpriced", 2), ("explicit", 1), ("opportunity", 1), ("unknown", 2), ("overlap", 2), ("regression", 2)],
)
def test_exact_current_membership_only_narrows_with_original_choice_or_complete_cost(
    tmp_path, monkeypatch, mode, expected
):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    if mode == "explicit":
        selected = bound.select(selected.plan, explicit=True)
    elif mode != "unpriced":
        selected, _ = priced(
            selected,
            unknown=mode == "unknown",
            bounds={"overlap": (90, 110), "regression": (130, 150)}.get(mode, (70, 80)),
        )
    current = parse_mlir_text(ir_text(bound.source.module))
    operations = selected.select_for(current, routing=bound.routing)
    assert len(operations) == expected
    assert operations[-1] is tuple(current.walk())[bound.original_ordinals[-1]]
    assert all(operation not in tuple(bound.source.module.walk()) for operation in operations)
    assert len({tuple(str(value.type) for value in op.results) for op in operations}) == 1
    report = selected.record()
    assert report["selected_ordinals"] == (
        [bound.original_ordinals[-1]] if expected == 1 else list(bound.original_ordinals)
    )
    if mode == "explicit":
        assert report["route"] == "explicit_unpriced" and report["timing"] == "UNKNOWN"
        with pytest.raises(ValueError, match="cannot mix"):
            bound.select(selected.plan, comparison=priced(bound.select(selected.plan))[0].comparison, explicit=True)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "package",
        "tool",
        "facts",
        "abi",
        "transport",
        "numeric",
        "budget",
        "selector",
        "selector_code",
        "runtime",
        "configuration",
        "identity",
    ],
)
def test_original_source_route_and_dependency_changes_refuse(tmp_path, monkeypatch, change):
    bound, selected, selector, hardware, abi, link, tool = fixture(tmp_path, monkeypatch)
    if change == "source":
        bound.source.module.attributes["prov.changed"] = builtin.UnitAttr()
    elif change == "package":
        (bound.routing.package_dir / "new-file").write_text("changed source roster")
    elif change == "tool":
        (bound.routing.package_dir / "driver.py").write_text("changed tool")
    elif change == "facts":
        hardware["generation"] += 1
    elif change == "abi":
        object.__setattr__(abi, "symbol", "different_kernel")
    elif change == "transport":
        object.__setattr__(link, "operand_placement", "preload_at_base")
    elif change == "numeric":
        bound.routing.numeric_policy["comparison"] = "approximate"
    elif change == "budget":
        object.__setattr__(bound.routing.device_ir_budget, "max_bytes", 1)
    elif change == "selector":
        selector.accept = False
    elif change == "selector_code":
        object.__setattr__(bound.routing, "select", no_evaluation)
    elif change == "runtime":
        tool.write_text("different runtime declaration")
    elif change == "configuration":
        object.__setattr__(bound, "configuration", (("index_bits", 32),))
    else:
        object.__setattr__(bound, "implementation_identity", "different package")
    with pytest.raises(ValueError, match="changed|stale|differs|binding"):
        selected.revalidate()


@pytest.mark.parametrize("field", ["hardware", "numerical_policy", "resources", "execution", "library"])
def test_owner_cannot_substitute_another_source_analysis_context(tmp_path, monkeypatch, field):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    forged = bind_source(bound.source.module, entry="forward", context=replace(bound.context, **{field: "different"}))
    object.__setattr__(bound, "source", forged)
    with pytest.raises(ValueError, match="binding"):
        selected.revalidate()


@pytest.mark.parametrize("field", ["placement", "implementation", "metadata"])
def test_valid_plan_for_different_implementation_cannot_replace_original_candidate(tmp_path, monkeypatch, field):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    alternatives = tuple(
        replace(alt, **{field: (("unrealized", True),) if field == "metadata" else "other"})
        for alt in selected.plan.device_alternatives
    )
    plan = realize_offload_plan(
        bound.source, choose(bound.source, alternatives), device_alternatives=alternatives, context=bound.context
    )
    with pytest.raises(ValueError, match="implementation differs"):
        bound.select(plan, explicit=True)
    with pytest.raises(ValueError, match="fields changed"):
        replace(selected, plan=plan).select_for(bound.source.module)


def test_original_selector_rejects_source_expansion_and_nonboolean_answers(tmp_path, monkeypatch):
    bound, selected, selector, *_ = fixture(tmp_path, monkeypatch)
    selector.accept = False
    with pytest.raises(ValueError, match="outcomes changed"):
        selected.revalidate()
    selector.accept = 1
    with pytest.raises(ValueError, match="exact Boolean"):
        selected.revalidate()


@pytest.mark.parametrize("phase", ["selector", "cost"])
@pytest.mark.parametrize("mutation", ["explicit", "comparison", "routing", "plan"])
def test_original_callbacks_cannot_change_live_selection_after_snapshot(tmp_path, monkeypatch, phase, mutation):
    bound, selected, selector, *_ = fixture(tmp_path, monkeypatch)
    if phase == "cost":
        selected, _ = priced(selected, unknown=True)
        decision, original = selected.comparison, selected.comparison._check

        def changed():
            result = original()
            if armed["value"]:
                mutate()
            return result

        # Reissue with the actual original verifier wrapper before arming it.
        armed = {"value": False}
        object.__setattr__(decision, "_check", changed)
        selected = bound.select(selected.plan, decision)
    else:
        armed = {"value": False}

    def mutate():
        if not armed["value"]:
            return
        if mutation == "explicit":
            object.__setattr__(selected, "explicit", True)
        elif mutation == "comparison":
            object.__setattr__(selected, "comparison", None)
        elif mutation == "routing":
            bound.routing.numeric_policy["comparison"] = "changed"
        else:
            object.__setattr__(selected, "plan", replace(selected.plan))

    selector.hook = mutate if phase == "selector" else None
    armed["value"] = True
    # comparison=None -> None is intentionally not a mutation; test removal only
    # for the cost callback and use an invalid object for the unpriced selector.
    if phase == "selector" and mutation == "comparison":

        def mutate():
            object.__setattr__(selected, "comparison", object())

        selector.hook = mutate
    with pytest.raises(ValueError, match="changed|conflict|verifier"):
        selected.record()


@pytest.mark.parametrize("limit,value", [("max_package_files", 1), ("max_package_bytes", 1), ("max_operations", 1)])
def test_bounded_inputs_refuse_before_original_selector(tmp_path, monkeypatch, limit, value):
    selector = Selector()
    with pytest.raises(ValueError, match="budget"):
        fixture(tmp_path, monkeypatch, selector=selector, **{limit: value})
    assert selector.calls == 0


def test_symlink_and_missing_tool_refuse_current_route(tmp_path, monkeypatch):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    tool = bound.routing.package_dir / "driver.py"
    original = tool.read_bytes()
    tool.unlink()
    with pytest.raises(ValueError):
        selected.revalidate()
    external = tmp_path / "outside.py"
    external.write_bytes(original)
    tool.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        selected.revalidate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("granularity", "group"),
        ("exact_selection", object()),
        ("catalog_manifest", "unadmitted"),
        ("catalog_builder", no_evaluation),
        ("select", None),
        ("linked_elf_admission", None),
    ],
)
def test_economic_scope_cannot_mix_original_alternate_routes(tmp_path, monkeypatch, field, value):
    bound, _, *_ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="only singleton|cannot mix|selector|required"):
        bound.revalidate(routing=replace(bound.routing, **{field: value}))


def test_selector_acceptance_outside_bound_entry_is_not_silently_dropped(tmp_path, monkeypatch):
    bound, _, *_ = fixture(tmp_path, monkeypatch)
    module = bound.source.module.clone()
    helper = module.body.block.first_op.clone()
    helper.sym_name = builtin.StringAttr("helper")
    module.body.block.add_op(helper)
    module.verify()
    with pytest.raises(ValueError, match="outside the bound flat entry"):
        bind_device_offload_source(module, routing=bound.routing)


def test_candidate_cannot_add_operations_declined_by_original_selector(tmp_path, monkeypatch):
    selector = Selector()
    selector.accept = False
    with pytest.raises(ValueError, match="expands the original selector"):
        fixture(tmp_path, monkeypatch, selector=selector)


def test_same_shape_current_source_rewiring_is_not_a_new_placement(tmp_path, monkeypatch):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    current = bound.source.module.clone()
    operation = tuple(current.walk())[bound.original_ordinals[-1]]
    previous = tuple(current.walk())[bound.original_ordinals[0]]
    assert previous.results[0].type == operation.operands[2].type
    operation.operands = (*operation.operands[:2], previous.results[0])
    current.verify()
    # Only the same-typed initializer edge changes: the second contraction now
    # accumulates the first result. Shape and the original selector stay equal.
    with pytest.raises(ValueError, match="differs|changed"):
        selected.select_for(current)


def test_finite_numerical_thresholds_are_identity_inputs_not_new_permissions(tmp_path, monkeypatch):
    policy = dict(comparison="exact", atol=0.0, rtol=0.0, cos=1.0)
    bound, selected, *_ = fixture(tmp_path, monkeypatch, numeric_policy=policy)
    assert json.loads(bound._record)["routing"]["numeric_policy"] == policy
    selected.revalidate()
    policy["atol"] = 0.001
    with pytest.raises(ValueError, match="context changed"):
        selected.revalidate()


def test_record_checks_the_actual_route_without_mutating_original_owner(tmp_path, monkeypatch):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    current = replace(bound.routing, numeric_policy=dict(bound.routing.numeric_policy))
    assert selected.record(routing=current) == selected.record()
    current.numeric_policy["comparison"] = "different actual invocation"
    assert bound.routing.numeric_policy["comparison"] == "exact"
    with pytest.raises(ValueError, match="context changed"):
        selected.record(routing=current)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numerical_thresholds_do_not_bind(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="unsupported original configuration"):
        fixture(tmp_path, monkeypatch, numeric_policy=dict(comparison="exact", atol=value))


def test_whole_module_digest_does_not_make_a_different_entry_the_same_plan(tmp_path, monkeypatch):
    bound, _, selector, *_ = fixture(tmp_path, monkeypatch)
    module = bound.source.module.clone()
    helper = module.body.block.first_op.clone()
    helper.sym_name = builtin.StringAttr("helper")
    module.body.block.add_op(helper)
    # Decline every original device operation so the whole-module selector mask
    # is empty. A helper-only CPU plan must still not stand for forward's graph.
    selector.accept = False
    other = bind_device_offload_source(module, routing=bound.routing)
    helper_source = bind_source(module, entry="helper", context=other.context)
    helper_plan = realize_offload_plan(
        helper_source, choose(helper_source, ()), device_alternatives=(), context=other.context
    )
    with pytest.raises(ValueError, match="another original entry or graph"):
        other.select(helper_plan, explicit=True)


@pytest.mark.parametrize("change", ["numeric", "resources", "source"])
def test_context_only_check_rejects_stale_original_inputs(tmp_path, monkeypatch, change):
    bound, selected, *_ = fixture(tmp_path, monkeypatch)
    if change == "numeric":
        bound.routing.numeric_policy["comparison"] = "changed"
    elif change == "resources":
        object.__setattr__(bound.routing, "device_ir_budget", DeviceIRBudget(max_bytes=7))
    else:
        bound.source.module.attributes["prov.changed"] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="changed|differs|stale"):
        selected.revalidate_context(routing=bound.routing)


def test_context_only_check_never_calls_original_selector_cost_or_policy(tmp_path, monkeypatch):
    bound, selected, selector, *_ = fixture(tmp_path, monkeypatch)
    selected, _ = priced(selected)

    def forbidden(*args, **kwargs):
        pytest.fail("context-only freshness must not call an original evaluator")

    # Retain the real original callback identities while making their bodies
    # fail if executed. Checking implementation selection must remain legal.
    original = selected.comparison._check
    armed = {"value": False}

    def checked_cost():
        if armed["value"]:
            forbidden()
        return original()

    object.__setattr__(selected.comparison, "_check", checked_cost)
    selected = bound.select(selected.plan, selected.comparison)
    armed["value"] = True
    selector.hook = forbidden
    monkeypatch.setattr(LinkedElfAdmissionService, "evaluate", forbidden)
    monkeypatch.setattr(LinkedElfAdmissionService, "revalidate", forbidden)
    before = selector.calls
    bound.revalidate_context(routing=bound.routing)
    selected.revalidate_context(routing=bound.routing)
    assert selector.calls == before
