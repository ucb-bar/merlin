"""Pure fanout intake keeps original SSA, callbacks and selected ceilings."""

from dataclasses import replace

import pytest
from test_source_pointwise_lowering import fixture as pointwise_selection
from xdsl.dialects import func

from merlin.compile.scheduling.multioutput_fusion_check import MultiOutputCompositionLaw
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.source_multioutput import SourceMultiOutputLowering, discover_multioutput
from merlin.transforms.certificates import ir_text


def source(*, observed=False, barrier=False):
    shape = "tensor<2x3xi32>"
    identity = "affine_map<(i,j)->(i,j)>"

    def generic(name, left, right, operation):
        return f"""%{name}=linalg.generic {{indexing_maps=[{identity},{identity},{identity}],
        iterator_types=["parallel","parallel"]}}
        ins(%{left},%{right}:{shape},{shape}) outs(%empty_{name}:{shape}) {{
        ^bb0(%a:i32,%b:i32,%unused:i32):
        %value=arith.{operation} %a,%b:i32
        linalg.yield %value:i32
        }}->{shape}"""

    types = ",".join([shape] * (3 if observed else 2))
    returns = "%e0,%e1,%p" if observed else "%e0,%e1"
    module = parse_mlir_text(f"""module {{
    func.func private @opaque()
    func.func @forward(%x:{shape},%y:{shape})->({types}) {{
    %empty_p=tensor.empty():{shape}
    %empty_e0=tensor.empty():{shape}
    %empty_e1=tensor.empty():{shape}
    {generic("p", "x", "y", "addi")}
    {generic("e0", "p", "x", "addi")}
    {"func.call @opaque():()->()" if barrier else ""}
    {generic("e1", "p", "y", "muli")}
    func.return {returns}:{types}
    }} }}""")
    module.verify()
    return module


def original_callbacks(tmp_path, on_validate, on_map):
    path = tmp_path / "original_multioutput_callbacks.py"
    text = "def validate():\n    on_validate()\n\ndef lower(source):\n    return on_map(source)\n"
    path.write_text(text)
    scope = {"on_validate": on_validate, "on_map": on_map}
    exec(compile(text, str(path), "exec"), scope)
    return scope["validate"], scope["lower"], path


def selection(tmp_path, mapper):
    original, singles, prepared = pointwise_selection(tmp_path)
    adapter = original.adapter
    law = MultiOutputCompositionLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "original-neutral-frontend-composition-premises",
        ("producer", "original-producer-form"),
        (("epilogue0", "original-epilogue0-form"), ("epilogue1", "original-epilogue1-form")),
        complete_projected_effects=True,
        synchronous_private_composition=True,
        suppressed_intermediate_transfers_no_faults=True,
        suppressed_intermediate_transfers_unobserved=True,
    )
    return (
        replace(
            original,
            borrowed_effects=BorrowedBufferEffects("original-neutral-frontend-borrowed-effects", True, True, True),
            multi_output=SourceMultiOutputLowering(mapper, law),
        ),
        singles,
        prepared,
    )


@pytest.mark.parametrize("mutation", ["clone_ssa", "validator_code", "mapper_code", "callback_file"])
def test_first_validate_cannot_retarget_equal_text_or_original_implementations(tmp_path, mutation):
    module = source()
    before = ir_text(module)
    validations, mappings = [], []

    def validate_hook():
        validations.append(1)
        if len(validations) != 1:
            return
        if mutation == "clone_ssa":
            original = next(
                op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward"
            )
            clone = original.clone()
            module.body.block.insert_op_before(clone, original)
            module.body.block.erase_op(original)
            module.verify()
            assert ir_text(module) == before
        elif mutation == "validator_code":
            validate.__code__ = replacement.__code__
        elif mutation == "mapper_code":
            mapper.__code__ = replacement_mapper.__code__
        else:
            path.write_bytes(path.read_bytes() + b"# changed original implementation\n")

    def map_hook(value):
        mappings.append(value)
        return None

    validate, mapper, path = original_callbacks(tmp_path, validate_hook, map_hook)
    selected, singles, prepared = selection(tmp_path, mapper)
    validate_code, mapper_code = validate.__code__, mapper.__code__

    def replacement():
        raise AssertionError("changed validator must not be dispatched")

    def replacement_mapper(value):
        raise AssertionError("changed mapper must not be dispatched")

    try:
        with pytest.raises(ValueError, match="changed|replan"):
            discover_multioutput(module, selected, validate)
    finally:
        validate.__code__, mapper.__code__ = validate_code, mapper_code
    assert validations == [1] and not mappings and not singles and not prepared


@pytest.mark.parametrize(
    "mutation", ["budgets", "selection", "mapper_code", "callback_file", "source_graph", "source_scalar_ir"]
)
def test_original_mapping_cannot_change_selected_policy_before_none_fallback(tmp_path, mutation):
    module, mappings = source(), []

    def map_hook(value):
        mappings.append(value)
        if mutation == "budgets":
            object.__setattr__(selected.budgets, "max_fusions", selected.budgets.max_fusions + 1)
        elif mutation == "selection":
            object.__setattr__(selected, "max_source_elements", selected.max_source_elements + 1)
        elif mutation == "mapper_code":
            mapper.__code__ = replacement.__code__
        elif mutation == "source_graph":
            object.__setattr__(value.producer.graph, "output_bits", value.producer.graph.output_bits + 8)
        elif mutation == "source_scalar_ir":
            object.__setattr__(value.producer, "scalar_ir", value.producer.scalar_ir + b"\n")
        else:
            path.write_bytes(path.read_bytes() + b"# changed mapper source\n")
        return None

    validate, mapper, path = original_callbacks(tmp_path, lambda: None, map_hook)
    selected, singles, prepared = selection(tmp_path, mapper)
    mapper_code = mapper.__code__

    def replacement(value):
        raise AssertionError("changed mapper must not be called again")

    try:
        with pytest.raises(ValueError, match="changed|replan"):
            discover_multioutput(module, selected, validate)
    finally:
        mapper.__code__ = mapper_code
    assert len(mappings) == 1 and not singles and not prepared


@pytest.mark.parametrize("unsupported", ["observed_producer", "opaque_barrier", "default_none", "excluded_member"])
def test_unsupported_fanout_preserves_original_without_mapping(tmp_path, unsupported):
    module = source(observed=unsupported == "observed_producer", barrier=unsupported == "opaque_barrier")
    before = ir_text(module)
    calls = []
    validate, mapper, _ = original_callbacks(tmp_path, lambda: calls.append("validate"), lambda _: calls.append("map"))
    selected, singles, prepared = selection(tmp_path, mapper)
    if unsupported == "default_none":
        selected = replace(selected, multi_output=None)
    excluded = frozenset()
    if unsupported == "excluded_member":
        excluded = frozenset(op for op in module.walk() if op.name == "linalg.generic")
    assert discover_multioutput(module, selected, validate, excluded_operations=excluded) == ((), ())
    assert ir_text(module) == before and "map" not in calls and not singles and not prepared
    if unsupported == "default_none":
        assert not calls


def test_valid_current_source_reaches_original_mapper_and_none_keeps_singles(tmp_path):
    module, mappings = source(), []
    before = ir_text(module)
    validate, mapper, _ = original_callbacks(tmp_path, lambda: None, lambda value: mappings.append(value))
    selected, singles, prepared = selection(tmp_path, mapper)
    regions, declined = discover_multioutput(module, selected, validate)
    assert regions == () and len(declined) == 1 and "no mapping" in declined[0]
    assert len(mappings) == 1 and len(mappings[0].outputs) == 2
    assert ir_text(module) == before and not singles and not prepared


@pytest.mark.parametrize(
    "permission",
    [
        "complete_projected_effects",
        "synchronous_private_composition",
        "suppressed_intermediate_transfers_no_faults",
        "suppressed_intermediate_transfers_unobserved",
    ],
)
def test_missing_composition_permission_declines_before_mapping(tmp_path, permission):
    module, calls = source(), []
    before = ir_text(module)
    validate, mapper, _ = original_callbacks(tmp_path, lambda: None, lambda _: calls.append("mapping"))
    selected, singles, prepared = selection(tmp_path, mapper)
    law = replace(selected.multi_output.composition_law, **{permission: False})
    selected = replace(selected, multi_output=replace(selected.multi_output, composition_law=law))
    regions, declined = discover_multioutput(module, selected, validate)
    assert regions == () and len(declined) == 1 and "unavailable" in declined[0]
    assert ir_text(module) == before and not calls and not singles and not prepared


@pytest.mark.parametrize(
    "identity", ["configuration_identity", "numerical_policy_identity", "adapter_qualification_identity"]
)
def test_foreign_original_law_refuses_before_mapping(tmp_path, identity):
    module, calls = source(), []
    validate, mapper, _ = original_callbacks(tmp_path, lambda: None, lambda _: calls.append("mapping"))
    selected, singles, prepared = selection(tmp_path, mapper)
    law = replace(selected.multi_output.composition_law, **{identity: "different-original-selection"})
    selected = replace(selected, multi_output=replace(selected.multi_output, composition_law=law))
    before = ir_text(module)
    with pytest.raises(ValueError, match="differs from the original selected adapter"):
        discover_multioutput(module, selected, validate)
    assert ir_text(module) == before and not calls and not singles and not prepared


def test_missing_typed_law_refuses_at_original_selection(tmp_path):
    _, mapper, _ = original_callbacks(tmp_path, lambda: None, lambda _: pytest.fail("mapping must not run"))
    selected, singles, prepared = selection(tmp_path, mapper)
    with pytest.raises(ValueError, match="original typed composition law"):
        replace(selected.multi_output, composition_law=None)
    assert not singles and not prepared


@pytest.mark.parametrize("missing", ["storage", "fusion", "scalar_qualification", "private_ownership", "disjointness"])
def test_missing_original_pointwise_admission_keeps_the_default_source_route(tmp_path, missing):
    from merlin.llvmlower.source_pointwise import discover_pointwise

    module, calls = source(), []
    before = ir_text(module)
    validate, mapper, _ = original_callbacks(
        tmp_path, lambda: calls.append("validate"), lambda _: calls.append("mapping")
    )
    selected, singles, prepared = selection(tmp_path, mapper)
    original_contract = None
    if missing in ("storage", "fusion"):
        original_contract = getattr(selected.selected_contracts, missing)
        object.__setattr__(selected.selected_contracts, missing, None)
    else:
        key, value = {
            "scalar_qualification": ("scalar_qualification_identity", None),
            "private_ownership": ("private_ownership", False),
            "disjointness": ("external_buffers_disjoint", False),
        }[missing]
        selected = replace(selected, **{key: value})
    try:
        regions, declined = discover_pointwise(module, selected, validate)
    finally:
        if original_contract is not None:
            object.__setattr__(selected.selected_contracts, missing, original_contract)
    assert regions == () and declined
    assert ir_text(module) == before and not calls and not singles and not prepared


def test_first_validation_cannot_enable_missing_law_via_method_replacement(tmp_path, monkeypatch):
    module, mapped = source(), []
    before = ir_text(module)
    validations = []

    def validate_hook():
        validations.append(1)
        if len(validations) == 1:
            monkeypatch.setattr(MultiOutputCompositionLaw, "flags", lambda self: (True,) * 4)

    def map_hook(value):
        mapped.append(value)
        (tmp_path / "unavailable-law-mapper-reached.txt").write_text(
            f"complete_projected_effects={missing.complete_projected_effects}\nmapping_calls={len(mapped)}\n"
        )

    validate, mapper, _ = original_callbacks(tmp_path, validate_hook, map_hook)
    selected, singles, prepared = selection(tmp_path, mapper)
    missing = replace(selected.multi_output.composition_law, complete_projected_effects=False)
    selected = replace(selected, multi_output=replace(selected.multi_output, composition_law=missing))
    assert missing.complete_projected_effects is False
    with pytest.raises(ValueError, match="implementation code changed|checker or implementation"):
        discover_multioutput(module, selected, validate)
    assert validations == [1] and not mapped and not singles and not prepared
    assert ir_text(module) == before


@pytest.mark.parametrize("helper", ["reader_returns_original_members", "checker_replaced_with_noop"])
def test_first_validation_cannot_replace_the_function_snapshot_boundary(tmp_path, monkeypatch, helper):
    from merlin.llvmlower import source_multioutput as frontend

    module, mapped, validations = source(), [], []
    original_members = frontend._functions()

    def validate_hook():
        validations.append(1)
        if len(validations) != 1:
            return
        if helper == "reader_returns_original_members":
            monkeypatch.setattr(frontend, "_functions", lambda: original_members)
        else:
            monkeypatch.setattr(frontend, "check_functions", lambda snapshot: None)

    def map_hook(value):
        mapped.append(value)
        (tmp_path / "changed-checker-mapper-reached.txt").write_text(f"mapping_calls={len(mapped)}\n")

    validate, mapper, _ = original_callbacks(tmp_path, validate_hook, map_hook)
    selected, singles, prepared = selection(tmp_path, mapper)
    before = ir_text(module)
    with pytest.raises(ValueError, match="changed|replan"):
        discover_multioutput(module, selected, validate)
    assert validations == [1] and not mapped and not singles and not prepared
    assert ir_text(module) == before
