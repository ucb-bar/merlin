"""Typed source graph extraction feeds the existing physical layout planner."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_layout_planning import layout_from_id, layout_id, plan_islands
from merlin.perf.physical_layout import _evaluate
from merlin.transforms.certificates import ir_text


def _type(shape, dtype="i32"):
    return "tensor<" + "x".join((*map(str, shape), dtype)) + ">"


def _source(
    shape,
    permutation,
    *,
    dtype="i32",
    generic_copy=False,
    barrier=False,
    broadcast=False,
    init_read=False,
    shared=False,
    bias_rank=None,
    bias_axis=None,
):
    rank = len(shape)
    moved = tuple(shape[axis] for axis in permutation)
    original, transposed = _type(shape, dtype), _type(moved, dtype)
    identity = "affine_map<(" + ",".join("d" + str(i) for i in range(rank)) + ")->("

    def mapping(axes):
        return identity + ",".join(axes) + ")>"

    same = mapping(["d" + str(i) for i in range(rank)])
    inverse = tuple(permutation.index(i) for i in range(rank))
    inputs, operand_types = "%t", transposed
    args = f"%v:{dtype},%unused:{dtype}"
    extra_arguments = ""
    maps = [same, same]
    if broadcast:
        bias_axis = rank - 1 if bias_axis is None else bias_axis
        bias_shape = tuple(size if axis == bias_axis else 1 for axis, size in enumerate(moved))
        bias_map = mapping(["d" + str(i) if i == bias_axis else "0" for i in range(rank)])
        if bias_rank is not None:
            bias_shape = moved[-bias_rank:]
            bias_map = mapping(["d" + str(i) for i in range(rank - bias_rank, rank)])
        bias_type = _type(bias_shape, dtype)
        extra_arguments = f",%bias:{bias_type},%scalar:tensor<{dtype}>"
        inputs, operand_types = "%t,%bias,%scalar", f"{transposed},{bias_type},tensor<{dtype}>"
        maps = [same, bias_map, mapping([]), same]
        args = f"%v:{dtype},%biasv:{dtype},%scalarv:{dtype},%unused:{dtype}"
    copy = f"""%t=linalg.transpose ins(%x:{original}) outs(%e:{transposed})
      permutation=[{",".join(map(str, permutation))}]"""
    if generic_copy:
        copy = f"""%t=linalg.generic {{indexing_maps=[{mapping(["d" + str(i) for i in inverse])},{same}],
        iterator_types=[{",".join(['"parallel"'] * rank)}]}}
        ins(%x:{original}) outs(%e:{transposed}) {{^bb0(%v:{dtype},%unused:{dtype}):
        linalg.yield %v:{dtype}
        }}->{transposed}"""
    operation = "arith.addf" if dtype.startswith("f") else "arith.addi"
    one = "1.0" if dtype.startswith("f") else "1"
    expression = "%unused" if init_read else "%one"
    body = f"%z={operation} %v,{expression}:{dtype}"
    if broadcast:
        body = f"%sum={operation} %v,%biasv:{dtype}\n %z={operation} %sum,%scalarv:{dtype}"
    extra = ""
    returns = "%back"
    result_type = original
    if shared:
        extra = f"""
        %e3=tensor.empty():{transposed}
        %right=linalg.generic {{indexing_maps=[{same},{same}],
          iterator_types=[{",".join(['"parallel"'] * rank)}]}}
          ins(%a:{transposed}) outs(%e3:{transposed}) {{^bb0(%v:{dtype},%unused:{dtype}):
          %z={operation} %v,%one:{dtype}
          linalg.yield %z:{dtype}
        }}->{transposed}
        %e4=tensor.empty():{original}
        %back2=linalg.transpose ins(%right:{transposed}) outs(%e4:{original})
          permutation=[{",".join(map(str, inverse))}]
        """
        returns, result_type = "%back,%back2", f"({original},{original})"
    effect = "func.call @opaque():()->()" if barrier else ""
    return f"""module {{
      func.func private @opaque()
      func.func @forward(%x:{original}{extra_arguments})->{result_type} {{
        %one=arith.constant {one}:{dtype}
        %e=tensor.empty():{transposed}
        {copy}
        %e1=tensor.empty():{transposed}
        %a=linalg.generic {{indexing_maps=[{",".join(maps)}],
          iterator_types=[{",".join(['"parallel"'] * rank)}]}}
          ins({inputs}:{operand_types}) outs(%e1:{transposed}) {{^bb0({args}):
          {body}
          linalg.yield %z:{dtype}
        }}->{transposed}
        {effect}
        %e2=tensor.empty():{original}
        %back=linalg.transpose ins(%a:{transposed}) outs(%e2:{original})
          permutation=[{",".join(map(str, inverse))}]
        {extra}
        func.return {returns}:{original + "," + original if shared else original}
      }}
    }}"""


def _plan(source, **options):
    module = parse_mlir_text(source)
    before = ir_text(module)
    islands = plan_islands(module, **options)
    assert ir_text(module) == before
    return module, islands


@pytest.mark.parametrize("permutation", [(1, 0), (1, 2, 0), (0, 3, 1, 2), (5, 4, 3, 2, 1, 0)])
def test_permutation_id_has_exact_canonical_roundtrip(permutation):
    assert layout_from_id(layout_id(permutation)) == permutation


@pytest.mark.parametrize("identifier", ["p_0_1", "perm_", "perm_00_1", "perm_1_1", "perm_-1_0", "perm_0_+1"])
def test_malformed_layout_identity_refuses(identifier):
    with pytest.raises(ValueError):
        layout_from_id(identifier)


@pytest.mark.parametrize(
    "shape,permutation,dtype",
    [
        ((2, 3, 5), (1, 2, 0), "i8"),
        ((1, 3, 5, 7), (0, 3, 1, 2), "f32"),
        ((2, 1, 3, 1, 5, 1), (5, 4, 3, 2, 1, 0), "i16"),
    ],
)
def test_source_generated_noninvolutive_domains_and_canonical_abi(shape, permutation, dtype):
    _, (island,) = _plan(_source(shape, permutation, dtype=dtype))
    assert len(island.operations) == 3
    assert len(island.operation_names) == 3
    assert tuple(dict(island.operation_names)) == island.operations
    assert all(not encoding.requires_capacity for value in island.graph.values for encoding in value.encodings)
    assert not island.graph.capacity_limits
    assert all(option.preference_weight == 0 for op in island.graph.ops for option in op.options)
    canonical = layout_id(tuple(range(len(shape))))
    before, refused = _evaluate(island.graph, {value.name: canonical for value in island.graph.values})
    assert not refused and before.cost.physical_movement_bytes > 0
    assert island.result.plan.cost.physical_movement_bytes == 0
    boundaries = [op for op in island.graph.ops if op.kind.startswith("boundary")]
    assert len(boundaries) == 2
    assert all(op.options[0].encoding_id == canonical for op in boundaries)
    assert all(
        conversion.read_scope == conversion.write_scope == "logical_payload" for conversion in island.graph.conversions
    )
    first = next(op for op in island.graph.ops if op.kind == "transpose")
    for option in first.options:
        chosen = layout_from_id(option.encoding_id)
        assert layout_from_id(dict(option.port_encodings)[first.inputs[0]]) == tuple(permutation[i] for i in chosen)


def test_generic_copy_permutation_is_counted_as_transpose_not_free_pointwise():
    _, (island,) = _plan(_source((2, 3, 5), (1, 2, 0), generic_copy=True))
    assert [op.kind for op in island.graph.ops] == [
        "boundary_input",
        "transpose",
        "pointwise",
        "transpose",
        "boundary_output",
    ]
    assert len(island.layouts) == 3
    canonical = layout_id((0, 1, 2))
    baseline, _ = _evaluate(island.graph, {name: canonical for _, name in island.value_names})
    assert baseline.cost.conversion_count == 2


def test_zero_broadcast_and_rank_zero_scalars_keep_exact_graph_membership():
    _, (island,) = _plan(_source((1, 3, 5, 7), (0, 3, 1, 2), broadcast=True, bias_axis=1))
    names = dict(island.value_names)
    generic = island.operations[1]
    assert generic.inputs[1] in names
    assert generic.inputs[1].type.get_shape() == (1, 7, 1, 1)
    assert generic.inputs[2] not in names  # Scalar has no tensor layout decision.
    assert generic.outputs[0] not in names  # Empty destination is never read.
    assert island.result.plan is not None
    assert island.result.plan.cost.physical_movement_bytes == 0
    bias_name = names[generic.inputs[1]]
    pointwise = next(op for op in island.graph.ops if op.kind == "pointwise")
    assert bias_name not in pointwise.propagated_ports
    assert all(dict(option.port_encodings)[bias_name] == layout_id((0, 1, 2, 3)) for option in pointwise.options)


@pytest.mark.parametrize(
    "shape,permutation,bias_rank",
    [
        ((2, 3), (1, 0), 1),
        ((1, 3, 5, 7), (0, 3, 1, 2), 2),
    ],
)
def test_lower_rank_bias_keeps_its_canonical_layout_outside_full_rank_coupling(shape, permutation, bias_rank):
    _, (island,) = _plan(_source(shape, permutation, broadcast=True, bias_rank=bias_rank))
    generic = island.operations[1]
    bias_name = dict(island.value_names)[generic.inputs[1]]
    bias = next(value for value in island.graph.values if value.name == bias_name)
    identity = layout_id(tuple(range(bias_rank)))
    assert [encoding.id for encoding in bias.encodings] == [identity]
    node = next(op for op in island.graph.ops if op.kind == "pointwise")
    assert bias_name not in node.propagated_ports
    assert all(dict(option.port_encodings)[bias_name] == identity for option in node.options)
    assert dict(island.result.plan.assignments)[bias_name] == identity
    assert island.result.plan.cost.physical_movement_bytes == 0


def test_attributed_copy_is_a_boundary_instead_of_unaccounted_free_transpose():
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(_source((2, 3, 5), (1, 2, 0), generic_copy=True))
    copied = next(op for op in module.walk() if op.name == "linalg.generic")
    copied.body.block.last_op.attributes["prov.label"] = StringAttr("retained attribution")
    islands = plan_islands(module)
    assert all(copied not in island.operations for island in islands)
    assert len(islands) == 1 and len(islands[0].operations) == 2
    assert islands[0].result.plan.cost.physical_movement_bytes > 0


def test_shared_producer_and_two_outputs_are_present_once():
    _, (island,) = _plan(_source((2, 3, 5), (1, 2, 0), shared=True))
    assert len(island.operations) == len(set(island.operations)) == 5
    assert len([op for op in island.graph.ops if op.kind == "boundary_output"]) == 2
    assert island.result.plan.cost.physical_movement_bytes == 0
    shared = dict(island.value_names)[island.operations[1].results[0]]
    assert sum(shared in op.inputs for op in island.graph.ops) == 2


def test_unknown_effect_splits_epochs_and_preserves_before_effect_snapshot():
    _, islands = _plan(_source((2, 3, 5), (1, 2, 0), barrier=True))
    assert [len(island.operations) for island in islands] == [2, 1]
    first = islands[0]
    output = first.operations[-1].results[0]
    name = dict(first.value_names)[output]
    boundary = next(op for op in first.graph.ops if op.kind == "boundary_output")
    assert boundary.inputs == (name,)
    assert boundary.options[0].encoding_id == layout_id((0, 1, 2))
    assert first.result.plan.cost.physical_movement_bytes > 0


def test_init_read_is_a_barrier_not_an_implicit_initialized_tensor():
    _, islands = _plan(_source((2, 3), (1, 0), init_read=True))
    assert len(islands) == 2
    assert all(op.name == "linalg.transpose" for island in islands for op in island.operations)


@pytest.mark.parametrize(
    "options,code",
    [
        ({"max_assignments": 1}, "assignment_limit_exceeded"),
        ({"max_permutations": 2}, "permutation_limit_exceeded"),
    ],
)
def test_explicit_search_or_domain_exhaustion_returns_no_plan_without_source_change(options, code):
    _, (island,) = _plan(_source((2, 3, 5), (1, 2, 0)), **options)
    assert island.result.plan is None
    assert not island.result.exhausted
    assert code in {refusal.code for refusal in island.result.refusals}
    if "max_permutations" in options:
        assert len(island.layouts) <= options["max_permutations"]


@pytest.mark.parametrize("options", [{"max_permutations": 0}, {"max_assignments": True}])
def test_invalid_work_budgets_reject(options):
    with pytest.raises(ValueError, match="budgets"):
        plan_islands(parse_mlir_text(_source((2, 3), (1, 0))), **options)
