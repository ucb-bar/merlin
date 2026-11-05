"""Original source identity transport is separate from actual work admission."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.compile.scheduling.projection_provenance import (
    projection_compute_member,
    projection_ir_attributes,
    projection_provenance_attributes,
    shared_projection_provenance,
)
from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir


def source():
    from merlin.frontends.linalg_mlir import parse_mlir_text

    body = []
    for index in range(2):
        body.append(f'''%z{index}=arith.constant 0:i32
          %s{index}="tensor.splat"(%z{index}):(i32)->tensor<2x2xi32>
          %r{index}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
          iterator_types=["parallel","parallel","reduction"]}}
          ins(%a,%w{index}:tensor<2x2xi8>,tensor<2x2xi8>) outs(%s{index}:tensor<2x2xi32>) {{
          ^bb0(%x:i8,%y:i8,%v:i32):
            %xx=arith.extsi %x:i8 to i32
            %yy=arith.extsi %y:i8 to i32
            %p=arith.muli %xx,%yy:i32
            %s=arith.addi %v,%p:i32
            linalg.yield %s:i32
          }}->tensor<2x2xi32>''')
    return parse_mlir_text('''module {func.func @source(%a:tensor<2x2xi8>,
      %w0:tensor<2x2xi8>,%w1:tensor<2x2xi8>)->(tensor<2x2xi32>,tensor<2x2xi32>){'''
      + '\n'.join(body) + '\nreturn %r0,%r1:tensor<2x2xi32>,tensor<2x2xi32>}}')


def annotated():
    module = source()
    group = find_shared_input_groups(module)[0]
    ops = tuple(module.body.block.first_op.body.block.ops)
    for index, ordinal in enumerate(group.contraction_ordinals):
        ops[ordinal].attributes.update(projection_ir_attributes({
            "prov.region_id": f"original.region.{index}", "prov.source_node_ids": [f"original.node.{index}"],
        }))
    return module, find_shared_input_groups(module)[0]


def test_exact_original_root_to_cb_interface_and_typed_attributes():
    from merlin.llvmlower.sibling_source_route import _command_buffer

    module, group = annotated()
    identities = shared_projection_provenance(module, group)
    assert [row["prov.region_id"] for row in identities] == ["original.region.0", "original.region.1"]
    cb, current = _command_buffer(module, "selected-target")
    assert current == group
    assert parse_interface_mlir(emit_interface_mlir(cb)) == cb
    attrs = [command["attributes"] for command in cb["commands"] if command["opcode"] == "MATMUL_RESIDENT"]
    assert tuple(attrs) == identities
    assert projection_ir_attributes(attrs[1]) == {
        "prov.region_id": StringAttr("original.region.1"),
        "prov.source_node_ids": ArrayAttr([StringAttr("original.node.1")]),
    }
    assert all("merlin.source_op_index" not in row for row in attrs)


def test_unannotated_legacy_interface_is_unchanged():
    from merlin.llvmlower.sibling_source_route import _command_buffer

    module = source()
    cb, group = _command_buffer(module, "selected-target")
    assert shared_projection_provenance(module, group) == ()
    assert all("attributes" not in c for c in cb["commands"] if c["opcode"] == "MATMUL_RESIDENT")
    assert parse_interface_mlir(emit_interface_mlir(cb)) == cb


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nodes", "stale"])
def test_incomplete_ambiguous_and_changed_source_refuse(damage):
    module, group = annotated()
    roots = [tuple(module.body.block.first_op.body.block.ops)[i] for i in group.contraction_ordinals]
    if damage == "missing":
        del roots[1].attributes["prov.source_node_ids"]
    elif damage == "duplicate":
        roots[1].attributes["prov.region_id"] = roots[0].attributes["prov.region_id"]
    elif damage == "nodes":
        roots[1].attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("x"), StringAttr("x")])
    else:
        roots[1].attributes["prov.region_id"] = StringAttr("changed")
    # A caller cannot retain a source proof across even a metadata mutation.
    with pytest.raises(ValueError, match="current typed source"):
        shared_projection_provenance(module, group)
    if damage != "stale":
        with pytest.raises(ValueError):
            shared_projection_provenance(module, find_shared_input_groups(module)[0])


@pytest.mark.parametrize("attrs", [
    {"prov.region_id": "x"},
    {"prov.region_id": "x", "prov.source_node_ids": []},
    {"prov.region_id": "x", "prov.source_node_ids": ["x", "x"]},
    {"prov.region_id": 'x,"forged"', "prov.source_node_ids": ["x"]},
    {"prov.region_id": "x", "prov.source_node_ids": ["x"], "merlin.source_op_index": 0},
])
def test_no_partial_or_injected_identity(attrs):
    with pytest.raises(ValueError):
        projection_provenance_attributes(attrs)


def test_actual_compute_bounds_are_complete_single_member_intervals():
    assert projection_compute_member((0, 32, 80), (0, 16), (16, 16)) == 0
    assert projection_compute_member((0, 32, 80), (32, 64), (16, 16)) == 1
    for col, width in [(None, (16, 16)), ((16, 32), (16, 16)), ((-1, 0), (16, 16)),
                       ((64, 79), (16, 16)), ((32, 32), (0, 16))]:
        with pytest.raises(ValueError):
            projection_compute_member((0, 32, 80), col, width)
