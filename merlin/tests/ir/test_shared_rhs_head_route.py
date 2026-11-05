"""Ordinary source selection keeps original products through checked row folds."""

import json
import struct
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fake_quant_layer import Oracle
from test_shared_rhs_views import source
from test_sibling_components import effects
from xdsl.dialects import builtin
from xdsl.dialects.linalg.ops import GenericOp

from merlin.llvmlower.head_source_route import (
    HeadBatchSourceRoute,
    prepare_host,
    project_heads,
    require_host,
    validate_head_member_ranges,
)
from merlin.llvmlower.sibling_source_route import _text
from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry


def fixture(*shape, scale="literal", planes=2):
    module = source(*shape, scale=scale, planes=planes)
    roots = []
    for op in module.walk():
        if type(op) is not GenericOp:
            continue
        try:
            static_zero_integer_matmul_geometry(op)
        except ValueError:
            continue
        i = len(roots)
        roots.append(op)
        op.attributes.update({"prov.region_id": builtin.StringAttr(f"original_{i}"),
                              "prov.source_node_ids": builtin.ArrayAttr([builtin.StringAttr(f"node_{i}")])})
    groups = [SimpleNamespace(root=root, members=(root,), index=i, placement="selected")
              for i, root in enumerate(roots)]
    positions = {id(op): i for i, op in enumerate(module.body.block.first_op.body.block.ops)}
    evidence = [{"source_op_index": positions[id(g.root)], "source_group": g.index,
                 "region_id": g.root.attributes["prov.region_id"].data, "semantic_family": "contraction"}
                for g in groups]
    return module, groups, evidence


def stored(tmp_path, module, *, g=2, r=3, planes=2):
    (tmp_path / "model.mlir").write_text(_text(module))
    raw = np.repeat(np.arange(2, 2 + g, dtype="<f4"), r).tobytes()
    shape = [g*r, 1, 1]
    header = json.dumps({"scale": {"dtype": "F32", "shape": shape,
                                     "data_offsets": [0, len(raw)]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw)
    manifest = {str(i): {"kind": "input", "name": f"runtime_{i}"}
                for i in range(len(module.body.block.first_op.function_type.inputs))}
    manifest[str(planes + 1)] = {"kind": "buffer", "weight": "scale", "dtype": "float32", "shape": shape}
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "session_contract.yaml").write_text("version: 1\nstates: []\nstreams: []\n")


@pytest.mark.parametrize("shape", [(2, 3, 2, 5, 7), (3, 2, 1, 7, 3)])
def test_checked_row_head_composition_keeps_original_planes_observers_and_range_identity(shape):
    module, groups, _ = fixture(*shape)
    before = _text(module)
    projection = project_heads(module, shared_rhs_row_views=True)
    g, r, m, k, n = shape
    assert len(projection.members) == 2
    assert sum(len(f["members"]) for f in projection.members) == len(groups) == 2*g*r
    for family in projection.members:
        assert family["batch_extent"] == g and family["geometry_m_k_n"] == [r*m, k, n]
        assert validate_head_member_ranges(family) == g*r
        assert [(v["projected_head"], v["row_start"], v["row_count"]) for v in family["members"]] == [
            (h // r, (h % r)*m, m) for h in range(g*r)]
    assert projection.module.body.block.first_op.function_type == module.body.block.first_op.function_type
    assert _text(module) == before
    projection.batching.revalidate()
    module.attributes["observer"] = builtin.StringAttr("new")
    with pytest.raises(ValueError, match="source or SSA changed"):
        projection.batching.revalidate()


def test_runtime_coefficient_needs_real_generated_caller_and_fresh_stored_span(tmp_path):
    module, _, _ = fixture(2, 3, 1, 5, 7, scale="runtime")
    with pytest.raises(ValueError, match="no checked repeated RHS"):
        project_heads(module, shared_rhs_row_views=True)
    stored(tmp_path, module)
    projection = project_heads(module, shared_rhs_row_views=True, capture=tmp_path)
    assert len(projection.members) == 2
    projection.batching.revalidate()
    (tmp_path / "session_contract.yaml").write_text("version: 1\nstates:\n- input_arg: 3\n")
    with pytest.raises(ValueError, match="generated-caller binding"):
        projection.batching.revalidate()
    (tmp_path / "session_contract.yaml").write_text("version: 1\nstates: []\nstreams: []\n")
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-4] + struct.pack("<f", 4.0))
    with pytest.raises(ValueError, match="stored argument"):
        projection.batching.revalidate()


@pytest.mark.parametrize("damage", ["overlap", "gap", "geometry", "partial", "missing"])
def test_original_row_partition_refuses_invalid_source_members(damage):
    module, _, _ = fixture(2, 3, 1, 5, 7)
    family = deepcopy(project_heads(module, shared_rhs_row_views=True).members[0])
    if damage == "overlap":
        family["members"][1]["row_start"] = 0
    elif damage == "gap":
        family["members"][1]["row_start"] += 1
    elif damage == "geometry":
        family["members"][0]["geometry_m_k_n"][0] += 1
    elif damage == "partial":
        family["members"][0].pop("projected_head")
    else:
        family["members"].pop()
    with pytest.raises(ValueError, match="range|roster|geometry"):
        validate_head_member_ranges(family)


def test_ordinary_request_routes_all_original_products_and_independently_joins_ranges(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.private_group_provenance import source_group_roster
    from merlin_experiments.phase1.feedback.private_head_provenance import join_heads, source_metrics

    from merlin.compile import mesh
    from merlin.llvmlower import device_offload, device_shim
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.targetgen.corpus_spec import CorpusBinding
    from merlin.targetgen.source_head_selection import resolve_request

    module, groups, evidence = fixture(2, 3, 1, 5, 7, scale="runtime")
    stored(tmp_path, module)
    required = set(range(len(groups)))
    selected, record = resolve_request(module, groups, required,
        preferences={"shared_input_siblings": False, "shared_rhs_row_views": True, "whole_head_batches": True},
        eligible_source_regions=evidence)
    assert selected.shared_rhs_row_views and selected.whole_batch
    assert record["matrix_products"] == 12 and record["composed_source_groups"] == list(range(12))
    # Original eligibility cannot be replaced by the four new row-root shapes.
    refused, _ = resolve_request(module, groups, required - {1},
        preferences={"shared_input_siblings": False, "shared_rhs_row_views": True},
        eligible_source_regions=evidence)
    assert refused is None
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "compiler.py").write_text("# independent target implementation\n")
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("logical", version=2))
    monkeypatch.setattr(mesh, "_mesh_tile_binding",
        lambda *a, **kw: CorpusBinding("neutral", 2, "int8", "i32", True, [], "exact_int"))
    ordinary = device_offload.rewrite_groups_to_device
    monkeypatch.setattr(device_offload, "rewrite_groups_to_device",
                        lambda *a, **kw: ordinary(*a, **{**kw, "oracle": Oracle(), "weight_args": ()}))
    device = DeviceRouting("neutral", str(package), "int8", "i32", granularity="group",
        sibling_source=selected, capture=tmp_path,
        select=lambda shape: shape.op == "linalg.batch_matmul")
    _, host = prepare_host(device, selected, tmp_path / "model.mlir", tmp_path / "route")
    require_host(host, selected)
    assert host.receipt["matrix_products"] == 12
    assert host.module.body.block.first_op.function_type == module.body.block.first_op.function_type
    roster = source_group_roster(groups, sorted(required), host_placement="host", module=module)
    joined = join_heads(host.rewrite.to_sidecar(), roster, module=module, target="neutral", capture=tmp_path,
                        whole_batch=True, shared_rhs_row_views=True)
    assert joined["matrix_products"] == 12
    with pytest.raises(ValueError, match="independently rederived ordinary routed roster"):
        # A flag or a relabeled old receipt cannot supply changed source ancestry.
        join_heads(host.rewrite.to_sidecar(), roster, module=module, target="neutral", capture=tmp_path,
                   whole_batch=True)
    _, metrics = source_metrics(host.rewrite.to_sidecar()["routed"], roster, joined)
    assert sum(v["source_total_elements"] for v in metrics.values()) == 12*7
    assert all(v["elements"] == 2*3*7 and v["matrix_invocations"] == 1 for v in metrics.values())
    symbols = {}
    for symbol, interface in host.rewrite.expected_interfaces.items():
        buffers, reason = device_shim.logical_buffers(interface["mlir"])
        assert not reason
        assert [b.shape for b in buffers] == [(2, 3, 5), (2, 5, 7), (2, 3, 7)]
        symbols[symbol] = buffers, host.rewrite.call_buffers[symbol]
    unit = device_shim.emit_logical_translation_unit("neutral", symbols, kernel_symbol_for=lambda _: "target")
    assert not unit.skipped and "for (long slice" not in unit.text
    # Every original source obligation is still checked, with its original ID.
    from merlin.targetgen.private_head_dispatch import mandatory_heads

    for component in joined["components"]:
        obligations = [r for r in evidence if r["source_group"] in component["covered_source_groups"]]
        rows = mandatory_heads(None, {}, obligations, component, mandatory=lambda f, c, obs: obs)
        assert len(rows) == 6 and [r["head_slice"] for r in rows] == list(range(6))


@pytest.mark.parametrize("shape", [(2, 3, 1, 5, 7), (3, 2, 2, 7, 3)])
def test_actual_native_projected_code_preserves_all_integer_planes_and_dense_observer(shape, tmp_path):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    g, r, m, k, n = shape
    module, _, _ = fixture(*shape)
    projection = project_heads(module, shared_rhs_row_views=True)
    rng = np.random.default_rng(711)
    lhs = [rng.integers(-31, 32, (g*r, m, k), dtype=np.int8) for _ in range(2)]
    rhs = (np.arange(g*k*n, dtype=np.float32).reshape(g, 1, k, n) - 17)*np.float32(1.5)
    scales = np.repeat(np.arange(2, 2+g, dtype=np.float32), r).reshape(g*r, 1, 1)
    quantized = np.clip(np.rint(np.repeat(rhs, r, axis=1).reshape(g*r, k, n)/scales), -127, 127).astype(np.int8)
    expected = [*(lhs[0].astype(np.int32) @ quantized.astype(np.int32)),
                *(lhs[1].astype(np.int32) @ quantized.astype(np.int32)), quantized]
    result = lower_model(_text(projection.module), tmp_path / "projected")
    invoke = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
    buffers = [np.full(value.size+32, 91, dtype=value.dtype) for value in expected]
    outputs = [buf[16:-16].reshape(value.shape) for buf, value in zip(buffers, expected, strict=True)]
    invoke([(value.ctypes.data, value.shape) for value in (*lhs, rhs, *outputs)])
    for actual, reference, buf in zip(outputs, expected, buffers, strict=True):
        np.testing.assert_array_equal(actual, reference)
        assert (buf[:16] == 91).all() and (buf[-16:] == 91).all()


def test_default_route_record_keeps_prior_bytes():
    selected = HeadBatchSourceRoute("a"*64, effects())
    assert "shared_rhs_row_views" not in selected.record()
