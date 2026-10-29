"""Whole-head ABIs compose with independently checked shared-input writers."""

from dataclasses import replace

import numpy as np
import pytest
from fake_quant_layer import Oracle
from test_joint_source_route import evidence, joint_source
from test_sibling_components import effects

from merlin.llvmlower import device_offload, device_shim
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.head_source_route import HeadBatchSourceRoute, prepare_host, require_host
from merlin.llvmlower.sibling_source_route import _sha, _text
from merlin.targetgen.corpus_spec import CorpusBinding


def test_whole_batch_request_keeps_disjoint_siblings_and_original_observers():
    from merlin.targetgen.source_head_selection import resolve_request

    module = joint_source()
    groups, records = evidence(module)
    selected, record = resolve_request(module, groups, set(range(7)),
                                      preferences={"shared_input_siblings": True, "whole_head_batches": True},
                                      eligible_source_regions=records)
    assert selected.whole_batch and selected.shared_input_siblings
    assert selected.record()["whole_batch"] is True
    assert record["composed_source_groups"] == list(range(6))
    assert record["unmatched_source_groups"] == [6]
    assert record["deferred_requests"] == []
    assert record["effect_or_placement_grant"] is False


def test_joint_host_rebuilds_only_matched_whole_head_interface_and_rechecks_custody(tmp_path, monkeypatch):
    from merlin.compile import mesh
    from merlin.runtime.linalg_numpy import evaluate

    module = joint_source()
    import json

    source = tmp_path / "model.mlir"
    source.write_text(_text(module))
    # All operands are explicit runtime inputs in this neutral source fixture.
    manifest = {str(i): {"kind": "input", "name": f"arg_{i}"}
                for i in range(len(module.body.block.first_op.function_type.inputs))}
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "weights.safetensors").write_bytes((8).to_bytes(8, "little") + b"{}      ")
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "compiler.py").write_text("# separately supplied neutral target consumer\n")
    selected = HeadBatchSourceRoute(_sha(_text(module)), effects(), shared_input_siblings=True, whole_batch=True)
    device = DeviceRouting("neutral", str(package), "int8", "i32", granularity="group",
                           sibling_source=selected, select=lambda _: True, capture=tmp_path)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("logical", version=2))
    monkeypatch.setattr(mesh, "_mesh_tile_binding",
                        lambda *args, **kwargs: CorpusBinding("neutral", 2, "int8", "i32", True, [], "exact_int"))
    ordinary = device_offload.rewrite_groups_to_device

    def rewrite(*args, **kwargs):
        result = ordinary(*args, **{**kwargs, "oracle": Oracle(), "weight_args": ()})
        return result

    monkeypatch.setattr(device_offload, "rewrite_groups_to_device", rewrite)
    _path, host = prepare_host(device, selected, source, tmp_path / "route")
    require_host(host, selected)
    family, = host.receipt["head_families"]
    symbol = family["physical_symbol"]
    head_buffers, reason = device_shim.logical_buffers(host.rewrite.expected_interfaces[symbol]["mlir"])
    assert not reason and [b.shape for b in head_buffers] == [(2, 2, 2)] * 3
    assert "matmul_batched" in host.rewrite.expected_interfaces[symbol]["mlir"]
    assert host.component_host.rewrite == host.rewrite
    assert host.receipt["matrix_products"] == 2
    for component in host.component_host.components:
        interface = host.rewrite.expected_interfaces[component.device_symbol]
        assert interface == component.rewrite.expected_interfaces[component.device_symbol]
        assert "matmul_batched" not in interface["mlir"]
    assert host.module.body.block.first_op.function_type == module.body.block.first_op.function_type
    from merlin_experiments.phase1.feedback.private_group_provenance import source_group_roster
    from merlin_experiments.phase1.feedback.private_head_provenance import join_heads

    groups, _records = evidence(module)
    roster = source_group_roster(groups, list(range(7)), host_placement="host", module=module)
    joined = join_heads(host.rewrite.to_sidecar(), roster, module=module, target="neutral", capture=source,
                        shared_input_siblings=True, effects=selected.effects, whole_batch=True)
    assert joined["components"][0]["whole_batch"] is True
    assert joined["components"][0]["kernel_invocations"] == 1
    assert len(joined["components"]) == 3
    assert len(joined["covered_source_groups"]) == 6
    # Actual source graph still has the exact externally visible output roster.
    args = [(np.arange(np.prod(t.get_shape())).reshape(t.get_shape()) % 9 - 4).astype(np.int8)
            for t in module.body.block.first_op.function_type.inputs]
    assert len(evaluate(module, args)) == 7
    corrupted = dict(host.rewrite.expected_interfaces)
    corrupted[symbol] = {**corrupted[symbol], "mlir": corrupted[symbol]["mlir"].replace("2x2x2", "3x2x2")}
    changed = replace(host, rewrite=replace(host.rewrite, expected_interfaces=corrupted))
    with pytest.raises(ValueError, match="custody changed"):
        require_host(changed, selected)


@pytest.mark.parametrize("shape", [(3, 2, 7, 5), (2, 1, 5, 3), (5, 3, 2, 7)])
def test_actual_generated_whole_head_bridge_executes_all_products_once(shape, tmp_path):
    import ctypes
    import shutil
    import subprocess

    from test_head_batching import source as head_source
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.head_source_route import project_heads, whole_head_interfaces

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("neutral executable interface check requires native compiler")
    h, m, k, n = shape
    source = parse_mlir_text(head_source(*shape))
    for index, root in enumerate(op for op in source.walk() if op.name == "linalg.generic"):
        root.attributes.update({"prov.region_id": StringAttr(f"original_{index}"),
                                "prov.source_node_ids": ArrayAttr([StringAttr(f"node_{index}")])})
    projection = project_heads(source)
    rewrite = device_offload.rewrite_groups_to_device(
        projection.module.clone(), "neutral", oracle=Oracle(), weight_args=(), select=lambda _: True,
    )
    rewrite = whole_head_interfaces(rewrite, projection.members,
        binding=CorpusBinding("neutral", 2, "int8", "i32", True, [], "exact_int"), enabled=True)
    symbol, = rewrite.entries
    buffers, reason = device_shim.logical_buffers(rewrite.expected_interfaces[symbol]["mlir"])
    assert not reason
    unit = device_shim.emit_logical_translation_unit(
        "neutral", {symbol: (buffers, rewrite.call_buffers[symbol])}, kernel_symbol_for=lambda _: "whole_kernel",
    )
    assert not unit.skipped and "for (long slice" not in unit.text
    # The neutral test primitive owns its complete logical batch. This is a
    # numerical ABI check, not an accelerator overlap or performance claim.
    primitive = f"""
static uint64_t calls;
uint64_t count(void){{return calls;}}
void whole_kernel(void *av,void *bv,void *cv){{
 const int8_t *a=av,*b=bv;int32_t *c=cv;calls++;
 for(int h=0;h<{h};h++)for(int m=0;m<{m};m++)for(int n=0;n<{n};n++){{
  uint32_t sum=0;
  for(int k=0;k<{k};k++)sum+=(uint32_t)a[(h*{m}+m)*{k}+k]*(uint32_t)b[(h*{k}+k)*{n}+n];
  c[(h*{m}+m)*{n}+n]=(int32_t)sum;
 }}
}}
"""
    # Passed tmp_path directories are removed between parametrizations; a new
    # shape must not reopen a pathname still held by the platform dynamic loader.
    cfile, library = tmp_path / "whole.c", tmp_path / f"whole_{h}_{m}_{k}_{n}.so"
    cfile.write_text(unit.text + primitive)
    subprocess.run([cc, "-std=c11", "-O2", "-shared", "-fPIC", cfile, "-o", library], check=True)
    loaded = ctypes.CDLL(str(library))

    class Descriptor(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64), ("sizes", ctypes.c_int64 * 3), ("strides", ctypes.c_int64 * 3)]

    def descriptor(value):
        return Descriptor(value.ctypes.data, value.ctypes.data, 0,
                          (ctypes.c_int64 * 3)(*value.shape),
                          (ctypes.c_int64 * 3)(*(s // value.itemsize for s in value.strides)))

    rng = np.random.default_rng(43)
    a = rng.integers(-7, 8, (h, m, k), dtype=np.int8)
    b = rng.integers(-7, 8, (h, k, n), dtype=np.int8)
    c = np.full((h, m, n), -777, np.int32)
    values = {buf.role: {"A0": a, "W": b, "Y0": c}[buf.name] for buf in buffers}
    # The source's actual declared role/order, rather than a guessed C order.
    supplied = []
    for row in rewrite.call_buffers[symbol]:
        supplied.append(descriptor(values[row["role"]]))
    result = Descriptor()
    getattr(loaded, "_mlir_ciface_" + symbol)(ctypes.byref(result), *(ctypes.byref(d) for d in supplied))
    np.testing.assert_array_equal(c, a.astype(np.int64) @ b.astype(np.int64))
    assert result.aligned == c.ctypes.data
    assert loaded.count() == 1
