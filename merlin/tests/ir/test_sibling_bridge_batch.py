"""Final pure ABI assembly validates one fresh owner, retaining every writer."""

import ctypes
import json
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from fake_quant_layer import Oracle
from test_joint_source_route import joint_source
from test_output_fusion import _path
from test_sibling_components import effects, multi_source
from test_sibling_quantized_readout import shared_readout_source
from xdsl.dialects.builtin import StringAttr

from merlin.compile import mesh
from merlin.llvmlower import device_offload, device_shim, sibling_source_route
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.head_source_route import HeadBatchSourceRoute
from merlin.llvmlower.sibling_component_route import build_components
from merlin.llvmlower.sibling_source_route import SiblingSourceRoute, _sha, _text
from merlin.targetgen.corpus_spec import CorpusBinding
from merlin.xdsl_dialects.lowering import output_fusion as OF


def prepared(tmp_path, monkeypatch, kind="raw"):
    source = {"raw": multi_source, "typed": lambda: shared_readout_source(raw_observer=True),
              "heads": joint_source}[kind]()
    source_path = tmp_path / "model.mlir"
    source_path.write_text(_text(source))
    manifest = {str(i): {"kind": "input", "name": f"arg_{i}"}
                for i in range(len(source.body.block.first_op.function_type.inputs))}
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "weights.safetensors").write_bytes((8).to_bytes(8, "little") + b"{}      ")
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "compiler.py").write_text("# independently selected neutral implementation\n")
    monkeypatch.setattr(OF, "declared_path", lambda _: _path())
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("logical", version=2))
    monkeypatch.setattr(mesh, "_mesh_tile_binding", lambda *a, **kw:
                        CorpusBinding("neutral", 16, "int8", "i32", True, [], "exact_int"))
    ordinary = device_offload.rewrite_groups_to_device

    def rewrite(*args, **kwargs):
        return ordinary(*args, **{**kwargs, "oracle": Oracle(), "weight_args": ()})

    monkeypatch.setattr(device_offload, "rewrite_groups_to_device", rewrite)
    selected = (HeadBatchSourceRoute(_sha(_text(source)), effects(), shared_input_siblings=True)
                if kind == "heads" else SiblingSourceRoute(_sha(_text(source)), effects(), partial=True))
    device = DeviceRouting("neutral", str(package), "int8", "i32", granularity="group",
                           sibling_source=selected, select=lambda _: True, capture=tmp_path)
    if kind == "heads":
        from merlin.llvmlower.head_source_route import prepare_host

        _, host = prepare_host(device, selected, source_path, tmp_path / "route")
    else:
        host = build_components(device, selected, source, original=source)
    components = sibling_source_route.component_hosts(host)
    bindings = {}
    for component in components:
        buffers, reason = device_shim.logical_buffers(component.interface)
        assert not reason
        bindings[component.device_symbol] = (buffers, component.rewrite.call_buffers[component.device_symbol])
    names = {symbol: f"kernel_{i}" for i, symbol in enumerate(bindings)}
    unit = device_shim.emit_logical_translation_unit("neutral", bindings, kernel_symbol_for=names.get)
    assert not unit.skipped
    return selected, host, unit, bindings, names


@pytest.mark.parametrize("kind", ["raw", "typed", "heads"])
def test_batch_matches_original_single_bridges_and_revalidates_complete_owner_once(tmp_path, monkeypatch, kind):
    selected, host, unit, _, _ = prepared(tmp_path, monkeypatch, kind)
    components = sibling_source_route.component_hosts(host)
    expected = unit
    for component in components:
        expected = sibling_source_route.with_borrowed_bridge(expected, selected, component)
    actual_checks, graph_reads = [], []
    original_require, original_text = sibling_source_route.require_host, sibling_source_route._text

    def require(actual, selection):
        actual_checks.append(actual)
        return original_require(actual, selection)

    def graph_text(module):
        graph_reads.append(module)
        return original_text(module)

    monkeypatch.setattr(sibling_source_route, "require_host", require)
    monkeypatch.setattr(sibling_source_route, "_text", graph_text)
    result = sibling_source_route.with_borrowed_bridges(unit, selected, host)
    assert result == expected
    assert actual_checks == [host]
    if kind != "heads":
        assert sum(module is host.source for module in graph_reads) == 1
        assert sum(module is host.module for module in graph_reads) == 1
    actual_checks.clear()
    result = sibling_source_route.with_borrowed_bridges(unit, selected, host,
                                                      symbols=(components[-1].device_symbol,))
    assert actual_checks == [host]
    assert result == sibling_source_route.with_borrowed_bridge(unit, selected, components[-1])


@pytest.mark.parametrize("damage", ["source", "host", "core", "receipt", "abi"])
def test_batch_refuses_fresh_mutations_before_constructing_any_bridge(tmp_path, monkeypatch, damage):
    selected, host, unit, _, _ = prepared(tmp_path, monkeypatch)
    sibling_source_route.with_borrowed_bridges(unit, selected, host)
    component = host.components[-1]
    if damage in {"source", "host", "core"}:
        graph = {"source": host.source, "host": host.module, "core": component.core}[damage]
        graph.attributes["changed_after_earlier_bridge"] = StringAttr("yes")
    elif damage == "receipt":
        component.receipt["changed_after_earlier_bridge"] = True
    else:
        component.rewrite.call_buffers[component.device_symbol][-1]["dtype"] = "i8"
    constructed = []
    monkeypatch.setattr(sibling_source_route, "borrowed_bridge_source", lambda *args: constructed.append(args))
    with pytest.raises(ValueError, match="changed"):
        sibling_source_route.with_borrowed_bridges(unit, selected, host)
    assert constructed == []


def test_batch_subset_and_partial_shim_refuse_without_changing_input(tmp_path, monkeypatch):
    selected, host, unit, _, _ = prepared(tmp_path, monkeypatch)
    symbol = host.components[0].device_symbol
    for symbols in ((symbol, symbol), ("foreign_writer",), [symbol]):
        with pytest.raises(ValueError, match="symbols|roster"):
            sibling_source_route.with_borrowed_bridges(unit, selected, host, symbols=symbols)
    partial = replace(unit, symbols=(symbol,))
    before = partial.text
    with pytest.raises(ValueError, match="lost its ordinary logical shim"):
        sibling_source_route.with_borrowed_bridges(partial, selected, host)
    assert partial.text == before
    # The original single-call API still reopens its complete source/host.
    sibling_source_route.with_borrowed_bridge(unit, selected, host.components[0])
    host.source.attributes["changed_after_single_call"] = StringAttr("yes")
    with pytest.raises(ValueError, match="changed"):
        sibling_source_route.with_borrowed_bridge(unit, selected, host.components[0])


@pytest.mark.parametrize("kind", ["raw", "typed", "heads"])
def test_device_emission_batch_reads_full_source_only_at_its_two_boundaries(tmp_path, monkeypatch, kind):
    from merlin.llvmlower.device_build import _sibling_emission_batch

    selected, host, _, _, _ = prepared(tmp_path, monkeypatch, kind)
    checks, reads = [], []
    original_require, original_text = sibling_source_route.require_host, sibling_source_route._text

    def require(actual, selection):
        checks.append(actual)
        return original_require(actual, selection)

    def text(module):
        reads.append(module)
        return original_text(module)

    monkeypatch.setattr(sibling_source_route, "require_host", require)
    monkeypatch.setattr(sibling_source_route, "_text", text)
    with _sibling_emission_batch((selected, host)) as interfaces:
        assert checks == [host]
        components = sibling_source_route.component_hosts(host)
        for component in components:
            assert interfaces[component.device_symbol] == (
                component.interface, len(component.rewrite.call_buffers[component.device_symbol]))
        with pytest.raises(TypeError):
            interfaces[components[0].device_symbol] = ("foreign", 0)
        assert checks == [host]
    assert checks == [host, host]
    if kind != "heads":
        assert sum(module is host.source for module in reads) == 2
        assert sum(module is host.module for module in reads) == 2


@pytest.mark.parametrize("callback_raises", [False, True])
def test_device_emission_batch_refuses_source_mutation_after_callbacks_even_on_error(
        tmp_path, monkeypatch, callback_raises):
    from merlin.llvmlower.device_build import _sibling_emission_batch

    selected, host, _, _, _ = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="changed"):
        with _sibling_emission_batch((selected, host)) as interfaces:
            assert interfaces
            # Model an arbitrary package callback mutating another source owner.
            host.source.attributes["changed_by_package_callback"] = StringAttr("yes")
            if callback_raises:
                raise RuntimeError("package callback failed after mutation")


def test_device_emission_batch_uses_frozen_interfaces_if_callback_restores_an_owner(tmp_path, monkeypatch):
    from merlin.llvmlower.device_build import _sibling_emission_batch

    selected, host, _, _, _ = prepared(tmp_path, monkeypatch)
    component = host.components[0]
    original = component.interface
    with _sibling_emission_batch((selected, host)) as interfaces:
        try:
            object.__setattr__(component, "interface", "temporary untrusted replacement")
            assert interfaces[component.device_symbol][0] == original
        finally:
            object.__setattr__(component, "interface", original)
    # Temporary mutation cannot replace any actual package input. Fresh final
    # validation succeeds only because the complete original owner is restored.


@pytest.mark.parametrize("kind", ["raw", "typed"])
def test_actual_compiled_bridges_keep_all_typed_buffers_and_dispatch_once(tmp_path, monkeypatch, kind):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native compiler required for descriptor transport")
    selected, host, original, bindings, names = prepared(tmp_path, monkeypatch, kind)
    unit = sibling_source_route.with_borrowed_bridges(original, selected, host)
    primitives, arrays = [], {}
    for component in sibling_source_route.component_hosts(host):
        buffers, _ = bindings[component.device_symbol]
        signature = ",".join(f"void *p{i}" for i in range(len(buffers)))
        inputs = [i for i, b in enumerate(buffers) if not b.role.startswith("out_")]
        scalar = "+".join(f"((int{8 if buffers[i].dtype == 'i8' else 32}_t*)p{i})[0]" for i in inputs)
        body = [f"calls++;int seed={scalar};"]
        values = []
        for i, buffer in enumerate(buffers):
            array = np.full(buffer.shape, i + 1, np.int8 if buffer.dtype == "i8" else np.int32)
            if buffer.role.startswith("out_"):
                array.fill(-77)
                body.append(f"for(unsigned j=0;j<{array.size};j++)"
                            f"((int{8 if buffer.dtype == 'i8' else 32}_t*)p{i})[j]=seed+{i}+j%13;")
            values.append(array)
        arrays[component.device_symbol] = values
        primitives.append(f"void {names[component.device_symbol]}({signature}){{{''.join(body)}}}")
    source, library = tmp_path / "bridges.c", tmp_path / f"bridges_{kind}.so"
    source.write_text(unit.text + "\nstatic uint64_t calls;uint64_t count(void){return calls;}\n" +
                      "\n".join(primitives))
    subprocess.run([cc, "-O2", "-std=c11", "-shared", "-fPIC", source, "-o", library], check=True)
    loaded = ctypes.CDLL(str(library))

    def descriptor(value):
        class Descriptor(ctypes.Structure):
            _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p), ("offset", ctypes.c_int64),
                        ("sizes", ctypes.c_int64 * value.ndim), ("strides", ctypes.c_int64 * value.ndim)]
        return Descriptor(value.ctypes.data, value.ctypes.data, 0,
                          (ctypes.c_int64 * value.ndim)(*value.shape),
                          (ctypes.c_int64 * value.ndim)(*(s // value.itemsize for s in value.strides)))

    for component in sibling_source_route.component_hosts(host):
        buffers, _ = bindings[component.device_symbol]
        values = arrays[component.device_symbol]
        descriptors = [descriptor(value) for value in values]
        seed = sum(int(value.reshape(-1)[0]) for value, buffer in zip(values, buffers, strict=True)
                   if not buffer.role.startswith("out_"))
        getattr(loaded, "_mlir_ciface_" + component.writer.borrowed_symbol)(
            *(ctypes.byref(d) for d in descriptors))
        for i, (value, buffer) in enumerate(zip(values, buffers, strict=True)):
            if buffer.role.startswith("out_"):
                expected = (seed + i + np.arange(value.size) % 13).reshape(value.shape)
                np.testing.assert_array_equal(value, expected)
            else:
                np.testing.assert_array_equal(value, i + 1)
    assert loaded.count() == len(arrays)
    assert unit.symbols == original.symbols and unit.kernels == original.kernels
    # The neutral kernels establish transport only, not target tensor arithmetic.
