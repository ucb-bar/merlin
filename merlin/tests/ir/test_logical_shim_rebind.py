"""Pure shim replay preserves the actual public ABI and observed bridge bytes."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from merlin.common import invocation_record
from merlin.common.digest import sha256_file
from merlin.llvmlower import device_shim as S
from merlin.targetgen.corpus_spec import CorpusBinding, build


def compile_original(tmp_path, text):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler required for actual shim producer custody")
    source, obj = tmp_path / "device_shim.c", tmp_path / "device_shim.o"
    source.write_text(text)
    with invocation_record.observe(tmp_path, stage="device_shim_compilation",
                                   argv=[cc, "-c", str(source), "-o", str(obj)],
                                   inputs=(source,), outputs=(obj,)) as observation:
        result = subprocess.run(observation.document["argv"], capture_output=True, check=True)
        observation.complete(result)
    return dict(shim_source=source, shim_object=obj,
                compilation={"path": str(observation.path), "sha256": sha256_file(observation.path)})


def produced(tmp_path):
    binding = CorpusBinding("neutral", 16, "int8", "i32", True, [], "exact_int")
    entry = dict(name="source", kind="op", source_role="model_derived", source_reference="original source")
    _, leaf = build(dict(**entry, op="matmul", M=2, K=3, N=2), binding)
    _, ranked = build(dict(**entry, op="gemv_batched", B=3, M=2, H=3, K=3, N=2), binding)
    call = [{"role": "input_0", "shape": [3, 2, 3], "dtype": "i8"},
            {"role": "weight_0", "shape": [3, 3, 2], "dtype": "i8"},
            {"role": "out_0", "shape": [3, 2, 2], "dtype": "i32"}]
    buffers, why = S.logical_buffers(leaf)
    assert not why
    interfaces, calls, kernels = {"source_entry": leaf}, {"source_entry": call}, {"source_entry": "kernel0"}
    old = S.emit_logical_translation_unit("neutral", {"source_entry": (buffers, call)},
                                          kernel_symbol_for=kernels.get)
    suffix = "\n/* Original independently owned descriptor bridge. */\nint retained_bridge(int x){return x+7;}\n"
    options = dict(interfaces=interfaces, call_buffers=calls, kernels=kernels,
                   replacements={"source_entry": ranked}, **compile_original(tmp_path, old.text + suffix))
    return old, suffix, options


def test_rebind_requires_exact_observed_prefix_and_keeps_suffix(tmp_path):
    old, suffix, options = produced(tmp_path)
    before = Path(options["shim_source"]).read_bytes()
    new = S.rebind_logical_translation_unit("neutral", **options)
    assert new.symbols == old.symbols and new.kernels == old.kernels and not new.skipped
    assert not new.writer_contracts and not new.result_identities
    assert new.text.endswith(suffix)
    assert "for (long slice = 0; slice < 3; ++slice)" in old.text
    assert "for (long slice" not in new.text
    assert "kernel0((void *)p0_addr, (void *)p1_addr, (void *)p2_addr);" in new.text
    assert Path(options["shim_source"]).read_bytes() == before
    assert S.rebind_logical_translation_unit("neutral", **{**options, "replacements": {}}).text == old.text + suffix


@pytest.mark.parametrize("damage", ["suffix", "object", "record", "symbol", "kernel", "layout", "dtype"])
def test_rebind_refuses_changed_custody_symbols_or_public_abi(tmp_path, damage):
    _old, _suffix, options = produced(tmp_path)
    if damage == "suffix":
        path = Path(options["shim_source"])
        path.write_text(path.read_text().replace("x+7", "x+8"))
    elif damage == "object":
        Path(options["shim_object"]).write_bytes(b"changed original object")
    elif damage == "record":
        path = Path(options["compilation"]["path"])
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "symbol":
        options["replacements"] = {"foreign_entry": next(iter(options["replacements"].values()))}
    elif damage == "kernel":
        options["kernels"] = {"source_entry": "another_kernel"}
    elif damage == "layout":
        changed = options["replacements"]["source_entry"].replace("3x2x3xi8", "3x3x2xi8")
        options["replacements"] = {"source_entry": changed}
    else:
        options["replacements"] = {"source_entry": options["replacements"]["source_entry"].replace("i32", "i8")}
    with pytest.raises(ValueError):
        S.rebind_logical_translation_unit("neutral", **options)


def test_rebind_public_descriptors_execute_exactly_in_source_order(tmp_path):
    old, suffix, options = produced(tmp_path)
    new = S.rebind_logical_translation_unit("neutral", **options)
    a = (np.arange(18).reshape(3, 2, 3) % 9 - 4).astype(np.int8)
    w = (np.arange(18).reshape(3, 3, 2) % 5 - 2).astype(np.int8)
    expected = a.astype(np.int32) @ w.astype(np.int32)

    class Descriptor(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p), ("offset", ctypes.c_int64),
                    ("sizes", ctypes.c_int64 * 3), ("strides", ctypes.c_int64 * 3)]

    def descriptor(value):
        return Descriptor(value.ctypes.data, value.ctypes.data, 0, (ctypes.c_int64 * 3)(*value.shape),
                          (ctypes.c_int64 * 3)(*(stride // value.itemsize for stride in value.strides)))

    for label, unit, parameters, heads, offsets in (
        ("old", replace(old, text=old.text + suffix), "void *ww,void *aa,void *yy", 1, (0, 0, 0)),
        ("new", new, "void *aa,void *ww,void *yy", 3, (6, 6, 4)),
    ):
        source, library = tmp_path / f"{label}.c", tmp_path / f"{label}.so"
        source.write_text(unit.text + f"""
static int calls; int call_count(void){{return calls;}}
void kernel0({parameters}){{
  calls++; const signed char *a=aa,*w=ww; int *y=yy;
  for(int h=0;h<{heads};h++)for(int m=0;m<2;m++)for(int n=0;n<2;n++){{
    int z=0;for(int k=0;k<3;k++)z+=a[h*{offsets[0]}+m*3+k]*w[h*{offsets[1]}+k*2+n];
    y[h*{offsets[2]}+m*2+n]=z;
  }}
}}
""")
        subprocess.run([shutil.which("cc"), "-O2", "-shared", "-fPIC", str(source), "-o", str(library)],
                       capture_output=True, check=True)
        loaded = ctypes.CDLL(str(library))
        result, output = Descriptor(), np.full((3, 2, 2), -77, dtype=np.int32)
        values = [descriptor(value) for value in (a, w, output)]
        entry = loaded._mlir_ciface_source_entry
        entry.argtypes = [ctypes.POINTER(Descriptor)] * 4
        entry(ctypes.byref(result), *(ctypes.byref(value) for value in values))
        np.testing.assert_array_equal(output, expected)
        assert loaded.call_count() == (3 if label == "old" else 1)
        assert result.aligned == output.ctypes.data and tuple(result.sizes) == output.shape
        assert loaded.retained_bridge(5) == 12
        # A wrong public shape still returns an empty descriptor before dispatch.
        values[0].sizes[1] += 1
        entry(ctypes.byref(result), *(ctypes.byref(value) for value in values))
        assert not result.aligned and loaded.call_count() == (3 if label == "old" else 1)


def test_rebind_complete_roster_retains_actual_typed_borrowed_bridges(tmp_path, monkeypatch):
    from test_sibling_bridge_batch import prepared

    from merlin.llvmlower import sibling_source_route

    selected, host, _unit, bindings, kernels = prepared(tmp_path, monkeypatch, "typed")
    head_dir = tmp_path / "head"
    head_dir.mkdir()
    _, _, head = produced(head_dir)
    interfaces = {component.device_symbol: component.interface
                  for component in sibling_source_route.component_hosts(host)}
    calls = {symbol: list(call) for symbol, (_buffers, call) in bindings.items()}
    interfaces.update(head["interfaces"])
    calls.update(head["call_buffers"])
    kernels.update(head["kernels"])
    groups = {symbol: (S.logical_buffers(interface)[0], calls[symbol])
              for symbol, interface in interfaces.items()}
    plain = S.emit_logical_translation_unit("neutral", groups, kernel_symbol_for=kernels.get)
    original = sibling_source_route.with_borrowed_bridges(plain, selected, host)
    production_dir = tmp_path / "complete_production"
    production_dir.mkdir()
    custody = compile_original(production_dir, original.text)
    rebound = S.rebind_logical_translation_unit("neutral", interfaces=interfaces, call_buffers=calls,
                                                kernels=kernels, replacements=head["replacements"], **custody)
    suffix = original.text[len(plain.text):]
    assert suffix and "_mlir_ciface_" in suffix and rebound.text.endswith(suffix)
    assert rebound.symbols == original.symbols and rebound.kernels == original.kernels
    for component in sibling_source_route.component_hosts(host):
        component_group = {component.device_symbol: groups[component.device_symbol]}
        standalone = S.emit_logical_translation_unit("neutral", component_group,
                                                     kernel_symbol_for=kernels.get)
        # The whole ordinary typed entry is unchanged, including every result.
        body = standalone.text.partition("/* " + component.device_symbol + ":")[2].partition("/* 1 entry")[0]
        assert body in original.text and body in rebound.text
