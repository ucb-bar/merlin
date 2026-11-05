"""Compile the actual connected wrapper, counted dense loops and pointwise images.

Neutral byte-machine execution only: no target compiler or hardware timing claim.
"""
import ctypes as C
from dataclasses import replace
import shutil
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

from merlin.compile.connected_host import ConnectedCStage, emit_connected_host_c
from merlin.compile.connected_regions import compile_connected_region
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects

from test_connected_mixed import fixture
from test_dense_source_catalog import implementation
from test_multioutput_fusion import SPACES, WORD


EFFECTS = BorrowedBufferEffects("neutral-compiled-synchronous-noescape", True, True, True)


class Backing(C.Structure):
    _fields_ = [("base", C.c_void_p), ("bytes", C.c_size_t)]


class Result(C.Structure):
    _fields_ = [("status", C.c_int), ("failed_stage", C.c_size_t)]


def build(tmp_path, *, widths=(1, 1, 4), origin=0, fail_stage=None, corrupt=False):
    region, settings, dense = fixture(tmp_path, widths=widths)
    settings["windows"] = tuple(replace(w, start=w.start + origin) for w in settings["windows"])
    program = compile_connected_region(region, **settings)
    calls = []

    def emit(invocation, symbol):
        index = len(calls)
        calls.append(invocation)
        args = "const uint8_t *const *inputs,const size_t *sizes,uint8_t *output,size_t output_bytes"
        if invocation.images is None:
            assert invocation.product is program.invocations[0].product
            selected, problem, storage, _ = dense
            code = implementation(invocation.product, problem, storage, selected.dense.dtypes, symbol + "_dense")
            invoke = f"{symbol}_dense((void*)inputs[0],(void*)inputs[1],output);"
            alignment = widths[:2]
        else:
            # Retain the EXACT checked image bytes. The neutral runtime explicitly
            # maps encoded window coordinates to the provided logical views.
            assert invocation.images == invocation.product.images()
            image = b"".join(invocation.images)
            binding = invocation.bindings
            width = widths[2]
            code = f"static const uint8_t {symbol}_image[] = {{{','.join(map(str,image))}}};\n"
            code += f"""static uint32_t {symbol}_word(const uint8_t *p) {{
              return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
            }}
            static int {symbol}_run({args}) {{
              const uint8_t *image={symbol}_image;
              for(size_t pos=0;pos<sizeof({symbol}_image);pos+={WORD.size}) {{
                const uint8_t *p=image+pos; unsigned op=p[0];
                if ({symbol}_word(p+1)!={SPACES.index(binding.inputs[0].space)} ||
                    {symbol}_word(p+9)!={SPACES.index(binding.output.space)}) return 81;
                uint32_t r={symbol}_word(p+13),w={symbol}_word(p+21),n={symbol}_word(p+25);
                if(r<{binding.inputs[0].start} || w<{binding.output.start}) return 82;
                size_t in=r-{binding.inputs[0].start},out=w-{binding.output.start};
                size_t output_width=op==1?1:{width};
                if(in>sizes[0] || n>(sizes[0]-in)/{width} ||
                   out>output_bytes || n>(output_bytes-out)/output_width) return 83;
                for(size_t i=0;i<n;++i) {{
                  uint32_t raw=0;
                  for(size_t b=0;b<{width};++b) raw|=(uint32_t)inputs[0][in+i*{width}+b]<<(8*b);
                  int32_t value=(int{width*8}_t)raw;
                  if(op==1) value=value < -128 ? -128 : value > 127 ? 127 : value;
                  else if(op==2) value=value < 0 ? 0 : value; else return 84;
                  for(size_t b=0;b<output_width;++b)
                    output[out+i*output_width+b]=(uint8_t)((uint32_t)value>>(8*b));
                }}
              }} return 0;
            }}\n"""
            invoke = f"return {symbol}_run(inputs,sizes,output,output_bytes);"
            if corrupt:
                invoke = "return 0;"  # Deliberately lying renderer: independent outputs must catch it.
            alignment = (1,)
        code += f"static int {symbol}_calls;\nint {symbol}_count_calls(void) {{ return {symbol}_calls; }}\n"
        code += f"int {symbol}({args}) {{ ++{symbol}_calls; "
        code += "return 77;" if fail_stage == index else invoke + "return 0;"
        code += "}\n"
        return ConnectedCStage(code, EFFECTS, alignment, widths[2] if index == 0 else 1,
                               "neutral-decoded-window-to-logical-pointer-v1")

    names = {w.backing for w in program.selected_contracts.storage.external}
    artifact = emit_connected_host_c(program, emit, symbol="chain", backing_origins={n: origin for n in names})
    assert len(calls) == 3 and "for (" in artifact.source
    source = tmp_path / "chain.c"
    source.write_text(artifact.source)
    compiler = shutil.which("cc")
    if not compiler or sys.byteorder != "little":
        pytest.skip("neutral native runtime requires C compiler and little-endian host")
    # The native loader may retain prior images even after pytest reuses a path.
    library = tmp_path / f"chain-{uuid4().hex}.so"
    result = subprocess.run([compiler, "-std=c11", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    dll = C.CDLL(str(library))
    dll.chain.argtypes = [C.POINTER(Backing), C.c_size_t]
    dll.chain.restype = Result
    buffers = {n: (C.c_ubyte * 512)(*([193] * 512)) for n in names}
    backings = (Backing * len(names))(*(Backing(C.addressof(buffers[row["name"]]), 512)
                                     for row in artifact.receipt["backings"]))
    return artifact, dll, buffers, backings


@pytest.mark.parametrize("widths,origin", [((1, 1, 4), 0), ((1, 1, 4), 128), ((2, 2, 2), 0)])
def test_compiled_chain_keeps_raw_liveout_fanout_tail_and_repeated_inputs(tmp_path, widths, origin):
    artifact, dll, buffers, backings = build(tmp_path, widths=widths, origin=origin)
    for seed in (1, 7):
        a = ((np.arange(21)*31+seed)%255-127).astype(f"<i{widths[0]}").reshape(3, 7)
        b = ((np.arange(35)*47+seed)%255-127).astype(f"<i{widths[1]}").reshape(7, 5)
        C.memmove(buffers["input0"], a.ctypes.data, a.nbytes)
        C.memmove(buffers["input1"], b.ctypes.data, b.nbytes)
        result = dll.chain(backings, len(backings))
        assert result.status == 0
        expected = (a.astype(np.int64) @ b.astype(np.int64)).astype(f"<i{widths[2]}")
        assert bytes(buffers["output1"])[:expected.nbytes] == expected.tobytes()
        assert bytes(buffers["output0"])[:15] == expected.clip(-128, 127).astype("i1").tobytes()
        assert bytes(buffers["output1"])[64:64+expected.nbytes] == expected.clip(0).tobytes()
        assert bytes(buffers["input0"])[:a.nbytes] == a.tobytes()
        assert bytes(buffers["input1"])[:b.nbytes] == b.tobytes()
        assert bytes(buffers["output1"])[expected.nbytes:64] == bytes([193])*(64-expected.nbytes)
    assert artifact.receipt["live_outputs"] == [2, 3, 4]
    assert artifact.receipt["execution_qualification"] == "not_established"
    assert [getattr(dll, f"chain_stage_{i}_count_calls")() for i in range(3)] == [2, 2, 2]


@pytest.mark.parametrize("failure", ["short_last", "alias_range", "overflow", "alignment", "metadata_alias"])
def test_all_boundary_failures_precede_first_launch(tmp_path, failure):
    artifact, dll, buffers, backings = build(tmp_path)
    names = [b["name"] for b in artifact.receipt["backings"]]
    if failure == "short_last":
        backings[names.index("output1")].bytes = 80
    elif failure == "alias_range":
        backings[1].base = backings[0].base + 8
    elif failure == "overflow":
        backings[0].base = C.c_size_t(-8).value
    elif failure == "alignment":
        backings[names.index("output1")].base += 1
    else:
        backings[0].base = C.addressof(backings)
    assert dll.chain(backings, len(backings)).status == -1
    assert [getattr(dll, f"chain_stage_{i}_count_calls")() for i in range(3)] == [0, 0, 0]
    assert all(value == 193 for buffer in buffers.values() for value in buffer)


def test_stage_failure_stops_successors_and_preserves_error(tmp_path):
    _, dll, buffers, backings = build(tmp_path, fail_stage=1)
    result = dll.chain(backings, len(backings))
    assert (result.status, result.failed_stage) == (77, 1)
    assert [getattr(dll, f"chain_stage_{i}_count_calls")() for i in range(3)] == [1, 1, 0]
    assert bytes(buffers["output0"]) == bytes([193])*512


def test_renderer_receipt_cannot_replace_independent_outputs(tmp_path):
    _, dll, buffers, backings = build(tmp_path, corrupt=True)
    C.memset(buffers["input0"], 1, 21)
    C.memset(buffers["input1"], 1, 35)
    assert dll.chain(backings, len(backings)).status == 0
    expected = np.full((3, 5), 7, dtype="<i4")
    assert bytes(buffers["output1"])[:60] == expected.tobytes()
    assert bytes(buffers["output0"])[:15] != expected.astype("i1").tobytes()
    assert bytes(buffers["output1"])[64:124] != expected.tobytes()


def test_mutation_and_missing_completion_premises_refuse(tmp_path):
    region, settings, _ = fixture(tmp_path)
    program = compile_connected_region(region, **settings)
    names = {w.backing for w in program.selected_contracts.storage.external}
    def bad(invocation, symbol):
        object.__setattr__(invocation.bindings, "inputs", ())
        return ConnectedCStage("code", EFFECTS, (1, 1), 1, "explicit")
    with pytest.raises(ValueError, match="changed"):
        emit_connected_host_c(program, bad, symbol="chain", backing_origins={n: 0 for n in names})
    with pytest.raises(ValueError, match="synchronous"):
        BorrowedBufferEffects("async", True, True, False)


def test_stage_alignment_domain_and_c_keyword_refuse(tmp_path):
    with pytest.raises(ValueError, match="alignments"):
        ConnectedCStage("code", EFFECTS, (1,), 1 << 64, "explicit").validate(1)
    region, settings, _ = fixture(tmp_path)
    program = compile_connected_region(region, **settings)
    with pytest.raises(ValueError, match="ordinary C symbol"):
        emit_connected_host_c(program, lambda *_: None, symbol="int", backing_origins={})
