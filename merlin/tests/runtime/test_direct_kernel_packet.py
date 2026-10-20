"""Execute the real neutral codec over actual output bytes, never an oracle."""

import base64
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime import harness_render as H
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, packet_output_blocks, render_direct_kernel
from merlin.runtime.out_packet import OutPacketError, decode_out_bin_packet, out_bin_packet_capacity
from merlin.targetgen.contract.readback_policy import COHERENT_PACKET_V1, FULL_VALUES_B64, ReadbackPolicy


@pytest.fixture
def packet_program():
    # A single genuine outlier affects its transport block, not other blocks.
    values = [((i * 7) % 255) - 128 for i in range(513)]
    values[300], values[-1] = 159, -(1 << 31)
    floats = [0x80000000, 0x7F800000, 0x7FC12345, 0, 0xFFFFFFFF]
    tensors = {
        "X": {"dtype": "i32", "shape": [3, 171],
              "preload_b64": base64.b64encode(struct.pack("<513i", *values)).decode()},
        "F": {"dtype": "f32", "shape": [5],
              "preload_b64": base64.b64encode(struct.pack("<5I", *floats)).decode()},
        "Y": {"dtype": "i32", "shape": [3, 171]}, "Z": {"dtype": "f32", "shape": [5]},
    }
    cb = {"tensors": tensors, "kernel_abi": {
        "args": [{"tensor": name, "access": access}
                 for name, access in (("X", "read"), ("F", "read"), ("Y", "write"), ("Z", "write"))],
        "outputs": ["Y", "Z"],
    }}
    return cb, values, floats


@pytest.fixture
def actual_packet(tmp_path, packet_program):
    cb, values, floats = packet_program
    source = tmp_path / "harness.c"
    text = render_direct_kernel(
        cb, inputs={"X": {}, "F": {}}, abi=DirectKernelAbi("copy", "complete", 8, "little", "void"),
        readback_policy=ReadbackPolicy(COHERENT_PACKET_V1), output_poison=0xA5,
    )
    source.write_text(text)
    (tmp_path / "htif.h").write_text(
        "void console_init(void);void htif_puts(const char*);void htif_exit(int) __attribute__((noreturn));\n"
    )
    wrapper = tmp_path / "wrapper.c"
    wrapper.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <string.h>\n"
        "extern unsigned char merlin_readback_packet[];extern volatile uint64_t merlin_readback_packet_used;\n"
        "static int completed;void console_init(void){}\n"
        "void copy(void*x,void*f,void*y,void*z){memcpy(y,x,2052);memcpy(z,f,20);}\n"
        "void complete(void){completed=1;}\n"
        "void htif_puts(const char*s){if(!completed || strcmp(s,\"DONE\\n\"))exit(4);}\n"
        "void htif_exit(int code){if(!code && completed && merlin_readback_packet_used)"
        "fwrite(merlin_readback_packet,1,merlin_readback_packet_used,stdout);exit(code);}\n"
        "int harness_main(void);int main(void){return harness_main();}\n"
    )
    compiler = shutil.which("cc")
    assert compiler, "codec controls require an actual C compiler"
    obj, elf = tmp_path / "harness.o", tmp_path / "harness.elf"
    subprocess.run([compiler, "-std=c11", "-O2", "-fstrict-aliasing", "-Dmain=harness_main",
                    "-I", str(tmp_path), "-I", str(runtime_dir() / "baremetal"),
                    "-c", str(source), "-o", str(obj)], capture_output=True, check=True)
    subprocess.run([compiler, "-no-pie", str(wrapper), str(obj), "-o", str(elf)],
                   capture_output=True, check=True)
    packet = subprocess.run([str(elf)], capture_output=True, check=True, timeout=5).stdout
    return packet, cb, values, floats, text


def test_actual_native_words_use_lossless_independent_block_widths(actual_packet):
    packet, cb, values, floats, text = actual_packet
    blocks = packet_output_blocks(cb)
    frames = decode_out_bin_packet(packet, tuple(frame for _, _, frame in blocks))
    assert [frames[frame.name].wire_bytes for _, _, frame in blocks] == [1, 2, 4, 4]
    actual = tuple(value for name, _, frame in blocks if name == "Y" for value in frames[frame.name].values)
    assert actual == tuple(values)
    assert frames[blocks[-1][2].name].f32_bits == tuple(floats)
    assert len(packet) < 513 * 4 + 20
    assert text.index("complete();") < text.index("merlin_packet_load(tensor_")
    assert text.index("merlin_out_bin_memory_finish(&merlin_packet_sink)") < text.index('htif_puts("DONE')
    assert "OUT_B64_BEGIN" not in text


def test_packet_arena_is_exact_file_backed_writable_storage(tmp_path, actual_packet):
    from merlin_experiments.phase1.feedback.native_output_readback import _elf_output_symbols

    _, cb, *_ = actual_packet
    capacity = out_bin_packet_capacity(tuple(frame for _, _, frame in packet_output_blocks(cb)))
    raw = (tmp_path / "harness.elf").read_bytes()
    symbols, _ = _elf_output_symbols(raw, {"merlin_readback_packet": capacity,
                                          "merlin_readback_packet_used": 8})
    header = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    segments = [struct.unpack_from("<IIQQQQQQ", raw, header[4] + index * 56) for index in range(header[9])]
    address = symbols["merlin_readback_packet"]
    rows = [row for row in segments if row[0] == 1 and row[1] & 6 == 6
            and row[3] <= address < address + capacity <= row[3] + row[5]]
    assert len(rows) == 1  # Arena capacity is loaded directly, outside zero-fill storage.
    assert raw[rows[0][2] + address - rows[0][3]] == 1


def test_complete_native_packet_reassembles_original_tensor_shapes(tmp_path, actual_packet):
    from merlin_experiments.phase1.feedback.native_memory_readback import _decode_neutral_packet

    packet, cb, values, floats, _ = actual_packet
    cb["kernel_abi"]["kind"] = "whole_program"
    capacity = out_bin_packet_capacity(tuple(frame for _, _, frame in packet_output_blocks(cb)))
    admission = {
        "packet_storage": {"capacity": capacity, "regions": [{"base": 4096, "bytes": 8},
                                                               {"base": 8192, "bytes": capacity}]},
        "outputs": [{"tensor": "Y", "logical_shape": [3, 171], "physical_word_bytes": 4, "bytes": 2052},
                    {"tensor": "Z", "logical_shape": [5], "physical_word_bytes": 4, "bytes": 20}],
    }
    path = tmp_path / "actual-host-codec.dump"
    envelope = b"GSIMPKT1" + struct.pack("<QQQQ", 4096, 8192, capacity, len(packet)) + packet + b"PKTEND1\n"
    path.write_bytes(envelope)
    outputs, raw, used = _decode_neutral_packet(path, admission, cb)
    assert outputs == {"Y": [values[index:index + 171] for index in range(0, 513, 171)], "Z": [floats]}
    assert raw == envelope and used == len(packet)


@pytest.mark.parametrize("defect", ["truncated", "extra", "checksum", "wrong_name", "wrong_geometry", "missing_block"])
def test_packet_preserves_exact_roster_count_and_transport_integrity(actual_packet, defect):
    packet, cb, *_ = actual_packet
    if defect == "truncated":
        packet = packet[:-1]
    elif defect == "extra":
        packet += b"extra"
    elif defect == "checksum":
        position = packet.index(b"OUT_BIN_END v1 ") + len(b"OUT_BIN_END v1 ")
        packet = packet[:position] + (b"0" if packet[position:position+1] != b"0" else b"1") + packet[position+1:]
    elif defect == "wrong_name":
        packet = packet.replace(b"merlin_block_0_0", b"merlin_block_0_8", 1)
    elif defect == "wrong_geometry":
        packet = packet.replace(b" 1 256 ", b" 2 128 ", 1)
    else:
        packet = packet[packet.index(b"OUT_BIN_BEGIN v1 merlin_block_0_1"):]
    with pytest.raises(OutPacketError):
        decode_out_bin_packet(packet, tuple(frame for _, _, frame in packet_output_blocks(cb)))


def test_selected_packet_accounts_for_full_arena_and_refuses_implicit_policy(packet_program):
    cb, *_ = packet_program
    contract = {"harness_abi": {
        "version": 2, "kind": "logical_pointer", "entry_symbol": "copy", "fence_symbol": "complete",
        "tensor_alignment": 8, "byte_order": "little", "main_convention": "void",
        "readback_transport": FULL_VALUES_B64,
    }, "runner": {"chipyard_rocc": {"engines": {"gsim": {"readback_transport": COHERENT_PACKET_V1}}}}}
    tensors = {name: {**spec, "role": "input" if name in {"X", "F"} else "output"}
               for name, spec in cb["tensors"].items()}
    logical = {**cb, "tensors": tensors, "kernel_abi": {**cb["kernel_abi"], "kind": "whole_program"}}
    values = {name: {"preload_b64": spec["preload_b64"]} for name, spec in tensors.items() if name in {"X", "F"}}
    # Use the original exact byte payload accepted by the neutral logical renderer.
    values = {name: base64.b64decode(row["preload_b64"]) for name, row in values.items()}
    capacity = out_bin_packet_capacity(tuple(frame for _, _, frame in packet_output_blocks(cb)))
    storage = sum(H._tensor(spec)[2] for spec in tensors.values()) + 8 + capacity
    with pytest.raises(ValueError, match="complete storage budget"):
        H.render_harness(logical, target="synthetic", contract=contract, inputs=values,
                         readback_policy=ReadbackPolicy(COHERENT_PACKET_V1), max_storage_bytes=storage - 1)
    with pytest.raises(ValueError, match="readback differs"):
        H.render_harness(logical, target="synthetic", contract={"harness_abi": contract["harness_abi"]},
                         inputs=values, readback_policy=ReadbackPolicy(COHERENT_PACKET_V1))
    assert "merlin_readback_packet" in H.render_harness(
        logical, target="synthetic", contract=contract, inputs=values,
        readback_policy=ReadbackPolicy(COHERENT_PACKET_V1), max_storage_bytes=storage,
    )


def test_packet_refuses_incompatible_byte_order(packet_program):
    cb, *_ = packet_program
    with pytest.raises(ValueError, match="little-endian"):
        render_direct_kernel(cb, inputs={"X": {}, "F": {}},
                             abi=DirectKernelAbi("copy", None, 8, "big", "void"),
                             readback_policy=ReadbackPolicy(COHERENT_PACKET_V1))
