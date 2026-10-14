"""The guest C encoder and independent host decoder agree on exact words."""

from __future__ import annotations

import subprocess
import base64
import hashlib
import struct
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime.out_packet import ExpectedOutput, OutPacketError
from merlin.runtime.out_packet_v2 import decode_out_packet_v2, out_packet_v2_capacity


def _build_packet(tmp_path: Path, words: list[int]) -> bytes:
    values = ",".join(str(word) for word in words)
    source = tmp_path / "packet.c"
    source.write_text(
        '#include <stdio.h>\n#include "out_bin_memory.h"\n#include "out_packet_v2.h"\n'
        f"static int32_t words[]={{ {values} }};\n"
        "static unsigned char arena[1024*1024];\n"
        "static volatile uint64_t used;\n"
        "static merlin_out_bin_memory sink;\n"
        "static int write_bytes(const void *p,size_t n){return merlin_out_bin_memory_append(&sink,p,n);}\n"
        "int main(void){\n"
        "  merlin_out_bin_memory_init(&sink,arena,sizeof(arena),&used);\n"
        f'  if(!merlin_out_bin_memory_cstr(&sink,"OUT_PKT_BEGIN v2 out 1 {len(words)} i32 4 s e\\n")) return 2;\n'
        "  merlin_out_packet_v2 codec;\n"
        f"  merlin_out_packet_v2_init(&codec,{len(words)},4,1,1,write_bytes);\n"
        f"  if(!merlin_out_packet_v2_words_i32(&codec,words,{len(words)})) return 3;\n"
        "  if(!merlin_out_packet_v2_finish(&codec)) return 4;\n"
        '  if(!merlin_out_bin_memory_cstr(&sink,"OUT_PKT_END v2 ")) return 5;\n'
        "  if(!merlin_out_bin_memory_hex16(&sink,codec.checksum)) return 6;\n"
        '  if(!merlin_out_bin_memory_cstr(&sink,"\\nDONE\\n")) return 7;\n'
        "  if(!merlin_out_bin_memory_finish(&sink)) return 8;\n"
        "  return fwrite(arena,1,used,stdout)==used ? 0 : 9;\n"
        "}\n",
        encoding="ascii",
    )
    binary = tmp_path / "packet"
    subprocess.run(
        ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I",
         str(merlin_dir() / "runtime" / "baremetal"), str(source), "-o", str(binary)],
        check=True, capture_output=True,
    )
    return subprocess.run([str(binary)], check=True, capture_output=True).stdout


def _roster(count: int) -> tuple[ExpectedOutput, ...]:
    return (ExpectedOutput("out", 1, count, "i32", True, 4, (count,)),)


def test_c_encoder_roundtrips_signed_boundaries_once(tmp_path: Path) -> None:
    words = [-2147483648, -129, -128, -127, -1, 0, 1, 127, 128, 255, 32767, 2147483647]
    packet = _build_packet(tmp_path, words)
    actual = decode_out_packet_v2(packet, _roster(len(words)))
    assert actual["out"].values == tuple(words)
    assert len(packet) <= out_packet_v2_capacity(_roster(len(words)))
    assert packet.count(b"OUT_PKT_BEGIN") == 1


def test_large_common_case_is_exact_and_compact(tmp_path: Path) -> None:
    words = [0] * 87_616
    for index, value in zip(range(0, len(words), 7301), range(140, 152)):
        words[index] = value
    packet = _build_packet(tmp_path, words)
    assert len(packet) < 88_000
    assert decode_out_packet_v2(packet, _roster(len(words)))["out"].values == tuple(words)


def test_truncation_checksum_roster_escape_and_extra_bytes_refuse(tmp_path: Path) -> None:
    words = [1, -128, 2]
    packet = _build_packet(tmp_path, words)
    for bad in (
        packet[:-1],
        packet + b"extra",
        packet.replace(b"OUT_PKT_BEGIN v2 out", b"OUT_PKT_BEGIN v2 bad", 1),
        packet.replace(b"OUT_PKT_END v2 ", b"OUT_PKT_END v2 0", 1),
        packet.replace(bytes([0x80, 0x80, 0xFF, 0xFF, 0xFF]), b"\x01", 1),
    ):
        with pytest.raises(OutPacketError):
            decode_out_packet_v2(bad, _roster(len(words)))
    with pytest.raises(OutPacketError):
        decode_out_packet_v2(packet, _roster(2))


def test_rendered_direct_harness_and_trusted_neutral_reader(tmp_path: Path) -> None:
    from merlin.runtime.direct_kernel_harness import DirectKernelAbi, packet_output_roster_v2, render_direct_kernel
    from merlin.targetgen.contract.readback_policy import COHERENT_PACKET_V2, ReadbackPolicy
    from merlin_experiments.phase1.feedback.native_memory_readback import _decode_neutral_packet

    words = [0] * 513
    words[0], words[256], words[-1] = -2147483648, 159, -128
    cb = {
        "tensors": {
            "X": {"dtype": "i32", "shape": [3, 171],
                  "preload_b64": base64.b64encode(struct.pack("<513i", *words)).decode()},
            "Y": {"dtype": "i32", "shape": [3, 171]},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "X", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
    }
    harness = render_direct_kernel(
        cb, inputs={"X": {}}, abi=DirectKernelAbi("copy", "complete", 8, "little", "void"),
        readback_policy=ReadbackPolicy(COHERENT_PACKET_V2),
    )
    assert harness.count("OUT_PKT_BEGIN v2") == 1
    (tmp_path / "harness.c").write_text(harness)
    (tmp_path / "htif.h").write_text(
        "void console_init(void);void htif_puts(const char*);"
        "void htif_exit(int) __attribute__((noreturn));\n"
    )
    (tmp_path / "wrapper.c").write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <string.h>\n"
        "extern unsigned char merlin_readback_packet[];"
        "extern volatile uint64_t merlin_readback_packet_used;\n"
        "void console_init(void){}\nvoid complete(void){}\n"
        "void copy(void*x,void*y){memcpy(y,x,2052);}\n"
        "void htif_puts(const char*s){if(strcmp(s,\"DONE\\n\"))exit(4);}\n"
        "void htif_exit(int code){if(!code && merlin_readback_packet_used)"
        "fwrite(merlin_readback_packet,1,merlin_readback_packet_used,stdout);exit(code);}\n"
        "int harness_main(void);int main(void){return harness_main();}\n"
    )
    include = merlin_dir() / "runtime" / "baremetal"
    subprocess.run(
        ["cc", "-std=c11", "-O2", "-Dmain=harness_main", "-I", str(tmp_path), "-I", str(include),
         "-c", str(tmp_path / "harness.c"), "-o", str(tmp_path / "harness.o")],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["cc", "-no-pie", str(tmp_path / "wrapper.c"), str(tmp_path / "harness.o"),
         "-o", str(tmp_path / "harness")],
        check=True, capture_output=True,
    )
    packet = subprocess.run([str(tmp_path / "harness")], check=True, capture_output=True).stdout
    decoded = decode_out_packet_v2(packet, packet_output_roster_v2(cb))
    assert decoded["Y"].values == tuple(words)
    capacity = out_packet_v2_capacity(packet_output_roster_v2(cb))
    assert len(packet) < 650 and capacity > len(packet)
    envelope = b"GSIMPKT1" + struct.pack("<QQQQ", 4096, 8192, capacity, len(packet)) + packet + b"PKTEND1\n"
    dump = tmp_path / "packet.dump"
    dump.write_bytes(envelope)
    admission = {
        "packet_storage": {
            "capacity": capacity, "framing": "neutral_dense_outputs_out_packet_v2", "block_words": 0,
            "regions": [{"base": 4096, "bytes": 8}, {"base": 8192, "bytes": capacity}],
        },
        "outputs": [{"tensor": "Y", "logical_shape": [3, 171],
                     "physical_word_bytes": 4, "bytes": 2052}],
    }
    outputs, raw, used = _decode_neutral_packet(dump, admission, cb)
    assert outputs["Y"] == [words[i:i + 171] for i in range(0, 513, 171)]
    assert raw == envelope and used == len(packet)


def test_whole_model_readout_uses_one_exact_frame_per_output() -> None:
    from merlin.targetgen.contract.harness_render import Buffer, _Readout
    from merlin.targetgen.contract.readback_policy import COHERENT_PACKET_V2, ReadbackPolicy

    cb = {
        "kernel_abi": {"kind": "whole_program"},
        "tensors": {"out": {"role": "output", "dtype": "i32", "shape": [3, 171]}},
    }
    readout = _Readout(cb, [Buffer("out", "output", (3, 171), "i32")], None,
                       ReadbackPolicy(COHERENT_PACKET_V2), None)
    lines = "\n".join(readout.lines())
    assert lines.count("OUT_PKT_BEGIN v2") == 1
    assert "merlin_out_packet_v2_word" in lines
    assert "merlin_out_b64_range" not in lines
    assert "out_packet_v2.h" in readout.includes()
    assert readout.packet_capacity == out_packet_v2_capacity(
        (ExpectedOutput("out", 3, 171, "i32", True, 4, (3, 171)),)
    )


def test_gsim_v2_export_reuses_format_agnostic_coherent_packet_mode(tmp_path: Path) -> None:
    from merlin.runtime.backends._chipyard_memory_readback import (
        memory_readback_transport, prepare_memory_readback,
    )

    elf = tmp_path / "program.elf"
    elf.write_bytes(b"selected ELF bytes")
    manifest = tmp_path / "regions.txt"
    manifest.write_text("0x1000 8\n0x2000 4096\n")
    transport = memory_readback_transport("gsim", policy_transport="coherent_packet_v2")
    assert transport == "gsim_coherent_packet_v2"
    request = {
        "schema": "oracle_memory_readback_v1", "transport": transport,
        "output_path": str(tmp_path / "output.dump"),
        "regions": [{"base": 4096, "bytes": 8}, {"base": 8192, "bytes": 4096}],
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "regions_path": str(manifest),
    }
    command = prepare_memory_readback(elf, "gsim", request)
    assert command.argv_suffix[-1] == "+dump-mode=coherent-packet"
    command.revalidate()
    with pytest.raises(ValueError, match="region"):
        prepare_memory_readback(elf, "gsim", {**request, "regions": request["regions"][:1]})
