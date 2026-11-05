"""``out_digest_v1``: the grader harness prints one XXH64 per output; the host checks it against values."""

from __future__ import annotations

import random
import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.out_digest import container_bytes, xxh64
from merlin.targetgen.contract import harness_render as hr
from merlin.targetgen.contract import readback_policy as RB


def test_xxh64_matches_the_published_vectors():
    assert xxh64(b"") == 0xEF46DB3751D8E999
    assert xxh64(b"a") == 0xD24EC4F1A98C6E5B
    assert xxh64(b"abc") == 0x44BC2CF5AD770999


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a host C compiler")
def test_the_harness_codec_and_the_host_agree_on_every_length_and_alignment(tmp_path):
    source = tmp_path / "check.c"
    source.write_text(
        '#include <stdio.h>\n#include "out_digest.h"\n'
        "static unsigned char b[5000];\n"
        "int main(void){ for (int i = 0; i < 5000; i++) b[i] = (unsigned char)(i * 131u + 7u);\n"
        "  for (int n = 0; n < 300; n++) for (int o = 0; o < 9; o++)\n"
        '    printf("%d %d %016llx\\n", n, o, (unsigned long long)merlin_out_digest(b + o, (uint64_t)n));\n'
        "  return 0; }\n"
    )
    exe = tmp_path / "check"
    subprocess.run(["cc", "-O2", "-I", str(runtime_dir() / "baremetal"), str(source), "-o", str(exe)], check=True)
    data = bytes((i * 131 + 7) & 0xFF for i in range(5000))
    for line in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines():
        n, o, digest = line.split()
        assert int(digest, 16) == xxh64(data[int(o) : int(o) + int(n)]), (n, o)


def _cb():
    return {
        "abi_version": "0.1",
        "tensors": {
            "W": {"shape": [16, 8], "dtype": "i8", "role": "weight"},
            "A": {"shape": [4, 16], "dtype": "i8", "role": "input"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "R", "dst": "acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "Y"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
            {"opcode": "EVICT", "operands": {"handle": "R"}},
        ],
    }


def test_the_default_abi_harness_prints_one_digest_and_no_values():
    abi = hr.logical_abi()
    hooks = hr.HostHooks("probe_kernel", "cycle_window_probe", "rdcycle %0", "fence")
    text = hr.render_with(_cb(), abi=abi, hooks=hooks, readback_policy=RB.ReadbackPolicy(RB.OUT_DIGEST_V1))
    assert '#include "out_digest.h"' in text
    assert 'printf("OUT_DIGEST Y %lu %016lx\\n", (unsigned long)128UL, ' in text
    assert "merlin_out_digest(T_Y, 128ULL)" in text
    assert 'printf("OUT ' not in text, "a digest harness prints no values"
    assert RB.OUT_DIGEST_V1 not in RB.READBACK_TRANSPORTS, "a digest is never offered as a full-value transport"


def test_the_digest_roster_is_closed_and_exact():
    cb = _cb()
    values = [random.Random(3).randint(-(2**31), 2**31 - 1) for _ in range(4 * 8)]
    digest = f"{xxh64(container_bytes(values, 4)):016x}"
    console = f"METRIC cycles 9\nOUT_DIGEST Y 128 {digest}\nDONE\n"
    roster = RB.require_digest_roster(cb, console, {})
    assert roster == {"Y": {"nbytes": 128, "digest": digest, "dtype": "i32"}}
    assert RB.digest_mismatches(roster, {"Y": values}) == []
    assert RB.digest_mismatches(roster, {"Y": [v + (i == 31) for i, v in enumerate(values)]}) == ["Y"]
    for bad, why in (
        (console.replace(" 128 ", " 124 "), "bytes"),
        ("DONE\n", "omitted"),
        (console + f"OUT_DIGEST Y 128 {digest}\n", "repeated"),
        (console + "OUT Y 1 1 5\n", "mix"),
    ):
        with pytest.raises(ValueError, match=why):
            RB.require_digest_roster(cb, bad, {})
