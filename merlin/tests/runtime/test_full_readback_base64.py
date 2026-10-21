"""Actual C native-word framing survives terminal transformations losslessly."""

import errno
import json
import os
import shutil
import subprocess
import termios
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.out_b64 import OutB64Error, parse_base64_console_details
from merlin.runtime.whole_model_readback import decode_outputs


def samples():
    return [
        ("f32", np.array([0, 0x80000000, 0x7FC12345, 0x090A0D09], dtype="<u4")),
        ("f64", np.array([0x8000000000000000, 0x7FF8000000001234], dtype="<u8")),
        ("f16", np.array([0x8000, 0x7E55], dtype="<u2")),
        ("bf16", np.array([0x8000, 0x7FC1], dtype="<u2")),
        ("i64", np.array([-(1 << 63), -1, (1 << 63) - 1], dtype="<i8")),
        ("i32", np.array([-(1 << 31), -1, (1 << 31) - 1], dtype="<i4")),
        ("i16", np.array([-32768, -1, 32767], dtype="<i2")),
        ("i8", np.array([-128, -1, 127], dtype="i1")),
        ("i1", np.array([0, 1, 0, 1], dtype="u1")),
        ("f32", np.arange(1000, dtype="<u4") * 0x10203),
    ]


def compile_writer(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    source = tmp_path / "writer.c"
    rows = []
    for i, (dtype, array) in enumerate(samples()):
        raw = array.tobytes()
        rows.append(f"const unsigned char a{i}[]={{" + ",".join(map(str, raw)) + "};")
        signed = 0 if dtype.startswith(("f", "bf")) or dtype == "i1" else 1
        rows.append(f'if(!merlin_readback_b64("out{i}",1,{array.size},{array.itemsize},{signed},a{i},sink))return 2;')
    source.write_text(
        '#include <stdio.h>\n#include "full_readback_b64.h"\n'
        "static void sink(const char *s){fputs(s,stdout);}\nint main(void){\n"
        + "\n".join(rows)
        + '\nputs("METRIC memref_rank_mismatch 0");puts("DONE");return 0;}\n'
    )
    program = tmp_path / "writer"
    subprocess.run(
        [cc, "-std=c11", "-O2", "-I", str(runtime_dir() / "baremetal"), str(source), "-o", str(program)], check=True
    )
    return program


def test_c_writer_pty_tabs_crlf_complete_typed_decode(tmp_path):
    program = compile_writer(tmp_path)
    baseline = subprocess.check_output([program])
    master, slave = os.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[1] |= termios.OPOST | termios.ONLCR | termios.TAB3
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    proc = subprocess.Popen([program], stdout=slave)
    os.close(slave)
    blocks = []
    try:
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            blocks.append(chunk)
    finally:
        os.close(master)
    assert proc.wait(timeout=10) == 0
    wire = b"".join(blocks)
    (tmp_path / "console.bin").write_bytes(wire)
    assert b"\r\n" in wire
    assert b"\t" not in wire
    original, _, _ = parse_base64_console_details(baseline)
    outputs, _, frames = parse_base64_console_details(wire)
    assert outputs == original
    specs = [([a.size], dtype) for dtype, a in samples()]
    actual = decode_outputs(outputs, frames, specs)
    for value, (dtype, array) in zip(actual, samples(), strict=True):
        if dtype == "bf16":
            assert np.array_equal(value.view(np.uint32), array.astype(np.uint32) << 16)
        else:
            assert value.tobytes() == array.tobytes()


@pytest.mark.parametrize(
    "mutation", ["truncated", "mixed", "duplicate", "chunk", "width", "after_done", "no_done", "marker"]
)
def test_strict_base64_controls(tmp_path, mutation):
    wire = subprocess.check_output([compile_writer(tmp_path)])
    if mutation == "truncated":
        wire = wire[: wire.index(b"OUT_B64_END")]
    elif mutation == "mixed":
        wire = b"OUT_BIN_BEGIN v1 x 1 1 1 u 1\n" + wire
    elif mutation == "duplicate":
        wire = wire.replace(b"out1 ", b"out0 ", 1)
    elif mutation == "chunk":
        wire = wire.replace(b"OUT_B64_CHUNK 00000000", b"OUT_B64_CHUNK 00000001", 1)
    elif mutation == "width":
        wire = wire.replace(b"out0 1 4 4 u", b"out0 1 4 2 u", 1)
    elif mutation == "after_done":
        wire = b"DONE\n" + wire
    elif mutation == "no_done":
        wire = wire.removesuffix(b"DONE\n")
    else:
        wire = wire.replace(b"OUT_B64_END", b"OUT_B64_UNKNOWN", 1)
    with pytest.raises(OutB64Error):
        parse_base64_console_details(wire)


def test_ordinary_tiny_wholemodel_base64_with_readonly_inputs(tmp_path):
    from test_session_freestanding import capture

    from merlin.compile.baremetal_model import read_full_outputs
    from merlin.runtime.backends import spike_model as M

    if (
        not Path(os.environ.get("MERLIN_RISCV_GCC", "/unselected")).is_file()
        or not Path(os.environ.get("MERLIN_SPIKE", "/unselected")).is_file()
    ):
        pytest.skip("explicit original compiler and Spike required")
    root = capture(tmp_path / "capture") / "stages/first"
    built = M.build(
        root,
        tmp_path / "build",
        arena_mb=1,
        backend="scalar",
        host_vectorize=False,
        full_readback=True,
        full_readback_transport="base64",
        readonly_input_indices=(0, 1, 2),
        code_reserve=16 << 20,
        dram_bytes=1 << 30,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-O2", "-mcmodel=medany"],
    )
    argv = [
        os.environ["MERLIN_SPIKE"],
        "--isa=rv64gc",
        "--priv=m",
        "-p1",
        f"-m0x80000000:{hex(built['mem_bytes'])}",
        str(built["elf"]),
    ]
    proc = subprocess.run(argv, capture_output=True, timeout=120)
    (tmp_path / "console.bin").write_bytes(proc.stdout)
    (tmp_path / "console.stderr").write_bytes(proc.stderr)
    assert proc.returncode == 0, proc.stderr
    values, metrics = read_full_outputs(
        proc.stdout,
        root,
        built["build_hash"],
        readonly_input_observation=built["readonly_input_observation"],
        full_readback_transport="base64",
    )
    assert [v.tolist() for v in values] == [[9, 13], [3, 4], [1, 2], [5, 7]]
    assert built["full_readback_transport"] == "base64"
    assert metrics["memref_rank_mismatch"] == "0"
    assert b"OUT_BIN" not in proc.stdout

    recipe = json.loads((tmp_path / "build/compilation_recipe.json").read_text())
    command = next(c for c in recipe["commands"] if c["requested_output"].endswith("model_main.o"))
    assert "-DMERLIN_FULL_READBACK_BASE64" in command["argv"]
    names = {Path(pin["path"]).name for pin in command["inputs"]}
    assert {"out_b64.h", "full_readback_b64.h", "model_outputs.h", "readonly_inputs.h"} <= names
    assert recipe["status"] == "completed"

    from merlin.common.digest import sha256_file
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe

    recipe_path = tmp_path / "build/compilation_recipe.json"
    verify_completed_recipe(recipe_path, executable=built["elf"], expected_recipe_sha256=sha256_file(recipe_path))

    # A repeated header identity is malformed even when all referenced bytes exist.
    command["inputs"].append(dict(next(p for p in command["inputs"] if Path(p["path"]).name == "model_outputs.h")))
    bad = tmp_path / "malformed/compilation_recipe.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="repeats an input"):
        verify_completed_recipe(bad, executable=built["elf"], expected_recipe_sha256=sha256_file(bad))
