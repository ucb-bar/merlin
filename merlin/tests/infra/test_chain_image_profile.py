"""Refresh only retained generated profiling support, preserving model objects."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.model_check.chain_image_objects import pin
from merlin.model_check.chain_image_profile import MARKER, RESET, prepare, verify
from merlin.runtime.whole_model_readback import GroupCallee, parse_group_profile, render_group_profile


@pytest.fixture
def retained(tmp_path):
    source = render_group_profile([GroupCallee("kernel", "c_interface", None, 2)], [("first", "kernel")])
    source = source.replace(RESET.strip() + "\n", "")
    source = source.replace(MARKER, MARKER + "\nstatic unsigned long long ticks;")
    source = source.replace('  __asm__ volatile("rdcycle %0" : "=r"(c));', '  c = ++ticks;')
    source += """
static char console[4096]; static unsigned at;
void htif_putc(char c) { console[at++] = c; console[at] = 0; }
void htif_puts(const char *s) { while (*s) htif_putc(*s++); }
void __real_kernel(void *r, const void *a, const void *b) { (void)r; (void)a; (void)b; ++ticks; }
void invoke(void) { __wrap_kernel(0,0,0); }
const char *dump(void) { at=0; merlin_group_profile_dump(0,ticks+1); return console; }
"""
    c, obj = tmp_path / "retained.c", tmp_path / "group_profile.o"
    c.write_text(source)
    gcc = str(Path(shutil.which("gcc")).resolve())
    argv = [gcc, "-O2", "-fPIC", "-c", str(c), "-o", str(obj)]
    subprocess.run(argv, check=True, capture_output=True)
    model = tmp_path / "model.o"
    model.write_bytes(b"retained graph/device bytes")
    original = pin(obj)
    program = {
        "objects": [original, pin(model)],
        "by_name": {"group_profile.o": original, "model.o": pin(model)},
        "recipe": {"commands": [{
            "requested_output": str(obj), "argv": argv, "inputs": [pin(c)], "executable": pin(Path(gcc)),
        }]},
    }
    return program, c


def refresh(retained, tmp_path):
    return prepare(retained[0], tmp_path / "refresh", nm=Path(shutil.which("nm")).resolve(), timeout=10)


def test_retained_profiler_gets_reset_without_recompiling_model(retained, tmp_path):
    original, _ = retained
    before = [dict(row) for row in original["objects"]]
    updated, record = refresh(retained, tmp_path)
    assert original["objects"] == before
    assert updated["objects"][1] == original["objects"][1]
    assert updated["objects"][0] != original["objects"][0]
    verify(record)
    so = tmp_path / "profile.so"
    subprocess.run(["cc", "-shared", record["object"]["path"], "-o", str(so)], check=True, capture_output=True)
    rt = ctypes.CDLL(str(so))
    rt.dump.restype = ctypes.c_char_p
    for _ in range(4):
        rt.merlin_group_profile_reset()
        rt.invoke()
        assert parse_group_profile(rt.dump().decode())["calls"] == 1
    # A freshly generated support object is reused without another compilation.
    assert refresh((updated, None), tmp_path) == (updated, None)


def test_retained_profile_source_drift_refused(retained, tmp_path):
    retained[1].write_text(retained[1].read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        refresh(retained, tmp_path)


def test_profile_support_receipt_reopens_actual_output(retained, tmp_path):
    _, record = refresh(retained, tmp_path)
    Path(record["object"]["path"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed"):
        verify(record)
