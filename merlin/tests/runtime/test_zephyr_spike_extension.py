"""`zephyr_model.run_on_spike(extension=...)`: a declared functional model reaches the spike argv, and
an absent one leaves the invocation exactly as it was."""

from __future__ import annotations

from types import SimpleNamespace

from merlin.runtime.backends import zephyr_model as zm
from merlin.targetgen.spike_extension import SpikeExtension


def _capture(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env")
        return SimpleNamespace(stdout="DONE\n", stderr="", returncode=0)

    monkeypatch.setattr(zm.subprocess, "run", fake_run)
    monkeypatch.setattr(zm, "_parse_console", lambda console, rc: {"console": console})
    return seen


def test_no_extension_keeps_the_plain_invocation(monkeypatch):
    seen = _capture(monkeypatch)
    zm.run_on_spike("img.elf", dram_base=0x80000000, mem_bytes=1 << 20, harts=2)
    assert not any(a.startswith(("--extlib", "--extension")) for a in seen["cmd"])
    assert seen["env"] is None
    assert seen["cmd"][-1] == "img.elf"


def test_a_declared_extension_adds_its_flags_before_the_image(monkeypatch, tmp_path):
    seen = _capture(monkeypatch)
    ext = SpikeExtension(
        target="t",
        extension_name="unit_model",
        extlib=tmp_path / "libm.so",
        library_dir=tmp_path,
        sha256="0" * 64,
        declared=True,
    )
    zm.run_on_spike("img.elf", dram_base=0x80000000, mem_bytes=1 << 20, harts=2, extension=ext)
    cmd = seen["cmd"]
    assert cmd[-3:] == [f"--extlib={tmp_path / 'libm.so'}", "--extension=unit_model", "img.elf"]
    assert seen["env"]["LD_LIBRARY_PATH"].split(":")[0] == str(tmp_path)
