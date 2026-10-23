"""Submitted code that a full-call provider imports runs only in the isolated build sandbox."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

full_call_build = pytest.importorskip("merlin.targetgen.full_call_build")

# What a hostile submitted module would do on import: patch the grader, then hunt for answers.
HOSTILE = """
import json, pathlib, sys
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
from merlin.targetgen import core_aten_capsules
core_aten_capsules.grade_readback = lambda *a, **k: "rewritten by the candidate"
seen = {}
for key in ("private_answer", "corpus_answer"):
    try:
        seen[key] = pathlib.Path(request[key]).read_text()
    except OSError as exc:
        seen[key] = type(exc).__name__
inputs = sorted(p.name for p in pathlib.Path(request["bundle"]).iterdir())
pathlib.Path(request["result"]).write_text(json.dumps({"error": json.dumps({"seen": seen, "inputs": inputs})}))
"""


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_hostile_submitted_code_cannot_reach_private_inputs_or_the_grader(tmp_path, monkeypatch):
    from merlin.targetgen import core_aten_capsules, plugins

    run = tmp_path / "run"
    bundle, corpus = run / ".private_full_call_runtime", tmp_path / "corpus"
    for directory in (bundle, corpus / "_private", tmp_path / "support", tmp_path / "submission"):
        directory.mkdir(parents=True)
    (bundle / "model.mlir").write_text("module {}\n")
    (bundle / "golden.npz").write_text("PRIVATE ANSWER BYTES")
    (corpus / "_private" / "golden.yaml").write_text("CORPUS ANSWER BYTES")
    (tmp_path / "facts.json").write_text("{}")
    monkeypatch.setattr(plugins, "resolve_support", lambda target: SimpleNamespace(base=tmp_path / "support"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    original = core_aten_capsules.grade_readback
    real_run = subprocess.run

    def hostile_child(command, **kwargs):
        request = json.loads((run / ".isolated" / "request.json").read_text())
        request.update(private_answer=str(bundle / "golden.npz"), corpus_answer=str(corpus / "_private/golden.yaml"))
        (run / ".isolated" / "request.json").write_text(json.dumps(request))
        tail = command.index(sys.executable)
        return real_run([*command[:tail], sys.executable, "-c", HOSTILE, command[-1]], **kwargs)

    monkeypatch.setattr(full_call_build.subprocess, "run", hostile_child)
    report = full_call_build.build_in_sandbox(
        {"target": "t"},
        bundle=bundle,
        work=run / ".isolated",
        package=tmp_path / "submission",
        provider=tmp_path / "support" / "provider.py",
        facts=tmp_path / "facts.json",
        denied=(corpus,),
        timeout=60,
    )
    observed = json.loads(report["error"])
    assert observed["seen"] == {"private_answer": "FileNotFoundError", "corpus_answer": "FileNotFoundError"}
    assert observed["inputs"] == ["model.mlir"]  # program inputs only; never the answers beside them
    assert core_aten_capsules.grade_readback is original  # the patch stayed in the sandboxed process


def test_sandbox_refuses_a_bind_that_would_expose_private_inputs(tmp_path, monkeypatch):
    from merlin.targetgen import plugins

    bundle = tmp_path / "run" / ".private_full_call_runtime"
    bundle.mkdir(parents=True)
    (tmp_path / "facts.json").write_text("{}")
    # The submission directory CONTAINS the private bundle: binding it would expose the answers.
    monkeypatch.setattr(plugins, "resolve_support", lambda target: SimpleNamespace(base=tmp_path / "run"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(full_call_build.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="would expose private grading inputs"):
        full_call_build.build_in_sandbox(
            {"target": "t"},
            bundle=bundle,
            work=tmp_path / "work",
            package=tmp_path / "run",
            provider=tmp_path / "provider.py",
            facts=tmp_path / "facts.json",
            denied=(),
            timeout=60,
        )


HOSTILE_ELF = r"""
#include <stdint.h>
#include "htif.h"
extern volatile uint64_t tohost, fromhost;
static volatile uint64_t request[8] __attribute__((aligned(64)));
static char seen[128];
static long proxy(long number, long a0, long a1, long a2, long a3, long a4) {
  request[0] = number; request[1] = a0; request[2] = a1; request[3] = a2; request[4] = a3; request[5] = a4;
  __asm__ volatile("fence rw, rw" ::: "memory");
  tohost = (uint64_t)(uintptr_t)request;
  while (fromhost == 0) ;
  fromhost = 0;
  __asm__ volatile("fence rw, rw" ::: "memory");
  return (long)request[0];
}
int main(void) {
  console_init();
  const char *path = PRIVATE_PATH;
  long length = 0;
  while (path[length]) length++;
  long fd = proxy(56 /* openat */, -100, (long)path, length + 1, 0 /* O_RDONLY */, 0);
  if (fd < 0) {
    htif_puts("HOSTILE open refused\n");
  } else {
    long got = proxy(63 /* read */, fd, (long)seen, sizeof(seen) - 1, 0, 0);
    seen[got > 0 ? got : 0] = 0;
    htif_puts("HOSTILE read: ");
    htif_puts(seen);
    htif_puts("\n");
  }
  htif_exit(0);
}
"""


def _hostile_image(tmp_path, private_file):
    import os

    from merlin.runtime.backends.spike import gcc_path
    from merlin.runtime.backends.spike_model import _harness_dir

    gcc, harness = gcc_path(), _harness_dir()
    if not gcc.is_file():
        pytest.skip("the RISC-V cross compiler is unavailable")
    (tmp_path / "hostile.c").write_text(HOSTILE_ELF)
    elf = tmp_path / "hostile.elf"
    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O1", "-ffreestanding", "-nostdlib", "-nostartfiles"]
    subprocess.run(
        [
            str(gcc),
            *flags,
            f"-I{harness}",
            f'-DPRIVATE_PATH="{private_file}"',
            "-T",
            str(harness / "link.ld"),
            str(harness / "crt.S"),
            str(harness / "htif.c"),
            str(tmp_path / "hostile.c"),
            "-o",
            str(elf),
        ],
        check=True,
        env={**os.environ},
    )
    return elf


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_htif_guest_cannot_open_private_files_from_the_spike_jail(tmp_path):
    from merlin.runtime.backends.spike import spike_path

    bundle = tmp_path / "run" / ".private_full_call_runtime"
    bundle.mkdir(parents=True)
    (bundle / "expected.txt").write_text("PRIVATE EXPECTED OUTPUT")
    elf = _hostile_image(tmp_path, bundle / "expected.txt")
    if not spike_path().is_file():
        pytest.skip("spike is unavailable")
    # Unjailed, the HTIF proxy really does hand the guest the file: the attack is real.
    plain = subprocess.run([str(spike_path()), "--isa=rv64gc", str(elf)], capture_output=True, text=True, timeout=60)
    assert "HOSTILE read: PRIVATE EXPECTED OUTPUT" in plain.stdout
    jail, image = full_call_build.oracle_jail(
        elf, tmp_path / "run" / ".oracle", private=[bundle], executables=[spike_path()]
    )
    jailed = subprocess.run(
        [*jail, str(spike_path()), "--isa=rv64gc", str(image)], capture_output=True, text=True, timeout=60
    )
    assert "HOSTILE open refused" in jailed.stdout and "PRIVATE" not in jailed.stdout + jailed.stderr


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_htif_guest_cannot_open_private_files_from_the_rtl_jail(tmp_path):
    import os

    from merlin.runtime.backends import base

    if not os.environ.get(full_call_build.ORACLE_ROOTS_ENV):
        pytest.skip("no RTL simulator jail roots are selected")
    try:
        backend = base.get_backend("gemmini")
        if not backend.available("vcs"):
            pytest.skip("the VCS engine is unavailable")
    except Exception:  # noqa: BLE001 -- no gemmini support selected in this environment
        pytest.skip("the gemmini support backend is unavailable")
    bundle = tmp_path / "run" / ".private_full_call_runtime"
    bundle.mkdir(parents=True)
    (bundle / "expected.txt").write_text("PRIVATE EXPECTED OUTPUT")
    elf = _hostile_image(tmp_path, bundle / "expected.txt")
    assert "HOSTILE read: PRIVATE EXPECTED OUTPUT" in backend.run_elf(elf, simulator="vcs", timeout=600)
    console = full_call_build.run_elf_jailed(
        "gemmini", "vcs", elf, tmp_path / "run" / ".oracle-vcs", private=[bundle], timeout=600
    )
    assert "HOSTILE open refused" in console and "PRIVATE" not in console


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_candidate_sandboxes_have_no_network(tmp_path):
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    probe = (
        "import socket, sys\n"
        "try:\n"
        f"    socket.create_connection(('127.0.0.1', {port}), timeout=5); print('CONNECTED')\n"
        "except OSError as exc:\n"
        "    print('BLOCKED', type(exc).__name__)\n"
    )
    roots = [sys.prefix, sys.base_prefix]
    try:
        for network, expected in ((False, "BLOCKED"), (True, "CONNECTED")):
            work = tmp_path / f"jail-{network}"
            work.mkdir()
            argv = full_call_build._jail(work, roots, {}, [], network=network)
            done = subprocess.run([*argv, sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
            assert done.stdout.startswith(expected), (done.stdout, done.stderr)
    finally:
        listener.close()


def test_sandbox_text_reaching_feedback_is_a_short_path_free_line():
    leak = "ValueError: read /scratch/x/merlin-support/core_aten/execution_provider.py " + "SECRET " * 80 + "\nline2"
    text = full_call_build.sanitized(leak)
    assert text.startswith("ValueError: read <path> SECRET") and len(text) <= len("ValueError: ") + 120
    assert "/" not in text and "line2" not in text
    assert full_call_build.sanitized("no eligible contraction in /a/b") == "no eligible contraction in <path>"
    record = full_call_build.sanitized_record([{"op": "/abs/path/leak", "n": 3, "deep": [[[["x"]]]]}] * 100)
    assert len(record) == 64 and record[0]["op"] == "<path>" and record[0]["n"] == 3


def test_jail_libraries_leave_pinned_copies_to_the_library_path(tmp_path, monkeypatch):
    pinned, live, other = tmp_path / "pinned", tmp_path / "live", tmp_path / "other"
    for directory, name in ((pinned, "libstdc++.so.6"), (live, "libstdc++.so.6"), (other, "libextra.so")):
        directory.mkdir(exist_ok=True)
        (directory / name).write_text("")
    listing = f"\tlibstdc++.so.6 => {live}/libstdc++.so.6 (0x1)\n\tlibextra.so => {other}/libextra.so (0x2)\n"
    monkeypatch.setattr(full_call_build.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, listing))
    assert full_call_build._libraries(["sim"], str(pinned)) == [other]


FORGING_ELF = (
    HOSTILE_ELF.split("int main(void) {")[0]
    + r"""
static long length(const char *s) { long n = 0; while (s[n]) n++; return n; }
static void put(long fd, const char *s) { proxy(64 /* write */, fd, (long)s, length(s), 0, 0); }
int main(void) {
  console_init();
  static const char line[] = "core   0: 0x0000000080000000 (0x0000000b) FORGED\n";
  put(2, line);
  const char *names[] = {"/dev/stderr", "/dev/fd/2", "/proc/self/fd/2"};
  for (int i = 0; i < 3; i++) {
    long fd = proxy(56 /* openat */, -100, (long)names[i], length(names[i]) + 1, 1 /* O_WRONLY */, 0);
    if (fd >= 0) put(fd, line);
  }
  htif_exit(0);
}
"""
)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_htif_guest_cannot_write_the_spike_trace(tmp_path, monkeypatch):
    from merlin.runtime.backends.spike import spike_path

    monkeypatch.setattr(sys.modules[__name__], "HOSTILE_ELF", FORGING_ELF)
    elf = _hostile_image(tmp_path, tmp_path / "unused")
    if not spike_path().is_file():
        pytest.skip("spike is unavailable")
    run = lambda prefix, trace: subprocess.run(  # noqa: E731
        [*prefix, str(spike_path()), "--isa=rv64gc", "-l", str(image)], stdout=subprocess.PIPE, stderr=trace, timeout=60
    )
    jail, image = full_call_build.oracle_jail(elf, tmp_path / "jail", private=[], executables=[spike_path()])
    procfs, _ = full_call_build.oracle_jail(elf, tmp_path / "jail", private=[], executables=[spike_path()], procfs=True)
    with (tmp_path / "open.trace").open("wb") as trace:
        run(procfs, trace)  # with /proc the guest reopens the trace through /dev/fd/2: the attack is real
    with (tmp_path / "jailed.trace").open("wb") as trace:
        assert b"FORGED" in run(jail, trace).stdout  # its own fd 2 is the console
    assert b"FORGED" in (tmp_path / "open.trace").read_bytes()
    assert b"FORGED" not in (tmp_path / "jailed.trace").read_bytes()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_simulator_jail_puts_the_operator_tool_path_first(tmp_path, monkeypatch):
    tools = tmp_path / "frozen" / "bin"
    tools.mkdir(parents=True)
    (tools / "dtc").write_text("#!/bin/sh\necho frozen-dtc\n")
    (tools / "dtc").chmod(0o755)
    (tmp_path / "image.elf").write_bytes(b"\x7fELF")
    monkeypatch.setenv(full_call_build.ORACLE_PATH_ENV, str(tools))
    jail, _ = full_call_build.oracle_jail(tmp_path / "image.elf", tmp_path / "jail", private=[])
    assert subprocess.run([*jail, "sh", "-c", "dtc"], capture_output=True, text=True).stdout == "frozen-dtc\n"
