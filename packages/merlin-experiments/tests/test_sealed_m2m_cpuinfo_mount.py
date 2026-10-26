"""The sealed capture guest gets exactly one byte-pinned /proc/cpuinfo, so torch's cpuinfo initialization succeeds."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_m2m_validation import _validate_snapshots


def _mount():
    return ("--tmpfs", "/proc", "--ro-bind", str(sealed_m2m._GUEST_CPUINFO), "/proc/cpuinfo")


def _argv(tmp_path, **kwargs):
    return sealed_m2m._sandbox_argv(
        Path("/usr/bin/bwrap"), tmp_path / "runtime", tmp_path / "source", tmp_path / "out",
        ("/bin/true",), Path("/capture-out"), **kwargs,
    )


def test_the_pinned_cpuinfo_bytes_match_their_literal():
    data = sealed_m2m._GUEST_CPUINFO.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sealed_m2m._GUEST_CPUINFO_SHA256
    assert data.startswith(b"processor\t: 0\n")


def test_guest_gets_only_the_fixed_cpuinfo_from_proc(tmp_path):
    argv = _argv(tmp_path)
    mount = _mount()
    start = argv.index("--tmpfs", argv.index("/capture-out")) + 2
    assert tuple(argv[start : start + len(mount)]) == mount
    assert argv[start + len(mount) : start + len(mount) + 2] == ["--dev", "/dev"]
    assert "--proc" not in argv
    assert "/proc/cpuinfo" not in argv[: argv.index("--tmpfs", argv.index("/capture-out")) + 2]


def test_the_probe_runs_with_the_same_proc_mount(tmp_path, monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(sealed_m2m.subprocess, "run", run)
    sealed_m2m._probe_sandbox(Path("/usr/bin/bwrap"))
    argv, mount = seen["argv"], _mount()
    assert tuple(argv[argv.index(mount[0], argv.index("/tmp")) :][: len(mount)]) == mount


def test_policy_digest_binds_the_proc_mount_and_its_bytes(tmp_path, monkeypatch):
    payloads = []
    monkeypatch.setattr(sealed_m2m, "_digest", lambda data: payloads.append(data) or "digest")
    sealed_m2m._policy(("python",), tmp_path)
    assert f"proc:cpuinfo:{sealed_m2m._GUEST_CPUINFO_SHA256}".encode() in payloads[0]
    monkeypatch.setattr(sealed_m2m, "_GUEST_CPUINFO_SHA256", "0" * 64)
    sealed_m2m._policy(("python",), tmp_path)
    assert payloads[0] != payloads[1]


@pytest.mark.parametrize("change", ["bytes", "missing", "symlink"])
def test_changed_or_indirect_cpuinfo_is_refused(tmp_path, monkeypatch, change):
    copy = tmp_path / "guest_cpuinfo.txt"
    if change == "bytes":
        copy.write_bytes(sealed_m2m._GUEST_CPUINFO.read_bytes() + b"x")
    elif change == "symlink":
        copy.symlink_to(sealed_m2m._GUEST_CPUINFO)
    monkeypatch.setattr(sealed_m2m, "_GUEST_CPUINFO", copy)
    with pytest.raises(sealed_m2m.SealedM2MError, match="guest cpuinfo"):
        _argv(tmp_path)
    with pytest.raises(sealed_m2m.SealedM2MError, match="guest cpuinfo"):
        sealed_m2m._probe_sandbox(Path("/usr/bin/bwrap"))


@pytest.mark.parametrize("state", ["missing", "file", "symlink", "occupied"])
def test_validation_requires_an_empty_proc_mount_point(tmp_path, state):
    runtime = tmp_path / "runtime"
    for name in ("source", "capture-out", "tmp", "dev"):
        (runtime / name).mkdir(parents=True)
    if state == "file":
        (runtime / "proc").write_text("x")
    elif state == "symlink":
        (runtime / "proc").symlink_to("/proc")
    elif state == "occupied":
        (runtime / "proc").mkdir()
        (runtime / "proc/cpuinfo").write_text("x")
    with pytest.raises(sealed_m2m.SealedM2MError, match="/proc is not empty"):
        _validate_snapshots(tmp_path / "source", runtime, Path("/capture-out"), schema="unused")


def _bwrap_or_skip():
    bwrap = shutil.which("bwrap")
    if not bwrap or not Path("/usr/bin/python3").exists():
        pytest.skip("bubblewrap and /usr/bin/python3 are required")
    probe = subprocess.run(
        [bwrap, "--unshare-all", "--unshare-user", "--disable-userns", "--ro-bind", "/", "/", "--", "/bin/true"],
        capture_output=True,
        timeout=30,
    )
    if probe.returncode:
        pytest.skip(f"this host cannot create the sandbox namespaces: {probe.stderr.decode()[-200:]}")
    return bwrap


def _guest(tmp_path, *, proc=True):
    guest = tmp_path / "guest"
    for name in ("usr", "tmp", "dev", *(("proc",) if proc else ())):
        (guest / name).mkdir(parents=True)
    for link, target in (("lib", "usr/lib"), ("lib64", "usr/lib64"), ("bin", "usr/bin")):
        (guest / link).symlink_to(target)
    return guest


def _run(bwrap, guest, mount, code):
    argv = [
        bwrap, "--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent", "--clearenv",
        "--ro-bind", str(guest), "/", "--ro-bind", "/usr", "/usr", "--tmpfs", "/tmp", *mount, "--dev", "/dev",
        "--", "/usr/bin/python3", "-S", "-c", code,
    ]
    return subprocess.run(argv, capture_output=True, timeout=30)


def test_real_bwrap_shows_only_the_fixed_cpuinfo_under_a_read_only_root(tmp_path):
    bwrap = _bwrap_or_skip()
    code = "import os; print(sorted(os.listdir('/proc'))); print(open('/proc/cpuinfo','rb').read().hex())"
    result = _run(bwrap, _guest(tmp_path), _mount(), code)
    assert result.returncode == 0, result.stderr.decode()
    lines = result.stdout.decode().split("\n")
    assert lines[0] == "['cpuinfo']"
    assert bytes.fromhex(lines[1]) == sealed_m2m._GUEST_CPUINFO.read_bytes()


def test_real_bwrap_without_the_mount_point_cannot_create_proc(tmp_path):
    bwrap = _bwrap_or_skip()
    result = _run(bwrap, _guest(tmp_path, proc=False), _mount(), "print(1)")
    assert result.returncode != 0
    assert b"/proc" in result.stderr


def test_real_bwrap_guest_cannot_write_the_pinned_cpuinfo(tmp_path):
    bwrap = _bwrap_or_skip()
    code = "open('/proc/cpuinfo','ab').write(b'x')"
    result = _run(bwrap, _guest(tmp_path), _mount(), code)
    assert result.returncode != 0
    assert sealed_m2m._file_digest(sealed_m2m._GUEST_CPUINFO) == sealed_m2m._GUEST_CPUINFO_SHA256
