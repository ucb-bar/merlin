"""Existing common-check channel scoped to one original compiler invocation.

The compiler sees only its input/output workspace and a public request channel.
The original owner selects semantics/toolchain outside that namespace. Replies
are descriptive: saved JSON cannot recreate live formal checking authority.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.common import owned_children
from merlin.common.cpu_pool import (
    CompilerCPUTransfer,
    borrow_compiler_cpu,
    check_compiler_cpu,
    shared_compute_enabled,
    shared_compute_pool,
    transfer_compiler_cpu,
)
from merlin.common.path_scrub import scrub_host_paths
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

SCHEMA = "merlin.compiler_call_channel.v1"
MAX_REQUEST_BYTES = 256 * 1024
MAX_REQUESTS = 64
MAX_PENDING_ENTRIES = 256
COMMON_ACTIONS = frozenset({"common-transform", "common-check"})
def _valid_request_id(value):
    parts = value.split("_")
    return (value.isascii() and len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit()
            and len(parts[2]) == 32 and all(character in "0123456789abcdef" for character in parts[2]))


def _sha(path):
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("compiler call input is not a plain file")
        while block := stream.read(1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("compiler call input moved during identity read")
    return digest.hexdigest()


def _sources(package):
    from merlin_experiments.phase2.retained_revisions import source_inventory

    return source_inventory(package, max_bytes=32 << 20, max_files=8192, exclude_generated=True)["sha256"]


def _read(descriptor, name, limit):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("compiler channel input exceeds its ordinary file budget")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit or (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("compiler channel request moved during read")
    return json.loads(data)


def _write(descriptor, name, body):
    data = json.dumps(body, sort_keys=True).encode()
    if len(data) > 2 * MAX_REQUEST_BYTES:
        raise ValueError("compiler channel response exceeds its inherited byte budget")
    temporary = ".host-" + os.urandom(16).hex()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
    finally:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except FileNotFoundError:
            pass


def _validate_request(request, request_id, deadline):
    if not isinstance(request, dict) or set(request) != {
        "protocol", "request_id", "requester", "requested_at_unix_ns", "deadline_unix_ns", "action", "transform"
    } or request["protocol"] != 3 or request["request_id"] != request_id:
        raise ValueError("compiler channel requires the original common request protocol and identity")
    if request["action"] not in COMMON_ACTIONS:
        raise ValueError("compiler channel admits only original transformation and preservation checks")
    if not isinstance(request["transform"], dict):
        raise ValueError("compiler channel requires an actual bounded transformation request")
    end = request["deadline_unix_ns"]
    if type(end) is not int or type(request["requested_at_unix_ns"]) is not int:
        raise ValueError("compiler channel client deadline must be explicit")
    remaining = min(deadline - time.monotonic(), (end - time.time_ns()) / 1_000_000_000, 540)
    if remaining < 1:
        raise TimeoutError("compiler channel exhausted its original invocation deadline")
    return math.floor(remaining)


def _dispatch(service, request, request_id, *, deadline, proof_root, package_sha256, input_sha256):
    """Reuse the same original service methods and frozen substrate gates."""
    started = time.time_ns()
    try:
        remaining = _validate_request(request, request_id, deadline)
        if request["action"] == "common-check" and service.profile["common_substrate"] not in (
            "merlin.common_compiler_substrate.v3",
            "merlin.common_compiler_substrate.v4",
            "merlin.common_compiler_substrate.v5",
            "merlin.common_compiler_substrate.v6",
            "merlin.common_compiler_substrate.v7",
        ):
            raise FileNotFoundError("common check was not admitted by the original compiler profile")
        method = service.check if request["action"] == "common-check" else service.apply
        body = method(request["transform"], work_root=proof_root, timeout_s=remaining)
    except Exception as error:
        body = {"status": "unavailable" if isinstance(error, FileNotFoundError) else "unknown",
                "reason": scrub_host_paths(str(error)), "all_pass": False}
    payload = request.get("transform") if isinstance(request, dict) else None
    if isinstance(payload, dict):
        fields = (("before_ir", "source_sha256"), ("after_ir", "target_sha256")) if "before_ir" in payload else (
            ("source_ir", "source_sha256"),)
        for name, key in fields:
            if isinstance(payload.get(name), str):
                body.setdefault(key, hashlib.sha256(payload[name].encode()).hexdigest())
    body.update(selfcheck_protocol=3, selfcheck_request_id=request_id, submission_sha256=package_sha256,
        invocation_input_sha256=input_sha256, grade_started_at_unix_ns=started, graded_at_unix_ns=time.time_ns(),
        compiler_tool_profile_sha256=service.profile["sha256"])
    return body


class _Cancelled(BaseException):
    pass



def _verify_owner_selection(owner):
    """Bind readiness to protected metadata without scanning the SDK payload.

    The parent fully checks the original service before and after the compiler
    call. A requested proof additionally constructs that service and checks all
    frozen bytes before dispatch; readiness itself grants no proof authority.
    """
    import yaml

    from merlin.common.bundle_snapshot import MARKER, bundle_snapshot_root
    from merlin.targetgen.tool_profile import verify_tool_profile

    if type(owner) is not dict or set(owner) != {
        "workspace", "bundle_manifest", "repo", "archive_sha256", "snapshot_sha256",
        "profile_sha256", "snapshot_marker_sha256",
    }:
        raise ValueError("compiler channel has no exact original owner selection")
    workspace, archive, repo = (Path(owner[name]) for name in ("workspace", "bundle_manifest", "repo"))
    for path in (workspace, archive, repo):
        if not path.is_absolute() or path.resolve() != path or path.is_symlink():
            raise ValueError("compiler channel owner is not an ordinary canonical selection")
    if not workspace.is_dir() or not repo.is_dir() or archive.is_relative_to(workspace):
        raise ValueError("compiler channel authority is not independent of its original workspace")
    if _sha(archive) != owner["archive_sha256"]:
        raise ValueError("compiler channel changed its original archive")
    profile = verify_tool_profile(yaml.safe_load(archive.read_bytes()))
    if profile is None or profile["sha256"] != owner["profile_sha256"]:
        raise ValueError("compiler channel changed its original profile")
    marker = bundle_snapshot_root(workspace) / MARKER
    if marker.resolve() != marker or _sha(marker) != owner["snapshot_marker_sha256"]:
        raise ValueError("compiler channel changed its original snapshot marker")
    document = json.loads(marker.read_bytes())
    if (document.get("version") != 4 or document.get("repo") != str(repo)
        or document.get("content_sha256") != owner["snapshot_sha256"]):
        raise ValueError("compiler channel changed its original snapshot owner")

def _owner_digest(owner):
    return hashlib.sha256(json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _serve(selection, control, cpu_transfer=None):
    if set(selection) != {"schema", "workspace", "package", "package_sha256", "input_path", "input_sha256",
                          "command_name", "deadline", "owner", "parent_pid"} or selection["schema"] != SCHEMA:
        raise ValueError("compiler channel has no original host selection")
    owner = selection["owner"]
    _verify_owner_selection(owner)
    context = (borrow_compiler_cpu(cpu_transfer, owner_sha256=_owner_digest(owner))
               if cpu_transfer is not None else contextlib.nullcontext())
    with context:
        _serve_requests(selection, control)


def _serve_requests(selection, control):
    owner = selection["owner"]
    service = None
    workspace, package, input_path = map(Path, (selection["workspace"], selection["package"], selection["input_path"]))
    if (input_path.resolve() != input_path or input_path.is_symlink()
        or _sources(package) != selection["package_sha256"]):
        raise ValueError("compiler channel changed its original compiler invocation")
    if _sha(input_path) != selection["input_sha256"]:
        raise ValueError("compiler channel input changed before dispatch")
    deadline = selection["deadline"]
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or deadline <= time.monotonic():
        raise TimeoutError("compiler channel has no remaining original invocation budget")
    channel = workspace / ".qa_channel"
    descriptor = os.open(channel, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    processed = set()
    try:
        (control / "ready.json").write_text(json.dumps({"schema": SCHEMA, "ready": True}))
        while time.monotonic() < deadline:
            pending = []
            with os.scandir(descriptor) as entries:
                for index, row in enumerate(entries):
                    if index >= MAX_PENDING_ENTRIES:
                        raise ValueError("compiler channel exceeded its bounded pending file roster")
                    if row.name == "STOP":
                        return
                    if row.name.startswith("req_") and row.name.endswith(".json") and row.name not in processed:
                        pending.append(row.name)
            for name in sorted(pending):
                if len(processed) >= MAX_REQUESTS:
                    raise ValueError("compiler channel exhausted its inherited request count")
                request_id = name[4:-5]
                if not _valid_request_id(request_id):
                    raise ValueError("compiler channel request filename is not an original client identity")
                request = _read(descriptor, name, MAX_REQUEST_BYTES)
                proof = control / "proofs" / request_id
                proof.mkdir(parents=True, mode=0o700)
                if service is None:
                    service = CommonTransformService(workspace=Path(owner["workspace"]),
                        bundle_manifest=Path(owner["bundle_manifest"]), repo=Path(owner["repo"]))
                    if (service.profile["sha256"] != owner["profile_sha256"]
                        or service.snapshot_digest != owner["snapshot_sha256"]
                        or service.snapshot_marker_digest != owner["snapshot_marker_sha256"]
                        or hashlib.sha256(service.bundle_bytes).hexdigest() != owner["archive_sha256"]):
                        raise ValueError("compiler channel changed its original frozen owner")
                body = _dispatch(service, request, request_id, deadline=deadline, proof_root=proof,
                    package_sha256=selection["package_sha256"], input_sha256=selection["input_sha256"])
                check_compiler_cpu()
                _write(descriptor, "resp_" + request_id + ".json", body)
                _write(descriptor, "done_" + request_id, "ok")
                processed.add(name)
            time.sleep(0.02)
    finally:
        _write(descriptor, "STOP", {"stopped": True})
        os.close(descriptor)
        # Parent reopens profile/package/input after observing completed child
        # reaping. Avoid hashing a large snapshot before cancellation can reap
        # a proof process which started its own session.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cpu-lease-fd", type=int)
    parser.add_argument("--cpu-slot-fd", type=int)
    parser.add_argument("--cpu-lifetime-fd", type=int)
    args = parser.parse_args(argv)
    descriptors = (args.cpu_lease_fd, args.cpu_slot_fd, args.cpu_lifetime_fd)
    if (any(value is not None for value in descriptors)
            and not all(type(value) is int and value >= 0 for value in descriptors)):
        raise ValueError("compiler CPU transfer requires all original inherited descriptors")
    transfer = CompilerCPUTransfer(*descriptors) if descriptors[0] is not None else None
    selected = args.selection
    if selected.resolve() != selected or selected.is_symlink():
        raise ValueError("compiler channel requires a canonical private host selection")
    control = selected.parent
    descriptor = os.open(control, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        selection = _read(descriptor, selected.name, 64 * 1024)
    finally:
        os.close(descriptor)
    owned_children.become_child_subreaper()
    def cancel(*args):
        raise _Cancelled()
    signal.signal(signal.SIGTERM, cancel)
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError("compiler-call broker parent lifetime binding unavailable")
    try:
        if os.getppid() != selection["parent_pid"]:
            raise _Cancelled()
        _serve(selection, control, transfer)
    except _Cancelled:
        pass
    finally:
        cleanup = owned_children.reap_owned_children(None, 2)
        (control / "closed.json").write_text(json.dumps(cleanup))
    return 0 if cleanup["complete"] else 1


@dataclass(frozen=True)
class CompilerCallTransport:
    workspace: Path
    deadline: float

    def environment(self):
        return (("MERLIN_COMMON_WORKSPACE", str(self.workspace)),)

    def env_prefix(self, prefix):
        return prefix + "".join("export " + key + "=" + shlex.quote(value) + "; "
                                for key, value in self.environment())

    def remaining_timeout(self):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("compiler call exhausted its original aggregate invocation deadline")
        return value


def _service_owner(service):
    return {"workspace": str(service.workspace), "bundle_manifest": str(service.bundle_manifest),
            "repo": str(service.repo), "archive_sha256": hashlib.sha256(service.bundle_bytes).hexdigest(),
            "snapshot_sha256": service.snapshot_digest, "profile_sha256": service.profile["sha256"],
            "snapshot_marker_sha256": service.snapshot_marker_digest}


@contextlib.contextmanager
def compiler_call_transport(service, *, workspace, package, input_path, command_name, timeout_s,
                            proof_parent=None, borrow_compute=False):
    """Optionally lease one original CPU through candidate work and its RPC child.

    The original broker borrows that same live slot through private inherited
    descriptors. It is reaped before the parent closes the lease. No request or
    candidate environment can select a CPU or recreate this transfer.
    """
    if type(service) is not CommonTransformService or type(borrow_compute) is not bool:
        raise ValueError("compiler CPU transport requires the original owner and explicit CPU mode")
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("compiler call transport requires its finite original invocation budget")
    deadline = time.monotonic() + timeout_s
    with contextlib.ExitStack() as stack:
        transfer = None
        if borrow_compute and shared_compute_enabled():
            stack.enter_context(shared_compute_pool(timeout_s=timeout_s))
            transfer = stack.enter_context(transfer_compiler_cpu(owner_sha256=_owner_digest(_service_owner(service))))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("compiler call exhausted its CPU admission deadline")
        with _compiler_call_transport(service, workspace=workspace, package=package, input_path=input_path,
                                     command_name=command_name, timeout_s=remaining, proof_parent=proof_parent,
                                     cpu_transfer=transfer) as transport:
            yield transport


@contextlib.contextmanager
def _compiler_call_transport(service, *, workspace, package, input_path, command_name, timeout_s,
                             proof_parent=None, cpu_transfer=None):
    """Stage only the existing common client channel for a single compiler call."""
    if type(service) is not CommonTransformService:
        raise ValueError("compiler call transport requires the exact original common owner")
    workspace, package, input_path = map(Path, (workspace, package, input_path))
    if (workspace.resolve() != workspace or workspace.is_symlink() or not workspace.is_dir()
        or input_path.resolve() != input_path or input_path.is_symlink()):
        raise ValueError("compiler call transport requires its isolated ordinary input/output workspace")
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("compiler call transport requires its finite original invocation budget")
    service._verify()
    package_sha256, input_sha256 = _sources(package), _sha(input_path)
    parent = Path(proof_parent) if proof_parent is not None else workspace.parent
    if (parent.resolve() != parent or parent.is_symlink()
        or parent.is_relative_to(workspace) or parent.is_relative_to(package)):
        raise ValueError("compiler proof owner cannot be in an untrusted mounted workspace")
    channel = workspace / ".qa_channel"
    channel.mkdir(mode=0o700)
    inode = channel.stat().st_dev, channel.stat().st_ino
    deadline = time.monotonic() + timeout_s
    startup_deadline = min(deadline, time.monotonic() + 30)
    with tempfile.TemporaryDirectory(prefix="original-proof-", dir=parent) as raw:
        control = Path(raw)
        selection = {"schema": SCHEMA, "workspace": str(workspace), "package": str(package),
            "package_sha256": package_sha256, "input_path": str(input_path), "input_sha256": input_sha256,
            "command_name": command_name, "deadline": deadline, "parent_pid": os.getpid(),
            "owner": _service_owner(service)}
        path = control / "selection.json"
        path.write_text(json.dumps(selection, sort_keys=True))
        path.chmod(0o400)
        from merlin.common.paths import python_import_roots
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
               "PYTHONDONTWRITEBYTECODE": "1"}
        with (control / "broker.stdout").open("wb") as stdout, (control / "broker.stderr").open("wb") as stderr:
            command = [sys.executable, "-m", __name__, "--selection", str(path)]
            descriptors = ()
            if cpu_transfer is not None:
                command += ["--cpu-lease-fd", str(cpu_transfer.metadata_fd), "--cpu-slot-fd", str(cpu_transfer.slot_fd),
                            "--cpu-lifetime-fd", str(cpu_transfer.lifetime_fd)]
                descriptors = cpu_transfer.pass_fds
            process = subprocess.Popen(command, cwd=control, env=env, stdout=stdout, stderr=stderr,
                pass_fds=descriptors, start_new_session=True)
            try:
                while not (control / "ready.json").is_file():
                    if process.poll() is not None:
                        raise RuntimeError("original compiler-call broker failed its host readiness gate")
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError("original compiler-call broker exhausted its startup deadline")
                    time.sleep(0.02)
                yield CompilerCallTransport(workspace, deadline)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                closed = control / "closed.json"
                if not closed.is_file() or json.loads(closed.read_bytes()).get("complete") is not True:
                    raise RuntimeError("original compiler-call broker did not close its owned proof processes")
                if channel.is_symlink() or inode != (channel.stat().st_dev, channel.stat().st_ino):
                    raise ValueError("compiler invocation replaced its original request channel")
                service._verify()
                if _sources(package) != package_sha256 or _sha(input_path) != input_sha256:
                    raise ValueError("compiler/input identity changed during its original proof transport")


if __name__ == "__main__":
    raise SystemExit(main())
