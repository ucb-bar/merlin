"""Exclusive FD-rooted DATA IO. Imports do not access any path."""
import json
import os
import stat
from pathlib import PurePosixPath

from policy import absolute, require

DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def facts(s):
    return {k: getattr(s, k) for k in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")}


def directory(path):
    p = absolute(str(path))
    fd = os.open("/", DIR)
    try:
        for part in p.parts[1:]:
            new = os.open(part, DIR, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def child_dirs(root, parts):
    fd = os.dup(root)
    try:
        for part in parts:
            require(part not in ("", ".", "..") and "/" not in part, "directory component")
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            new = os.open(part, DIR, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def new_file(parent, name):
    require(PurePosixPath(name).name == name and name not in (".", ".."), "file component")
    return os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o400, dir_fd=parent)


def write_all(fd, raw):
    view = memoryview(raw)
    while view:
        n = os.write(fd, view)
        require(n > 0, "write made no progress")
        view = view[n:]


def save(root, name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = new_file(root, name)
    try:
        write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(root)


def stable_file(fd, parent, name, size):
    st = os.fstat(fd)
    live = os.stat(name, dir_fd=parent, follow_symlinks=False)
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size == size
            and facts(st) == facts(live), "output FD/path/stat association")
    return facts(st)


class IntakeStop(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(evidence["error"])


def error_details(error):
    """Preserve primary notes and structured cleanup facts across wrappers/JSON."""
    retained = error.evidence if isinstance(error, IntakeStop) else {}
    return {"error_type": type(error).__name__, "error": str(error),
            "error_notes": list(getattr(error, "__notes__", retained.get("error_notes", []))),
            "deadline_cleanup_errors": list(getattr(error, "deadline_cleanup_errors",
                                                      retained.get("deadline_cleanup_errors", [])))}


def stop(root, error, evidence, pending=b""):
    evidence = {**evidence, **error_details(error),
                "accepted": False, "retry": False, "old_holds_cleared": False}
    try:
        save(root, "FIRST-FAILURE.json", evidence)
        if pending:
            fd = new_file(root, "pending-read-buffer.bin")
            try:
                write_all(fd, pending)
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException as failure:
        evidence["preservation_failure"] = {"type": type(failure).__name__, "error": str(failure)}
    raise IntakeStop(evidence) from error
