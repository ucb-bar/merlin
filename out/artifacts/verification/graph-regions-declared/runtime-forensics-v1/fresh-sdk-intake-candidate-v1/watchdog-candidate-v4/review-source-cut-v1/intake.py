"""One exclusive NEW DATA intake. Backend injection is only for synthetic tests."""
import os
import time

import fdio
from http_fetch import fetch
from policy import absolute, identity, require, validate
from zip_intake import extract
from outer_deadline import fetch_deadline


def run(request, backend, *, clock=time.monotonic):
    validate(request)
    path = absolute(request["new_root"])
    parent = fdio.directory(path.parent)
    root = download = archive = None
    state = {"phase": "prepare", "pending": b"", "request_identity": identity(request)}
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        root = os.open(path.name, fdio.DIR, dir_fd=parent)
        root_stat = os.fstat(root)
        fdio.save(root, "ATTEMPT.json", {"request": request, "attempt_consumed": True, "retry": False})
        os.fsync(parent)
        disk = os.fstatvfs(root)
        require(disk.f_bavail * disk.f_frsize >= request["extract"]["disk_reserve_bytes"], "disk reserve unavailable")
        os.mkdir("download", 0o700, dir_fd=root)
        download = os.open("download", fdio.DIR, dir_fd=root)
        archive = fdio.new_file(download, "lean-4.35.0-linux.zip.part")
        with fetch_deadline(request["fetch"]["total_deadline_seconds"]):
            fetched = fetch(request["fetch"], archive, state, backend, clock)
        archive_stat = fdio.stable_file(archive, download, "lean-4.35.0-linux.zip.part", request["fetch"]["exact_bytes"])
        fdio.save(root, "FETCH.json", {**fetched, "archive_stat": archive_stat})
        state.update(phase="ZIP", pending=b"")
        manifest = extract(archive, root, request, state)
        fdio.stable_file(archive, download, "lean-4.35.0-linux.zip.part", request["fetch"]["exact_bytes"])
        live_root = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require((root_stat.st_dev, root_stat.st_ino, root_stat.st_mode) ==
                (live_root.st_dev, live_root.st_ino, live_root.st_mode), "root association changed")
        manifest["request_identity"] = identity(request)
        fdio.save(root, "MEMBER-MANIFEST.json", manifest)
        fdio.save(root, "RESULT.json", {"status": "AUTHENTICATED_NEW_ARCHIVE_DATA_ONLY",
                  "manifest_identity": identity(manifest), "authority": dict(request["authority"])})
        return manifest
    except BaseException as error:
        pending = state.pop("pending", b"")
        if archive is not None:
            try:
                state["partial_archive_stat"] = fdio.facts(os.fstat(archive))
            except OSError as stat_error:
                state["partial_archive_stat_error"] = str(stat_error)
        if root is not None:
            fdio.stop(root, error, state, pending)
        raise fdio.IntakeStop({**state, **fdio.error_details(error), "retry": False}) from error
    finally:
        for fd in (archive, download, root, parent):
            if fd is not None:
                os.close(fd)
